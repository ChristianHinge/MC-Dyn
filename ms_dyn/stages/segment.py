"""Organ segmentation with Moose — one NIfTI per model, no label merging."""

from __future__ import annotations

import logging
from pathlib import Path

from ms_dyn.models import CasePaths

log = logging.getLogger(__name__)


def _find_moose_seg_nii(seg_dir: Path) -> Path:
    """
    Locate a segmentation NIfTI produced by Moose when seg_paths is empty.

    Moose writes results to seg_dir; the exact sub-path may vary by version.
    """
    candidates = list(seg_dir.rglob("*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"No segmentation NIfTI found in {seg_dir} after running Moose. "
            "Check Moose output above for errors."
        )
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        if "seg" in c.stem.lower():
            return c
    return candidates[0]


def run_moose(
    ct_nii: Path,
    seg_dir: Path,
    model_names: list[str],
    accelerator: str,
) -> list[tuple[Path, dict[int, str]]]:
    """
    Run Moose segmentation on CT.

    Returns a list of (seg_nii_path, label_map) — one entry per model.
    Each model produces its own NIfTI file with its own label integer space.
    Labels must NOT be merged across models as the integers overlap.

    moose() returns (List[output_paths], List[Model]).
    Model.organ_indices is Dict[int, str] — no file parsing needed.
    """
    seg_dir.mkdir(parents=True, exist_ok=True)

    try:
        from moosez import moose  # type: ignore[import]
    except ImportError as e:
        raise ImportError("moosez is not installed. Install it with: pip install moosez") from e

    log.info("[moose] Running models %s on %s", model_names, ct_nii)
    seg_paths, used_models = moose(
        input_data=str(ct_nii),
        model_names=model_names,
        output_dir=str(seg_dir),
        accelerator=accelerator,
    )

    if not seg_paths:
        raise RuntimeError(
            "Moose returned no segmentation paths. Check Moose output for errors."
        )

    results: list[tuple[Path, dict[int, str]]] = []
    for seg_path, model in zip(seg_paths, used_models):
        p = Path(seg_path) if isinstance(seg_path, str) else seg_path
        log.info("[moose] %s → %s (%d labels)", model.folder_name, p, len(model.organ_indices))
        results.append((p, model.organ_indices))

    return results


def find_aorta(
    model_results: list[tuple[Path, dict[int, str]]],
) -> tuple[Path, int] | None:
    """
    Search all model results for an aorta label.

    Returns (ct_seg_path, label_int) for the first model that has an aorta label,
    or None if not found.
    """
    for seg_path, label_map in model_results:
        for label_int, name in label_map.items():
            if "aorta" in name.lower():
                log.debug("Found aorta label %d=%r in %s", label_int, name, seg_path.name)
                return seg_path, label_int
    log.warning("No aorta label found across all Moose models. Skipping input function.")
    return None


def seg_pet_path(ct_seg_path: Path) -> Path:
    """
    Derive the PET-space resampled path from a CT-space seg path.

    e.g. .../seg/clin_CT_cardiac_segmentation_ct.nii.gz
         → .../seg/clin_CT_cardiac_segmentation_pet.nii.gz
    """
    name = ct_seg_path.name
    if name.endswith("_ct.nii.gz"):
        pet_name = name[: -len("_ct.nii.gz")] + "_pet.nii.gz"
    else:
        pet_name = ct_seg_path.stem + "_pet.nii.gz"
    return ct_seg_path.parent / pet_name


def run_segment(
    paths: CasePaths,
    model_names: list[str],
    accelerator: str,
) -> tuple[list[tuple[Path, dict[int, str]]], tuple[Path, int] | None]:
    """
    Full segment stage: run Moose on CT.

    Returns:
        model_results: list of (ct_seg_path, label_map) — one per model
        aorta_ct:      (ct_seg_path, label_int) for the aorta, or None
    """
    model_results = run_moose(paths.ct_nii, paths.seg_dir, model_names, accelerator)
    aorta_ct = find_aorta(model_results)
    return model_results, aorta_ct


