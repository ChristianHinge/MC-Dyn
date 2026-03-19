"""Organ segmentation with Moose and aorta input function extraction."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from ms_dyn.models import CasePaths

log = logging.getLogger(__name__)


def _find_moose_seg_nii(seg_dir: Path) -> Path:
    """
    Locate the segmentation NIfTI produced by Moose.

    Moose writes results to seg_dir; the exact sub-path may vary by version.
    We search for the first .nii.gz file that isn't a temporary artifact.
    """
    candidates = list(seg_dir.rglob("*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"No segmentation NIfTI found in {seg_dir} after running Moose. "
            "Check Moose output above for errors."
        )
    if len(candidates) == 1:
        return candidates[0]
    # Prefer files named 'seg' or containing the model name
    for c in candidates:
        if "seg" in c.stem.lower():
            return c
    return candidates[0]


def _find_moose_labels_json(seg_dir: Path) -> Path:
    """Locate Moose's labels.json mapping int labels → organ names."""
    candidates = list(seg_dir.rglob("labels.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No labels.json found in {seg_dir} after running Moose. "
            "Cannot map segmentation labels to organ names."
        )
    return candidates[0]


def run_moose(
    ct_nii: Path,
    seg_dir: Path,
    model_names: list[str],
    accelerator: str,
) -> tuple[Path, dict[int, str]]:
    """
    Run Moose segmentation on CT and return (seg_nii_path, label_map).

    label_map: {label_int: organ_name}

    Moose API (moosez >= 3.0):
        from moosez import moose
        moose(input_path, model_names, output_directory, accelerator)
    """
    seg_dir.mkdir(parents=True, exist_ok=True)

    try:
        from moosez import moose  # type: ignore[import]
        log.info("[moose] Running models %s on %s", model_names, ct_nii)
        moose(
            input_path=str(ct_nii),
            model_names=model_names,
            output_directory=str(seg_dir),
            accelerator=accelerator,
        )
    except ImportError as e:
        raise ImportError(
            "moosez is not installed. Install it with: pip install moosez"
        ) from e

    seg_nii = _find_moose_seg_nii(seg_dir)
    labels_json = _find_moose_labels_json(seg_dir)

    with open(labels_json) as f:
        raw_labels: dict = json.load(f)

    # Moose labels.json format: {"1": "organ_name", ...} or {"organ_name": 1}
    label_map: dict[int, str] = {}
    for k, v in raw_labels.items():
        if isinstance(v, int):
            label_map[v] = k          # {"organ": int_label}
        elif isinstance(k, str) and k.isdigit():
            label_map[int(k)] = str(v)  # {"int_str": "organ"}
        else:
            try:
                label_map[int(k)] = str(v)
            except ValueError:
                pass

    log.info("[moose] Segmentation complete: %s (%d labels)", seg_nii, len(label_map))
    return seg_nii, label_map


def find_aorta_label(label_map: dict[int, str]) -> int | None:
    """Find the integer label corresponding to the aorta in Moose's label map."""
    for label_int, name in label_map.items():
        if "aorta" in name.lower():
            log.debug("Found aorta label: %d → %r", label_int, name)
            return label_int
    log.warning("No aorta label found in Moose label map. Skipping input function extraction.")
    return None


def extract_aorta_input_function(
    pet_nii: Path,
    seg_pet_nii: Path,
    pet_json: Path,
    aorta_label: int,
    output_path: Path,
) -> None:
    """
    Extract the aortic input function TAC using nifti_dynamic's CLI.

    nifti_dynamic CLI:
        extract_input_function --pet <pet> --totalseg <seg> --output <out>
            --sidecar <json> --aorta-index <label> --skip-visualization
    """
    cmd = [
        "extract_input_function",
        "--pet", str(pet_nii),
        "--totalseg", str(seg_pet_nii),
        "--output", str(output_path),
        "--aorta-index", str(aorta_label),
        "--skip-visualization",
    ]
    if pet_json.exists():
        cmd += ["--sidecar", str(pet_json)]

    log.info("[nifti_dynamic] Extracting aorta input function: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"extract_input_function failed (exit {result.returncode}):\n{result.stderr}"
        )
    log.info("[nifti_dynamic] Aorta input function written to %s", output_path)


def run_segment(
    paths: CasePaths,
    model_names: list[str],
    accelerator: str,
) -> tuple[Path, dict[int, str], int | None]:
    """
    Full segment stage:
    1. Run Moose on CT
    2. Identify aorta label for later input function extraction

    Returns (seg_nii_path, label_map, aorta_label_int_or_None).
    The aorta input function is extracted in the extract stage after resampling.
    """
    seg_nii, label_map = run_moose(paths.ct_nii, paths.seg_dir, model_names, accelerator)
    aorta_label = find_aorta_label(label_map)
    return seg_nii, label_map, aorta_label
