"""Extract TACs from 4D PET using nifti_dynamic and Moose segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from ms_dyn.models import CasePaths, FrameTiming
from ms_dyn.stages.segment import extract_aorta_input_function

log = logging.getLogger(__name__)


def _voxel_volume_ml(img: nib.Nifti1Image) -> float:
    """Volume of one voxel in mL (= cm³)."""
    zooms = img.header.get_zooms()[:3]   # (dx, dy, dz) in mm
    return float(np.prod(zooms)) / 1000.0  # mm³ → mL


def extract_organ_tacs(
    pet_nii: Path,
    seg_pet_nii: Path,
    timing: FrameTiming,
    label_map: dict[int, str],
    case_id: str,
    task: str,
    max_roi_size_factor: float = 2.0,
) -> pd.DataFrame:
    """
    Extract per-organ TACs from 4D PET using nifti_dynamic.extract_multiple_tacs.

    extract_multiple_tacs takes the full integer-labelled segmentation and returns
    dicts of {label_int: mean_tac_array}, handling memory efficiently for large organs.

    Returns a long-format DataFrame matching the tacs.csv schema.
    """
    try:
        from nifti_dynamic.tacs import extract_multiple_tacs  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "nifti_dynamic is not installed. Install it with: pip install nifti_dynamic"
        ) from e

    pet_img = nib.load(pet_nii)
    seg_img = nib.load(seg_pet_nii)
    # extract_multiple_tacs expects seg as a numpy array with integer labels
    seg_data = np.asarray(seg_img.dataobj).astype(np.int16)

    if seg_data.shape != pet_img.shape[:3]:
        raise RuntimeError(
            f"Segmentation shape {seg_data.shape} does not match "
            f"PET spatial shape {pet_img.shape[:3]}. "
            "Ensure the resample stage completed correctly."
        )

    n_frames = pet_img.shape[3]
    if n_frames != len(timing):
        raise RuntimeError(
            f"PET has {n_frames} frames but timing has {len(timing)} entries."
        )

    voxel_vol = _voxel_volume_ml(seg_img)

    # extract_multiple_tacs returns {label_int: np.ndarray(n_frames)} dicts
    tacs_mean, tacs_std, tacs_n = extract_multiple_tacs(
        pet_img, seg_data, max_roi_size_factor=max_roi_size_factor
    )

    rows: list[dict] = []
    for label_int, organ_name in label_map.items():
        if label_int not in tacs_mean:
            log.warning("[%s] Organ %r (label %d) not in extracted TACs — skipping",
                        case_id, organ_name, label_int)
            continue

        mean_arr = np.asarray(tacs_mean[label_int])
        std_arr = np.asarray(tacs_std[label_int])
        n_arr = np.asarray(tacs_n[label_int])
        volume_ml = round(float(n_arr[0]) * voxel_vol, 4)

        for frame_idx in range(n_frames):
            rows.append({
                "case_id": case_id,
                "task": task,
                "organ": organ_name,
                "frame_idx": frame_idx,
                "time_start_s": timing.frame_start_s[frame_idx],
                "time_mid_s": timing.frame_mid_s[frame_idx],
                "time_end_s": timing.frame_end_s[frame_idx],
                "mean_value": float(mean_arr[frame_idx]),
                "std_value": float(std_arr[frame_idx]),
                "volume_ml": volume_ml,
            })

    n_organs = len(rows) // max(n_frames, 1)
    log.info("[%s] Extracted TACs for %d organs (%d rows)", case_id, n_organs, len(rows))
    return pd.DataFrame(rows)


def extract_aorta_tac(
    paths: CasePaths,
    timing: FrameTiming,
    aorta_label: int,
) -> pd.DataFrame:
    """
    Extract aortic input function TAC using nifti_dynamic's extract_input_function CLI.

    Returns a DataFrame with the same schema as organ TACs, with task="nifti_dynamic".
    """
    output_csv = paths.output_dir / "aorta_input_function.csv"
    extract_aorta_input_function(
        pet_nii=paths.pet_nii,
        seg_pet_nii=paths.seg_pet_nii,
        pet_json=paths.pet_json,
        aorta_label=aorta_label,
        output_path=output_csv,
    )

    if not output_csv.exists():
        log.warning("[%s] Aorta input function CSV not produced — skipping", paths.case_id)
        return pd.DataFrame()

    # nifti_dynamic extract_input_function outputs per-VOI TACs
    # The CLI saves using save_tac: columns = time, mu, std, n_voxels
    # It may produce one file per VOI segment; we load whatever is in the output
    raw = pd.read_csv(output_csv)
    log.debug("[%s] Aorta IF CSV columns: %s", paths.case_id, list(raw.columns))

    rows: list[dict] = []
    # Expected long-format from save_tac: time, mu, std, n_voxels
    if {"time", "mu", "std"}.issubset(raw.columns):
        n = len(raw)
        for i, row in raw.iterrows():
            frame_idx = int(i)
            if frame_idx >= len(timing):
                break
            rows.append({
                "case_id": paths.case_id,
                "task": "nifti_dynamic",
                "organ": "aorta_input_function",
                "frame_idx": frame_idx,
                "time_start_s": timing.frame_start_s[frame_idx],
                "time_mid_s": timing.frame_mid_s[frame_idx],
                "time_end_s": timing.frame_end_s[frame_idx],
                "mean_value": float(row["mu"]),
                "std_value": float(row["std"]),
                "volume_ml": float(row.get("n_voxels", float("nan"))),
            })

    return pd.DataFrame(rows)
