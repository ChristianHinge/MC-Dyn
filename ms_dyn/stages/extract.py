"""Extract TACs from 4D PET using nifti_dynamic and Moose segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from ms_dyn.models import CasePaths, FrameTiming

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
    seg_nii: Path,
) -> pd.DataFrame:
    """
    Extract aortic input function TACs using the nifti_dynamic Python API.

    seg_nii is the CT-space Moose segmentation containing the aorta label.
    It is resampled to PET space internally by nifti_dynamic.

    Produces 8 TACs per case:
      - 4 anatomical segments: ASCENDING, TOP, DESCENDING, DESCENDING_BOTTOM
      - 2 extraction modes per segment:
          * 1ml  — 3px-wide cylindrical VOI (1 mL)
          * full — all voxels in that anatomical segment

    Organ names: aorta_if_{segment}_1ml / aorta_if_{segment}_full
    Task: nifti_dynamic
    """
    try:
        from nifti_dynamic.aorta_rois import AortaSegment, pipeline  # type: ignore[import]
        from nifti_dynamic.tacs import extract_tac  # type: ignore[import]
        from nibabel.processing import resample_from_to
    except ImportError as e:
        raise ImportError("nifti_dynamic is not installed.") from e

    aorta_if_dir = paths.output_dir / "aorta_if"
    aorta_if_dir.mkdir(parents=True, exist_ok=True)

    # Load PET and CT-space segmentation
    dynpet = nib.load(paths.pet_nii)
    seg_img = nib.load(seg_nii)

    # Resample CT-space seg to PET space (same as nifti_dynamic CLI does internally)
    seg_pet = resample_from_to(seg_img, (dynpet.shape[:3], dynpet.affine), order=0)

    # Extract binary aorta mask
    aorta_mask = nib.Nifti1Image(
        (seg_pet.get_fdata() == aorta_label).astype("uint8"),
        affine=seg_pet.affine,
    )
    if aorta_mask.get_fdata().sum() == 0:
        raise RuntimeError(
            f"No aorta voxels found with label {aorta_label} in {seg_nii.name}"
        )

    frame_times_start = np.array(timing.frame_start_s)

    # Run pipeline once for 1ml VOIs (segment=None → all 4 segments)
    aorta_segments, aorta_vois_1ml = pipeline(
        aorta_mask=aorta_mask,
        dpet=dynpet,
        frame_times_start=frame_times_start,
        volume_ml=1.0,
        cylinder_width=3,
        segment=None,
        image_path=None,
    )
    log.info("[%s] Aorta segmented into 4 regions; extracting 8 input function TACs", paths.case_id)

    n_frames = dynpet.shape[3]
    voxel_vol = _voxel_volume_ml(seg_pet)
    rows: list[dict] = []

    for seg_enum in AortaSegment:
        seg_name = seg_enum.name.lower()

        for mode, source_img in (("1ml", aorta_vois_1ml), ("full", aorta_segments)):
            mask_arr = source_img.get_fdata() == seg_enum.value
            if not mask_arr.any():
                log.warning("[%s] Aorta segment %s has no voxels in %s mode", paths.case_id, seg_name, mode)
                continue

            mean_arr, std_arr, n_arr = extract_tac(dynpet, mask_arr)
            volume_ml = round(float(n_arr[0]) * voxel_vol, 4) if mode == "full" else 1.0
            organ_name = f"aorta_if_{seg_name}_{mode}"

            for frame_idx in range(n_frames):
                rows.append({
                    "case_id": paths.case_id,
                    "task": "nifti_dynamic",
                    "organ": organ_name,
                    "frame_idx": frame_idx,
                    "time_start_s": timing.frame_start_s[frame_idx],
                    "time_mid_s": timing.frame_mid_s[frame_idx],
                    "time_end_s": timing.frame_end_s[frame_idx],
                    "mean_value": float(mean_arr[frame_idx]),
                    "std_value": float(std_arr[frame_idx]),
                    "volume_ml": volume_ml,
                })

    # Save aorta_segments and aorta_vois for inspection
    nib.save(aorta_segments, aorta_if_dir / "aorta_segments.nii.gz")
    nib.save(aorta_vois_1ml, aorta_if_dir / "aorta_vois_1ml.nii.gz")
    log.info("[%s] Aorta IF NIfTIs saved to %s", paths.case_id, aorta_if_dir)

    return pd.DataFrame(rows)
