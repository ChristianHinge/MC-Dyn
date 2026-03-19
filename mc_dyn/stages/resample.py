"""Resample CT-space segmentation into PET voxel space using nibabel."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import nibabel.processing as nproc
import numpy as np

log = logging.getLogger(__name__)


def resample_seg_to_pet(seg_nii: Path, pet_nii: Path, output_path: Path) -> None:
    """
    Resample a multi-label segmentation from CT space to PET voxel space.

    Uses nearest-neighbour interpolation (order=0) to preserve integer label values.
    PET frame 0 is used as the reference geometry.
    """
    seg_img = nib.load(seg_nii)
    pet_img = nib.load(pet_nii)

    if pet_img.ndim != 4:
        raise RuntimeError(f"Expected 4D PET image, got {pet_img.ndim}D: {pet_nii}")

    # Extract frame 0 as 3D reference for geometry
    pet_data_0 = np.asarray(pet_img.dataobj[..., 0])
    pet_frame0 = nib.Nifti1Image(pet_data_0, pet_img.affine, pet_img.header)

    log.info(
        "Resampling seg %s → PET space %s",
        seg_img.shape,
        pet_frame0.shape,
    )

    # order=0 = nearest-neighbour; preserves integer label values
    seg_pet_img = nproc.resample_from_to(seg_img, pet_frame0, order=0)

    seg_pet_arr = np.asarray(seg_pet_img.dataobj)
    n_labels = len(np.unique(seg_pet_arr)) - 1  # exclude 0
    log.info("Resampled seg shape: %s, non-zero labels: %d", seg_pet_img.shape, n_labels)

    nib.save(seg_pet_img, output_path)
