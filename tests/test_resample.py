"""Tests for segmentation resampling."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mc_dyn.stages.resample import resample_seg_to_pet


def _make_seg(path: Path, shape=(20, 20, 20), affine=None) -> Path:
    if affine is None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
    data = np.zeros(shape, dtype=np.int16)
    data[5:15, 5:15, 5:15] = 1
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def _make_pet(path: Path, shape=(10, 10, 10, 5), affine=None) -> Path:
    if affine is None:
        affine = np.diag([4.0, 4.0, 4.0, 1.0])
    data = np.ones(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def test_output_shape_matches_pet(tmp_path: Path) -> None:
    seg_path = _make_seg(tmp_path / "seg.nii.gz")
    pet_path = _make_pet(tmp_path / "pet.nii.gz")
    out_path = tmp_path / "seg_pet.nii.gz"

    resample_seg_to_pet(seg_path, pet_path, out_path)

    result = nib.load(out_path)
    pet_img = nib.load(pet_path)
    assert result.shape == pet_img.shape[:3]


def test_integer_labels_preserved(tmp_path: Path) -> None:
    """Nearest-neighbour interpolation must preserve integer labels."""
    seg_path = _make_seg(tmp_path / "seg.nii.gz")
    pet_path = _make_pet(tmp_path / "pet.nii.gz")
    out_path = tmp_path / "seg_pet.nii.gz"

    resample_seg_to_pet(seg_path, pet_path, out_path)

    result_data = np.asarray(nib.load(out_path).dataobj)
    unique_values = set(np.unique(result_data).tolist())
    assert unique_values.issubset({0, 1}), f"Unexpected label values: {unique_values}"


def test_non_4d_pet_raises(tmp_path: Path) -> None:
    seg_path = _make_seg(tmp_path / "seg.nii.gz")
    # Create a 3D "PET"
    data = np.ones((10, 10, 10), dtype=np.float32)
    pet_path = tmp_path / "pet3d.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), pet_path)
    out_path = tmp_path / "out.nii.gz"

    with pytest.raises(RuntimeError, match="4D"):
        resample_seg_to_pet(seg_path, pet_path, out_path)
