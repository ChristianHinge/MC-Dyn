"""Tests for TAC extraction."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mc_dyn.models import FrameTiming

pytest.importorskip("nifti_dynamic", reason="nifti_dynamic not installed")

from mc_dyn.stages.extract import extract_organ_tacs


AFFINE = np.diag([4.0, 4.0, 4.0, 1.0])


def _make_pet_uniform(path: Path, value: float, shape=(10, 10, 10, 3)) -> Path:
    """PET where every voxel in every frame has the same value."""
    data = np.full(shape, value, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, AFFINE), path)
    return path


def _make_seg_single_label(path: Path, shape=(10, 10, 10), label: int = 1) -> Path:
    """Segmentation with one label covering the first octant."""
    data = np.zeros(shape, dtype=np.int16)
    data[0:5, 0:5, 0:5] = label
    nib.save(nib.Nifti1Image(data, AFFINE), path)
    return path


def _timing(n: int) -> FrameTiming:
    return FrameTiming(
        frame_start_s=[float(i * 60) for i in range(n)],
        frame_duration_s=[60.0] * n,
    )


def test_mean_value_correct(tmp_path: Path) -> None:
    n_frames = 3
    pet_val = 42.0
    pet_path = _make_pet_uniform(tmp_path / "pet.nii.gz", pet_val, (10, 10, 10, n_frames))
    seg_path = _make_seg_single_label(tmp_path / "seg.nii.gz", label=1)
    timing = _timing(n_frames)
    label_map = {1: "liver"}

    df = extract_organ_tacs(pet_path, seg_path, timing, label_map, "sub-001_ses-01", "moosez_test")

    assert len(df) == n_frames
    assert (df["organ"] == "liver").all()
    assert np.allclose(df["mean_value"].values, pet_val, atol=1e-3)


def test_empty_mask_skipped(tmp_path: Path) -> None:
    n_frames = 2
    pet_path = _make_pet_uniform(tmp_path / "pet.nii.gz", 10.0, (10, 10, 10, n_frames))
    # Segmentation with label 99 but no voxels in PET space
    data = np.zeros((10, 10, 10), dtype=np.int16)
    nib.save(nib.Nifti1Image(data, AFFINE), tmp_path / "seg.nii.gz")
    timing = _timing(n_frames)
    label_map = {99: "ghost_organ"}

    df = extract_organ_tacs(tmp_path / "pet.nii.gz", tmp_path / "seg.nii.gz",
                             timing, label_map, "sub-001_ses-01", "moosez_test")
    assert df.empty


def test_output_columns(tmp_path: Path) -> None:
    n_frames = 2
    pet_path = _make_pet_uniform(tmp_path / "pet.nii.gz", 5.0, (10, 10, 10, n_frames))
    seg_path = _make_seg_single_label(tmp_path / "seg.nii.gz", label=1)
    timing = _timing(n_frames)

    df = extract_organ_tacs(pet_path, seg_path, timing, {1: "liver"}, "sub-001_ses-01", "moosez_test")

    expected_cols = {
        "case_id", "task", "organ", "frame_idx",
        "time_start_s", "time_mid_s", "time_end_s",
        "mean_value", "std_value", "volume_ml",
    }
    assert expected_cols.issubset(set(df.columns))


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    n_frames = 2
    pet_path = _make_pet_uniform(tmp_path / "pet.nii.gz", 1.0, (10, 10, 10, n_frames))
    # Seg with different spatial shape
    data = np.zeros((5, 5, 5), dtype=np.int16)
    data[0:3, 0:3, 0:3] = 1
    nib.save(nib.Nifti1Image(data, AFFINE), tmp_path / "seg.nii.gz")
    timing = _timing(n_frames)

    with pytest.raises(RuntimeError, match="shape"):
        extract_organ_tacs(tmp_path / "pet.nii.gz", tmp_path / "seg.nii.gz",
                           timing, {1: "liver"}, "sub-001_ses-01", "moosez_test")
