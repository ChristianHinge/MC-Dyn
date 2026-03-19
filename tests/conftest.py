"""Shared pytest fixtures for synthetic NIfTI test data."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


SHAPE_3D = (10, 10, 10)
SHAPE_4D = (10, 10, 10, 5)
AFFINE = np.diag([4.0, 4.0, 4.0, 1.0])   # 4 mm isotropic


@pytest.fixture
def tmp_case(tmp_path: Path) -> Path:
    """Create a minimal case directory structure (NIfTI input)."""
    case_dir = tmp_path / "sub001"
    pet_dir = case_dir / "pet"
    ct_dir = case_dir / "ct"
    pet_dir.mkdir(parents=True)
    ct_dir.mkdir(parents=True)
    return case_dir


@pytest.fixture
def synthetic_pet(tmp_path: Path) -> Path:
    """4D PET NIfTI with known values: all voxels = frame_index + 1."""
    data = np.zeros(SHAPE_4D, dtype=np.float32)
    for t in range(SHAPE_4D[3]):
        data[..., t] = float(t + 1)
    img = nib.Nifti1Image(data, AFFINE)
    path = tmp_path / "pet.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def synthetic_ct(tmp_path: Path) -> Path:
    """3D CT NIfTI filled with zeros."""
    data = np.zeros(SHAPE_3D, dtype=np.int16)
    img = nib.Nifti1Image(data, AFFINE)
    path = tmp_path / "ct.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def synthetic_seg(tmp_path: Path) -> Path:
    """
    3D segmentation with 2 labels:
      label 1 = first octant (0:5, 0:5, 0:5)
      label 2 = second octant (5:10, 0:5, 0:5)
    """
    data = np.zeros(SHAPE_3D, dtype=np.int16)
    data[0:5, 0:5, 0:5] = 1
    data[5:10, 0:5, 0:5] = 2
    img = nib.Nifti1Image(data, AFFINE)
    path = tmp_path / "seg.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def synthetic_timing_json(tmp_path: Path) -> Path:
    """Minimal BIDS-compatible JSON sidecar with frame timing."""
    n_frames = SHAPE_4D[3]
    sidecar = {
        "FrameTimesStart": [float(i * 60) for i in range(n_frames)],
        "FrameDuration": [60.0] * n_frames,
        "PatientID": "TEST001",
        "StudyDate": "20240101",
        "PatientAge": "045Y",
        "PatientSex": "M",
        "PatientWeight": 70.0,
        "PatientSize": 1.75,
    }
    path = tmp_path / "pet.json"
    path.write_text(json.dumps(sidecar))
    return path


@pytest.fixture
def label_map() -> dict[int, str]:
    return {1: "liver", 2: "spleen"}
