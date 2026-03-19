"""Tests for input format detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mc_dyn.stages.detect import detect_input_format


def _make_nifti(directory: Path) -> Path:
    f = directory / "pet.nii.gz"
    f.touch()
    return f


def test_detects_nifti(tmp_path: Path) -> None:
    pet_dir = tmp_path / "pet"
    ct_dir = tmp_path / "ct"
    pet_dir.mkdir()
    ct_dir.mkdir()
    _make_nifti(pet_dir)
    _make_nifti(ct_dir)

    fmt = detect_input_format(pet_dir, ct_dir)
    assert not fmt.is_dicom


def test_empty_directory_raises(tmp_path: Path) -> None:
    pet_dir = tmp_path / "pet"
    ct_dir = tmp_path / "ct"
    pet_dir.mkdir()
    ct_dir.mkdir()

    with pytest.raises(ValueError, match="No DICOM or NIfTI files found"):
        detect_input_format(pet_dir, ct_dir)


def test_mixed_raises(tmp_path: Path, monkeypatch) -> None:
    import pydicom.misc

    pet_dir = tmp_path / "pet"
    ct_dir = tmp_path / "ct"
    pet_dir.mkdir()
    ct_dir.mkdir()

    _make_nifti(pet_dir)
    dicom_file = pet_dir / "file.dcm"
    dicom_file.touch()

    # Mock is_dicom to return True for the .dcm file
    def fake_is_dicom(path: Path) -> bool:
        return Path(path).suffix == ".dcm"

    monkeypatch.setattr(pydicom.misc, "is_dicom", fake_is_dicom)

    with pytest.raises(ValueError, match="Both DICOM and NIfTI files found"):
        detect_input_format(pet_dir, ct_dir)
