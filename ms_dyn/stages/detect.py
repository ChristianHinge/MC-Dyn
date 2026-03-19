"""Detect whether a case's input is DICOM or NIfTI+JSON."""

from __future__ import annotations

import logging
from pathlib import Path

import pydicom.misc

from ms_dyn.models import InputFormat

log = logging.getLogger(__name__)


def _has_dicom_files(directory: Path) -> bool:
    """Return True if any file in directory looks like a DICOM file."""
    for f in directory.iterdir():
        if f.is_file():
            try:
                if pydicom.misc.is_dicom(f):
                    return True
            except Exception:
                continue
    return False


def _has_nifti_files(directory: Path) -> bool:
    return any(
        f.suffix in (".gz", ".nii") and f.name.endswith((".nii.gz", ".nii"))
        for f in directory.iterdir()
        if f.is_file()
    )


def detect_input_format(pet_dir: Path, ct_dir: Path) -> InputFormat:
    """
    Inspect pet_dir and ct_dir to determine input format.

    Returns InputFormat(is_dicom=True) if DICOM files are found,
    InputFormat(is_dicom=False) if NIfTI files are found.
    Raises ValueError if format is ambiguous or unsupported.
    """
    pet_dicom = _has_dicom_files(pet_dir)
    pet_nifti = _has_nifti_files(pet_dir)

    if pet_dicom and not pet_nifti:
        log.debug("Detected DICOM input in %s", pet_dir)
        return InputFormat(is_dicom=True)

    if pet_nifti and not pet_dicom:
        log.debug("Detected NIfTI input in %s", pet_dir)
        return InputFormat(is_dicom=False)

    if pet_dicom and pet_nifti:
        raise ValueError(
            f"Both DICOM and NIfTI files found in {pet_dir} — please provide one format only."
        )

    raise ValueError(
        f"No DICOM or NIfTI files found in {pet_dir}. "
        "Expected either DICOM files or a .nii.gz file."
    )
