"""Convert DICOM to NIfTI using dcm2niix, or copy existing NIfTI files."""

from __future__ import annotations

import glob
import json
import logging
import shutil
import subprocess
from pathlib import Path

import nibabel as nib

from mc_dyn.models import CasePaths, InputFormat

log = logging.getLogger(__name__)


def _run_dcm2niix(dicom_dir: Path, output_dir: Path, filename: str) -> list[Path]:
    """Run dcm2niix and return produced .nii.gz paths."""
    cmd = [
        "dcm2niix",
        "-z", "y",          # gzip output
        "-f", filename,      # output filename stem
        "-o", str(output_dir),
        str(dicom_dir),
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"dcm2niix failed (exit {result.returncode}):\n{result.stderr}"
        )
    return list(output_dir.glob(f"{filename}*.nii.gz"))


def _find_nifti(directory: Path) -> Path:
    """Find a single .nii.gz file in directory; raise if not exactly one."""
    hits = list(directory.glob("*.nii.gz")) + list(directory.glob("*.nii"))
    if len(hits) == 0:
        raise FileNotFoundError(f"No NIfTI file found in {directory}")
    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple NIfTI files found in {directory}: {[h.name for h in hits]}. "
            "Please ensure each pet/ and ct/ folder contains exactly one NIfTI file."
        )
    return hits[0]


def _find_json_sidecar(nii_path: Path) -> Path | None:
    """Look for a JSON sidecar alongside a NIfTI file."""
    candidates = [
        nii_path.with_suffix("").with_suffix(".json"),   # file.nii.gz → file.json
        nii_path.with_suffix(".json"),                   # file.nii → file.json
    ]
    for c in candidates:
        if c.exists():
            return c
    # Also check same directory for any .json file
    jsons = list(nii_path.parent.glob("*.json"))
    if len(jsons) == 1:
        return jsons[0]
    return None


def convert_case(paths: CasePaths, fmt: InputFormat) -> None:
    """
    Ensure pet.nii.gz, ct.nii.gz, and pet.json exist in paths.output_dir.

    For DICOM input: converts both series via dcm2niix.
    For NIfTI input: copies/links existing files into output_dir.
    """
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    if fmt.is_dicom:
        _convert_dicom(paths)
    else:
        _copy_nifti(paths)

    # Validate PET is 4D
    pet_img = nib.load(paths.pet_nii)
    if pet_img.ndim != 4:
        raise RuntimeError(
            f"PET image must be 4D (got {pet_img.ndim}D). "
            "Check that the input is a dynamic PET scan."
        )

    log.info("[%s] PET: %s, frames=%d", paths.case_id, pet_img.shape[:3], pet_img.shape[3])


def _convert_dicom(paths: CasePaths) -> None:
    # Convert PET
    pet_niis = _run_dcm2niix(paths.pet_dir, paths.output_dir, "pet")
    if len(pet_niis) != 1:
        raise RuntimeError(
            f"dcm2niix produced {len(pet_niis)} NIfTI file(s) for PET "
            f"(expected exactly 1): {[p.name for p in pet_niis]}. "
            "Ensure the PET DICOM directory contains a single series."
        )

    # Move the produced file to the canonical name if needed
    produced_pet = pet_niis[0]
    if produced_pet != paths.pet_nii:
        produced_pet.rename(paths.pet_nii)

    # Copy accompanying JSON sidecar
    json_src = _find_json_sidecar(produced_pet if produced_pet.exists() else paths.pet_nii)
    if json_src and json_src != paths.pet_json:
        shutil.copy2(json_src, paths.pet_json)
    elif not json_src and not paths.pet_json.exists():
        # dcm2niix may have named it differently — search
        jsons = list(paths.output_dir.glob("pet*.json"))
        if jsons:
            jsons[0].rename(paths.pet_json)

    # Convert CT
    ct_niis = _run_dcm2niix(paths.ct_dir, paths.output_dir, "ct")
    if len(ct_niis) != 1:
        raise RuntimeError(
            f"dcm2niix produced {len(ct_niis)} NIfTI file(s) for CT "
            f"(expected exactly 1): {[p.name for p in ct_niis]}. "
            "Ensure the CT DICOM directory contains a single series."
        )
    produced_ct = ct_niis[0]
    if produced_ct != paths.ct_nii:
        produced_ct.rename(paths.ct_nii)


def _copy_nifti(paths: CasePaths) -> None:
    # PET
    src_pet = _find_nifti(paths.pet_dir)
    if src_pet != paths.pet_nii:
        shutil.copy2(src_pet, paths.pet_nii)

    # PET JSON sidecar
    src_json = _find_json_sidecar(src_pet)
    if src_json:
        if src_json != paths.pet_json:
            shutil.copy2(src_json, paths.pet_json)
    else:
        log.warning("[%s] No JSON sidecar found alongside %s", paths.case_id, src_pet)

    # CT
    src_ct = _find_nifti(paths.ct_dir)
    if src_ct != paths.ct_nii:
        shutil.copy2(src_ct, paths.ct_nii)
