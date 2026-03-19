"""Extract subject metadata and frame timing from DICOM or JSON sidecar."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pydicom

from ms_dyn.models import CasePaths, FrameTiming, InputFormat, StudyMetadata

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Age parsing
# ---------------------------------------------------------------------------

_AGE_RE = re.compile(r"^(\d+)([DWMY])$", re.IGNORECASE)

def _parse_dicom_age(raw: str | None) -> float | None:
    """Parse DICOM PatientAge string (e.g. '045Y', '006M', '002W', '010D') to years."""
    if not raw:
        return None
    m = _AGE_RE.match(raw.strip())
    if not m:
        log.warning("Cannot parse PatientAge: %r", raw)
        return None
    value, unit = int(m.group(1)), m.group(2).upper()
    if unit == "Y":
        return float(value)
    if unit == "M":
        return round(value / 12, 1)
    if unit == "W":
        return round(value / 52.18, 2)
    if unit == "D":
        return round(value / 365.25, 3)
    return None


# ---------------------------------------------------------------------------
# Metadata from DICOM
# ---------------------------------------------------------------------------

def _first_dicom(directory: Path) -> pydicom.Dataset | None:
    for f in sorted(directory.iterdir()):
        if f.is_file():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                return ds
            except Exception:
                continue
    return None


def _injected_dose_mbq_from_dicom(ds: "pydicom.Dataset") -> float | None:
    """Extract injected dose in MBq from RadiopharmaceuticalInformationSequence."""
    seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if seq and len(seq) > 0:
        dose_bq = getattr(seq[0], "RadionuclideTotalDose", None)
        if dose_bq is not None:
            return round(float(dose_bq) / 1e6, 3)  # Bq → MBq
    return None


def _extract_from_dicom(pet_dir: Path, case_id: str) -> tuple[StudyMetadata, str | None, str | None]:
    """Returns (StudyMetadata, patient_id, study_date)."""
    ds = _first_dicom(pet_dir)
    if ds is None:
        raise RuntimeError(f"[{case_id}] No readable DICOM file found in {pet_dir}")

    patient_id: str | None = getattr(ds, "PatientID", None)
    study_date: str | None = getattr(ds, "StudyDate", None)

    age = _parse_dicom_age(getattr(ds, "PatientAge", None))
    sex_raw: str | None = getattr(ds, "PatientSex", None)
    sex = sex_raw.upper() if sex_raw and sex_raw.upper() in ("M", "F") else None
    weight: float | None = getattr(ds, "PatientWeight", None)
    size: float | None = getattr(ds, "PatientSize", None)
    height_cm = round(size * 100, 1) if size else None

    meta = StudyMetadata(
        case_id=case_id,
        age=float(age) if age is not None else None,
        sex=sex,
        weight_kg=float(weight) if weight is not None else None,
        height_cm=height_cm,
        injected_dose_mbq=_injected_dose_mbq_from_dicom(ds),
    )
    return meta, str(patient_id) if patient_id else None, str(study_date) if study_date else None


# ---------------------------------------------------------------------------
# Metadata from JSON sidecar
# ---------------------------------------------------------------------------

def _extract_from_json(json_path: Path, case_id: str) -> tuple[StudyMetadata, str | None, str | None]:
    """Returns (StudyMetadata, patient_id, study_date).

    dcm2niix copies DICOM tags verbatim into the JSON sidecar using their
    standard keyword names.
    """
    with open(json_path) as f:
        sidecar = json.load(f)

    patient_id: str | None = sidecar.get("PatientID")
    study_date: str | None = sidecar.get("StudyDate")

    age = _parse_dicom_age(sidecar.get("PatientAge"))
    sex_raw: str | None = sidecar.get("PatientSex")
    sex = sex_raw.upper() if sex_raw and sex_raw.upper() in ("M", "F") else None
    weight = sidecar.get("PatientWeight")
    size = sidecar.get("PatientSize")
    height_cm = round(float(size) * 100, 1) if size else None

    # dcm2niix copies RadionuclideTotalDose (Bq) into JSON
    dose_bq = sidecar.get("RadionuclideTotalDose")
    injected_dose_mbq = round(float(dose_bq) / 1e6, 3) if dose_bq is not None else None

    meta = StudyMetadata(
        case_id=case_id,
        age=float(age) if age is not None else None,
        sex=sex,
        weight_kg=float(weight) if weight is not None else None,
        height_cm=height_cm,
        injected_dose_mbq=injected_dose_mbq,
    )
    return meta, str(patient_id) if patient_id else None, str(study_date) if study_date else None


# ---------------------------------------------------------------------------
# Frame timing
# ---------------------------------------------------------------------------

def extract_frame_timing(json_path: Path) -> FrameTiming:
    """
    Parse frame timing from a BIDS/dcm2niix JSON sidecar.
    Raises RuntimeError if required fields are missing.
    """
    if not json_path.exists():
        raise RuntimeError(
            f"JSON sidecar not found: {json_path}. "
            "Frame timing is mandatory — provide a dcm2niix-style sidecar alongside the NIfTI."
        )

    with open(json_path) as f:
        sidecar = json.load(f)

    starts = sidecar.get("FrameTimesStart")
    durations = sidecar.get("FrameDuration")

    if starts is None:
        raise RuntimeError(
            f"'FrameTimesStart' missing from {json_path}. "
            "Cannot determine frame timing without this field."
        )
    if durations is None:
        raise RuntimeError(
            f"'FrameDuration' missing from {json_path}. "
            "Cannot determine frame timing without this field."
        )

    starts = [float(v) for v in starts]
    durations = [float(v) for v in durations]

    if len(starts) != len(durations):
        raise RuntimeError(
            f"FrameTimesStart ({len(starts)} entries) and "
            f"FrameDuration ({len(durations)} entries) must have the same length in {json_path}."
        )

    return FrameTiming(frame_start_s=starts, frame_duration_s=durations)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_metadata(
    paths: CasePaths,
    fmt: InputFormat,
) -> tuple[StudyMetadata, str | None, str | None]:
    """
    Extract study metadata and identify patient_id + study_date for anonymization.

    Returns (StudyMetadata, patient_id, study_date).
    """
    if fmt.is_dicom:
        return _extract_from_dicom(paths.pet_dir, paths.case_id)
    else:
        if paths.pet_json.exists():
            return _extract_from_json(paths.pet_json, paths.case_id)
        else:
            log.warning("[%s] No JSON sidecar — metadata fields will be None", paths.case_id)
            meta = StudyMetadata(
                case_id=paths.case_id,
                age=None, sex=None, weight_kg=None, height_cm=None,
                injected_dose_mbq=None,
            )
            return meta, None, None
