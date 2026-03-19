"""Core data models shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawCase:
    """A discovered case before anonymized ID assignment."""
    original_path: Path   # the directory containing pet/ and ct/
    pet_dir: Path
    ct_dir: Path


@dataclass
class CasePaths:
    """All paths for a single case, post-ID assignment."""
    case_id: str          # anonymized BIDS ID, e.g. "sub-001_ses-01"
    original_path: Path
    pet_dir: Path
    ct_dir: Path
    output_dir: Path      # output_dir/cases/{case_id}/

    @property
    def pet_nii(self) -> Path:
        return self.output_dir / "pet.nii.gz"

    @property
    def ct_nii(self) -> Path:
        return self.output_dir / "ct.nii.gz"

    @property
    def pet_json(self) -> Path:
        return self.output_dir / "pet.json"

    @property
    def seg_dir(self) -> Path:
        return self.output_dir / "seg"

    @property
    def seg_pet_nii(self) -> Path:
        return self.output_dir / "seg_pet.nii.gz"

    @property
    def tacs_csv(self) -> Path:
        return self.output_dir / "tacs.csv"

    @property
    def state_file(self) -> Path:
        return self.output_dir / ".pipeline_state.json"


@dataclass
class InputFormat:
    is_dicom: bool    # False means NIfTI+JSON


@dataclass
class FrameTiming:
    frame_start_s: list[float]
    frame_duration_s: list[float]

    @property
    def frame_mid_s(self) -> list[float]:
        return [s + d / 2 for s, d in zip(self.frame_start_s, self.frame_duration_s)]

    @property
    def frame_end_s(self) -> list[float]:
        return [s + d for s, d in zip(self.frame_start_s, self.frame_duration_s)]

    def __len__(self) -> int:
        return len(self.frame_start_s)


@dataclass
class StudyMetadata:
    case_id: str
    age: float | None
    sex: str | None          # "M" or "F"
    weight_kg: float | None
    height_cm: float | None
    injected_dose_mbq: float | None

    @property
    def bmi(self) -> float | None:
        if self.weight_kg and self.height_cm:
            return round(self.weight_kg / (self.height_cm / 100) ** 2, 1)
        return None
