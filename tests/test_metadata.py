"""Tests for metadata extraction and frame timing parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mc_dyn.stages.metadata import _parse_dicom_age, extract_frame_timing


class TestParseDicomAge:
    def test_years(self):
        assert _parse_dicom_age("045Y") == 45.0

    def test_months(self):
        result = _parse_dicom_age("006M")
        assert abs(result - 0.5) < 0.01

    def test_weeks(self):
        result = _parse_dicom_age("002W")
        assert result is not None and result < 0.1

    def test_days(self):
        result = _parse_dicom_age("010D")
        assert result is not None and result < 0.05

    def test_none_input(self):
        assert _parse_dicom_age(None) is None

    def test_empty_string(self):
        assert _parse_dicom_age("") is None

    def test_lowercase(self):
        assert _parse_dicom_age("030y") == 30.0


class TestFrameTiming:
    def test_valid_sidecar(self, tmp_path: Path):
        sidecar = {
            "FrameTimesStart": [0.0, 60.0, 120.0],
            "FrameDuration": [60.0, 60.0, 60.0],
        }
        p = tmp_path / "pet.json"
        p.write_text(json.dumps(sidecar))
        timing = extract_frame_timing(p)
        assert len(timing) == 3
        assert timing.frame_mid_s == [30.0, 90.0, 150.0]
        assert timing.frame_end_s == [60.0, 120.0, 180.0]

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="JSON sidecar not found"):
            extract_frame_timing(tmp_path / "nonexistent.json")

    def test_missing_frame_times_start_raises(self, tmp_path: Path):
        p = tmp_path / "pet.json"
        p.write_text(json.dumps({"FrameDuration": [60.0]}))
        with pytest.raises(RuntimeError, match="FrameTimesStart"):
            extract_frame_timing(p)

    def test_missing_frame_duration_raises(self, tmp_path: Path):
        p = tmp_path / "pet.json"
        p.write_text(json.dumps({"FrameTimesStart": [0.0]}))
        with pytest.raises(RuntimeError, match="FrameDuration"):
            extract_frame_timing(p)

    def test_mismatched_lengths_raises(self, tmp_path: Path):
        p = tmp_path / "pet.json"
        p.write_text(json.dumps({
            "FrameTimesStart": [0.0, 60.0],
            "FrameDuration": [60.0],
        }))
        with pytest.raises(RuntimeError, match="same length"):
            extract_frame_timing(p)
