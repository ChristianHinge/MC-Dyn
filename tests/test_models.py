"""Tests for data model helpers."""

from __future__ import annotations

from ms_dyn.models import FrameTiming, StudyMetadata


class TestFrameTiming:
    def test_mid_times(self):
        t = FrameTiming(frame_start_s=[0.0, 60.0], frame_duration_s=[60.0, 60.0])
        assert t.frame_mid_s == [30.0, 90.0]

    def test_end_times(self):
        t = FrameTiming(frame_start_s=[0.0, 60.0], frame_duration_s=[30.0, 60.0])
        assert t.frame_end_s == [30.0, 120.0]

    def test_len(self):
        t = FrameTiming(frame_start_s=[0.0, 60.0, 120.0], frame_duration_s=[60.0] * 3)
        assert len(t) == 3


class TestStudyMetadata:
    def test_bmi_calculated(self):
        m = StudyMetadata("x", age=30.0, sex="M", weight_kg=70.0, height_cm=175.0)
        assert m.bmi is not None
        assert abs(m.bmi - 22.9) < 0.1

    def test_bmi_none_when_missing(self):
        m = StudyMetadata("x", age=None, sex=None, weight_kg=None, height_cm=None)
        assert m.bmi is None

    def test_bmi_none_when_height_missing(self):
        m = StudyMetadata("x", age=30.0, sex="M", weight_kg=70.0, height_cm=None)
        assert m.bmi is None
