"""Tests for checkpoint manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from ms_dyn.checkpoint import CheckpointManager, STAGES


def test_initial_state_all_pending(tmp_path: Path) -> None:
    state_file = tmp_path / ".pipeline_state.json"
    ckpt = CheckpointManager(state_file, "sub-001_ses-01")
    for stage in STAGES:
        assert not ckpt.is_completed(stage)


def test_mark_completed(tmp_path: Path) -> None:
    state_file = tmp_path / ".pipeline_state.json"
    ckpt = CheckpointManager(state_file, "sub-001_ses-01")
    ckpt.mark_completed("detect")
    assert ckpt.is_completed("detect")
    assert not ckpt.is_completed("convert")


def test_state_persists_across_instances(tmp_path: Path) -> None:
    state_file = tmp_path / ".pipeline_state.json"
    ckpt1 = CheckpointManager(state_file, "sub-001_ses-01")
    ckpt1.mark_completed("detect")
    ckpt1.mark_completed("convert")

    ckpt2 = CheckpointManager(state_file, "sub-001_ses-01")
    assert ckpt2.is_completed("detect")
    assert ckpt2.is_completed("convert")
    assert not ckpt2.is_completed("segment")


def test_mark_failed(tmp_path: Path) -> None:
    state_file = tmp_path / ".pipeline_state.json"
    ckpt = CheckpointManager(state_file, "sub-001_ses-01")
    ckpt.mark_failed("segment", "OOM error")
    assert not ckpt.is_completed("segment")


def test_reset_clears_all(tmp_path: Path) -> None:
    state_file = tmp_path / ".pipeline_state.json"
    ckpt = CheckpointManager(state_file, "sub-001_ses-01")
    for s in STAGES[:3]:
        ckpt.mark_completed(s)
    ckpt.reset()
    for s in STAGES:
        assert not ckpt.is_completed(s)


def test_atomic_write_creates_file(tmp_path: Path) -> None:
    state_file = tmp_path / "nested" / ".pipeline_state.json"
    ckpt = CheckpointManager(state_file, "sub-001_ses-01")
    ckpt.mark_completed("detect")
    assert state_file.exists()
