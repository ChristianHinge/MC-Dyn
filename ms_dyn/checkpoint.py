"""Per-case pipeline checkpoint management with atomic writes."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

STAGES = ["detect", "convert", "metadata", "segment", "resample", "extract", "export"]

Status = Literal["pending", "completed", "failed"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "stages": {
            s: {"status": "pending", "timestamp": None, "error": None}
            for s in STAGES
        },
    }


class CheckpointManager:
    def __init__(self, state_file: Path, case_id: str) -> None:
        self._path = state_file
        self._case_id = case_id
        self._state: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                log.warning("Corrupt checkpoint file %s — resetting", self._path)
        return _default_state(self._case_id)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._state, f, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def is_completed(self, stage: str) -> bool:
        return self._state["stages"].get(stage, {}).get("status") == "completed"

    def mark_completed(self, stage: str) -> None:
        self._state["stages"][stage] = {
            "status": "completed",
            "timestamp": _now(),
            "error": None,
        }
        self._save()
        log.debug("[%s] Stage %s completed", self._case_id, stage)

    def mark_failed(self, stage: str, error: str) -> None:
        self._state["stages"][stage] = {
            "status": "failed",
            "timestamp": _now(),
            "error": error,
        }
        self._save()
        log.debug("[%s] Stage %s failed: %s", self._case_id, stage, error)

    def reset(self) -> None:
        """Reset all stages to pending (for --overwrite)."""
        self._state = _default_state(self._case_id)
        self._save()
