"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    overwrite: bool = False
    cases_filter: list[str] = field(default_factory=list)   # relative paths from input_dir
    moose_models: list[str] = field(default_factory=lambda: ["clin_ct_organs", "clin_ct_cardiac"])
    accelerator: str = "cuda"   # "cuda" | "cpu" | "mps"
    max_roi_size_factor: float = 2.0
