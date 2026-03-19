"""CLI entry point for mc-dyn."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from mc_dyn.config import PipelineConfig
from mc_dyn.pipeline import run_batch


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@click.group()
def main() -> None:
    """MC-Dyn: Multi Center Dynamic PET/CT TAC extraction pipeline."""


@main.command("run")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Re-run all stages, ignoring existing checkpoints.",
)
@click.option(
    "--cases",
    multiple=True,
    metavar="PATH",
    help="Limit to specific cases (relative path from INPUT_DIR). Can be repeated.",
)
@click.option(
    "--accelerator",
    default="cuda",
    show_default=True,
    type=click.Choice(["cuda", "cpu", "mps"]),
    help="Hardware accelerator for Moose.",
)
@click.option(
    "--max-roi-size-factor",
    default=2.0,
    show_default=True,
    type=float,
    help="nifti_dynamic max ROI size factor.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Debug logging.")
def run_cmd(
    input_dir: Path,
    output: Path,
    overwrite: bool,
    cases: tuple[str, ...],
    accelerator: str,
    max_roi_size_factor: float,
    verbose: bool,
) -> None:
    """Process dynamic PET/CT studies and extract TACs.

    INPUT_DIR must contain subject directories with pet/ and ct/ subdirectories
    (at any depth). Both DICOM and NIfTI+JSON inputs are supported.

    By default, previously completed pipeline stages are skipped (resume mode).
    Use --overwrite to force re-processing from scratch.
    """
    _setup_logging(verbose)

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output,
        overwrite=overwrite,
        cases_filter=list(cases),
        accelerator=accelerator,
        max_roi_size_factor=max_roi_size_factor,
    )

    run_batch(config)
