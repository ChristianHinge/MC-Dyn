"""Write per-case CSVs and aggregate all cases into combined outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mc_dyn.models import StudyMetadata

log = logging.getLogger(__name__)

TAC_COLUMNS = [
    "case_id", "task", "organ", "frame_idx",
    "time_start_s", "time_mid_s", "time_end_s",
    "mean_value", "std_value", "volume_ml",
]

STUDY_COLUMNS = ["case_id", "age", "sex", "weight_kg", "height_cm", "bmi", "injected_dose_mbq"]


def write_case_tacs(tacs: pd.DataFrame, output_path: Path) -> None:
    """Write per-case tacs.csv with canonical column order."""
    cols = [c for c in TAC_COLUMNS if c in tacs.columns]
    tacs[cols].to_csv(output_path, index=False)
    log.info("Wrote %d TAC rows to %s", len(tacs), output_path)


def aggregate_outputs(
    case_output_dirs: list[Path],
    output_dir: Path,
    study_metadata: list[StudyMetadata],
) -> None:
    """
    Concatenate all per-case tacs.csv into output_dir/tacs.csv.
    Write output_dir/studies.csv from collected StudyMetadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate TACs
    tac_dfs: list[pd.DataFrame] = []
    for case_dir in case_output_dirs:
        tac_csv = case_dir / "tacs.csv"
        if tac_csv.exists():
            tac_dfs.append(pd.read_csv(tac_csv))
        else:
            log.warning("Missing tacs.csv for %s — skipping in combined output", case_dir.name)

    if tac_dfs:
        combined_tacs = pd.concat(tac_dfs, ignore_index=True)
        cols = [c for c in TAC_COLUMNS if c in combined_tacs.columns]
        combined_tacs[cols].to_csv(output_dir / "tacs.csv", index=False)
        log.info("Combined tacs.csv: %d rows, %d cases", len(combined_tacs), len(tac_dfs))
    else:
        log.warning("No per-case tacs.csv files found — combined tacs.csv not written")

    # Write studies.csv
    if study_metadata:
        rows = [
            {
                "case_id": m.case_id,
                "age": m.age,
                "sex": m.sex,
                "weight_kg": m.weight_kg,
                "height_cm": m.height_cm,
                "bmi": m.bmi,
                "injected_dose_mbq": m.injected_dose_mbq,
            }
            for m in study_metadata
        ]
        pd.DataFrame(rows)[STUDY_COLUMNS].to_csv(output_dir / "studies.csv", index=False)
        log.info("Wrote studies.csv: %d cases", len(rows))
