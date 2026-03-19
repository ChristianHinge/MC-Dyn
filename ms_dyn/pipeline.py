"""Batch orchestration and per-case pipeline runner."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ms_dyn.checkpoint import CheckpointManager
from ms_dyn.config import PipelineConfig
from ms_dyn.models import CasePaths, RawCase, StudyMetadata
from ms_dyn.stages import detect, convert, metadata, segment, resample, extract, export
from ms_dyn.stages.metadata import extract_frame_timing

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def _discover_raw_cases(input_dir: Path) -> list[RawCase]:
    """Recursively find all directories that contain both pet/ and ct/ subdirs."""
    raw: list[RawCase] = []
    for candidate in sorted(input_dir.rglob("*")):
        if candidate.is_dir():
            pet_dir = candidate / "pet"
            ct_dir = candidate / "ct"
            if pet_dir.is_dir() and ct_dir.is_dir():
                raw.append(RawCase(
                    original_path=candidate,
                    pet_dir=pet_dir,
                    ct_dir=ct_dir,
                ))
    return raw


def _assign_case_ids(
    raw_cases: list[RawCase],
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> tuple[list[CasePaths], pd.DataFrame]:
    """
    Assign anonymized BIDS-like IDs (sub-001_ses-01) to each case.

    Reads PatientID + StudyDate from DICOM/JSON sidecar.
    Returns (list[CasePaths], mapping_dataframe).
    """
    mapping_rows: list[dict] = []
    patient_counter: dict[str, int] = {}   # patient_id → sub number
    session_counter: dict[str, dict[str, int]] = {}  # patient_id → {study_date → ses number}

    cases: list[CasePaths] = []

    for raw in raw_cases:
        rel_path = str(raw.original_path.relative_to(input_dir))

        # Try to get PatientID and StudyDate
        patient_id: str | None = None
        study_date: str | None = None
        try:
            fmt = detect.detect_input_format(raw.pet_dir, raw.ct_dir)
            # Peek at metadata without a full CasePaths (use a placeholder)
            placeholder = CasePaths(
                case_id="__tmp__",
                original_path=raw.original_path,
                pet_dir=raw.pet_dir,
                ct_dir=raw.ct_dir,
                output_dir=output_dir / "cases" / "__tmp__",
            )
            # Copy JSON if NIfTI input (needed for _extract_from_json)
            if not fmt.is_dicom:
                from ms_dyn.stages.convert import _find_nifti, _find_json_sidecar
                src_nii = _find_nifti(raw.pet_dir)
                src_json = _find_json_sidecar(src_nii)
                if src_json:
                    placeholder.output_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(src_json, placeholder.pet_json)

            _, patient_id, study_date = metadata.extract_metadata(placeholder, fmt)
        except Exception as exc:
            log.warning("Could not read metadata from %s: %s", rel_path, exc)

        patient_id = patient_id or rel_path
        study_date = study_date or "unknown"

        # Assign sub number
        if patient_id not in patient_counter:
            patient_counter[patient_id] = len(patient_counter) + 1
        sub_num = patient_counter[patient_id]

        # Assign ses number
        if patient_id not in session_counter:
            session_counter[patient_id] = {}
        if study_date not in session_counter[patient_id]:
            session_counter[patient_id][study_date] = len(session_counter[patient_id]) + 1
        ses_num = session_counter[patient_id][study_date]

        case_id = f"sub-{sub_num:03d}_ses-{ses_num:02d}"
        case_output = output_dir / "cases" / case_id

        cases.append(CasePaths(
            case_id=case_id,
            original_path=raw.original_path,
            pet_dir=raw.pet_dir,
            ct_dir=raw.ct_dir,
            output_dir=case_output,
        ))
        mapping_rows.append({
            "case_id": case_id,
            "original_path": rel_path,
            "patient_id": patient_id,
            "study_date": study_date,
        })
        log.info("Assigned %s → %s", rel_path, case_id)

    mapping_df = pd.DataFrame(mapping_rows)
    return cases, mapping_df


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def run_case(paths: CasePaths, config: PipelineConfig) -> StudyMetadata | None:
    """
    Run all pipeline stages for a single case with checkpointing.

    Returns StudyMetadata on success, None on failure.
    Stages already completed are skipped unless config.overwrite is True.
    """
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(paths.state_file, paths.case_id)

    if config.overwrite:
        ckpt.reset()

    # ---- Stage results shared across stages ----
    fmt = None
    study_meta: StudyMetadata | None = None
    timing = None
    seg_nii_path = None
    label_map: dict[int, str] = {}
    aorta_label: int | None = None
    all_tacs = None

    def _run(stage_name: str, fn, *args, **kwargs):
        if ckpt.is_completed(stage_name):
            log.debug("[%s] Skipping completed stage: %s", paths.case_id, stage_name)
            return True
        try:
            result = fn(*args, **kwargs)
            ckpt.mark_completed(stage_name)
            return result
        except Exception as exc:
            ckpt.mark_failed(stage_name, str(exc))
            log.error("[%s] Stage %s failed: %s", paths.case_id, stage_name, exc, exc_info=True)
            return None

    # detect
    log.info("[%s] Stage: detect", paths.case_id)
    fmt = detect.detect_input_format(paths.pet_dir, paths.ct_dir)
    _run("detect", lambda: None)  # detect always re-runs (fast, stateless)
    ckpt.mark_completed("detect")

    # convert
    log.info("[%s] Stage: convert", paths.case_id)
    if not ckpt.is_completed("convert"):
        try:
            convert.convert_case(paths, fmt)
            ckpt.mark_completed("convert")
        except Exception as exc:
            ckpt.mark_failed("convert", str(exc))
            log.error("[%s] Stage convert failed: %s", paths.case_id, exc, exc_info=True)
            return None

    # metadata
    log.info("[%s] Stage: metadata", paths.case_id)
    if not ckpt.is_completed("metadata"):
        try:
            study_meta, _, _ = metadata.extract_metadata(paths, fmt)
            timing = extract_frame_timing(paths.pet_json)
            ckpt.mark_completed("metadata")
        except Exception as exc:
            ckpt.mark_failed("metadata", str(exc))
            log.error("[%s] Stage metadata failed: %s", paths.case_id, exc, exc_info=True)
            return None
    else:
        try:
            study_meta, _, _ = metadata.extract_metadata(paths, fmt)
            timing = extract_frame_timing(paths.pet_json)
        except Exception as exc:
            log.error("[%s] Cannot re-read metadata/timing: %s", paths.case_id, exc)
            return None

    # segment
    log.info("[%s] Stage: segment", paths.case_id)
    if not ckpt.is_completed("segment"):
        try:
            seg_nii_path, label_map, aorta_label = segment.run_segment(
                paths, config.moose_models, config.accelerator
            )
            ckpt.mark_completed("segment")
        except Exception as exc:
            ckpt.mark_failed("segment", str(exc))
            log.error("[%s] Stage segment failed: %s", paths.case_id, exc, exc_info=True)
            return None
    else:
        # Re-derive label_map from Moose output
        try:
            from ms_dyn.stages.segment import _find_moose_seg_nii, _find_moose_labels_json
            import json as _json
            labels_json = _find_moose_labels_json(paths.seg_dir)
            with open(labels_json) as f:
                raw_labels = _json.load(f)
            label_map = {}
            for k, v in raw_labels.items():
                if isinstance(v, int):
                    label_map[v] = k
                elif isinstance(k, str) and k.isdigit():
                    label_map[int(k)] = str(v)
            seg_nii_path = _find_moose_seg_nii(paths.seg_dir)
            aorta_label = segment.find_aorta_label(label_map)
        except Exception as exc:
            log.error("[%s] Cannot re-read segment outputs: %s", paths.case_id, exc)
            return None

    # resample
    log.info("[%s] Stage: resample", paths.case_id)
    if not ckpt.is_completed("resample"):
        try:
            resample.resample_seg_to_pet(seg_nii_path, paths.pet_nii, paths.seg_pet_nii)
            ckpt.mark_completed("resample")
        except Exception as exc:
            ckpt.mark_failed("resample", str(exc))
            log.error("[%s] Stage resample failed: %s", paths.case_id, exc, exc_info=True)
            return None

    # extract
    log.info("[%s] Stage: extract", paths.case_id)
    if not ckpt.is_completed("extract"):
        try:
            # Determine task name from the first Moose model
            task_name = f"moosez_{config.moose_models[0]}" if config.moose_models else "moosez"
            all_tacs = extract.extract_organ_tacs(
                pet_nii=paths.pet_nii,
                seg_pet_nii=paths.seg_pet_nii,
                timing=timing,
                label_map=label_map,
                case_id=paths.case_id,
                task=task_name,
                max_roi_size_factor=config.max_roi_size_factor,
            )
            # Also extract aorta input function if available
            if aorta_label is not None:
                try:
                    aorta_df = extract.extract_aorta_tac(paths, timing, aorta_label)
                    if not aorta_df.empty:
                        import pandas as _pd
                        all_tacs = _pd.concat([all_tacs, aorta_df], ignore_index=True)
                except Exception as exc:
                    log.warning("[%s] Aorta IF extraction failed (non-fatal): %s", paths.case_id, exc)

            ckpt.mark_completed("extract")
        except Exception as exc:
            ckpt.mark_failed("extract", str(exc))
            log.error("[%s] Stage extract failed: %s", paths.case_id, exc, exc_info=True)
            return None

    # export
    log.info("[%s] Stage: export", paths.case_id)
    if not ckpt.is_completed("export"):
        try:
            if all_tacs is None or all_tacs.empty:
                log.warning("[%s] No TAC data to export", paths.case_id)
            else:
                export.write_case_tacs(all_tacs, paths.tacs_csv)
            ckpt.mark_completed("export")
        except Exception as exc:
            ckpt.mark_failed("export", str(exc))
            log.error("[%s] Stage export failed: %s", paths.case_id, exc, exc_info=True)
            return None

    log.info("[%s] All stages complete", paths.case_id)
    return study_meta


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(config: PipelineConfig) -> None:
    """Discover all cases, assign IDs, and run the pipeline for each."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Discovering cases in %s", config.input_dir)
    raw_cases = _discover_raw_cases(config.input_dir)
    if not raw_cases:
        log.warning("No cases found (no directories with both pet/ and ct/ subdirs).")
        return

    log.info("Found %d candidate cases", len(raw_cases))

    # Filter if --cases specified
    if config.cases_filter:
        raw_cases = [
            r for r in raw_cases
            if str(r.original_path.relative_to(config.input_dir)) in config.cases_filter
        ]
        log.info("Filtered to %d cases", len(raw_cases))

    cases, mapping_df = _assign_case_ids(raw_cases, config.input_dir, config.output_dir, config)

    # Write mapping file
    mapping_path = config.output_dir / "case_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    log.info("Case mapping written to %s", mapping_path)

    # Process each case
    all_metadata: list[StudyMetadata] = []
    succeeded = 0
    failed = 0

    for paths in cases:
        log.info("=" * 60)
        log.info("Processing case: %s (%s)", paths.case_id, paths.original_path)
        meta = run_case(paths, config)
        if meta is not None:
            all_metadata.append(meta)
            succeeded += 1
        else:
            failed += 1

    log.info("=" * 60)
    log.info("Batch complete: %d succeeded, %d failed", succeeded, failed)

    # Aggregate outputs
    case_dirs = [p.output_dir for p in cases]
    export.aggregate_outputs(case_dirs, config.output_dir, all_metadata)
    log.info("Combined outputs written to %s", config.output_dir)
