# MC-Dyn: Multi Center Dynamic PET/CT TAC Extraction

MC-Dyn is an open-source pipeline for automated, harmonized extraction of Time-Activity Curves (TACs) from dynamic whole-body PET/CT studies. It is designed to standardize the technical decisions that confound cross-study comparisons, and to produce simple, privacy-safe outputs that are straightforward to pool across sites.

## Features

- Accepts **DICOM** or **NIfTI + BIDS JSON sidecar** input
- Automated CT organ segmentation via [MOOSE](https://github.com/ENHANCE-PET/MOOSE) (120+ structures)
- Aortic input function extraction with 4 anatomical segments × 2 VOI modes (1 mL cylindrical, full segment)
- Outputs flat CSV files — inherently anonymous under GDPR (no direct identifiers)
- Assigns anonymized BIDS-like case IDs (`sub-001_ses-01`) from DICOM PatientID + StudyDate
- Checkpointed pipeline — resumes from the last completed stage by default

## Installation

```bash
pip install mc-dyn
```

Requires `dcm2niix` on PATH for DICOM input.

## Usage

```bash
# Process all cases
mc-dyn run /data/input --output /data/output

# With multiple Moose models (run each separately, no label merging)
mc-dyn run /data/input --output /data/output \
    --model clin_ct_organs \
    --model clin_ct_cardiac

# Force re-run all stages
mc-dyn run /data/input --output /data/output --overwrite

# Limit to specific cases
mc-dyn run /data/input --output /data/output --cases subject_01
```

## Input structure

Each case must have a `pet/` and `ct/` subdirectory (at any depth under the input root):

```
input_dir/
└── subject_01/
    ├── pet/    ← DICOM files, or .nii.gz + dcm2niix JSON sidecar
    └── ct/     ← DICOM files, or .nii.gz
```

## Outputs

```
output_dir/
├── tacs.csv          # All TACs: case_id, task, organ, frame timings, mean, std, volume_ml
├── studies.csv       # Subject metadata: age, sex, weight, height, BMI, injected dose
├── case_mapping.csv  # Anonymization map
└── cases/
    └── sub-001_ses-01/
        ├── tacs.csv
        ├── aorta_if/        # aorta_segments.nii.gz, aorta_vois_1ml.nii.gz
        └── ...
```

## Docker

```bash
docker build -t mc-dyn .
docker run --gpus all \
  -v /data/input:/input:ro \
  -v /data/output:/output \
  mc-dyn run /input --output /output \
    --model clin_ct_organs \
    --model clin_ct_cardiac
```
