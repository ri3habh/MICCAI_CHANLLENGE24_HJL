# AortaSeg24 Validation Harness — Design

**Date:** 2026-07-15
**Status:** Approved (design), pending spec review
**Owner:** Rishabh Sharma

## Context

This repository is the HJL team's solution to the MICCAI 2024 **AortaSeg24** challenge: an
inference-only nnU-Net ensemble that segments the aorta and ~23 of its branches from CT
angiography. `run_inference.py` already wraps the frozen weights into a runnable CLI (single
model or full 3-model ensemble).

In the larger **Toralis** endovascular-planning pipeline (see `Toralis MVP.png`), the very first
step — *auto segmentation of Raw DICOM into vessel masks* — is marked **"BIGGEST GAP!"**. This
model is the candidate that fills that gap. Toralis needs **9 vessel masks**:

- abdominal_aorta
- left_renal, right_renal
- left_common_iliac, right_common_iliac
- left_internal_iliac, right_internal_iliac
- left_external_iliac, right_external_iliac

These 9 are a **subset** of the model's ~23 output classes, so coverage is not in question —
per-vessel **accuracy on Toralis's own data** is. We have **100 CTAs with ground-truth labels**
(stored as `.mha`/`.nii.gz`, which the model reads directly) and access to a 24 GB RTX 4090.

## Goal

Produce a per-vessel, per-case accuracy report for the frozen full ensemble across all 100
labeled CTAs, so the team can decide: **ship as-is / fix specific vessels / fine-tune**. This
de-risks every downstream stage of the Toralis pipeline before any effort is spent on mesh,
centerline, or measurement work.

Success criterion: a committed CSV + summary that states, for each of the 9 vessels, the Dice /
HD95 / volume-difference distribution across the cohort, plus a ranked list of worst cases for
manual QA.

## Non-Goals (explicitly out of scope)

- Mesh generation, centerline processing, per-vessel measurements, the web UI, the LLM chatbot.
- Fine-tuning or retraining the model (a *possible follow-up*, decided by this harness's output).
- DICOM → `.mha` conversion (data is already in a format the model reads).
- Changing the model, weights, or the ensemble logic in any way.

## Configuration (decisions made during brainstorming)

- **Model:** full ensemble — `lowres → ResEnc-M(0,1,2) → ResEnc-L(4,all) → majority vote`, exactly
  as `run_inference.run_ensemble` does. Device: `cuda`.
- **Metrics:** Dice, HD95 (95th-percentile Hausdorff distance, in mm using voxel spacing), and
  volume difference (mL and %).

## Architecture

Pipeline over the cohort:

```
images + labels
     │
     ▼
[1 pair]  ── match each image to its GT label file (configurable naming rule)
     │
     ▼   (for each case, resumable)
[2 infer] ── full ensemble, prediction cached to disk; skip if already present
     │
     ▼
[3 remap] ── map prediction AND ground truth into shared {background + 9 vessels} space
     │
     ▼
[4 metric] ── Dice / HD95 / volume per vessel, with explicit empty-vessel handling
     │
     ▼
[5 report] ── append rows → aggregate → CSV + summary.md
```

Each numbered component is independently testable and communicates through files/plain data
structures.

### Component 1 — Case pairing

- **Input:** images dir, labels dir, a naming rule (e.g. `subject001_CTA.mha` ↔
  `subject001_label.mha`; exact rule confirmed against the real Drive folder during build).
- **Output:** a list of `(case_id, image_path, label_path)` triples.
- **Behavior:** fail loud if any image has no matching label (or vice versa); print the count of
  paired cases before starting. Depends on nothing but the filesystem.

### Component 2 — Cached inference runner

- **Input:** a paired case, the `--resources` weights folder, device.
- **Output:** a prediction volume written to `predictions/<case_id>.mha` (or `.nii.gz`).
- **Behavior:** reuses the existing ensemble path (`build_predictor` + the
  `run_ensemble` flow from `run_inference.py`) rather than reimplementing it. **Resumable:** if
  `predictions/<case_id>.*` already exists and is readable, skip inference for that case. Frees GPU
  memory between models/cases (`torch.cuda.empty_cache()`), as the existing code does.
- **Rationale:** 100 full-ensemble runs take hours; the harness must survive a crash/reboot and
  continue.

### Component 3 — Label harmonization

- **Input:** a prediction volume, a ground-truth volume, the model's `dataset.json`.
- **Output:** two integer arrays in a shared label space `{0=background, 1..9 = the 9 vessels}`.
- **Behavior:** read the model's class→name map from `dataset.json`; define an explicit mapping
  table from (a) model class indices and (b) ground-truth label indices onto the 9 canonical
  vessel names. Remap both volumes. Any model class not among the 9 collapses to background.
- **KEY UNKNOWN — verify against real files, do not assume:** the exact integer↔name correspondence
  for the model output *and* for the ground-truth labels. The mapping table is the crux of
  correctness; it is validated by inspecting real `dataset.json` + a real label file before the
  batch runs, and encoded as a single reviewed constant.

### Component 4 — Metrics

- **Input:** the two remapped arrays + voxel spacing.
- **Output:** for each of the 9 vessels: `dice`, `hd95_mm`, `vol_gt_ml`, `vol_pred_ml`,
  `vol_diff_ml`, `vol_diff_pct`, and a `status` field.
- **Empty-vessel handling (explicit):**
  - GT empty & pred empty → `status = absent_both`, Dice = N/A (excluded from means).
  - GT empty & pred non-empty → `status = false_positive`, Dice = 0.
  - GT non-empty & pred empty → `status = missed`, Dice = 0, HD95 = N/A.
  - both non-empty → `status = scored`, all metrics computed.
- HD95 uses physical spacing (mm). Volumes use voxel volume from spacing.

### Component 5 — Report

- **Output files:**
  - `results/per_case_vessel.csv` — one row per (case_id, vessel) with all metrics + status.
  - `results/summary.csv` — per vessel: mean/median/IQR of Dice & HD95 over `scored` cases, plus
    counts of each status.
  - `results/summary.md` — human-readable: the summary table, weak vessels flagged (e.g. mean
    Dice below a configurable threshold), and the N worst cases per vessel listed by case_id for
    manual QA in ITK-SNAP.

## Error handling

- Per-case try/except: an I/O error, shape/geometry mismatch, or inference failure logs the
  case_id and reason to `results/failures.log` and continues; it never aborts the batch.
- Geometry sanity check before metrics: prediction and GT must share shape and (within tolerance)
  spacing/origin; mismatches are recorded as failures, not silently metric'd.
- All prediction outputs and result files are written atomically enough to be safely resumed.

## Testing strategy

- **Unit tests** on small synthetic volumes (no GPU): Dice/HD95/volume correctness against
  hand-computed values; the remap table; every empty-vessel branch of Component 4.
- **Smoke test:** run the full harness end-to-end on **1–2 real cases** and eyeball the CSV before
  committing to the 100-case batch.
- The heavy 100-case run is an operation, not a test; it is launched only after the smoke test
  passes.

## Deliverables

1. The harness code (pairing, cached inference wrapper, remap, metrics, report).
2. The reviewed label-mapping constant (Component 3).
3. `results/` outputs from the full 100-case run.
4. A short written readout of what the numbers imply for the Toralis auto-seg gap.

## Open questions to resolve during implementation

- Exact filename convention pairing images ↔ labels in the Drive folder.
- The concrete integer↔vessel mapping for both model output and ground-truth labels.
- Weak-vessel Dice threshold for flagging in `summary.md` (default proposal: 0.80).
