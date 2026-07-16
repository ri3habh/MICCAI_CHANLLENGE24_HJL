# AortaSeg24 Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, Docker-packaged harness that runs the frozen AortaSeg24 ensemble on 100 labeled CTAs and reports per-vessel Dice / HD95 / volume vs. ground truth, so the team can quantify the Toralis "auto segmentation" gap.

**Architecture:** A small `validation/` Python package with five single-responsibility modules (pairing, labelmap, metrics, inference, report) orchestrated by one CLI (`run_validation.py`). Inference reuses the *unmodified* ensemble from `run_inference.py`; predictions are cached to disk so the multi-hour batch is resumable. Metrics are computed in the **saved-file frame** (read prediction `.mha` and GT `.mha` with plain SimpleITK) to sidestep the model's internal axis-flip bookkeeping. The whole thing ships as a CUDA Docker image with weights baked in, driven by a `--preflight` check and a `HANDOFF.md` for a coworker running it on a company RTX 4090.

**Tech Stack:** Python 3.10, PyTorch + nnU-Netv2 (existing), SimpleITK, medpy (Dice/HD95), numpy, pandas — all already in `requirements.txt`. Docker with an NVIDIA CUDA base.

## Global Constraints

- **Do not modify** the model, weights, `new_inference_code.py`, or the ensemble logic. The harness only orchestrates and measures.
- **Reuse, don't reimplement** inference: import `_load_repo`, `run_ensemble` from `run_inference.py` and `write_array_as_image_file` from `new_inference_code.py`.
- **Resumable:** any case whose prediction already exists on disk is not re-inferred.
- **The 9 canonical vessels, in this exact order (label indices 1–9):** `abdominal_aorta`, `left_renal`, `right_renal`, `left_common_iliac`, `right_common_iliac`, `left_internal_iliac`, `right_internal_iliac`, `left_external_iliac`, `right_external_iliac`. Background = 0.
- **Metrics:** Dice, HD95 (mm, using voxel spacing), volume (mL) + volume diff (mL and %).
- **SimpleITK axis order:** `GetArrayFromImage` returns `[z, y, x]`; `GetSpacing()` returns `(x, y, z)`. Any voxel-spacing passed to medpy must be reversed to `(z, y, x)`.
- **Python package layout:** all new code under `validation/`; all tests under `tests/`; run tests with `pytest`.

---

### Task 1: Package scaffold + case pairing

**Files:**
- Create: `validation/__init__.py`
- Create: `validation/pairing.py`
- Create: `tests/__init__.py`
- Create: `tests/test_pairing.py`
- Modify: `requirements.in` (add `pytest`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Case: case_id: str; image_path: pathlib.Path; label_path: pathlib.Path`
  - `pair_cases(images_dir: Path, labels_dir: Path, image_glob: str = "*.mha", label_template: str = "{stem}.mha") -> list[Case]` — sorted by `case_id`. `{stem}` in `label_template` is the image filename with its extension removed (handles `.nii.gz` as a double extension). Raises `FileNotFoundError` listing every image whose derived label file is missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pairing.py
from pathlib import Path
import pytest
from validation.pairing import pair_cases, Case


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_pairs_matching_stems(tmp_path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    _touch(imgs / "subject002.mha")
    _touch(imgs / "subject001.mha")
    _touch(labs / "subject001.mha")
    _touch(labs / "subject002.mha")
    cases = pair_cases(imgs, labs)
    assert [c.case_id for c in cases] == ["subject001", "subject002"]
    assert cases[0].image_path == imgs / "subject001.mha"
    assert cases[0].label_path == labs / "subject001.mha"


def test_strips_double_extension(tmp_path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    _touch(imgs / "caseA.nii.gz")
    _touch(labs / "caseA.nii.gz")
    cases = pair_cases(imgs, labs, image_glob="*.nii.gz", label_template="{stem}.nii.gz")
    assert cases[0].case_id == "caseA"


def test_custom_label_template(tmp_path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    _touch(imgs / "subject001_CTA.mha")
    _touch(labs / "subject001_CTA_label.mha")
    cases = pair_cases(imgs, labs, label_template="{stem}_label.mha")
    assert cases[0].label_path == labs / "subject001_CTA_label.mha"


def test_missing_label_raises(tmp_path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    _touch(imgs / "subject001.mha")
    with pytest.raises(FileNotFoundError) as e:
        pair_cases(imgs, labs)
    assert "subject001" in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pairing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation.pairing'`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/__init__.py
# (empty — marks the package)
```

```python
# tests/__init__.py
# (empty — marks the test package)
```

```python
# validation/pairing.py
"""Pair each CT image file with its ground-truth label file."""
from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path


@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: Path
    label_path: Path


def _stem(filename: str) -> str:
    """Filename with extension removed. Treats .nii.gz as one extension."""
    if filename.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")]
    return Path(filename).stem


def pair_cases(
    images_dir: Path,
    labels_dir: Path,
    image_glob: str = "*.mha",
    label_template: str = "{stem}.mha",
) -> list[Case]:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    image_paths = sorted(Path(p) for p in glob(str(images_dir / image_glob)))

    cases: list[Case] = []
    missing: list[str] = []
    for img in image_paths:
        stem = _stem(img.name)
        label_path = labels_dir / label_template.format(stem=stem)
        if not label_path.is_file():
            missing.append(f"{img.name} -> expected {label_path}")
            continue
        cases.append(Case(case_id=stem, image_path=img, label_path=label_path))

    if missing:
        raise FileNotFoundError(
            "No label file found for the following image(s):\n  " + "\n  ".join(missing)
        )
    return sorted(cases, key=lambda c: c.case_id)
```

Add `pytest` to `requirements.in` (append on its own line):

```
pytest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pairing.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/__init__.py validation/pairing.py tests/__init__.py tests/test_pairing.py requirements.in
git commit -m "feat(validation): add case pairing"
```

---

### Task 2: Label harmonization

**Files:**
- Create: `validation/labelmap.py`
- Create: `tests/test_labelmap.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `VESSELS: list[str]` — the 9 canonical names in index order (index 0 unused/background). Concretely `VESSELS[0] == "abdominal_aorta"` maps to output label 1.
  - `remap_volume(vol: np.ndarray, mapping: dict[int, int]) -> np.ndarray` — returns an array where each source integer is replaced by `mapping.get(value, 0)`; values not in `mapping` become 0 (background).
  - `build_model_mapping(dataset_json: dict, aliases: dict[str, list[str]] = VESSEL_ALIASES) -> dict[int, int]` — reads `dataset_json["labels"]` (name→index, per nnU-Net), and returns `{model_index: canonical_index(1..9)}` by matching each model class name against the alias lists (case-insensitive, ignoring spaces/underscores). Multiple model classes may map to one vessel (e.g. abdominal aortic zones → `abdominal_aorta`).
  - `GT_CLASS_TO_CANONICAL: dict[int, int]` — the ground-truth integer→canonical(1..9) map. **This is a reviewed constant, verified against a real label file in Step 6 before any batch runs.**
  - `VESSEL_ALIASES: dict[str, list[str]]` — canonical name → known source names/synonyms.
  - `canonical_index(name: str) -> int` — 1-based index of a vessel name in `VESSELS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labelmap.py
import numpy as np
from validation.labelmap import (
    VESSELS, remap_volume, build_model_mapping, canonical_index, VESSEL_ALIASES,
)


def test_vessels_order_and_count():
    assert len(VESSELS) == 9
    assert VESSELS[0] == "abdominal_aorta"
    assert canonical_index("right_external_iliac") == 9


def test_remap_collapses_unknown_to_background():
    vol = np.array([[0, 1, 2], [3, 99, 5]], dtype=np.int16)
    mapping = {1: 1, 2: 2, 5: 4}  # 3 and 99 unmapped -> 0
    out = remap_volume(vol, mapping)
    assert out.tolist() == [[0, 1, 2], [0, 0, 4]]


def test_remap_many_to_one():
    vol = np.array([10, 11, 12, 0], dtype=np.int16)
    mapping = {10: 1, 11: 1, 12: 1}  # three model zones -> abdominal_aorta
    assert remap_volume(vol, mapping).tolist() == [1, 1, 1, 0]


def test_build_model_mapping_matches_names_case_insensitively():
    dataset_json = {
        "labels": {
            "background": 0,
            "Right Renal Artery": 7,
            "left_renal": 8,
            "Some Aortic Zone We Ignore": 3,
        }
    }
    aliases = {
        "right_renal": ["right renal artery", "right_renal"],
        "left_renal": ["left renal artery", "left_renal"],
    }
    mapping = build_model_mapping(dataset_json, aliases)
    assert mapping[7] == canonical_index("right_renal")   # 3
    assert mapping[8] == canonical_index("left_renal")    # 2
    assert 3 not in mapping  # unmatched model class excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labelmap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation.labelmap'`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/labelmap.py
"""Map model output classes and ground-truth labels into a shared 9-vessel space.

Canonical label space: 0 = background, 1..9 = VESSELS in order.
"""
from __future__ import annotations

import numpy as np

VESSELS: list[str] = [
    "abdominal_aorta",       # -> 1
    "left_renal",            # -> 2
    "right_renal",           # -> 3
    "left_common_iliac",     # -> 4
    "right_common_iliac",    # -> 5
    "left_internal_iliac",   # -> 6
    "right_internal_iliac",  # -> 7
    "left_external_iliac",   # -> 8
    "right_external_iliac",  # -> 9
]


def canonical_index(name: str) -> int:
    return VESSELS.index(name) + 1


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


# Known source names / synonyms per canonical vessel. Extend after inspecting the
# real dataset.json in Step 6. abdominal_aorta may match several aortic-zone classes.
VESSEL_ALIASES: dict[str, list[str]] = {
    "abdominal_aorta": ["abdominal aorta", "abdominal_aorta", "aorta abdominal"],
    "left_renal": ["left renal", "left renal artery", "left_renal"],
    "right_renal": ["right renal", "right renal artery", "right_renal"],
    "left_common_iliac": ["left common iliac", "left_common_iliac"],
    "right_common_iliac": ["right common iliac", "right_common_iliac"],
    "left_internal_iliac": ["left internal iliac", "left_internal_iliac"],
    "right_internal_iliac": ["right internal iliac", "right_internal_iliac"],
    "left_external_iliac": ["left external iliac", "left_external_iliac"],
    "right_external_iliac": ["right external iliac", "right_external_iliac"],
}

# Ground-truth integer -> canonical index. VERIFY against a real label file (Step 6);
# these values are the AortaSeg24 default guess and MUST be confirmed before the batch.
GT_CLASS_TO_CANONICAL: dict[int, int] = {}


def remap_volume(vol: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(vol)
    for src, dst in mapping.items():
        out[vol == src] = dst
    return out


def build_model_mapping(dataset_json: dict, aliases: dict[str, list[str]] = VESSEL_ALIASES) -> dict[int, int]:
    alias_lookup: dict[str, int] = {}
    for vessel, names in aliases.items():
        idx = canonical_index(vessel)
        for n in names:
            alias_lookup[_norm(n)] = idx

    mapping: dict[int, int] = {}
    for name, model_idx in dataset_json["labels"].items():
        if int(model_idx) == 0:
            continue
        canon = alias_lookup.get(_norm(name))
        if canon is not None:
            mapping[int(model_idx)] = canon
    return mapping
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_labelmap.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/labelmap.py tests/test_labelmap.py
git commit -m "feat(validation): add label harmonization to 9-vessel space"
```

- [ ] **Step 6: Verify mappings against the real weights + a real label (manual, blocks the batch)**

This resolves the spec's KEY UNKNOWN. Run against the downloaded weights and one real GT file:

```bash
python - <<'PY'
import json, sys
import SimpleITK as sitk
import numpy as np
from validation.labelmap import build_model_mapping, VESSELS

ds = json.load(open("resources/Dataset040_Aortaseg24/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/dataset.json"))
print("MODEL LABELS (name -> index):")
for name, idx in ds["labels"].items():
    print(f"  {idx:>3}  {name}")
print("\nRESOLVED model->canonical mapping:")
print(build_model_mapping(ds))
print("\nCanonical vessels:", list(enumerate(VESSELS, start=1)))

gt = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[1])) if len(sys.argv) > 1 else None
if gt is not None:
    print("\nGT unique integer labels:", sorted(np.unique(gt).tolist()))
PY
# optionally: append the path to one real label file as an argument to the command above
```

Confirm: (a) every one of the 9 vessels appears in the resolved model mapping — if any is missing, add the real class name to `VESSEL_ALIASES` and re-run; (b) fill `GT_CLASS_TO_CANONICAL` in `validation/labelmap.py` from the printed GT label integers and their meaning (from the dataset's label documentation). Commit the confirmed constants:

```bash
git add validation/labelmap.py
git commit -m "chore(validation): confirm model + GT label mappings against real files"
```

---

### Task 3: Per-vessel metrics

**Files:**
- Create: `validation/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `VESSELS`, `canonical_index` from `validation.labelmap`.
- Produces:
  - `score_vessel(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing_zyx: tuple[float, float, float]) -> dict` — returns keys `status, dice, hd95_mm, vol_gt_ml, vol_pred_ml, vol_diff_ml, vol_diff_pct`. `status` is one of `"scored" | "missed" | "false_positive" | "absent_both"`. Non-applicable numbers are `float("nan")`.
  - `score_case(pred_canon: np.ndarray, gt_canon: np.ndarray, spacing_zyx, case_id: str) -> list[dict]` — one row dict per vessel, each including `case_id` and `vessel`.

Empty-vessel rules (from spec): gt empty & pred empty → `absent_both` (dice NaN); gt empty & pred non-empty → `false_positive` (dice 0); gt non-empty & pred empty → `missed` (dice 0, hd95 NaN); both non-empty → `scored` (all computed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import math
import numpy as np
from validation.metrics import score_vessel, score_case
from validation.labelmap import VESSELS


SP = (1.0, 1.0, 1.0)


def test_perfect_overlap():
    m = np.zeros((10, 10, 10), dtype=np.uint8)
    m[2:5, 2:5, 2:5] = 1
    r = score_vessel(m, m, SP)
    assert r["status"] == "scored"
    assert r["dice"] == 1.0
    assert r["hd95_mm"] == 0.0
    assert r["vol_diff_ml"] == 0.0


def test_two_offset_single_voxels_hd95():
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10, 10), dtype=np.uint8)
    pred[5, 5, 5] = 1
    gt[5, 5, 8] = 1  # 3 voxels apart along last axis, spacing 1mm
    r = score_vessel(pred, gt, SP)
    assert r["status"] == "scored"
    assert math.isclose(r["hd95_mm"], 3.0, rel_tol=1e-6)


def test_volume_uses_spacing():
    m = np.zeros((4, 4, 4), dtype=np.uint8)
    m[0, 0, 0] = 1  # one voxel
    r = score_vessel(m, m, (2.0, 1.0, 1.0))  # voxel = 2 mm^3 = 0.002 mL
    assert math.isclose(r["vol_gt_ml"], 0.002, rel_tol=1e-9)


def test_missed_vessel():
    gt = np.zeros((4, 4, 4), dtype=np.uint8)
    gt[1, 1, 1] = 1
    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    r = score_vessel(pred, gt, SP)
    assert r["status"] == "missed"
    assert r["dice"] == 0.0
    assert math.isnan(r["hd95_mm"])


def test_false_positive_vessel():
    gt = np.zeros((4, 4, 4), dtype=np.uint8)
    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    pred[1, 1, 1] = 1
    r = score_vessel(pred, gt, SP)
    assert r["status"] == "false_positive"
    assert r["dice"] == 0.0


def test_absent_both():
    z = np.zeros((4, 4, 4), dtype=np.uint8)
    r = score_vessel(z, z, SP)
    assert r["status"] == "absent_both"
    assert math.isnan(r["dice"])


def test_score_case_one_row_per_vessel():
    pred = np.zeros((6, 6, 6), dtype=np.uint8)
    gt = np.zeros((6, 6, 6), dtype=np.uint8)
    pred[1, 1, 1] = 1  # canonical label 1 present in pred+gt
    gt[1, 1, 1] = 1
    rows = score_case(pred, gt, SP, "caseX")
    assert len(rows) == len(VESSELS)
    assert all(row["case_id"] == "caseX" for row in rows)
    first = next(r for r in rows if r["vessel"] == VESSELS[0])
    assert first["status"] == "scored"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/metrics.py
"""Per-vessel Dice / HD95 / volume metrics with explicit empty-vessel handling."""
from __future__ import annotations

import math

import numpy as np
from medpy.metric.binary import dc, hd95

from validation.labelmap import VESSELS, canonical_index

NAN = float("nan")


def _volume_ml(mask: np.ndarray, spacing_zyx) -> float:
    voxel_mm3 = float(spacing_zyx[0]) * float(spacing_zyx[1]) * float(spacing_zyx[2])
    return float(mask.sum()) * voxel_mm3 / 1000.0


def score_vessel(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing_zyx) -> dict:
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    pred_any, gt_any = bool(pred_bin.any()), bool(gt_bin.any())

    vol_gt = _volume_ml(gt_bin, spacing_zyx)
    vol_pred = _volume_ml(pred_bin, spacing_zyx)
    vol_diff = vol_pred - vol_gt
    vol_diff_pct = (vol_diff / vol_gt * 100.0) if vol_gt > 0 else NAN

    if not gt_any and not pred_any:
        return dict(status="absent_both", dice=NAN, hd95_mm=NAN,
                    vol_gt_ml=vol_gt, vol_pred_ml=vol_pred,
                    vol_diff_ml=vol_diff, vol_diff_pct=NAN)
    if gt_any and not pred_any:
        return dict(status="missed", dice=0.0, hd95_mm=NAN,
                    vol_gt_ml=vol_gt, vol_pred_ml=vol_pred,
                    vol_diff_ml=vol_diff, vol_diff_pct=vol_diff_pct)
    if pred_any and not gt_any:
        return dict(status="false_positive", dice=0.0, hd95_mm=NAN,
                    vol_gt_ml=vol_gt, vol_pred_ml=vol_pred,
                    vol_diff_ml=vol_diff, vol_diff_pct=NAN)

    dice = float(dc(pred_bin, gt_bin))
    hd = float(hd95(pred_bin, gt_bin, voxelspacing=spacing_zyx))
    return dict(status="scored", dice=dice, hd95_mm=hd,
                vol_gt_ml=vol_gt, vol_pred_ml=vol_pred,
                vol_diff_ml=vol_diff, vol_diff_pct=vol_diff_pct)


def score_case(pred_canon: np.ndarray, gt_canon: np.ndarray, spacing_zyx, case_id: str) -> list[dict]:
    rows: list[dict] = []
    for vessel in VESSELS:
        idx = canonical_index(vessel)
        row = score_vessel(pred_canon == idx, gt_canon == idx, spacing_zyx)
        row["case_id"] = case_id
        row["vessel"] = vessel
        rows.append(row)
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: 7 passed. (If `medpy` import errors locally, run inside the Docker image built in Task 6; medpy is in `requirements.txt`.)

- [ ] **Step 5: Commit**

```bash
git add validation/metrics.py tests/test_metrics.py
git commit -m "feat(validation): add per-vessel dice/hd95/volume metrics"
```

---

### Task 4: Cached inference wrapper

**Files:**
- Create: `validation/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `run_inference._load_repo`, `run_inference.run_ensemble`, `new_inference_code.load_image_file_as_array`, `new_inference_code.write_array_as_image_file`.
- Produces:
  - `load_case_array(image_path: Path) -> tuple` — same 6-tuple as `load_image_file_as_array` (`img, spacing, direction, origin, props, ori_axcode`) but for an explicit single file of any SimpleITK-readable type (`.mha`/`.nii.gz`/`.tiff`). Reproduces the model's exact preprocessing (identical flips).
  - `predict_case(case, resources: Path, predictions_dir: Path, device, ensemble_fn=None, loader=None, writer=None) -> Path` — returns the path to `predictions_dir/<case_id>/output.mha`. If that file already exists, returns it **without** running inference (resumability). `ensemble_fn`/`loader`/`writer` are injectable for testing; production defaults wire the real functions.

- [ ] **Step 1: Write the failing test** (caching + wiring, no GPU/weights needed)

```python
# tests/test_inference.py
from pathlib import Path
import numpy as np
from validation.inference import predict_case
from validation.pairing import Case


def _fake_loader(image_path):
    img = np.zeros((1, 4, 4, 4), dtype=np.float32)
    return img, (1.0, 1.0, 1.0), (1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 0), {}, ("R", "A", "S")


def test_predict_runs_then_caches(tmp_path):
    calls = {"n": 0}

    def fake_ensemble(img, props, resources, device):
        calls["n"] += 1
        return np.ones((4, 4, 4), dtype=np.uint8)

    written = {}

    def fake_writer(*, location, array, spacing, origin, direction, ori_axcode):
        location.mkdir(parents=True, exist_ok=True)
        (location / "output.mha").write_bytes(b"x")
        written["array"] = array

    case = Case("c1", tmp_path / "c1.mha", tmp_path / "c1_lab.mha")
    preds = tmp_path / "predictions"

    out1 = predict_case(case, tmp_path, preds, device="cpu",
                        ensemble_fn=fake_ensemble, loader=_fake_loader, writer=fake_writer)
    assert out1 == preds / "c1" / "output.mha"
    assert out1.is_file()
    assert calls["n"] == 1

    # second call: file exists -> no re-inference
    out2 = predict_case(case, tmp_path, preds, device="cpu",
                        ensemble_fn=fake_ensemble, loader=_fake_loader, writer=fake_writer)
    assert out2 == out1
    assert calls["n"] == 1  # unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation.inference'`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/inference.py
"""Cached, resumable single-case inference using the frozen ensemble (unmodified)."""
from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np


def load_case_array(image_path: Path):
    """Explicit-path twin of new_inference_code.load_image_file_as_array.

    Same preprocessing (SimpleITKIO read + the two axis flips) so the model sees
    exactly what it expects, but accepts any SimpleITK-readable single file.
    """
    import SimpleITK
    from monai import transforms
    from monai.data import ITKReader
    from nibabel.orientations import aff2axcodes
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    image_path = str(image_path)
    result = SimpleITK.ReadImage(image_path)
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    _, meta_data = transforms.LoadImage(reader=ITKReader())(image_path)
    img, props = SimpleITKIO().read_images([image_path])
    ori_axcode = aff2axcodes(meta_data["affine"])
    img = np.flip(img, axis=2)
    img = np.flip(img, axis=3)
    return img, spacing, direction, origin, props, ori_axcode


def predict_case(case, resources, predictions_dir, device,
                 ensemble_fn=None, loader=None, writer=None):
    predictions_dir = Path(predictions_dir)
    out_dir = predictions_dir / case.case_id
    out_file = out_dir / "output.mha"
    if out_file.is_file():
        return out_file  # resumable: already predicted

    if ensemble_fn is None:
        from run_inference import run_ensemble as ensemble_fn  # noqa
    if loader is None:
        loader = load_case_array
    if writer is None:
        from new_inference_code import write_array_as_image_file as writer  # noqa

    img, spacing, direction, origin, props, ori_axcode = loader(case.image_path)
    ret = ensemble_fn(img, props, Path(resources), device)
    del img
    writer(location=out_dir, array=ret, spacing=spacing, origin=origin,
           direction=direction, ori_axcode=ori_axcode)
    return out_file
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inference.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/inference.py tests/test_inference.py
git commit -m "feat(validation): add cached resumable inference wrapper"
```

---

### Task 5: Report generation

**Files:**
- Create: `validation/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: `VESSELS` from `validation.labelmap`; row dicts from `score_case`.
- Produces:
  - `write_reports(rows: list[dict], outdir: Path, weak_threshold: float = 0.80, worst_n: int = 5) -> None` — writes `results/per_case_vessel.csv`, `results/summary.csv`, `results/summary.md` under `outdir`.
  - `summarize(rows: list[dict]) -> pandas.DataFrame` — per-vessel aggregate over `status == "scored"` rows: `mean_dice, median_dice, iqr_dice, mean_hd95, median_hd95, n_scored, n_missed, n_false_positive, n_absent_both`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from pathlib import Path
from validation.report import summarize, write_reports
from validation.labelmap import VESSELS


def _row(case, vessel, status, dice=float("nan"), hd95=float("nan")):
    return dict(case_id=case, vessel=vessel, status=status, dice=dice, hd95_mm=hd95,
                vol_gt_ml=1.0, vol_pred_ml=1.0, vol_diff_ml=0.0, vol_diff_pct=0.0)


def test_summarize_counts_and_means():
    v = VESSELS[0]
    rows = [
        _row("c1", v, "scored", dice=0.8, hd95=2.0),
        _row("c2", v, "scored", dice=0.6, hd95=4.0),
        _row("c3", v, "missed", dice=0.0),
    ]
    df = summarize(rows).set_index("vessel")
    assert df.loc[v, "n_scored"] == 2
    assert df.loc[v, "n_missed"] == 1
    assert abs(df.loc[v, "mean_dice"] - 0.7) < 1e-9


def test_write_reports_creates_files(tmp_path):
    rows = [_row("c1", VESSELS[0], "scored", dice=0.9, hd95=1.0)]
    write_reports(rows, tmp_path)
    results = tmp_path / "results"
    assert (results / "per_case_vessel.csv").is_file()
    assert (results / "summary.csv").is_file()
    md = (results / "summary.md").read_text()
    assert VESSELS[0] in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation.report'`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/report.py
"""Aggregate per-vessel rows into CSVs and a human-readable summary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from validation.labelmap import VESSELS


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for vessel in VESSELS:
        v = df[df["vessel"] == vessel]
        scored = v[v["status"] == "scored"]
        dice = scored["dice"].astype(float)
        hd = scored["hd95_mm"].astype(float)
        out.append(dict(
            vessel=vessel,
            mean_dice=float(dice.mean()) if len(dice) else float("nan"),
            median_dice=float(dice.median()) if len(dice) else float("nan"),
            iqr_dice=float(dice.quantile(0.75) - dice.quantile(0.25)) if len(dice) else float("nan"),
            mean_hd95=float(hd.mean()) if len(hd) else float("nan"),
            median_hd95=float(hd.median()) if len(hd) else float("nan"),
            n_scored=int((v["status"] == "scored").sum()),
            n_missed=int((v["status"] == "missed").sum()),
            n_false_positive=int((v["status"] == "false_positive").sum()),
            n_absent_both=int((v["status"] == "absent_both").sum()),
        ))
    return pd.DataFrame(out)


def write_reports(rows: list[dict], outdir: Path, weak_threshold: float = 0.80, worst_n: int = 5) -> None:
    outdir = Path(outdir)
    results = outdir / "results"
    results.mkdir(parents=True, exist_ok=True)

    per_case = pd.DataFrame(rows)
    per_case.to_csv(results / "per_case_vessel.csv", index=False)

    summary = summarize(rows)
    summary.to_csv(results / "summary.csv", index=False)

    weak = summary[summary["mean_dice"] < weak_threshold]["vessel"].tolist()
    lines = ["# AortaSeg24 Validation Summary", ""]
    lines.append(summary.round(3).to_markdown(index=False))
    lines.append("")
    lines.append(f"## Weak vessels (mean Dice < {weak_threshold})")
    lines.append(", ".join(weak) if weak else "_none_")
    lines.append("")
    lines.append(f"## Worst {worst_n} scored cases per vessel (by Dice)")
    scored = per_case[per_case["status"] == "scored"]
    for vessel in VESSELS:
        v = scored[scored["vessel"] == vessel].sort_values("dice").head(worst_n)
        ids = ", ".join(f"{r.case_id} ({r.dice:.2f})" for r in v.itertuples())
        lines.append(f"- **{vessel}**: {ids or '_no scored cases_'}")
    (results / "summary.md").write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: 2 passed. (If `tabulate` for `to_markdown` is missing, add `tabulate` to `requirements.in` and re-run.)

- [ ] **Step 5: Commit**

```bash
git add validation/report.py tests/test_report.py requirements.in
git commit -m "feat(validation): add CSV + markdown reporting"
```

---

### Task 6: CLI orchestrator + preflight

**Files:**
- Create: `validation/run_validation.py`
- Create: `tests/test_run_validation.py`

**Interfaces:**
- Consumes: `pair_cases`, `predict_case`, label mapping (`build_model_mapping`, `GT_CLASS_TO_CANONICAL`, `remap_volume`), `score_case`, `write_reports`.
- Produces:
  - `preflight(args) -> int` — checks GPU, weights+dataset.json, pairing count, free disk; prints a report; returns 0 on success, non-zero on first failure.
  - `run(args) -> int` — the full batch: pair → load repo → per case predict (cached) → read pred+GT via SimpleITK → remap both → score_case → collect → `write_reports`. Per-case failures are logged to `results/failures.log` and skipped.
  - `main(argv=None) -> int` — argparse front door.
  - `_read_seg(path) -> tuple[np.ndarray, tuple]` — returns `(array_zyx, spacing_zyx)` via SimpleITK (spacing reversed from `GetSpacing`).

- [ ] **Step 1: Write the failing test** (argument parsing + preflight failure paths, no GPU)

```python
# tests/test_run_validation.py
from validation.run_validation import build_parser, preflight


def test_parser_requires_images_labels_outdir():
    p = build_parser()
    args = p.parse_args(["--images", "a", "--labels", "b", "--outdir", "c"])
    assert args.device == "cuda"
    assert args.weak_threshold == 0.80


def test_preflight_fails_on_missing_weights(tmp_path, capsys):
    p = build_parser()
    args = p.parse_args([
        "--images", str(tmp_path), "--labels", str(tmp_path),
        "--outdir", str(tmp_path), "--resources", str(tmp_path / "nope"),
        "--device", "cpu",
    ])
    rc = preflight(args)
    assert rc != 0
    assert "weights" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation.run_validation'`.

- [ ] **Step 3: Write minimal implementation**

```python
# validation/run_validation.py
"""CLI: validate the frozen AortaSeg24 ensemble against labeled CTAs.

Usage (inside the Docker container):
  python -m validation.run_validation --preflight --images /data/images --labels /data/labels --outdir /out
  python -m validation.run_validation           --images /data/images --labels /data/labels --outdir /out
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AortaSeg24 validation harness.")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--resources", default="resources")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--image-glob", default="*.mha")
    p.add_argument("--label-template", default="{stem}.mha")
    p.add_argument("--weak-threshold", type=float, default=0.80)
    p.add_argument("--limit", type=int, default=None, help="Only process the first N cases (smoke test).")
    p.add_argument("--preflight", action="store_true", help="Run checks and exit.")
    return p


def _resencl_dataset_json(resources: Path) -> Path:
    return (resources / "Dataset040_Aortaseg24"
            / "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres" / "dataset.json")


def _read_seg(path: Path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # [z, y, x]
    spacing_zyx = tuple(reversed(img.GetSpacing()))
    return arr, spacing_zyx


def preflight(args) -> int:
    from validation.pairing import pair_cases
    resources = Path(args.resources)

    print("== preflight ==")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("FAIL: --device cuda but torch.cuda.is_available() is False.")
            return 2
        print(f"OK: GPU {torch.cuda.get_device_name(0)}")

    ds = _resencl_dataset_json(resources)
    if not ds.is_file():
        print(f"FAIL: model weights/dataset.json not found at {ds}")
        return 3
    json.load(open(ds))
    print(f"OK: weights present ({ds})")

    try:
        cases = pair_cases(Path(args.images), Path(args.labels),
                           args.image_glob, args.label_template)
    except FileNotFoundError as e:
        print(f"FAIL: pairing: {e}")
        return 4
    print(f"OK: {len(cases)} image/label pairs found")

    free_gb = shutil.disk_usage(args.outdir if Path(args.outdir).exists() else ".").free / 1e9
    print(f"INFO: {free_gb:.1f} GB free at output location")
    print("preflight PASSED")
    return 0


def run(args) -> int:
    from validation.pairing import pair_cases
    from validation.inference import predict_case
    from validation.labelmap import (
        build_model_mapping, remap_volume, GT_CLASS_TO_CANONICAL,
    )
    from validation.metrics import score_case
    from validation.report import write_reports

    import torch
    from run_inference import _load_repo

    resources = Path(args.resources)
    outdir = Path(args.outdir)
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    failures_log = outdir / "results" / "failures.log"
    device = torch.device(args.device)

    cases = pair_cases(Path(args.images), Path(args.labels),
                       args.image_glob, args.label_template)
    if args.limit:
        cases = cases[: args.limit]

    _load_repo()
    model_mapping = build_model_mapping(json.load(open(_resencl_dataset_json(resources))))
    if not GT_CLASS_TO_CANONICAL:
        print("ERROR: GT_CLASS_TO_CANONICAL is empty — complete Task 2 Step 6 first.")
        return 5

    rows: list[dict] = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        try:
            pred_path = predict_case(case, resources, outdir / "predictions", device)
            pred_arr, sp_pred = _read_seg(pred_path)
            gt_arr, sp_gt = _read_seg(case.label_path)
            if pred_arr.shape != gt_arr.shape:
                raise ValueError(f"shape mismatch pred{pred_arr.shape} vs gt{gt_arr.shape}")
            pred_canon = remap_volume(pred_arr, model_mapping)
            gt_canon = remap_volume(gt_arr, GT_CLASS_TO_CANONICAL)
            rows.extend(score_case(pred_canon, gt_canon, sp_pred, case.case_id))
            elapsed = time.time() - t0
            eta = elapsed / i * (len(cases) - i)
            print(f"[{i}/{len(cases)}] {case.case_id} done "
                  f"({elapsed/i:.0f}s/case, ETA {eta/60:.0f} min)")
        except Exception as e:  # noqa: BLE001 — never abort the batch
            with open(failures_log, "a", encoding="utf-8") as fh:
                fh.write(f"{case.case_id}: {e}\n{traceback.format_exc()}\n")
            print(f"[{i}/{len(cases)}] {case.case_id} FAILED: {e} (logged, continuing)")

    write_reports(rows, outdir, weak_threshold=args.weak_threshold)
    print(f"Wrote results to {outdir / 'results'}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return preflight(args) if args.preflight else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_validation.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the whole unit suite**

Run: `pytest -v`
Expected: all tests from Tasks 1–6 pass.

- [ ] **Step 6: Commit**

```bash
git add validation/run_validation.py tests/test_run_validation.py
git commit -m "feat(validation): add CLI orchestrator with preflight"
```

---

### Task 7: Docker handoff bundle + HANDOFF.md

**Files:**
- Create: `Dockerfile.validation`
- Create: `HANDOFF.md`
- Create: `.dockerignore` (if absent)

**Interfaces:**
- Consumes: the whole `validation/` package, `run_inference.py`, `new_inference_code.py`, `nnunetv2/`, `resources/` (weights).
- Produces: a runnable GPU image whose default command is the validation CLI.

Note: the existing `Dockerfile` is CPU-only (its `FROM pytorch/pytorch` line is dead — immediately overridden by `FROM python:3.10-slim`) and copies the deleted `post_processing.py`. This task ships a **separate, corrected** `Dockerfile.validation` with a real CUDA base and does not touch the original.

- [ ] **Step 1: Write the corrected Dockerfile**

```dockerfile
# Dockerfile.validation — GPU image for the AortaSeg24 validation harness.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
WORKDIR /opt/app

COPY requirements.txt /opt/app/
RUN python -m pip install -U pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps nnunetv2 \
    && python -m pip install pytest tabulate

# Application code
COPY nnunetv2 /opt/app/nnunetv2
COPY new_inference_code.py run_inference.py /opt/app/
COPY validation /opt/app/validation
COPY tests /opt/app/tests

# Model weights baked in (downloaded once by the author; see HANDOFF.md).
COPY resources /opt/app/resources

ENTRYPOINT ["python", "-m", "validation.run_validation"]
```

- [ ] **Step 2: Build the image and run the unit tests inside it**

Run:
```bash
docker build -f Dockerfile.validation -t aortaseg-validate .
docker run --rm --entrypoint pytest aortaseg-validate -v
```
Expected: image builds; all unit tests pass inside the container. (Requires `resources/` populated with the downloaded weights before building — see HANDOFF.md Step 0.)

- [ ] **Step 3: Author-side preflight + smoke test inside the container**

Run (author's machine or the coworker's — needs GPU + one real case):
```bash
docker run --rm --gpus all \
  -v /path/to/images:/data/images -v /path/to/labels:/data/labels -v $(pwd)/out:/out \
  aortaseg-validate --preflight --images /data/images --labels /data/labels --outdir /out

docker run --rm --gpus all \
  -v /path/to/images:/data/images -v /path/to/labels:/data/labels -v $(pwd)/out:/out \
  aortaseg-validate --limit 2 --images /data/images --labels /data/labels --outdir /out
```
Expected: preflight prints PASSED; the 2-case run writes `out/results/summary.md`. Eyeball it before the full batch.

- [ ] **Step 4: Write HANDOFF.md**

```markdown
# Running the AortaSeg24 Validation (coworker guide)

You need: a machine with an NVIDIA GPU (e.g. RTX 4090), Docker, and the NVIDIA
Container Toolkit (so `--gpus all` works). You do **not** need Python, nnU-Net,
or any model download — the weights are already inside the image.

## 0. Load the image
You received `aortaseg-validate.tar.gz`. Load it:
```
docker load -i aortaseg-validate.tar.gz
```

## 1. Point at your data
You have the 100 CTAs and their label masks. Put (or mount) them as two folders:
- images: one CT per file (`.mha` or `.nii.gz`)
- labels: the matching ground-truth mask per case

If your filenames are e.g. `subject001_CTA.mha` / `subject001_CTA_label.mha`, add
`--image-glob "*_CTA.mha" --label-template "{stem}_label.mha"` to the commands below.

## 2. Preflight (takes seconds — do this first)
```
docker run --rm --gpus all \
  -v /ABS/PATH/images:/data/images \
  -v /ABS/PATH/labels:/data/labels \
  -v /ABS/PATH/out:/out \
  aortaseg-validate --preflight --images /data/images --labels /data/labels --outdir /out
```
It must print `preflight PASSED`. If not, fix what it reports (GPU not visible,
wrong folder, filename mismatch) before continuing.

## 3. Run the full batch (hours — it prints progress + ETA)
Same command, drop `--preflight`:
```
docker run --rm --gpus all \
  -v /ABS/PATH/images:/data/images \
  -v /ABS/PATH/labels:/data/labels \
  -v /ABS/PATH/out:/out \
  aortaseg-validate --images /data/images --labels /data/labels --outdir /out
```
Safe to stop and rerun: finished cases are cached and skipped. If the machine
reboots, just run the same command again.

## 4. Send back
Zip and return only the small `out/results/` folder:
`per_case_vessel.csv`, `summary.csv`, `summary.md`, `failures.log`.
(The big `out/predictions/` folder can stay on your machine.)
```

- [ ] **Step 5: Export the shippable bundle**

Run:
```bash
docker save aortaseg-validate | gzip > aortaseg-validate.tar.gz
```
Expected: a single `aortaseg-validate.tar.gz` to send to the coworker alongside `HANDOFF.md`.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile.validation HANDOFF.md .dockerignore
git commit -m "feat(validation): add GPU Docker bundle and coworker handoff guide"
```

---

## Self-Review

**Spec coverage:**
- Case pairing → Task 1. ✓
- Cached/resumable inference reusing existing ensemble → Task 4 + orchestrator Task 6. ✓
- Label harmonization (model + GT → 9 vessels), verified against real files → Task 2 (incl. Step 6). ✓
- Metrics Dice/HD95/volume + empty-vessel statuses → Task 3. ✓
- Report CSV + summary.md with weak-vessel flag + worst cases → Task 5. ✓
- Error handling: per-case try/except → failures.log, geometry (shape) check → Task 6 `run`. ✓
- Handoff: Docker (CUDA), bundled weights, one command, preflight, HANDOFF.md, resumability, return-just-results → Tasks 6 & 7. ✓
- Testing: unit tests per module + smoke `--limit 2` + author-side container dry-run → Tasks 1–6 tests, Task 7 Steps 2–3. ✓

**Notes / accepted scope:**
- Geometry check is shape-equality (spacing/origin tolerance check not implemented) — acceptable because GT and prediction share the source CT grid; a mismatch surfaces as a logged failure rather than silent bad metrics.
- The GT integer→vessel constant is intentionally empty until Task 2 Step 6 fills it; the orchestrator refuses to run (returns 5) until it is populated, so the batch cannot run against an unverified mapping.
- HD95 unit uses medpy's `voxelspacing` in `(z,y,x)` order, matching the SimpleITK array axis order — enforced by `_read_seg` reversing `GetSpacing()`.
