"""CLI: validate the frozen AortaSeg24 ensemble against labeled CTAs.

Usage (inside the Docker container):
  python -m validation.run_validation --preflight --images /data/images --labels /data/labels --outdir /out
  python -m validation.run_validation           --images /data/images --labels /data/labels --outdir /out
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AortaSeg24 validation harness.")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--resources", default="resources")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--image-glob", default="*.mha")
    p.add_argument("--label-template", default="{stem}.mha",
                   help="Label filename from '{stem}' (image name minus ext) or '{case_id}' "
                        "(stem minus --image-suffix).")
    p.add_argument("--image-suffix", default="",
                   help="Suffix stripped from the image stem to form {case_id}, "
                        "e.g. '_CTA' so subject001_CTA -> case_id subject001.")
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
    from validation.labelmap import build_model_mapping, unmapped_vessels
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
    with open(ds) as fh:
        dataset_json = json.load(fh)
    print(f"OK: weights present ({ds})")

    model_mapping = build_model_mapping(dataset_json)
    missing = unmapped_vessels(model_mapping)
    if missing:
        print("FAIL: model mapping is incomplete — missing canonical vessels: "
              + ", ".join(missing))
        print("      Extend VESSEL_ALIASES in validation/labelmap.py and complete "
              "the real-data verification.")
        return 6
    print("OK: model mapping covers all 9 canonical vessels")

    try:
        cases = pair_cases(Path(args.images), Path(args.labels),
                           args.image_glob, args.label_template, args.image_suffix)
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
    from validation.labelmap import (
        build_model_mapping, remap_volume, unmapped_vessels, GT_CLASS_TO_CANONICAL,
    )

    import torch

    resources = Path(args.resources)
    outdir = Path(args.outdir)
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    failures_log = outdir / "results" / "failures.log"
    device = torch.device(args.device)

    cases = pair_cases(Path(args.images), Path(args.labels),
                       args.image_glob, args.label_template, args.image_suffix)
    if args.limit:
        cases = cases[: args.limit]

    if not GT_CLASS_TO_CANONICAL:
        print("ERROR: GT_CLASS_TO_CANONICAL is empty — complete Task 2 Step 6 first.")
        return 5

    with open(_resencl_dataset_json(resources)) as fh:
        dataset_json = json.load(fh)
    model_mapping = build_model_mapping(dataset_json)
    missing = unmapped_vessels(model_mapping)
    if missing:
        print("ERROR: model mapping is incomplete — missing canonical vessels: "
              + ", ".join(missing))
        print("       Extend VESSEL_ALIASES in validation/labelmap.py and complete "
              "the real-data verification.")
        return 6

    from validation.inference import predict_case
    from validation.metrics import score_case
    from validation.report import write_reports
    from run_inference import _load_repo
    _load_repo()

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
