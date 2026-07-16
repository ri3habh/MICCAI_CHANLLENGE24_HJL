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
