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
