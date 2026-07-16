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
