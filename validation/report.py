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
