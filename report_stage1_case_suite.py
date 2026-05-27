#!/usr/bin/env python3
"""Aggregate Stage-1 case-suite summaries into CSV and Markdown reports."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

METRIC_COLUMNS = [
    "case",
    "split",
    "episodes",
    "samples",
    "success_rate",
    "collision_rate",
    "safety_stop_rate",
    "mean_final_dist",
    "mean_min_clearance",
    "p05_min_clearance",
    "reverse_episode_rate",
    "mean_reverse_steps",
    "mean_stall_steps",
    "mean_mode_switches",
    "mean_final_heading_err",
]


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.4g}"
    return str(x)


def _load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        case = summary.get("case", summary_path.parent.name)
        splits = summary.get("splits", {})
        for split, metrics in splits.items():
            row = {"case": case, "split": split}
            row.update(metrics)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_COLUMNS})


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Case", "Split", "Ep", "Samples", "Succ ↑", "Coll ↓", "Stop ↓", "Final dist ↓",
        "Min clear ↑", "p05 clear ↑", "Rev ep", "Rev steps", "Stall ↓", "Mode sw", "Head err ↓",
    ]
    keys = METRIC_COLUMNS
    with open(path, "w") as f:
        f.write("# Stage-1 Case Suite Teacher/Data Report\n\n")
        f.write("This report summarizes the MPC-teacher rollouts used to create reproducible offline-training datasets.\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            vals = [_fmt(row.get(k, "")) for k in keys]
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n## Recommended use\n\n")
        f.write("Use train/val/test.pt for offline training and model selection. Use scenes_test.json to replay learned policies on exactly the same held-out scenarios.\n")


def main() -> None:
    ap = argparse.ArgumentParser("Report Stage-1 case suite metrics")
    ap.add_argument("--root", type=str, default="data/stage1_case_suite")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "teacher_report"
    rows = _load_rows(root)
    if not rows:
        raise RuntimeError(f"No summary.json files found under {root}")
    write_csv(out.with_suffix(".csv"), rows)
    write_markdown(out.with_suffix(".md"), rows)
    print(json.dumps({"rows": len(rows), "csv": str(out.with_suffix('.csv')), "md": str(out.with_suffix('.md'))}, indent=2))


if __name__ == "__main__":
    main()
