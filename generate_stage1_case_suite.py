#!/usr/bin/env python3
"""Generate a reproducible Stage-1 case suite for offline policy training/evaluation.

This is a thin orchestrator around ``generate_stage1_ackermann_rollouts.py``.  It
creates one directory per case, with train/val/test tensors plus exact scene
banks.  The generated scene banks can be reused later to evaluate learned
policies on the identical scenarios.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

CASE_OVERRIDES: Dict[str, List[str]] = {
    "easy_open": ["--n-obs-min", "2", "--n-obs-max", "6", "--corridor-half-width", "1.20", "--mpc-p-reverse", "0.05"],
    "dense_clutter": ["--n-obs-min", "18", "--n-obs-max", "35", "--corridor-half-width", "0.75", "--mpc-p-reverse", "0.20", "--mpc-samples", "768"],
    "narrow_gate": ["--n-obs-min", "12", "--n-obs-max", "24", "--corridor-half-width", "0.50", "--mpc-safe-margin", "0.25", "--mpc-samples", "1024"],
    "s_turn": ["--n-obs-min", "10", "--n-obs-max", "22", "--corridor-half-width", "0.65", "--mpc-p-reverse", "0.18", "--mpc-samples", "768"],
    "front_trap": ["--n-obs-min", "8", "--n-obs-max", "18", "--corridor-half-width", "0.65", "--mpc-p-reverse", "0.45", "--v-reverse-max", "0.35", "--mpc-samples", "1024"],
    "parallel_parking": ["--n-obs-min", "8", "--n-obs-max", "16", "--corridor-half-width", "0.55", "--mpc-p-reverse", "0.55", "--v-reverse-max", "0.35", "--goal-tol", "0.28", "--max-steps", "220", "--mpc-samples", "1536"],
    "terminal_bay": ["--n-obs-min", "10", "--n-obs-max", "20", "--corridor-half-width", "0.55", "--mpc-safe-margin", "0.25", "--mpc-p-reverse", "0.30", "--mpc-samples", "1024"],
    "noisy_scan": ["--n-obs-min", "10", "--n-obs-max", "22", "--scan-noise", "0.05", "--scan-dropout", "0.08", "--corridor-half-width", "0.70", "--mpc-samples", "768"],
    "long_horizon_mixed": ["--n-obs-min", "14", "--n-obs-max", "28", "--corridor-half-width", "0.55", "--max-steps", "240", "--mpc-p-reverse", "0.35", "--mpc-samples", "1280"],
}

DEFAULT_CASES = [
    "easy_open",
    "dense_clutter",
    "narrow_gate",
    "s_turn",
    "front_trap",
    "parallel_parking",
    "terminal_bay",
    "noisy_scan",
    "long_horizon_mixed",
]


def main() -> None:
    ap = argparse.ArgumentParser("Generate a reproducible Stage-1 case suite")
    ap.add_argument("--root", type=str, default="data/stage1_case_suite")
    ap.add_argument("--cases", type=str, default=",".join(DEFAULT_CASES), help="Comma-separated case names")
    ap.add_argument("--train-episodes", type=int, default=200)
    ap.add_argument("--val-episodes", type=int, default=40)
    ap.add_argument("--test-episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--sample-stride", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--mpc-horizon", type=int, default=12)
    ap.add_argument("--mpc-samples", type=int, default=512)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args, passthrough = ap.parse_known_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    manifest = {
        "root": str(root),
        "cases": [],
        "train_episodes": int(args.train_episodes),
        "val_episodes": int(args.val_episodes),
        "test_episodes": int(args.test_episodes),
        "seed": int(args.seed),
        "note": "Each case directory contains train/val/test.pt and scenes_{split}.json for exact replay.",
    }

    for idx, case in enumerate(cases):
        case_root = root / case
        case_seed = int(args.seed + 1000 * idx)
        cmd = [
            args.python,
            "-m",
            "generate_stage1_ackermann_rollouts",
            "--root", str(case_root),
            "--case", case,
            "--train-episodes", str(args.train_episodes),
            "--val-episodes", str(args.val_episodes),
            "--test-episodes", str(args.test_episodes),
            "--seed", str(case_seed),
            "--horizon", str(args.horizon),
            "--sample-stride", str(args.sample_stride),
            "--max-steps", str(args.max_steps),
            "--mpc-horizon", str(args.mpc_horizon),
            "--mpc-samples", str(args.mpc_samples),
        ]
        cmd.extend(CASE_OVERRIDES.get(case, []))
        cmd.extend(passthrough)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        manifest["cases"].append({"case": case, "root": str(case_root), "seed": case_seed, "overrides": CASE_OVERRIDES.get(case, [])})

    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate a compact teacher-data report if possible.
    report_cmd = [args.python, "-m", "report_stage1_case_suite", "--root", str(root), "--out", str(root / "teacher_report")]
    print(" ".join(report_cmd), flush=True)
    if not args.dry_run:
        subprocess.run(report_cmd, check=True)


if __name__ == "__main__":
    main()
