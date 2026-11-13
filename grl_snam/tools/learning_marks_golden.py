"""The learning-curve marks golden: synthetic curves and the marks :mod:`grl_snam.learning.curves`
gives them.

    python -m grl_snam.tools.learning_marks_golden --out tests/fixtures/learning_marks_golden.json
    python -m grl_snam.tools.learning_marks_golden --check tests/fixtures/learning_marks_golden.json

Another implementation of the marks (a C++ reader, a dashboard) checks itself against the file:
the curves are stored point by point, so it needs nothing from here. Each case lists ``expect``
(this implementation's best, elbow, plateau and divergence, as ``{index, x, y}`` or ``null``)
and ``truth`` (what the curve was built to show, e.g. the noise-free minimum of an overfitting
curve), and the file states the parameters and the tolerance: an index may differ by
``tolerance.index`` and a smoothed y by ``tolerance.y_abs``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from grl_snam.learning.curves import DEFAULT, marks

GOLDEN_SCHEMA = "grl_snam.learning_marks_golden/v1"
TOLERANCE = {"index": 1, "y_abs": 1e-9}


def _noise(seed: int, n: int, sd: float) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sd, n)


def cases() -> list[dict]:
    """The synthetic curves: ``{name, mode, about, x, y, truth}``, y rounded to 6 decimals."""
    out = []

    def add(name, mode, about, x, y, **truth):
        out.append(
            {
                "name": name,
                "mode": mode,
                "about": about,
                "x": [float(v) for v in x],
                "y": [round(float(v), 6) for v in y],
                "truth": truth,
            }
        )

    i = np.arange(60.0)
    add(
        "decay_to_floor",
        "min",
        "a held-out loss that falls and settles: an elbow early, a plateau, no divergence",
        10 * i,
        0.2 + 0.8 * np.exp(-i / 8) + _noise(1, 60, 0.01),
        settles_after_x=300.0,
        diverges=False,
    )
    clean = 0.3 + 0.7 * np.exp(-i / 6) + 0.012 * np.maximum(0, i - 25)
    add(
        "overfit_u",
        "min",
        "falls, then rises as the net overfits: best mid-way, divergence after it",
        10 * i,
        clean + _noise(2, 60, 0.008),
        best_x=float(10 * i[int(np.argmin(clean))]),
        diverges=True,
    )
    add(
        "flat_noise",
        "min",
        "a learner that learned nothing: noise about a constant, no elbow and no divergence",
        10 * i,
        1.0 + _noise(3, 60, 0.02),
        diverges=False,
    )
    add(
        "still_falling",
        "min",
        "a straight descent with noise, still learning: no elbow, no plateau, best at the end",
        10 * i,
        1.0 - 0.6 * i / 60 + _noise(4, 60, 0.01),
        best_x=590.0,
        diverges=False,
    )
    xs = np.cumsum(np.r_[0, np.full(19, 5.0), np.full(20, 25.0)])  # denser early, irregular
    add(
        "fitness_saturating",
        "max",
        "a held-out fitness (higher is better) that rises and saturates, on an irregular x",
        xs,
        0.8 * (1 - np.exp(-xs / 60)) + _noise(5, 40, 0.01),
        diverges=False,
    )
    y = 0.2 + 0.8 * np.exp(-i / 8) + _noise(6, 60, 0.01)
    y[40] += 1.0
    add(
        "spike_on_floor",
        "min",
        "one bad evaluation on the floor: the median smooth removes it, so no divergence",
        10 * i,
        y,
        spike_x=400.0,
        diverges=False,
    )
    j = np.arange(40.0)
    add(
        "step_drop",
        "min",
        "flat, a sudden drop over three evaluations, flat again: the elbow at the drop",
        j,
        np.where(j < 15, 1.0, np.where(j > 17, 0.3, 1.0 - 0.7 * (j - 14) / 4))
        + _noise(7, 40, 0.005),
        drop_x=(15.0, 18.0),
        diverges=False,
    )
    add(
        "three_points",
        "min",
        "too short for anything but best",
        [0.0, 1.0, 2.0],
        [1.0, 0.8, 0.9],
        diverges=False,
    )
    return out


def golden() -> dict:
    """The golden file's content."""
    out = []
    for c in cases():
        m = marks(c["x"], c["y"], c["mode"], DEFAULT).to_dict()
        c = dict(c)
        c["expect"] = {
            k: m[k] for k in ("best", "elbow", "plateau", "divergence", "iqr", "noise", "n")
        }
        out.append(c)
    return {
        "schema": GOLDEN_SCHEMA,
        "about": "Synthetic held-out curves and their marks (best, Kneedle elbow, plateau, "
        "divergence after a w-point running median). The definitions are in "
        "grl_snam/learning/curves.py; regenerate with python -m "
        "grl_snam.tools.learning_marks_golden --out <this file>.",
        "params": DEFAULT.to_dict(),
        "tolerance": TOLERANCE,
        "cases": out,
    }


def compare(have: dict, want: dict) -> list[str]:
    """Where ``have`` (a golden) disagrees with ``want`` beyond the tolerance."""
    tol = want.get("tolerance", TOLERANCE)
    errs = []
    if have.get("params") != want.get("params"):
        errs.append(f"params differ: {have.get('params')} vs {want.get('params')}")
    hw = {c["name"]: c for c in have["cases"]}
    for wc in want["cases"]:
        hc = hw.get(wc["name"])
        if hc is None:
            errs.append(f"{wc['name']}: missing")
            continue
        for k in ("best", "elbow", "plateau", "divergence"):
            a, b = hc["expect"][k], wc["expect"][k]
            if (a is None) != (b is None):
                errs.append(f"{wc['name']}.{k}: {a} vs {b}")
            elif a is not None and abs(a["index"] - b["index"]) > tol["index"]:
                errs.append(f"{wc['name']}.{k}: index {a['index']} vs {b['index']}")
            elif a is not None and a["index"] == b["index"] and abs(a["y"] - b["y"]) > tol["y_abs"]:
                errs.append(f"{wc['name']}.{k}: y {a['y']} vs {b['y']}")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="write the golden here")
    g.add_argument("--check", help="compare this golden with a fresh one; exit 1 on a difference")
    args = ap.parse_args(argv)
    fresh = golden()
    if args.out:
        Path(args.out).write_text(json.dumps(fresh, indent=1) + "\n")
        print(f"wrote {args.out} ({len(fresh['cases'])} cases)")
        return 0
    errs = compare(fresh, json.loads(Path(args.check).read_text()))
    for e in errs:
        print(e, file=sys.stderr)
    print(f"{args.check}: {'differs' if errs else 'matches'}")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
