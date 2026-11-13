"""The retrain's learning curves, from its ``history.jsonl`` files (``grl-snam train --history``).

Per checkpoint: the training terms (their mean over the steps since the last checkpoint), the same
terms on the held-out pairs, the held-out seek drive (reach, end gap, clearance), each head's mean
and spread on the probe batch, the trunk's ``||dtheta|| / ||theta||`` since the start and the
gradient norm. Prints Markdown tables (every ``--every`` steps), writes the numbers as JSON and,
with ``--png``, one figure.

    python experiments/learning_proof/learning_curves.py runs/retrain/seed*/history.jsonl \\
        --json curves.json --png curves.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

TERMS = ("loss", "goal", "col", "reg")


def curve(path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    steps = [r for r in recs if r["kind"] == "step"]
    cks = [r for r in recs if r["kind"] == "checkpoint"]
    end = next(r for r in recs if r["kind"] == "end")
    out = []
    prev = 0
    for c in cks:
        window = [s for s in steps if prev < s["step"] <= c["step"]]
        prev = c["step"]
        row = {"step": c["step"], "best": c["best"]}
        for t in TERMS:
            row[f"train_{t}"] = float(np.mean([s["terms"][t] for s in window])) if window else None
            row[f"heldout_{t}"] = c["heldout"][f"obj_{t}"]
        for k in ("reach", "gap_m", "min_clear_m", "pen_ticks", "time_to_goal_s"):
            row[k] = c["heldout"][k]
        for h in ("alpha", "beta", "gamma"):
            row[f"{h}_mean"] = c["heads"][h]["mean"]
            row[f"{h}_spread"] = c["heads"][h]["spread"]
            row[f"{h}_dtheta_init"] = c["heads"][h]["dtheta_rel_init"]
        row["trunk_dtheta_init"] = c["heads"]["trunk"]["dtheta_rel_init"]
        row["grad_norm"] = (
            float(np.mean([s["grad_norm"]["mean"] for s in window])) if window else None
        )
        row["clipped_frac"] = (
            float(np.mean([s["grad_norm"]["clipped_frac"] for s in window])) if window else None
        )
        out.append(row)
    return {
        "path": path,
        "best_step": end["best_step"],
        "best_fitness": end["best_fitness"],
        "rows": out,
    }


def table(c, every):
    cols = [
        ("step", "{:d}"), ("train_goal", "{:.3f}"), ("heldout_goal", "{:.3f}"),
        ("train_col", "{:.3f}"), ("heldout_col", "{:.3f}"), ("reach", "{:.3f}"),
        ("gap_m", "{:.0f}"), ("min_clear_m", "{:.1f}"), ("alpha_mean", "{:.2f}"),
        ("alpha_spread", "{:.3f}"), ("beta_mean", "{:.2f}"), ("gamma_mean", "{:.2f}"),
        ("trunk_dtheta_init", "{:.3f}"), ("grad_norm", "{:.2f}"),
    ]  # fmt: skip
    lines = ["| " + " | ".join(k for k, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in c["rows"]:
        if r["step"] % every and not r["best"] and r is not c["rows"][-1]:
            continue
        cells = []
        for k, f in cols:
            v = r[k]
            cells.append("—" if v is None else f.format(v))
        if r["best"] and r["step"] == c["best_step"]:
            cells[0] += " ★"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(curves, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("goal term", ("train_goal", "heldout_goal")),
        ("collision term", ("train_col", "heldout_col")),
        ("held-out reach (seek drive)", ("reach",)),
        ("held-out end gap, m", ("gap_m",)),
        ("alpha (mean; band = spread on the probe)", ("alpha_mean",)),
        ("trunk ||dtheta||/||theta|| since step 0", ("trunk_dtheta_init",)),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for a, (title, keys) in zip(ax.flat, panels, strict=True):
        for i, c in enumerate(curves):
            x = [r["step"] for r in c["rows"]]
            for k in keys:
                y = np.array([np.nan if r[k] is None else r[k] for r in c["rows"]], float)
                ls = "--" if k.startswith("train") else "-"
                lab = (
                    f"seed {i}" + (" train" if k.startswith("train") else "")
                    if len(keys) > 1
                    else f"seed {i}"
                )
                a.plot(x, y, ls, color=colors[i % len(colors)], lw=1.2, label=lab)
                if k == "alpha_mean":
                    s = np.array([r["alpha_spread"] for r in c["rows"]])
                    a.fill_between(x, y - s, y + s, color=colors[i % len(colors)], alpha=0.15)
            a.axvline(c["best_step"], color=colors[i % len(colors)], lw=0.6, ls=":")
        a.set_title(title, fontsize=10)
        a.set_xlabel("step")
    ax.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "grl-snam train on austin_south: training (dashed) and held-out (solid); dotted = held-out pick"
    )
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("histories", nargs="+")
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--json", default="")
    ap.add_argument("--png", default="")
    args = ap.parse_args(argv)
    curves = [curve(p) for p in args.histories]
    for p, c in zip(args.histories, curves, strict=True):
        print(
            f"\n#### {os.path.basename(os.path.dirname(p)) or p} (pick: step {c['best_step']}, held-out reach {c['best_fitness']:.3f})\n"
        )
        print(table(c, args.every))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(curves, f, indent=1)
    if args.png:
        figure(curves, args.png)


if __name__ == "__main__":
    main()
