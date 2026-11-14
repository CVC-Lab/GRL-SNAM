"""Print an arms.json (sdf_arms.py) as Markdown tables: each arm's level, then the paired
differences, per scorer.

    python experiments/learning_proof/arms_table.py arms.json
"""

from __future__ import annotations

import json
import sys

ORDER = (
    "published",
    "trained",
    "untrained",
    "final",
    "constant",
    "constant_trained",
    "constant_final",
)
LABEL = {
    "reached": ("reach", 100.0, "%", 1),
    "pen_ticks": ("pen ticks", 1.0, "", 2),
    "path_m": ("path m (both arrived)", 1.0, "", 1),
    "min_clear_m": ("min clearance m", 1.0, "", 2),
    "time_to_goal_s": ("time to goal s (both arrived)", 1.0, "", 2),
    "gap_m": ("end gap m", 1.0, "", 1),
}


def fmt(r, scale, unit, nd):
    if r is None or r.get("mean") is None:
        return "n/a"
    star = "*" if r.get("excludes_zero") else ""
    return (
        f"{r['mean'] * scale:+.{nd}f} [{r['lo'] * scale:+.{nd}f}, {r['hi'] * scale:+.{nd}f}]{star}"
    )


def lvl(r, scale, nd):
    if r is None or r.get("mean") is None:
        return "n/a"
    return f"{r['mean'] * scale:.{nd}f} [{r['lo'] * scale:.{nd}f}, {r['hi'] * scale:.{nd}f}]"


def main(path):
    a = json.load(open(path))
    for scorer, res in a["scorers"].items():
        print(f"\n### {scorer}\n")
        arms = [x for x in ORDER if x in res["levels"]["reached"]]
        print("| metric | " + " | ".join(arms) + " |")
        print("|---|" + "---|" * len(arms))
        for m, (name, sc, unit, nd) in LABEL.items():
            row = [lvl(res["levels"][m][x], sc, nd) for x in arms]
            print(f"| {name}{' (' + unit + ')' if unit else ''} | " + " | ".join(row) + " |")
        comps = list(res["diffs"]["reached"])
        print("\n| metric | " + " | ".join(comps) + " |")
        print("|---|" + "---|" * len(comps))
        for m, (name, sc, unit, nd) in LABEL.items():
            row = [fmt(res["diffs"][m][c], sc, unit, nd) for c in comps]
            print(f"| {name}{' (' + unit + ' pts)' if unit else ''} | " + " | ".join(row) + " |")
        n = res["diffs"]["reached"]["trained-untrained"]["n_units"]
        print(
            f"\n{n} units; * = the 95% CI excludes 0. Per-seed trained-untrained reach: "
            + ", ".join(
                f"{x * 100:+.1f}" for x in res["diffs"]["reached"]["trained-untrained"]["per_seed"]
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
