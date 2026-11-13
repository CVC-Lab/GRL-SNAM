"""The held-out per-term losses of every arm (the training objective's terms, per test pair), with
the same arms, test pairs and paired bootstrap as ``sdf_arms.py``. The histories hold these terms
for the trained runs only; this puts the untrained, constant and published arms beside them.

Per pair: ``goal`` (squared end distance after the training horizon), ``col`` (the clearance
softplus), ``reg`` (the pull toward the bias regime) and ``loss = goal + 3 col + 0.1 reg``.

    python experiments/learning_proof/objective_arms.py --sdf nav_sdf.npz --runs runs/retrain \\
        --published coef_sdf.pt --constants arms.json --out objective.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import grl_snam.sdf_nav as sdf_nav
from grl_snam.tools import sdf_eval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdf_arms import arms_from_runs, constant_coef_mlp  # noqa: E402
from stats import level, paired  # noqa: E402

TERMS = ("loss", "goal", "col", "reg")


@torch.no_grad()
def per_pair(model, scene, s, g, meta):
    al, be, ga = model(sdf_nav.coef_feats(scene.field, s, g))
    oT, _, clr = sdf_nav.sdf_rollout(
        scene.field, s, torch.zeros_like(s), g, al, be, ga, int(meta["horizon"]), nsub=1,
        rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"],
    )  # fmt: skip
    goal = ((oT - g) ** 2).sum(-1)
    col = F.softplus((0.02 - clr) / 0.02)
    reg = (be - 3.0) ** 2 + (ga - 4.0) ** 2 + (al - 1.0) ** 2
    return {"goal": goal, "col": col, "reg": reg, "loss": goal + 3.0 * col + 0.1 * reg}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument(
        "--constants", required=True, help="sdf_arms.py's arms.json (the constant arms)"
    )
    ap.add_argument("--test-pairs", type=int, default=1024)
    ap.add_argument("--test-seed", type=int, default=777)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    arms, seeds, picks, header = arms_from_runs(args.runs, args.published)
    c = json.load(open(args.constants))
    arms["constant"] = [constant_coef_mlp(*c["constant"])]
    arms["constant_trained"] = [constant_coef_mlp(*m) for m in c["constant_trained"]]
    scene = sdf_eval.load_scene(args.sdf)
    meta = torch.load(args.published, map_location="cpu", weights_only=False)["meta"]
    pool = sdf_eval.free_pool(scene, meta["rr"])
    _, held = sdf_eval.split_pool(
        pool, scene.region_n, header["val_blocks"], header["heldout_blocks"]
    )
    s, g = sdf_eval.sample_pairs(scene, held, args.test_pairs, args.test_seed, rr=meta["rr"])
    units = {a: [per_pair(m, scene, s, g, meta) for m in ms] for a, ms in arms.items()}

    def stack(a, t):
        return np.stack([u[t].numpy().astype(np.float64) for u in units[a]])

    out = {"seeds": seeds, "picks": picks, "levels": {}, "diffs": {}}
    pairs = [("trained", b) for b in ("untrained", "constant", "published", "constant_trained")]
    pairs += [("final", "untrained"), ("published", "untrained")]
    for t in TERMS:
        out["levels"][t] = {a: level(stack(a, t)) for a in units}
        out["diffs"][t] = {f"{a}-{b}": paired(stack(a, t), stack(b, t)) for a, b in pairs}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for t in TERMS:
        lv = "  ".join(f"{a} {r['mean']:.3f}" for a, r in out["levels"][t].items())
        print(f"{t}: {lv}")
        for k, r in out["diffs"][t].items():
            print(
                f"   {k}: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{'*' if r['excludes_zero'] else ''}"
            )


if __name__ == "__main__":
    main()
