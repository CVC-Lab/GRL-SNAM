"""``coef_eval --compare-mu`` with paired CIs: the same arms (coef-train's bicycle fine-tune on the
ice-band city, a blind 5-feature net and a mu-widened 6-feature one, ``--seeds`` training seeds),
scored per agent against the untrained seed net on ``--draws`` evaluation draws, so each
difference gets the two-level bootstrap of ``stats.paired``. ``coef_eval`` itself prints only
means and the raw per-seed penetration. Each trained net is also scored against a constant net at
its own mean coefficients on the same drive (``constant_trained``): did it learn to use what it
sees, or only move its mean?

    python experiments/learning_proof/coef_eval_paired.py --seeds 5 --out coef_eval_paired.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

import grl_snam.sdf_nav as sdf_nav
from grl_snam.tools import coef_eval, coef_train

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdf_arms import constant_coef_mlp  # noqa: E402
from stats import level, paired  # noqa: E402


@torch.no_grad()
def per_agent(model, field, meta, rand_on, friction, *, n, ticks, seed):
    """coef_eval.evaluate's drive, kept per agent: reached, penetration ticks, worst clearance."""
    rr, d_hat, dt, vmax = meta["rr"], meta["d_hat"], meta["dt"], meta["vmax"]
    kw = dict(coef_eval.VEH)
    wants_mu = getattr(model, "in_dim", 5) == 6
    rng = np.random.default_rng(seed)
    o = torch.from_numpy(rand_on(n, rng))
    goal = torch.from_numpy(rand_on(n, rng))
    th = torch.from_numpy(rng.uniform(-np.pi, np.pi, n).astype(np.float32))
    sp = torch.zeros(n)
    reached = torch.zeros(n, dtype=torch.bool)
    pen, worst = torch.zeros(n), torch.full((n,), 1e9)
    pen_at = coef_eval.pen_threshold(rr)
    coef = torch.zeros(3)
    for _ in range(ticks):
        feat = sdf_nav.coef_feats(field, o, goal, friction=friction if wants_mu else None)
        al, be, ga = model(feat)
        coef += torch.stack([al.mean(), be.mean(), ga.mean()]) / ticks
        o, th, sp, _ = sdf_nav.bicycle_rollout(
            field, o, th, sp, goal, al, be, ga, 1, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax,
            friction=friction, **kw,
        )  # fmt: skip
        c = coef_eval._clearance(field, o, th, kw)
        pen += (c < pen_at).float()
        worst = torch.minimum(worst, c)
        reached |= (goal - o).norm(dim=1) < 0.15
    return {
        "reach": reached.float().numpy(),
        "pen": pen.numpy(),
        "clear": worst.numpy(),
        "coef": coef,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--draws", type=int, default=3, help="eval draws (seeds 4242, 4243, ...)")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--ticks", type=int, default=90)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    field, meta, rand_on, friction = coef_eval.ice_band_world(args.grid, 0.18)
    torch.manual_seed(0)
    seed_net = sdf_nav.CoefMLP()
    arms = {
        "blind (5-feature)": lambda net: copy.deepcopy(net),
        "mu-ahead (6-feature)": lambda net: sdf_nav.widen_coef_mlp(net),
    }

    def score(model):
        runs = [
            per_agent(
                model, field, meta, rand_on, friction, n=args.n, ticks=args.ticks, seed=4242 + k
            )
            for k in range(args.draws)
        ]
        out = {m: np.concatenate([r[m] for r in runs]) for m in ("reach", "pen", "clear")}
        out["coef"] = torch.stack([r["coef"] for r in runs]).mean(0).tolist()
        return out

    base = score(seed_net)
    out = {"args": vars(args), "seed_net": {m: level(base[m]) for m in ("reach", "pen", "clear")}}
    out["seed_net"]["coef"] = base["coef"]
    out["arms"] = {}
    for label, factory in arms.items():
        got, const = [], []
        for s in range(args.seeds):
            torch.manual_seed(s)
            model = coef_train.train_bicycle(
                model=factory(seed_net), friction=friction, steps=args.steps, horizon=16, n=64,
                window=4, grid=args.grid, seed=s, w_coll=3.0,
            )  # fmt: skip
            got.append(score(model))
            const.append(score(constant_coef_mlp(*got[-1]["coef"])))
        rec = {"coef": [g["coef"] for g in got]}
        for m in ("reach", "pen", "clear"):
            a = np.stack([g[m] for g in got])
            c = np.stack([g[m] for g in const])
            rec[m] = {
                "level": level(a),
                "constant_trained": level(c),
                "minus_seed": paired(a, base[m]),
                "minus_constant_trained": paired(a, c),
            }
        out["arms"][label] = rec
        for m in ("reach", "pen", "clear"):
            r = rec[m]
            print(label, m, {k: (round(r[k]["mean"], 3), round(r[k]["lo"], 3), round(r[k]["hi"], 3))
                             for k in ("level", "constant_trained", "minus_seed", "minus_constant_trained")},
                  flush=True)  # fmt: skip
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
