"""Constant coefficients on Austin's seek drive: how far a net that ignores what it sees gets.

Every ``(alpha, beta, gamma)`` in the grid drives as a constant net (``sdf_arms.constant_coef_mlp``)
on fixed pairs, in the sim frame (``--metres-per-unit``, 20 by default, as ``grl-snam train``):

- ``--split train``: pairs started outside the run's held-out blocks (the training draw), drawn
  with ``--pair-seed``. A choice made on these (``grl-snam train --reg-centre``) has not seen the
  held-out or test pairs.
- ``--split test``: the test pairs of ``sdf_arms.py`` (inside the held-out blocks, ``--pair-seed``
  777 there).

The held-out blocks come from a run's history header (``--split-history``).

    python experiments/learning_proof/constant_sweep.py --sdf nav_sdf.npz \\
        --split-history runs/retrain/seed0/history.jsonl --split train --out sweep_train.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from grl_snam.learning_history import read_history
from grl_snam.tools import sdf_eval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdf_arms import constant_coef_mlp  # noqa: E402


def pairs(args, header):
    """The sweep's pairs, in the sim frame: ``(scene, starts, goals)``."""
    task = sdf_eval.load_scene(args.sdf)
    scene = sdf_eval.in_frame(task, sdf_eval.sim_scale(task.scale, args.metres_per_unit))
    blocks, held = header["val_blocks"], header["heldout_blocks"]
    rr = 3.0 * scene.scale
    if args.split == "test":
        s, g = sdf_eval.heldout_pairs(scene, blocks, held, args.pairs, args.pair_seed, rr=rr)
        return scene, s, g
    rt, k = 3.0 * task.scale, scene.task_k
    train, _ = sdf_eval.split_pool(sdf_eval.free_pool(task, rt), task.region_n, blocks, held)
    s, g = sdf_eval.sample_pairs(task, train, args.pairs, args.pair_seed, rr=rt)
    return scene, s * k, g * k


def drive(job):
    args, abg = job
    torch.set_num_threads(1)
    header = read_history(args.split_history)[0]
    scene, s, g = pairs(args, header)
    S = scene.scale
    d = sdf_eval.seek_drive(
        constant_coef_mlp(*abg), scene, s, g, rr=3.0 * S, d_hat=0.35, dt=0.06, vmax=0.9,
        nsub=3, ticks=args.ticks, reach_tol=args.reach_tol * scene.task_k,
    )  # fmt: skip
    out = sdf_eval.drive_summary(d, 0.06)
    out.update(alpha=abg[0], beta=abg[1], gamma=abg[2])
    return out


def floats(text):
    return [float(x) for x in text.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--split-history", required=True, help="a run's history: its held-out blocks")
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--pairs", type=int, default=1024)
    ap.add_argument("--pair-seed", type=int, default=4242)
    ap.add_argument("--metres-per-unit", type=float, default=sdf_eval.SIM_METRES_PER_UNIT)
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--reach-tol", type=float, default=0.05, help="in the npz's units")
    ap.add_argument("--alpha", type=floats, default=floats("1,2,4,8"))
    ap.add_argument("--beta", type=floats, default=floats("0.75,1.5,3"))
    ap.add_argument("--gamma", type=floats, default=floats("1,2,4"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    grid = list(itertools.product(args.alpha, args.beta, args.gamma))
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(drive, [(args, abg) for abg in grid]))
    best = max(rows, key=lambda r: (r["clean_reach"], -r["gap_m"]))
    for r in sorted(rows, key=lambda r: -r["clean_reach"]):
        print(
            f"alpha {r['alpha']:g} beta {r['beta']:g} gamma {r['gamma']:g}: clean reach "
            f"{100 * r['clean_reach']:.1f}% reach {100 * r['reach']:.1f}% clearance "
            f"{r['min_clear_m']:.2f} m"
        )
    out = {k: v for k, v in vars(args).items() if k != "out"}
    with open(args.out, "w") as f:
        json.dump({"args": out, "rows": rows, "best": best}, f, indent=1, default=float)
    print("best:", {k: best[k] for k in ("alpha", "beta", "gamma", "clean_reach")})
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    raise SystemExit(main())
