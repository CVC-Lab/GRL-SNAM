"""E1, planted dissipation: the stagewise episodes.

Two datasets on the SAME obstacle fields (episode i uses seed ``seed + i`` in both):

- ``planted``: the translational damping is gamma = 2.5 while the ring's centre is at x < 4.5 and
  6.0 from there to the goal (``spline_stagewise6`` ``gamma_o_regions``);
- ``control``: gamma = 4.0 everywhere, scored over the same two x-regions.

Tangential friction is off (``fric_mu = 0``), so the only velocity-dependent force on the centre is
the planted damping. Each episode runs the Case-1 setup of ``scripts/stagewise_dataset.py`` (fixed
start (0, 2.5) and goal (9, -1.2), ``sample_obstacles_case2_harder``), in its own directory: the
generator's ``main()`` writes every episode's checkpoints to one shared file. The checkpoints keep
everything ``train_coef_energy`` reads and drop the ring's sample points (``Pw``, ``X``).

    python experiments/learning_proof/e1_generate.py --out runs/e1 --episodes 40 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

M_O = 1.5
X_SPLIT = 4.5
DATASETS = {
    "planted": [(-1e9, X_SPLIT, 2.5), (X_SPLIT, 1e9, 6.0)],
    "control": [(-1e9, X_SPLIT, 4.0), (X_SPLIT, 1e9, 4.0)],
}


def planted_spec(regions):
    return {
        "M_o": M_O,
        "regions": [
            {"name": "lo", "x_lo": regions[0][0], "x_hi": regions[0][1], "gamma": regions[0][2]},
            {"name": "hi", "x_lo": regions[1][0], "x_hi": regions[1][1], "gamma": regions[1][2]},
        ],
    }


def _episode(job):
    name, root, i, seed, steps = job
    import torch

    torch.set_num_threads(1)
    import scripts.spline_stagewise6 as she
    import scripts.stagewise_dataset as sd

    cfg = sd.GenCfg()
    cfg.total_steps = steps
    sd.set_all_seeds(seed)
    C, R, W = sd.sample_obstacles_case2_harder(cfg)
    world = she.WorldObstacles(C, R, W, d_hat=cfg.d_hat)
    snap = os.path.join(root, "episodes", f"ep{i:03d}", "snapshot.png")
    params = {"fric_mu": 0.0, "gamma_o_regions": DATASETS[name]}
    ep = sd.run_episode(cfg, world, cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius, snap, params)
    ck = ep["logs"]["checkpoints_jsonl"]
    with open(ck) as f:
        rows = [json.loads(line) for line in f]
    with open(ck, "w") as f:
        for r in rows:
            r.pop("Pw", None)
            r.pop("X", None)
            f.write(json.dumps(r) + "\n")
    path = sd.save_episode(root, name, i, ep)
    return {"case": name, "idx": i, "seed": seed, "path": path, "success": ep["success"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    jobs = []
    for name in DATASETS:
        root = os.path.join(args.out, name)
        os.makedirs(os.path.join(root, "episodes"), exist_ok=True)
        with open(os.path.join(root, "planted.json"), "w") as f:
            json.dump(planted_spec(DATASETS[name]), f, indent=2)
        jobs += [(name, root, i, args.seed + i, args.steps) for i in range(args.episodes)]
    with Pool(args.workers) as pool:
        recs = pool.map(_episode, jobs)
    for name in DATASETS:
        root = os.path.join(args.out, name)
        mine = sorted((r for r in recs if r["case"] == name), key=lambda r: r["idx"])
        with open(os.path.join(root, "manifest.json"), "w") as f:
            json.dump(mine, f, indent=2)
        ok = sum(r["success"] for r in mine)
        print(f"{name}: {len(mine)} episodes, {ok} reached the goal -> {root}")


if __name__ == "__main__":
    main()
