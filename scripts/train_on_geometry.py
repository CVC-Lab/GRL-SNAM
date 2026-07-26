"""Self-supervised CoefEnergyNet training on a real geometry's obstacle set.

Trains the learned navigation coefficients directly on a scene — NO expert labels
and no pre-generated dataset. The GRL-SNAM surrogate rollout
(``surrogate_robust.integrate_surrogate_v2``) is fully differentiable, so we
backprop a reach-goal + no-penetration + speed-cap loss straight through an
H-step rollout and let ``CoefEnergyNet`` learn to predict per-obstacle repulsion
(``alphas``), goal pull (``beta``) and damping (``gamma``) that navigate the real
obstacles. The result is a checkpoint the live demo drives, optionally refined
online with ``HistSecantController`` at inference.

Input is the ``obstacles.npz`` produced by ``extract_obstacles.py`` (obstacle
circles + free-point pool, in world units). This step needs only ``torch`` +
``numpy`` + the GRL-SNAM repo on ``PYTHONPATH`` — no graphics/VTK — so it runs on
a plain CPU box or a GPU cluster.

SCALE: the surrogate's coefficients are tuned for a ~10-unit world. A large scene
(e.g. a 3 km city) is normalized per WORKING REGION: pick a region half-extent
around a center, map that region to ~``TARGET_EXTENT`` units for the rollout, and
map back for rendering. The checkpoint records ``scale`` + ``center`` so the demo
converts world<->normalized consistently.

Usage:
    python scripts/train_on_geometry.py obstacles.npz -o coef_energy.pt \\
        --steps 5000 --region 430
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

# eval_coef_energy pulls in imageio + scripts.* purely at import time; stub them so
# build_local_feats imports standalone (no dataset generators needed for training).
for _m in ["imageio", "imageio.v3", "scripts.ring_dataset_maxmin", "scripts.spline_stagewise6"]:
    sys.modules.setdefault(_m, types.ModuleType(_m))

from eval_coef_energy import build_local_feats  # noqa: E402
from surrogate_robust import integrate_surrogate_v2  # noqa: E402
from train_coef_energy import CoefEnergyNet  # noqa: E402

TARGET_EXTENT = 10.0  # normalize a working region to ~10 units (the tuned regime)


def _nearby(centers_n, p, win=3.0, k=32):
    """The <=k closest obstacles to normalized point p within a window."""
    d2 = ((centers_n - p) ** 2).sum(1)
    idx = np.argsort(d2)[:k]
    return centers_n[idx[d2[idx] < win * win]]


def make_batch(pool, centers_n, radn, rr, bs=64):
    """A batch of LOCAL navigation problems in normalized space.

    Each sample starts at a random free point and aims at a nearby goal (1.5-3.0
    units away, random direction). Local goals keep the rollout at a realistic
    speed and teach obstacle-avoiding local progress that CHAINS into a full route
    at inference — training on far goals instead rewards rushing straight across
    the map (and clipping buildings)."""
    si = np.random.randint(0, len(pool), bs)
    o0 = pool[si]
    ang = np.random.uniform(0, 2 * np.pi, bs).astype(np.float32)
    dist = np.random.uniform(1.0, 2.0, bs).astype(np.float32)  # reachable within the horizon
    goal = (o0 + np.stack([np.cos(ang), np.sin(ang)], 1) * dist[:, None]).astype(np.float32)

    nbs = [_nearby(centers_n, o0[i]) for i in range(bs)]
    maxN = max(1, max(len(n) for n in nbs))
    C = np.zeros((bs, maxN, 2), np.float32)
    R = np.zeros((bs, maxN), np.float32)
    ofs, gfs = [], []
    for i, nb in enumerate(nbs):
        C[i, : len(nb)] = nb
        R[i, : len(nb)] = radn
        of, gf = build_local_feats(
            o0[i], goal[i], nb, np.full(len(nb), radn, np.float32), np.ones(len(nb), np.float32)
        )
        ofs.append(torch.cat([of, torch.zeros(1, maxN - of.shape[1], 6)], 1))
        gfs.append(gf)
    Ct = torch.from_numpy(C)
    mask = Ct.abs().sum(-1) > 0
    return (torch.from_numpy(o0), torch.zeros(bs, 2), torch.from_numpy(goal), Ct,
            torch.from_numpy(R), mask, torch.cat(ofs, 0), torch.cat(gfs, 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("obstacles", help="obstacles.npz from extract_obstacles.py")
    ap.add_argument("-o", "--out", default="coef_energy.pt", help="output checkpoint")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--region", type=float, default=430.0,
                    help="working-region half-extent (world units) around the scene center")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=28, help="rollout steps per train sample")
    ap.add_argument("--dt", type=float, default=0.06)
    ap.add_argument("--d-hat-world", type=float, default=25.0,
                    help="IPC barrier reach in WORLD units (~one street width). Kept LOCAL: a "
                         "large reach makes every point sit inside many overlapping barriers "
                         "(a 'sea of repulsion') that stalls the agent in a dense scene.")
    ap.add_argument("--w-reg", type=float, default=0.3,
                    help="weight anchoring coefficients to the known-good navigating regime")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.obstacles)
    centers = d["centers"].astype(np.float32)
    pool_w = d["free_pool"].astype(np.float32)
    mnx, mny, mxx, mxy = [float(v) for v in d["bounds"]]
    cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    S = TARGET_EXTENT / (2.0 * args.region)  # world -> normalized
    radn = float(d["radius_world"]) * S
    rr = float(d["robot_radius_world"]) * S

    # Normalize obstacles + the drivable pool into the ~10-unit regime, centered.
    ctr = np.array([cx, cy], np.float32)
    centers_n = (centers - ctr) * S
    sel = (np.abs(pool_w[:, 0] - cx) < args.region) & (np.abs(pool_w[:, 1] - cy) < args.region)
    pool = (pool_w[sel] - ctr) * S
    if len(pool) == 0:
        raise SystemExit("no free points in the working region — widen --region")
    print("SETUP obstacles=%d radn=%.3f rr=%.3f free_pool=%d S=%.5f region=%.0f"
          % (len(centers_n), radn, rr, len(pool), S, args.region))

    model = CoefEnergyNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    H, dt = args.horizon, args.dt
    d_hat = args.d_hat_world * S  # barrier reach in the normalized regime (local, not global)
    t0 = time.time()
    for it in range(args.steps):
        o0, v0, goal, C, R, mask, of, gf = make_batch(pool, centers_n, radn, rr, args.batch)
        al, be, ga = model(of, mask, gf)
        B = o0.shape[0]
        oT, vT, clr = integrate_surrogate_v2(
            o0, v0, goal, C, R, mask, al, be, ga,
            torch.full((B,), d_hat), torch.full((B,), dt),
            torch.full((B,), H, dtype=torch.long),
            robot_radius=torch.full((B,), rr), margin_factor=0.5,
        )
        L_goal = ((oT - goal) ** 2).sum(-1).mean()                                   # reach it
        # Penalize COLLISION only (clr below a thin margin), NOT proximity: streets are
        # narrow, so navigating them needs low positive clearance. A speed cap or an
        # over-eager clearance penalty makes staying put (max clearance / min speed) beat
        # moving, and the net collapses to a crawl (gamma >> beta). No speed cap.
        L_pen = torch.nn.functional.softplus((0.02 - clr) / 0.02).mean()
        # Anchor coefficients to the known-good navigating regime (beta~3, gamma~4,
        # alpha~3) so the self-supervised optimizer stays in the stable basin while the
        # task terms adapt them per situation.
        L_reg = ((be - 3.0) ** 2).mean() + ((ga - 4.0) ** 2).mean() + \
                (((al - 3.0) ** 2) * mask).sum() / mask.sum().clamp_min(1)
        loss = L_goal + 3.0 * L_pen + args.w_reg * L_reg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if it % 400 == 0:
            print("TRAIN it=%d/%d L_goal=%.3f L_pen=%.3f (%.0fs)"
                  % (it, args.steps, float(L_goal), float(L_pen), time.time() - t0), flush=True)
    train_s = time.time() - t0
    print("TIME %d steps in %.1fs = %.1f ms/step" % (args.steps, train_s, 1000 * train_s / args.steps))

    meta = {"scale": S, "center": [cx, cy], "region": args.region, "radn": radn, "rr": rr,
            "d_hat": d_hat, "dt": dt, "horizon": H, "steps": args.steps,
            "bounds": [mnx, mny, mxx, mxy], "target_extent": TARGET_EXTENT}
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, args.out)
    print("SAVED %s" % args.out)

    # Sanity eval: full navigation with the trained net (single-step rollout loop).
    model.eval()
    reached, steps_used = 0, []
    M = args.eval_episodes
    for _ in range(M):
        s = pool[np.random.randint(0, len(pool))]
        g = pool[np.random.randint(0, len(pool))]
        if np.linalg.norm(g - s) < 3.0:
            continue
        o = torch.from_numpy(s).unsqueeze(0)
        v = torch.zeros(1, 2)
        goalt = torch.from_numpy(g).unsqueeze(0)
        for t in range(500):
            p = o[0].numpy()
            nb = np.ascontiguousarray(_nearby(centers_n, p))
            Rw = np.full(len(nb), radn, np.float32)
            of, gf = build_local_feats(p, g, nb, Rw, np.ones(len(nb), np.float32))
            mk = (torch.ones(1, of.shape[1], dtype=torch.bool) if of.shape[1]
                  else torch.zeros(1, 0, dtype=torch.bool))
            with torch.no_grad():
                al, be, ga = model(of, mk, gf)
            C = torch.from_numpy(nb).unsqueeze(0) if len(nb) else torch.zeros(1, 0, 2)
            Rt = torch.from_numpy(Rw).unsqueeze(0) if len(nb) else torch.zeros(1, 0)
            m2 = torch.ones(1, len(nb), dtype=torch.bool) if len(nb) else torch.zeros(1, 0, dtype=torch.bool)
            o, v, _ = integrate_surrogate_v2(
                o, v, goalt, C, Rt, m2, al, be, ga,
                torch.tensor([d_hat]), torch.tensor([dt]), torch.tensor([1]),
                robot_radius=torch.tensor([rr]), margin_factor=0.5)
            if np.linalg.norm(o[0].numpy() - g) < 0.4:
                reached += 1
                steps_used.append(t)
                break
    print("EVAL reached %d/%d (median steps %s)"
          % (reached, M, int(np.median(steps_used)) if steps_used else "-"))
    result = {"steps": args.steps, "train_seconds": train_s,
              "ms_per_step": 1000 * train_s / args.steps, "eval_reached": reached, "eval_total": M}
    with open(os.path.splitext(args.out)[0] + "_result.json", "w") as fh:
        json.dump(result, fh)


if __name__ == "__main__":
    main()
