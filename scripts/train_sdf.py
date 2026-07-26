"""Self-supervised training of the SDF navigation coefficients on a real scene.

Reads a ``nav_sdf.npz`` (from ``build_sdf.py``) and trains ``sdf_nav.CoefMLP`` to
predict ``(alpha, beta, gamma)`` for the differentiable SDF surrogate by
backpropagating a reach-goal + no-collision objective through the rollout — no
labels. The coefficient net is biased toward the known-good navigating regime, so
it starts near-optimal and converges in a few hundred–thousand steps.

Runs on CPU (the SDF removes the per-step obstacle-search cost of the circle
surrogate, so CPU is fast) or on GPU automatically if a CUDA torch is present.

Usage:
    python scripts/train_sdf.py <bundle_dir>/nav_sdf.npz -o coef_sdf.pt --steps 1500
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import sdf_nav


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sdf_npz")
    ap.add_argument("-o", "--out", default="coef_sdf.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--dt", type=float, default=0.06)
    ap.add_argument("--d-hat", type=float, default=0.35, help="wall barrier reach (normalized)")
    ap.add_argument("--robot-radius-world", type=float, default=3.0)
    ap.add_argument("--vmax", type=float, default=0.9)
    ap.add_argument("--nsub-infer", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--bundle", default=None, help="scene dir (for the eval penetration check); optional")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.sdf_npz)
    phi, nxg, nyg = d["phi"], d["normal_x"], d["normal_y"]
    bounds = [float(v) for v in d["bounds"]]
    center = [float(v) for v in d["center"]]
    S = float(d["scale"]); region = float(d["region"])
    rr = args.robot_radius_world * S
    field = sdf_nav.SDFField(phi, nxg, nyg, bounds, center, S, device=dev)
    mnx, mny, mxx, mxy = bounds
    print("device=%s field=%s phi[%.2f,%.2f] rr=%.3f S=%.5f" % (dev, phi.shape, phi.min(), phi.max(), rr, S))

    # free-point pool (normalized, centered) within the working region
    ny_, nx_ = phi.shape
    gxs = np.linspace(mnx, mxx, nx_); gys = np.linspace(mny, mxy, ny_)
    GX, GY = np.meshgrid(gxs, gys)
    free = phi > (rr + 0.02)                                   # drivable = clearance beyond the robot
    reg = (np.abs(GX - center[0]) < region) & (np.abs(GY - center[1]) < region)
    sel = free & reg
    pool = np.stack([(GX[sel] - center[0]) * S, (GY[sel] - center[1]) * S], 1).astype(np.float32)
    poolt = torch.from_numpy(pool).to(dev)
    print("pool=%d" % len(pool))

    model = sdf_nav.CoefMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    kw = dict(rr=rr, d_hat=args.d_hat, dt=args.dt, vmax=args.vmax)
    import time
    t0 = time.time()
    for it in range(args.steps):
        si = torch.randint(0, len(poolt), (args.batch,), device=dev)
        o0 = poolt[si]
        ang = torch.rand(args.batch, device=dev) * 6.2832
        dd = 1.0 + torch.rand(args.batch, device=dev)                       # 1..2 local goal
        goal = o0 + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * dd.unsqueeze(-1)
        al, be, ga = model(sdf_nav.coef_feats(field, o0, goal))
        oT, vT, clr = sdf_nav.sdf_rollout(field, o0, torch.zeros_like(o0), goal, al, be, ga,
                                          args.horizon, nsub=1, **kw)
        L_goal = ((oT - goal) ** 2).sum(-1).mean()
        L_col = F.softplus((0.02 - clr) / 0.02).mean()
        L_reg = ((be - 3.0) ** 2).mean() + ((ga - 4.0) ** 2).mean() + ((al - 1.0) ** 2).mean()
        loss = L_goal + 3.0 * L_col + 0.1 * L_reg
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if it % 300 == 0:
            print("it=%d/%d L_goal=%.3f L_col=%.3f al=%.2f be=%.2f ga=%.2f (%.0fs)"
                  % (it, args.steps, float(L_goal), float(L_col), float(al.mean()),
                     float(be.mean()), float(ga.mean()), time.time() - t0), flush=True)
    tt = time.time() - t0
    print("TIME %d steps %.1fs = %.1f ms/step on %s" % (args.steps, tt, 1000 * tt / args.steps, dev))

    meta = {"scale": S, "center": center, "region": region, "rr": rr, "d_hat": args.d_hat,
            "dt": args.dt, "horizon": args.horizon, "nsub": args.nsub_infer, "vmax": args.vmax,
            "steps": args.steps, "bounds": bounds, "kind": "sdf", "sdf_npz": os.path.abspath(args.sdf_npz)}
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, args.out)
    print("SAVED %s" % args.out)
    json.dump({"steps": args.steps, "ms_per_step": 1000 * tt / args.steps, "device": str(dev)},
              open(os.path.splitext(args.out)[0] + "_result.json", "w"))


if __name__ == "__main__":
    main()
