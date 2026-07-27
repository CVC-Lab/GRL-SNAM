"""Self-supervised training of the SDF navigation coefficients.

Reads a ``nav_sdf.npz`` (from :mod:`grl_snam.tools.sdf`) and trains
``sdf_nav.CoefMLP`` to predict ``(alpha, beta, gamma)`` for the differentiable SDF
surrogate by back-propagating a reach-goal + no-collision objective through the
rollout — no labels. The net is biased toward the known-good navigating regime, so
it starts near-optimal and converges in a few hundred to a few thousand steps. Runs
on CPU (fast — the SDF removes the per-step obstacle search) or CUDA if present.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import sdf_nav


def train_sdf(
    sdf_npz: str,
    out: str = "coef_sdf.pt",
    *,
    steps: int = 1500,
    batch: int = 128,
    horizon: int = 28,
    dt: float = 0.06,
    d_hat: float = 0.35,
    robot_radius_world: float = 3.0,
    vmax: float = 0.9,
    nsub_infer: int = 3,
    lr: float = 3e-4,
    threads: int = 6,
    seed: int = 0,
    log_every: int = 300,
) -> str:
    """Train the SDF coefficient net on ``sdf_npz`` and save a checkpoint ``.pt``
    (state dict + physics ``meta``) plus a ``_result.json``. Returns the checkpoint path."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = np.load(sdf_npz)
    phi, nxg, nyg = d["phi"], d["normal_x"], d["normal_y"]
    bounds = [float(v) for v in d["bounds"]]
    center = [float(v) for v in d["center"]]
    scale = float(d["scale"])
    region = float(d["region"])
    rr = robot_radius_world * scale
    field = sdf_nav.SDFField(phi, nxg, nyg, bounds, center, scale, device=dev)
    mnx, mny, mxx, mxy = bounds
    print(
        "device=%s field=%s phi[%.2f,%.2f] rr=%.3f S=%.5f"
        % (dev, phi.shape, phi.min(), phi.max(), rr, scale)
    )

    # drivable free-point pool (normalized, centered) within the working region
    ny_, nx_ = phi.shape
    gxs = np.linspace(mnx, mxx, nx_)
    gys = np.linspace(mny, mxy, ny_)
    grid_x, grid_y = np.meshgrid(gxs, gys)
    free = phi > (rr + 0.02)
    reg = (np.abs(grid_x - center[0]) < region) & (np.abs(grid_y - center[1]) < region)
    sel = free & reg
    pool = np.stack([(grid_x[sel] - center[0]) * scale, (grid_y[sel] - center[1]) * scale], 1)
    poolt = torch.from_numpy(pool.astype(np.float32)).to(dev)
    print("pool=%d drivable points" % len(pool))

    model = sdf_nav.CoefMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    kw = dict(rr=rr, d_hat=d_hat, dt=dt, vmax=vmax)
    t0 = time.time()
    for it in range(steps):
        si = torch.randint(0, len(poolt), (batch,), device=dev)
        o0 = poolt[si]
        ang = torch.rand(batch, device=dev) * 6.2832
        dd = 1.0 + torch.rand(batch, device=dev)  # 1..2 local goal
        goal = o0 + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * dd.unsqueeze(-1)
        al, be, ga = model(sdf_nav.coef_feats(field, o0, goal))
        oT, _vT, clr = sdf_nav.sdf_rollout(
            field, o0, torch.zeros_like(o0), goal, al, be, ga, horizon, nsub=1, **kw
        )
        loss_goal = ((oT - goal) ** 2).sum(-1).mean()
        loss_col = F.softplus((0.02 - clr) / 0.02).mean()
        loss_reg = ((be - 3.0) ** 2).mean() + ((ga - 4.0) ** 2).mean() + ((al - 1.0) ** 2).mean()
        loss = loss_goal + 3.0 * loss_col + 0.1 * loss_reg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if it % log_every == 0:
            print(
                "it=%d/%d L_goal=%.3f L_col=%.3f al=%.2f be=%.2f ga=%.2f (%.0fs)"
                % (
                    it,
                    steps,
                    float(loss_goal),
                    float(loss_col),
                    float(al.mean()),
                    float(be.mean()),
                    float(ga.mean()),
                    time.time() - t0,
                ),
                flush=True,
            )
    dt_total = time.time() - t0
    print(
        "TIME %d steps %.1fs = %.1f ms/step on %s"
        % (steps, dt_total, 1000 * dt_total / max(1, steps), dev)
    )

    meta = {
        "scale": scale,
        "center": center,
        "region": region,
        "rr": rr,
        "d_hat": d_hat,
        "dt": dt,
        "horizon": horizon,
        "nsub": nsub_infer,
        "vmax": vmax,
        "steps": steps,
        "bounds": bounds,
        "kind": "sdf",
        "sdf_npz": os.path.abspath(sdf_npz),
    }
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, out)
    print("SAVED %s" % out)
    with open(os.path.splitext(out)[0] + "_result.json", "w") as fh:
        json.dump(
            {"steps": steps, "ms_per_step": 1000 * dt_total / max(1, steps), "device": str(dev)}, fh
        )
    return out
