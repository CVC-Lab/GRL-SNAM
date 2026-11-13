"""Self-supervised training of the SDF navigation coefficients.

Reads a ``nav_sdf.npz`` (from :mod:`grl_snam.tools.sdf`) and trains
``sdf_nav.CoefMLP`` to predict ``(alpha, beta, gamma)`` for the differentiable SDF
surrogate by back-propagating a reach-goal + no-collision objective through the
rollout — no labels. The net is biased toward the known-good navigating regime, so
it starts near-optimal and converges in a few hundred to a few thousand steps. Runs
on CPU (fast — the SDF removes the per-step obstacle search) or CUDA if present.

The learning instruments (:mod:`grl_snam.learning_history`) are opt-in, as in coef-train:

- ``history``: the per-step loss terms, the gradient norm before the clip and the learning
  rate, and per checkpoint each head's mean and spread on a fixed probe batch and its
  ``||dtheta|| / ||theta||``.
- ``checkpoint_every``: the policy every N steps as ``<out stem>.stepNNNNNN.pt``, step 0 (the
  untrained net) included.
- ``val_frac``: hold out that fraction of the working region's ``val_blocks x val_blocks``
  blocks (:func:`grl_snam.tools.sdf_eval.block_split`). Training starts are drawn only outside
  them. Each checkpoint scores fixed held-out pairs started inside them: the objective's terms,
  and the seek drive's reach, penetration, path length, clearance and time to goal. The
  checkpoint with the highest held-out reach is kept as ``<out stem>.best_heldout.pt``; a best at
  step 0 means no step beat the untrained net on held-out pairs.

With none of them on, training draws and weights are what they were. ``val_frac`` changes the
training pool, so it changes the weights; the instruments alone do not.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import grl_snam.sdf_nav as sdf_nav

from ..learning_history import CoefTrainMonitor, CoefTrainMonitorCfg, coef_heads
from . import sdf_eval


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
    device: str | None = None,
    history: str | None = None,
    checkpoint_every: int = 0,
    val_frac: float = 0.0,
    val_seed: int = 0,
    val_blocks: int = 8,
    val_pairs: int = 256,
    val_ticks: int = 200,
    reach_tol: float = 0.05,
) -> str:
    """Train the SDF coefficient net on ``sdf_npz`` and save a checkpoint ``.pt``
    (state dict + physics ``meta``) plus a ``_result.json``. Returns the checkpoint path.
    ``device`` ``None`` picks CUDA when present; the instruments are in the module docstring."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
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
    pool = pool.astype(np.float32)
    held_blocks, held_pool = None, None
    if val_frac > 0.0:
        held_blocks = sdf_eval.block_split(val_blocks, val_frac, val_seed)
        pool, held_pool = sdf_eval.split_pool(pool, region * scale, val_blocks, held_blocks)
        print(
            "held out %d of %d blocks: %d points"
            % (len(held_blocks), val_blocks**2, len(held_pool))
        )
    poolt = torch.from_numpy(pool).to(dev)
    print("pool=%d drivable points" % len(pool))

    model = sdf_nav.CoefMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    kw = dict(rr=rr, d_hat=d_hat, dt=dt, vmax=vmax)
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
    monitor = None
    if history or checkpoint_every > 0 or val_frac > 0.0:
        scene = sdf_eval.SdfScene(field, phi, bounds, center, scale, region)
        monitor, probe = _monitor(
            out, meta, scene, pool, held_pool, held_blocks, kw,
            history=history, checkpoint_every=checkpoint_every, seed=seed, lr=lr,
            val_frac=val_frac, val_seed=val_seed, val_blocks=val_blocks, val_pairs=val_pairs,
            val_ticks=val_ticks, reach_tol=reach_tol, nsub=nsub_infer,
        )  # fmt: skip
        monitor.begin(model, probe)
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if monitor is not None:
            terms = {"loss": loss.item(), "goal": loss_goal.item(), "col": loss_col.item()}
            terms["reg"] = loss_reg.item()
            monitor.window(terms, grad_norm, opt.param_groups[0]["lr"], 5.0)
            monitor.end_step(it, model)
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
    if monitor is not None:
        monitor.finish(steps, model)
        if monitor.best_step is not None:
            print("best held-out reach %.4f at step %d" % (monitor.best, monitor.best_step))
    dt_total = time.time() - t0
    print(
        "TIME %d steps %.1fs = %.1f ms/step on %s"
        % (steps, dt_total, 1000 * dt_total / max(1, steps), dev)
    )

    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, out)
    print("SAVED %s" % out)
    with open(os.path.splitext(out)[0] + "_result.json", "w") as fh:
        json.dump(
            {"steps": steps, "ms_per_step": 1000 * dt_total / max(1, steps), "device": str(dev)}, fh
        )
    return out


def _stem(out: str) -> str:
    return out[: -len(".pt")] if out.endswith(".pt") else out


def _monitor(out, meta, scene, pool, held_pool, held_blocks, kw, **o):
    """The instruments for :func:`train_sdf` (see the module docstring): the monitor, and the
    probe that reads the heads on its fixed batch (pairs from the training pool)."""
    stem = _stem(out)
    every = o["checkpoint_every"]

    def save(model, path):
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"model_state_dict": state, "meta": meta}, path)

    fitness = None
    hdr = {"trainer": "grl-snam train", "seed": o["seed"], "lr": o["lr"], "steps": meta["steps"]}
    if held_pool is not None:
        hs, hg = sdf_eval.sample_pairs(scene, held_pool, o["val_pairs"], o["val_seed"], rr=kw["rr"])
        drive = dict(kw, nsub=o["nsub"], ticks=o["val_ticks"], reach_tol=o["reach_tol"])

        def fitness(model):
            terms = sdf_eval.objective_terms(model, scene, hs, hg, horizon=meta["horizon"], **kw)
            d = sdf_eval.seek_drive(model, scene, hs, hg, **drive)
            got = {f"obj_{k}": v for k, v in terms.items()}
            got.update(sdf_eval.drive_summary(d, kw["dt"]))
            got["fitness"] = got["reach"]
            return got

        hdr.update(
            val_frac=o["val_frac"], val_seed=o["val_seed"], val_blocks=o["val_blocks"],
            heldout_blocks=held_blocks, heldout_pairs=o["val_pairs"], heldout_ticks=o["val_ticks"],
            reach_tol=o["reach_tol"], fitness="heldout seek-drive reach",
        )  # fmt: skip
    cfg = CoefTrainMonitorCfg(
        history=o["history"] or None,
        checkpoint_every=every,
        checkpoint_prefix=stem if every > 0 else None,
        checkpoint_ext=".pt",
        best_path=f"{stem}.best_heldout.pt" if held_pool is not None else None,
    )
    monitor = CoefTrainMonitor(cfg, save=save, fitness=fitness, meta=hdr)
    po, pg = sdf_eval.sample_pairs(scene, pool, cfg.probe_agents, cfg.probe_seed, rr=kw["rr"])
    return monitor, lambda m: coef_heads(m, sdf_nav.coef_feats(scene.field, po, pg))
