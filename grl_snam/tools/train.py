"""Self-supervised training of the SDF navigation coefficients.

Reads a ``nav_sdf.npz`` (from :mod:`grl_snam.tools.sdf`) and trains
``sdf_nav.CoefMLP`` to predict ``(alpha, beta, gamma)`` for the differentiable SDF
surrogate by back-propagating a reach-goal + no-collision objective through the
rollout — no labels. The net is biased toward the known-good navigating regime, so
it starts near-optimal and converges in a few hundred to a few thousand steps. Runs
on CPU (fast — the SDF removes the per-step obstacle search) or CUDA if present.

The objective is ``goal + 3 col + 0.1 reg`` (:func:`grl_snam.tools.sdf_eval.loss_terms`), in a
sim frame of ``metres_per_unit`` metres per unit (docs/learning-validation.md §9):

- ``metres_per_unit`` (default 20, :data:`grl_snam.tools.sdf_eval.SIM_METRES_PER_UNIT`): the
  physics constants (``d_hat``, ``vmax``, the carrot) are numbers in sim units, tuned where one
  unit is 20 m. ``grl-snam sdf`` frames a map at ~10 units across its region, 300 m per unit on
  Austin, where the same numbers are a 270 m/s robot with a 105 m barrier band. The field is
  read in the sim frame (the same geometry in metres); training and held-out pairs are drawn in
  the npz's own frame, as before, and converted, so they are the same places. ``0`` is the npz's
  own frame (the trainer before §9). The checkpoint's ``meta["scale"]`` is the sim frame's and
  ``meta["sdf_scale"]`` the npz's; a consumer reads the npz in ``meta["scale"]``.
- ``objective`` ``task`` (default): the seek drive's carrot rule each step, the goal term the
  share of possible progress toward arrival not made, the collision margin and temperature in
  metres (``col_margin_m``, ``col_temp_m``). ``end-distance``: the objective before §9 (the
  coefficients once, the squared end distance, a 0.02-unit margin).
- ``leg_ticks`` (``task``'s default 1000, the held-out drive's length): each batch row drives one
  leg, start to goal, across optimizer steps, ``horizon`` steps a step from where it stopped,
  and is redrawn when it arrives or after ``leg_ticks`` ticks. The objective's windows then see
  the whole of a 300-600 m leg, stalls behind buildings included, where a fresh draw sees its
  first 30 m. ``0``: fresh pairs every step, from rest (``end-distance`` always).
- ``reg_centre`` (``task`` only; default the net's bias regime ``(1, 3, 4)``): the
  ``(alpha, beta, gamma)`` the ``reg`` term pulls toward. The net still starts at its bias; only
  the pull moves (docs/learning-validation.md §10).
- ``pick`` ``clean-reach`` (default): the held-out pick is the share of held-out pairs reached
  with no penetration tick, a fitness the objective also serves; ``reach`` is the pick before §9.

Two settings decide the gradient, and neither changes a forward value:

- ``bptt_window`` (default 7): the backward is truncated every that many rollout steps. Through
  the full 28-step horizon it is the derivative of a chaotic map. On dense geometry in the
  field's units (Austin: 1 unit = 300 m, so rr is 0.01 units and one SDF cell is 0.0098), an
  agent near a wall bounces among buildings like a dispersing billiard. The barrier's curvature
  and the turn of the wall normal from cell to cell then multiply a perturbation by ~1.65 a
  step, and the gradient grows as ~1.65**window (median norm 7.8e4 at 28 steps, 2.2 at 7). Full
  BPTT clipped every optimizer step, so Adam followed whichever few samples exploded.
- ``col_grad`` (default on): the collision term carries gradient. Without it the rollout's
  clearance is detached, so ``col`` is a constant to the optimizer. A clean gradient then lowers
  ``goal`` by weakening the wall barrier while ``col`` climbs, and the loss rises.

``metres_per_unit=0, objective="end-distance", bptt_window=0, col_grad=False`` is the trainer
before the fix; seed 0 for 200 steps on the CPU reproduces the published ``coef_sdf.pt`` bit for
bit (docs/learning-validation.md §2.2, §8). With ``bptt_window=7, col_grad=True`` and
``pick="reach"`` it is §8's trainer.

The learning instruments (:mod:`grl_snam.learning_history`) are opt-in, as in coef-train:

- ``history``: the per-step loss terms, the gradient norm before the clip and the learning
  rate, and per checkpoint each head's mean and spread on a fixed probe batch and its
  ``||dtheta|| / ||theta||``. ``grad_terms_every`` adds each weighted term's own gradient norm
  every that many steps (``grad_norm.terms.<term>``; extra backward passes, same update).
- ``checkpoint_every``: the policy every N steps as ``<out stem>.stepNNNNNN.pt``, step 0 (the
  untrained net) included.
- ``val_frac``: hold out that fraction of the working region's ``val_blocks x val_blocks``
  blocks (:func:`grl_snam.tools.sdf_eval.block_split`). Training starts are drawn only outside
  them. Each checkpoint scores fixed held-out pairs started inside them: the objective's terms,
  and the seek drive's reach, penetration, path length, clearance and time to goal. The
  checkpoint with the best ``pick`` is kept as ``<out stem>.best_heldout.pt``; a best at step 0
  means no step beat the untrained net on held-out pairs. ``val_ticks`` (default 1000: 60 s, in
  which a 300-600 m Austin leg at 18 m/s has room for a detour) and ``reach_tol`` (in the npz's
  units: 0.05 is 15 m on Austin) set the drive.

With none of them on, training draws and weights are what they were. ``val_frac`` changes the
training pool, so it changes the weights; the instruments alone do not.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

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
    val_ticks: int = 1000,
    reach_tol: float = 0.05,
    bptt_window: int = 7,
    col_grad: bool = True,
    grad_terms_every: int = 0,
    metres_per_unit: float = sdf_eval.SIM_METRES_PER_UNIT,
    objective: str = "task",
    col_margin_m: float = 0.4,
    col_temp_m: float = 0.4,
    pick: str = "clean-reach",
    leg_ticks: int | None = None,
    reg_centre: tuple[float, float, float] | None = None,
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

    if objective not in sdf_eval.OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r} ({', '.join(sdf_eval.OBJECTIVES)})")
    if pick not in PICKS:
        raise ValueError(f"unknown pick {pick!r} ({', '.join(PICKS)})")
    d = np.load(sdf_npz)
    phi, nxg, nyg = d["phi"], d["normal_x"], d["normal_y"]
    bounds = [float(v) for v in d["bounds"]]
    center = [float(v) for v in d["center"]]
    npz_scale = float(d["scale"])
    region = float(d["region"])
    # The task (pools, pairs, legs, reach_tol) is drawn in the npz's frame; the sim runs in
    # `scale`'s. k converts: k == 1.0 (metres_per_unit 0) multiplies nothing.
    scale = sdf_eval.sim_scale(npz_scale, metres_per_unit)
    k = scale / npz_scale
    rr_task = robot_radius_world * npz_scale
    rr = robot_radius_world * scale
    task_field = sdf_nav.SDFField(phi, nxg, nyg, bounds, center, npz_scale, device=dev)
    task_scene = sdf_eval.SdfScene(task_field, phi, bounds, center, npz_scale, region)
    scene = sdf_eval.in_frame(task_scene, scale)
    field = scene.field
    mnx, mny, mxx, mxy = bounds
    print(
        "device=%s field=%s phi[%.2f,%.2f] rr=%.3f S=%.5f (%.1f m/unit; npz %.1f m/unit)"
        % (dev, phi.shape, scene.phi.min(), scene.phi.max(), rr, scale, 1 / scale, 1 / npz_scale)
    )

    # drivable free-point pool (normalized, centered) within the working region
    ny_, nx_ = phi.shape
    gxs = np.linspace(mnx, mxx, nx_)
    gys = np.linspace(mny, mxy, ny_)
    grid_x, grid_y = np.meshgrid(gxs, gys)
    free = phi > (rr_task + 0.02)
    reg = (np.abs(grid_x - center[0]) < region) & (np.abs(grid_y - center[1]) < region)
    sel = free & reg
    pool = np.stack(
        [(grid_x[sel] - center[0]) * npz_scale, (grid_y[sel] - center[1]) * npz_scale], 1
    )
    pool = pool.astype(np.float32)
    held_blocks, held_pool = None, None
    if val_frac > 0.0:
        held_blocks = sdf_eval.block_split(val_blocks, val_frac, val_seed)
        pool, held_pool = sdf_eval.split_pool(pool, region * npz_scale, val_blocks, held_blocks)
        print(
            "held out %d of %d blocks: %d points"
            % (len(held_blocks), val_blocks**2, len(held_pool))
        )
    poolt = torch.from_numpy(pool).to(dev)
    if k != 1.0:
        poolt = poolt * k
    print("pool=%d drivable points" % len(pool))

    model = sdf_nav.CoefMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    kw = dict(rr=rr, d_hat=d_hat, dt=dt, vmax=vmax)
    obj = dict(kind=objective, window=bptt_window, col_grad=col_grad)
    if leg_ticks is None:
        leg_ticks = 1000 if objective == "task" else 0
    if leg_ticks > 0 and objective != "task":
        raise ValueError("leg_ticks is the task objective's")
    if reg_centre is not None and objective != "task":
        raise ValueError("reg_centre is the task objective's")
    if objective == "task":
        obj.update(
            carrot_m=36.0, col_margin_m=col_margin_m, col_temp_m=col_temp_m,
            reach_tol_m=round(reach_tol / npz_scale, 6),
        )  # fmt: skip
        if reg_centre is not None:  # absent: the bias regime (1, 3, 4), as before
            obj["reg_centre"] = [float(x) for x in reg_centre]
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
        "sdf_scale": npz_scale,
        "metres_per_unit": 1.0 / scale,
        "objective": obj,
        "pick": pick,
        "leg_ticks": leg_ticks,
    }
    monitor = None
    if history or checkpoint_every > 0 or val_frac > 0.0:
        monitor, probe = _monitor(
            out, meta, scene, task_scene, pool, held_pool, held_blocks, kw,
            history=history, checkpoint_every=checkpoint_every, seed=seed, lr=lr,
            val_frac=val_frac, val_seed=val_seed, val_blocks=val_blocks, val_pairs=val_pairs,
            val_ticks=val_ticks, reach_tol=reach_tol, nsub=nsub_infer, objective=obj, pick=pick,
            rr_task=rr_task,
        )  # fmt: skip
        monitor.begin(model, probe)

    def draw(n):
        si = torch.randint(0, len(poolt), (n,), device=dev)
        o0 = poolt[si]
        ang = torch.rand(n, device=dev) * 6.2832
        dd = 1.0 + torch.rand(n, device=dev)  # 1..2 local goal (npz units)
        if k != 1.0:
            dd = dd * k
        return o0, o0 + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * dd.unsqueeze(-1)

    legs = None
    if leg_ticks > 0:  # each row drives one leg across steps (see the module docstring)
        o_l, g_l = draw(batch)
        legs = dict(o=o_l, g=g_l, v=torch.zeros_like(o_l), age=torch.zeros(batch, device=dev))
    t0 = time.time()
    for it in range(steps):
        state = None
        if legs is None:
            o0, goal = draw(batch)
            lk = {}
        else:
            o0, goal, state = legs["o"], legs["g"], {}
            lk = dict(v0=legs["v"], state=state)
        t, (al, be, ga) = sdf_eval.loss_terms(
            model, field, o0, goal, horizon=horizon, **kw, **obj, **lk
        )
        loss, loss_goal, loss_col, loss_reg = t["loss"], t["goal"], t["col"], t["reg"]
        term_grads = None
        if monitor is not None and grad_terms_every > 0 and it % grad_terms_every == 0:
            term_grads = _term_grad_norms(model, t)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if legs is not None:
            legs.update(o=state["o"], v=state["v"], age=legs["age"] + horizon)
            done = ((legs["g"] - legs["o"]).norm(dim=-1) < reach_tol * k) | (
                legs["age"] >= leg_ticks
            )
            nd = int(done.sum())
            if nd:
                o_n, g_n = draw(nd)
                legs["o"][done], legs["g"][done] = o_n, g_n
                legs["v"][done] = 0.0
                legs["age"][done] = 0.0
        if monitor is not None:
            terms = {"loss": loss.item(), "goal": loss_goal.item(), "col": loss_col.item()}
            terms["reg"] = loss_reg.item()
            monitor.window(terms, grad_norm, opt.param_groups[0]["lr"], 5.0, term_grads)
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
            print("best held-out %s %.4f at step %d" % (pick, monitor.best, monitor.best_step))
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


def _term_grad_norms(model, terms: dict) -> dict[str, float]:
    """Each weighted loss term's gradient norm (``goal``, ``3 col``, ``0.1 reg``), before any
    clip. Extra backward passes that leave ``.grad`` and the update untouched."""
    params = [p for p in model.parameters() if p.requires_grad]
    out = {}
    for name, w in (("goal", 1.0), ("col", 3.0), ("reg", 0.1)):
        term = terms[name]
        if not term.requires_grad:
            out[name] = 0.0
            continue
        gs = torch.autograd.grad(w * term, params, retain_graph=True, allow_unused=True)
        sq = sum(float((g.double() ** 2).sum()) for g in gs if g is not None)
        out[name] = sq**0.5
    return out


#: The held-out pick (``train_sdf``'s ``pick``): the history field it maximizes, and its name.
PICKS = {
    "clean-reach": ("clean_reach", "heldout seek-drive reach without penetration"),
    "reach": ("reach", "heldout seek-drive reach"),
}


def _stem(out: str) -> str:
    return out[: -len(".pt")] if out.endswith(".pt") else out


def _monitor(out, meta, scene, task_scene, pool, held_pool, held_blocks, kw, **o):
    """The instruments for :func:`train_sdf` (see the module docstring): the monitor, and the
    probe that reads the heads on its fixed batch (pairs from the training pool). ``pool`` and
    ``held_pool`` are in ``task_scene``'s frame (the npz's); pairs are drawn there and converted
    to ``scene``'s."""
    k = scene.scale / task_scene.scale

    def pairs(p, n, seed):
        s, g = sdf_eval.sample_pairs(task_scene, p, n, seed, rr=o["rr_task"])
        return (s * k, g * k) if k != 1.0 else (s, g)

    stem = _stem(out)
    every = o["checkpoint_every"]

    def save(model, path):
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"model_state_dict": state, "meta": meta}, path)

    fitness = None
    hdr = {"trainer": "grl-snam train", "seed": o["seed"], "lr": o["lr"], "steps": meta["steps"]}
    hdr["objective"] = o["objective"]
    hdr.update(
        metres_per_unit=meta["metres_per_unit"], sdf_scale=meta["sdf_scale"],
        leg_ticks=meta["leg_ticks"],
    )  # fmt: skip
    if held_pool is not None:
        hs, hg = pairs(held_pool, o["val_pairs"], o["val_seed"])
        drive = dict(kw, nsub=o["nsub"], ticks=o["val_ticks"], reach_tol=o["reach_tol"] * k)
        field_key, fitness_name = PICKS[o["pick"]]

        def fitness(model):
            terms = sdf_eval.objective_terms(
                model, scene, hs, hg, horizon=meta["horizon"], **kw, **o["objective"]
            )
            d = sdf_eval.seek_drive(model, scene, hs, hg, **drive)
            got = {f"obj_{k}": v for k, v in terms.items()}
            got.update(sdf_eval.drive_summary(d, kw["dt"]))
            got["fitness"] = got[field_key]
            return got

        hdr.update(
            val_frac=o["val_frac"], val_seed=o["val_seed"], val_blocks=o["val_blocks"],
            heldout_blocks=held_blocks, heldout_pairs=o["val_pairs"], heldout_ticks=o["val_ticks"],
            reach_tol=o["reach_tol"], fitness=fitness_name, pick=o["pick"],
        )  # fmt: skip
    cfg = CoefTrainMonitorCfg(
        history=o["history"] or None,
        checkpoint_every=every,
        checkpoint_prefix=stem if every > 0 else None,
        checkpoint_ext=".pt",
        best_path=f"{stem}.best_heldout.pt" if held_pool is not None else None,
    )
    monitor = CoefTrainMonitor(cfg, save=save, fitness=fitness, meta=hdr)
    po, pg = pairs(pool, cfg.probe_agents, cfg.probe_seed)
    if o["objective"]["kind"] == "task":  # the heads where the net is used: at the seek carrot
        pg = sdf_eval.carrot_of(po, pg, o["objective"]["carrot_m"] * scene.scale)
    return monitor, lambda m: coef_heads(m, sdf_nav.coef_feats(scene.field, po, pg))
