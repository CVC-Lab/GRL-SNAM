"""Where ``grl-snam train``'s gradient blew up (docs/learning-validation.md §8.1): the training
objective's gradient at the untrained net, on fixed batches, under each candidate cause.

- ``frames``: each scene's frame against the dynamics' length scales (the SDF cell, rr, the
  distance one step covers at vmax, the share of the map inside the barrier band);
- ``window``: the gradient norm (each weighted term's and the total's) as the backward is
  truncated every 28 (full BPTT), 14, 7, 4, 2 and 1 steps, with the collision term's gradient on;
  and the trainer before the fix (full BPTT, collision term detached);
- ``alternatives``: the doc's first guesses at full BPTT: the inference substeps (``nsub`` 3) and
  the clamped barrier of ``train_coef_energy.ipc_piecewise`` (``max_grad`` 200);
- ``city``: the same objective on the procedural city (``coef_train._scene``), the frame the
  physics constants were set in;
- ``per_sample``: each pair's own ``|dL/d(alpha, beta, gamma)|`` at full BPTT, and the clearance
  of the pair that dominates;
- ``jacobian``: one rollout step's Jacobian ``d(o', v')/d(o, v)`` (spectral norm) by distance to
  the nearest wall, with the barrier's curvature or the wall normal's turn switched off in the
  backward.

    python experiments/learning_proof/gradient_source.py --sdf nav_sdf.npz --out gradient_source.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import grl_snam.sdf_nav as sdf_nav
from grl_snam.sdf_nav import _ipc_dbdd
from grl_snam.tools import coef_train, sdf_eval

PHYS = dict(d_hat=0.35, dt=0.06, vmax=0.9, horizon=28)
CLIP = 5.0


def batches(starts_fn, n_batches, batch, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        o0 = starts_fn(g, batch)
        ang = torch.rand(batch, generator=g) * 6.2832
        dd = 1.0 + torch.rand(batch, generator=g)
        out.append((o0, o0 + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * dd[:, None]))
    return out


def quantiles(v):
    v = np.asarray(v, np.float64)
    return {
        "q50": float(np.median(v)),
        "q90": float(np.quantile(v, 0.9)),
        "max": float(v.max()),
        "clipped_frac": float((v > CLIP).mean()),
    }


def grad_norms(field, data, rr, **objective):
    """Per batch, the gradient norm of the loss and of each weighted term, at the untrained net."""
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP()
    params = list(model.parameters())
    out = {"loss": [], "goal": [], "col": []}
    for o0, goal in data:
        t, _ = sdf_eval.loss_terms(model, field, o0, goal, rr=rr, **PHYS, **objective)
        for k, w in (("loss", 1.0), ("goal", 1.0), ("col", 3.0)):
            if not t[k].requires_grad:
                out[k].append(0.0)
                continue
            gs = torch.autograd.grad(w * t[k], params, retain_graph=True)
            out[k].append(float(torch.sqrt(sum((x.double() ** 2).sum() for x in gs))))
    return {k: quantiles(v) for k, v in out.items()}


def per_sample(field, data, rr):
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP()
    sens, worst_clr = [], []
    for o0, goal in data:
        al, be, ga = model(sdf_nav.coef_feats(field, o0, goal))
        c = [x.detach().requires_grad_() for x in (al, be, ga)]
        oT, _, clr = sdf_nav.sdf_rollout(
            field, o0, torch.zeros_like(o0), goal, *c, PHYS["horizon"], nsub=1, rr=rr,
            d_hat=PHYS["d_hat"], dt=PHYS["dt"], vmax=PHYS["vmax"],
        )  # fmt: skip
        g = torch.autograd.grad(((oT - goal) ** 2).sum(-1).sum(), c)
        s = torch.stack(g, -1).norm(dim=-1)
        sens.append(s.numpy())
        worst_clr.append(float(clr[int(s.argmax())]))
    s = np.concatenate(sens)
    return {
        "pairs": int(s.size),
        "q50": float(np.median(s)),
        "q99": float(np.quantile(s, 0.99)),
        "max": float(s.max()),
        "worst_pair_min_clearance": {"median": float(np.median(worst_clr)), "per_batch": worst_clr},
        "share_of_batches_where_worst_pair_penetrates": float(np.mean(np.array(worst_clr) < 0.0)),
    }


def jacobian(scene, rr, n=4096, seed=7):
    """One step's Jacobian norm by clearance band, on states the untrained net's rollouts visit."""
    field = scene.field
    pool = torch.from_numpy(sdf_eval.free_pool(scene, rr))
    g = torch.Generator().manual_seed(seed)
    o0 = pool[torch.randint(0, len(pool), (n,), generator=g)]
    ang = torch.rand(n, generator=g) * 6.2832
    goal = (
        o0
        + torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        * (1.0 + torch.rand(n, generator=g))[:, None]
    )
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP()
    kw = dict(rr=rr, d_hat=PHYS["d_hat"], dt=PHYS["dt"], vmax=PHYS["vmax"])
    with torch.no_grad():
        al, be, ga = model(sdf_nav.coef_feats(field, o0, goal))
        o, v, states = o0.clone(), torch.zeros_like(o0), []
        for t in range(PHYS["horizon"]):
            o, v, _ = sdf_nav.sdf_rollout(field, o, v, goal, al, be, ga, 1, nsub=1, **kw)
            if t in (5, 12, 20, 27):
                states.append((o.clone(), v.clone()))
    O, V = torch.cat([s[0] for s in states]), torch.cat([s[1] for s in states])
    G, AL, BE, GA = goal.repeat(4, 1), al.repeat(4), be.repeat(4), ga.repeat(4)

    def step(o, v, no_turn=False, no_curv=False):
        phi, nrm = field.sample(o)
        d = phi - rr
        nrm = nrm.detach() if no_turn else nrm
        d = d.detach() if no_curv else d
        a = -(AL * _ipc_dbdd(d, PHYS["d_hat"]))[:, None] * nrm - BE[:, None] * (o - G)
        v = v + PHYS["dt"] * (a - GA[:, None] * v)
        sp = v.norm(dim=-1, keepdim=True)
        v = torch.where(sp > PHYS["vmax"], v * PHYS["vmax"] / sp, v)
        return torch.cat([o + PHYS["dt"] * v, v], -1)

    def norms(**kw):
        x = torch.cat([O, V], -1).requires_grad_()
        y = step(x[:, :2], x[:, 2:], **kw)
        rows = [torch.autograd.grad(y[:, k].sum(), x, retain_graph=True)[0] for k in range(4)]
        return torch.linalg.matrix_norm(torch.stack(rows, 1), ord=2)

    with torch.no_grad():
        D = field.sample(O)[0] - rr
    res = {
        "full": norms(),
        "no_normal_turn": norms(no_turn=True),
        "no_barrier_curvature": norms(no_curv=True),
        "neither": norms(no_turn=True, no_curv=True),
    }
    bands = [(-9.0, 0.0, "d<0"), (0.0, 0.01, "0<d<rr"), (0.01, 0.05, "rr<d<0.05")]
    bands += [(0.05, 0.35, "0.05<d<d_hat"), (0.35, 99.0, "d>d_hat")]
    out = {}
    for lo, hi, name in bands:
        m = (D >= lo) & (D < hi)
        if bool(m.any()):
            out[name] = {"share": float(m.float().mean())}
            out[name].update(
                {k: {"median": float(r[m].median()), "geomean": float(r[m].log().mean().exp())}
                 for k, r in res.items()}
            )  # fmt: skip
    return out


def frame(field, phi, bounds, scale, rr):
    n = phi.shape[-1]
    cell = (bounds[2] - bounds[0]) / (n - 1) * scale
    step = PHYS["vmax"] * PHYS["dt"]
    return {
        "metres_per_unit": 1.0 / scale,
        "cell_units": cell,
        "rr_units": rr,
        "rr_cells": rr / cell,
        "step_units": step,
        "step_cells": step / cell,
        "share_within_d_hat": float((phi - rr < PHYS["d_hat"]).mean()),
        "share_inside_geometry": float((phi < rr).mean()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sdf", required=True, help="Austin's nav_sdf.npz")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    scene = sdf_eval.load_scene(args.sdf)
    rr = 3.0 * scene.scale
    pool = torch.from_numpy(sdf_eval.free_pool(scene, rr))
    data = batches(lambda g, b: pool[torch.randint(0, len(pool), (b,), generator=g)],
                   args.batches, args.batch, args.seed)  # fmt: skip
    cf, cmeta, rand_on = coef_train._scene(96, 0)
    rng = np.random.default_rng(0)
    cdata = batches(
        lambda g, b: torch.from_numpy(rand_on(b, rng)), args.batches, args.batch, args.seed
    )
    out = {
        "schema": "grl_snam.learnproof_gradient_source/v1",
        "args": vars(args),
        "physics": PHYS,
        "clip": CLIP,
        "frames": {
            "austin": frame(scene.field, scene.phi, scene.bounds, scene.scale, rr),
            "city": frame(cf, cf.field[0, 0].numpy(), (cf.mnx, cf.mny, cf.mxx, cf.mxy), cf.S,
                          cmeta["rr"]),
        },  # fmt: skip
    }
    print(json.dumps(out["frames"], indent=1), flush=True)
    out["window"] = {"before_fix": grad_norms(scene.field, data, rr, window=0, col_grad=False)}
    for w in (28, 14, 7, 4, 2, 1):
        out["window"][str(w)] = grad_norms(scene.field, data, rr, window=w, col_grad=True)
        print("window", w, out["window"][str(w)]["loss"], flush=True)
    alt = {}
    orig_rollout, orig_dbdd = sdf_nav.sdf_rollout, sdf_nav._ipc_dbdd
    sdf_nav.sdf_rollout = lambda *a, **k: orig_rollout(*a, **{**k, "nsub": 3})
    try:
        alt["nsub3"] = grad_norms(scene.field, data, rr, window=0, col_grad=False)
    finally:
        sdf_nav.sdf_rollout = orig_rollout
    from grl_snam.train_coef_energy import ipc_piecewise

    sdf_nav._ipc_dbdd = lambda d, dh: ipc_piecewise(d, dh)[1]
    try:
        alt["ipc_piecewise_clamp"] = grad_norms(scene.field, data, rr, window=0, col_grad=False)
    finally:
        sdf_nav._ipc_dbdd = orig_dbdd
    out["alternatives"] = alt
    print("alternatives", {k: v["loss"] for k, v in alt.items()}, flush=True)
    out["city"] = {
        "before_fix": grad_norms(cf, cdata, cmeta["rr"], window=0, col_grad=False),
        "fixed": grad_norms(cf, cdata, cmeta["rr"], window=7, col_grad=True),
    }
    print("city", {k: v["loss"] for k, v in out["city"].items()}, flush=True)
    out["per_sample"] = per_sample(scene.field, data, rr)
    print(
        "per_sample",
        {k: v for k, v in out["per_sample"].items() if k != "worst_pair_min_clearance"},
    )
    out["jacobian"] = jacobian(scene, rr)
    print("jacobian", json.dumps(out["jacobian"], indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
