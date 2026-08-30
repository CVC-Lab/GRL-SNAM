"""Self-supervised training of the CoefMLP navigation policy — NO labels, NO
dataset. Differentiable rollouts over a scene's SDF (sdf_nav.sdf_rollout) give a
gradient straight from "did the agent reach its goal without hitting a wall" into
the coefficient net. Exports the result to the versioned .cvcnav a pure-C++ host
loads (coef_mlp::default_weights_path).

    python -m grl_snam.tools.coef_train --steps 400 --out coef_mlp.cvcnav

The net is bias-initialized toward a known-good basin, so this refines rather than
learns-from-scratch; a longer run on a bigger box (more steps / horizon / scenes)
sharpens it further.

NOTE: uses truncated BPTT through the differentiable rollout. Some fragile CPU
torch builds segfault in the grid_sample backward on long graphs — run on a
stable torch / a GPU box. The shipped reference policy
(libcvc share/cvc/nav/coef_mlp.cvcnav) is the bias-basin CoefMLP exported with
grl_snam.tools.coef_export; retrain here and re-export to replace it.
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    raise SystemExit("coef_train needs torch")

import sdf_nav

from .. import planner
from ..fog_stories import STORIES, shrunk
from .coef_export import write_coef_mlp


def _scene(grid, seed):
    story = shrunk(STORIES["city"], n=grid, max_steps=100)
    truth = story.truth_grid()
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(truth, story.bounds, meta["scale"])
    field = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape
    S = meta["scale"]
    cx, cy = meta["center"]

    def rand_on(n, rng):
        idx = rng.integers(0, len(rows), n)
        wx = mnx + cols[idx] / (nx - 1) * (mxx - mnx)
        wy = mny + rows[idx] / (ny - 1) * (mxy - mny)
        return np.stack([(wx - cx) * S, (wy - cy) * S], 1).astype(np.float32)

    return field, meta, rand_on


def train(
    steps=400, horizon=28, n=192, lr=1e-3, seed=0, grid=96, w_coll=6.0, window=7,
    model=None, friction=None,
):
    """Truncated BPTT: the rollout is `horizon` steps but the autograd graph is
    detached every `window` steps (bounded memory; a long full-BPTT graph OOMs).

    ``model`` fine-tunes an existing net instead of a fresh init -- the seam a
    grip-widened net needs, since training this policy from scratch is known to
    collapse reach against the shipped seed, while a
    :func:`sdf_nav.widen_coef_mlp` copy starts output-identical to a net that
    works. ``friction`` then feeds mu in as the sixth feature.

    CAVEAT, and it is the whole reason this is a seam and not a result: this
    loop integrates :func:`sdf_nav.sdf_rollout`, a HOLONOMIC POINT. Grip does
    not enter a point's dynamics at all -- there is no actuator envelope to
    limit -- so mu here is an input the loss has no reason to use. Teaching
    anticipation needs a bicycle-rollout loop with grip in the dynamics AND a
    term that punishes arriving at ice too fast. That harness does not exist
    yet; do not read a fine-tune on this one as evidence either way.
    """
    torch.manual_seed(seed)
    field, meta, rand_on = _scene(grid, seed)
    rr, d_hat, dt, vmax = meta["rr"], meta["d_hat"], meta["dt"], meta["vmax"]
    model = sdf_nav.CoefMLP() if model is None else model
    if friction is not None and getattr(model, "in_dim", 5) != 6:
        raise ValueError(
            "friction= needs a 6-feature model; call sdf_nav.widen_coef_mlp(model) first"
        )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        o = torch.from_numpy(rand_on(n, rng))
        goal = torch.from_numpy(rand_on(n, rng))
        v = torch.zeros(n, 2)
        last_goal, last_coll = 0.0, 0.0
        coll = torch.zeros(())
        for t in range(horizon):
            feat = sdf_nav.coef_feats(field, o, goal, friction=friction)
            al, be, ga = model(feat)
            o, v, _ = sdf_nav.sdf_rollout(
                field, o, v, goal, al, be, ga, 1, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax
            )
            phi_o, _ = field.sample(o)
            coll = coll + torch.relu(rr - phi_o).mean()  # penalize wall penetration
            if (t + 1) % window == 0 or t == horizon - 1:
                goal_loss = (goal - o).norm(dim=1).mean()
                loss = goal_loss + w_coll * coll / window
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                last_goal, last_coll = goal_loss.item(), float(coll.detach())
                o, v, coll = o.detach(), v.detach(), torch.zeros(())
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}: goal_dist {last_goal:6.3f}  coll {last_coll:6.3f}")
    model.eval()
    return model


def reach_rate(model, grid=96, n=400, ticks=200, seed=123):
    """A quick bicycle-drive eval: fraction of agents that reach their goal."""
    from grl_snam.squad import AgentSpec
    from grl_snam.swarm import Swarm

    story = shrunk(STORIES["city"], n=grid, max_steps=10_000_000)
    truth = story.truth_grid()
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    rng = np.random.default_rng(seed)
    s, g = rng.integers(0, len(rows), n), rng.integers(0, len(rows), n)
    specs = [
        AgentSpec(f"a{i}", w(rows[s[i]], cols[s[i]]), w(rows[g[i]], cols[g[i]])) for i in range(n)
    ]
    sw = Swarm(story, specs, model=model, truth_occ=truth, prior_occ=truth, belief_mode="shared")
    sw._sense_shared = lambda: None
    for _ in range(ticks):
        sw.step()
    return float(sw.reached.float().mean())


def train_native(out, *, grid=96, steps=400, rollout="surrogate", use_cuda=False, lr=None, seed=0):
    """Train via the pure-C++ ``cvc::nav`` trainer (NO torch) on the SAME city scene
    the torch path uses, writing the ``.cvcnav`` to ``out``. This is what
    ``GRL_SNAM_TRAIN_BACKEND=native`` (or ``--backend native``) dispatches to; the
    torch :func:`train` stays canonical. ``rollout`` is ``"surrogate"`` (default) or
    ``"bicycle"`` (the deployment integrator; auto-uses a lower lr). Needs a pycvc
    built with ``cvc::nav::coef_train``."""
    from grl_snam import nav_native

    if not nav_native.HAS_TRAIN:
        raise SystemExit(
            "native trainer needs a pycvc built with cvc::nav::coef_train; "
            "use --backend torch or install a newer pycvc"
        )
    story = shrunk(STORIES["city"], n=grid, max_steps=100)
    meta = story.meta()
    if lr is None:
        lr = 1e-5 if str(rollout).lower() == "bicycle" else 2e-4
    return nav_native.train_coef_mlp(
        story.truth_grid().astype(np.uint8),
        out,
        bounds=story.bounds,
        scale=meta["scale"],
        rr=meta["rr"],
        d_hat=meta["d_hat"],
        dt=meta["dt"],
        vmax=meta["vmax"],
        steps=steps,
        lr=lr,
        seed=seed,
        rollout=rollout,
        use_cuda=use_cuda,
    )


def main(argv=None):
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="coef_mlp.cvcnav")
    # Feature flag: which trainer runs. `torch` (canonical) or `native` (the
    # torch-free libcvc cvc::nav trainer, via pycvc). Env default lets a whole
    # pipeline opt in without changing call sites.
    ap.add_argument(
        "--backend",
        choices=["torch", "native"],
        default=os.environ.get("GRL_SNAM_TRAIN_BACKEND", "torch"),
    )
    ap.add_argument("--rollout", choices=["surrogate", "bicycle"], default="surrogate")
    ap.add_argument("--cuda", action="store_true", help="native backend: use the GPU trainer")
    args = ap.parse_args(argv)

    if args.backend == "native":
        lr = args.lr if args.lr != 1e-3 else None  # 1e-3 is the torch default; let native pick
        print(f"training natively ({args.rollout}, {args.steps} steps, cuda={args.cuda})...")
        train_native(
            args.out,
            steps=args.steps,
            rollout=args.rollout,
            use_cuda=args.cuda,
            lr=lr,
            seed=args.seed,
        )
        print(f"wrote {args.out} (native cvc::nav)")
        return

    print(f"training ({args.steps} steps)...")
    model = train(args.steps, args.horizon, args.n, args.lr, args.seed)
    write_coef_mlp(model, args.out)
    print(f"wrote {args.out}   reach_rate={reach_rate(model):.2%}")


if __name__ == "__main__":
    main()
