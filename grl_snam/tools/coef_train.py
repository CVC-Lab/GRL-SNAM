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
    """The training world. NOTE ``grid``: ``shrunk`` scales the city's rects
    with the grid, so a small enough world has no buildings left. Measured
    fraction of random starts inside the barrier band (phi - rr < d_hat): 0.00
    at n=32, 0.31 at n=96. Below ~64 the barrier is identically zero, ``alpha``
    scales nothing, its gradient is exactly 0, and training silently fits the
    goal spring alone — it looks like it is working and teaches nothing about
    walls. Use the default unless you have checked the small world still has
    geometry the agents can reach."""
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
    steps=400,
    horizon=28,
    n=192,
    lr=1e-3,
    seed=0,
    grid=96,
    w_coll=6.0,
    window=7,
    model=None,
    friction=None,
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


def train_bicycle(
    steps=300,
    horizon=24,
    n=128,
    lr=1e-3,
    seed=0,
    grid=96,
    w_coll=3.0,
    window=6,
    model=None,
    friction=None,
    veh=None,
    d_safe=None,
    scene=None,
):
    """Fine-tune the coefficients through the VEHICLE, with grip in the dynamics.

    :func:`train` integrates :func:`sdf_nav.sdf_rollout` -- a holonomic point,
    which has no actuator envelope for grip to limit -- so mu there is an input
    the loss has no reason to use. This loop integrates
    :func:`sdf_nav.bicycle_rollout` instead, where mu scales BOTH ``a_max`` and
    ``a_lat_max``, and that is what makes anticipation learnable.

    THE LOSS. Both terms are normalized to O(1), and that is load-bearing
    rather than cosmetic. The original form -- ``goal_dist + w_coll * mean(relu(
    rr - phi))`` -- could not train this at all, for two multiplying reasons:

    * **Units.** ``goal_dist`` spans the WORLD (~5.5 normalized); the penalty is
      a breach depth bounded by ``rr`` = 0.15. ~36x apart before anything else.
    * **Sparsity.** ``goal_dist`` is nonzero for every agent every tick; a breach
      depth is nonzero only while actually inside geometry. Measured at random
      free-space starts: **0.00%** of agents. Averaging over the batch divides
      the term by another ~100x, and you cannot up-weight a term that is zero.

    Together that put the collision term at **0.2-0.7% of the loss**, so training
    minimised goal distance alone -- and since slowing costs goal distance, it
    was correctly learning NOT to slow down. A balanced weight would have been
    ~3,700 on ice and ~55,000 on dry: no single constant works across surfaces.

    So the penalty is now a **margin shortfall**, ``relu(d_safe - clearance) /
    d_safe``, nonzero for any agent within ``d_safe`` of geometry (21% of random
    starts at the default ``d_safe = d_hat``, against 0.00%), and ``goal_dist``
    is divided by the world half-extent. With both O(1), ``w_coll`` finally
    expresses a preference and spans the trade with small numbers. Three
    training seeds each, 120 steps, ice-bearing city; the raw column is there
    because one row is not honestly summarised by its mean:

        w_coll   reach          pen/agent      raw pen
        seed     14.6%            1.56
        0        18.9% +-0.2      2.05 +-0.17  2.25 2.05 1.83   goal only:
        1        20.1% +-0.2      1.54 +-0.01  1.55 1.55 1.53   more reach,
        3        20.1% +-0.2      1.52 +-0.03  1.48 1.54 1.54   MORE collisions
        10       11.6% +-5.8      0.57 +-0.70  0.06 0.10 1.56   BIMODAL
        30        5.2% +-2.7      0.09 +-0.06  0.05 0.05 0.17

    w_coll 1-3 is a reliable Pareto improvement on the seed -- more reach AND
    less penetration, tight across seeds -- and 3 is the default. w_coll = 10 is
    bimodal rather than noisy: two seeds find a cautious policy, the third never
    leaves the seed, so train several and select, or use 30, which reaches ~0.09
    dependably and spends two thirds of the reach doing it.

    This demonstrates the LOSS, not the mu feature. Against a blind 5-feature
    net -- same dynamics, both arms driving on ice -- mu is a wash (w=3:
    20.1%/1.52 seeing mu vs 19.6%/1.58 blind; w=10: 11.6%/0.57 vs 11.1%/0.55).
    The gain is the objective; anticipation is still unproven.

    ``clearance`` is the min over the BODY when ``veh`` carries a footprint, and
    the reference point otherwise -- the training signal is then the same
    quantity ``FogScenario.body_clearance_m`` reports, so what is optimised and
    what is published cannot drift apart.

    Pick ``w_coll`` deliberately: it selects a point on a safety-for-reach curve,
    and every metric this project publishes measures only the reach side.

    ``scene`` overrides the procedural city with any ``(field, meta, rand_on)``
    triple, where ``rand_on(n, rng)`` returns ``(n, 2)`` normalized starts and
    ``meta`` carries rr/d_hat/dt/vmax/region/scale. **Check ``d_hat`` against the
    new map before trusting a run.** It is a distance in normalized units, not a
    fraction: the city's 0.35 covers 98.7% of free space on the Austin SDF, so
    the wall barrier never turns off there. (Measured: shrinking it does NOT
    help -- route-less reach on Austin is ~3% at every d_hat because the failure
    is stalling in local minima, not the barrier. That map needs a route spine,
    not a tuning pass.)

    Pass a :func:`sdf_nav.widen_coef_mlp` copy as ``model`` together with
    ``friction`` so the net can SEE mu; a 5-input net trains fine here too, it
    just cannot anticipate -- it can only react once it is already sliding.

    ``grid`` MUST be coarse enough to still contain geometry. ``shrunk`` scales
    the city's rects with the grid, and by n=32 almost nothing survives: with no
    cell inside the barrier band, ``alpha`` scales a force that is identically
    zero and its gradient is exactly 0 -- the net trains on the goal spring
    alone and silently learns nothing about walls. Measured on the shrunk city,
    fraction of starts inside the band: 0.00 at n=32, 0.31 at n=96. Keep the
    default unless you have checked that a smaller world still has walls in it.
    """
    torch.manual_seed(seed)
    # `scene` is the seam for training on real geometry -- an Austin
    # nav_sdf.npz, a generated terrain, anything yielding (field, meta,
    # rand_on). Without it this trainer can only see the procedural city,
    # which is also the only world its published numbers describe.
    field, meta, rand_on = _scene(grid, seed) if scene is None else scene
    rr, d_hat, dt, vmax = meta["rr"], meta["d_hat"], meta["dt"], meta["vmax"]
    kw = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8, allow_reverse=True)
    kw.update(veh or {})
    # Both loss terms are normalized to O(1) so w_coll is a PREFERENCE, not a
    # unit conversion. See the docstring for the measurement behind this.
    region_n = float(meta["region"]) * float(meta["scale"])
    d_safe = float(d_hat if d_safe is None else d_safe)
    body_offsets = kw.get("body_offsets")
    body_rr = kw.get("body_rr")

    def _clearance(o_, th_):
        """Min clearance over the BODY, differentiable. Falls back to the
        reference point when no footprint is configured."""
        if not body_offsets:
            phi_, _ = field.sample(o_)
            return phi_ - rr
        head = torch.stack([torch.cos(th_), torch.sin(th_)], -1)
        worst = None
        for off in body_offsets:
            phi_, _ = field.sample(o_ + float(off) * head)
            c = phi_ - float(body_rr if body_rr else rr)
            worst = c if worst is None else torch.minimum(worst, c)
        return worst

    model = sdf_nav.CoefMLP() if model is None else model
    wants_mu = getattr(model, "in_dim", 5) == 6
    if wants_mu and friction is None:
        raise ValueError("a 6-feature model needs friction= to supply its mu column")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    last = (0.0, 0.0)
    for step in range(steps):
        o = torch.from_numpy(rand_on(n, rng))
        goal = torch.from_numpy(rand_on(n, rng))
        th = torch.from_numpy(rng.uniform(-np.pi, np.pi, n).astype(np.float32))
        # A spread of starting speeds, not rest: training only from standstill
        # would fit the coefficients to the launch transient, where a_long sits
        # against the actuator clamp, rather than to the cruising regime the
        # vehicle spends its time in.
        sp = torch.from_numpy(rng.uniform(0.0, float(vmax), n).astype(np.float32))
        coll = torch.zeros(())
        for t in range(horizon):
            feat = sdf_nav.coef_feats(field, o, goal, friction=friction if wants_mu else None)
            al, be, ga = model(feat)
            o, th, sp, _ = sdf_nav.bicycle_rollout(
                field,
                o,
                th,
                sp,
                goal,
                al,
                be,
                ga,
                1,
                rr=rr,
                d_hat=d_hat,
                dt=dt,
                vmax=vmax,
                friction=friction,
                **kw,
            )
            # Margin shortfall, not breach depth: nonzero for any agent within
            # d_safe of geometry, which is where the signal has to live.
            coll = coll + (torch.relu(d_safe - _clearance(o, th)) / d_safe).mean()
            if (t + 1) % window == 0 or t == horizon - 1:
                goal_loss = (goal - o).norm(dim=1).mean() / region_n
                loss = goal_loss + w_coll * coll / window
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                # The two summands AS APPLIED, so a caller can check the
                # objective is balanced without scraping stdout.
                last = (goal_loss.item(), float(w_coll * coll.detach() / window))
                o, th, sp, coll = o.detach(), th.detach(), sp.detach(), torch.zeros(())
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}: goal_dist {last[0]:6.3f}  coll {last[1]:6.3f}")
    model.eval()
    model.last_loss_terms = last
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
    ap.add_argument(
        "--rollout",
        choices=["surrogate", "bicycle"],
        default="surrogate",
        help="surrogate: the holonomic point. bicycle: integrate the vehicle, which is "
        "the only path where grip and the footprint can reach the loss.",
    )
    ap.add_argument(
        "--w-coll",
        type=float,
        default=3.0,
        help="bicycle rollout only: the safety-for-reach dial. 1-3 improves on the seed "
        "in both; 30 cuts penetration ~94%% and spends two thirds of the reach. This "
        "selects an operating point -- it is not a hyperparameter to tune away.",
    )
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

    print(f"training ({args.rollout}, {args.steps} steps)...")
    if args.rollout == "bicycle":
        # Grip and the footprint have no way into the loss through the holonomic
        # surrogate, so this is the path that can learn about either. Pass a
        # FrictionField or a footprint programmatically -- there is no flag for
        # them because both need a world, not a scalar.
        model = train_bicycle(
            args.steps, args.horizon, args.n, args.lr, args.seed, w_coll=args.w_coll
        )
    else:
        model = train(args.steps, args.horizon, args.n, args.lr, args.seed)
    write_coef_mlp(model, args.out)
    print(f"wrote {args.out}   reach_rate={reach_rate(model):.2%}")


if __name__ == "__main__":
    main()
