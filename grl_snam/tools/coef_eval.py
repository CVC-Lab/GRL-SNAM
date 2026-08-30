"""Score a coefficient policy on BOTH axes it trades between, over several seeds.

    python -m grl_snam.tools.coef_eval --sweep-w 0,1,3,10,30      # the dial
    python -m grl_snam.tools.coef_eval --compare-footprint        # body vs point
    python -m grl_snam.tools.coef_eval --compare-mu               # is mu earning?

Every headline metric this project publishes -- ``reach_rate``,
``ScenarioResult.reached`` -- measures only reach. A policy trained for safety
therefore looks like a pure regression on all of them while being several times
less likely to hit anything, so a reach number on its own cannot say whether a
change was good. This reports the pair, and the RAW per-seed values with it.

The raw column is not decoration. The ``w_coll = 10`` row of the sweep below
returns penetrations of 0.06, 0.10 and 1.56: two seeds converge on a cautious
policy and the third never leaves the seed. Its mean, 0.57, describes no policy
that exists. Three seeds is the floor, and a single-seed number from here should
not be quoted.

Penetration counts agent-ticks with clearance below half the stopping margin,
and clearance is the MIN over the body when a footprint is configured -- the
same quantity ``FogScenario.body_clearance_m`` reports.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import sdf_nav
from grl_snam import planner
from grl_snam.fog_stories import STORIES, shrunk
from grl_snam.material import FrictionField

VEH = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8, allow_reverse=True)
# A three-disc body at full radius. body_gain MUST be 1/n_body: the barrier is
# summed over discs while `al` was fit for a single sample point, so without it
# the learned force is multiplied by the disc count.
FOOTPRINT = dict(body_offsets=(-0.05, 0.0, 0.05), body_rr=0.15, body_gain=1.0 / 3.0)


def ice_band_world(grid=96, mu=0.18, seed=0):
    """The city with a band of ice across its middle third, so most routes cross
    a grip transition rather than sitting entirely on one surface. A uniform
    field would make the mu feature a constant and could not test anticipation."""
    story = shrunk(STORIES["city"], n=grid, max_steps=100)
    truth = story.truth_grid()
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(truth, story.bounds, meta["scale"])
    field = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    labels, sizes = planner.free_components(truth, 2)
    rows, cols = np.nonzero(labels == max(sizes, key=sizes.get))
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape
    scale, (cx, cy) = meta["scale"], meta["center"]

    def rand_on(n, rng):
        idx = rng.integers(0, len(rows), n)
        wx = mnx + cols[idx] / (nx - 1) * (mxx - mnx)
        wy = mny + rows[idx] / (ny - 1) * (mxy - mny)
        return np.stack([(wx - cx) * scale, (wy - cy) * scale], 1).astype(np.float32)

    grip = np.full(truth.shape, 1.0, np.float32)
    grip[:, nx // 3 : 2 * nx // 3] = float(mu)
    friction = FrictionField(grip, story.bounds, meta["center"], meta["scale"])
    return field, meta, rand_on, friction


def _clearance(field, o, th, veh):
    """Min clearance over the body, or the reference point if there is none."""
    offs = veh.get("body_offsets") if veh else None
    if not offs:
        phi, _ = field.sample(o)
        return phi - 0.15
    rr_b = float(veh.get("body_rr", 0.15))
    head = torch.stack([torch.cos(th), torch.sin(th)], -1)
    worst = None
    for off in offs:
        phi, _ = field.sample(o + float(off) * head)
        c = phi - rr_b
        worst = c if worst is None else torch.minimum(worst, c)
    return worst


def evaluate(model, field, meta, rand_on, *, friction=None, veh=None, n=192, ticks=90, seed=4242):
    """Drive n agents and report reach, penetration, worst clearance, final gap."""
    rr, d_hat, dt, vmax = meta["rr"], meta["d_hat"], meta["dt"], meta["vmax"]
    kw = dict(VEH, **(veh or {}))
    wants_mu = getattr(model, "in_dim", 5) == 6
    rng = np.random.default_rng(seed)
    o = torch.from_numpy(rand_on(n, rng))
    goal = torch.from_numpy(rand_on(n, rng))
    th = torch.from_numpy(rng.uniform(-np.pi, np.pi, n).astype(np.float32))
    sp = torch.zeros(n)
    reached = torch.zeros(n, dtype=torch.bool)
    pen, worst = torch.zeros(n), torch.full((n,), 1e9)
    margin = 0.5 * float(kw.get("body_rr", rr) if kw.get("body_offsets") else rr)
    with torch.no_grad():
        for _ in range(ticks):
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
            c = _clearance(field, o, th, kw)
            pen += (c < margin).float()
            worst = torch.minimum(worst, c)
            reached |= (goal - o).norm(dim=1) < 0.15
    return dict(
        reach=float(reached.float().mean()),
        pen=float(pen.mean()),
        clear=float(worst.mean()),
        gap=float((goal - o).norm(dim=1).mean()),
    )


def run_arms(arms, *, seeds=3, grid=96, steps=120, horizon=16, n_train=64, window=4, mu=0.18):
    """Train and score each arm over `seeds` seeds. An arm is
    (label, model_factory, train_kwargs, eval_veh); model_factory takes the seed
    net and returns the net to train, so an arm can widen it or leave it alone."""
    from grl_snam.tools import coef_train

    field, meta, rand_on, friction = ice_band_world(grid, mu)
    torch.manual_seed(0)
    seed_net = sdf_nav.CoefMLP()
    rows = []
    for label, factory, tkw, eveh in arms:
        got = []
        for s in range(seeds):
            torch.manual_seed(s)
            model = factory(seed_net)
            model = coef_train.train_bicycle(
                model=model,
                friction=friction,
                steps=steps,
                horizon=horizon,
                n=n_train,
                window=window,
                grid=grid,
                seed=s,
                **tkw,
            )
            got.append(evaluate(model, field, meta, rand_on, friction=friction, veh=eveh))
        rows.append((label, got))
    return rows, (field, meta, rand_on, friction, seed_net)


def _print(rows, baseline=None):
    hdr = f"{'arm':>26} {'reach':>8} {'pen':>8} {'clear':>8}   raw pen"
    print(hdr)
    print("-" * len(hdr))
    if baseline is not None:
        print(
            f"{'seed (untrained)':>26} {baseline['reach']*100:7.1f}% {baseline['pen']:8.2f} "
            f"{baseline['clear']:8.3f}"
        )
    for label, got in rows:
        r = np.mean([g["reach"] for g in got]) * 100
        p = [g["pen"] for g in got]
        c = np.mean([g["clear"] for g in got])
        print(
            f"{label:>26} {r:7.1f}% {np.mean(p):8.2f} {c:8.3f}   "
            f"{[round(x, 2) for x in p]}  +-{np.std(p):.2f}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3, help="3 is the floor; 1 is not quotable")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--grid", type=int, default=96)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep-w", type=str, help="comma-separated w_coll values")
    g.add_argument(
        "--compare-footprint",
        action="store_true",
        help="train on the point vs on the body; BOTH scored with the body",
    )
    g.add_argument(
        "--compare-mu",
        action="store_true",
        help="blind 5-feature net vs a widened 6-feature one, same dynamics",
    )
    args = ap.parse_args(argv)

    plain = lambda net: __import__("copy").deepcopy(net)  # noqa: E731
    wide = lambda net: sdf_nav.widen_coef_mlp(net)  # noqa: E731

    if args.sweep_w:
        arms = [(f"w_coll={w}", wide, dict(w_coll=float(w)), None) for w in args.sweep_w.split(",")]
        eveh = None
    elif args.compare_footprint:
        # Both arms are SCORED with the footprint; only the training differs.
        arms = [
            ("trained on the POINT", plain, dict(w_coll=3.0), FOOTPRINT),
            ("trained on the BODY", plain, dict(w_coll=3.0, veh=dict(VEH, **FOOTPRINT)), FOOTPRINT),
        ]
        eveh = FOOTPRINT
    else:
        arms = [
            ("blind (5-feature)", plain, dict(w_coll=3.0), None),
            ("mu-ahead (6-feature)", wide, dict(w_coll=3.0), None),
        ]
        eveh = None

    rows, (field, meta, rand_on, friction, seed_net) = run_arms(
        arms, seeds=args.seeds, grid=args.grid, steps=args.steps
    )
    base = evaluate(seed_net, field, meta, rand_on, friction=friction, veh=eveh)
    _print(rows, base)


if __name__ == "__main__":
    main()
