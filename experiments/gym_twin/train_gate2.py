"""The gym twin's training gate, second design (docs/gym-twin.md, gate 4).

The first run (``train_gate.py``) failed: its rollouts started anywhere and lasted 30 ticks, so in
114 of 150 iterations the policy got no gradient at all. The coefficients act only near walls
(alpha's barrier) and when closing on the goal (beta's pull, gamma's damping); in open ground the
drive accelerates at its limit and steers at the goal whatever they are. Its loss also paid for
closing fast and not for stopping. This design changes four things:

* **Where rollouts start.** Each iteration drives a training episode in the gym with the current
  policy (``--episode-ticks``), recording every tick's state. It then starts rollouts from the
  states where the coefficients act: within ``--wall-m`` of a wall, or within ``--approach-m``
  of the goal and not there yet. If there are none, from random ticks. Up to ``--starts`` per
  agent are batched, each with its agent's goal.
* **What the loss pays for.** :func:`grl_snam.gym_twin.arrival_loss`: ending inside the reach,
  stopped, plus a small pull while still outside it; and the clearance barrier.
* **How long.** ``--horizon`` ticks (90 by default, 5.4 s), long enough to brake to a stop from
  the approach.
* **A curve.** Every ``--probe-every`` iterations the policy is judged in the gym on a probe set
  of held-out layouts and seeds, disjoint from the final judging set.

The worlds are denser than the first run's (18 boxes on a 120 m square), so walls are met. The
arms are judged as before by driving the gym itself: ``nav`` (nav@1), ``constant``, ``untrained``
and ``trained``, with paired bootstrap intervals (:mod:`grl_snam.learning.stats`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time

import numpy as np
import torch

import pycvc
from grl_snam.gym_twin import (
    GymTwin,
    arrival_loss,
    clearance_loss,
    constant_policy,
    drive_gym,
    mlp_policy,
)
from grl_snam.learning.stats import paired_diff, summarize
from grl_snam.sdf_nav import CoefMLP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gate import judge, world_doc  # noqa: E402

MEASURES = ("left_m", "progress_m", "arrived", "collided_ticks")


def pick_starts(twin: GymTwin, states, goals, k: int, wall_m: float, approach_m: float, rng):
    """Up to ``k`` start rows per agent from the recorded ``states`` (a list of ``[n, 14]``): the
    ticks where the coefficients act (near a wall, or closing on the goal), else random ticks.
    Returns ``(state [B, 14], goal [B, 2], kinds)``."""
    if not states:
        return None, None, []
    S = torch.stack(states, 0)  # [T, n, 14]
    T, n, _ = S.shape
    rows, gl, kinds = [], [], []
    for a in range(n):
        xy = S[:, a, :2]
        clear = twin.clearance(xy)
        gd = (xy - goals[a][None]).norm(dim=-1)
        near_wall = [t for t in range(T) if float(clear[t]) < wall_m]
        approach = [t for t in range(T) if twin.reach < float(gd[t]) < approach_m]
        pool = [(t, "wall") for t in near_wall] + [(t, "approach") for t in approach]
        if not pool:
            pool = [(t, "random") for t in range(T)]
        for t, kind in rng.sample(pool, min(k, len(pool))):
            rows.append(S[t, a])
            gl.append(goals[a])
            kinds.append(kind)
    return torch.stack(rows, 0), torch.stack(gl, 0), kinds


def summary_of(rows) -> dict:
    return {m: summarize([r[m] for r in rows]) for m in MEASURES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--half", type=float, default=60.0, help="half the square's side (m)")
    ap.add_argument("--boxes", type=int, default=18)
    ap.add_argument("--train-worlds", type=int, nargs=2, default=(0, 8), metavar=("FROM", "TO"))
    ap.add_argument(
        "--heldout-worlds", type=int, nargs=2, default=(100, 104), metavar=("FROM", "TO")
    )
    ap.add_argument("--probe-worlds", type=int, nargs=2, default=(200, 202), metavar=("FROM", "TO"))
    ap.add_argument("--train-seeds", type=int, nargs=2, default=(0, 1000), metavar=("FROM", "TO"))
    ap.add_argument(
        "--heldout-seeds", type=int, nargs=2, default=(5000, 5010), metavar=("FROM", "TO")
    )
    ap.add_argument(
        "--probe-seeds", type=int, nargs=2, default=(7000, 7003), metavar=("FROM", "TO")
    )
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--episode-ticks", type=int, default=400, help="gym ticks driven per iteration")
    ap.add_argument("--starts", type=int, default=4, help="rollout starts per agent per iteration")
    ap.add_argument("--wall-m", type=float, default=8.0)
    ap.add_argument("--approach-m", type=float, default=30.0)
    ap.add_argument("--horizon", type=int, default=90, help="ticks backpropagated per rollout")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-clear", type=float, default=1.0)
    ap.add_argument("--w-stop", type=float, default=0.5)
    ap.add_argument("--ticks", type=int, default=600, help="ticks each judged episode runs")
    ap.add_argument("--probe-every", type=int, default=30)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--out", default="gym_twin_gate2.json")
    a = ap.parse_args(argv)

    torch.manual_seed(a.torch_seed)
    rng = random.Random(a.torch_seed)
    tmp = tempfile.mkdtemp(prefix="gym_twin_gate2_")
    train_worlds = list(range(*a.train_worlds))
    held_worlds = list(range(*a.heldout_worlds))
    probe_worlds = list(range(*a.probe_worlds))
    agents = [{"plant": "dynamic4w"} for _ in range(a.agents)]
    gyms = {
        w: pycvc.WorldGym(world_doc(w, a.half, a.boxes, tmp), agents=agents, max_steps=100000)
        for w in train_worlds + held_worlds + probe_worlds
    }

    model = CoefMLP().double()
    init = CoefMLP().double()
    init.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    policy = mlp_policy(model)
    probe_seeds = list(range(*a.probe_seeds))
    history, curve = [], []
    t0 = time.time()

    def probe(it):
        rows = judge(gyms, probe_worlds, probe_seeds, policy, a.ticks)
        s = summary_of(rows)
        curve.append({"iter": it, **{m: s[m]["mean"] for m in MEASURES}})
        print(
            f"  probe @ {it:4d}: progress {s['progress_m']['mean']:7.2f} m  "
            f"arrived {s['arrived']['mean']:.3f}  left {s['left_m']['mean']:6.2f} m",
            flush=True,
        )

    probe(0)
    seeds = list(range(*a.train_seeds))
    for it in range(a.iters):
        w = train_worlds[it % len(train_worlds)]
        g = gyms[w]
        g.reset(seeds[it % len(seeds)])
        twin = GymTwin(g)
        recorded = []
        drive_gym(g, twin, policy, a.episode_ticks, states=recorded)
        s0, goal, kinds = pick_starts(
            twin, recorded, twin.goal, a.starts, a.wall_m, a.approach_m, rng
        )
        states, _ = twin.rollout(policy, s0, a.horizon, goal=goal)
        la = arrival_loss(twin, states, goal, w_stop=a.w_stop)
        lc = clearance_loss(twin, states)
        loss = la + a.w_clear * lc
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        opt.step()
        history.append(
            {
                "iter": it,
                "world": w,
                "loss": loss.item(),
                "arrival": la.item(),
                "clearance": lc.item(),
                "grad_norm": gn,
                "rows": len(kinds),
                "wall": kinds.count("wall"),
                "approach": kinds.count("approach"),
                "random": kinds.count("random"),
            }
        )
        if it % 10 == 0 or it + 1 == a.iters:
            print(
                f"iter {it:4d} loss {loss.item():8.3f} arrival {la.item():8.3f} "
                f"clear {lc.item():7.3f} |g| {gn:9.3g} rows {len(kinds)} "
                f"(wall {kinds.count('wall')}, approach {kinds.count('approach')}) "
                f"{time.time() - t0:6.0f} s",
                flush=True,
            )
        if (it + 1) % a.probe_every == 0 or it + 1 == a.iters:
            probe(it + 1)

    held_seeds = list(range(*a.heldout_seeds))
    arms = {
        "constant": constant_policy((1.0, 3.0, 4.0)),
        "untrained": mlp_policy(init),
        "trained": mlp_policy(model),
    }
    results = {name: judge(gyms, held_worlds, held_seeds, p, a.ticks) for name, p in arms.items()}
    results["nav"] = judge(gyms, held_worlds, held_seeds, None, a.ticks, nav=True)
    summary = {arm: summary_of(rows) for arm, rows in results.items()}
    diffs = {
        f"trained-{ref}": {
            m: paired_diff([r[m] for r in results[ref]], [r[m] for r in results["trained"]])
            for m in MEASURES
        }
        for ref in ("untrained", "constant", "nav")
    }
    live = sum(1 for h in history if h["grad_norm"] > 0)
    report = {
        "args": vars(a),
        "history": history,
        "curve": curve,
        "summary": summary,
        "paired": diffs,
        "episodes": results,
        "live_gradient_iters": live,
        "seconds": time.time() - t0,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"iterations with a gradient: {live} of {len(history)}")
    for arm in ("nav", "constant", "untrained", "trained"):
        s = summary[arm]
        print(
            f"{arm:10s} left {s['left_m']['mean']:7.2f} m  progress {s['progress_m']['mean']:7.2f} m"
            f"  arrived {s['arrived']['mean']:.3f}  collided {s['collided_ticks']['mean']:7.2f}"
        )
    for k, v in diffs.items():
        p, r = v["progress_m"], v["arrived"]
        print(
            f"{k:18s} progress {p['mean']:+7.2f} m [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]"
            f"  arrived {r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
