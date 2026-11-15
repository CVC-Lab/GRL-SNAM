"""The gym twin's training gate (docs/gym-twin.md, gate 4).

Does a coefficient policy (:class:`grl_snam.sdf_nav.CoefMLP`) trained through the twin of libcvc's
gym (:mod:`grl_snam.gym_twin`) drive better in the gym itself than its untrained init and than
the constant coefficients it is biased toward? It is judged on held-out worlds and seeds, with
paired bootstrap intervals (:mod:`grl_snam.learning.stats`), as doc 35's protocol asks.

* **Worlds.** Flat squares with box obstacles, one layout per world seed: on flat ground the
  twin's plant is the gym's ``dynamic4w`` model (gate 1), so what differs is the driver and
  contact. Training uses world seeds ``--train-worlds``; judging uses ``--heldout-worlds``,
  never trained on.
* **Episodes.** ``--agents`` Humvees on ``dynamic4w``; the gym draws their starts and goals from
  the episode seed (nav@1's held goals). Training seeds and judging seeds do not overlap.
* **Training.** Each iteration resets a training world on the next training seed, optionally
  drives the gym a random number of ticks with the current policy (so the twin starts where the
  policy actually drives), then backpropagates ``--horizon`` ticks of the twin: progress to the
  goal plus a clearance barrier.
* **Arms, each judged in the gym** by driving it ``--ticks`` ticks with the twin's controller:
  ``constant`` (CoefMLP's bias), ``untrained`` (the init), ``trained``; and ``nav`` (the gym's own
  navigator, nav@1, untouched), for reference.

Writes a JSON report (``--out``) with each arm's measures per episode, their summaries and the
paired differences.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time

import numpy as np
import torch

import pycvc
from grl_snam.gym_twin import (
    GymTwin,
    clearance_loss,
    constant_policy,
    drive_gym,
    mlp_policy,
    progress_loss,
)
from grl_snam.learning.stats import paired_diff, summarize
from grl_snam.sdf_nav import CoefMLP

HEAD = """requires: [scene.physics]
world:
  vehicles:
    archetypes:
      humvee: { vehicle: humvee, controllers: [nav@1, none] }
scene:
  physics: {}
  world: { source: square, cell: 2 }
  nodes:
    - node: square
      world: square
      source: { box: { size: [HALF2, HALF2, 1] } }
      transform: { position: [0, 0, -0.5] }
      body: static
"""
BOX = """    - node: box{k}
      source: {{ box: {{ size: [{sx:.1f}, {sy:.1f}, 4] }} }}
      transform: {{ position: [{x:.1f}, {y:.1f}, 2] }}
      body: static
"""


def world_doc(seed: int, half: float, boxes: int, out_dir: str) -> str:
    """A flat square ``2 half`` m across with ``boxes`` box obstacles placed from ``seed``."""
    rng = random.Random(1000003 * seed + 17)
    text = HEAD.replace("HALF2", f"{2 * half:.0f}")
    for k in range(boxes):
        sx, sy = rng.uniform(6.0, 18.0), rng.uniform(6.0, 18.0)
        x, y = rng.uniform(-0.7 * half, 0.7 * half), rng.uniform(-0.7 * half, 0.7 * half)
        text += BOX.format(k=k, sx=sx, sy=sy, x=x, y=y)
    path = os.path.join(out_dir, f"world_{seed}.ari")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def episode_measures(gym, obs_log, start_obs) -> dict:
    """An episode's measures, averaged over its agents: the distance left to the goal (m), the
    progress made (m), the fraction of agents that arrived, and collided agent-ticks."""
    last = obs_log[-1] if obs_log else start_obs
    d0 = np.hypot(start_obs["goal_x"] - start_obs["x"], start_obs["goal_y"] - start_obs["y"])
    d1 = np.hypot(last["goal_x"] - last["x"], last["goal_y"] - last["y"])
    arrived = np.zeros_like(d1, dtype=bool)
    collided = 0
    for o in obs_log:
        arrived |= np.asarray(o["reached"], dtype=bool)
        collided += int(np.asarray(o["collided"]).sum())
    return {
        "left_m": float(d1.mean()),
        "progress_m": float((d0 - d1).mean()),
        "arrived": float(arrived.mean()),
        "collided_ticks": float(collided) / max(1, len(d1)),
        "ticks": len(obs_log),
    }


def judge(gyms, worlds, seeds, policy, ticks, *, nav=False) -> list[dict]:
    rows = []
    for w in worlds:
        g = gyms[w]
        for s in seeds:
            start = g.reset(s)
            if nav:
                log = []
                nan = np.full(g.agents(), np.nan)
                for _ in range(ticks):
                    out = g.step(nan, nan)
                    log.append(out[0])
                    if out[2] or out[3]:
                        break
            else:
                t = GymTwin(g)
                _, log = drive_gym(g, t, policy, ticks)
            m = episode_measures(g, log, start)
            m.update(world=w, seed=s)
            rows.append(m)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--half", type=float, default=80.0, help="half the square's side (m)")
    ap.add_argument("--boxes", type=int, default=8)
    ap.add_argument("--train-worlds", type=int, nargs=2, default=(0, 6), metavar=("FROM", "TO"))
    ap.add_argument(
        "--heldout-worlds", type=int, nargs=2, default=(100, 104), metavar=("FROM", "TO")
    )
    ap.add_argument("--train-seeds", type=int, nargs=2, default=(0, 400), metavar=("FROM", "TO"))
    ap.add_argument(
        "--heldout-seeds", type=int, nargs=2, default=(5000, 5010), metavar=("FROM", "TO")
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=30, help="ticks backpropagated per iteration")
    ap.add_argument(
        "--warmup", type=int, default=120, help="most gym ticks driven before a rollout"
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-clear", type=float, default=1.0)
    ap.add_argument("--ticks", type=int, default=600, help="ticks each judged episode runs")
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--out", default="gym_twin_gate.json")
    a = ap.parse_args(argv)

    torch.manual_seed(a.torch_seed)
    rng = random.Random(a.torch_seed)
    tmp = tempfile.mkdtemp(prefix="gym_twin_gate_")
    train_worlds = list(range(*a.train_worlds))
    held_worlds = list(range(*a.heldout_worlds))
    agents = [{"plant": "dynamic4w"} for _ in range(a.agents)]
    gyms = {
        w: pycvc.WorldGym(world_doc(w, a.half, a.boxes, tmp), agents=agents, max_steps=100000)
        for w in train_worlds + held_worlds
    }

    model = CoefMLP().double()
    init = CoefMLP().double()
    init.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    policy = mlp_policy(model)
    history = []
    t0 = time.time()
    seeds = list(range(*a.train_seeds))
    for it in range(a.iters):
        w = train_worlds[it % len(train_worlds)]
        g = gyms[w]
        g.reset(seeds[it % len(seeds)])
        twin = GymTwin(g)
        k = rng.randrange(a.warmup + 1) if a.warmup > 0 else 0
        if k:
            drive_gym(g, twin, policy, k)
        s0 = twin.state(g)
        states, _ = twin.rollout(policy, s0, a.horizon)
        lp = progress_loss(twin, states)
        lc = clearance_loss(twin, states)
        loss = lp + a.w_clear * lc
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        opt.step()
        history.append(
            {
                "iter": it,
                "world": w,
                "warmup": k,
                "loss": loss.item(),
                "progress": lp.item(),
                "clearance": lc.item(),
                "grad_norm": gn,
            }
        )
        if it % 20 == 0 or it + 1 == a.iters:
            print(
                f"iter {it:4d} loss {loss.item():8.3f} progress {lp.item():8.3f} "
                f"clear {lc.item():7.3f} |g| {gn:9.3g}  {time.time() - t0:6.0f} s",
                flush=True,
            )

    held_seeds = list(range(*a.heldout_seeds))
    arms = {
        "constant": constant_policy((1.0, 3.0, 4.0)),
        "untrained": mlp_policy(init),
        "trained": mlp_policy(model),
    }
    results = {name: judge(gyms, held_worlds, held_seeds, p, a.ticks) for name, p in arms.items()}
    results["nav"] = judge(gyms, held_worlds, held_seeds, None, a.ticks, nav=True)
    measures = ("left_m", "progress_m", "arrived", "collided_ticks")
    summary = {
        arm: {m: summarize([r[m] for r in rows]) for m in measures} for arm, rows in results.items()
    }
    diffs = {}
    for ref in ("untrained", "constant", "nav"):
        diffs[f"trained-{ref}"] = {
            m: paired_diff([r[m] for r in results[ref]], [r[m] for r in results["trained"]])
            for m in measures
        }
    report = {
        "args": vars(a),
        "history": history,
        "summary": summary,
        "paired": diffs,
        "episodes": results,
        "seconds": time.time() - t0,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for arm in ("nav", "constant", "untrained", "trained"):
        s = summary[arm]
        print(
            f"{arm:10s} left {s['left_m']['mean']:7.2f} m  progress {s['progress_m']['mean']:7.2f} m"
            f"  arrived {s['arrived']['mean']:.3f}  collided {s['collided_ticks']['mean']:7.2f}"
        )
    for k, v in diffs.items():
        p = v["progress_m"]
        print(
            f"{k:18s} progress {p['mean']:+7.2f} m [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]"
            f"  arrived {v['arrived']['mean']:+.3f} [{v['arrived']['ci95'][0]:+.3f}, "
            f"{v['arrived']['ci95'][1]:+.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
