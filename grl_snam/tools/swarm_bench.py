"""Reproduce the vectorized-swarm numbers, and demonstrate the threading model.

Two things this prints:

``throughput``
    how many agents fit a 60 Hz / 30 Hz tick, for the drive path (the 3-of-4
    ticks with ``sense_every=4``, on a static shared map) and the current
    steady state (shared belief update + single-EDT rebuild every 4th tick).
    The single shared EDT rebuild is O(1) in N — that is the shared-belief win;
    the remaining per-agent cost is the Python raycast loop, which the planned
    C++ ``sense_batch`` kernel removes.

``threading``
    the sim runs on a background :class:`~grl_snam.sim_thread.SimThread` while
    this thread reads immutable snapshots lock-free (never blocking the sim) and
    fires a live retarget the running agents react to — the answer to "can nav
    run off the render thread and still react to the live scene?".

    python -m grl_snam.tools.swarm_bench [throughput|threading|all] [--grid N]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep of the swarm
    raise SystemExit("swarm_bench needs torch")

import sdf_nav

from .. import planner
from ..fog_stories import STORIES, shrunk
from ..sim_thread import Pause, RetargetGoal, SimThread
from ..squad import AgentSpec
from ..swarm import Swarm


def _world_scene(grid: int):
    story = shrunk(STORIES["city"], n=grid, max_steps=10_000_000)
    truth = story.truth_grid()
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    def specs(n, seed=0):
        rng = np.random.default_rng(seed)
        s = rng.integers(0, len(rows), n)
        g = rng.integers(0, len(rows), n)
        return [
            AgentSpec(f"a{i}", w(rows[s[i]], cols[s[i]]), w(rows[g[i]], cols[g[i]]))
            for i in range(n)
        ]

    return story, truth, specs, w, (rows, cols)


def _model():
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _bench_one(story, truth, specs, n, *, freeze, sense_every, ticks):
    sw = Swarm(
        story, specs(n), model=_model(), truth_occ=truth, prior_occ=truth, sense_every=sense_every
    )
    if freeze:
        sw._sense_shared = lambda: None
    for _ in range(3):
        sw.step()
    ts = []
    for _ in range(ticks):
        t = time.perf_counter()
        sw.step()
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return ts.mean(), float(np.percentile(ts, 95)), ts.max()


def throughput(grid: int, ticks: int, counts) -> None:
    story, truth, specs, _, _ = _world_scene(grid)
    print(
        f"grid {truth.shape}  {truth.mean() * 100:.1f}% blocked  cpu={os.cpu_count()}  "
        f"torch_threads={torch.get_num_threads()}"
    )

    def show(tag, n, mean, p95, mx):
        f60 = "Y" if mean < 16.7 else "."
        f30 = "Y" if mean < 33.3 else "."
        print(
            f"  N={n:5d}  {tag:11s}  mean {mean:7.2f}ms ({1000 / mean:6.1f} fps)  "
            f"60Hz[{f60}] 30Hz[{f30}]  p95 {p95:7.2f}ms  max {mx:7.2f}ms"
        )

    print("\n== drive path (static shared map — the 3-of-4 ticks) ==", flush=True)
    for n in counts:
        show(
            "drive", n, *_bench_one(story, truth, specs, n, freeze=True, sense_every=1, ticks=ticks)
        )

    print("\n== steady state (shared belief + single-EDT rebuild every 4th tick) ==", flush=True)
    for n in counts:
        if n > 1024:
            print(
                f"  N={n:5d}  +sense/4     skipped (Python raycast loop O(N); use C++ sense_batch)"
            )
            continue
        show(
            "+sense/4",
            n,
            *_bench_one(story, truth, specs, n, freeze=False, sense_every=4, ticks=ticks),
        )


def threading(grid: int, n: int, hz: float) -> None:
    story, truth, specs, w, (rows, cols) = _world_scene(grid)
    sw = Swarm(
        story, specs(n, seed=1), model=_model(), truth_occ=truth, prior_occ=truth, sense_every=4
    )
    sw._sense_shared = lambda: None  # isolate the GIL-released drive path
    print(f"\n== threading ==  grid {truth.shape}  N={sw.N}  sim @ {hz:.0f} Hz  render @ 200 fps")

    sim = SimThread(sw, hz=hz)
    sim.start()
    try:
        time.sleep(0.25)
        reads, lat, torn, last_gen = 0, [], 0, -1
        t0 = sim.ticks
        t_end = time.perf_counter() + 1.5
        while time.perf_counter() < t_end:
            t = time.perf_counter()
            f = sim.buffer.read()  # lock-free; never blocks the sim
            lat.append((time.perf_counter() - t) * 1e6)
            if len(f.pos) != sw.N or f.gen < last_gen:
                torn += 1
            last_gen = f.gen
            reads += 1
            time.sleep(1.0 / 200.0)
        dticks = sim.ticks - t0
        print(
            f"  [render] {reads} lock-free reads @ {reads / 1.5:.0f} fps "
            f"(latency mean {np.mean(lat):.2f} us / p99 {np.percentile(lat, 99):.2f} us)"
        )
        print(
            f"  [sim]    advanced {dticks} ticks CONCURRENTLY "
            f"({dticks / 1.5:.0f} tps; step {sim.step_ms:.2f} ms; behind {sim.behind})"
        )
        print(f"  [tears]  {torn}  (0 == every read frame was self-consistent)")

        newg = w(rows[len(rows) // 2], cols[len(cols) // 2])
        d0 = float(
            np.hypot(sim.buffer.read().pos[0, 0] - newg[0], sim.buffer.read().pos[0, 1] - newg[1])
        )
        sim.send(RetargetGoal(0, newg))
        time.sleep(2.0)
        d1 = float(
            np.hypot(sim.buffer.read().pos[0, 0] - newg[0], sim.buffer.read().pos[0, 1] - newg[1])
        )
        verb = "CLOSING — reacted to live command" if d1 < d0 - 1.0 else "no progress"
        print(f"  [retarget] agent 0 -> new goal: distance {d0:.1f} m -> {d1:.1f} m  ({verb})")

        sim.send(Pause(True))
        time.sleep(0.05)
        a = sim.buffer.read().tick
        time.sleep(0.2)
        print(f"  [pause]  tick {a} -> {sim.buffer.read().tick}  (frozen)")
    finally:
        sim.stop()
    print(f"  [shutdown] clean; {sim.ticks} total sim ticks")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="all", choices=["throughput", "threading", "all"])
    ap.add_argument("--grid", type=int, default=384)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--n", type=int, default=1024, help="agent count for the threading demo")
    ap.add_argument("--hz", type=float, default=60.0)
    args = ap.parse_args(argv)

    if args.mode in ("throughput", "all"):
        throughput(args.grid, args.ticks, (256, 512, 1024, 2048, 4096, 8192))
    if args.mode in ("threading", "all"):
        threading(args.grid, args.n, args.hz)


if __name__ == "__main__":
    main()
