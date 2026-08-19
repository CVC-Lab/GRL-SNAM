"""Benchmark the belief modes (all-shared / clustered-shared / all-private) and
the ``nsub`` knob, with the C++ ``sense_batch`` kernel live.

The finding: the vectorized drive is cheap; the sense raycast is the steady-state
wall, and its parallelism equals the belief-plane count — so how much belief is
shared trades footprint against sense parallelism (see the table this prints, and
docs/CVCNAV_CPP_PORT_ROADMAP.md §11).

    python -m grl_snam.tools.belief_bench [--grid 384] [--ticks 20] [--drive-only]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    raise SystemExit("belief_bench needs torch")

import sdf_nav

from .. import nav_native, planner
from ..fog_stories import STORIES, shrunk
from ..squad import AgentSpec
from ..swarm import Swarm


def _scene(grid):
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

    return story, truth, specs


def _model():
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _bench(story, truth, specs, n, mode, nsub, clusters, ticks, freeze):
    kw = {} if clusters is None else {"clusters": clusters}
    sw = Swarm(
        story,
        specs(n),
        model=_model(),
        truth_occ=truth,
        sense_every=4,
        belief_mode=mode,
        nsub=nsub,
        **kw,
    )
    if freeze:
        sw._sense_shared = lambda: None
    for _ in range(5):
        sw.step()
    ts = []
    for _ in range(ticks):
        t = time.perf_counter()
        sw.step()
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return ts.mean(), float(np.percentile(ts, 95))


def _row(tag, n, nsub, mean, p95):
    f60 = "Y" if mean < 16.7 else "."
    f30 = "Y" if mean < 33.3 else "."
    print(
        f"  N={n:5d} nsub={nsub}  {tag:12s}  mean {mean:8.2f}ms ({1000/mean:6.1f}fps "
        f"60[{f60}]30[{f30}])  p95 {p95:8.2f}ms"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=384)
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--nsub", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument(
        "--drive-only", action="store_true", help="also run the frozen-sense drive sweep"
    )
    args = ap.parse_args(argv)

    story, truth, specs = _scene(args.grid)
    print(
        f"grid {truth.shape}  cpu={os.cpu_count()} torch_threads={torch.get_num_threads()} "
        f"sense_batch={nav_native.HAS_SENSE_BATCH}  story.nsub={story.nsub}"
    )
    modes = [
        ("all-shared", "shared", None, (256, 512, 1024, 2048, 4096)),
        ("clustered/8", "clustered", 8, (256, 512, 1024, 2048, 4096)),
        ("all-private", "private", None, (256, 512, 1024, 2048)),
    ]
    print("\n### STEADY-STATE (sense + rebuild every 4th tick) ###")
    for tag, mode, clusters, counts in modes:
        for nsub in args.nsub:
            print(f"\n== {tag}  nsub={nsub} ==", flush=True)
            for n in counts:
                try:
                    _row(
                        tag,
                        n,
                        nsub,
                        *_bench(story, truth, specs, n, mode, nsub, clusters, args.ticks, False),
                    )
                except Exception as e:  # OOM at the private memory wall
                    print(f"  N={n:5d} nsub={nsub}  {tag:12s}  SKIPPED ({type(e).__name__})")
                    break

    if args.drive_only:
        print("\n### DRIVE-ONLY (frozen sense: nsub's effect on the vectorized drive) ###")
        for nsub in args.nsub:
            print(f"\n== drive-only (shared)  nsub={nsub} ==", flush=True)
            for n in (1024, 2048, 4096, 8192):
                _row(
                    "drive-only",
                    n,
                    nsub,
                    *_bench(story, truth, specs, n, "shared", nsub, None, args.ticks, True),
                )


if __name__ == "__main__":
    main()
