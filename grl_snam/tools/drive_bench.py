"""Isolate the INFERENCE (drive) cost — torch vs the C++ native drive — with sense
frozen so a Swarm tick is only sample -> CoefMLP -> bicycle_rollout.

    python -m grl_snam.tools.drive_bench            # compare torch vs native across N
    GRL_SNAM_NAV_DRIVE=native python -m grl_snam.tools.drive_bench --once --n 8

This is "C++ for inference" (GRL_SNAM_NAV_DRIVE=native, the opt-in native drive),
distinct from the kernels (which are the default; see squad_bench). torch pays a
~fixed per-tick tensor-dispatch overhead independent of N, so the native drive
wins biggest at LOW agent counts — the single-agent free-drive and fog-story demos
(measured ~3.6x at N=1, ~2.2x at N=8); mid-N is a wash and it pulls ahead again at
very large N (threaded per-agent C++). Float-equivalent to torch, off by default."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _run_once(n: int, ticks: int):
    import numpy as np
    import torch

    import sdf_nav
    from grl_snam import planner
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.squad import AgentSpec
    from grl_snam.swarm import Swarm

    story = shrunk(STORIES["city"], n=96, max_steps=10_000_000)
    truth = story.truth_grid()
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    rng = np.random.default_rng(0)
    s, g = rng.integers(0, len(rows), n), rng.integers(0, len(rows), n)
    specs = [
        AgentSpec(f"a{i}", w(rows[s[i]], cols[s[i]]), w(rows[g[i]], cols[g[i]])) for i in range(n)
    ]
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP().eval()
    sw = Swarm(story, specs, model=model, truth_occ=truth, prior_occ=truth, belief_mode="shared")
    sw._sense_shared = lambda: None  # freeze sense: the tick is now ONLY the drive

    for _ in range(3):
        sw.step()
    t0 = time.perf_counter()
    for _ in range(ticks):
        sw.step()
    return (time.perf_counter() - t0) / ticks, sw._native_drive


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="agent count")
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--sizes", type=int, nargs="*", default=[1, 8, 64, 1024])
    ap.add_argument("--once", action="store_true", help="time the CURRENT backend at --n only")
    args = ap.parse_args(argv)

    if args.once:
        dt, native = _run_once(args.n, args.ticks)
        be = "native" if native else "torch"
        print(f"[drive={be:6s}] N={args.n:5d}: {dt*1000:8.3f} ms/tick")
        return

    for n in args.sizes:
        res = {}
        for be in ("torch", "native"):
            env = dict(os.environ, GRL_SNAM_NAV_DRIVE=be)
            out = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grl_snam.tools.drive_bench",
                    "--once",
                    "--n",
                    str(n),
                    "--ticks",
                    str(args.ticks),
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            lines = out.stdout.strip().splitlines()
            line = lines[-1] if lines else out.stderr.strip()[-200:]
            try:
                res[be] = float(line.split("ms/tick")[0].split()[-1])
            except Exception:
                res[be] = float("nan")
        if res.get("torch") and res.get("native") and res["native"] > 0:
            print(
                f"N={n:5d}: torch {res['torch']:7.3f} ms  native {res['native']:7.3f} ms  "
                f"-> {res['torch']/res['native']:.2f}x"
            )


if __name__ == "__main__":
    main()
