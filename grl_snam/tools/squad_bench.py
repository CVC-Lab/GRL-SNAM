"""Benchmark the city-squad demo sim (``Squad.step``) — the speedup the native
``cvc::nav`` kernels give the EXISTING Python demos with NO code change.

    python -m grl_snam.tools.squad_bench                 # compare python vs native
    python -m grl_snam.tools.squad_bench --n 32          # more agents
    GRL_SNAM_NAV_BACKEND=native python -m grl_snam.tools.squad_bench --once

Just having a ``cvc::nav``-capable ``pycvc`` on the path flips the per-tick kernels
(A*, EDT, ``build_sdf``, ``sense_batch``, inflate, neighbours) to threaded C++,
transparently and bit-identically — no demo code changes. The drive stays torch
(the native drive is a Swarm opt-in). The speedup applies to ACTIVE-navigation
ticks (where the kernels run; a parked/coasting tick is cheap either way) and
GROWS with agent count because A* is per-agent. Measured (192², active nav):
8 agents 3.3 -> 18 Hz (5.4x), 32 agents 0.56 -> 4.3 Hz (7.7x)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _run_once(grid: int, n: int, ticks: int, seed: int, route_clearance=None):
    import numpy as np
    import torch

    import sdf_nav
    from grl_snam import nav_native, planner
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.squad import AgentSpec, Squad

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
    agents = [
        AgentSpec(f"a{i}", w(rows[s[i]], cols[s[i]]), w(rows[g[i]], cols[g[i]])) for i in range(n)
    ]
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP().eval()
    squad = Squad(story, agents, model, truth_occ=truth, prior_occ=truth)
    # OFF by default, and deliberately so: a per-cell route surcharge turns off
    # A* batching (each agent's search becomes cost-specific), which is part of
    # what this benchmark measures. Enabling it silently would make the native
    # speedup look worse for a reason that has nothing to do with the backend.
    # The flag exists so the cost of standoff routing can be measured on purpose
    # -- it matters for a real-time demo.
    if route_clearance is not None:
        from grl_snam.squad import attach_clearance_routing

        attach_clearance_routing(squad, *route_clearance)

    for _ in range(2):  # warmup: alloc + first replan
        squad.step()
    t0 = time.perf_counter()
    for _ in range(ticks):
        squad.step()
    return (time.perf_counter() - t0) / ticks, nav_native.enabled()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=192)
    ap.add_argument("--n", type=int, default=8, help="agent count")
    ap.add_argument("--ticks", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--once", action="store_true", help="time the CURRENT backend only")
    ap.add_argument(
        "--route-clearance",
        nargs="?",
        const="6.0,1.5",
        default=None,
        metavar="D_SAFE,GAMMA",
        help="time WITH clearance-weighted routing (default off: it disables A* "
        "batching, which this benchmark measures)",
    )
    args = ap.parse_args(argv)

    rc = tuple(float(v) for v in args.route_clearance.split(",")) if args.route_clearance else None
    if args.once:
        dt, enabled = _run_once(args.grid, args.n, args.ticks, args.seed, rc)
        be = "native" if enabled else "python"
        print(f"[{be:6s}] grid={args.grid} n={args.n}: {dt*1000:8.1f} ms/tick  {1/dt:6.2f} Hz")
        return

    # Compare: run each backend in its OWN process so GRL_SNAM_NAV_BACKEND is
    # honored from import time (nav_native probes the env once at load).
    res: dict[str, float] = {}
    for be in ("python", "native"):
        env = dict(os.environ, GRL_SNAM_NAV_BACKEND=be)
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "grl_snam.tools.squad_bench",
                "--once",
                "--grid",
                str(args.grid),
                "--n",
                str(args.n),
                "--ticks",
                str(args.ticks),
                "--seed",
                str(args.seed),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        lines = out.stdout.strip().splitlines()
        line = lines[-1] if lines else out.stderr.strip()[-200:]
        print(line)
        try:
            res[be] = float(line.split("ms/tick")[0].split()[-1])
        except Exception:
            res[be] = float("nan")
    if res.get("python") and res.get("native") and res["native"] > 0:
        print(
            f"  -> native speedup {res['python']/res['native']:.1f}x  "
            f"(city squad, {args.n} agents @ {args.grid}^2, active navigation)"
        )


if __name__ == "__main__":
    main()
