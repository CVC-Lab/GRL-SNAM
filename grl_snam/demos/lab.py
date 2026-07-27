"""The general GRL-SNAM lab demo — an analytic terrain with an agent walking a draped
loop — static, animated, or offscreen, built on the CVC graphics stack (``pycvc_gl``).

This is the home of the former ``grl_snam_lab.demo`` after consolidating onto
``pycvc_gl`` (which supplies ``Lab``, ``terrain_mesh``, cameras, etc.). The pure
geometry (``terrain_height``, ``TRACK``, ``agent_position``) needs no bindings and is
imported by the sparse-obstacle planner demo; the scene builders import ``pycvc_gl``
lazily so this module imports anywhere.

Entry points: :func:`run_standalone` (window / PNG, the ``grl-snam lab-demo`` command),
:func:`run_in_volrover` (into a running VolRover3 scene), and ``step(dt)`` (the animated
job loaded with ``grl-snam demo lab``).
"""

from __future__ import annotations

import math

TERRAIN_BOUNDS = (-100.0, -100.0, 100.0, 100.0)
TRACK_RADIUS = 70.0
LOOP_SECONDS = 22.0
AGENT_LIFT = 0.4
_N_TRACK = 180


def terrain_height(x: float, y: float) -> float:
    """Analytic terrain height at world ``(x, y)`` — gentle rolling hills. Single source
    of truth: the mesh samples it and the agent/track drape onto it."""
    h = 14.0 * math.exp(-((x * x + y * y) / 3000.0))
    h += 8.0 * math.exp(-(((x - 55.0) ** 2 + (y + 35.0) ** 2) / 900.0))
    h += 6.0 * math.exp(-(((x + 45.0) ** 2 + (y - 50.0) ** 2) / 1200.0))
    h += 2.5 * math.sin(x * 0.06) * math.cos(y * 0.05)
    return h


def _terrain_heights(n: int = 56):
    """Sample ``terrain_height`` on an n x n grid over TERRAIN_BOUNDS (row -> y, col -> x)."""
    min_x, min_y, max_x, max_y = TERRAIN_BOUNDS
    sx = (max_x - min_x) / (n - 1)
    sy = (max_y - min_y) / (n - 1)
    return [[terrain_height(min_x + j * sx, min_y + i * sy) for j in range(n)] for i in range(n)]


def _loop_point(theta: float):
    r = TRACK_RADIUS + 8.0 * math.sin(3.0 * theta)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return (x, y, terrain_height(x, y) + AGENT_LIFT)


TRACK = [_loop_point(2.0 * math.pi * k / _N_TRACK) for k in range(_N_TRACK + 1)]


def _agent_marker_mesh(size: float = 5.0):
    s = size
    verts = [0.0, 0.0, s, -s, -s, 0.0, s, -s, 0.0, 0.0, s, 0.0]
    tris = [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]
    return verts, tris


def agent_position(t: float, speed: float = 1.0):
    """The agent's ``(x, y, z)`` on the demo loop at time ``t`` (draped onto the terrain)."""
    theta = 2.0 * math.pi * ((t * speed / LOOP_SECONDS) % 1.0)
    return _loop_point(theta)


def demo_scene(lab):
    """Add the canonical demo — terrain + the agent's track + the agent marker — to ``lab``."""
    lab.add_terrain(_terrain_heights(), bounds=TERRAIN_BOUNDS, color=(0.42, 0.53, 0.34))
    lab.add_path("agent0_track", TRACK, color=(0.95, 0.75, 0.10))
    verts, tris = _agent_marker_mesh()
    lab.add_mesh("agent0", verts, tris, color=(1.0, 0.20, 0.20))
    lab.move("agent0", *agent_position(0.0))
    lab.pump()
    return lab


def animate_agent(lab, t: float):
    """Walk the ``agent0`` marker to its position at time ``t`` (transform mutate, no rebuild)."""
    lab.move("agent0", *agent_position(t))
    lab.pump()
    return lab


def run_standalone(png: str | None = None, width: int = 1024, height: int = 768):
    """Static demo in a self-owned Lab; offscreen PNG when ``png`` is given, else a window."""
    from pycvc_gl.lab import Lab

    lab = demo_scene(Lab())
    if png:
        lab.render_png(png, width, height)
        print(f"grl-snam lab-demo: wrote {png} ({lab.num_nodes()} nodes)")
    else:
        print(f"grl-snam lab-demo: showing {lab.num_nodes()} nodes — close the window to exit")
        lab.show("grl-snam lab demo", width, height)
    return lab


def run_in_volrover():
    """Build the static demo in a running VolRover3 scene (call from the REPL / job)."""
    from grl_snam.demos._common import require_host

    _pycvc, vrhost = require_host()
    from pycvc_gl.lab import Lab

    lab = demo_scene(Lab(app=vrhost.app(), scene=vrhost.scene()))
    lab.pump()
    print(
        f"grl-snam lab: live VolRover3 scene now has {lab.num_nodes()} node(s) — look at the viewport."
    )
    return lab


_S: dict = {}


def setup() -> None:
    from grl_snam.demos._common import require_host

    _pycvc, vrhost = require_host()
    from pycvc_gl.lab import Lab

    lab = demo_scene(Lab(app=vrhost.app(), scene=vrhost.scene()))
    _S.update(lab=lab, t=0.0)
    print("grl-snam lab: animated agent walking the draped loop.", flush=True)


def step(dt: float) -> None:
    """Animated job: walk the agent one tick (``grl-snam demo lab``)."""
    if not _S:
        setup()
    _S["t"] += dt
    animate_agent(_S["lab"], _S["t"])


def main(argv: list[str] | None = None) -> int:
    """Console entry (``grl-snam lab-demo``): standalone viz. ``lab-demo out.png`` => snapshot."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="grl-snam lab-demo", description="GRL-SNAM lab demo (terrain + agent track + marker)."
    )
    ap.add_argument(
        "png", nargs="?", help="write an offscreen PNG here instead of opening a window"
    )
    args = ap.parse_args(argv)
    try:
        run_standalone(png=args.png)
    except Exception as exc:  # pycvc/pycvc_gl missing, or no GL display
        print(f"grl-snam lab-demo: {exc}")
        return 1
    return 0
