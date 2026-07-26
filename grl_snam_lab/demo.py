"""Runnable demos for the GRL-SNAM lab — static and animated.

Entry points:

* ``run_in_volrover()`` — build the STATIC demo (terrain + full track + agent) in a
  RUNNING volrover3's live scene. Call it from the volrover3 Python Console REPL::

      >>> import grl_snam_lab
      >>> grl_snam_lab.run_in_volrover()

* ``run_standalone(png=...)`` — the same static scene in a self-owned window / PNG
  (the ``grl-snam-lab-demo`` console script).

* ANIMATED job — ``examples/volrover_lab_animated.py`` is a JobScheduler ``step(dt)``
  script that walks the agent along the track in the live volrover3 window (Load
  Script -> Run as Job). It reuses ``demo_scene`` + ``animate_agent`` below.

``demo_scene(lab)`` populates any ``Lab`` (the agent marker is added ONCE);
``animate_agent(lab, t)`` moves the agent to its position at time ``t`` by mutating
the node's transform in place — no destroy/recreate, just ``lab.move(...)`` — the
shared core for the static, animated, and frame-capture paths. volrover3 has no
GRL-SNAM dependency; these run only because GRL-SNAM is importable in the
interpreter's environment.
"""

from __future__ import annotations

import math

from .lab import Lab

__all__ = [
    "demo_scene",
    "agent_position",
    "animate_agent",
    "terrain_height",
    "run_in_volrover",
    "run_standalone",
    "TRACK",
]

# ── World geometry ───────────────────────────────────────────────────────────
# The terrain is an ANALYTIC height field h(x,y) over a 200x200 world box; the
# drawn terrain mesh samples it, and the agent + track DRAPE onto it (so the
# robot sits on the ground, not floating). The agent walks a SMOOTH analytic
# loop (a real circle, not a coarse polygon), so its position AND its heading
# vary continuously — the earlier 32-gon made the chase camera jerk at each
# vertex because the velocity jumped there.

TERRAIN_BOUNDS = (-100.0, -100.0, 100.0, 100.0)  # (min_x, min_y, max_x, max_y)
TRACK_RADIUS = 70.0  # loop radius (world units)
LOOP_SECONDS = 22.0  # time for the agent to walk one full lap at speed 1.0
AGENT_LIFT = 0.4  # small lift so the marker base clears z-fighting with terrain
_N_TRACK = 180  # drawn-polyline resolution (fine => looks like a smooth circle)


def terrain_height(x: float, y: float) -> float:
    """Analytic terrain height at world ``(x, y)`` — gentle rolling hills (a central
    rise, two bumps, a low ripple). The single source of truth for the terrain: the
    mesh samples it and the agent/track drape onto it."""
    h = 14.0 * math.exp(-((x * x + y * y) / 3000.0))  # central hill
    h += 8.0 * math.exp(-(((x - 55.0) ** 2 + (y + 35.0) ** 2) / 900.0))  # bump
    h += 6.0 * math.exp(-(((x + 45.0) ** 2 + (y - 50.0) ** 2) / 1200.0))  # bump
    h += 2.5 * math.sin(x * 0.06) * math.cos(y * 0.05)  # ripple
    return h


def _terrain_heights(n: int = 56):
    """Sample ``terrain_height`` on an n x n grid over TERRAIN_BOUNDS (row-major,
    row index -> y, col index -> x), matching ``terrain_mesh``'s layout."""
    min_x, min_y, max_x, max_y = TERRAIN_BOUNDS
    sx = (max_x - min_x) / (n - 1)
    sy = (max_y - min_y) / (n - 1)
    return [[terrain_height(min_x + j * sx, min_y + i * sy) for j in range(n)] for i in range(n)]


def _loop_point(theta: float):
    """A point on the agent's demo loop at angle ``theta``, draped onto the terrain.

    The radius WANDERS (a 3-lobed modulation) so the heading changes at a varying
    rate — this is a stand-in for a real GRL-SNAM path, and it gives the
    position-driven chase camera genuine direction changes to smooth. (The camera
    never sees this function; it only sees the emitted positions.)"""
    r = TRACK_RADIUS + 8.0 * math.sin(3.0 * theta)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return (x, y, terrain_height(x, y) + AGENT_LIFT)


# The drawn track: a fine, closed, draped ring (looks like a smooth circle on the
# terrain). Shared by the static path (drawn whole) and the animation.
TRACK = [_loop_point(2.0 * math.pi * k / _N_TRACK) for k in range(_N_TRACK + 1)]


def _agent_marker_mesh(size: float = 5.0):
    """A small upward tetrahedron with its BASE at local z=0 (apex at +size), so
    placing it via setPosition(x, y, terrain_height) sits it ON the ground. Built
    ONCE at the origin and then MOVED via the node transform (see animate_agent)."""
    s = size
    verts = [0.0, 0.0, s, -s, -s, 0.0, s, -s, 0.0, 0.0, s, 0.0]
    tris = [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]
    return verts, tris


def _theta(t: float, speed: float) -> float:
    return 2.0 * math.pi * ((t * speed / LOOP_SECONDS) % 1.0)


def agent_position(t: float, speed: float = 1.0):
    """The agent's (x, y, z) at time ``t`` on the demo loop, draped onto the
    terrain. ``speed`` scales time (1.0 => one lap per LOOP_SECONDS). This is only
    a stand-in path for the demo — the live GRL-SNAM planner will supply real
    positions; feed whichever stream you have to ``ChaseCamera`` (grl_snam_lab.camera),
    which derives the camera heading from the positions themselves."""
    return _loop_point(_theta(t, speed))


def animate_agent(lab: Lab, t: float) -> Lab:
    """Move the ``agent0`` marker to its position at time ``t`` by mutating the
    node's transform — the marker walks the track with NO destroy/recreate (the
    geometry + VTK actor are reused; only the transform changes). ``demo_scene``
    must have added ``agent0`` first. Returns ``lab``."""
    lab.move("agent0", *agent_position(t))
    lab.pump()
    return lab


def demo_scene(lab: Lab) -> Lab:
    """Add the canonical demo — terrain + the agent's track + the agent marker
    (added ONCE at the origin, then placed at its t=0 position) — to ``lab``.
    Returns ``lab`` so calls chain."""
    lab.add_terrain(
        _terrain_heights(), bounds=(-100.0, -100.0, 100.0, 100.0), color=(0.42, 0.53, 0.34)
    )
    lab.add_path("agent0_track", TRACK, color=(0.95, 0.75, 0.10))
    # Build the agent marker ONCE (at the origin); animate_agent MOVES it.
    verts, tris = _agent_marker_mesh()
    lab.add_mesh("agent0", verts, tris, color=(1.0, 0.20, 0.20))
    animate_agent(lab, 0.0)  # place it at the track start
    return lab


def run_in_volrover() -> Lab:
    """Build the STATIC demo in volrover3's RUNNING scene (call from the REPL). Adopts
    the host's live app + scene via ``vrhost``; raises a clear error outside volrover3.
    For the ANIMATED version, load ``examples/volrover_lab_animated.py`` as a job."""
    try:
        import vrhost  # host-injected inside volrover3; not a GRL-SNAM dependency
    except ImportError as exc:
        raise RuntimeError(
            "grl_snam_lab.run_in_volrover(): `vrhost` not found — this must run "
            "INSIDE volrover3's embedded Python console. For a standalone window "
            "use grl_snam_lab.run_standalone()."
        ) from exc

    lab = demo_scene(Lab(app=vrhost.app(), scene=vrhost.scene()))
    lab.pump()
    print(
        f"grl_snam_lab: live volrover3 scene now has {lab.num_nodes()} node(s) "
        "(terrain + agent0_track + agent0) — look at the viewport. For the animated "
        "walk, Load Script -> Run as Job: examples/volrover_lab_animated.py"
    )
    return lab


def run_standalone(png: str | None = None, width: int = 1024, height: int = 768) -> Lab:
    """Build the static demo in a self-owned Lab; render an offscreen PNG when ``png``
    is given (headless), else open a blocking interactive window (needs a display)."""
    lab = demo_scene(Lab())
    if png:
        lab.render_png(png, width, height)
        print(f"grl_snam_lab: wrote {png} ({lab.num_nodes()} nodes)")
    else:
        print(f"grl_snam_lab: showing {lab.num_nodes()} nodes — close the window to exit")
        lab.show("grl_snam_lab demo", width, height)
    return lab


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``grl-snam-lab-demo``): standalone viz demo.

    ``grl-snam-lab-demo`` opens a window; ``grl-snam-lab-demo out.png`` renders an
    offscreen snapshot. Inside volrover3 use the REPL:
    ``import grl_snam_lab; grl_snam_lab.run_in_volrover()``."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="grl-snam-lab-demo",
        description="GRL-SNAM lab demo (terrain + agent track + marker), standalone.",
    )
    ap.add_argument(
        "png", nargs="?", help="write an offscreen PNG here instead of opening a window"
    )
    args = ap.parse_args(argv)
    try:
        run_standalone(png=args.png)
    except Exception as exc:  # pycvc/pycvc_gl missing, or no GL display
        print(f"grl-snam-lab-demo: {exc}")
        return 1
    return 0
