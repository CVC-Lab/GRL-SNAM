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
    "run_in_volrover",
    "run_standalone",
    "TRACK",
]

# The agent's closed trajectory: a radius-78 loop undulating in Z, draped above a
# 200x200 terrain. Shared by the static path (drawn whole) and the animation (the
# agent walks it).
TRACK = [
    (78.0 * math.cos(k * 0.2), 78.0 * math.sin(k * 0.2), 16.0 + 4.0 * math.sin(k * 0.6))
    for k in range(32)
]


def _terrain_heights(n: int = 32):
    return [
        [10.0 * math.exp(-(((i - n / 2) ** 2 + (j - n / 2) ** 2) / 90.0)) for j in range(n)]
        for i in range(n)
    ]


def _agent_marker_mesh(size: float = 6.0):
    """A small upward tetrahedron centred at the ORIGIN — a marker that reads from
    any angle (a single point is nearly invisible at scene scale). Built ONCE and
    then MOVED via the node transform (see ``animate_agent``), never rebuilt."""
    s = size
    verts = [0.0, 0.0, s, -s, -s, 0.0, s, -s, 0.0, 0.0, s, 0.0]
    tris = [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]
    return verts, tris


def agent_position(t: float, speed: float = 6.0):
    """The agent's (x,y,z) at time ``t`` — linearly interpolated along the closed
    ``TRACK`` so it moves smoothly, at ``speed`` track-points per second, looping."""
    n = len(TRACK)
    f = (t * speed) % n
    i = int(f)
    j = (i + 1) % n
    a = f - i
    p, q = TRACK[i], TRACK[j]
    return tuple(p[k] * (1.0 - a) + q[k] * a for k in range(3))


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
