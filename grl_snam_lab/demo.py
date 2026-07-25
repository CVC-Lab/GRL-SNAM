"""Runnable demos for the GRL-SNAM lab.

Two entry points, one scene:

* ``run_in_volrover()`` — build the demo in a RUNNING volrover3's live scene
  (detected via the host-injected ``vrhost`` module). Call it from the volrover3
  Python Console REPL::

      >>> import grl_snam_lab
      >>> grl_snam_lab.run_in_volrover()

  and terrain + an agent track + a marker appear in the live viewport. volrover3
  does not depend on GRL-SNAM — this runs only because GRL-SNAM is importable in
  the interpreter's environment.

* ``run_standalone(png=...)`` — build the SAME scene in a self-owned window
  (``show()``) or an offscreen PNG (``render_png``). This is the ``grl-snam-lab-demo``
  console script (see pyproject ``[project.scripts]``); it needs pycvc/pycvc_gl +
  a GL context but no volrover3.

``demo_scene(lab)`` populates any ``Lab`` and is shared by both, so the exact same
nodes render whether standalone or embedded.
"""

from __future__ import annotations

import math

from .lab import Lab

__all__ = ["demo_scene", "run_in_volrover", "run_standalone", "main"]


def demo_scene(lab: Lab) -> Lab:
    """Add the canonical demo — terrain + an agent track + a marker — to ``lab``.

    Domain-general: a Gaussian-bump heightfield, a looping agent trajectory draped
    above it, and a marker at the agent's current position. Returns ``lab`` so the
    call chains.
    """
    # Terrain: a gentle Gaussian bump over a 200x200 XY box.
    n = 32
    heights = [
        [10.0 * math.exp(-(((i - n / 2) ** 2 + (j - n / 2) ** 2) / 90.0)) for j in range(n)]
        for i in range(n)
    ]
    lab.add_terrain(heights, bounds=(-100.0, -100.0, 100.0, 100.0), color=(0.42, 0.53, 0.34))

    # An agent track: a looping trajectory draped above the terrain.
    track = [
        (78.0 * math.cos(k * 0.2), 78.0 * math.sin(k * 0.2), 16.0 + 4.0 * math.sin(k * 0.6))
        for k in range(32)
    ]
    lab.add_path("agent0_track", track, color=(0.95, 0.75, 0.10))

    # A marker at the agent's current position.
    lab.add_markers("agent0", [track[0]], color=(1.0, 0.20, 0.20))
    return lab


def run_in_volrover() -> Lab:
    """Build the demo in volrover3's RUNNING scene (call from the volrover3 REPL).

    Adopts the host's live app + scene via ``vrhost``, so every node appears in the
    live volrover3 window (its render timer picks up the changes). Does NOT call
    ``show()`` — volrover3 owns the render loop. Raises a clear error if ``vrhost``
    isn't present (i.e. not running inside volrover3).
    """
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
        "(terrain + agent0_track + agent0) — look at the viewport."
    )
    return lab


def run_standalone(png: str | None = None, width: int = 1024, height: int = 768) -> Lab:
    """Build the demo in a self-owned Lab. Render an offscreen PNG when ``png`` is
    given (headless), otherwise open a blocking interactive window (needs a display).
    """
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
    offscreen snapshot (works headless). Inside volrover3, use the REPL instead:
    ``import grl_snam_lab; grl_snam_lab.run_in_volrover()``.
    """
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
