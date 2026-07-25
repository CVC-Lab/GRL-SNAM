# examples/volrover_lab_demo.py — the GRL-SNAM lab, LIVE inside volrover3.
#
# The companion to examples/lab_demo.py (which builds its OWN standalone window).
# This one instead drives grl_snam_lab.Lab against volrover3's RUNNING scene, so
# terrain + an agent track + a marker appear in the live volrover3 viewport.
#
# LAYERING (important): volrover3 does NOT depend on GRL-SNAM. This demo lives
# here, in GRL-SNAM, and is run by volrover3's GENERIC script runner (the Python
# Console dock) only when GRL-SNAM is installed in the interpreter's environment.
# The `vrhost` module is injected by the volrover3 host at runtime — it is not a
# GRL-SNAM dependency; outside volrover3 this script says so and exits cleanly.
#
# HOW TO RUN (inside a running volrover3, with GRL-SNAM importable):
#   * Python Console dock -> REPL tab:
#         exec(open(".../GRL-SNAM/examples/volrover_lab_demo.py").read())
#   * or Jobs tab -> "Load Script..." and pick this file.

import math

# vrhost is provided by volrover3's embedded interpreter; a soft dependency, so we
# fail with a helpful message rather than an ImportError traceback when run outside.
try:
    import vrhost
except ImportError as exc:  # not inside volrover3
    raise SystemExit(
        "volrover_lab_demo: `import vrhost` failed — run this INSIDE volrover3's "
        "embedded Python console (the host injects vrhost). To see the lab as a "
        "standalone window instead, run examples/lab_demo.py."
    ) from exc

from grl_snam_lab import Lab

# Adopt volrover3's LIVE app + scene: every add_* mutates the running scene graph,
# so it shows up in the live window (its render timer drains the changes). We do
# NOT call lab.show() here — volrover3 already owns the render loop.
lab = Lab(app=vrhost.app(), scene=vrhost.scene())

# 1. Terrain — a gentle Gaussian bump over a 200x200 XY box.
N = 32
heights = [
    [10.0 * math.exp(-(((i - N / 2) ** 2 + (j - N / 2) ** 2) / 90.0)) for j in range(N)]
    for i in range(N)
]
lab.add_terrain(heights, bounds=(-100.0, -100.0, 100.0, 100.0), color=(0.42, 0.53, 0.34))

# 2. An agent track — a looping trajectory draped above the terrain.
track = [
    (78.0 * math.cos(k * 0.2), 78.0 * math.sin(k * 0.2), 16.0 + 4.0 * math.sin(k * 0.6))
    for k in range(32)
]
lab.add_path("agent0_track", track, color=(0.95, 0.75, 0.10))

# 3. A marker at the agent's current position.
lab.add_markers("agent0", [track[0]], color=(1.0, 0.20, 0.20))

lab.pump()
print(
    f"grl_snam_lab: live scene now has {lab.num_nodes()} node(s) "
    "(terrain + agent0_track + agent0) — look at the volrover3 viewport."
)
