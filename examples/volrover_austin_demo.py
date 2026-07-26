# examples/volrover_austin_demo.py — a robot patrols REAL Austin, TX geometry in
# a LIVE volrover3 window, filmed by a third-person chase camera.
#
# Loads a geometry_bundle (terrain heightfield + glTF city mesh) — by default the
# CVC-DBG "austin_south" bundle (3 km x 3 km around the UT campus) — drapes an
# agent onto the real terrain, and follows it with grl_snam_lab.camera.ChaseCamera
# (position-driven, ready for live GRL-SNAM paths).
#
# Load it from volrover3's Python Console -> Jobs tab -> "Load Script..." ->
# "Run as Job" (the glTF is ~978k triangles, so the first tick takes a few
# seconds to load). Point it at a different bundle with the GRL_SNAM_SCENE_BUNDLE
# env var.
#
# ATTRIBUTION: the geometry is derived from OpenStreetMap (c) OpenStreetMap
# contributors, ODbL (https://openstreetmap.org/copyright); SRTM terrain is US
# public domain. Credit OpenStreetMap in any published render.
#
# LAYERING: volrover3 does NOT depend on GRL-SNAM (or on the CVC-DBG assets). This
# runs under volrover3's generic job runner when GRL-SNAM is installed and the
# bundle is present.

import math
import os

from grl_snam_lab.camera import ChaseCamera
from grl_snam_lab.lab import Lab
from grl_snam_lab.scenes import load_geometry_bundle

try:
    import pycvc
    import vrhost
except ImportError as exc:  # pragma: no cover - only meaningful inside volrover3
    raise RuntimeError(
        "volrover_austin_demo: `vrhost`/`pycvc` not found — load this INSIDE "
        "volrover3's embedded Python (Jobs tab -> Load Script -> Run as Job)."
    ) from exc

_BUNDLE = os.environ.get(
    "GRL_SNAM_SCENE_BUNDLE",
    "/home/joe/src/cvc/CVC-DBG/platoon-sim/scene_viewer/exports/scenes/austin_south",
)

_app = vrhost.app()
_lab = Lab(app=_app, scene=vrhost.scene())
# Real Austin terrain + buildings; `_sample(x, y)` is the terrain height (drape).
_sample = load_geometry_bundle(_lab, _BUNDLE)

# ── the patrolling agent, draped on the real terrain ─────────────────────────
_RADIUS, _WANDER, _LOOP_S, _LIFT = 430.0, 90.0, 44.0, 4.0


def _agent_pos(t):
    th = 2.0 * math.pi * ((t / _LOOP_S) % 1.0)
    r = _RADIUS + _WANDER * math.sin(3.0 * th)
    x, y = r * math.cos(th), r * math.sin(th)
    return (x, y, _sample(x, y) + _LIFT)


def _marker(size=18.0):
    s = size
    return ([0, 0, s, -s, -s, 0, s, -s, 0, 0, s, 0], [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2])


_mv, _mt = _marker()
_lab.add_mesh("agent0", _mv, _mt, color=(0.95, 0.15, 0.15))
_lab.add_path(
    "agent0_track",
    [_agent_pos(_LOOP_S * k / 240.0) for k in range(241)],
    color=(0.95, 0.75, 0.10),
)

# ── third-person chase camera (scaled for the city) ──────────────────────────
_CAM = "volrover3.camera"
_chase = ChaseCamera(back=150.0, height=95.0, look_up=8.0, up=(0.0, 0.0, 1.0))


def _cset(k, v):
    pycvc.state_set(_app, _CAM + "." + k, "%.6f" % float(v))


def _drive(eye, tgt, up):
    vx, vy, vz = (tgt[i] - eye[i] for i in range(3))
    m = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
    _cset("position.x", eye[0])
    _cset("position.y", eye[1])
    _cset("position.z", eye[2])
    _cset("view_direction.x", vx / m)
    _cset("view_direction.y", vy / m)
    _cset("view_direction.z", vz / m)
    _cset("up_vector.x", up[0])
    _cset("up_vector.y", up[1])
    _cset("up_vector.z", up[2])
    _cset("fov", 60.0)


_DT = 1.0 / 30.0
_t = 0.0
for _i in range(30):  # prime the camera to a settled pose
    _e, _g, _u = _chase.update(_agent_pos(_i * _DT), _DT)
_drive(_e, _g, _u)
_t = 30 * _DT
_lab.move("agent0", *_agent_pos(_t))
_lab.pump()

print(
    "grl_snam_lab: Austin city chase-cam demo loaded (OSM/ODbL + SRTM terrain). "
    "The red agent patrols the streets; camera follows. Pause/stop from the Jobs tab."
)


def step(dt):
    global _t
    _t += dt
    pos = _agent_pos(_t)
    _lab.move("agent0", *pos)
    eye, tgt, up = _chase.update(pos, dt)
    _drive(eye, tgt, up)
    _lab.pump()
