# examples/volrover_lab_animated.py — the GRL-SNAM agent WALKS its track in a
# LIVE volrover3 window, filmed by a third-person CHASE CAMERA.
#
# Load it from volrover3's Python Console -> Jobs tab -> "Load Script..." ->
# "Run as Job". The scheduler calls step(dt) each tick; the agent advances by
# MUTATING its node transform (no destroy/recreate), and a position-driven chase
# camera follows it. Pause / stop from the Jobs tab.
#
# The camera is driven ONLY by the stream of agent positions (grl_snam_lab.camera
# .ChaseCamera) — it estimates the heading from a smoothed velocity and damps the
# pose, so it works unchanged when the LIVE GRL-SNAM planner supplies real (noisy)
# positions instead of this demo path.
#
# LAYERING: volrover3 does NOT depend on GRL-SNAM. This runs under volrover3's
# GENERIC job runner only when GRL-SNAM is installed. `vrhost` + `pycvc` are
# provided by the volrover3 host at runtime.

import math

from grl_snam_lab.camera import ChaseCamera
from grl_snam_lab.demo import agent_position, demo_scene
from grl_snam_lab.lab import Lab

try:
    import pycvc
    import vrhost  # host-injected inside volrover3; a soft dependency
except ImportError as exc:  # pragma: no cover - only meaningful inside volrover3
    raise RuntimeError(
        "volrover_lab_animated: `vrhost`/`pycvc` not found — load this INSIDE "
        "volrover3's embedded Python (Jobs tab -> Load Script -> Run as Job)."
    ) from exc

_app = vrhost.app()
_lab = demo_scene(Lab(app=_app, scene=vrhost.scene()))
_lab.pump()

# ── third-person chase camera → volrover3's live camera (its state tree) ──────
_CAM = "volrover3.camera"
_chase = ChaseCamera(back=55.0, height=40.0, look_up=3.0, up=(0.0, 0.0, 1.0))


def _cset(k, v):
    pycvc.state_set(_app, _CAM + "." + k, "%.6f" % float(v))


def _drive_camera(eye, tgt, up):
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
    _cset("fov", 55.0)


# Prime the camera over the first ~1s of motion so it starts settled (no swing).
_DT = 1.0 / 30.0
_t = 0.0
for _i in range(30):
    _e, _g, _u = _chase.update(agent_position(_i * _DT), _DT)
_drive_camera(_e, _g, _u)
_t = 30 * _DT
_lab.move("agent0", *agent_position(_t))
_lab.pump()

print(
    "grl_snam_lab: animated chase-cam job loaded — the red agent walks the track, "
    "the camera follows. Pause/stop from the Jobs tab."
)


def step(dt):
    """Scheduler tick (dt = seconds since last tick): advance the agent and let the
    position-driven chase camera follow. Only the agent0 node's transform and the
    camera state change — nothing is added or removed."""
    global _t
    _t += dt
    pos = agent_position(_t)
    _lab.move("agent0", *pos)
    eye, tgt, up = _chase.update(pos, dt)  # camera reads ONLY the position stream
    _drive_camera(eye, tgt, up)
    _lab.pump()
