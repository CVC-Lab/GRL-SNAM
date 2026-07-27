# examples/volrover_grl_snam_planner.py — a REAL GRL-SNAM navigation demo in a
# live volrover3 window: the GRL-SNAM surrogate planner determines a path from a
# START to a GOAL that navigates AROUND obstacles, and an agent drives it,
# filmed by the third-person chase camera.
#
# The path is produced by iterating grl_snam's differentiable surrogate rollout
# (dynamics.integrate_surrogate_v2) one step at a time — the same physics-informed
# integrator downstream training runs through (semi-implicit Euler + IPC obstacle
# barrier). Control coefficients here are hand-set to a known-good regime; with a
# trained CoefEnergyNet checkpoint you'd predict them per step and refine online
# with grl_snam.adaptation.HistSecantController (the "continuous learning" loop) —
# see the CONTINUOUS-LEARNING note below.
#
# HOW TO RUN (inside volrover3): Jobs tab -> "Load Script..." -> pick this file.
# Needs torch (a GRL-SNAM dependency) in the interpreter. Set GRL_SNAM_ROOT if the
# repo isn't at the default path.
#
# LAYERING: volrover3 has no GRL-SNAM dependency; this runs under its generic job
# runner when GRL-SNAM + torch are installed.

import math
import os
import sys

# grl_snam imports surrogate_robust from the repo root, so ensure it's importable.
_ROOT = os.environ.get("GRL_SNAM_ROOT", "/home/joe/src/cvc/GRL-SNAM")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
from grl_snam.dynamics import integrate_surrogate_v2  # noqa: E402

from grl_snam_lab.camera import ChaseCamera  # noqa: E402
from grl_snam_lab.demo import terrain_height, _terrain_heights, TERRAIN_BOUNDS  # noqa: E402
from grl_snam_lab.lab import Lab  # noqa: E402

try:
    import pycvc
    import vrhost
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "volrover_grl_snam_planner: run INSIDE volrover3 (Jobs tab -> Load Script)."
    ) from exc

# ── scene: start, goal, obstacles (all in the terrain's x,y plane) ───────────
START = (-70.0, -60.0)
GOAL = (70.0, 60.0)
OBSTACLES = [  # (x, y, radius)
    (10.0, 20.0, 14.0),
    (-20.0, 10.0, 12.0),
    (25.0, -15.0, 12.0),
    (-10.0, -25.0, 10.0),
]
ROBOT_RADIUS = 3.0
LIFT = 1.0


def _drape(x, y):
    return terrain_height(x, y) + LIFT


# ── GRL-SNAM plan: iterate the surrogate rollout to a path of (x,y) ──────────
def plan_path():
    f64 = torch.float64
    o = torch.tensor([[START[0], START[1]]], dtype=f64)
    v = torch.zeros(1, 2, dtype=f64)
    goal = torch.tensor([[GOAL[0], GOAL[1]]], dtype=f64)
    C = torch.tensor([[[ox, oy] for ox, oy, _ in OBSTACLES]], dtype=f64)
    R = torch.tensor([[r for _, _, r in OBSTACLES]], dtype=f64)
    mask = torch.ones(1, len(OBSTACLES), dtype=torch.bool)
    rr = torch.full((1,), ROBOT_RADIUS, dtype=f64)  # (B,) tensor, not a scalar
    # known-good regime (goal spring / damping / barrier activation) for this scale
    alphas = torch.full((1, len(OBSTACLES)), 3.0, dtype=f64)
    beta = torch.full((1,), 2.0, dtype=f64)
    gamma = torch.full((1,), 4.0, dtype=f64)
    d_hat = torch.full((1,), 16.0, dtype=f64)
    dt = torch.full((1,), 0.06, dtype=f64)
    H1 = torch.tensor([1], dtype=torch.long)
    path = [START]
    min_clear = float("inf")
    for _ in range(9000):
        o, v, clr = integrate_surrogate_v2(
            o,
            v,
            goal,
            C,
            R,
            mask,
            alphas,
            beta,
            gamma,
            d_hat,
            dt,
            H1,
            robot_radius=rr,
            margin_factor=0.5,
        )
        min_clear = min(min_clear, float(clr[0]))
        path.append((float(o[0, 0]), float(o[0, 1])))
        if torch.linalg.norm(o - goal).item() < 1.5:
            break
    return path, min_clear


_path2d, _clear = plan_path()
print(
    "grl_snam planner: %d-waypoint path, min obstacle clearance %.1f — building scene..."
    % (len(_path2d), _clear)
)

# ── build the live scene ─────────────────────────────────────────────────────
_app = vrhost.app()
_lab = Lab(app=_app, scene=vrhost.scene())
_lab.add_terrain(_terrain_heights(), bounds=TERRAIN_BOUNDS, color=(0.32, 0.40, 0.27))

# obstacles as translucent red pillars (children of the terrain, so they align)
from vtkmodules.vtkFiltersSources import vtkCylinderSource  # noqa: E402
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter  # noqa: E402
from vtkmodules.vtkCommonTransforms import vtkTransform  # noqa: E402
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper  # noqa: E402


def _add_obstacle(name, x, y, r, height=40.0):
    cyl = vtkCylinderSource()
    cyl.SetRadius(r)
    cyl.SetHeight(height)
    cyl.SetResolution(28)
    tr = vtkTransform()  # cylinder is Y-up; stand it up along Z at (x,y)
    tr.Translate(x, y, terrain_height(x, y) + height * 0.5)
    tr.RotateX(90.0)
    tf = vtkTransformPolyDataFilter()
    tf.SetTransform(tr)
    tf.SetInputConnection(cyl.GetOutputPort())
    tf.Update()
    m = vtkPolyDataMapper()
    m.SetInputData(tf.GetOutput())
    m.SetStatic(1)
    a = vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(0.85, 0.20, 0.18)
    a.GetProperty().SetOpacity(0.45)
    b = tf.GetOutput().GetBounds()
    _lab.add_prop(name, a, (b[0], b[2], b[4], b[1], b[3], b[5]), parent="terrain")


for _i, (_ox, _oy, _r) in enumerate(OBSTACLES):
    _add_obstacle("obstacle%d" % _i, _ox, _oy, _r)

# start (green) + goal (gold) markers, the planned path (yellow), and the agent
_lab.add_markers("start", [(*START, _drape(*START))], color=(0.15, 0.85, 0.25))
_lab.add_markers("goal", [(*GOAL, _drape(*GOAL))], color=(0.95, 0.80, 0.10))
_lab.add_path("plan", [(x, y, _drape(x, y)) for x, y in _path2d], color=(0.95, 0.85, 0.20))
_S = 5.0
_lab.add_mesh(
    "agent0",
    [0, 0, _S, -_S, -_S, 0, _S, -_S, 0, 0, _S, 0],
    [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
    color=(0.95, 0.15, 0.15),
)

# ── chase camera along the planned path ──────────────────────────────────────
_CAM = "volrover3.camera"
_chase = ChaseCamera(back=55.0, height=42.0, look_up=3.0, up=(0.0, 0.0, 1.0))
_SPEED = 28.0  # waypoints per second along the plan


def _agent_at(t):
    f = t * _SPEED
    i = min(int(f), len(_path2d) - 1)
    j = min(i + 1, len(_path2d) - 1)
    a = f - i
    x = _path2d[i][0] * (1 - a) + _path2d[j][0] * a
    y = _path2d[i][1] * (1 - a) + _path2d[j][1] * a
    return (x, y, _drape(x, y))


def _cset(k, val):
    pycvc.state_set(_app, _CAM + "." + k, "%.6f" % float(val))


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
    _cset("fov", 55.0)


_DT = 1.0 / 30.0
_t = 0.0
for _k in range(30):  # prime the camera
    _e, _g, _u = _chase.update(_agent_at(_k * _DT), _DT)
_drive(_e, _g, _u)
_lab.move("agent0", *_agent_at(0.0))
_lab.pump()
print(
    "grl_snam planner demo: agent drives the planned path A->B around obstacles. Jobs tab -> Stop."
)

_T_END = (len(_path2d) - 1) / _SPEED


def step(dt):
    global _t
    _t += dt
    if _t > _T_END + 1.5:  # loop
        _t = 0.0
    pos = _agent_at(_t)
    _lab.move("agent0", *pos)
    _drive(*_chase.update(pos, dt))
    _lab.pump()


# ── CONTINUOUS-LEARNING note ─────────────────────────────────────────────────
# With a trained policy checkpoint you replace the hand-set coefficients with
# per-step predictions and adapt them ONLINE (no retraining):
#   from grl_snam import CoefEnergyNet
#   from grl_snam.adaptation import HistSecantController   # rank-1 secant Jacobian
# Each step: predict (alphas,beta,gamma) from the local obstacle features, form an
# observable [clearance, goal-distance, speed], and let HistSecantController nudge
# the coefficients toward (clear, closing, moving) — the path then bends in real
# time. That online correction IS the "continuous learning after training."
