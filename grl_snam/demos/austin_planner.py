"""Live demo: the GRL-SNAM surrogate planner in its native sparse-obstacle regime.

On an analytic terrain with a handful of circular obstacles, the differentiable
surrogate rollout (``grl_snam.dynamics.integrate_surrogate_v2`` — the same integrator
downstream training runs through) is iterated to a collision-free path A->B, and an
agent drives it under a chase camera. Coefficients are hand-set to a known-good regime
here; with a trained ``CoefEnergyNet`` you would predict them per step and adapt online
with ``grl_snam.adaptation.HistSecantController``.
"""

from __future__ import annotations

import torch

from grl_snam.demos._common import CameraDriver, require_host
from grl_snam.demos.lab import TERRAIN_BOUNDS, _terrain_heights, terrain_height
from grl_snam.dynamics import integrate_surrogate_v2

START = (-70.0, -60.0)
GOAL = (70.0, 60.0)
OBSTACLES = [(10.0, 20.0, 14.0), (-20.0, 10.0, 12.0), (25.0, -15.0, 12.0), (-10.0, -25.0, 10.0)]
ROBOT_RADIUS = 3.0
LIFT = 1.0
_S: dict = {}


def _drape(x, y):
    return terrain_height(x, y) + LIFT


def plan_path():
    """Iterate the surrogate rollout to a collision-free path A->B; return (path, min_clear)."""
    f64 = torch.float64
    o = torch.tensor([[START[0], START[1]]], dtype=f64)
    v = torch.zeros(1, 2, dtype=f64)
    goal = torch.tensor([[GOAL[0], GOAL[1]]], dtype=f64)
    c = torch.tensor([[[ox, oy] for ox, oy, _ in OBSTACLES]], dtype=f64)
    r = torch.tensor([[rr for _, _, rr in OBSTACLES]], dtype=f64)
    mask = torch.ones(1, len(OBSTACLES), dtype=torch.bool)
    rr = torch.full((1,), ROBOT_RADIUS, dtype=f64)
    alphas = torch.full((1, len(OBSTACLES)), 3.0, dtype=f64)
    beta = torch.full((1,), 2.0, dtype=f64)
    gamma = torch.full((1,), 4.0, dtype=f64)
    d_hat = torch.full((1,), 16.0, dtype=f64)
    dt = torch.full((1,), 0.06, dtype=f64)
    h1 = torch.tensor([1], dtype=torch.long)
    path = [START]
    min_clear = float("inf")
    for _ in range(9000):
        o, v, clr = integrate_surrogate_v2(
            o,
            v,
            goal,
            c,
            r,
            mask,
            alphas,
            beta,
            gamma,
            d_hat,
            dt,
            h1,
            robot_radius=rr,
            margin_factor=0.5,
        )
        min_clear = min(min_clear, float(clr[0]))
        path.append((float(o[0, 0]), float(o[0, 1])))
        if torch.linalg.norm(o - goal).item() < 1.5:
            break
    return path, min_clear


def _obstacle_actor(x, y, r, height=40.0):
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
    from vtkmodules.vtkFiltersSources import vtkCylinderSource
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

    cyl = vtkCylinderSource()
    cyl.SetRadius(r)
    cyl.SetHeight(height)
    cyl.SetResolution(28)
    tr = vtkTransform()
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
    return a, tf.GetOutput().GetBounds()


def setup() -> None:
    pycvc, vrhost = require_host()
    from pycvc_gl.camera import ChaseCamera
    from pycvc_gl.lab import Lab

    path2d, clear = plan_path()
    print(
        "grl_snam planner: %d-waypoint path, min clearance %.1f — building scene..."
        % (len(path2d), clear),
        flush=True,
    )
    app = vrhost.app()
    lab = Lab(app=app, scene=vrhost.scene())
    lab.add_terrain(_terrain_heights(), bounds=TERRAIN_BOUNDS, color=(0.32, 0.40, 0.27))
    for i, (ox, oy, r) in enumerate(OBSTACLES):
        actor, b = _obstacle_actor(ox, oy, r)
        lab.add_prop(
            "obstacle%d" % i, actor, (b[0], b[2], b[4], b[1], b[3], b[5]), parent="terrain"
        )
    lab.add_markers("start", [(*START, _drape(*START))], color=(0.15, 0.85, 0.25))
    lab.add_markers("goal", [(*GOAL, _drape(*GOAL))], color=(0.95, 0.80, 0.10))
    lab.add_path("plan", [(x, y, _drape(x, y)) for x, y in path2d], color=(0.95, 0.85, 0.20))
    s = 5.0
    lab.add_mesh(
        "agent0",
        [0, 0, s, -s, -s, 0, s, -s, 0, 0, s, 0],
        [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
        color=(0.95, 0.15, 0.15),
    )
    chase = ChaseCamera(back=55.0, height=42.0, look_up=3.0, up=(0.0, 0.0, 1.0))
    lab.move("agent0", *_pose(path2d, 0.0))
    lab.pump()
    print("grl_snam planner: agent drives the planned path A->B around obstacles.", flush=True)
    _S.update(
        lab=lab,
        path=path2d,
        chase=chase,
        cam=CameraDriver(app, pycvc, fov=55.0),
        t=0.0,
        speed=28.0,
        t_end=(len(path2d) - 1) / 28.0,
    )


def _pose(path2d, t, speed=28.0):
    f = t * speed
    i = min(int(f), len(path2d) - 1)
    j = min(i + 1, len(path2d) - 1)
    a = f - i
    x = path2d[i][0] * (1 - a) + path2d[j][0] * a
    y = path2d[i][1] * (1 - a) + path2d[j][1] * a
    return (x, y, _drape(x, y))


def step(dt: float) -> None:
    if not _S:
        setup()
    _S["t"] += dt
    if _S["t"] > _S["t_end"] + 1.5:
        _S["t"] = 0.0
    pos = _pose(_S["path"], _S["t"], _S["speed"])
    _S["lab"].move("agent0", *pos)
    eye, tgt, up = _S["chase"].update(pos, dt)
    _S["cam"].look(eye, tgt, up)
    _S["lab"].pump()
