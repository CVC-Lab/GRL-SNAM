# examples/volrover_grl_snam_austin_learned.py — GRL-SNAM STAGEWISE navigation on
# REAL Austin, TX in a LIVE volrover3 window. A vehicle drives START -> GOAL across
# the city, filmed by a low third-person chase camera.
#
# ARCHITECTURE (as in the GRL-SNAM paper — a stage planner + a learned local policy):
#   * PLANNER / spine:  an A*/occupancy route through the free space is the
#     collision-free "stage" spine (pycvc_gl.scenes.plan_ground_route). This is the
#     part that does the global, environment-based pathfinding around the buildings.
#   * LEARNED local policy:  at every step the trained CoefEnergyNet predicts the
#     navigation energy coefficients, HistSecantController adapts them online, and the
#     differentiable surrogate (integrate_surrogate_v2) integrates the motion. The
#     learned step DRIVES the vehicle whenever it stays inside the street corridor;
#     otherwise the drive advances along the clean route (a footprint check + the
#     route spine guarantee the vehicle never enters a building).
#
# SCOPE (be honest): a dense rectilinear city is *hard* for a point-agent potential
# field — thousands of circular obstacle barriers conflict — so on Austin the route
# spine carries the global path and the learned policy contributes local reactive
# control. The learned surrogate's *full* end-to-end navigation is best seen in its
# native sparse-obstacle regime: examples/volrover_grl_snam_planner.py.
#
# NEEDS: the volrover3 embedded env (pycvc / pycvc_gl / vtk-python), torch (a GRL-SNAM
# dep), the GRL-SNAM repo importable, the Austin bundle on disk, and a trained
# checkpoint (scripts/train_on_geometry.py; see docs/training-navigation-on-geometry.md).
#   GRL_SNAM_SCENE_BUNDLE  scene dir (default austin_south)
#   GRL_SNAM_CHECKPOINT    trained coef_energy checkpoint (.pt)
#   GRL_SNAM_ROOT          the GRL-SNAM repo, if not on the default path
#
# RUN (inside volrover3): Python Console -> Jobs tab -> "Load Script..." -> Run as Job.
#
# ATTRIBUTION: geometry derived from OpenStreetMap (c) OpenStreetMap contributors, ODbL
# (https://openstreetmap.org/copyright); SRTM terrain is US public domain.

import math
import os
import sys
import types

import numpy as np

_ROOT = os.environ.get("GRL_SNAM_ROOT", "/home/joe/src/cvc/GRL-SNAM")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# eval_coef_energy pulls imageio + scripts.* at import time; stub them (not needed here).
for _m in ["imageio", "imageio.v3", "scripts.ring_dataset_maxmin", "scripts.spline_stagewise6"]:
    sys.modules.setdefault(_m, types.ModuleType(_m))

import torch  # noqa: E402

from grl_snam.adaptation import HistSecantController  # noqa: E402
from grl_snam.dynamics import integrate_surrogate_v2  # noqa: E402
from grl_snam.network import CoefEnergyNet  # noqa: E402
from eval_coef_energy import build_local_feats  # noqa: E402

from pycvc_gl.camera import ChaseCamera  # noqa: E402
from pycvc_gl.lab import Lab  # noqa: E402
from pycvc_gl.scenes import (  # noqa: E402
    building_occupancy,
    load_geometry_bundle,
    plan_ground_route,
    resample_polyline,
    terrain_grid,
)
from pycvc_gl.vehicle import VehiclePose  # noqa: E402

try:
    import pycvc
    import vrhost
except ImportError as exc:  # pragma: no cover - only meaningful inside volrover3
    raise RuntimeError(
        "volrover_grl_snam_austin_learned: run INSIDE volrover3 (Jobs tab -> Load Script)."
    ) from exc

_BUNDLE = os.environ.get(
    "GRL_SNAM_SCENE_BUNDLE",
    "/home/joe/src/cvc/CVC-DBG/platoon-sim/scene_viewer/exports/scenes/austin_south",
)
_CKPT = os.environ.get("GRL_SNAM_CHECKPOINT", os.path.join(_ROOT, "checkpoints", "coef_energy_austin.pt"))

# ── trained policy + its scene/physics metadata ──────────────────────────────
_ck = torch.load(_CKPT, map_location="cpu")
_meta = _ck["meta"]
_S = float(_meta["scale"])
_CTR = np.asarray(_meta.get("center", [0.0, 0.0]), np.float32)
_RADN = float(_meta["radn"]); _RR = float(_meta["rr"]); _DHAT = float(_meta["d_hat"]); _DT = float(_meta["dt"])
_BLOCK = int(_meta.get("block", 2))
_model = CoefEnergyNet(); _model.load_state_dict(_ck["model_state_dict"]); _model.eval()
torch.set_num_threads(2)


def _w2n(p):
    return (np.asarray(p, np.float32) - _CTR) * _S


def _n2w(p):
    return np.asarray(p, np.float32) / _S + _CTR


# ── live scene: real Austin terrain + buildings ──────────────────────────────
_app = vrhost.app()
_lab = Lab(app=_app, scene=vrhost.scene())
_lab.set_axis_visible(False)
_sample = load_geometry_bundle(_lab, _BUNDLE)
_bounds = terrain_grid(os.path.join(_BUNDLE, "terrain.json"))[1]
_mnx, _mny, _mxx, _mxy = _bounds
print("grl_snam_austin_learned: loading occupancy grids (cached next to the .glb)...")
_occR = building_occupancy(os.path.join(_BUNDLE, "buildings.glb"), _bounds, nx=512, ny=512, inflate_m=12.0)
_occ0 = building_occupancy(os.path.join(_BUNDLE, "buildings.glb"), _bounds, nx=512, ny=512, inflate_m=0.0)
_NY, _NX = _occ0.shape


def _in_building(xw, yw):
    c = int((xw - _mnx) / (_mxx - _mnx) * (_NX - 1))
    r = int((yw - _mny) / (_mxy - _mny) * (_NY - 1))
    return 0 <= r < _NY and 0 <= c < _NX and bool(_occ0[r, c])


# obstacle circles the learned policy repels from — coarsen the footprint mask the
# SAME way training did (block -> one circle per cell), in NORMALIZED coords.
_cny, _cnx = _NY // _BLOCK, _NX // _BLOCK
_cocc = _occ0[: _cny * _BLOCK, : _cnx * _BLOCK].reshape(_cny, _BLOCK, _cnx, _BLOCK).any(axis=(1, 3))
_csx = (_mxx - _mnx) / _cnx; _csy = (_mxy - _mny) / _cny
_ys, _xs = np.where(_cocc)
_CENTERS_N = _w2n(np.stack([_mnx + (_xs + 0.5) * _csx, _mny + (_ys + 0.5) * _csy], 1)).astype(np.float32)


def _nearby(p_n, win=3.0, k=32):
    d2 = ((_CENTERS_N - p_n) ** 2).sum(1)
    idx = np.argsort(d2)[:k]
    return np.ascontiguousarray(_CENTERS_N[idx[d2[idx] < win * win]])


# ── plan the collision-free A* spine (START -> GOAL across the working region) ──
_REGION = float(_meta.get("region", 430.0))
_START = (-_REGION * 0.9, -_REGION * 0.75)
_GOAL = (_REGION * 0.9, _REGION * 0.8)
print("grl_snam_austin_learned: planning route %s -> %s ..." % (_START, _GOAL))
_route = plan_ground_route(_occR, _bounds, [_START, _GOAL], close_loop=False)
if not _route or len(_route) < 2:
    raise RuntimeError("route planning failed for this bundle/region")
_ROUTE = np.asarray(resample_polyline(_route, spacing=0.12 / _S), np.float32)  # dense spine (world)
print("grl_snam_austin_learned: %d-point collision-free spine." % len(_ROUTE))

_lab.add_markers("start", [(_START[0], _START[1], _sample(*_START) + 1.0)], color=(0.15, 0.85, 0.25))
_lab.add_markers("goal", [(_GOAL[0], _GOAL[1], _sample(*_GOAL) + 1.0)], color=(0.95, 0.80, 0.10))
_lab.add_path("spine", [(w[0], w[1], _sample(w[0], w[1]) + 0.5) for w in _ROUTE], color=(0.30, 0.55, 0.95))


def _vehicle_mesh():
    L, W, H = 4.6, 2.0, 1.6
    hx, hy = L / 2.0, W / 2.0
    v = [-hx, -hy, 0, hx, -hy, 0, hx, hy, 0, -hx, hy, 0,
         -hx, -hy, H, hx, -hy, H, hx, hy, H, -hx, hy, H]
    t = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 1, 2, 6, 1, 6, 5,
         0, 7, 4, 0, 3, 7, 3, 2, 6, 3, 6, 7, 0, 5, 1, 0, 4, 5]
    return v, t


_vv, _vt = _vehicle_mesh()
_lab.add_mesh("agent0", _vv, _vt, color=(0.90, 0.12, 0.12))
_vpose = VehiclePose(_sample, lift=0.25)

# ── learned navigation state (stagewise: spine index + learned local policy) ──
_CORRIDOR = 0.35 / _S   # how far a learned step may stray from the spine (world)
_LOOK = 8               # sub-goal look-ahead along the spine (indices)
_o = torch.from_numpy(_w2n(_ROUTE[0])).unsqueeze(0).float()
_v = torch.zeros(1, 2)
_ri = 0
_ctrl = HistSecantController(k_alpha=2, safe_margin=max(2 * _RR, 0.05))
_done = False


def _nav_step():
    """One stagewise step. The learned policy proposes a move toward the look-ahead
    sub-goal; accept it only if it stays in the street corridor and out of footprints,
    else advance along the clean spine. Returns the vehicle world (x, y)."""
    global _o, _v, _ri, _done
    if _done:
        w = _ROUTE[_ri]
        return float(w[0]), float(w[1])
    sub = _ROUTE[min(_ri + _LOOK, len(_ROUTE) - 1)]
    g_n = _w2n(sub)
    p = _o[0].numpy()
    nb = _nearby(p)
    Rw = np.full(len(nb), _RADN, np.float32)
    of, gf = build_local_feats(p, g_n, nb, Rw, np.ones(len(nb), np.float32))
    mk = torch.ones(1, of.shape[1], dtype=torch.bool) if of.shape[1] else torch.zeros(1, 0, dtype=torch.bool)
    with torch.no_grad():
        al, be, ga = _model(of, mk, gf)
    if len(nb) >= 2:
        clr = float(np.min(np.linalg.norm(p[None] - nb, axis=1) - _RADN))
        al, be, ga = _ctrl.update(al, be, ga, p, _v[0].numpy(), g_n, nb, Rw, np.ones(len(nb), np.float32),
                                  clr, float(np.linalg.norm(p - g_n)), float(np.linalg.norm(_v[0].numpy())))
    else:
        _ctrl.prev = None; _ctrl.J = None
    C = torch.from_numpy(nb).unsqueeze(0) if len(nb) else torch.zeros(1, 0, 2)
    Rt = torch.from_numpy(Rw).unsqueeze(0) if len(nb) else torch.zeros(1, 0)
    m2 = torch.ones(1, len(nb), dtype=torch.bool) if len(nb) else torch.zeros(1, 0, dtype=torch.bool)
    o_new, v_new, _ = integrate_surrogate_v2(_o, _v, torch.from_numpy(g_n).unsqueeze(0), C, Rt, m2,
                                             al, be, ga, torch.tensor([_DHAT]), torch.tensor([_DT]),
                                             torch.tensor([1]), robot_radius=torch.tensor([_RR]), margin_factor=0.5)
    w_new = _n2w(o_new[0].numpy())
    seg = _ROUTE[_ri: min(_ri + 3 * _LOOK, len(_ROUTE))]
    j = _ri + int(np.argmin(np.linalg.norm(seg - w_new, axis=1)))
    if (not _in_building(float(w_new[0]), float(w_new[1]))) and np.linalg.norm(w_new - _ROUTE[j]) < _CORRIDOR and j > _ri:
        _o, _v, _ri = o_new, v_new, j            # learned step accepted
    else:
        _ri = min(_ri + 1, len(_ROUTE) - 1)      # clean spine fallback
        w_new = _ROUTE[_ri].copy()
        _o = torch.from_numpy(_w2n(w_new)).unsqueeze(0); _v = torch.zeros(1, 2)
    if _ri >= len(_ROUTE) - 1:
        _done = True
    return float(w_new[0]), float(w_new[1])


# ── low third-person chase camera ────────────────────────────────────────────
_CAM = "volrover3.camera"
_chase = ChaseCamera(back=34.0, height=13.0, look_up=2.5, up=(0.0, 0.0, 1.0))


def _cset(k, val):
    pycvc.state_set(_app, _CAM + "." + k, "%.6f" % float(val))


def _drive_cam(eye, tgt, up):
    vx, vy, vz = (tgt[i] - eye[i] for i in range(3))
    m = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
    _cset("position.x", eye[0]); _cset("position.y", eye[1]); _cset("position.z", eye[2])
    _cset("view_direction.x", vx / m); _cset("view_direction.y", vy / m); _cset("view_direction.z", vz / m)
    _cset("up_vector.x", up[0]); _cset("up_vector.y", up[1]); _cset("up_vector.z", up[2])
    _cset("fov", 60.0)


def _place(x, y, dt):
    _lab.node("agent0").setTransform(_vpose.update(x, y, dt))


_STEPS_PER_FRAME = 3
_FDT = 1.0 / 30.0
for _i in range(30):  # settle vehicle + camera at the start
    _place(_START[0], _START[1], _FDT)
    _e, _g, _u = _chase.update((_START[0], _START[1], _sample(*_START)), _FDT)
_drive_cam(_e, _g, _u)
_lab.pump()
print("grl_snam_austin_learned: driving Austin START->GOAL (A* spine + learned "
      "CoefEnergyNet/HistSecant local policy). Pause/stop from the Jobs tab.")


def step(dt):
    for _ in range(_STEPS_PER_FRAME):
        x, y = _nav_step()
    _place(x, y, dt)
    eye, tgt, up = _chase.update((x, y, _sample(x, y)), dt)
    _drive_cam(eye, tgt, up)
    _lab.pump()
