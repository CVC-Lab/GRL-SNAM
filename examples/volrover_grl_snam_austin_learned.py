# examples/volrover_grl_snam_austin_learned.py — the LEARNED GRL-SNAM navigator
# driving REAL Austin, TX in a LIVE volrover3 window, using the SDF obstacle model.
#
# A vehicle drives START -> GOAL across the city. At every step the trained SDF
# coefficient net (sdf_nav.CoefMLP) predicts the navigation coefficients and the
# DIFFERENTIABLE SDF surrogate (sdf_nav.sdf_rollout) integrates the motion — the
# learned policy genuinely NAVIGATES the streets (a signed distance field repels
# along the true wall normal, so it doesn't clip corners the way the circular-
# obstacle surrogate did). An A*/occupancy route supplies the global topology
# (GRL-SNAM's stagewise decomposition — pure potential fields have local minima);
# the learned surrogate drives within the street corridor, and a footprint check +
# the route spine guarantee the vehicle never enters a building.
#
# PREP (once, outside volrover3): build the SDF and train the coefficients ->
#   python scripts/build_sdf.py <bundle> --source edt   # or --source cvc (mesh-exact 3-D)
#   python scripts/train_sdf.py <bundle>/nav_sdf.npz -o checkpoints/coef_sdf_austin.pt
# See docs/training-navigation-on-geometry.md.
#
# ENV: GRL_SNAM_SCENE_BUNDLE (scene dir), GRL_SNAM_CHECKPOINT (coef_sdf .pt),
#      GRL_SNAM_SDF (prebuilt nav_sdf.npz; else built from the occupancy at load),
#      GRL_SNAM_ROOT (repo path if not default).
#
# RUN (inside volrover3): Python Console -> Jobs tab -> "Load Script..." -> Run as Job.
# ATTRIBUTION: OpenStreetMap (c) contributors, ODbL; SRTM terrain US public domain.

import math
import os
import sys

import numpy as np

_ROOT = os.environ.get("GRL_SNAM_ROOT", "/home/joe/src/cvc/GRL-SNAM")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

import sdf_nav  # noqa: E402
from pycvc_gl.camera import ChaseCamera  # noqa: E402
from pycvc_gl.lab import Lab  # noqa: E402
from pycvc_gl.scenes import (  # noqa: E402
    building_occupancy, load_geometry_bundle, plan_ground_route, resample_polyline, terrain_grid,
)
from pycvc_gl.vehicle import VehiclePose  # noqa: E402

try:
    import pycvc
    import vrhost
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("run INSIDE volrover3 (Jobs tab -> Load Script)") from exc

_BUNDLE = os.environ.get("GRL_SNAM_SCENE_BUNDLE",
                         os.path.expanduser("~/scenes/austin_south"))  # set to your bundle dir
_CKPT = os.environ.get("GRL_SNAM_CHECKPOINT", os.path.join(_ROOT, "checkpoints", "coef_sdf_austin.pt"))
torch.set_num_threads(2)

# ── trained SDF policy + physics metadata ────────────────────────────────────
_ck = torch.load(_CKPT, map_location="cpu"); _m = _ck["meta"]
_S = float(_m["scale"]); _CTR = np.asarray(_m["center"], np.float32)
_RR = float(_m["rr"]); _DHAT = float(_m["d_hat"]); _DT = float(_m["dt"]); _NSUB = int(_m["nsub"]); _VMAX = float(_m["vmax"])
_REGION = float(_m["region"])
_model = sdf_nav.CoefMLP(); _model.load_state_dict(_ck["model_state_dict"]); _model.eval()

# ── live scene: real Austin terrain + buildings ──────────────────────────────
_app = vrhost.app()
_lab = Lab(app=_app, scene=vrhost.scene()); _lab.set_axis_visible(False)
_sample = load_geometry_bundle(_lab, _BUNDLE)
_bounds = terrain_grid(os.path.join(_BUNDLE, "terrain.json"))[1]; _mnx, _mny, _mxx, _mxy = _bounds
print("grl_snam_austin_learned(SDF): loading occupancy + SDF...")
_occR = building_occupancy(os.path.join(_BUNDLE, "buildings.glb"), _bounds, nx=512, ny=512, inflate_m=12.0)
_occ0 = building_occupancy(os.path.join(_BUNDLE, "buildings.glb"), _bounds, nx=512, ny=512, inflate_m=0.0)
_NY, _NX = _occ0.shape

_sdf_npz = os.environ.get("GRL_SNAM_SDF", "")
if _sdf_npz and os.path.exists(_sdf_npz):
    _d = np.load(_sdf_npz); _phi, _nxg, _nyg = _d["phi"], _d["normal_x"], _d["normal_y"]
else:  # build the footprint-EDT SDF from the occupancy (no extra deps)
    _phi, _nxg, _nyg = sdf_nav.build_sdf(_occ0, _bounds, _S)
_field = sdf_nav.SDFField(_phi, _nxg, _nyg, _bounds, _CTR, _S, device="cpu")


def _w2n(p):
    return np.array([(p[0] - _CTR[0]) * _S, (p[1] - _CTR[1]) * _S], np.float32)


def _n2w(on):
    return np.array([on[0] / _S + _CTR[0], on[1] / _S + _CTR[1]], np.float32)


def _in_building(xw, yw):
    c = int((xw - _mnx) / (_mxx - _mnx) * (_NX - 1)); r = int((yw - _mny) / (_mxy - _mny) * (_NY - 1))
    return 0 <= r < _NY and 0 <= c < _NX and bool(_occ0[r, c])


# ── global route (the stage planner): collision-free spine START -> GOAL ──────
_START = (-_REGION * 0.9, -_REGION * 0.75); _GOAL = (_REGION * 0.9, _REGION * 0.8)
print("grl_snam_austin_learned(SDF): planning route...")
_route = plan_ground_route(_occR, _bounds, [_START, _GOAL], close_loop=False)
if not _route or len(_route) < 2:
    raise RuntimeError("route planning failed for this bundle/region")
_ROUTE = np.asarray(resample_polyline(_route, spacing=0.12 / _S), np.float32)
_lab.add_markers("start", [(_START[0], _START[1], _sample(*_START) + 1.0)], color=(0.15, 0.85, 0.25))
_lab.add_markers("goal", [(_GOAL[0], _GOAL[1], _sample(*_GOAL) + 1.0)], color=(0.95, 0.80, 0.10))
_lab.add_path("spine", [(w[0], w[1], _sample(w[0], w[1]) + 0.5) for w in _ROUTE], color=(0.30, 0.55, 0.95))


def _vehicle_mesh():
    L, W, Hh = 4.6, 2.0, 1.6; hx, hy = L / 2, W / 2
    v = [-hx, -hy, 0, hx, -hy, 0, hx, hy, 0, -hx, hy, 0, -hx, -hy, Hh, hx, -hy, Hh, hx, hy, Hh, -hx, hy, Hh]
    t = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 1, 2, 6, 1, 6, 5, 0, 7, 4, 0, 3, 7, 3, 2, 6, 3, 6, 7, 0, 5, 1, 0, 4, 5]
    return v, t


_vv, _vt = _vehicle_mesh(); _lab.add_mesh("agent0", _vv, _vt, color=(0.90, 0.12, 0.12))
_vpose = VehiclePose(_sample, lift=0.25)

# ── learned SDF navigation state ─────────────────────────────────────────────
_CORR = 0.35 / _S; _LOOK = 8
_o = torch.from_numpy(_w2n(_ROUTE[0])).unsqueeze(0).float(); _v = torch.zeros(1, 2)
_ri = 0; _stall = 0; _done = False
_kw = dict(rr=_RR, d_hat=_DHAT, dt=_DT, vmax=_VMAX)


@torch.no_grad()
def _nav_step():
    """One learned-SDF step: the surrogate drives toward the look-ahead sub-goal and
    is accepted while it stays in the street corridor and out of footprints; else the
    drive advances along the clean spine. Returns the vehicle world (x, y)."""
    global _o, _v, _ri, _stall, _done
    if _done:
        w = _ROUTE[_ri]; return float(w[0]), float(w[1])
    sub = _ROUTE[min(_ri + _LOOK, len(_ROUTE) - 1)]; goal = torch.from_numpy(_w2n(sub)).unsqueeze(0)
    al, be, ga = _model(sdf_nav.coef_feats(_field, _o, goal))
    o2, v2, _ = sdf_nav.sdf_rollout(_field, _o, _v, goal, al, be, ga, 1, nsub=_NSUB, **_kw)
    w = _n2w(o2[0].numpy())
    lo = max(0, _ri - 2); seg = _ROUTE[lo:min(_ri + 3 * _LOOK, len(_ROUTE))]
    j = lo + int(np.argmin(np.linalg.norm(seg - w, axis=1)))
    in_corr = (not _in_building(float(w[0]), float(w[1]))) and np.linalg.norm(w - _ROUTE[j]) < _CORR
    if in_corr:
        _o, _v = o2, v2; _stall = _stall + 1 if j <= _ri else 0; _ri = max(_ri, j)
    else:
        _stall += 1
    if (not in_corr) or _stall > 25:                                  # spine nudge if strayed or stuck
        _ri = min(_ri + 1, len(_ROUTE) - 1); w = _ROUTE[_ri].copy()
        _o = torch.from_numpy(_w2n(w)).unsqueeze(0); _v = torch.zeros(1, 2); _stall = 0
    if np.linalg.norm(w - np.asarray(_GOAL, np.float32)) < 40.0 or _ri >= len(_ROUTE) - 1:
        _done = True
    return float(w[0]), float(w[1])


# ── chase camera ─────────────────────────────────────────────────────────────
_CAM = "volrover3.camera"; _chase = ChaseCamera(back=34.0, height=13.0, look_up=2.5, up=(0.0, 0.0, 1.0))


def _cset(k, val):
    pycvc.state_set(_app, _CAM + "." + k, "%.6f" % float(val))


def _drive_cam(eye, tgt, up):
    vx, vy, vz = (tgt[i] - eye[i] for i in range(3)); mm = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
    _cset("position.x", eye[0]); _cset("position.y", eye[1]); _cset("position.z", eye[2])
    _cset("view_direction.x", vx / mm); _cset("view_direction.y", vy / mm); _cset("view_direction.z", vz / mm)
    _cset("up_vector.x", up[0]); _cset("up_vector.y", up[1]); _cset("up_vector.z", up[2]); _cset("fov", 60.0)


def _place(x, y, dt):
    _lab.node("agent0").setTransform(_vpose.update(x, y, dt))


_STEPS_PER_FRAME = 3; _FDT = 1.0 / 30.0
for _i in range(30):
    _place(_START[0], _START[1], _FDT)
    _e, _g, _u = _chase.update((_START[0], _START[1], _sample(*_START)), _FDT)
_drive_cam(_e, _g, _u); _lab.pump()
print("grl_snam_austin_learned(SDF): the learned SDF policy is driving Austin. Pause/stop from Jobs.")


def step(dt):
    for _ in range(_STEPS_PER_FRAME):
        x, y = _nav_step()
    _place(x, y, dt)
    eye, tgt, up = _chase.update((x, y, _sample(x, y)), dt)
    _drive_cam(eye, tgt, up); _lab.pump()
