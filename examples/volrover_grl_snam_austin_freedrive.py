# examples/volrover_grl_snam_austin_freedrive.py — the learned GRL-SNAM navigator
# finding its OWN way across REAL Austin, TX in a LIVE volrover3 window, with NO A*
# route. Pick a START and a GOAL; the trained SDF policy drives there end-to-end,
# purely by reacting to the buildings: a moving carrot toward the goal + the SDF
# barrier deflecting around footprints. This is "the vehicle finds its own path
# from A to B based on environmental factors" — no precomputed path at all.
#
# vs volrover_grl_snam_austin_learned.py (which uses an A* route as a stage spine):
# this one is pure end-to-end learned navigation. A pure potential field can stall
# in a local minimum (a dead-end / U-shaped block) for adversarial A->B pairs, so
# the START/GOAL below are chosen to be navigable; retarget as you like.
#
# PREP (once): build the SDF + train the coefficients (see docs) ->
#   python scripts/build_sdf.py <bundle> --source edt
#   python scripts/train_sdf.py <bundle>/nav_sdf.npz -o checkpoints/coef_sdf_austin.pt
# ENV: GRL_SNAM_SCENE_BUNDLE, GRL_SNAM_CHECKPOINT (coef_sdf .pt), GRL_SNAM_SDF
#      (prebuilt nav_sdf.npz; else built from occupancy at load), GRL_SNAM_ROOT.
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
from pycvc_gl.scenes import building_occupancy, load_geometry_bundle, terrain_grid  # noqa: E402
from pycvc_gl.vehicle import VehiclePose  # noqa: E402

try:
    import pycvc
    import vrhost
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("run INSIDE volrover3 (Jobs tab -> Load Script)") from exc

_BUNDLE = os.environ.get("GRL_SNAM_SCENE_BUNDLE",
                         "/home/joe/src/cvc/CVC-DBG/platoon-sim/scene_viewer/exports/scenes/austin_south")
_CKPT = os.environ.get("GRL_SNAM_CHECKPOINT", os.path.join(_ROOT, "checkpoints", "coef_sdf_austin.pt"))
torch.set_num_threads(2)

_ck = torch.load(_CKPT, map_location="cpu"); _m = _ck["meta"]
_S = float(_m["scale"]); _CTR = np.asarray(_m["center"], np.float32)
_RR = float(_m["rr"]); _DHAT = float(_m["d_hat"]); _DT = float(_m["dt"]); _NSUB = int(_m["nsub"]); _VMAX = float(_m["vmax"])
_model = sdf_nav.CoefMLP(); _model.load_state_dict(_ck["model_state_dict"]); _model.eval()

# START -> GOAL (world x,y). Validated navigable on austin_south; retarget freely.
_START = (-361.0, 114.0)
_GOAL = (185.0, 50.0)

_app = vrhost.app()
_lab = Lab(app=_app, scene=vrhost.scene()); _lab.set_axis_visible(False)
_sample = load_geometry_bundle(_lab, _BUNDLE)
_bounds = terrain_grid(os.path.join(_BUNDLE, "terrain.json"))[1]; _mnx, _mny, _mxx, _mxy = _bounds
print("grl_snam_austin_freedrive: loading occupancy + SDF...")
_occ0 = building_occupancy(os.path.join(_BUNDLE, "buildings.glb"), _bounds, nx=512, ny=512, inflate_m=0.0)
_NY, _NX = _occ0.shape
_sdf_npz = os.environ.get("GRL_SNAM_SDF", "")
if _sdf_npz and os.path.exists(_sdf_npz):
    _d = np.load(_sdf_npz); _phi, _nxg, _nyg = _d["phi"], _d["normal_x"], _d["normal_y"]
else:
    _phi, _nxg, _nyg = sdf_nav.build_sdf(_occ0, _bounds, _S)
_field = sdf_nav.SDFField(_phi, _nxg, _nyg, _bounds, _CTR, _S, device="cpu")
_kw = dict(rr=_RR, d_hat=_DHAT, dt=_DT, vmax=_VMAX)


def _w2n(p):
    return np.array([(p[0] - _CTR[0]) * _S, (p[1] - _CTR[1]) * _S], np.float32)


def _n2w(on):
    return np.array([on[0] / _S + _CTR[0], on[1] / _S + _CTR[1]], np.float32)


_lab.add_markers("start", [(_START[0], _START[1], _sample(*_START) + 1.0)], color=(0.15, 0.85, 0.25))
_lab.add_markers("goal", [(_GOAL[0], _GOAL[1], _sample(*_GOAL) + 1.0)], color=(0.95, 0.80, 0.10))


def _vehicle_mesh():
    L, W, Hh = 4.6, 2.0, 1.6; hx, hy = L / 2, W / 2
    v = [-hx, -hy, 0, hx, -hy, 0, hx, hy, 0, -hx, hy, 0, -hx, -hy, Hh, hx, -hy, Hh, hx, hy, Hh, -hx, hy, Hh]
    t = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 1, 2, 6, 1, 6, 5, 0, 7, 4, 0, 3, 7, 3, 2, 6, 3, 6, 7, 0, 5, 1, 0, 4, 5]
    return v, t


_vv, _vt = _vehicle_mesh(); _lab.add_mesh("agent0", _vv, _vt, color=(0.90, 0.12, 0.12))
_vpose = VehiclePose(_sample, lift=0.25)

# ── end-to-end learned navigation state (no route — carrot toward the true goal) ──
_o = torch.from_numpy(_w2n(_START)).unsqueeze(0).float(); _v = torch.zeros(1, 2)
_GN = _w2n(_GOAL); _done = False


@torch.no_grad()
def _nav_step():
    global _o, _v, _done
    if _done:
        w = _n2w(_o[0].numpy()); return float(w[0]), float(w[1])
    p = _o[0].numpy(); dvec = _GN - p; dist = float(np.linalg.norm(dvec))
    carrot = (p + dvec / (dist + 1e-6) * min(1.8, dist)).astype(np.float32)   # local goal toward the true goal
    gt = torch.from_numpy(carrot).unsqueeze(0)
    al, be, ga = _model(sdf_nav.coef_feats(_field, _o, gt))
    _o, _v, _ = sdf_nav.sdf_rollout(_field, _o, _v, gt, al, be, ga, 1, nsub=_NSUB, **_kw)
    if float(np.linalg.norm(_o[0].numpy() - _GN)) < 0.4:
        _done = True
    w = _n2w(_o[0].numpy()); return float(w[0]), float(w[1])


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
print("grl_snam_austin_freedrive: the learned SDF policy is finding its own way "
      "%s -> %s across Austin (no route). Pause/stop from Jobs." % (_START, _GOAL))


def step(dt):
    for _ in range(_STEPS_PER_FRAME):
        x, y = _nav_step()
    _place(x, y, dt)
    eye, tgt, up = _chase.update((x, y, _sample(x, y)), dt)
    _drive_cam(eye, tgt, up); _lab.pump()
