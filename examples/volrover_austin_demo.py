# examples/volrover_austin_demo.py — a robot patrols REAL Austin, TX geometry in
# a LIVE volrover3 window, GROUNDED: it drives the STREETS (routed around the
# buildings, draped on the terrain), filmed by a low third-person chase camera.
#
# Loads a geometry_bundle (terrain heightfield + glTF city mesh) — by default an
# "austin_south" OSM/SRTM bundle (3 km x 3 km around the UT campus) — rasterizes the
# buildings into an occupancy grid, A*-plans a grounded patrol loop through the
# free space (streets), drapes it on the real terrain, and follows the vehicle
# with grl_snam_lab.camera.ChaseCamera (position-driven, ready for live GRL-SNAM
# paths).
#
# Load it from volrover3's Python Console -> Jobs tab -> "Load Script..." ->
# "Run as Job" (the glTF is ~978k triangles, so the first tick takes several
# seconds to load + rasterize). Point it at a different bundle with the
# GRL_SNAM_SCENE_BUNDLE env var.
#
# ATTRIBUTION: the geometry is derived from OpenStreetMap (c) OpenStreetMap
# contributors, ODbL (https://openstreetmap.org/copyright); SRTM terrain is US
# public domain. Credit OpenStreetMap in any published render.
#
# LAYERING: volrover3 does NOT depend on GRL-SNAM (or on the scene assets). This
# runs under volrover3's generic job runner when GRL-SNAM is installed and the
# bundle is present.

import math
import os

from grl_snam_lab.camera import ChaseCamera
from grl_snam_lab.lab import Lab
from grl_snam_lab.scenes import (
    building_occupancy,
    load_geometry_bundle,
    plan_ground_route,
    resample_polyline,
    terrain_grid,
)
from grl_snam_lab.vehicle import VehiclePose

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
    os.path.expanduser("~/scenes/austin_south"),  # set GRL_SNAM_SCENE_BUNDLE to your bundle dir
)

_app = vrhost.app()
_lab = Lab(app=_app, scene=vrhost.scene())
_lab.set_axis_visible(False)  # hide the XYZ gnomon — a distraction for this demo
# Real Austin terrain + buildings; `_sample(x, y)` is the terrain height (drape).
_sample = load_geometry_bundle(_lab, _BUNDLE)

# ── plan a GROUNDED patrol route through the streets (around the buildings) ───
# Rasterize the city into an occupancy grid, then A*-route a closed patrol loop
# through the free space and drape it on the terrain. The vehicle stays on the
# ground and never drives through a building.
_bounds = terrain_grid(os.path.join(_BUNDLE, "terrain.json"))[1]
print("volrover_austin_demo: building/loading the building occupancy grid...")
# SOLID footprint mask (filled), grown by ~12 m so a vehicle keeps clear of walls.
# Cached next to the .glb, so this is instant after the first run.
_occ = building_occupancy(os.path.join(_BUNDLE, "buildings.glb"), _bounds, nx=512, ny=512, inflate_m=12.0)

_R = 430.0  # patrol radius; waypoints are snapped to the nearest open street
_WAYPTS = [(_R * math.cos(a), _R * math.sin(a)) for a in
           [0.0, math.pi / 3, 2 * math.pi / 3, math.pi, 4 * math.pi / 3, 5 * math.pi / 3]]
_route2d = resample_polyline(plan_ground_route(_occ, _bounds, _WAYPTS, close_loop=True), spacing=4.0)
if len(_route2d) < 2:  # pathing failed (bad bundle) — fall back to a draped ring
    _route2d = [( _R * math.cos(2 * math.pi * k / 240), _R * math.sin(2 * math.pi * k / 240))
                for k in range(241)]
print("volrover_austin_demo: grounded route = %d pts along the streets." % len(_route2d))

_PPS = 3.0  # route points/sec -> ~12 m/s at 4 m spacing


def _pose(t):
    # (x, y) along the uniformly-resampled loop (seamless wrap). Draping (z) and
    # heading are handled downstream — VehiclePose derives heading + slope from the
    # position stream, and the chase camera drapes its own target.
    n = len(_route2d)
    f = (t * _PPS) % n
    i = int(f)
    j = (i + 1) % n
    a = f - i
    x = _route2d[i][0] * (1 - a) + _route2d[j][0] * a
    y = _route2d[i][1] * (1 - a) + _route2d[j][1] * a
    return x, y


def _vehicle_mesh():
    # A car-sized box in LOCAL coords: forward = +X, centered in X/Y, base at z=0.
    L, W, H = 4.6, 2.0, 1.6
    hx, hy = L / 2.0, W / 2.0
    v = [-hx, -hy, 0, hx, -hy, 0, hx, hy, 0, -hx, hy, 0,
         -hx, -hy, H, hx, -hy, H, hx, hy, H, -hx, hy, H]
    t = [0, 1, 2, 0, 2, 3,   4, 6, 5, 4, 7, 6,   1, 2, 6, 1, 6, 5,
         0, 7, 4, 0, 3, 7,   3, 2, 6, 3, 6, 7,   0, 5, 1, 0, 4, 5]
    return v, t


_vv, _vt = _vehicle_mesh()
_lab.add_mesh("agent0", _vv, _vt, color=(0.90, 0.12, 0.12))
_lab.add_path("agent0_track", [(x, y, _sample(x, y) + 0.6) for x, y in _route2d], color=(0.95, 0.75, 0.10))

# Smoothly yaw the vehicle to its heading AND pitch/roll it to the terrain normal
# (banks over crests, leans on side-slopes) with camera-style damping.
_vpose = VehiclePose(_sample, lift=0.25)


def _place_vehicle(x, y, dt):
    _lab.node("agent0").setTransform(_vpose.update(x, y, dt))


# ── low third-person chase camera (grounded, street-level) ───────────────────
_CAM = "volrover3.camera"
_chase = ChaseCamera(back=34.0, height=13.0, look_up=2.5, up=(0.0, 0.0, 1.0))


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


def _cam_pos(x, y):
    # feed the chase camera the vehicle's draped ground position (x, y, z)
    return (x, y, _sample(x, y))


_DT = 1.0 / 30.0
_t = 0.0
for _i in range(30):  # prime the camera + vehicle smoothing to a settled pose
    _x, _y = _pose(_i * _DT)
    _place_vehicle(_x, _y, _DT)
    _e, _g, _u = _chase.update(_cam_pos(_x, _y), _DT)
_drive(_e, _g, _u)
_t = 30 * _DT
_lab.pump()

print(
    "grl_snam_lab: Austin GROUNDED chase-cam demo loaded (OSM/ODbL + SRTM terrain). "
    "The red agent drives the streets, routed around the buildings; camera follows. "
    "Pause/stop from the Jobs tab."
)


def step(dt):
    global _t
    _t += dt
    x, y = _pose(_t)
    _place_vehicle(x, y, dt)  # yaw to heading + pitch/roll to the terrain, smoothed
    eye, tgt, up = _chase.update(_cam_pos(x, y), dt)
    _drive(eye, tgt, up)
    _lab.pump()
