"""Live demo: a grounded A* street patrol on real Austin, TX. The city mesh is
rasterized to an occupancy grid, an A* route threads a closed patrol loop through the
free space (streets), and a vehicle drives it draped on the terrain under a low chase
camera — a non-learned baseline (and a clean showcase of the scene + routing stack).
Env: GRL_SNAM_SCENE_BUNDLE.
"""

from __future__ import annotations

import math
import os

from grl_snam.demos._common import CameraDriver, require_host, vehicle_box_mesh

_S: dict = {}


def setup() -> None:
    pycvc, vrhost = require_host()
    from pycvc_gl.camera import ChaseCamera
    from pycvc_gl.lab import Lab
    from pycvc_gl.scenes import (
        building_occupancy,
        load_geometry_bundle,
        plan_ground_route,
        resample_polyline,
        terrain_grid,
    )
    from pycvc_gl.vehicle import VehiclePose

    from grl_snam.route import cells_for_metres, plan_clearance_route

    bundle = os.environ.get("GRL_SNAM_SCENE_BUNDLE", os.path.expanduser("~/scenes/austin_south"))
    app = vrhost.app()
    lab = Lab(app=app, scene=vrhost.scene())
    lab.set_axis_visible(False)
    sample = load_geometry_bundle(lab, bundle)
    bounds = terrain_grid(os.path.join(bundle, "terrain.json"))[1]
    print("grl_snam austin_patrol: building occupancy + A* patrol route...", flush=True)
    occ = building_occupancy(
        os.path.join(bundle, "buildings.glb"), bounds, 512, 512, inflate_m=12.0
    )
    r = 430.0
    waypts = [
        (r * math.cos(a), r * math.sin(a))
        for a in [0.0, math.pi / 3, 2 * math.pi / 3, math.pi, 4 * math.pi / 3, 5 * math.pi / 3]
    ]
    # A patrol that threads the middle of the streets rather than clipping the
    # corners: same standoff surcharge the learned demo routes through.
    standoff_m = float(os.environ.get("GRL_SNAM_ROUTE_STANDOFF_M", "12"))
    _loop = (
        plan_clearance_route(
            occ,
            bounds,
            waypts,
            close_loop=True,
            d_safe=cells_for_metres(bounds, occ.shape, standoff_m),
        )
        if standoff_m > 0
        else plan_ground_route(occ, bounds, waypts, close_loop=True)
    )
    route2d = resample_polyline(_loop, spacing=4.0)
    if len(route2d) < 2:
        route2d = [
            (r * math.cos(2 * math.pi * k / 240), r * math.sin(2 * math.pi * k / 240))
            for k in range(241)
        ]
    print("grl_snam austin_patrol: grounded route = %d pts along the streets." % len(route2d))
    vv, vt = vehicle_box_mesh()
    lab.add_mesh("agent0", vv, vt, color=(0.90, 0.12, 0.12))
    lab.add_path(
        "agent0_track", [(x, y, sample(x, y) + 0.6) for x, y in route2d], color=(0.95, 0.75, 0.10)
    )
    vpose = VehiclePose(sample, lift=0.25)
    chase = ChaseCamera(back=34.0, height=13.0, look_up=2.5, up=(0.0, 0.0, 1.0))
    lab.node("agent0").setTransform(vpose.update(route2d[0][0], route2d[0][1], 1.0 / 30.0))
    lab.pump()
    print("grl_snam austin_patrol: the vehicle patrols the streets, routed around buildings.")
    _S.update(
        lab=lab,
        route=route2d,
        sample=sample,
        vpose=vpose,
        chase=chase,
        cam=CameraDriver(app, pycvc),
        t=0.0,
        pps=3.0,
    )


def _pose(t):
    route = _S["route"]
    n = len(route)
    f = (t * _S["pps"]) % n
    i = int(f)
    j = (i + 1) % n
    a = f - i
    return route[i][0] * (1 - a) + route[j][0] * a, route[i][1] * (1 - a) + route[j][1] * a


def step(dt: float) -> None:
    if not _S:
        setup()
    _S["t"] += dt
    x, y = _pose(_S["t"])
    _S["lab"].node("agent0").setTransform(_S["vpose"].update(x, y, dt))
    eye, tgt, up = _S["chase"].update((x, y, _S["sample"](x, y)), dt)
    _S["cam"].look(eye, tgt, up)
    _S["lab"].pump()
