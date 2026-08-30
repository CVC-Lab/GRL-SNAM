"""Live demo: the learned SDF navigator driving real Austin with an A* route spine
(GRL-SNAM's stagewise decomposition). An A*/occupancy route supplies global topology;
the trained SDF surrogate drives locally within the street corridor, accepted while it
stays out of footprints, else nudged along the clean spine — robust for adversarial
start/goal pairs where a pure potential field would stall. Live metrics publish each
step. Prep + env are the same as ``austin_freedrive`` (build-sdf + train).
"""

from __future__ import annotations

import os

import numpy as np
import torch

import sdf_nav
from grl_snam.demos._common import CameraDriver, MetricsPublisher, require_host, vehicle_box_mesh
from grl_snam.metrics import NavMetrics

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
    ckpt = os.environ.get("GRL_SNAM_CHECKPOINT", "checkpoints/coef_sdf.pt")
    torch.set_num_threads(2)
    ck = torch.load(ckpt, map_location="cpu")
    meta = ck["meta"]
    model = sdf_nav.CoefMLP()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    scale = float(meta["scale"])
    center = np.asarray(meta["center"], np.float32)
    region = float(meta["region"])
    kw = dict(
        rr=float(meta["rr"]),
        d_hat=float(meta["d_hat"]),
        dt=float(meta["dt"]),
        vmax=float(meta["vmax"]),
    )
    nsub = int(meta["nsub"])

    app = vrhost.app()
    lab = Lab(app=app, scene=vrhost.scene())
    lab.set_axis_visible(False)
    sample = load_geometry_bundle(lab, bundle)
    bounds = terrain_grid(os.path.join(bundle, "terrain.json"))[1]
    mnx, mny, mxx, mxy = bounds
    print("grl_snam austin_learned: occupancy + SDF + route...", flush=True)
    occ_r = building_occupancy(
        os.path.join(bundle, "buildings.glb"), bounds, 512, 512, inflate_m=12.0
    )
    occ0 = building_occupancy(
        os.path.join(bundle, "buildings.glb"), bounds, 512, 512, inflate_m=0.0
    )
    ny, nx = occ0.shape
    sdf_npz = os.environ.get("GRL_SNAM_SDF", "")
    if sdf_npz and os.path.exists(sdf_npz):
        d = np.load(sdf_npz)
        phi, nxg, nyg = d["phi"], d["normal_x"], d["normal_y"]
    else:
        phi, nxg, nyg = sdf_nav.build_sdf(occ0, bounds, scale)
    field = sdf_nav.SDFField(phi, nxg, nyg, bounds, center, scale, device="cpu")

    start = (-region * 0.9, -region * 0.75)
    goal = (region * 0.9, region * 0.8)
    # Clearance-weighted spine, not the shortest path. The shortest route hugs
    # building corners because that is what shortest means, and the local drive
    # then spends the whole run fighting its own wall barrier. A ~12 m standoff
    # target lifts route-guided reach ~0.80 -> ~0.90 on the procedural city and
    # gives visibly smoother paths here. Set GRL_SNAM_ROUTE_STANDOFF_M=0 for the
    # old shortest-path behaviour.
    standoff_m = float(os.environ.get("GRL_SNAM_ROUTE_STANDOFF_M", "12"))
    if standoff_m > 0:
        route = plan_clearance_route(
            occ_r,
            bounds,
            [start, goal],
            close_loop=False,
            d_safe=cells_for_metres(bounds, occ_r.shape, standoff_m),
        )
    else:
        route = plan_ground_route(occ_r, bounds, [start, goal], close_loop=False)
    if not route or len(route) < 2:
        raise RuntimeError("route planning failed for this bundle/region")
    route = np.asarray(resample_polyline(route, spacing=0.12 / scale), np.float32)
    lab.add_markers("start", [(start[0], start[1], sample(*start) + 1.0)], color=(0.15, 0.85, 0.25))
    lab.add_markers("goal", [(goal[0], goal[1], sample(*goal) + 1.0)], color=(0.95, 0.80, 0.10))
    lab.add_path(
        "spine", [(w[0], w[1], sample(w[0], w[1]) + 0.5) for w in route], color=(0.30, 0.55, 0.95)
    )
    vv, vt = vehicle_box_mesh()
    lab.add_mesh("agent0", vv, vt, color=(0.90, 0.12, 0.12))
    vpose = VehiclePose(sample, lift=0.25)
    chase = ChaseCamera(back=34.0, height=13.0, look_up=2.5, up=(0.0, 0.0, 1.0))

    o = torch.from_numpy(
        np.array([(route[0][0] - center[0]) * scale, (route[0][1] - center[1]) * scale], np.float32)
    ).unsqueeze(0)
    _S.update(
        lab=lab,
        model=model,
        field=field,
        route=route,
        meta=meta,
        kw=kw,
        nsub=nsub,
        scale=scale,
        center=center,
        occ0=occ0,
        ny=ny,
        nx=nx,
        bounds=bounds,
        goal=goal,
        sample=sample,
        vpose=vpose,
        chase=chase,
        cam=CameraDriver(app, pycvc),
        metrics=MetricsPublisher(app, pycvc),
        o=o,
        v=torch.zeros(1, 2),
        ri=0,
        stall=0,
        done=False,
        corr=0.35 / scale,
        look=8,
        init=len(route),
    )
    lab.node("agent0").setTransform(vpose.update(start[0], start[1], 1.0 / 30.0))
    lab.pump()
    print("grl_snam austin_learned: learned SDF policy driving the A* street corridor.", flush=True)


def _n2w(on, center, scale):
    return np.array([on[0] / scale + center[0], on[1] / scale + center[1]], np.float32)


def _in_building(xw, yw):
    mnx, mny, mxx, mxy = _S["bounds"]
    c = int((xw - mnx) / (mxx - mnx) * (_S["nx"] - 1))
    r = int((yw - mny) / (mxy - mny) * (_S["ny"] - 1))
    return 0 <= r < _S["ny"] and 0 <= c < _S["nx"] and bool(_S["occ0"][r, c])


@torch.no_grad()
def _nav_step() -> NavMetrics:
    center, scale = _S["center"], _S["scale"]
    route, goal = _S["route"], _S["goal"]
    al = be = ga = 0.0
    if _S["done"]:
        w = route[_S["ri"]]
    else:
        sub = route[min(_S["ri"] + _S["look"], len(route) - 1)]
        gt = torch.from_numpy(
            np.array([(sub[0] - center[0]) * scale, (sub[1] - center[1]) * scale], np.float32)
        ).unsqueeze(0)
        alt, bet, gat = _S["model"](sdf_nav.coef_feats(_S["field"], _S["o"], gt))
        al, be, ga = float(alt.mean()), float(bet.mean()), float(gat.mean())
        o2, v2, _ = sdf_nav.sdf_rollout(
            _S["field"], _S["o"], _S["v"], gt, alt, bet, gat, 1, nsub=_S["nsub"], **_S["kw"]
        )
        w = _n2w(o2[0].numpy(), center, scale)
        lo = max(0, _S["ri"] - 2)
        seg = route[lo : min(_S["ri"] + 3 * _S["look"], len(route))]
        j = lo + int(np.argmin(np.linalg.norm(seg - w, axis=1)))
        in_corr = (not _in_building(float(w[0]), float(w[1]))) and np.linalg.norm(
            w - route[j]
        ) < _S["corr"]
        if in_corr:
            _S["o"], _S["v"] = o2, v2
            _S["stall"] = _S["stall"] + 1 if j <= _S["ri"] else 0
            _S["ri"] = max(_S["ri"], j)
        else:
            _S["stall"] += 1
        if (not in_corr) or _S["stall"] > 25:
            _S["ri"] = min(_S["ri"] + 1, len(route) - 1)
            w = route[_S["ri"]].copy()
            _S["o"] = torch.from_numpy(
                np.array([(w[0] - center[0]) * scale, (w[1] - center[1]) * scale], np.float32)
            ).unsqueeze(0)
            _S["v"] = torch.zeros(1, 2)
            _S["stall"] = 0
        mode = "route" if in_corr else "spine"
        if np.linalg.norm(w - np.asarray(goal, np.float32)) < 40.0 or _S["ri"] >= len(route) - 1:
            _S["done"] = True
    phi, _ = _S["field"].sample(_S["o"])
    dg = float(np.linalg.norm(np.asarray(goal, np.float32) - w))
    return NavMetrics(
        step=_S["ri"],
        x=float(w[0]),
        y=float(w[1]),
        goal_x=goal[0],
        goal_y=goal[1],
        goal_dist_m=dg,
        clearance_m=(float(phi[0]) - _S["kw"]["rr"]) / scale,
        alpha=al,
        beta=be,
        gamma=ga,
        mode=locals().get("mode", "route"),
        stall=_S["stall"],
        inside_building=_in_building(float(w[0]), float(w[1])),
        progress=_S["ri"] / max(1, _S["init"] - 1),
    )


def step(dt: float) -> None:
    if not _S:
        setup()
    m = None
    for _ in range(3):
        m = _nav_step()
    _S["lab"].node("agent0").setTransform(_S["vpose"].update(m.x, m.y, dt))
    eye, tgt, up = _S["chase"].update((m.x, m.y, _S["sample"](m.x, m.y)), dt)
    _S["cam"].look(eye, tgt, up)
    _S["metrics"].publish(m, dt)
    _S["lab"].pump()
