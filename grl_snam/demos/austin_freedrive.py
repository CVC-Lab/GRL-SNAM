"""Live demo: the learned SDF navigator finding its OWN way A->B across real Austin,
TX in a running VolRover3 window — no precomputed route. The trained policy reacts to
the buildings through the SDF (a moving carrot toward the goal + the wall barrier),
with a bug-style wall-follow escape for potential-field dead-ends. Live metrics (the
network's coefficients, clearance, mode) publish to the state tree + console every step.

Runs out of the box off cvcpkg — no training run: it uses the published pretrained
coefficients (`grl-snam-weights` → share/grl-snam-weights/coef_sdf.pt) and the published
Austin geometry (`scene-austin-south` → share/cvc-scenes/austin_south), both discovered
from the active prefix. So: `cvcpkg install grl-snam-weights scene-austin-south` (plus
the demo host), then run inside VolRover3 (Jobs tab -> Load Script, or `grl-snam demo
austin-freedrive`).

Overrides (all optional): GRL_SNAM_CHECKPOINT — a .pt to use instead of the published
weights, e.g. from your own `grl-snam build-sdf <bundle>` + `grl-snam train
<bundle>/nav_sdf.npz -o checkpoints/coef_sdf.pt`; GRL_SNAM_SCENE_BUNDLE (bundle dir);
GRL_SNAM_SDF (prebuilt nav_sdf.npz; else built from occupancy); GRL_SNAM_START /
GRL_SNAM_GOAL ("x,y"; defaults are a navigable pair on austin_south).
"""

from __future__ import annotations

import os

import numpy as np
import torch

import sdf_nav
from grl_snam.demos._common import (
    SimPacer,
    current_host,
    default_nav_weights_pt,
    default_scene_bundle,
    vehicle_box_mesh,
)
from grl_snam.nav import SdfNavigator

# Sim playback rate relative to the world clock: 1.0 == real time (the drive takes
# as many wall seconds as it does sim seconds), matching the native cvc::nav demos.
# Raise it for a faster-to-watch pace without ever re-coupling to the frame rate.
SIM_SPEED = 1.0

_S: dict = {}


def _xy(env, default):
    v = os.environ.get(env)
    if not v:
        return default
    a, b = v.split(",")
    return (float(a), float(b))


def setup() -> None:
    host = current_host()
    from pycvc_gl.camera import NativeChaseCamera as ChaseCamera
    from pycvc_gl.scenes import building_occupancy, load_geometry_bundle, terrain_grid
    from pycvc_gl.vehicle import VehiclePose

    bundle = default_scene_bundle()
    ckpt = default_nav_weights_pt()
    torch.set_num_threads(2)
    ck = torch.load(ckpt, map_location="cpu")
    meta = ck["meta"]
    model = sdf_nav.CoefMLP()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    lab = host.make_lab()
    lab.set_axis_visible(False)
    sample = load_geometry_bundle(lab, bundle)
    bounds = terrain_grid(os.path.join(bundle, "terrain.json"))[1]
    sdf_npz = os.environ.get("GRL_SNAM_SDF", "")
    if sdf_npz and os.path.exists(sdf_npz):
        d = np.load(sdf_npz)
        phi, nxg, nyg = d["phi"], d["normal_x"], d["normal_y"]
    else:
        print("grl_snam austin_freedrive: rasterizing occupancy + building SDF...", flush=True)
        occ = building_occupancy(
            os.path.join(bundle, "buildings.glb"), bounds, 512, 512, inflate_m=0.0
        )
        phi, nxg, nyg = sdf_nav.build_sdf(occ, bounds, float(meta["scale"]))
    field = sdf_nav.SDFField(
        phi, nxg, nyg, bounds, meta["center"], float(meta["scale"]), device="cpu"
    )

    start = _xy("GRL_SNAM_START", (-361.0, 114.0))
    goal = _xy("GRL_SNAM_GOAL", (185.0, 50.0))
    nav = SdfNavigator(field, model, meta, reach_tol=0.5).start(start, goal)

    lab.add_markers("start", [(start[0], start[1], sample(*start) + 1.0)], color=(0.15, 0.85, 0.25))
    lab.add_markers("goal", [(goal[0], goal[1], sample(*goal) + 1.0)], color=(0.95, 0.80, 0.10))
    vv, vt = vehicle_box_mesh()
    lab.add_mesh("agent0", vv, vt, color=(0.90, 0.12, 0.12))
    vpose = VehiclePose(sample, lift=0.25)
    chase = ChaseCamera(back=34.0, height=13.0, look_up=2.5, up=(0.0, 0.0, 1.0))
    cam = host.camera()
    metrics = host.metrics()

    lab.node("agent0").setTransform(vpose.update(start[0], start[1], 1.0 / 30.0))
    lab.pump()
    print(
        "grl_snam austin_freedrive: the learned SDF policy is finding its own way "
        "%s -> %s across Austin (no route)." % (start, goal),
        flush=True,
    )
    _S.update(
        lab=lab,
        nav=nav,
        sample=sample,
        vpose=vpose,
        chase=chase,
        cam=cam,
        metrics=metrics,
        # Fixed-timestep sim pacing off the world clock. One nav.step() advances
        # meta["dt"] seconds of sim time, so ticking it at that wall rate is
        # real time — not once per drawn frame (which ran ~10x fast on a quick
        # display and varied with the frame rate).
        pacer=SimPacer(),
        sim_dt=float(nav.dt),
        speed=SIM_SPEED,
        last_m=None,
    )


def step(dt: float) -> None:
    if not _S:
        setup()
    nav = _S["nav"]
    # Run the navigator a whole number of fixed sim ticks for the real time this
    # frame took (0 on a fast frame, more on a slow one), not once per frame.
    m = _S["last_m"]
    for _ in range(_S["pacer"].ticks(dt, _S["sim_dt"], _S["speed"])):
        m = nav.step()
    if m is None:
        m = nav.step()  # first frame: guarantee a pose to draw
    _S["last_m"] = m
    x, y = m.x, m.y
    _S["lab"].node("agent0").setTransform(_S["vpose"].update(x, y, dt))
    eye, tgt, up = _S["chase"].update((x, y, _S["sample"](x, y)), dt)
    _S["cam"].look(eye, tgt, up)
    _S["metrics"].publish(m, dt)
    _S["lab"].pump()
