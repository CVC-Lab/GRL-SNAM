"""The finale: eight vehicles across real Austin, in 3-D, with a live HUD.

Three things happen per frame, and they are separable on purpose:

* **the world** — terrain heightfield plus the city mesh, loaded once;
* **the vehicles** — each seated on the terrain by :class:`pycvc_gl.vehicle.VehiclePose`,
  which conforms yaw/pitch/roll to the slope rather than sliding a flat box
  across it;
* **the camera** — :func:`~grl_snam.cinema.frame_group` keeps all eight in
  shot, :func:`~grl_snam.cinema.clear_eye` keeps the line of sight out of the
  buildings, and :class:`~grl_snam.cinema.SmoothCamera` turns the result into a
  move instead of a twitch.

The HUD is drawn **in the scene** as VTK 2-D actors, not burned in by the
encoder. That needs the renderer to arrive in Python as a live ``vtkRenderer``
(transfix/libcvc#185); before that bridge existed the only option was ffmpeg
``drawtext``, which cannot occlude, cannot follow a projected 3-D position, and
cannot respond to the camera.

The overhead picture-in-picture is the SAME renderer, re-aimed between shots,
composited by ffmpeg. It cannot be a second :class:`SceneRenderer`:
``SceneGraph::setRenderer`` is single-attachment, so constructing a second one
over the same scene moves every actor to it and silently blanks the first
(measured: mean luma 81.8 -> 0.0). One renderer, two cameras per frame.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from grl_snam.cinema import (
    CameraState,
    SmoothCamera,
    building_height_grid,
    clear_eye,
    clear_shot,
    frame_group,
    shot_angles,
)
from grl_snam.clock import WorldClock
from grl_snam.fog_trace import Trace

PIP_FRAC = 0.22  # overhead inset, as a fraction of frame width
MARK_Z = 350.0  # overhead-marker altitude: above every roof in the bundle

# A wide lens is what lets the camera stand IN the city rather than above it: a
# group is framed from 4.67 radii at 30 degrees and 2.40 at 55.
FOV_DEG = 55.0
# Hard ceiling on the eye. Austin's roofs here are 10-30 m, so a camera at 900 m
# is looking at a map, not a city.
EYE_CEILING_M = 210.0
# Degrees of bearing change per CLIP second. The bearing search is free to
# decide a 120-degree swing is optimal for visibility; taken in one frame that
# reads as a cut, and repeatedly it reads as tumbling.
MAX_BEARING_RATE_DEG_S = 11.0
# Ceiling on the corrective climb, separately from the framing ceiling. Without
# it a single building beside the subject can demand a kilometre of altitude.
MAX_LIFT_M = 85.0


def act_duration_s(bundle) -> float:
    """World seconds in a recorded act, without loading the whole trace set."""
    bundle = Path(bundle)
    spec = json.loads((bundle / "squad.json").read_text())
    return Trace.load(bundle / spec["agents"][0]["key"]).duration_s


def _load(bundle: Path):
    spec = json.loads((bundle / "squad.json").read_text())
    return spec, {a["key"]: Trace.load(bundle / a["key"]) for a in spec["agents"]}


def build_world(lab, bundle_dir: str, *, buildings: bool = True):
    """Terrain + city. Returns the terrain height sampler ``h(x, y)``."""
    from pycvc_gl.scenes import load_geometry_bundle

    return load_geometry_bundle(
        lab,
        bundle_dir,
        terrain_color=(0.20, 0.23, 0.19),
        building_color=(0.62, 0.64, 0.70),
        buildings=buildings,
    )


def add_vehicles(lab, spec, sampler, *, length=14.0, width=6.5, height=4.5):
    """A box per agent, seated on the terrain by VehiclePose.

    Larger than life: a 4.5 m car is two pixels from a camera framing eight of
    them across a kilometre of city, and an invisible protagonist is not a
    protagonist.
    """
    from pycvc_gl.vehicle import VehiclePose

    poses = {}
    for a in spec["agents"]:
        k, col = a["key"], tuple(a["color"])
        v, t = _box(length, width, height)
        lab.add_mesh(f"car_{k}", v, t, col)
        # A 14 m vehicle is a couple of pixels in a shot framing a kilometre of
        # city. A thin beacon rising from each one is findable at any zoom and
        # disappears into the vehicle when the camera comes in close.
        bv, bt = _box(1.0, 1.0, 1.0)  # unit box; scaled per frame to hold its size on screen
        lab.add_mesh(f"beacon_{k}", bv, bt, col)
        # ...and a beacon seen end-on from straight above is a single dot, so
        # the overhead inset gets its own marker: a flat arrowhead, held well
        # above the skyline so no roof can hide it, carrying heading as well as
        # position. Shown only in the overhead shot -- from the main camera it
        # would read as a UFO.
        av, at = _arrow(80.0)
        lab.add_mesh(f"mark_{k}", av, at, col)
        # Where this agent is trying to GET to. In the pursuit act that point
        # moves, and a chase whose quarry is not drawn is just eight vehicles
        # driving oddly -- the same failure as a target marker pinned to the
        # opening waypoint, which is what the last review caught.
        gv, gt = _pyramid()
        lab.add_mesh(f"goal_{k}", gv, gt, col)
        poses[k] = VehiclePose(sampler, lift=1.2)
    return poses


def _goal_offset(i: int, n: int, radius_m: float = 5.0):
    """A small per-agent nudge off the exact goal point.

    Two vehicles share one target in the pursuit act, so their goal markers
    land on the same cell and z-fight into a flickering mess. Spreading them
    around a tiny circle keeps both readable and still reads as one target.
    """
    a = 2.0 * np.pi * i / max(n, 1)
    return radius_m * float(np.cos(a)), radius_m * float(np.sin(a))


def _pyramid():
    """A unit pyramid with its APEX AT THE ORIGIN, pointing down (+Z is up).

    Vehicles stand up; goals hang above and point down at the spot they mark.
    Two colours of vertical pole are indistinguishable at any useful zoom --
    which is what the first render looked like -- so hunter and quarry need
    different silhouettes, not just different positions.
    """
    v = [
        0.0, 0.0, 0.0,
        -0.5, -0.5, 1.0,  0.5, -0.5, 1.0,  0.5, 0.5, 1.0,  -0.5, 0.5, 1.0,
    ]  # fmt: skip
    t = [
        0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4,   # four faces down to the apex
        1, 2, 3, 1, 3, 4,                      # the top
    ]  # fmt: skip
    return v, t


def _arrow(size):
    """A flat arrowhead in the z=0 plane pointing +X, centred on the origin."""
    h, w = size * 0.5, size * 0.34
    v = [
        h, 0.0, 0.0,  -h * 0.55, w, 0.0,  -h * 0.15, 0.0, 0.0,  -h * 0.55, -w, 0.0,
    ]  # fmt: skip
    return v, [0, 1, 2, 0, 2, 3]


def _yaw_at(x, y, z, heading):
    c, s = float(np.cos(heading)), float(np.sin(heading))
    return [c, -s, 0.0, float(x), s, c, 0.0, float(y), 0.0, 0.0, 1.0, float(z), 0.0, 0.0, 0.0, 1.0]


def _box(lx, ly, lz):
    hx, hy = lx * 0.5, ly * 0.5
    v = [
        -hx, -hy, 0.0,  hx, -hy, 0.0,  hx,  hy, 0.0,  -hx,  hy, 0.0,
        -hx, -hy, lz,   hx, -hy, lz,   hx,  hy, lz,   -hx,  hy, lz,
    ]  # fmt: skip
    t = [
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
        1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0,
    ]  # fmt: skip
    return v, t


def make_hud(ren, spec, width, height):
    """In-scene 2-D text: one line per vehicle plus a title."""
    from vtkmodules.vtkRenderingCore import vtkTextActor

    def _panel(prop, size, color):
        prop.SetFontSize(size)
        prop.SetColor(*color)
        # A city rendered in pale concrete is the worst possible backdrop for
        # white text. VTK can back each line itself, which beats compositing a
        # separate panel actor and keeps the HUD one object per line.
        prop.SetBackgroundColor(0.02, 0.03, 0.05)
        prop.SetBackgroundOpacity(0.62)
        prop.SetFontFamilyToCourier()

    # Every size here is a FRACTION of frame height. vtkTextActor font sizes are
    # absolute pixels, so a HUD tuned at 720p quietly shrinks to three-quarters
    # of its intended size when the same code renders at 900p — legible in the
    # test render, small in the deliverable. The ratios below are the 720p
    # design (30 px title, 18 px rows, 26 px leading) expressed against height.
    title_px = max(12, round(height * 0.0417))
    row_px = max(9, round(height * 0.0250))
    lead_px = max(12, round(height * 0.0361))
    left_px = round(height * 0.0333)

    actors = {}
    title = vtkTextActor()
    _panel(title.GetTextProperty(), title_px, (0.92, 0.94, 1.0))
    title.SetPosition(left_px, height - round(height * 0.0667))
    ren.AddViewProp(title)
    actors["_title"] = title

    for i, a in enumerate(spec["agents"]):
        tx = vtkTextActor()
        _panel(tx.GetTextProperty(), row_px, a["color"])
        tx.SetPosition(left_px, height - round(height * 0.1333) - i * lead_px)
        ren.AddViewProp(tx)
        actors[a["key"]] = tx
    return actors


def capture_finale(
    bundle: str | Path,
    bundle_dir: str,
    out: str | Path,
    *,
    fps: int = 24,
    speed: float = 6.0,
    size: tuple[int, int] = (1600, 900),
    occ: np.ndarray | None = None,
    world_bounds=None,
    elevation_deg: float = 26.0,
    frame_goals: bool = False,
    camera_in: CameraState | None = None,
    u_range: tuple[float, float] = (0.0, 1.0),
    progress=None,
) -> tuple[Path, CameraState]:
    import pycvc_gl
    from pycvc_gl.lab import Lab

    bundle = Path(bundle)
    spec, traces = _load(bundle)
    keys = [a["key"] for a in spec["agents"]]
    out = Path(out)

    # ── main 3-D view ───────────────────────────────────────────────────────
    lab = Lab()
    sampler = build_world(lab, bundle_dir)
    poses = add_vehicles(lab, spec, sampler)
    scene = lab._scene
    _hide_chrome(scene, lab)

    main = pycvc_gl.SceneRenderer(scene, size[0], size[1], True)
    ren = main.renderer()
    ren.GradientBackgroundOn()
    ren.SetBackground(0.015, 0.02, 0.035)
    ren.SetBackground2(0.10, 0.14, 0.22)
    _add_sun(ren)
    hud = make_hud(ren, spec, size[0], size[1])

    # The overhead inset is rendered by the SAME renderer at full size and
    # scaled down at compose time. A second SceneRenderer would detach the
    # scene from this one.
    #
    # It is a FIXED shot of the whole map, not a second tracking shot: the
    # question it answers is "where are they on the map", and a minimap that
    # zooms with the group answers that only if you already know the answer.
    # Fixed also means the inset never lags -- the smoothed focal point the
    # main camera uses would drag it off-centre during fast convergence.
    pip_w = int(size[0] * PIP_FRAC) // 2 * 2

    # VTK's view angle governs the viewport HEIGHT, so fit the map's north-south
    # span to it and let the wider 16:9 frame overhang east and west. That
    # overhang is dead sky, so crop it back off: a square map wants a square
    # minimap, and letterboxing it would spend inset pixels on nothing.
    if world_bounds is not None:
        mnx, mny, mxx, mxy = (float(b) for b in world_bounds)
    else:
        mnx, mny, mxx, mxy = -600.0, -600.0, 600.0, 600.0
    alt = (mxy - mny) * 0.5 / np.tan(np.radians(45.0 * 0.5)) * 1.04
    map_focal = np.array([(mnx + mxx) * 0.5, (mny + mxy) * 0.5, 0.0])
    map_eye = map_focal + np.array([0.0, 0.0, alt])
    seen_w = (mxy - mny) * 1.04 * size[0] / size[1]  # world metres across the frame
    crop_w = min(size[0], int(size[0] * (mxx - mnx) / seen_w) // 2 * 2)
    pip_h = int(pip_w * size[1] / crop_w) // 2 * 2  # inset takes the MAP's aspect

    height_grid = None
    if occ is not None and world_bounds is not None:
        from grl_snam.tools.sdf import _gltf_mesh

        verts, _t = _gltf_mesh(str(Path(bundle_dir) / "buildings.glb"))
        height_grid = building_height_grid(verts, world_bounds, occ.shape[0], occ=occ)

    any_tr = traces[keys[0]]
    clock = WorldClock(fixed_dt=any_tr.fixed_dt, mode="replay")
    # Long taus: the framing target jumps every frame as the bounding sphere
    # shrinks, and that jitter is what makes the move feel nervous.
    cam = SmoothCamera(eye_tau=3.2, focal_tau=1.6)
    if camera_in is not None and camera_in.eye is not None:
        # Pick up exactly where the previous act put the camera, so the two are
        # one move rather than two shots. The damper does the rest: whatever the
        # new act wants to frame, it is approached over ~3 s instead of cut to.
        cam.prime(camera_in.eye, camera_in.focal)
    n_frames = max(1, int(any_tr.duration_s / max(speed, 1e-9) * fps))
    dt_frame = 1.0 / fps

    seats: dict[str, tuple[float, float, float]] = {}
    u0, u1 = (float(u_range[0]), float(u_range[1]))
    az_prev = [shot_angles(u0, low_deg=elevation_deg)[1]]
    az_pref = [az_prev[0] if camera_in is None else float(camera_in.azimuth_deg)]
    frames = out.parent / f"_finale_{out.stem}"
    shutil.rmtree(frames, ignore_errors=True)
    (frames / "main").mkdir(parents=True, exist_ok=True)
    (frames / "pip").mkdir(parents=True, exist_ok=True)

    try:
        for f in range(n_frames):
            t = f / fps * speed
            clock.seek_time(t)
            pts = []
            goals_now: dict[str, tuple[float, float, float]] = {}
            for k in keys:
                pose = traces[k].pose_at(t)
                m = poses[k].update(pose.x, pose.y, dt_frame * speed)
                scene.getGraphics(f"car_{k}").setTransform(list(m))
                scene.getGraphics(f"mark_{k}").setTransform(
                    _yaw_at(pose.x, pose.y, MARK_Z, pose.heading_rad)
                )
                z = float(sampler(pose.x, pose.y))
                pts.append((pose.x, pose.y, z))
                seats[k] = (pose.x, pose.y, z)
                tr = traces[k]
                i, _ = tr._tick_index(t)
                # Distance to the GOAL, computed from the recorded goal
                # position -- not `goal_dist_m`, which is the distance to the
                # route carrot and therefore sits at one lookahead (~14 m) for
                # the entire drive. A HUD that reads "goal 14 m" from the far
                # side of the city is worse than no HUD.
                g = tr.goal_at(t)
                dist = (
                    float(np.hypot(pose.x - g[0], pose.y - g[1]))
                    if g is not None
                    else float(tr.rows["goal_dist_m"][i])
                )
                hud[k].SetInput(f"{k}  {tr.rows['speed_mps'][i]:5.1f} m/s   goal {dist:6.0f} m")
                if g is not None:
                    ox, oy = _goal_offset(keys.index(k), len(keys))
                    goals_now[k] = (g[0] + ox, g[1] + oy, float(sampler(g[0] + ox, g[1] + oy)))
            hud["_title"].SetInput(f"AUSTIN  ·  {len(keys)} vehicles  ·  t = {clock.t():6.1f} s")

            # Size and seat the goal posts, and (for a chase) frame them with
            # the vehicles so the closing gap is visible rather than implied.
            for k in keys:
                gp = goals_now.get(k)
                if gp is None:
                    continue
                scene.getGraphics(f"goal_{k}").setVisible(True)
                if frame_goals:
                    pts.append(gp)

            _show(scene, keys, "mark_", False)
            u = u0 + (u1 - u0) * (f / max(n_frames - 1, 1))
            elev, azim = shot_angles(u, low_deg=elevation_deg)
            if height_grid is not None:
                # Carry the bearing forward and add only the schedule's DRIFT to
                # it. Preferring the scheduled bearing outright would snap the
                # camera back the instant an obstruction cleared; this way a
                # detour around a stadium is kept and drifted onward from.
                az_pref[0] += azim - az_prev[0]
                # The beacon height the renderer will use for this frame. dist
                # depends only on the group's radius and the lens, not on the
                # bearing, so it is the same for every candidate and can be
                # computed once.
                _p = np.asarray(pts, np.float64).reshape(-1, 3)
                _rad = max(float(np.linalg.norm(_p - _p.mean(axis=0), axis=1).max()), 60.0)
                _dist = _rad / (np.tan(np.radians(FOV_DEG * 0.5)) * 0.80)
                _rise = 0.8 * float(np.clip(_dist * 0.065, 6.0, 95.0))
                eye, focal, _r, used = clear_shot(
                    pts,
                    height_grid,
                    world_bounds,
                    elevation_deg=elev,
                    azimuth_deg=az_pref[0],
                    fill=0.80,
                    margin_m=25.0,
                    fov_deg=FOV_DEG,
                    max_height_m=EYE_CEILING_M,
                    max_lift_m=MAX_LIFT_M,
                    subject_rise_m=_rise,
                )
                # Walk toward the chosen bearing at a bounded rate instead of
                # snapping to it. The search re-runs every frame, so a bearing
                # it cannot reach this frame it simply chooses again next frame
                # -- the camera pans there over about a second rather than
                # cutting there in one.
                step_cap = MAX_BEARING_RATE_DEG_S / fps
                delta = (used - az_pref[0] + 180.0) % 360.0 - 180.0
                step = float(np.clip(delta, -step_cap, step_cap))
                az_pref[0] += step
                if abs(step) < abs(delta):
                    eye, focal, _r = frame_group(
                        pts,
                        elevation_deg=elev,
                        azimuth_deg=az_pref[0],
                        fill=0.80,
                        fov_deg=FOV_DEG,
                        max_height_m=EYE_CEILING_M,
                    )
                    eye = clear_eye(
                        eye,
                        focal,
                        height_grid,
                        world_bounds,
                        margin_m=25.0,
                        tail_m=30.0,
                        max_lift_m=MAX_LIFT_M,
                    )
            else:
                eye, focal, _r = frame_group(
                    pts,
                    elevation_deg=elev,
                    azimuth_deg=azim,
                    fill=0.80,
                    fov_deg=FOV_DEG,
                    max_height_m=EYE_CEILING_M,
                )
            az_prev[0] = azim
            eye, focal = cam.update(eye, focal, dt_frame)

            # Size the beacons off the FINAL camera distance, so they stay a
            # constant fraction of the frame. A fixed 90 m pole is a useful
            # pointer when the shot spans a kilometre and an absurd tower when
            # the group has converged and the camera is 80 m away.
            dist = float(np.linalg.norm(eye - focal))
            bw = float(np.clip(dist * 0.006, 0.8, 9.0))
            # Capped well under the 130 m the first render used: against a city
            # whose roofs are 10-30 m, that read as eight columns rather than
            # eight markers.
            bh = float(np.clip(dist * 0.065, 6.0, 95.0))
            for k in keys:
                x, y, z = seats[k]
                scene.getGraphics(f"beacon_{k}").setTransform(
                    [bw, 0.0, 0.0, x, 0.0, bw, 0.0, y, 0.0, 0.0, bh, z, 0.0, 0.0, 0.0, 1.0]
                )
                gp = goals_now.get(k)
                g_node = scene.getGraphics(f"goal_{k}")
                if gp is None:
                    g_node.setVisible(False)
                    continue
                # Thinner and shorter than a vehicle beacon on purpose: a goal
                # is a place, not a protagonist, and it must not be mistaken
                # for a ninth vehicle.
                gx, gy, gz = gp
                # Hovers with its point on the target, clear of the rooftops so
                # it is not swallowed by whatever the target is driving past.
                gw = max(bw * 2.2, 5.0)
                gh = max(bh * 0.40, 9.0)
                g_node.setTransform(
                    [
                        gw,
                        0.0,
                        0.0,
                        gx,
                        0.0,
                        gw,
                        0.0,
                        gy,
                        0.0,
                        0.0,
                        gh,
                        gz + gh * 0.55,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                )
            main.setCamera(*eye, *focal, 0.0, 0.0, 1.0, FOV_DEG, 1.0, 20000.0)
            main.writePNG(str(frames / "main" / f"f_{f:05d}.png"))

            # Re-aim the SAME renderer straight down for the inset, then put it
            # back. The HUD actors are 2-D and would show up in the inset too,
            # so hide them for this shot.
            for actor in hud.values():
                actor.SetVisibility(False)
            _show(scene, keys, "mark_", True)
            _show(scene, keys, "beacon_", False)
            main.setCamera(*map_eye, *map_focal, 0.0, 1.0, 0.0, 45.0, 1.0, 30000.0)
            main.writePNG(str(frames / "pip" / f"f_{f:05d}.png"))
            for actor in hud.values():
                actor.SetVisibility(True)
            _show(scene, keys, "beacon_", True)

            if progress and f % 25 == 0:
                progress(f, n_frames)
    finally:
        main.close()

    _compose(frames, out, fps=fps, size=size, pip=(pip_w, pip_h), crop_w=crop_w)
    shutil.rmtree(frames, ignore_errors=True)
    return out, cam.state(az_pref[0])


def _show(scene, keys, prefix, visible):
    for k in keys:
        g = scene.getGraphics(f"{prefix}{k}")
        if g is not None:
            g.setVisible(visible)


def _hide_chrome(scene, lab):
    """Bounding boxes and labels off, everywhere.

    Uses getGraphics (a GraphicsNode) rather than geometry_node: the city
    arrives as a VTK prop, not a cvcGL geometry, so geometry_node() returns
    None for it -- and the flags live on GraphicsNode anyway.
    """
    for n in list(scene.graphics_names()):
        g = scene.getGraphics(n)
        if g is None:
            continue
        g.setShowBBox(False)
        g.setShowExtentLabels(False)
        g.setShowLabel(False)
    root = scene.getGraphicsRoot()
    if root is not None:
        root.setShowBBox(False)
        root.setShowExtentLabels(False)
        root.setShowLabel(False)
    scene.setGridVisible(False)
    lab.set_axis_visible(False)


def _add_sun(ren):
    from vtkmodules.vtkRenderingCore import vtkLight

    key = vtkLight()
    key.SetLightTypeToSceneLight()
    key.SetPosition(-1200.0, -900.0, 1400.0)
    key.SetFocalPoint(0.0, 0.0, 0.0)
    key.SetIntensity(1.05)
    key.SetColor(1.0, 0.96, 0.90)
    ren.AddLight(key)
    fill = vtkLight()
    fill.SetLightTypeToSceneLight()
    fill.SetPosition(900.0, 1100.0, 600.0)
    fill.SetIntensity(0.35)
    fill.SetColor(0.75, 0.83, 1.0)
    ren.AddLight(fill)


def _compose(frames: Path, out: Path, *, fps: int, size, pip, crop_w: int | None = None) -> None:
    """Overlay the overhead inset on the main view, bottom-right."""
    pw, ph = pip
    x, y = size[0] - pw - 26, size[1] - ph - 26
    # Crop the overhead frame to the map before scaling: the renderer sees more
    # east-west than the map covers, and that overhang is empty sky.
    crop = f"crop={crop_w}:{size[1]}:{(size[0] - crop_w) // 2}:0," if crop_w else ""
    filt = (
        f"[1:v]{crop}scale={pw}:{ph},"
        f"drawbox=x=0:y=0:w={pw}:h={ph}:color=0x8899bb@0.9:t=2[pipv];"
        f"[0:v][pipv]overlay={x}:{y}"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(frames / "main" / "f_%05d.png"),
            "-framerate", str(fps), "-i", str(frames / "pip" / "f_%05d.png"),
            "-filter_complex", filt,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
            str(out),
        ],
        check=True,
    )  # fmt: skip
