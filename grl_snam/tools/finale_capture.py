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
    contain,
    frame_group,
    shot_angles,
)
from grl_snam.clock import WorldClock
from grl_snam.fog_trace import Trace
from grl_snam.tools import wall_vis

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
#: Degrees off the group's own heading to sit. Dead astern of the objective is a
#: head-on shot with no depth; a shoulder gives a three-quarter view where the
#: street grid still reads.
SHOULDER_DEG = 34.0
# Ceiling on the corrective climb, separately from the framing ceiling. Without
# it a single building beside the subject can demand a kilometre of altitude.
MAX_LIFT_M = 85.0


def leader_arrival_s(
    bundle, *, key: str | None = None, tol_m: float = 4.0, hold_s: float = 4.5
) -> float | None:
    """World time at which the lead agent reaches its goal, plus a short hold.

    A convoy has nothing left to show once the leader parks: the followers
    close on a stationary target, jostle for the standoff beside it and mill
    about. Measured on the recorded convoy, the leader arrives at 67.3 s of a
    192 s run, so 35% of the clip was that milling.
    """
    bundle = Path(bundle)
    spec = json.loads((bundle / "squad.json").read_text())
    k = key or spec["agents"][0]["key"]
    tr = Trace.load(bundle / k)
    r = tr.rows
    d = np.hypot(r["x"] - r["goal_x"], r["y"] - r["goal_y"])
    hit = np.nonzero(d < tol_m)[0]
    if not hit.size:
        return None
    return float(hit[0] * tr.fixed_dt + hold_s)


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
    fog: bool = True,
    end_world_s: float | None = None,
    on_frame=None,
    camera_in: CameraState | None = None,
    u_range: tuple[float, float] = (0.0, 1.0),
    progress=None,
) -> tuple[Path, CameraState]:
    import pycvc_gl
    from pycvc_gl.lab import Lab

    bundle = Path(bundle)
    spec, traces = _load(bundle)
    keys = [a["key"] for a in spec["agents"]]
    # Which navigator produced this trace. Read from the manifest rather than
    # inferred: the two look identical in a still frame and differ enormously in
    # what they can do -- on real Austin the route spine arrives and the
    # reactive controller alone never does.
    nav_label = traces[keys[0]].manifest.get("nav", "route+sdf")
    out = Path(out)

    # ── main 3-D view ───────────────────────────────────────────────────────
    lab = Lab()
    sampler = build_world(lab, bundle_dir)
    poses = add_vehicles(lab, spec, sampler)
    scene = lab._scene
    fog_node = fog_live = wall = None
    if fog and traces[keys[0]].has_fov():
        tr0 = traces[keys[0]]
        fog_node, fog_live = add_fog_decal(lab, scene, sampler, tr0.bounds, tr0.shape)
        # Prefer the true mesh cast when the post-pass has been run; the
        # projected raster mask is the fallback.
        faces = add_wall_faces(scene, wall_vis.load(bundle))
        wall = None if faces is not None else add_wall_mask(scene, tr0.bounds, tr0.shape)
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
    cam = SmoothCamera(eye_tau=3.2, focal_tau=1.6, widen_tau=0.35)
    if camera_in is not None and camera_in.eye is not None:
        # Pick up exactly where the previous act put the camera, so the two are
        # one move rather than two shots. The damper does the rest: whatever the
        # new act wants to frame, it is approached over ~3 s instead of cut to.
        cam.prime(camera_in.eye, camera_in.focal)
    span_s = (
        any_tr.duration_s if end_world_s is None else min(any_tr.duration_s, float(end_world_s))
    )
    n_frames = max(1, int(span_s / max(speed, 1e-9) * fps))
    dt_frame = 1.0 / fps

    # Pre-drape every driven path once. Route spines are draped per frame from
    # the current polyline (they are 2-5 points, so it is cheap).
    track_pts = {}
    for i, k in enumerate(keys):
        r = traces[k].rows
        track_pts[k] = _drape(sampler, np.stack([r["x"], r["y"]], 1), 3.0 + i * 0.45)
    node_track = add_line_node(scene, lab._app, "tracks", 4.0)
    node_spine = add_line_node(scene, lab._app, "spines", 3.0)
    agent_col = {a["key"]: tuple(a["color"]) for a in spec["agents"]}

    seats: dict[str, tuple[float, float, float]] = {}
    u0, u1 = (float(u_range[0]), float(u_range[1]))
    az_prev = [shot_angles(u0, low_deg=elevation_deg)[1]]
    az_pref = [az_prev[0] if camera_in is None else float(camera_in.azimuth_deg)]
    # Raw RGB straight into two ffmpeg processes. writePNG was 95% of the
    # frame cost -- measured 333 ms for the two 1600x900 encodes against 33 ms
    # of actual GL render -- and it also left thousands of files on disk per
    # clip. frameRGB() is the same pixels without the PNG round trip; the
    # precedent is fog_capture.open_encoder, which already does exactly this.
    frames = out.parent / f"_finale_{out.stem}"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True, exist_ok=True)
    raw_main = frames / "main.mp4"
    raw_pip = frames / "pip.mp4"
    enc_main = _open_raw_encoder(raw_main, fps=fps, size=size)
    enc_pip = _open_raw_encoder(raw_pip, fps=fps, size=size)

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
            if fog_live is not None:
                i_tick_wall, _fw = traces[keys[0]]._tick_index(t)
                tier = fog_tiers(traces, keys, t, traces[keys[0]].shape)
                np.take(FOG_LUT, tier, axis=0, out=fog_live)
                fog_node.texture_modified()
                if faces is not None:
                    order, cur, fs = faces["order"], faces["cursor"], faces["fs"]
                    j = cur[0]
                    while j < len(order) and fs[order[j]] <= i_tick_wall:
                        j += 1
                    if j != cur[0]:
                        faces["rgb"][order[cur[0] : j]] = WALL_LUT[2, :3]
                        cur[0] = j
                        faces["arr"].Modified()
                        faces["pd"].Modified()
                elif wall is not None:
                    tex, wbuf, wimg, warr, pad = wall
                    # A ray stops at the first cell it hits, so only the near
                    # face rim of a block is ever marked seen and a big building
                    # would stay dark with a lit edge. Widen by one cell for the
                    # WALLS only; the ground decal keeps the exact measured set.
                    wide = tier
                    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                        wide = np.maximum(wide, np.roll(tier, sh, axis=ax))
                    np.take(WALL_LUT, wide, axis=0, out=wbuf[pad:-pad, pad:-pad])
                    warr.Modified()
                    wimg.Modified()
                    tex.Modified()

            hud["_title"].SetInput(
                f"AUSTIN · {len(keys)} vehicles · nav: {nav_label} · t = {clock.t():6.1f} s"
            )

            # Size and seat the goal posts, and (for a chase) frame them with
            # the vehicles so the closing gap is visible rather than implied.
            for k in keys:
                gp = goals_now.get(k)
                if gp is None:
                    continue
                scene.getGraphics(f"goal_{k}").setVisible(True)
                if frame_goals:
                    pts.append(gp)

            # Driven path so far, per-agent hue. Strided: 4 ticks is ~2.4 m at
            # 10 m/s, far below a pixel at this camera distance.
            i_tick, _f = traces[keys[0]]._tick_index(t)
            segs, cols = [], []
            for k in keys:
                # NOT `pts` -- that name holds the vehicle positions the camera
                # frames itself on, and rebinding it here silently pointed the
                # whole shot at the last agent's TRAIL instead of at the squad.
                trail = track_pts[k][: max(2, i_tick + 1) : 4]
                if len(trail) >= 2:
                    segs.append(trail)
                    cols.append(agent_col[k])
            set_lines(node_track, lab._app, segs, cols)

            # The BELIEF-space route spine. Inset only, deliberately: the
            # recorded route is string-pulled in belief space, so early in a run
            # it is a single straight segment over a kilometre long -- honest
            # (the planner has not looked yet and believes it is clear) and
            # correct in the 2-D map, but in the 3-D shot it is a line through
            # the skyline that reads as a rendering bug.
            sp_segs, sp_cols = [], []
            for i, k in enumerate(keys):
                r = traces[k].route_at(t) if hasattr(traces[k], "route_at") else None
                if r is None:
                    ki = traces[k].snapshot_index_at(t)
                    r = traces[k].routes[ki] if 0 <= ki < len(traces[k].routes) else None
                if r is None or len(r) < 2:
                    continue
                sp_segs.append(_drape(sampler, _resample(r), 6.0 + i * 0.45))
                c = agent_col[k]
                sp_cols.append(tuple(0.35 + 0.25 * v for v in c))  # dim
            set_lines(node_spine, lab._app, sp_segs, sp_cols)
            node_spine.setVisible(False)  # main shot: off

            _show(scene, keys, "mark_", False)
            u = u0 + (u1 - u0) * (f / max(n_frames - 1, 1))
            elev, azim = shot_angles(u, low_deg=elevation_deg)
            if height_grid is not None:
                # Carry the bearing forward and add only the schedule's DRIFT to
                # it. Preferring the scheduled bearing outright would snap the
                # camera back the instant an obstruction cleared; this way a
                # detour around a stadium is kept and drifted onward from.
                # Where the camera WANTS to be: on the objective's side, off
                # the shoulder, plus whatever slow drift the schedule has
                # accumulated. Falls back to pure drift when the objective is
                # too close to give a bearing.
                gb = goal_bearing_deg(traces, keys, t, pts)
                drift = azim - shot_angles(u0, low_deg=elevation_deg)[1]
                want = az_pref[0] + (azim - az_prev[0]) if gb is None else gb + SHOULDER_DEG + drift
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
                    azimuth_deg=want,
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
            # Guarantee containment on the camera that actually renders. The
            # framing distance above was computed for the UNDAMPED target; once
            # the damper has had its say the group can easily no longer fit.
            eye = contain(eye, focal, pts, fov_deg=FOV_DEG)
            if height_grid is not None:
                eye = clear_eye(
                    eye, focal, height_grid, world_bounds,
                    margin_m=25.0, tail_m=30.0, max_lift_m=MAX_LIFT_M,
                )  # fmt: skip
            if on_frame is not None:
                on_frame(f, eye, focal, pts)

            # Size the beacons off the FINAL camera distance, so they stay a
            # constant fraction of the frame. A fixed 90 m pole is a useful
            # pointer when the shot spans a kilometre and an absurd tower when
            # the group has converged and the camera is 80 m away.
            dist = float(np.linalg.norm(eye - focal))
            # Sized for legibility at the distance the shot actually sits at.
            # Framing eight agents spread across a kilometre puts the camera a
            # measured 706 m out on average, where a 14 m vehicle is ~17 px and
            # a thin beacon is a couple of pixels wide -- in frame, and still
            # invisible. Containment guarantees they are IN shot; this is what
            # makes them findable once they are.
            bw = float(np.clip(dist * 0.011, 1.6, 17.0))
            # Capped well under the 130 m the first render used: against a city
            # whose roofs are 10-30 m, that read as eight columns rather than
            # eight markers.
            bh = float(np.clip(dist * 0.078, 9.0, 125.0))
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
            ren.GetActiveCamera().ParallelProjectionOff()  # main shot stays perspective
            main.setCamera(*eye, *focal, 0.0, 0.0, 1.0, FOV_DEG, 1.0, 20000.0)
            enc_main.stdin.write(main.frameRGB())

            # Re-aim the SAME renderer straight down for the inset, then put it
            # back. The HUD actors are 2-D and would show up in the inset too,
            # so hide them for this shot.
            for actor in hud.values():
                actor.SetVisibility(False)
            _show(scene, keys, "mark_", True)
            _show(scene, keys, "beacon_", False)
            node_spine.setVisible(True)
            # ORTHOGRAPHIC for the inset. Under perspective, the heading
            # arrows -- which are held above the skyline so no roof can hide
            # them -- are nearer the lens than the ground, so each one projects
            # outward from the image centre by an amount that grows with its
            # distance from centre. The result is an arrow visibly offset from
            # the vehicle it belongs to, which reads as a bug because it is one.
            # A parallel projection has no parallax: altitude cannot displace
            # anything, so the arrow lands exactly on its vehicle. It also makes
            # the inset a true map rather than a perspective view of one, which
            # is what a minimap should be.
            main.setCamera(*map_eye, *map_focal, 0.0, 1.0, 0.0, 45.0, 1.0, 30000.0)
            vcam = ren.GetActiveCamera()
            vcam.ParallelProjectionOn()
            vcam.SetParallelScale((mxy - mny) * 0.5 * 1.04)
            enc_pip.stdin.write(main.frameRGB())
            for actor in hud.values():
                actor.SetVisibility(True)
            _show(scene, keys, "beacon_", True)
            node_spine.setVisible(False)

            if progress and f % 25 == 0:
                progress(f, n_frames)
    finally:
        for e in (enc_main, enc_pip):
            try:
                e.stdin.close()
                e.wait(timeout=600)
            except Exception:
                e.kill()
        main.close()

    _compose(raw_main, raw_pip, out, fps=fps, size=size, pip=(pip_w, pip_h), crop_w=crop_w)
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


def add_fog_decal(lab, scene, sampler, bounds, shape, *, n: int = 129, lift_m: float = 1.5):
    """A terrain-draped surface carrying the recorded field of view as a texture.

    The 2-D clips show three-tier fog -- never seen, remembered, visible now --
    and the 3-D ones showed none of it. This paints the SAME measured grid onto
    the ground.

    A texture, not geometry, because at finale playback (0.25 world s per frame
    against a 0.24 s sensor cadence) the visible set changes EVERY frame, so the
    2-D renderer's "re-mesh only when it changed" trick buys nothing here.
    Re-meshing the lit cells costs ~172 ms/frame; re-writing the texture through
    libcvc's zero-copy path costs a few. The decal's own resolution is
    independent of the fog's -- all the detail lives in the texels.

    Lab.add_mesh cannot carry UVs, so the geometry is built directly.
    """
    import pycvc

    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    gx = np.linspace(mnx, mxx, n)
    gy = np.linspace(mny, mxy, n)
    X, Y = np.meshgrid(gx, gy)
    # The bundle's terrain sampler is scalar-only, so this is a loop -- but it
    # runs ONCE at setup (129^2 = 16,641 lookups), never per frame.
    Z = (
        np.array(
            [sampler(float(x), float(y)) for x, y in zip(X.ravel(), Y.ravel())], np.float64
        ).reshape(n, n)
        + lift_m
    )

    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).ravel().tolist()
    idx = np.arange(n * n).reshape(n, n)
    a, b_, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    tris = np.concatenate(
        [np.stack([a, b_, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    ).ravel().tolist()  # fmt: skip
    uvs = np.stack([(X.ravel() - mnx) / (mxx - mnx), (Y.ravel() - mny) / (mxy - mny)], 1)

    g = pycvc.geometry(lab._app)
    g.add_vertices(verts)
    g.add_triangles(tris)
    g.set_uvs(uvs.ravel().tolist())
    scene.addGraphics("fogdecal", g)
    node = scene.getGraphics("fogdecal")

    ny, nx = int(shape[0]), int(shape[1])
    buf = np.zeros((ny, nx, 4), np.uint8)
    img = pycvc.image.from_numpy(buf)
    node.set_texture(img)
    node.setOpacity(0.999)  # forces the actor into VTK's translucent pass
    for f in ("setShowBBox", "setShowExtentLabels", "setShowLabel"):
        getattr(node, f)(False)
    return node, img.numpy()


#: never seen / remembered / visible-now, as RGBA. Mirrors the 2-D clips' three
#: tiers: unseen ground is hidden outright, memory is dim, live sensing is lit.
FOG_LUT = np.array(
    [
        [6, 7, 10, 235],
        [40, 47, 60, 175],
        [120, 190, 225, 95],
    ],
    np.uint8,
)


def goal_bearing_deg(traces, keys, t, pts, *, min_range_m: float = 40.0):
    """Bearing from the group to whatever it is heading for.

    The camera used to hold a bearing fixed by the shot schedule, which is
    unrelated to where anyone is going -- so as the group crossed the map the
    camera could end up parked over the destination looking back at an empty
    street, or square behind the squad watching it recede. Placing it on the
    objective's side means the vehicles drive TOWARD the lens and the street
    they are about to take is the street you are looking down.

    Returns None when the objective is too close to define a direction, so the
    caller can hold its current bearing instead of spinning about a degenerate
    one.
    """
    gs = [g for g in (traces[k].goal_at(t) for k in keys) if g is not None]
    if not gs:
        return None
    g = np.asarray(gs, np.float64).mean(axis=0)
    c = np.asarray(pts, np.float64)[:, :2].mean(axis=0)
    d = g - c
    if float(np.hypot(*d)) < min_range_m:
        return None
    return float(np.degrees(np.arctan2(d[1], d[0])))


def fog_tiers(traces, keys, t, shape):
    """Union the squad's recorded FOV into one three-tier code array.

    Coverage is shared, knowledge is not -- so the UNION of ``ever_seen`` is a
    real measured quantity ("somebody has looked here"), while a union of the
    agents' private occupancy beliefs would be a map no agent holds. This uses
    only the former.
    """
    ny, nx = int(shape[0]), int(shape[1])
    seen = np.zeros((ny, nx), bool)
    vis = np.zeros((ny, nx), bool)
    for k in keys:
        got = traces[k].fov_at(t)
        if got is None:
            continue
        v, s = got
        np.logical_or(seen, s, out=seen)
        np.logical_or(vis, v, out=vis)
    return (seen.astype(np.uint8) + vis.astype(np.uint8)).clip(0, 2)


#: Building surface, by tier. Not an overlay -- a brightness the wall is drawn
#: AT, so an unseen block recedes into the dark instead of standing lit above a
#: black carpet, which is what a ground-only fog decal leaves behind.
WALL_LUT = np.array(
    [
        [34, 38, 50, 255],
        [122, 128, 142, 255],
        [252, 248, 238, 255],
    ],
    np.uint8,
)


def add_wall_faces(scene, first_seen):
    """Colour the city PER FACE from a true mesh ray-cast.

    This supersedes the projected 2-D mask, and the difference is the one that
    was wrong before: the simulator's sensor is a 2-D cast over an occupancy
    raster, so it knows a building CELL was seen and nothing about height --
    a 90 m tower lit to its roof because somebody drove past its base. Here a
    face is lit only if a ray actually reached that face, so the lower storeys
    a vehicle could see light up and the upper ones do not, and a wall behind
    another wall stays dark.

    ``first_seen`` is per-face and monotone, so replay is a comparison: a face
    is lit iff ``0 <= first_seen <= tick``.
    """
    import pycvc_gl
    from vtkmodules.util.numpy_support import numpy_to_vtk

    act = pycvc_gl.prop(scene, "buildings")
    if act is None:
        return None
    pd = act.GetMapper().GetInput()
    n = int(pd.GetNumberOfPolys())
    if first_seen is None or len(first_seen) != n:
        return None
    rgb = np.empty((n, 3), np.uint8)
    rgb[:] = WALL_LUT[0, :3]
    arr = numpy_to_vtk(rgb, deep=0)
    arr.SetName("wall_seen")
    pd.GetCellData().SetScalars(arr)
    m = act.GetMapper()
    m.SetScalarModeToUseCellData()
    m.SetColorModeToDirectScalars()
    m.ScalarVisibilityOn()
    # first_seen is monotone, so replay only ever ADDS faces. Walking a
    # presorted order and advancing a cursor touches only the faces that light
    # up this frame, instead of recolouring all 978,242 every time (measured:
    # 385 -> 275 ms per frame-pair).
    fs = np.asarray(first_seen)
    order = np.argsort(np.where(fs < 0, np.iinfo(np.int32).max, fs), kind="stable")
    n_lit = int((fs >= 0).sum())
    return {"rgb": rgb, "arr": arr, "pd": pd, "fs": fs, "order": order[:n_lit], "cursor": [0]}


def add_wall_mask(scene, bounds, shape, *, pad: int = 1):
    """Project the recorded coverage onto the city mesh as a texture.

    A GLSL shader is reachable here (vtkOpenGLShaderProperty accepts fragment
    replacements) and was tried; it buys nothing this does not, so this takes
    the plain route: world-XY texture coordinates generated once over the 978k
    -face mesh, and one small RGBA texture re-written per frame.

    The texture domain is pushed out by half a cell plus a pad ring, and that
    is not arbitrary. The belief grid is a grid of POINTS -- cell ``c`` is
    centred at ``mn + c*cw`` -- while texels are AREAS, so sampling without the
    half-cell shift lands the whole mask one cell off. The pad ring stays
    "never seen" so the city outside the simulated box clamps to unseen rather
    than smearing the border row outward across it.

    Returns ``(texture, live_rgba_view)`` or ``None`` if there is no city mesh.
    """
    import pycvc_gl
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkRenderingCore import vtkTexture

    act = pycvc_gl.prop(scene, "buildings")
    if act is None:
        return None
    pd = act.GetMapper().GetInput()
    pts = vtk_to_numpy(pd.GetPoints().GetData())

    ny, nx = int(shape[0]), int(shape[1])
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    cw = (mxx - mnx) / (nx - 1)
    tw, th = nx + 2 * pad, ny + 2 * pad
    u0, v0 = mnx - cw * (0.5 + pad), mny - cw * (0.5 + pad)
    tc = np.empty((pts.shape[0], 2), np.float32)
    tc[:, 0] = (pts[:, 0] - u0) / (cw * tw)
    tc[:, 1] = (pts[:, 1] - v0) / (cw * th)
    np.clip(tc, 0.0, 1.0, out=tc)  # city outside the sim box clamps to unseen
    pd.GetPointData().SetTCoords(numpy_to_vtk(tc, deep=1))
    pd.Modified()

    buf = np.zeros((th, tw, 4), np.uint8)
    buf[:] = WALL_LUT[0]
    arr = numpy_to_vtk(buf.reshape(-1, 4), deep=0)
    arr.SetName("wall_mask")
    img = vtkImageData()
    img.SetDimensions(tw, th, 1)
    img.GetPointData().SetScalars(arr)
    tex = vtkTexture()
    tex.SetInputData(img)
    tex.InterpolateOn()
    tex.EdgeClampOn()
    act.GetProperty().SetTexture("wall_mask", tex)
    act.GetProperty().SetColor(1.0, 1.0, 1.0)  # let the texture carry the value
    return tex, buf, img, arr, pad


def _drape(sampler, xy, lift):
    """Put a 2-D polyline on the terrain. Scalar sampler, so this is a loop --
    it runs at SETUP, never per frame."""
    return np.array(
        [(float(x), float(y), float(sampler(float(x), float(y))) + lift) for x, y in xy],
        np.float64,
    )


def _resample(xy, step_m=14.0):
    """Tessellate each straight segment so a draped line follows the ground.

    NOT re-derivation: the recorded polyline's own vertices are left exactly
    where they are, and the added points lie ON its segments. Without this a
    two-point route drawn over a heightfield tunnels straight through hills.
    """
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    if len(xy) < 2:
        return xy
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        d = float(np.hypot(*(b - a)))
        k = max(1, int(d / step_m))
        for i in range(1, k + 1):
            out.append(a + (b - a) * (i / k))
    return np.array(out)


def add_line_node(scene, app, name, width):
    """A long-lived node whose geometry is replaced in place each frame.

    Deliberately NOT Lab.add_path: that goes through SceneGraph::addGraphics,
    which removes and rebuilds the node every call (7-14 ms each, and it leaks
    a boost::signals2 connection into m_boundsConns on every re-add). Sixteen
    of those a frame would cost more than the render.
    """
    import pycvc

    g = pycvc.geometry(app)
    scene.addGraphics(name, g)
    node = scene.getGraphics(name)
    node.setLineWidth(float(width))
    for f in ("setShowBBox", "setShowExtentLabels", "setShowLabel"):
        getattr(node, f)(False)
    return node


def set_lines(node, app, segments, colors):
    """Push merged polylines (list of (N,3) arrays) into an existing node."""
    import pycvc

    if not segments:
        node.setVisible(False)
        return
    g = pycvc.geometry(app)
    verts, lines, cols, base = [], [], [], 0
    for pts, col in zip(segments, colors):
        n = len(pts)
        if n < 2:
            continue
        verts.append(pts)
        seg = np.stack([np.arange(n - 1), np.arange(1, n)], 1) + base
        lines.append(seg)
        cols.append(np.tile(np.asarray(col, np.float64), (n, 1)))
        base += n
    if not verts:
        node.setVisible(False)
        return
    V = np.concatenate(verts)
    C = np.concatenate(cols)
    L = np.concatenate(lines)
    g.add_vertices(V.ravel().tolist())
    g.add_lines(L.ravel().astype(int).tolist())
    g.set_colors(C.ravel().tolist())
    node.setGeometry(g)
    # setGeometry resets the render mode from the geometry's own auto-mode, and
    # a lines-only cvc::geometry still reports SURFACE_TRI -- so without this it
    # draws nothing at all.
    node.setRenderMode(pycvc_gl_lines_mode())
    node.setUseSingleColor(False)
    node.setVisible(True)


def pycvc_gl_lines_mode():
    import pycvc_gl

    return pycvc_gl.GeometryRenderMode_LINES


def _open_raw_encoder(out_mp4: Path, *, fps: int, size):
    """ffmpeg taking raw bottom-up RGB on stdin (what frameRGB hands back)."""
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "-",
         "-vf", "vflip", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         str(out_mp4)],
        stdin=subprocess.PIPE,
    )  # fmt: skip


def _compose(main_mp4: Path, pip_mp4: Path, out: Path, *, fps: int, size, pip, crop_w=None) -> None:
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
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(main_mp4), "-i", str(pip_mp4),
         "-filter_complex", filt,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
         str(out)],
        check=True,
    )  # fmt: skip
