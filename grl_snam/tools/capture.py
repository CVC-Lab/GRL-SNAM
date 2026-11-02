"""Drive the learned SDF policy on a real scene and render it to an mp4 OFFSCREEN
(no window) via VTK — the same engine cvcGL/VolRover3 use — with an on-frame **HUD**
of live metrics (the network's coefficients, wall clearance, mode, penetration).

Two entry points:

* :func:`capture_drive` — a single fixed A->B run.
* :func:`capture_multigoal` — the dynamic multi-goal free-drive (goals chosen in the
  SDF's free space so every leg is reachable; goals re-targeted live; drone chase cam).

``plant`` picks what drives:

* ``"point"`` (the default) — the Python :class:`grl_snam.nav.SdfNavigator` on the point-mass
  surrogate the policy trains on. Each leg is truncated at closest approach so no orbiting
  tail shows, and the video is paced by arc length.
* ``"kinematic"``, ``"dynamic"`` or ``"jolt"`` — the same policy on libcvc's sim_world
  (:class:`grl_snam.plant_nav.PlantNavigator`) driving nav's kinematic bicycle, the dynamic
  single-track vehicle (tires, load transfer, actuator lag) or a Jolt vehicle (``vehicle``,
  on ground of grip ``ground_mu``, the scene's buildings as static boxes). The whole run is
  kept, failures included, and the video is paced by the clock (``speed`` world seconds per
  played second), so a stall or a spin takes the time it took. The HUD adds the vehicle's
  sideslip, tire use, lateral g and energy books, and a ``.json`` beside the mp4 records the
  run's outcome.

Needs VTK, a trained checkpoint, a scene bundle (``terrain.json`` + ``buildings.glb``), and
ffmpeg; ``pycvc_gl``'s scene helpers unless ``sdf_npz`` is given; and for the plants a pycvc
with libcvc's sim_world and plant bindings.
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import textwrap

import numpy as np
import torch

import grl_snam.sdf_nav as sdf_nav

from ..metrics import NavStats, hud_lines
from ..nav import SdfNavigator, select_reachable_goals
from .sdf import terrain_bounds

_GOAL_COLORS = [(0.20, 0.80, 0.35), (0.30, 0.65, 0.95), (0.95, 0.75, 0.15), (0.85, 0.35, 0.85)]


#: What ``capture`` can drive: the Python surrogate, or a plant (grl_snam.plant_nav.PLANTS).
PLANTS = ("point", "kinematic", "dynamic", "jolt")
#: A leg on a plant ends after this many ticks without getting closer to its goal.
STALL_TICKS = 340
#: On a plant a goal is reached within this many metres (the trained metric's own radius,
#: reach_tol normalized units, can be hundreds of metres on a city).
REACH_M = 15.0


def _load(bundle: str, checkpoint: str, sdf_npz: str | None):
    """Load the trained navigator + SDF field for a bundle."""
    ck = torch.load(checkpoint, map_location="cpu")
    meta = ck["meta"]
    model = sdf_nav.CoefMLP()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    bounds = terrain_bounds(bundle)
    if sdf_npz and os.path.exists(sdf_npz):
        d = np.load(sdf_npz)
        phi, nxg, nyg = d["phi"], d["normal_x"], d["normal_y"]
    else:
        from pycvc_gl.scenes import building_occupancy

        occ = building_occupancy(
            os.path.join(bundle, "buildings.glb"), bounds, 512, 512, inflate_m=0.0
        )
        phi, nxg, nyg = sdf_nav.build_sdf(occ, bounds, float(meta["scale"]))
    field = sdf_nav.SDFField(
        phi, nxg, nyg, bounds, meta["center"], float(meta["scale"]), device="cpu"
    )
    return field, model, meta


def _occupancy(field) -> np.ndarray:
    """The field's obstacles, (rows, cols) bool on its own grid (row 0 at min y): phi < 0."""
    return field.field[0, 0].numpy() < 0.0


def _navigator(
    field, model, meta, plant, *, reach_tol, vehicle, ground_mu, routes, reach_m=REACH_M,
    fog=False, weights=None,
):  # fmt: skip
    """SdfNavigator for ``"point"``, else a PlantNavigator on the plant (arriving within
    ``reach_m``; ``fog``: the map starts unknown; ``weights``: a ``.cvcnav`` to drive)."""
    if plant == "point":
        if fog or weights:
            raise ValueError("--fog and --weights need a plant: the point surrogate is torch's")
        return SdfNavigator(field, model, meta, reach_tol=reach_tol)
    from ..plant_nav import PlantNavigator, plant_label

    if plant not in PLANTS:
        raise ValueError(f"plant must be one of {PLANTS}")
    return PlantNavigator(
        field,
        model,
        meta,
        _occupancy(field),
        plant_label(plant, vehicle),
        reach_m=reach_m,
        ground_mu=ground_mu,
        routes=routes,
        known_map=not fog,
        weights=weights,
    )


def _vehicle_box(plant: str, vehicle: str | None):
    """The rendered vehicle's (length, width, height), m: the plant vehicle's chassis box, or
    the Humvee-sized default."""
    if plant in ("point", "kinematic") or not vehicle:
        return (4.6, 2.0, 1.6)
    from .. import nav_native

    hx, hy, hz = nav_native.vehicle_desc(vehicle).half_extents()
    return (2 * hx, 2 * hy, 2 * hz)


def _summary(legs, goals, plant, vehicle, ground_mu, dt, stall=STALL_TICKS) -> dict:
    """A run's outcome for the ``.json`` beside the mp4: per leg, how it ended (``reached``,
    ``stalled``: ``stall`` ticks without getting closer, or ``out of ticks``) and what the
    vehicle went through."""
    out = {"plant": plant, "vehicle": vehicle, "ground_mu": ground_mu, "legs": []}
    for gi, ms in legs:
        if not ms:
            continue
        leg = {
            "goal": [round(float(c), 2) for c in goals[gi]],
            "reached": bool(ms[-1].reached),
            "ticks": len(ms),
            "world_s": round(len(ms) * dt, 2),
            "closest_m": round(min(m.goal_dist_m for m in ms), 2),
            "final_dist_m": round(ms[-1].goal_dist_m, 2),
            "ended": (
                "reached"
                if ms[-1].reached
                else "stalled" if ms[-1].stall > stall else "out of ticks"
            ),
            "penetration_steps": sum(1 for m in ms if m.inside_building),
            "min_clearance_m": round(min(m.clearance_m for m in ms), 2),
            "max_speed_mps": round(max(m.speed_mps for m in ms), 2),
        }
        ps = [m.plant for m in ms if m.plant and "slip_deg" in m.plant]
        if ps:
            leg["max_abs_slip_deg"] = round(max(abs(p["slip_deg"]) for p in ps), 1)
            leg["max_tire_use"] = round(max(p["tire_use"] for p in ps), 3)
            leg["ticks_past_peak"] = sum(1 for p in ps if p["tire_use"] > 1.0)
            leg["max_abs_lateral_g"] = round(max(abs(p["lateral_g"]) for p in ps), 3)
            leg["contact_ticks"] = sum(1 for p in ps if p.get("touching"))
            leg["books_J"] = {k: round(v, 1) for k, v in ps[-1]["books"].items()}
        out["legs"].append(leg)
    return out


def _hsamp_fn(bundle: str):
    """A bilinear terrain-height sampler ``h(x, y)`` + the terrain polydata builder."""
    d = json.load(open(os.path.join(bundle, "terrain.json")))
    b = d["bounds"]
    grid = list(reversed(d["grid"]))
    rows, cols = d["rows"], d["cols"]
    mnx, mny, mxx, mxy = b["min_x"], b["min_y"], b["max_x"], b["max_y"]
    dx = (mxx - mnx) / (cols - 1)
    dy = (mxy - mny) / (rows - 1)

    def hsamp(x, y):
        fx = min(max((x - mnx) / dx, 0), cols - 1)
        fy = min(max((y - mny) / dy, 0), rows - 1)
        c0, r0 = int(fx), int(fy)
        c1, r1 = min(c0 + 1, cols - 1), min(r0 + 1, rows - 1)
        tx, ty = fx - c0, fy - r0
        return (grid[r0][c0] * (1 - tx) + grid[r0][c1] * tx) * (1 - ty) + (
            grid[r1][c0] * (1 - tx) + grid[r1][c1] * tx
        ) * ty

    return hsamp, (grid, rows, cols, mnx, mny, dx, dy)


def _smooth(a, k=5):
    a = np.asarray(a, np.float32)
    out = a.copy()
    for i in range(len(a)):
        out[i] = a[max(0, i - k) : i + k + 1].mean(0)
    return out


def render_drive(
    bundle: str,
    legs,
    goals,
    out_mp4: str,
    *,
    minutes: float = 3.0,
    fps: int = 15,
    hud: bool = True,
    size=(960, 540),
    cam="drone",
    pace: str = "arc",
    dt: float = 0.06,
    speed: float = 1.0,
    vehicle_box=(4.6, 2.0, 1.6),
    caption: str | None = None,
) -> int:
    """Render a drive (``legs`` = list of ``(goal_index, [NavMetrics,...])``) to mp4.

    ``pace="arc"`` paces by arc length (constant cruise + per-leg accel/decel) over
    ``minutes``; ``pace="clock"`` plays the run as it happened, ``speed`` world seconds
    (``dt`` per metric) per played second, sped up only as far as fitting ``minutes`` needs,
    with the vehicle at its own heading. Places a shaded vehicle (``vehicle_box``: length,
    width, height in m) + goal spires, drives a drone chase camera, draws the HUD from the
    metrics carried by each leg and ``caption`` along the bottom. Returns the number of
    frames written."""
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals, vtkQuadricDecimation
    from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
    from vtkmodules.vtkFiltersSources import vtkConeSource, vtkCubeSource
    from vtkmodules.vtkIOGeometry import vtkGLTFReader
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkLight,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkTextActor,
        vtkWindowToImageFilter,
    )

    w, h = size
    # One frames dir per video, emptied first (a shorter clip must not pick up a longer
    # one's tail; `rm` through os.system did nothing on Windows).
    frames_dir = os.path.splitext(out_mp4)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for old in glob.glob(os.path.join(frames_dir, "f_*.png")):
        os.remove(old)
    hsamp, (grid, rows, cols, mnx, mny, dx, dy) = _hsamp_fn(bundle)

    pts = vtkPoints()
    pts.SetNumberOfPoints(rows * cols)
    for r in range(rows):
        for c in range(cols):
            pts.SetPoint(r * cols + c, mnx + c * dx, mny + r * dy, float(grid[r][c]))
    tris = vtkCellArray()
    for r in range(rows - 1):
        for c in range(cols - 1):
            vv = r * cols + c
            for cell in ((vv, vv + 1, vv + cols), (vv + 1, vv + cols + 1, vv + cols)):
                tris.InsertNextCell(3)
                for idx in cell:
                    tris.InsertCellPoint(idx)
    tpd = vtkPolyData()
    tpd.SetPoints(pts)
    tpd.SetPolys(tris)
    tn = vtkPolyDataNormals()
    tn.SetInputData(tpd)
    tn.Update()
    tm = vtkPolyDataMapper()
    tm.SetInputConnection(tn.GetOutputPort())
    tm.ScalarVisibilityOff()
    terrain = vtkActor()
    terrain.SetMapper(tm)
    terrain.GetProperty().SetColor(0.33, 0.39, 0.27)
    terrain.GetProperty().SetAmbient(0.25)

    gr = vtkGLTFReader()
    gr.SetFileName(os.path.join(bundle, "buildings.glb"))
    gr.Update()
    gf = vtkCompositeDataGeometryFilter()
    gf.SetInputConnection(gr.GetOutputPort())
    gf.Update()
    dec = vtkQuadricDecimation()
    dec.SetInputData(gf.GetOutput())
    dec.SetTargetReduction(0.65)
    dec.Update()
    bm = vtkPolyDataMapper()
    bm.SetInputData(dec.GetOutput())
    bm.SetStatic(1)
    bm.ScalarVisibilityOff()
    bld = vtkActor()
    bld.SetMapper(bm)
    bp = bld.GetProperty()
    bp.SetColor(0.72, 0.72, 0.77)
    bp.SetAmbient(0.4)
    bp.SetDiffuse(0.72)

    cs = vtkCubeSource()
    cs.SetXLength(float(vehicle_box[0]))
    cs.SetYLength(float(vehicle_box[1]))
    cs.SetZLength(float(vehicle_box[2]))
    cs.Update()
    vm = vtkPolyDataMapper()
    vm.SetInputConnection(cs.GetOutputPort())
    veh = vtkActor()
    veh.SetMapper(vm)
    veh.GetProperty().SetColor(0.90, 0.13, 0.12)
    veh.GetProperty().SetAmbient(0.3)
    veh.GetProperty().SetDiffuse(0.8)

    def shaded_cone(height, rad, col, down=True):
        c = vtkConeSource()
        c.SetHeight(height)
        c.SetRadius(rad)
        c.SetResolution(26)
        c.SetDirection(0, 0, -1 if down else 1)
        c.Update()
        n = vtkPolyDataNormals()
        n.SetInputConnection(c.GetOutputPort())
        n.Update()
        m = vtkPolyDataMapper()
        m.SetInputConnection(n.GetOutputPort())
        a = vtkActor()
        a.SetMapper(m)
        pr = a.GetProperty()
        pr.SetColor(*col)
        pr.SetAmbient(0.22)
        pr.SetDiffuse(0.85)
        pr.SetSpecular(0.25)
        pr.SetSpecularPower(18)
        return a

    veh_beacon = shaded_cone(24, 5.0, (0.98, 0.42, 0.10), down=True)
    goal_beacons = [
        shaded_cone(150, 9.0, _GOAL_COLORS[i % len(_GOAL_COLORS)], down=False)
        for i in range(len(goals))
    ]

    ren = vtkRenderer()
    ren.SetBackground(0.16, 0.19, 0.13)
    ren.SetBackground2(0.55, 0.68, 0.82)
    ren.GradientBackgroundOn()
    for a in (terrain, bld, veh, veh_beacon):
        ren.AddActor(a)
    for gi, gb in enumerate(goal_beacons):
        t = vtkTransform()
        t.Translate(goals[gi][0], goals[gi][1], hsamp(*goals[gi]) + 74)
        gb.SetUserTransform(t)
        ren.AddActor(gb)
    sun = vtkLight()
    sun.SetPosition(mnx - 500, mny - 500, 2500)
    sun.SetFocalPoint(mnx + (cols - 1) * dx / 2, mny + (rows - 1) * dy / 2, 0)
    sun.SetIntensity(1.0)
    sun.SetLightTypeToSceneLight()
    ren.AddLight(sun)

    # The path driven so far, a line just above the ground (clock pacing: it shows a loop,
    # a dead end or a slide for what it was).
    trail_pts = vtkPoints()
    trail_pd = vtkPolyData()
    trail_pd.SetPoints(trail_pts)
    trail_pd.SetLines(vtkCellArray())
    tmap = vtkPolyDataMapper()
    tmap.SetInputData(trail_pd)
    trail = vtkActor()
    trail.SetMapper(tmap)
    trail.GetProperty().SetColor(1.0, 0.85, 0.25)
    trail.GetProperty().SetLineWidth(3.0)
    trail.GetProperty().LightingOff()
    if pace == "clock":
        ren.AddActor(trail)

    hud_actor = None
    if hud:
        hud_actor = vtkTextActor()
        hud_actor.SetDisplayPosition(16, h - 24)
        tp = hud_actor.GetTextProperty()
        tp.SetFontFamilyToCourier()
        tp.SetFontSize(15)
        tp.SetColor(0.92, 0.97, 0.85)
        tp.SetLineOffset(-2)
        tp.SetVerticalJustificationToTop()
        tp.SetBackgroundColor(0.05, 0.07, 0.03)
        tp.SetBackgroundOpacity(0.55)
        ren.AddActor2D(hud_actor)
    if caption:
        cap = vtkTextActor()
        cap.SetDisplayPosition(16, 14)
        cap.SetInput(textwrap.fill(caption, width=max(40, w // 9)))
        cp = cap.GetTextProperty()
        cp.SetFontFamilyToArial()
        cp.SetFontSize(17)
        cp.SetColor(1.0, 0.93, 0.62)
        cp.SetBackgroundColor(0.05, 0.05, 0.05)
        cp.SetBackgroundOpacity(0.7)
        ren.AddActor2D(cap)

    rw = vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(w, h)
    camera = ren.GetActiveCamera()
    camera.SetClippingRange(1.0, 9000.0)

    total_frames = int(minutes * 60 * fps)
    seq = []  # (x, y, goal_index, NavMetrics, vehicle yaw or None)
    if pace == "clock":
        # One frame per 1/fps played seconds at `speed`, faster only to fit `minutes`.
        flat = [(gi, m) for gi, ms in legs for m in ms]
        world_s = len(flat) * float(dt)
        speed = max(float(speed), world_s * fps / max(total_frames, 1))
        n = max(1, int(world_s / speed * fps))
        for k in range(n):
            gi, m = flat[min(len(flat) - 1, int(k * speed / (fps * dt)))]
            seq.append((m.x, m.y, gi, m, m.heading_rad))
    else:
        # arc-length pacing so speed is constant (per-leg sin ease = accel/decel); total
        # frames set to `minutes` for a steady cruise regardless of path length.
        smoothed = []
        lens = []
        for gi, ms in legs:
            seg = np.array([[m.x, m.y] for m in ms], np.float32)
            if len(seg) < 2:
                continue
            seg_s = _smooth(seg)
            dd = np.linalg.norm(np.diff(seg_s, axis=0), axis=1)
            arcl = np.concatenate([[0.0], np.cumsum(dd)])
            smoothed.append((gi, seg_s, arcl, ms))
            lens.append(float(arcl[-1]))
        tot = sum(lens) or 1.0
        for (gi, seg_s, arcl, ms), leg_len in zip(smoothed, lens):
            if leg_len < 1.0:
                continue
            n = max(120, int(round(total_frames * leg_len / tot)))
            u = np.linspace(0, 1, n)
            spd = 0.35 + 0.65 * np.sin(np.pi * u)
            cum = np.cumsum(spd)
            cum = cum / cum[-1] * leg_len
            xs = np.interp(cum, arcl, seg_s[:, 0])
            ys = np.interp(cum, arcl, seg_s[:, 1])
            midx = np.clip(np.searchsorted(arcl, cum), 0, len(ms) - 1)  # nearest step for the HUD
            for k in range(n):
                seq.append((float(xs[k]), float(ys[k]), gi, ms[int(midx[k])], None))

    stats = NavStats()
    w2i = vtkWindowToImageFilter()
    w2i.SetInput(rw)
    head = np.array([1.0, 0.0])
    back, ht, ahead = (74.0, 52.0, 24.0) if cam == "drone" else (34.0, 16.0, 22.0)
    prev = np.array([seq[0][0], seq[0][1]])
    if seq[0][4] is not None:
        head = np.array([math.cos(seq[0][4]), math.sin(seq[0][4])])
    for fi, (px, py, ag, m, yaw) in enumerate(seq):
        p = np.array([px, py])
        dvec = p - prev
        prev = p
        nn = np.linalg.norm(dvec)
        if nn > 1e-4:
            head = 0.6 * head + 0.4 * (dvec / nn)
            head /= np.linalg.norm(head) + 1e-9
        zt = hsamp(px, py) + 0.9
        if pace == "clock":
            trail_pts.InsertNextPoint(px, py, zt - 0.4)
            n_tp = trail_pts.GetNumberOfPoints()
            if n_tp >= 2:
                line = vtkCellArray()
                line.InsertNextCell(n_tp)
                for j in range(n_tp):
                    line.InsertCellPoint(j)
                trail_pd.SetLines(line)
                trail_pts.Modified()
                trail_pd.Modified()
        tf = vtkTransform()
        tf.Translate(px, py, zt + 0.8)
        # The camera follows the motion; the body points where the vehicle does (a spin shows).
        tf.RotateZ(math.degrees(yaw if yaw is not None else math.atan2(head[1], head[0])))
        veh.SetUserTransform(tf)
        bt = vtkTransform()
        bt.Translate(px, py, zt + 24.0)
        veh_beacon.SetUserTransform(bt)
        for gi, gb in enumerate(goal_beacons):
            gb.GetProperty().SetAmbient(0.62 if gi == ag else 0.22)
        camera.SetPosition(px - head[0] * back, py - head[1] * back, zt + ht)
        camera.SetFocalPoint(px + head[0] * ahead, py + head[1] * ahead, zt - 2.0)
        camera.SetViewUp(0, 0, 1)
        if hud_actor is not None:
            if yaw is None:
                # report the PLAYBACK ground speed (matches what the viewer sees), not the
                # sim-time speed: frame-to-frame displacement x frame rate.
                m.speed_mps = float(nn) * fps
            stats.update(m)
            lines = hud_lines(m, stats)
            if pace == "clock":
                lines.append(f"t {m.step * dt:6.1f} s   playback x{speed:.1f}")
            hud_actor.SetInput("\n".join(lines))
        rw.Render()
        w2i.Modified()
        w2i.Update()
        wr = vtkPNGWriter()
        wr.SetFileName("%s/f_%05d.png" % (frames_dir, fi))
        wr.SetInputConnection(w2i.GetOutputPort())
        wr.Write()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            "%s/f_%%05d.png" % frames_dir,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=%d:%d" % (w, h),
            out_mp4,
        ],
        check=True,
    )
    return len(seq)


def capture_drive(
    bundle: str,
    checkpoint: str,
    start,
    goal,
    out: str = "drive.mp4",
    *,
    sdf_npz: str | None = None,
    minutes: float = 1.0,
    fps: int = 15,
    hud: bool = True,
    cam: str = "chase",
    plant: str = "point",
    vehicle: str | None = "humvee",
    ground_mu: float = 0.9,
    routes: bool = False,
    fog: bool = False,
    reach_m: float = REACH_M,
    weights: str | None = None,
    max_steps: int = 3000,
    speed: float = 1.0,
    caption: str | None = None,
) -> str:
    """Drive a single fixed A->B run on ``plant`` and render it (with HUD) to ``out``."""
    field, model, meta = _load(bundle, checkpoint, sdf_npz)
    nav = _navigator(
        field, model, meta, plant, reach_tol=0.5, vehicle=vehicle, ground_mu=ground_mu,
        routes=routes, reach_m=reach_m, fog=fog, weights=weights,
    )  # fmt: skip
    nav.start(start, goal)
    if plant == "point":
        ms, best_i, _bo, _bv = nav.drive_to_goal(max_steps=max_steps)
        legs = [(0, ms[: best_i + 1])]
    else:
        ms, best_i, _bo, _bv = nav.drive_to_goal(max_steps=max_steps, stall=STALL_TICKS)
        legs = [(0, ms)]
    print(
        "drive: %d steps, best-approach %.1f m, penetration %d"
        % (len(ms), ms[best_i].goal_dist_m, sum(1 for m in ms if m.inside_building))
    )
    return _finish(
        bundle, nav, legs, [goal], out, plant, vehicle, ground_mu, fog,
        minutes=minutes, fps=fps, hud=hud, cam=cam, speed=speed, caption=caption,
    )  # fmt: skip


def _finish(
    bundle, nav, legs, goals, out, plant, vehicle, ground_mu, fog, *, minutes, fps, hud, cam,
    speed, caption,
) -> str:  # fmt: skip
    """Render ``legs`` and, on a plant, write the run's outcome beside the mp4."""
    clock = plant != "point"
    n = render_drive(
        bundle, legs, goals, out, minutes=minutes, fps=fps, hud=hud, cam=cam,
        pace="clock" if clock else "arc", dt=nav.dt, speed=speed,
        vehicle_box=_vehicle_box(plant, vehicle), caption=caption,
    )  # fmt: skip
    print("wrote %s (%d frames, %.0fs)" % (out, n, n / fps))
    if clock:
        summary = _summary(legs, goals, plant, vehicle, ground_mu, nav.dt)
        summary["fog"] = bool(fog)
        summary["reach_m"] = round(nav.reach_tol / nav.S, 2)
        summary["caption"] = caption
        summary["nav_stats"] = nav.nav_stats()
        path = os.path.splitext(out)[0] + ".json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=1)
        for i, leg in enumerate(summary["legs"]):
            print(
                "  leg %d: %s" % (i, json.dumps({k: v for k, v in leg.items() if k != "books_J"}))
            )
    return out


def capture_multigoal(
    bundle: str,
    checkpoint: str,
    out: str = "multigoal.mp4",
    *,
    sdf_npz: str | None = None,
    minutes: float = 3.0,
    fps: int = 15,
    hud: bool = True,
    n_goals: int = 4,
    plant: str = "point",
    vehicle: str | None = "humvee",
    ground_mu: float = 0.9,
    routes: bool = False,
    fog: bool = False,
    reach_m: float = REACH_M,
    weights: str | None = None,
    max_steps: int = 1300,
    speed: float = 1.0,
    caption: str | None = None,
) -> str:
    """Dynamic multi-goal free-drive (goals chosen in the SDF's free space, re-targeted
    live, drone chase cam) on ``plant``, rendered with HUD to ``out``. ``max_steps`` bounds
    each leg. On a plant each leg starts where the last one ended."""
    field, model, meta = _load(bundle, checkpoint, sdf_npz)
    start, goals = select_reachable_goals(field, model, meta, n_corners=n_goals)
    print("start %s  goals %s" % (np.round(start).tolist(), [np.round(g).tolist() for g in goals]))
    nav = _navigator(
        field, model, meta, plant, reach_tol=0.8, vehicle=vehicle, ground_mu=ground_mu,
        routes=routes, reach_m=reach_m, fog=fog, weights=weights,
    )  # fmt: skip
    legs = []
    total_pen = 0
    if plant == "point":
        o = torch.from_numpy(nav.w2n(start)).unsqueeze(0).float()
        v = torch.zeros(1, 2)
        for gi, g in enumerate(goals):
            nav.o = o.clone()
            nav.v = v.clone()
            nav.set_goal(g, goal_index=gi)
            ms, best_i, o, v = nav.drive_to_goal(max_steps=max_steps)
            legs.append((gi, ms[: best_i + 1]))
            total_pen += sum(1 for m in ms[: best_i + 1] if m.inside_building)
    else:
        nav.start(start, goals[0])
        for gi, g in enumerate(goals):
            nav.set_goal(g, goal_index=gi)
            ms, _bi, _o, _v = nav.drive_to_goal(max_steps=max_steps, stall=STALL_TICKS)
            legs.append((gi, ms))
            total_pen += sum(1 for m in ms if m.inside_building)
    print(
        "multigoal drive: %d steps over %d legs, penetration=%d"
        % (sum(len(ms) for _, ms in legs), len(legs), total_pen)
    )
    return _finish(
        bundle, nav, legs, goals, out, plant, vehicle, ground_mu, fog,
        minutes=minutes, fps=fps, hud=hud, cam="drone", speed=speed, caption=caption,
    )  # fmt: skip
