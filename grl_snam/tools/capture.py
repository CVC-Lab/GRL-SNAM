"""Drive the learned SDF policy on a real scene and render it to an mp4 OFFSCREEN
(no window) via VTK — the same engine cvcGL/VolRover3 use — with an on-frame **HUD**
of live metrics (the network's coefficients, wall clearance, mode, penetration).

Two entry points, both built on the one shared :class:`grl_snam.nav.SdfNavigator`:

* :func:`capture_drive` — a single fixed A->B run.
* :func:`capture_multigoal` — the dynamic multi-goal free-drive (goals chosen in the
  SDF's free space so every leg is reachable; goals re-targeted live; drone chase cam;
  each leg truncated at closest approach so no orbiting tail shows).

Needs the volrover env (``pycvc_gl`` scene helpers + VTK), a trained checkpoint, a
scene bundle (``terrain.json`` + ``buildings.glb``), and ffmpeg.
"""

from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np
import torch

import sdf_nav

from ..metrics import NavStats, hud_lines
from ..nav import SdfNavigator, select_reachable_goals

_GOAL_COLORS = [(0.20, 0.80, 0.35), (0.30, 0.65, 0.95), (0.95, 0.75, 0.15), (0.85, 0.35, 0.85)]


def _load(bundle: str, checkpoint: str, sdf_npz: str | None):
    """Load the trained navigator + SDF field for a bundle."""
    from pycvc_gl.scenes import building_occupancy, terrain_grid

    ck = torch.load(checkpoint, map_location="cpu")
    meta = ck["meta"]
    model = sdf_nav.CoefMLP()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    bounds = terrain_grid(os.path.join(bundle, "terrain.json"))[1]
    if sdf_npz and os.path.exists(sdf_npz):
        d = np.load(sdf_npz)
        phi, nxg, nyg = d["phi"], d["normal_x"], d["normal_y"]
    else:
        occ = building_occupancy(
            os.path.join(bundle, "buildings.glb"), bounds, 512, 512, inflate_m=0.0
        )
        phi, nxg, nyg = sdf_nav.build_sdf(occ, bounds, float(meta["scale"]))
    field = sdf_nav.SDFField(
        phi, nxg, nyg, bounds, meta["center"], float(meta["scale"]), device="cpu"
    )
    return field, model, meta


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
) -> int:
    """Render a drive (``legs`` = list of ``(goal_index, [NavMetrics,...])``) to mp4.

    Paces by arc length (constant cruise + per-leg accel/decel), places a shaded
    vehicle + goal spires, drives a drone chase camera, and draws the HUD from the
    metrics carried by each leg. Returns the number of frames written."""
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
    frames_dir = os.path.join(os.path.dirname(out_mp4) or ".", "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.system("rm -f %s/f_*.png" % frames_dir)
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
    cs.SetXLength(4.6)
    cs.SetYLength(2.0)
    cs.SetZLength(1.6)
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

    rw = vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(w, h)
    camera = ren.GetActiveCamera()
    camera.SetClippingRange(1.0, 9000.0)

    # arc-length pacing so speed is constant (per-leg sin ease = accel/decel); total
    # frames set to `minutes` for a steady cruise regardless of path length.
    total_frames = int(minutes * 60 * fps)
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
    seq = []  # (x, y, goal_index, NavMetrics)
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
        midx = np.clip(np.searchsorted(arcl, cum), 0, len(ms) - 1)  # nearest nav-step for the HUD
        for k in range(n):
            seq.append((float(xs[k]), float(ys[k]), gi, ms[int(midx[k])]))

    stats = NavStats()
    w2i = vtkWindowToImageFilter()
    w2i.SetInput(rw)
    head = np.array([1.0, 0.0])
    back, ht, ahead = (74.0, 52.0, 24.0) if cam == "drone" else (34.0, 16.0, 22.0)
    prev = np.array([seq[0][0], seq[0][1]])
    for fi, (px, py, ag, m) in enumerate(seq):
        p = np.array([px, py])
        dvec = p - prev
        prev = p
        nn = np.linalg.norm(dvec)
        if nn > 1e-4:
            head = 0.6 * head + 0.4 * (dvec / nn)
            head /= np.linalg.norm(head) + 1e-9
        zt = hsamp(px, py) + 0.9
        tf = vtkTransform()
        tf.Translate(px, py, zt + 0.8)
        tf.RotateZ(math.degrees(math.atan2(head[1], head[0])))
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
            # report the PLAYBACK ground speed (matches what the viewer sees), not the
            # sim-time speed: frame-to-frame displacement x frame rate.
            m.speed_mps = float(nn) * fps
            stats.update(m)
            hud_actor.SetInput("\n".join(hud_lines(m, stats)))
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
) -> str:
    """Drive a single fixed A->B run and render it (with HUD) to ``out``."""
    field, model, meta = _load(bundle, checkpoint, sdf_npz)
    nav = SdfNavigator(field, model, meta, reach_tol=0.5)
    nav.start(start, goal)
    ms, best_i, _bo, _bv = nav.drive_to_goal(max_steps=3000)
    print(
        "drive: %d steps, best-approach %.1f m, penetration %d"
        % (len(ms), ms[best_i].goal_dist_m, sum(1 for m in ms if m.inside_building))
    )
    n = render_drive(
        bundle, [(0, ms[: best_i + 1])], [goal], out, minutes=minutes, fps=fps, hud=hud, cam=cam
    )
    print("wrote %s (%d frames)" % (out, n))
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
) -> str:
    """Dynamic multi-goal free-drive (goals chosen in the SDF's free space, re-targeted
    live, drone chase cam) rendered with HUD to ``out``."""
    field, model, meta = _load(bundle, checkpoint, sdf_npz)
    start, goals = select_reachable_goals(field, model, meta, n_corners=n_goals)
    print("start %s  goals %s" % (np.round(start).tolist(), [np.round(g).tolist() for g in goals]))
    nav = SdfNavigator(field, model, meta, reach_tol=0.8)
    o = torch.from_numpy(nav.w2n(start)).unsqueeze(0).float()
    v = torch.zeros(1, 2)
    legs = []
    total_pen = 0
    for gi, g in enumerate(goals):
        nav.o = o.clone()
        nav.v = v.clone()
        nav.set_goal(g, goal_index=gi)
        ms, best_i, o, v = nav.drive_to_goal(max_steps=1300)
        legs.append((gi, ms[: best_i + 1]))
        total_pen += sum(1 for m in ms[: best_i + 1] if m.inside_building)
    print(
        "multigoal drive: %d steps over %d legs, penetration=%d"
        % (sum(len(ms) for _, ms in legs), len(legs), total_pen)
    )
    n = render_drive(bundle, legs, goals, out, minutes=minutes, fps=fps, hud=hud, cam="drone")
    print("wrote %s (%d frames, %.0fs)" % (out, n, n / fps))
    return out
