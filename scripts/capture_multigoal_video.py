"""Capture the DYNAMIC multi-goal free-drive to an mp4: the learned SDF policy starts
near the centre of a real scene and heads for a goal that is RE-TARGETED live through
several points toward the corners — no precomputed route. Renders the actual 3-D scene
(terrain relief + glTF buildings + vehicle + shaded goal spires) from a drone chase
camera OFFSCREEN via VTK (the engine cvcGL/VolRover3 use), and ffmpeg's it to video.

Two robustness details make the drive clean (no circling, no far-corner wandering):
  * goals are chosen in the SDF's OWN free space (so selection and navigation agree —
    every goal is actually reachable), corner-ward but reachable, in a perimeter loop;
  * each leg is rendered only up to its CLOSEST APPROACH to the goal, so any
    unproductive orbit/back-track after the last real progress is never shown (and the
    goal simply re-targets — on-narrative for a dynamic free-drive).

Needs the volrover env (pycvc_gl/VTK), torch, the GRL-SNAM repo, a scene bundle
(terrain.json + buildings.glb), a trained SDF checkpoint, and ffmpeg.

    python scripts/capture_multigoal_video.py <bundle> checkpoints/coef_sdf.pt \
        --sdf <bundle>/nav_sdf.npz --minutes 3 -o multigoal.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess

import numpy as np
import torch

import sdf_nav
from pycvc_gl.scenes import building_occupancy, terrain_grid


def _leg(field, model, meta, o, v, goal_w, kw, maxst=1300):
    """Drive toward goal_w; return the state and trajectory truncated at the CLOSEST
    approach to the goal (so an orbit/back-track tail is never rendered)."""
    S = meta["scale"]; cx, cy = meta["center"]

    def w2n(p): return np.array([(p[0] - cx) * S, (p[1] - cy) * S], np.float32)

    def n2w(on): return np.array([on[0] / S + cx, on[1] / S + cy], np.float32)

    gn = w2n(goal_w); best = 1e9; stall = 0; mode = "seek"; turn = 1.0; dhit = 0.0
    out = []; best_o = o.clone(); best_v = v.clone(); besti = 0; bg = 1e9; since = 0
    with torch.no_grad():
        for _ in range(maxst):
            p = o[0].numpy(); dg = float(np.linalg.norm(p - gn)); gdir = (gn - p) / (dg + 1e-6)
            if dg < best - 1e-3:
                best = dg; stall = 0
            else:
                stall += 1
            if mode == "seek" and stall > 70:
                _, nn = field.sample(torch.from_numpy(p).unsqueeze(0)); nn = nn[0].numpy()
                t = np.array([-nn[1], nn[0]], np.float32)
                turn = 1.0 if np.dot(t, gdir) >= 0 else -1.0; dhit = dg; mode = "wall"; stall = 0
            if mode == "wall":
                _, nn = field.sample(torch.from_numpy(p).unsqueeze(0)); nn = nn[0].numpy()
                t = turn * np.array([-nn[1], nn[0]], np.float32)
                carrot = (p + (0.6 * t + 0.4 * nn) * 1.6).astype(np.float32)
                if dg < dhit - 1.2 or stall > 240:
                    mode = "seek"; best = dg; stall = 0
            else:
                carrot = (p + gdir * min(1.8, dg)).astype(np.float32)
            al, be, ga = model(sdf_nav.coef_feats(field, o, torch.from_numpy(carrot).unsqueeze(0)))
            o, v, _ = sdf_nav.sdf_rollout(field, o, v, torch.from_numpy(carrot).unsqueeze(0), al, be, ga, 1,
                                         nsub=meta["nsub"], **kw)
            out.append(n2w(o[0].numpy()))
            pg = float(np.linalg.norm(o[0].numpy() - gn))
            if pg < bg - 1e-3:
                bg = pg; besti = len(out) - 1; best_o = o.clone(); best_v = v.clone(); since = 0
            else:
                since += 1
            if pg < 0.8:
                besti = len(out) - 1; best_o = o.clone(); best_v = v.clone(); break
            if since > 340:
                break
    return best_o, best_v, out[: besti + 1]


def select_goals(field, meta, bounds, clear_m=16.0):
    """Pick 4 corner-ward goals in the SDF's free space, chained from an open centre
    cell in a perimeter loop (NE, SE, SW, NW) so consecutive legs are short/reachable."""
    S = meta["scale"]; cx, cy = meta["center"]; region = meta["region"]
    mnx, mny, mxx, mxy = bounds
    kw = dict(rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"])

    def w2n(p): return np.array([(p[0] - cx) * S, (p[1] - cy) * S], np.float32)

    def clearance(pw):
        d, _ = field.sample(torch.from_numpy(w2n(pw)).unsqueeze(0)); return float(d[0]) / S

    start = None
    for rad in (0, 20, 40, 60, 80, 120):
        for a in range(0, 360, 30):
            pw = np.array([cx + math.cos(math.radians(a)) * rad, cy + math.sin(math.radians(a)) * rad], np.float32)
            if clearance(pw) > clear_m:
                start = pw; break
        if start is not None:
            break

    def candidates(sx, sy):
        diag = math.atan2(sy, sx); out = []
        for frac in (0.62, 0.56, 0.50, 0.44, 0.38, 0.32):
            for da in (0, -18, 18, -32, 32):
                ang = diag + math.radians(da); r = region * frac
                gw = np.array([cx + math.cos(ang) * r, cy + math.sin(ang) * r], np.float32)
                if mnx < gw[0] < mxx and mny < gw[1] < mxy and clearance(gw) > clear_m:
                    out.append(gw)
        return out

    o = torch.from_numpy(w2n(start)).unsqueeze(0); v = torch.zeros(1, 2); goals = []
    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        chosen = None; chosen_ba = 1e9; chosen_state = None
        for gw in candidates(sx, sy)[:8]:
            bo, bv, seg = _leg(field, model_ref["m"], meta, o.clone(), v.clone(), gw, kw, maxst=800)
            ba = float(np.linalg.norm(bo[0].numpy() - w2n(gw))) / S
            if ba < chosen_ba:
                chosen_ba = ba; chosen = gw; chosen_state = (bo, bv)
            if ba < 70:
                chosen = gw; chosen_state = (bo, bv); break
        o, v = chosen_state; goals.append(chosen)
    return start, goals


model_ref = {"m": None}  # module-level handle so select_goals can reuse the loaded model


def render_video(bundle, legs, goals, out_mp4, minutes=3.0, fps=15, size=(960, 540)):
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals, vtkQuadricDecimation
    from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkConeSource
    from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
    from vtkmodules.vtkIOGeometry import vtkGLTFReader
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (vtkRenderer, vtkRenderWindow, vtkActor, vtkPolyDataMapper,
                                             vtkWindowToImageFilter, vtkLight)
    W, H = size
    fr = os.path.join(os.path.dirname(out_mp4) or ".", "_frames"); os.makedirs(fr, exist_ok=True)
    os.system("rm -f %s/f_*.png" % fr)
    d = json.load(open(os.path.join(bundle, "terrain.json"))); b = d["bounds"]
    grid = list(reversed(d["grid"])); rows, cols = d["rows"], d["cols"]
    mnx, mny, mxx, mxy = b["min_x"], b["min_y"], b["max_x"], b["max_y"]
    dx = (mxx - mnx) / (cols - 1); dy = (mxy - mny) / (rows - 1)

    def hsamp(x, y):
        fx = min(max((x - mnx) / dx, 0), cols - 1); fy = min(max((y - mny) / dy, 0), rows - 1)
        c0, r0 = int(fx), int(fy); c1 = min(c0 + 1, cols - 1); r1 = min(r0 + 1, rows - 1)
        tx, ty = fx - c0, fy - r0
        return ((grid[r0][c0] * (1 - tx) + grid[r0][c1] * tx) * (1 - ty)
                + (grid[r1][c0] * (1 - tx) + grid[r1][c1] * tx) * ty)

    pts = vtkPoints(); pts.SetNumberOfPoints(rows * cols)
    for r in range(rows):
        for c in range(cols):
            pts.SetPoint(r * cols + c, mnx + c * dx, mny + r * dy, float(grid[r][c]))
    tris = vtkCellArray()
    for r in range(rows - 1):
        for c in range(cols - 1):
            vv = r * cols + c
            for cell in ((vv, vv + 1, vv + cols), (vv + 1, vv + cols + 1, vv + cols)):
                tris.InsertNextCell(3); [tris.InsertCellPoint(i) for i in cell]
    tpd = vtkPolyData(); tpd.SetPoints(pts); tpd.SetPolys(tris)
    tn = vtkPolyDataNormals(); tn.SetInputData(tpd); tn.Update()
    tm = vtkPolyDataMapper(); tm.SetInputConnection(tn.GetOutputPort()); tm.ScalarVisibilityOff()
    terrain = vtkActor(); terrain.SetMapper(tm)
    terrain.GetProperty().SetColor(0.33, 0.39, 0.27); terrain.GetProperty().SetAmbient(0.25)

    gr = vtkGLTFReader(); gr.SetFileName(os.path.join(bundle, "buildings.glb")); gr.Update()
    gf = vtkCompositeDataGeometryFilter(); gf.SetInputConnection(gr.GetOutputPort()); gf.Update()
    dec = vtkQuadricDecimation(); dec.SetInputData(gf.GetOutput()); dec.SetTargetReduction(0.65); dec.Update()
    bm = vtkPolyDataMapper(); bm.SetInputData(dec.GetOutput()); bm.SetStatic(1); bm.ScalarVisibilityOff()
    bld = vtkActor(); bld.SetMapper(bm)
    bp = bld.GetProperty(); bp.SetColor(0.72, 0.72, 0.77); bp.SetAmbient(0.4); bp.SetDiffuse(0.72)

    cs = vtkCubeSource(); cs.SetXLength(4.6); cs.SetYLength(2.0); cs.SetZLength(1.6); cs.Update()
    vm = vtkPolyDataMapper(); vm.SetInputConnection(cs.GetOutputPort())
    veh = vtkActor(); veh.SetMapper(vm)
    veh.GetProperty().SetColor(0.90, 0.13, 0.12); veh.GetProperty().SetAmbient(0.3); veh.GetProperty().SetDiffuse(0.8)

    def shaded_cone(h, rad, col, down=True):
        c = vtkConeSource(); c.SetHeight(h); c.SetRadius(rad); c.SetResolution(26)
        c.SetDirection(0, 0, -1 if down else 1); c.Update()
        n = vtkPolyDataNormals(); n.SetInputConnection(c.GetOutputPort()); n.Update()
        m = vtkPolyDataMapper(); m.SetInputConnection(n.GetOutputPort())
        a = vtkActor(); a.SetMapper(m); p = a.GetProperty()
        p.SetColor(*col); p.SetAmbient(0.22); p.SetDiffuse(0.85); p.SetSpecular(0.25); p.SetSpecularPower(18)
        return a

    veh_beacon = shaded_cone(24, 5.0, (0.98, 0.42, 0.10), down=True)
    goal_cols = [(0.20, 0.80, 0.35), (0.30, 0.65, 0.95), (0.95, 0.75, 0.15), (0.85, 0.35, 0.85)]
    goal_beacons = [shaded_cone(150, 9.0, goal_cols[i % len(goal_cols)], down=False) for i in range(len(goals))]

    ren = vtkRenderer(); ren.SetBackground(0.16, 0.19, 0.13); ren.SetBackground2(0.55, 0.68, 0.82); ren.GradientBackgroundOn()
    ren.AddActor(terrain); ren.AddActor(bld); ren.AddActor(veh); ren.AddActor(veh_beacon)
    for gi, gb in enumerate(goal_beacons):
        t = vtkTransform(); t.Translate(goals[gi][0], goals[gi][1], hsamp(*goals[gi]) + 74)
        gb.SetUserTransform(t); ren.AddActor(gb)
    sun = vtkLight(); sun.SetPosition(mnx - 500, mny - 500, 2500); sun.SetFocalPoint((mnx + mxx) / 2, (mny + mxy) / 2, 0)
    sun.SetIntensity(1.0); sun.SetLightTypeToSceneLight(); ren.AddLight(sun)
    rw = vtkRenderWindow(); rw.SetOffScreenRendering(1); rw.AddRenderer(ren); rw.SetSize(W, H)
    cam = ren.GetActiveCamera(); cam.SetClippingRange(1.0, 9000.0)

    def smooth(a, k=5):
        a = np.asarray(a, np.float32); o = a.copy()
        for i in range(len(a)):
            o[i] = a[max(0, i - k): i + k + 1].mean(0)
        return o

    # Pace by ARC LENGTH so speed is constant (fixes 'slow then fast'); a per-leg
    # sin ease decelerates into each goal and accelerates out. Total frames are set
    # to `minutes` so the film runs the requested length at a steady cruise.
    total_frames = int(minutes * 60 * fps)
    lens = []
    smoothed = []
    for gi, seg in legs:
        segS = smooth(seg); dd = np.linalg.norm(np.diff(segS, axis=0), axis=1)
        arcl = np.concatenate([[0.0], np.cumsum(dd)]); smoothed.append((gi, segS, arcl)); lens.append(float(arcl[-1]))
    tot = sum(lens) or 1.0
    seq = []
    for (gi, segS, arcl), L in zip(smoothed, lens):
        if L < 1.0:
            continue
        n = max(120, int(round(total_frames * L / tot)))
        u = np.linspace(0, 1, n); spd = 0.35 + 0.65 * np.sin(np.pi * u)
        cum = np.cumsum(spd); cum = cum / cum[-1] * L
        xs = np.interp(cum, arcl, segS[:, 0]); ys = np.interp(cum, arcl, segS[:, 1])
        for k in range(n):
            seq.append((float(xs[k]), float(ys[k]), gi))

    w2i = vtkWindowToImageFilter(); w2i.SetInput(rw)
    head = np.array([1.0, 0.0]); BACK, HT, AHEAD = 74.0, 52.0, 24.0
    prevp = np.array([seq[0][0], seq[0][1]])
    for fi, (px, py, ag) in enumerate(seq):
        p = np.array([px, py]); dvec = p - prevp; prevp = p; nn = np.linalg.norm(dvec)
        if nn > 1e-4:
            head = 0.6 * head + 0.4 * (dvec / nn); head /= (np.linalg.norm(head) + 1e-9)
        zt = hsamp(px, py) + 0.9
        tf = vtkTransform(); tf.Translate(px, py, zt + 0.8)
        tf.RotateZ(math.degrees(math.atan2(head[1], head[0]))); veh.SetUserTransform(tf)
        bt = vtkTransform(); bt.Translate(px, py, zt + 24.0); veh_beacon.SetUserTransform(bt)
        for gi, gb in enumerate(goal_beacons):
            gb.GetProperty().SetAmbient(0.62 if gi == ag else 0.22)  # active goal spire glows
        cam.SetPosition(px - head[0] * BACK, py - head[1] * BACK, zt + HT)
        cam.SetFocalPoint(px + head[0] * AHEAD, py + head[1] * AHEAD, zt - 2.0); cam.SetViewUp(0, 0, 1)
        rw.Render(); w2i.Modified(); w2i.Update()
        wr = vtkPNGWriter(); wr.SetFileName("%s/f_%05d.png" % (fr, fi)); wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", "%s/f_%%05d.png" % fr,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=%d:%d" % (W, H), out_mp4], check=True)
    return len(seq)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle"); ap.add_argument("checkpoint")
    ap.add_argument("--sdf", default=None, help="prebuilt nav_sdf.npz (else built from occupancy)")
    ap.add_argument("--minutes", type=float, default=3.0, help="target film length")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("-o", "--out", default="multigoal.mp4")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu"); meta = ck["meta"]
    model = sdf_nav.CoefMLP(); model.load_state_dict(ck["model_state_dict"]); model.eval()
    model_ref["m"] = model
    bounds = terrain_grid(os.path.join(args.bundle, "terrain.json"))[1]
    occ0 = building_occupancy(os.path.join(args.bundle, "buildings.glb"), bounds, 512, 512, inflate_m=0.0)
    if args.sdf and os.path.exists(args.sdf):
        dd = np.load(args.sdf); phi, nxg, nyg = dd["phi"], dd["normal_x"], dd["normal_y"]
    else:
        phi, nxg, nyg = sdf_nav.build_sdf(occ0, bounds, meta["scale"])
    field = sdf_nav.SDFField(phi, nxg, nyg, bounds, meta["center"], meta["scale"], device="cpu")
    kw = dict(rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"])

    start, goals = select_goals(field, meta, bounds)
    print("start %s  goals %s" % (np.round(start).tolist(), [np.round(g).tolist() for g in goals]))
    S = meta["scale"]; cx, cy = meta["center"]

    def w2n(p): return np.array([(p[0] - cx) * S, (p[1] - cy) * S], np.float32)

    o = torch.from_numpy(w2n(start)).unsqueeze(0); v = torch.zeros(1, 2); legs = []
    for gi, g in enumerate(goals):
        o, v, seg = _leg(field, model, meta, o, v, g, kw)
        pts = ([start.tolist()] if gi == 0 else []) + [p.tolist() for p in seg]
        legs.append((gi, np.asarray(pts, np.float32)))
    n = render_video(args.bundle, legs, goals, args.out, minutes=args.minutes, fps=args.fps)
    print("wrote %s (%d bytes, %d frames, %.0fs)" % (args.out, os.path.getsize(args.out), n, n / args.fps))


if __name__ == "__main__":
    main()
