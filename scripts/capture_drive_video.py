"""Capture a learned SDF drive to an mp4: run the end-to-end navigator (with the
wall-follow local-minimum escape) from START to GOAL on a real scene, then render
the ACTUAL 3-D scene — terrain relief + glTF buildings + the vehicle — from a chase
camera OFFSCREEN via VTK (the same engine cvcGL/VolRover3 use), and ffmpeg the
frames into a video.

This is how the demo video is produced without a live window: same geometry, same
renderer, a scripted chase camera. Needs the volrover env (pycvc_gl/VTK), torch,
the GRL-SNAM repo, a scene bundle, a trained SDF checkpoint, and ffmpeg.

    python scripts/capture_drive_video.py <bundle> checkpoints/coef_sdf.pt \
        --start -361 114 --goal 185 50 -o drive.mp4
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess

import numpy as np
import torch

import sdf_nav
from pycvc_gl.scenes import building_occupancy, terrain_grid


def navigate(field, occ0, bounds, model, meta, start, goal, maxst=3000):
    """End-to-end SDF nav with wall-follow escape; returns the world trajectory."""
    S = meta["scale"]; cx, cy = meta["center"]; rr = meta["rr"]
    mnx, mny, mxx, mxy = bounds; ny, nx = occ0.shape
    kw = dict(rr=rr, d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"])

    def w2n(p): return np.array([(p[0] - cx) * S, (p[1] - cy) * S], np.float32)

    def n2w(o): return np.array([o[0] / S + cx, o[1] / S + cy], np.float32)

    def nrm_at(on):
        _, n = field.sample(torch.from_numpy(on).unsqueeze(0).float()); return n[0].numpy()

    o = torch.from_numpy(w2n(start)).unsqueeze(0); v = torch.zeros(1, 2); gn = w2n(goal)
    tr = [np.asarray(start, np.float32)]; best = 1e9; stall = 0; mode = "seek"; turn = 1.0; dhit = 0.0
    with torch.no_grad():
        for _ in range(maxst):
            p = o[0].numpy(); dg = float(np.linalg.norm(p - gn)); gdir = (gn - p) / (dg + 1e-6)
            if dg < best - 1e-3:
                best = dg; stall = 0
            else:
                stall += 1
            if mode == "seek" and stall > 70:
                t = np.array([-nrm_at(p)[1], nrm_at(p)[0]], np.float32)
                turn = 1.0 if np.dot(t, gdir) >= 0 else -1.0; dhit = dg; mode = "wall"; stall = 0
            if mode == "wall":
                n = nrm_at(p); t = turn * np.array([-n[1], n[0]], np.float32)
                carrot = (p + (0.6 * t + 0.4 * n) * 1.6).astype(np.float32)
                if dg < dhit - 1.2 or stall > 240:
                    mode = "seek"; best = dg; stall = 0
            else:
                carrot = (p + gdir * min(1.8, dg)).astype(np.float32)
            al, be, ga = model(sdf_nav.coef_feats(field, o, torch.from_numpy(carrot).unsqueeze(0)))
            o, v, _ = sdf_nav.sdf_rollout(field, o, v, torch.from_numpy(carrot).unsqueeze(0), al, be, ga, 1,
                                         nsub=meta["nsub"], **kw)
            tr.append(n2w(o[0].numpy()));
            if dg < 0.4:
                break
    return np.asarray(tr)


def render_video(bundle, traj, start, goal, out_mp4, frames=170, size=(960, 540)):
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
    from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkConeSource
    from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
    from vtkmodules.vtkIOGeometry import vtkGLTFReader
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkRenderingCore import (vtkRenderer, vtkRenderWindow, vtkActor, vtkPolyDataMapper,
                                             vtkWindowToImageFilter, vtkLight)
    import json
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
            v = r * cols + c
            for cell in ((v, v + 1, v + cols), (v + 1, v + cols + 1, v + cols)):
                tris.InsertNextCell(3); [tris.InsertCellPoint(i) for i in cell]
    tpd = vtkPolyData(); tpd.SetPoints(pts); tpd.SetPolys(tris)
    tn = vtkPolyDataNormals(); tn.SetInputData(tpd); tn.Update()
    tm = vtkPolyDataMapper(); tm.SetInputConnection(tn.GetOutputPort()); tm.ScalarVisibilityOff()
    terrain = vtkActor(); terrain.SetMapper(tm)
    terrain.GetProperty().SetColor(0.33, 0.39, 0.27); terrain.GetProperty().SetAmbient(0.28)

    gr = vtkGLTFReader(); gr.SetFileName(os.path.join(bundle, "buildings.glb")); gr.Update()
    gf = vtkCompositeDataGeometryFilter(); gf.SetInputConnection(gr.GetOutputPort()); gf.Update()
    bm = vtkPolyDataMapper(); bm.SetInputData(gf.GetOutput()); bm.SetStatic(1); bm.ScalarVisibilityOff()
    bld = vtkActor(); bld.SetMapper(bm)
    bp = bld.GetProperty(); bp.SetColor(0.72, 0.72, 0.77); bp.SetAmbient(0.42); bp.SetDiffuse(0.72)

    cs = vtkCubeSource(); cs.SetXLength(4.6); cs.SetYLength(2.0); cs.SetZLength(1.6); cs.Update()
    vm = vtkPolyDataMapper(); vm.SetInputConnection(cs.GetOutputPort())
    veh = vtkActor(); veh.SetMapper(vm); veh.GetProperty().SetColor(0.90, 0.13, 0.12); veh.GetProperty().SetAmbient(0.4)
    cn = vtkConeSource(); cn.SetHeight(16); cn.SetRadius(3.2); cn.SetResolution(20); cn.SetDirection(0, 0, -1); cn.Update()
    bcm = vtkPolyDataMapper(); bcm.SetInputConnection(cn.GetOutputPort())
    beacon = vtkActor(); beacon.SetMapper(bcm); beacon.GetProperty().SetColor(0.98, 0.32, 0.12); beacon.GetProperty().SetAmbient(0.9)

    def pillar(x, y, col):
        c = vtkCubeSource(); c.SetXLength(6); c.SetYLength(6); c.SetZLength(60); c.Update()
        m = vtkPolyDataMapper(); m.SetInputConnection(c.GetOutputPort())
        a = vtkActor(); a.SetMapper(m); a.GetProperty().SetColor(*col); a.GetProperty().SetOpacity(0.55)
        tf = vtkTransform(); tf.Translate(x, y, hsamp(x, y) + 30); a.SetUserTransform(tf); return a

    ren = vtkRenderer(); ren.SetBackground(0.16, 0.19, 0.13); ren.SetBackground2(0.55, 0.68, 0.82); ren.GradientBackgroundOn()
    for a in (terrain, bld, veh, beacon, pillar(*start, (0.15, 0.85, 0.25)), pillar(*goal, (0.95, 0.78, 0.10))):
        ren.AddActor(a)
    sun = vtkLight(); sun.SetPosition(mnx, mny, 2500); sun.SetFocalPoint((mnx + mxx) / 2, (mny + mxy) / 2, 0)
    sun.SetIntensity(0.9); sun.SetLightTypeToSceneLight(); ren.AddLight(sun)
    rw = vtkRenderWindow(); rw.SetOffScreenRendering(1); rw.AddRenderer(ren); rw.SetSize(W, H)
    cam = ren.GetActiveCamera(); cam.SetClippingRange(1.0, 9000.0)

    T = traj.copy()
    for i in range(len(T)):
        lo = max(0, i - 4); T[i] = traj[lo:i + 5].mean(0)
    idx = np.linspace(0, len(T) - 2, frames).astype(int)
    w2i = vtkWindowToImageFilter(); w2i.SetInput(rw)
    head = np.array([1.0, 0.0]); BACK, HT, AHEAD = 74.0, 52.0, 24.0
    for fi, i in enumerate(idx):
        p = T[i]; dvec = T[min(i + 3, len(T) - 1)] - p; nn = np.linalg.norm(dvec)
        if nn > 1e-3:
            head = 0.7 * head + 0.3 * (dvec / nn); head /= (np.linalg.norm(head) + 1e-9)
        z = hsamp(p[0], p[1]) + 0.9
        tf = vtkTransform(); tf.Translate(p[0], p[1], z + 0.8)
        tf.RotateZ(math.degrees(math.atan2(head[1], head[0]))); veh.SetUserTransform(tf)
        bt = vtkTransform(); bt.Translate(p[0], p[1], z + 20.0); beacon.SetUserTransform(bt)
        cam.SetPosition(p[0] - head[0] * BACK, p[1] - head[1] * BACK, z + HT)
        cam.SetFocalPoint(p[0] + head[0] * AHEAD, p[1] + head[1] * AHEAD, z - 2.0); cam.SetViewUp(0, 0, 1)
        rw.Render(); w2i.Modified(); w2i.Update()
        wr = vtkPNGWriter(); wr.SetFileName("%s/f_%04d.png" % (fr, fi)); wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
    subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", "%s/f_%%04d.png" % fr,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=%d:%d" % (W, H), out_mp4], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle"); ap.add_argument("checkpoint")
    ap.add_argument("--sdf", default=None, help="prebuilt nav_sdf.npz (else built from occupancy)")
    ap.add_argument("--start", type=float, nargs=2, required=True)
    ap.add_argument("--goal", type=float, nargs=2, required=True)
    ap.add_argument("--frames", type=int, default=170)
    ap.add_argument("-o", "--out", default="drive.mp4")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu"); meta = ck["meta"]
    model = sdf_nav.CoefMLP(); model.load_state_dict(ck["model_state_dict"]); model.eval()
    bounds = terrain_grid(os.path.join(args.bundle, "terrain.json"))[1]
    occ0 = building_occupancy(os.path.join(args.bundle, "buildings.glb"), bounds, 512, 512, inflate_m=0.0)
    if args.sdf and os.path.exists(args.sdf):
        d = np.load(args.sdf); phi, nxg, nyg = d["phi"], d["normal_x"], d["normal_y"]
    else:
        phi, nxg, nyg = sdf_nav.build_sdf(occ0, bounds, meta["scale"])
    field = sdf_nav.SDFField(phi, nxg, nyg, bounds, meta["center"], meta["scale"], device="cpu")
    traj = navigate(field, occ0, bounds, model, meta, np.asarray(args.start), np.asarray(args.goal))
    print("navigated %d steps; rendering %d frames -> %s" % (len(traj), args.frames, args.out))
    render_video(args.bundle, traj, args.start, args.goal, args.out, frames=args.frames)
    print("wrote %s (%d bytes)" % (args.out, os.path.getsize(args.out)))


if __name__ == "__main__":
    main()
