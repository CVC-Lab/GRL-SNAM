#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E9. Qualitative Analyses for CoefEnergyNet

Generates visual artifacts that make the Hamiltonian story visible:
- Vector fields of the fused force and its components (Fg, Fbs, Fμ) at start/mid/end
- Streamlines / quivers over the local 2·d_hat window
- Trajectory overlays vs rigid & deformable A* baselines
- Time-series of alphas/beta/gamma, min-clearance, force magnitudes
- Per-episode safety stats (barrier/tube) written to JSON

Usage:
  python qual_coef_e9.py --ckpt checkpoints/best.pt --case case1-tight --alpha_mode weight
"""

import os, json, argparse
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v3 as iio

# ---- Project imports (your codebase)
from train_coef_energy import CoefEnergyNet
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi

# ---------- small utils ----------
def mkdir(p): os.makedirs(p, exist_ok=True); return p

def compute_workspace_bounds(C, R, margin=1.0):
    if len(C) == 0: return -5.0, 5.0, -5.0, 5.0
    cx, cy = C[:,0], C[:,1]
    xmin = float(np.min(cx - R) - margin)
    xmax = float(np.max(cx + R) + margin)
    ymin = float(np.min(cy - R) - margin)
    ymax = float(np.max(cy + R) + margin)
    return xmin, xmax, ymin, ymax

def path_length(path_xy):
    if len(path_xy) < 2: return 0.0
    diffs = np.diff(np.asarray(path_xy), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def clearance_to_circles(xy, C, R):
    if len(C) == 0: return np.inf
    d = np.linalg.norm(C - xy[None,:], axis=1) - R
    return float(np.min(d))

def clearance_profile(path_xy, C, R):
    return [clearance_to_circles(xy, C, R) for xy in path_xy]

def success_reached(last_center, goal_xy, tol):
    return np.linalg.norm(np.asarray(last_center) - np.asarray(goal_xy)) <= tol

def draw_snapshot(png_path, world, path_so_far, start, goal):
    """Write a lightweight frame PNG showing obstacles + path-so-far."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.2, 4.8))
    th = np.linspace(0, 2*np.pi, 128)
    for (cx, cy), r in zip(world.C_np, world.R_np):
        plt.plot(cx + r*np.cos(th), cy + r*np.sin(th), 'k-', lw=1.0, alpha=0.9)
    if len(path_so_far) > 0:
        P = np.asarray(path_so_far)
        plt.plot(P[:,0], P[:,1], '-', lw=2.0, color='tab:orange', alpha=0.95)
        plt.scatter([P[-1,0]], [P[-1,1]], s=25, color='tab:orange')
    plt.scatter([start[0]], [start[1]], s=40, marker='s', color='k')
    plt.scatter([goal[0]],  [goal[1]],  s=70, marker='*', color='crimson')
    plt.axis('equal'); plt.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=140)
    plt.close()

# ---------- A* helpers (same as your eval script, trimmed) ----------
def world_to_ij(xy, bounds, res):
    xmin, xmax, ymin, ymax = bounds
    j = int(np.round((xy[0] - xmin)/res))
    i = int(np.round((xy[1] - ymin)/res))
    return i, j

def ij_to_world(i, j, bounds, res):
    xmin, xmax, ymin, ymax = bounds
    x = xmin + j*res
    y = ymin + i*res
    return np.array([x, y], dtype=np.float32)

def rasterize_inflated_obstacles(C, R, inflation, bounds, res):
    xmin, xmax, ymin, ymax = bounds
    W = int(np.ceil((xmax - xmin)/res)) + 1
    H = int(np.ceil((ymax - ymin)/res)) + 1
    occ = np.zeros((H, W), dtype=bool)
    if len(C) == 0: return occ, H, W
    yy = np.linspace(ymin, ymax, H)
    xx = np.linspace(xmin, xmax, W)
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    for (cx, cy), r in zip(C, R):
        mask = (X - cx)**2 + (Y - cy)**2 <= (r + inflation)**2
        occ |= mask
    return occ, H, W

def a_star_path(occ, bounds, res, start_xy, goal_xy):
    from heapq import heappush, heappop
    H, W = occ.shape
    si, sj = world_to_ij(start_xy, bounds, res)
    gi, gj = world_to_ij(goal_xy, bounds, res)
    if not (0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): return np.inf, []
    if occ[si, sj] or occ[gi, gj]: return np.inf, []
    nbrs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
            (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]
    g = np.full((H,W), np.inf, dtype=np.float64)
    came = np.full((H,W,2), -1, dtype=np.int32)
    def h(i,j): return np.hypot(i-gi, j-gj)
    pq = []
    g[si,sj] = 0.0
    heappush(pq, (h(si,sj), (si,sj)))
    visited = np.zeros_like(occ, dtype=bool)
    while pq:
        _, (i,j) = heappop(pq)
        if visited[i,j]: continue
        visited[i,j] = True
        if (i,j) == (gi,gj): break
        for di,dj,c in nbrs:
            ni, nj = i+di, j+dj
            if not (0<=ni<H and 0<=nj<W): continue
            if occ[ni,nj]: continue
            tentative = g[i,j] + c
            if tentative < g[ni,nj]:
                g[ni,nj] = tentative
                came[ni,nj] = [i,j]
                heappush(pq, (tentative + h(ni,nj), (ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf, []
    # reconstruct
    path_ij = []
    ci, cj = gi, gj
    path_ij.append((ci,cj))
    while not (ci == si and cj == sj):
        pi, pj = came[ci, cj]
        if pi < 0: break
        path_ij.append((pi,pj)); ci, cj = pi, pj
    path_ij.reverse()
    # to world
    length_steps = 0.0
    for k in range(1,len(path_ij)):
        i0,j0 = path_ij[k-1]; i1,j1 = path_ij[k]
        length_steps += np.hypot(i1-i0, j1-j0)
    length_m = length_steps * res
    path_xy = np.stack([ij_to_world(i,j,bounds,res) for (i,j) in path_ij], axis=0) if len(path_ij) else []
    return float(length_m), path_xy

def clearance_field_on_grid(C, R, bounds, res):
    xmin, xmax, ymin, ymax = bounds
    W = int(np.ceil((xmax - xmin)/res)) + 1
    H = int(np.ceil((ymax - ymin)/res)) + 1
    yy = np.linspace(ymin, ymax, H)
    xx = np.linspace(xmin, xmax, W)
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    if len(C) == 0:
        return np.full((H,W), np.inf, dtype=np.float32)
    clear = np.full((H,W), np.inf, dtype=np.float32)
    for (cx, cy), r in zip(C, R):
        d = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32) - float(r)
        clear = np.minimum(clear, d)
    return clear

def deformable_a_star_path(clearance, bounds, res, start_xy, goal_xy,
                           r_rest, r_min, lam=5.0, beta=0.5, eps=1e-3):
    from heapq import heappush, heappop
    H, W = clearance.shape
    si, sj = world_to_ij(start_xy, bounds, res)
    gi, gj = world_to_ij(goal_xy, bounds, res)
    if not (0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): return np.inf, []
    feas = clearance >= r_min
    if not (feas[si, sj] and feas[gi, gj]): return np.inf, []
    nbrs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
            (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]
    def h(i,j): return np.hypot(i-gi, j-gj) * res
    def pen(c):
        alpha = np.maximum(0.0, (r_rest / (c + eps)) - 1.0)
        return lam * (alpha**2)
    g = np.full((H,W), np.inf, dtype=np.float64)
    came = np.full((H,W,2), -1, dtype=np.int32)
    pq = []
    g[si,sj] = 0.0
    heappush(pq, (h(si,sj), (si,sj)))
    visited = np.zeros((H,W), dtype=bool)
    while pq:
        _, (i,j) = heappop(pq)
        if visited[i,j]: continue
        visited[i,j] = True
        if (i,j) == (gi,gj): break
        if not feas[i,j]: continue
        pij = pen(clearance[i,j])
        for di,dj,cstep in nbrs:
            ni, nj = i+di, j+dj
            if not (0<=ni<H and 0<=nj<W): continue
            if not feas[ni,nj]: continue
            pnn = pen(clearance[ni,nj])
            base = cstep * res
            deform = beta * 0.5*(pij + pnn) * base
            tentative = g[i,j] + (base + deform)
            if tentative < g[ni,nj]:
                g[ni,nj] = tentative
                came[ni,nj] = [i,j]
                heappush(pq, (tentative + h(ni,nj), (ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf, []
    path_ij = []
    ci, cj = gi, gj
    path_ij.append((ci,cj))
    while not (ci == si and cj == sj):
        pi, pj = came[ci, cj]
        if pi < 0: break
        path_ij.append((pi,pj)); ci, cj = pi, pj
    path_ij.reverse()
    length_m = 0.0
    for k in range(1,len(path_ij)):
        i0,j0 = path_ij[k-1]; i1,j1 = path_ij[k]
        length_m += np.hypot(i1-i0, j1-j0) * res
    path_xy = np.stack([ij_to_world(i,j,bounds,res) for (i,j) in path_ij], axis=0) if len(path_ij) else []
    return float(length_m), path_xy

# ---------- Model I/O ----------
def build_local_feats(o_w: np.ndarray, goal_w: np.ndarray, C_w: np.ndarray, R_w: np.ndarray, W_w: np.ndarray):
    o = torch.as_tensor(o_w, dtype=torch.float32)
    g = torch.as_tensor(goal_w, dtype=torch.float32)
    C = torch.as_tensor(C_w, dtype=torch.float32) if C_w.size else torch.zeros(0,2, dtype=torch.float32)
    R = torch.as_tensor(R_w, dtype=torch.float32) if R_w.size else torch.zeros(0, dtype=torch.float32)
    W = torch.as_tensor(W_w, dtype=torch.float32) if W_w.size else torch.zeros(0, dtype=torch.float32)
    if C.ndim == 1: C = C.reshape(0,2)
    dg = (g - o); gdist = torch.linalg.norm(dg).unsqueeze(0)
    goal_feats = torch.stack([dg[0], dg[1], gdist[0], torch.tensor(1.0)], dim=0).unsqueeze(0)  # [1,4]
    if C.shape[0] == 0:
        obs_feats = torch.zeros(1,0,6, dtype=torch.float32)
    else:
        dxdy = (g.unsqueeze(0) - C)  # (N,2)
        obs_feats = torch.cat([C, R.unsqueeze(-1), W.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)  # [1,N,6]
    return obs_feats, goal_feats

def map_alpha_to_world(W_in: np.ndarray, R_in: np.ndarray, alphas: np.ndarray, mode: str, k_rad: float = 0.05):
    al = np.maximum(alphas, 0.0)
    W_out = W_in.copy(); R_out = R_in.copy()
    if mode in ("weight", "both") and al.size: W_out = W_out * al
    if mode in ("radius", "both") and al.size: R_out = R_out + k_rad * al
    return W_out, R_out

# ---------- Force field (qualitative proxy) ----------
def F_goal(xy: np.ndarray, goal: np.ndarray, beta: float) -> np.ndarray:
    v = goal - xy
    n = np.linalg.norm(v) + 1e-6
    return beta * (v / n)

def F_barrier(xy: np.ndarray, C: np.ndarray, R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Smooth repulsive field from disks; decays ~ 1/r^2 outside the boundary.
    """
    if len(C) == 0: return np.zeros(2, dtype=np.float32)
    F = np.zeros(2, dtype=np.float32)
    for (cx,cy), r, w in zip(C, R, W):
        dvec = xy - np.array([cx,cy], dtype=np.float32)
        dist = np.linalg.norm(dvec) + 1e-6
        gap = max(dist - r, 1e-3)
        dirn = dvec / dist
        # quadratic near boundary, 1/r^2 far
        mag = w * (1.0/(gap*gap))
        F += dirn * mag
    return F

def F_damping(vel: np.ndarray, gamma: float) -> np.ndarray:
    return -gamma * vel

def render_local_field(figpath: str, center_xy: np.ndarray, goal: np.ndarray,
                       C: np.ndarray, R: np.ndarray, W: np.ndarray,
                       beta: float, gamma: float, field_half: float = 1.0,
                       grid_n: int = 21):
    """
    Renders quiver plots for Fg, Fbs, and fused F = Fg + Fbs (no velocity term for static snapshot).
    """
    xs = np.linspace(center_xy[0]-field_half, center_xy[0]+field_half, grid_n)
    ys = np.linspace(center_xy[1]-field_half, center_xy[1]+field_half, grid_n)
    X, Y = np.meshgrid(xs, ys)
    Fg = np.zeros((grid_n, grid_n, 2), dtype=np.float32)
    Fb = np.zeros((grid_n, grid_n, 2), dtype=np.float32)
    Ff = np.zeros((grid_n, grid_n, 2), dtype=np.float32)

    for i in range(grid_n):
        for j in range(grid_n):
            p = np.array([X[i,j], Y[i,j]], dtype=np.float32)
            fg = F_goal(p, goal, beta)
            fb = F_barrier(p, C, R, W)
            Fg[i,j] = fg
            Fb[i,j] = fb
            Ff[i,j] = fg + fb

    mag = np.linalg.norm(Ff, axis=-1) + 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    titles = [r"$F_g$", r"$F_{bs}$", r"$F = F_g + F_{bs}$"]
    fields = [Fg, Fb, Ff]
    for ax, title, Fcomp in zip(axes, titles, fields):
        ax.quiver(X, Y, Fcomp[...,0], Fcomp[...,1], color="tab:gray", width=0.002, scale=35)
        # obstacles
        th = np.linspace(0, 2*np.pi, 128)
        for (cx,cy), r in zip(C, R):
            ax.plot(cx + r*np.cos(th), cy + r*np.sin(th), 'k-', lw=1.0, alpha=0.8)
        ax.plot([goal[0]], [goal[1]], '*', color='crimson', ms=10)
        ax.plot([center_xy[0]],[center_xy[1]], 's', color='tab:blue', ms=6)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
    axes[2].set_title(titles[2] + f"   (β={beta:.2f}, γ={gamma:.2f})")
    plt.savefig(figpath, dpi=180)
    plt.close(fig)

# ---------- overlay ----------
def plot_paths_overlay(outdir, world, our_path, path_star_rigid, path_star_deform, start, goal):
    plt.figure(figsize=(5.8,5.2))
    th = np.linspace(0, 2*np.pi, 128)
    for (cx,cy), r in zip(world.C_np, world.R_np):
        plt.plot(cx + r*np.cos(th), cy + r*np.sin(th), lw=1.2, alpha=0.8, color='k')
    if path_star_rigid is None or len(path_star_rigid) == 0:
        plt.text(start[0], start[1], "No Rigid A*", color="tab:blue", fontsize=9)
    else:
        plt.plot(path_star_rigid[:,0], path_star_rigid[:,1], '--', lw=2, label='Rigid A*', alpha=0.9, color="tab:blue")
    if path_star_deform is None or len(path_star_deform) == 0:
        pass
    else:
        plt.plot(path_star_deform[:,0], path_star_deform[:,1], '-.', lw=2, label='Deformable A*', alpha=0.9, color="tab:green")
    if len(our_path):
        P = np.asarray(our_path)
        plt.plot(P[:,0], P[:,1], '-', lw=2.3, label='CoefEnergyNet', alpha=0.95, color="tab:orange")
    plt.scatter([start[0]],[start[1]], s=40, marker='s', label='Start', color='k')
    plt.scatter([goal[0]],[goal[1]], s=70, marker='*', label='Goal', color='crimson')
    plt.axis('equal'); plt.legend(frameon=False, loc='best')
    plt.title("Trajectory overlay vs baselines")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "trajectory_overlay.png"), dpi=200)
    plt.close()

# ---------- time-series panels ----------
def plot_timeseries(outdir, taxis, min_clear, Fg_mag, Fbs_mag, F_total_mag, beta_hist, gamma_hist, alpha_stats):
    plt.figure(figsize=(11,7.5))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(taxis, min_clear, label="min-clearance")
    ax1.axhline(0.0, color='r', ls='--', lw=1.0, label="collision threshold")
    ax1.set_ylabel("Clearance (m)"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = plt.subplot(3,1,2)
    ax2.plot(taxis, Fg_mag, label="|Fg|")
    ax2.plot(taxis, Fbs_mag, label="|Fbs|")
    ax2.plot(taxis, F_total_mag, label="|F_total|", lw=2)
    ax2.set_ylabel("Force magnitude"); ax2.legend(); ax2.grid(alpha=0.3)

    ax3 = plt.subplot(3,1,3)
    ax3.plot(taxis, beta_hist, label="beta (goal)")
    ax3.plot(taxis, gamma_hist, label="gamma (damping)")
    if alpha_stats:
        ax3.plot(taxis, alpha_stats["mean"], label="alpha mean", ls="--")
        ax3.fill_between(taxis, alpha_stats["min"], alpha_stats["max"], alpha=0.15, label="alpha range")
    ax3.set_xlabel("time step"); ax3.set_ylabel("coefficients"); ax3.legend(); ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "timeseries_forces_clearance_coeffs.png"), dpi=220)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Qualitative analyses for CoefEnergyNet (E9)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--case", type=str, default="case1-tight")
    ap.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","none"])
    ap.add_argument("--seed", type=int, default=2312)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--k_rad", type=float, default=0.05)
    ap.add_argument("--gif_stride", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = CoefEnergyNet().to(device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state); model.eval()

    # build env
    cfg = gen.GenCfg(); cfg.seed = args.seed; gen.set_all_seeds(cfg.seed)
    C, R, W = (gen.sample_obstacles_case1_tight(cfg) if args.case.startswith("case1")
               else gen.sample_obstacles_case2_harder(cfg))
    world = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)
    r_rest = getattr(cfg, "radius", cfg.d_hat)
    r_min  = getattr(cfg, "radius_min", 0.3 * r_rest)
    bounds = compute_workspace_bounds(world.C_np, world.R_np, margin=2.0*max(r_rest, cfg.d_hat))
    res    = max(r_rest, cfg.d_hat) * 0.5

    # A* baselines
    occ_rigid, _, _ = rasterize_inflated_obstacles(world.C_np, world.R_np, r_rest, bounds, res)
    L_star_rigid, path_star_rigid = a_star_path(occ_rigid, bounds, res, cfg.start, cfg.goal)
    clear_grid = clearance_field_on_grid(world.C_np, world.R_np, bounds, res)
    L_star_def, path_star_def = deformable_a_star_path(clear_grid, bounds, res, cfg.start, cfg.goal,
                                                       r_rest=r_rest, r_min=r_min, lam=5.0, beta=0.6)

    # bookkeeping
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = mkdir(f"snaps_qual_e9/{stamp}")

    # planner
    k_bulk  = getattr(cfg, "k_bulk", 1.0)
    gamma_s = getattr(cfg, "gamma_s", 0.1)
    planner = gen.planner_from_cfg(cfg, world, k_bulk, gamma_s, cfg.d_hat, getattr(cfg, "radius", cfg.d_hat))

    # rollout & record
    frames_png = []
    path = []
    beta_hist, gamma_hist = [], []
    alpha_min, alpha_max, alpha_mean = [], [], []
    Fg_mag, Fbs_mag, F_total_mag = [], [], []
    min_clear_hist = []

    frames_png = []
    def capture_frame(t_idx):
        png_path = os.path.join(outdir, f"frame_{t_idx:04d}.png")
        draw_snapshot(png_path, world, path, cfg.start, cfg.goal)
        frames_png.append(png_path)

    capture_frame(0)  

    for t in range(args.steps):
        sys = planner.sys
        o_w = sys.o.detach().cpu().numpy()
        Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)

        obs_feats, goal_feats = build_local_feats(o_w, cfg.goal, Cw, Rw, Ww)
        obs_mask = (torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
                    if obs_feats.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device))

        with torch.no_grad():
            alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))

        al_np  = (alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw))
        beta_f = float(beta.squeeze(0).item())
        gamma_f= float(gamma.squeeze(0).item())

        # apply to world slice (for visualization + dynamics)
        W_adj, R_adj = Ww.copy(), Rw.copy()
        if args.alpha_mode != "none":
            W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, args.alpha_mode, args.k_rad)
        world_step = ssi.WorldObstacles(Cw, R_adj, W_adj, d_hat=cfg.d_hat)

        # (qual) instantaneous forces at center
        fg = F_goal(o_w, cfg.goal, beta_f)
        fbs= F_barrier(o_w, Cw, R_adj, W_adj)
        ft = fg + fbs
        Fg_mag.append(np.linalg.norm(fg))
        Fbs_mag.append(np.linalg.norm(fbs))
        F_total_mag.append(np.linalg.norm(ft))

        # dynamics params
        planner.stage_field.w_goal = max(0.0, beta_f)
        planner.sys.gamma_o = max(0.0, gamma_f)

        info = planner.step(cfg.dt, world_step)
        center_xy = np.asarray(info["center"], float)
        path.append(center_xy)

        # stats
        min_clear = clearance_to_circles(center_xy, world.C_np, world.R_np)
        min_clear_hist.append(min_clear)
        beta_hist.append(beta_f); gamma_hist.append(gamma_f)
        if al_np.size:
            alpha_min.append(float(np.min(al_np)))
            alpha_max.append(float(np.max(al_np)))
            alpha_mean.append(float(np.mean(al_np)))
        else:
            alpha_min.append(0.0); alpha_max.append(0.0); alpha_mean.append(0.0)

        # sparse frames for quick GIF
        if t % max(1, args.gif_stride) == 0:
            capture_frame(t+1)

        if np.linalg.norm(center_xy - cfg.goal) <= cfg.goal_tol:
            break

    # vector-field snapshots at start/mid/end
    if len(path) > 0:
        idxs = [0, len(path)//2, len(path)-1]
        labels = ["start","mid","end"]
        # To reconstruct coefficients at those instants, we re-query the model
        for idx, tag in zip(idxs, labels):
            # local slice at that pose
            xy = path[idx]
            Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)
            obs_feats, goal_feats = build_local_feats(xy, cfg.goal, Cw, Rw, Ww)
            mask = None
            if obs_feats.shape[1]:
                mask = torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)

            with torch.no_grad():
                alphas, beta, gamma = model(obs_feats.to(device), mask, goal_feats.to(device))

            al_np  = (alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw))
            beta_f = float(beta.squeeze(0).item()); gamma_f = float(gamma.squeeze(0).item())
            W_adj, R_adj = Ww.copy(), Rw.copy()
            if args.alpha_mode != "none":
                W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, args.alpha_mode, args.k_rad)
            figpath = os.path.join(outdir, f"field_{tag}.png")
            render_local_field(figpath, xy, cfg.goal, Cw, R_adj, W_adj, beta_f, gamma_f,
                               field_half=getattr(cfg, "d_hat", 0.5), grid_n=25)

    # baselines overlay
    plot_paths_overlay(outdir, world, path, path_star_rigid, path_star_def, cfg.start, cfg.goal)

    # GIF
    try:
        gif_frames = [iio.imread(p) for p in frames_png]
        iio.imwrite(os.path.join(outdir,"rollout.gif"), gif_frames, loop=0, fps=10)
    except Exception as e:
        print(f"[WARN] GIF failed: {e}")

    # time-series plots
    taxis = np.arange(len(path))
    alpha_stats = {"min": np.array(alpha_min), "max": np.array(alpha_max), "mean": np.array(alpha_mean)}
    plot_timeseries(outdir, taxis, np.array(min_clear_hist),
                    np.array(Fg_mag), np.array(Fbs_mag), np.array(F_total_mag),
                    np.array(beta_hist), np.array(gamma_hist), alpha_stats)

    # per-episode safety summary
    min_clear = float(np.min(min_clear_hist)) if len(min_clear_hist) else np.inf
    L_exec = path_length(path)
    reached = success_reached(path[-1] if len(path) else cfg.start, cfg.goal, cfg.goal_tol)
    # choose ref A* (prefer deformable)
    L_ref = L_star_def if np.isfinite(L_star_def) else (L_star_rigid if np.isfinite(L_star_rigid) else None)
    detour = (L_exec / L_ref) if (L_ref is not None and L_ref > 0) else np.nan
    spl    = (L_ref / max(L_ref, L_exec)) if (reached and L_ref is not None and L_ref > 0) else 0.0

    episode_json = {
        "ckpt": args.ckpt,
        "case": args.case,
        "alpha_mode": args.alpha_mode,
        "steps_taken": int(len(path)),
        "success": bool(reached),
        "path_length": float(L_exec),
        "astar_rigid_L": float(L_star_rigid) if np.isfinite(L_star_rigid) else float("inf"),
        "astar_deform_L": float(L_star_def) if np.isfinite(L_star_def) else float("inf"),
        "detour_vs_ref": float(detour) if np.isfinite(detour) else float("nan"),
        "SPL_vs_ref": float(spl),
        "min_clearance": float(min_clear) if np.isfinite(min_clear) else float("nan"),
        "series": {
            "min_clearance": list(map(float, min_clear_hist)),
            "Fg_mag": list(map(float, Fg_mag)),
            "Fbs_mag": list(map(float, Fbs_mag)),
            "F_total_mag": list(map(float, F_total_mag)),
            "beta": list(map(float, beta_hist)),
            "gamma": list(map(float, gamma_hist)),
            "alpha_min": list(map(float, alpha_min)),
            "alpha_max": list(map(float, alpha_max)),
            "alpha_mean": list(map(float, alpha_mean)),
        },
        "artifacts": {
            "overlay_png": os.path.join(outdir, "trajectory_overlay.png"),
            "field_start_png": os.path.join(outdir, "field_start.png"),
            "field_mid_png": os.path.join(outdir, "field_mid.png"),
            "field_end_png": os.path.join(outdir, "field_end.png"),
            "timeseries_png": os.path.join(outdir, "timeseries_forces_clearance_coeffs.png"),
            "gif": os.path.join(outdir, "rollout.gif"),
        }
    }
    with open(os.path.join(outdir, "qual_episode_summary.json"), "w") as f:
        json.dump(episode_json, f, indent=2)

    print("\n[E9] Qualitative artifacts saved to:", outdir)
    print(f"  success={episode_json['success']}  SPL={episode_json['SPL_vs_ref']:.3f}  detour={episode_json['detour_vs_ref']:.3f}")
    print("  Overlay:", episode_json["artifacts"]["overlay_png"])
    print("  Field snapshots:", episode_json["artifacts"]["field_start_png"], episode_json["artifacts"]["field_mid_png"], episode_json["artifacts"]["field_end_png"])
    print("  Timeseries:", episode_json["artifacts"]["timeseries_png"])

if __name__ == "__main__":
    main()
