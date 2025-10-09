#!/usr/bin/env python3
"""
Experiment 1: Main comparison on Test-ID and Test-OOD
Methods: GRL-SNAM (CoefEnergyNet), Rigid A*, Deformable A*, DWA/CBF-style reactive

Outputs:
- per-episode CSV/JSON
- aggregate JSON (per split + combined)
- plots: bars/violins for Success/SPL/Detour, safety, Pareto (SPL vs clearance),
         method trajectory overlays vs both A* baselines.

Usage:
python eval_exp1_main_comparison.py \
  --ckpt checkpoints/best.pt \
  --case_id case1-tight --case_ood case2-dense \
  --n_envs 10 --n_trials 8 --alpha_mode weight
"""

import os, json, argparse, csv
from datetime import datetime
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt

# ==== project imports (unchanged) ====
from train_coef_energy import CoefEnergyNet
from src.utils.online_stage_manager import StageManagerOnline
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi
from eval_coef_energy import HistSecantController
import re

def mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p

def safe_name(s: str) -> str:
    # keep letters, digits, dot, underscore, hyphen; turn everything else into '_'
    return re.sub(r'[^0-9A-Za-z._-]+', '_', s)



# ========= shared geometry/grid utilities (adapted from your earlier code) =========
from collections import deque

def _neighbors8(i,j,H,W):
    for di,dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < H and 0 <= nj < W:
            yield ni, nj

def snap_to_free_cell(occ, i, j, max_radius=3):
    """BFS outward until we find an unoccupied cell; returns (ni,nj) or (i,j) if none."""
    H, W = occ.shape
    if not occ[i, j]:
        return i, j
    q = deque([(i, j)])
    seen = {(i, j)}
    r = 0
    while q and r <= max_radius:
        for _ in range(len(q)):
            ci, cj = q.popleft()
            for ni, nj in _neighbors8(ci, cj, H, W):
                if (ni, nj) in seen: 
                    continue
                seen.add((ni, nj))
                if not occ[ni, nj]:
                    return ni, nj
                q.append((ni, nj))
        r += 1
    return i, j

def mkdir(p): os.makedirs(p, exist_ok=True); return p

def compute_workspace_bounds(C, R, margin=1.0):
    if len(C) == 0: return -5.0, 5.0, -5.0, 5.0
    cx, cy = C[:,0], C[:,1]
    xmin = float(np.min(cx - R) - margin)
    xmax = float(np.max(cx + R) + margin)
    ymin = float(np.min(cy - R) - margin)
    ymax = float(np.max(cy + R) + margin)
    return xmin, xmax, ymin, ymax

def world_to_ij(xy, bounds, res):
    xmin, xmax, ymin, ymax = bounds
    # floor to cell index, then clamp to [0, H/W-1]
    jf = (xy[0] - xmin) / res
    if_ = (xy[1] - ymin) / res
    return int(np.floor(np.clip(if_, 0, np.floor((ymax - ymin)/res)))), \
           int(np.floor(np.clip(jf, 0, np.floor((xmax - xmin)/res))))

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

def choose_reference_length(L_deform: float, L_rigid: float):
    """Pick a single reference for *all* methods on this episode."""
    if np.isfinite(L_deform):
        return float(L_deform), "A*_deform"
    if np.isfinite(L_rigid):
        return float(L_rigid), "A*_rigid"
    return None, "none"

def metric_pack(path_xy, L_ref, *, reached: bool, C, R, barrier_thresh: float, tube_thresh: float):
    L_exec = float(path_length(path_xy))
    S = 1.0 if reached else 0.0

    if (L_ref is None) or not (L_ref > 0 and np.isfinite(L_ref)):
        spl = np.nan
        detour = np.nan
    else:
        spl = S * (L_ref / max(L_ref, L_exec))
        detour = L_exec / L_ref

    # Use the flat-output safety helper you defined in this file
    safety = compute_safety_stats(path_xy, C, R,
                                  barrier_thresh=float(barrier_thresh),
                                  tube_thresh=float(tube_thresh))

    return dict(
        success=float(S),
        path_length=L_exec,
        SPL=float(spl) if np.isfinite(spl) else np.nan,
        detour=float(detour) if np.isfinite(detour) else np.nan,
        # take min_clearance straight from safety (no need to recompute)
        min_clear=float(safety["min_clearance"]) if np.isfinite(safety["min_clearance"]) else np.nan,
        barrier_viol_rate=float(safety["barrier_viol_rate"]),
        tube_viol_rate=float(safety["tube_viol_rate"]),
        barrier_penalty_sum=float(safety["barrier_penalty_sum"]),
    )


# def summarize_method(path, L_ref, reached, C, R):
#     """Return dict for one method with success/SPL/detour/min_clear; NaNs if no path."""
#     if path is None or len(path) == 0:
#         return dict(success=0.0, SPL=np.nan, detour=np.nan, min_clear=np.nan)
#     L_exec, detour, spl = metric_pack(path, L_ref, reached)
#     # min clearance along this path vs *raw* obstacles
#     def _clearance_profile(P):
#         return [float(np.min(np.linalg.norm(C - p[None,:], axis=1) - R)) if len(C) else np.inf
#                 for p in P]
#     mc = float(np.nanmin(_clearance_profile(path))) if len(path) else np.nan
#     return dict(success=float(reached), SPL=float(spl), detour=float(detour), min_clear=mc)
def summarize_method(tag, m):
    """Uniform pretty print for quick debugging (optional)."""
    return (f"{tag}: S={int(m['success'])}  L={m['path_length']:.3f}  "
            f"SPL={m['SPL'] if np.isfinite(m['SPL']) else np.nan:.3f}  "
            f"detour={m['detour'] if np.isfinite(m['detour']) else np.nan:.3f}  "
            f"minClr={m['min_clear'] if np.isfinite(m['min_clear']) else np.nan:.3f}")


def a_star_path(occ, bounds, res, start_xy, goal_xy):
    from heapq import heappush, heappop
    H, W = occ.shape
    si, sj = world_to_ij(start_xy, bounds, res)
    gi, gj = world_to_ij(goal_xy, bounds, res)
    if not (0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): return np.inf, []
    si, sj = snap_to_free_cell(occ, si, sj, max_radius=3)
    gi, gj = snap_to_free_cell(occ, gi, gj, max_radius=3)
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
    path_ij = [(gi,gj)]
    ci,cj = gi,gj
    while not (ci==si and cj==sj):
        pi,pj = came[ci,cj]
        if pi < 0: break
        path_ij.append((pi,pj)); ci,cj = pi,pj
    path_ij.reverse()
    length_steps = sum(np.hypot(path_ij[k][0]-path_ij[k-1][0], path_ij[k][1]-path_ij[k-1][1])
                       for k in range(1,len(path_ij)))
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
    if len(C) == 0: return np.full((H,W), np.inf, dtype=np.float32)
    clear = np.full((H,W), np.inf, dtype=np.float32)
    for (cx, cy), r in zip(C, R):
        d = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32) - float(r)
        clear = np.minimum(clear, d)
    return clear

def continuous_clearance_point(xy, C, R):
    if C.size == 0:
        return np.inf
    return float(np.min(np.linalg.norm(C - xy[None,:], axis=1) - R))

def deformable_a_star_path(clearance, bounds, res, start_xy, goal_xy,
                           r_rest, r_min, lam=5.0, beta=0.6, eps=1e-3, C=None, R=None):
    """Feasible if c>=r_min; edge cost = step + beta * avg_pen * step; pen ~ (max(0, r_rest/c -1))^2."""
    from heapq import heappush, heappop
    H, W = clearance.shape
    si, sj = world_to_ij(start_xy, bounds, res)
    gi, gj = world_to_ij(goal_xy, bounds, res)
    if not (0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): return np.inf, []
    feas = clearance >= r_min
    # NEW: continuous feasibility at exact coordinates (if C/R provided)
    start_ok = (C is None or R is None) or (continuous_clearance_point(np.asarray(start_xy), C, R) >= r_min)
    goal_ok  = (C is None or R is None) or (continuous_clearance_point(np.asarray(goal_xy),  C, R) >= r_min)
    if not start_ok or not goal_ok:
        # fall back to snapping on the feas grid
        def snap_to_feas(i,j, max_radius=3):
            if feas[i,j]: return i,j
            return snap_to_free_cell(~feas, i, j, max_radius=max_radius)  # reuse the same BFS on inverted mask
        si, sj = snap_to_feas(si, sj, 3)
        gi, gj = snap_to_feas(gi, gj, 3)
        if not (feas[si, sj] and feas[gi, gj]):
            return np.inf, []
    else:
        # if exact ok but grid says not, also snap just in case
        if not feas[si, sj]:
            si, sj = snap_to_free_cell(~feas, si, sj, 3)
        if not feas[gi, gj]:
            gi, gj = snap_to_free_cell(~feas, gi, gj, 3)
        if not (feas[si, sj] and feas[gi, gj]):
            return np.inf, []

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
            if not (0<=ni<H and 0<=nj<W) or not feas[ni,nj]: continue
            base = cstep * res
            deform = beta * 0.5*(pij + pen(clearance[ni,nj])) * base
            tentative = g[i,j] + base + deform
            if tentative < g[ni,nj]:
                g[ni,nj] = tentative
                came[ni,nj] = [i,j]
                heappush(pq, (tentative + h(ni,nj), (ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf, []
    path_ij = [(gi,gj)]
    ci,cj = gi,gj
    while not (ci==si and cj==sj):
        pi,pj = came[ci,cj]
        if pi < 0: break
        path_ij.append((pi,pj)); ci,cj = pi,pj
    path_ij.reverse()
    length_m = sum(np.hypot(path_ij[k][0]-path_ij[k-1][0], path_ij[k][1]-path_ij[k-1][1])*res
                   for k in range(1,len(path_ij)))
    path_xy = np.stack([ij_to_world(i,j,bounds,res) for (i,j) in path_ij], axis=0) if len(path_ij) else []
    return float(length_m), path_xy

# ========= metrics & helpers =========
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

def _barrier_penalty_soft(clearance: float, thresh: float, scale: float = 1.0) -> float:
    if thresh <= 0.0: return 0.0
    gap = max(0.0, thresh - float(clearance))
    return float(scale * (gap * gap))

def compute_safety_stats(path_xy, C, R, barrier_thresh, tube_thresh):
    T = len(path_xy)
    if T == 0:
        return dict(min_clearance=np.inf, barrier_viol_rate=0.0, tube_viol_rate=0.0,
                    barrier_penalty_sum=0.0, barrier_viol_count=0, tube_viol_count=0)
    clears = np.asarray([clearance_to_circles(np.asarray(xy,float), C, R) for xy in path_xy], float)
    barrier_mask = clears < float(barrier_thresh)
    tube_mask    = clears < float(tube_thresh)
    penalties = [_barrier_penalty_soft(c, barrier_thresh, 1.0) for c in clears]
    return dict(
        min_clearance=float(np.min(clears)),
        barrier_viol_rate=float(np.count_nonzero(barrier_mask))/T,
        tube_viol_rate=float(np.count_nonzero(tube_mask))/T,
        barrier_penalty_sum=float(np.sum(penalties)),
        barrier_viol_count=int(np.count_nonzero(barrier_mask)),
        tube_viol_count=int(np.count_nonzero(tube_mask))
    )

# ========= CoefEnergyNet feature helpers =========
def build_local_feats(o_w: np.ndarray, goal_w: np.ndarray, C_w: np.ndarray, R_w: np.ndarray, W_w: np.ndarray):
    o = torch.as_tensor(o_w, dtype=torch.float32)
    g = torch.as_tensor(goal_w, dtype=torch.float32)
    C = torch.as_tensor(C_w, dtype=torch.float32) if C_w.size else torch.zeros(0,2, dtype=torch.float32)
    R = torch.as_tensor(R_w, dtype=torch.float32) if R_w.size else torch.zeros(0, dtype=torch.float32)
    W = torch.as_tensor(W_w, dtype=torch.float32) if W_w.size else torch.zeros(0, dtype=torch.float32)
    if C.ndim == 1: C = C.reshape(0,2)
    dg = (g - o); gdist = torch.linalg.norm(dg).unsqueeze(0)
    goal_feats = torch.stack([dg[0], dg[1], gdist[0], torch.tensor(1.0)], dim=0).unsqueeze(0)
    if C.shape[0] == 0: obs_feats = torch.zeros(1,0,6, dtype=torch.float32)
    else:
        dxdy = (g.unsqueeze(0) - C)
        obs_feats = torch.cat([C, R.unsqueeze(-1), W.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)
    return obs_feats, goal_feats

def map_alpha_to_world(W_in: np.ndarray, R_in: np.ndarray, alphas: np.ndarray, mode: str, k_rad: float = 0.05):
    al = np.maximum(alphas, 0.0)
    W_out = W_in.copy(); R_out = R_in.copy()
    if mode in ("weight","both") and al.size: W_out = W_out * al
    if mode in ("radius","both") and al.size: R_out = R_out + k_rad * al
    return W_out, R_out

# ========= DWA/CBF-style reactive baseline =========
def reactive_dwa_cbf_path(C, R, start_xy, goal_xy, *,
                          dt=0.02, steps=1500, v_max=0.7,
                          k_goal=1.5, k_rep=0.9, d_safe=0.8, r_min=0.3):
    """
    Point-mass center controller:
      v = k_goal * (goal - x)/||goal-x||  +  sum_j k_rep * f_rep_j
    with f_rep_j = (max(0, d_safe - d_j)/(d_j+1e-3))^2 * n_j, n_j = (x - c_j)/d_j
    Feasibility: stop if clearance < r_min (counts as violation but still move minimally).
    """
    x = np.array(start_xy, dtype=np.float32)
    path = [x.copy()]
    for t in range(steps):
        to_goal = goal_xy - x
        dist = np.linalg.norm(to_goal) + 1e-6
        v = k_goal * (to_goal / dist)
        # repulsion
        for (cx,cy), r in zip(C, R):
            dvec = x - np.array([cx,cy], dtype=np.float32)
            dj = np.linalg.norm(dvec) + 1e-6
            clear = dj - r
            if clear < d_safe:
                gain = ((d_safe - clear) / (dj))**2
                v += k_rep * gain * (dvec / dj)
        # speed clamp
        sp = np.linalg.norm(v)
        if sp > v_max: v = v * (v_max / sp)
        # step (freeze if too infeasible)
        if len(C):
            minc = clearance_to_circles(x, C, R)
            if minc < r_min:  # emergency brake
                v *= 0.2
        x = x + dt * v
        path.append(x.copy())
        if np.linalg.norm(x - goal_xy) <= 0.2: break
    return np.array(path, dtype=np.float32)

# ========= plotting =========
def plot_paths_overlay(outdir, world, our_path, path_star_rigid, path_star_deform, reactive_path, start, goal):
    plt.figure(figsize=(6.2,5.6))
    th = np.linspace(0, 2*np.pi, 128)
    for (cx,cy), r in zip(world.C_np, world.R_np):
        plt.plot(cx + r*np.cos(th), cy + r*np.sin(th), lw=1.2, alpha=0.8, color='k')
    if path_star_rigid is None or len(path_star_rigid)==0:
        plt.text(start[0], start[1], "No Rigid A*", color="tab:blue", fontsize=8)
    else:
        plt.plot(path_star_rigid[:,0], path_star_rigid[:,1], '--', lw=2, label='Rigid A*', alpha=0.9, color="tab:blue")
    if path_star_deform is None or len(path_star_deform)==0:
        plt.text(start[0], start[1]-0.5, "No Deformable A*", color="tab:green", fontsize=8)
    else:
        plt.plot(path_star_deform[:,0], path_star_deform[:,1], '-.', lw=2, label='Deformable A*', alpha=0.9, color="tab:green")
    if reactive_path is not None and len(reactive_path):
        plt.plot(reactive_path[:,0], reactive_path[:,1], ':', lw=2.3, label='DWA/CBF', alpha=0.9, color="tab:purple")
    if len(our_path):
        P = np.asarray(our_path)
        plt.plot(P[:,0], P[:,1], '-', lw=2.3, label='GRL-SNAM', alpha=0.95, color="tab:orange")
    plt.scatter([start[0]],[start[1]], s=50, marker='s', label='Start', color='k')
    plt.scatter([goal[0]],[goal[1]], s=70, marker='*', label='Goal', color='crimson')
    plt.axis('equal'); plt.legend(frameon=False, loc='best')
    plt.title("Trajectories: ours vs baselines")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "trajectory_overlay.png"), dpi=200)
    plt.close()

def plot_bars_violins(outdir, table_rows: List[Dict[str, Any]], split_name: str):
    # Collect per-method stats
    methods = ["GRL-SNAM", "RigidA*", "DeformA*", "DWA/CBF"]
    metrics = ["success", "SPL", "detour"]
    agg = {m:{k:[] for k in metrics} for m in methods}
    for r in table_rows:
        for m in methods:
            agg[m]["success"].append(float(r[f"{m}_success"]))
            if not np.isnan(r[f"{m}_SPL"]): agg[m]["SPL"].append(float(r[f"{m}_SPL"]))
            if not np.isnan(r[f"{m}_detour"]): agg[m]["detour"].append(float(r[f"{m}_detour"]))
    # bar: success
    plt.figure(figsize=(6.4,4.2))
    means = [np.mean(agg[m]["success"]) if len(agg[m]["success"]) else 0 for m in methods]
    stds  = [np.std(agg[m]["success"]) if len(agg[m]["success"]) else 0 for m in methods]
    plt.bar(methods, means, yerr=stds, capsize=5, alpha=0.85)
    plt.ylim(0,1.0); plt.ylabel("Success"); plt.title(f"Success ({safe_name(split_name)})"); plt.grid(axis='y', alpha=0.25)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{safe_name(split_name)}_success_bar.png"), dpi=200); plt.close()
    # violin: SPL
    plt.figure(figsize=(6.8,4.2))
    data = [agg[m]["SPL"] for m in methods]
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(np.arange(1,len(methods)+1), methods); plt.ylabel("SPL (vs ref)"); plt.title(f"SPL ({safe_name(split_name)})")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{safe_name(split_name)}_spl_violin.png"), dpi=200); plt.close()
    # violin: detour
    plt.figure(figsize=(6.8,4.2))
    data = [agg[m]["detour"] for m in methods]
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(np.arange(1,len(methods)+1), methods); plt.ylabel("Detour (L_exec/L_ref)"); plt.title(f"Detour ({safe_name(split_name)})")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{safe_name(split_name)}_detour_violin.png"), dpi=200); plt.close()

def plot_pareto(outdir, rows: List[Dict[str, Any]], method_key: str, split_name: str):
    mkdir(outdir)
    xs, ys = [], []
    for r in rows:
        c = r.get(f"{safe_name(method_key)}_min_clear", np.nan)
        s = r.get(f"{safe_name(method_key)}_SPL", np.nan)
        if np.isfinite(c) and np.isfinite(s): xs.append(c); ys.append(s)
    if not xs: return
    plt.figure(figsize=(5.2,4.5))
    plt.scatter(xs, ys, s=28, alpha=0.7)
    plt.xlabel("Min clearance (m)"); plt.ylabel("SPL"); plt.title(f"{safe_name(method_key)}: SPL vs Clearance ({safe_name(split_name)})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    fname = f"{safe_name(split_name)}_{safe_name(method_key)}_pareto.png"
    plt.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close()

# ========= env cases (ID / OOD) =========
def get_case(case: str):
    if case == "case1-tight":   return gen.sample_obstacles_case1_tight
    if case == "case2-harder":  return gen.sample_obstacles_case2_harder
    if case == "case2-dense":
        def _dense(cfg):
            cfg2 = gen.GenCfg(); cfg2.__dict__.update(cfg.__dict__)
            cfg2.n_obs_range = (25, 45)
            return gen.sample_obstacles_case2_harder(cfg2)
        return _dense
    # fallback
    return gen.sample_obstacles_case1_tight

# ========= main evaluation per-episode =========
# def rollout_grl_snam(
#     model,
#     cfg_local,
#     world,
#     *,
#     alpha_mode="weight",
#     k_rad=0.05,
#     steps=1200,
#     correction=True,          # turn on to use HistSecantController
#     device="cpu",
#     warmup_steps=5,           # let dynamics settle before corrections
#     stall_tol=1e-3,           # "no progress" threshold (m per step)
#     stall_patience=25         # steps of no progress before reset
# ):
#     planner = gen.planner_from_cfg(
#         cfg_local, world,
#         getattr(cfg_local, "k_bulk", 20.0),
#         getattr(cfg_local, "gamma_s", 0.05),
#         getattr(cfg_local, "d_hat", 0.5),
#         getattr(cfg_local, "radius", getattr(cfg_local, "d_hat", 0.5))
#     )
#     dt = getattr(cfg_local, "dt", 0.02)

#     # --- Controller: one instance, keeps history
#     ctrl = None
#     if correction:
#         ctrl = HistSecantController(
#             k_alpha=2, lr_beta=0.15, lr_gamma=0.10, lr_alpha=0.4,
#             safe_margin=0.5 * getattr(cfg_local, "radius", 0.16),
#             v_min=0.25, v_max=0.5, prog_eps=0.01, ema=0.9
#         )

#         def _reset_ctrl():
#             ctrl.prev = None
#             ctrl.J = None

#     path, frames = [], []

#     def _capture_frame(sys, t_idx):
#         entry = {
#             "center": sys.o.detach().cpu().to(torch.float32),
#             "theta": (sys.theta.detach().cpu().to(torch.float32)
#                       if hasattr(sys, "theta") else torch.tensor(0.0)),
#             "t": int(t_idx),
#         }
#         if hasattr(sys, "Pw") and sys.Pw is not None:
#             entry["Pw"] = sys.Pw.detach().cpu().to(torch.float32)
#         elif hasattr(sys, "Ploc") and sys.Ploc is not None:
#             o = entry["center"].numpy(); theta = float(entry["theta"])
#             c, s = np.cos(theta), np.sin(theta)
#             Rm = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
#             entry["Pw"] = (sys.Ploc.detach().cpu().to(torch.float32) @ Rm.T) + torch.tensor(o)
#         else:
#             entry["Pw"] = planner.sys.world_points()
#         return entry

#     frames.append(_capture_frame(planner.sys, 0))

#     # progress tracking (for event-driven resets)
#     prev_dist = float("inf")
#     stall_ctr = 0

#     for t in range(steps):
#         sys = planner.sys
#         o_w = sys.o.detach().cpu().numpy()
#         Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)

#         # build net inputs
#         obs_feats, goal_feats = build_local_feats(o_w, cfg_local.goal, Cw, Rw, Ww)
#         obs_mask = (
#             torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
#             if obs_feats.shape[1] else torch.zeros(1, 0, dtype=torch.bool, device=device)
#         )

#         with torch.no_grad():
#             alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))

#         # defaults from the net
#         al_np  = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
#         beta_f = float(beta.squeeze(0).item())
#         gamma_f= float(gamma.squeeze(0).item())

#         # --- Optional controller update (stateful, after warmup)
#         if ctrl is not None and t >= warmup_steps:
#             try:
#                 # current speed vector if available
#                 v_vec = np.zeros(2, dtype=np.float32)
#                 if hasattr(planner.sys, "v_o") and planner.sys.v_o is not None:
#                     v_vec = planner.sys.v_o.detach().cpu().numpy().astype(np.float32)

#                 # quick local clearance at center (no margins)
#                 clr_now = float("inf")
#                 if Cw.shape[0] > 0:
#                     clr_now = float(np.min(np.linalg.norm(o_w[None, :] - Cw, axis=1) - Rw))

#                 dist_now = float(np.linalg.norm(o_w - cfg_local.goal))
#                 speed_now = float(np.linalg.norm(v_vec))

#                 # EVENT-DRIVEN RESET: (a) progress stall, (b) nearest-obstacle set changes a lot
#                 progress = prev_dist - dist_now
#                 stall_ctr = (stall_ctr + 1) if (progress < stall_tol) else 0
#                 prev_dist = dist_now

#                 # nearest-k indices now vs. previous (if any)
#                 idx_now = ctrl._select_alpha_indices(o_w, Cw, Rw, Ww)
#                 idx_prev = ctrl.prev["idx"] if (ctrl.prev is not None and "idx" in ctrl.prev) else None
#                 idx_changed = (idx_prev is not None) and (idx_now.shape != idx_prev.shape or np.any(idx_now != idx_prev))

#                 if stall_ctr > stall_patience or idx_changed:
#                     _reset_ctrl()
#                     stall_ctr = 0

#                 # stateful update (preserves history across steps)
#                 al_upd, b_upd, g_upd = ctrl.update(
#                     alphas, beta, gamma,
#                     o_w, v_vec, cfg_local.goal, Cw, Rw, Ww,
#                     clr_now, dist_now, speed_now
#                 )
#                 # adopt updates (controller already returns tensors on correct device/dtype)
#                 if al_upd is not None:
#                     al_np  = al_upd.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
#                 if b_upd is not None:
#                     beta_f = float(b_upd.squeeze(0).item())
#                 if g_upd is not None:
#                     gamma_f = float(g_upd.squeeze(0).item())
#             except Exception as _e:
#                 # Non-fatal; keep raw net outputs
#                 # print(f"[WARN] controller update failed at t={t}: {_e}")
#                 pass

#         # map α once
#         W_adj, R_adj = Ww.copy(), Rw.copy()
#         if alpha_mode != "none":
#             W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, alpha_mode, k_rad=k_rad)

#         # build adjusted slice + inject scalars
#         world_step = ssi.WorldObstacles(Cw, R_adj, W_adj, d_hat=cfg_local.d_hat)
#         if hasattr(planner, "stage_field") and hasattr(planner.stage_field, "w_goal"):
#             planner.stage_field.w_goal = max(0.0, beta_f)
#         if hasattr(planner.sys, "gamma_o"):
#             planner.sys.gamma_o = max(0.0, gamma_f)

#         info = planner.step(dt, world_step)

#         frames.append(_capture_frame(planner.sys, t + 1))
#         center_xy = np.asarray(info["center"], float)
#         path.append(center_xy)

#         if np.linalg.norm(center_xy - cfg_local.goal) <= cfg_local.goal_tol:
#             break

#     return np.asarray(path, dtype=np.float32), frames, planner



def rollout_grl_snam(model, cfg_local, world, *, alpha_mode="weight", k_rad=0.05, steps=1200, alpha="weight", correction=True, device="cpu"):
    # def _exit_with_width(entry, cur_bounds, next_bounds, C, R, **kw):
    #     # import/forward to your existing solver inside this module:
    #     # return _pick_exit_overlap_shortest_with_width(entry=..., cur_bounds=..., next_bounds=..., C_inf=C, R_inf=R, **kw)
    #     return ssi.pick_exit_overlap_shortest_with_width(
    #         entry=entry,
    #         cur_bounds=cur_bounds,
    #         next_bounds=next_bounds,
    #         C_inf=C, R_inf=R,
    #         **kw
    #     )
    sm = StageManagerOnline(
        stage_size=cfg_local.stage_size,
        inflate=cfg_local.inflate,
        near_eps= 0.75 * cfg_local.radius,
        advance_frac=0.7,
        detour_fracs=(0.25, 0.5, 0.75),
        exit_solver=None,
        # exit_kwargs=dict(
        #     tube=cfg_local.inflate, edge_margin=1e-3, exit_margin=1e-3,
        #     grid_res=16, knn_k=7, nsamp_edge=24, roi_pad_scale=(0.8,0.8),
        # )
        exit_kwargs=None
    )
    planner = gen.planner_from_cfg(cfg_local, world, cfg_local.k_bulk,
                                   cfg_local.gamma_s, cfg_local.d_hat, cfg_local.radius)
    planner.sm = sm
    dt = cfg_local.dt
    path, frames = [], []

    def _capture_frame(sys, t_idx):
        entry = {
            "center": sys.o.detach().cpu().to(torch.float32),
            "theta": (sys.theta.detach().cpu().to(torch.float32) if hasattr(sys, "theta") else torch.tensor(0.0)),
            "t": int(t_idx),
        }
        if hasattr(sys, "Pw") and sys.Pw is not None:
            entry["Pw"] = sys.Pw.detach().cpu().to(torch.float32)
        elif hasattr(sys, "Ploc") and sys.Ploc is not None:
            o = entry["center"].numpy(); theta = float(entry["theta"])
            c, s = np.cos(theta), np.sin(theta)
            Rm = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
            entry["Pw"] = (sys.Ploc.detach().cpu().to(torch.float32) @ Rm.T) + torch.tensor(o)
        else:
            entry["Pw"] = planner.sys.world_points()
        return entry

    frames.append(_capture_frame(planner.sys, 0))
    new_stage_timer = 0
    o0 = planner.sys.o.detach().cpu().numpy()
    goal_xy = np.asarray(cfg_local.goal, dtype=float)
    planner.sm.reset(o0, goal_xy, world)
    
    for t in range(steps):
        sys = planner.sys
        o_w = sys.o.detach().cpu().numpy()
        Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)



        obs_feats, goal_feats = build_local_feats(o_w, cfg_local.goal, Cw, Rw, Ww)
        obs_mask = (torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
                    if obs_feats.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device))

        with torch.no_grad():
            alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))

        # al_np  = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
        # beta_f = float(beta.squeeze(0).item())
        # gamma_f= float(gamma.squeeze(0).item())
        
        if correction and new_stage_timer >= 5:
            if reinitialize:
                controller = HistSecantController(k_alpha=1, lr_beta=1.0, lr_gamma=1.0, lr_alpha=1.0, safe_margin=0.5*getattr(cfg_local,"radius",0.16), prog_eps=0.1, v_min=0.05, v_max=5.0, ema=0.9)
                reinitialize = False
                
            dist_now = np.linalg.norm(o_w - cfg_local.goal)
            speed_now = float(np.linalg.norm(planner.sys.v_o.detach().cpu().numpy())) if hasattr(planner.sys, "v_o") else 0.0
            # quick min clearance from current slice
            clr_now = np.inf
            if Cw.shape[0] > 0:
                clr_now = float(np.min(np.linalg.norm(o_w[None,:] - Cw, axis=1) - Rw))

            # create once (outside loop)
            # controller = HistSecantController(k_alpha=2, safe_margin=0.08, v_min=0.25)
            # print(speed_now, clr_now)

            # update params without extra sims
            alphas_use, beta_use, gamma_use = controller.update(
                alphas, beta, gamma, o_w, speed_now, cfg_local.goal, Cw, Rw, Ww, clr_now, dist_now, speed_now
            )
            # print(t, alphas_use, beta_use, gamma_use)
            al_np = alphas_use.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
            beta_f = float(beta_use.squeeze(0).item())
            gamma_f = float(gamma_use.squeeze(0).item())
        else:
            al_np = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
            beta_f = float(beta.squeeze(0).item())
            gamma_f = float(gamma.squeeze(0).item())
        # map α
        W_adj, R_adj = Ww.copy(), Rw.copy()
        if alpha_mode != "none":
            W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, alpha_mode)
            # inject goal weight and damping
            planner.stage_field.w_goal = max(0.0, beta_f)
            planner.sys.gamma_o = max(0.0, gamma_f)

        world_step = ssi.WorldObstacles(Cw, R_adj, W_adj, d_hat=cfg_local.d_hat)
        planner.stage_field.w_goal = max(0.0, beta_f)
        planner.sys.gamma_o = max(0.0, gamma_f)

        info = planner.step(dt, world_step)

        frames.append(_capture_frame(planner.sys, t+1))
        center_xy = np.asarray(info["center"], float)
        path.append(center_xy)

        path.append(np.asarray(info["center"], float))

        if np.linalg.norm(np.asarray(info["center"]) - goal_xy) <= cfg_local.goal_tol:
            break

    return np.array(path, dtype=np.float32), frames, planner

def evaluate_episode(
    model, cfg_env, world, start_xy, goal_xy, *,
    alpha_mode, k_rad, steps, device,
    grid_cache  # Dict[str, Any]: {"bounds","res","occ_rigid","clearance"}
):
    # --- Params and cached grids
    r_rest = getattr(cfg_env, "radius", cfg_env.d_hat)
    r_min  = getattr(cfg_env, "radius_min", 0.3 * r_rest)
    bounds = grid_cache["bounds"]
    res    = grid_cache["res"]
    occ_rigid      = grid_cache["occ_rigid"]
    clearance_grid = grid_cache["clearance"]

    # ========== Baselines ==========
    # A* (rigid / inflated)
    L_rigid, path_rigid = a_star_path(
        occ_rigid, bounds, res, start_xy, goal_xy
    )

    # A* (deformable / clearance-penalized)
    L_deform, path_deform = deformable_a_star_path(
        clearance_grid, bounds, res, start_xy, goal_xy,
        r_rest=r_rest, r_min=r_min, lam=5.0, beta=0.6, eps=1e-3
    )

    # Reactive (DWA/CBF-like) baseline
    reactive_path = reactive_dwa_cbf_path(
        world.C_np, world.R_np, start_xy, goal_xy,
        dt=cfg_env.dt, steps=steps, v_max=0.9,
        k_goal=1.6, k_rep=1.0, d_safe=max(0.8 * r_rest, 0.5), r_min=r_min
    )

    # ========== Our rollout (GRL-SNAM) ==========
    cfg_local = gen.GenCfg()
    cfg_local.__dict__.update(cfg_env.__dict__)
    cfg_local.start = start_xy.astype(np.float32)
    cfg_local.goal  = goal_xy.astype(np.float32)

    our_path, our_frames, our_planner = rollout_grl_snam(
        model, cfg_local, world,
        alpha_mode=alpha_mode, k_rad=k_rad, steps=steps, device=device
    )

    # ========== Reference length (one per episode, used by ALL methods) ==========
    L_ref, ref_tag = choose_reference_length(
        L_deform, L_rigid, start_xy, goal_xy, fallback="nan"
    )

    # ========== Success flags for each method ==========
    # Our & reactive: geometric success under cfg_env.goal_tol
    our_success = success_reached(our_path[-1] if len(our_path) else start_xy,
                                  goal_xy, cfg_env.goal_tol)
    react_success = success_reached(reactive_path[-1] if len(reactive_path) else start_xy,
                                    goal_xy, cfg_env.goal_tol)
    # A*: path existence => success (they are planners, not simulators)
    rigid_success  = np.isfinite(L_rigid)
    deform_success = np.isfinite(L_deform)

    # ========== Per-method summaries (success, SPL, detour, min_clear) ==========
    m_ours     = summarize_method(our_path,        L_ref, our_success,    C=world.C_np, R=world.R_np)
    m_rigid    = summarize_method(path_rigid,      L_ref, rigid_success,  C=world.C_np, R=world.R_np)
    m_deform   = summarize_method(path_deform,     L_ref, deform_success, C=world.C_np, R=world.R_np)
    m_reactive = summarize_method(reactive_path,   L_ref, react_success,  C=world.C_np, R=world.R_np)

    # (Optional) If you also want the raw exec lengths for tables:
    L_exec_ours     = path_length(our_path)
    L_exec_reactive = path_length(reactive_path)
    L_exec_rigid    = path_length(path_rigid)
    L_exec_deform   = path_length(path_deform)

    return dict(
        # paths (for overlay plots & GIFs)
        our_path=our_path,
        rigid_path=path_rigid,
        deform_path=path_deform,
        reactive_path=reactive_path,

        # lengths
        L_rigid=L_rigid,
        L_deform=L_deform,
        L_exec_ours=L_exec_ours,
        L_exec_reactive=L_exec_reactive,
        L_exec_rigid=L_exec_rigid,
        L_exec_deform=L_exec_deform,

        # unified reference used for SPL/detour
        L_ref=L_ref,
        ref_tag=ref_tag,

        # per-method metric packs
        m_ours=m_ours,
        m_rigid=m_rigid,
        m_deform=m_deform,
        m_reactive=m_reactive,

        # artifacts for movie generation upstream
        our_frames=our_frames,
        our_planner=our_planner,
    )

def evaluate_episode(model, cfg_env, world, start_xy, goal_xy, *, alpha_mode, k_rad, steps, device,
                     grid_cache: Dict[str, Any]):
    # Params / cached grids
    r_rest = getattr(cfg_env, "radius", cfg_env.d_hat)
    r_min  = getattr(cfg_env, "radius_min", 0.3 * r_rest)
    bounds = grid_cache["bounds"]; res = grid_cache["res"]
    occ_rigid = grid_cache["occ_rigid"]; clearance_grid = grid_cache["clearance"]

    # ---------- Baselines ----------
    # A* (rigid)
    L_rigid, path_rigid = a_star_path(occ_rigid, bounds, res, start_xy, goal_xy)
    rigid_success = np.isfinite(L_rigid) and len(path_rigid) > 1  # path found

    # A* (deformable)
    L_deform, path_deform = deformable_a_star_path(
        clearance_grid, bounds, res, start_xy, goal_xy,
        r_rest=r_rest, r_min=r_min, lam=5.0, beta=0.6, eps=1e-3, C=world.C_np, R=world.R_np
    )
    deform_success = np.isfinite(L_deform) and len(path_deform) > 1

    # Reactive (DWA/CBF)
    reactive_path = reactive_dwa_cbf_path(
        world.C_np, world.R_np, start_xy, goal_xy,
        dt=cfg_env.dt, steps=steps, v_max=0.9,
        k_goal=1.6, k_rep=1.0, d_safe=max(0.8*r_rest, 0.5), r_min=r_min
    )
    reactive_success = success_reached(reactive_path[-1] if len(reactive_path) else start_xy,
                                       goal_xy, cfg_env.goal_tol)

    # Ours (GRL-SNAM)
    cfg_local = gen.GenCfg()
    cfg_local.__dict__.update(cfg_env.__dict__)
    cfg_local.start = start_xy.astype(np.float32)
    cfg_local.goal  = goal_xy.astype(np.float32)
    our_path, our_frames, our_planner = rollout_grl_snam(
        model, cfg_local, world, alpha_mode=alpha_mode, k_rad=k_rad, steps=steps, device=device
    )
    our_success = success_reached(our_path[-1] if len(our_path) else start_xy, goal_xy, cfg_env.goal_tol)

    # ---------- Single reference for this episode ----------
    L_ref, ref_tag = choose_reference_length(L_deform, L_rigid)

    # ---------- Metrics (all using the *same* L_ref when defined) ----------
    m_ours     = metric_pack(our_path,       L_ref, reached=our_success,
                             C=world.C_np, R=world.R_np,
                             barrier_thresh=cfg_env.d_hat, tube_thresh=r_min)
    # For A* methods, success = path exists (grid A* connects start->goal by construction)
    m_rigid    = metric_pack(path_rigid,     L_rigid if rigid_success else None,
                             reached=rigid_success, C=world.C_np, R=world.R_np,
                             barrier_thresh=cfg_env.d_hat, tube_thresh=r_min)
    m_deform   = metric_pack(path_deform,    L_deform if deform_success else None,
                             reached=deform_success, C=world.C_np, R=world.R_np,
                             barrier_thresh=cfg_env.d_hat, tube_thresh=r_min)
    m_reactive = metric_pack(reactive_path,  L_ref, reached=reactive_success,
                             C=world.C_np, R=world.R_np,
                             barrier_thresh=cfg_env.d_hat, tube_thresh=r_min)

    # (optional) quick stdout
    # print(summarize_method("GRL", m_ours), "| ref:", ref_tag)

    return dict(
        our_path=our_path, rigid_path=path_rigid, deform_path=path_deform, reactive_path=reactive_path,
        L_rigid=L_rigid, L_deform=L_deform, ref_tag=ref_tag, ref_available=(L_ref is not None),
        m_ours=m_ours, m_rigid=m_rigid, m_deform=m_deform, m_reactive=m_reactive,
        our_frames=our_frames, our_planner=our_planner
    )

# ========= driver =========
def run_split(split_name: str, case_fn, args, device, model, run_dir) -> Dict[str, Any]:
    print(f"\n=== Split: {safe_name(split_name)}  ({args.n_envs} envs × {args.n_trials} trials) ===")
    rows = []
    rng = np.random.default_rng(args.seed + (0 if split_name=="TestID" else 1337))
    for env_id in range(args.n_envs):
        # env
        cfg = gen.GenCfg()
        cfg.seed = int(rng.integers(0, 10_000_000)); gen.set_all_seeds(cfg.seed)
        cfg.d_hat   = getattr(cfg, "d_hat", 0.5)
        cfg.radius  = getattr(cfg, "radius", 0.5)
        cfg.radius_min = getattr(cfg, "radius_min", 0.3*cfg.radius)
        cfg.k_bulk  = getattr(cfg, "k_bulk", 20.0)
        cfg.gamma_s = getattr(cfg, "gamma_s", 0.05)
        cfg.dt      = getattr(cfg, "dt", 0.02)
        cfg.total_steps = args.steps
        cfg.inflate = getattr(cfg, "s_min", 0.3) * cfg.radius

        C, R, W = case_fn(cfg)
        world = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)

        # workspace + grids for A*
        r_rest = cfg.radius; bounds = compute_workspace_bounds(world.C_np, world.R_np, margin=2.0*r_rest)
        res = max(0.2, 0.35 * max(r_rest, cfg.d_hat))  # slightly finer than 0.5×
        occ_rigid, _, _ = rasterize_inflated_obstacles(world.C_np, world.R_np, r_rest, bounds, res)
        clearance_grid = clearance_field_on_grid(world.C_np, world.R_np, bounds, res)
        grid_cache = dict(bounds=bounds, res=res, occ_rigid=occ_rigid, clearance=clearance_grid)

        for trial_id in range(args.n_trials):
            # sampled start/goal (simple: sample until outside inflated obstacles & far enough)
            def sample_valid_point(margin, bounds, tries=1000):
                xmin,xmax,ymin,ymax = bounds
                for _ in range(tries):
                    p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=np.float32)
                    # not inside inflated obstacles
                    if len(C)==0: return p
                    d = np.linalg.norm(C - p[None,:], axis=1) - (R + margin)
                    if np.all(d >= 0.0): return p
                return p
            #TODO: change it to be wiser
            start = sample_valid_point(cfg.radius_min, [8.0, 9.0, -3.0, -2.0])
            xmin,xmax,ymin,ymax = bounds
            goal  = sample_valid_point(cfg.radius_min,  [xmin - 0.1, 0.1*xmax + 0.9 * xmin, 0.1*ymin + 0.9*ymax, ymax + 0.1])
            if np.linalg.norm(start-goal) < 2.0*cfg.radius: goal = goal + np.array([2.5*cfg.radius,0.0],dtype=np.float32)

            ep_dir = mkdir(os.path.join(run_dir, f"{safe_name(split_name)}_env{env_id:03d}_trial{trial_id:03d}"))

            result = evaluate_episode(model, cfg, world, start, goal,
                                      alpha_mode=args.alpha_mode, k_rad=args.k_rad, steps=args.steps,
                                      device=device, grid_cache=grid_cache)

            # save overlay
            plot_paths_overlay(ep_dir, world, result["our_path"], result["rigid_path"],
                               result["deform_path"], result["reactive_path"], start, goal)

            # === GIF rendering like your previous script ===
            pngs = []
            # pick a stride so we don't dump too many frames (≈30 frames)
            gif_stride = max(1, len(result["our_frames"]) // 30)
            for t_idx in range(0, len(result["our_frames"]), gif_stride):
                png_path = os.path.join(ep_dir, f"frame_{t_idx:04d}.png")
                # same signature you used before (no explicit bounds needed for the original saver)
                gen.save_episode_snapshot(
                    png_path,
                    result["our_planner"],
                    result["our_frames"][:t_idx+1],
                    world,
                    start,  # cfg_local.start equivalent
                    goal,   # cfg_local.goal  equivalent
                    cfg     # we can pass the env cfg; it has dt etc.
                )
                pngs.append(png_path)

            # write GIF
            gif_path = os.path.join(ep_dir, "rollout.gif")
            try:
                gif_frames = [iio.imread(p) for p in pngs]
                iio.imwrite(gif_path, gif_frames, loop=0, fps=10)
            except Exception as e:
                print(f"[WARN] GIF failed: {e}")

            # optional MP4 (requires ffmpeg on PATH)
            mp4_path = os.path.join(ep_dir, "rollout.mp4")
            try:
                with imageio.get_writer(mp4_path, format="FFMPEG", mode="I", fps=30, codec="libx264", quality=7) as w:
                    for fr in gif_frames:
                        w.append_data(fr)
            except Exception as e:
                print(f"[WARN] MP4 failed: {e}")
            
            # record row
            row = {
                "split": split_name, "env_id": env_id, "trial_id": trial_id, "n_obs": len(C),
                # ours
                "GRL-SNAM_success": result["m_ours"]["success"],
                "GRL-SNAM_SPL": result["m_ours"]["SPL"],
                "GRL-SNAM_detour": result["m_ours"]["detour"],
                "GRL-SNAM_min_clear": result["m_ours"]["min_clear"],
                # rigid A*
                "RigidA*_success": float(np.isfinite(result["L_rigid"])),
                "RigidA*_SPL": 1.0 if np.isfinite(result["L_rigid"]) else 0.0,
                "RigidA*_detour": 1.0,   # by definition ref / ref
                "RigidA*_min_clear": float(np.min(clearance_profile(result["rigid_path"], world.C_np, world.R_np))) if len(result["rigid_path"]) else np.nan,
                # deform A*
                "DeformA*_success": float(np.isfinite(result["L_deform"])),
                "DeformA*_SPL": 1.0 if np.isfinite(result["L_deform"]) else 0.0,
                "DeformA*_detour": 1.0,
                "DeformA*_min_clear": float(np.min(clearance_profile(result["deform_path"], world.C_np, world.R_np))) if len(result["deform_path"]) else np.nan,
                # reactive
                "DWA/CBF_success": result["m_reactive"]["success"],
                "DWA/CBF_SPL": result["m_reactive"]["SPL"],
                "DWA/CBF_detour": result["m_reactive"]["detour"],
                "DWA/CBF_min_clear": result["m_reactive"]["min_clear"],
                # refs
                "L_rigid": float(result["L_rigid"]) if np.isfinite(result["L_rigid"]) else float("inf"),
                "L_deform": float(result["L_deform"]) if np.isfinite(result["L_deform"]) else float("inf"),
                "traj_png": os.path.join(ep_dir, "trajectory_overlay.png"),
                "gif_path": gif_path,
                "traj_png": os.path.join(ep_dir, "trajectory_overlay.png"),
            }
            rows.append(row)
            print(f"[{safe_name(split_name)} e{env_id} t{trial_id}] "
                  f"S(GRL)={row['GRL-SNAM_success']:.0f} SPL={row['GRL-SNAM_SPL']:.3f} "
                  f"S(DWA)={row['DWA/CBF_success']:.0f}  A*rigid={'Y' if np.isfinite(result['L_rigid']) else 'N'} "
                  f"A*def={'Y' if np.isfinite(result['L_deform']) else 'N'}")

    # write CSV/JSON
    csv_path = os.path.join(run_dir, f"{safe_name(split_name)}_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(run_dir, f"{safe_name(split_name)}_episodes.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # plots for split
    plot_bars_violins(run_dir, rows, split_name)
    for mk in ["GRL-SNAM","DWA/CBF"]:
        plot_pareto(run_dir, rows, mk, split_name)

    # aggregate
    def _m(name): 
        vals = [float(r[name]) for r in rows if name in r and np.isfinite(r[name])]
        return float(np.mean(vals)) if vals else float("nan")
    agg = {
        "split": split_name,
        "episodes": len(rows),
        "success_mean": {m: _m(f"{m}_success") for m in ["GRL-SNAM","RigidA*","DeformA*","DWA/CBF"]},
        "spl_mean":     {m: _m(f"{m}_SPL")     for m in ["GRL-SNAM","RigidA*","DeformA*","DWA/CBF"]},
        "detour_mean":  {m: _m(f"{m}_detour")  for m in ["GRL-SNAM","RigidA*","DeformA*","DWA/CBF"]},
        "min_clear_mean": {m: _m(f"{m}_min_clear") for m in ["GRL-SNAM","RigidA*","DeformA*","DWA/CBF"]},
    }
    with open(os.path.join(run_dir, f"{safe_name(split_name)}_aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    return dict(rows=rows, aggregate=agg)

def main():
    ap = argparse.ArgumentParser("Experiment 1: Main comparison (ID & OOD)")
    ap.add_argument("--ckpt", type=str, required=True, help="CoefEnergyNet checkpoint")
    ap.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","none"])
    ap.add_argument("--k_rad", type=float, default=0.05)
    ap.add_argument("--case_id", type=str, default="case1-tight")
    ap.add_argument("--case_ood", type=str, default="case2-dense")
    ap.add_argument("--n_envs", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=2312)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoefEnergyNet().to(device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state); model.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = mkdir(f"exp1_main_comparison/{stamp}")

    # splits
    id_results  = run_split("TestID",  get_case(args.case_id),  args, device, model, run_dir)
    ood_results = run_split("TestOOD", get_case(args.case_ood), args, device, model, run_dir)

    # combined summary
    def _pull(agg, key): return agg["aggregate"][key]
    overall = {
        "TestID":  id_results["aggregate"],
        "TestOOD": ood_results["aggregate"],
        "ckpt": args.ckpt, "alpha_mode": args.alpha_mode,
        "n_envs": args.n_envs, "n_trials": args.n_trials, "steps": args.steps
    }
    with open(os.path.join(run_dir, "summary_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\n=== DONE: Experiment 1 ===")
    print(f"Artifacts -> {run_dir}")

if __name__ == "__main__":
    main()