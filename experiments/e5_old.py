#!/usr/bin/env python3
"""
E5: Safety & Constraint Satisfaction — with DWA/CBF reactive baselines
and A* (rigid & deformable) oracles. Produces per-episode overlays and
aggregate safety/performance statistics suitable for a paper table.

Usage:
  python eval_E5_safety.py --ckpt checkpoints/best.pt --case case1-tight \
      --n_envs 3 --n_trials 5 --alpha_mode weight
"""

import os, json, argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt

# ==== your project imports (unchanged) ====
from train_coef_energy import CoefEnergyNet
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi
# ======================= DWA (fixed) =======================

from dataclasses import dataclass

@dataclass
class DWAParams:
    v_min: float = 0.0
    v_max: float = 2.0
    w_max: float = 2.5           # rad/s
    a_v: float = 2.0             # m/s^2
    a_w: float = 4.0             # rad/s^2
    dt_sim: float = 0.05         # rollout integrator step
    T_horizon: float = 1.5       # seconds
    nv: int = 7                  # samples in v
    nw: int = 13                 # samples in w
    # objective weights
    w_heading: float = 1.0
    w_clear: float = 0.8
    w_speed: float = 0.2
    # safety
    r_min: float = 0.2           # tube/no-penetration radius
    clear_fail: float = 1e-6     # forbid <= this clearance

def _wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def _clearance_point(xy, C, R):
    if C.size == 0: return np.inf
    return float(np.min(np.linalg.norm(C - xy[None,:], axis=1) - R))

def _traj_clearance(traj_xy, C, R):
    if len(traj_xy) == 0: return np.inf
    return float(min(_clearance_point(p, C, R) for p in traj_xy))

def _dynamic_window(v_cur, w_cur, p: DWAParams, dt):
    v_lo = max(p.v_min, v_cur - p.a_v * dt)
    v_hi = min(p.v_max, v_cur + p.a_v * dt)
    w_lo = max(-p.w_max, w_cur - p.a_w * dt)
    w_hi = min( p.w_max, w_cur + p.a_w * dt)
    return v_lo, v_hi, w_lo, w_hi

def _simulate(x0, y0, th0, v_cmd, w_cmd, p: DWAParams):
    """Kinematic unicycle rollout; returns [(x,y), ...], final (x,y,th)."""
    x, y, th = float(x0), float(y0), float(th0)
    traj = []
    nsteps = max(1, int(p.T_horizon / p.dt_sim))
    for _ in range(nsteps):
        x  += v_cmd * np.cos(th) * p.dt_sim
        y  += v_cmd * np.sin(th) * p.dt_sim
        th += w_cmd * p.dt_sim
        th  = _wrap_pi(th)
        traj.append((x,y))
    return np.asarray(traj, dtype=np.float32), (x, y, th)

def _score_trajectory(traj_xy, x_end, y_end, th_end, goal_xy, p: DWAParams, C, R, v_cmd):
    if len(traj_xy) == 0:
        return -np.inf, {"S_heading":0,"S_clear":0,"S_speed":0}
    # heading score in [0,1]: 1 when end heading points at goal
    bearing = np.arctan2(goal_xy[1] - y_end, goal_xy[0] - x_end)
    heading_err = abs(_wrap_pi(bearing - th_end))
    S_heading = 1.0 - (heading_err / np.pi)         # 1 for perfect align, 0 for opposite
    # clearance score in [0,1]: saturating with tanh relative to tube r_min
    cmin = _traj_clearance(traj_xy, C, R)
    S_clear = float(np.tanh(max(0.0, cmin) / (p.r_min + 1e-6)))
    # speed score
    S_speed = float(v_cmd / max(p.v_max, 1e-6))
    # final score
    score = (p.w_heading * S_heading) + (p.w_clear * S_clear) + (p.w_speed * S_speed)
    return score, {"S_heading":S_heading, "S_clear":S_clear, "S_speed":S_speed}

def dwa_select_control(state, goal_xy, C, R, p: DWAParams, dt_outer, v_cur=0.0, w_cur=0.0, debug=False):
    """
    state: (x,y,theta) in WORLD.
    returns: v*, w*, traj_xy*, dbg
    """
    x, y, th = float(state[0]), float(state[1]), float(state[2])
    v_lo, v_hi, w_lo, w_hi = _dynamic_window(v_cur, w_cur, p, dt_outer)
    Vs = np.linspace(v_lo, v_hi, max(2, p.nv))
    Ws = np.linspace(w_lo, w_hi, max(2, p.nw))

    best = {"score": -np.inf, "v": 0.0, "w": 0.0, "traj": np.zeros((0,2), np.float32),
            "end": (x,y,th), "terms": {}}
    for v in Vs:
        for w in Ws:
            traj, (xe, ye, the) = _simulate(x, y, th, v, w, p)
            cmin = _traj_clearance(traj, C, R)
            if cmin <= max(p.clear_fail, p.r_min):   # reject penetrations / too tight
                continue
            sc, terms = _score_trajectory(traj, xe, ye, the, goal_xy, p, C, R, v)
            if sc > best["score"]:
                best.update({"score": sc, "v": v, "w": w, "traj": traj, "end": (xe,ye,the), "terms": terms})
    if debug:
        print(f"[DWA] best score={best['score']:.3f} v={best['v']:.2f} w={best['w']:.2f} "
              f"S_heading={best['terms'].get('S_heading',0):.2f} S_clear={best['terms'].get('S_clear',0):.2f} "
              f"S_speed={best['terms'].get('S_speed',0):.2f}")
    return best["v"], best["w"], best["traj"], best

def run_dwa_episode(planner, world, cfg_local, params: DWAParams, steps: int, device="cpu", debug_first_steps=3):
    path = []
    min_clear = np.inf
    collisions = 0
    success = False
    v_cur = 0.0
    w_cur = 0.0

    for t in range(steps):
        sys = planner.sys
        x, y = float(sys.o[0].item()), float(sys.o[1].item())
        th   = float(sys.theta.item()) if hasattr(sys, "theta") else 0.0

        # select control
        v_cmd, w_cmd, traj_star, dbg = dwa_select_control(
            state=(x,y,th),
            goal_xy=cfg_local.goal,
            C=world.C_np, R=world.R_np,
            p=params, dt_outer=cfg_local.dt,
            v_cur=v_cur, w_cur=w_cur,
            debug=(t < debug_first_steps)
        )

        # apply to system
        vx = v_cmd * np.cos(th)
        vy = v_cmd * np.sin(th)
        sys.v_o = torch.tensor([vx, vy], dtype=sys.dtype)
        if hasattr(sys, "omega"):
            sys.omega = torch.tensor(w_cmd, dtype=sys.dtype)

        info = planner.step(cfg_local.dt, world)
        center_xy = np.asarray(info["center"], float)
        path.append(center_xy)

        v_cur, w_cur = v_cmd, w_cmd
        dc = _clearance_point(center_xy, world.C_np, world.R_np)
        min_clear = min(min_clear, dc)
        if dc < 0.0:
            collisions += 1

        if np.linalg.norm(center_xy - cfg_local.goal) <= cfg_local.goal_tol:
            success = True
            break

    return {
        "path": path,
        "success": success,
        "timesteps": len(path),
        "min_clearance": float(min_clear),
        "collisions": int(collisions),
    }


# ---------- utilities reused ----------
def mkdir(p): os.makedirs(p, exist_ok=True); return p

def path_length(path_xy):
    if len(path_xy) < 2: return 0.0
    diffs = np.diff(np.asarray(path_xy), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def clearance_to_circles(xy, C, R):
    if len(C) == 0: return np.inf
    d = np.linalg.norm(C - xy[None,:], axis=1) - R
    return float(np.min(d))

def clearance_profile(path_xy, C, R): return [clearance_to_circles(p, C, R) for p in path_xy]

def success_reached(xy, goal_xy, tol): return np.linalg.norm(np.asarray(xy)-np.asarray(goal_xy)) <= tol

def compute_workspace_bounds(C, R, margin=1.0):
    if len(C) == 0: return -5.0,5.0,-5.0,5.0
    cx, cy = C[:,0], C[:,1]
    xmin = float(np.min(cx - R) - margin)
    xmax = float(np.max(cx + R) + margin)
    ymin = float(np.min(cy - R) - margin)
    ymax = float(np.max(cy + R) + margin)
    return xmin, xmax, ymin, ymax

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
    pq=[]; g[si,sj]=0.0; heappush(pq,(h(si,sj),(si,sj))); vis=np.zeros_like(occ,bool)
    while pq:
        _,(i,j)=heappop(pq)
        if vis[i,j]: continue
        vis[i,j]=True
        if (i,j)==(gi,gj): break
        for di,dj,c in nbrs:
            ni, nj = i+di, j+dj
            if not(0<=ni<H and 0<=nj<W): continue
            if occ[ni,nj]: continue
            tentative = g[i,j] + c
            if tentative < g[ni,nj]:
                g[ni,nj]=tentative; came[ni,nj]=[i,j]
                heappush(pq,(tentative + h(ni,nj),(ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf,[]
    # reconstruct
    path_ij=[(gi,gj)]
    ci, cj = gi, gj
    while not(ci==si and cj==sj):
        pi, pj = came[ci,cj]
        if pi<0: break
        path_ij.append((pi,pj)); ci, cj = pi, pj
    path_ij.reverse()
    # metric length
    length_steps=0.0
    for k in range(1,len(path_ij)):
        i0,j0=path_ij[k-1]; i1,j1=path_ij[k]
        length_steps += np.hypot(i1-i0,j1-j0)
    length_m = length_steps * res
    path_xy = np.stack([ij_to_world(i,j,bounds,res) for (i,j) in path_ij], axis=0) if path_ij else []
    return float(length_m), path_xy

def clearance_field_on_grid(C, R, bounds, res):
    xmin, xmax, ymin, ymax = bounds
    W = int(np.ceil((xmax - xmin)/res)) + 1
    H = int(np.ceil((ymax - ymin)/res)) + 1
    yy = np.linspace(ymin, ymax, H)
    xx = np.linspace(xmin, xmax, W)
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    if len(C)==0: return np.full((H,W), np.inf, np.float32)
    clear = np.full((H,W), np.inf, np.float32)
    for (cx,cy), r in zip(C, R):
        d = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(np.float32) - float(r)
        clear = np.minimum(clear, d)
    return clear

def deformable_a_star_path(clearance, bounds, res, start_xy, goal_xy,
                           r_rest, r_min, lam=5.0, beta=0.5, eps=1e-3):
    from heapq import heappush, heappop
    H,W = clearance.shape
    si,sj = world_to_ij(start_xy,bounds,res)
    gi,gj = world_to_ij(goal_xy,bounds,res)
    if not(0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): return np.inf,[]
    feas = clearance >= r_min
    if not(feas[si,sj] and feas[gi,gj]): return np.inf,[]
    nbrs=[(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
          (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]
    def h(i,j): return np.hypot(i-gi,j-gj)*res
    def pen(c):
        alpha = max(0.0, (r_rest/(c+eps)) - 1.0)
        return lam*(alpha**2)
    g = np.full((H,W), np.inf, np.float64)
    came = np.full((H,W,2), -1, np.int32)
    pq=[]; g[si,sj]=0.0; heappush(pq,(h(si,sj),(si,sj)))
    vis=np.zeros((H,W),bool)
    while pq:
        _,(i,j)=heappop(pq)
        if vis[i,j]: continue
        vis[i,j]=True
        if (i,j)==(gi,gj): break
        if not feas[i,j]: continue
        pij = pen(clearance[i,j])
        for di,dj,cstep in nbrs:
            ni,nj=i+di,j+dj
            if not(0<=ni<H and 0<=nj<W): continue
            if not feas[ni,nj]: continue
            base = cstep*res
            deform = 0.5*beta*(pij + pen(clearance[ni,nj]))*base
            tentative = g[i,j] + (base + deform)
            if tentative < g[ni,nj]:
                g[ni,nj]=tentative; came[ni,nj]=[i,j]
                heappush(pq,(tentative + h(ni,nj),(ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf,[]
    path_ij=[(gi,gj)]
    ci,cj=gi,gj
    while not(ci==si and cj==sj):
        pi,pj=came[ci,cj]
        if pi<0: break
        path_ij.append((pi,pj)); ci,cj=pi,pj
    path_ij.reverse()
    length_m=0.0
    for k in range(1,len(path_ij)):
        i0,j0=path_ij[k-1]; i1,j1=path_ij[k]
        length_m += np.hypot(i1-i0,j1-j0)*res
    path_xy = np.stack([ij_to_world(i,j,bounds,res) for (i,j) in path_ij], axis=0) if path_ij else []
    return float(length_m), path_xy

# ---------- safety accounting ----------
def _soft_penalty(clearance, thresh, scale=1.0):
    gap = max(0.0, float(thresh) - float(clearance))
    return float(scale * (gap * gap))

def compute_safety_stats(path_xy, C, R, barrier_thresh, tube_thresh, penalty_scale=1.0):
    if len(path_xy)==0:
        return dict(min_clearance=np.inf, barrier=dict(viol_rate=0.0, penalty_sum=0.0),
                    tube=dict(viol_rate=0.0))
    clears = np.array([clearance_to_circles(p, C, R) for p in path_xy], float)
    barrier_mask = clears < float(barrier_thresh)
    tube_mask    = clears < float(tube_thresh)
    penalties = [_soft_penalty(c, barrier_thresh, penalty_scale) for c in clears]
    return dict(
        min_clearance=float(np.min(clears)) if clears.size else np.inf,
        barrier=dict(
            viol_rate=float(np.mean(barrier_mask)) if clears.size else 0.0,
            penalty_sum=float(np.sum(penalties))
        ),
        tube=dict(
            viol_rate=float(np.mean(tube_mask)) if clears.size else 0.0
        )
    )

# ---------- OUR policy rollout (CoefEnergyNet) ----------
def build_local_feats(o_w, goal_w, C_w, R_w, W_w):
    # tensors for robot & goal
    o = torch.as_tensor(o_w, dtype=torch.float32)
    g = torch.as_tensor(goal_w, dtype=torch.float32)

    # make sure we can safely use .size (lists -> arrays)
    C_np = np.asarray(C_w)
    R_np = np.asarray(R_w)
    W_np = np.asarray(W_w)

    # goal features: [dx, dy, dist, 1]
    dg = (g - o)
    gdist = torch.linalg.norm(dg).unsqueeze(0)
    goal_feats = torch.stack([dg[0], dg[1], gdist[0], torch.tensor(1.0)], dim=0).unsqueeze(0)

    # obstacle features or empty tensor if none
    if C_np.size == 0:
        # shape: (batch=1, obs=0, feat=6)
        obs_feats = torch.zeros((1, 0, 6), dtype=torch.float32)
    else:
        C = torch.as_tensor(C_np, dtype=torch.float32)                    # (N,2)
        R = torch.as_tensor(R_np, dtype=torch.float32)                    # (N,)
        W = torch.as_tensor(W_np, dtype=torch.float32)                    # (N,)
        dxdy = (g.unsqueeze(0) - C)                                       # (N,2)
        obs_feats = torch.cat([C, R.unsqueeze(-1), W.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)  # (1,N,6)

    return obs_feats, goal_feats

def map_alpha_to_world(W_in, R_in, alphas, mode, k_rad=0.05):
    al = np.maximum(alphas, 0.0)
    W_out = W_in.copy(); R_out = R_in.copy()
    if mode in ("weight","both") and al.size: W_out = W_out * al
    if mode in ("radius","both") and al.size: R_out = R_out + k_rad * al
    return W_out, R_out

def rollout_ours(model, cfg_local, world, alpha_mode="weight", k_rad=0.05, steps=800, device="cpu"):
    planner = gen.planner_from_cfg(cfg_local, world, cfg_local.k_bulk,
                                   cfg_local.gamma_s, cfg_local.d_hat, cfg_local.radius)
    path=[]; success=False
    for t in range(steps):
        sys = planner.sys
        o_w = sys.o.detach().cpu().numpy()
        Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)
        obs_feats, goal_feats = build_local_feats(o_w, cfg_local.goal, Cw, Rw, Ww)
        obs_mask = torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device) if obs_feats.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device)
        with torch.no_grad():
            alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))
        al_np  = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
        beta_f = float(beta.squeeze(0).item()); gamma_f=float(gamma.squeeze(0).item())
        W_adj, R_adj = Ww.copy(), Rw.copy()
        if alpha_mode!="none": W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, alpha_mode, k_rad)
        world_step = ssi.WorldObstacles(Cw, R_adj, W_adj, d_hat=cfg_local.d_hat)
        planner.stage_field.w_goal = max(0.0, beta_f)
        planner.sys.gamma_o = max(0.0, gamma_f)
        info = planner.step(cfg_local.dt, world_step)
        center_xy = np.asarray(info["center"], float)
        path.append(center_xy)
        if success_reached(center_xy, cfg_local.goal, cfg_local.goal_tol):
            success=True; break
    return path, success

# ---------- DWA baseline ----------
def dwa_rollout(start, goal, C, R, dt=0.05, steps=800,
                v_max=1.0, w_max=1.5, a_lin=1.0, a_ang=2.0,
                r_rest=0.5, w_clear=1.0, w_head=1.0, w_speed=0.2):
    """Point-robot DWA on (x,y,theta) with ring radius r_rest for clearance."""
    x = np.array([start[0], start[1], 0.0], float)
    path=[start.copy()]
    v=0.0; w=0.0
    for t in range(steps):
        # dynamic window
        v_low  = max(-v_max, v - a_lin*dt); v_high = min(v_max, v + a_lin*dt)
        w_low  = max(-w_max, w - a_ang*dt); w_high = min(w_max, w + a_ang*dt)
        v_cands = np.linspace(v_low, v_high, 5)
        w_cands = np.linspace(w_low, w_high, 7)
        best_score=-1e9; best=(v,w)
        for vc in v_cands:
            for wc in w_cands:
                # simulate short horizon
                x_sim = x.copy()
                min_clear = np.inf
                for _ in range(6):
                    x_sim[0] += vc*np.cos(x_sim[2])*dt
                    x_sim[1] += vc*np.sin(x_sim[2])*dt
                    x_sim[2] += wc*dt
                    c = clearance_to_circles(x_sim[:2], C, R + r_rest)
                    min_clear = min(min_clear, c)
                    if c <= 0.0: # collision in rollout
                        min_clear = -1.0
                        break
                # scores
                to_goal = goal - x_sim[:2]
                head = np.cos(np.arctan2(to_goal[1], to_goal[0]) - x_sim[2])
                score = (w_clear * min_clear) + (w_head * head) + (w_speed * abs(vc))
                if min_clear <= 0: score -= 1e3  # hard collision penalty
                if score > best_score:
                    best_score=score; best=(vc,wc)
        v,w = best
        # apply
        x[0] += v*np.cos(x[2])*dt
        x[1] += v*np.sin(x[2])*dt
        x[2] += w*dt
        path.append(x[:2].copy())
        if np.linalg.norm(x[:2]-goal) <= 0.5*r_rest:
            return path, True
    return path, False

# ---------- CBF baseline (QP-lite via half-space projection) ----------
def cbf_rollout(start, goal, C, R, dt=0.05, steps=800,
                v_max=1.0, r_rest=0.5, alpha=1.5):
    """
    Point kinematics x_{t+1}=x_t + v*dt. Choose v close to v_des toward goal,
    then project onto half-spaces {a_i^T v >= b_i} enforcing \dot h + alpha h >= 0,
    with h_i = ||x - c_i|| - (R_i + r_rest).
    """
    x = np.array(start, float)
    path=[x.copy()]
    for t in range(steps):
        to_goal = goal - x
        dist = np.linalg.norm(to_goal) + 1e-8
        v_des = (to_goal / dist) * min(v_max, dist/dt)  # cap
        # Build constraints A v >= b
        A=[]; b=[]
        for (ci, ri) in zip(C, R):
            dvec = x - ci
            d = np.linalg.norm(dvec) + 1e-8
            n = dvec / d
            h = d - (ri + r_rest)
            # h_dot = n^T v  (since \dot d = n^T v for point-mass)
            # constraint: n^T v + alpha*h >= 0
            A.append(n); b.append(-alpha*h)
        v = v_des.copy()
        # sequential projection onto violated half-spaces
        for (ai, bi) in zip(A, b):
            if np.dot(ai, v) < bi:
                # project: v := v + ((bi - ai^T v)/||ai||^2) * ai
                v = v + ((bi - np.dot(ai, v)) / (np.dot(ai, ai)+1e-12)) * ai
        # step
        v_norm = np.linalg.norm(v)
        if v_norm > v_max: v = v*(v_max/v_norm)
        x = x + v*dt
        path.append(x.copy())
        if np.linalg.norm(x - goal) <= 0.5*r_rest:
            return path, True
    return path, False

# ---------- plotting ----------
def plot_paths_overlay(outdir, world, start, goal, paths: Dict[str, np.ndarray]):
    plt.figure(figsize=(6.2,6.2))
    th = np.linspace(0,2*np.pi,128)
    for (cx,cy), r in zip(world.C_np, world.R_np):
        plt.plot(cx + r*np.cos(th), cy + r*np.sin(th), lw=1.1, alpha=0.85, color='k')
    colors = {
        "Ours":"tab:orange", "A* Rigid":"tab:blue", "A* Deform":"tab:green",
        "DWA":"tab:purple", "CBF":"tab:red"
    }
    styles = {"A* Rigid":"--", "A* Deform":"-.", "Ours":"-","DWA":"-","CBF":"-"}
    for name, P in paths.items():
        if P is None or len(P)==0: continue
        P = np.asarray(P)
        plt.plot(P[:,0], P[:,1], styles.get(name,"-"), lw=2.2,
                 label=name, alpha=0.95, color=colors.get(name,None))
    plt.scatter([start[0]],[start[1]], s=50, marker='s', label='Start', color='k')
    plt.scatter([goal[0]],[goal[1]], s=70, marker='*', label='Goal', color='crimson')
    plt.axis('equal'); plt.legend(frameon=False, loc='best')
    plt.title("Trajectories: Ours vs DWA/CBF and A* baselines")
    plt.tight_layout()
    out_png = os.path.join(outdir, "trajectory_overlay.png")
    plt.savefig(out_png, dpi=200); plt.close()
    return out_png

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("E5 Safety/Constraint eval with DWA/CBF baselines")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--case", type=str, default="case1-tight")
    ap.add_argument("--n_envs", type=int, default=3)
    ap.add_argument("--n_trials", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","none"])
    ap.add_argument("--k_rad", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load model (safe if available)
    model = CoefEnergyNet().to(device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = mkdir(f"snaps_E5/{stamp}")
    per_episode = []

    for env_id in range(args.n_envs):
        # ---------- environment ----------
        cfg = gen.GenCfg()
        cfg.seed = np.random.randint(1e9); gen.set_all_seeds(cfg.seed)
        if args.case.startswith("case1"):
            C, R, W = gen.sample_obstacles_case1_tight(cfg)
        elif args.case.startswith("case2"):
            C, R, W = gen.sample_obstacles_case2_harder(cfg)
        else:
            C, R, W = gen.sample_obstacles_case1_tight(cfg)
        world = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)

        # bounds/grid for A*
        r_rest = getattr(cfg, "radius", cfg.d_hat)                # nominal ring radius
        r_min  = getattr(cfg, "radius_min", 0.3 * r_rest)         # min feasible radius
        bounds = compute_workspace_bounds(world.C_np, world.R_np, margin=2.0 * max(r_rest, cfg.d_hat))
        res    = max(r_rest, cfg.d_hat) * 0.5

        rng = np.random.default_rng(cfg.seed + 1234)

        for trial in range(args.n_trials):
            # ---------- start/goal sampling (valid & separated) ----------
            def sample_valid(margin):
                for _ in range(1000):
                    x = rng.uniform(bounds[0], bounds[1])
                    y = rng.uniform(bounds[2], bounds[3])
                    if clearance_to_circles(np.array([x, y]), world.C_np, world.R_np) >= margin:
                        return np.array([x, y], float)
                return np.array([bounds[0] + 1, bounds[2] + 1], float)

            start = sample_valid(r_min)
            goal  = sample_valid(r_min)
            if np.linalg.norm(start - goal) < 2.0 * r_rest:
                goal = goal + np.array([2.0 * r_rest, 0.0])

            ep_dir = mkdir(os.path.join(run_dir, f"env{env_id:03d}_trial{trial:03d}"))

            # ---------- A* baselines ----------
            occ_rigid, _, _ = rasterize_inflated_obstacles(world.C_np, world.R_np, r_rest, bounds, res)
            L_star_rigid, path_star_rigid = a_star_path(occ_rigid, bounds, res, start, goal)

            clearance_grid = clearance_field_on_grid(world.C_np, world.R_np, bounds, res)
            L_star_deform, path_star_deform = deformable_a_star_path(
                clearance_grid, bounds, res, start, goal,
                r_rest=r_rest, r_min=r_min, lam=5.0, beta=0.6
            )

            # ---------- OURS rollout ----------
            cfg_local = gen.GenCfg()
            cfg_local.d_hat     = cfg.d_hat
            cfg_local.radius    = r_rest
            cfg_local.k_bulk    = getattr(cfg, "k_bulk", 20.0)
            cfg_local.gamma_s   = getattr(cfg, "gamma_s", 0.05)
            cfg_local.dt        = getattr(cfg, "dt", 0.01)
            cfg_local.total_steps = args.steps
            cfg_local.start     = start.astype(np.float32)
            cfg_local.goal      = goal.astype(np.float32)

            path_ours, succ_ours = rollout_ours(
                model, cfg_local, world,
                alpha_mode=args.alpha_mode, k_rad=args.k_rad,
                steps=args.steps, device=device
            )

            # ---------- DWA rollout ----------
            # fresh planner/system (so it doesn't inherit our policy's internal state)
            planner_dwa = gen.planner_from_cfg(cfg_local, world, cfg_local.k_bulk,
                                               cfg_local.gamma_s, cfg_local.d_hat, cfg_local.radius)
            dwa_params = DWAParams()  # use your tuned values inside this dataclass
            dwa_out = run_dwa_episode(
                planner=planner_dwa, world=world, cfg_local=cfg_local,
                params=dwa_params, steps=args.steps, device=device
            )
            path_dwa, succ_dwa = dwa_out["path"], dwa_out["success"]

            # ---------- CBF rollout ----------
            path_cbf, succ_cbf = cbf_rollout(
                start, goal, world.C_np, world.R_np,
                dt=cfg_local.dt, steps=args.steps, v_max=1.0,
                r_rest=r_rest, alpha=1.5
            )

            # ---------- metrics ----------
            def metrics_for(path, succ, L_ref):
                L_exec = path_length(path)
                min_clear = float("nan") if len(path) == 0 else \
                    min(clearance_profile(path, world.C_np, world.R_np))
                safety = compute_safety_stats(
                    path, world.C_np, world.R_np,
                    barrier_thresh=float(cfg_local.d_hat),
                    tube_thresh=float(r_min)
                )
                spl = 0.0
                if L_ref is not None and L_ref > 0 and succ:
                    spl = float(L_ref / max(L_ref, L_exec))
                prog_T = float(np.linalg.norm(start - goal) -
                               np.linalg.norm((path[-1] if path else start) - goal))
                return dict(
                    success=bool(succ),
                    path_length=float(L_exec),
                    min_clearance=min_clear,
                    barrier_viol_rate=safety["barrier"]["viol_rate"],
                    barrier_penalty_sum=safety["barrier"]["penalty_sum"],
                    tube_viol_rate=safety["tube"]["viol_rate"],
                    SPL=spl,
                    progress_at_T=prog_T,
                )

            L_ref = (
                L_star_deform if np.isfinite(L_star_deform)
                else (L_star_rigid if np.isfinite(L_star_rigid) else None)
            )

            m_ours = metrics_for(path_ours, succ_ours, L_ref)
            m_dwa  = metrics_for(path_dwa,  succ_dwa,  L_ref)
            m_cbf  = metrics_for(path_cbf,  succ_cbf,  L_ref)

            # ---------- overlay ----------
            overlay_png = plot_paths_overlay(
                ep_dir, world, start, goal,
                paths={
                    "Ours": np.array(path_ours) if len(path_ours) else None,
                    "A* Rigid": path_star_rigid if len(path_star_rigid) else None,
                    "A* Deform": path_star_deform if len(path_star_deform) else None,
                    "DWA": np.array(path_dwa) if len(path_dwa) else None,
                    "CBF": np.array(path_cbf) if len(path_cbf) else None,
                }
            )

            per_episode.append({
                "env_id": env_id, "trial": trial, "case": args.case,
                "start": start.tolist(), "goal": goal.tolist(),
                "astar_rigid_len": float(L_star_rigid) if np.isfinite(L_star_rigid) else float("inf"),
                "astar_deform_len": float(L_star_deform) if np.isfinite(L_star_deform) else float("inf"),
                "ours": m_ours, "dwa": m_dwa, "cbf": m_cbf,
                "overlay": overlay_png
            })

    # ---------- aggregation ----------
    def agg(key, method):
        vals = []
        for ep in per_episode:
            v = ep[method].get(key, None)
            if v is None: continue
            vals.append(float(v))
        return float(np.nanmean(vals)) if vals else float("nan")

    aggregate = {
        "n_envs": args.n_envs, "n_trials": args.n_trials, "case": args.case,
        # success
        "ours_success_rate": agg("success","ours"),
        "dwa_success_rate":  agg("success","dwa"),
        "cbf_success_rate":  agg("success","cbf"),
        # safety
        "ours_barrier_viol_rate_mean": agg("barrier_viol_rate","ours"),
        "dwa_barrier_viol_rate_mean":  agg("barrier_viol_rate","dwa"),
        "cbf_barrier_viol_rate_mean":  agg("barrier_viol_rate","cbf"),
        "ours_tube_viol_rate_mean":    agg("tube_viol_rate","ours"),
        "dwa_tube_viol_rate_mean":     agg("tube_viol_rate","dwa"),
        "cbf_tube_viol_rate_mean":     agg("tube_viol_rate","cbf"),
        "ours_barrier_penalty_sum_mean": agg("barrier_penalty_sum","ours"),
        # efficiency-ish
        "ours_SPL_mean": agg("SPL","ours"),
        "dwa_SPL_mean":  agg("SPL","dwa"),
        "cbf_SPL_mean":  agg("SPL","cbf"),
        "ours_progress_at_T_mean": agg("progress_at_T","ours"),
        "dwa_progress_at_T_mean":  agg("progress_at_T","dwa"),
        "cbf_progress_at_T_mean":  agg("progress_at_T","cbf"),
    }

    out_dir = mkdir(os.path.join(run_dir, "results"))
    with open(os.path.join(out_dir, "per_episode.json"), "w") as f:
        json.dump(per_episode, f, indent=2)
    with open(os.path.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    print("\n=== E5 Safety & Constraint Satisfaction (with DWA/CBF) ===")
    print(json.dumps(aggregate, indent=2))
    print(f"Artifacts: {run_dir}")

if __name__ == "__main__":
    main()
