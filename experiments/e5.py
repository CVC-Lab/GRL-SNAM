#!/usr/bin/env python3
"""
E5: Safety & Constraint Satisfaction — Comprehensive Evaluation
Methods: GRL-SNAM, Rigid A*, Deformable A*, Potential Field, CBF, DWA

Focus: Safety constraints, barrier violations, clearance profiles, constraint satisfaction
with fair local information constraints for reactive methods.

Usage:
  python eval_E5_safety_comprehensive.py --ckpt checkpoints/best.pt --case case1-tight \
      --n_envs 5 --n_trials 8 --alpha_mode weight
"""

import os, json, argparse, csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import scipy.optimize
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt

# ==== project imports ====
from train_coef_energy import CoefEnergyNet
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi
from eval_coef_energy import HistSecantController
import re

def mkdir(p): os.makedirs(p, exist_ok=True); return p
def safe_name(s: str) -> str: return re.sub(r'[^0-9A-Za-z._-]+', '_', s)

def _to_serializable_path(p):
    if p is None:
        return None
    try:
        # Works for numpy arrays *and* lists-of-lists
        return np.asarray(p).tolist()
    except Exception:
        # Last resort: leave as-is if it’s already JSON-friendly
        return p

# ========= shared utilities =========
def compute_workspace_bounds(C, R, margin=1.0):
    if len(C) == 0: return -5.0, 5.0, -5.0, 5.0
    cx, cy = C[:,0], C[:,1]
    return (float(np.min(cx - R) - margin), float(np.max(cx + R) + margin),
            float(np.min(cy - R) - margin), float(np.max(cy + R) + margin))

def world_to_ij(xy, bounds, res):
    xmin, xmax, ymin, ymax = bounds
    jf = (xy[0] - xmin) / res; if_ = (xy[1] - ymin) / res
    return (int(np.floor(np.clip(if_, 0, np.floor((ymax - ymin)/res)))), 
            int(np.floor(np.clip(jf, 0, np.floor((xmax - xmin)/res)))))

def ij_to_world(i, j, bounds, res):
    xmin, ymin = bounds[0], bounds[2]
    return np.array([xmin + j*res, ymin + i*res], dtype=np.float32)

def path_length(path_xy):
    if len(path_xy) < 2: return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))

def clearance_to_circles(xy, C, R):
    if len(C) == 0: return np.inf
    return float(np.min(np.linalg.norm(C - xy[None,:], axis=1) - R))

def success_reached(last_center, goal_xy, tol):
    return np.linalg.norm(np.asarray(last_center) - np.asarray(goal_xy)) <= tol

def compute_smoothness(path_xy):
    """Compute path smoothness as average turning angle"""
    if len(path_xy) < 3: return 0.0
    P = np.array(path_xy)
    v = P[1:] - P[:-1]
    v_norm = np.linalg.norm(v, axis=1)
    v_norm[v_norm < 1e-6] = 1e-6
    v_unit = v / v_norm[:, None]
    cos_angles = np.sum(v_unit[:-1] * v_unit[1:], axis=1)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles)
    return float(np.mean(angles))

# ========= Safety-focused metrics =========
def _soft_penalty(clearance: float, thresh: float, scale: float = 1.0) -> float:
    gap = max(0.0, thresh - float(clearance))
    return float(scale * (gap * gap))

def compute_comprehensive_safety_stats(path_xy, C, R, barrier_thresh, tube_thresh, penalty_scale=1.0):
    """Comprehensive safety statistics focused on constraint violations"""
    T = len(path_xy)
    if T == 0:
        return dict(
            n_steps=0, min_clearance=np.inf, mean_clearance=np.inf,
            barrier=dict(viol_count=0, viol_rate=0.0, first_viol_step=None, 
                        penalty_sum=0.0, penalty_mean=0.0, max_viol_depth=0.0),
            tube=dict(viol_count=0, viol_rate=0.0, first_viol_step=None,
                     max_viol_depth=0.0),
            collision=dict(count=0, rate=0.0, first_collision_step=None),
            clearance_std=0.0,
            safety_margin_score=1.0  # perfect safety for empty path
        )
    
    # Compute clearance profile
    clears = np.array([clearance_to_circles(np.asarray(xy,float), C, R) for xy in path_xy])
    
    # Barrier violations (sensing/planning horizon)
    barrier_mask = clears < float(barrier_thresh)
    barrier_viols = np.where(barrier_mask)[0]
    barrier_depths = np.maximum(0, barrier_thresh - clears[barrier_mask])
    
    # Tube violations (minimum feasible radius)
    tube_mask = clears < float(tube_thresh)
    tube_viols = np.where(tube_mask)[0]  
    tube_depths = np.maximum(0, tube_thresh - clears[tube_mask])
    
    # Collision violations (actual penetration)
    collision_mask = clears < 0.0
    collision_viols = np.where(collision_mask)[0]
    
    # Penalty computation
    penalties = [_soft_penalty(c, barrier_thresh, penalty_scale) for c in clears]
    
    # Safety margin score (how much "safety buffer" is maintained on average)
    safe_clears = clears[clears > 0]  # only consider non-penetrating points
    if len(safe_clears) > 0:
        # Normalized by barrier threshold: 1.0 = always at barrier, >1.0 = safer
        safety_margin_score = float(np.mean(safe_clears) / max(barrier_thresh, 1e-6))
    else:
        safety_margin_score = 0.0  # all penetrating
    
    return dict(
        n_steps=T,
        min_clearance=float(np.min(clears)),
        mean_clearance=float(np.mean(clears)),
        clearance_std=float(np.std(clears)),
        safety_margin_score=safety_margin_score,
        
        barrier=dict(
            viol_count=int(len(barrier_viols)),
            viol_rate=float(len(barrier_viols)) / T,
            first_viol_step=int(barrier_viols[0]) if len(barrier_viols) > 0 else None,
            penalty_sum=float(np.sum(penalties)),
            penalty_mean=float(np.mean(penalties)),
            max_viol_depth=float(np.max(barrier_depths)) if len(barrier_depths) > 0 else 0.0,
        ),
        
        tube=dict(
            viol_count=int(len(tube_viols)),
            viol_rate=float(len(tube_viols)) / T,
            first_viol_step=int(tube_viols[0]) if len(tube_viols) > 0 else None,
            max_viol_depth=float(np.max(tube_depths)) if len(tube_depths) > 0 else 0.0,
        ),
        
        collision=dict(
            count=int(len(collision_viols)),
            rate=float(len(collision_viols)) / T,
            first_collision_step=int(collision_viols[0]) if len(collision_viols) > 0 else None,
        ),
    )

def safety_metric_pack(path_xy, L_ref, *, reached: bool, C, R, barrier_thresh: float, tube_thresh: float):
    """Comprehensive metrics pack with safety focus"""
    L_exec = path_length(path_xy)
    S = 1.0 if reached else 0.0
    
    if (L_ref is None) or not (L_ref > 0 and np.isfinite(L_ref)):
        spl, detour = np.nan, np.nan
    else:
        spl = S * (L_ref / max(L_ref, L_exec))
        detour = L_exec / L_ref
    
    safety = compute_comprehensive_safety_stats(path_xy, C, R, barrier_thresh, tube_thresh)
    
    return dict(
        # Performance metrics
        success=float(S), 
        path_length=L_exec,
        SPL=float(spl) if np.isfinite(spl) else np.nan,
        detour=float(detour) if np.isfinite(detour) else np.nan,
        smoothness=compute_smoothness(path_xy),
        
        # Core safety metrics  
        min_clearance=safety["min_clearance"],
        mean_clearance=safety["mean_clearance"],
        clearance_std=safety["clearance_std"],
        safety_margin_score=safety["safety_margin_score"],
        
        # Constraint violation metrics
        barrier_viol_rate=safety["barrier"]["viol_rate"],
        barrier_penalty_sum=safety["barrier"]["penalty_sum"],
        tube_viol_rate=safety["tube"]["viol_rate"],
        collision_rate=safety["collision"]["rate"],
        
        # Timing/criticality metrics
        first_barrier_viol_step=safety["barrier"]["first_viol_step"],
        first_tube_viol_step=safety["tube"]["first_viol_step"],
        first_collision_step=safety["collision"]["first_collision_step"],
        
        computational_time=0.0  # Will be filled by timing wrapper
    )

# ========= A* implementations =========
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
    g = np.full((H,W), np.inf)
    came = np.full((H,W,2), -1, dtype=int)
    h = lambda i,j: np.hypot(i-gi, j-gj) * res
    
    pq = [(h(si,sj), (si,sj))]
    g[si,sj] = 0.0
    visited = np.zeros_like(occ, dtype=bool)
    
    while pq:
        _, (i,j) = heappop(pq)
        if visited[i,j]: continue
        visited[i,j] = True
        if (i,j) == (gi,gj): break
        
        for di,dj,c in nbrs:
            ni, nj = i+di, j+dj
            if not (0<=ni<H and 0<=nj<W) or occ[ni,nj]: continue
            tentative = g[i,j] + c * res
            if tentative < g[ni,nj]:
                g[ni,nj] = tentative
                came[ni,nj] = [i,j]
                heappush(pq, (tentative + h(ni,nj), (ni,nj)))
    
    if not np.isfinite(g[gi,gj]): return np.inf, []
    
    path_ij = [(gi,gj)]
    ci, cj = gi, gj
    while not (ci==si and cj==sj):
        pi, pj = came[ci,cj]
        if pi < 0: break
        path_ij.append((pi,pj))
        ci, cj = pi, pj
    path_ij.reverse()
    
    path_xy = np.array([ij_to_world(i,j,bounds,res) for (i,j) in path_ij])
    return g[gi,gj], path_xy

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
        d = np.sqrt((X - cx)**2 + (Y - cy)**2) - r
        clear = np.minimum(clear, d)
    return clear

def deformable_a_star_path(clearance, bounds, res, start_xy, goal_xy, r_rest, r_min, lam=5.0, beta=0.6, eps=1e-3):
    from heapq import heappush, heappop
    H, W = clearance.shape
    si, sj = world_to_ij(start_xy, bounds, res)
    gi, gj = world_to_ij(goal_xy, bounds, res)
    
    if not (0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): return np.inf, []
    
    feas = clearance >= r_min
    if not (feas[si,sj] and feas[gi,gj]): return np.inf, []
    
    nbrs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
            (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]
    
    h = lambda i,j: np.hypot(i-gi, j-gj) * res
    pen = lambda c: lam * max(0, (r_rest / (c + eps)) - 1)**2
    
    g = np.full((H,W), np.inf)
    came = np.full((H,W,2), -1, dtype=int)
    pq = [(h(si,sj), (si,sj))]
    g[si,sj] = 0.0
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
            deform_cost = beta * 0.5 * (pij + pen(clearance[ni,nj])) * base
            tentative = g[i,j] + base + deform_cost
            
            if tentative < g[ni,nj]:
                g[ni,nj] = tentative
                came[ni,nj] = [i,j]
                heappush(pq, (tentative + h(ni,nj), (ni,nj)))
    
    if not np.isfinite(g[gi,gj]): return np.inf, []
    
    path_ij = [(gi,gj)]
    ci, cj = gi, gj
    while not (ci==si and cj==sj):
        pi, pj = came[ci,cj]
        if pi < 0: break
        path_ij.append((pi,pj))
        ci, cj = pi, pj
    path_ij.reverse()
    
    path_xy = np.array([ij_to_world(i,j,bounds,res) for (i,j) in path_ij])
    return g[gi,gj], path_xy

# ========= Stage-aware baseline implementations =========
def create_stage_manager_for_baselines(start_xy, goal_xy, stage_size=(2.6, 2.0), overlap=0.3, obstacles=None, inflate=0.35):
    return ssi.StageManager(
        np.array(start_xy), np.array(goal_xy),
        stage_size=stage_size, overlap_ratio=overlap,
        obstacles=obstacles, inflate=inflate
    )

def get_local_obstacles(stage_manager, C_global, R_global, W_global=None):
    st = stage_manager.current()
    xmin, xmax, ymin, ymax = st.bounds
    mask = ((C_global[:,0] >= xmin-0.5) & (C_global[:,0] <= xmax+0.5) & 
            (C_global[:,1] >= ymin-0.5) & (C_global[:,1] <= ymax+0.5))
    
    if np.any(mask):
        C_local = C_global[mask]
        R_local = R_global[mask] 
        W_local = W_global[mask] if W_global is not None else None
        return C_local, R_local, W_local
    else:
        return np.zeros((0,2)), np.zeros((0)), np.zeros((0))

# ========= Potential Field (Stage-aware) =========
def potential_field_path_staged(C_global, R_global, start_xy, goal_xy, *, 
                               stage_size=(2.6, 2.0), overlap=0.3, inflate=0.35,
                               dt=0.02, steps=1500, v_max=0.7,
                               k_goal=1.5, k_rep=0.9, d_safe=0.8, r_min=0.3):
    stage_manager = create_stage_manager_for_baselines(
        start_xy, goal_xy, stage_size, overlap, 
        obstacles=(C_global, R_global), inflate=inflate
    )
    
    x = np.array(start_xy, dtype=np.float32)
    path = [x.copy()]
    
    for t in range(steps):
        C_local, R_local, _ = get_local_obstacles(stage_manager, C_global, R_global)
        current_stage = stage_manager.current()
        
        if stage_manager.current_stage_idx < len(stage_manager.stages) - 1:
            target = current_stage.exit_point
        else:
            target = goal_xy
            
        to_target = target - x
        dist_target = np.linalg.norm(to_target) + 1e-6
        F_att = k_goal * (to_target / dist_target)
        
        F_rep = np.zeros(2, dtype=np.float32)
        for (cx, cy), r in zip(C_local, R_local):
            obs_pos = np.array([cx, cy], dtype=np.float32)
            to_obs = x - obs_pos
            dist_obs = np.linalg.norm(to_obs) + 1e-6
            clearance = dist_obs - r
            
            if clearance < d_safe:
                magnitude = k_rep * ((d_safe - clearance) / clearance**2) if clearance > 1e-3 else k_rep * 1000
                direction = to_obs / dist_obs
                F_rep += magnitude * direction
        
        # Stage boundary forces
        xmin, xmax, ymin, ymax = current_stage.bounds
        margin = 0.3
        F_boundary = np.zeros(2, dtype=np.float32)
        if x[0] < xmin + margin: F_boundary[0] += (xmin + margin - x[0]) * 2.0
        if x[0] > xmax - margin: F_boundary[0] -= (x[0] - (xmax - margin)) * 2.0  
        if x[1] < ymin + margin: F_boundary[1] += (ymin + margin - x[1]) * 2.0
        if x[1] > ymax - margin: F_boundary[1] -= (x[1] - (ymax - margin)) * 2.0
        
        v = F_att + F_rep + F_boundary
        
        speed = np.linalg.norm(v)
        if speed > v_max:
            v = v * (v_max / speed)
        
        if len(C_local) > 0:
            min_clearance = clearance_to_circles(x, C_local, R_local)
            if min_clearance < r_min:
                v *= 0.1
        
        x = x + dt * v
        path.append(x.copy())
        stage_manager.advance_if_needed(x)
        
        if np.linalg.norm(x - goal_xy) <= 0.2:
            break
    
    return np.array(path, dtype=np.float32)

# ========= CBF (Stage-aware) =========
class CBFControllerStaged:
    def __init__(self, gamma=1.0, safety_margin=0.1):
        self.gamma = gamma
        self.safety_margin = safety_margin
        self.u_max = 2.0
        self.C_local = np.zeros((0,2))
        self.R_local = np.zeros(0)
    
    def update_local_obstacles(self, C_local, R_local):
        self.C_local = np.array(C_local) if len(C_local) > 0 else np.zeros((0,2))
        self.R_local = np.array(R_local) + self.safety_margin if len(R_local) > 0 else np.zeros(0)
    
    def barrier_function(self, x):
        if len(self.C_local) == 0:
            return np.inf
        distances = np.linalg.norm(self.C_local - x, axis=1)
        return np.min(distances - self.R_local)
    
    def barrier_gradient(self, x):
        if len(self.C_local) == 0:
            return np.zeros(2)
        
        distances = np.linalg.norm(self.C_local - x, axis=1)
        clearances = distances - self.R_local
        min_idx = np.argmin(clearances)
        
        direction = (x - self.C_local[min_idx]) / distances[min_idx]
        return direction
    
    def safe_control(self, x, u_nominal):
        h = self.barrier_function(x)
        grad_h = self.barrier_gradient(x)
        
        if h > 0 and np.dot(grad_h, u_nominal) >= -self.gamma * h:
            return u_nominal
        
        try:
            def objective(u):
                return np.sum((u - u_nominal)**2)
            
            def constraint(u):
                return np.dot(grad_h, u) + self.gamma * h
            
            bounds = [(-self.u_max, self.u_max), (-self.u_max, self.u_max)]
            cons = {'type': 'ineq', 'fun': constraint}
            
            result = scipy.optimize.minimize(
                objective, u_nominal, method='SLSQP', 
                bounds=bounds, constraints=cons
            )
            
            return result.x if result.success else u_nominal * 0.1
            
        except Exception:
            if np.dot(grad_h, grad_h) > 1e-6:
                violation = -self.gamma * h - np.dot(grad_h, u_nominal)
                if violation > 0:
                    correction = (violation / np.dot(grad_h, grad_h)) * grad_h
                    return u_nominal + correction
            return u_nominal * 0.1

def cbf_path_staged(C_global, R_global, start_xy, goal_xy, *, 
                   stage_size=(2.6, 2.0), overlap=0.3, inflate=0.35,
                   dt=0.02, steps=1500, k_goal=2.0, gamma_cbf=1.0, safety_margin=0.15):
    stage_manager = create_stage_manager_for_baselines(
        start_xy, goal_xy, stage_size, overlap,
        obstacles=(C_global, R_global), inflate=inflate
    )
    
    controller = CBFControllerStaged(gamma=gamma_cbf, safety_margin=safety_margin)
    x = np.array(start_xy, dtype=np.float32)
    path = [x.copy()]
    
    for t in range(steps):
        C_local, R_local, _ = get_local_obstacles(stage_manager, C_global, R_global)
        controller.update_local_obstacles(C_local, R_local)
        
        current_stage = stage_manager.current()
        
        if stage_manager.current_stage_idx < len(stage_manager.stages) - 1:
            target = current_stage.exit_point
        else:
            target = goal_xy
            
        to_target = target - x
        dist_target = np.linalg.norm(to_target) + 1e-6
        u_nominal = k_goal * (to_target / dist_target)
        
        u_safe = controller.safe_control(x, u_nominal)
        
        x = x + dt * u_safe
        path.append(x.copy())
        
        stage_manager.advance_if_needed(x)
        
        if np.linalg.norm(x - goal_xy) <= 0.2:
            break
    
    return np.array(path, dtype=np.float32)

# ========= DWA (Stage-aware) =========
class DWAPlannerStaged:
    def __init__(self, dt=0.02, predict_time=2.0):
        self.dt = dt
        self.predict_time = predict_time
        self.predict_steps = int(predict_time / dt)
        
        self.v_max = 1.2
        self.omega_max = 1.5
        self.v_acc_max = 2.0
        self.omega_acc_max = 3.0
        
        self.v_resolution = 0.1
        self.omega_resolution = 0.1
        
        self.w_heading = 0.4
        self.w_distance = 0.3  
        self.w_velocity = 0.3
        self.w_clear = 0.5
        self.w_bounds = 2.0
        
        self.safety_margin = 0.2
        self.eps = 1e-6
        
        self.C_local = np.zeros((0,2))
        self.R_local = np.zeros(0)
    
    def update_local_obstacles(self, C_local, R_local):
        self.C_local = np.array(C_local) if len(C_local) > 0 else np.zeros((0,2))
        self.R_local = np.array(R_local) if len(R_local) > 0 else np.zeros(0)
    
    def dynamic_window(self, v_current, omega_current):
        v_min = max(0, v_current - self.v_acc_max * self.dt)
        v_max = min(self.v_max, v_current + self.v_acc_max * self.dt)
        
        omega_min = max(-self.omega_max, omega_current - self.omega_acc_max * self.dt)
        omega_max = min(self.omega_max, omega_current + self.omega_acc_max * self.dt)
        
        return v_min, v_max, omega_min, omega_max
    
    def simulate_trajectory(self, x, theta, v, omega):
        trajectory = []
        curr_x, curr_theta = x.copy(), theta
        
        for _ in range(self.predict_steps):
            curr_x += np.array([v * np.cos(curr_theta), v * np.sin(curr_theta)]) * self.dt
            curr_theta += omega * self.dt
            trajectory.append(curr_x.copy())
        
        return np.array(trajectory)
    
    def trajectory_cost(self, trajectory, target, v, omega, stage_bounds, theta0):
        if len(trajectory) == 0:
            return np.inf

        xmin, xmax, ymin, ymax = stage_bounds
        min_clear = np.inf
        bound_pen = 0.0

        for pt in trajectory:
            # Stage boundary penalty
            dxl = max(0.0, xmin - pt[0])
            dxr = max(0.0, pt[0] - xmax)
            dyl = max(0.0, ymin - pt[1])
            dyr = max(0.0, pt[1] - ymax)
            bound_pen += (dxl*dxl + dxr*dxr + dyl*dyl + dyr*dyr)

            # Local obstacle clearance
            if len(self.C_local) > 0:
                clr = clearance_to_circles(pt, self.C_local, self.R_local)
                min_clear = min(min_clear, clr)
                if clr <= 0.0:  # Hard collision rejection
                    return np.inf

        # Heading cost
        theta_f = theta0 + omega * self.predict_time
        final_pos = trajectory[-1]
        to_target = target - final_pos
        phi = np.arctan2(to_target[1], to_target[0])
        heading_err = np.arctan2(np.sin(phi - theta_f), np.cos(phi - theta_f))

        # Distance and velocity costs
        dist_cost = np.linalg.norm(to_target)
        vel_cost = (self.v_max - abs(v)) / self.v_max

        # Clearance cost (reward higher clearance)
        clear_cost = self.w_clear / (self.eps + max(min_clear, self.eps))

        total = (self.w_heading * abs(heading_err) +
                 self.w_distance * dist_cost +
                 self.w_velocity * vel_cost +
                 self.w_bounds * bound_pen +
                 clear_cost)
        return total
    
    def plan_step(self, pos, theta, v_current, omega_current, target, stage_bounds):
        v_min, v_max, om_min, om_max = self.dynamic_window(v_current, omega_current)
        
        # Grid sampling
        v_samples = np.linspace(v_min, v_max, max(2, int((v_max-v_min)/self.v_resolution)+1))
        om_samples = np.linspace(om_min, om_max, max(2, int((om_max-om_min)/self.omega_resolution)+1))

        best_cost = np.inf
        best_v, best_om = 0.0, 0.0

        for v in v_samples:
            for om in om_samples:
                traj = self.simulate_trajectory(pos, theta, v, om)
                cost = self.trajectory_cost(traj, target, v, om, stage_bounds, theta)
                if cost < best_cost:
                    best_cost, best_v, best_om = cost, v, om

        # Fallback for tight situations
        if not np.isfinite(best_cost):
            dir_vec = target - pos
            d = np.linalg.norm(dir_vec) + 1e-6
            return min(self.v_resolution, self.v_max*0.3), 0.0

        return best_v, best_om

def dwa_path_staged(C_global, R_global, start_xy, goal_xy, *, 
                   stage_size=(2.6, 2.0), overlap=0.3, inflate=0.35,
                   dt=0.02, steps=1500):
    stage_manager = create_stage_manager_for_baselines(
        start_xy, goal_xy, stage_size, overlap,
        obstacles=(C_global, R_global), inflate=inflate
    )
    
    planner = DWAPlannerStaged(dt=dt)
    
    x = np.array(start_xy, dtype=np.float32)
    theta = np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
    v, omega = 0.0, 0.0
    
    path = [x.copy()]
    
    for t in range(steps):
        C_local, R_local, _ = get_local_obstacles(stage_manager, C_global, R_global)
        planner.update_local_obstacles(C_local, R_local)
        
        current_stage = stage_manager.current()
        
        if stage_manager.current_stage_idx < len(stage_manager.stages) - 1:
            target = current_stage.exit_point
        else:
            target = goal_xy
        
        v, omega = planner.plan_step(x, theta, v, omega, target, current_stage.bounds)
        
        x += np.array([v * np.cos(theta), v * np.sin(theta)]) * dt
        theta += omega * dt
        
        path.append(x.copy())
        stage_manager.advance_if_needed(x)
        
        if np.linalg.norm(x - goal_xy) <= 0.2:
            break
    
    return np.array(path, dtype=np.float32)

# ========= GRL-SNAM rollout =========
def build_local_feats(o_w, goal_w, C_w, R_w, W_w):
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

def map_alpha_to_world(W_in, R_in, alphas, mode, k_rad=0.05):
    al = np.maximum(alphas, 0.0)
    W_out = W_in.copy(); R_out = R_in.copy()
    if mode in ("weight","both") and al.size: W_out = W_out * al
    if mode in ("radius","both") and al.size: R_out = R_out + k_rad * al
    return W_out, R_out

def rollout_grl_snam(model, cfg_local, world, alpha_mode="weight", k_rad=0.05, steps=1500, device="cpu"):
    planner = gen.planner_from_cfg(cfg_local, world, cfg_local.k_bulk,
                                   cfg_local.gamma_s, cfg_local.d_hat, cfg_local.radius)
    path = []
    new_stage_timer = 0
    
    for t in range(steps):
        sys = planner.sys
        o_w = sys.o.detach().cpu().numpy()
        Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)
        obs_feats, goal_feats = build_local_feats(o_w, cfg_local.goal, Cw, Rw, Ww)
        obs_mask = torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device) if obs_feats.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))
            
        # al_np = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
        # beta_f = float(beta.squeeze(0).item())
        # gamma_f = float(gamma.squeeze(0).item())
        if new_stage_timer >= 5:
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
        
        W_adj, R_adj = Ww.copy(), Rw.copy()
        if alpha_mode != "none": 
            W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, alpha_mode, k_rad)
            
        world_step = ssi.WorldObstacles(Cw, R_adj, W_adj, d_hat=cfg_local.d_hat)
        planner.stage_field.w_goal = max(0.0, beta_f)
        planner.sys.gamma_o = max(0.0, gamma_f)
        info = planner.step(cfg_local.dt, world_step)
        
        center_xy = np.asarray(info["center"], float)
        path.append(center_xy)
        
        if success_reached(center_xy, cfg_local.goal, cfg_local.goal_tol):
            return np.array(path, dtype=np.float32), True
            
    return np.array(path, dtype=np.float32), False

# ========= Timing wrapper =========
def timing_wrapper(func, *args, **kwargs):
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# ========= Evaluation function =========
def evaluate_all_methods_safety_focused(model, cfg_env, world, start_xy, goal_xy, *, 
                                       alpha_mode, k_rad, steps, device, grid_cache):
    """Safety-focused evaluation of all methods"""
    
    r_rest = getattr(cfg_env, "radius", cfg_env.d_hat)
    r_min = getattr(cfg_env, "radius_min", 0.3 * r_rest)
    bounds = grid_cache["bounds"]
    res = grid_cache["res"]
    stage_size = getattr(cfg_env, "stage_size", (2.6, 2.0))
    overlap = getattr(cfg_env, "overlap", 0.3)
    inflate = getattr(cfg_env, "inflate", 0.35)
    
    results = {}
    
    # 1. Rigid A* (Global planning baseline)
    occ_rigid = grid_cache["occ_rigid"]
    (L_rigid, path_rigid), time_rigid = timing_wrapper(
        a_star_path, occ_rigid, bounds, res, start_xy, goal_xy
    )
    rigid_success = np.isfinite(L_rigid) and len(path_rigid) > 1
    
    # 2. Deformable A* (Global planning baseline)  
    clearance_grid = grid_cache["clearance"]
    (L_deform, path_deform), time_deform = timing_wrapper(
        deformable_a_star_path, clearance_grid, bounds, res, start_xy, goal_xy, r_rest, r_min
    )
    deform_success = np.isfinite(L_deform) and len(path_deform) > 1
    
    # Reference length (straight line for fair comparison)
    L_ref = float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(start_xy)))
    if L_ref <= 1e-8:
        L_ref = None
    
    # 3. Potential Field (Stage-aware)
    path_pf, time_pf = timing_wrapper(
        potential_field_path_staged, world.C_np, world.R_np, start_xy, goal_xy,
        stage_size=stage_size, overlap=overlap, inflate=inflate,
        dt=cfg_env.dt, steps=steps, d_safe=max(0.8*r_rest, 0.5), r_min=r_min
    )
    pf_success = success_reached(path_pf[-1] if len(path_pf) else start_xy, goal_xy, cfg_env.goal_tol)
    
    # 4. CBF (Stage-aware)
    path_cbf, time_cbf = timing_wrapper(
        cbf_path_staged, world.C_np, world.R_np, start_xy, goal_xy,
        stage_size=stage_size, overlap=overlap, inflate=inflate,
        dt=cfg_env.dt, steps=steps, safety_margin=r_min
    )
    cbf_success = success_reached(path_cbf[-1] if len(path_cbf) else start_xy, goal_xy, cfg_env.goal_tol)
    
    # 5. DWA (Stage-aware)
    path_dwa, time_dwa = timing_wrapper(
        dwa_path_staged, world.C_np, world.R_np, start_xy, goal_xy,
        stage_size=stage_size, overlap=overlap, inflate=inflate,
        dt=cfg_env.dt, steps=steps
    )
    dwa_success = success_reached(path_dwa[-1] if len(path_dwa) else start_xy, goal_xy, cfg_env.goal_tol)
    
    # 6. GRL-SNAM (Our method)
    cfg_local = gen.GenCfg()
    cfg_local.__dict__.update(cfg_env.__dict__)
    cfg_local.start = start_xy.astype(np.float32)
    cfg_local.goal = goal_xy.astype(np.float32)
    cfg_local.stage_size = stage_size
    cfg_local.overlap = overlap
    cfg_local.inflate = inflate
    
    (our_path, our_success), time_ours = timing_wrapper(
        rollout_grl_snam, model, cfg_local, world,
        alpha_mode=alpha_mode, k_rad=k_rad, steps=steps, device=device
    )
    
    # Package results with safety-focused metrics
    methods_data = [
        ('GRL-SNAM', our_path, our_success, time_ours, 'local_staged'),
        ('RigidA*', path_rigid, rigid_success, time_rigid, 'global_planning'),
        ('DeformA*', path_deform, deform_success, time_deform, 'global_planning'),  
        ('PotentialField', path_pf, pf_success, time_pf, 'local_staged'),
        ('CBF', path_cbf, cbf_success, time_cbf, 'local_staged'),
        ('DWA', path_dwa, dwa_success, time_dwa, 'local_staged'),
    ]
    
    for name, path, success, comp_time, info_type in methods_data:
        metrics = safety_metric_pack(path, L_ref, reached=success,
                                   C=world.C_np, R=world.R_np, 
                                   barrier_thresh=cfg_env.d_hat, tube_thresh=r_min)
        metrics['computational_time'] = comp_time
        metrics['information_type'] = info_type
        
        results[name] = {
            'path': path,
            'success': success,
            'metrics': metrics,
            **metrics  # Flatten metrics for easy access
        }
    
    return results, L_ref

# ========= Safety-focused visualization =========
def plot_safety_focused_overlay(outdir, world, start, goal, results, method_names):
    """Create safety-focused trajectory visualization with clearance profiles"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color scheme
    colors = {
        'GRL-SNAM': '#ff7f0e', 'RigidA*': '#1f77b4', 'DeformA*': '#2ca02c',
        'PotentialField': '#d62728', 'CBF': '#9467bd', 'DWA': '#8c564b'
    }
    
    # Left plot: Trajectory overlay with obstacle clearance visualization
    ax1.set_aspect('equal')
    
    # Draw obstacles with safety margins
    th = np.linspace(0, 2*np.pi, 64)
    for (cx, cy), r in zip(world.C_np, world.R_np):
        # Raw obstacle
        circle = plt.Circle((cx, cy), r, fill=True, color='black', alpha=0.8)
        ax1.add_patch(circle)
        # Safety margin (barrier threshold)
        safety_circle = plt.Circle((cx, cy), r + 0.5, fill=False, color='red', 
                                 linestyle='--', linewidth=1, alpha=0.6)
        ax1.add_patch(safety_circle)
    
    # Draw paths
    for name in method_names:
        path = results[name]['path']
        if path is not None and len(path) > 0:
            # Color path by clearance
            clearances = [clearance_to_circles(p, world.C_np, world.R_np) for p in path]
            
            # Plot path with safety-based coloring
            for i in range(len(path)-1):
                clr = clearances[i]
                if clr < 0:  # Collision
                    color = 'red'
                    alpha = 1.0
                    linewidth = 3
                elif clr < 0.2:  # Danger zone  
                    color = 'orange'
                    alpha = 0.8
                    linewidth = 2.5
                else:  # Safe
                    color = colors.get(name, 'gray')
                    alpha = 0.7
                    linewidth = 2
                    
                ax1.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], 
                        color=color, alpha=alpha, linewidth=linewidth)
            
            # Add method label at end of path
            ax1.text(path[-1][0], path[-1][1], name, fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=colors.get(name, 'gray'), alpha=0.7))
    
    ax1.scatter(*start, s=100, marker='s', color='green', label='Start', zorder=10)
    ax1.scatter(*goal, s=120, marker='*', color='gold', label='Goal', zorder=10)
    ax1.set_title('Safety-Aware Trajectory Comparison\n(Red=Collision, Orange=Danger, Colors=Safe)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Clearance profiles over time
    for name in method_names:
        path = results[name]['path']
        if path is not None and len(path) > 0:
            clearances = [clearance_to_circles(p, world.C_np, world.R_np) for p in path]
            timesteps = range(len(clearances))
            ax2.plot(timesteps, clearances, label=f"{name} (min={results[name]['min_clearance']:.3f})", 
                    color=colors.get(name, 'gray'), linewidth=2, alpha=0.8)
    
    # Safety thresholds
    ax2.axhline(y=0, color='red', linestyle='-', alpha=0.8, label='Collision Boundary')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, label='Barrier Threshold')
    ax2.axhline(y=0.15, color='purple', linestyle=':', alpha=0.6, label='Tube Threshold')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Clearance (m)')
    ax2.set_title('Clearance Profiles Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 2.0)
    
    plt.tight_layout()
    safety_png = os.path.join(outdir, "safety_focused_comparison.png")
    plt.savefig(safety_png, dpi=200, bbox_inches='tight')
    plt.close()
    
    return safety_png

def plot_safety_dashboard(outdir, all_results, split_name):
    """Create comprehensive safety performance dashboard"""
    methods = list(all_results[0].keys()) if all_results else []
    
    # Extract safety metrics
    safety_data = {method: {
        'success': [r[method]['success'] for r in all_results],
        'barrier_viol_rate': [r[method]['barrier_viol_rate'] for r in all_results],
        'collision_rate': [r[method]['collision_rate'] for r in all_results], 
        'min_clearance': [r[method]['min_clearance'] for r in all_results if np.isfinite(r[method]['min_clearance'])],
        'safety_margin_score': [r[method]['safety_margin_score'] for r in all_results if np.isfinite(r[method]['safety_margin_score'])],
        'computational_time': [r[method]['computational_time'] for r in all_results],
    } for method in methods}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Success vs Safety Trade-off
    ax = axes[0]
    for method in methods:
        success_rates = safety_data[method]['success']
        barrier_rates = safety_data[method]['barrier_viol_rate']
        avg_success = np.mean(success_rates)
        avg_barrier_viol = np.mean(barrier_rates)
        ax.scatter(avg_barrier_viol, avg_success, s=100, label=method, alpha=0.8)
        ax.text(avg_barrier_viol + 0.01, avg_success, method, fontsize=8)
    
    ax.set_xlabel('Average Barrier Violation Rate')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success vs Safety Trade-off')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Collision Rates
    ax = axes[1]
    collision_means = [np.mean(safety_data[m]['collision_rate']) for m in methods]
    collision_stds = [np.std(safety_data[m]['collision_rate']) for m in methods]
    bars = ax.bar(methods, collision_means, yerr=collision_stds, capsize=5, alpha=0.8)
    ax.set_ylabel('Collision Rate')
    ax.set_title('Average Collision Rates by Method')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 3. Minimum Clearance Distribution
    ax = axes[2]
    clear_data = [safety_data[m]['min_clearance'] for m in methods if len(safety_data[m]['min_clearance']) > 0]
    if clear_data:
        ax.violinplot(clear_data, positions=range(len(clear_data)), showmeans=True)
        ax.set_xticks(range(len(clear_data)))
        ax.set_xticklabels([m for m in methods if len(safety_data[m]['min_clearance']) > 0], rotation=45)
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.6, label='Collision')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, label='Barrier')
    ax.set_ylabel('Minimum Clearance (m)')
    ax.set_title('Minimum Clearance Distributions')
    ax.legend()
    
    # 4. Safety Margin Score
    ax = axes[3]
    margin_means = [np.mean(safety_data[m]['safety_margin_score']) for m in methods if safety_data[m]['safety_margin_score']]
    margin_stds = [np.std(safety_data[m]['safety_margin_score']) for m in methods if safety_data[m]['safety_margin_score']]
    if margin_means:
        bars = ax.bar(methods, margin_means, yerr=margin_stds, capsize=5, alpha=0.8)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, label='Barrier Level')
        ax.set_ylabel('Safety Margin Score')
        ax.set_title('Average Safety Margin Maintenance')
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.legend()
    
    # 5. Computational Efficiency vs Safety
    ax = axes[4]
    for method in methods:
        comp_times = safety_data[method]['computational_time']
        barrier_rates = safety_data[method]['barrier_viol_rate']
        if comp_times and barrier_rates:
            ax.scatter(np.mean(comp_times), np.mean(barrier_rates), 
                      s=100, label=method, alpha=0.8)
            ax.text(np.mean(comp_times), np.mean(barrier_rates) + 0.01, method, fontsize=8)
    
    ax.set_xlabel('Average Computation Time (s)')
    ax.set_ylabel('Average Barrier Violation Rate')  
    ax.set_title('Computational Efficiency vs Safety')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 6. Safety Performance Summary Table
    ax = axes[5]
    ax.axis('off')
    
    # Create summary table
    table_data = []
    for method in methods:
        row = [
            method,
            f"{np.mean(safety_data[method]['success']):.3f}",
            f"{np.mean(safety_data[method]['barrier_viol_rate']):.3f}",
            f"{np.mean(safety_data[method]['collision_rate']):.3f}",
            f"{np.mean(safety_data[method]['min_clearance']):.3f}" if safety_data[method]['min_clearance'] else "N/A"
        ]
        table_data.append(row)
    
    headers = ['Method', 'Success', 'Barrier Viol', 'Collision', 'Min Clear']
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title('Safety Performance Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{safe_name(split_name)}_safety_dashboard.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()

# ========= Main evaluation loop =========
def run_safety_evaluation(split_name, case_fn, args, device, model, run_dir):
    print(f"\n=== {split_name}: Safety-Focused Evaluation ===")
    print(f"Environments: {args.n_envs}, Trials per env: {args.n_trials}")
    
    all_results = []
    rng = np.random.default_rng(args.seed + (0 if "ID" in split_name else 1337))
    
    for env_id in range(args.n_envs):
        # Generate environment
        cfg = gen.GenCfg()
        cfg.seed = int(rng.integers(0, 10_000_000))
        gen.set_all_seeds(cfg.seed)
        cfg.d_hat = getattr(cfg, "d_hat", 0.5)
        cfg.radius = getattr(cfg, "radius", 0.5) 
        cfg.radius_min = getattr(cfg, "radius_min", 0.3 * cfg.radius)
        
        C, R, W = case_fn(cfg)
        world = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)
        
        # Precompute grids  
        r_rest = cfg.radius
        bounds = compute_workspace_bounds(world.C_np, world.R_np, margin=2.0*r_rest)
        res = max(0.2, 0.35 * max(r_rest, cfg.d_hat))
        occ_rigid, _, _ = rasterize_inflated_obstacles(world.C_np, world.R_np, r_rest, bounds, res)
        clearance_grid = clearance_field_on_grid(world.C_np, world.R_np, bounds, res)
        grid_cache = dict(bounds=bounds, res=res, occ_rigid=occ_rigid, clearance=clearance_grid)
        
        for trial_id in range(args.n_trials):
            # Sample challenging start/goal (closer to obstacles for safety testing)
            def sample_point_with_clearance_range(min_clear, max_clear, tries=1000):
                xmin, xmax, ymin, ymax = bounds
                for _ in range(tries):
                    p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
                    if len(C) == 0:
                        return p
                    clr = clearance_to_circles(p, world.C_np, world.R_np)
                    if min_clear <= clr <= max_clear:
                        return p
                # Fallback to any valid point
                for _ in range(tries):
                    p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
                    if len(C) == 0 or clearance_to_circles(p, world.C_np, world.R_np) >= min_clear:
                        return p
                return np.array([bounds[0] + 1, bounds[2] + 1])

            # Sample points with moderate clearance (not too easy, not impossible)
            start = sample_point_with_clearance_range(cfg.radius_min, 1.5*cfg.radius)
            goal = sample_point_with_clearance_range(cfg.radius_min, 1.5*cfg.radius)
            if np.linalg.norm(start - goal) < 2.0*cfg.radius:
                goal = goal + np.array([2.5*cfg.radius, 0.0])
            
            # Episode directory
            ep_dir = mkdir(os.path.join(run_dir, f"{safe_name(split_name)}_env{env_id:03d}_trial{trial_id:03d}"))
            
            # Evaluate all methods
            results, L_ref = evaluate_all_methods_safety_focused(
                model, cfg, world, start, goal,
                alpha_mode=args.alpha_mode, k_rad=args.k_rad, 
                steps=args.steps, device=device, grid_cache=grid_cache
            )
            
            # Create safety-focused visualizations
            method_names = list(results.keys())
            safety_png = plot_safety_focused_overlay(ep_dir, world, start, goal, results, method_names)
            
            all_results.append(results)
            
            # Progress logging with safety focus
            safety_summary = []
            for m in method_names:
                s = int(results[m]['success'])
                b = results[m]['barrier_viol_rate']
                c = results[m]['collision_rate']
                safety_summary.append(f"{m}:S{s}/B{b:.2f}/C{c:.2f}")
            
            print(f"[{split_name} e{env_id} t{trial_id}] " + " | ".join(safety_summary))
    
    # Create safety dashboard
    plot_safety_dashboard(run_dir, all_results, split_name)
    
    # Save detailed results
    results_path = os.path.join(run_dir, f"{safe_name(split_name)}_safety_results.json")
    json_results = []
    for result_set in all_results:
        json_set = {}
        for method, data in result_set.items():
            json_set[method] = {
                'path': _to_serializable_path(data.get('path')),
                'success': float(data['success']),
                'min_clearance': float(data['min_clearance']) if np.isfinite(data['min_clearance']) else None,
                'barrier_viol_rate': float(data['barrier_viol_rate']),
                'collision_rate': float(data['collision_rate']),
                'safety_margin_score': float(data['safety_margin_score']),
                'computational_time': float(data['computational_time']),
                'information_type': data['information_type'],
            }
        json_results.append(json_set)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return all_results

def main():
    parser = argparse.ArgumentParser("E5: Safety & Constraint Satisfaction - Comprehensive")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","none"])  
    parser.add_argument("--k_rad", type=float, default=0.05)
    parser.add_argument("--case", type=str, default="case1-tight")
    parser.add_argument("--n_envs", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=2312)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = CoefEnergyNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = mkdir(f"E5_safety_comprehensive/{timestamp}")
    
    # Define case function
    case_funcs = {
        "case1-tight": gen.sample_obstacles_case1_tight,
        "case2-harder": gen.sample_obstacles_case2_harder,
    }
    case_fn = case_funcs.get(args.case, gen.sample_obstacles_case1_tight)
    
    # Run safety evaluation
    all_results = run_safety_evaluation("Safety_Test", case_fn, args, device, model, run_dir)
    
    # Create final safety summary
    def safety_summary_stats(method_key):
        success_rates = [r[method_key]['success'] for r in all_results]
        barrier_rates = [r[method_key]['barrier_viol_rate'] for r in all_results] 
        collision_rates = [r[method_key]['collision_rate'] for r in all_results]
        comp_times = [r[method_key]['computational_time'] for r in all_results]
        
        return {
            'success_rate_mean': float(np.mean(success_rates)),
            'success_rate_std': float(np.std(success_rates)),
            'barrier_viol_rate_mean': float(np.mean(barrier_rates)),
            'barrier_viol_rate_std': float(np.std(barrier_rates)),
            'collision_rate_mean': float(np.mean(collision_rates)),
            'collision_rate_std': float(np.std(collision_rates)),
            'computational_time_mean': float(np.mean(comp_times)),
            'computational_time_std': float(np.std(comp_times)),
        }
    
    methods = list(all_results[0].keys()) if all_results else []
    summary = {
        "experiment": "E5: Safety & Constraint Satisfaction",
        "timestamp": timestamp,
        "parameters": {
            "case": args.case,
            "n_envs": args.n_envs,
            "n_trials": args.n_trials,
            "steps": args.steps,
            "alpha_mode": args.alpha_mode
        },
        "methods_evaluated": methods,
        "safety_summary": {method: safety_summary_stats(method) for method in methods}
    }
    
    with open(os.path.join(run_dir, "safety_experiment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== E5 Safety Experiment Complete ===")
    print(f"Results saved to: {run_dir}")
    print(f"Methods compared: {len(methods)}")
    print(f"Total episodes: {args.n_envs * args.n_trials}")
    
    # Print safety summary table
    print("\n=== Safety Performance Summary ===")
    print(f"{'Method':<15} {'Success':<8} {'Barrier':<8} {'Collision':<10} {'CompTime':<8}")
    print("-" * 60)
    for method in methods:
        stats = summary["safety_summary"][method]
        print(f"{method:<15} {stats['success_rate_mean']:.3f}    "
              f"{stats['barrier_viol_rate_mean']:.3f}    "
              f"{stats['collision_rate_mean']:.4f}     "
              f"{stats['computational_time_mean']:.3f}s")

if __name__ == "__main__":
    main()