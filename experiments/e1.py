#!/usr/bin/env python3
"""
Experiment 1: Comprehensive baseline comparison
Methods: GRL-SNAM, Rigid A*, Deformable A*, Potential Field, CBF, DWA

Accurate implementations of all baseline methods with proper tuning parameters.
"""

import os, json, argparse, csv
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import scipy.optimize
from scipy.spatial.distance import cdist
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
    if isinstance(p, np.ndarray):
        return p.tolist()
    # p might already be a list/tuple or a list of arrays
    try:
        return np.asarray(p).tolist()
    except Exception:
        # last resort: wrap as list of primitives
        return [q.tolist() if hasattr(q, "tolist") else (list(q) if isinstance(q, (list, tuple)) else q)
                for q in p] if isinstance(p, (list, tuple)) else None

def local_slice(x, C, R, *, sense_radius=None, k_nearest=None):
    """Return obstacles within sense_radius and (optionally) the k nearest of those."""
    C = np.asarray(C); R = np.asarray(R)
    if C.size == 0: return C, R
    d = np.linalg.norm(C - np.asarray(x)[None, :], axis=1)
    mask = np.ones(len(C), dtype=bool)
    if sense_radius is not None:
        mask &= (d <= sense_radius)
    C2, R2, d2 = C[mask], R[mask], d[mask]
    if (k_nearest is not None) and (len(C2) > k_nearest):
        idx = np.argpartition(d2, k_nearest)[:k_nearest]
        C2, R2 = C2[idx], R2[idx]
    return C2, R2

def straight_line_ref(start_xy, goal_xy, eps=1e-8):
    d = float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(start_xy)))
    return (d if d > eps else None), "straight_line"

def to_serializable_path(p):
    if p is None:
        return None
    try:
        # Works for numpy arrays *and* lists-of-lists
        return np.asarray(p).tolist()
    except Exception:
        # Last resort: leave as-is if it’s already JSON-friendly
        return p

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

def compute_safety_stats(path_xy, C, R, barrier_thresh, tube_thresh):
    T = len(path_xy)
    if T == 0:
        return dict(min_clearance=np.inf, barrier_viol_rate=0.0, tube_viol_rate=0.0,
                   barrier_penalty_sum=0.0, barrier_viol_count=0, tube_viol_count=0)
    clears = np.array([clearance_to_circles(np.asarray(xy,float), C, R) for xy in path_xy])
    barrier_mask = clears < float(barrier_thresh)
    tube_mask = clears < float(tube_thresh)
    penalties = [max(0, barrier_thresh - c)**2 for c in clears]
    return dict(
        min_clearance=float(np.min(clears)),
        barrier_viol_rate=float(np.sum(barrier_mask))/T,
        tube_viol_rate=float(np.sum(tube_mask))/T,
        barrier_penalty_sum=float(np.sum(penalties)),
        barrier_viol_count=int(np.sum(barrier_mask)),
        tube_viol_count=int(np.sum(tube_mask))
    )

def metric_pack(path_xy, L_ref, *, reached: bool, C, R, barrier_thresh: float, tube_thresh: float):
    L_exec = path_length(path_xy)
    S = 1.0 if reached else 0.0
    
    if (L_ref is None) or not (L_ref > 0 and np.isfinite(L_ref)):
        spl, detour = np.nan, np.nan
    else:
        spl = S * (L_ref / max(L_ref, L_exec))
        detour = L_exec / L_ref
    
    safety = compute_safety_stats(path_xy, C, R, barrier_thresh, tube_thresh)
    return dict(
        success=float(S), path_length=L_exec,
        SPL=float(spl) if np.isfinite(spl) else np.nan,
        detour=float(detour) if np.isfinite(detour) else np.nan,
        min_clear=safety["min_clearance"],
        barrier_viol_rate=safety["barrier_viol_rate"],
        tube_viol_rate=safety["tube_viol_rate"],
        barrier_penalty_sum=safety["barrier_penalty_sum"],
        smoothness=compute_smoothness(path_xy),
        computational_time=0.0  # Will be filled by timing wrapper
    )

def compute_smoothness(path_xy):
    """Compute path smoothness as average turning angle"""
    if len(path_xy) < 3: return 0.0
    P = np.array(path_xy)
    v = P[1:] - P[:-1]  # velocity vectors
    v_norm = np.linalg.norm(v, axis=1)
    v_norm[v_norm < 1e-6] = 1e-6  # avoid division by zero
    v_unit = v / v_norm[:, None]
    
    # compute turning angles between consecutive segments
    cos_angles = np.sum(v_unit[:-1] * v_unit[1:], axis=1)
    cos_angles = np.clip(cos_angles, -1, 1)  # numerical stability
    angles = np.arccos(cos_angles)
    return float(np.mean(angles))

# ========= A* implementations (existing, kept for completeness) =========
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
    """Standard A* implementation on occupancy grid"""
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
    
    # reconstruct path
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
    """Compute exact clearance field on grid"""
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
    """A* with clearance-based cost penalties"""
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
    
    # reconstruct
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
    """Create stage manager identical to GRL-SNAM for fair comparison"""
    return ssi.StageManager(
        np.array(start_xy), np.array(goal_xy),
        stage_size=stage_size, overlap_ratio=overlap,
        obstacles=obstacles, inflate=inflate
    )

def get_local_obstacles(stage_manager, C_global, R_global, W_global=None):
    """Extract obstacles within current stage bounds (same as GRL-SNAM's stage_slice)"""
    st = stage_manager.current()
    xmin, xmax, ymin, ymax = st.bounds
    # Add small margin like in original stage_slice
    mask = ((C_global[:,0] >= xmin-0.5) & (C_global[:,0] <= xmax+0.5) & 
            (C_global[:,1] >= ymin-0.5) & (C_global[:,1] <= ymax+0.5))
    
    if np.any(mask):
        C_local = C_global[mask]
        R_local = R_global[mask] 
        W_local = W_global[mask] if W_global is not None else None
        return C_local, R_local, W_local
    else:
        return np.zeros((0,2)), np.zeros((0)), np.zeros((0))

# ========= Reactive Potential Field (Stage-aware) =========
def potential_field_path_staged(C_global, R_global, start_xy, goal_xy, *, 
                               stage_size=(2.6, 2.0), overlap=0.3, inflate=0.35,
                               dt=0.02, steps=1500, v_max=0.7,
                               k_goal=1.5, k_rep=0.9, d_safe=0.8, r_min=0.3):
    """Stage-aware potential field navigation matching GRL-SNAM's local information"""
    # Create stage manager identical to GRL-SNAM
    stage_manager = create_stage_manager_for_baselines(
        start_xy, goal_xy, stage_size, overlap, 
        obstacles=(C_global, R_global), inflate=inflate
    )
    
    x = np.array(start_xy, dtype=np.float32)
    path = [x.copy()]
    
    for t in range(steps):
        # Get current stage's local obstacles (same as GRL-SNAM sees)
        C_local, R_local, _ = get_local_obstacles(stage_manager, C_global, R_global)
        
        # Current stage info
        current_stage = stage_manager.current()
        
        # Attractive force toward stage exit (intermediate target) or final goal
        if stage_manager.current_stage_idx < len(stage_manager.stages) - 1:
            # Navigate to stage exit point
            target = current_stage.exit_point
        else:
            # Final stage - navigate to goal
            target = goal_xy
            
        to_target = target - x
        dist_target = np.linalg.norm(to_target) + 1e-6
        F_att = k_goal * (to_target / dist_target)
        
        # Repulsive forces from LOCAL obstacles only
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
        
        # Stage boundary repulsion (keep within current stage)
        xmin, xmax, ymin, ymax = current_stage.bounds
        margin = 0.3
        F_boundary = np.zeros(2, dtype=np.float32)
        if x[0] < xmin + margin: F_boundary[0] += (xmin + margin - x[0]) * 2.0
        if x[0] > xmax - margin: F_boundary[0] -= (x[0] - (xmax - margin)) * 2.0  
        if x[1] < ymin + margin: F_boundary[1] += (ymin + margin - x[1]) * 2.0
        if x[1] > ymax - margin: F_boundary[1] -= (x[1] - (ymax - margin)) * 2.0
        
        # Total velocity command
        v = F_att + F_rep + F_boundary
        
        # Speed limiting
        speed = np.linalg.norm(v)
        if speed > v_max:
            v = v * (v_max / speed)
        
        # Emergency braking near LOCAL obstacles
        if len(C_local) > 0:
            min_clearance = clearance_to_circles(x, C_local, R_local)
            if min_clearance < r_min:
                v *= 0.1
        
        # Update position
        x = x + dt * v
        path.append(x.copy())
        
        # Advance stage if needed (same logic as GRL-SNAM)
        stage_manager.advance_if_needed(x)
        
        # Final goal check
        if np.linalg.norm(x - goal_xy) <= 0.2:
            break
    
    return np.array(path, dtype=np.float32)

# ========= Control Barrier Functions (CBF) - Stage-aware =========
class CBFControllerStaged:
    def __init__(self, gamma=1.0, safety_margin=0.1):
        self.gamma = gamma
        self.safety_margin = safety_margin
        self.u_max = 2.0
        # Local obstacles updated each step
        self.C_local = np.zeros((0,2))
        self.R_local = np.zeros(0)
    
    def update_local_obstacles(self, C_local, R_local):
        """Update local obstacle set (called each timestep)"""
        self.C_local = np.array(C_local) if len(C_local) > 0 else np.zeros((0,2))
        self.R_local = np.array(R_local) + self.safety_margin if len(R_local) > 0 else np.zeros(0)
    
    def barrier_function(self, x):
        """Barrier function h(x) = min_i(||x - c_i|| - r_i) for LOCAL obstacles only"""
        if len(self.C_local) == 0:
            return np.inf
        distances = np.linalg.norm(self.C_local - x, axis=1)
        return np.min(distances - self.R_local)
    
    def barrier_gradient(self, x):
        """Gradient of barrier function for LOCAL obstacles"""
        if len(self.C_local) == 0:
            return np.zeros(2)
        
        distances = np.linalg.norm(self.C_local - x, axis=1)
        clearances = distances - self.R_local
        min_idx = np.argmin(clearances)
        
        direction = (x - self.C_local[min_idx]) / distances[min_idx]
        return direction
    
    def safe_control(self, x, u_nominal):
        """Solve QP to find safe control input"""
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
    """Stage-aware CBF navigation matching GRL-SNAM's local information"""
    stage_manager = create_stage_manager_for_baselines(
        start_xy, goal_xy, stage_size, overlap,
        obstacles=(C_global, R_global), inflate=inflate
    )
    
    controller = CBFControllerStaged(gamma=gamma_cbf, safety_margin=safety_margin)
    x = np.array(start_xy, dtype=np.float32)
    path = [x.copy()]
    
    for t in range(steps):
        # Get LOCAL obstacles within current stage
        C_local, R_local, _ = get_local_obstacles(stage_manager, C_global, R_global)
        controller.update_local_obstacles(C_local, R_local)
        
        # Current stage info
        current_stage = stage_manager.current()
        
        # Navigate to stage exit or final goal
        if stage_manager.current_stage_idx < len(stage_manager.stages) - 1:
            target = current_stage.exit_point
        else:
            target = goal_xy
            
        # Nominal control toward target
        to_target = target - x
        dist_target = np.linalg.norm(to_target) + 1e-6
        u_nominal = k_goal * (to_target / dist_target)
        
        # Apply CBF safety filter using LOCAL obstacles only
        u_safe = controller.safe_control(x, u_nominal)
        
        # Update state
        x = x + dt * u_safe
        path.append(x.copy())
        
        # Advance stage if needed
        stage_manager.advance_if_needed(x)
        
        # Final goal check
        if np.linalg.norm(x - goal_xy) <= 0.2:
            break
    
    return np.array(path, dtype=np.float32)

# ========= Dynamic Window Approach (DWA) - Stage-aware =========
class DWAPlannerStaged:
    def __init__(self, dt=0.02, predict_time=2.0, stage_size=(2.6, 2.0), overlap=0.3, inflate=0.35):
        self.dt = dt
        self.predict_time = predict_time
        self.predict_steps = int(predict_time / dt)
        self.stage_size = stage_size
        self.overlap = overlap
        self.inflate = inflate
        
        # Robot dynamics constraints
        self.v_max = 1.2
        self.omega_max = 1.5
        self.v_acc_max = 2.0
        self.omega_acc_max = 3.0
        
        # DWA parameters
        self.v_resolution = 0.1
        self.omega_resolution = 0.1
        
        # Objective weights
        self.w_heading = 0.4
        self.w_distance = 0.3  
        self.w_velocity = 0.3
        self.safety_margin = 0.2
        
        # Local obstacles (updated each step)
        self.C_local = np.zeros((0,2))
        self.R_local = np.zeros(0)
        
        self.w_clear    = 0.5
        self.w_bounds   = 2.0
        self.eps = 1e-6
    
    def update_local_obstacles(self, C_local, R_local):
        """Update local obstacle set for current stage"""
        self.C_local = np.array(C_local) if len(C_local) > 0 else np.zeros((0,2))
        self.R_local = np.array(R_local) if len(R_local) > 0 else np.zeros(0)
    
    def dynamic_window(self, v_current, omega_current):
        """Compute admissible velocity space given current state and dynamics"""
        v_min = max(0, v_current - self.v_acc_max * self.dt)
        v_max = min(self.v_max, v_current + self.v_acc_max * self.dt)
        
        omega_min = max(-self.omega_max, omega_current - self.omega_acc_max * self.dt)
        omega_max = min(self.omega_max, omega_current + self.omega_acc_max * self.dt)
        
        return v_min, v_max, omega_min, omega_max
    
    def simulate_trajectory(self, x, theta, v, omega):
        """Simulate robot trajectory for given control inputs"""
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
        collision = False

        for pt in trajectory:
            # Softer stage boundary penalties (don't immediately reject)
            if (pt[0] < xmin - 1.0 or pt[0] > xmax + 1.0 or 
                pt[1] < ymin - 1.0 or pt[1] > ymax + 1.0):
                bound_pen += 100.0  # Large penalty but not infinite
            else:
                # Soft quadratic penalty for being outside preferred bounds
                dxl = max(0.0, xmin - pt[0])
                dxr = max(0.0, pt[0] - xmax) 
                dyl = max(0.0, ymin - pt[1])
                dyr = max(0.0, pt[1] - ymax)
                bound_pen += 0.5 * (dxl*dxl + dxr*dxr + dyl*dyl + dyr*dyr)

            # Local obstacle clearance - be more lenient
            if len(self.C_local) > 0:
                clr = clearance_to_circles(pt, self.C_local, self.R_local)
                min_clear = min(min_clear, clr)
                # Only reject for significant penetration, not just touching
                if clr < -0.1:  # Allow small violations
                    collision = True
                    break

        # Hard collision rejection
        if collision:
            return np.inf

        # Heading cost 
        theta_f = theta0 + omega * self.predict_time
        final_pos = trajectory[-1]
        to_target = target - final_pos
        target_heading = np.arctan2(to_target[1], to_target[0])
        heading_err = abs(np.arctan2(np.sin(target_heading - theta_f), 
                                    np.cos(target_heading - theta_f)))

        # Distance and velocity costs
        dist_cost = np.linalg.norm(to_target)
        vel_cost = (self.v_max - abs(v)) / (self.v_max + 1e-6)

        # Balanced clearance cost - reward good clearance but don't dominate
        if min_clear < np.inf:
            clear_cost = max(0.0, 2.0 - min_clear)  # Linear penalty for clearance < 2.0
        else:
            clear_cost = 0.0

        # Balanced cost function
        total = (2.0 * heading_err +           # Heading importance
                1.0 * dist_cost +             # Distance importance  
                0.5 * vel_cost +              # Velocity importance
                0.1 * bound_pen +             # Boundary penalty
                1.0 * clear_cost)             # Clearance penalty
        
        return total

    def _grid(self, lo, hi, step):
        if hi <= lo + 1e-12:
            return np.array([lo])
        
        # Ensure minimum number of samples
        n_samples = max(3, int(np.ceil((hi - lo) / step)) + 1)
        return np.linspace(lo, hi, n_samples)

    def plan_step(self, pos, theta, v_current, omega_current, target, stage_bounds):
        v_min, v_max, om_min, om_max = self.dynamic_window(v_current, omega_current)
        
        # Ensure we have reasonable sampling resolution
        v_samples = self._grid(v_min, v_max, min(self.v_resolution, (v_max - v_min) / 5))
        om_samples = self._grid(om_min, om_max, min(self.omega_resolution, (om_max - om_min) / 5))
        
        best_cost = np.inf
        best_v, best_om = 0.0, 0.0
        valid_found = False

        for v in v_samples:
            for om in om_samples:
                traj = self.simulate_trajectory(pos, theta, v, om)
                cost = self.trajectory_cost(traj, target, v, om, stage_bounds, theta)
                
                if np.isfinite(cost):
                    valid_found = True
                    if cost < best_cost:
                        best_cost, best_v, best_om = cost, v, om

        # Improved fallback when no valid trajectory found
        if not valid_found:
            # Try simple proportional control toward target
            to_target = target - pos
            target_dist = np.linalg.norm(to_target) + 1e-6
            target_heading = np.arctan2(to_target[1], to_target[0])
            
            # Compute desired angular velocity to face target
            heading_error = np.arctan2(np.sin(target_heading - theta), 
                                    np.cos(target_heading - theta))
            
            # Conservative fallback velocities
            fallback_v = min(0.3, v_max * 0.5) if target_dist > 0.5 else 0.1
            fallback_om = np.clip(2.0 * heading_error, om_min, om_max)
            
            return fallback_v, fallback_om

        return best_v, best_om

def dwa_path_staged(C_global, R_global, start_xy, goal_xy, *, 
                   stage_size=(2.6, 2.0), overlap=0.3, inflate=0.35,
                   dt=0.02, steps=1500):
    """Stage-aware DWA navigation matching GRL-SNAM's local information"""
    # Create stage manager identical to GRL-SNAM
    stage_manager = create_stage_manager_for_baselines(
        start_xy, goal_xy, stage_size, overlap,
        obstacles=(C_global, R_global), inflate=inflate
    )
    
    planner = DWAPlannerStaged(dt=dt, stage_size=stage_size, overlap=overlap, inflate=inflate)
    
    # State: [x, y, theta, v, omega]
    x = np.array(start_xy, dtype=np.float32)
    theta = np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
    v, omega = 0.0, 0.0
    
    path = [x.copy()]
    
    for t in range(steps):
        # Get LOCAL obstacles within current stage
        C_local, R_local, _ = get_local_obstacles(stage_manager, C_global, R_global)
        planner.update_local_obstacles(C_local, R_local)
        
        # Current stage info
        current_stage = stage_manager.current()
        
        # Navigate to stage exit or final goal
        if stage_manager.current_stage_idx < len(stage_manager.stages) - 1:
            target = current_stage.exit_point
        else:
            target = goal_xy
        
        # Plan next control input using LOCAL information only
        v, omega = planner.plan_step(x, theta, v, omega, target, current_stage.bounds)
        
        # Update state
        x += np.array([v * np.cos(theta), v * np.sin(theta)]) * dt
        theta += omega * dt
        
        path.append(x.copy())
        
        # Advance stage if needed (same as GRL-SNAM)
        stage_manager.advance_if_needed(x)
        
        # Final goal check
        if np.linalg.norm(x - goal_xy) <= 0.2:
            break
    
    return np.array(path, dtype=np.float32)

# ========= Enhanced visualization and analysis =========
def plot_comprehensive_comparison(outdir, world, results, start, goal, method_names):
    """Create comprehensive trajectory comparison plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Color scheme for methods
    colors = {
        'GRL-SNAM': '#ff7f0e',      # orange
        'RigidA*': '#1f77b4',       # blue  
        'DeformA*': '#2ca02c',      # green
        'PotentialField': '#d62728', # red
        'CBF': '#9467bd',           # purple
        'DWA': '#8c564b'            # brown
    }
    
    # Plot each method separately first
    for idx, method in enumerate(method_names[:6]):  # Max 6 subplots
        ax = axes[idx]
        
        # Draw obstacles
        th = np.linspace(0, 2*np.pi, 64)
        for (cx, cy), r in zip(world.C_np, world.R_np):
            circle = plt.Circle((cx, cy), r, fill=False, color='black', linewidth=1.5)
            ax.add_patch(circle)
        
        # Draw path
        path = results[method]['path']
        if path is not None and len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], '-', color=colors[method], 
                   linewidth=2.5, label=method, alpha=0.9)
        
        # Start and goal
        ax.scatter(*start, s=100, marker='s', color='green', label='Start', zorder=10)
        ax.scatter(*goal, s=120, marker='*', color='gold', label='Goal', zorder=10)
        
        ax.set_aspect('equal')
        ax.set_title(f'{method}\nSuccess: {results[method]["success"]:.0f}, SPL (vs straight-line): {results[method]["SPL"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(method_names), 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comprehensive_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Overlay plot with all methods
    plt.figure(figsize=(10, 8))
    
    # Draw obstacles
    for (cx, cy), r in zip(world.C_np, world.R_np):
        circle = plt.Circle((cx, cy), r, fill=False, color='black', linewidth=1.5)
        plt.gca().add_patch(circle)
    
    # Draw all paths
    for method in method_names:
        path = results[method]['path']
        if path is not None and len(path) > 0:
            plt.plot(path[:, 0], path[:, 1], '-', color=colors[method], 
                    linewidth=2.2, label=f'{method} (S={results[method]["success"]:.0f})', alpha=0.8)
    
    plt.scatter(*start, s=100, marker='s', color='green', label='Start', zorder=10)
    plt.scatter(*goal, s=120, marker='*', color='gold', label='Goal', zorder=10)
    
    plt.axis('equal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('All Methods Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlay_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()

def plot_performance_metrics(outdir, all_results, split_name):
    """Create comprehensive performance metric visualizations"""
    methods = list(all_results[0].keys()) if all_results else []
    n_methods = len(methods)
    
    # Extract metrics
    metrics_data = {method: {
        'success': [r[method]['success'] for r in all_results],
        'SPL': [r[method]['SPL'] for r in all_results if np.isfinite(r[method]['SPL'])],
        'detour': [r[method]['detour'] for r in all_results if np.isfinite(r[method]['detour'])],
        'min_clear': [r[method]['min_clear'] for r in all_results if np.isfinite(r[method]['min_clear'])],
        'smoothness': [r[method]['smoothness'] for r in all_results if np.isfinite(r[method]['smoothness'])],
        'path_length': [r[method]['path_length'] for r in all_results]
    } for method in methods}
    
    # Create performance dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Success Rate
    ax = axes[0]
    success_rates = [np.mean(metrics_data[m]['success']) for m in methods]
    success_stds = [np.std(metrics_data[m]['success']) for m in methods]
    bars = ax.bar(methods, success_rates, yerr=success_stds, capsize=5, alpha=0.8)
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate by Method')
    ax.set_ylim(0, 1.1)
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 2. SPL Distribution
    ax = axes[1]
    spl_data = [metrics_data[m]['SPL'] for m in methods if len(metrics_data[m]['SPL']) > 0]
    if spl_data:
        ax.boxplot(spl_data, labels=[m for m in methods if len(metrics_data[m]['SPL']) > 0])
    ax.set_ylabel('SPL')
    ax.set_title('SPL Distribution')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 3. Path Length vs Success
    ax = axes[2]
    for method in methods:
        success = metrics_data[method]['success']
        lengths = metrics_data[method]['path_length']
        colors_scatter = ['green' if s else 'red' for s in success]
        ax.scatter(lengths, success, alpha=0.6, label=method, c=colors_scatter)
    ax.set_xlabel('Path Length')
    ax.set_ylabel('Success (0/1)')
    ax.set_title('Path Length vs Success')
    ax.legend()
    
    # 4. Safety Analysis (Clearance)
    ax = axes[3]
    clear_data = [metrics_data[m]['min_clear'] for m in methods if len(metrics_data[m]['min_clear']) > 0]
    if clear_data:
        ax.violinplot(clear_data, positions=range(len(clear_data)), showmeans=True)
        ax.set_xticks(range(len(clear_data)))
        ax.set_xticklabels([m for m in methods if len(metrics_data[m]['min_clear']) > 0], rotation=45)
    ax.set_ylabel('Minimum Clearance')
    ax.set_title('Safety: Minimum Clearance Distribution')
    
    # 5. Smoothness Comparison
    ax = axes[4]
    smooth_means = [np.mean(metrics_data[m]['smoothness']) if metrics_data[m]['smoothness'] else 0 
                   for m in methods]
    smooth_stds = [np.std(metrics_data[m]['smoothness']) if len(metrics_data[m]['smoothness']) > 1 else 0 
                  for m in methods]
    bars = ax.bar(methods, smooth_means, yerr=smooth_stds, capsize=5, alpha=0.8)
    ax.set_ylabel('Average Turning Angle (rad)')
    ax.set_title('Path Smoothness (Lower = Smoother)')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 6. Pareto: SPL vs Safety
    ax = axes[5]
    for method in methods:
        spl_vals = metrics_data[method]['SPL']
        clear_vals = metrics_data[method]['min_clear']
        if len(spl_vals) > 0 and len(clear_vals) > 0:
            # Align lengths
            min_len = min(len(spl_vals), len(clear_vals))
            ax.scatter(clear_vals[:min_len], spl_vals[:min_len], alpha=0.7, label=method, s=50)
    ax.set_xlabel('Minimum Clearance')
    ax.set_ylabel('SPL')
    ax.set_title('Pareto Front: Performance vs Safety')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{safe_name(split_name)}_performance_dashboard.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()

# ========= Main evaluation functions =========
def timing_wrapper(func, *args, **kwargs):
    """Wrapper to measure execution time"""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def evaluate_all_methods(model, cfg_env, world, start_xy, goal_xy, *, 
                        alpha_mode, k_rad, steps, device, grid_cache):
    """Evaluate all baseline methods on single episode with fair local information constraints"""
    
    # Shared parameters
    r_rest = getattr(cfg_env, "radius", cfg_env.d_hat)
    r_min = getattr(cfg_env, "radius_min", 0.3 * r_rest)
    bounds = grid_cache["bounds"]
    res = grid_cache["res"]
    stage_size = getattr(cfg_env, "stage_size", (2.6, 2.0))
    overlap = getattr(cfg_env, "overlap", 0.3)
    inflate = getattr(cfg_env, "inflate", 0.35)
    
    results = {}
    
    # 1. Rigid A* - uses global information (planning-time method)
    occ_rigid = grid_cache["occ_rigid"]
    (L_rigid, path_rigid), time_rigid = timing_wrapper(
        a_star_path, occ_rigid, bounds, res, start_xy, goal_xy
    )
    rigid_success = np.isfinite(L_rigid) and len(path_rigid) > 1
    
    # 2. Deformable A* - uses global information (planning-time method)
    clearance_grid = grid_cache["clearance"]
    (L_deform, path_deform), time_deform = timing_wrapper(
        deformable_a_star_path, clearance_grid, bounds, res, start_xy, goal_xy, r_rest, r_min
    )
    deform_success = np.isfinite(L_deform) and len(path_deform) > 1
    
    # Reference length for SPL/detour calculation (from planning methods)
    # L_ref = L_deform if deform_success else (L_rigid if rigid_success else None)
    # NEW: straight-line reference
    L_ref = float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(start_xy)))
    if L_ref <= 1e-8:   # degenerate case: start≈goal
        L_ref = None
    
    # 3. Potential Field - STAGED (same local info as GRL-SNAM)
    path_pf, time_pf = timing_wrapper(
        potential_field_path_staged, world.C_np, world.R_np, start_xy, goal_xy,
        stage_size=stage_size, overlap=overlap, inflate=inflate,
        dt=cfg_env.dt, steps=steps, d_safe=max(0.8*r_rest, 0.5), r_min=r_min
    )
    pf_success = success_reached(path_pf[-1] if len(path_pf) else start_xy, goal_xy, cfg_env.goal_tol)
    
    # 4. CBF - STAGED (same local info as GRL-SNAM)
    path_cbf, time_cbf = timing_wrapper(
        cbf_path_staged, world.C_np, world.R_np, start_xy, goal_xy,
        stage_size=stage_size, overlap=overlap, inflate=inflate,
        dt=cfg_env.dt, steps=steps, safety_margin=r_min
    )
    cbf_success = success_reached(path_cbf[-1] if len(path_cbf) else start_xy, goal_xy, cfg_env.goal_tol)
    
    # 5. DWA - STAGED (same local info as GRL-SNAM)
    path_dwa, time_dwa = timing_wrapper(
        dwa_path_staged, world.C_np, world.R_np, start_xy, goal_xy,
        stage_size=stage_size, overlap=overlap, inflate=inflate,
        dt=cfg_env.dt, steps=steps
    )
    dwa_success = success_reached(path_dwa[-1] if len(path_dwa) else start_xy, goal_xy, cfg_env.goal_tol)
    
    # 6. GRL-SNAM (Our method) - uses staged local information
    cfg_local = gen.GenCfg()
    cfg_local.__dict__.update(cfg_env.__dict__)
    cfg_local.start = start_xy.astype(np.float32)
    cfg_local.goal = goal_xy.astype(np.float32)
    cfg_local.stage_size = stage_size
    cfg_local.overlap = overlap
    cfg_local.inflate = inflate
    
    def rollout_ours():
        planner = gen.planner_from_cfg(
            cfg_local, world, cfg_local.k_bulk, cfg_local.gamma_s, cfg_local.d_hat, cfg_local.radius
        )
        dt = cfg_local.dt
        path, frames = [], []

        def _capture_frame(sys, t_idx):
            entry = {
                "center": sys.o.detach().cpu().to(torch.float32),
                "theta": (sys.theta.detach().cpu().to(torch.float32)
                        if hasattr(sys, "theta") else torch.tensor(0.0)),
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

        for t in range(steps):
            sys = planner.sys
            o_w = sys.o.detach().cpu().numpy()
            Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)

            # NN inputs
            obs_feats, goal_feats = build_local_feats(o_w, cfg_local.goal, Cw, Rw, Ww)
            obs_mask = (torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
                        if obs_feats.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device))

            with torch.no_grad():
                alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))

            # al_np  = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
            # beta_f = float(beta.squeeze(0).item())
            # gamma_f= float(gamma.squeeze(0).item())
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

            # map α once
            W_adj, R_adj = Ww.copy(), Rw.copy()
            if alpha_mode != "none":
                W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, alpha_mode, k_rad)

            # step
            world_step = ssi.WorldObstacles(Cw, R_adj, W_adj, d_hat=cfg_local.d_hat)
            planner.stage_field.w_goal = max(0.0, beta_f)
            planner.sys.gamma_o        = max(0.0, gamma_f)
            info = planner.step(dt, world_step)

            frames.append(_capture_frame(planner.sys, t+1))
            center_xy = np.asarray(info["center"], float)
            path.append(center_xy)

            if np.linalg.norm(center_xy - cfg_local.goal) <= cfg_local.goal_tol:
                break

        return np.array(path, dtype=np.float32), frames, planner
    
    (our_path, our_frames, our_planner), time_ours = timing_wrapper(rollout_ours)
    our_success = success_reached(our_path[-1] if len(our_path) else start_xy, goal_xy, cfg_env.goal_tol)
    
    # Package results with method categorization
    methods_data = [
        ('GRL-SNAM', our_path, our_success, time_ours, 'local_staged'),
        ('RigidA*', path_rigid, rigid_success, time_rigid, 'global_planning'),
        ('DeformA*', path_deform, deform_success, time_deform, 'global_planning'),  
        ('PotentialField', path_pf, pf_success, time_pf, 'local_staged'),
        ('CBF', path_cbf, cbf_success, time_cbf, 'local_staged'),
        ('DWA', path_dwa, dwa_success, time_dwa, 'local_staged'),
    ]
    
    for name, path, success, comp_time, info_type in methods_data:
        metrics = metric_pack(path, L_ref, reached=success,
                            C=world.C_np, R=world.R_np, 
                            barrier_thresh=cfg_env.d_hat, tube_thresh=r_min)
        metrics['computational_time'] = comp_time
        metrics['information_type'] = info_type  # Track info constraint type
        
        results[name] = {
            'path': path,
            'success': success,
            'metrics': metrics,
            **metrics  # Flatten metrics for easy access
        }
    
    return results, L_ref

def run_split(split_name, case_fn, args, device, model, run_dir):
    """Run evaluation split with all baseline methods"""
    print(f"\n=== {split_name}: {args.n_envs} envs × {args.n_trials} trials ===")
    
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
        
        # Precompute grids for A* methods
        r_rest = cfg.radius
        bounds = compute_workspace_bounds(world.C_np, world.R_np, margin=2.0*r_rest)
        res = max(0.2, 0.35 * max(r_rest, cfg.d_hat))
        occ_rigid, _, _ = rasterize_inflated_obstacles(world.C_np, world.R_np, r_rest, bounds, res)
        clearance_grid = clearance_field_on_grid(world.C_np, world.R_np, bounds, res)
        grid_cache = dict(bounds=bounds, res=res, occ_rigid=occ_rigid, clearance=clearance_grid)
        
        for trial_id in range(args.n_trials):
            # Sample start/goal
            def sample_valid_point(margin, bounds, tries=1000):
                xmin, xmax, ymin, ymax = bounds
                for _ in range(tries):
                    p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
                    if len(C) == 0:
                        return p
                    if np.all(np.linalg.norm(C - p[None,:], axis=1) - (R + margin) >= 0):
                        return p
                return p

            start = sample_valid_point(cfg.radius_min, bounds=(0, 8, -3, -2))
            goal = sample_valid_point(cfg.radius_min, bounds=(0, 8, 2, 3))
            if np.linalg.norm(start - goal) < 2.0*cfg.radius:
                goal = goal + np.array([2.5*cfg.radius, 0.0])
            
            # Episode directory
            ep_dir = mkdir(os.path.join(run_dir, f"{safe_name(split_name)}_env{env_id:03d}_trial{trial_id:03d}"))
            
            # Evaluate all methods
            results, L_ref = evaluate_all_methods(
                model, cfg, world, start, goal,
                alpha_mode=args.alpha_mode, k_rad=args.k_rad, 
                steps=args.steps, device=device, grid_cache=grid_cache
            )
            
            # Create visualizations
            method_names = list(results.keys())
            plot_comprehensive_comparison(ep_dir, world, results, start, goal, method_names)
            
            # Store results for aggregate analysis
            all_results.append(results)
            
            # Progress logging
            success_summary = " ".join([f"{m}:{int(results[m]['success'])}" for m in method_names])
            print(f"[{split_name} e{env_id} t{trial_id}] {success_summary}")
    
    # Create aggregate performance analysis
    plot_performance_metrics(run_dir, all_results, split_name)
    
    # Save detailed results
    results_path = os.path.join(run_dir, f"{safe_name(split_name)}_detailed_results.json")
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result_set in all_results:
        json_set = {}
        for method, data in result_set.items():
            json_set[method] = {
                'path': to_serializable_path(data.get('path')),
                'success': float(data['success']),
                'SPL': float(data['SPL']) if np.isfinite(data['SPL']) else None,
                'detour': float(data['detour']) if np.isfinite(data['detour']) else None,
                'min_clear': float(data['min_clear']) if np.isfinite(data['min_clear']) else None,
                'computational_time': float(data['computational_time']),
                'smoothness': float(data['smoothness']) if np.isfinite(data['smoothness']) else None,
            }
        json_results.append(json_set)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return all_results

def main():
    parser = argparse.ArgumentParser("Comprehensive Baseline Comparison")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","none"])  
    parser.add_argument("--k_rad", type=float, default=0.05)
    parser.add_argument("--case_id", type=str, default="case2-harder")
    parser.add_argument("--case_ood", type=str, default="case2-harder")
    parser.add_argument("--n_envs", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2000)
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
    run_dir = mkdir(f"comprehensive_baseline_comparison/{timestamp}")
    
    # Define case functions
    case_funcs = {
        "case1-tight": gen.sample_obstacles_case1_tight,
        "case2-harder": gen.sample_obstacles_case2_harder,
        "case2-dense": lambda cfg: gen.sample_obstacles_case2_harder(cfg)  # Can add density modifications
    }
    
    # Run evaluations
    id_results = run_split("TestID", case_funcs[args.case_id], args, device, model, run_dir)
    ood_results = run_split("TestOOD", case_funcs[args.case_ood], args, device, model, run_dir)
    
    # Create final summary
    summary = {
        "experiment": "Comprehensive Baseline Comparison",
        "timestamp": timestamp,
        "parameters": {
            "ckpt": args.ckpt,
            "alpha_mode": args.alpha_mode,
            "n_envs": args.n_envs,
            "n_trials": args.n_trials,
            "steps": args.steps
        },
        "methods_evaluated": ["GRL-SNAM", "RigidA*", "DeformA*", "PotentialField", "CBF", "DWA"],
        "splits": ["TestID", "TestOOD"]
    }
    
    with open(os.path.join(run_dir, "experiment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Experiment Complete ===")
    print(f"Results saved to: {run_dir}")
    print(f"Methods compared: {len(summary['methods_evaluated'])}")
    print(f"Total episodes: {args.n_envs * args.n_trials * 2}")

if __name__ == "__main__":
    main()