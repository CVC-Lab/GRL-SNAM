#!/usr/bin/env python3
"""
OPTIMIZED E7. Robustness to sensing + dynamics shift for CoefEnergyNet.

Key optimizations:
1. Reduced default parameter sweeps (much fewer combinations)
2. Early termination when goal reached
3. Progress indicators
4. Precomputed A* grids
5. Reasonable default values

Usage:
  python eval_robustness_shifts_optimized.py --ckpt checkpoints/best.pt --case case1-tight \
      --alpha_mode weight --quick_test
"""

import os, json, argparse, time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple, List
from tqdm import tqdm

# Project imports (match your repo)
from train_coef_energy import CoefEnergyNet
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi

# -----------------------------
# Optimized utilities
# -----------------------------
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

def _barrier_penalty_soft(clearance: float, thresh: float, scale: float = 1.0) -> float:
    if thresh <= 0.0: return 0.0
    gap = max(0.0, thresh - float(clearance))
    return float(scale * (gap * gap))

def compute_safety_stats(path_xy: List[np.ndarray], C: np.ndarray, R: np.ndarray,
                         barrier_thresh: float, tube_thresh: float,
                         *, penalty_scale: float = 1.0) -> Dict[str, float]:
    """Barrier/tube violations against TRUE world."""
    clears = clearance_profile(path_xy, C, R) if path_xy else []
    if not clears:
        return dict(min_clearance=np.inf, barrier_rate=0.0, tube_rate=0.0,
                    barrier_penalty_sum=0.0, barrier_penalty_mean=0.0)
    c = np.asarray(clears, float)
    barrier_mask = c < float(barrier_thresh)
    tube_mask    = c < float(tube_thresh)
    penalties = [_barrier_penalty_soft(ci, barrier_thresh, penalty_scale) for ci in c]
    return dict(
        min_clearance=float(np.min(c)),
        barrier_rate=float(np.mean(barrier_mask)),
        tube_rate=float(np.mean(tube_mask)),
        barrier_penalty_sum=float(np.sum(penalties)),
        barrier_penalty_mean=float(np.mean(penalties))
    )

# ---------------------------------
# Model feature helpers (your API)
# ---------------------------------
def build_local_feats(o_w: np.ndarray, goal_w: np.ndarray,
                      C_w: np.ndarray, R_w: np.ndarray, W_w: np.ndarray):
    o = torch.as_tensor(o_w, dtype=torch.float32)
    g = torch.as_tensor(goal_w, dtype=torch.float32)
    C = torch.as_tensor(C_w, dtype=torch.float32) if C_w.size else torch.zeros(0,2, dtype=torch.float32)
    R = torch.as_tensor(R_w, dtype=torch.float32) if R_w.size else torch.zeros(0, dtype=torch.float32)
    W = torch.as_tensor(W_w, dtype=torch.float32) if W_w.size else torch.zeros(0, dtype=torch.float32)
    if C.ndim == 1: C = C.reshape(0,2)
    dg = (g - o); gdist = torch.linalg.norm(dg).unsqueeze(0)
    goal_feats = torch.stack([dg[0], dg[1], gdist[0], torch.tensor(1.0)], dim=0).unsqueeze(0)
    if C.shape[0] == 0:
        obs_feats = torch.zeros(1,0,6, dtype=torch.float32)
    else:
        dxdy = (g.unsqueeze(0) - C)
        obs_feats = torch.cat([C, R.unsqueeze(-1), W.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)
    return obs_feats, goal_feats

def map_alpha_to_world(W_in: np.ndarray, R_in: np.ndarray, alphas: np.ndarray,
                       mode: str, k_rad: float = 0.05):
    al = np.maximum(alphas, 0.0)
    W_out = W_in.copy(); R_out = R_in.copy()
    if mode in ("weight","both") and al.size: W_out = W_out * al
    if mode in ("radius","both") and al.size: R_out = R_out + k_rad * al
    return W_out, R_out

# ---------------------------------
# Sensing corruption generators
# ---------------------------------
def corrupt_sensing(C: np.ndarray, R: np.ndarray, W: np.ndarray,
                    *, pos_sigma: float = 0.0, rad_sigma: float = 0.0,
                    miss_prob: float = 0.0, n_false: int = 0,
                    bounds: Tuple[float,float,float,float] = None,
                    rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a *sensed* obstacle set (C~, R~, W~)"""
    if rng is None: rng = np.random.default_rng()
    C = C.copy(); R = R.copy(); W = W.copy()

    # (1) Missed detections
    if C.size and miss_prob > 0.0:
        keep = rng.random(len(C)) > miss_prob
        C, R, W = C[keep], R[keep], W[keep]

    # (2) Position noise
    if C.size and pos_sigma > 0.0:
        C = C + rng.normal(0.0, pos_sigma, size=C.shape)

    # (3) Radius noise
    if R.size and rad_sigma > 0.0:
        scale = 1.0 + rng.normal(0.0, rad_sigma, size=R.shape)
        R = np.maximum(0.05, R * scale)

    # (4) False positives
    if n_false > 0 and bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        C_fp = np.stack([
            rng.uniform(xmin, xmax, size=n_false),
            rng.uniform(ymin, ymax, size=n_false)
        ], axis=-1)
        R_fp = rng.uniform(0.08, 0.25, size=n_false)
        W_fp = rng.uniform(0.4, 1.0, size=n_false)
        C = np.concatenate([C, C_fp], axis=0) if C.size else C_fp
        R = np.concatenate([R, R_fp], axis=0) if R.size else R_fp
        W = np.concatenate([W, W_fp], axis=0) if W.size else W_fp

    return C, R, W

# ---------------------------------
# Dynamics shift injectors
# ---------------------------------
def inject_dynamics_shift(planner, *, vel_sigma: float = 0.0, gamma_scale: float = 1.0,
                          rng: np.random.Generator = None):
    """Applies dynamics perturbations for this step"""
    if rng is None: rng = np.random.default_rng()
    if hasattr(planner.sys, "gamma_o"):
        base = float(planner.sys.gamma_o.item()) if torch.is_tensor(planner.sys.gamma_o) else float(planner.sys.gamma_o)
        planner.sys.gamma_o = base * gamma_scale
    if vel_sigma > 0.0 and hasattr(planner.sys, "v_o"):
        dv = rng.normal(0.0, vel_sigma, size=2).astype(np.float32)
        vo = planner.sys.v_o.detach().cpu().numpy().astype(np.float32)
        planner.sys.v_o = torch.tensor(vo + dv, dtype=planner.sys.dtype)

# ---------------------------------
# Optimized A* (precomputed grid)
# ---------------------------------
class AStarCache:
    """Precompute A* grid once per environment to avoid repeated computation."""
    def __init__(self, world_C, world_R, r_rest, bounds, res):
        self.bounds = bounds
        self.res = res
        self.occ_grid = self._rasterize_inflated_obstacles(world_C, world_R, r_rest, bounds, res)
    
    def _rasterize_inflated_obstacles(self, C, R, inflation, bounds, res):
        xmin, xmax, ymin, ymax = bounds
        W = int(np.ceil((xmax - xmin)/res)) + 1
        H = int(np.ceil((ymax - ymin)/res)) + 1
        occ = np.zeros((H, W), dtype=bool)
        if len(C) == 0: return occ
        yy = np.linspace(ymin, ymax, H)
        xx = np.linspace(xmin, xmax, W)
        Y, X = np.meshgrid(yy, xx, indexing='ij')
        for (cx, cy), r in zip(C, R):
            occ |= ((X - cx)**2 + (Y - cy)**2 <= (r + inflation)**2)
        return occ
    
    def _world_to_ij(self, xy):
        xmin, xmax, ymin, ymax = self.bounds
        j = int(np.round((xy[0] - xmin)/self.res))
        i = int(np.round((xy[1] - ymin)/self.res))
        return i, j
    
    def _ij_to_world(self, i, j):
        xmin, xmax, ymin, ymax = self.bounds
        return np.array([xmin + j*self.res, ymin + i*self.res], np.float32)
    
    def find_path(self, start_xy, goal_xy):
        """Returns (length, path_xy) or (inf, [])"""
        from heapq import heappush, heappop
        H, W = self.occ_grid.shape
        si, sj = self._world_to_ij(start_xy)
        gi, gj = self._world_to_ij(goal_xy)
        
        if not (0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): 
            return np.inf, []
        if self.occ_grid[si, sj] or self.occ_grid[gi, gj]: 
            return np.inf, []
        
        nbrs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]
        g = np.full((H,W), np.inf); came = np.full((H,W,2), -1, int)
        def h(i,j): return np.hypot(i-gi, j-gj)
        
        pq = []; g[si,sj] = 0.0; heappush(pq, (h(si,sj), (si,sj)))
        visited = np.zeros_like(self.occ_grid, bool)
        
        while pq:
            _, (i,j) = heappop(pq)
            if visited[i,j]: continue
            visited[i,j] = True
            if (i,j) == (gi,gj): break
            
            for di,dj,c in nbrs:
                ni, nj = i+di, j+dj
                if not (0<=ni<H and 0<=nj<W): continue
                if self.occ_grid[ni,nj]: continue
                tentative = g[i,j] + c
                if tentative < g[ni,nj]:
                    g[ni,nj] = tentative; came[ni,nj] = [i,j]
                    heappush(pq, (tentative + h(ni,nj), (ni,nj)))
        
        if not np.isfinite(g[gi,gj]): return np.inf, []
        
        # Reconstruct path
        path_ij = []; ci, cj = gi, gj; path_ij.append((ci,cj))
        while not (ci == si and cj == sj):
            pi, pj = came[ci, cj]; 
            if pi < 0: break
            path_ij.append((pi,pj)); ci, cj = pi, pj
        path_ij.reverse()
        
        # Convert to world coordinates and compute length
        length = 0.0
        for k in range(1,len(path_ij)):
            i0,j0 = path_ij[k-1]; i1,j1 = path_ij[k]
            length += np.hypot(i1-i0, j1-j0) * self.res
        
        path_xy = np.stack([self._ij_to_world(i,j) for (i,j) in path_ij], 0) if path_ij else []
        return float(length), path_xy

# ---------------------------------
# Case configuration
# ---------------------------------
def get_case(case: str):
    if case.startswith("case1"):
        return gen.sample_obstacles_case1_tight
    elif case.startswith("case2"):
        return gen.sample_obstacles_case2_harder
    else:
        return gen.sample_obstacles_case1_tight

# ---------------------------------
# Plotting (robustness curves)
# ---------------------------------
def plot_robustness_curves(outdir: str, xlevels: List[float], series: Dict[str, List[float]], 
                          title: str, ylabel: str, xlabel: str = "Corruption level"):
    plt.figure(figsize=(8,5))
    for k,v in series.items():
        plt.plot(xlevels, v, marker='o', linewidth=2, markersize=6, label=k)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout()
    fname = f"{title.replace(' ','_').replace('(','').replace(')','').lower()}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight')
    plt.close()

# ---------------------------------
# Main eval
# ---------------------------------
def main():
    ap = argparse.ArgumentParser("OPTIMIZED E7 Robustness evaluation")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","nonce"])
    ap.add_argument("--case", type=str, default="case1-tight")
    ap.add_argument("--seed", type=int, default=2312)
    ap.add_argument("--steps", type=int, default=1800)  # Reduced from 1800
    ap.add_argument("--k_rad", type=float, default=0.05)
    ap.add_argument("--n_envs", type=int, default=1)  # Reduced from 3
    ap.add_argument("--n_trials", type=int, default=1)  # Reduced from 3
    ap.add_argument("--gif_stride", type=int, default=30)
    
    # MUCH more reasonable default sweeps
    ap.add_argument("--sensing_sweep", type=str,
                    default="pos=0,0.05 rad=0,0.1 miss=0,0.2 fp=0,2",  # 2×2×2×2 = 16 combinations
                    help="Space-separated keys, each with comma-separated levels")
    ap.add_argument("--dyn_sweep", type=str,
                    default="vel=0,0.3 gamma=1.0,0.6",  # 2×2 = 4 combinations
                    help="Keys: vel (disturbance), gamma (friction scale)")
    
    # Quick test option
    ap.add_argument("--quick_test", action="store_true", 
                    help="Ultra-fast test with minimal parameters")
    
    ap.add_argument("--sensing_only", action="store_true")
    ap.add_argument("--dynamics_only", action="store_true")
    ap.add_argument("--save_overlays", action="store_true")
    args = ap.parse_args()

    # Quick test overrides
    if args.quick_test:
        args.sensing_sweep = "pos=0,0.1 rad=0,0.2"  # 2×2 = 4 combinations
        args.dyn_sweep = "vel=0,0.5 gamma=1.0,0.5"  # 2×2 = 4 combinations  
        args.n_envs = 1
        args.n_trials = 1
        args.steps = 400
        print("[QUICK TEST MODE] Minimal parameter sweep for fast validation")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = CoefEnergyNet().to(device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state); model.eval()
    print("✓ Model loaded")

    # Parse sweeps
    def parse_sweep(s: str) -> Dict[str, List[float]]:
        out = {}
        for token in s.strip().split():
            k, vals = token.split("=")
            out[k] = [float(x) for x in vals.split(",")]
        return out
    
    sensing_grid = parse_sweep(args.sensing_sweep) if not args.dynamics_only else {"pos":[0.0]}
    dyn_grid     = parse_sweep(args.dyn_sweep) if not args.sensing_only else {"vel":[0.0], "gamma":[1.0]}

    # Build level combinations
    def cartesian(d: Dict[str, List[float]]):
        keys = list(d.keys())
        grids = np.meshgrid(*[d[k] for k in keys], indexing="ij")
        combos = []
        for idx in np.ndindex(*[len(d[k]) for k in keys]):
            combos.append({k: float(grids[i][idx]) for i,k in enumerate(keys)})
        return keys, combos

    sense_keys, sense_levels = cartesian(sensing_grid)
    dyn_keys, dyn_levels     = cartesian(dyn_grid)

    total_combinations = len(sense_levels) * len(dyn_levels)
    total_episodes = total_combinations * args.n_envs * args.n_trials
    
    print(f"Parameter sweep:")
    print(f"  Sensing: {len(sense_levels)} combinations {sense_levels}")
    print(f"  Dynamics: {len(dyn_levels)} combinations {dyn_levels}")
    print(f"  Total: {total_combinations} combinations")
    print(f"  Episodes: {total_episodes} ({args.n_envs} envs × {args.n_trials} trials)")
    print(f"  Max steps per episode: {args.steps}")

    if total_episodes > 1000:
        print(f"[WARNING] {total_episodes} episodes will take a long time!")
        print("Consider using --quick_test or reducing --n_envs/--n_trials")

    # Bookkeeping
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = mkdir(f"snaps_robustness/{stamp}")
    per_episode_rows = []

    def level_name(d: Dict[str,float]): 
        return "_".join([f"{k}{v}".replace(".","p") for k,v in d.items()])

    # Global RNG
    master_rng = np.random.default_rng(args.seed)
    
    # Progress tracking
    start_time = time.time()
    combination_idx = 0
    
    # Loop over sensing and dynamics levels with progress bar
    total_pbar = tqdm(total=total_combinations, desc="Robustness sweep", unit="combination")
    
    for s_idx, s_lv in enumerate(sense_levels):
        for d_idx, d_lv in enumerate(dyn_levels):
            combination_idx += 1
            level_dir = mkdir(os.path.join(run_dir, f"sense_{level_name(s_lv)}__dyn_{level_name(d_lv)}"))
            
            # Update progress bar description
            total_pbar.set_description(f"Sens:{level_name(s_lv)} Dyn:{level_name(d_lv)}")

            # Aggregate per-level
            succ_list = []; spl_list = []; detour_list = []; minclear_list = []
            barr_rate_list = []; tube_rate_list = []

            # Environments loop
            for env_id in range(args.n_envs):
                cfg = gen.GenCfg()
                cfg.seed = int(master_rng.integers(0, 10_000_000))
                gen.set_all_seeds(cfg.seed)
                cfg.d_hat = getattr(cfg, "d_hat", 0.5)
                C, R, W = get_case(args.case)(cfg)
                world_true = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)

                # Setup params and A* cache
                r_rest = getattr(cfg, "radius", cfg.d_hat)
                r_min  = getattr(cfg, "radius_min", 0.3 * r_rest)
                bounds = compute_workspace_bounds(world_true.C_np, world_true.R_np, margin=2.0*max(r_rest, cfg.d_hat))
                res = max(r_rest, cfg.d_hat) * 0.5
                
                # Precompute A* grid for this environment
                astar_cache = AStarCache(world_true.C_np, world_true.R_np, r_rest, bounds, res)

                # Trials per environment
                for trial_id in range(args.n_trials):
                    rng = np.random.default_rng(int(master_rng.integers(0, 10_000_000)))
                    ep_dir = mkdir(os.path.join(level_dir, f"env{env_id:02d}_trial{trial_id:02d}"))

                    # Sample start/goal (simple but valid)
                    def sample_valid_point(bounds, C, R, margin, rng, tries=500):
                        xmin,xmax,ymin,ymax = bounds
                        for _ in range(tries):
                            p = np.array([rng.uniform(xmin,xmax), rng.uniform(ymin,ymax)], np.float32)
                            if clearance_to_circles(p, C, R) >= margin: return p
                        return np.array([xmin+1,ymin+1], np.float32)
                    
                    start = sample_valid_point(bounds, world_true.C_np, world_true.R_np, r_min, rng)
                    goal  = sample_valid_point(bounds, world_true.C_np, world_true.R_np, r_min, rng)
                    while np.linalg.norm(start-goal) < 2.0*r_rest:
                        goal = sample_valid_point(bounds, world_true.C_np, world_true.R_np, r_min, rng)

                    # Fresh planner
                    cfg_ep = gen.GenCfg()
                    cfg_ep.seed = int(rng.integers(0, 10_000_000))
                    gen.set_all_seeds(cfg_ep.seed)
                    cfg_ep.d_hat   = cfg.d_hat
                    cfg_ep.radius  = getattr(cfg, "radius", 0.5)
                    cfg_ep.k_bulk  = getattr(cfg, "k_bulk", 20.0)
                    cfg_ep.gamma_s = getattr(cfg, "gamma_s", 0.05)
                    cfg_ep.dt      = getattr(cfg, "dt", 0.01)
                    cfg_ep.total_steps = args.steps
                    cfg_ep.start = start.astype(np.float32)
                    cfg_ep.goal  = goal.astype(np.float32)

                    planner = gen.planner_from_cfg(cfg_ep, world_true, cfg_ep.k_bulk,
                                                   cfg_ep.gamma_s, cfg_ep.d_hat, cfg_ep.radius)

                    # Rollout with early termination
                    path = []; min_clear = np.inf; collisions = 0; success = False

                    for t in range(args.steps):
                        sys = planner.sys
                        o_w = sys.o.detach().cpu().numpy()
                        Cw, Rw, Ww = planner.stage_slice(world_true.C_np, world_true.R_np, world_true.W_np)

                        # Sensing corruption
                        C_s, R_s, W_s = corrupt_sensing(
                            Cw, Rw, Ww,
                            pos_sigma=s_lv.get("pos", 0.0),
                            rad_sigma=s_lv.get("rad", 0.0),
                            miss_prob=s_lv.get("miss", 0.0),
                            n_false=int(round(s_lv.get("fp", 0.0))),
                            bounds=bounds, rng=rng
                        )

                        # Model prediction
                        obs_feats, goal_feats = build_local_feats(o_w, cfg_ep.goal, C_s, R_s, W_s)
                        obs_mask = (torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
                                    if obs_feats.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device))

                        with torch.no_grad():
                            alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))

                        al_np  = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(R_s)
                        beta_f = float(beta.squeeze(0).item())
                        gamma_f= float(gamma.squeeze(0).item())

                        W_adj, R_adj = W_s.copy(), R_s.copy()
                        if args.alpha_mode != "none":
                            W_adj, R_adj = map_alpha_to_world(W_s, R_s, al_np, args.alpha_mode, args.k_rad)

                        world_step = ssi.WorldObstacles(C_s, R_adj, W_adj, d_hat=cfg_ep.d_hat)
                        planner.stage_field.w_goal = max(0.0, beta_f)
                        planner.sys.gamma_o = max(0.0, gamma_f)

                        # Dynamics shift
                        inject_dynamics_shift(planner,
                                              vel_sigma=d_lv.get("vel", 0.0),
                                              gamma_scale=d_lv.get("gamma", 1.0),
                                              rng=rng)

                        info = planner.step(cfg_ep.dt, world_step)
                        center_xy = np.asarray(info["center"], float)
                        path.append(center_xy)

                        d_clear = clearance_to_circles(center_xy, world_true.C_np, world_true.R_np)
                        min_clear = min(min_clear, d_clear)
                        if d_clear < 0.0: collisions += 1

                        # EARLY TERMINATION - key optimization!
                        if np.linalg.norm(center_xy - cfg_ep.goal) <= cfg_ep.goal_tol:
                            success = True; break

                    # Use precomputed A* for reference
                    L_ref, _ = astar_cache.find_path(start, goal)
                    L_exec = path_length(path)
                    reached = success_reached(path[-1] if path else start, goal, cfg_ep.goal_tol)
                    detour = (L_exec / L_ref) if (np.isfinite(L_ref) and L_ref>0) else np.nan
                    spl    = (L_ref / max(L_ref, L_exec)) if (reached and np.isfinite(L_ref) and L_ref>0) else 0.0

                    safety = compute_safety_stats(
                        path, world_true.C_np, world_true.R_np,
                        barrier_thresh=float(cfg_ep.d_hat), tube_thresh=float(r_min)
                    )

                    row = {
                        "env_id": env_id, "trial_id": trial_id,
                        "sense_pos": s_lv.get("pos", 0.0), "sense_rad": s_lv.get("rad", 0.0),
                        "sense_miss": s_lv.get("miss", 0.0), "sense_fp": s_lv.get("fp", 0.0),
                        "dyn_vel": d_lv.get("vel", 0.0), "dyn_gamma": d_lv.get("gamma", 1.0),
                        "success": bool(reached), "SPL_vs_ref": float(spl),
                        "detour_vs_ref": float(detour) if np.isfinite(detour) else float("nan"),
                        "path_length": float(L_exec), "L_ref_rigid": float(L_ref) if np.isfinite(L_ref) else float("inf"),
                        "min_clearance": float(safety["min_clearance"]) if np.isfinite(safety["min_clearance"]) else float("nan"),
                        "barrier_rate": float(safety["barrier_rate"]), "tube_rate": float(safety["tube_rate"]),
                        "steps_taken": len(path), "early_termination": success
                    }
                    per_episode_rows.append(row)

                    # Aggregate
                    succ_list.append(1.0 if reached else 0.0)
                    spl_list.append(spl)
                    detour_list.append(detour if np.isfinite(detour) else np.nan)
                    minclear_list.append(safety["min_clearance"])
                    barr_rate_list.append(safety["barrier_rate"])
                    tube_rate_list.append(safety["tube_rate"])

            total_pbar.update(1)
            
            # Per-level summary
            def _nanmean(a): 
                arr=np.asarray(a,float)
                return float(np.nanmean(arr)) if arr.size else float("nan")
            
            level_agg = {
                "sensing": s_lv, "dynamics": d_lv,
                "success_mean": _nanmean(succ_list),
                "SPL_mean": _nanmean(spl_list),
                "detour_mean": _nanmean(detour_list),
                "min_clear_mean": _nanmean(minclear_list),
                "barrier_rate_mean": _nanmean(barr_rate_list),
                "tube_rate_mean": _nanmean(tube_rate_list),
            }
            with open(os.path.join(level_dir, "aggregate.json"), "w") as f:
                json.dump(level_agg, f, indent=2)

    total_pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\n✓ Completed {len(per_episode_rows)} episodes in {elapsed:.1f}s ({elapsed/len(per_episode_rows):.2f}s per episode)")

    # Write results
    csv_path = os.path.join(run_dir, "episodes.csv")
    fields = ["env_id","trial_id","success","SPL_vs_ref","detour_vs_ref","path_length",
              "L_ref_rigid","min_clearance","barrier_rate","tube_rate","steps_taken","early_termination",
              "sense_pos","sense_rad","sense_miss","sense_fp","dyn_vel","dyn_gamma"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_episode_rows:
            w.writerow(r)

    with open(os.path.join(run_dir, "episodes.json"), "w") as f:
        json.dump(per_episode_rows, f, indent=2)

    # Generate plots
    plots_dir = mkdir(os.path.join(run_dir, "robustness_plots"))
    
    # Helper to collect data for plotting
    def collect_for_param(param_name, param_values, fixed_others=None):
        if fixed_others is None:
            fixed_others = {}
        
        success_vals = []
        spl_vals = []
        barrier_vals = []
        
        for val in param_values:
            matching_rows = []
            for row in per_episode_rows:
                match = (row[param_name] == val)
                for other_param, other_val in fixed_others.items():
                    match &= (row[other_param] == other_val)
                if match:
                    matching_rows.append(row)
            
            if matching_rows:
                success_vals.append(np.mean([r["success"] for r in matching_rows]))
                spl_vals.append(np.mean([r["SPL_vs_ref"] for r in matching_rows]))
                barrier_vals.append(np.mean([r["barrier_rate"] for r in matching_rows]))
            else:
                success_vals.append(0.0)
                spl_vals.append(0.0)
                barrier_vals.append(0.0)
        
        return success_vals, spl_vals, barrier_vals

    # Create plots for each parameter
    baseline_sensing = {k: sensing_grid[k][0] for k in sensing_grid.keys()}
    baseline_dyn = {k: dyn_grid[k][0] for k in dyn_grid.keys()}
    
    for param in sensing_grid.keys():
        values = sensing_grid[param]
        if len(values) > 1:
            fixed_others = {f"dyn_{k}": v for k, v in baseline_dyn.items()}
            success, spl, barrier = collect_for_param(f"sense_{param}", values, fixed_others)
            
            plot_robustness_curves(plots_dir, values, {"Success": success}, 
                                 f"Success vs {param} (sensing)", "Success Rate", f"{param} corruption")
            plot_robustness_curves(plots_dir, values, {"SPL": spl}, 
                                 f"SPL vs {param} (sensing)", "SPL", f"{param} corruption")
            plot_robustness_curves(plots_dir, values, {"Barrier Violation": barrier}, 
                                 f"Safety vs {param} (sensing)", "Violation Rate", f"{param} corruption")

    for param in dyn_grid.keys():
        values = dyn_grid[param]
        if len(values) > 1:
            fixed_others = {f"sense_{k}": v for k, v in baseline_sensing.items()}
            success, spl, barrier = collect_for_param(f"dyn_{param}", values, fixed_others)
            
            plot_robustness_curves(plots_dir, values, {"Success": success}, 
                                 f"Success vs {param} (dynamics)", "Success Rate", f"{param} shift")
            plot_robustness_curves(plots_dir, values, {"SPL": spl}, 
                                 f"SPL vs {param} (dynamics)", "SPL", f"{param} shift")
            plot_robustness_curves(plots_dir, values, {"Barrier Violation": barrier}, 
                                 f"Safety vs {param} (dynamics)", "Violation Rate", f"{param} shift")

    # Global summary
    def _nanmean(a):
        arr=np.asarray(a,float); return float(np.nanmean(arr)) if arr.size else float("nan")
    
    agg_global = {
        "n_episodes": len(per_episode_rows),
        "success_mean": _nanmean([r["success"] for r in per_episode_rows]),
        "SPL_mean": _nanmean([r["SPL_vs_ref"] for r in per_episode_rows]),
        "detour_mean": _nanmean([r["detour_vs_ref"] for r in per_episode_rows if np.isfinite(r["detour_vs_ref"])]),
        "barrier_rate_mean": _nanmean([r["barrier_rate"] for r in per_episode_rows]),
        "early_termination_rate": _nanmean([r["early_termination"] for r in per_episode_rows]),
        "avg_steps": _nanmean([r["steps_taken"] for r in per_episode_rows])
    }
    
    with open(os.path.join(run_dir, "aggregate_global.json"), "w") as f:
        json.dump(agg_global, f, indent=2)

    print(f"\n=== OPTIMIZED ROBUSTNESS EVALUATION COMPLETE ===")
    print(f"Episodes: {len(per_episode_rows)}")
    print(f"Success Rate: {agg_global['success_mean']:.1%}")
    print(f"Average SPL: {agg_global['SPL_mean']:.3f}")
    print(f"Early Termination Rate: {agg_global['early_termination_rate']:.1%}")
    print(f"Average Steps: {agg_global['avg_steps']:.1f}")
    print(f"Results: {run_dir}")
    print(f"Plots: {plots_dir}")

if __name__ == "__main__":
    main()