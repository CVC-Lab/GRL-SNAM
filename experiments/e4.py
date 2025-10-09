#!/usr/bin/env python3
"""
E4. Deformability Advantage (Narrow-Passage Stress Tests)

Sweeps the bottleneck ratio rho = w / r_rest (gap width over nominal ring radius),
evaluates GRL-SNAM (CoefEnergyNet) vs. two A* baselines:
  - Rigid A*: inflated obstacles by r_rest (feasibility == existence of path)
  - Deformable A*: feasibility c>=r_min with clearance-penalized cost

Outputs:
  - Per-episode CSV/JSON with metrics
  - Aggregated stats per ratio
  - Plots:
      * Success vs ratio (all 3 methods)
      * SPL vs ratio (ours; reference = deformable A* if feasible else rigid A*)
      * Detour vs ratio (ours vs reference)
      * Min-clearance vs ratio (ours)
      * Feasibility bands for the two A* baselines
      * Example overlays for a few ratios
"""

import os, json, argparse, csv
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt

# --- Project imports
from train_coef_energy import CoefEnergyNet
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi

# ===================== Reused helpers (trimmed) =====================

def mkdir(p): os.makedirs(p, exist_ok=True); return p

def compute_workspace_bounds(C, R, margin=1.0):
    if len(C) == 0: return -6.0, 6.0, -4.0, 4.0
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
    return np.array([xmin + j*res, ymin + i*res], dtype=np.float32)

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
    pq=[]; g[si,sj]=0.0; heappush(pq,(h(si,sj),(si,sj)))
    visited = np.zeros_like(occ, dtype=bool)
    while pq:
        _, (i,j) = heappop(pq)
        if visited[i,j]: continue
        visited[i,j] = True
        if (i,j)==(gi,gj): break
        for di,dj,c in nbrs:
            ni, nj = i+di, j+dj
            if not (0<=ni<H and 0<=nj<W): continue
            if occ[ni,nj]: continue
            tentative = g[i,j]+c
            if tentative < g[ni,nj]:
                g[ni,nj]=tentative; came[ni,nj]=[i,j]
                heappush(pq,(tentative+h(ni,nj),(ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf,[]
    # reconstruct
    path_ij=[(gi,gj)]
    ci,cj=gi,gj
    while not (ci==si and cj==sj):
        pi,pj = came[ci,cj]
        if pi<0: break
        path_ij.append((pi,pj)); ci,cj=pi,pj
    path_ij.reverse()
    # metric length (m)
    length_steps = sum(np.hypot(path_ij[k][0]-path_ij[k-1][0], path_ij[k][1]-path_ij[k-1][1]) for k in range(1,len(path_ij)))
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
    if len(C)==0: return np.full((H,W), np.inf, dtype=np.float32)
    clear = np.full((H,W), np.inf, dtype=np.float32)
    for (cx,cy),r in zip(C,R):
        d = np.sqrt((X-cx)**2+(Y-cy)**2).astype(np.float32) - float(r)
        clear = np.minimum(clear, d)
    return clear

def deformable_a_star_path(clearance, bounds, res, start_xy, goal_xy, r_rest, r_min, lam=5.0, beta=0.6, eps=1e-3):
    from heapq import heappush, heappop
    H,W = clearance.shape
    si,sj = world_to_ij(start_xy,bounds,res)
    gi,gj = world_to_ij(goal_xy,bounds,res)
    if not (0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): return np.inf,[]
    feas = clearance >= r_min
    if not (feas[si,sj] and feas[gi,gj]): return np.inf,[]
    nbrs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
            (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]
    def h(i,j): return np.hypot(i-gi,j-gj)*res
    def pen(c):
        alpha = np.maximum(0.0, (r_rest/(c+eps))-1.0)
        return lam*(alpha**2)
    g = np.full((H,W), np.inf); came = np.full((H,W,2), -1, dtype=np.int32)
    pq=[]; g[si,sj]=0.0; heappush(pq,(h(si,sj),(si,sj)))
    visited = np.zeros((H,W), dtype=bool)
    while pq:
        _,(i,j)=heappop(pq)
        if visited[i,j]: continue
        visited[i,j]=True
        if (i,j)==(gi,gj): break
        if not feas[i,j]: continue
        pij = pen(clearance[i,j])
        for di,dj,cstep in nbrs:
            ni,nj=i+di,j+dj
            if not (0<=ni<H and 0<=nj<W): continue
            if not feas[ni,nj]: continue
            base = cstep*res
            deform = beta*0.5*(pij+pen(clearance[ni,nj]))*base
            tentative = g[i,j] + base + deform
            if tentative < g[ni,nj]:
                g[ni,nj]=tentative; came[ni,nj]=[i,j]
                heappush(pq,(tentative+h(ni,nj),(ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf,[]
    path=[(gi,gj)]
    ci,cj=gi,gj
    while not (ci==si and cj==sj):
        pi,pj=came[ci,cj]
        if pi<0: break
        path.append((pi,pj)); ci,cj=pi,pj
    path.reverse()
    length_m = sum(np.hypot(path[k][0]-path[k-1][0], path[k][1]-path[k-1][1])*res for k in range(1,len(path)))
    path_xy = np.stack([ij_to_world(i,j,bounds,res) for (i,j) in path], axis=0) if path else []
    return float(length_m), path_xy

def build_local_feats(o_w: np.ndarray, goal_w: np.ndarray, C_w: np.ndarray, R_w: np.ndarray, W_w: np.ndarray):
    o = torch.as_tensor(o_w, dtype=torch.float32)
    g = torch.as_tensor(goal_w, dtype=torch.float32)
    C = torch.as_tensor(C_w, dtype=torch.float32) if C_w.size else torch.zeros(0,2, dtype=torch.float32)
    R = torch.as_tensor(R_w, dtype=torch.float32) if R_w.size else torch.zeros(0, dtype=torch.float32)
    W = torch.as_tensor(W_w, dtype=torch.float32) if W_w.size else torch.zeros(0, dtype=torch.float32)
    if C.ndim==1: C=C.reshape(0,2)
    dg=(g-o); gdist=torch.linalg.norm(dg).unsqueeze(0)
    goal_feats = torch.stack([dg[0], dg[1], gdist[0], torch.tensor(1.0)], dim=0).unsqueeze(0)
    if C.shape[0]==0:
        obs_feats=torch.zeros(1,0,6, dtype=torch.float32)
    else:
        dxdy=(g.unsqueeze(0)-C)
        obs_feats=torch.cat([C, R.unsqueeze(-1), W.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)
    return obs_feats, goal_feats

def map_alpha_to_world(W_in: np.ndarray, R_in: np.ndarray, alphas: np.ndarray, mode: str, k_rad: float=0.05):
    al = np.maximum(alphas, 0.0)
    W_out, R_out = W_in.copy(), R_in.copy()
    if mode in ("weight","both") and al.size: W_out = W_out * al
    if mode in ("radius","both") and al.size: R_out = R_out + k_rad*al
    return W_out, R_out

def path_length(path_xy):
    if len(path_xy)<2: return 0.0
    diffs = np.diff(np.asarray(path_xy), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def clearance_to_circles(xy, C, R):
    if len(C)==0: return np.inf
    d = np.linalg.norm(C-xy[None,:], axis=1) - R
    return float(np.min(d))

def success_reached(last_center, goal_xy, tol):
    return np.linalg.norm(np.asarray(last_center)-np.asarray(goal_xy)) <= tol

# ===================== Corridor generator =====================

def make_corridor_env(r_rest: float,
                             rho: float,
                             *,
                             r_min: float,
                             length: float = 14.0,
                             y_half: float = 2.6,
                             wall_r: float = 0.85,
                             wall_dx: float = 1.0,
                             slit_pillar_scale: float = 1.05,
                             seed: int = 0):
    """
    Deterministic 'hourglass' corridor:
      - Two pillar-disks at y=0 creating a slit of width gap = 2*rho*r_rest (by construction).
      - Top/bottom walls are chains of disks far enough to not shrink the slit.
      - Start/goal placed on centerline well away from pillars.
    Guarantees:
      * Deformable feasible  <=>  rho >= (r_min / r_rest)
      * Rigid feasible       <=>  rho >= 1
    """
    rng = np.random.default_rng(seed)

    # ----- geometry you can *reason about* -----
    gap = 2.0 * rho * r_rest                     # desired slit width
    r_p = slit_pillar_scale * r_rest             # pillar radius (slightly > r_rest for realism)
    # center-to-center separation for pillars to realize 'gap':
    d_centers = 2.0 * r_p + gap
    cL = np.array([-0.5 * d_centers, 0.0], dtype=np.float64)
    cR = np.array([+0.5 * d_centers, 0.0], dtype=np.float64)

    # Outer walls: rows of small disks at y = ±y_half, spaced along x
    xs = np.arange(-0.5 * length, 0.5 * length + 1e-6, wall_dx)
    top_centers = np.stack([xs, np.full_like(xs, +y_half)], axis=1)
    bot_centers = np.stack([xs, np.full_like(xs, -y_half)], axis=1)
    wall_centers = np.concatenate([top_centers, bot_centers], axis=0)
    wall_radii   = np.full(wall_centers.shape[0], wall_r, dtype=np.float64)
    wall_weights = np.ones_like(wall_radii)

    # Pillars
    C = np.vstack([wall_centers, cL[None, :], cR[None, :]])
    R = np.concatenate([wall_radii, np.array([r_p, r_p], dtype=np.float64)], axis=0)
    W = np.ones_like(R)

    # Bounds: leave buffer above/below walls
    xmin, xmax = -0.5 * length - 1.0, 0.5 * length + 1.0
    ymin, ymax = -y_half - 1.0, y_half + 1.0
    bounds = (xmin, xmax, ymin, ymax)

    # Place start/goal on centerline with padding from pillars
    pad = max(2.5 * r_rest, 2.0)  # distance from each pillar
    start_xy = np.array([xmin + 1.0, 0.0], dtype=np.float64)
    goal_xy  = np.array([xmax - 1.0, 0.0], dtype=np.float64)
    # If pillars are too close to ends, nudge inward
    start_xy[0] = min(start_xy[0], cL[0] - pad)
    goal_xy[0]  = max(goal_xy[0],  cR[0] + pad)

    # --- sanity: analytic checks for the regimes you expect ---
    rigid_feasible_theory  = rho >= 1.0
    deform_feasible_theory = rho >= (r_min / r_rest)

    # A tiny slack so later raster inflation doesn’t erase the slit
    # (clearance grid still uses exact distance to circles)
    # You can tune if needed:
    slack_report = {
        "gap": float(gap),
        "theory_rigid_feasible": bool(rigid_feasible_theory),
        "theory_deform_feasible": bool(deform_feasible_theory),
        "r_pillar": float(r_p)
    }

    return (C, R, W, bounds, start_xy, goal_xy, slack_report, cL, cR)


# ===================== Plotting =====================

def plot_overlay_example(outdir, world, our_path, path_rigid, path_deform, start, goal, title):
    plt.figure(figsize=(6.2,4.6))
    th = np.linspace(0,2*np.pi,128)
    for (cx,cy), r in zip(world.C_np, world.R_np):
        plt.plot(cx + r*np.cos(th), cy + r*np.sin(th), lw=1.2, color='k', alpha=0.7)
    if len(our_path): P=np.asarray(our_path); plt.plot(P[:,0],P[:,1],'-',lw=2.2,label='GRL-SNAM',alpha=0.95,color='tab:orange')
    if path_rigid is not None and len(path_rigid)>0:
        plt.plot(path_rigid[:,0],path_rigid[:,1],'--',lw=2,label='Rigid A*',alpha=0.9,color='tab:blue')
    if path_deform is not None and len(path_deform)>0:
        plt.plot(path_deform[:,0],path_deform[:,1],':',lw=2,label='Deformable A*',alpha=0.9,color='tab:green')
    plt.scatter([start[0]],[start[1]],s=40,marker='s',color='k',label='Start')
    plt.scatter([goal[0]],[goal[1]],s=60,marker='*',color='crimson',label='Goal')
    plt.axis('equal'); plt.legend(frameon=False, loc='best'); plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir,'overlay.png'), dpi=200); plt.close()

def plot_ratio_curves(outdir, ratios, ours_succ, rigid_feas, deform_feas, ours_spl, ours_detour, ours_minclr):
    ratios = np.asarray(ratios, float)
    def _plot(y, name, ylabel):
        plt.figure(figsize=(6,4))
        plt.plot(ratios, y, marker='o')
        plt.xlabel('Bottleneck ratio  ρ = w / r_rest'); plt.ylabel(ylabel); plt.grid(True,alpha=0.3)
        plt.title(name); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{name.replace(' ','_').lower()}.png"), dpi=200)
        plt.close()
    # Success / feasibility
    plt.figure(figsize=(6.2,4.2))
    plt.plot(ratios, ours_succ, 'o-', label='GRL-SNAM success')
    plt.plot(ratios, rigid_feas, 's--', label='Rigid A* feasibility')
    plt.plot(ratios, deform_feas, 'd-.', label='Deformable A* feasibility')
    plt.ylim(0,1.05); plt.xlabel('ρ = w / r_rest'); plt.ylabel('Rate'); plt.grid(True,alpha=0.3)
    plt.title('Success / Feasibility vs ρ'); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(outdir,'success_feas_vs_ratio.png'), dpi=200); plt.close()

    _plot(ours_spl, 'SPL vs ρ (GRL-SNAM)', 'SPL (vs reference)')
    _plot(ours_detour, 'Detour vs ρ (GRL-SNAM)', 'Detour (Lexec / Lref)')
    _plot(ours_minclr, 'Min-Clearance vs ρ (GRL-SNAM)', 'Min clearance (m)')

# ===================== Core evaluation per episode =====================

def rollout_episode(model, cfg_local, world, alpha_mode, k_rad, device):
    """Single rollout using CoefEnergyNet in the provided world/start/goal."""
    planner = gen.planner_from_cfg(cfg_local, world, cfg_local.k_bulk, cfg_local.gamma_s, cfg_local.d_hat, cfg_local.radius)
    path=[]; min_clear=np.inf; collisions=0; success=False

    for t in range(cfg_local.total_steps):
        sys = planner.sys
        o_w = sys.o.detach().cpu().numpy()
        Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)
        obs_feats, goal_feats = build_local_feats(o_w, cfg_local.goal, Cw, Rw, Ww)
        obs_mask = (torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
                    if obs_feats.shape[1] else torch.zeros(1,0,dtype=torch.bool, device=device))
        with torch.no_grad():
            alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))
        al_np = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
        W_adj, R_adj = Ww.copy(), Rw.copy()
        if alpha_mode!='none':
            W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, alpha_mode, k_rad)
        world_step = ssi.WorldObstacles(Cw, R_adj, W_adj, d_hat=cfg_local.d_hat)
        planner.stage_field.w_goal = max(0.0, float(beta.squeeze(0).item()))
        planner.sys.gamma_o        = max(0.0, float(gamma.squeeze(0).item()))
        info = planner.step(cfg_local.dt, world_step)
        center_xy = np.asarray(info["center"], float)
        path.append(center_xy)
        d_clear = clearance_to_circles(center_xy, world.C_np, world.R_np)
        min_clear = min(min_clear, d_clear)
        if d_clear < 0.0: collisions += 1
        if np.linalg.norm(center_xy - cfg_local.goal) <= cfg_local.goal_tol:
            success=True; break

    L_exec = path_length(path)
    reached = success_reached(path[-1] if path else cfg_local.start, cfg_local.goal, cfg_local.goal_tol)
    return dict(path=path, L_exec=L_exec, min_clear=min_clear, collisions=collisions, success=reached, steps=len(path))

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser("E4: Deformability advantage (narrow-passage)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","none"])
    ap.add_argument("--k_rad", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.6,0.7,0.8,0.9,1.0,1.1,1.2])
    ap.add_argument("--envs_per_ratio", type=int, default=3)
    ap.add_argument("--trials_per_env", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--gif_examples", type=int, default=3)
    ap.add_argument("--scenario", type=str, default="all", choices=["hourglass","keyhole","serpentine","all"])
    ap.add_argument("--pillar_scale", type=float, default=1.4,
                    help="Scale factor for the two center pillars (multiplies r_rest).")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    # Load model
    model = CoefEnergyNet().to(device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state); model.eval()

    # Base cfg defaults
    base_cfg = gen.GenCfg()
    base_cfg.d_hat    = getattr(base_cfg, "d_hat", 0.5)
    base_cfg.radius   = getattr(base_cfg, "radius", base_cfg.d_hat)  # r_rest
    base_cfg.radius_min = getattr(base_cfg, "radius_min", 0.3*base_cfg.radius)
    base_cfg.k_bulk   = getattr(base_cfg, "k_bulk", 20.0)
    base_cfg.gamma_s  = getattr(base_cfg, "gamma_s", 0.05)
    base_cfg.dt       = getattr(base_cfg, "dt", 0.01)

    r_rest = base_cfg.radius
    r_min  = base_cfg.radius_min

    # Output dirs
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = mkdir(f"snaps_deformability/{stamp}")
    per_episode: List[Dict] = []
    agg_per_ratio: Dict[float, Dict] = {}

    # Sweep ratios
    example_count = 0
    example_dir = mkdir(os.path.join(run_dir,"examples"))

    for rho in args.ratios:
        ratio_rows=[]
        succ_list=[]; spl_list=[]; detour_list=[]; minclr_list=[]
        rigid_feas_list=[]; deform_feas_list=[]

        # --- helper: choose slit offsets for each scenario
        def _slit_offsets_for(scenario: str):
            if scenario == "hourglass":
                return (0.0,)
            elif scenario == "keyhole":
                return (+0.35,)
            elif scenario == "serpentine":
                return (+0.35, -0.35)
            else:
                return (0.0,)

        # --- helper: quick sanity check that straight line hits the bottleneck region
        def _segment_intersects_bottleneck(start_xy, goal_xy, cL, cR, w_band=0.4):
            """
            True if the centerline segment crosses the vertical strip between pillar centers
            and passes within |y|<=w_band of the slit center at x ~ (cL.x+cR.x)/2.
            """
            s = np.asarray(start_xy, float); g = np.asarray(goal_xy, float)
            x_mid = 0.5*(cL[0] + cR[0])
            # param t where segment hits x_mid
            if g[0] == s[0]:
                return False
            t = (x_mid - s[0]) / (g[0] - s[0])
            if t < 0.0 or t > 1.0:
                return False
            y_hit = s[1] + t*(g[1] - s[1])
            return (abs(y_hit) <= w_band)

        # ----------------- main loop per ratio -----------------
        for env_id in range(args.envs_per_ratio):
            seed_env = int(rng.integers(0, 10_000_000))
            C,R,W,bounds,start_xy,goal_xy, rep, cL, cR = make_corridor_env(
                r_rest, rho, r_min=r_min,
                length=getattr(args, "length", 14.0),
                y_half=getattr(args, "y_half", 2.6),
                slit_pillar_scale=args.pillar_scale,
                seed=seed_env
            )
            world = ssi.WorldObstacles(C, R, W, d_hat=base_cfg.d_hat)

            # --- grid: make the bottleneck resolvable
            # coarse 0.5*r_rest was too big; use fine res tied to r_min
            res = max(0.05 * r_rest, 0.25 * r_min)

            occ_rigid,_,_  = rasterize_inflated_obstacles(world.C_np, world.R_np, r_rest, bounds, res)
            clearance_grid  = clearance_field_on_grid(world.C_np, world.R_np, bounds, res)

            for trial_id in range(args.trials_per_env):
                # A* baselines
                L_rigid,  path_rigid   = a_star_path(occ_rigid, bounds, res, start_xy, goal_xy)
                rigid_feasible         = np.isfinite(L_rigid)
                rigid_feas_list.append(1.0 if rigid_feasible else 0.0)

                L_deform, path_deform  = deformable_a_star_path(
                    clearance_grid, bounds, res, start_xy, goal_xy,
                    r_rest=r_rest, r_min=r_min, lam=5.0, beta=0.6, eps=1e-3
                )
                deform_feasible        = np.isfinite(L_deform)
                deform_feas_list.append(1.0 if deform_feasible else 0.0)

                # Regime tag
                if deform_feasible and not rigid_feasible:
                    regime = "deform_only"
                elif deform_feasible and rigid_feasible:
                    regime = "both_feasible"
                elif (not deform_feasible) and (not rigid_feasible):
                    regime = "both_infeasible"
                else:
                    regime = "rigid_only"

                # Episode cfg (unchanged)
                cfg_local = gen.GenCfg()
                cfg_local.d_hat       = base_cfg.d_hat
                cfg_local.radius      = r_rest
                cfg_local.k_bulk      = base_cfg.k_bulk
                cfg_local.gamma_s     = base_cfg.gamma_s
                cfg_local.dt          = base_cfg.dt
                cfg_local.total_steps = args.steps
                cfg_local.start       = start_xy.astype(np.float32)
                cfg_local.goal        = goal_xy.astype(np.float32)

                out = rollout_episode(model, cfg_local, world, args.alpha_mode, args.k_rad, device)

                # Reference for SPL/Detour
                L_ref = L_deform if deform_feasible else (L_rigid if rigid_feasible else None)
                if L_ref and L_ref > 0:
                    detour = out["L_exec"]/L_ref
                    spl    = (L_ref / max(L_ref, out["L_exec"])) if out["success"] else 0.0
                else:
                    detour, spl = np.nan, 0.0

                row = {
                    "ratio": float(rho),
                    "env_id": int(env_id),
                    "trial_id": int(trial_id),
                    "success": bool(out["success"]),
                    "path_length": float(out["L_exec"]),
                    "min_clearance": float(out["min_clear"]),
                    "collisions": int(out["collisions"]),
                    "SPL_vs_ref": float(spl),
                    "detour_vs_ref": float(detour) if np.isfinite(detour) else float("nan"),
                    "astar_rigid_feasible": bool(rigid_feasible),
                    "astar_deform_feasible": bool(deform_feasible),
                    "astar_rigid_L": float(L_rigid) if rigid_feasible else float("inf"),
                    "astar_deform_L": float(L_deform) if deform_feasible else float("inf"),
                    "steps": int(out["steps"]),
                    "bounds": list(bounds),
                    "regime": regime,
                    # for quick debugging in logs:
                    "gap": rep["gap"],
                    "theory_rigid_feasible": rep["theory_rigid_feasible"],
                    "theory_deform_feasible": rep["theory_deform_feasible"]
                }
                per_episode.append(row)
                ratio_rows.append(row)

                succ_list.append(1.0 if out["success"] else 0.0)
                spl_list.append(spl)
                detour_list.append(detour if np.isfinite(detour) else np.nan)
                minclr_list.append(out["min_clear"])

                if example_count < args.gif_examples:
                    ep_dir = mkdir(os.path.join(example_dir, f"rho_{rho:.2f}_env{env_id}_trial{trial_id}"))
                    plot_overlay_example(
                        ep_dir, world, out["path"],
                        path_rigid if rigid_feasible else None,
                        path_deform if deform_feasible else None,
                        start_xy, goal_xy,
                        title=f"ρ={rho:.2f} · {regime}"
                    )
                    example_count += 1

        # --- Aggregate per ratio (same as before) ---
        def _nanmean(x):
            arr = np.asarray(x, float)
            return float(np.nanmean(arr)) if arr.size else float('nan')

        agg_per_ratio[rho] = {
            "success_rate_mean": _nanmean(succ_list),
            "spl_mean": _nanmean(spl_list),
            "detour_mean": _nanmean(detour_list),
            "min_clear_mean": _nanmean(minclr_list),
            "rigid_feas_rate": _nanmean(rigid_feas_list),
            "deform_feas_rate": _nanmean(deform_feas_list),
            "episodes": len(ratio_rows)
        }

    # Write artifacts
    with open(os.path.join(run_dir, "episodes.json"), "w") as f:
        json.dump(per_episode, f, indent=2)

    # CSV
    csv_fields = ["ratio","env_id","trial_id","success","SPL_vs_ref","detour_vs_ref",
                  "path_length","min_clearance","collisions","steps",
                  "astar_rigid_feasible","astar_deform_feasible",
                  "astar_rigid_L","astar_deform_L"]
    with open(os.path.join(run_dir,"episodes.csv"),"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=csv_fields); w.writeheader()
        for r in per_episode: w.writerow({k:r.get(k,"") for k in csv_fields})

    with open(os.path.join(run_dir, "aggregate_by_ratio.json"), "w") as f:
        json.dump(agg_per_ratio, f, indent=2)

    # Plots vs ratio
    ratios_sorted = sorted(agg_per_ratio.keys())
    ours_succ   = [agg_per_ratio[r]["success_rate_mean"] for r in ratios_sorted]
    ours_spl    = [agg_per_ratio[r]["spl_mean"] for r in ratios_sorted]
    ours_detour = [agg_per_ratio[r]["detour_mean"] for r in ratios_sorted]
    ours_minclr = [agg_per_ratio[r]["min_clear_mean"] for r in ratios_sorted]
    rigid_feas  = [agg_per_ratio[r]["rigid_feas_rate"] for r in ratios_sorted]
    deform_feas = [agg_per_ratio[r]["deform_feas_rate"] for r in ratios_sorted]

    plot_ratio_curves(run_dir, ratios_sorted, ours_succ, rigid_feas, deform_feas,
                      ours_spl, ours_detour, ours_minclr)

    # Print quick summary
    print("\n=== Deformability Stress Test Summary ===")
    for r in ratios_sorted:
        a = agg_per_ratio[r]
        print(f"ρ={r:.2f} | ours succ={a['success_rate_mean']:.2f} | "
              f"rigid feas={a['rigid_feas_rate']:.2f} | deform feas={a['deform_feas_rate']:.2f} | "
              f"SPL={a['spl_mean']:.3f} | detour={a['detour_mean']:.3f} | minclr={a['min_clear_mean']:.3f}")
    print(f"\nArtifacts written to: {run_dir}")

if __name__ == "__main__":
    main()
