#!/usr/bin/env python3
"""
E3. Category Baselines: CoefEnergyNet vs Rigid A*, Deformable A*, Reactive (CBF-like)

Usage examples
--------------
python eval_e3_category_baselines.py --ckpt checkpoints/best.pt --case case1-tight --n_envs 5 --n_trials 10
python eval_e3_category_baselines.py --ckpt checkpoints/best.pt --case case2-harder --alpha_mode weight
python eval_e3_category_baselines.py --ckpt checkpoints/best.pt --case case1-dense --disable_reactive
"""

import os, json, argparse, csv, math, time
from datetime import datetime
from collections import defaultdict
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt

# Project imports
from train_coef_energy import CoefEnergyNet
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi

# --------------------------- Small utils ---------------------------

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
            (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]
    def h(i,j): return math.hypot(i-gi, j-gj)
    g = np.full((H,W), np.inf); came = np.full((H,W,2), -1, dtype=np.int32)
    pq=[]; g[si,sj]=0.0; heappush(pq,(h(si,sj),(si,sj))); vis=np.zeros_like(occ, bool)
    while pq:
        _,(i,j)=heappop(pq)
        if vis[i,j]: continue
        vis[i,j]=True
        if (i,j)==(gi,gj): break
        for di,dj,c in nbrs:
            ni,nj=i+di,j+dj
            if not (0<=ni<H and 0<=nj<W): continue
            if occ[ni,nj]: continue
            t = g[i,j]+c
            if t<g[ni,nj]:
                g[ni,nj]=t; came[ni,nj]=[i,j]; heappush(pq,(t+h(ni,nj),(ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf, []
    # backtrack
    path=[]; ci,cj=gi,gj; path.append((ci,cj))
    while not (ci==si and cj==sj):
        pi,pj=came[ci,cj]
        if pi<0: break
        path.append((pi,pj)); ci,cj=pi,pj
    path.reverse()
    length_steps=0.0
    for k in range(1,len(path)):
        i0,j0=path[k-1]; i1,j1=path[k]
        length_steps += math.hypot(i1-i0, j1-j0)
    length_m = length_steps*res
    path_xy = np.stack([ij_to_world(i,j,bounds,res) for (i,j) in path],0) if path else []
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
    for (cx,cy), r in zip(C,R):
        d = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(np.float32) - float(r)
        clear = np.minimum(clear, d)
    return clear

def deformable_a_star_path(clearance, bounds, res, start_xy, goal_xy, r_rest, r_min, lam=5.0, beta=0.6, eps=1e-3):
    from heapq import heappush, heappop
    H,W = clearance.shape
    si,sj = world_to_ij(start_xy, bounds, res)
    gi,gj = world_to_ij(goal_xy, bounds, res)
    if not (0<=si<H and 0<=sj<W and 0<=gi<H and 0<=gj<W): return np.inf,[]
    feas = clearance >= r_min
    if not (feas[si,sj] and feas[gi,gj]): return np.inf,[]
    nbrs=[(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
          (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]
    def h(i,j): return math.hypot(i-gi, j-gj)*res
    def pen(c):
        a = max(0.0, (r_rest/(c+eps))-1.0)
        return lam*(a*a)
    g=np.full((H,W), np.inf); came=np.full((H,W,2), -1, np.int32)
    pq=[]; g[si,sj]=0.0; heappush(pq,(h(si,sj),(si,sj))); vis=np.zeros((H,W), bool)
    while pq:
        _,(i,j)=heappop(pq)
        if vis[i,j]: continue
        vis[i,j]=True
        if (i,j)==(gi,gj): break
        if not feas[i,j]: continue
        pij=pen(clearance[i,j])
        for di,dj,cstep in nbrs:
            ni,nj=i+di,j+dj
            if not (0<=ni<H and 0<=nj<W): continue
            if not feas[ni,nj]: continue
            base=cstep*res; deform=beta*0.5*(pij+pen(clearance[ni,nj]))*base
            t=g[i,j]+base+deform
            if t<g[ni,nj]:
                g[ni,nj]=t; came[ni,nj]=[i,j]; heappush(pq,(t+h(ni,nj),(ni,nj)))
    if not np.isfinite(g[gi,gj]): return np.inf,[]
    # backtrack and metric length
    path=[]; ci,cj=gi,gj; path.append((ci,cj))
    while not (ci==si and cj==sj):
        pi,pj=came[ci,cj]
        if pi<0: break
        path.append((pi,pj)); ci,cj=pi,pj
    path.reverse()
    L=0.0
    for k in range(1,len(path)):
        i0,j0=path[k-1]; i1,j1=path[k]
        L+= math.hypot(i1-i0, j1-j0)*res
    path_xy=np.stack([ij_to_world(i,j,bounds,res) for (i,j) in path],0) if path else []
    return float(L), path_xy

def path_length(path_xy):
    if len(path_xy)<2: return 0.0
    d=np.diff(np.asarray(path_xy),axis=0)
    return float(np.sum(np.linalg.norm(d,axis=1)))

def clearance_to_circles(xy, C, R):
    if len(C)==0: return np.inf
    d = np.linalg.norm(C-xy[None,:], axis=1) - R
    return float(np.min(d))

def turning_cost(path_xy):
    P=np.asarray(path_xy)
    if len(P)<3: return 0.0
    v=P[1:]-P[:-1]
    ang=np.arctan2(v[:,1],v[:,0])
    d=np.diff(ang); d=(d+np.pi)%(2*np.pi)-np.pi
    return float(np.mean(np.abs(d)))

def success_reached(last_center, goal_xy, tol):
    return np.linalg.norm(np.asarray(last_center)-np.asarray(goal_xy)) <= tol

def compute_safety_stats(path_xy, C, R, barrier_thresh, tube_thresh):
    T=len(path_xy)
    if T==0:
        return dict(min_clearance=np.inf, barrier_rate=0.0, tube_rate=0.0, barrier_penalty_sum=0.0)
    clears=[clearance_to_circles(np.asarray(xy,float),C,R) for xy in path_xy]
    c=np.asarray(clears,float)
    barrier=(c<barrier_thresh).mean()
    tube=(c<tube_thresh).mean()
    pen=np.sum(np.maximum(0.0, barrier_thresh-c)**2)
    return dict(min_clearance=float(np.min(c)), barrier_rate=float(barrier),
                tube_rate=float(tube), barrier_penalty_sum=float(pen))

# --------------------------- CoefEnergyNet helpers ---------------------------

def build_local_feats(o_w, goal_w, C_w, R_w, W_w):
    o=torch.as_tensor(o_w, dtype=torch.float); g=torch.as_tensor(goal_w, dtype=torch.float)
    C=torch.as_tensor(C_w, dtype=torch.float) if C_w.size else torch.zeros(0,2)
    R=torch.as_tensor(R_w, dtype=torch.float) if R_w.size else torch.zeros(0)
    W=torch.as_tensor(W_w, dtype=torch.float) if W_w.size else torch.zeros(0)
    if C.ndim==1: C=C.reshape(0,2)
    dg=g-o; gdist=torch.linalg.norm(dg).unsqueeze(0)
    goal_feats=torch.stack([dg[0],dg[1],gdist[0],torch.tensor(1.0)],0).unsqueeze(0)
    if C.shape[0]==0: obs_feats=torch.zeros(1,0,6)
    else:
        dxdy=(g.unsqueeze(0)-C); obs_feats=torch.cat([C,R.unsqueeze(-1),W.unsqueeze(-1),dxdy],-1).unsqueeze(0)
    return obs_feats, goal_feats

def map_alpha_to_world(W_in, R_in, alphas, mode, k_rad=0.05):
    al = np.maximum(alphas, 0.0)
    W_out=W_in.copy(); R_out=R_in.copy()
    if mode in ("weight","both") and al.size: W_out = W_out * al
    if mode in ("radius","both") and al.size: R_out = R_out + k_rad*al
    return W_out, R_out

# --------------------------- Reactive baseline (CBF-ish) ---------------------------

def reactive_velocity(o_xy: np.ndarray, goal: np.ndarray, C: np.ndarray, R: np.ndarray,
                      v_goal=1.0, k_att=1.0, k_rep=1.5, d0=1.5, eps=1e-6):
    """
    Potential-field style: v = k_att * (goal - o)/||.||  +  Σ k_rep * (1/d - 1/d0)_+ * (o - c)/d
    Only obstacles with d < d0 contribute. Clipped to v_goal.
    """
    g = goal - o_xy
    gnorm = np.linalg.norm(g)+eps
    v = k_att * g/gnorm
    if C.size:
        d = np.linalg.norm(C - o_xy[None,:], axis=1) - R
        mask = d < d0
        if np.any(mask):
            Cc = C[mask]; dd = np.clip(d[mask], eps, None)
            dir_away = (o_xy[None,:] - Cc) / (np.linalg.norm(o_xy[None,:]-Cc, axis=1, keepdims=True)+eps)
            rep = np.maximum(0.0, (1.0/dd) - (1.0/d0))[:,None] * dir_away
            v += k_rep * np.sum(rep, axis=0)
    n=np.linalg.norm(v)+eps
    v = v_goal * v / n
    return v.astype(np.float32)

# --------------------------- Plot overlays ---------------------------

def plot_paths_overlay(outdir, world, our_path, path_rigid, path_deform, path_react, start, goal):
    plt.figure(figsize=(5.8,5.6))
    th=np.linspace(0,2*np.pi,128)
    for (cx,cy),r in zip(world.C_np, world.R_np):
        plt.plot(cx+r*np.cos(th), cy+r*np.sin(th), lw=1.2, alpha=0.8, color='k')
    if path_rigid is not None and len(path_rigid):
        plt.plot(path_rigid[:,0], path_rigid[:,1], '--', lw=2, label='Rigid A*', color='tab:blue', alpha=0.9)
    else:
        plt.text(start[0], start[1], "No Rigid A*", color='tab:blue', fontsize=8)
    if path_deform is not None and len(path_deform):
        plt.plot(path_deform[:,0], path_deform[:,1], '-.', lw=2, label='Deformable A*', color='tab:green', alpha=0.9)
    else:
        plt.text(start[0], start[1]-0.5, "No Deform A*", color='tab:green', fontsize=8)
    if path_react is not None and len(path_react):
        P=np.asarray(path_react); plt.plot(P[:,0], P[:,1], ':', lw=2.4, label='Reactive', color='tab:purple', alpha=0.9)
    if our_path:
        P=np.asarray(our_path); plt.plot(P[:,0], P[:,1], '-', lw=2.4, label='CoefEnergyNet', color='tab:orange', alpha=0.95)
    plt.scatter([start[0]],[start[1]], s=50, marker='s', color='k', label='Start')
    plt.scatter([goal[0]],[goal[1]], s=70, marker='*', color='crimson', label='Goal')
    plt.axis('equal'); plt.legend(frameon=False, loc='best'); plt.title("Trajectories: ours vs baselines")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"trajectory_overlay.png"), dpi=200); plt.close()

# --------------------------- Main eval ---------------------------

def main():
    ap = argparse.ArgumentParser("E3 Category Baselines")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","none"])
    ap.add_argument("--case", type=str, default="case1-tight")
    ap.add_argument("--seed", type=int, default=2312)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--k_rad", type=float, default=0.05)
    ap.add_argument("--n_envs", type=int, default=3)
    ap.add_argument("--n_trials", type=int, default=3)
    ap.add_argument("--gif_stride", type=int, default=30)
    ap.add_argument("--disable_reactive", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = CoefEnergyNet().to(device)
    if not os.path.exists(args.ckpt):
        print(f"[ERROR] Missing checkpoint: {args.ckpt}"); return
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state); model.eval()

    # Run directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = mkdir(f"snaps_e3/{stamp}")
    per_episode: List[Dict[str, Any]] = []

    rng = np.random.default_rng(args.seed)
    print(f"[E3] case={args.case} envs={args.n_envs} trials/env={args.n_trials}")

    for env_id in range(args.n_envs):
        # Build env
        cfg = gen.GenCfg(); cfg.seed = int(rng.integers(0,10_000_000)); gen.set_all_seeds(cfg.seed)
        if args.case.startswith("case2"):
            C,R,W = gen.sample_obstacles_case2_harder(cfg)
        else:
            C,R,W = gen.sample_obstacles_case1_tight(cfg)
        world = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)

        # Baseline params
        r_rest = getattr(cfg,"radius", cfg.d_hat)
        r_min  = getattr(cfg,"radius_min", 0.3*r_rest)
        bounds = compute_workspace_bounds(world.C_np, world.R_np, margin=2.0*max(r_rest,cfg.d_hat))
        res    = max(r_rest, cfg.d_hat) * 0.5

        # Precompute grids for A*
        occ_rigid, _, _ = rasterize_inflated_obstacles(world.C_np, world.R_np, r_rest, bounds, res)
        clearance_grid = clearance_field_on_grid(world.C_np, world.R_np, bounds, res)

        for trial_id in range(args.n_trials):
            ep_dir = mkdir(os.path.join(run_dir, f"env{env_id:03d}_trial{trial_id:03d}"))

            # sample a reasonable start/goal (reject inside obstacles inflated by r_min)
            def sample_valid(margin=r_min, tries=500):
                for _ in range(tries):
                    x=rng.uniform(bounds[0], bounds[1]); y=rng.uniform(bounds[2], bounds[3])
                    p=np.array([x,y], np.float32)
                    d=np.linalg.norm(world.C_np - p[None,:], axis=1) - (world.R_np + margin) if len(world.C_np) else np.array([np.inf])
                    if np.all(d>0.0): return p
                return np.array([bounds[0]+1.0, bounds[2]+1.0], np.float32)
            start = sample_valid(); goal = sample_valid()
            while np.linalg.norm(start-goal) < 2.0*r_rest:
                goal = sample_valid()

            # ---------- Baselines (timed) ----------
            t0=time.perf_counter()
            L_rigid, path_rigid = a_star_path(occ_rigid, bounds, res, start, goal)
            t_rigid = (time.perf_counter()-t0)*1000.0

            t0=time.perf_counter()
            L_deform, path_deform = deformable_a_star_path(
                clearance_grid, bounds, res, start, goal, r_rest=r_rest, r_min=r_min, lam=5.0, beta=0.6
            )
            t_deform = (time.perf_counter()-t0)*1000.0

            # ---------- Reactive rollout (optional) ----------
            react_path=[]
            t_react=0.0
            if not args.disable_reactive:
                cfg_re = gen.GenCfg(); cfg_re.__dict__.update(cfg.__dict__)
                cfg_re.start = start.astype(np.float32); cfg_re.goal = goal.astype(np.float32)
                planner_re = gen.planner_from_cfg(cfg_re, world, getattr(cfg,"k_bulk",20.0),
                                                  getattr(cfg,"gamma_s",0.05), cfg.d_hat, getattr(cfg,"radius",cfg.d_hat))
                t0=time.perf_counter()
                for _ in range(args.steps):
                    o = planner_re.sys.o.detach().cpu().numpy().astype(np.float32)
                    v = reactive_velocity(o, cfg_re.goal, world.C_np, world.R_np, v_goal=1.0, k_att=1.2, k_rep=1.8, d0=max(1.2*r_rest, 0.8))
                    planner_re.sys.v_o = torch.tensor(v, dtype=planner_re.sys.dtype)
                    info = planner_re.step(cfg_re.dt, world)
                    cxy = np.asarray(info["center"], float); react_path.append(cxy)
                    if np.linalg.norm(cxy - cfg_re.goal) <= cfg_re.goal_tol: break
                t_react = (time.perf_counter()-t0)*1000.0
            # ---------- Our policy rollout (timed) ----------
            cfg_local = gen.GenCfg(); cfg_local.__dict__.update(cfg.__dict__)
            cfg_local.start = start.astype(np.float32); cfg_local.goal = goal.astype(np.float32)
            planner = gen.planner_from_cfg(cfg_local, world, getattr(cfg,"k_bulk",20.0),
                                           getattr(cfg,"gamma_s",0.05), cfg.d_hat, getattr(cfg,"radius",cfg.d_hat))

            frames=[]; path=[]; min_clear=np.inf; collisions=0; success=False
            def capture(sys): 
                return {"center": sys.o.detach().cpu().to(torch.float32),
                        "theta": (sys.theta.detach().cpu().to(torch.float32) if hasattr(sys,"theta") else torch.tensor(0.0))}
            frames.append(capture(planner.sys))

            t0=time.perf_counter()
            for t in range(args.steps):
                sys=planner.sys; o_w=sys.o.detach().cpu().numpy()
                Cw,Rw,Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)
                obs,goal_feats = build_local_feats(o_w, cfg_local.goal, Cw, Rw, Ww)
                omask = (torch.ones(1,obs.shape[1], dtype=torch.bool, device=device)
                         if obs.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device))
                with torch.no_grad():
                    alphas,beta,gamma = model(obs.to(device), omask, goal_feats.to(device))
                al_np = alphas.squeeze(0).detach().cpu().numpy() if obs.shape[1] else np.zeros_like(Rw)
                beta_f=float(beta.squeeze(0).item()); gamma_f=float(gamma.squeeze(0).item())
                W_adj,R_adj = Ww.copy(),Rw.copy()
                if args.alpha_mode!="none":
                    W_adj,R_adj = map_alpha_to_world(Ww,Rw,al_np,args.alpha_mode,args.k_rad)
                world_step = ssi.WorldObstacles(Cw,R_adj,W_adj,d_hat=cfg_local.d_hat)
                if hasattr(planner,"stage_field"): planner.stage_field.w_goal = max(0.0, beta_f)
                if hasattr(planner.sys,"gamma_o"): planner.sys.gamma_o = max(0.0, gamma_f)
                info = planner.step(cfg_local.dt, world_step)
                cxy = np.asarray(info["center"], float); path.append(cxy)
                d_clear = clearance_to_circles(cxy, world.C_np, world.R_np)
                min_clear = min(min_clear, d_clear); collisions += int(d_clear<0.0)
                if np.linalg.norm(cxy - cfg_local.goal) <= cfg_local.goal_tol:
                    success=True; break
            t_ours = (time.perf_counter()-t0)*1000.0

            # ---------- Metrics ----------
            # SPL/Detour vs best available A* (prefer deformable)
            L_ref = L_deform if np.isfinite(L_deform) else (L_rigid if np.isfinite(L_rigid) else None)
            L_exec = path_length(path)
            reached = success_reached(path[-1] if path else start, goal, cfg_local.goal_tol)
            detour = (L_exec/L_ref) if (L_ref is not None and L_ref>0) else np.nan
            spl    = (L_ref/max(L_ref,L_exec)) if (reached and L_ref is not None and L_ref>0) else 0.0
            smooth = turning_cost(path)
            safety = compute_safety_stats(path, world.C_np, world.R_np, barrier_thresh=cfg.d_hat, tube_thresh=r_min)

            # Reactive metrics (length only; no A* dependency)
            L_react = path_length(react_path) if react_path else np.inf

            # Save overlay
            plot_paths_overlay(ep_dir, world, path, path_rigid, path_deform, react_path, start, goal)

            ep = dict(
                case=args.case, env_id=env_id, trial_id=trial_id,
                start=start.tolist(), goal=goal.tolist(),
                n_obstacles=int(len(world.C_np)),
                ours_success=bool(reached), ours_len=float(L_exec), ours_SPL=float(spl),
                ours_detour=float(detour) if np.isfinite(detour) else float("nan"),
                ours_min_clear=float(safety["min_clearance"]) if np.isfinite(safety["min_clearance"]) else float("nan"),
                ours_barrier_rate=float(safety["barrier_rate"]), ours_tube_rate=float(safety["tube_rate"]),
                ours_smooth=float(smooth), ours_collisions=int(collisions),
                astar_rigid_len=float(L_rigid) if np.isfinite(L_rigid) else float("inf"),
                astar_deform_len=float(L_deform) if np.isfinite(L_deform) else float("inf"),
                reactive_len=float(L_react) if np.isfinite(L_react) else float("inf"),
                t_ms_ours=float(t_ours), t_ms_rigid=float(t_rigid), t_ms_deform=float(t_deform),
                t_ms_reactive=float(t_react)
            )
            per_episode.append(ep)

    # --------------------------- Save + Plots ---------------------------

    # Write JSON/CSV
    with open(os.path.join(run_dir,"episodes.json"),"w") as f: json.dump(per_episode,f,indent=2)
    fields = list(per_episode[0].keys()) if per_episode else []
    if fields:
        with open(os.path.join(run_dir,"episodes.csv"),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow(r) for r in per_episode]

    # Aggregates
    def arr(key, mask=None):
        xs=[r[key] for r in per_episode if key in r]
        return np.asarray(xs, float) if not mask else np.asarray([r[key] for r in per_episode if mask(r)], float)

    summary = dict(
        n_episodes=len(per_episode),
        success_rate=float(np.mean(arr("ours_success"))) if per_episode else 0.0,
        mean_SPL=float(np.nanmean(arr("ours_SPL"))) if per_episode else float("nan"),
        mean_detour=float(np.nanmean(arr("ours_detour"))) if per_episode else float("nan"),
        mean_min_clear=float(np.nanmean(arr("ours_min_clear"))) if per_episode else float("nan"),
        mean_barrier_rate=float(np.nanmean(arr("ours_barrier_rate"))) if per_episode else float("nan"),
        mean_tube_rate=float(np.nanmean(arr("ours_tube_rate"))) if per_episode else float("nan"),
        mean_len_ours=float(np.nanmean(arr("ours_len"))) if per_episode else float("nan"),
        mean_len_rigid=float(np.nanmean(arr("astar_rigid_len"))) if per_episode else float("nan"),
        mean_len_deform=float(np.nanmean(arr("astar_deform_len"))) if per_episode else float("nan"),
        mean_len_reactive=float(np.nanmean(arr("reactive_len"))) if per_episode else float("nan"),
        mean_t_ms_ours=float(np.nanmean(arr("t_ms_ours"))) if per_episode else float("nan"),
        mean_t_ms_rigid=float(np.nanmean(arr("t_ms_rigid"))) if per_episode else float("nan"),
        mean_t_ms_deform=float(np.nanmean(arr("t_ms_deform"))) if per_episode else float("nan"),
        mean_t_ms_reactive=float(np.nanmean(arr("t_ms_reactive"))) if per_episode else float("nan"),
    )
    with open(os.path.join(run_dir,"summary.json"),"w") as f: json.dump(summary,f,indent=2)

    # -------- Plots (paper figures) --------
    def bar_with_err(ax, names, vals):
        ax.bar(names,[np.nanmean(v) for v in vals], yerr=[np.nanstd(v) for v in vals],
               capsize=4, alpha=0.8); ax.grid(True, alpha=0.25)

    # 1) Success & SPL (ours only, category uses A* as reference)
    plt.figure(figsize=(6,4))
    plt.bar(["Success"], [summary["success_rate"]], color="tab:orange"); plt.ylim(0,1); plt.grid(True, alpha=0.3)
    plt.ylabel("Rate"); plt.title("CoefEnergyNet Success"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir,"fig_success.png"), dpi=200); plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(["SPL_vs_ref"], [summary["mean_SPL"]], color="tab:orange"); plt.ylim(0,1); plt.grid(True, alpha=0.3)
    plt.ylabel("SPL"); plt.title("CoefEnergyNet SPL vs A* reference"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir,"fig_spl.png"), dpi=200); plt.close()

    # 2) Path length comparison (ours vs baselines)
    plt.figure(figsize=(6.8,4.2))
    data = [arr("ours_len"), arr("astar_rigid_len"), arr("astar_deform_len"), arr("reactive_len")]
    plt.boxplot(data, labels=["Ours","Rigid*","Deform*","Reactive"])
    plt.ylabel("Path length (m)"); plt.title("Length distribution across methods")
    plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir,"fig_length_box.png"), dpi=200); plt.close()

    # 3) Timing comparison
    plt.figure(figsize=(6.8,4.2))
    names=["Ours","Rigid*","Deform*","Reactive"]
    means=[summary["mean_t_ms_ours"], summary["mean_t_ms_rigid"], summary["mean_t_ms_deform"], summary["mean_t_ms_reactive"]]
    plt.bar(names, means); plt.ylabel("Time (ms/episode)"); plt.title("Computation (lower is better)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(run_dir,"fig_timing.png"), dpi=200); plt.close()

    # 4) Safety vs Performance (Pareto)
    valid = [(r["ours_min_clear"], r["ours_SPL"]) for r in per_episode if np.isfinite(r["ours_min_clear"])]
    if valid:
        x,y = zip(*valid)
        plt.figure(figsize=(5.4,4.6))
        plt.scatter(x,y, s=30, c='tab:orange', alpha=0.7)
        plt.xlabel("Min clearance (m)"); plt.ylabel("SPL"); plt.title("SPL vs Min Clearance (Ours)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(run_dir,"fig_pareto_spl_clear.png"), dpi=200); plt.close()

    # 5) Detour histogram (ours vs ref)
    det = arr("ours_detour")
    plt.figure(figsize=(6.4,4.2))
    plt.hist(det[np.isfinite(det)], bins=20, edgecolor='k', alpha=0.75, color='tab:orange')
    plt.axvline(1.0, color='r', linestyle='--', label='Optimal'); plt.legend()
    plt.xlabel("Detour (L_exec / L_ref)"); plt.ylabel("Count"); plt.title("Detour distribution"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir,"fig_detour_hist.png"), dpi=200); plt.close()

    print("\n[E3] Summary")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts → {run_dir}")

if __name__ == "__main__":
    main()
