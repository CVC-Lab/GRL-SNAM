#!/usr/bin/env python3
# Dataset generator that reuses the overlap-constrained stage logic from spline_stagewise6.

import sys
import os, json, math, random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.spline_stagewise6 as she 


# ==============================
# Config
# ==============================

class GenCfg:
    out_root: str = "./nav_dataset_overlap_hyperring"
    seed: int = 2025

    # simulation
    total_steps: int = 2000
    dt: float = 0.03
    snapshot_every: int = 5

    # start/goal and stage geometry
    start = np.array([-1.0, -0.5], float)
    goal  = np.array([ 9.0,  0.2], float)

    start, goal = goal, start

    stage_size = (2.6, 2.0)
    overlap = 0.30

    # tube radius (robot radius + margin); used for:
    #  - inflating obstacles at planning + runtime
    #  - barrier d_hat
    inflate = 0.05

    # hyperelastic robot base params (perturbed for Case 2)
    n_ctrl: int = 20
    K: int = 240
    d_hat: float = 1.0
    seed_robot: int = 7
    radius: float = 0.35  # base ring radius at s=1

    # Hyperelastic parameters
    k_bulk: float = 1.5
    gamma_s: float = 2.0
    M_s: float = 1.0

    # Rigid body params
    M_o: float = 1.5
    gamma_o: float = 4.0
    I: float = 0.6
    gamma_th: float = 1.6

    # success thresholds
    goal_tol: float = 0.30
    min_clear_tol: float = 1e-6

    # dataset sizes
    num_env_variations: int = 10
    num_robot_variations: int = 10

    # *** Case 1 (tight corridor) obstacle generation ***
    env_x_bounds = (-0.5, 9.0)
    env_y_bounds = (-1.8, 1.8)
    radius_range = (0.25, 0.80)
    weight_range = (0.6, 1.5)
    n_obs_range = (12, 22)
    poisson_max_tries: int = 6000
    min_separation_slack: float = 0.08
    min_start_clear: float = 0.9
    min_goal_clear: float = 0.9

    gap_mult_range = (0.70, 0.80)
    gap_margin_range = (0.01, 0.03)
    corridor_wiggle_amp = 0.35
    corridor_wiggle_k = 0.6
    n_gates_range = (3, 5)
    n_intruders_range = (2, 4)

    # *** Case 2 (robot perturbations) ***
    k_bulk_mult_range    = (0.90, 1.15)
    gamma_s_mult_range   = (0.85, 1.15)
    d_hat_mult_range     = (0.95, 1.05)
    radius_abs_range     = (0.30, 1.00)

    # viz
    overlay_inflated_in_snapshot: bool = True


# ==============================
# Utility / RNG
# ==============================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rand_uniform(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))


# ==============================
# Tight obstacle generator (Case 1)
# ==============================

def sample_obstacles_case1_tight(cfg: GenCfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tight, solvable corridor with enforced narrow gates + intruders."""
    rng = np.random.default_rng(cfg.seed)

    robot_diam = 2.0 * cfg.radius
    gap_mult   = rng.uniform(*cfg.gap_mult_range)
    gap_margin = rng.uniform(*cfg.gap_margin_range)
    target_gap = gap_mult * robot_diam + gap_margin

    # ensure solvable with the tube used in planning
    min_gap = 2.0 * cfg.inflate + 0.01
    if target_gap < min_gap:
        target_gap = min_gap
    half_gap   = 0.5 * target_gap

    eps = 0.012
    corridor_half_w = half_gap + 0.01

    p0 = cfg.start.astype(float); p1 = cfg.goal.astype(float)
    dir01 = p1 - p0; dir01 = dir01 / (np.linalg.norm(dir01) + 1e-12)
    ortho = np.array([-dir01[1], dir01[0]])
    amp   = cfg.corridor_wiggle_amp
    kfreq = cfg.corridor_wiggle_k
    ts = np.linspace(0.0, 1.0, 80)

    def centerline_point(t):
        base = (1 - t) * p0 + t * p1
        return base + amp * np.sin(2 * np.pi * kfreq * t) * ortho

    centerline = np.stack([centerline_point(t) for t in ts], axis=0)

    def dist_to_polyline(pt):
        return float(np.linalg.norm(centerline - pt[None, :], axis=1).min())

    def disk_outside_corridor(c, r):
        return dist_to_polyline(c) >= (corridor_half_w + r)

    def non_overlap(c, r, Cs, Rs, slack):
        if Cs.size == 0: return True
        d2 = np.sum((Cs - c[None, :])**2, axis=1)
        min_allow2 = (Rs + r + slack)**2
        return np.all(d2 >= min_allow2)

    def clear_of_pts(c, r, pts, clear):
        if pts.size == 0: return True
        d = np.linalg.norm(pts - c[None, :], axis=1)
        return np.all(d >= (r + clear))

    xmin, xmax = cfg.env_x_bounds
    ymin, ymax = cfg.env_y_bounds

    Cs = np.zeros((0, 2), float)
    Rs = np.zeros((0,), float)
    Ws = np.zeros((0,), float)

    # gates
    n_gates = int(rng.integers(cfg.n_gates_range[0], cfg.n_gates_range[1] + 1))
    for _ in range(n_gates):
        t_gate = float(rng.uniform(0.18, 0.82))
        g = centerline_point(t_gate)
        rL = float(rng.uniform(0.32, 0.55))
        rR = float(rng.uniform(0.32, 0.55))
        cL = g - ortho * (half_gap + rL + eps)
        cR = g + ortho * (half_gap + rR + eps)
        jitter = rng.uniform(-0.12, 0.12)
        cL = cL + jitter * dir01
        cR = cR + jitter * dir01
        if not non_overlap(cL, rL, Cs, Rs, cfg.min_separation_slack): continue
        if not non_overlap(cR, rR, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(cL, rL, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        if not clear_of_pts(cR, rR, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, cL, cR])
        Rs = np.concatenate([Rs, [rL, rR]])
        Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range)),
                                  float(rng.uniform(*cfg.weight_range))]])

    # intruders
    n_intr = int(rng.integers(cfg.n_intruders_range[0], cfg.n_intruders_range[1] + 1))
    for _ in range(n_intr):
        t_int = float(rng.uniform(0.20, 0.80))
        g = centerline_point(t_int)
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        rI = float(rng.uniform(0.28, 0.48))
        depth = float(rng.uniform(0.65, 0.90)) * half_gap
        depth = min(depth, half_gap - rI - 2*eps)
        cI = g + side * ortho * (half_gap - depth + rI + eps)
        cI = cI + rng.uniform(-0.10, 0.10) * dir01
        if not non_overlap(cI, rI, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(cI, rI, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, cI]); Rs = np.concatenate([Rs, [rI]])
        Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range))]])

    # clutter strictly outside corridor
    n_target = int(rng.integers(max(10, cfg.n_obs_range[0]), cfg.n_obs_range[1] + 1))
    tries = 0
    while Cs.shape[0] < n_target and tries < cfg.poisson_max_tries:
        tries += 1
        c = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], float)
        r = float(rng.uniform(cfg.radius_range[0], cfg.radius_range[1]))
        if not disk_outside_corridor(c, r): continue
        if not non_overlap(c, r, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(c, r, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, c]); Rs = np.concatenate([Rs, [r]])
        Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range))]])

    # purge near start/goal
    keep = np.ones(Cs.shape[0], dtype=bool)
    for k in range(Cs.shape[0]):
        if (np.linalg.norm(Cs[k] - p0) < (Rs[k] + cfg.min_start_clear)) or \
           (np.linalg.norm(Cs[k] - p1) < (Rs[k] + cfg.min_goal_clear)):
            keep[k] = False
    Cs, Rs, Ws = Cs[keep], Rs[keep], Ws[keep]
    return Cs.astype(np.float64), Rs.astype(np.float64), Ws.astype(np.float64)

def sample_obstacles_case2_harder(cfg: GenCfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Harder but still solvable:
      - stronger corridor wiggle (piecewise sinusoid)
      - more gates + deeper intruders
      - clustered clutter near corridor edges
      - short 'wall belts' (rows of disks) forming pockets outside the corridor
    The corridor width always respects the planning tube (>= 2*inflate + slack).
    """
    rng = np.random.default_rng(cfg.seed)

    # --- Corridor geometry (piecewise wiggle) ---
    p0 = cfg.start.astype(float); p1 = cfg.goal.astype(float)
    L  = np.linalg.norm(p1 - p0) + 1e-12
    t_breaks = np.array([0.0, 0.33, 0.66, 1.0])
    amps  = rng.uniform(cfg.corridor_wiggle_amp*0.9, cfg.corridor_wiggle_amp*1.5, size=3)
    freqs = rng.uniform(cfg.corridor_wiggle_k*0.9,  cfg.corridor_wiggle_k*1.6,  size=3)

    d01 = (p1 - p0) / L
    ortho = np.array([-d01[1], d01[0]])

    def centerline_point(t):
        # piecewise sinusoid to create sharper bends
        seg = min(2, int(np.searchsorted(t_breaks, t) - 1))
        amp, kfreq = amps[seg], freqs[seg]
        base = (1 - t) * p0 + t * p1
        return base + amp * np.sin(2*np.pi*kfreq*(t - t_breaks[seg])/(t_breaks[seg+1]-t_breaks[seg] + 1e-12)) * ortho

    ts = np.linspace(0.0, 1.0, 120)
    centerline = np.stack([centerline_point(t) for t in ts], axis=0)

    def dist_to_polyline(pt):
        return float(np.linalg.norm(centerline - pt[None, :], axis=1).min())

    # --- Corridor width (enforce solvable tube) ---
    robot_diam = 2.0 * cfg.radius
    gap_mult   = rng.uniform(0.65, 0.9)          # slightly tighter baseline than before
    gap_margin = rng.uniform(0.01, 0.04)
    target_gap = max(gap_mult * robot_diam + gap_margin, 2.0*cfg.inflate + 0.012)
    half_gap   = 0.5 * target_gap
    corridor_half_w = half_gap + 0.01
    eps = 0.012

    xmin, xmax = cfg.env_x_bounds
    ymin, ymax = cfg.env_y_bounds

    def disk_outside_corridor(c, r):
        return dist_to_polyline(c) >= (corridor_half_w + r)

    def non_overlap(c, r, Cs, Rs, slack):
        if Cs.size == 0: return True
        d2 = np.sum((Cs - c[None, :])**2, axis=1)
        min_allow2 = (Rs + r + slack)**2
        return np.all(d2 >= min_allow2)

    def clear_of_pts(c, r, pts, clear):
        if pts.size == 0: return True
        d = np.linalg.norm(pts - c[None, :], axis=1)
        return np.all(d >= (r + clear))

    Cs = np.zeros((0, 2), float)
    Rs = np.zeros((0,), float)
    Ws = np.zeros((0,), float)

    # --- Gates: more of them, varied radii, mild jitter along the path
    n_gates = int(rng.integers(max(4, cfg.n_gates_range[0]+1), cfg.n_gates_range[1] + 2))
    for _ in range(n_gates):
        t_gate = float(rng.uniform(0.10, 0.90))
        g = centerline_point(t_gate)
        rL = float(rng.uniform(0.34, 0.60))
        rR = float(rng.uniform(0.34, 0.60))
        cL = g - ortho * (half_gap + rL + eps)
        cR = g + ortho * (half_gap + rR + eps)
        jitter = rng.uniform(-0.18, 0.18) * L  # stronger along-path jitter
        cL = cL + jitter * d01
        cR = cR + jitter * d01
        if not non_overlap(cL, rL, Cs, Rs, cfg.min_separation_slack): continue
        if not non_overlap(cR, rR, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(cL, rL, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        if not clear_of_pts(cR, rR, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, cL, cR])
        Rs = np.concatenate([Rs, [rL, rR]])
        Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range)),
                                  float(rng.uniform(*cfg.weight_range))]])

    # --- Intruders: deeper into corridor, slightly larger
    n_intr = int(rng.integers(cfg.n_intruders_range[0]+1, cfg.n_intruders_range[1] + 2))
    for _ in range(n_intr):
        t_int = float(rng.uniform(0.12, 0.88))
        g = centerline_point(t_int)
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        rI = float(rng.uniform(0.32, 0.52))
        depth = float(rng.uniform(0.80, 0.98)) * half_gap           # deeper!
        depth = min(depth, half_gap - rI - 2*eps)
        if depth < 0.02: continue
        cI = g + side * ortho * (half_gap - depth + rI + eps)
        cI = cI + rng.uniform(-0.18, 0.18) * d01 * L
        if not non_overlap(cI, rI, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(cI, rI, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, cI]); Rs = np.concatenate([Rs, [rI]])
        Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range))]])

    # --- Clustered clutter near corridor edges (tempting pockets)
    n_clusters = rng.integers(2, 4)
    for _ in range(n_clusters):
        # pick a center on the corridor, then push outward a bit
        tC = float(rng.uniform(0.15, 0.85))
        base = centerline_point(tC)
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        offset = corridor_half_w + rng.uniform(0.12, 0.40)
        cluster_center = base + side * ortho * offset
        n_in_cluster = rng.integers(3, 7)
        for _k in range(n_in_cluster):
            ang = rng.uniform(0, 2*np.pi)
            rad = rng.uniform(0.00, 0.40)
            c = cluster_center + rad * np.array([np.cos(ang), np.sin(ang)])
            r = float(rng.uniform(cfg.radius_range[0], min(cfg.radius_range[1], 0.65)))
            if not non_overlap(c, r, Cs, Rs, cfg.min_separation_slack): continue
            if not clear_of_pts(c, r, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
            if not disk_outside_corridor(c, r):  # keep clusters just outside the corridor
                continue
            Cs = np.vstack([Cs, c]); Rs = np.concatenate([Rs, [r]])
            Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range))]])

    # --- Short 'wall belts' (rows of overlapping disks) to form cul-de-sacs outside corridor
    n_belts = rng.integers(1, 3)
    for _ in range(n_belts):
        tB = float(rng.uniform(0.20, 0.80))
        base = centerline_point(tB)
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        belt_dir = d01 * rng.choice([1.0, -1.0])  # parallel or anti-parallel to path
        belt_start = base + side * ortho * (corridor_half_w + rng.uniform(0.25, 0.60))
        n_links = rng.integers(4, 7)
        link_step = rng.uniform(0.45, 0.75)
        r_link = rng.uniform(0.30, 0.55)
        for k in range(n_links):
            c = belt_start + k * link_step * belt_dir
            r = float(r_link * rng.uniform(0.85, 1.15))
            if not non_overlap(c, r, Cs, Rs, cfg.min_separation_slack): continue
            if not clear_of_pts(c, r, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
            if not disk_outside_corridor(c, r):  # keep belts off the corridor
                continue
            Cs = np.vstack([Cs, c]); Rs = np.concatenate([Rs, [r]])
            Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range))]])

    # --- Background clutter (a bit denser than before), but still outside corridor
    n_target = rng.integers(max(16, cfg.n_obs_range[0]+4), cfg.n_obs_range[1] + 8)
    tries, tries_max = 0, cfg.poisson_max_tries
    while Cs.shape[0] < n_target and tries < tries_max:
        tries += 1
        c = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], float)
        r = float(rng.uniform(cfg.radius_range[0], cfg.radius_range[1]))
        if not disk_outside_corridor(c, r): continue
        if not non_overlap(c, r, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(c, r, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, c]); Rs = np.concatenate([Rs, [r]])
        Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range))]])

    # --- Purge near start/goal (keep spawn/target clear)
    keep = np.ones(Cs.shape[0], dtype=bool)
    for k in range(Cs.shape[0]):
        if (np.linalg.norm(Cs[k] - p0) < (Rs[k] + cfg.min_start_clear)) or \
           (np.linalg.norm(Cs[k] - p1) < (Rs[k] + cfg.min_goal_clear)):
            keep[k] = False
    Cs, Rs, Ws = Cs[keep], Rs[keep], Ws[keep]
    return Cs.astype(np.float64), Rs.astype(np.float64), Ws.astype(np.float64)


# ==============================
# Stage manager & planner (reuse she.StageManager)
# ==============================

@dataclass
class StageLite:
    center: np.ndarray
    size: Tuple[float, float]
    entry_point: np.ndarray
    exit_point: np.ndarray
    stage_id: int
    def __post_init__(self):
        W, H = self.size; cx, cy = self.center
        self.bounds = (cx - W/2, cx + W/2, cy - H/2, cy + H/2)

class PlannerOverlapHyper:
    """
    Uses spline_stagewise6.StageManager (which picks exits on the overlap boundary
    via the shortest-path-with-width logic) and the hyperelastic ring dynamics.
    """
    def __init__(self, start_xy, goal_xy, stage_size=(2.6,2.0), overlap=0.3,
                 n_ctrl=20, K=240, d_hat=1.0, seed=7,
                 radius=0.35,
                 k_bulk=1.5, gamma_s=2.0, M_s=1.0,
                 M_o=1.5, gamma_o=4.0, I=0.6, gamma_th=1.6,
                 obstacles=None, inflate=0.35):

        # Stage manager from she (overlap-constrained exit computation)
        self.sm = she.StageManager(
            np.array(start_xy), np.array(goal_xy),
            stage_size=stage_size, overlap_ratio=overlap,
            obstacles=obstacles, inflate=inflate
        )

        # Hyperelastic ring system
        self.sys = she.HyperelasticRingSystem(
            n_ctrl=n_ctrl, K=K, d_hat=d_hat, seed=seed, radius=radius,
            k_bulk=k_bulk, gamma_s=gamma_s, M_s=M_s,
            M_o=M_o, gamma_o=gamma_o, I=I, gamma_th=gamma_th
        )

        # initialize pose
        self.sys.o = torch.tensor(start_xy, dtype=self.sys.dtype)
        d = np.array(goal_xy) - np.array(start_xy)
        self.sys.theta = torch.tensor(math.atan2(d[1], d[0]), dtype=self.sys.dtype)

        self.stage_field = she.StageForceFieldTorch(
            r_safe=0.35, r_contact=0.18,
            w_goal=1.0, w_radial=2.0, w_tangential=1.2, w_flow=0.8, w_boundary=0.6
        )

        self.inflate = float(inflate)

    def stage_slice(self, C,R,W) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        st = self.sm.current()
        xmin,xmax,ymin,ymax = st.bounds
        m = (C[:,0] >= xmin-0.5) & (C[:,0] <= xmax+0.5) & (C[:,1] >= ymin-0.5) & (C[:,1] <= ymax+0.5)
        if np.any(m): return C[m],R[m],W[m]
        return np.zeros((0,2)), np.zeros((0)), np.zeros((0))

    def step(self, dt, world_obs: she.WorldObstacles):
        st = self.sm.current()

        # Tube-aware runtime distances: inflate radii by the same 'inflate' used by StageManager
        C,R,W = self.stage_slice(world_obs.C_np, world_obs.R_np, world_obs.W_np)
        R_eff = R + self.inflate if len(R) else R
        obs_t  = she.ObstacleProviderTorch(C, R_eff, W, dtype=self.sys.dtype)

        # Barrier activation at tube radius (matches planning-time tube)
        barrier = she.IPCBarrier(obs_t, d_hat=self.inflate)

        info = self.sys.step(dt, obs_t, barrier, self.stage_field, st.bounds, tuple(st.exit_point))
        self.sm.advance_if_needed(np.array(info["center"], float))
        return info


# ==============================
# Episode / dataset I/O
# ==============================

def perturb_robot_params(cfg: GenCfg) -> Dict[str, float]:
    return {
        "k_bulk":  cfg.k_bulk  * rand_uniform(*cfg.k_bulk_mult_range),
        "gamma_s": cfg.gamma_s * rand_uniform(*cfg.gamma_s_mult_range),
        "d_hat":   cfg.d_hat   * rand_uniform(*cfg.d_hat_mult_range),
        "radius":  rand_uniform(*cfg.radius_abs_range),
    }

def stage_term_weights(stage_field) -> Dict[str, float]:
    return {
        "w_goal": float(getattr(stage_field, "w_goal", 1.0)),
        "w_radial": float(getattr(stage_field, "w_radial", 2.0)),
        "w_tangential": float(getattr(stage_field, "w_tangential", 1.0)),
        "w_flow": float(getattr(stage_field, "w_flow", 0.6)),
        "w_boundary": float(getattr(stage_field, "w_boundary", 0.6)),
    }

def episode_success(cfg: GenCfg, final_center: np.ndarray, *_args, **_kwargs) -> bool:
    """Define success as simply reaching the goal within tolerance."""
    goal_dist = np.linalg.norm(np.asarray(final_center, float) - cfg.goal)
    return goal_dist <= cfg.goal_tol

def save_episode_snapshot(
    snapshot_path: str,
    planner: PlannerOverlapHyper,
    frames: list,
    world_obs: she.WorldObstacles,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    cfg: GenCfg,
    title="Maximin (decoupled ROI) stage exits with Hyperelastic Ring: start/mid/end"
):
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    if not frames:
        return None
    start_Pw = frames[0]["Pw"].numpy()
    end_Pw   = frames[-1]["Pw"].numpy()
    mid_Pw   = frames[len(frames)//2]["Pw"].numpy() if len(frames) > 2 else None

    fig = plt.figure(figsize=(7.6, 3.8), dpi=140)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", "box")

    # raw obstacles
    she.draw_obstacles(ax, world_obs.C_np, world_obs.R_np, ec="k")
    # overlay inflated keep-out (what the planner respects)
    if cfg.overlay_inflated_in_snapshot:
        she.draw_obstacles(ax, world_obs.C_np, world_obs.R_np + cfg.inflate, ec="#00cfdc", alpha=0.35)

    she.render_curve(ax, planner.sys.B, start_Pw, "start",  "#1f77b4")
    if mid_Pw is not None:
        she.render_curve(ax, planner.sys.B, mid_Pw,   "mid",    "#ff7f0e")
    she.render_curve(ax, planner.sys.B, end_Pw,   "end",    "#2ca02c")

    for i, st in enumerate(planner.sm.stages):
        xmin, xmax, ymin, ymax = st.bounds
        ax.add_patch(plt.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, fill=False,
                                   lw=1.6, ec="#00cfdc" if i==planner.sm.current_stage_idx else (0.4,0.6,0.6,0.6)))
        ax.plot([st.entry_point[0]],[st.entry_point[1]],"o",ms=4,color="#85ff85")
        ax.plot([st.exit_point[0]],[st.exit_point[1]],"o",ms=4,color="#ff8585")

    ax.plot([start_xy[0]],[start_xy[1]],"o",color="green",ms=6)
    ax.plot([goal_xy[0]],[goal_xy[1]],"*",color="gold",ms=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.set_xlim(-2.5, 10.0); ax.set_ylim(-4.0, 4.0)

    fig.savefig(snapshot_path, bbox_inches="tight")
    plt.close(fig)
    return snapshot_path

def planner_from_cfg(
    cfg: GenCfg,
    world_obs: she.WorldObstacles,
    k_bulk: float,
    gamma_s: float,
    d_hat: float,
    radius: float
) -> PlannerOverlapHyper:
    planner = PlannerOverlapHyper(
        start_xy=cfg.start,
        goal_xy=cfg.goal,
        stage_size=cfg.stage_size,
        overlap=cfg.overlap,
        n_ctrl=cfg.n_ctrl,
        K=cfg.K,
        d_hat=d_hat,
        seed=cfg.seed_robot,
        radius=radius,
        k_bulk=k_bulk, gamma_s=gamma_s, M_s=cfg.M_s,
        M_o=cfg.M_o, gamma_o=cfg.gamma_o, I=cfg.I, gamma_th=cfg.gamma_th,
        obstacles=(world_obs.C_np, world_obs.R_np),
        inflate=cfg.inflate,
    )
    return planner

def run_episode(
    cfg: GenCfg,
    world_obs: she.WorldObstacles,
    k_bulk: float,
    gamma_s: float,
    d_hat: float,
    radius: float,
    snapshot_path: str,
) -> Dict[str, Any]:
    planner = planner_from_cfg(cfg, world_obs, k_bulk, gamma_s, d_hat, radius)

    frames = []
    energies = []
    frame_state = []
    min_d_history = []

    meta = {
        "start": cfg.start.tolist(),
        "goal": cfg.goal.tolist(),
        "stage_size": list(cfg.stage_size),
        "overlap": float(cfg.overlap),
        "inflate": float(cfg.inflate),
        "dt": float(cfg.dt),
        "total_steps": int(cfg.total_steps),
        "n_ctrl": int(cfg.n_ctrl),
        "K": int(cfg.K),
    }

    tw = stage_term_weights(planner.stage_field)

    for t in range(cfg.total_steps):
        info = planner.step(cfg.dt, world_obs)

        # tube-aware already (radii inflated in planner.step)
        min_d_history.append(float(info["min_d"]))

        if (t % cfg.snapshot_every) == 0 or t in (0, cfg.total_steps // 2, cfg.total_steps - 1):
            sys = planner.sys
            with torch.no_grad():
                Pw = sys.world_points().detach().cpu()
                X  = (sys.B @ Pw).detach().cpu()
                frames.append({
                    "t": t,
                    "Pw": Pw,
                    "X":  X,
                    "o":  sys.o.detach().cpu(),
                    "theta": float(sys.theta.item()),
                    "v_o": sys.v_o.detach().cpu(),
                    "omega": float(sys.omega.item()),
                    "scale": float(sys.s.item()),
                    "sdot": float(sys.sdot.item()),
                    "center": sys.o.detach().cpu(),
                })

        if (t % cfg.snapshot_every) == 0:
            st = planner.sm.current()
            frame_state.append({
                "t": t,
                "stage_idx": int(planner.sm.current_stage_idx),
                "bounds": list(st.bounds),
                "entry_point": st.entry_point.tolist(),
                "exit_point": st.exit_point.tolist(),
                "center": st.center.tolist(),
            })

        energies.append({
            "t": t,
            "U_barrier": float(info["U_barrier"]),
            "U_bulk": float(info["U_bulk"]),
            "min_d": float(info["min_d"]),
        })

    final_center = planner.sys.o.detach().cpu().numpy()
    final_theta  = float(planner.sys.theta.item())
    final_scale  = float(planner.sys.s.item())
    success = episode_success(cfg, final_center, min_d_history)

    snap = save_episode_snapshot(snapshot_path, planner, frames, world_obs, cfg.start, cfg.goal, cfg)

    episode = {
        "meta": meta,
        "params": {
            "k_bulk": float(k_bulk),
            "gamma_s": float(gamma_s),
            "d_hat": float(d_hat),
            "radius": float(radius),
            "gamma": float(planner.sys.gamma_o),
            "gamma_th": float(planner.sys.gamma_th),
            "gamma_o": float(planner.sys.gamma_o),
            **tw,
        },
        "obstacles": {
            "centers": world_obs.C_np.astype(np.float32),
            "radii":   world_obs.R_np.astype(np.float32),
            "weights": world_obs.W_np.astype(np.float32),
            "d_hat_env": float(world_obs.d_hat),
        },
        "frames": frames,
        "energies": energies,
        "frame_state": frame_state,
        "success": bool(success),
        "final_center": final_center.tolist(),
        "final_theta": final_theta,
        "final_scale": final_scale,
        "snapshot_path": snap,
    }
    return episode

def save_episode(root: str, case_tag: str, idx: int, episode: Dict[str, Any]) -> str:
    os.makedirs(os.path.join(root, "episodes"), exist_ok=True)
    path = os.path.join(root, "episodes", f"episode_{case_tag}_{idx:02d}.pt")
    ep = dict(episode)
    obs = ep["obstacles"]
    obs_t = {
        "centers": torch.from_numpy(obs["centers"]),
        "radii":   torch.from_numpy(obs["radii"]),
        "weights": torch.from_numpy(obs["weights"]),
        "d_hat_env": obs["d_hat_env"],
    }
    ep["obstacles"] = obs_t
    torch.save(ep, path)
    return path

def write_manifest(root: str, records: List[Dict[str, Any]]):
    man_path = os.path.join(root, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(records, f, indent=2)
    return man_path


# ==============================
# Dataset loader
# ==============================

class NavigatorEpisodes(Dataset):
    def __init__(self, root: str, filter_case: Optional[str] = None, success_only: Optional[bool] = None):
        super().__init__()
        manifest = os.path.join(root, "manifest.json")
        with open(manifest, "r") as f:
            self.records = json.load(f)
        if filter_case is not None:
            self.records = [r for r in self.records if r["case"] == filter_case]
        if success_only is not None:
            self.records = [r for r in self.records if r["success"] == bool(success_only)]
        self.root = root
    def __len__(self): return len(self.records)
    def __getitem__(self, idx: int):
        rec = self.records[idx]
        ep = torch.load(rec["path"], map_location="cpu")
        return ep


# ==============================
# Main
# ==============================

def main():
    cfg = GenCfg()
    set_all_seeds(cfg.seed)

    os.makedirs(cfg.out_root, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_root, "episodes"), exist_ok=True)
    records: List[Dict[str, Any]] = []

    # ---------- Case 1: environment variations (TIGHT) ----------
    for i in range(cfg.num_env_variations):
        k_bulk, gamma_s, d_hat, radius = cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius
        C, R, W = sample_obstacles_case2_harder(cfg)
        world = she.WorldObstacles(C, R, W, d_hat=cfg.d_hat)

        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_env_{i:02d}_snapshot.png")
        ep = run_episode(cfg, world, k_bulk, gamma_s, d_hat, radius, snapshot_path=snap_path)
        path = save_episode(cfg.out_root, "env", i, ep)
        records.append({
            "case": "env",
            "idx": i,
            "path": path,
            "success": ep["success"],
            "n_obstacles": int(world.C_np.shape[0]),
            "snapshot": ep["snapshot_path"],
        })
        print(f"[Case1 env #{i:02d}] saved: {path} | success={ep['success']} | n_obs={world.C_np.shape[0]}")

    # ---------- Case 2: robot variations (fixed obstacle field) ----------
    rng_state = np.random.get_state()
    np.random.seed(cfg.seed + 777)
    C0, R0, W0 = sample_obstacles_case1_tight(cfg)
    np.random.set_state(rng_state)

    world_fixed = she.WorldObstacles(C0, R0, W0, d_hat=cfg.d_hat)
    for i in range(cfg.num_robot_variations):
        p = perturb_robot_params(cfg)
        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_robot_{i:02d}_snapshot.png")
        ep = run_episode(cfg, world_fixed, p["k_bulk"], p["gamma_s"], p["d_hat"], p["radius"], snapshot_path=snap_path)
        path = save_episode(cfg.out_root, "robot", i, ep)
        records.append({
            "case": "robot",
            "idx": i,
            "path": path,
            "success": ep["success"],
            "k_bulk": p["k_bulk"],
            "gamma_s": p["gamma_s"],
            "d_hat": p["d_hat"],
            "radius": p["radius"],
            "snapshot": ep["snapshot_path"],
        })
        print(f"[Case2 robot #{i:02d}] saved: {path} | success={ep['success']} | "
              f"k_bulk={p['k_bulk']:.3f} gamma_s={p['gamma_s']:.3f} d_hat={p['d_hat']:.3f} r={p['radius']:.3f}")

    man_path = write_manifest(cfg.out_root, records)
    print(f"\nWrote manifest: {man_path}")
    print("Done.")


if __name__ == "__main__":
    main()
