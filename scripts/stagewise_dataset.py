#!/usr/bin/env python3
# Dataset generator that reuses the overlap-constrained stage logic from spline_stagewise6.

import sys
import os, json, math, random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.spline_stagewise6 as she 


# ==============================
# Config
# ==============================

class GenCfg:
    out_root: str = "./nav_stagewise_hyperring"
    seed: int = 2025

    gap_extra_path: float = 0.5 # <- widen free corridor by this many meters
    
    # simulation
    total_steps: int = 2000
    dt: float = 0.03
    snapshot_every: int = 5

    # start/goal and stage geometry
    start = np.array([-1.0, -0.5], float)
    goal  = np.array([ 9.0,  0.2], float)
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

    gap_mult_range = (0.70, 1.40)
    gap_margin_range = (0.05, 0.09)
    corridor_wiggle_amp = 0.55
    corridor_wiggle_k = 0.8
    n_gates_range = (3, 5)
    n_intruders_range = (2, 4)

    # *** Case 2 (robot perturbations) ***
    k_bulk_mult_range    = (0.90, 1.15)
    gamma_s_mult_range   = (0.85, 1.15)
    d_hat_mult_range     = (0.95, 1.05)
    radius_abs_range     = (0.30, 0.80)

    # viz
    overlay_inflated_in_snapshot: bool = True
    
    # --- extra dynamics knobs (Case 2+ robot/system variations) ---
    # contact/barrier threshold (usually ~ inflate; allow slight mismatch for robustness tests)
    d_hat_abs_range     = (0.02, 0.90)

    # squeeze law in bulk energy U_bulk(s): A_target = alpha + (1-alpha)*tanh(beta*clearance)
    alpha_squeeze_range = (0.15, 0.60)
    beta_squeeze_range  = (1.5, 4.0)

    # tangential friction proxy (used inside the hyperelastic step if supported)
    fric_mu_range       = (0.05, 0.30)
    v_t_clip_range      = (1.0,  3.0)

    # damping multipliers
    gamma_o_mult_range  = (0.70, 1.50)
    gamma_th_mult_range = (0.70, 1.50)


# ==============================
# Utility / RNG
# ==============================

def safety_violation_stats_explicit(ep, which_margin="tube"):
    """
    Returns (violated, min_clearance, first_t) where clearance is measured
    against RAW obstacles expanded by the chosen margin:
      - which_margin="tube": use ep["meta"]["inflate"] (your planning tube)
      - which_margin="barrier": use ep["params"]["d_hat"] (runtime barrier)
    """
    # obstacles saved raw
    C = ep["obstacles"]["centers"]  # np.ndarray (N,2) float32
    R = ep["obstacles"]["radii"]    # np.ndarray (N,)  float32

    if which_margin == "barrier":
        margin = float(ep["params"].get("d_hat", 0.0))
    else:
        margin = float(ep["meta"].get("inflate", 0.0))

    def frame_min_clearance(Pw_np):
        # Pw_np: (K,2) world points along ring/curve
        # clearance = min_p min_obs (||p - c|| - (R + margin))
        if C.size == 0:
            return np.inf
        # vectorized: for each p, compute distances to all obstacles
        dmin = np.inf
        for p in Pw_np:
            d = np.linalg.norm(C - p[None, :], axis=1) - (R + margin)
            dm = float(d.min())
            if dm < dmin:
                dmin = dm
        return dmin

    violated = False
    min_min_d = np.inf
    first_t = None

    for f in ep.get("frames", []):
        Pw = f["Pw"].numpy() if hasattr(f["Pw"], "numpy") else np.asarray(f["Pw"], dtype=float)
        t = int(f["t"])
        dmin = frame_min_clearance(Pw)
        if dmin < min_min_d:
            min_min_d = dmin
        if (not violated) and dmin < 0.0:
            violated = True
            first_t = t

    return violated, float(min_min_d), first_t


def make_checkpoint(t, stage_idx, st, info, sys, B,
                    C_step, R_step, W_step, inflate, barrier_params):
    with torch.no_grad():
        Pw = sys.world_points().detach().cpu()
        X  = (B @ Pw).detach().cpu()
    return {
        "t": int(t),
        "dt": None,  # you can set per-call if dt varies
        "stage_idx": int(stage_idx),
        "stage_bounds": [float(v) for v in st.bounds],
        "stage_entry": [float(st.entry_point[0]), float(st.entry_point[1])],
        "stage_exit":  [float(st.exit_point[0]),  float(st.exit_point[1])],  # <-- stage goal for this frame
        # state
        "center": list(map(float, info["center"])),
        "theta": float(info["theta"]),
        "scale": float(info["scale"]),
        "min_d": float(info["min_d"]),
        # energies
        "U_barrier": float(info["U_barrier"]),
        "U_bulk": float(info["U_bulk"]),
        # generalized forces used to integrate this step
        "F_s_total": float(info.get("F_s_total", 0.0)),
        "tau_total": float(info.get("tau_total", 0.0)),
        "F_o_total": [float(x) for x in info.get("F_o_total", [0.0, 0.0])],
        # constraints used on THIS step (post-inflate)
        "tube_inflate": float(inflate),
        "barrier": barrier_params,
        "obstacles_effective": {
            "C": np.asarray(C_step, float).tolist(),
            "R_eff": (np.asarray(R_step, float) + float(inflate)).tolist() if len(R_step) else [],
            "W": np.asarray(W_step, float).tolist(),
        },
        # shapes (comment out if too heavy)
        "Pw": Pw.numpy().tolist(),
        "X":  X.numpy().tolist(),
    }


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rand_uniform(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))

def export_sys_params(sys):
    return {
        "n_ctrl": sys.B.shape[1],
        "K_samples": sys.B.shape[0],
        "d_hat": float(sys.d_hat),
        "k_bulk": float(sys.k_bulk),
        "gamma_s": float(sys.gamma_s),
        "M_s": float(sys.M_s),
        "M_o": float(sys.M_o),
        "gamma_o": float(sys.gamma_o),
        "I": float(sys.I),
        "gamma_th": float(sys.gamma_th),
        "radius_base": float(sys.P0_loc.norm(dim=1).mean().item()),  # average radius of control ring at s=1
    }

def export_stage_field_params(sf):
    return {
        "r_safe": float(sf.r_safe),
        "r_contact": float(sf.r_contact),
        "w_goal": float(sf.w_goal),
        "w_radial": float(sf.w_radial),
        "w_tangential": float(sf.w_tangential),
        "w_flow": float(sf.w_flow),
        "w_boundary": float(sf.w_boundary),
    }

def export_barrier_params(barrier):
    return {
        "barrier_d_hat": float(barrier.d_hat),
        "violation_penalty": float(barrier.vp),
        "barrier_eps": float(barrier.eps),
        "barrier_max_grad": float(barrier.max_grad),
        "barrier_max_b": float(barrier.max_b),
    }

def export_obstacles_np(C, R, W):
    return {
        "C": np.asarray(C, float).tolist(),
        "R": np.asarray(R, float).tolist(),
        "W": np.asarray(W, float).tolist(),
    }


# ==============================
# Tight obstacle generator (Case 1)
# ==============================

def sample_obstacles_case1_tight(cfg: GenCfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tight, solvable corridor with enforced narrow gates + intruders."""
    rng = np.random.default_rng()

    robot_diam = 2.0 * cfg.radius
    gap_mult   = rng.uniform(*cfg.gap_mult_range)
    gap_margin = rng.uniform(*cfg.gap_margin_range)
    target_gap = gap_mult * robot_diam + gap_margin

    # ensure solvable with the tube used in planning
    min_gap = 2.0 * cfg.inflate + 0.01
    if target_gap < min_gap:
        target_gap = min_gap
    
    # >>> widen corridor by requested extra gap
    target_gap += float(cfg.gap_extra_path)
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
    rng = np.random.default_rng()

    # --- Corridor geometry (piecewise wiggle) ---
    p0 = cfg.start.astype(float); p1 = cfg.goal.astype(float)
    L  = np.linalg.norm(p1 - p0) + 1e-12
    t_breaks = np.array([0.0, 0.33, 0.66, 1.0])
    amps  = rng.uniform(cfg.corridor_wiggle_amp*1.2, cfg.corridor_wiggle_amp*1.7, size=3)
    freqs = rng.uniform(cfg.corridor_wiggle_k*1.2,  cfg.corridor_wiggle_k*1.7,  size=3)

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
    robot_diam = 2.2 * cfg.radius
    gap_mult   = rng.uniform(0.85, 1.4)          # slightly tighter baseline than before
    gap_margin = rng.uniform(0.05, 0.1)
    target_gap = max(gap_mult * robot_diam + gap_margin, 2.0*cfg.inflate + 0.012)
    # >>> widen corridor by requested extra gap
    target_gap += float(cfg.gap_extra_path)
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
# Tight obstacle generator (Case 3)
# ==============================

def _build_occ_grid_xy(cfg, C, R, grid_res=0.10, margin=None):
    """Return (occ, xs, ys) with occ shape (ny, nx), True=blocked."""
    if margin is None:
        margin = float(cfg.inflate)
    xmin, xmax = cfg.env_x_bounds
    ymin, ymax = cfg.env_y_bounds
    xs = np.arange(xmin, xmax + 1e-9, grid_res)
    ys = np.arange(ymin, ymax + 1e-9, grid_res)
    nx, ny = len(xs), len(ys)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx)

    occ = np.zeros((ny, nx), dtype=bool)
    if C.size:
        # Inflate obstacles by margin and rasterize disks
        for (cx, cy), rr in zip(C, R + margin):
            occ |= (XX - cx) ** 2 + (YY - cy) ** 2 <= rr ** 2
    return occ, xs, ys  # (ny, nx), (nx,), (ny,)

def _label_components(free_mask: np.ndarray) -> np.ndarray:
    """4-connected component labeling. Returns labels (int32), 0=background."""
    ny, nx = free_mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    lbl = 0
    for j in range(ny):
        for i in range(nx):
            if not free_mask[j, i] or labels[j, i] != 0:
                continue
            lbl += 1
            # BFS flood fill
            dq = deque([(i, j)])
            labels[j, i] = lbl
            while dq:
                ci, cj = dq.popleft()
                for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nx and 0 <= nj < ny and free_mask[nj, ni] and labels[nj, ni] == 0:
                        labels[nj, ni] = lbl
                        dq.append((ni, nj))
    return labels

def _idx_to_pos(i: int, j: int, xs, ys) -> np.ndarray:
    return np.array([xs[i], ys[j]], float)

def sample_start_goal_case3(
    cfg,
    C: np.ndarray,
    R: np.ndarray,
    grid_res: float = 0.10,
    min_world_separation: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Picks (start, goal) in free space (with inflate margin) from the same
    connected component, chosen to be far apart, so a collision-free grid
    path exists by construction.
    """
    occ, xs, ys = _build_occ_grid_xy(cfg, C, R, grid_res=grid_res, margin=cfg.inflate)
    free = ~occ
    if not free.any():
        # no free space; fall back
        return cfg.start.copy(), cfg.goal.copy()

    # Label free space once.
    labels = _label_components(free)
    # Choose the largest free-space component.
    lbls, counts = np.unique(labels[labels > 0], return_counts=True)
    if lbls.size == 0:
        return cfg.start.copy(), cfg.goal.copy()
    comp = lbls[np.argmax(counts)]

    # All cells in this component
    js, is_ = np.where(labels == comp)  # rows (y idx), cols (x idx)
    pts = np.stack([is_, js], axis=1)   # (N, 2) in (i, j)

    # Heuristic: bias start near the left side, goal near the right side
    xs_all = xs[is_]
    x_lo_cut = np.quantile(xs_all, 0.20)
    x_hi_cut = np.quantile(xs_all, 0.80)
    left_idx  = np.where(xs_all <= x_lo_cut)[0]
    right_idx = np.where(xs_all >= x_hi_cut)[0]
    rng = np.random.default_rng()

    if left_idx.size and right_idx.size:
        s_idx = pts[left_idx[rng.integers(0, left_idx.size)]]
        # choose farthest right candidate from s_idx (cheap 1-vs-many)
        s_xy = _idx_to_pos(s_idx[0], s_idx[1], xs, ys)
        cand_right = pts[right_idx]
        cand_xy = np.stack([xs[cand_right[:,0]], ys[cand_right[:,1]]], axis=1)
        d2 = np.sum((cand_xy - s_xy[None,:])**2, axis=1)
        g_idx = cand_right[int(np.argmax(d2))]
    else:
        # Fallback: farthest-point pair within the component using a 2-pass trick
        k = min(2048, pts.shape[0])
        sample_ids = rng.choice(pts.shape[0], size=k, replace=False)
        S = pts[sample_ids]
        # pick a random anchor, find farthest A
        anchor = S[rng.integers(0, k)]
        A_xy = _idx_to_pos(anchor[0], anchor[1], xs, ys)
        S_xy = np.stack([xs[S[:,0]], ys[S[:,1]]], axis=1)
        d2A = np.sum((S_xy - A_xy[None,:])**2, axis=1)
        A = S[int(np.argmax(d2A))]
        A_xy = _idx_to_pos(A[0], A[1], xs, ys)
        # farthest from A is B
        d2B = np.sum((S_xy - A_xy[None,:])**2, axis=1)
        B = S[int(np.argmax(d2B))]
        s_idx, g_idx = A, B

    s = _idx_to_pos(int(s_idx[0]), int(s_idx[1]), xs, ys)
    g = _idx_to_pos(int(g_idx[0]), int(g_idx[1]), xs, ys)

    # Enforce a minimum world-space separation; if too close, stretch goal to farthest.
    if np.linalg.norm(g - s) < min_world_separation:
        cand_xy = np.stack([xs[pts[:,0]], ys[pts[:,1]]], axis=1)
        d2 = np.sum((cand_xy - s[None,:])**2, axis=1)
        g = cand_xy[int(np.argmax(d2))]

    return s, g

def build_centerline_from_waypoints(waypoints: List[np.ndarray], n=160) -> np.ndarray:
    W = np.array(waypoints, float)
    seglen = np.linalg.norm(np.diff(W, axis=0), axis=1)
    L = np.sum(seglen) + 1e-9
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        d = t * L
        acc = 0.0
        for i, Ls in enumerate(seglen):
            if acc + Ls >= d:
                u = (d - acc) / (Ls + 1e-12)
                p = (1-u) * W[i] + u * W[i+1]
                pts.append(p)
                break
            acc += Ls
    return np.stack(pts, axis=0)

def sample_obstacles_polyline_corridor(
    cfg: GenCfg, waypoints: List[np.ndarray],
    tighten=(0.65, 0.9), # like gap_mult
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    centerline = build_centerline_from_waypoints(waypoints, n=160)

    def dist_to_polyline(pt):
        return float(np.linalg.norm(centerline - pt[None, :], axis=1).min())

    # tube width (respect inflate)
    robot_diam = 2.0 * cfg.radius
    gap_mult   = rng.uniform(*tighten)
    gap_margin = rng.uniform(0.01, 0.04)
    target_gap = max(gap_mult * robot_diam + gap_margin, 2.0*cfg.inflate + 0.012)
    # >>> widen corridor by requested extra gap
    target_gap += float(cfg.gap_extra_path)
    half_gap   = 0.5 * target_gap
    
    corridor_half_w = half_gap + 0.01
    eps = 0.012

    xmin, xmax = cfg.env_x_bounds
    ymin, ymax = cfg.env_y_bounds

    def disk_outside_corridor(c, r): return dist_to_polyline(c) >= (corridor_half_w + r)
    def non_overlap(c, r, Cs, Rs, slack):
        if Cs.size == 0: return True
        d2 = np.sum((Cs - c[None, :])**2, axis=1)
        return np.all(d2 >= (Rs + r + slack)**2)
    def clear_of_pts(c, r, pts, clear):
        if pts.size == 0: return True
        d = np.linalg.norm(pts - c[None, :], axis=1)
        return np.all(d >= (r + clear))

    Cs = np.zeros((0,2)); Rs = np.zeros((0,)); Ws = np.zeros((0,))
    # Gates & intruders are placed by sampling points along the polyline and offsetting orthogonally.
    # Compute local tangent/normal from finite differences:
    T = np.gradient(centerline, axis=0)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    N = np.stack([-T[:,1], T[:,0]], axis=1)

    # Gates
    for _ in range(np.random.randint(4, 7)):
        i = np.random.randint(10, len(centerline)-10)
        g = centerline[i]; n = N[i]
        rL = float(np.random.uniform(0.34, 0.60))
        rR = float(np.random.uniform(0.34, 0.60))
        cL = g - n * (half_gap + rL + eps)
        cR = g + n * (half_gap + rR + eps)
        if not non_overlap(cL, rL, Cs, Rs, cfg.min_separation_slack): continue
        if not non_overlap(cR, rR, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(cL, rL, np.stack([cfg.start, cfg.goal]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        if not clear_of_pts(cR, rR, np.stack([cfg.start, cfg.goal]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, cL, cR]); Rs = np.concatenate([Rs, [rL, rR]])
        Ws = np.concatenate([Ws, [float(np.random.uniform(*cfg.weight_range)),
                                  float(np.random.uniform(*cfg.weight_range))]])

    # Intruders
    for _ in range(np.random.randint(3, 6)):
        i = np.random.randint(10, len(centerline)-10)
        g = centerline[i]; n = N[i]
        side = 1.0 if np.random.rand() < 0.5 else -1.0
        rI = float(np.random.uniform(0.32, 0.52))
        depth = float(np.random.uniform(0.80, 0.98)) * half_gap
        depth = min(depth, half_gap - rI - 2*eps)
        if depth < 0.02: continue
        cI = g + side * n * (half_gap - depth + rI + eps)
        if not non_overlap(cI, rI, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(cI, rI, np.stack([cfg.start, cfg.goal]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, cI]); Rs = np.concatenate([Rs, [rI]])
        Ws = np.concatenate([Ws, [float(np.random.uniform(*cfg.weight_range))]])

    # Background clutter (outside corridor)
    n_target = np.random.randint(18, 28)
    tries = 0
    while Cs.shape[0] < n_target and tries < cfg.poisson_max_tries:
        tries += 1
        c = np.array([rand_uniform(xmin, xmax), rand_uniform(ymin, ymax)], float)
        r = float(rand_uniform(cfg.radius_range[0], cfg.radius_range[1]))
        if not disk_outside_corridor(c, r): continue
        if not non_overlap(c, r, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(c, r, np.stack([cfg.start, cfg.goal]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, c]); Rs = np.concatenate([Rs, [r]])
        Ws = np.concatenate([Ws, [float(rand_uniform(*cfg.weight_range))]])

    # Purge near start/goal
    keep = np.ones(Cs.shape[0], dtype=bool)
    for k in range(Cs.shape[0]):
        if (np.linalg.norm(Cs[k] - cfg.start) < (Rs[k] + cfg.min_start_clear)) or \
           (np.linalg.norm(Cs[k] - cfg.goal)  < (Rs[k] + cfg.min_goal_clear)):
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
        # Make the system’s internal contact threshold match the tube
        self.sys.d_hat = torch.tensor(d_hat, dtype=self.sys.dtype)

    def _stage_slice(self, C,R,W) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        st = self.sm.current()
        xmin,xmax,ymin,ymax = st.bounds
        m = (C[:,0] >= xmin-0.5) & (C[:,0] <= xmax+0.5) & (C[:,1] >= ymin-0.5) & (C[:,1] <= ymax+0.5)
        if np.any(m): return C[m],R[m],W[m]
        return C[:0], R[:0], W[:0]

    def step(self, dt, world_obs: she.WorldObstacles):
        idx_before = self.sm.current_stage_idx
        st_before  = self.sm.current()

        # Slice & inflate for the stage we’re integrating in
        C, R, W = self._stage_slice(world_obs.C_np, world_obs.R_np, world_obs.W_np)
        R_eff = R + self.inflate if len(R) else R
        obs_t = she.ObstacleProviderTorch(C, R_eff, W, dtype=self.sys.dtype)

        # Use the system's (possibly perturbed) d_hat, not inflate
        barrier = she.IPCBarrier(obs_t, d_hat=float(self.sys.d_hat))

        info = self.sys.step(dt, obs_t, barrier, self.stage_field, st_before.bounds, tuple(st_before.exit_point))

        # Advance stage if we crossed the exit
        advanced  = self.sm.advance_if_needed(np.array(info["center"], float))
        idx_after = self.sm.current_stage_idx
        st_after  = self.sm.current()

        runtime = {
            "C": C, "R": R, "W": W,
            "barrier": {
                "barrier_d_hat": float(barrier.d_hat),
                "violation_penalty": float(barrier.vp),
                "barrier_eps": float(barrier.eps),
                "barrier_max_grad": float(barrier.max_grad),
                "barrier_max_b": float(barrier.max_b),
            },
        }
        # info, index before step, index after step, stage used (before), stage after, runtime
        return info, idx_before, idx_after, st_before, st_after, runtime

# ==============================
# Episode / dataset I/O
# ==============================

def perturb_robot_params(cfg: GenCfg) -> Dict[str, float]:
    return {
        "k_bulk":   cfg.k_bulk  * rand_uniform(*cfg.k_bulk_mult_range),
        "gamma_s":  cfg.gamma_s * rand_uniform(*cfg.gamma_s_mult_range),
        "d_hat":    rand_uniform(*cfg.d_hat_abs_range),          # allow to deviate from inflate
        "radius":   rand_uniform(*cfg.radius_abs_range),

        # NEW:
        "alpha_sq": rand_uniform(*cfg.alpha_squeeze_range),
        "beta_sq":  rand_uniform(*cfg.beta_squeeze_range),
        "fric_mu":  rand_uniform(*cfg.fric_mu_range),
        "v_t_clip": rand_uniform(*cfg.v_t_clip_range),
        "gamma_o":  cfg.gamma_o * rand_uniform(*cfg.gamma_o_mult_range),
        "gamma_th": cfg.gamma_th * rand_uniform(*cfg.gamma_th_mult_range),
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
    ax.set_title("Maximin (decoupled ROI) stage exits with Hyperelastic Ring: start/mid/end")
    ax.set_xlim(-2.5, 10.0); ax.set_ylim(-2.4, 2.4)

    fig.savefig(snapshot_path, bbox_inches="tight")
    plt.close(fig)
    return snapshot_path

def planner_from_cfg(cfg: GenCfg, world_obs: she.WorldObstacles,
                     k_bulk: float, gamma_s: float, d_hat: float, radius: float,
                     alpha_sq: Optional[float]=None, beta_sq: Optional[float]=None,
                     fric_mu: Optional[float]=None, v_t_clip: Optional[float]=None,
                     gamma_o: Optional[float]=None, gamma_th: Optional[float]=None) -> PlannerOverlapHyper:

    planner = PlannerOverlapHyper(
        start_xy=cfg.start, goal_xy=cfg.goal,
        stage_size=cfg.stage_size, overlap=cfg.overlap,
        n_ctrl=cfg.n_ctrl, K=cfg.K, d_hat=d_hat, seed=cfg.seed_robot,
        radius=radius, k_bulk=k_bulk, gamma_s=gamma_s, M_s=cfg.M_s,
        M_o=cfg.M_o, gamma_o=cfg.gamma_o, I=cfg.I, gamma_th=cfg.gamma_th,
        obstacles=(world_obs.C_np, world_obs.R_np),
        inflate=cfg.inflate,
    )
    # ---- Tuning knobs (only if the underlying class exposes them) ----
    # Contact threshold (barrier): set both the system and the step barrier via inflate if you want strict tube == inflate.
    if hasattr(planner.sys, "d_hat"):      planner.sys.d_hat = torch.tensor(d_hat, dtype=planner.sys.dtype)

    if gamma_o is not None and hasattr(planner.sys, "gamma_o"):  planner.sys.gamma_o = float(gamma_o)
    if gamma_th is not None and hasattr(planner.sys, "gamma_th"): planner.sys.gamma_th = float(gamma_th)

    if alpha_sq is not None: setattr(planner.sys, "alpha_squeeze", float(alpha_sq))
    if beta_sq  is not None: setattr(planner.sys, "beta_squeeze",  float(beta_sq))

    if fric_mu  is not None: setattr(planner.sys, "fric_mu",  float(fric_mu))
    if v_t_clip is not None: setattr(planner.sys, "v_t_clip", float(v_t_clip))

    return planner

def run_episode(
    cfg: GenCfg,
    world_obs: she.WorldObstacles,
    k_bulk: float,
    gamma_s: float,
    d_hat: float,
    radius: float,
    snapshot_path: str,
    params: Optional[Dict[str, float]] = None,   # NEW: extra knobs for Case 2+
) -> Dict[str, Any]:
    params = params or {}

    # Build planner with optional overrides (alpha/beta squeeze, friction, damping)
    planner = planner_from_cfg(
        cfg, world_obs,
        k_bulk, gamma_s, d_hat, radius,
        alpha_sq=params.get("alpha_sq"),
        beta_sq=params.get("beta_sq"),
        fric_mu=params.get("fric_mu"),
        v_t_clip=params.get("v_t_clip"),
        gamma_o=params.get("gamma_o"),
        gamma_th=params.get("gamma_th"),
    )

    frames: List[Dict[str, Any]] = []
    energies: List[Dict[str, Any]] = []
    frame_state: List[Dict[str, Any]] = []
    min_d_history: List[float] = []

    # ---- RUN HEADER (once) ----
    run_header = {
        "start": cfg.start.tolist(),
        "goal": cfg.goal.tolist(),
        "stage_size": list(cfg.stage_size),
        "overlap": float(cfg.overlap),
        "dt": float(cfg.dt),
        "total_steps": int(cfg.total_steps),
        "inflate": float(cfg.inflate),
        "system": export_sys_params(planner.sys),
        "stage_field": export_stage_field_params(planner.stage_field),
        "world_obstacles": export_obstacles_np(world_obs.C_np, world_obs.R_np, world_obs.W_np),
    }
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    run_header_path = os.path.join(os.path.dirname(snapshot_path), "run_header.json")
    with open(run_header_path, "w") as f:
        json.dump(run_header, f, indent=2)

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

    checkpoints: List[Dict[str, Any]] = []
    stage_headers: List[Dict[str, Any]] = []
    last_stage_idx: Optional[int] = None

    for t in range(cfg.total_steps):
        # info, index before step, index after step, stage used (before), stage after, runtime
        info, idx_before, idx_after, st_before, st_after, runtime = planner.step(cfg.dt, world_obs)

        # Track min clearance history (optional diagnostic)
        min_d_history.append(float(info["min_d"]))

        # Stage entry header when we ENTER a new stage (idx_after changed)
        if last_stage_idx is None or idx_after != last_stage_idx:
            # Pre-inflate slice for header (uses current stage = st_after)
            C_step, R_step, W_step = planner._stage_slice(world_obs.C_np, world_obs.R_np, world_obs.W_np)
            stage_headers.append({
                "t_enter": int(t),
                "stage_idx": int(idx_after),
                "bounds": list(st_after.bounds),
                "entry_point": st_after.entry_point.tolist(),
                "exit_point":  st_after.exit_point.tolist(),
                "obstacles_stage": export_obstacles_np(C_step, R_step, W_step),
            })
            last_stage_idx = idx_after

        # Framewise checkpoint: log the stage we actually integrated in (st_before)
        C_eff, R_eff, W_eff = runtime["C"], runtime["R"], runtime["W"]
        ck = make_checkpoint(
            t, idx_before, st_before, info, planner.sys, planner.sys.B,
            C_eff, R_eff, W_eff, cfg.inflate, runtime["barrier"]
        )
        ck["dt"] = float(cfg.dt)
        checkpoints.append(ck)

        # Snapshots / frame states (unchanged)
        if (t % cfg.snapshot_every) == 0 or t in (0, cfg.total_steps // 2, cfg.total_steps - 1):
            sys = planner.sys
            with torch.no_grad():
                Pw = sys.world_points().detach().cpu()
                X  = (sys.B @ Pw).detach().cpu()
                frames.append({
                    "t": t, "Pw": Pw, "X": X,
                    "o": sys.o.detach().cpu(),
                    "theta": float(sys.theta.item()),
                    "v_o": sys.v_o.detach().cpu(),
                    "omega": float(sys.omega.item()),
                    "scale": float(sys.s.item()),
                    "sdot": float(sys.sdot.item()),
                    "center": sys.o.detach().cpu(),
                })

        if (t % cfg.snapshot_every) == 0:
            frame_state.append({
                "t": t,
                "stage_idx": int(idx_before),
                "bounds": list(st_before.bounds),
                "entry_point": st_before.entry_point.tolist(),
                "exit_point": st_before.exit_point.tolist(),
                "center": st_before.center.tolist(),
            })

        energies.append({
            "t": t,
            "U_barrier": float(info["U_barrier"]),
            "U_bulk": float(info["U_bulk"]),
            "min_d": float(info["min_d"]),
        })

    # Write per-episode aux logs
    log_dir = os.path.dirname(snapshot_path)
    stage_headers_path = os.path.join(log_dir, "stage_headers.json")
    checkpoints_jsonl_path = os.path.join(log_dir, "stagewise_checkpoints.jsonl")
    with open(stage_headers_path, "w") as f:
        json.dump(stage_headers, f, indent=2)
    with open(checkpoints_jsonl_path, "w") as f:
        for ck in checkpoints:
            f.write(json.dumps(ck) + "\n")

    # Final stats / success
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
            "gamma_o": float(planner.sys.gamma_o),
            "gamma_th": float(planner.sys.gamma_th),
            **tw,
            # include optional perturbations actually used (if any)
            **({k: v for k, v in params.items()
                if k in ("alpha_sq", "beta_sq", "fric_mu", "v_t_clip", "gamma_o", "gamma_th")}),
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
        "logs": {
            "run_header": run_header_path,
            "stage_headers": stage_headers_path,
            "checkpoints_jsonl": checkpoints_jsonl_path,
        },
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

    # Helper to get both tube + barrier safety summaries
    def safety_report(ep):
        tube_viol, tube_min_d, tube_first_t = safety_violation_stats_explicit(ep, which_margin="tube")
        bar_viol,  bar_min_d,  bar_first_t  = safety_violation_stats_explicit(ep, which_margin="barrier")
        return tube_viol, tube_min_d, tube_first_t, bar_viol, bar_min_d, bar_first_t

    # ---------- Case 1: environment variations (TIGHT) ----------
    for i in range(cfg.num_env_variations):
        k_bulk, gamma_s, d_hat, radius = cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius
        C, R, W = sample_obstacles_case2_harder(cfg)
        world = she.WorldObstacles(C, R, W, d_hat=cfg.d_hat)

        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_env_{i:02d}_snapshot.png")
        ep = run_episode(cfg, world, k_bulk, gamma_s, d_hat, radius, snapshot_path=snap_path)
        tube_viol, tube_min_d, tube_first_t, bar_viol, bar_min_d, bar_first_t = safety_report(ep)
        path = save_episode(cfg.out_root, "env", i, ep)
        print(
            f"[Case env #{i:02d}] saved: {path} | success={ep['success']} | "
            f"n_obs={world.C_np.shape[0]} | "
            f"tube_violation={tube_viol} (tube_min_d={tube_min_d:.3f}"
            f"{'' if tube_first_t is None else f', first_t={tube_first_t}'}) | "
            f"barrier_violation={bar_viol} (barrier_min_d={bar_min_d:.3f}"
            f"{'' if bar_first_t is None else f', first_t={bar_first_t}'})"
        )
        records.append({
            "case": "env",
            "idx": i,
            "path": path,
            "success": ep["success"],
            "n_obstacles": int(world.C_np.shape[0]),
            "snapshot": ep["snapshot_path"],
        })

    # ---------- Case 2: robot variations (fixed obstacle field) ----------
    rng_state = np.random.get_state()
    np.random.seed(cfg.seed + 777)
    C0, R0, W0 = sample_obstacles_case1_tight(cfg)
    np.random.set_state(rng_state)

    world_fixed = she.WorldObstacles(C0, R0, W0, d_hat=cfg.d_hat)
    for i in range(cfg.num_robot_variations):
        p = perturb_robot_params(cfg)
        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_robot_{i:02d}_snapshot.png")
        ep = run_episode(
            cfg, world_fixed, p["k_bulk"], p["gamma_s"], p["d_hat"], p["radius"],
            snapshot_path=snap_path, params=p
        )
        tube_viol, tube_min_d, tube_first_t, bar_viol, bar_min_d, bar_first_t = safety_report(ep)
        path = save_episode(cfg.out_root, "robot", i, ep)
        print(
            f"[Case robot #{i:02d}] saved: {path} | success={ep['success']} | "
            f"n_obs={world_fixed.C_np.shape[0]} | "
            f"tube_violation={tube_viol} (tube_min_d={tube_min_d:.3f}"
            f"{'' if tube_first_t is None else f', first_t={tube_first_t}'}) | "
            f"barrier_violation={bar_viol} (barrier_min_d={bar_min_d:.3f}"
            f"{'' if bar_first_t is None else f', first_t={bar_first_t}'})"
        )
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

    # ---------- Case 3: randomized start/goal, guaranteed feasible corridor ----------
    for i in range(6):
        C, R, W = sample_obstacles_case1_tight(cfg)
        s, g = sample_start_goal_case3(cfg, C, R, grid_res=0.10, min_world_separation=4.0)

        # swap cfg start/goal just for this episode (so planner + purges see them)
        old_s, old_g = cfg.start.copy(), cfg.goal.copy()
        cfg.start, cfg.goal = s, g

        world = she.WorldObstacles(C, R, W, d_hat=cfg.d_hat)
        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_startgoal_{i:02d}_snapshot.png")
        ep = run_episode(cfg, world, cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius, snapshot_path=snap_path)

        cfg.start, cfg.goal = old_s, old_g  # restore

        tube_viol, tube_min_d, tube_first_t, bar_viol, bar_min_d, bar_first_t = safety_report(ep)
        path = save_episode(cfg.out_root, "startgoal", i, ep)
        print(
            f"[Case startgoal #{i:02d}] saved: {path} | success={ep['success']} | "
            f"n_obs={world.C_np.shape[0]} | "
            f"tube_violation={tube_viol} (tube_min_d={tube_min_d:.3f}"
            f"{'' if tube_first_t is None else f', first_t={tube_first_t}'}) | "
            f"barrier_violation={bar_viol} (barrier_min_d={bar_min_d:.3f}"
            f"{'' if bar_first_t is None else f', first_t={bar_first_t}'})"
        )
        records.append({
            "case": "startgoal",
            "idx": i,
            "path": path,
            "success": ep["success"],
            "n_obstacles": int(world.C_np.shape[0]),
            "start": s.tolist(),
            "goal": g.tolist(),
            "snapshot": ep["snapshot_path"],
        })

    # ---------- Case 4: path shape = 'L' ----------
    for i in range(4):
        s = cfg.start.copy()
        g = cfg.goal.copy()
        corner = np.array([(s[0] + g[0]) * 0.5, s[1] + 2.0])  # go up, then right
        waypoints = [s, corner, g]

        old_s, old_g = cfg.start.copy(), cfg.goal.copy()
        cfg.start, cfg.goal = s, g
        C, R, W = sample_obstacles_polyline_corridor(cfg, waypoints)
        cfg.start, cfg.goal = old_s, old_g

        world = she.WorldObstacles(C, R, W, d_hat=cfg.d_hat)
        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_L_{i:02d}_snapshot.png")
        ep = run_episode(cfg, world, cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius, snapshot_path=snap_path)
        tube_viol, tube_min_d, tube_first_t, bar_viol, bar_min_d, bar_first_t = safety_report(ep)
        path = save_episode(cfg.out_root, "Lshape", i, ep)
        print(
            f"[Case Lshape #{i:02d}] saved: {path} | success={ep['success']} | "
            f"n_obs={world.C_np.shape[0]} | "
            f"tube_violation={tube_viol} (tube_min_d={tube_min_d:.3f}"
            f"{'' if tube_first_t is None else f', first_t={tube_first_t}'}) | "
            f"barrier_violation={bar_viol} (barrier_min_d={bar_min_d:.3f}"
            f"{'' if bar_first_t is None else f', first_t={bar_first_t}'})"
        )
        records.append({
            "case": "Lshape",
            "idx": i,
            "path": path,
            "success": ep["success"],
            "n_obstacles": int(world.C_np.shape[0]),
            "snapshot": ep["snapshot_path"],
        })

    # ---------- Case 5: path shape = 'ZigZag' ----------
    for i in range(4):
        s = cfg.start.copy()
        g = cfg.goal.copy()
        mid1 = s + np.array([2.2,  1.6])
        mid2 = s + np.array([4.6, -1.4])
        waypoints = [s, mid1, mid2, g]

        old_s, old_g = cfg.start.copy(), cfg.goal.copy()
        cfg.start, cfg.goal = s, g
        C, R, W = sample_obstacles_polyline_corridor(cfg, waypoints, tighten=(0.65, 0.85))
        cfg.start, cfg.goal = old_s, old_g

        world = she.WorldObstacles(C, R, W, d_hat=cfg.d_hat)
        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_Z_{i:02d}_snapshot.png")
        ep = run_episode(cfg, world, cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius, snapshot_path=snap_path)
        tube_viol, tube_min_d, tube_first_t, bar_viol, bar_min_d, bar_first_t = safety_report(ep)
        path = save_episode(cfg.out_root, "zigzag", i, ep)
        print(
            f"[Case zigzag #{i:02d}] saved: {path} | success={ep['success']} | "
            f"n_obs={world.C_np.shape[0]} | "
            f"tube_violation={tube_viol} (tube_min_d={tube_min_d:.3f}"
            f"{'' if tube_first_t is None else f', first_t={tube_first_t}'}) | "
            f"barrier_violation={bar_viol} (barrier_min_d={bar_min_d:.3f}"
            f"{'' if bar_first_t is None else f', first_t={bar_first_t}'})"
        )
        records.append({
            "case": "zigzag",
            "idx": i,
            "path": path,
            "success": ep["success"],
            "n_obstacles": int(world.C_np.shape[0]),
            "snapshot": ep["snapshot_path"],
        })

    man_path = write_manifest(cfg.out_root, records)
    print(f"\nWrote manifest: {man_path}")
    print("Done.")


if __name__ == "__main__":
    main()
