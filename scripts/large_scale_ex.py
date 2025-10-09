#!/usr/bin/env python3
# One large-scale example + MP4 movie with frame updates.
# Assumes 'spline_stagewise4' (ssi) is available on PYTHONPATH.

import sys, os, math, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
import matplotlib as mpl
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.8
mpl.rcParams['agg.path.chunksize'] = 10000

# If you need to add your scripts folder (adjust if different):
sys.path.append("/mnt/data/adityas/DPO/scripts")
import spline_stagewise4 as ssi


# ========= Base config you provided (trimmed replica for inheritance) =========
class GenCfg:
    out_root: str = "./nav_dataset_maximin"
    seed: int = 2025

    # simulation
    total_steps: int = 2000
    dt: float = 0.03
    snapshot_every: int = 5  # we will also use this as video frame stride

    # start/goal and stage geometry
    start = np.array([-1.0, -0.5], float)
    goal  = np.array([ 9.0,  0.2], float)
    stage_size = (2.6, 2.0)
    overlap = 0.30

    # tube inflation (planning clearance)
    inflate = 0.05

    # robot base params
    n_ctrl: int = 20
    K: int = 240
    lam_reg: float = 3.0
    mu: float = 0.25
    d_hat: float = 1.0
    seed_robot: int = 7
    radius: float = 0.35

    # success thresholds
    goal_tol: float = 0.30
    min_clear_tol: float = 1e-6

    # obstacle field (Case 1 tight corridor generator)
    env_x_bounds = (-0.5, 9.0)
    env_y_bounds = (-1.8, 1.8)
    radius_range = (0.28, 0.80)
    weight_range = (0.6, 1.5)
    n_obs_range = (12, 22)
    poisson_max_tries: int = 6000
    min_separation_slack: float = 0.08
    min_start_clear: float = 0.9
    min_goal_clear: float = 0.9

    # tight gap controls
    gap_mult_range = (0.70, 0.80)
    gap_margin_range = (0.01, 0.03)
    corridor_wiggle_amp = 0.35
    corridor_wiggle_k = 0.6
    n_gates_range = (3, 5)
    n_intruders_range = (2, 4)
    intruder_depth_frac_range = (0.60, 0.95)

    # graph parameters
    grid_res: int = 12
    knn_k: int = 7
    nsamp_edge: int = 20
    min_node_clear: float = 1e-3
    follow_corridor: bool = True

    # decoupled-search ROI
    roi_pad_scale_x: float = 0.8
    roi_pad_scale_y: float = 0.8
    search_union_current_next: bool = True

    # hard extra margins beyond tube
    min_edge_margin: float = 1e-3
    min_exit_margin: float = 1e-3

    # viz
    overlay_inflated_in_snapshot: bool = True


# ========= "Bigger scale, more obstacles; same core configs" =========
class BigCfg(GenCfg):
    # Output folder for this single example
    out_root = "./nav_large_example"

    # Make the world ~4x longer and ~2x taller
    start = np.array([-2.0, -1.2], float)
    goal  = np.array([ 32.0,  1.0], float)
    env_x_bounds = (-3.0, 34.0)
    env_y_bounds = (-4.0,  4.0)

    gap_mult_range = (0.55, 0.65)         # lower => narrower nominal gap
    gap_margin_range = (0.005, 0.02)      # smaller extra margin
    intruder_depth_frac_range = (0.85, 0.98)  # intruders push deeper in
    
    # Stage window scaled up to cover longer path comfortably
    stage_size = (5.2, 3.8)   # ~2x original
    overlap = 0.30            # keep same overlap fraction

    # Simulation: a bit longer to let it traverse
    total_steps = 3600
    snapshot_every = 8        # render every 3 sim steps for smoother video (dt=0.03 => ≈10 fps)
    dt = 0.03

    # Much denser obstacle field
    n_obs_range = (70, 120)
    radius_range = (0.30, 1.20)
    corridor_wiggle_amp = 0.60  # wider wiggle to force route finding
    n_gates_range = (6, 9)
    n_intruders_range = (5, 8)

    # Slightly denser graph for exit selection
    grid_res = 20
    knn_k = 9
    nsamp_edge = 28

    # Keep the planning tube inflation and robot params the same spirit
    inflate = 0.05
    n_ctrl = 20
    K = 240
    lam_reg = 3.0
    mu = 0.25
    d_hat = 1.0
    radius = 0.35

    # overlay
    overlay_inflated_in_snapshot = True


# ========= Helpers (directly in this file so it’s one runnable script) =========
def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rand_uniform(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))


def sample_obstacles_case1_tight(cfg: GenCfg):
    """Tight, solvable corridor with enforced narrow gates + intruders (scaled by cfg)."""
    rng = np.random.default_rng()

    robot_diam = 2.0 * cfg.radius
    gap_mult   = rng.uniform(*cfg.gap_mult_range)
    gap_margin = rng.uniform(*cfg.gap_margin_range)
    target_gap = max(gap_mult * robot_diam + gap_margin, 2.0 * cfg.inflate + 0.01)
    half_gap   = 0.5 * target_gap
    eps = 0.012
    corridor_half_w = half_gap + 0.01

    p0 = cfg.start.astype(float); p1 = cfg.goal.astype(float)
    dir01 = p1 - p0; dir01 = dir01 / (np.linalg.norm(dir01) + 1e-12)
    ortho = np.array([-dir01[1], dir01[0]])
    amp   = cfg.corridor_wiggle_amp
    kfreq = cfg.corridor_wiggle_k
    ts = np.linspace(0.0, 1.0, 120)

    def centerline_point(t):
        base = (1 - t) * p0 + t * p1
        return base + amp * np.sin(2 * np.pi * kfreq * t) * ortho

    centerline = np.stack([centerline_point(t) for t in ts], axis=0)

    def dist_to_polyline(pt):
        return float(np.linalg.norm(centerline - pt[None, :], axis=1).min())

    def disk_outside_corridor(c, r):  # forbid clutter inside corridor
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

    # Gates
    n_gates = int(rng.integers(cfg.n_gates_range[0], cfg.n_gates_range[1] + 1))
    for _ in range(n_gates):
        t_gate = float(rng.uniform(0.10, 0.90))
        g = centerline_point(t_gate)
        rL = float(rng.uniform(0.40, 0.85))
        rR = float(rng.uniform(0.40, 0.85))
        cL = g - ortho * (half_gap + rL + eps)
        cR = g + ortho * (half_gap + rR + eps)
        jitter = rng.uniform(-0.25, 0.25)
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

    # Intruders (push partially into corridor)
    n_intr = int(rng.integers(cfg.n_intruders_range[0], cfg.n_intruders_range[1] + 1))
    for _ in range(n_intr):
        t_int = float(rng.uniform(0.15, 0.85))
        g = centerline_point(t_int)
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        rI = float(rng.uniform(0.35, 0.75))
        depth = float(rng.uniform(0.60, 0.95)) * half_gap
        depth = min(depth, half_gap - rI - 2*eps)
        cI = g + side * ortho * (half_gap - depth + rI + eps)
        cI = cI + rng.uniform(-0.18, 0.18) * dir01
        if not non_overlap(cI, rI, Cs, Rs, cfg.min_separation_slack): continue
        if not clear_of_pts(cI, rI, np.stack([p0, p1]), min(cfg.min_start_clear, cfg.min_goal_clear)): continue
        Cs = np.vstack([Cs, cI]); Rs = np.concatenate([Rs, [rI]])
        Ws = np.concatenate([Ws, [float(rng.uniform(*cfg.weight_range))]])

    # Clutter strictly outside the corridor
    n_target = int(rng.integers(max(50, cfg.n_obs_range[0]), cfg.n_obs_range[1] + 1))
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

    # Purge too close to start/goal
    keep = np.ones(Cs.shape[0], dtype=bool)
    for k in range(Cs.shape[0]):
        if (np.linalg.norm(Cs[k] - p0) < (Rs[k] + cfg.min_start_clear)) or \
           (np.linalg.norm(Cs[k] - p1) < (Rs[k] + cfg.min_goal_clear)):
            keep[k] = False
    Cs, Rs, Ws = Cs[keep], Rs[keep], Ws[keep]
    return Cs.astype(np.float64), Rs.astype(np.float64), Ws.astype(np.float64)


def planner_from_cfg(cfg, world_obs, lam_reg, mu, d_hat, radius):
    """Small helper: make planner + scale the initial loop radius."""
    # Stage manager + deformable spline system
    planner = PlannerMaximin(
        start_xy=cfg.start,
        goal_xy=cfg.goal,
        stage_size=cfg.stage_size,
        overlap=cfg.overlap,
        n_ctrl=cfg.n_ctrl,
        K=cfg.K,
        d_hat=d_hat,
        lam_reg=lam_reg,
        mu=mu,
        seed=cfg.seed_robot,
        obstacles=(world_obs.C_np, world_obs.R_np),
        inflate=cfg.inflate,
        graph_cfg=cfg,
    )
    # Scale initial loop to requested radius prior
    with torch.no_grad():
        sys = planner.sys
        P0 = sys.P0_loc
        cur_norm = torch.mean(torch.linalg.norm(P0, dim=-1)).item() + 1e-8
        scale = float(radius / cur_norm)
        sys.P0_loc *= scale
        sys.Ploc   *= scale
    return planner


# ========= Planner/Stage classes (thin wrapper around your logic) =========
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Dict

@dataclass
class Stage:
    center: np.ndarray
    size: Tuple[float, float]
    entry_point: np.ndarray
    exit_point: np.ndarray
    stage_id: int
    def __post_init__(self):
        W, H = self.size; cx, cy = self.center
        self.bounds = (cx - W/2, cx + W/2, cy - H/2, cy + H/2)

def _union_bounds(b1, b2):
    x1a,x1b,y1a,y1b = b1; x2a,x2b,y2a,y2b = b2
    return (min(x1a,x2a), max(x1b,x2b), min(y1a,y2a), max(y1b,y2b))

def _pad_bounds(b, px, py):
    xmin,xmax,ymin,ymax = b
    return (xmin - px, xmax + px, ymin - py, ymax + py)

# --- (The graph helpers are lengthy; we’ll reuse your widest-path routine inline to keep behavior identical.) ---
# To keep this script focused, we import your previously defined helpers where possible.
# If those helpers live in your dataset script, you can paste them here, or import as a module.

# Minimal imports of your maximin graph helpers (paste your originals if needed)
from math import inf
import heapq

def _sample_rect_boundary(bounds, n_per_edge: int = 18) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, n_per_edge, endpoint=True)
    ys = np.linspace(ymin, ymax, n_per_edge, endpoint=True)
    top    = np.stack([xs, np.full_like(xs, ymax)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, ymin)], axis=1)
    left   = np.stack([np.full_like(ys, xmin), ys], axis=1)
    right  = np.stack([np.full_like(ys, xmax), ys], axis=1)
    B = np.concatenate([top, bottom, left, right], axis=0)
    return np.unique(B, axis=0)

def _clearance_points(P: np.ndarray, C: np.ndarray, R: np.ndarray) -> np.ndarray:
    if C.size == 0: return np.full((P.shape[0],), np.inf, dtype=float)
    diff = P[:, None, :] - C[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    d = dist - R[None, :]
    return d.min(axis=1)

def _segment_clearance(a: np.ndarray, b: np.ndarray, C: np.ndarray, R: np.ndarray, nsamp: int = 20) -> float:
    ts = np.linspace(0.0, 1.0, nsamp)
    P = (1.0 - ts)[:, None] * a[None, :] + ts[:, None] * b[None, :]
    return float(_clearance_points(P, C, R).min())

def _los_segment(a: np.ndarray, b: np.ndarray, C: np.ndarray, R: np.ndarray) -> bool:
    if C.size == 0: return True
    ab = b - a
    L2 = float(np.dot(ab, ab)) + 1e-12
    ac = C - a[None, :]
    t = np.clip((ac @ ab) / L2, 0.0, 1.0)
    p = a[None, :] + t[:, None] * ab[None, :]
    d2 = np.sum((p - C)**2, axis=1)
    return bool(np.all(d2 > (R * R)))

def _sample_interior(bounds, grid_res: int = 12) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)

def _knn_indices(P: np.ndarray, k: int = 6) -> List[List[int]]:
    n = P.shape[0]
    nbrs: List[List[int]] = [[] for _ in range(n)]
    D2 = ((P[:, None, :] - P[None, :, :])**2).sum(axis=-1)
    np.fill_diagonal(D2, np.inf)
    kk = min(k, n-1)
    idxs = np.argpartition(D2, kth=kk-1, axis=1)[:, :kk]
    for i in range(n):
        nbrs[i] = [int(j) for j in idxs[i]]
    return nbrs

def _maximin_widest_path(entry_idx: int, target_idx_set: set,
                         P: np.ndarray, neighbors: List[List[int]],
                         edge_clear_fn):
    n = P.shape[0]
    best = np.full(n, -np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[entry_idx] = np.inf
    heap = [(-best[entry_idx], entry_idx)]
    visited = np.zeros(n, dtype=bool)
    while heap:
        _, u = heapq.heappop(heap)
        if visited[u]: continue
        visited[u] = True
        for v in neighbors[u]:
            if visited[v]: continue
            ce = edge_clear_fn(u, v)
            cand = min(best[u], ce)
            if cand > best[v]:
                best[v] = cand
                parent[v] = u
                heapq.heappush(heap, (-best[v], v))
    if not target_idx_set: return -1, parent, {"best": best}
    tgt_list = list(target_idx_set)
    scores = best[tgt_list]
    if np.all(scores == -np.inf): return -1, parent, {"best": best}
    best_tgt = tgt_list[int(np.argmax(scores))]
    return best_tgt, parent, {"best": best}

def _reconstruct_path(parent: np.ndarray, tgt: int) -> np.ndarray:
    if tgt < 0: return np.zeros((0,2), float)
    path = []
    cur = tgt
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path = path[::-1]
    return np.array(path, int)

def pick_maximin_boundary_exit_decoupled(entry, next_bounds, search_bounds, C_inf, R_inf,
                                         grid_res=12, k=7, min_node_clear=1e-3,
                                         nsamp_edge=20, min_exit_clearance=0.0, min_edge_clearance=0.0):
    """Same decoupled-ROI widest-exit search as your dataset script."""
    B_next = _sample_rect_boundary(next_bounds, n_per_edge=max(14, grid_res))
    I_roi  = _sample_interior(search_bounds, grid_res=grid_res)
    P = np.vstack([entry[None, :], B_next, I_roi])

    clr = _clearance_points(P, C_inf, R_inf)
    keep = clr > max(min_node_clear, 1e-4)
    keep[0] = True
    P, clr = P[keep], clr[keep]
    if P.shape[0] < 3:
        return None, None, -np.inf

    entry_idx = int(np.argmin(np.linalg.norm(P - entry[None, :], axis=1)))

    xmin,xmax,ymin,ymax = next_bounds
    epsb = 1e-6
    on_next = (np.isclose(P[:,0], xmin, atol=epsb) | np.isclose(P[:,0], xmax, atol=epsb) |
               np.isclose(P[:,1], ymin, atol=epsb) | np.isclose(P[:,1], ymax, atol=epsb))
    ok_exit = on_next & (clr >= (min_exit_clearance + 1e-6))
    targets = set(np.nonzero(ok_exit)[0].tolist())
    targets.discard(entry_idx)
    if not targets:
        return None, None, -np.inf

    nbrs = _knn_indices(P, k=k)
    for u in range(P.shape[0]):
        nbrs[u] = [v for v in nbrs[u] if _los_segment(P[u], P[v], C_inf, R_inf)]

    req_edge = float(min_edge_clearance)
    edge_cache: Dict[Tuple[int,int], float] = {}
    def edge_clear(u, v):
        key = (u, v) if u < v else (v, u)
        if key in edge_cache: return edge_cache[key]
        ce = _segment_clearance(P[u], P[v], C_inf, R_inf, nsamp=nsamp_edge)
        edge_cache[key] = (ce if ce >= req_edge - 1e-6 else -np.inf)
        return edge_cache[key]

    tgt_idx, parent, aux = _maximin_widest_path(entry_idx, targets, P, nbrs, edge_clear)
    if tgt_idx < 0 or not np.isfinite(aux["best"][tgt_idx]) or aux["best"][tgt_idx] <= 0.0:
        return None, None, -np.inf

    path_idx = _reconstruct_path(parent, tgt_idx)
    return P[tgt_idx].copy(), P[path_idx], float(aux["best"][tgt_idx])


class StageManagerMaximin:
    def __init__(self, start_xy, goal_xy, stage_size=(2.6,2.0), overlap_ratio=0.3,
                 obstacles=None, inflate=0.35, graph_cfg: Optional[GenCfg]=None):
        self.start = np.array(start_xy, float)
        self.goal  = np.array(goal_xy,  float)
        self.stage_size = tuple(stage_size)
        self.overlap_ratio = float(overlap_ratio)
        self.C = np.zeros((0, 2), float); self.R = np.zeros((0,), float)
        if obstacles is not None:
            C, R = obstacles
            self.C = np.array(C, float)
            self.R = np.array(R, float) + float(inflate)
        self.gcfg = graph_cfg or GenCfg()
        self.inflate = float(inflate)
        self.stages: List[Stage] = []
        self.current_stage_idx = 0
        self._build()

    def _build(self):
        d = self.goal - self.start
        L = np.linalg.norm(d)
        if L < 1e-6:
            self.stages = [Stage((self.start + self.goal) / 2, self.stage_size,
                                 self.start.copy(), self.goal.copy(), 0)]
            return
        step = self.stage_size[0] * (1.0 - self.overlap_ratio)
        n = max(1, int(np.ceil(L / step)))
        centers = []
        for i in range(n):
            t = 0.0 if n == 1 else i / (n - 1)
            centers.append(self.start + t * d)
        centers = [np.array(c, float) for c in centers]

        W, H = self.stage_size
        pad_x = self.gcfg.roi_pad_scale_x * W
        pad_y = self.gcfg.roi_pad_scale_y * H

        for i in range(n):
            c_i = centers[i]
            entry = self.start if i == 0 else self.stages[-1].exit_point
            if i == n - 1:
                exitp = self.goal
            else:
                nxt_center = centers[i + 1]
                next_bounds = (nxt_center[0] - W/2, nxt_center[0] + W/2,
                               nxt_center[1] - H/2, nxt_center[1] + H/2)
                cur_bounds  = (c_i[0] - W/2, c_i[0] + W/2,
                               c_i[1] - H/2, c_i[1] + H/2)
                roi = _union_bounds(cur_bounds, next_bounds) if self.gcfg.search_union_current_next else next_bounds
                search_bounds = _pad_bounds(roi, pad_x, pad_y)

                cand, poly, _ = pick_maximin_boundary_exit_decoupled(
                    entry=np.array(entry, float),
                    next_bounds=next_bounds,
                    search_bounds=search_bounds,
                    C_inf=self.C, R_inf=self.R,
                    grid_res=self.gcfg.grid_res,
                    k=self.gcfg.knn_k,
                    min_node_clear=self.gcfg.min_node_clear,
                    nsamp_edge=self.gcfg.nsamp_edge,
                    min_exit_clearance=self.gcfg.min_exit_margin,
                    min_edge_clearance=self.gcfg.min_edge_margin,
                )
                if cand is not None:
                    exitp = cand
                    if self.gcfg.follow_corridor and poly is not None and poly.shape[0] >= 2:
                        seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
                        if seg_len.size > 0 and np.sum(seg_len) > 1e-9:
                            s = np.cumsum(seg_len)
                            j = int(np.searchsorted(s, s[-1]*0.33))
                            j = min(max(j, 0), poly.shape[0]-1)
                            desired_center = poly[j]
                            centers[i + 1] = 0.7 * centers[i + 1] + 0.3 * desired_center
                else:
                    exitp = self.start + (i + 1) / n * d

            self.stages.append(Stage(c_i, self.stage_size, np.array(entry, float), np.array(exitp, float), i))

    def current(self) -> Stage:
        return self.stages[self.current_stage_idx]

    def advance_if_needed(self, robot_xy: np.ndarray, thresh=0.45) -> bool:
        st = self.current()
        if (np.linalg.norm(robot_xy - st.exit_point) < thresh) and (self.current_stage_idx < len(self.stages) - 1):
            self.current_stage_idx += 1
            return True
        if self.current_stage_idx < len(self.stages) - 1:
            nxt = self.stages[self.current_stage_idx + 1]
            path_dir = self.goal - self.start
            if np.dot(robot_xy - st.center, path_dir) > np.dot(nxt.center - st.center, path_dir):
                self.current_stage_idx += 1
                return True
        return False


class PlannerMaximin:
    def __init__(self, start_xy, goal_xy, stage_size=(2.5,2.0), overlap=0.3,
                 n_ctrl=20, K=240, d_hat=1.0, lam_reg=3.0, mu=0.25, seed=7,
                 obstacles=None, inflate=0.35, graph_cfg: Optional[GenCfg]=None):

        self.sm = StageManagerMaximin(
            np.array(start_xy), np.array(goal_xy),
            stage_size=stage_size, overlap_ratio=overlap,
            obstacles=obstacles, inflate=inflate, graph_cfg=graph_cfg
        )

        self.sys = ssi.DeformableSplineSystem(n_ctrl=n_ctrl, K=K, d_hat=d_hat, lam_reg=lam_reg, mu=mu, seed=seed)
        # initialize pose
        self.sys.o = torch.tensor(start_xy, dtype=self.sys.dtype)
        d = np.array(goal_xy)-np.array(start_xy)
        self.sys.theta = torch.tensor(math.atan2(d[1],d[0]), dtype=self.sys.dtype)

        self.stage_field = ssi.StageForceFieldTorch(r_safe=0.30, r_contact=0.16,
                                                    w_goal=1.0, w_radial=2.0, w_tangential=1.2, w_flow=0.8, w_boundary=0.6)

    def _stage_slice(self, C,R,W):
        st = self.sm.current(); xmin,xmax,ymin,ymax = st.bounds
        m = (C[:,0] >= xmin-0.5) & (C[:,0] <= xmax+0.5) & (C[:,1] >= ymin-0.5) & (C[:,1] <= ymax+0.5)
        if np.any(m): return C[m],R[m],W[m]
        return C,R,W

    def step(self, dt, world_obs: ssi.WorldObstacles):
        st = self.sm.current()
        C,R,W = self._stage_slice(world_obs.C_np, world_obs.R_np, world_obs.W_np)
        obs_t = ssi.ObstacleProviderTorch(C,R,W, dtype=self.sys.dtype)
        barrier = ssi.IPCBarrier(obs_t, d_hat=world_obs.d_hat)
        info = self.sys.step(dt, obs_t, barrier, self.stage_field, st.bounds, tuple(st.exit_point))
        self.sm.advance_if_needed(np.array(info["center"],float))
        return info


# ========= Movie runner =========
def run_large_example_and_movie():
    cfg = BigCfg()
    set_all_seeds(cfg.seed)
    os.makedirs(cfg.out_root, exist_ok=True)

    # Build a single, large environment
    C, R, W = sample_obstacles_case1_tight(cfg)
    world = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)

    # Planner with same core robot/configs (just scaled world)
    planner = planner_from_cfg(cfg, world,
                               lam_reg=cfg.lam_reg, mu=cfg.mu, d_hat=cfg.d_hat, radius=cfg.radius)

    # Figure & axes
    fig = plt.figure(figsize=(14, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", "box")

    # MP4 path
    mp4_path = os.path.join(cfg.out_root, "large_example.mp4")

    # FPS ~ 1/(dt*snapshot_every); clamp to >= 8 for a smoother look
    fps = 24 # max(8, int(round(1.0 / (cfg.dt * cfg.snapshot_every))))

    # Fallback: if ffmpeg not available, we’ll dump PNG frames instead
    ffmpeg_ok = writers.is_available("ffmpeg")
    png_dir = os.path.join(cfg.out_root, "frames_png")

    def draw_frame(t_step: int):
        ax.clear()
        ax.set_aspect("equal", "box")

        # Obstacles (raw + inflated)
        ssi.draw_obstacles(ax, world.C_np, world.R_np, ec="k")
        if cfg.overlay_inflated_in_snapshot:
            ssi.draw_obstacles(ax, world.C_np, world.R_np + cfg.inflate, ec="#00cfdc", alpha=0.25)

        # Stage windows and entry/exit markers
        for i, st in enumerate(planner.sm.stages):
            xmin, xmax, ymin, ymax = st.bounds
            ax.add_patch(plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                fill=False, lw=1.4,
                ec="#00cfdc" if i == planner.sm.current_stage_idx else (0.4,0.6,0.6,0.4)
            ))
            ax.plot([st.entry_point[0]],[st.entry_point[1]], "o", ms=3, color="#85ff85")
            ax.plot([st.exit_point[0]],[st.exit_point[1]], "o", ms=3, color="#ff8585")

        # Current spline shape
        sys = planner.sys
        with torch.no_grad():
            Pw = sys.world_points().detach().cpu().numpy()
        # ssi.render_curve(ax, planner.sys.B, Pw, label=None, color="#1f77b4")
        Pw = planner.sys.world_points().detach().cpu().numpy()
        B  = planner.sys.B.detach().cpu().numpy()
        X  = (B @ Pw)  # (K,2) curve samples

        (line,) = ax.plot(X[:,0], X[:,1],
                        lw=5.0,        # <-- make the spline look bigger
                        alpha=0.98,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        zorder=5)

        # add a light stroke for contrast (optional but helps visibility)
        import matplotlib.patheffects as pe
        line.set_path_effects([
            pe.Stroke(linewidth=7.0, foreground="white"),
            pe.Normal()
        ])

        # Robot COM, start/goal
        c = planner.sys.o.detach().cpu().numpy()
        ax.plot([c[0]],[c[1]], "o", ms=4, color="black")
        ax.plot([cfg.start[0]],[cfg.start[1]], "o", color="green", ms=6)
        ax.plot([cfg.goal[0]],[cfg.goal[1]], "*", color="gold", ms=10)

        # Nice bounds
        ax.set_xlim(cfg.env_x_bounds[0]-1.0, cfg.env_x_bounds[1]+1.0)
        ax.set_ylim(cfg.env_y_bounds[0]-1.0, cfg.env_y_bounds[1]+1.0)

        ax.set_title(f"Large-scale navigation  |  t={t_step*cfg.dt:.2f}s   "
                     f"stage {planner.sm.current_stage_idx+1}/{len(planner.sm.stages)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if ffmpeg_ok:
        writer = FFMpegWriter(
            fps=fps,
            codec="mpeg4",   
            bitrate=-1,
            extra_args=["-vtag", "mp4v", "-pix_fmt", "yuv420p"]  # ensure MP4 compatibility
        )
        with writer.saving(fig, mp4_path, dpi=120):
            for t in range(cfg.total_steps):
                info = planner.step(cfg.dt, world)
                # Only draw at snapshot cadence
                if (t % cfg.snapshot_every) != 0 and t not in (0, cfg.total_steps - 1):
                    continue
                draw_frame(t)
                writer.grab_frame()
        plt.close(fig)
        print(f"[OK] MP4 saved to: {mp4_path}")
    else:
        # PNG fallback
        os.makedirs(png_dir, exist_ok=True)
        frame_id = 0
        for t in range(cfg.total_steps):
            info = planner.step(cfg.dt, world)
            if (t % cfg.snapshot_every) != 0 and t not in (0, cfg.total_steps - 1):
                continue
            draw_frame(t)
            png_path = os.path.join(png_dir, f"frame_{frame_id:05d}.png")
            fig.savefig(png_path, bbox_inches="tight")
            frame_id += 1
        plt.close(fig)
        print(f"[WARN] ffmpeg not available. Wrote PNG frames to: {png_dir}")
        print("       To make an mp4 manually you can run something like:")
        print(f"       ffmpeg -y -framerate {fps} -i {png_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {mp4_path}")

    # Also dump a small manifest for reproducibility
    manifest = {
        "seed": cfg.seed,
        "dt": cfg.dt,
        "snapshot_every": cfg.snapshot_every,
        "fps": fps,
        "start": cfg.start.tolist(),
        "goal": cfg.goal.tolist(),
        "stage_size": list(cfg.stage_size),
        "env_bounds": [*cfg.env_x_bounds, *cfg.env_y_bounds],
        "inflate": cfg.inflate,
        "robot": {"n_ctrl": cfg.n_ctrl, "K": cfg.K, "lam_reg": cfg.lam_reg, "mu": cfg.mu, "d_hat": cfg.d_hat, "radius": cfg.radius},
        "graph": {
            "grid_res": cfg.grid_res, "knn_k": cfg.knn_k, "nsamp_edge": cfg.nsamp_edge,
            "min_node_clear": cfg.min_node_clear, "follow_corridor": bool(cfg.follow_corridor),
            "roi_pad_scale_x": cfg.roi_pad_scale_x, "roi_pad_scale_y": cfg.roi_pad_scale_y,
            "search_union_current_next": bool(cfg.search_union_current_next),
            "min_edge_margin": cfg.min_edge_margin, "min_exit_margin": cfg.min_exit_margin,
        },
        "outputs": {"mp4": mp4_path if ffmpeg_ok else None, "png_dir": png_dir if not ffmpeg_ok else None}
    }
    with open(os.path.join(cfg.out_root, "large_example_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    run_large_example_and_movie()
