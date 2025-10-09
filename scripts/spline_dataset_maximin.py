
#!/usr/bin/env python3
# Generate a dataset using a max–min (widest-path) exit selection per stage
# with decoupled search region, hard tube clearance, and solvable tight corridors.

import sys
import os, json, math, random, heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.spline_stagewise4 as ssi


# ==============================
# Config
# ==============================

class GenCfg:
    out_root: str = "./nav_dataset_maximin"
    seed: int = 2025

    # simulation
    total_steps: int = 2000
    dt: float = 0.03
    snapshot_every: int = 5

    # start/goal and stage geometry (stage centers may be corridor-followed)
    start = np.array([-1.0, -0.5], float)
    goal  = np.array([ 9.0,  0.2], float)
    stage_size = (2.6, 2.0)
    overlap = 0.30

    # ---- tube radius (robot radius + margin) ----
    # We inflate obstacles by this much; guarantees a free tube of radius = inflate.
    inflate = 0.05

    # robot base params (perturbed for Case 2)
    n_ctrl: int = 20
    K: int = 240
    lam_reg: float = 3.0
    mu: float = 0.25
    d_hat: float = 1.0
    seed_robot: int = 7
    radius: float = 0.35  # initial loop radius (shape prior for the spline)

    # success thresholds
    goal_tol: float = 0.30
    min_clear_tol: float = 1e-6

    # dataset sizes
    num_env_variations: int = 10
    num_robot_variations: int = 10

    # *** Case 1 (tight corridor) obstacle generation ***
    env_x_bounds = (-0.5, 9.0)
    env_y_bounds = (-1.8, 1.8)
    radius_range = (0.28, 0.80)
    weight_range = (0.6, 1.5)
    n_obs_range = (12, 22)        # total count target (gates + intruders + clutter)
    poisson_max_tries: int = 6000
    min_separation_slack: float = 0.08
    min_start_clear: float = 0.9
    min_goal_clear: float = 0.9

    # tight gap controls (Case 1) — we clamp to >= 2*inflate + eps to ensure solvable
    gap_mult_range = (0.70, 0.80)    # gap ≈ mult * (2*radius) + margin
    gap_margin_range = (0.01, 0.03)
    corridor_wiggle_amp = 0.35
    corridor_wiggle_k = 0.6
    n_gates_range = (3, 5)
    n_intruders_range = (2, 4)

    # *** Case 2 (robot perturbations) ***
    lam_reg_mult_range = (0.92, 1.08)
    mu_mult_range      = (0.92, 1.08)
    d_hat_mult_range   = (0.95, 1.05)
    radius_abs_range   = (0.30, 0.50)

    # widest-path graph parameters
    grid_res: int = 12      # interior grid samples per axis
    knn_k: int = 7          # neighbors per node
    nsamp_edge: int = 20    # samples to estimate edge clearance
    min_node_clear: float = 1e-3
    follow_corridor: bool = True      # recenter next stage toward corridor

    # decoupled-search ROI (bigger than next stage)
    roi_pad_scale_x: float = 0.8      # pad ~0.8*W on both sides
    roi_pad_scale_y: float = 0.8      # pad ~0.8*H on both sides
    search_union_current_next: bool = True  # use union(current, next) before padding

    # hard extra margins beyond tube (set small >0 to avoid grazing)
    min_edge_margin: float = 1e-3     # extra beyond inflate along edges
    min_exit_margin: float = 1e-3     # extra beyond inflate at exit

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
    rng = np.random.default_rng()

    # target gap tied to robot size, but clamp so it's always >= tube diameter
    robot_diam = 2.0 * cfg.radius
    gap_mult   = rng.uniform(*cfg.gap_mult_range)
    gap_margin = rng.uniform(*cfg.gap_margin_range)
    target_gap = gap_mult * robot_diam + gap_margin

    # enforce solvability w.r.t. the tube used in planning
    min_gap = 2.0 * cfg.inflate + 0.01  # tiny slack
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


# ==============================
# Widest-path graph helpers
# ==============================

def _sample_rect_boundary(bounds, n_per_edge: int = 18) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, n_per_edge, endpoint=True)
    ys = np.linspace(ymin, ymax, n_per_edge, endpoint=True)
    top    = np.stack([xs, np.full_like(xs, ymax)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, ymin)], axis=1)
    left   = np.stack([np.full_like(ys, xmin), ys], axis=1)
    right  = np.stack([np.full_like(ys, xmax), ys], axis=1)
    B = np.concatenate([top, bottom, left, right], axis=0)
    B = np.unique(B, axis=0)
    return B

def _clearance_points(P: np.ndarray, C: np.ndarray, R: np.ndarray) -> np.ndarray:
    if C.size == 0:
        return np.full((P.shape[0],), np.inf, dtype=float)
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
                         edge_clear_fn) -> Tuple[int, np.ndarray, Dict[str, np.ndarray]]:
    n = P.shape[0]
    best = np.full(n, -np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[entry_idx] = np.inf
    heap = [(-best[entry_idx], entry_idx)]
    visited = np.zeros(n, dtype=bool)
    while heap:
        negb, u = heapq.heappop(heap)
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
    if not target_idx_set:
        return -1, parent, {"best": best}
    tgt_list = list(target_idx_set)
    scores = best[tgt_list]
    if np.all(scores == -np.inf):
        return -1, parent, {"best": best}
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

def _union_bounds(b1, b2):
    x1a,x1b,y1a,y1b = b1; x2a,x2b,y2a,y2b = b2
    return (min(x1a,x2a), max(x1b,x2b), min(y1a,y2a), max(y1b,y2b))

def _pad_bounds(b, px, py):
    xmin,xmax,ymin,ymax = b
    return (xmin - px, xmax + px, ymin - py, ymax + py)

def pick_maximin_boundary_exit_decoupled(
    entry: np.ndarray,
    next_bounds: Tuple[float,float,float,float],   # boundary where exit must lie
    search_bounds: Tuple[float,float,float,float], # larger ROI for graph
    C_inf: np.ndarray,
    R_inf: np.ndarray,
    grid_res: int = 12,
    k: int = 7,
    min_node_clear: float = 1e-3,
    nsamp_edge: int = 20,
    min_exit_clearance: float = 0.0,   # extra beyond tube
    min_edge_clearance: float = 0.0,   # extra beyond tube
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Search in a larger ROI but choose exit on next_bounds; enforce hard clearances."""
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


# ==============================
# Stage manager with widest-path exits (decoupled search)
# ==============================

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

class StageManagerMaximin:
    """Build stages using max–min exits; search over padded union ROI; optionally corridor-follow centers."""
    def __init__(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        stage_size=(2.6, 2.0),
        overlap_ratio=0.3,
        obstacles: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (C, R)
        inflate: float = 0.35,
        graph_cfg: Optional[GenCfg] = None,
    ):
        self.start = np.array(start_xy, float)
        self.goal  = np.array(goal_xy,  float)
        self.stage_size = tuple(stage_size)
        self.overlap_ratio = float(overlap_ratio)

        # store inflated obstacles for planning/search
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

        # initial straight-line centers
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

                # bigger search ROI: union(current, next) padded
                if self.gcfg.search_union_current_next:
                    roi = _union_bounds(cur_bounds, next_bounds)
                else:
                    roi = next_bounds
                search_bounds = _pad_bounds(roi, pad_x, pad_y)

                cand, poly, bottle = pick_maximin_boundary_exit_decoupled(
                    entry=np.array(entry, float),
                    next_bounds=next_bounds,
                    search_bounds=search_bounds,
                    C_inf=self.C, R_inf=self.R,
                    grid_res=self.gcfg.grid_res,
                    k=self.gcfg.knn_k,
                    min_node_clear=self.gcfg.min_node_clear,
                    nsamp_edge=self.gcfg.nsamp_edge,
                    # require exactly the tube plus tiny extra margins
                    min_exit_clearance=self.gcfg.min_exit_margin,
                    min_edge_clearance=self.gcfg.min_edge_margin,
                )

                if cand is not None:
                    exitp = cand
                    # corridor-follow next center toward the discovered path
                    if self.gcfg.follow_corridor and poly is not None and poly.shape[0] >= 2:
                        seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
                        if seg_len.size > 0 and np.sum(seg_len) > 1e-9:
                            s = np.cumsum(seg_len)
                            j = int(np.searchsorted(s, s[-1]*0.33))  # ~first third
                            j = min(max(j, 0), poly.shape[0]-1)
                            desired_center = poly[j]
                            centers[i + 1] = 0.7 * centers[i + 1] + 0.3 * desired_center
                else:
                    # fallback to straight-line fraction
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


# ==============================
# Planner wrapper using StageManagerMaximin
# ==============================

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

        self.stage_field = ssi.StageForceFieldTorch(r_safe=0.35, r_contact=0.18,
                                                    w_goal=1.0, w_radial=2.0, w_tangential=1.2, w_flow=0.8, w_boundary=0.6)

    def stage_slice(self, C,R,W) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        st = self.sm.current(); xmin,xmax,ymin,ymax = st.bounds
        m = (C[:,0] >= xmin-0.5) & (C[:,0] <= xmax+0.5) & (C[:,1] >= ymin-0.5) & (C[:,1] <= ymax+0.5)
        if np.any(m): return C[m],R[m],W[m]
        return C,R,W

    def step(self, dt, world_obs: ssi.WorldObstacles, obs_weighs=None):
        st = self.sm.current()
        C,R,W = self.stage_slice(world_obs.C_np, world_obs.R_np, world_obs.W_np)
        obs_t = ssi.ObstacleProviderTorch(C,R,W, dtype=self.sys.dtype)
        barrier = ssi.IPCBarrier(obs_t, d_hat=world_obs.d_hat)
        if obs_weighs is not None:
            barrier.set_weights(obs_weighs)
        info = self.sys.step(dt, obs_t, barrier, self.stage_field, st.bounds, tuple(st.exit_point))
        self.sm.advance_if_needed(np.array(info["center"],float))
        return info


# ==============================
# Episode / dataset I/O
# ==============================

def perturb_robot_params(cfg: GenCfg) -> Dict[str, float]:
    return {
        "lam_reg": cfg.lam_reg * rand_uniform(*cfg.lam_reg_mult_range),
        "mu":      cfg.mu      * rand_uniform(*cfg.mu_mult_range),
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

def episode_success(cfg: GenCfg, final_center: np.ndarray, min_d_history: List[float]) -> bool:
    goal_dist = np.linalg.norm(np.asarray(final_center, float) - cfg.goal)
    reached = (goal_dist <= cfg.goal_tol)
    collision_free = (len(min_d_history) > 0) and (np.min(min_d_history) > cfg.min_clear_tol)
    return bool(reached and collision_free)

def save_episode_snapshot(
    snapshot_path: str,
    planner: PlannerMaximin,
    frames: list,
    world_obs: ssi.WorldObstacles,
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
    ssi.draw_obstacles(ax, world_obs.C_np, world_obs.R_np, ec="k")
    # overlay inflated keep-out (what the planner respects)
    if cfg.overlay_inflated_in_snapshot:
        ssi.draw_obstacles(ax, world_obs.C_np, world_obs.R_np + cfg.inflate, ec="#00cfdc", alpha=0.35)

    ssi.render_curve(ax, planner.sys.B, start_Pw, "start",  "#1f77b4")
    if mid_Pw is not None:
        ssi.render_curve(ax, planner.sys.B, mid_Pw,   "mid",    "#ff7f0e")
    ssi.render_curve(ax, planner.sys.B, end_Pw,   "end",    "#2ca02c")

    for i, st in enumerate(planner.sm.stages):
        xmin, xmax, ymin, ymax = st.bounds
        ax.add_patch(plt.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, fill=False,
                                   lw=1.6, ec="#00cfdc" if i==planner.sm.current_stage_idx else (0.4,0.6,0.6,0.6)))
        ax.plot([st.entry_point[0]],[st.entry_point[1]],"o",ms=4,color="#85ff85")
        ax.plot([st.exit_point[0]],[st.exit_point[1]],"o",ms=4,color="#ff8585")

    ax.plot([start_xy[0]],[start_xy[1]],"o",color="green",ms=6)
    ax.plot([goal_xy[0]],[goal_xy[1]],"*",color="gold",ms=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Maximin (decoupled ROI) stage exits: start/mid/end")
    ax.set_xlim(-2.5, 10.0); ax.set_ylim(-2.4, 2.4)

    fig.savefig(snapshot_path, bbox_inches="tight")
    plt.close(fig)
    return snapshot_path

def planner_from_cfg(
    cfg: GenCfg,
    world_obs: ssi.WorldObstacles,
    lam_reg: float,
    mu: float,
    d_hat: float,
    radius: float
) -> PlannerMaximin:
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
    with torch.no_grad():
        sys = planner.sys
        P0 = sys.P0_loc
        cur_norm = torch.mean(torch.linalg.norm(P0, dim=-1)).item() + 1e-8
        scale = float(radius / cur_norm)
        sys.P0_loc *= scale
        sys.Ploc   *= scale
    return planner

def run_episode(
    cfg: GenCfg,
    world_obs: ssi.WorldObstacles,
    lam_reg: float,
    mu: float,
    d_hat: float,
    radius: float,
    snapshot_path: str,
) -> Dict[str, Any]:
    planner = planner_from_cfg(cfg, world_obs, lam_reg, mu, d_hat, radius)

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
        "graph": {
            "grid_res": cfg.grid_res,
            "knn_k": cfg.knn_k,
            "nsamp_edge": cfg.nsamp_edge,
            "min_node_clear": cfg.min_node_clear,
            "follow_corridor": bool(cfg.follow_corridor),
            "roi_pad_scale_x": cfg.roi_pad_scale_x,
            "roi_pad_scale_y": cfg.roi_pad_scale_y,
            "search_union_current_next": bool(cfg.search_union_current_next),
            "min_edge_margin": cfg.min_edge_margin,
            "min_exit_margin": cfg.min_exit_margin,
        }
    }

    tw = stage_term_weights(planner.stage_field)

    for t in range(cfg.total_steps):
        info = planner.step(cfg.dt, world_obs)
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
                    "Ploc": sys.Ploc.detach().cpu(),
                    "Vloc": sys.Vloc.detach().cpu(),
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
            "U_reg": float(info["U_reg"]),
            "U_area": float(info["U_area"]),
            "min_d": float(info["min_d"]),
        })

    final_center = planner.sys.o.detach().cpu().numpy()
    final_theta  = float(planner.sys.theta.item())
    success = episode_success(cfg, final_center, min_d_history)

    snap = save_episode_snapshot(snapshot_path, planner, frames, world_obs, cfg.start, cfg.goal, cfg)

    episode = {
        "meta": meta,
        "params": {
            "lam_reg": float(lam_reg),
            "mu": float(mu),
            "d_hat": float(d_hat),
            "radius": float(radius),
            "gamma": float(planner.sys.gamma),
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
        lam_reg, mu, d_hat, radius = cfg.lam_reg, cfg.mu, cfg.d_hat, cfg.radius
        C, R, W = sample_obstacles_case1_tight(cfg)
        world = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)

        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_env_{i:02d}_snapshot.png")
        ep = run_episode(cfg, world, lam_reg, mu, d_hat, radius, snapshot_path=snap_path)
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
    # Use one moderately tight field from the same generator but fixed RNG seed for reproducibility
    rng_state = np.random.get_state()
    np.random.seed(cfg.seed + 777)
    C0, R0, W0 = sample_obstacles_case1_tight(cfg)
    np.random.set_state(rng_state)

    world_fixed = ssi.WorldObstacles(C0, R0, W0, d_hat=cfg.d_hat)
    for i in range(cfg.num_robot_variations):
        p = perturb_robot_params(cfg)
        snap_path = os.path.join(cfg.out_root, "episodes", f"episode_robot_{i:02d}_snapshot.png")
        ep = run_episode(cfg, world_fixed, p["lam_reg"], p["mu"], p["d_hat"], p["radius"], snapshot_path=snap_path)
        path = save_episode(cfg.out_root, "robot", i, ep)
        records.append({
            "case": "robot",
            "idx": i,
            "path": path,
            "success": ep["success"],
            "lam_reg": p["lam_reg"],
            "mu": p["mu"],
            "d_hat": p["d_hat"],
            "radius": p["radius"],
            "snapshot": ep["snapshot_path"],
        })
        print(f"[Case2 robot #{i:02d}] saved: {path} | success={ep['success']} | "
              f"lam={p['lam_reg']:.3f} mu={p['mu']:.3f} d_hat={p['d_hat']:.3f} r={p['radius']:.3f}")

    man_path = write_manifest(cfg.out_root, records)
    print(f"\nWrote manifest: {man_path}")
    print("Done.")


if __name__ == "__main__":
    main()
