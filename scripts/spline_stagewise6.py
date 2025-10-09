# stagewise_hyperelastic_icp_with_stage_field.py
# Hyperelastic ring (uniform scale only) + ICP barrier + stage field
# Produces a snapshot (and optional GIF) showing start/mid/end.

from __future__ import annotations
import os, math, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
import heapq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------- utils ---------------------------

def rot2(theta: torch.Tensor):
    c = torch.cos(theta); s = torch.sin(theta)
    return torch.stack([torch.stack([c,-s],-1), torch.stack([s,c],-1)], -2)

# ---------- small geometry helpers ----------

def _segment_circle_intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, r: float) -> bool:
    ab = b - a
    ac = c - a
    L2 = np.dot(ab, ab) + 1e-12
    t = np.clip(np.dot(ac, ab) / L2, 0.0, 1.0)
    p = a + t * ab
    return np.sum((p - c) ** 2) <= (r * r)

def _los_clear(a: np.ndarray, b: np.ndarray, C: np.ndarray, R: np.ndarray) -> bool:
    if C.size == 0:
        return True
    for (ci, ri) in zip(C, R):
        if _segment_circle_intersects(a, b, ci, ri):
            return False
    return True

def _sample_rect_boundary(bounds, n_per_edge: int = 16) -> np.ndarray:
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

def _pick_visible_boundary_point(
    entry: np.ndarray,
    next_bounds: Tuple[float, float, float, float],
    C_inf: np.ndarray,
    R_inf: np.ndarray,
    prefer_dir: np.ndarray,
    n_per_edge: int = 18,
) -> Optional[np.ndarray]:
    B = _sample_rect_boundary(next_bounds, n_per_edge)
    vis = [b for b in B if _los_clear(entry, b, C_inf, R_inf)]
    if not vis:
        return None
    V = np.array(vis, float)
    vdir = V - entry
    vnorm = np.linalg.norm(vdir, axis=1, keepdims=True) + 1e-12
    vhat = vdir / vnorm
    phat = prefer_dir / (np.linalg.norm(prefer_dir) + 1e-12)
    cosang = (vhat @ phat.reshape(2,))
    dist = np.linalg.norm(V - entry, axis=1)
    order = np.lexsort((dist, -cosang))
    return V[order[0]]

# ---------- graph + overlap helpers (new) ----------

def _rect_overlap(b1, b2):
    x1a,x1b,y1a,y1b = b1; x2a,x2b,y2a,y2b = b2
    xa, xb = max(x1a,x2a), min(x1b,x2b)
    ya, yb = max(y1a,y2a), min(y1b,y2b)
    if xa < xb and ya < yb:
        return (xa, xb, ya, yb)
    return None

def _pad_bounds(b, px, py):
    xmin,xmax,ymin,ymax = b
    return (xmin - px, xmax + px, ymin - py, ymax + py)

def _sample_interior(bounds, grid_res: int = 12) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)

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

def _knn_indices(P: np.ndarray, k: int = 6) -> list[list[int]]:
    n = P.shape[0]
    D2 = ((P[:, None, :] - P[None, :, :])**2).sum(axis=-1)
    np.fill_diagonal(D2, np.inf)
    kk = min(k, n-1)
    idxs = np.argpartition(D2, kth=kk-1, axis=1)[:, :kk]
    return [[int(j) for j in idxs[i]] for i in range(n)]

def _dijkstra_shortest(entry_idx: int, target_set: set, P: np.ndarray, nbrs: list[list[int]], edge_len_fn):
    n = P.shape[0]
    dist = np.full(n, np.inf, float); parent = np.full(n, -1, int)
    dist[entry_idx] = 0.0
    heap = [(0.0, entry_idx)]
    visited = np.zeros(n, dtype=bool)
    while heap:
        du, u = heapq.heappop(heap)
        if visited[u]: continue
        visited[u] = True
        if u in target_set:
            return u, parent, dist
        for v in nbrs[u]:
            if visited[v]: continue
            w = edge_len_fn(u, v)
            if not np.isfinite(w): 
                continue
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return -1, parent, dist

def _reconstruct_path(parent: np.ndarray, tgt: int) -> np.ndarray:
    if tgt < 0: return np.zeros((0,), int)
    path = []
    cur = tgt
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    return np.array(path[::-1], int)

def _pick_exit_overlap_shortest_with_width(
    entry: np.ndarray,
    cur_bounds: tuple[float,float,float,float],
    next_bounds: tuple[float,float,float,float],
    C_inf: np.ndarray, R_inf: np.ndarray,
    tube: float,                         # required minimum clearance (>= inflate)
    edge_margin: float = 1e-3,           # extra beyond tube on edges
    exit_margin: float = 1e-3,           # extra beyond tube at exit node
    grid_res: int = 14,
    knn_k: int = 7,
    nsamp_edge: int = 20,
    roi_pad_scale: tuple[float,float] = (0.8, 0.8),
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Among all points on the *overlap boundary* between current and next frames,
    find a path from 'entry' with minimum length subject to clearance >= tube.
    Returns (exit_point, polyline) in world coordinates, or (None, None).
    """
    # overlap region where exit must lie
    ov = _rect_overlap(cur_bounds, next_bounds)
    if ov is None:
        return None, None

    # nodes: entry + boundary samples (of overlap) + interior samples (ROI)
    # ROI = padded union so we allow slight detours while still local.
    union = (min(cur_bounds[0], next_bounds[0]), max(cur_bounds[1], next_bounds[1]),
             min(cur_bounds[2], next_bounds[2]), max(cur_bounds[3], next_bounds[3]))
    pad_x = roi_pad_scale[0]*(cur_bounds[1]-cur_bounds[0])
    pad_y = roi_pad_scale[1]*(cur_bounds[3]-cur_bounds[2])
    roi = _pad_bounds(union, pad_x, pad_y)

    B_ov = _sample_rect_boundary(ov, n_per_edge=max(18, grid_res))
    I_roi = _sample_interior(roi, grid_res=grid_res)
    P = np.vstack([entry[None, :], B_ov, I_roi])

    # node clearance filter
    node_clr = _clearance_points(P, C_inf, R_inf)
    min_node = max(tube, 1e-4)  # at least tube width
    keep = node_clr > min_node
    keep[0] = True  # always keep entry
    P, node_clr = P[keep], node_clr[keep]
    if P.shape[0] < 3:
        return None, None

    # re-identify indices
    entry_idx = int(np.argmin(np.linalg.norm(P - entry[None, :], axis=1)))
    # overlap boundary mask (recomputed after filtering)
    xmin,xmax,ymin,ymax = ov
    epsb = 1e-6
    on_ov = (np.isclose(P[:,0], xmin, atol=epsb) | np.isclose(P[:,0], xmax, atol=epsb) |
             np.isclose(P[:,1], ymin, atol=epsb) | np.isclose(P[:,1], ymax, atol=epsb))
    # enforce slightly higher margin at exit nodes
    ok_exit = on_ov & (node_clr >= (tube + exit_margin))
    targets = set(np.nonzero(ok_exit)[0].tolist())
    targets.discard(entry_idx)
    if not targets:
        return None, None

    # graph: KNN with LOS; edge feasible if segment clearance >= tube+edge_margin
    nbrs = _knn_indices(P, k=knn_k)
    for u in range(P.shape[0]):
        nbrs[u] = [v for v in nbrs[u] if _los_clear(P[u], P[v], C_inf, R_inf)]

    req_edge = float(tube + edge_margin)
    edge_cache: dict[tuple[int,int], float] = {}
    def edge_len(u, v):
        key = (u, v) if u < v else (v, u)
        if key in edge_cache: return edge_cache[key]
        ce = _segment_clearance(P[u], P[v], C_inf, R_inf, nsamp=nsamp_edge)
        edge_cache[key] = (np.linalg.norm(P[u]-P[v]) if ce >= req_edge else np.inf)
        return edge_cache[key]

    # shortest path to any overlap-boundary node with width >= tube
    tgt_idx, parent, dist = _dijkstra_shortest(entry_idx, targets, P, nbrs, edge_len)
    if tgt_idx < 0 or not np.isfinite(dist[tgt_idx]):
        return None, None

    path_idx = _reconstruct_path(parent, tgt_idx)
    return P[tgt_idx].copy(), P[path_idx].copy()

# ---------- graph + overlap helpers (new) ----------

def _rect_overlap(b1, b2):
    x1a,x1b,y1a,y1b = b1; x2a,x2b,y2a,y2b = b2
    xa, xb = max(x1a,x2a), min(x1b,x2b)
    ya, yb = max(y1a,y2a), min(y1b,y2b)
    if xa < xb and ya < yb:
        return (xa, xb, ya, yb)
    return None

def _pad_bounds(b, px, py):
    xmin,xmax,ymin,ymax = b
    return (xmin - px, xmax + px, ymin - py, ymax + py)

def _sample_interior(bounds, grid_res: int = 12) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)

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

def _knn_indices(P: np.ndarray, k: int = 6) -> list[list[int]]:
    n = P.shape[0]
    D2 = ((P[:, None, :] - P[None, :, :])**2).sum(axis=-1)
    np.fill_diagonal(D2, np.inf)
    kk = min(k, n-1)
    idxs = np.argpartition(D2, kth=kk-1, axis=1)[:, :kk]
    return [[int(j) for j in idxs[i]] for i in range(n)]

def _dijkstra_shortest(entry_idx: int, target_set: set, P: np.ndarray, nbrs: list[list[int]], edge_len_fn):
    n = P.shape[0]
    dist = np.full(n, np.inf, float); parent = np.full(n, -1, int)
    dist[entry_idx] = 0.0
    heap = [(0.0, entry_idx)]
    visited = np.zeros(n, dtype=bool)
    while heap:
        du, u = heapq.heappop(heap)
        if visited[u]: continue
        visited[u] = True
        if u in target_set:
            return u, parent, dist
        for v in nbrs[u]:
            if visited[v]: continue
            w = edge_len_fn(u, v)
            if not np.isfinite(w): 
                continue
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return -1, parent, dist

def _reconstruct_path(parent: np.ndarray, tgt: int) -> np.ndarray:
    if tgt < 0: return np.zeros((0,), int)
    path = []
    cur = tgt
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    return np.array(path[::-1], int)

def _pick_exit_overlap_shortest_with_width(
    entry: np.ndarray,
    cur_bounds: tuple[float,float,float,float],
    next_bounds: tuple[float,float,float,float],
    C_inf: np.ndarray, R_inf: np.ndarray,
    tube: float,                         # required minimum clearance (>= inflate)
    edge_margin: float = 1e-3,           # extra beyond tube on edges
    exit_margin: float = 1e-3,           # extra beyond tube at exit node
    grid_res: int = 16,
    knn_k: int = 7,
    nsamp_edge: int = 20,
    roi_pad_scale: tuple[float,float] = (0.8, 0.8),
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Among all points on the *overlap boundary* between current and next frames,
    find a path from 'entry' with minimum length subject to clearance >= tube.
    Returns (exit_point, polyline) in world coordinates, or (None, None).
    """
    # overlap region where exit must lie
    ov = _rect_overlap(cur_bounds, next_bounds)
    if ov is None:
        return None, None

    # nodes: entry + boundary samples (of overlap) + interior samples (ROI)
    # ROI = padded union so we allow slight detours while still local.
    union = (min(cur_bounds[0], next_bounds[0]), max(cur_bounds[1], next_bounds[1]),
             min(cur_bounds[2], next_bounds[2]), max(cur_bounds[3], next_bounds[3]))
    pad_x = roi_pad_scale[0]*(cur_bounds[1]-cur_bounds[0])
    pad_y = roi_pad_scale[1]*(cur_bounds[3]-cur_bounds[2])
    roi = _pad_bounds(union, pad_x, pad_y)

    B_ov = _sample_rect_boundary(ov, n_per_edge=max(18, grid_res))
    I_roi = _sample_interior(roi, grid_res=grid_res)
    P = np.vstack([entry[None, :], B_ov, I_roi])

    # node clearance filter
    node_clr = _clearance_points(P, C_inf, R_inf)
    min_node = max(tube, 1e-4)  # at least tube width
    keep = node_clr > min_node
    keep[0] = True  # always keep entry
    P, node_clr = P[keep], node_clr[keep]
    if P.shape[0] < 3:
        return None, None

    # re-identify indices
    entry_idx = int(np.argmin(np.linalg.norm(P - entry[None, :], axis=1)))
    # overlap boundary mask (recomputed after filtering)
    xmin,xmax,ymin,ymax = ov
    epsb = 1e-6
    on_ov = (np.isclose(P[:,0], xmin, atol=epsb) | np.isclose(P[:,0], xmax, atol=epsb) |
             np.isclose(P[:,1], ymin, atol=epsb) | np.isclose(P[:,1], ymax, atol=epsb))
    # enforce slightly higher margin at exit nodes
    ok_exit = on_ov & (node_clr >= (tube + exit_margin))
    targets = set(np.nonzero(ok_exit)[0].tolist())
    targets.discard(entry_idx)
    if not targets:
        return None, None

    # graph: KNN with LOS; edge feasible if segment clearance >= tube+edge_margin
    nbrs = _knn_indices(P, k=knn_k)
    for u in range(P.shape[0]):
        nbrs[u] = [v for v in nbrs[u] if _los_clear(P[u], P[v], C_inf, R_inf)]

    req_edge = float(tube + edge_margin)
    edge_cache: dict[tuple[int,int], float] = {}
    def edge_len(u, v):
        key = (u, v) if u < v else (v, u)
        if key in edge_cache: return edge_cache[key]
        ce = _segment_clearance(P[u], P[v], C_inf, R_inf, nsamp=nsamp_edge)
        edge_cache[key] = (np.linalg.norm(P[u]-P[v]) if ce >= req_edge else np.inf)
        return edge_cache[key]

    # shortest path to any overlap-boundary node with width >= tube
    tgt_idx, parent, dist = _dijkstra_shortest(entry_idx, targets, P, nbrs, edge_len)
    if tgt_idx < 0 or not np.isfinite(dist[tgt_idx]):
        return None, None

    path_idx = _reconstruct_path(parent, tgt_idx)
    return P[tgt_idx].copy(), P[path_idx].copy()

# ---------- Stage + StageManager ----------

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

class StageManager:
    def __init__(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        stage_size=(2.6, 2.0),
        overlap_ratio=0.3,
        obstacles: Optional[Tuple[np.ndarray, np.ndarray]] = None,   # (C, R)
        inflate: float = 0.4,
        n_boundary_samples: int = 18,
    ):
        self.start = np.array(start_xy, float)
        self.goal  = np.array(goal_xy,  float)
        self.stage_size = tuple(stage_size)
        self.overlap_ratio = float(overlap_ratio)
        self.inflate = float(inflate)
        if obstacles is None:
            self.C = np.zeros((0, 2), float)
            self.R = np.zeros((0,), float)
        else:
            C, R = obstacles
            self.C = np.array(C, float)
            self.R = np.array(R, float) + self.inflate
        self._n_boundary_samples = int(n_boundary_samples)
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

        W, H = self.stage_size
        step = W * (1.0 - self.overlap_ratio)
        n = max(1, int(np.ceil(L / step)))
        path_dir = d / (L + 1e-12)

        # initial straight-line centers
        centers = []
        for i in range(n):
            t = 0.0 if n == 1 else i / (n - 1)
            centers.append(self.start + t * d)

        for i in range(n):
            c_i = centers[i]
            cur_bounds = (c_i[0] - W/2, c_i[0] + W/2, c_i[1] - H/2, c_i[1] + H/2)
            entry = self.start if i == 0 else self.stages[-1].exit_point

            if i == n - 1:
                exitp = self.goal
            else:
                nxt_center = centers[i + 1]
                next_bounds = (nxt_center[0] - W/2, nxt_center[0] + W/2,
                               nxt_center[1] - H/2, nxt_center[1] + H/2)

                # primary: shortest path with min width (tube = inflate)
                cand, poly = _pick_exit_overlap_shortest_with_width(
                    entry=np.array(entry, float),
                    cur_bounds=cur_bounds,
                    next_bounds=next_bounds,
                    C_inf=self.C, R_inf=self.R,
                    tube=self.inflate,
                    edge_margin=1e-3, exit_margin=1e-3,
                    grid_res=16, knn_k=7, nsamp_edge=24,
                    roi_pad_scale=(0.8, 0.8),
                )

                # use the planner's tube = inflation (what you already store)
                if cand is None:
                    cand, poly = _pick_exit_overlap_shortest_with_width(
                        entry=np.array(entry, float),
                        cur_bounds=cur_bounds,
                        next_bounds=next_bounds,
                        C_inf=self.C, R_inf=self.R,
                        tube=0.0 if self.R.size == 0 else 0.0 + 0.0,   # no obstacles -> no constraint
                        edge_margin=1e-3, exit_margin=1e-3,
                        grid_res=16, knn_k=7, nsamp_edge=24,
                        roi_pad_scale=(0.8, 0.8),
                    )

                if cand is not None:
                    exitp = cand
                else:
                    # fallback: visible boundary on next frame
                    exit_nom = self.start + (i + 1) / n * d
                    cand2 = _pick_visible_boundary_point(
                        entry=np.array(entry, float),
                        next_bounds=next_bounds,
                        C_inf=self.C, R_inf=self.R,
                        prefer_dir=path_dir,
                        n_per_edge=22,
                    )
                    exitp = cand2 if cand2 is not None else exit_nom

            self.stages.append(Stage(c_i, self.stage_size,
                                     np.array(entry, float),
                                     np.array(exitp, float), i))

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

# --------------------------- obstacles & ICP (torch) ---------------------------

class ObstacleProviderTorch:
    def __init__(self, centers, radii, weights=None, device="cpu", dtype=torch.float32, eps=1e-12):
        self.C = torch.as_tensor(centers, dtype=dtype, device=device)  # (C,2)
        self.R = torch.as_tensor(radii,   dtype=dtype, device=device)  # (C,)
        if weights is None: weights = torch.ones_like(self.R)
        self.W = torch.as_tensor(weights, dtype=dtype, device=device)
        self.eps = float(eps)
    def compute_all(self, q: torch.Tensor):
        if self.C.shape[0]==0:
            return q.new_zeros(q.shape[0], 0), q.new_zeros(q.shape[0],0, 2)
        diff = q.unsqueeze(-2) - self.C              # (...,C,2)
        r = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-12)
        d = r.squeeze(-1) - self.R                   # (...,C)
        g = diff / r                                  # (...,C,2)
        return d, g

class IPCBarrier:
    def __init__(self, obstacles: ObstacleProviderTorch, d_hat=1.0, violation_penalty=-5e2,
                 eps=1e-9, max_grad=200.0, max_b=200.0):
        self.obs = obstacles
        self.d_hat = float(d_hat)
        self.vp = float(violation_penalty)
        self.eps = float(eps)
        self.max_grad = float(max_grad)
        self.max_b = float(max_b)

    def _piecewise(self, d: torch.Tensor):
        # scalar d_hat (as before)
        dh = d.new_tensor(self.d_hat)
        safe = torch.clamp(d, min=self.eps)
        b_in = -(d - dh) ** 2 * torch.log(safe / dh)
        dbdd_in = (dh - d) * (2.0 * torch.log(safe / dh) - dh / safe) + 1.0

        b = torch.where(d <= self.eps, self.vp, torch.where(d < dh, b_in, torch.zeros_like(d)))
        dbdd = torch.where(d <= self.eps, self.vp, torch.where(d < dh, dbdd_in, torch.zeros_like(d)))

        b = torch.clamp(b, 0.0, self.max_b)
        dbdd = torch.clamp(dbdd, -self.max_grad, self.max_grad)
        return b, dbdd

    def barrier_and_grad(self, q: torch.Tensor):
        """
        Uses per-obstacle weights W from the obstacle provider:
            b_total = sum_j W_j * b_j
            g_total = sum_j W_j * (db/dd)_j * n_j
        Backward-compatible: if all W_j == 1, behavior is unchanged.
        """
        d, g = self.obs.compute_all(q)             # d: (K,C), g: (K,C,2)
        if d.shape[1] == 0:
            # No obstacles => no force, no barrier
            return q.new_zeros(q.shape[0]), q.new_zeros(q.shape)

        b, dbdd = self._piecewise(d)               # (K,C), (K,C)
        # Broadcast W: (C,) -> (K,C)
        W = self.obs.W.to(device=d.device, dtype=d.dtype)
        if W.ndim == 1:
            W = W.unsqueeze(0).expand_as(d)
        # Weighted sums
        b_sum = (W * b).sum(dim=-1)                # (K,)
        g_sum = ((W * dbdd).unsqueeze(-1) * g).sum(dim=-2)  # (K,2)
        return b_sum, g_sum


# --------------------------- periodic cubic B-spline ---------------------------

def periodic_cubic_uniform_matrices(n_ctrl:int, n_samples:int, device="cpu", dtype=torch.float32):
    K=n_samples
    t_vals = torch.linspace(0.0, float(n_ctrl), K+1, device=device, dtype=dtype)[:-1]
    k0 = torch.floor(t_vals).long(); u=(t_vals-k0.to(dtype)); u2=u*u; u3=u2*u
    w_m1=(1-3*u+3*u2-u3)/6.0; w_0=(4-6*u2+3*u3)/6.0; w_p1=(1+3*u+3*u2-3*u3)/6.0; w_p2=u3/6.0
    dw_m1=(-3+6*u-3*u2)/6.0; dw_0=(-12*u+9*u2)/6.0; dw_p1=(3+6*u-9*u2)/6.0; dw_p2=(3*u2)/6.0
    B=torch.zeros(K,n_ctrl,device=device,dtype=dtype); D=torch.zeros_like(B)
    i_m1=(k0-1)%n_ctrl; i_0=k0%n_ctrl; i_p1=(k0+1)%n_ctrl; i_p2=(k0+2)%n_ctrl; ar=torch.arange(K,device=device)
    for idx,wt,dw in [(i_m1,w_m1,dw_m1),(i_0,w_0,dw_0),(i_p1,w_p1,dw_p1),(i_p2,w_p2,dw_p2)]:
        B[ar,idx]+=wt; D[ar,idx]+=dw
    return B,D

# ---- polygon area and gradient wrt samples ----
def polygon_area_and_grad(X: torch.Tensor):
    x,y = X[:,0], X[:,1]
    x_next = torch.roll(x,-1,0); y_next = torch.roll(y,-1,0)
    A = 0.5*torch.sum(x*y_next - y*x_next)
    y_prev = torch.roll(y,1,0); x_prev = torch.roll(x,1,0)
    dA_dx = 0.5*(y_next - y_prev); dA_dy = 0.5*(x_prev - x_next)
    return A, torch.stack([dA_dx,dA_dy],-1)

# --------------------------- Stage force field (torch) ---------------------------

class StageForceFieldTorch:
    def __init__(self, r_safe=0.4, r_contact=0.2,
                 w_goal=1.0, w_radial=2.0, w_tangential=1.0,
                 w_flow=0.8, w_boundary=0.6):
        self.r_safe=float(r_safe); self.r_contact=float(r_contact)
        self.w_goal=float(w_goal); self.w_radial=float(w_radial)
        self.w_tangential=float(w_tangential); self.w_flow=float(w_flow)
        self.w_boundary=float(w_boundary)
        self.J = torch.tensor([[0.,-1.],[1.,0.]], dtype=torch.float32)

    def _goal(self, X, target):
        v = target.unsqueeze(0) - X
        n = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
        return v / n

    def _radial(self, X, C, R):
        if C.numel() == 0: return torch.zeros_like(X)
        diff = X.unsqueeze(1) - C.unsqueeze(0)
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-8)
        d = dist - R.unsqueeze(0)
        contact = (d <= self.r_contact).unsqueeze(-1)
        strong = (diff / dist.unsqueeze(-1)) * contact * 1e4
        d_ref = X.new_tensor(self.r_safe)
        in_band = ((d > self.r_contact) & (d < d_ref)).unsqueeze(-1)
        db_dd = -(2*(d - d_ref)*torch.log(d.clamp_min(1e-8)/d_ref)
                 + (d - d_ref)**2 / d.clamp_min(1e-8))
        grad = db_dd.unsqueeze(-1) * (diff / dist.unsqueeze(-1))
        grad = torch.where(in_band, grad, torch.zeros_like(grad))
        return (strong + grad).sum(dim=1)

    def _tangent(self, X, target, C, R):
        if C.numel() == 0: return torch.zeros_like(X)
        K, Cn = X.shape[0], C.shape[0]
        diff = X.unsqueeze(1) - C.unsqueeze(0)
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-8)
        d = dist - R.unsqueeze(0)
        n_hat = diff / dist.unsqueeze(-1)
        J = self.J.to(device=X.device, dtype=X.dtype)
        t_hat = (J @ n_hat.transpose(-1, -2)).transpose(-1, -2)
        gdir = (target.unsqueeze(0) - X)
        gdir = gdir / torch.linalg.norm(gdir, dim=1, keepdim=True).clamp_min(1e-8)
        gdir = gdir.unsqueeze(1).expand(-1, Cn, -1)
        sign = torch.sign((t_hat * gdir).sum(dim=-1, keepdim=True))
        inner = torch.clamp(self.r_safe - d, min=0.0) / self.r_safe
        strength = torch.where(d < self.r_safe, inner, X.new_tensor(0.1)).unsqueeze(-1)
        influence = (d < (self.r_safe + 0.5)).unsqueeze(-1).float()
        F = sign * strength * t_hat * influence
        return F.sum(dim=1)

    def _flow(self, X, target, C, R):
        if C.numel() == 0: return torch.zeros_like(X)
        K, Cn = X.shape[0], C.shape[0]
        diff = X.unsqueeze(1) - C.unsqueeze(0)
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-8)
        d = dist - R.unsqueeze(0)
        n_hat = diff / dist.unsqueeze(-1)
        J = self.J.to(device=X.device, dtype=X.dtype)
        t_hat = (J @ n_hat.transpose(-1, -2)).transpose(-1, -2)
        obs_to_goal = (target - C)
        goal_unit = obs_to_goal / torch.linalg.norm(obs_to_goal, dim=-1, keepdim=True).clamp_min(1e-8)
        goal_unit = goal_unit.unsqueeze(0).expand(K, -1, -1)
        sign = torch.sign((t_hat * goal_unit).sum(dim=-1, keepdim=True))
        flow_dir = sign * t_hat
        far = (d > self.r_safe).float().unsqueeze(-1)
        radial_comp = goal_unit
        flow_dir = 0.7*flow_dir + 0.3*far*radial_comp
        flow_strength = torch.where(d < self.r_contact, X.new_tensor(1.0),
                             torch.where(d < self.r_safe, X.new_tensor(0.8),
                                         torch.clamp(0.3*(1.0-(d-self.r_safe)), min=0.0)))
        F = flow_dir * flow_strength.unsqueeze(-1)    # (K,C,2)
        return F.sum(dim=1)                           # (K,2)

    def _boundary(self, X, bounds):
        xmin,xmax,ymin,ymax = [X.new_tensor(v) for v in bounds]
        m = X.new_tensor(0.5)
        F = torch.zeros_like(X)
        left  = X[:,0] < (xmin+m); right = X[:,0] > (xmax-m)
        bot   = X[:,1] < (ymin+m); top   = X[:,1] > (ymax-m)
        F[left,0]  += ((xmin+m) - X[left,0]) / m * 2.0
        F[right,0] += (-(X[right,0] - (xmax-m)) / m) * 2.0
        F[bot,1]   += ((ymin+m) - X[bot,1]) / m * 2.0
        F[top,1]   += (-(X[top,1] - (ymax-m)) / m) * 2.0
        return F
    
    def _goal_rimon_koditschek(self, X, bounds, target, C, R, kappa=4):
        v = target.unsqueeze(0) - X
        n = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
        if C.numel() == 0: return v / n, 0.0

        xmin,xmax,ymin,ymax = bounds
        L = (xmax + ymax - xmin - ymin) / 2
        # betas
        gamma = n / L*L
        x_ = X.unsqueeze(-2)                            # (...,1,2)
        dxc = x_ - C                              # (...,M,2)
        #print(dxc.shape)
        beta_i = ((dxc*dxc).sum(-1) - R*R) / L*L       # (...,M)
        beta_i = torch.clamp(beta_i, min=1e-8)
        #print(beta_i.shape, beta_i.min(), beta_i.max())
        beta   = beta_i.prod(dim=-1, keepdim=True)      # (...,1)

        denom = (gamma.pow(kappa) + beta).clamp_min(1e-8)
        phi   = (gamma / denom.pow(1.0/kappa)).clamp(0.0, 1.0-1e-6)

        # gradients
        nv = -2.0 * v                               # (...,2)
        nbeta  = 2.0 * beta * (dxc / beta_i.unsqueeze(-1)).sum(dim=-2)  # (...,2)
        grad = (beta * nv - (gamma / kappa) * nbeta) / denom.pow(1.0/kappa + 1.0)
        #print(denom.max(), beta.max())
        return 0.5 * v / n - 0.5 * grad, phi
        

    def field(self, X, target, bounds, C, R):
        Fg = self._goal(X, target)
        #Fg, Ug = self._goal_rimon_koditschek(X, bounds, target, C, R)

        # clearance gate (scalar in [0.2, 1.0])
        # if C.numel() > 0:
        #     d = torch.linalg.norm(X.unsqueeze(1)-C.unsqueeze(0), dim=-1) - R.unsqueeze(0)  # (K,C)
        #     min_clr = d.min(dim=1).values.clamp(min=0.0)
        #     gate = (0.2 + 0.8 * (min_clr / (self.r_safe + 1e-6)).clamp(max=1.0)).unsqueeze(-1)
        #     F = F * gate

        return Fg

# --------------------------- Hyperelastic ring system (scale-only) ---------------------------

class HyperelasticRingSystem:
    """
    Single internal DoF: uniform scale s(t). Rigid (o, theta).
    """
    def __init__(self, n_ctrl=20, K=220, device="cpu", dtype=torch.float32,
                 d_hat=1.0,
                 # hyperelastic/bulk:
                 k_bulk=1.5,         # bulk modulus (area resistance)
                 gamma_s=2.0,        # scale damping
                 M_s=1.0,            # "mass" for scale dynamics
                 # rigid:
                 M_o=1.5, gamma_o=4.0,
                 I=0.6,  gamma_th=1.6,
                 seed=7, radius=0.35):
        torch.manual_seed(seed); self.device=device; self.dtype=dtype
        self.B,self.D = periodic_cubic_uniform_matrices(n_ctrl,K,device,dtype)
        self.w = torch.full((K,), 1.0/float(K), dtype=dtype, device=device)
        ang = torch.linspace(0,2*math.pi,n_ctrl+1,device=device,dtype=dtype)[:-1]
        self.P0_loc = radius * torch.stack([torch.cos(ang), torch.sin(ang)],-1)  # base shape (unit ring * radius)
        # generalized state
        self.s     = torch.tensor(1.0, dtype=dtype, device=device)     # scale
        self.sdot  = torch.tensor(0.0, dtype=dtype, device=device)
        self.theta = torch.tensor(0.0, dtype=dtype, device=device)
        self.omega = torch.tensor(0.0, dtype=dtype, device=device)
        self.o     = torch.tensor([0.0,0.0], dtype=dtype, device=device)
        self.v_o   = torch.tensor([0.0,0.0], dtype=dtype, device=device)
        self.s_min = 0.3; self.s_max = 3.0
        # params
        self.d_hat=float(d_hat)
        self.k_bulk=float(k_bulk); self.gamma_s=float(gamma_s); self.M_s=float(M_s)
        self.M_o=float(M_o); self.gamma_o=float(gamma_o)
        self.I=float(I); self.gamma_th=float(gamma_th)
        self.J = torch.tensor([[0.,-1.],[1.,0.]], dtype=dtype, device=device)
        # area reference
        with torch.no_grad():
            R0 = rot2(self.theta)
            Pw0 = (self.P0_loc @ R0.T) + self.o
            X0 = self.B @ Pw0
            A0, _ = polygon_area_and_grad(X0)
        self.A_ref = float(torch.abs(A0).item()) + 1e-6   # reference area at s=1

    # ---- kinematics ----
    def world_points(self):
        R = rot2(self.theta)
        return self.o + self.s * (self.P0_loc @ R.T)  # (n_ctrl,2)

    def sample_curve(self):
        Pw = self.world_points()                      # (n_ctrl,2)
        X = self.B @ Pw                                # (K,2)
        Xp = self.D @ Pw                               # (K,2)
        ell = Xp.norm(dim=1).clamp_min(1e-8)
        T_hat = Xp/ell.unsqueeze(-1)
        return Pw,X,Xp,ell,T_hat

    # ---- conservative gradient wrt Pw (same structure as original) ----
    @staticmethod
    def _gradPw_from_barrier_terms(B, D, w, ell, bsum, g_sum, T_hat):
        term1 = (w * ell).unsqueeze(-1) * g_sum      # (K,2)
        term2 = (w * bsum).unsqueeze(-1) * T_hat     # (K,2)
        gradP_world = B.T @ term1 + D.T @ term2      # (n_ctrl,2)
        return gradP_world

    # ---- energy helpers ----
    def _bulk_dU_ds(self, s, min_clear: float):
        # adaptive area target (same idea as your squeeze logic)
        alpha, beta = 0.25, 2.5
        squeeze = alpha + (1 - alpha) * torch.tanh(beta * torch.clamp(torch.as_tensor(min_clear, dtype=self.dtype, device=self.device), min=0.0))
        A_target = squeeze * self.A_ref
        # area scales as s^2 * A_ref
        A = (s**2) * self.A_ref
        dA_ds = 2.0 * s * self.A_ref
        U = 0.5 * self.k_bulk * (A - A_target)**2
        dU_ds = self.k_bulk * (A - A_target) * dA_ds
        return U, dU_ds

    def step(
        self,
        dt: float,
        obstacle_provider: ObstacleProviderTorch,
        barrier: IPCBarrier,
        stage_force: StageForceFieldTorch,
        stage_bounds,               # (xmin, xmax, ymin, ymax)
        target_xy: Tuple[float, float]
    ):
        Pw, X, Xp, ell, T_hat = self.sample_curve()
        w = self.w
        R = rot2(self.theta)
        J = self.J.to(device=X.device, dtype=X.dtype)
        target = torch.as_tensor(target_xy, dtype=self.dtype, device=self.device)

        # --- ICP barrier ---
        bsum, g_sum = barrier.barrier_and_grad(X)                 # (K,), (K,2)
        gradPw_bar = self._gradPw_from_barrier_terms(self.B, self.D, w, ell, bsum, g_sum, T_hat)

        # --- min clearance for adaptivity ---
        dists, _ = obstacle_provider.compute_all(X)
        if dists.shape[1] == 0:
            min_d = 1e5 * torch.ones(dists.shape[0], dtype=self.dtype, device=dists.device)
        else:
            min_d = dists.min(dim=1).values
        min_clear = float(min_d.min().item())

        # --- conservative rigid forces (o, theta) from barrier ---
        X_minus_o = X - self.o.unsqueeze(0)
        term1 = (w * ell).unsqueeze(-1) * g_sum
        term2 = (w * bsum).unsqueeze(-1) * T_hat
        dU_do = torch.sum(term1, dim=0) + torch.sum(term2*0.0, dim=0)  # second term's dℓ/ d o handled via J-term below (same as original torque)
        F_o_cons = -dU_do
        gJx = torch.einsum("kd,dd,kd->k", g_sum, J, X_minus_o)
        dU_dtheta = torch.sum(w * ell * gJx)
        tau_cons = -dU_dtheta

        # --- map grad wrt Pw to scale s (chain rule) ---
        dPw_ds = (self.P0_loc @ R.T)                                # (n_ctrl,2)
        dU_ds_barrier = torch.sum(gradPw_bar * dPw_ds).sum()        # scalar

        # --- friction (dissipative) ---
        contact_mask = (min_d < self.d_hat).to(X.dtype)
        Pdot_world = self.sdot * (self.P0_loc @ R.T) + self.omega * (self.s * (self.P0_loc @ R.T) @ J.T) + self.v_o
        Xdot = self.B @ Pdot_world
        v_t = torch.sum(Xdot * T_hat, dim=1)
        p_k = g_sum.norm(dim=1)
        p_k = p_k.clamp(max=200.0)          # cap normal proxy
        v_t = v_t.clamp(-2.0, 2.0)   # cap tangential speed per sample
        #TODO:make 0.25 be a parameter
        f_fric = -0.1 * (p_k * v_t * contact_mask).unsqueeze(-1) * T_hat  # small μ built-in; tune if needed
        F_world_from_fric = self.B.T @ (w.unsqueeze(-1) * f_fric)
        F_o_from_fric = torch.sum(w.unsqueeze(-1) * f_fric, dim=0)
        tau_fric = torch.sum(w * torch.sum(f_fric * (J @ X_minus_o.T).T, dim=1))
        dU_ds_fric = -torch.sum(F_world_from_fric * dPw_ds).sum()   # virtual work -> generalized force

        # --- stage non-conservative field on samples ---
        F_stage = stage_force.field(X, target, stage_bounds, obstacle_provider.C, obstacle_provider.R)  # (K,2)
        F_world_from_stage = self.B.T @ (w.unsqueeze(-1) * F_stage)
        F_o_from_stage = torch.sum(w.unsqueeze(-1) * F_stage, dim=0)
        tau_stage = torch.sum(w * torch.sum(F_stage * (J @ X_minus_o.T).T, dim=1))
        dU_ds_stage = -torch.sum(F_world_from_stage * dPw_ds).sum()

        # --- hyperelastic bulk (area) ---
        U_bulk, dU_ds_bulk = self._bulk_dU_ds(self.s, min_clear)

        # ---- integrate generalized dynamics ----
        # scale s
        
        F_s_total = -(dU_ds_barrier + 0 * dU_ds_fric + 0 * dU_ds_stage + dU_ds_bulk) - self.gamma_s * self.sdot
        self.sdot = self.sdot + dt * (F_s_total / self.M_s)
        # TODO: make min max to be parameters and record violation.
        self.s    = torch.clamp(self.s + dt * self.sdot, min=self.s_min, max=self.s_max)  # keep sane limits

        # rotation
        tau_total = tau_cons + tau_fric + tau_stage - self.gamma_th * self.omega
        self.omega = self.omega + dt * (tau_total / self.I)
        self.theta = self.theta + dt * self.omega

        # translation
        F_o_total = F_o_cons + F_o_from_fric + F_o_from_stage - self.gamma_o * self.v_o
        if self.v_o[0] ** 2 + self.v_o[1] ** 2 >=5.0:
            print(F_o_cons, F_o_from_fric, F_o_from_stage)
        self.v_o  = self.v_o + dt * (F_o_total / self.M_o)
        self.o    = self.o + dt * self.v_o

        # diagnostics
        # U_barrier = torch.sum(w * bsum * ell)
        
        # with torch.no_grad():
        #     dbg = {
        #         "norm_gradPw_bar": float(gradPw_bar.norm().item()),
        #         "norm_F_stage": float(F_stage.norm(dim=1).mean().item()),
        #         "norm_F_world_stage": float(F_world_from_stage.norm().item()),
        #         "norm_f_fric": float(f_fric.norm(dim=1).mean().item()),
        #         "F_s_total": float(F_s_total.item()),
        #         "tau_total": float(tau_total.item()),
        #         "F_o_total_norm": float(F_o_total.norm().item()),
        #         "min_clear": float(min_d.min().item()),
        #     }

        # return {
        #     "U_barrier": float(U_barrier.item()),
        #     "U_bulk": float(U_bulk.item()),
        #     "center": self.o.detach().cpu().numpy().tolist(),
        #     "theta": float(self.theta.item()),
        #     "scale": float(self.s.item()),
        #     "min_d": float(min_d.min().item()),
        # }

                # diagnostics
        U_barrier = torch.sum(w * bsum * ell)

        with torch.no_grad():
            dbg = {
                "norm_gradPw_bar": float(gradPw_bar.norm().item()),
                "norm_F_stage": float(F_stage.norm(dim=1).mean().item()),
                "norm_F_world_stage": float(F_world_from_stage.norm().item()),
                "norm_f_fric": float(f_fric.norm(dim=1).mean().item()),
                "F_s_total": float(F_s_total.item()),
                "tau_total": float(tau_total.item()),
                "F_o_total_x": float(F_o_total[0].item()),
                "F_o_total_y": float(F_o_total[1].item()),
                "F_o_total_norm": float(F_o_total.norm().item()),
                "min_clear": float(min_d.min().item()),
            }

        return {
            "U_barrier": float(U_barrier.item()),
            "U_bulk": float(U_bulk.item()),
            "center": self.o.detach().cpu().numpy().tolist(),
            "theta": float(self.theta.item()),
            "scale": float(self.s.item()),
            "min_d": float(min_d.min().item()),
            # forces actually used in integration (stagewise)
            "F_s_total": dbg["F_s_total"],
            "tau_total": dbg["tau_total"],
            "F_o_total": [dbg["F_o_total_x"], dbg["F_o_total_y"]],
            # keep dbg if you like
            "dbg": dbg,
        }

# --------------------------- world obstacles holder ---------------------------

class WorldObstacles:
    def __init__(self, centers, radii, weights=None, d_hat=0.28):
        self.C_np=np.asarray(centers,float); self.R_np=np.asarray(radii,float)
        self.W_np=np.asarray(weights if weights is not None else np.ones_like(self.R_np),float)
        self.d_hat=float(d_hat)

# --------------------------- viz helpers ---------------------------

def draw_obstacles(ax, C,R, **kw):
    C=np.asarray(C); R=np.asarray(R)
    for (cx,cy),rr in zip(C,R):
        ax.add_patch(plt.Circle((cx,cy), rr, fill=False, lw=2, **kw))

def render_curve(ax, B, Pw, label=None, color=None):
    Xv=(B @ torch.tensor(Pw)).numpy(); ax.plot(Xv[:,0],Xv[:,1],lw=2,label=label,color=color)

# --------------------------- planner wrapper ---------------------------

class StagewiseHyperelasticPlanner:
    def __init__(self, start_xy, goal_xy, stage_size=(2.5,2.0), overlap=0.3,
                 n_ctrl=20, K=240, d_hat=1.0, seed=7, obstacles=None, inflate=0.35):

        self.sm = StageManager(
            np.array(start_xy), np.array(goal_xy),
            stage_size=stage_size, overlap_ratio=overlap,
            obstacles=obstacles, inflate=inflate
        )

        self.sys = HyperelasticRingSystem(n_ctrl=n_ctrl, K=K, d_hat=d_hat, seed=seed)
        self.initial_timer_counter=0
        # initialize pose
        self.sys.o = torch.tensor(start_xy, dtype=self.sys.dtype)
        self.inflate = float(inflate)
        d = np.array(goal_xy)-np.array(start_xy); self.sys.theta = torch.tensor(math.atan2(d[1],d[0]), dtype=self.sys.dtype)
        self.stage_field = StageForceFieldTorch(r_safe=0.35, r_contact=0.18,
                                                w_goal=1.0, w_radial=2.0, w_tangential=1.2, w_flow=0.8, w_boundary=0.5)
        self.sys.d_hat = torch.tensor(self.inflate, dtype=self.sys.dtype)  # friction/contact gate uses this
        
    # def stage_slice(self, C,R,W) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    #     st = self.sm.current(); 
    #     xmin,xmax,ymin,ymax = st.bounds
    #     m = (C[:,0] >= xmin-0.5) & (C[:,0] <= xmax+0.5) & (C[:,1] >= ymin-0.5) & (C[:,1] <= ymax+0.5)
    #     if np.any(m): return C[m],R[m],W[m]
    #     return [], [], []
    
    def stage_slice(self, C, R, W):
        """Crop obstacles to the current live bounds (keeps old signature)."""
        st = self.sm.current(); 
        b = st.bounds
        if b is None or len(C) == 0:
            return C, R, W
        infl = getattr(self, "boundary_inflation", 0.2)
        x0,x1,y0,y1 = b
        x0 -= infl; x1 += infl; y0 -= infl; y1 += infl

        # circle-vs-AABB test (branchless, vectorized)
        # distance from center to box along each axis (0 if inside slab)
        zero = np.zeros_like(R, dtype=C.dtype)
        dx = np.maximum.reduce([x0 - C[:, 0], zero, C[:, 0] - x1])
        dy = np.maximum.reduce([y0 - C[:, 1], zero, C[:, 1] - y1])

        # intersect iff dx^2 + dy^2 <= r^2  (use < if you don't want "touching" to count)
        m = (dx*dx + dy*dy) <= (R*R)
        return C[m], R[m], W[m] if hasattr(W, "__len__") and len(W)==len(C) else W

    def step(self, dt, world_obs: WorldObstacles):
        st = self.sm.current()
        C,R,W = self.stage_slice(world_obs.C_np, world_obs.R_np, world_obs.W_np)
        if len(R) > 0:
            R_eff = R + self.inflate
        else:
            R_eff = R
        obs_t  = ObstacleProviderTorch(C, R_eff, W, dtype=self.sys.dtype)
        barrier = IPCBarrier(obs_t, d_hat=self.inflate)   # activate barrier at tube distance
        #barrier = IPCBarrier(obs_t, d_hat=world_obs.d_hat)
        self.initial_timer_counter += 1
        if self.initial_timer_counter <= 10:
            info = self.sys.step(0.01 * self.initial_timer_counter * dt, obs_t, barrier, self.stage_field, st.bounds, tuple(st.exit_point))
        else:
            info = self.sys.step(dt, obs_t, barrier, self.stage_field, st.bounds, tuple(st.exit_point))
        advanced = self.sm.advance_if_needed(np.array(info["center"],float))
        if advanced: self.initial_timer_counter=0
        return info

# --------------------------- demo ---------------------------

def run_demo(out_dir="./out_stage_hyperelastic", total_steps=900, dt=0.015,
             stage_size=(2.6,2.0), overlap=0.3, snapshot_every=5):
    os.makedirs(out_dir,exist_ok=True)

    centers=np.array([
        [1.0,0.5],[2.5,-0.8],[3.2,0.3],[4.0,1.0],[4.8,-0.5],
        [5.5,0.8],[6.2,-1.2],[6.8,0.9],[7.5,-0.7],[8.2,0.4],
        [2.0,1.2],[3.5,-1.3],[5.0,1.5],[6.5,-1.1],[7.8,1.2]
    ],float)
    radii=np.array([0.40,0.50,0.30,0.60,0.40,0.50,0.30,0.40,0.50,
                    0.30,0.30,0.40,0.30,0.40,0.30],float)
    weights=np.array([1.0,1.2,0.8,1.5,0.9,1.1,0.7,1.0,1.3,0.8,
                    0.6,1.0,0.7,1.1,0.6],float)

    world_obs=WorldObstacles(centers,radii,weights,d_hat=1.0)
    start=np.array([-1.0,-0.5]); goal=np.array([9.0,0.2])

    planner=StagewiseHyperelasticPlanner(start, goal, stage_size=stage_size, overlap=overlap,
                                         n_ctrl=20, K=240, d_hat=1.0, seed=7,
                                         obstacles=(world_obs.C_np, world_obs.R_np), inflate=0.1)

    frames=[]; mid=None
    for t in range(total_steps):
        planner.step(dt, world_obs)
        if t%snapshot_every==0 or t in (0,total_steps//2,total_steps-1):
            Pw=planner.sys.world_points().detach().cpu().numpy()
            frames.append(Pw)
            if t==total_steps//2: mid=Pw.copy()

    # snapshot
    fig=plt.figure(figsize=(7.6,3.8),dpi=140); ax=fig.add_subplot(111); ax.set_aspect("equal","box")
    draw_obstacles(ax,centers,radii, ec="k")
    if frames: render_curve(ax, planner.sys.B, frames[0], "start", "#1f77b4")
    if mid is not None: render_curve(ax, planner.sys.B, mid, "mid", "#ff7f0e")
    render_curve(ax, planner.sys.B, frames[-1], "end", "#2ca02c")
    for i,st in enumerate(planner.sm.stages):
        xmin,xmax,ymin,ymax=st.bounds
        ax.add_patch(plt.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, fill=False,
                                   lw=1.6, ec="#00cfdc" if i==planner.sm.current_stage_idx else (0.4,0.6,0.6,0.6)))
        ax.plot([st.entry_point[0]],[st.entry_point[1]],"o",ms=4,color="#85ff85")
        ax.plot([st.exit_point[0]],[st.exit_point[1]],"o",ms=4,color="#ff8585")
    ax.plot([start[0]],[start[1]],"o",color="green",ms=6); ax.plot([goal[0]],[goal[1]],"*",color="gold",ms=10)
    ax.legend(loc="upper left",fontsize=8)
    ax.set_title("Stagewise hyperelastic ring: ICP + stage field (goal+radial+tangent+flow) + bulk area")
    ax.set_xlim(-2.5,10.0); ax.set_ylim(-2.4,2.4)
    snap=os.path.join(out_dir,"stagewise_hyperelastic_snapshot.png"); fig.savefig(snap,bbox_inches="tight"); plt.close(fig)

    # gif (optional)
    gif=os.path.join(out_dir,"stagewise_hyperelastic_motion.gif")
    try:
        import imageio
        imgs=[]
        for Pw in frames:
            fig=plt.figure(figsize=(6.4,3.6),dpi=110); ax=fig.add_subplot(111); ax.set_aspect("equal","box")
            draw_obstacles(ax,centers,radii, ec="k"); render_curve(ax, planner.sys.B, Pw)
            ax.set_xlim(-2.5,10.0); ax.set_ylim(-2.4,2.4); ax.set_xticks([]); ax.set_yticks([])
            fig.canvas.draw()
            buf=np.frombuffer(fig.canvas.tostring_argb(),dtype=np.uint8)
            arr=buf.reshape(fig.canvas.get_width_height()[::-1]+(4,))[:,:,1:]; imgs.append(arr); plt.close(fig)
        imageio.mimsave(gif, imgs, fps=max(6,int(1.0/dt)))
    except Exception:
        gif=None

    return {"snapshot":snap, "gif":gif, "stages":len(planner.sm.stages),
            "final_center":planner.sys.o.detach().cpu().numpy().tolist(),
            "final_theta":float(planner.sys.theta.item()),
            "final_scale":float(planner.sys.s.item())}

# --------------------------- CLI ---------------------------

def parse_args():
    ap=argparse.ArgumentParser("Stagewise hyperelastic ring with stage field")
    ap.add_argument("--out",type=str,default="./out_stage_hyperelastic")
    ap.add_argument("--steps",type=int,default=3000)
    ap.add_argument("--dt",type=float,default=0.04)
    ap.add_argument("--stage_w",type=float,default=2.6)
    ap.add_argument("--stage_h",type=float,default=2.0)
    ap.add_argument("--overlap",type=float,default=0.3)
    return ap.parse_args()

if __name__=="__main__":
    args=parse_args()
    out=run_demo(out_dir=args.out, total_steps=args.steps, dt=args.dt,
                 stage_size=(args.stage_w,args.stage_h), overlap=args.overlap)
    print("Outputs:"); print(out)
