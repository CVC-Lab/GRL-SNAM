import numpy as np
import math, heapq, inspect
from collections import deque

def _inside_rect(p, b):
    return (b[0] <= p[0] <= b[1]) and (b[2] <= p[1] <= b[3])

def _grid_from_bounds(bounds, grid_res=(64, 40)):
    x0,x1,y0,y1 = bounds
    nx = max(12, int(grid_res[0])); ny = max(12, int(grid_res[1]))
    xs = np.linspace(x0, x1, nx, dtype=float)
    ys = np.linspace(y0, y1, ny, dtype=float)
    return xs, ys, nx, ny

def _inflate_mask(xs, ys, C, R, inflate, dilate_cells=1):
    """True = free, False = blocked. Also adds a cheap cell-dilation near obstacles."""
    ny, nx = len(ys), len(xs)
    if len(C) == 0:
        free = np.ones((ny, nx), dtype=bool)
    else:
        X, Y = np.meshgrid(xs, ys)              # (ny,nx) each
        rr = (R + float(inflate)).astype(float) # per obstacle
        free = np.ones((ny, nx), dtype=bool)
        for ci, ri in zip(C, rr):
            if ri <= 0: 
                continue
            d2 = (X - ci[0])**2 + (Y - ci[1])**2
            free &= (d2 >= ri*ri)
    if dilate_cells > 0:
        # 3x3 max-pool on occupied to erode the free set a bit (keeps a buffer)
        occ = (~free).astype(np.uint8)
        pad = np.pad(occ, 1, mode="edge")
        for _ in range(int(dilate_cells)):
            grow = np.zeros_like(occ)
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    grow = np.maximum(grow, pad[1+di:1+di+occ.shape[0], 1+dj:1+dj+occ.shape[1]])
            free &= (grow == 0)
            pad = np.pad((~free).astype(np.uint8), 1, mode="edge")
    return free

def _boundary_candidates(free):
    """Return boundary indices that are free AND have a free neighbor inside."""
    ny, nx = free.shape
    cand = []
    for i in range(ny):
        for j in (0, nx-1):
            if free[i,j] and ((j==0 and free[i,min(j+1,nx-1)]) or (j==nx-1 and free[i,max(j-1,0)])):
                cand.append((i,j))
    for j in range(nx):
        for i in (0, ny-1):
            if free[i,j] and ((i==0 and free[min(i+1,ny-1),j]) or (i==ny-1 and free[max(i-1,0),j])):
                cand.append((i,j))
    return list(set(cand))


def _closest_index(xs, ys, p):
    j = np.searchsorted(xs, p[0]) - 1
    i = np.searchsorted(ys, p[1]) - 1
    j = int(np.clip(j, 0, len(xs)-1))
    i = int(np.clip(i, 0, len(ys)-1))
    # snap to truly closest among neighbors
    best = (i, j); bd = float("inf")
    for di in (-1,0,1):
        for dj in (-1,0,1):
            ii = int(np.clip(i+di, 0, len(ys)-1))
            jj = int(np.clip(j+dj, 0, len(xs)-1))
            d = (xs[jj] - p[0])**2 + (ys[ii] - p[1])**2
            if d < bd:
                bd, best = d, (ii, jj)
    return best  # (i,row,y) first, (j,col,x)

def _boundary_indices(ny, nx):
    idxs = []
    for i in range(ny):
        idxs.append((i,0)); idxs.append((i,nx-1))
    for j in range(nx):
        idxs.append((0,j)); idxs.append((ny-1,j))
    return list(set(idxs))

def _neighbors_8(i,j, ny, nx):
    for di in (-1,0,1):
        for dj in (-1,0,1):
            if di==0 and dj==0: continue
            ii, jj = i+di, j+dj
            if 0 <= ii < ny and 0 <= jj < nx:
                w = math.sqrt(2.0) if di!=0 and dj!=0 else 1.0
                yield ii, jj, w

def _dijkstra_grid(free, weights, start_ij, goals_set):
    """Fallback Dijkstra on grid with 8-connectivity."""
    ny, nx = free.shape
    INF = 1e30
    dist = np.full((ny,nx), INF, dtype=float)
    prev = np.full((ny,nx,2), -1, dtype=int)
    i0,j0 = start_ij
    if not free[i0,j0]:
        # pick nearest free neighbor as start
        best, bd = None, float("inf")
        for i in range(ny):
            for j in range(nx):
                if free[i,j]:
                    d = (i-i0)**2 + (j-j0)**2
                    if d<bd: bd, best = d, (i,j)
        if best is None: return None
        i0,j0 = best
    dist[i0,j0] = 0.0
    h = [(0.0, i0, j0)]
    goal_hit = None
    while h:
        d,i,j = heapq.heappop(h)
        if d != dist[i,j]: 
            continue
        if (i,j) in goals_set:
            goal_hit = (i,j); break
        for ii,jj,w in _neighbors_8(i,j, ny, nx):
            if not free[ii,jj]: 
                continue
            nd = d + 0.5*(weights[i,j] + weights[ii,jj]) * w
            if nd < dist[ii,jj]:
                dist[ii,jj] = nd
                prev[ii,jj] = (i,j)
                heapq.heappush(h, (nd, ii, jj))
    if goal_hit is None:
        return None
    # reconstruct path
    path = []
    cur = goal_hit
    while cur != (-1,-1):
        path.append(cur)
        pi, pj = prev[cur]
        if pi < 0: break
        cur = (pi, pj)
    path.reverse()
    return path  # list[(i,j)]

def _edge_weight_field(xs, ys, free):
    """Optional bias away from obstacles: base=1, slight penalty near occupied."""
    ny, nx = free.shape
    w = np.ones((ny,nx), dtype=float)
    # 3x3 min-pooling on inverted free to mark near-occupied as costly
    inv = (~free).astype(np.float32)
    pad = np.pad(inv, 1, mode="edge")
    near = np.zeros_like(inv)
    for di in (-1,0,1):
        for dj in (-1,0,1):
            near = np.maximum(near, pad[1+di:1+di+inv.shape[0], 1+dj:1+dj+inv.shape[1]])
    w += 0.15*near  # light repulsion
    return w

def _snap_to_free_edge(exit_xy, bounds, xs, ys, free, max_slide_cells=6):
    """Slide along the closest edge to the nearest free boundary cell (if needed)."""
    x0,x1,y0,y1 = bounds
    # pick the closest edge
    dleft  = abs(exit_xy[0]-x0); dright = abs(x1-exit_xy[0])
    dbot   = abs(exit_xy[1]-y0); dtop   = abs(y1-exit_xy[1])
    edge = np.argmin([dleft, dright, dbot, dtop])  # 0=L,1=R,2=B,3=T

    # indices for search along that edge
    if edge in (0,1):     # vertical edge: vary i (rows/ys), fix j (col/xs)
        j = 0 if edge==0 else len(xs)-1
        best = None; bd = 1e9
        for di in range(-max_slide_cells, max_slide_cells+1):
            # candidate rows around the closest row
            i_center = np.searchsorted(ys, exit_xy[1])
            i = int(np.clip(i_center+di, 0, len(ys)-1))
            if free[i,j]:
                xy = np.array([xs[j], ys[i]], float)
                d  = np.hypot(*(xy-exit_xy))
                if d<bd:
                    bd, best = d, xy
        if best is not None:
            # tiny inwards epsilon to avoid sitting exactly on edge
            eps = 1e-4
            best[0] = np.clip(best[0], x0+eps, x1-eps)
            best[1] = np.clip(best[1], y0+eps, y1-eps)
            return best
    else:                 # horizontal edge: vary j, fix i
        i = 0 if edge==2 else len(ys)-1
        best = None; bd = 1e9
        for dj in range(-max_slide_cells, max_slide_cells+1):
            j_center = np.searchsorted(xs, exit_xy[0])
            j = int(np.clip(j_center+dj, 0, len(xs)-1))
            if free[i,j]:
                xy = np.array([xs[j], ys[i]], float)
                d  = np.hypot(*(xy-exit_xy))
                if d<bd:
                    bd, best = d, xy
        if best is not None:
            eps = 1e-4
            best[0] = np.clip(best[0], x0+eps, x1-eps)
            best[1] = np.clip(best[1], y0+eps, y1-eps)
            return best
    # fall back: return the original (will still be clipped by caller)
    return exit_xy

def _clearance_field(xs, ys, C, R, inflate):
    """
    Returns clearance (meters) for each grid cell: distance to nearest obstacle
    after applying 'inflate'. Shape: (ny, nx). If no obstacles: large constant.
    """
    ny, nx = len(ys), len(xs)
    if len(C) == 0:
        return np.full((ny, nx), 1e6, dtype=float)  # effectively 'infinite' clearance
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    # vectorized min distance to any inflated disk
    clr = np.full((ny, nx), np.inf, dtype=float)
    for (cx, cy), r in zip(C, R):
        rr = float(r) + float(inflate)
        d = np.hypot(X - cx, Y - cy) - rr
        clr = np.minimum(clr, d)
    # no negative clearance inside keep-out: clamp to small epsilon
    return np.maximum(clr, 1e-6)

def _clearance_weight_field(xs, ys, free, C, R, inflate,
                            goal_xy=None, goal_pull=0.0,
                            gamma=1.2, p=1.5, eps=1e-3):
    """
    Clearance-aware weights:
      cost ≈ base + gamma / (eps + clearance)^p  + goal_pull * goal_bias
    - 'gamma' controls how aggressively we avoid tight spaces
    - 'p' sets how sharply cost grows near walls (1.0–2.0 is typical)
    - 'goal_pull' (0..0.6) gently attracts paths toward goal
    """
    ny, nx = free.shape
    W = np.ones((ny, nx), dtype=float)  # base cost
    # clearance penalty
    clr = _clearance_field(xs, ys, C, R, inflate)
    W = W + gamma / np.power(eps + clr, p)

    # optional goal bias (lower cost near the goal)
    if goal_xy is not None and goal_pull > 0:
        X, Y = np.meshgrid(xs, ys)
        G = np.hypot(X - goal_xy[0], Y - goal_xy[1])
        G = (G - G.min()) / (G.max() - G.min() + 1e-12)  # [0,1]
        W = W + goal_pull * G

    # make occupied cells unusable regardless of weight
    W[~free] = np.inf
    return W

def _bfs_component(free, start_ij, connectivity=8):
    ny, nx = free.shape
    si,sj = start_ij
    if not free[si,sj]:
        return np.zeros_like(free, dtype=bool)  # nothing reachable
    vis = np.zeros_like(free, dtype=bool)
    q = deque([(si,sj)])
    vis[si,sj] = True
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)] if connectivity == 4 else \
           [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while q:
        i,j = q.popleft()
        for di,dj in nbrs:
            ii, jj = i+di, j+dj
            if 0 <= ii < ny and 0 <= jj < nx and free[ii,jj] and not vis[ii,jj]:
                vis[ii,jj] = True
                q.append((ii,jj))
    return vis


def exit_to_boundary_dijkstra(
    entry_xy: np.ndarray,
    cur_bounds: tuple,
    goal_xy: np.ndarray,
    C: np.ndarray, R: np.ndarray,
    inflate: float = 0.0,
    grid_res=(64, 40),
    prefer_goal_if_inside=True,
    # --- new knobs ---
    goal_target_topk: int | None = None,  # e.g., 64  (keep nearest K boundary cells)
    goal_target_frac: float = 0.25,       # or keep nearest fraction (0,1]
    goal_pull: float = 0.35,              # weight of goal-distance bias in cell costs
    # NEW knobs for clearance
    gamma_clear=1.0,     # strength of clearance penalty
    p_clear=2.0,         # sharpness near walls (1.0..2.0)
    edge_clear_min=0.1, # meters: discard boundary cells tighter than this
):
    xs, ys, nx, ny = _grid_from_bounds(cur_bounds, grid_res)
    free = _inflate_mask(xs, ys, C, R, inflate, dilate_cells=1)

    entry = np.asarray(entry_xy, float)
    goal  = np.asarray(goal_xy,  float)

    # ---------- clearance-aware weights ----------
    weights = _clearance_weight_field(xs, ys, free, C, R, inflate,
                                      goal_xy=goal, goal_pull=goal_pull,
                                      gamma=gamma_clear, p=p_clear, eps=1e-3)

    # ---------- choose targets (goal-in-box or boundary) ----------
    si, sj = _closest_index(xs, ys, entry)
    if not free[si, sj]:
        free[si, sj] = True  # allow start cell

    targets = []
    if prefer_goal_if_inside and _inside_rect(goal, cur_bounds):
        gi, gj = _closest_index(xs, ys, goal)
        if free[gi, gj]:
            targets = [(gi, gj)]

    if not targets:
        # boundary candidates that are free AND interior-adjacent
        all_bd = _boundary_candidates(free)

        if not all_bd:
            return None

        # FILTER: require some minimum clearance on the edge
        clr = _clearance_field(xs, ys, C, R, inflate)
        safe_bd = [(i,j) for (i,j) in all_bd if clr[i,j] >= edge_clear_min]
        bd = safe_bd if safe_bd else all_bd  # if all are tight, keep all

        # RANK: combine goal distance and “safety” so safe+goalward wins
        ranked = []
        for (i,j) in bd:
            gx, gy = xs[j], ys[i]
            d_goal2 = (gx - goal[0])**2 + (gy - goal[1])**2
            # smaller score is better: goal distance + penalty for low clearance
            score = d_goal2 + (1.5 * max(0.0, edge_clear_min - clr[i,j]) / edge_clear_min)
            ranked.append(((i,j), score))
        ranked.sort(key=lambda t: t[1])

        if goal_target_topk is None:
            K = max(1, int(math.ceil(len(bd) * float(goal_target_frac))))
        else:
            K = max(1, min(len(bd), int(goal_target_topk)))
        targets = [ij for (ij, _) in ranked[:K]]

    goals_set = set(targets)
    path = _dijkstra_grid(free, weights, (si,sj), set(goals_set))
    if path is None:
        return None

    i_hit, j_hit = path[-1]
    exit_xy = np.array([xs[j_hit], ys[i_hit]], dtype=float)

    # Snap to nearest free edge cell (keeps exits off obstacles on the boundary)
    if not (prefer_goal_if_inside and _inside_rect(goal, cur_bounds)):
        exit_xy = _snap_to_free_edge(exit_xy, cur_bounds, xs, ys, free, max_slide_cells=8)

    return exit_xy

def exit_to_boundary_reachable(
    entry_xy: np.ndarray,
    cur_bounds: tuple,
    goal_xy: np.ndarray,
    C: np.ndarray, R: np.ndarray,
    inflate: float = 0.0,
    grid_res=(64, 40),
    connectivity: int = 8,
    prefer_goal_if_inside: bool = True,
):
    """
    Connectivity-only exit selection:
      1) Build inflated free mask inside cur_bounds.
      2) BFS from entry cell to get reachable component.
      3) Candidate exits = reachable boundary cells.
      4) Pick the one closest to goal (or first, if you prefer).
      5) Return exact edge point (snapped to free edge).
    Returns None if no boundary cell is reachable (fully enclosed component).
    """
    xs, ys, _, _= _grid_from_bounds(cur_bounds, grid_res)
    free = _inflate_mask(xs, ys, C, R, 1.1 * inflate)

    entry = np.asarray(entry_xy, float)
    goal  = np.asarray(goal_xy, float)

    # if goal is inside current box and prefer_goal_if_inside, go straight there
    x0,x1,y0,y1 = cur_bounds
    if prefer_goal_if_inside and (x0<=goal[0]<=x1) and (y0<=goal[1]<=y1):
        return goal.copy()

    si, sj = _closest_index(xs, ys, entry)
    # ensure we start in free (if not, open the nearest free cell)
    if not free[si,sj]:
        # quick nearest-free search
        ny, nx = free.shape
        best, bd = None, 1e18
        for i in range(ny):
            for j in range(nx):
                if free[i,j]:
                    d = (xs[j]-entry[0])**2 + (ys[i]-entry[1])**2
                    if d < bd: bd, best = d, (i,j)
        if best is None:
            return None
        si, sj = best

    comp = _bfs_component(free, (si,sj), connectivity=connectivity)
    bd_cells = _boundary_candidates(free)
    # reachable boundary cells only
    cand = [(i,j) for (i,j) in bd_cells if comp[i,j]]
    if not cand:
        return None  # enclosed region

    # choose the reachable boundary cell closest to the GOAL
    cand.sort(key=lambda ij: (xs[ij[1]]-goal[0])**2 + (ys[ij[0]]-goal[1])**2)
    i_hit, j_hit = cand[0]
    exit_xy = np.array([xs[j_hit], ys[i_hit]], dtype=float)

    # snap to a free edge point near that cell
    exit_xy = _snap_to_free_edge(exit_xy, cur_bounds, xs, ys, free, max_slide_cells=8)
    return exit_xy

def exit_to_boundary_exhaustive(
    entry_xy, cur_bounds, goal_xy,
    Cw, Rw,
    inflate=0.0,
    grid_res=(64,40),
    connectivity=8,
    prefer_goal_if_inside=True,
    min_step_frac=0.12,   # require at least this fraction of min(W,H) movement
):
    """
    Enumerate all *reachable* free boundary cells and pick the best by a robust score.
    Allows exits that move temporarily away from the goal when necessary.
    """
    xs, ys, _, _ = _grid_from_bounds(cur_bounds, grid_res)
    free   = _inflate_mask(xs, ys, Cw, Rw, inflate)

    entry = np.asarray(entry_xy, float)
    goal  = np.asarray(goal_xy,  float)

    # goal inside → go there
    x0,x1,y0,y1 = cur_bounds
    if prefer_goal_if_inside and (x0<=goal[0]<=x1) and (y0<=goal[1]<=y1):
        return goal.copy()

    si, sj = _closest_index(xs, ys, entry)
    if not free[si,sj]:
        # move to nearest free
        ny, nx = free.shape
        best, bd = None, 1e18
        for i in range(ny):
            for j in range(nx):
                if free[i,j]:
                    d = (xs[j]-entry[0])**2 + (ys[i]-entry[1])**2
                    if d < bd: bd, best = d, (i,j)
        if best is None:
            return None
        si, sj = best

    comp = _bfs_component(free, (si,sj), connectivity=connectivity)
    bd_cells = [(i,j) for (i,j) in _boundary_candidates(free) if comp[i,j]]
    if not bd_cells:
        return None  # enclosed pocket

    # scoring: prefer goal progress, but accept detours; require min step
    W = x1 - x0; H = y1 - y0
    min_step = min_step_frac * min(W, H)
    g0 = float(np.linalg.norm(goal - entry))

    best = (None, -1e18)  # (exit_xy, score)
    for (i,j) in bd_cells:
        p = np.array([xs[j], ys[i]], float)
        p = _snap_to_free_edge(p, cur_bounds, xs, ys, free, max_slide_cells=8)

        step = float(np.linalg.norm(p - entry))
        if step < min_step:
            continue  # too tiny → likely jitter

        # goal progress (can be negative; that’s okay as last resort)
        g1 = float(np.linalg.norm(goal - p))
        progress = g0 - g1

        # clearance proxy: distance to nearest inflated obstacle in this crop
        if len(Cw):
            clr = float(np.min(np.linalg.norm(p - Cw, axis=1) - (Rw + inflate)))
        else:
            clr = 1.0

        # score: prioritize positive progress; if none exist, fall back to safety+step
        # two-mode scoring implemented by the weights below:
        score = 1.0*progress + 0.6*clr + 0.1*step
        if score > best[1]:
            best = (p, score)

    return best[0]