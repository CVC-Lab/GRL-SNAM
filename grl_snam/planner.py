"""Belief-space global route planning: A* over what the agent THINKS is there.

The reactive SDF navigator is a local steering law — it has no notion of
topology, and a fresh wall square across the route is a potential-field local
minimum it can only escape by luck or wall-following. The working architecture
(measured on Austin) is a global route spine + learned local control between
sub-goals. Under fog, the spine must be planned on **belief**, not truth: the
agent routes around what it believes, drives into what it has not seen, and
replans when its sensor changes its mind. The SDF rebuild and the route replan
are the same event.

Pure numpy, 8-connected A* with corner-cut prevention, radius inflation and
line-of-sight string-pulling — deliberately the same shape as
``pycvc_gl.scenes`` so the demo behaves like the shipped Austin pipeline, but
with no dependency on the compiled bindings (testable anywhere).
"""

from __future__ import annotations

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)


def inflate(occ: np.ndarray, cells: int) -> np.ndarray:
    """Binary dilation by ``cells`` 4-connected steps (matches the shipped
    planner's inflation; no scipy dependency)."""
    out = occ.astype(bool).copy()
    for _ in range(max(0, cells)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _line_of_sight(occ: np.ndarray, a, b) -> bool:
    """Bresenham walk; True if no occupied cell between a and b."""
    r0, c0 = a
    r1, c1 = b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if occ[r, c]:
            return False
        if (r, c) == (r1, c1):
            return True
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _nearest_free(occ: np.ndarray, r: int, c: int, max_radius: int = 12):
    """Snap a cell to the nearest free cell (start/goal may sit inside an
    inflated boundary)."""
    ny, nx = occ.shape
    r = min(max(r, 0), ny - 1)
    c = min(max(c, 0), nx - 1)
    if not occ[r, c]:
        return r, c
    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in (-rad, rad):
                for rr_, cc_ in ((r + dr, c + dc), (r + dc, c + dr)):
                    if 0 <= rr_ < ny and 0 <= cc_ < nx and not occ[rr_, cc_]:
                        return rr_, cc_
    return None


def astar(occ: np.ndarray, start, goal):
    """8-connected A* over free cells; diagonal moves must not cut corners.
    Returns a list of (r, c) or None when unreachable."""
    ny, nx = occ.shape
    start = _nearest_free(occ, *start)
    goal = _nearest_free(occ, *goal)
    if start is None or goal is None:
        return None

    def h(n):
        return math.hypot(n[0] - goal[0], n[1] - goal[1])

    open_q = [(h(start), 0.0, start, None)]
    came: dict = {}
    g_best = {start: 0.0}
    while open_q:
        _f, g, node, parent = heapq.heappop(open_q)
        if node in came:
            continue
        came[node] = parent
        if node == goal:
            path = [node]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        r, c = node
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ny and 0 <= nc < nx) or occ[nr, nc]:
                    continue
                if dr and dc and (occ[r + dr, c] or occ[r, c + dc]):
                    continue  # no corner cutting
                ng = g + (SQRT2 if dr and dc else 1.0)
                nxt = (nr, nc)
                if ng < g_best.get(nxt, float("inf")):
                    g_best[nxt] = ng
                    heapq.heappush(open_q, (ng + h(nxt), ng, nxt, node))
    return None


def simplify(occ: np.ndarray, path):
    """String-pull: keep only the corners needed to preserve line of sight."""
    if not path or len(path) < 3:
        return path
    out = [path[0]]
    anchor = 0
    for i in range(2, len(path)):
        if not _line_of_sight(occ, path[anchor], path[i]):
            out.append(path[i - 1])
            anchor = i - 1
    out.append(path[-1])
    return out


class BeliefRoutePlanner:
    """Plan (and re-plan) a world-coordinate route over a belief occupancy."""

    def __init__(self, bounds, shape, *, inflate_cells: int = 1):
        self.mnx, self.mny, self.mxx, self.mxy = (float(b) for b in bounds)
        self.ny, self.nx = int(shape[0]), int(shape[1])
        self.inflate_cells = int(inflate_cells)

    def _w2c(self, x, y):
        c = (x - self.mnx) / (self.mxx - self.mnx) * (self.nx - 1)
        r = (y - self.mny) / (self.mxy - self.mny) * (self.ny - 1)
        return int(round(r)), int(round(c))

    def _c2w(self, r, c):
        x = self.mnx + c / (self.nx - 1) * (self.mxx - self.mnx)
        y = self.mny + r / (self.ny - 1) * (self.mxy - self.mny)
        return float(x), float(y)

    def route_length(self, route) -> float:
        if not route or len(route) < 2:
            return 0.0
        pts = np.asarray(route, np.float64)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def route_valid(self, occ: np.ndarray, route) -> bool:
        """Is an existing route still collision-free under NEW (inflated)
        occupancy? The hysteresis test: a route is only abandoned when belief
        actually invalidates it, or a decisively better one exists."""
        if not route or len(route) < 2:
            return False
        grid = inflate(occ, self.inflate_cells)
        cells = [self._w2c(x, y) for x, y in route]
        for a, b in zip(cells, cells[1:]):
            if not (0 <= a[0] < self.ny and 0 <= a[1] < self.nx):
                return False
            if not _line_of_sight(grid, a, b):
                return False
        return True

    def plan(self, occ: np.ndarray, start_world, goal_world):
        """World route [start..goal] over the (belief) occupancy, or None if
        the goal is unreachable *in belief* — which the caller should surface,
        not paper over (an unroutable click deserves a 'no route', not a
        silently discontinuous path)."""
        grid = inflate(occ, self.inflate_cells)
        cells = astar(grid, self._w2c(*start_world), self._w2c(*goal_world))
        if cells is None:
            return None
        cells = simplify(grid, cells)
        return [self._c2w(r, c) for r, c in cells]


def free_components(occ: np.ndarray, inflate_cells: int):
    """Label connected free space UNDER THE PLANNER'S OWN INFLATION.

    Picking endpoints from raw free space is the trap: a cell can be free and
    still unreachable once the route is inflated for clearance, so a run drives
    most of the way and then reports no route for the last stretch (measured on
    Austin: 1550 m driven, then no_route 147 m short). Endpoints have to come
    from the same space the planner will actually search.

    Returns ``(labels, sizes)`` with 0 = blocked, 1..n = component ids.
    """
    grid = ~inflate(occ, inflate_cells)
    ny, nx = grid.shape
    labels = np.zeros((ny, nx), np.int32)
    sizes: dict[int, int] = {}
    nxt = 0
    for r0 in range(ny):
        for c0 in range(nx):
            if not grid[r0, c0] or labels[r0, c0]:
                continue
            nxt += 1
            n = 0
            stack = [(r0, c0)]
            labels[r0, c0] = nxt
            while stack:
                r, c = stack.pop()
                n += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < ny and 0 <= cc < nx and grid[rr, cc] and not labels[rr, cc]:
                        labels[rr, cc] = nxt
                        stack.append((rr, cc))
            sizes[nxt] = n
    return labels, sizes


def far_pair_in_free_space(occ: np.ndarray, bounds, inflate_cells: int, *, rng=None):
    """A start/goal pair that is guaranteed routable: both drawn from the
    LARGEST inflated-free component, and far apart within it."""
    labels, sizes = free_components(occ, inflate_cells)
    if not sizes:
        raise ValueError("no free space at this inflation — the map is closed")
    best = max(sizes, key=lambda k: sizes[k])
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    ny, nx = occ.shape

    def to_world(r, c):
        return (
            mnx + c / (nx - 1) * (mxx - mnx),
            mny + r / (ny - 1) * (mxy - mny),
        )

    # Two passes of "farthest point from here" — a cheap graph diameter that
    # keeps both endpoints inside the component rather than at opposite map
    # corners which may be in different ones.
    i0 = 0
    for _ in range(2):
        d = (rows - rows[i0]) ** 2 + (cols - cols[i0]) ** 2
        i0 = int(np.argmax(d))
    a = i0
    d = (rows - rows[a]) ** 2 + (cols - cols[a]) ** 2
    b = int(np.argmax(d))
    return to_world(rows[a], cols[a]), to_world(rows[b], cols[b]), sizes[best]
