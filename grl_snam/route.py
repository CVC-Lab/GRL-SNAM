"""Grounded routes that keep their distance from buildings.

``pycvc_gl.scenes.plan_ground_route`` plans the SHORTEST grounded route and then
line-of-sight simplifies it. Both halves pull the spine against the walls: the
shortest path hugs corners because that is what shortest means, and the simplify
pass then straightens whatever standoff survived. A local drive following that
spine spends the whole run fighting its own wall barrier.

:func:`plan_clearance_route` is the same interface with the clearance surcharge
from :func:`sdf_nav.clearance_cost` applied, and simplification OFF by default --
skipping it is not an oversight, it is the point. Measured on the procedural city
squad (n=20 x 5 seeds), routing this way lifts route-guided reach ~0.80 -> ~0.90
and produces visibly smoother, wall-avoiding paths.

**Budget-sensitive.** The standoff route is longer, so it pays only when the run
has tick headroom to finish it; under a tight budget it can regress below the
shortest path. Demos have headroom. Benchmarks with a fixed tick budget may not.
"""

from __future__ import annotations

import numpy as np

import sdf_nav
from grl_snam import planner


def plan_clearance_route(
    occ,
    bounds2d,
    waypoints_xy,
    close_loop: bool = True,
    *,
    d_safe: float = 6.0,
    gamma: float = 1.5,
    simplify: bool = False,
):
    """Route through ``occ``'s free space visiting ``waypoints_xy`` in order.

    Drop-in for ``pycvc_gl.scenes.plan_ground_route`` -- same arguments, same
    ``[(x, y), ...]`` world-coordinate return, ``[]`` when no leg is routable.

    ``d_safe`` is the standoff target in CELLS and ``gamma`` the price per cell of
    shortfall. Because the surcharge is additive and never forbids a cell, a
    corridor narrower than ``d_safe`` throughout degrades to the shortest path
    rather than stranding the route.

    ``simplify`` re-enables the line-of-sight pass. Leave it off: it straightens
    the route back toward the walls, undoing the standoff this function exists to
    buy.
    """
    occ = np.asarray(occ)
    ny, nx = occ.shape
    min_x, min_y, max_x, max_y = (float(v) for v in bounds2d)

    def to_cell(x, y):
        c = int(round((x - min_x) / (max_x - min_x) * (nx - 1)))
        r = int(round((y - min_y) / (max_y - min_y) * (ny - 1)))
        return (max(0, min(ny - 1, r)), max(0, min(nx - 1, c)))

    def to_world(rc):
        r, c = rc
        return (
            min_x + c / (nx - 1) * (max_x - min_x),
            min_y + r / (ny - 1) * (max_y - min_y),
        )

    wps = list(waypoints_xy)
    if close_loop and wps:
        wps = wps + [wps[0]]
    if len(wps) < 2:
        return []

    # One raster for every leg: the surcharge depends only on the occupancy.
    cost = sdf_nav.clearance_cost(occ, d_safe=d_safe, gamma=gamma)

    cells = []
    for x, y in wps:
        r, c = to_cell(x, y)
        cells.append(planner._nearest_free(occ, r, c))
    if any(rc is None for rc in cells):
        return []

    full: list = []
    for a, b in zip(cells, cells[1:]):
        seg = planner.astar(occ, a, b, cost=cost)
        if not seg:
            continue
        if simplify:
            seg = planner.simplify(occ, seg)
        if full:
            seg = seg[1:]  # drop the duplicated junction cell
        full += list(seg)
    return [to_world(rc) for rc in full]


def cells_for_metres(bounds2d, shape, metres: float) -> float:
    """Convert a standoff in METRES to the cell units ``d_safe`` expects.

    ``d_safe`` is a count of cells, so the same number means different distances
    on different maps -- the procedural city's cells are ~2 m, a 1024-cell Austin
    raster's are ~0.6 m, and a demo that hardcodes ``d_safe=6`` silently asks for
    12 m on one and 3.5 m on the other. This is the same class of mistake as
    reading ``d_hat`` as a fraction: prefer stating the standoff you want and
    converting it here.
    """
    min_x, min_y, max_x, max_y = (float(v) for v in bounds2d)
    ny, nx = int(shape[0]), int(shape[1])
    cell_x = (max_x - min_x) / max(1, nx - 1)
    cell_y = (max_y - min_y) / max(1, ny - 1)
    return float(metres) / max(1e-9, 0.5 * (cell_x + cell_y))
