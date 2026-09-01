"""The clearance-routing helper every demo now plans through."""

import numpy as np

import sdf_nav
from grl_snam import planner
from grl_snam.route import cells_for_metres, plan_clearance_route


def _clearance(occ):
    """Cells to the nearest blocked cell, via the SAME EDT clearance_cost uses.

    Not scipy: it is not a dependency of this package and CI has no such wheel,
    which is how the first version of this file turned CI red. Measuring with
    the production EDT is also the more honest check.
    """
    return np.sqrt(sdf_nav._edt2(np.asarray(occ) != 0))


def _min_along(clr, route):
    """Minimum clearance ALONG the polyline, not at its vertices.

    Sampling vertices is wrong and flatters simplified routes: simplification
    leaves few points, so a vertex-only minimum skips the wall-hugging segments
    between them. Measured that way a simplified route looked BETTER (8.06 vs
    7.0) than the one that actually keeps its distance.
    """
    pts = np.asarray(route, float)
    worst = float("inf")
    for a, b in zip(pts, pts[1:]):
        n = max(2, int(np.hypot(*(b - a)) * 4))
        for t in np.linspace(0.0, 1.0, n):
            x, y = a + t * (b - a)
            worst = min(worst, clr[int(round(y)), int(round(x))])
    return worst


def test_clearance_route_keeps_far_more_standoff_than_shortest():
    """The whole point. A shortest path hugs the corner because that is what
    shortest means; the surcharge buys distance from it."""
    occ = np.zeros((64, 64), np.uint8)
    occ[20:44, 28:36] = 1
    clr = _clearance(occ)

    route = plan_clearance_route(occ, (0.0, 0.0, 64.0, 64.0), [(5, 32), (58, 32)], close_loop=False)
    assert route, "no route found"
    short = planner.astar(occ, (32, 5), (32, 58))
    assert short

    got = _min_along(clr, route)
    base = min(clr[r, c] for r, c in short)
    assert got > base + 1.0, f"clearance route ({got}) barely beat shortest ({base})"


def test_simplify_would_undo_the_standoff():
    """Line-of-sight simplification straightens the route back toward the walls,
    which is why plan_clearance_route leaves it OFF. Pinned so nobody 'restores'
    it as a tidy-up."""
    occ = np.zeros((64, 64), np.uint8)
    occ[20:44, 28:36] = 1
    clr = _clearance(occ)
    kw = dict(close_loop=False)
    keep = plan_clearance_route(occ, (0.0, 0.0, 64.0, 64.0), [(5, 32), (58, 32)], **kw)
    taut = plan_clearance_route(
        occ, (0.0, 0.0, 64.0, 64.0), [(5, 32), (58, 32)], simplify=True, **kw
    )
    m_keep = _min_along(clr, keep)
    m_taut = _min_along(clr, taut)
    # Measured on this world: 6.40 kept vs 1.00 simplified -- the simplified
    # route is exactly as bad as the shortest path, i.e. it gives back every bit
    # of standoff the surcharge just paid for.
    assert m_keep > 3.0 * m_taut, f"simplify kept too much standoff: {m_keep} vs {m_taut}"


def test_a_narrow_corridor_degrades_to_a_route_rather_than_stranding():
    """The surcharge is additive and never forbids a cell, so a corridor
    narrower than d_safe everywhere still routes."""
    occ = np.zeros((40, 40), np.uint8)
    occ[:, :18] = 1
    occ[:, 22:] = 1  # a 4-cell corridor, far under d_safe=6
    route = plan_clearance_route(
        occ, (0.0, 0.0, 40.0, 40.0), [(20, 2), (20, 37)], close_loop=False, d_safe=6.0
    )
    assert route, "a sub-d_safe corridor must still route, not strand the agent"


def test_metre_conversion_tracks_resolution():
    """d_safe is in CELLS, so the same number is a different distance on every
    map -- the same footgun as reading d_hat as a fraction."""
    coarse = cells_for_metres((0.0, 0.0, 200.0, 200.0), (100, 100), 12.0)
    fine = cells_for_metres((0.0, 0.0, 200.0, 200.0), (400, 400), 12.0)
    assert fine > coarse
    assert abs(coarse - 12.0 / (200.0 / 99)) < 1e-6
