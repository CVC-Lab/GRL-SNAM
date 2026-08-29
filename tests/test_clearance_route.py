"""Clearance-weighted routing: the demo-nav lever. sdf_nav.clearance_cost turns
the SDF clearance field into a per-cell A* surcharge; fed to planner.astar (or a
scenario's route_cost_fn via squad.attach_clearance_routing) it trades a little
path length for standoff, which the local drive follows far more reliably."""
import numpy as np

import sdf_nav
from grl_snam import planner


def _min_clearance(occ, path):
    """Min cells-to-nearest-obstacle over a path of (r, c) cells."""
    clr = np.sqrt(sdf_nav._edt2(np.asarray(occ) != 0))
    return min(float(clr[r, c]) for r, c in path)


def test_clearance_cost_values():
    occ = np.zeros((20, 20), np.uint8)
    occ[:, 9:11] = 1  # a wall down the middle
    c = sdf_nav.clearance_cost(occ, d_safe=5.0, gamma=1.5)
    assert c.shape == occ.shape
    assert c.dtype == np.float64
    assert (c >= 0).all()
    assert c[10, 9] == 7.5  # at the wall: gamma * d_safe
    assert abs(c[10, 8] - 6.0) < 1e-9  # one cell away: gamma * (d_safe - 1)
    assert c[10, 0] == 0.0  # >= d_safe cells of clearance: no surcharge
    # monotone non-increasing moving away from the wall
    row = [c[10, cc] for cc in range(8, -1, -1)]
    assert all(row[i] >= row[i + 1] for i in range(len(row) - 1))


def test_clearance_cost_open_field_is_free():
    occ = np.zeros((16, 16), np.uint8)  # no obstacles
    c = sdf_nav.clearance_cost(occ, d_safe=5.0, gamma=1.5)
    assert np.all(c == 0.0)  # everywhere is beyond d_safe -> no surcharge anywhere


def test_clearance_route_keeps_more_standoff():
    # A pillar; start and goal on opposite sides at the pillar's mid-row, so the
    # route must detour around it.
    occ = np.zeros((30, 30), np.uint8)
    occ[12:18, 12:18] = 1
    start, goal = (15, 2), (15, 27)
    plain = planner.astar(occ, start, goal)
    weighted = planner.astar(occ, start, goal, cost=sdf_nav.clearance_cost(occ, d_safe=5.0, gamma=2.0))
    assert plain and weighted, "both routes exist"
    mc_plain = _min_clearance(occ, plain)
    mc_weighted = _min_clearance(occ, weighted)
    # The weighted route pays length to hold a wider berth from the pillar.
    assert mc_weighted > mc_plain
    assert len(weighted) >= len(plain)


def test_clearance_route_degrades_to_shortest_when_forced():
    # A one-cell-wide corridor: clearance is low EVERYWHERE on the only route, so
    # the surcharge cannot buy standoff and must not strand the agent.
    occ = np.ones((11, 11), np.uint8)
    occ[5, :] = 0  # a single open row (the only corridor)
    start, goal = (5, 0), (5, 10)
    plain = planner.astar(occ, start, goal)
    weighted = planner.astar(occ, start, goal, cost=sdf_nav.clearance_cost(occ, d_safe=5.0, gamma=3.0))
    assert plain and weighted
    assert [tuple(p) for p in weighted] == [tuple(p) for p in plain]  # same path, not stranded
