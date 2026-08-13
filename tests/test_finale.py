"""The finale's staging geometry.

The scene bundle lives outside this repository, so these exercise the layout
rules with a trivial snapper rather than real city geometry — which is the
part that had the bug anyway.
"""

import numpy as np

from grl_snam.tools.finale_record import COLORS, LINE_HALF, LINE_X, N_AGENTS, staging_line


def _identity_snap(x, y):
    return float(x), float(y)


def test_the_squad_forms_a_line_not_a_ring():
    """A ring walls its own stragglers out.

    Every agent approaches from the west, so on a ring the first arrivals park
    between the late ones and the far-side slots. Measured on the ring, three
    agents orbited the rendezvous at 10 m/s for the last minute, 115 m / 121 m
    / 329 m short. A line has no far side.
    """
    slots = staging_line(_identity_snap)
    assert len(slots) == N_AGENTS
    assert len({s for s in slots}) == N_AGENTS, "slots must be distinct"
    assert all(x == LINE_X for x, _y in slots), "a line: one x, many y"


def test_slot_order_matches_approach_order_so_paths_never_cross():
    """Starts ascend in y and so must the slots; otherwise agents swap sides."""
    starts_y = np.linspace(-430.0, 430.0, N_AGENTS)
    slots_y = [y for _x, y in staging_line(_identity_snap)]
    assert all(slots_y[i] < slots_y[i + 1] for i in range(N_AGENTS - 1))
    assert all(starts_y[i] < starts_y[i + 1] for i in range(N_AGENTS - 1))


def test_slots_are_further_apart_than_a_parked_vehicle_blocks():
    """A parked peer is stamped as an obstacle and inflated ~6 m; a 14 m
    vehicle therefore blocks roughly 26 m. Slots closer than that would let an
    early arrival deny its neighbour's."""
    ys = [y for _x, y in staging_line(_identity_snap)]
    gaps = np.diff(ys)
    assert gaps.min() > 26.0, f"slot spacing {gaps.min():.1f} m is inside a parked vehicle"
    assert ys[0] == -LINE_HALF and ys[-1] == LINE_HALF


def test_every_agent_has_its_own_colour():
    assert len(COLORS) >= N_AGENTS
    assert len({tuple(c) for c in COLORS[:N_AGENTS]}) == N_AGENTS
