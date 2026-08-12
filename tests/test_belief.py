"""Fog-of-war belief mapping: truth vs belief, occlusion, stale ghosts,
discovery-triggered rebuilds, unknown-space policy, and the decaying dynamic
layer. Pure numpy — no torch, no bindings."""

import numpy as np

from grl_snam.belief import BeliefGrid, DynamicLayer, composite_occupancy

N = 64
BOUNDS = (-100.0, -100.0, 100.0, 100.0)


def _world(r, c):
    """Cell -> world center, for placing the sensor."""
    x = BOUNDS[0] + c / (N - 1) * (BOUNDS[2] - BOUNDS[0])
    y = BOUNDS[1] + r / (N - 1) * (BOUNDS[3] - BOUNDS[1])
    return x, y


def _empty_truth():
    return np.zeros((N, N), bool)


def _wall_truth(col=40):
    t = _empty_truth()
    t[10:54, col] = True  # a vertical wall
    return t


# ── log-odds fundamentals ────────────────────────────────────────────────────


def test_prior_is_maximally_uncertain():
    b = BeliefGrid((N, N), BOUNDS)
    assert np.allclose(b.p(), 0.5)
    assert not b.known().any()
    assert b.confidence_at(0.0, 0.0) == 0.0


def test_sensing_free_space_lowers_p_and_sensing_walls_raises_it():
    b = BeliefGrid((N, N), BOUNDS)
    truth = _wall_truth()
    x, y = _world(32, 20)
    b.sense(truth, (x, y), range_m=80.0, n_rays=360)
    pr = b.p()
    assert pr[32, 25] < 0.3, "free space along the ray should be believed free"
    assert pr[32, 40] > 0.7, "the wall hit should be believed occupied"


def test_occlusion_you_cannot_see_behind_the_wall():
    b = BeliefGrid((N, N), BOUNDS)
    truth = _wall_truth(col=40)
    x, y = _world(32, 20)  # sensor west of the wall
    b.sense(truth, (x, y), range_m=200.0, n_rays=720)
    pr = b.p()
    # Directly east of the wall, along the sensing axis: must remain unknown.
    assert abs(pr[32, 50] - 0.5) < 1e-6, "belief leaked through an occluding wall"


def test_range_limit_is_respected():
    b = BeliefGrid((N, N), BOUNDS)
    truth = _empty_truth()
    x, y = _world(32, 32)
    b.sense(truth, (x, y), range_m=20.0, n_rays=360)
    pr = b.p()
    # ~20 m is ~6.3 cells at this resolution; a cell 20 cells away is beyond range
    assert abs(pr[32, 60] - 0.5) < 1e-6, "sensed beyond the range limit"
    assert pr[32, 34] < 0.5, "did not sense within range"


# ── the fog-of-war stories ───────────────────────────────────────────────────


def test_stale_map_ghost_persists_until_observed():
    """Scenario 1: the map says wall, reality says gone. The ghost must hold
    until a ray actually crosses the spot — the cost of a stale map."""
    b = BeliefGrid((N, N), BOUNDS)
    b.logodds[10:54, 40] = 6.0  # prior map: wall firmly believed
    truth = _empty_truth()  # ground reality: demolished
    assert b.to_occupancy(unknown="optimistic")[32, 40], "prior wall missing"

    # Sense from far away, out of range of the wall: ghost must survive.
    x, y = _world(32, 5)
    b.sense(truth, (x, y), range_m=30.0, n_rays=360)
    assert b.to_occupancy(unknown="optimistic")[32, 40], "ghost vanished unobserved"

    # Now walk close enough to see through where it stood — repeated looks
    # must overcome the strong prior (evidence accumulates in log-odds).
    v0 = b.version
    for c in (20, 26, 32, 38):
        x, y = _world(32, c)
        b.sense(truth, (x, y), range_m=40.0, n_rays=720)
    assert not b.to_occupancy(unknown="optimistic")[32, 40], "ghost not cleared"
    assert b.version > v0, "clearing the ghost did not bump the rebuild version"


def test_new_blocker_is_discovered_and_bumps_version():
    """Scenario 2: reality grew a wall the map does not know about."""
    b = BeliefGrid((N, N), BOUNDS)
    truth = _wall_truth(col=44)
    assert not b.to_occupancy(unknown="optimistic")[32, 44]
    v0 = b.version
    x, y = _world(32, 30)
    flips = b.sense(truth, (x, y), range_m=80.0, n_rays=720)
    assert flips > 0
    assert b.version > v0
    assert b.to_occupancy(unknown="optimistic")[32, 44], "blocker not discovered"


def test_unknown_space_policy_optimistic_vs_pessimistic():
    b = BeliefGrid((N, N), BOUNDS)
    # A fully unknown map:
    assert not b.to_occupancy(unknown="optimistic").any(), "optimistic must plan through unknown"
    assert b.to_occupancy(unknown="pessimistic").all(), "pessimistic must block unknown"
    # After observing a corridor, pessimistic opens exactly the seen cells.
    truth = _empty_truth()
    x, y = _world(32, 32)
    b.sense(truth, (x, y), range_m=40.0, n_rays=720)
    pess = b.to_occupancy(unknown="pessimistic")
    assert not pess[32, 32], "observed-free must be traversable under pessimistic"
    assert pess[2, 2], "never-observed must stay blocked under pessimistic"


def test_no_observation_is_not_zero_occupancy():
    """The renderer-parity lesson restated for planning: an unobserved cell is
    UNKNOWN, not free — the two policies must disagree about it."""
    b = BeliefGrid((N, N), BOUNDS)
    opt = b.to_occupancy(unknown="optimistic")
    pes = b.to_occupancy(unknown="pessimistic")
    assert (opt != pes).all()


# ── the dynamic layer ────────────────────────────────────────────────────────


def test_dynamic_marks_decay_with_world_time():
    """Scenario 3: moving units decay instead of smearing permanent ghosts."""
    d = DynamicLayer((N, N), ttl_s=4.0)
    d.mark(32, 40, t_now=100.0)
    assert d.occupancy(t_now=100.5)[32, 40]
    assert d.occupancy(t_now=103.9)[32, 40]
    assert not d.occupancy(t_now=104.5)[32, 40], "mark outlived its ttl"


def test_moving_unit_leaves_no_permanent_smear():
    d = DynamicLayer((N, N), ttl_s=2.0)
    for i, c in enumerate(range(10, 40)):  # unit sweeps across
        d.mark(32, c, t_now=float(i))
    occ = d.occupancy(t_now=30.0)
    assert not occ[32, 10:26].any(), "old track should have decayed"
    assert occ[32, 38], "recent position should persist"


def test_composite_merges_belief_and_dynamic():
    b = BeliefGrid((N, N), BOUNDS)
    b.logodds[20, 20] = 6.0  # believed static wall
    d = DynamicLayer((N, N), ttl_s=4.0)
    d.mark(40, 40, t_now=10.0)
    occ = composite_occupancy(b, d, t_now=11.0, unknown="optimistic")
    assert occ[20, 20] and occ[40, 40]
    occ_later = composite_occupancy(b, d, t_now=99.0, unknown="optimistic")
    assert occ_later[20, 20] and not occ_later[40, 40]


# ── wiring to the SDF (the rebuild is the replan) ────────────────────────────


def test_belief_occupancy_feeds_build_sdf():
    torch = __import__("pytest").importorskip("torch")  # noqa: F841
    import sdf_nav

    b = BeliefGrid((N, N), BOUNDS)
    truth = _wall_truth(col=40)
    x, y = _world(32, 20)
    b.sense(truth, (x, y), range_m=200.0, n_rays=720)
    occ = b.to_occupancy(unknown="optimistic")
    phi, nx_g, ny_g = sdf_nav.build_sdf(occ, BOUNDS, scale=0.05)
    # The believed wall must be negative (inside) in the derived SDF...
    assert phi[32, 40] <= 0.0
    # ...and space observed free must have positive clearance.
    assert phi[32, 20] > 0.0
