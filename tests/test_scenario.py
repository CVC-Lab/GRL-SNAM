"""End-to-end fog-of-war scenarios with bicycle dynamics — the three demo
stories run headlessly: a stale-map ghost, a blocker that appears mid-drive,
and a transient unit. These are the tests that say "the Friday demo works"
without a renderer in the loop."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sdf_nav  # noqa: E402  (imports torch; must follow the skip guard)
from grl_snam.scenario import Event, FogScenario  # noqa: E402

N = 96
BOUNDS = (-100.0, -100.0, 100.0, 100.0)
SCALE = 0.05


def _meta():
    return dict(
        scale=SCALE,
        center=(0.0, 0.0),
        region=100.0,
        rr=0.15,
        d_hat=0.35,
        dt=0.06,
        nsub=2,
        vmax=0.9,
        bounds=list(BOUNDS),
    )


def _model():
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _cell_to_world(r, c):
    x = BOUNDS[0] + c / (N - 1) * (BOUNDS[2] - BOUNDS[0])
    y = BOUNDS[1] + r / (N - 1) * (BOUNDS[3] - BOUNDS[1])
    return x, y


def _wall(occ, r0, r1, c0, c1):
    occ[r0:r1, c0:c1] = True
    return occ


def test_open_field_baseline_reaches_all_waypoints_cleanly():
    truth = np.zeros((N, N), bool)
    sc = FogScenario(
        truth,
        BOUNDS,
        SCALE,
        _model(),
        _meta(),
        waypoints=[(40.0, 0.0), (40.0, 40.0)],
        unknown="optimistic",
    ).start((-40.0, 0.0))
    res = sc.run(max_steps=6000)
    assert res.waypoints_reached == 2, "baseline failed to reach its waypoints"
    assert res.truth_penetration_steps == 0


def test_stale_map_ghost_forces_detour_then_clears():
    """Story 1: the prior map has a wall reality no longer has. The agent must
    first respect the ghost (belief is all it has), then — once its sensor
    crosses the site — clear it and rebuild."""
    truth = np.zeros((N, N), bool)  # reality: open field
    prior = np.zeros((N, N), bool)
    _wall(prior, 20, 76, 47, 50)  # the map's ghost wall, square across the route

    sc = FogScenario(
        truth,
        BOUNDS,
        SCALE,
        _model(),
        _meta(),
        waypoints=[(60.0, 0.0)],
        prior_occ=prior,
        unknown="optimistic",
        sensor=dict(range_m=35.0, n_rays=240),
    ).start((-60.0, 0.0))
    res = sc.run(max_steps=6000)

    assert res.waypoints_reached == 1, "never reached the goal past the ghost"
    assert res.rebuilds > 0, "the ghost was never cleared into a rebuild"
    assert res.truth_penetration_steps == 0
    # The believed wall at the route midline must be gone by the end.
    assert not sc.belief.to_occupancy(unknown="optimistic")[48, 48]


def test_new_blocker_appears_and_is_avoided_against_truth():
    """Story 2: reality grows a wall across the planned route mid-drive. The
    metric that matters is penetration against TRUTH — hitting a wall you did
    not know about still counts."""
    truth = np.zeros((N, N), bool)
    # A ~30 m blocker across the route. Deliberately NOT a region-spanning
    # wall: a purely reactive navigator on an untrained model oscillates on a
    # worst-case 80 m wall centered on the goal (that geometry is what the
    # A*-spine composition is for); what THIS test pins is discovery, honest
    # truth-collision scoring, and the replan around a realistic obstacle.
    sc = FogScenario(
        truth,
        BOUNDS,
        SCALE,
        _model(),
        _meta(),
        waypoints=[(60.0, 0.0)],
        events=[Event(step=40, kind="add_rect", args=(41, 56, 46, 50))],
        unknown="optimistic",
        sensor=dict(range_m=45.0, n_rays=240),
        sense_every=3,
    ).start((-60.0, 0.0))
    res = sc.run(max_steps=8000)

    assert res.rebuilds > 0, "the new blocker was never discovered"
    assert res.waypoints_reached == 1, "did not reach the goal after replanning"
    # Grace: discovery takes a sensor frame; what must NOT happen is driving
    # through the wall as if it were not there.
    assert (
        res.truth_penetration_steps <= 2
    ), f"{res.truth_penetration_steps} steps inside the undiscovered wall"


def test_unit_marks_trigger_rebuild_and_decay():
    """Story 3: a transient unit near the route forces a rebuild while fresh
    and stops mattering once it decays."""
    truth = np.zeros((N, N), bool)
    ur, uc = 48, 40
    sc = FogScenario(
        truth,
        BOUNDS,
        SCALE,
        _model(),
        _meta(),
        waypoints=[(60.0, 0.0)],
        events=[Event(step=10, kind="unit_at", args=(ur, uc))],
        unknown="optimistic",
        unit_ttl_s=1.0,
        sense_every=2,
    ).start((-60.0, 0.0))

    saw_dynamic = False
    for _ in range(600):
        sc.step()
        occ = sc.dyn.occupancy(sc._t())
        if occ[ur, uc]:
            saw_dynamic = True
        # sc.done, NOT sc.nav.reached: the nav's goal is the route sub-goal
        # ~lookahead metres ahead, which sits inside reach_tol by construction
        # — nav.reached is true from step one and would end the loop before
        # the unit event at step 10 even fires.
        if sc.done:
            break
    assert saw_dynamic, "the unit never entered the dynamic layer"
    # ttl of 1 world-second at dt=0.06 is ~17 steps; long gone by the end.
    assert not sc.dyn.occupancy(sc._t()).any(), "unit mark failed to decay"


def test_determinism_same_scenario_same_trajectory():
    """The replay property (roadmap 22.5): identical inputs, identical run —
    the fixed quantum is what makes a benchmark comparable to itself."""
    truth = np.zeros((N, N), bool)
    _wall(truth, 30, 66, 60, 63)

    def run():
        torch.manual_seed(0)
        sc = FogScenario(
            truth,
            BOUNDS,
            SCALE,
            _model(),
            _meta(),
            waypoints=[(70.0, 10.0)],
            unknown="optimistic",
        ).start((-60.0, -10.0))
        return sc.run(max_steps=1500).positions

    a, b = run(), run()
    assert a.shape == b.shape
    assert np.array_equal(a, b), "same scenario diverged between runs"


def test_pessimistic_policy_stays_inside_observed_space():
    """Under the pessimistic policy the agent must not outrun its own sensor:
    it should keep clearance from *unknown* space, not only from walls."""
    truth = np.zeros((N, N), bool)
    sc = FogScenario(
        truth,
        BOUNDS,
        SCALE,
        _model(),
        _meta(),
        waypoints=[(50.0, 0.0)],
        unknown="pessimistic",
        sensor=dict(range_m=50.0, n_rays=360),
        sense_every=2,
    ).start((-50.0, 0.0))
    for _ in range(1200):
        rec = sc.step()
        if sc.nav.reached:
            break
        r, c = sc.belief.world_to_cell(rec.x, rec.y)
        pv = 1.0 / (1.0 + np.exp(-float(sc.belief.logodds[r, c])))
        assert pv < 0.5, "agent stood on ground it had never confirmed free"


def test_body_metric_is_honest_about_resolution():
    """A bitmap has no sub-cell information: its distance field steps in whole
    cells. So the body metric can only tell the body apart from its reference
    point when a cell is no bigger than the body -- and the city story's 2.1 m
    cells against a 0.42 m-wide vehicle cannot. The scenario must SAY so, or a
    deliverable will quote `body_penetration` as if it measured something."""
    truth = np.zeros((N, N), bool)
    sc = FogScenario(truth, BOUNDS, SCALE, _model(), _meta(), waypoints=[(40.0, 0.0)]).start(
        (-40.0, 0.0)
    )
    cell_w = (BOUNDS[2] - BOUNDS[0]) / (N - 1)
    assert cell_w > 2 * sc._body_half_width_m()
    assert not sc.body_metric_resolvable(), "n=96 must not claim to resolve a body"


def _fine_scenario():
    """A vehicle-scale lattice, where the body metric actually resolves."""
    fine_n = 1001  # 0.2 m cells over the 200 m world
    truth = np.zeros((fine_n, fine_n), bool)
    truth[:, fine_n // 2 :] = True  # wall over the right half
    sc = FogScenario(truth, BOUNDS, SCALE, _model(), _meta(), waypoints=[(40.0, 0.0)]).start(
        (-40.0, 0.0)
    )
    wall_x = BOUNDS[0] + (fine_n // 2) / (fine_n - 1) * (BOUNDS[2] - BOUNDS[0])
    return sc, wall_x


def test_body_collision_sees_what_the_point_test_misses_at_vehicle_scale():
    """The whole reason the metric exists, on a grid fine enough to show it.

    Park the vehicle so the rear axle is in free space and the NOSE overlaps the
    wall: the point test reports clean. That is exactly how a footprint's
    benefit becomes invisible and a safety-for-reach trade reads as a pure loss.
    """
    sc, wall_x = _fine_scenario()
    assert sc.body_metric_resolvable()
    nose_m = max(sc._body_probe_m())
    x = wall_x - 0.5 * nose_m  # rear axle clear, nose past the wall face

    assert not sc._truth_hit((x, 0.0)), "the reference point should read clean"
    assert sc._body_hit(x, 0.0, 0.0), "the nose is in the wall and was not seen"
    assert sc.body_clearance_m(x, 0.0, 0.0) < 0.0

    # Heading reversed the nose points away, so the body is clean again --
    # this is a BODY test, not merely a fattened point.
    assert not sc._body_hit(x, 0.0, math.pi)
    assert sc.body_clearance_m(x, 0.0, math.pi) > 0.0


def test_body_clearance_is_continuous_where_the_boolean_is_flat():
    """The continuous metric is the point of this for deliverables: two
    configurations can both post zero penetrations while one runs half the
    margin, and only clearance shows it."""
    sc, wall_x = _fine_scenario()
    prev = None
    for back in (2.0, 4.0, 8.0):
        clr = sc.body_clearance_m(wall_x - back, 0.0, 0.0)
        assert clr > 0.0, "well clear of the wall must not read as a collision"
        if prev is not None:
            assert clr > prev, "clearance must grow as the body backs away"
        prev = clr


def test_body_collision_is_a_superset_of_the_point_test():
    """Anywhere the reference point is inside truth, the body is too. Without
    this the two metrics could disagree in the direction that matters and a
    'safer' configuration could post a WORSE body number for no real reason."""
    sc, wall_x = _fine_scenario()
    for dx in (1.0, 5.0, 20.0):
        deep = wall_x + dx
        assert sc._truth_hit((deep, 0.0))
        for heading in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            assert sc._body_hit(deep, 0.0, heading)
