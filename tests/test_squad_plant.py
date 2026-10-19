"""A squad on libcvc's native runtime and a physics plant (libcvc's P6).

``Squad(plant=...)`` runs the squad in the pure-C++ ``cvc::nav::sim_world``, one private
belief plane per agent, with the agents driving a Jolt plant's vehicles
(:class:`grl_snam.nav_native.JoltPlant`) or nav's C++ bicycle (``"kinematic"``). These
tests need a pycvc built from libcvc with P6's physics bindings; with any other build
they skip.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

from grl_snam import nav_native, planner  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402
from grl_snam.squad import AgentSpec, FollowGoal, Squad  # noqa: E402

pytestmark = pytest.mark.skipif(
    not nav_native.HAS_PLANT, reason="pycvc build lacks libcvc P6's physics bindings"
)


def _city(n=96, max_steps=1500):
    """The city story and its truth, with a helper that snaps a point into the largest
    free component (a physics vehicle cannot be added inside a building)."""
    story = shrunk(STORIES["city"], n=n, max_steps=max_steps)
    truth = story.truth_grid()
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape
    pts = np.stack(
        [mnx + cols / (nx - 1) * (mxx - mnx), mny + rows / (ny - 1) * (mxy - mny)], axis=1
    )

    def free(x, y):
        i = int(np.argmin(((pts - (x, y)) ** 2).sum(1)))
        return float(pts[i, 0]), float(pts[i, 1])

    return story, truth, free


def _crossing(free):
    return [
        AgentSpec("a", free(-70, -60), free(70, 60), (0.3, 0.7, 1.0)),
        AgentSpec("b", free(-70, 60), free(70, -60), (1.0, 0.5, 0.2)),
    ]


def test_obstacle_boxes_cover_exactly_the_occupied_cells():
    """The physics world's buildings are the truth grid's occupied cells, no more and
    no less: rasterize the merged boxes back and they are the grid."""
    rng = np.random.default_rng(3)
    occ = rng.random((23, 31)) < 0.3
    occ[5:12, 4:20] = True  # a big block merges into few boxes
    bounds = (-15.0, -11.0, 15.0, 11.0)
    boxes = nav_native.obstacle_boxes(occ, bounds)
    ny, nx = occ.shape
    xs = bounds[0] + np.arange(nx) / (nx - 1) * (bounds[2] - bounds[0])
    ys = bounds[1] + np.arange(ny) / (ny - 1) * (bounds[3] - bounds[1])
    cover = np.zeros((ny, nx), int)
    for x0, y0, x1, y1 in boxes:
        cover[np.ix_((ys > y0) & (ys < y1), (xs > x0) & (xs < x1))] += 1
    assert (cover == occ).all(), "every occupied cell in exactly one box, no free cell in any"
    assert len(boxes) < int(occ.sum()), "the merge joins cells into rectangles"


def test_a_squad_drives_the_jolt_plant_to_its_goals():
    """Two RoboRacers cross the city on the Jolt plant: both arrive, never inside a
    building, and their energy ledgers say what they drew."""
    story, truth, free = _city()
    torch.manual_seed(0)
    sq = Squad(story, _crossing(free), truth_occ=truth, prior_occ=truth, plant="jolt")
    assert sq.native.plant_kind == "jolt"
    assert sq.native.planes == 2, "one private belief plane per agent"
    assert len(sq.plant.obstacles) > 0, "the buildings are bodies in the physics world"
    r = sq.run(max_steps=1500)
    assert r.reached == {"a": True, "b": True}, r.reached
    assert r.penetration == {"a": 0, "b": 0}, r.penetration
    for k in ("a", "b"):
        assert 0.0 < r.energy_J[k] < 1e5, r.energy_J  # a 1/10 car's battery: a few kJ
        assert len(r.tracks[k]) == r.ticks
    # The plant arm's drift against nav's bicycle is published, binned by lateral g.
    bins = sq.plant.drift_summary()
    assert sum(b["ticks"] for b in bins) == 2 * r.ticks


def test_the_kinematic_native_path_is_deterministic_and_carries_no_energy():
    story, truth, free = _city(max_steps=600)
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        sq = Squad(story, _crossing(free), truth_occ=truth, prior_occ=truth, plant="kinematic")
        assert sq.native.plant_kind == "kinematic" and sq.plant is None
        runs.append(sq.run(max_steps=600))
    a, b = runs
    assert a.ticks == b.ticks
    for k in ("a", "b"):
        np.testing.assert_array_equal(a.tracks[k], b.tracks[k])
    assert a.energy_J == {}, "nav's bicycle has no energy ledger"
    assert all(a.reached.values())


def test_the_jolt_plant_is_deterministic():
    """The same squad twice on the plant: the same tracks and the same physics state."""
    story, truth, free = _city(max_steps=300)
    out = []
    for _ in range(2):
        torch.manual_seed(0)
        sq = Squad(story, _crossing(free), truth_occ=truth, prior_occ=truth, plant="jolt")
        r = sq.run(max_steps=300, stop_when_done=False)
        out.append((r, sq.plant.state_hash()))
    (a, ha), (b, hb) = out
    assert ha == hb
    for k in ("a", "b"):
        np.testing.assert_array_equal(a.tracks[k], b.tracks[k])


def test_a_start_inside_a_building_is_refused_on_a_physics_plant():
    story, truth, _ = _city(n=48)
    agents = [AgentSpec("a", (-70.0, -60.0), (70.0, 60.0))]  # (9, 7) is a building at n=48
    with pytest.raises(ValueError, match="inside an obstacle"):
        Squad(story, agents, truth_occ=truth, prior_occ=truth, plant="jolt")
    Squad(story, agents, truth_occ=truth, prior_occ=truth, plant="kinematic")  # nav drives out


def test_a_follower_tracks_its_leader_on_the_plant():
    """A FollowGoal reads its leader's position from the native runtime."""
    story, truth, free = _city(max_steps=900)
    agents = [
        AgentSpec("lead", free(-70, -60), free(70, 60)),
        AgentSpec("follow", free(-80, -70), (0.0, 0.0), moving_goal=FollowGoal("lead")),
    ]
    torch.manual_seed(0)
    sq = Squad(story, agents, truth_occ=truth, prior_occ=truth, plant="jolt")
    sq.run(max_steps=900, stop_when_done=False)
    lx, ly = sq.position_of("lead")
    fx, fy = sq.position_of("follow")
    assert np.hypot(lx - fx, ly - fy) < 12.0, ((lx, ly), (fx, fy))


def test_the_plant_is_selectable_on_a_native_world():
    story, truth, free = _city(max_steps=60)
    torch.manual_seed(0)
    sq = Squad(story, _crossing(free), truth_occ=truth, prior_occ=truth, plant="kinematic")
    w = sq.native
    w.set_plant(nav_native.JoltPlant("roboracer", bounds=story.bounds, obstacles=truth))
    assert w.plant_kind == "jolt"
    for _ in range(10):
        w.step()
    w.set_plant(None)
    assert w.plant_kind == "kinematic"
    w.step()
    assert w.tick == 11


def test_the_terrain_limits_of_a_vehicle():
    cfg = nav_native.JoltPlant.terrain_config_of("humvee")
    assert 0.6 < cfg["max_grade"] < 0.8, cfg
    assert abs(cfg["tip_ratio"] - 0.91 / 0.742) < 0.05, cfg
    rr = nav_native.JoltPlant.terrain_config_of("roboracer")
    assert rr["max_grade"] > 0.0 and rr["tip_ratio"] > 0.0, rr


def test_the_plant_needs_a_squad_without_overrides():
    story, truth, free = _city(max_steps=60)
    agents = _crossing(free)
    agents[1].vmax = 3.0
    with pytest.raises(NotImplementedError, match="vmax"):
        Squad(story, agents, truth_occ=truth, prior_occ=truth, plant="jolt")
