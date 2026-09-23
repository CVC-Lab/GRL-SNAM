"""Per-material nav-stats buckets — the discrete material dimension of the base
scorecard, the Python twin of cvc::nav veh_nav_stats.time_over_material_s /
dist_over_material_m (libcvc src/cvc/tests/nav_stats_test.cpp).

Covers: the accumulator (hand-computed, mirroring the C++ test's material_id split),
the from_nav_stats -> VehStats -> aggregate_nav threading and the derived
terrain-risk share, the MaterialIdRaster nearest sampler + coordinate chain, the
city raster builder, the Swarm collection path, and the byte-identical no-material
default.
"""

from __future__ import annotations

import numpy as np

from grl_snam.material_palette import (
    MATERIAL_ID,
    NUM_MATERIALS,
    RISK_MATERIAL_IDS,
    terrain_risk_share,
)
from grl_snam.metrics import NavMetrics, NavStats
from grl_snam.scorecard import EpisodeStats, aggregate_nav


def _feed(ns, steps, dt):
    for x, y, mid in steps:
        ns.update(NavMetrics(x=x, y=y, material_id=mid), dt=dt)


def test_navstats_material_buckets_hand_computed():
    """Time gets dt per collected step; dist gets the travelled segment; both indexed
    by the id at the current pose (parity with nav_stats.cpp step())."""
    ns = NavStats()
    ns.seed_start(0.0, 0.0)  # prime prev at start so the first segment counts
    OPEN, SOIL = MATERIAL_ID["open_air"], MATERIAL_ID["soil"]
    # (x, y, material_id): segs 10, 15, 100 ; dt = 2.0 each
    _feed(ns, [(10.0, 0.0, OPEN), (25.0, 0.0, SOIL), (25.0, 100.0, SOIL)], dt=2.0)

    assert ns.time_over_material_s[OPEN] == 2.0
    assert ns.dist_over_material_m[OPEN] == 10.0
    assert ns.time_over_material_s[SOIL] == 4.0  # two steps * dt
    assert ns.dist_over_material_m[SOIL] == 115.0  # 15 + 100
    # every other bucket untouched
    assert sum(ns.time_over_material_s) == 6.0
    assert sum(ns.dist_over_material_m) == 125.0 == ns.total_path_m


def test_no_material_id_is_byte_identical():
    """Default m.material_id == -1 (and/or no dt) moves no bucket — the opt-in claim."""
    ns = NavStats()
    ns.seed_start(0.0, 0.0)
    ns.update(NavMetrics(x=5.0, y=0.0))  # no material_id, no dt
    ns.update(NavMetrics(x=9.0, y=0.0), dt=1.0)  # dt but material_id still -1
    assert ns.time_over_material_s == [0.0] * NUM_MATERIALS
    assert ns.dist_over_material_m == [0.0] * NUM_MATERIALS
    assert ns.total_path_m == 9.0  # base path accounting unchanged


def test_dist_bucket_accrues_without_dt():
    """dist needs no dt; only the time bucket does."""
    ns = NavStats()
    ns.seed_start(0.0, 0.0)
    ns.update(NavMetrics(x=4.0, y=0.0, material_id=MATERIAL_ID["rock"]))  # no dt
    assert ns.dist_over_material_m[MATERIAL_ID["rock"]] == 4.0
    assert ns.time_over_material_s[MATERIAL_ID["rock"]] == 0.0


def test_from_nav_stats_threads_buckets_and_scorecard_lights_up():
    ns = NavStats()
    ns.seed_start(0.0, 0.0)
    OPEN, SOIL = MATERIAL_ID["open_air"], MATERIAL_ID["soil"]
    ns.update(NavMetrics(x=10.0, y=0.0, material_id=OPEN, reached=True, goal_index=0), dt=2.0)
    ns.update(NavMetrics(x=25.0, y=0.0, material_id=SOIL), dt=2.0)
    ns.update(NavMetrics(x=25.0, y=100.0, material_id=SOIL), dt=2.0)

    ep = EpisodeStats.from_nav_stats([ns], straights_m=[100.0], arrival_times_s=[6.0])
    v = ep.per_vehicle[0]
    assert v.time_over_material_s[SOIL] == 4.0
    assert v.dist_over_material_m[SOIL] == 115.0  # VehStats carries dist too (C++ parity)

    card = aggregate_nav([ep])
    share = card.material_time_share
    assert len(share) == NUM_MATERIALS
    assert abs(sum(share) - 1.0) < 1e-9  # total time normalizes to 1
    assert abs(share[OPEN] - 2.0 / 6.0) < 1e-9
    assert abs(share[SOIL] - 4.0 / 6.0) < 1e-9
    # soil is a terrain-risk class; open_air is not
    assert abs(terrain_risk_share(share) - 4.0 / 6.0) < 1e-9


def test_material_id_raster_nearest_and_bounds():
    from grl_snam.material import MaterialIdRaster

    # 3x3 id plane, world spans [0,20]x[0,20], center (10,10), scale 1 (world==norm/scale+center)
    ids = np.array([[7, 7, 8], [7, 9, 8], [4, 4, 12]], dtype=np.int16)
    r = MaterialIdRaster(ids, bounds=(0.0, 0.0, 20.0, 20.0), center=(10.0, 10.0), scale=1.0)
    # world (0,0) -> cell (row0,col0)=7 ; (20,20) -> (row2,col2)=12 ; (10,10) center -> (row1,col1)=9
    got = r.ids_at_world([0.0, 20.0, 10.0], [0.0, 20.0, 10.0])
    assert list(got) == [7, 12, 9]
    # out of bounds (beyond half a cell past the extent) -> -1
    assert list(r.ids_at_world([-15.0, 35.0], [10.0, 10.0])) == [-1, -1]
    # normalized entry: on = (world - center) * scale
    on = np.array([[(20.0 - 10.0) * 1.0, (0.0 - 10.0) * 1.0]])  # world (20,0) -> (row0,col2)=8
    assert int(r.ids_at_norm(on)[0]) == 8


def test_city_material_ids_paints_terrain_and_buildings():
    from grl_snam.material import city_material_ids

    truth = np.zeros((48, 48), dtype=bool)
    truth[10:20, 10:20] = True  # a building block
    r = city_material_ids(
        truth, bounds=(0.0, 0.0, 47.0, 47.0), center=(23.5, 23.5), scale=1.0, seed=1
    )
    assert r.ids[15, 15] == MATERIAL_ID["reinforced_concrete"]  # building -> concrete
    present = set(np.unique(r.ids).tolist())
    assert MATERIAL_ID["open_air"] in present
    assert present & set(RISK_MATERIAL_IDS)  # at least one terrain-risk blob painted
    # buildings never overwritten by a blob
    assert (r.ids[truth] == MATERIAL_ID["reinforced_concrete"]).all()


def test_swarm_collects_material_buckets():
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.material import city_material_ids
    from grl_snam.squad import AgentSpec
    from grl_snam.swarm import Swarm

    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    truth = story.truth_grid()
    meta = story.meta()
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    free_r, free_c = np.nonzero(~truth)
    rng = np.random.default_rng(0)
    specs = []
    for i in range(8):
        s = rng.integers(0, len(free_r))
        g = rng.integers(0, len(free_r))
        specs.append(AgentSpec(f"a{i}", w(free_r[s], free_c[s]), w(free_r[g], free_c[g])))

    mids = city_material_ids(truth, story.bounds, meta["center"], meta["scale"], seed=0)
    sw = Swarm(
        story, specs, truth_occ=truth, prior_occ=truth, collect_stats=True, material_ids=mids
    )
    for _ in range(40):
        sw.step()
    ep = sw.episode_stats()
    total_time = sum(sum(v.time_over_material_s) for v in ep.per_vehicle)
    assert total_time > 0.0  # something was bucketed
    # each vehicle's material time ~ its collected steps * dt (open_air dominates)
    for v in ep.per_vehicle:
        assert min(v.time_over_material_s) >= 0.0
        assert v.time_over_material_s[MATERIAL_ID["open_air"]] > 0.0


def test_swarm_without_material_ids_has_zero_buckets():
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.squad import AgentSpec
    from grl_snam.swarm import Swarm

    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    truth = story.truth_grid()
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape
    free_r, free_c = np.nonzero(~truth)
    rng = np.random.default_rng(0)

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    specs = [
        AgentSpec(
            f"a{i}",
            w(free_r[rng.integers(0, len(free_r))], free_c[rng.integers(0, len(free_c))]),
            w(free_r[rng.integers(0, len(free_r))], free_c[rng.integers(0, len(free_c))]),
        )
        for i in range(6)
    ]
    sw = Swarm(story, specs, truth_occ=truth, prior_occ=truth, collect_stats=True)
    for _ in range(20):
        sw.step()
    ep = sw.episode_stats()
    for v in ep.per_vehicle:
        assert v.time_over_material_s == [0.0] * NUM_MATERIALS
        assert v.dist_over_material_m == [0.0] * NUM_MATERIALS
