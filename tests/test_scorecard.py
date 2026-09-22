"""Base nav scorecard — aggregator parity with the C++ cvc::dbg::nav_scorecard.

The corpus + every expected value here is byte-identical to CVC-DBG/cvcdbg
tests/nav_stats_test.cpp::test_scorecard, so the two languages produce the same base
scorecard from the same inputs (the shared-schema contract, gated like the other
grl-snam <-> cvc::nav parity tests).
"""

import math

from grl_snam.metrics import NavMetrics, NavStats
from grl_snam.scorecard import EpisodeStats, VehStats, aggregate_nav


def _v(arrived, ttg, path, straight, turn, fuel, contacts):
    return VehStats(
        arrived=arrived,
        time_to_goal_s=ttg,
        total_path_m=path,
        straight_m=straight,
        turn_total_rad=turn,
        fuel_used=fuel,
        veh_contacts=contacts,
    )


def test_aggregate_nav_matches_cpp():
    e0 = EpisodeStats(
        success=True,
        makespan_s=20,
        penetration_pct=0,
        min_sep_m=5,
        per_vehicle=[_v(True, 10, 100, 100, 2, 5, 0), _v(True, 20, 150, 100, 4, 7, 1)],
    )
    e1 = EpisodeStats(
        success=False,
        makespan_s=30,
        penetration_pct=10,
        min_sep_m=3,
        per_vehicle=[_v(True, 30, 200, 100, 6, 9, 0), _v(False, -1, 120, 100, 3, 8, 2)],
    )
    s = aggregate_nav([e0, e1], "ckpt-42")

    assert s.checkpoint == "ckpt-42"
    assert s.n_episodes == 2 and s.n_vehicle_runs == 4
    assert math.isclose(s.success_rate, 0.5)
    assert math.isclose(s.arrival_rate, 0.75)
    assert math.isclose(s.mean_time_to_goal_s, 20.0)  # (10+20+30)/3
    assert math.isclose(s.p95_time_to_goal_s, 30.0)
    assert math.isclose(s.mean_makespan_s, 25.0)
    assert math.isclose(s.mean_path_ratio, 1.425)  # (1.0+1.5+2.0+1.2)/4
    assert math.isclose(s.mean_turn_total_rad, 3.75)  # (2+4+6+3)/4
    assert math.isclose(s.mean_fuel, 7.25)  # (5+7+9+8)/4
    assert math.isclose(s.mean_penetration_pct, 5.0)
    assert math.isclose(s.veh_contacts_per_run, 0.75)  # (0+1+0+2)/4
    assert math.isclose(s.mean_min_sep_m, 4.0)


def test_scorecard_json_keys():
    s = aggregate_nav([EpisodeStats(success=True, per_vehicle=[_v(True, 1, 1, 1, 0, 0, 0)])], "c")
    d = s.to_dict()
    for k in (
        "checkpoint",
        "success_rate",
        "arrival_rate",
        "mean_time_to_goal_s",
        "mean_path_ratio",
        "material_time_share",
    ):
        assert k in d
    assert "rf" not in d  # base-only shape (grl-snam-non-DBG)


def test_navstats_accumulates_turn_and_fuel():
    # heading 0 -> pi/2 -> pi/2 ; speed 10 -> 10 -> 0 : turn = pi/2, fuel = |10-10|+|0-10| = 10
    ns = NavStats()
    ns.update(NavMetrics(heading_rad=0.0, speed_mps=10.0))
    ns.update(NavMetrics(heading_rad=math.pi / 2, speed_mps=10.0))
    ns.update(NavMetrics(heading_rad=math.pi / 2, speed_mps=0.0))
    assert math.isclose(ns.turn_total_rad, math.pi / 2, abs_tol=1e-9)
    assert math.isclose(ns.fuel_used, 10.0, abs_tol=1e-9)


def test_episode_from_nav_stats():
    # Two vehicles driven; both reach a goal. from_nav_stats maps NavStats -> EpisodeStats.
    def drive(path_headings, reached_goal):
        ns = NavStats()
        for h, s in path_headings:
            m = NavMetrics(heading_rad=h, speed_mps=s)
            if reached_goal:
                m.reached = True
                m.goal_index = 0
            ns.update(m)
        return ns

    v0 = drive([(0.0, 5.0), (0.0, 5.0)], reached_goal=True)
    v1 = drive([(0.0, 5.0), (0.0, 5.0)], reached_goal=False)
    ep = EpisodeStats.from_nav_stats(
        [v0, v1], straights_m=[50.0, 50.0], arrival_times_s=[12.0, -1.0], min_sep_m=8.0
    )
    assert ep.per_vehicle[0].arrived and not ep.per_vehicle[1].arrived
    assert not ep.success  # not all arrived
    assert math.isclose(ep.min_sep_m, 8.0)
    sc = aggregate_nav([ep], "c")
    assert math.isclose(sc.arrival_rate, 0.5)
