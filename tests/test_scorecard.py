"""Base nav scorecard — aggregator parity with the C++ cvc::nav::nav_scorecard.

The corpus + every expected value here is byte-identical to transfix/libcvc
src/cvc/tests/nav_stats_test.cpp, so the two languages produce the same base
scorecard from the same inputs (the shared-schema contract, gated like the other
grl-snam <-> cvc::nav parity tests).
"""

import math

from grl_snam.material_palette import NUM_MATERIALS
from grl_snam.metrics import NavMetrics, NavStats
from grl_snam.scorecard import (
    EpisodeStats,
    NavCoverage,
    NavScorecard,
    VehStats,
    aggregate_nav,
    compute_coverage,
)


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


def test_navstats_seed_start_counts_first_segment():
    # C++ parity: seeding the start pose makes the first update count the start->first-step
    # leg in total_path_m, while turn/fuel still skip the first sample (prev_head None). Without
    # the seed the first segment is dropped (the prior, HUD-compatible behavior).
    seeded, plain = NavStats(), NavStats()
    seeded.seed_start(0.0, 0.0)
    for ns in (seeded, plain):
        ns.update(NavMetrics(x=3.0, y=0.0, heading_rad=0.0, speed_mps=5.0))
        ns.update(NavMetrics(x=3.0, y=4.0, heading_rad=0.0, speed_mps=5.0))
    assert math.isclose(seeded.total_path_m, 7.0)  # 3 (start->m0) + 4 (m0->m1)
    assert math.isclose(plain.total_path_m, 4.0)  # first leg dropped
    # turn/fuel skip the first sample either way — seeding only affects the path.
    assert seeded.turn_total_rad == plain.turn_total_rad == 0.0
    assert seeded.fuel_used == plain.fuel_used == 0.0


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
        [v0, v1],
        straights_m=[50.0, 50.0],
        arrival_times_s=[12.0, -1.0],
        veh_contacts=[0, 2],
        min_sep_m=8.0,
    )
    assert ep.per_vehicle[0].arrived and not ep.per_vehicle[1].arrived
    assert not ep.success  # not all arrived
    assert math.isclose(ep.min_sep_m, 8.0)
    # veh_contacts flows through the builder (NavStats doesn't carry it), so the
    # scorecard sees real contacts instead of a silent 0.
    assert ep.per_vehicle[1].veh_contacts == 2
    sc = aggregate_nav([ep], "c")
    assert math.isclose(sc.arrival_rate, 0.5)
    assert math.isclose(sc.veh_contacts_per_run, 1.0)  # (0 + 2) / 2 runs


# --- Track 1 parity: the same corpora + values as the C++ nav_stats_test feature-ON cases ---


def test_formation_scorecard_matches_cpp():
    # Mirrors nav_stats_test.cpp::FormationSlotAndScorecard (scorecard half): convoy 0, anchor arrived,
    # one follower. e0 follower in-slot (slot_error 1.5), e1 follower out (slot_error 20, not arrived).
    def anchor():
        return VehStats(arrived=True, convoy_id=0, formation_parent=-1)

    def follower(arrived_slot, slot_err):
        return VehStats(
            convoy_id=0,
            formation_parent=0,
            formation_arrived=arrived_slot,
            slot_error_mean_m=slot_err,
        )

    e0 = EpisodeStats(per_vehicle=[anchor(), follower(True, 1.5)])
    e1 = EpisodeStats(per_vehicle=[anchor(), follower(False, 20.0)])
    s = aggregate_nav([e0, e1], "ckpt-F")
    assert math.isclose(s.form_arrival_rate, 0.5)  # 1 of 2 followers in-slot
    assert math.isclose(s.form_mission_rate, 0.5)  # e0 ok, e1 not
    assert math.isclose(s.mean_slot_error_m, 10.75)  # (1.5 + 20)/2


def test_progress_scorecard_matches_cpp():
    # Mirrors ProgressStallAndClosestApproach (scorecard half): closest 0/4, stall 2/6.
    e0 = EpisodeStats(per_vehicle=[VehStats(closest_approach_m=0.0, stall_steps=2)])
    e1 = EpisodeStats(per_vehicle=[VehStats(closest_approach_m=4.0, stall_steps=6)])
    s = aggregate_nav([e0, e1], "ckpt-P")
    assert math.isclose(s.mean_stall_steps, 4.0)  # (2+6)/2
    assert math.isclose(s.mean_closest_approach_m, 2.0)  # (0+4)/2


def test_compute_coverage_matches_cpp():
    # Mirrors CoverageReducerAndSenseFlips (reducer half): 2 planes x 2x2 grid.
    truth = [0, 1, 0, 0]
    belief = [0, 1, 1, 0, 0, 0, 0, 1]  # plane0 then plane1
    everseen = [1, 1, 1, 0, 1, 1, 0, 0]  # 5 of 8
    lastvis = [1, 0, 0, 0, 1, 0, 0, 0]  # 2 of 8
    cov = compute_coverage(truth, belief, everseen, lastvis, planes=2, cells=4)
    assert math.isclose(cov.explored_frac, 0.625)  # 5/8
    assert math.isclose(cov.visible_frac, 0.25)  # 2/8
    assert math.isclose(cov.believed_free_frac, 0.625)  # 5 belief==0 cells / 8
    assert math.isclose(cov.phantom_frac, 0.25)  # 2/8 believed-occ where truth free
    # bad-arg guard -> all zero
    off = compute_coverage(None, belief, everseen, lastvis, 2, 4)
    assert off.explored_frac == 0.0 and off.phantom_frac == 0.0


def test_coverage_and_sense_flips_scorecard_matches_cpp():
    # Mirrors CoverageReducerAndSenseFlips (scorecard half): mean_coverage 0.5 each, mean_sense_flips 5.
    e0 = EpisodeStats(
        coverage=NavCoverage(0.625, 0.25, 0.625, 0.25), per_vehicle=[VehStats(sense_flips=8)]
    )
    e1 = EpisodeStats(
        coverage=NavCoverage(0.375, 0.75, 0.375, 0.75),
        per_vehicle=[VehStats(arrived=True, sense_flips=2)],
    )
    s = aggregate_nav([e0, e1], "ckpt-C")
    assert math.isclose(s.mean_coverage.explored_frac, 0.5)  # (0.625+0.375)/2
    assert math.isclose(s.mean_coverage.visible_frac, 0.5)  # (0.25+0.75)/2
    assert math.isclose(s.mean_coverage.believed_free_frac, 0.5)
    assert math.isclose(s.mean_coverage.phantom_frac, 0.5)
    assert math.isclose(s.mean_sense_flips, 5.0)  # (8+2)/2


def test_drive_telemetry_scorecard_matches_cpp():
    # Mirrors DriveTelemetryReduction (scorecard half): one drive-carrying run.
    v = VehStats(
        drive_steps=3,
        alpha_mean=2.0,
        beta_mean=4.0,
        gamma_mean=2.0,
        mu_mean=0.5,
        mrisk_mean=0.5,
        ext_force_mean=2.0,
    )
    s = aggregate_nav([EpisodeStats(per_vehicle=[v])], "ckpt-D")
    assert math.isclose(s.mean_alpha, 2.0)
    assert math.isclose(s.mean_beta, 4.0)
    assert math.isclose(s.mean_gamma, 2.0)
    assert math.isclose(s.mean_mu, 0.5)
    assert math.isclose(s.mean_mrisk, 0.5)
    assert math.isclose(s.mean_ext_force, 2.0)


def test_scorecard_json_has_track1_keys():
    s = aggregate_nav([EpisodeStats(success=True, per_vehicle=[VehStats(arrived=True)])], "c")
    d = s.to_dict()
    for k in (
        "form_arrival_rate",
        "form_mission_rate",
        "mean_slot_error_m",
        "mean_closest_approach_m",
        "mean_stall_steps",
        "mean_coverage",
        "mean_sense_flips",
        "mean_alpha",
        "mean_mu",
        "mean_ext_force",
    ):
        assert k in d
    assert set(d["mean_coverage"]) == {
        "explored_frac",
        "visible_frac",
        "believed_free_frac",
        "phantom_frac",
    }


# --- Track 4 Phase 0: the scorecard reader (inverse of to_dict/to_json; loads C++ scorecard_json) ---


def test_scorecard_reader_round_trips():
    # A fully-populated scorecard (all Track-1 field families non-default) survives to_dict->from_dict
    # and to_json->from_json unchanged. Since to_dict's keys are field-for-field the C++
    # cvc::nav::scorecard_json keys, this is also the guarantee that a recorded C++ scorecard row loads.
    e0 = EpisodeStats(
        success=True,
        makespan_s=20,
        penetration_pct=3,
        min_sep_m=5,
        coverage=NavCoverage(0.6, 0.3, 0.6, 0.2),
        per_vehicle=[
            VehStats(
                arrived=True,
                convoy_id=0,
                formation_parent=-1,
                sense_flips=4,
                drive_steps=2,
                alpha_mean=1.0,
                beta_mean=2.0,
                gamma_mean=1.0,
                mu_mean=0.5,
                mrisk_mean=0.25,
                ext_force_mean=1.5,
            ),
            VehStats(
                arrived=True,
                convoy_id=0,
                formation_parent=0,
                formation_arrived=True,
                slot_error_mean_m=1.5,
                closest_approach_m=3.0,
                stall_steps=2,
            ),
        ],
    )
    sc = aggregate_nav([e0], "ckpt-RT")
    d = sc.to_dict()
    assert NavScorecard.from_dict(d).to_dict() == d  # dict round-trip is the exact inverse
    assert NavScorecard.from_json(sc.to_json()).to_dict() == d  # JSON round-trip too


def test_scorecard_reader_tolerates_partial_and_ignores_rf():
    # A partial producer (missing keys) loads with dataclass defaults; an embedded "rf" sub-object
    # (the DBG scorecard_json shape) is ignored by the base reader.
    sc = NavScorecard.from_dict(
        {"checkpoint": "x", "success_rate": 0.5, "rf": {"outage_rate": 0.9}}
    )
    assert sc.checkpoint == "x"
    assert math.isclose(sc.success_rate, 0.5)
    assert sc.n_episodes == 0  # missing scalar -> default
    assert math.isclose(sc.form_arrival_rate, 0.0)  # missing Track-1 field -> default
    assert math.isclose(sc.mean_coverage.phantom_frac, 0.0)  # missing nested -> default
    # missing list field -> the length-NUM_MATERIALS zero default, NOT [] (so share[m] never IndexErrors)
    assert sc.material_time_share == [0.0] * NUM_MATERIALS
    assert not hasattr(sc, "rf")  # rf sub-object dropped, not smuggled onto the base row
