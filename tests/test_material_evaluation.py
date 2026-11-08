import pytest

from grl_snam.material_evaluation import (
    MaterialRouteMetrics,
    aggregate_material_evaluation_reports,
    paired_route_statistics,
    select_waypoint_risk_weight,
    summarize_material_routes,
)


def _row(route_id, arm, *, seed=1, split="test", success=True, risk=1.0, hard=0.0):
    return MaterialRouteMetrics(
        seed=seed,
        route_id=route_id,
        split=split,
        challenge_kind="soft",
        arm=arm,
        success=success,
        collision=False,
        timeout=not success,
        direct_distance_m=10.0,
        path_length_m=12.0,
        risk_distance=risk,
        hard_hazard_distance_m=hard,
        steps=4,
        hard_hazard_steps=int(hard > 0.0),
        gate_active_steps=2,
        prediction_latency_ms=1.0,
    )


def test_summary_keeps_derived_material_metrics():
    summary = summarize_material_routes([_row(1, "learned"), _row(2, "learned", hard=2.0)])

    assert summary["success"] == 1.0
    assert summary["hard_hazard_free_success"] == 0.5
    assert summary["gate_active_fraction"] == 0.5


def test_waypoint_weight_selection_uses_validation_only():
    candidates = {
        0.5: [_row(1, "learned", split="validation", risk=3.0)],
        1.0: [_row(1, "learned", split="validation", risk=1.0)],
    }

    selected, summaries = select_waypoint_risk_weight(candidates)

    assert selected == 1.0
    assert set(summaries) == {"0.5", "1.0"}
    with pytest.raises(ValueError, match="validation"):
        select_waypoint_risk_weight({1.0: [_row(1, "learned")]})


def test_paired_statistics_cluster_training_seeds_by_route():
    rows = []
    for seed in (1, 2, 3):
        for route_id in (10, 11):
            rows.append(_row(route_id, "fixed_lambda", seed=seed, risk=3.0))
            rows.append(_row(route_id, "learned", seed=seed, risk=1.0))

    report = paired_route_statistics(rows, bootstrap_samples=200, bootstrap_seed=7)

    assert report["route_count"] == 2
    assert report["seed_count"] == 3
    metric = report["arms"]["learned"]["metrics"]["risk_per_m"]
    assert metric["mean_delta"] == pytest.approx(-2.0 / 12.0)
    assert metric["ci95"] == pytest.approx([-2.0 / 12.0, -2.0 / 12.0])


def test_report_aggregation_keeps_seed_route_pairing():
    reports = []
    for seed in (1, 2, 3):
        rows = []
        for route_id in (10, 11):
            for arm in ("fixed_lambda", "force_only", "waypoint", "learned"):
                risk = 3.0 if arm == "fixed_lambda" else 1.0
                rows.append(_row(route_id, arm, seed=seed, risk=risk).to_dict())
        reports.append(
            {
                "seed": seed,
                "selected_waypoint_risk_weight": 1.0,
                "rows": rows,
            }
        )

    aggregate = aggregate_material_evaluation_reports(
        reports,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert aggregate["seeds"] == [1, 2, 3]
    assert aggregate["summary"]["learned"]["n"] == 6
    assert aggregate["paired"]["route_count"] == 2
    assert aggregate["paired"]["seed_count"] == 3
    assert aggregate["paired"]["arms"]["learned"]["metrics"]["risk_per_m"][
        "mean_delta"
    ] == pytest.approx(-2.0 / 12.0)
