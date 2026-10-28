"""Paired reporting for material-controller route evaluations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields

import numpy as np

__all__ = [
    "MATERIAL_EVALUATION_ARMS",
    "MaterialRouteMetrics",
    "aggregate_material_evaluation_reports",
    "material_route_metrics_from_dict",
    "paired_route_statistics",
    "select_waypoint_risk_weight",
    "summarize_material_routes",
]

MATERIAL_EVALUATION_ARMS = ("fixed_lambda", "force_only", "waypoint", "learned")
_METRICS = (
    "success",
    "collision",
    "timeout",
    "risk_per_m",
    "hard_hazard_distance_m",
    "hard_hazard_step_fraction",
    "hard_hazard_free_success",
    "direct_efficiency",
    "path_length_m",
    "prediction_latency_ms",
    "gate_active_fraction",
)


@dataclass(frozen=True)
class MaterialRouteMetrics:
    """One arm on one frozen route under one trained model seed."""

    seed: int
    route_id: int
    split: str
    challenge_kind: str
    arm: str
    success: bool
    collision: bool
    timeout: bool
    direct_distance_m: float
    path_length_m: float
    risk_distance: float
    hard_hazard_distance_m: float
    steps: int
    hard_hazard_steps: int
    gate_active_steps: int
    prediction_latency_ms: float

    def __post_init__(self) -> None:
        if self.arm not in MATERIAL_EVALUATION_ARMS:
            raise ValueError(f"unknown material evaluation arm {self.arm!r}")
        if self.split not in ("train", "validation", "test"):
            raise ValueError(f"unknown route split {self.split!r}")
        if sum((self.success, self.collision, self.timeout)) != 1:
            raise ValueError("exactly one of success, collision, or timeout must be true")
        if self.steps < 0 or not 0 <= self.hard_hazard_steps <= self.steps:
            raise ValueError("hard_hazard_steps must be in [0, steps]")
        if not 0 <= self.gate_active_steps <= self.steps:
            raise ValueError("gate_active_steps must be in [0, steps]")
        scalars = (
            self.direct_distance_m,
            self.path_length_m,
            self.risk_distance,
            self.hard_hazard_distance_m,
            self.prediction_latency_ms,
        )
        if not np.isfinite(scalars).all() or any(value < 0.0 for value in scalars):
            raise ValueError("route metrics must be finite and non-negative")

    @property
    def risk_per_m(self) -> float:
        return self.risk_distance / max(self.path_length_m, 1e-9)

    @property
    def hard_hazard_step_fraction(self) -> float:
        return self.hard_hazard_steps / max(self.steps, 1)

    @property
    def hard_hazard_free_success(self) -> bool:
        return self.success and self.hard_hazard_distance_m <= 1e-9

    @property
    def direct_efficiency(self) -> float:
        if not self.success:
            return 0.0
        return self.direct_distance_m / max(self.path_length_m, self.direct_distance_m)

    @property
    def gate_active_fraction(self) -> float:
        return self.gate_active_steps / max(self.steps, 1)

    def to_dict(self) -> dict:
        row = asdict(self)
        for name in (
            "risk_per_m",
            "hard_hazard_step_fraction",
            "hard_hazard_free_success",
            "direct_efficiency",
            "gate_active_fraction",
        ):
            row[name] = getattr(self, name)
        return row


def _value(row: MaterialRouteMetrics, name: str) -> float:
    return float(getattr(row, name))


def summarize_material_routes(rows: Iterable[MaterialRouteMetrics]) -> dict:
    """Aggregate one arm's route rows without discarding the paired rows."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot summarize an empty material evaluation")
    arms = {row.arm for row in rows}
    if len(arms) != 1:
        raise ValueError("summary rows must contain exactly one arm")
    summary = {"arm": rows[0].arm, "n": len(rows), "seeds": sorted({row.seed for row in rows})}
    for name in _METRICS:
        summary[name] = float(np.mean([_value(row, name) for row in rows]))
    summary["successful_path_length_m"] = (
        float(np.mean([row.path_length_m for row in rows if row.success]))
        if any(row.success for row in rows)
        else None
    )
    return summary


def material_route_metrics_from_dict(row: Mapping[str, object]) -> MaterialRouteMetrics:
    """Rebuild a route row while ignoring its serialized derived metrics."""
    names = {field.name for field in fields(MaterialRouteMetrics)}
    missing = names - set(row)
    if missing:
        raise ValueError(f"material route row is missing {sorted(missing)}")
    return MaterialRouteMetrics(**{name: row[name] for name in names})


def aggregate_material_evaluation_reports(
    reports: Iterable[Mapping[str, object]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20261015,
) -> dict:
    """Combine independently trained seed reports without losing route pairing."""
    reports = list(reports)
    if not reports:
        raise ValueError("at least one material evaluation report is required")
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("material evaluation reports must have unique training seeds")
    rows = [
        material_route_metrics_from_dict(row)
        for report in reports
        for row in report.get("rows", [])
    ]
    if not rows or any(row.split != "test" for row in rows):
        raise ValueError("aggregate material reports must contain held-out test rows only")
    result = {
        "seed_count": len(seeds),
        "seeds": sorted(seeds),
        "selected_waypoint_risk_weight": {
            str(report["seed"]): float(report["selected_waypoint_risk_weight"])
            for report in reports
        },
        "summary": {
            arm: summarize_material_routes(row for row in rows if row.arm == arm)
            for arm in MATERIAL_EVALUATION_ARMS
        },
        "paired": paired_route_statistics(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "rows": [row.to_dict() for row in rows],
    }
    return result


def select_waypoint_risk_weight(
    validation_rows: Mapping[float, Iterable[MaterialRouteMetrics]],
) -> tuple[float, dict]:
    """Choose one waypoint weight using validation rows from the learned arm only.

    The predeclared ordering is lexicographic: success, collision avoidance,
    hard-hazard-free success, timeout avoidance, hard distance, risk per metre,
    direct efficiency, then the smaller weight.
    """
    if not validation_rows:
        raise ValueError("at least one waypoint-risk candidate is required")
    summaries = {}
    for weight, candidate_rows in validation_rows.items():
        rows = list(candidate_rows)
        if not rows or any(row.split != "validation" for row in rows):
            raise ValueError("waypoint weight selection may use validation rows only")
        if any(row.arm != "learned" for row in rows):
            raise ValueError("waypoint weight selection must use the learned arm")
        summaries[float(weight)] = summarize_material_routes(rows)

    def key(item):
        weight, summary = item
        return (
            summary["success"],
            -summary["collision"],
            summary["hard_hazard_free_success"],
            -summary["timeout"],
            -summary["hard_hazard_distance_m"],
            -summary["risk_per_m"],
            summary["direct_efficiency"],
            -weight,
        )

    selected = max(summaries.items(), key=key)[0]
    return selected, {str(weight): summaries[weight] for weight in sorted(summaries)}


def _sign_test_two_sided(values: np.ndarray) -> float:
    nonzero = values[values != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = min(int((nonzero > 0.0).sum()), int((nonzero < 0.0).sum()))
    tail = sum(math.comb(n, index) for index in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_route_statistics(
    rows: Iterable[MaterialRouteMetrics],
    *,
    baseline: str = "fixed_lambda",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20261015,
) -> dict:
    """Paired per-route deltas and route-cluster bootstrap confidence intervals.

    Seed-route observations are paired first. Their deltas are then averaged
    within route, and the 95% interval resamples routes, keeping all training
    seeds for a route together.
    """
    rows = list(rows)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    by_key = {(row.seed, row.route_id, row.arm): row for row in rows}
    route_ids = sorted({row.route_id for row in rows})
    seeds = sorted({row.seed for row in rows})
    arms = sorted({row.arm for row in rows})
    if baseline not in arms:
        raise ValueError(f"baseline arm {baseline!r} is absent")
    expected = {(seed, route_id, arm) for seed in seeds for route_id in route_ids for arm in arms}
    missing = expected - set(by_key)
    if missing:
        raise ValueError(f"paired evaluation is incomplete; missing {len(missing)} rows")

    rng = np.random.default_rng(bootstrap_seed)
    report = {
        "baseline": baseline,
        "route_count": len(route_ids),
        "seed_count": len(seeds),
        "arms": {},
    }
    for arm in arms:
        if arm == baseline:
            continue
        per_route = []
        matrix = np.empty((len(route_ids), len(_METRICS)), np.float64)
        for route_index, route_id in enumerate(route_ids):
            record = {"route_id": route_id}
            for metric_index, metric in enumerate(_METRICS):
                deltas = [
                    _value(by_key[(seed, route_id, arm)], metric)
                    - _value(by_key[(seed, route_id, baseline)], metric)
                    for seed in seeds
                ]
                delta = float(np.mean(deltas))
                matrix[route_index, metric_index] = delta
                record[metric] = delta
            per_route.append(record)

        indices = rng.integers(0, len(route_ids), size=(bootstrap_samples, len(route_ids)))
        boot = matrix[indices].mean(axis=1)
        metrics = {}
        for metric_index, metric in enumerate(_METRICS):
            values = matrix[:, metric_index]
            metrics[metric] = {
                "mean_delta": float(values.mean()),
                "ci95": [
                    float(np.quantile(boot[:, metric_index], 0.025)),
                    float(np.quantile(boot[:, metric_index], 0.975)),
                ],
                "paired_sign_test_p": _sign_test_two_sided(values),
            }
        report["arms"][arm] = {"metrics": metrics, "per_route": per_route}
    return report
