"""Scan-local waypoint selection using geometry and observed material."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MaterialProfile",
    "MaterialWaypointParams",
    "choose_material_local_waypoint",
]

MaterialProfile = Callable[[float, float, float], tuple[float, float | None]]


@dataclass(frozen=True)
class MaterialWaypointParams:
    """Weights and distance limits for scan-local waypoint selection."""

    horizon_m: float = 25.0
    wall_margin_m: float = 2.0
    risk_weight: float = 1.0
    progress_weight: float = 1.0
    reach_weight: float = 1.6
    smoothness_weight: float = 0.15
    min_usable_m: float = 0.75

    def __post_init__(self) -> None:
        positive = {
            "horizon_m": self.horizon_m,
            "min_usable_m": self.min_usable_m,
        }
        nonnegative = {
            "wall_margin_m": self.wall_margin_m,
            "risk_weight": self.risk_weight,
            "progress_weight": self.progress_weight,
            "reach_weight": self.reach_weight,
            "smoothness_weight": self.smoothness_weight,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")


def _point(name: str, value) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain two finite coordinates")
    return point


def choose_material_local_waypoint(
    pos_xy,
    vel_xy,
    final_goal_xy,
    ray_hit_distances,
    *,
    sensor_radius_m: float,
    material_profile: MaterialProfile,
    params: MaterialWaypointParams = MaterialWaypointParams(),
) -> tuple[tuple[float, float], dict[str, float | bool]]:
    """Choose one scan-ray endpoint using geometry and observed material.

    Ray zero points along world ``+x`` and subsequent rays advance
    counter-clockwise at equal angular spacing. ``material_profile(ux, uy,
    distance)`` returns the mean *observed* risk before the first blocked cell
    and that cell's distance, or ``None`` when the ray remains feasible. The
    callback boundary deliberately gives this scorer no global material map.
    """
    pos = _point("pos_xy", pos_xy)
    velocity = _point("vel_xy", vel_xy)
    goal = _point("final_goal_xy", final_goal_xy)
    hit_distances = np.asarray(ray_hit_distances, dtype=np.float64)
    if hit_distances.ndim != 1 or len(hit_distances) == 0:
        raise ValueError("ray_hit_distances must be a non-empty vector")
    if not np.isfinite(hit_distances).all() or (hit_distances < 0.0).any():
        raise ValueError("ray_hit_distances must be finite and >= 0")
    if not math.isfinite(sensor_radius_m) or sensor_radius_m <= 0.0:
        raise ValueError("sensor_radius_m must be finite and > 0")

    goal_delta = goal - pos
    goal_distance = float(np.linalg.norm(goal_delta))
    if goal_distance < 1e-9:
        return tuple(float(v) for v in goal), {
            "mean_risk": 0.0,
            "usable_m": 0.0,
            "material_limited": False,
        }

    goal_u = goal_delta / goal_distance
    speed = float(np.linalg.norm(velocity))
    heading_u = velocity / speed if speed > 1e-6 else goal_u
    free_horizon = min(
        params.horizon_m,
        max(params.min_usable_m, sensor_radius_m - params.wall_margin_m),
        goal_distance,
    )

    candidates: list[tuple[float, float, float, float, float, bool]] = []
    for ray_index, wall_distance in enumerate(hit_distances):
        angle = 2.0 * math.pi * ray_index / len(hit_distances)
        ux, uy = math.cos(angle), math.sin(angle)
        geometry_usable = min(
            free_horizon,
            max(0.0, float(wall_distance) - params.wall_margin_m),
        )
        if geometry_usable < params.min_usable_m:
            continue

        mean_risk, material_stop = material_profile(ux, uy, geometry_usable)
        mean_risk = float(mean_risk)
        if not math.isfinite(mean_risk) or not 0.0 <= mean_risk <= 1.0:
            raise ValueError("material_profile mean risk must be finite and in [0, 1]")
        usable = geometry_usable
        material_limited = material_stop is not None
        if material_limited:
            material_stop = float(material_stop)
            if not math.isfinite(material_stop) or material_stop < 0.0:
                raise ValueError("material_profile stop distance must be finite and >= 0")
            usable = min(usable, max(0.0, material_stop - params.wall_margin_m))
        if usable < params.min_usable_m:
            continue

        progress = ux * goal_u[0] + uy * goal_u[1]
        smoothness = ux * heading_u[0] + uy * heading_u[1]
        score = (
            params.progress_weight * progress
            + params.reach_weight * (usable / free_horizon)
            + params.smoothness_weight * smoothness
            - params.risk_weight * mean_risk
        )
        candidates.append((score, usable, ux, uy, mean_risk, material_limited))

    if not candidates:
        waypoint = pos + params.min_usable_m * goal_u
        return tuple(float(v) for v in waypoint), {
            "mean_risk": 0.0,
            "usable_m": params.min_usable_m,
            "material_limited": False,
        }

    _score, usable, ux, uy, mean_risk, material_limited = max(candidates, key=lambda row: row[0])
    waypoint = pos + usable * np.asarray((ux, uy))
    return tuple(float(v) for v in waypoint), {
        "mean_risk": mean_risk,
        "usable_m": float(usable),
        "material_limited": bool(material_limited),
    }
