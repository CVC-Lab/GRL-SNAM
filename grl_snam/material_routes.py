"""Deterministic material-challenge route corpus generation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import planner, sdf_nav
from .material_palette import MATERIALS

__all__ = [
    "MaterialRouteGenerationParams",
    "generate_material_route_manifest",
    "validate_material_route_manifest",
    "write_material_route_manifest",
]

_SPLITS = ("train", "validation", "test")
_SCHEMA = "grl-snam-material-routes/1"


@dataclass(frozen=True)
class MaterialRouteGenerationParams:
    """Selection limits shared by public material-route corpora."""

    min_distance_m: float = 100.0
    max_distance_m: float = 220.0
    endpoint_clearance_m: float = 5.0
    min_endpoint_separation_m: float = 60.0
    risk_threshold: float = 0.30
    max_detour_ratio: float = 2.5
    max_attempts_per_route: int = 100_000
    first_route_id: int = 400

    def __post_init__(self) -> None:
        finite_nonnegative = {
            "min_distance_m": self.min_distance_m,
            "endpoint_clearance_m": self.endpoint_clearance_m,
            "min_endpoint_separation_m": self.min_endpoint_separation_m,
        }
        for name, value in finite_nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not math.isfinite(self.max_distance_m) or self.max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be finite and > 0")
        if self.max_distance_m < self.min_distance_m:
            raise ValueError("max_distance_m must be >= min_distance_m")
        if not math.isfinite(self.risk_threshold) or not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be finite and in [0, 1]")
        if not math.isfinite(self.max_detour_ratio) or self.max_detour_ratio < 1.0:
            raise ValueError("max_detour_ratio must be finite and >= 1")
        if self.max_attempts_per_route < 1:
            raise ValueError("max_attempts_per_route must be >= 1")
        if self.first_route_id < 0:
            raise ValueError("first_route_id must be >= 0")


def _split_counts(name: str, values: Mapping[str, int] | None) -> dict[str, int]:
    if values is None:
        return {split: 0 for split in _SPLITS}
    if set(values) != set(_SPLITS):
        raise ValueError(f"{name} must have exactly the keys {', '.join(_SPLITS)}")
    result = {split: int(values[split]) for split in _SPLITS}
    if any(result[split] < 0 or result[split] != values[split] for split in _SPLITS):
        raise ValueError(f"{name} values must be non-negative integers")
    return result


def _bounds(bounds) -> tuple[float, float, float, float]:
    result = tuple(float(value) for value in bounds)
    if len(result) != 4 or not all(math.isfinite(value) for value in result):
        raise ValueError("bounds must contain four finite values")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("bounds must have positive width and height")
    return result


def _line_cells(a, b) -> tuple[np.ndarray, np.ndarray]:
    # ``a`` and ``b`` are grid indices, so sample in grid-cell units. Dividing
    # their distance by metres per cell undersamples coarse rasters and can skip
    # every material cell between the endpoints.
    distance_cells = max(abs(int(b[0]) - int(a[0])), abs(int(b[1]) - int(a[1])))
    samples = max(2, distance_cells + 1)
    rr = np.rint(np.linspace(a[0], b[0], samples)).astype(np.intp)
    cc = np.rint(np.linspace(a[1], b[1], samples)).astype(np.intp)
    return rr, cc


def _cell_to_world(cell, bounds, shape) -> tuple[float, float]:
    row, col = int(cell[0]), int(cell[1])
    min_x, min_y, max_x, max_y = bounds
    rows, cols = shape
    return (
        min_x + col / (cols - 1) * (max_x - min_x),
        min_y + row / (rows - 1) * (max_y - min_y),
    )


def _path_length_m(path, bounds, shape) -> float:
    min_x, min_y, max_x, max_y = bounds
    rows, cols = shape
    cell_x = (max_x - min_x) / (cols - 1)
    cell_y = (max_y - min_y) / (rows - 1)
    return sum(
        math.hypot((b[1] - a[1]) * cell_x, (b[0] - a[0]) * cell_y) for a, b in zip(path, path[1:])
    )


def _challenge_name(
    material_id: np.ndarray | None,
    risk: np.ndarray,
    hard: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    kind: str,
) -> str:
    if material_id is None:
        return kind
    eligible = hard[rr, cc] if kind == "hard" else risk[rr, cc] >= risk[rr, cc].max()
    ids = material_id[rr[eligible], cc[eligible]]
    if len(ids) == 0:
        return kind
    values, frequencies = np.unique(ids, return_counts=True)
    material = int(values[int(np.argmax(frequencies))])
    return MATERIALS[material] if 0 <= material < len(MATERIALS) else kind


def _separated(start, goal, selected, minimum_m: float) -> bool:
    if minimum_m <= 0.0:
        return True
    for route in selected:
        for candidate in (start, goal):
            for existing in (route["start_xy"], route["goal_xy"]):
                if math.dist(candidate, existing) < minimum_m:
                    return False
    return True


def generate_material_route_manifest(
    occupancy,
    risk,
    hard,
    bounds,
    *,
    counts: Mapping[str, int],
    hard_counts: Mapping[str, int] | None = None,
    seed: int,
    material_id=None,
    params: MaterialRouteGenerationParams = MaterialRouteGenerationParams(),
    source: Mapping[str, object] | None = None,
) -> dict:
    """Generate a split-locked material challenge corpus from public rasters.

    Endpoints lie on low-risk cells in the largest material-safe free-space
    component. A route is accepted only when its direct segment crosses soft
    risk or a hard material hazard, crosses no physical occupancy, and A* can
    connect the endpoints while treating hard material as blocked. Thus the
    challenge is caused by material rather than by a building on the direct
    line, while a hazard-free detour is known to exist.
    """
    split_counts = _split_counts("counts", counts)
    split_hard = _split_counts("hard_counts", hard_counts)
    for split in _SPLITS:
        if split_hard[split] > split_counts[split]:
            raise ValueError(f"hard_counts[{split!r}] exceeds counts[{split!r}]")

    occ = np.asarray(occupancy, dtype=bool)
    risk = np.asarray(risk, dtype=np.float32)
    hard = np.asarray(hard, dtype=bool)
    if occ.ndim != 2 or risk.shape != occ.shape or hard.shape != occ.shape:
        raise ValueError("occupancy, risk, and hard must be same-shape 2-D rasters")
    if min(occ.shape) < 2:
        raise ValueError("route rasters must have at least two cells per axis")
    if not np.isfinite(risk).all() or (risk < 0.0).any() or (risk > 1.0).any():
        raise ValueError("risk must be finite and in [0, 1]")
    ids = None if material_id is None else np.asarray(material_id)
    if ids is not None and ids.shape != occ.shape:
        raise ValueError("material_id must have the same shape as occupancy")

    world_bounds = _bounds(bounds)
    blocked = occ | hard
    labels, sizes = planner.free_components(blocked, 0)
    if not sizes:
        raise ValueError("no material-safe free-space component exists")
    largest = max(sizes, key=sizes.get)
    clearance_m, _normal_x, _normal_y = sdf_nav.build_sdf(blocked, world_bounds, 1.0)
    endpoints = (
        (labels == largest)
        & (clearance_m >= params.endpoint_clearance_m)
        & (risk < params.risk_threshold)
    )
    endpoint_cells = np.argwhere(endpoints)
    if len(endpoint_cells) < 2:
        raise ValueError("fewer than two eligible route endpoints")

    rng = np.random.default_rng(seed)
    manifest = {split: [] for split in _SPLITS}
    selected: list[dict] = []
    for split in _SPLITS:
        requested = ["hard"] * split_hard[split] + ["soft"] * (
            split_counts[split] - split_hard[split]
        )
        if requested:
            requested = [requested[index] for index in rng.permutation(len(requested))]
        for requested_kind in requested:
            for _attempt in range(params.max_attempts_per_route):
                indices = rng.integers(0, len(endpoint_cells), size=2)
                if indices[0] == indices[1]:
                    continue
                start_cell = endpoint_cells[int(indices[0])]
                goal_cell = endpoint_cells[int(indices[1])]
                start = _cell_to_world(start_cell, world_bounds, occ.shape)
                goal = _cell_to_world(goal_cell, world_bounds, occ.shape)
                direct_distance = math.dist(start, goal)
                if not params.min_distance_m <= direct_distance <= params.max_distance_m:
                    continue
                if not _separated(start, goal, selected, params.min_endpoint_separation_m):
                    continue

                rr, cc = _line_cells(start_cell, goal_cell)
                if occ[rr, cc].any():
                    continue
                kind = "hard" if hard[rr, cc].any() else "soft"
                if kind == "soft" and not (risk[rr, cc] >= params.risk_threshold).any():
                    continue
                if kind != requested_kind:
                    continue
                detour = planner.astar(blocked, tuple(start_cell), tuple(goal_cell))
                if detour is None:
                    continue
                if (
                    _path_length_m(detour, world_bounds, occ.shape)
                    > params.max_detour_ratio * direct_distance
                ):
                    continue

                route = {
                    "id": params.first_route_id + len(selected),
                    "start_xy": [float(value) for value in start],
                    "goal_xy": [float(value) for value in goal],
                    "direct_distance_m": float(direct_distance),
                    "challenge_kind": kind,
                    "material_challenge": _challenge_name(ids, risk, hard, rr, cc, kind),
                    "split_locked_before_training": True,
                }
                manifest[split].append(route)
                selected.append(route)
                break
            else:
                raise RuntimeError(
                    f"could not generate {requested_kind} route {len(manifest[split]) + 1}/"
                    f"{split_counts[split]} for {split!r} after "
                    f"{params.max_attempts_per_route} attempts"
                )

    manifest["schema"] = _SCHEMA
    manifest["protocol"] = {
        "name": "material_challenge_routes_v1",
        "seed": int(seed),
        "test_locked_before_training": True,
        "counts": split_counts,
        "hard_counts": split_hard,
        "selection": (
            "low-risk endpoints in the largest hard-safe free component; direct segment "
            "crosses material but no physical occupancy; hard-safe A* detour exists"
        ),
        "soft_risk_threshold": params.risk_threshold,
        "minimum_direct_distance_m": params.min_distance_m,
        "maximum_direct_distance_m": params.max_distance_m,
        "endpoint_clearance_m": params.endpoint_clearance_m,
        "minimum_endpoint_separation_m": params.min_endpoint_separation_m,
        "maximum_hard_safe_detour_ratio": params.max_detour_ratio,
        "source": dict(source or {}),
    }
    validate_material_route_manifest(manifest)
    return manifest


def validate_material_route_manifest(manifest: Mapping[str, object]) -> None:
    """Reject malformed or train/test-leaking material route manifests."""
    if manifest.get("schema") != _SCHEMA:
        raise ValueError(f"route manifest schema must be {_SCHEMA!r}")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("test_locked_before_training") is not True:
        raise ValueError("route protocol must lock the test split before training")
    counts = _split_counts("protocol counts", protocol.get("counts"))
    hard_counts = _split_counts("protocol hard_counts", protocol.get("hard_counts"))

    route_ids: set[int] = set()
    endpoints: list[tuple[float, float]] = []
    observed_hard = {split: 0 for split in _SPLITS}
    for split in _SPLITS:
        routes = manifest.get(split)
        if not isinstance(routes, list) or len(routes) != counts[split]:
            raise ValueError(f"{split} route count does not match protocol")
        for route in routes:
            if not isinstance(route, Mapping):
                raise ValueError(f"{split} routes must be objects")
            route_id = route.get("id")
            if not isinstance(route_id, int) or isinstance(route_id, bool) or route_id in route_ids:
                raise ValueError("route ids must be unique integers")
            route_ids.add(route_id)
            points = []
            for field in ("start_xy", "goal_xy"):
                point = np.asarray(route.get(field), dtype=np.float64)
                if point.shape != (2,) or not np.isfinite(point).all():
                    raise ValueError(f"route {route_id} {field} must contain two finite values")
                points.append((float(point[0]), float(point[1])))
                endpoints.append(points[-1])
            direct = route.get("direct_distance_m")
            if not isinstance(direct, (int, float)) or not math.isclose(
                float(direct), math.dist(*points), rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"route {route_id} has inconsistent direct_distance_m")
            kind = route.get("challenge_kind")
            if kind not in ("soft", "hard"):
                raise ValueError(f"route {route_id} challenge_kind must be 'soft' or 'hard'")
            observed_hard[split] += kind == "hard"
            if not isinstance(route.get("material_challenge"), str):
                raise ValueError(f"route {route_id} material_challenge must be a string")
            if route.get("split_locked_before_training") is not True:
                raise ValueError(f"route {route_id} is not split-locked")
    if observed_hard != hard_counts:
        raise ValueError("hard route counts do not match protocol")

    minimum = protocol.get("minimum_endpoint_separation_m")
    if not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_endpoint_separation_m must be finite and >= 0")
    for index, endpoint in enumerate(endpoints):
        for other in endpoints[index + 1 :]:
            if math.dist(endpoint, other) < float(minimum):
                raise ValueError("route endpoints violate minimum separation")


def write_material_route_manifest(path, manifest: Mapping[str, object]) -> None:
    """Validate and write a canonical, finite JSON route manifest."""
    validate_material_route_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
