"""Public, sensor-limited Austin material-controller evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..belief import BeliefGrid
from ..belief_material_controller import (
    BeliefMaterialCoefficients,
    BeliefMaterialControlCommand,
    BeliefMaterialController,
    BeliefMaterialSensor,
)
from ..material import GateParams, MaterialParams
from ..material_evaluation import (
    MATERIAL_EVALUATION_ARMS,
    MaterialRouteMetrics,
    paired_route_statistics,
    select_waypoint_risk_weight,
    summarize_material_routes,
)
from ..material_routes import validate_material_route_manifest
from ..material_waypoint import MaterialWaypointParams, choose_material_local_waypoint
from .austin_material_routes import (
    load_austin_material_route_inputs,
)

__all__ = [
    "AustinMaterialEvaluationParams",
    "AustinMaterialEvaluationScene",
    "constant_material_rollout_patch",
    "evaluate_austin_material_model",
    "load_austin_material_evaluation_scene",
    "run_austin_material_route",
    "scan_visible_geometry",
]


@dataclass(frozen=True)
class AustinMaterialEvaluationScene:
    """Austin route-grid occupancy with aligned public material planes."""

    occupancy: np.ndarray
    risk: np.ndarray
    hard: np.ndarray
    bounds: tuple[float, float, float, float]

    @property
    def cell_size_m(self) -> float:
        rows, cols = self.occupancy.shape
        min_x, min_y, max_x, max_y = self.bounds
        return min((max_x - min_x) / (cols - 1), (max_y - min_y) / (rows - 1))


@dataclass(frozen=True)
class AustinMaterialEvaluationParams:
    """Frozen protocol shared by validation selection and held-out evaluation."""

    sensor_range_m: float = 60.0
    sense_interval_steps: int = 10
    max_sensor_obstacles: int = 36
    waypoint_risk_candidates: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    max_steps: int = 250
    goal_tolerance_m: float = 12.0
    dt_s: float = 1.0
    max_speed_mps: float = 3.0
    fixed_lambda_soft: float = 1.5
    fixed_lambda_hard: float = 1.0
    material_gate_trigger: float = 0.30
    geometry_d_hat_m: float = 3.0
    patch_extent_m: float = 100.0
    segment_sample_m: float = 1.0
    bootstrap_samples: int = 10_000

    def __post_init__(self) -> None:
        positive = {
            "sensor_range_m": self.sensor_range_m,
            "max_sensor_obstacles": self.max_sensor_obstacles,
            "max_steps": self.max_steps,
            "goal_tolerance_m": self.goal_tolerance_m,
            "dt_s": self.dt_s,
            "max_speed_mps": self.max_speed_mps,
            "geometry_d_hat_m": self.geometry_d_hat_m,
            "patch_extent_m": self.patch_extent_m,
            "segment_sample_m": self.segment_sample_m,
            "bootstrap_samples": self.bootstrap_samples,
        }
        if any(not math.isfinite(float(value)) or value <= 0 for value in positive.values()):
            raise ValueError("Austin evaluation positive parameters must be finite and > 0")
        if self.sense_interval_steps < 1:
            raise ValueError("sense_interval_steps must be positive")
        if not self.waypoint_risk_candidates:
            raise ValueError("waypoint_risk_candidates must not be empty")
        if any(
            weight < 0.0 or not math.isfinite(weight) for weight in self.waypoint_risk_candidates
        ):
            raise ValueError("waypoint risk candidates must be finite and non-negative")
        if (
            not math.isfinite(self.material_gate_trigger)
            or not 0.0 <= self.material_gate_trigger <= 1.0
        ):
            raise ValueError("material_gate_trigger must be finite and in [0, 1]")


def load_austin_material_evaluation_scene(bundle: str | Path) -> AustinMaterialEvaluationScene:
    """Load the audited route grid shared by corpus generation and evaluation."""
    route_inputs = load_austin_material_route_inputs(bundle)
    return AustinMaterialEvaluationScene(
        occupancy=route_inputs.occupancy,
        risk=route_inputs.risk,
        hard=route_inputs.hard,
        bounds=route_inputs.bounds,
    )


def _world_to_cell(point, bounds, shape) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    rows, cols = shape
    col = int(round((point[0] - min_x) / (max_x - min_x) * (cols - 1)))
    row = int(round((point[1] - min_y) / (max_y - min_y) * (rows - 1)))
    return min(max(row, 0), rows - 1), min(max(col, 0), cols - 1)


def _cell_to_world(cell, bounds, shape) -> tuple[float, float]:
    row, col = cell
    min_x, min_y, max_x, max_y = bounds
    rows, cols = shape
    return (
        min_x + col / (cols - 1) * (max_x - min_x),
        min_y + row / (rows - 1) * (max_y - min_y),
    )


def scan_visible_geometry(
    belief: BeliefGrid,
    truth_occupancy: np.ndarray,
    position: np.ndarray,
    n_rays: int,
    range_m: float,
    max_obstacles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hit_cells = np.argwhere(belief.last_visible & truth_occupancy)
    if len(hit_cells):
        centers = np.asarray(
            [
                _cell_to_world(
                    cell, (belief.mnx, belief.mny, belief.mxx, belief.mxy), truth_occupancy.shape
                )
                for cell in hit_cells
            ],
            dtype=np.float32,
        )
        distances = np.linalg.norm(centers - position[None, :], axis=1)
        order = np.argsort(distances)[:max_obstacles]
        centers = centers[order]
        distances = distances[order]
    else:
        centers = np.empty((0, 2), np.float32)
        distances = np.empty((0,), np.float32)

    cell_size = min(
        (belief.mxx - belief.mnx) / (belief.nx - 1),
        (belief.mxy - belief.mny) / (belief.ny - 1),
    )
    radius = 0.5 * cell_size
    radii = np.full(len(centers), radius, np.float32)
    weights = np.ones(len(centers), np.float32)
    ray_hits = np.full(n_rays, range_m, np.float32)
    for center, distance in zip(centers, distances):
        angle = math.atan2(center[1] - position[1], center[0] - position[0]) % (2.0 * math.pi)
        ray_index = int(round(angle / (2.0 * math.pi) * n_rays)) % n_rays
        ray_hits[ray_index] = min(ray_hits[ray_index], max(0.0, float(distance) - radius))
    return centers, radii, weights, ray_hits


def constant_material_rollout_patch(runtime, position, device) -> np.ndarray:
    """Return a six-plane constant patch for a one-step rollout.

    A one-step material rollout samples its field only at the initial position,
    so a constant patch is exact while avoiding a large, otherwise-unused crop.
    """
    field = runtime.grid.field(device=str(device))
    with torch.inference_mode():
        at = torch.as_tensor(position, device=device, dtype=field.field.dtype).reshape(1, 2)
        risk, phi, grad_risk, grad_phi = field.sample(at)
    values = torch.cat((risk, phi, grad_risk.reshape(-1), grad_phi.reshape(-1)))
    return np.broadcast_to(values.detach().cpu().numpy()[:, None, None], (6, 3, 3)).copy()


def _fixed_command(
    command: BeliefMaterialControlCommand,
    lambda_soft: float,
    lambda_hard: float,
) -> BeliefMaterialControlCommand:
    soft = lambda_soft if command.gate is None or command.gate.active else 0.0
    effective = BeliefMaterialCoefficients(
        alphas=command.predicted.alphas,
        beta=command.predicted.beta,
        gamma=command.predicted.gamma,
        lam_soft=soft,
        lam_hard=lambda_hard,
        mu_lat=command.predicted.mu_lat,
    )
    return BeliefMaterialControlCommand(
        predicted=command.predicted,
        effective=effective,
        gate=command.gate,
    )


def _segment_metrics(
    scene: AustinMaterialEvaluationScene,
    start: np.ndarray,
    end: np.ndarray,
    spacing_m: float,
) -> tuple[bool, float, float, bool]:
    length = float(np.linalg.norm(end - start))
    count = max(1, int(math.ceil(length / spacing_m)))
    fractions = np.linspace(0.0, 1.0, count + 1)
    points = start[None, :] + fractions[:, None] * (end - start)[None, :]
    cells = [_world_to_cell(point, scene.bounds, scene.occupancy.shape) for point in points]
    rows = np.asarray([cell[0] for cell in cells], np.intp)
    cols = np.asarray([cell[1] for cell in cells], np.intp)
    collision = bool(scene.occupancy[rows, cols].any())
    risk_distance = float(scene.risk[rows, cols].mean()) * length
    hard_fraction = float(scene.hard[rows, cols].mean())
    return collision, risk_distance, hard_fraction * length, bool(hard_fraction > 0.0)


def run_austin_material_route(
    controller: BeliefMaterialController,
    scene: AustinMaterialEvaluationScene,
    route: dict,
    *,
    arm: str,
    split: str,
    seed: int,
    waypoint_risk_weight: float,
    params: AustinMaterialEvaluationParams = AustinMaterialEvaluationParams(),
) -> MaterialRouteMetrics:
    """Run one route with truth used only by sensing, physics, and post-hoc metrics."""
    if arm not in MATERIAL_EVALUATION_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    start = np.asarray(route["start_xy"], np.float64)
    final_goal = np.asarray(route["goal_xy"], np.float64)
    position = start.copy()
    velocity = np.zeros(2, np.float64)
    belief = BeliefGrid(scene.occupancy.shape, scene.bounds)
    material_params = MaterialParams(
        lam_soft=params.fixed_lambda_soft,
        lam_hard=params.fixed_lambda_hard,
        k_sharp=5.0,
        d_hat_sdf_m=3.0,
        gate=GateParams(material_trigger=params.material_gate_trigger),
    )
    sensor = BeliefMaterialSensor(
        belief,
        scene.occupancy,
        scene.risk,
        scene.hard,
        center=(0.0, 0.0),
        scale=1.0,
        sigma=1.0,
        material_params=material_params,
    )
    centers = np.empty((0, 2), np.float32)
    radii = np.empty((0,), np.float32)
    weights = np.empty((0,), np.float32)
    local_waypoint = final_goal.copy()
    path_length = risk_distance = hard_distance = 0.0
    hard_steps = gate_steps = 0
    latency_ms = []
    collision = success = False

    for step in range(params.max_steps):
        if step % params.sense_interval_steps == 0:
            update = sensor.sense(
                position,
                range_m=params.sensor_range_m,
                fov_rad=2.0 * math.pi,
            )
            centers, radii, weights, ray_hits = scan_visible_geometry(
                belief,
                scene.occupancy,
                position,
                update.n_rays,
                params.sensor_range_m,
                params.max_sensor_obstacles,
            )
            uses_material_waypoint = arm in ("waypoint", "learned")

            def profile(ux, uy, distance):
                if uses_material_waypoint:
                    return sensor.runtime.ray_profile(position, ux, uy, distance)
                return 0.0, None

            local_waypoint, _waypoint_info = choose_material_local_waypoint(
                position,
                velocity,
                final_goal,
                ray_hits,
                sensor_radius_m=params.sensor_range_m,
                material_profile=profile,
                params=MaterialWaypointParams(
                    risk_weight=waypoint_risk_weight if uses_material_waypoint else 0.0
                ),
            )
            local_waypoint = np.asarray(local_waypoint, np.float64)

        started = time.perf_counter()
        command = controller.decide(
            position,
            local_waypoint,
            centers,
            radii,
            weights,
            sensor.runtime,
            belief.material_seen,
        )
        latency_ms.append((time.perf_counter() - started) * 1000.0)
        if arm in ("fixed_lambda", "waypoint"):
            command = _fixed_command(
                command,
                params.fixed_lambda_soft,
                params.fixed_lambda_hard,
            )
        if command.gate is not None and command.gate.active:
            gate_steps += 1

        device = next(controller.model.parameters()).device
        rollout = controller.rollout(
            command,
            position,
            velocity,
            local_waypoint,
            centers,
            radii,
            constant_material_rollout_patch(sensor.runtime, position, device),
            sensor.runtime,
            d_hat=params.geometry_d_hat_m,
            dt=params.dt_s,
            steps=1,
        )
        next_velocity = np.asarray(rollout.velocity, np.float64)
        speed = float(np.linalg.norm(next_velocity))
        if not np.isfinite(next_velocity).all() or speed <= 1e-12:
            next_velocity = np.zeros(2, np.float64)
        elif speed > params.max_speed_mps:
            next_velocity *= params.max_speed_mps / speed
        next_position = position + params.dt_s * next_velocity
        collided, risk_step, hard_step_distance, entered_hard = _segment_metrics(
            scene,
            position,
            next_position,
            params.segment_sample_m,
        )
        displacement = float(np.linalg.norm(next_position - position))
        path_length += displacement
        risk_distance += risk_step
        hard_distance += hard_step_distance
        hard_steps += int(entered_hard)
        position = next_position
        velocity = next_velocity
        if collided:
            collision = True
            break
        if np.linalg.norm(position - final_goal) <= params.goal_tolerance_m:
            success = True
            break

    steps = step + 1
    timeout = not success and not collision
    return MaterialRouteMetrics(
        seed=seed,
        route_id=int(route["id"]),
        split=split,
        challenge_kind=str(route["challenge_kind"]),
        arm=arm,
        success=success,
        collision=collision,
        timeout=timeout,
        direct_distance_m=float(route["direct_distance_m"]),
        path_length_m=path_length,
        risk_distance=risk_distance,
        hard_hazard_distance_m=hard_distance,
        steps=steps,
        hard_hazard_steps=hard_steps,
        gate_active_steps=gate_steps,
        prediction_latency_ms=float(np.mean(latency_ms)),
    )


def evaluate_austin_material_model(
    controller: BeliefMaterialController,
    scene: AustinMaterialEvaluationScene,
    manifest: dict,
    *,
    seed: int,
    params: AustinMaterialEvaluationParams = AustinMaterialEvaluationParams(),
    route_runner=run_austin_material_route,
) -> dict:
    """Select waypoint weight on validation, then evaluate four arms on test."""
    validate_material_route_manifest(manifest)
    validation_rows = {}
    for weight in params.waypoint_risk_candidates:
        validation_rows[weight] = [
            route_runner(
                controller,
                scene,
                route,
                arm="learned",
                split="validation",
                seed=seed,
                waypoint_risk_weight=weight,
                params=params,
            )
            for route in manifest["validation"]
        ]
    selected_weight, selection_summary = select_waypoint_risk_weight(validation_rows)

    rows = [
        route_runner(
            controller,
            scene,
            route,
            arm=arm,
            split="test",
            seed=seed,
            waypoint_risk_weight=selected_weight,
            params=params,
        )
        for arm in MATERIAL_EVALUATION_ARMS
        for route in manifest["test"]
    ]
    return {
        "seed": seed,
        "params": asdict(params),
        "selected_waypoint_risk_weight": selected_weight,
        "validation_selection": selection_summary,
        "rows": [row.to_dict() for row in rows],
        "summary": {
            arm: summarize_material_routes(row for row in rows if row.arm == arm)
            for arm in MATERIAL_EVALUATION_ARMS
        },
        "paired": paired_route_statistics(
            rows,
            bootstrap_samples=params.bootstrap_samples,
            bootstrap_seed=seed,
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    user = parser.parse_args(argv)

    scene = load_austin_material_evaluation_scene(user.bundle)
    manifest = json.loads(user.routes.read_text(encoding="utf-8"))
    controller = BeliefMaterialController.from_cvcnm(user.checkpoint, device=user.device)
    report = evaluate_austin_material_model(controller, scene, manifest, seed=user.seed)
    report["provenance"] = {
        "checkpoint_sha256": _sha256(user.checkpoint),
        "routes_sha256": _sha256(user.routes),
    }
    user.out.parent.mkdir(parents=True, exist_ok=True)
    user.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "out": str(user.out), "summary": report["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
