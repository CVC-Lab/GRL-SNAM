"""Train a public belief-material controller on the pinned Austin route split.

Only ``manifest["train"]`` contributes optimization samples.  The geometry
backbone is warm-started from a public ``CoefEnergyNet`` checkpoint and frozen;
the observed-risk encoder and the two material coefficient heads are optimized
through :class:`grl_snam.material_train_torch.MaterialTrainerTorch`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .. import planner
from ..belief import BeliefGrid
from ..belief_material_controller import BeliefMaterialSensor
from ..belief_material_training import (
    BeliefMaterialTrainingSample,
    build_belief_material_training_batch,
)
from ..material import GateParams, MaterialParams
from ..material_nav import CoefEnergyNetMaterial, load_geometry_weights
from ..material_routes import validate_material_route_manifest
from ..material_train_torch import MaterialTrainerTorch, load_cvcnm
from .austin_material_evaluation import (
    AustinMaterialEvaluationScene,
    constant_material_rollout_patch,
    load_austin_material_evaluation_scene,
    scan_visible_geometry,
)
from .matnet_export import BeliefHeader

__all__ = [
    "AustinMaterialTrainingParams",
    "build_austin_material_training_samples",
    "train_austin_material_model",
]


@dataclass(frozen=True)
class AustinMaterialTrainingParams:
    """Pinned public training settings for the belief-material heads."""

    sensor_range_m: float = 60.0
    max_sensor_obstacles: int = 36
    samples_per_route: int = 16
    teacher_risk_weight: float = 4.0
    teacher_lookahead_m: float = 30.0
    max_speed_mps: float = 3.0
    dt_s: float = 1.0
    horizon_steps: int = 1
    target_gamma: float = 1.0
    geometry_d_hat_m: float = 3.0
    patch_extent_m: float = 100.0
    patch_size: int = 25
    feature_scale_m: float = 20.0
    fixed_lambda_soft: float = 1.5
    fixed_lambda_hard: float = 1.0
    material_gate_trigger: float = 0.30
    lambda_soft_max: float = 5.0
    lambda_hard_max: float = 10.0
    updates: int = 400
    batch_size: int = 48
    learning_rate: float = 3e-4
    cvar_alpha: float = 0.95

    def __post_init__(self) -> None:
        positive = {
            "sensor_range_m": self.sensor_range_m,
            "max_sensor_obstacles": self.max_sensor_obstacles,
            "samples_per_route": self.samples_per_route,
            "teacher_lookahead_m": self.teacher_lookahead_m,
            "max_speed_mps": self.max_speed_mps,
            "dt_s": self.dt_s,
            "horizon_steps": self.horizon_steps,
            "target_gamma": self.target_gamma,
            "geometry_d_hat_m": self.geometry_d_hat_m,
            "patch_extent_m": self.patch_extent_m,
            "feature_scale_m": self.feature_scale_m,
            "updates": self.updates,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lambda_soft_max": self.lambda_soft_max,
            "lambda_hard_max": self.lambda_hard_max,
        }
        if any(not math.isfinite(float(value)) or value <= 0 for value in positive.values()):
            raise ValueError("Austin material training positive parameters must be finite and > 0")
        if self.patch_size < 2:
            raise ValueError("patch_size must be >= 2")
        if not math.isfinite(self.teacher_risk_weight) or self.teacher_risk_weight < 0.0:
            raise ValueError("teacher_risk_weight must be finite and >= 0")
        if not 0.0 < self.fixed_lambda_soft < self.lambda_soft_max:
            raise ValueError("fixed_lambda_soft must lie strictly inside its learned range")
        if not 0.0 < self.fixed_lambda_hard < self.lambda_hard_max:
            raise ValueError("fixed_lambda_hard must lie strictly inside its learned range")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie strictly between zero and one")
        if (
            not math.isfinite(self.material_gate_trigger)
            or not 0.0 <= self.material_gate_trigger <= 1.0
        ):
            raise ValueError("material_gate_trigger must be finite and in [0, 1]")


def _world_to_cell(point, bounds, shape) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    rows, cols = shape
    col = int(round((point[0] - min_x) / (max_x - min_x) * (cols - 1)))
    row = int(round((point[1] - min_y) / (max_y - min_y) * (rows - 1)))
    return min(max(row, 0), rows - 1), min(max(col, 0), cols - 1)


def _cell_to_world(cell, bounds, shape) -> np.ndarray:
    row, col = cell
    min_x, min_y, max_x, max_y = bounds
    rows, cols = shape
    return np.asarray(
        [
            min_x + col / (cols - 1) * (max_x - min_x),
            min_y + row / (rows - 1) * (max_y - min_y),
        ],
        dtype=np.float64,
    )


def _polyline_position(points: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    distance = min(max(float(distance), 0.0), float(cumulative[-1]))
    upper = min(int(np.searchsorted(cumulative, distance, side="right")), len(points) - 1)
    lower = max(0, upper - 1)
    span = float(cumulative[upper] - cumulative[lower])
    if span <= 1e-12:
        return points[upper].copy()
    fraction = (distance - cumulative[lower]) / span
    return points[lower] + fraction * (points[upper] - points[lower])


def _teacher_polyline(
    scene: AustinMaterialEvaluationScene,
    route: dict,
    risk_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    blocked = scene.occupancy | scene.hard
    cells = planner.astar(
        blocked,
        _world_to_cell(route["start_xy"], scene.bounds, blocked.shape),
        _world_to_cell(route["goal_xy"], scene.bounds, blocked.shape),
        cost=risk_weight * scene.risk,
    )
    if cells is None:
        raise ValueError(f"training route {route['id']} has no hard-safe teacher path")
    points = np.stack([_cell_to_world(cell, scene.bounds, blocked.shape) for cell in cells])
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if cumulative[-1] <= 0.0:
        raise ValueError(f"training route {route['id']} has a zero-length teacher path")
    return points, cumulative


def build_austin_material_training_samples(
    scene: AustinMaterialEvaluationScene,
    manifest: dict,
    *,
    params: AustinMaterialTrainingParams = AustinMaterialTrainingParams(),
) -> tuple[list[BeliefMaterialTrainingSample], dict]:
    """Build observed-state supervision from train routes, never validation/test."""
    validate_material_route_manifest(manifest)
    samples: list[BeliefMaterialTrainingSample] = []
    route_lengths = []
    gate_active = 0
    visible_fractions = []
    obstacle_counts = []
    material_params = MaterialParams(
        lam_soft=params.fixed_lambda_soft,
        lam_hard=params.fixed_lambda_hard,
        k_sharp=5.0,
        d_hat_sdf_m=params.geometry_d_hat_m,
        gate=GateParams(material_trigger=params.material_gate_trigger),
    )

    for route in manifest["train"]:
        points, cumulative = _teacher_polyline(scene, route, params.teacher_risk_weight)
        route_lengths.append(float(cumulative[-1]))
        last_distance = max(0.0, float(cumulative[-1]) - params.max_speed_mps * params.dt_s)
        sample_distances = np.linspace(
            0.0,
            last_distance,
            params.samples_per_route,
            endpoint=True,
        )
        belief = BeliefGrid(scene.occupancy.shape, scene.bounds)
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

        for sample_index, distance in enumerate(sample_distances):
            position = _polyline_position(points, cumulative, distance)
            target = _polyline_position(
                points,
                cumulative,
                distance + params.max_speed_mps * params.dt_s * params.horizon_steps,
            )
            local_goal = _polyline_position(
                points,
                cumulative,
                distance + params.teacher_lookahead_m,
            )
            target_velocity = (target - position) / (params.dt_s * params.horizon_steps)
            speed = float(np.linalg.norm(target_velocity))
            if speed > params.max_speed_mps:
                target_velocity *= params.max_speed_mps / speed
                target = position + (params.dt_s * params.horizon_steps * target_velocity)
            velocity = target_velocity.copy()
            if sample_index == 0:
                velocity[:] = 0.0

            update = sensor.sense(
                position,
                range_m=params.sensor_range_m,
                fov_rad=2.0 * math.pi,
            )
            centers, radii, weights, _ray_hits = scan_visible_geometry(
                belief,
                scene.occupancy,
                position,
                update.n_rays,
                params.sensor_range_m,
                params.max_sensor_obstacles,
            )
            seen = np.asarray(belief.material_seen, dtype=bool).copy()
            active = bool(sensor.runtime.gate(position, local_goal).active)
            gate_active += int(active)
            visible_fractions.append(float(seen.mean()))
            obstacle_counts.append(len(centers))
            samples.append(
                BeliefMaterialTrainingSample(
                    position=position.astype(np.float32),
                    velocity=velocity.astype(np.float32),
                    goal=local_goal.astype(np.float32),
                    obstacle_centers=centers,
                    obstacle_radii=radii,
                    obstacle_weights=weights,
                    runtime=sensor.runtime,
                    seen=seen,
                    rollout_patch=constant_material_rollout_patch(
                        sensor.runtime,
                        position,
                        "cpu",
                    ),
                    robot_radius=0.0,
                    d_hat=params.geometry_d_hat_m,
                    dt=params.dt_s,
                    horizon=params.horizon_steps,
                    target_position=target.astype(np.float32),
                    target_velocity=target_velocity.astype(np.float32),
                    target_gamma=params.target_gamma,
                    gate_active=active,
                )
            )

    if not samples:
        raise ValueError("the route manifest contains no training samples")
    diagnostics = {
        "routes": len(manifest["train"]),
        "samples": len(samples),
        "challenge_counts": {
            kind: sum(route["challenge_kind"] == kind for route in manifest["train"])
            for kind in ("soft", "hard")
        },
        "teacher_path_length_m": {
            "minimum": float(np.min(route_lengths)),
            "mean": float(np.mean(route_lengths)),
            "maximum": float(np.max(route_lengths)),
        },
        "gate_active_fraction": gate_active / len(samples),
        "mean_visible_fraction": float(np.mean(visible_fractions)),
        "mean_obstacle_tokens": float(np.mean(obstacle_counts)),
    }
    return samples, diagnostics


def _set_initial_lambda(head: torch.nn.Sequential, value: float, maximum: float) -> None:
    probability = value / maximum
    bias = math.log(probability / (1.0 - probability))
    with torch.no_grad():
        head[-1].weight.zero_()
        head[-1].bias.fill_(bias)


def _new_model(
    geometry_checkpoint: str | Path,
    seed: int,
    params: AustinMaterialTrainingParams,
) -> tuple[CoefEnergyNetMaterial, int]:
    torch.manual_seed(seed)
    model = CoefEnergyNetMaterial(
        d_tok=64,
        d_risk=64,
        nhead=4,
        num_layers=2,
        patch_size=params.patch_size,
        risk_channels=3,
        lam_soft_max=params.lambda_soft_max,
        lam_hard_max=params.lambda_hard_max,
    )
    model.patch_size = params.patch_size
    model.belief_header = BeliefHeader(
        material_range_m=0.0,
        patch_extent_m=params.patch_extent_m,
    )
    matched = load_geometry_weights(model, str(geometry_checkpoint))
    geometry_tensors = sum(
        name.startswith(
            ("obs_enc.", "goal_enc.", "fuser.", "alpha_head.", "beta_head.", "gamma_head.")
        )
        for name in model.state_dict()
    )
    if matched != geometry_tensors:
        raise ValueError(
            f"geometry checkpoint matched {matched}/{geometry_tensors} geometry tensors"
        )
    _set_initial_lambda(model.lam_soft_head, params.fixed_lambda_soft, params.lambda_soft_max)
    _set_initial_lambda(model.lam_hard_head, params.fixed_lambda_hard, params.lambda_hard_max)
    model.fuser.enable_nested_tensor = False
    if hasattr(model.fuser, "use_nested_tensor"):
        model.fuser.use_nested_tensor = False
    return model, matched


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_austin_material_model(
    scene: AustinMaterialEvaluationScene,
    manifest: dict,
    *,
    geometry_checkpoint: str | Path,
    output_dir: str | Path,
    seed: int,
    device: str,
    params: AustinMaterialTrainingParams = AustinMaterialTrainingParams(),
) -> dict:
    """Train material heads, write a version-2 ``.cvcnm``, and return provenance."""
    np.random.seed(seed)
    samples, diagnostics = build_austin_material_training_samples(
        scene,
        manifest,
        params=params,
    )
    model, matched = _new_model(geometry_checkpoint, seed, params)
    trainer = MaterialTrainerTorch(
        model,
        device=device,
        grad_clip=5.0,
        w_traj=1.0,
        w_vel=0.5,
        w_fric=0.1,
        w_clear=5e-3,
        w_lreg=0.01,
        w_goal=2.0,
        w_len=0.01,
        w_risk=1.0,
        w_hard=5.0,
        cvar_alpha=params.cvar_alpha,
        w_multi=0.0,
        lam_soft_max=params.lambda_soft_max,
        lam_hard_max=params.lambda_hard_max,
        d_hat_sdf=params.geometry_d_hat_m,
        k_sharp=5.0,
        seed=seed,
    )
    trainable_prefixes = ("risk_enc.", "lam_soft_head.", "lam_hard_head.")
    for name, parameter in trainer.model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))

    rng = np.random.default_rng(seed)
    history = []
    for update in range(1, params.updates + 1):
        indices = rng.integers(0, len(samples), size=params.batch_size)
        batch = build_belief_material_training_batch(
            [samples[int(index)] for index in indices],
            patch_extent_m=params.patch_extent_m,
            patch_size=params.patch_size,
            feature_scale_m=params.feature_scale_m,
        )
        loss = trainer.step(batch, params.learning_rate)
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite material loss at update {update}: {loss}")
        if update == 1 or update % 50 == 0 or update == params.updates:
            row = {
                "update": update,
                "loss": float(loss),
                "eta": float(trainer.last_eta),
            }
            history.append(row)
            print(json.dumps({"event": "train", **row}), flush=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "material.cvcnm"
    trainer.save(checkpoint)
    # Read the file back through the shared deployable loader before reporting success.
    loaded = load_cvcnm(checkpoint, device="cpu")
    report = {
        "schema": "grl-snam-austin-material-training/1",
        "seed": seed,
        "trainer": "MaterialTrainerTorch",
        "device": str(trainer.device),
        "geometry_frozen": True,
        "matched_geometry_tensors": matched,
        "trainable_prefixes": list(trainable_prefixes),
        "params": asdict(params),
        "dataset": diagnostics,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "geometry_checkpoint_sha256": _sha256(geometry_checkpoint),
        "checkpoint_contract": {
            "risk_channels": loaded.risk_channels,
            "patch_size": loaded.patch_size,
            "belief": asdict(loaded.belief_header),
        },
    }
    report_path = output / "training.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cvar-alpha", type=float, default=0.95)
    parser.add_argument("--threads", type=int, default=2)
    user = parser.parse_args(argv)

    torch.set_num_threads(user.threads)
    device = user.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scene = load_austin_material_evaluation_scene(user.bundle)
    manifest = json.loads(user.routes.read_text(encoding="utf-8"))
    params = AustinMaterialTrainingParams(
        updates=user.updates,
        batch_size=user.batch_size,
        learning_rate=user.lr,
        cvar_alpha=user.cvar_alpha,
    )
    report = train_austin_material_model(
        scene,
        manifest,
        geometry_checkpoint=user.geometry_checkpoint,
        output_dir=user.out,
        seed=user.seed,
        device=device,
        params=params,
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
