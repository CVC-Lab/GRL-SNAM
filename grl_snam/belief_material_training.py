"""Build shared MaterialTrainer batches from observed material state."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .belief_material import BeliefMaterialParams
from .belief_material_controller import build_belief_material_controller_inputs
from .material import MaterialRuntime

__all__ = [
    "BeliefMaterialTrainingSample",
    "build_belief_material_training_batch",
]


@dataclass(frozen=True)
class BeliefMaterialTrainingSample:
    """One observed-state example for the shared material trainers.

    All geometry and trajectory arrays must use one coordinate frame. In particular,
    ``rollout_patch`` is the existing six-channel patch centred at ``position`` with channels
    ``[risk, phi, grad_r_x, grad_r_y, grad_phi_x, grad_phi_y]``. One patch-cell offset must mean
    one position-coordinate unit, matching :func:`material_nav.bilinear_sample_patch`; this
    adapter deliberately does not synthesize or rescale that dynamics input.
    """

    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_radii: np.ndarray
    obstacle_weights: np.ndarray
    runtime: MaterialRuntime
    seen: np.ndarray
    rollout_patch: np.ndarray
    robot_radius: float
    d_hat: float
    dt: float
    horizon: int
    target_position: np.ndarray
    target_velocity: np.ndarray
    target_gamma: float
    gate_active: bool | None = None


def _vec2(name: str, value) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32)
    if out.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _finite_scalar(name: str, value, *, minimum: float | None = None) -> np.float32:
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        condition = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {condition}")
    return np.float32(value)


def build_belief_material_training_batch(
    samples: Sequence[BeliefMaterialTrainingSample],
    *,
    patch_extent_m: float = 100.0,
    patch_size: int = 25,
    feature_scale_m: float = 20.0,
    belief: BeliefMaterialParams = BeliefMaterialParams(),
) -> dict[str, np.ndarray]:
    """Collate observed-state samples for both shared MaterialTrainer front ends.

    The returned mapping contains every ``material_train_native.BATCH_KEYS`` array plus
    ``gate_active``. Obstacle lists are zero-padded to the largest list in this batch and
    ``obs_mask`` marks the real tokens. The model patch always comes from the belief runtime and
    seen mask; the six-channel rollout patch is supplied by the trajectory dataset in its own
    pinned coordinate frame.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    if patch_size < 2:
        raise ValueError("patch_size must be >= 2")
    if not math.isfinite(patch_extent_m) or patch_extent_m <= 0.0:
        raise ValueError("patch_extent_m must be finite and > 0")

    prepared = []
    rollout_shape = None
    for index, sample in enumerate(samples):
        position = _vec2(f"samples[{index}].position", sample.position)
        velocity = _vec2(f"samples[{index}].velocity", sample.velocity)
        goal = _vec2(f"samples[{index}].goal", sample.goal)
        target_position = _vec2(f"samples[{index}].target_position", sample.target_position)
        target_velocity = _vec2(f"samples[{index}].target_velocity", sample.target_velocity)
        obs_feats, obs_mask, goal_feats, risk_patch = build_belief_material_controller_inputs(
            position,
            goal,
            sample.obstacle_centers,
            sample.obstacle_radii,
            sample.obstacle_weights,
            sample.runtime,
            sample.seen,
            patch_extent_m=patch_extent_m,
            patch_size=patch_size,
            feature_scale_m=feature_scale_m,
            belief=belief,
        )
        centers = np.asarray(sample.obstacle_centers, dtype=np.float32)
        if centers.size == 0:
            centers = np.empty((0, 2), dtype=np.float32)
        radii = np.asarray(sample.obstacle_radii, dtype=np.float32)

        rollout_patch = np.asarray(sample.rollout_patch, dtype=np.float32)
        if (
            rollout_patch.ndim != 3
            or rollout_patch.shape[0] != 6
            or min(rollout_patch.shape[1:]) < 2
        ):
            raise ValueError(
                f"samples[{index}].rollout_patch must have shape (6, H>=2, W>=2), "
                f"got {rollout_patch.shape}"
            )
        if not np.isfinite(rollout_patch).all():
            raise ValueError(f"samples[{index}].rollout_patch must be finite")
        if rollout_shape is None:
            rollout_shape = rollout_patch.shape
        elif rollout_patch.shape != rollout_shape:
            raise ValueError(
                "all rollout patches in a batch must have the same shape: "
                f"{rollout_patch.shape} vs {rollout_shape}"
            )

        robot_radius = _finite_scalar(
            f"samples[{index}].robot_radius", sample.robot_radius, minimum=0.0
        )
        d_hat = _finite_scalar(f"samples[{index}].d_hat", sample.d_hat)
        if d_hat <= 0.0:
            raise ValueError(f"samples[{index}].d_hat must be > 0")
        dt = _finite_scalar(f"samples[{index}].dt", sample.dt)
        if dt <= 0.0:
            raise ValueError(f"samples[{index}].dt must be > 0")
        horizon = int(sample.horizon)
        if horizon != sample.horizon or horizon < 0:
            raise ValueError(f"samples[{index}].horizon must be an integer >= 0")
        target_gamma = _finite_scalar(f"samples[{index}].target_gamma", sample.target_gamma)

        if sample.gate_active is None:
            gate_active = (
                bool(sample.runtime.gate(position, goal).active)
                if sample.runtime.params.gate_enabled
                else True
            )
        else:
            gate_active = bool(sample.gate_active)

        prepared.append(
            {
                "obs_feats": obs_feats,
                "obs_mask": obs_mask,
                "goal_feats": goal_feats,
                "risk_patch": risk_patch,
                "o0": position,
                "v0": velocity,
                "goal": goal,
                "C": centers,
                "R": radii,
                "rollout_patch": rollout_patch,
                "rr": robot_radius,
                "d_hat": d_hat,
                "dt": dt,
                "H": np.int32(horizon),
                "o_tgt": target_position,
                "v_tgt": target_velocity,
                "gamma_o": target_gamma,
                "gate_active": np.uint8(gate_active),
            }
        )

    batch_size = len(prepared)
    max_obstacles = max(item["C"].shape[0] for item in prepared)
    obs_feats = np.zeros((batch_size, max_obstacles, 6), dtype=np.float32)
    obs_mask = np.zeros((batch_size, max_obstacles), dtype=np.uint8)
    centers = np.zeros((batch_size, max_obstacles, 2), dtype=np.float32)
    radii = np.zeros((batch_size, max_obstacles), dtype=np.float32)
    for index, item in enumerate(prepared):
        count = item["C"].shape[0]
        obs_feats[index, :count] = item["obs_feats"]
        obs_mask[index, :count] = item["obs_mask"]
        centers[index, :count] = item["C"]
        radii[index, :count] = item["R"]

    def stack(name, dtype=np.float32):
        return np.ascontiguousarray(np.stack([item[name] for item in prepared]), dtype=dtype)

    return {
        "obs_feats": obs_feats,
        "obs_mask": obs_mask,
        "goal_feats": stack("goal_feats"),
        "risk_patch": stack("risk_patch"),
        "o0": stack("o0"),
        "v0": stack("v0"),
        "goal": stack("goal"),
        "C": centers,
        "R": radii,
        "rollout_patch": stack("rollout_patch"),
        "rr": stack("rr"),
        "d_hat": stack("d_hat"),
        "dt": stack("dt"),
        "H": stack("H", np.int32),
        "o_tgt": stack("o_tgt"),
        "v_tgt": stack("v_tgt"),
        "gamma_o": stack("gamma_o"),
        "gate_active": stack("gate_active", np.uint8),
    }
