#!/usr/bin/env python3
"""
Stage-1 sim-to-real bridge for GRL-SNAM.

This module provides a fast procedural simulator that is closer to the ROS2
car interface than the original velocity-frame toy rollouts:

  procedural circular-obstacle scene
      -> synthetic LaserScan
      -> local scan adapter / pseudo-obstacle tokens
      -> policy command (v, omega)
      -> Ackermann-feasible projection
      -> bicycle-model rollout

The goal is not photorealism. The goal is interface alignment: train and test
on the same local-frame observation and command envelope that a ROS2 policy node
will see later from /scan, /tf, and /odom.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class AckermannCfg:
    wheelbase: float = 0.324
    v_max: float = 0.80
    # Reverse is deliberately smaller than forward speed; tune to the car.
    v_reverse_max: float = 0.25
    v_min: float = 0.05
    omega_max: float = 2.0
    max_steering_angle: float = 0.396
    min_turning_radius: float = 0.80
    dt: float = 0.10

    # Optional command slew limits. These are applied in closed-loop eval and can
    # be emulated in training if previous commands are added to the state later.
    max_accel: float = 1.5
    max_steer_rate: float = 2.0

    # Visualization footprint (pose is rear-axle center in the bicycle model).
    body_width: float = 0.30
    rear_axle_to_front: float = 0.36
    rear_axle_to_back: float = 0.14


@dataclass
class LaserScanCfg:
    n_beams: int = 181
    fov: float = math.radians(270.0)
    range_min: float = 0.05
    range_max: float = 6.0
    noise_std: float = 0.015
    dropout_prob: float = 0.01
    local_radius: float = 4.0


@dataclass
class SceneCfg:
    world_x: Tuple[float, float] = (-1.5, 9.5)
    world_y: Tuple[float, float] = (-3.0, 3.0)
    n_obs_range: Tuple[int, int] = (8, 20)
    radius_range: Tuple[float, float] = (0.18, 0.65)
    start: Tuple[float, float] = (0.0, 0.0)
    goal_x_range: Tuple[float, float] = (5.5, 8.8)
    goal_y_range: Tuple[float, float] = (-1.5, 1.5)
    min_start_clear: float = 0.80
    min_goal_clear: float = 0.80
    min_obs_sep: float = 0.08
    robot_radius: float = 0.23
    corridor_half_width: float = 0.70
    p_keep_corridor_clear: float = 0.75


@dataclass
class AdapterCfg:
    cluster_dist: float = 0.18
    min_cluster_size: int = 2
    max_tokens: int = 24
    pseudo_radius_pad: float = 0.05
    proximity_radius: float = 4.0


@dataclass
class RecoveryCfg:
    # Fraction of on-the-fly training samples biased toward near-contact recovery.
    p_recovery_sample: float = 0.25
    # Stagnation detection for closed-loop diagnostics / optional port injection.
    stall_speed: float = 0.06
    stall_progress: float = 0.01
    stall_clearance: float = 0.65
    stall_steps_trigger: int = 8
    # Minimum commanded motion encouraged by the training loss away from the goal.
    move_speed_floor: float = 0.10
    # Optional online fallback. Keep disabled for pure learned-policy evaluation.
    enable_port_injection: bool = False
    reverse_bias: float = 0.22
    escape_steering_scale: float = 0.90
    # Stop only when the circular clearance surrogate is already violated.
    imminent_stop_clearance: float = -0.02
    # Horizon length used by the hybrid sequence policy.
    train_horizon: int = 12


@dataclass
class Stage1Cfg:
    scene: SceneCfg = field(default_factory=SceneCfg)
    scan: LaserScanCfg = field(default_factory=LaserScanCfg)
    adapter: AdapterCfg = field(default_factory=AdapterCfg)
    ackermann: AckermannCfg = field(default_factory=AckermannCfg)
    recovery: RecoveryCfg = field(default_factory=RecoveryCfg)
    seed: int = 2026


# -----------------------------------------------------------------------------
# Geometry and dynamics
# -----------------------------------------------------------------------------

def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def world_to_local(points_w: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    """Map world points to robot/body frame. pose=(x,y,theta)."""
    points_w = np.asarray(points_w, dtype=np.float32)
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    R = rot2(th)
    return (points_w - np.array([x, y], dtype=np.float32)) @ R


def local_to_world(points_l: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    points_l = np.asarray(points_l, dtype=np.float32)
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    R = rot2(th)
    return points_l @ R.T + np.array([x, y], dtype=np.float32)


def project_twist_np(
    v: float,
    omega: float,
    cfg: AckermannCfg,
) -> Tuple[float, float, float]:
    """Project (v, omega) onto the Ackermann-feasible command set.

    Returns (v_projected, omega_projected, steering_angle).
    """
    v = float(np.clip(v, -cfg.v_reverse_max, cfg.v_max))
    omega = float(np.clip(omega, -cfg.omega_max, cfg.omega_max))
    if abs(v) < cfg.v_min:
        return 0.0, 0.0, 0.0

    # Curvature and steering constraints are both imposed. The min-turning-radius
    # constraint should be set from the calibrated car.
    kappa = omega / v
    kappa_max = 1.0 / max(cfg.min_turning_radius, 1e-6)
    kappa = float(np.clip(kappa, -kappa_max, kappa_max))
    delta = math.atan(cfg.wheelbase * kappa)
    delta = float(np.clip(delta, -cfg.max_steering_angle, cfg.max_steering_angle))
    omega = v / cfg.wheelbase * math.tan(delta)
    return float(v), float(omega), float(delta)


def project_twist_torch(
    raw_cmd: torch.Tensor,
    cfg: AckermannCfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable-ish torch projection for training.

    raw_cmd: [B,2] where columns are (v, omega). Returns batched
    (v_projected, omega_projected, steering_angle).
    """
    v = torch.clamp(raw_cmd[:, 0], -cfg.v_reverse_max, cfg.v_max)
    omega = torch.clamp(raw_cmd[:, 1], -cfg.omega_max, cfg.omega_max)
    moving = (torch.abs(v) >= cfg.v_min).to(v.dtype)
    v_safe = torch.where(torch.abs(v) < cfg.v_min, torch.sign(v + 1e-6) * cfg.v_min, v)
    kappa = omega / v_safe
    kappa = torch.clamp(kappa, -1.0 / cfg.min_turning_radius, 1.0 / cfg.min_turning_radius)
    delta = torch.atan(cfg.wheelbase * kappa)
    delta = torch.clamp(delta, -cfg.max_steering_angle, cfg.max_steering_angle)
    omega_proj = v_safe / cfg.wheelbase * torch.tan(delta)
    return moving * v_safe, moving * omega_proj, moving * delta


def ackermann_step_np(
    pose: Sequence[float],
    v: float,
    omega: float,
    dt: float,
) -> np.ndarray:
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    x2 = x + dt * v * math.cos(th)
    y2 = y + dt * v * math.sin(th)
    th2 = wrap_angle(th + dt * omega)
    return np.array([x2, y2, th2], dtype=np.float32)


def ackermann_step_torch(
    pose: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    x, y, th = pose[:, 0], pose[:, 1], pose[:, 2]
    x2 = x + dt * v * torch.cos(th)
    y2 = y + dt * v * torch.sin(th)
    th2 = th + dt * omega
    th2 = torch.atan2(torch.sin(th2), torch.cos(th2))
    return torch.stack([x2, y2, th2], dim=-1)


# -----------------------------------------------------------------------------
# Procedural scenes
# -----------------------------------------------------------------------------

def _sample_goal(rng: np.random.Generator, cfg: SceneCfg) -> np.ndarray:
    return np.array([
        rng.uniform(*cfg.goal_x_range),
        rng.uniform(*cfg.goal_y_range),
    ], dtype=np.float32)


def _clear_of_points(c: np.ndarray, r: float, pts: np.ndarray, margins: Sequence[float]) -> bool:
    for p, m in zip(pts, margins):
        if np.linalg.norm(c - p) < (r + m):
            return False
    return True


def sample_scene(cfg: SceneCfg, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate a simple navigable scene with circular obstacles.

    A loose corridor between start and goal is kept clear with high probability.
    This gives nontrivial obstacle distributions while avoiding impossible scenes.
    """
    start = np.asarray(cfg.start, dtype=np.float32)
    goal = _sample_goal(rng, cfg)
    n_obs = int(rng.integers(cfg.n_obs_range[0], cfg.n_obs_range[1] + 1))
    C: List[np.ndarray] = []
    R: List[float] = []

    d = goal - start
    L2 = float(np.dot(d, d) + 1e-9)

    def dist_to_start_goal_segment(c: np.ndarray) -> float:
        t = float(np.clip(np.dot(c - start, d) / L2, 0.0, 1.0))
        p = start + t * d
        return float(np.linalg.norm(c - p))

    tries = 0
    while len(C) < n_obs and tries < 5000:
        tries += 1
        c = np.array([
            rng.uniform(*cfg.world_x),
            rng.uniform(*cfg.world_y),
        ], dtype=np.float32)
        r = float(rng.uniform(*cfg.radius_range))

        if not _clear_of_points(c, r, np.stack([start, goal]), [cfg.min_start_clear, cfg.min_goal_clear]):
            continue
        if C:
            dmin = np.min(np.linalg.norm(np.stack(C) - c[None, :], axis=1) - np.asarray(R))
            if dmin < r + cfg.min_obs_sep:
                continue
        if rng.uniform() < cfg.p_keep_corridor_clear:
            if dist_to_start_goal_segment(c) < cfg.corridor_half_width + r:
                continue
        C.append(c)
        R.append(r)

    if not C:
        C_arr = np.zeros((0, 2), dtype=np.float32)
        R_arr = np.zeros((0,), dtype=np.float32)
    else:
        C_arr = np.stack(C).astype(np.float32)
        R_arr = np.asarray(R, dtype=np.float32)
    W_arr = np.ones_like(R_arr, dtype=np.float32)
    return {"start": start, "goal": goal, "C": C_arr, "R": R_arr, "W": W_arr}


def min_clearance_to_obstacles(
    xy: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    robot_radius: float = 0.0,
) -> float:
    if C.shape[0] == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(C - xy[None, :], axis=1) - R - robot_radius))


# -----------------------------------------------------------------------------
# Synthetic LaserScan and adapter
# -----------------------------------------------------------------------------

def scan_angles(cfg: LaserScanCfg) -> np.ndarray:
    return np.linspace(-0.5 * cfg.fov, 0.5 * cfg.fov, cfg.n_beams, dtype=np.float32)


def render_laserscan(
    pose: Sequence[float],
    C_world: np.ndarray,
    R: np.ndarray,
    scan_cfg: LaserScanCfg,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Render a 2D LaserScan against circular obstacles.

    Returned ranges are float32 with inf for no return, mirroring ROS LaserScan
    conventions. Obstacles are transformed to the robot frame first.
    """
    angles = scan_angles(scan_cfg)
    ranges = np.full((scan_cfg.n_beams,), np.inf, dtype=np.float32)
    if C_world.shape[0] == 0:
        return ranges

    C_local = world_to_local(C_world, pose)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)

    for c, rad in zip(C_local, R):
        # Solve ||t d - c||^2 = r^2 for each ray direction d.
        cdot = dirs @ c
        c2 = float(c @ c)
        disc = cdot * cdot - (c2 - float(rad * rad))
        hit = disc >= 0.0
        if not np.any(hit):
            continue
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = cdot - sqrt_disc
        t2 = cdot + sqrt_disc
        t = np.where(t1 > scan_cfg.range_min, t1, t2)
        valid = hit & (t > scan_cfg.range_min) & (t < scan_cfg.range_max)
        ranges[valid] = np.minimum(ranges[valid], t[valid].astype(np.float32))

    if rng is not None:
        finite = np.isfinite(ranges)
        if scan_cfg.noise_std > 0:
            ranges[finite] += rng.normal(0.0, scan_cfg.noise_std, size=int(finite.sum())).astype(np.float32)
            ranges = np.clip(ranges, scan_cfg.range_min, scan_cfg.range_max)
        if scan_cfg.dropout_prob > 0:
            drop = finite & (rng.uniform(size=ranges.shape) < scan_cfg.dropout_prob)
            ranges[drop] = np.inf

    return ranges.astype(np.float32)


def scan_to_points(ranges: np.ndarray, scan_cfg: LaserScanCfg) -> np.ndarray:
    angles = scan_angles(scan_cfg)
    ranges = np.asarray(ranges, dtype=np.float32)
    valid = np.isfinite(ranges)
    valid &= ranges >= scan_cfg.range_min
    valid &= ranges <= min(scan_cfg.range_max, scan_cfg.local_radius)
    rr = ranges[valid]
    aa = angles[valid]
    if rr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([rr * np.cos(aa), rr * np.sin(aa)], axis=-1).astype(np.float32)


def scan_to_obstacle_features(
    ranges: np.ndarray,
    scan_cfg: LaserScanCfg,
    adapter_cfg: AdapterCfg,
) -> np.ndarray:
    """Convert LaserScan to pseudo-obstacle tokens in robot frame.

    Features are [cx, cy, r, ||c||, angle(c), proximity]. This mirrors the
    real ROS adapter we will later use for /scan.
    """
    angles = scan_angles(scan_cfg)
    ranges = np.asarray(ranges, dtype=np.float32)
    valid = np.isfinite(ranges)
    valid &= ranges >= scan_cfg.range_min
    valid &= ranges <= min(scan_cfg.range_max, scan_cfg.local_radius)

    pts_all = np.zeros((ranges.shape[0], 2), dtype=np.float32)
    finite_for_xy = np.isfinite(ranges)
    pts_all[finite_for_xy, 0] = ranges[finite_for_xy] * np.cos(angles[finite_for_xy])
    pts_all[finite_for_xy, 1] = ranges[finite_for_xy] * np.sin(angles[finite_for_xy])
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    clusters: List[np.ndarray] = []
    cur: List[np.ndarray] = [pts_all[valid_idx[0]]]
    prev_i = int(valid_idx[0])
    prev_p = pts_all[prev_i]
    for ii in valid_idx[1:]:
        ii = int(ii)
        p = pts_all[ii]
        # Break cluster if beams are not contiguous or Euclidean gap is large.
        if ii == prev_i + 1 and np.linalg.norm(p - prev_p) < adapter_cfg.cluster_dist:
            cur.append(p)
        else:
            if len(cur) >= adapter_cfg.min_cluster_size:
                clusters.append(np.asarray(cur, dtype=np.float32))
            cur = [p]
        prev_i = ii
        prev_p = p
    if len(cur) >= adapter_cfg.min_cluster_size:
        clusters.append(np.asarray(cur, dtype=np.float32))

    feats: List[List[float]] = []
    for cl in clusters:
        c = cl.mean(axis=0)
        rad = float(np.linalg.norm(cl - c[None, :], axis=1).max() + adapter_cfg.pseudo_radius_pad)
        norm = float(np.linalg.norm(c))
        ang = float(math.atan2(float(c[1]), float(c[0])))
        prox = max(0.0, (adapter_cfg.proximity_radius - max(norm - rad, 0.0)) / adapter_cfg.proximity_radius)
        feats.append([float(c[0]), float(c[1]), rad, norm, ang, prox])

    if not feats:
        return np.zeros((0, 6), dtype=np.float32)
    arr = np.asarray(feats, dtype=np.float32)
    # Keep nearest / most relevant tokens.
    order = np.argsort(arr[:, 3])
    arr = arr[order[: adapter_cfg.max_tokens]]
    return arr.astype(np.float32)


def scan_feature_clearances(obs_feats: np.ndarray, robot_radius: float = 0.23) -> Tuple[float, float]:
    """Return (min_clearance, front_clearance) from local pseudo-obstacle tokens.

    These are finite, clipped scalar context features for the policy; the exact
    closed-loop collision metric still uses world geometry during evaluation.
    """
    if obs_feats is None or len(obs_feats) == 0:
        return 4.0, 4.0
    arr = np.asarray(obs_feats, dtype=np.float32)
    clear = arr[:, 3] - arr[:, 2] - float(robot_radius)
    min_clear = float(np.min(clear))
    front = (arr[:, 0] > -0.20) & (np.abs(arr[:, 1]) < 1.10)
    if np.any(front):
        front_clear = float(np.min(clear[front]))
    else:
        front_clear = 4.0
    return float(np.clip(min_clear, -1.0, 4.0)), float(np.clip(front_clear, -1.0, 4.0))


def local_goal_features(
    goal_world: np.ndarray,
    pose: Sequence[float],
    current_speed: float = 0.0,
    min_clearance: float = 4.0,
    front_clearance: float = 4.0,
    stall_count: int = 0,
    stall_trigger: int = 8,
) -> np.ndarray:
    """Local policy context.

    Format: [gx, gy, ||g||, goal_angle, current_speed, min_clearance,
             front_clearance, normalized_stall_count].
    """
    g_l = world_to_local(np.asarray(goal_world, dtype=np.float32)[None, :], pose)[0]
    dist = float(np.linalg.norm(g_l))
    ang = float(math.atan2(float(g_l[1]), float(g_l[0])))
    stall_norm = float(np.clip(float(stall_count) / max(1.0, float(stall_trigger)), 0.0, 1.0))
    return np.array([
        g_l[0], g_l[1], dist, ang, current_speed,
        float(np.clip(min_clearance, -1.0, 4.0)),
        float(np.clip(front_clearance, -1.0, 4.0)),
        stall_norm,
    ], dtype=np.float32)


# -----------------------------------------------------------------------------
# Scan-only expert. Used for imitation labels and baseline evaluation.
# -----------------------------------------------------------------------------

def expert_from_scan_features(
    obs_feats: np.ndarray,
    goal_feats: np.ndarray,
    ack_cfg: AckermannCfg,
    robot_radius: float = 0.23,
) -> np.ndarray:
    """Reactive scan-only expert with explicit reverse recovery behavior.

    It uses only the same local tokens and scalar context as the policy. When a
    stall hint is active and a front obstacle is close, it chooses a short reverse
    maneuver with steering away from the nearest front obstacle. This gives the
    learner examples of the nonholonomic recovery behavior needed by a real car.
    """
    gf = np.zeros((8,), dtype=np.float32)
    gf[: min(len(goal_feats), 8)] = np.asarray(goal_feats, dtype=np.float32)[: min(len(goal_feats), 8)]
    gx, gy, gdist, _, current_speed, min_clear_feat, front_clear_feat, stall_hint = [float(x) for x in gf]
    if gdist < 0.20:
        return np.array([0.0, 0.0], dtype=np.float32)

    goal_dir = np.array([gx, gy], dtype=np.float32)
    goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)

    rep = np.zeros(2, dtype=np.float32)
    min_front_clear = float(front_clear_feat)
    min_clear = float(min_clear_feat)
    nearest_front_y = 0.0
    nearest_front_norm = float("inf")

    for f in obs_feats:
        cx, cy, rad, norm, _, prox = [float(x) for x in f]
        if norm < 1e-6:
            continue
        clear = norm - rad - robot_radius
        min_clear = min(min_clear, clear)
        if cx > -0.20 and abs(cy) < 1.1:
            if clear < min_front_clear:
                min_front_clear = clear
            if norm < nearest_front_norm:
                nearest_front_norm = norm
                nearest_front_y = cy
        # Stronger repulsion for front obstacles; lateral sign induces turning.
        influence = max(0.0, 1.25 - clear) / 1.25
        front_gate = 1.0 / (1.0 + math.exp(-3.0 * cx))
        away = -np.array([cx, cy], dtype=np.float32) / (norm + 1e-6)
        rep += (0.15 + 1.25 * front_gate) * influence * influence * away

    # Explicit recovery label: reverse and steer away from the nearest frontal
    # obstacle if a near-contact stall has been detected.
    if stall_hint > 0.5 and min_front_clear < 0.55:
        # If obstacle is on the left (cy>0), reverse with opposite steering sign.
        steer_sign = -1.0 if nearest_front_y >= 0.0 else 1.0
        delta = steer_sign * 0.85 * ack_cfg.max_steering_angle
        v_raw = -ack_cfg.v_reverse_max
        omega_raw = v_raw / ack_cfg.wheelbase * math.tan(delta)
        v, omega, _ = project_twist_np(v_raw, omega_raw, ack_cfg)
        return np.array([v, omega], dtype=np.float32)

    desired = goal_dir + rep
    if np.linalg.norm(desired) < 1e-6:
        desired = goal_dir
    desired_angle = math.atan2(float(desired[1]), float(desired[0]))

    omega_raw = 1.8 * desired_angle
    speed_angle_scale = max(0.15, math.cos(min(abs(desired_angle), math.pi / 2.0)))
    speed_goal_scale = min(1.0, gdist / 1.5)
    speed_clear_scale = 1.0
    if np.isfinite(min_front_clear):
        speed_clear_scale = float(np.clip((min_front_clear - 0.05) / 0.85, 0.10, 1.0))
    elif np.isfinite(min_clear):
        speed_clear_scale = float(np.clip((min_clear + 0.20) / 1.20, 0.25, 1.0))

    v_raw = ack_cfg.v_max * speed_angle_scale * speed_goal_scale * speed_clear_scale
    v, omega, _ = project_twist_np(v_raw, omega_raw, ack_cfg)
    return np.array([v, omega], dtype=np.float32)


# -----------------------------------------------------------------------------
# Horizon teacher helpers
# -----------------------------------------------------------------------------

def classify_maneuver_mode(v: float, delta: float, cfg: AckermannCfg) -> int:
    """Return a coarse maneuver label for sequence-level supervision.

    Modes:
      0 STOP_SAFE, 1 REVERSE_ALIGN, 2 CREEP_FORWARD,
      3 FORWARD_ARC_LEFT, 4 FORWARD_ARC_RIGHT, 5 GO_FORWARD.
    """
    v = float(v); delta = float(delta)
    if abs(v) < cfg.v_min:
        return 0
    if v < -cfg.v_min:
        return 1
    if v < 0.22:
        return 2
    if delta > 0.25 * cfg.max_steering_angle:
        return 3
    if delta < -0.25 * cfg.max_steering_angle:
        return 4
    return 5


def fast_open_loop_teacher_sequence(
    scene: Dict[str, np.ndarray],
    pose: np.ndarray,
    obs_feats: np.ndarray,
    goal_feats: np.ndarray,
    current_speed: float,
    current_delta: float,
    stall_count: int,
    cfg: Stage1Cfg,
    horizon: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Cheap H-step teacher used for on-the-fly training.

    This avoids re-rendering a full synthetic LaserScan at every horizon step.
    The first command is the scan-only expert. Subsequent commands are a
    kinematic continuation: recovery samples keep a short reverse-align option,
    otherwise the sequence uses pure-pursuit-like goal steering with the same
    Ackermann slew/curvature limits.
    """
    H = int(horizon if horizon is not None else cfg.recovery.train_horizon)
    pose_k = np.asarray(pose, dtype=np.float32).copy()
    speed_k = float(current_speed)
    delta_k = float(current_delta)
    min_clear_feat, front_clear_feat = scan_feature_clearances(obs_feats, cfg.scene.robot_radius)
    cmd0 = expert_from_scan_features(obs_feats, goal_feats, cfg.ackermann, cfg.scene.robot_radius)

    # Determine whether this is a recovery option from the same logic as the expert.
    recovery_active = bool(float(goal_feats[7]) > 0.5 and front_clear_feat < 0.55)
    nearest_y = 0.0
    nearest_norm = float('inf')
    for f in obs_feats:
        cx, cy, rad, norm, *_ = [float(x) for x in f]
        if cx > -0.20 and abs(cy) < 1.10 and norm < nearest_norm:
            nearest_norm = norm
            nearest_y = cy
    steer_sign = -1.0 if nearest_y >= 0.0 else 1.0
    recovery_delta = steer_sign * 0.85 * cfg.ackermann.max_steering_angle

    cmds: List[List[float]] = []
    poses: List[np.ndarray] = []
    betas: List[float] = []
    first_delta = 0.0
    for k in range(H):
        if recovery_active and k < max(2, min(5, H // 2)):
            v_des = -cfg.ackermann.v_reverse_max
            delta_des = recovery_delta
            omega_des = v_des / cfg.ackermann.wheelbase * math.tan(delta_des)
        elif k == 0:
            v_des, omega_des = float(cmd0[0]), float(cmd0[1])
            _, _, delta_des = project_twist_np(v_des, omega_des, cfg.ackermann)
        else:
            # Pure-pursuit-like continuation using the current predicted pose.
            g_l = world_to_local(scene['goal'][None, :], pose_k)[0]
            gdist = float(np.linalg.norm(g_l))
            alpha = math.atan2(float(g_l[1]), float(g_l[0]))
            lookahead = max(0.65, min(3.0, gdist))
            delta_des = math.atan2(2.0 * cfg.ackermann.wheelbase * math.sin(alpha), lookahead)
            delta_des = float(np.clip(delta_des, -cfg.ackermann.max_steering_angle, cfg.ackermann.max_steering_angle))
            speed_scale = max(0.20, math.cos(min(abs(alpha), math.pi / 2.0)))
            clear_scale = float(np.clip((min(front_clear_feat, min_clear_feat) + 0.20) / 1.20, 0.25, 1.0))
            v_des = cfg.ackermann.v_max * min(1.0, gdist / 1.5) * speed_scale * clear_scale
            omega_des = v_des / cfg.ackermann.wheelbase * math.tan(delta_des)

        v, omega, delta = project_twist_np(v_des, omega_des, cfg.ackermann)
        ddelta_max = cfg.ackermann.max_steer_rate * cfg.ackermann.dt
        delta = float(np.clip(delta, delta_k - ddelta_max, delta_k + ddelta_max))
        dv_max = cfg.ackermann.max_accel * cfg.ackermann.dt
        v = float(np.clip(v, speed_k - dv_max, speed_k + dv_max))
        omega = v / cfg.ackermann.wheelbase * math.tan(delta) if abs(v) >= cfg.ackermann.v_min else 0.0
        if k == 0:
            first_delta = delta
        cmds.append([v, omega])
        pose_k = ackermann_step_np(pose_k, v, omega, cfg.ackermann.dt)
        poses.append(pose_k.copy())
        speed_k, delta_k = v, delta

        beta = max(0.0, min(1.0, (0.90 - min(front_clear_feat, min_clear_feat)) / 0.90))
        if recovery_active and k < max(2, min(5, H // 2)):
            beta *= 0.65
        betas.append(float(beta))

    cmd_arr = np.asarray(cmds, dtype=np.float32)
    pose_arr = np.asarray(poses, dtype=np.float32)
    beta_arr = np.asarray(betas, dtype=np.float32)
    mode = classify_maneuver_mode(float(cmd_arr[0, 0]), float(first_delta), cfg.ackermann) if H > 0 else 0
    return cmd_arr, pose_arr, mode, beta_arr


def expert_sequence_from_state(
    scene: Dict[str, np.ndarray],
    pose: np.ndarray,
    current_speed: float,
    current_delta: float,
    stall_count: int,
    cfg: Stage1Cfg,
    horizon: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Roll out the scan-only expert for H steps and return sequence labels.

    Returns:
      cmd_seq:  [H,2]  (v, omega)
      pose_seq: [H,3]  poses after each expert step
      mode:     int    coarse label based on the first expert command
      beta_seq: [H]    heuristic soft-barrier weights in [0,1]
    """
    rng = rng or np.random.default_rng(cfg.seed + 123)
    H = int(horizon if horizon is not None else cfg.recovery.train_horizon)
    pose_k = np.asarray(pose, dtype=np.float32).copy()
    speed_k = float(current_speed)
    delta_k = float(current_delta)
    stall_k = int(stall_count)
    cmds: List[List[float]] = []
    poses: List[np.ndarray] = []
    betas: List[float] = []
    first_delta = 0.0

    prev_dist_goal = float(np.linalg.norm(scene["goal"] - pose_k[:2]))
    for k in range(H):
        ranges = render_laserscan(pose_k, scene["C"], scene["R"], cfg.scan, rng)
        obs_feats = scan_to_obstacle_features(ranges, cfg.scan, cfg.adapter)
        min_clear_feat, front_clear_feat = scan_feature_clearances(obs_feats, cfg.scene.robot_radius)
        goal_feats = local_goal_features(
            scene["goal"], pose_k, current_speed=speed_k,
            min_clearance=min_clear_feat, front_clearance=front_clear_feat,
            stall_count=stall_k, stall_trigger=cfg.recovery.stall_steps_trigger,
        )
        raw_cmd = expert_from_scan_features(obs_feats, goal_feats, cfg.ackermann, cfg.scene.robot_radius)
        v, omega, delta = project_twist_np(float(raw_cmd[0]), float(raw_cmd[1]), cfg.ackermann)

        # Apply the same slew limits as closed-loop rollout so the teacher is feasible.
        ddelta_max = cfg.ackermann.max_steer_rate * cfg.ackermann.dt
        delta = float(np.clip(delta, delta_k - ddelta_max, delta_k + ddelta_max))
        omega = v / cfg.ackermann.wheelbase * math.tan(delta) if abs(v) >= cfg.ackermann.v_min else 0.0
        dv_max = cfg.ackermann.max_accel * cfg.ackermann.dt
        v = float(np.clip(v, speed_k - dv_max, speed_k + dv_max))
        omega = v / cfg.ackermann.wheelbase * math.tan(delta) if abs(v) >= cfg.ackermann.v_min else 0.0
        if k == 0:
            first_delta = delta
        cmds.append([v, omega])

        pose_k = ackermann_step_np(pose_k, v, omega, cfg.ackermann.dt)
        poses.append(pose_k.copy())
        speed_k, delta_k = v, delta

        # Soft planning barrier target: high near front/contact, lower in free space.
        beta = max(0.0, min(1.0, (0.90 - min(front_clear_feat, min_clear_feat)) / 0.90))
        if stall_k >= cfg.recovery.stall_steps_trigger and front_clear_feat < cfg.recovery.stall_clearance:
            # Reduce soft barrier during explicit reverse recovery so the vehicle can back out.
            beta *= 0.65
        betas.append(float(beta))

        new_dist_goal = float(np.linalg.norm(scene["goal"] - pose_k[:2]))
        progress = prev_dist_goal - new_dist_goal
        true_clear = min_clearance_to_obstacles(pose_k[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
        if abs(v) < cfg.recovery.stall_speed and progress < cfg.recovery.stall_progress and true_clear < cfg.recovery.stall_clearance:
            stall_k += 1
        else:
            stall_k = max(0, stall_k - 1)
        prev_dist_goal = new_dist_goal

    cmd_arr = np.asarray(cmds, dtype=np.float32)
    pose_arr = np.asarray(poses, dtype=np.float32)
    beta_arr = np.asarray(betas, dtype=np.float32)
    mode = classify_maneuver_mode(float(cmd_arr[0, 0]), float(first_delta), cfg.ackermann) if H > 0 else 0
    return cmd_arr, pose_arr, mode, beta_arr

# -----------------------------------------------------------------------------
# Dataset and collate
# -----------------------------------------------------------------------------

class Stage1AckermannDataset(Dataset):
    """On-the-fly random scan dataset for Stage-1 training.

    Each sample is a local observation produced by synthetic LaserScan plus an
    Ackermann-feasible expert command. The next pose under the expert command is
    included so the learner can be trained with a dynamics-consistency term.
    """
    def __init__(
        self,
        n_samples: int = 20000,
        cfg: Optional[Stage1Cfg] = None,
        seed: int = 2026,
        randomize_pose: bool = True,
        p_recovery: Optional[float] = None,
    ):
        self.n_samples = int(n_samples)
        self.cfg = cfg or Stage1Cfg(seed=seed)
        self.seed = int(seed)
        self.randomize_pose = bool(randomize_pose)
        self.p_recovery = float(self.cfg.recovery.p_recovery_sample if p_recovery is None else p_recovery)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed + int(idx) * 9973)
        scene = sample_scene(self.cfg.scene, rng)
        start = scene["start"]
        goal = scene["goal"]
        recovery_sample = bool(rng.uniform() < self.p_recovery)

        # Sample states along rough start-goal corridor, with random lateral and
        # heading perturbations. Recovery samples put the vehicle near a frontal
        # obstacle with low speed so the expert demonstrates reverse+steer.
        if self.randomize_pose:
            tau = float(rng.uniform(0.0, 0.95))
            base = (1.0 - tau) * start + tau * goal
            d = goal - start
            d = d / (np.linalg.norm(d) + 1e-6)
            n = np.array([-d[1], d[0]], dtype=np.float32)
            xy = base + n * rng.normal(0.0, 0.65) + d * rng.normal(0.0, 0.20)
            theta_nom = math.atan2(float(d[1]), float(d[0]))
            theta = wrap_angle(theta_nom + rng.normal(0.0, 0.55))
        else:
            xy = start.copy()
            theta = math.atan2(float(goal[1] - start[1]), float(goal[0] - start[0]))

        if recovery_sample:
            # Keep the pose roughly along the route but deliberately place a
            # pseudo-obstacle just in front. This creates labels for reversing
            # out of a nonholonomic deadlock.
            theta = wrap_angle(theta + rng.normal(0.0, 0.25))
            pose_tmp = np.array([xy[0], xy[1], theta], dtype=np.float32)
            front_l = np.array([[rng.uniform(0.45, 0.75), rng.uniform(-0.35, 0.35)]], dtype=np.float32)
            front_w = local_to_world(front_l, pose_tmp)[0]
            r_front = float(rng.uniform(0.18, 0.35))
            scene["C"] = np.concatenate([scene["C"], front_w.reshape(1, 2).astype(np.float32)], axis=0)
            scene["R"] = np.concatenate([scene["R"], np.array([r_front], dtype=np.float32)], axis=0)
            scene["W"] = np.ones_like(scene["R"], dtype=np.float32)

        if recovery_sample:
            current_speed = float(rng.uniform(-0.03, 0.04))
            stall_count = int(rng.integers(self.cfg.recovery.stall_steps_trigger, self.cfg.recovery.stall_steps_trigger + 5))
        else:
            current_speed = float(rng.uniform(-0.05, self.cfg.ackermann.v_max))
            stall_count = int(rng.integers(0, max(1, self.cfg.recovery.stall_steps_trigger // 2)))

        pose = np.array([xy[0], xy[1], theta], dtype=np.float32)
        ranges = render_laserscan(pose, scene["C"], scene["R"], self.cfg.scan, rng)
        obs_feats = scan_to_obstacle_features(ranges, self.cfg.scan, self.cfg.adapter)
        min_clear_feat, front_clear_feat = scan_feature_clearances(obs_feats, self.cfg.scene.robot_radius)
        goal_feats = local_goal_features(
            goal, pose, current_speed=current_speed,
            min_clearance=min_clear_feat, front_clearance=front_clear_feat,
            stall_count=stall_count, stall_trigger=self.cfg.recovery.stall_steps_trigger,
        )
        cmd_seq, pose_seq, mode, beta_seq = fast_open_loop_teacher_sequence(
            scene, pose, obs_feats, goal_feats, current_speed=current_speed, current_delta=0.0,
            stall_count=stall_count, cfg=self.cfg, horizon=self.cfg.recovery.train_horizon
        )
        cmd = cmd_seq[0]
        next_pose = pose_seq[0]

        return {
            "obs_feats": torch.tensor(obs_feats, dtype=torch.float32),
            "goal_feats": torch.tensor(goal_feats, dtype=torch.float32),
            "cmd": torch.tensor(cmd, dtype=torch.float32),
            "cmd_seq": torch.tensor(cmd_seq, dtype=torch.float32),
            "pose": torch.tensor(pose, dtype=torch.float32),
            "next_pose": torch.tensor(next_pose, dtype=torch.float32),
            "pose_seq": torch.tensor(pose_seq, dtype=torch.float32),
            "beta_seq": torch.tensor(beta_seq, dtype=torch.float32),
            "mode": torch.tensor(mode, dtype=torch.long),
            "ranges": torch.tensor(ranges, dtype=torch.float32),
            "recovery_sample": torch.tensor(1 if recovery_sample else 0, dtype=torch.long),
        }


def collate_stage1(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    maxN = max((b["obs_feats"].shape[0] for b in batch), default=0)
    D = 6
    obs = torch.zeros(B, maxN, D, dtype=torch.float32)
    mask = torch.zeros(B, maxN, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["obs_feats"].shape[0]
        if n:
            obs[i, :n] = b["obs_feats"]
            mask[i, :n] = True
    return {
        "obs_feats": obs,
        "obs_mask": mask,
        "goal_feats": torch.stack([b["goal_feats"] for b in batch], 0),
        "cmd": torch.stack([b["cmd"] for b in batch], 0),
        "cmd_seq": torch.stack([b["cmd_seq"] for b in batch], 0),
        "pose": torch.stack([b["pose"] for b in batch], 0),
        "next_pose": torch.stack([b["next_pose"] for b in batch], 0),
        "pose_seq": torch.stack([b["pose_seq"] for b in batch], 0),
        "beta_seq": torch.stack([b["beta_seq"] for b in batch], 0),
        "mode": torch.stack([b["mode"] for b in batch], 0),
        "ranges": torch.stack([b["ranges"] for b in batch], 0),
        "recovery_sample": torch.stack([b.get("recovery_sample", torch.tensor(0, dtype=torch.long)) for b in batch], 0),
    }


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def apply_recovery_port(
    raw_cmd: np.ndarray,
    obs_feats: np.ndarray,
    stall_count: int,
    cfg: Stage1Cfg,
) -> np.ndarray:
    """Optional online port-injection recovery bias.

    This is a supervisory fallback, not the primary learned behavior. It adds a
    reverse-speed and escape-steering bias after repeated stagnation.
    """
    if not cfg.recovery.enable_port_injection or stall_count < cfg.recovery.stall_steps_trigger:
        return raw_cmd
    min_clear, front_clear = scan_feature_clearances(obs_feats, cfg.scene.robot_radius)
    if front_clear > cfg.recovery.stall_clearance:
        return raw_cmd
    nearest_y = 0.0
    nearest_norm = float("inf")
    for f in obs_feats:
        cx, cy, rad, norm, *_ = [float(x) for x in f]
        if cx > -0.20 and abs(cy) < 1.10 and norm < nearest_norm:
            nearest_norm = norm
            nearest_y = cy
    steer_sign = -1.0 if nearest_y >= 0.0 else 1.0
    delta_bias = steer_sign * cfg.recovery.escape_steering_scale * cfg.ackermann.max_steering_angle
    v_bias = -cfg.recovery.reverse_bias
    omega_bias = v_bias / cfg.ackermann.wheelbase * math.tan(delta_bias)
    return np.asarray(raw_cmd, dtype=np.float32) + np.array([v_bias, omega_bias], dtype=np.float32)


def rollout_policy(
    policy_fn,
    scene: Dict[str, np.ndarray],
    cfg: Stage1Cfg,
    max_steps: int = 160,
    goal_tol: float = 0.35,
    safety_stop: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Closed-loop rollout with synthetic scan and Ackermann dynamics.

    policy_fn(obs_feats, goal_feats) -> np.array([v, omega]). It receives local
    pseudo-obstacle tokens and local goal/recovery context features.
    """
    rng = rng or np.random.default_rng(cfg.seed + 4242)
    start = scene["start"]
    goal = scene["goal"]
    d = goal - start
    pose = np.array([start[0], start[1], math.atan2(float(d[1]), float(d[0]))], dtype=np.float32)
    current_speed = 0.0
    current_delta = 0.0
    stall_count = 0
    prev_dist_goal = float(np.linalg.norm(goal - pose[:2]))

    traj = [pose.copy()]
    cmds = []
    min_clear = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
    collision = min_clear < 0.0
    stopped_by_safety = False
    stall_steps = 0
    reverse_steps = 0

    for _ in range(max_steps):
        dist_goal = float(np.linalg.norm(goal - pose[:2]))
        if dist_goal <= goal_tol:
            break
        true_clear = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
        if safety_stop and true_clear < cfg.recovery.imminent_stop_clearance:
            stopped_by_safety = True
            break

        ranges = render_laserscan(pose, scene["C"], scene["R"], cfg.scan, rng)
        obs_feats = scan_to_obstacle_features(ranges, cfg.scan, cfg.adapter)
        min_clear_feat, front_clear_feat = scan_feature_clearances(obs_feats, cfg.scene.robot_radius)
        goal_feats = local_goal_features(
            goal, pose, current_speed=current_speed,
            min_clearance=min_clear_feat, front_clearance=front_clear_feat,
            stall_count=stall_count, stall_trigger=cfg.recovery.stall_steps_trigger,
        )
        raw = policy_fn(obs_feats, goal_feats)
        raw = apply_recovery_port(raw, obs_feats, stall_count, cfg)
        v, omega, delta = project_twist_np(float(raw[0]), float(raw[1]), cfg.ackermann)

        # Apply simple steering-rate and acceleration limits in rollout.
        ddelta_max = cfg.ackermann.max_steer_rate * cfg.ackermann.dt
        delta = float(np.clip(delta, current_delta - ddelta_max, current_delta + ddelta_max))
        omega = v / cfg.ackermann.wheelbase * math.tan(delta) if abs(v) >= cfg.ackermann.v_min else 0.0
        dv_max = cfg.ackermann.max_accel * cfg.ackermann.dt
        v = float(np.clip(v, current_speed - dv_max, current_speed + dv_max))
        current_speed = v
        current_delta = delta

        pose = ackermann_step_np(pose, v, omega, cfg.ackermann.dt)
        traj.append(pose.copy())
        cmds.append([v, omega, delta])
        if v < -cfg.ackermann.v_min:
            reverse_steps += 1
        clr = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
        min_clear = min(min_clear, clr)
        if clr < 0.0:
            collision = True
            break

        new_dist_goal = float(np.linalg.norm(goal - pose[:2]))
        progress = prev_dist_goal - new_dist_goal
        stalled_now = (abs(v) < cfg.recovery.stall_speed and progress < cfg.recovery.stall_progress and clr < cfg.recovery.stall_clearance)
        if stalled_now:
            stall_count += 1
            stall_steps += 1
        else:
            stall_count = max(0, stall_count - 1)
        prev_dist_goal = new_dist_goal

    traj_arr = np.asarray(traj, dtype=np.float32)
    cmds_arr = np.asarray(cmds, dtype=np.float32) if cmds else np.zeros((0, 3), dtype=np.float32)
    path_len = float(np.linalg.norm(np.diff(traj_arr[:, :2], axis=0), axis=1).sum()) if len(traj_arr) > 1 else 0.0
    final_dist = float(np.linalg.norm(goal - traj_arr[-1, :2]))
    return {
        "traj": traj_arr,
        "cmds": cmds_arr,
        "success": bool(final_dist <= goal_tol and not collision and not stopped_by_safety),
        "collision": bool(collision),
        "stopped_by_safety": bool(stopped_by_safety),
        "min_clearance": float(min_clear),
        "path_length": path_len,
        "final_dist": final_dist,
        "steps": int(len(traj_arr) - 1),
        "stall_steps": int(stall_steps),
        "reverse_steps": int(reverse_steps),
    }



def vehicle_box_corners_world(pose: Sequence[float], cfg: AckermannCfg) -> np.ndarray:
    """Return the 4 corners of the car footprint in world coordinates.

    The state pose is interpreted as the rear-axle center, consistent with the
    bicycle rollout used in Stage-1. The rectangle is returned in closed-loop
    plotting order (front-left, front-right, rear-right, rear-left).
    """
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    hf = float(cfg.rear_axle_to_front)
    hb = float(cfg.rear_axle_to_back)
    hw = 0.5 * float(cfg.body_width)
    box_local = np.array([
        [hf,  hw],
        [hf, -hw],
        [-hb, -hw],
        [-hb,  hw],
    ], dtype=np.float32)
    return local_to_world(box_local, (x, y, th))


def _plot_vehicle_boxes(ax, traj: np.ndarray, ack_cfg: AckermannCfg, color: str,
                        stride: int = 8, alpha: float = 0.10, lw: float = 0.8,
                        linestyle: str = '-'):
    if traj is None or len(traj) == 0:
        return
    n = len(traj)
    idxs = list(range(0, n, max(1, stride)))
    if (n - 1) not in idxs:
        idxs.append(n - 1)
    for j, idx in enumerate(idxs):
        corners = vehicle_box_corners_world(traj[idx, :3], ack_cfg)
        poly = np.vstack([corners, corners[0]])
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, linewidth=0.0)
        ax.plot(poly[:, 0], poly[:, 1], color=color, alpha=min(1.0, alpha + 0.35),
                lw=lw, linestyle=linestyle)


def _plot_vehicle_box_single(ax, pose: Sequence[float], ack_cfg: AckermannCfg, color: str,
                             alpha: float = 0.20, lw: float = 1.2, linestyle: str = '-'):
    corners = vehicle_box_corners_world(pose, ack_cfg)
    poly = np.vstack([corners, corners[0]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, linewidth=0.0)
    ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw, linestyle=linestyle)


def save_rollout_plot(
    path: str,
    scene: Dict[str, np.ndarray],
    rollout: Dict[str, Any],
    title: str = "Stage-1 Ackermann rollout",
    cfg: Optional[Stage1Cfg] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(8.0, 4.2), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", "box")
    C, R = scene["C"], scene["R"]
    for c, r in zip(C, R):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), fill=False, lw=1.5))
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r + 0.23), fill=False, lw=0.8, alpha=0.35))
    tr = rollout["traj"]
    ax.plot(tr[:, 0], tr[:, 1], lw=2.0, label="policy")
    if cfg is not None:
        _plot_vehicle_boxes(ax, tr, cfg.ackermann, color="C0", stride=max(1, len(tr)//10), alpha=0.10)
        _plot_vehicle_box_single(ax, tr[0, :3], cfg.ackermann, color="C0", alpha=0.18)
        _plot_vehicle_box_single(ax, tr[-1, :3], cfg.ackermann, color="C0", alpha=0.22)
    ax.plot([scene["start"][0]], [scene["start"][1]], "o", ms=6, label="start")
    ax.plot([scene["goal"][0]], [scene["goal"][1]], "*", ms=10, label="goal")
    ax.set_xlim(-1.5, 9.5)
    ax.set_ylim(-3.2, 3.2)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def stage1_cfg_to_dict(cfg: Stage1Cfg) -> Dict[str, Any]:
    return {
        "scene": asdict(cfg.scene),
        "scan": asdict(cfg.scan),
        "adapter": asdict(cfg.adapter),
        "ackermann": asdict(cfg.ackermann),
        "recovery": asdict(cfg.recovery),
        "seed": cfg.seed,
    }
