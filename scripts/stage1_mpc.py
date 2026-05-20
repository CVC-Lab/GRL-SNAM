#!/usr/bin/env python3
"""
Sampling-based MPC baseline for the Stage-1 Ackermann/LaserScan simulator.

This module is intentionally dependency-light: it uses random shooting over
Ackermann-feasible speed/steering sequences, evaluates them with the same bicycle
model and obstacle clearance metric used by Stage-1 evaluation, and executes the
first control. It is meant as a reference/controller baseline, not as a learned
policy.

By default this is a geometry-aware MPC: it sees the procedural obstacle circles.
That makes it an upper-bound baseline against the scan-token policy. For strict
sensor-fair comparisons, replace the clearance term with pseudo-obstacle circles
built from scan_to_obstacle_features at each control step.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from scripts.stage1_ackermann_scan import (
    AckermannCfg,
    Stage1Cfg,
    ackermann_step_np,
    min_clearance_to_obstacles,
    project_twist_np,
    render_laserscan,
    wrap_angle,
)


@dataclass
class MPCConfig:
    horizon: int = 12
    n_samples: int = 512
    safe_margin: float = 0.20
    v_nominal: float = 0.55

    # Cost weights.
    w_final_goal: float = 12.0
    w_running_goal: float = 0.15
    w_heading: float = 0.35
    w_clearance: float = 20.0
    w_collision: float = 1.0e5
    w_control: float = 0.04
    w_smooth: float = 0.10
    w_time: float = 0.02

    # Sampling mixture. Slow and reverse candidates help around clutter/deadlocks.
    p_slow: float = 0.20
    p_reverse: float = 0.25
    seed: int = 9102


def mpc_cfg_to_dict(cfg: MPCConfig) -> Dict[str, Any]:
    return asdict(cfg)


def _min_clearance_batch(xy: np.ndarray, C: np.ndarray, R: np.ndarray, robot_radius: float) -> np.ndarray:
    if C.shape[0] == 0:
        return np.full((xy.shape[0],), np.inf, dtype=np.float32)
    d = np.linalg.norm(xy[:, None, :] - C[None, :, :], axis=-1) - R[None, :] - float(robot_radius)
    return d.min(axis=1).astype(np.float32)


def _sample_control_sequences(
    pose: np.ndarray,
    goal: np.ndarray,
    current_speed: float,
    current_delta: float,
    cfg: Stage1Cfg,
    mpc_cfg: MPCConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return target speed and steering sequences of shape [N,H]."""
    N, H = int(mpc_cfg.n_samples), int(mpc_cfg.horizon)
    ack = cfg.ackermann

    # Goal-directed steering seed from local bearing.
    dx = float(goal[0] - pose[0])
    dy = float(goal[1] - pose[1])
    alpha = wrap_angle(math.atan2(dy, dx) - float(pose[2]))
    lookahead = max(0.7, min(3.0, math.hypot(dx, dy)))
    delta_goal = math.atan2(2.0 * ack.wheelbase * math.sin(alpha), lookahead)
    delta_goal = float(np.clip(delta_goal, -ack.max_steering_angle, ack.max_steering_angle))
    v_goal = min(ack.v_max, max(0.10, mpc_cfg.v_nominal))

    # Random shooting around the goal seed plus broad exploration.
    v_seq = rng.uniform(-ack.v_reverse_max, ack.v_max, size=(N, H)).astype(np.float32)
    delta_seq = rng.uniform(-ack.max_steering_angle, ack.max_steering_angle, size=(N, H)).astype(np.float32)

    # Bias half the samples around pure-pursuit-like commands.
    n_bias = max(1, N // 2)
    v_seq[:n_bias] = np.clip(
        rng.normal(v_goal, 0.20, size=(n_bias, H)), 0.0, ack.v_max
    ).astype(np.float32)
    delta_seq[:n_bias] = np.clip(
        rng.normal(delta_goal, 0.20, size=(n_bias, H)),
        -ack.max_steering_angle,
        ack.max_steering_angle,
    ).astype(np.float32)

    # Slow/braking candidates.
    n_slow = int(mpc_cfg.p_slow * N)
    if n_slow > 0:
        v_seq[-n_slow:] = rng.uniform(0.0, min(0.30, ack.v_max), size=(n_slow, H)).astype(np.float32)

    # Reverse recovery candidates.
    n_rev = int(mpc_cfg.p_reverse * N)
    if n_rev > 0:
        lo = max(0, N - n_slow - n_rev)
        hi = max(lo, N - n_slow)
        if hi > lo:
            v_seq[lo:hi] = rng.uniform(-ack.v_reverse_max, -ack.v_min, size=(hi - lo, H)).astype(np.float32)
            # Reverse samples need aggressive steering diversity.
            delta_seq[lo:hi] = rng.uniform(-ack.max_steering_angle, ack.max_steering_angle, size=(hi - lo, H)).astype(np.float32)

    # Deterministic anchors.
    anchors = [
        (v_goal, delta_goal),
        (min(ack.v_max, v_goal), 0.0),
        (0.25, delta_goal),
        (0.15, 0.0),
        (float(current_speed), float(current_delta)),
        (min(ack.v_max, v_goal), -ack.max_steering_angle),
        (min(ack.v_max, v_goal), ack.max_steering_angle),
        (-ack.v_reverse_max, -ack.max_steering_angle),
        (-ack.v_reverse_max, ack.max_steering_angle),
        (-0.5 * ack.v_reverse_max, 0.0),
    ]
    for i, (vv, dd) in enumerate(anchors[:N]):
        v_seq[i, :] = float(np.clip(vv, -ack.v_reverse_max, ack.v_max))
        delta_seq[i, :] = float(np.clip(dd, -ack.max_steering_angle, ack.max_steering_angle))

    return v_seq, delta_seq


def evaluate_sequences(
    pose: Sequence[float],
    goal: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    current_speed: float,
    current_delta: float,
    v_targets: np.ndarray,
    delta_targets: np.ndarray,
    cfg: Stage1Cfg,
    mpc_cfg: MPCConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized rollout/cost for candidate target control sequences.

    Returns:
      costs: [N]
      first_v: [N]
      first_delta: [N]
      first_omega: [N]
    """
    ack = cfg.ackermann
    N, H = v_targets.shape
    x = np.full((N,), float(pose[0]), dtype=np.float32)
    y = np.full((N,), float(pose[1]), dtype=np.float32)
    th = np.full((N,), float(pose[2]), dtype=np.float32)
    speed = np.full((N,), float(current_speed), dtype=np.float32)
    delta = np.full((N,), float(current_delta), dtype=np.float32)

    costs = np.zeros((N,), dtype=np.float32)
    min_clr_all = np.full((N,), np.inf, dtype=np.float32)
    prev_v = speed.copy()
    prev_delta = delta.copy()
    first_v = np.zeros((N,), dtype=np.float32)
    first_delta = np.zeros((N,), dtype=np.float32)
    first_omega = np.zeros((N,), dtype=np.float32)

    dv_max = ack.max_accel * ack.dt
    ddelta_max = ack.max_steer_rate * ack.dt

    for k in range(H):
        v_cmd = np.clip(v_targets[:, k], speed - dv_max, speed + dv_max)
        v_cmd = np.clip(v_cmd, -ack.v_reverse_max, ack.v_max)
        delta_cmd = np.clip(delta_targets[:, k], delta - ddelta_max, delta + ddelta_max)
        delta_cmd = np.clip(delta_cmd, -ack.max_steering_angle, ack.max_steering_angle)
        omega = v_cmd / ack.wheelbase * np.tan(delta_cmd)

        if k == 0:
            first_v[:] = v_cmd
            first_delta[:] = delta_cmd
            first_omega[:] = omega

        x = x + ack.dt * v_cmd * np.cos(th)
        y = y + ack.dt * v_cmd * np.sin(th)
        th = np.arctan2(np.sin(th + ack.dt * omega), np.cos(th + ack.dt * omega))

        speed = v_cmd
        delta = delta_cmd

        xy = np.stack([x, y], axis=-1)
        dist = np.linalg.norm(xy - goal[None, :], axis=1)
        clr = _min_clearance_batch(xy, C, R, cfg.scene.robot_radius)
        min_clr_all = np.minimum(min_clr_all, clr)

        clear_deficit = np.maximum(0.0, mpc_cfg.safe_margin - clr)
        costs += mpc_cfg.w_running_goal * dist * dist
        costs += mpc_cfg.w_clearance * clear_deficit * clear_deficit
        costs += mpc_cfg.w_collision * (clr < 0.0).astype(np.float32)
        reverse_pen = (v_cmd < -ack.v_min).astype(np.float32)
        costs += mpc_cfg.w_control * (v_cmd * v_cmd + 0.2 * delta_cmd * delta_cmd + 0.15 * reverse_pen)
        costs += mpc_cfg.w_smooth * ((v_cmd - prev_v) ** 2 + 0.5 * (delta_cmd - prev_delta) ** 2)
        costs += mpc_cfg.w_time

        prev_v = v_cmd
        prev_delta = delta_cmd

    final_xy = np.stack([x, y], axis=-1)
    final_dist = np.linalg.norm(final_xy - goal[None, :], axis=1)
    final_heading_des = np.arctan2(goal[1] - y, goal[0] - x)
    heading_err = np.arctan2(np.sin(final_heading_des - th), np.cos(final_heading_des - th))
    costs += mpc_cfg.w_final_goal * final_dist * final_dist
    costs += mpc_cfg.w_heading * heading_err * heading_err

    return costs, first_v, first_delta, first_omega


def mpc_action(
    pose: Sequence[float],
    scene: Dict[str, np.ndarray],
    current_speed: float,
    current_delta: float,
    cfg: Stage1Cfg,
    mpc_cfg: MPCConfig,
    rng: np.random.Generator,
) -> Tuple[float, float, float, Dict[str, float]]:
    v_seq, delta_seq = _sample_control_sequences(
        np.asarray(pose, dtype=np.float32), scene["goal"], current_speed, current_delta, cfg, mpc_cfg, rng
    )
    costs, first_v, first_delta, first_omega = evaluate_sequences(
        pose=pose,
        goal=scene["goal"],
        C=scene["C"],
        R=scene["R"],
        current_speed=current_speed,
        current_delta=current_delta,
        v_targets=v_seq,
        delta_targets=delta_seq,
        cfg=cfg,
        mpc_cfg=mpc_cfg,
    )
    j = int(np.argmin(costs))
    v = float(first_v[j])
    delta = float(first_delta[j])
    omega = float(first_omega[j])
    # Re-apply the twist projection for consistency with the policy command path.
    v, omega, delta_from_projection = project_twist_np(v, omega, cfg.ackermann)
    # Keep steering-rate-limited delta from the MPC rollout; projection mainly protects curvature.
    delta = float(np.clip(delta, -cfg.ackermann.max_steering_angle, cfg.ackermann.max_steering_angle))
    omega = v / cfg.ackermann.wheelbase * math.tan(delta) if abs(v) >= cfg.ackermann.v_min else 0.0
    info = {"best_cost": float(costs[j]), "mean_cost": float(costs.mean()), "std_cost": float(costs.std())}
    return v, omega, delta, info


def rollout_mpc(
    scene: Dict[str, np.ndarray],
    cfg: Stage1Cfg,
    mpc_cfg: Optional[MPCConfig] = None,
    max_steps: int = 160,
    goal_tol: float = 0.35,
    safety_stop: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    mpc_cfg = mpc_cfg or MPCConfig()
    rng = rng or np.random.default_rng(mpc_cfg.seed)
    start = scene["start"]
    goal = scene["goal"]
    d = goal - start
    pose = np.array([start[0], start[1], math.atan2(float(d[1]), float(d[0]))], dtype=np.float32)
    current_speed = 0.0
    current_delta = 0.0

    traj = [pose.copy()]
    cmds = []
    mpc_info = []
    min_clear = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
    collision = min_clear < 0.0
    stopped_by_safety = False
    reverse_steps = 0
    stall_steps = 0
    prev_dist_goal = float(np.linalg.norm(goal - pose[:2]))

    for _ in range(max_steps):
        dist_goal = float(np.linalg.norm(goal - pose[:2]))
        if dist_goal <= goal_tol:
            break
        true_clear = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
        if safety_stop and true_clear < cfg.recovery.imminent_stop_clearance:
            stopped_by_safety = True
            break
        # Render scan for interface parity/noise consumption, although this MPC baseline is geometry-aware.
        _ = render_laserscan(pose, scene["C"], scene["R"], cfg.scan, rng)

        v, omega, delta, info = mpc_action(pose, scene, current_speed, current_delta, cfg, mpc_cfg, rng)
        pose = ackermann_step_np(pose, v, omega, cfg.ackermann.dt)
        current_speed = v
        current_delta = delta
        traj.append(pose.copy())
        cmds.append([v, omega, delta])
        mpc_info.append(info)
        if v < -cfg.ackermann.v_min:
            reverse_steps += 1
        clr = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
        min_clear = min(min_clear, clr)
        if clr < 0.0:
            collision = True
            break
        new_dist_goal = float(np.linalg.norm(goal - pose[:2]))
        progress = prev_dist_goal - new_dist_goal
        if abs(v) < cfg.recovery.stall_speed and progress < cfg.recovery.stall_progress and clr < cfg.recovery.stall_clearance:
            stall_steps += 1
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
        "mpc_info": mpc_info,
    }
