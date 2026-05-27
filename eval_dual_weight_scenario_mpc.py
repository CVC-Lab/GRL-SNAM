#!/usr/bin/env python3
"""Closed-loop scenario evaluation for dual-weight energy policies vs MPC.

This script builds procedural circular-obstacle navigation scenes, runs the
DualWeightOnlineController in fixed/forward/sensitivity modes, runs a simple
geometry-aware sampling MPC baseline, and plots trajectory/command/clearance and
lambda traces.  It complements eval_dual_weight_sensitivity.py, which evaluates
per-snapshot fields from an offline dataset but does not roll out the vehicle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from eval_dual_weight_sensitivity import load_model
    from scripts.dual_weight_energy_nav import (
        AckermannLimitConfig,
        DualWeightOnlineController,
        SensitivityUpdateConfig,
        LAMBDA_NAMES,
    )
    from scripts.integrators import AckermannIntegratorConfig, ackermann_pose_step_np, ackermann_pose_step_np_vec
except Exception:  # pragma: no cover
    from eval_dual_weight_sensitivity import load_model
    from dual_weight_energy_nav import (
        AckermannLimitConfig,
        DualWeightOnlineController,
        SensitivityUpdateConfig,
        LAMBDA_NAMES,
    )
    from scripts.integrators import AckermannIntegratorConfig, ackermann_pose_step_np, ackermann_pose_step_np_vec


# -----------------------------------------------------------------------------
# Geometry, maps, and synthetic batch construction
# -----------------------------------------------------------------------------

def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def world_to_local(P_w: np.ndarray, pose: np.ndarray) -> np.ndarray:
    P_w = np.asarray(P_w, dtype=np.float32)
    R = rot2(float(pose[2]))
    return (P_w - pose[None, :2]) @ R


def local_to_world(P_l: np.ndarray, pose: np.ndarray) -> np.ndarray:
    P_l = np.asarray(P_l, dtype=np.float32)
    R = rot2(float(pose[2]))
    return P_l @ R.T + pose[None, :2]


def make_coord_grid(H: int, W: int, half: float) -> torch.Tensor:
    xs = torch.linspace(-half, half, W)
    ys = torch.linspace(-half, half, H)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).float()


def barrier_map_from_sd(sd: torch.Tensor, hat_d: float, eps: float = 1e-6, max_b: float = 50.0) -> torch.Tensor:
    dh = sd.new_tensor(float(hat_d))
    safe = torch.clamp(sd, min=eps)
    b_in = -(sd - dh) ** 2 * torch.log(safe / dh)
    b = torch.where(sd < dh, b_in, torch.zeros_like(sd))
    b = torch.where(sd <= eps, dh * dh * (-torch.log(sd.new_tensor(eps) / dh)) + (eps - sd) ** 2, b)
    return torch.clamp(b, 0.0, max_b)


def quadratic_goal(grid_xy: torch.Tensor, goal_l: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((grid_xy - goal_l.view(1, 1, 2)) ** 2).sum(dim=-1)


def build_local_batch(
    pose: np.ndarray,
    goal: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    H: int,
    W: int,
    hat_d: float,
    device: str,
) -> Dict[str, Any]:
    """Build a B=1 NavMax-style batch around current pose."""
    grid_xy = make_coord_grid(H, W, hat_d)
    C_l = world_to_local(C, pose) if C.size else np.zeros((0, 2), dtype=np.float32)
    goal_l = world_to_local(goal.reshape(1, 2), pose)[0]

    keep = np.ones((C_l.shape[0],), dtype=bool)
    if C_l.shape[0]:
        keep = np.max(np.abs(C_l), axis=1) <= (hat_d + R)
        C_l = C_l[keep]
        R_l = R[keep]
    else:
        R_l = np.zeros((0,), dtype=np.float32)

    barrier_maps: List[torch.Tensor] = []
    obs_feats: List[List[float]] = []
    for c, r in zip(C_l, R_l):
        c_t = torch.tensor(c, dtype=torch.float32)
        sd = torch.linalg.norm(grid_xy - c_t.view(1, 1, 2), dim=-1) - float(r)
        barrier_maps.append(barrier_map_from_sd(sd, hat_d))
        norm = float(np.linalg.norm(c))
        ang = float(math.atan2(float(c[1]), float(c[0])))
        prox = float((hat_d - max(norm - float(r), 0.0)) / max(hat_d, 1e-6))
        obs_feats.append([float(c[0]), float(c[1]), float(r), norm, ang, float(np.clip(prox, 0.0, 1.0))])

    if barrier_maps:
        barrier_stack = torch.stack(barrier_maps, dim=0)
        obs = torch.tensor(obs_feats, dtype=torch.float32)
        obs_mask = torch.ones(obs.shape[0], dtype=torch.bool)
    else:
        barrier_stack = torch.zeros(0, H, W, dtype=torch.float32)
        obs = torch.zeros(0, 6, dtype=torch.float32)
        obs_mask = torch.zeros(0, dtype=torch.bool)

    goal_l_t = torch.tensor(goal_l, dtype=torch.float32)
    goal_map = quadratic_goal(grid_xy, goal_l_t).unsqueeze(0)
    coord_map = torch.stack([grid_xy[..., 0] / hat_d, grid_xy[..., 1] / hat_d], dim=0)
    gl_norm = torch.linalg.norm(goal_l_t)
    gl_ang = torch.atan2(goal_l_t[1], goal_l_t[0])
    goal_feats = torch.stack([goal_l_t[0], goal_l_t[1], gl_norm, gl_ang])
    dir_xy = goal_l_t / gl_norm.clamp_min(1e-8)

    batch = {
        "grid_xy": grid_xy.unsqueeze(0).to(device),
        "goal_map": goal_map.unsqueeze(0).to(device),
        "coord_map": coord_map.unsqueeze(0).to(device),
        "barrier_stack": barrier_stack.unsqueeze(0).to(device),
        "obs_mask": obs_mask.unsqueeze(0).to(device),
        "obs_feats": obs.unsqueeze(0).to(device),
        "obs_weights": torch.ones(1, obs.shape[0], device=device),
        "goal_feats": goal_feats.unsqueeze(0).to(device),
        "pos_xy": torch.zeros(1, 2, device=device),
        "dir_xy": dir_xy.view(1, 2).to(device),
        "meta": {
            "hat_d": torch.tensor([hat_d], dtype=torch.float32, device=device),
            "episode": torch.tensor([0], dtype=torch.long, device=device),
            "stage": torch.tensor([0], dtype=torch.long, device=device),
            "snap": torch.tensor([0], dtype=torch.long, device=device),
            "success": torch.tensor([1], dtype=torch.long, device=device),
        },
    }
    return batch


def min_clearance(xy: np.ndarray, C: np.ndarray, R: np.ndarray, robot_radius: float) -> float:
    if C.size == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(C - xy[None, :], axis=1) - R - robot_radius))


def vehicle_step(pose: np.ndarray, u: np.ndarray, limit_cfg: AckermannLimitConfig, wheelbase: float, scheme: str = "semi_implicit") -> np.ndarray:
    cfg = AckermannIntegratorConfig(scheme=scheme, wheelbase=wheelbase, dt=limit_cfg.dt)
    return ackermann_pose_step_np(pose, u, cfg)


# -----------------------------------------------------------------------------
# Procedural scenarios and MPC baseline
# -----------------------------------------------------------------------------

@dataclass
class Scenario:
    start: np.ndarray
    goal: np.ndarray
    C: np.ndarray
    R: np.ndarray
    name: str
    start_theta: Optional[float] = None
    goal_theta: Optional[float] = None


def scenario_from_record(record: Dict[str, Any]) -> Scenario:
    """Convert a saved Stage-1 scene-bank record into this evaluator's Scenario."""
    start = np.asarray(record.get("start", [0.0, 0.0]), dtype=np.float32)[:2]
    goal = np.asarray(record.get("goal", [1.0, 0.0]), dtype=np.float32)[:2]
    C = np.asarray(record.get("C", []), dtype=np.float32).reshape(-1, 2) if len(record.get("C", [])) else np.zeros((0, 2), dtype=np.float32)
    R = np.asarray(record.get("R", []), dtype=np.float32).reshape(-1) if len(record.get("R", [])) else np.zeros((0,), dtype=np.float32)
    name = str(record.get("case", record.get("name", "scene_bank")))
    st = record.get("start_heading", None)
    gt = record.get("goal_heading", None)
    return Scenario(
        start=start, goal=goal, C=C, R=R, name=name,
        start_theta=None if st is None else float(st),
        goal_theta=None if gt is None else float(gt),
    )


def load_scene_bank(path: str, case_filter: str = "") -> List[Scenario]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "scenes" in obj:
        records = list(obj["scenes"])
    elif isinstance(obj, dict):
        # Support {"test": [...]} or similar custom banks.
        records = []
        for v in obj.values():
            if isinstance(v, list):
                records.extend(v)
    elif isinstance(obj, list):
        records = obj
    else:
        raise ValueError(f"Could not interpret scene bank {path!r}")
    cases = {c.strip() for c in case_filter.split(",") if c.strip()}
    if cases:
        records = [r for r in records if str(r.get("case", "")) in cases]
    return [scenario_from_record(r) for r in records]


def sample_scenario(kind: str, rng: np.random.Generator) -> Scenario:
    start = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([7.5, 0.0], dtype=np.float32)
    if kind == "front_trap":
        C = np.array([[1.05, 0.0], [2.25, 0.70], [2.25, -0.70], [4.0, 0.45], [5.6, -0.45]], dtype=np.float32)
        R = np.array([0.38, 0.45, 0.45, 0.42, 0.42], dtype=np.float32)
    elif kind == "narrow_gate":
        C = np.array([[2.2, 0.65], [2.2, -0.65], [4.4, 0.65], [4.4, -0.65], [5.8, 0.0]], dtype=np.float32)
        R = np.array([0.38, 0.38, 0.42, 0.42, 0.30], dtype=np.float32)
    elif kind == "s_turn":
        C = np.array([[1.4, 0.55], [2.5, -0.55], [3.6, 0.55], [4.7, -0.55], [5.8, 0.55]], dtype=np.float32)
        R = np.array([0.40, 0.40, 0.40, 0.40, 0.40], dtype=np.float32)
    elif kind == "parallel_parking":
        # The car starts ahead of the slot, heading +x.  The goal lies behind
        # and toward the curb, so the local goal has negative x and requires a
        # reverse command when --signed-speed is enabled.
        start = np.array([1.25, 0.55], dtype=np.float32)
        goal = np.array([0.00, -0.55], dtype=np.float32)
        C = np.array([
            [-2.05, -0.55], [-1.55, -0.55],   # rear parked car
            [ 1.55, -0.55], [ 2.05, -0.55],   # front parked car
            [-1.75, -1.18], [-0.60, -1.20], [0.60, -1.20], [1.75, -1.18],
            [ 0.95,  0.15],
        ], dtype=np.float32)
        R = np.array([0.33, 0.33, 0.33, 0.33, 0.18, 0.18, 0.18, 0.18, 0.20], dtype=np.float32)
        return Scenario(start=start, goal=goal, C=C, R=R, name=kind, start_theta=0.0, goal_theta=0.0)
    else:
        n = int(rng.integers(7, 14))
        C_list, R_list = [], []
        tries = 0
        while len(C_list) < n and tries < 5000:
            tries += 1
            c = np.array([rng.uniform(1.0, 7.0), rng.uniform(-1.8, 1.8)], dtype=np.float32)
            r = float(rng.uniform(0.22, 0.48))
            if np.linalg.norm(c - start) < r + 0.85 or np.linalg.norm(c - goal) < r + 0.85:
                continue
            # keep a loose central passage sometimes, but not always
            if rng.uniform() < 0.65 and abs(c[1]) < r + 0.35:
                continue
            if C_list:
                d = np.min(np.linalg.norm(np.stack(C_list) - c[None, :], axis=1) - np.asarray(R_list))
                if d < r + 0.12:
                    continue
            C_list.append(c); R_list.append(r)
        C = np.asarray(C_list, dtype=np.float32)
        R = np.asarray(R_list, dtype=np.float32)
    return Scenario(start=start, goal=goal, C=C, R=R, name=kind)


@dataclass
class MPCConfig:
    horizon: int = 12
    samples: int = 512
    v_nominal: float = 0.55
    p_reverse: float = 0.20
    safe_margin: float = 0.15
    w_goal: float = 14.0
    w_running: float = 0.20
    w_clear: float = 50.0
    w_collision: float = 1.0e5
    w_control: float = 0.04
    w_smooth: float = 0.10
    integrator: str = "semi_implicit"


def mpc_action(
    pose: np.ndarray,
    goal: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    u_prev: np.ndarray,
    limit_cfg: AckermannLimitConfig,
    wheelbase: float,
    robot_radius: float,
    cfg: MPCConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    N, H = int(cfg.samples), int(cfg.horizon)
    v_seq = rng.uniform(limit_cfg.v_min, limit_cfg.v_max, size=(N, H)).astype(np.float32)
    d_seq = rng.uniform(-limit_cfg.steering_max, limit_cfg.steering_max, size=(N, H)).astype(np.float32)
    # goal-biased anchors
    dx, dy = float(goal[0] - pose[0]), float(goal[1] - pose[1])
    alpha = wrap_angle(math.atan2(dy, dx) - float(pose[2]))
    lookahead = max(0.7, min(3.0, math.hypot(dx, dy)))
    d_goal = float(np.clip(math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead), -limit_cfg.steering_max, limit_cfg.steering_max))
    n_bias = max(1, N // 2)
    v_seq[:n_bias] = np.clip(rng.normal(cfg.v_nominal, 0.18, size=(n_bias, H)), 0.0, limit_cfg.v_max)
    d_seq[:n_bias] = np.clip(rng.normal(d_goal, 0.20, size=(n_bias, H)), -limit_cfg.steering_max, limit_cfg.steering_max)
    n_rev = int(cfg.p_reverse * N)
    if n_rev > 0:
        v_seq[-n_rev:] = rng.uniform(limit_cfg.v_min, -0.04, size=(n_rev, H)).astype(np.float32)
    anchors = [(cfg.v_nominal, d_goal), (0.20, d_goal), (cfg.v_nominal, 0.0), (limit_cfg.v_min, -limit_cfg.steering_max), (limit_cfg.v_min, limit_cfg.steering_max)]
    for i, (v, d) in enumerate(anchors[:N]):
        v_seq[i, :] = np.clip(v, limit_cfg.v_min, limit_cfg.v_max)
        d_seq[i, :] = np.clip(d, -limit_cfg.steering_max, limit_cfg.steering_max)

    x = np.full(N, pose[0], dtype=np.float32); y = np.full(N, pose[1], dtype=np.float32); th = np.full(N, pose[2], dtype=np.float32)
    v_prev = np.full(N, u_prev[0], dtype=np.float32); d_prev = np.full(N, u_prev[1], dtype=np.float32)
    cost = np.zeros(N, dtype=np.float32)
    dv_max = limit_cfg.accel_max * limit_cfg.dt
    dd_max = limit_cfg.steering_rate_max * limit_cfg.dt
    first_v = np.zeros(N, dtype=np.float32); first_d = np.zeros(N, dtype=np.float32)
    for k in range(H):
        v = np.clip(v_seq[:, k], v_prev - dv_max, v_prev + dv_max)
        v = np.clip(v, limit_cfg.v_min, limit_cfg.v_max)
        d = np.clip(d_seq[:, k], d_prev - dd_max, d_prev + dd_max)
        d = np.clip(d, -limit_cfg.steering_max, limit_cfg.steering_max)
        if k == 0:
            first_v[:] = v; first_d[:] = d
        x, y, th = ackermann_pose_step_np_vec(
            x, y, th, v, d,
            AckermannIntegratorConfig(scheme=getattr(cfg, "integrator", "semi_implicit"), wheelbase=wheelbase, dt=limit_cfg.dt),
        )
        xy = np.stack([x, y], axis=-1)
        dist = np.linalg.norm(xy - goal[None, :], axis=1)
        if C.size:
            clear = np.min(np.linalg.norm(xy[:, None, :] - C[None, :, :], axis=-1) - R[None, :] - robot_radius, axis=1)
        else:
            clear = np.full(N, np.inf, dtype=np.float32)
        deficit = np.maximum(0.0, cfg.safe_margin - clear)
        cost += cfg.w_running * dist * dist + cfg.w_clear * deficit * deficit + cfg.w_collision * (clear < 0.0).astype(np.float32)
        cost += cfg.w_control * (v * v + 0.2 * d * d) + cfg.w_smooth * ((v - v_prev) ** 2 + 0.5 * (d - d_prev) ** 2)
        v_prev = v; d_prev = d
    final = np.linalg.norm(np.stack([x, y], axis=-1) - goal[None, :], axis=1)
    cost += cfg.w_goal * final * final
    j = int(np.argmin(cost))
    return np.array([first_v[j], first_d[j]], dtype=np.float32)


# -----------------------------------------------------------------------------
# Rollout and plotting
# -----------------------------------------------------------------------------

def rollout_dual(
    controller: DualWeightOnlineController,
    scenario: Scenario,
    mode: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    theta0 = float(scenario.start_theta) if scenario.start_theta is not None else math.atan2(float(scenario.goal[1] - scenario.start[1]), float(scenario.goal[0] - scenario.start[0]))
    pose = np.array([scenario.start[0], scenario.start[1], theta0], dtype=np.float32)
    controller.reset()
    traj = [pose.copy()]
    cmds: List[np.ndarray] = []
    lambdas: List[np.ndarray] = []
    min_clr = min_clearance(pose[:2], scenario.C, scenario.R, args.robot_radius)
    stall_steps = 0
    reverse_steps = 0
    prev_dist = float(np.linalg.norm(scenario.goal - pose[:2]))
    field_snaps: Dict[int, Dict[str, np.ndarray]] = {}
    for t in range(args.max_steps):
        dist = float(np.linalg.norm(scenario.goal - pose[:2]))
        heading_err_now = 0.0 if scenario.goal_theta is None else abs(wrap_angle(float(pose[2]) - float(scenario.goal_theta)))
        if dist <= args.goal_tol and (scenario.goal_theta is None or heading_err_now <= args.goal_heading_tol):
            break
        batch = build_local_batch(pose, scenario.goal, scenario.C, scenario.R, args.H, args.W, args.hat_d, args.device)
        out = controller.step(batch, adaptation=mode)
        u = out["u"][0].detach().cpu().numpy().astype(np.float32)
        pose = vehicle_step(pose, u, controller.limit_cfg, args.wheelbase, scheme=args.ackermann_integrator)
        traj.append(pose.copy()); cmds.append(u); lambdas.append(out["lambda"][0].detach().cpu().numpy())
        c = min_clearance(pose[:2], scenario.C, scenario.R, args.robot_radius)
        min_clr = min(min_clr, c)
        if u[0] < -0.04:
            reverse_steps += 1
        new_dist = float(np.linalg.norm(scenario.goal - pose[:2]))
        if abs(u[0]) < 0.05 and (prev_dist - new_dist) < 1e-3:
            stall_steps += 1
        prev_dist = new_dist
        if t in args.field_steps_set:
            field_snaps[t] = {
                "U": out["U"][0, 0].detach().cpu().numpy(),
                "F": out["F"][0].detach().cpu().numpy(),
                "lambda": out["lambda"][0].detach().cpu().numpy(),
                "pose": pose.copy(),
                "hat_d": float(args.hat_d),
            }
        if c < -0.02:
            break
    traj_arr = np.asarray(traj, dtype=np.float32)
    cmds_arr = np.asarray(cmds, dtype=np.float32) if cmds else np.zeros((0, 2), dtype=np.float32)
    lam_arr = np.asarray(lambdas, dtype=np.float32) if lambdas else np.zeros((0, len(LAMBDA_NAMES)), dtype=np.float32)
    final_dist = float(np.linalg.norm(scenario.goal - traj_arr[-1, :2]))
    final_heading_err = 0.0 if scenario.goal_theta is None else abs(wrap_angle(float(traj_arr[-1, 2]) - float(scenario.goal_theta)))
    heading_ok = True if scenario.goal_theta is None else final_heading_err <= float(getattr(args, "goal_heading_tol", 0.55))
    path_len = float(np.linalg.norm(np.diff(traj_arr[:, :2], axis=0), axis=1).sum()) if len(traj_arr) > 1 else 0.0
    return {
        "traj": traj_arr,
        "cmds": cmds_arr,
        "lambda": lam_arr,
        "field_snaps": field_snaps,
        "success": bool(final_dist <= args.goal_tol and min_clr >= -0.02 and heading_ok),
        "collision": bool(min_clr < -0.02),
        "final_dist": final_dist,
        "final_heading_err": float(final_heading_err),
        "min_clearance": float(min_clr),
        "path_length": path_len,
        "steps": int(len(traj_arr) - 1),
        "stall_steps": int(stall_steps),
        "reverse_steps": int(reverse_steps),
    }


def rollout_mpc_scenario(scenario: Scenario, args: argparse.Namespace, rng: np.random.Generator) -> Dict[str, Any]:
    limit_cfg = AckermannLimitConfig(args.v_min, args.v_max, args.steering_max, args.accel_max, args.steering_rate_max, args.dt)
    mpc_cfg = MPCConfig(horizon=args.mpc_horizon, samples=args.mpc_samples, p_reverse=args.mpc_p_reverse, integrator=args.ackermann_integrator)
    theta0 = float(scenario.start_theta) if scenario.start_theta is not None else math.atan2(float(scenario.goal[1] - scenario.start[1]), float(scenario.goal[0] - scenario.start[0]))
    pose = np.array([scenario.start[0], scenario.start[1], theta0], dtype=np.float32)
    u_prev = np.zeros(2, dtype=np.float32)
    traj = [pose.copy()]; cmds = []
    min_clr = min_clearance(pose[:2], scenario.C, scenario.R, args.robot_radius)
    for t in range(args.max_steps):
        dist_now = float(np.linalg.norm(scenario.goal - pose[:2]))
        heading_err_now = 0.0 if scenario.goal_theta is None else abs(wrap_angle(float(pose[2]) - float(scenario.goal_theta)))
        if dist_now <= args.goal_tol and (scenario.goal_theta is None or heading_err_now <= args.goal_heading_tol):
            break
        u = mpc_action(pose, scenario.goal, scenario.C, scenario.R, u_prev, limit_cfg, args.wheelbase, args.robot_radius, mpc_cfg, rng)
        pose = vehicle_step(pose, u, limit_cfg, args.wheelbase, scheme=args.ackermann_integrator)
        u_prev = u
        traj.append(pose.copy()); cmds.append(u)
        min_clr = min(min_clr, min_clearance(pose[:2], scenario.C, scenario.R, args.robot_radius))
        if min_clr < -0.02:
            break
    traj_arr = np.asarray(traj, dtype=np.float32)
    cmds_arr = np.asarray(cmds, dtype=np.float32) if cmds else np.zeros((0, 2), dtype=np.float32)
    final_dist = float(np.linalg.norm(scenario.goal - traj_arr[-1, :2]))
    final_heading_err = 0.0 if scenario.goal_theta is None else abs(wrap_angle(float(traj_arr[-1, 2]) - float(scenario.goal_theta)))
    heading_ok = True if scenario.goal_theta is None else final_heading_err <= float(getattr(args, "goal_heading_tol", 0.55))
    return {
        "traj": traj_arr, "cmds": cmds_arr, "lambda": np.zeros((0, len(LAMBDA_NAMES))), "field_snaps": {},
        "success": bool(final_dist <= args.goal_tol and min_clr >= -0.02 and heading_ok),
        "collision": bool(min_clr < -0.02),
        "final_dist": final_dist,
        "final_heading_err": float(final_heading_err),
        "min_clearance": float(min_clr),
        "path_length": float(np.linalg.norm(np.diff(traj_arr[:, :2], axis=0), axis=1).sum()) if len(traj_arr) > 1 else 0.0,
        "steps": int(len(traj_arr) - 1),
        "stall_steps": 0,
        "reverse_steps": int((cmds_arr[:, 0] < -0.04).sum()) if len(cmds_arr) else 0,
    }


def summarize_runs(records: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ["success", "collision", "final_dist", "final_heading_err", "min_clearance", "path_length", "steps", "stall_steps", "reverse_steps"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.asarray([float(r[k]) for r in records], dtype=np.float32)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


def draw_vehicle(ax, pose: np.ndarray, color: str, width: float = 0.30, front: float = 0.36, back: float = 0.14, alpha: float = 0.12):
    box_l = np.array([[front, width/2], [front, -width/2], [-back, -width/2], [-back, width/2]], dtype=np.float32)
    box_w = local_to_world(box_l, pose)
    poly = np.vstack([box_w, box_w[0]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha)
    ax.plot(poly[:, 0], poly[:, 1], color=color, lw=0.8, alpha=min(1.0, alpha + 0.35))


def save_scenario_plot(path: str, scenario: Scenario, by_method: Dict[str, Dict[str, Any]], args: argparse.Namespace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(12.5, 8.0), dpi=150)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_aspect("equal", "box")
    for c, r in zip(scenario.C, scenario.R):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), fill=False, lw=1.5))
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r + args.robot_radius), fill=False, lw=0.6, alpha=0.25))
    for i, (name, ro) in enumerate(by_method.items()):
        tr = ro["traj"]
        ax.plot(tr[:, 0], tr[:, 1], lw=2.0, label=name)
        stride = max(1, len(tr)//8)
        for p in tr[::stride]:
            draw_vehicle(ax, p, f"C{i}", alpha=0.07)
    ax.plot([scenario.start[0]], [scenario.start[1]], "o", ms=6, label="start")
    ax.plot([scenario.goal[0]], [scenario.goal[1]], "*", ms=10, label="goal")
    ax.set_title(f"scenario: {scenario.name}")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="upper left")

    ax2 = fig.add_subplot(2, 2, 2)
    for name, ro in by_method.items():
        tr = ro["traj"]
        d = np.linalg.norm(tr[:, :2] - scenario.goal[None, :], axis=1)
        ax2.plot(np.arange(len(d)), d, label=name)
    ax2.set_title("distance to goal"); ax2.set_xlabel("step"); ax2.grid(True, alpha=0.25); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 2, 3)
    for name, ro in by_method.items():
        tr = ro["traj"]
        c = [min_clearance(p[:2], scenario.C, scenario.R, args.robot_radius) for p in tr]
        ax3.plot(np.arange(len(c)), c, label=name)
    ax3.axhline(0.0, lw=1.0, linestyle=":")
    ax3.set_title("clearance minus robot radius"); ax3.set_xlabel("step"); ax3.grid(True, alpha=0.25); ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 2, 4)
    for name, ro in by_method.items():
        cmd = ro["cmds"]
        if len(cmd):
            ax4.plot(np.arange(len(cmd)), cmd[:, 0], label=f"{name} v")
            ax4.plot(np.arange(len(cmd)), cmd[:, 1], linestyle="--", label=f"{name} delta")
    ax4.axhline(args.steering_max, lw=0.8, linestyle=":")
    ax4.axhline(-args.steering_max, lw=0.8, linestyle=":")
    ax4.set_title("speed and steering"); ax4.set_xlabel("step"); ax4.grid(True, alpha=0.25); ax4.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def save_lambda_plot(path: str, by_method: Dict[str, Dict[str, Any]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(11.0, 7.0), dpi=150)
    for j, method in enumerate([m for m in by_method if by_method[m]["lambda"].size], start=1):
        ax = fig.add_subplot(3, 1, j)
        lam = by_method[method]["lambda"]
        for k, n in enumerate(LAMBDA_NAMES):
            ax.plot(np.arange(lam.shape[0]), lam[:, k], label=n)
        ax.set_title(method); ax.grid(True, alpha=0.25); ax.legend(fontsize=6, ncol=4)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def save_energy_snapshots(path_prefix: str, by_method: Dict[str, Dict[str, Any]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    steps = sorted(set().union(*[set(ro["field_snaps"].keys()) for ro in by_method.values()]))
    for step in steps:
        methods = [m for m, ro in by_method.items() if step in ro["field_snaps"]]
        if not methods:
            continue
        fig = plt.figure(figsize=(4.2 * len(methods), 7.5), dpi=150)
        for i, m in enumerate(methods):
            snap = by_method[m]["field_snaps"][step]
            U, F = snap["U"], snap["F"]
            ax = fig.add_subplot(2, len(methods), i + 1)
            im = ax.imshow(U, origin="lower"); ax.set_title(f"{m}: U")
            ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.045)
            ax2 = fig.add_subplot(2, len(methods), len(methods) + i + 1)
            norm = np.sqrt(F[0]**2 + F[1]**2)
            im2 = ax2.imshow(norm, origin="lower")
            H, W = U.shape; stride = max(4, H//12); yy, xx = np.mgrid[0:H:stride, 0:W:stride]
            ax2.quiver(xx, yy, F[0, ::stride, ::stride], F[1, ::stride, ::stride], angles="xy", scale_units="xy", scale=1.0)
            ax2.set_title(f"{m}: |F| + arrows"); ax2.set_xticks([]); ax2.set_yticks([]); fig.colorbar(im2, ax=ax2, fraction=0.045)
        fig.suptitle(f"energy/field snapshots at step {step}")
        fig.tight_layout(); fig.savefig(f"{path_prefix}_step_{step:04d}.png", bbox_inches="tight"); plt.close(fig)


def save_global_energy_snapshots(path_prefix: str, scenario: Scenario, by_method: Dict[str, Dict[str, Any]], args: argparse.Namespace):
    """Plot local energy/force fields transformed into world coordinates.

    This makes the energy controller diagnostic global-scale: each snapshot is
    drawn over the same scenario map and trajectory, rather than as a local image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    steps = sorted(set().union(*[set(ro["field_snaps"].keys()) for ro in by_method.values()]))
    if not steps:
        return
    for step in steps:
        methods = [m for m, ro in by_method.items() if step in ro["field_snaps"]]
        if not methods:
            continue
        fig = plt.figure(figsize=(5.2 * len(methods), 4.8), dpi=150)
        for i, m in enumerate(methods):
            snap = by_method[m]["field_snaps"][step]
            U, F, pose = snap["U"], snap["F"], snap["pose"]
            H, W = U.shape
            hat_d = float(snap.get("hat_d", args.hat_d))
            xs = np.linspace(-hat_d, hat_d, W, dtype=np.float32)
            ys = np.linspace(-hat_d, hat_d, H, dtype=np.float32)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            P_l = np.stack([xx.ravel(), yy.ravel()], axis=-1)
            P_w = local_to_world(P_l, pose)
            # Rotate local force vectors into world frame.
            th = float(pose[2]); c, ss = math.cos(th), math.sin(th)
            Fx_w = c * F[0] - ss * F[1]
            Fy_w = ss * F[0] + c * F[1]
            ax = fig.add_subplot(1, len(methods), i + 1)
            for cc, rr in zip(scenario.C, scenario.R):
                ax.add_patch(plt.Circle((float(cc[0]), float(cc[1])), float(rr), fill=False, lw=1.2))
                ax.add_patch(plt.Circle((float(cc[0]), float(cc[1])), float(rr + args.robot_radius), fill=False, lw=0.5, alpha=0.25))
            tr = by_method[m]["traj"]
            ax.plot(tr[:, 0], tr[:, 1], lw=2.0, label=m)
            ax.plot([scenario.start[0]], [scenario.start[1]], "o", ms=5)
            ax.plot([scenario.goal[0]], [scenario.goal[1]], "*", ms=8)
            sc = ax.scatter(P_w[:, 0], P_w[:, 1], c=U.ravel(), s=3, alpha=0.45)
            stride = max(6, H // 10)
            Pq_l = np.stack([xx[::stride, ::stride].ravel(), yy[::stride, ::stride].ravel()], axis=-1)
            Pq_w = local_to_world(Pq_l, pose)
            ax.quiver(Pq_w[:, 0], Pq_w[:, 1], Fx_w[::stride, ::stride].ravel(), Fy_w[::stride, ::stride].ravel(),
                      angles="xy", scale_units="xy", scale=8.0, width=0.003, alpha=0.7)
            draw_vehicle(ax, pose, f"C{i}", alpha=0.18)
            ax.set_aspect("equal", "box")
            ax.set_title(f"{m}: global U/F at step {step}")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc="upper left")
            fig.colorbar(sc, ax=ax, fraction=0.045)
        fig.tight_layout()
        fig.savefig(f"{path_prefix}_step_{step:04d}.png", bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Closed-loop dual-weight energy vs MPC scenario evaluator")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="eval_dual_weight_scenario_mpc")
    ap.add_argument("--scenario", type=str, default="front_trap", choices=["front_trap", "narrow_gate", "s_turn", "parallel_parking", "random"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--scene-bank", type=str, default="",
                    help="Optional scenes_test.json from generate_stage1_ackermann_rollouts; evaluates exact saved scenarios.")
    ap.add_argument("--case-filter", type=str, default="",
                    help="Optional comma-separated case names to keep when --scene-bank contains multiple cases.")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--goal-tol", type=float, default=0.35)
    ap.add_argument("--goal-heading-tol", type=float, default=0.55)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--hat-d", type=float, default=3.5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--modes", type=str, default="fixed,forward,sensitivity,mpc")
    ap.add_argument("--plot-count", type=int, default=3)
    ap.add_argument("--field-steps", type=str, default="0,15,40,80")
    ap.add_argument("--no-global-energy", action="store_true", help="Disable global-frame energy/force snapshots.")
    # vehicle limits
    ap.add_argument("--wheelbase", type=float, default=0.324)
    ap.add_argument("--robot-radius", type=float, default=0.23)
    ap.add_argument("--v-min", type=float, default=-0.25)
    ap.add_argument("--v-max", type=float, default=0.80)
    ap.add_argument("--steering-max", type=float, default=0.396)
    ap.add_argument("--accel-max", type=float, default=1.5)
    ap.add_argument("--steering-rate-max", type=float, default=1.5)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--ackermann-integrator", type=str, default="semi_implicit",
                    choices=["explicit", "semi_implicit", "midpoint"],
                    help="Pose integrator for closed-loop Ackermann rollouts and MPC shooting.")
    ap.add_argument("--energy-integrator", type=str, default="semi_implicit",
                    choices=["explicit", "semi_implicit", "midpoint", "velocity_verlet"],
                    help="Local pH/energy integrator used by sensitivity rollouts.")
    ap.add_argument("--integrator-damping", type=float, default=0.20)
    ap.add_argument("--momentum-clip", type=float, default=1.50)
    ap.add_argument("--signed-speed", action="store_true", help="Allow negative local x-force to command reverse speed.")
    # sensitivity knobs
    ap.add_argument("--sens-eta", type=float, default=0.08)
    ap.add_argument("--sens-horizon", type=int, default=8)
    ap.add_argument("--sens-grad-clip", type=float, default=1.0)
    ap.add_argument("--sens-w-goal", type=float, default=1.0)
    ap.add_argument("--sens-w-path", type=float, default=0.15)
    ap.add_argument("--sens-w-barrier", type=float, default=0.25)
    ap.add_argument("--sens-w-align", type=float, default=0.25)
    ap.add_argument("--sens-w-act", type=float, default=0.05)
    ap.add_argument("--sens-w-stall", type=float, default=0.25)
    ap.add_argument("--force-floor", type=float, default=0.06)
    # MPC knobs
    ap.add_argument("--mpc-horizon", type=int, default=12)
    ap.add_argument("--mpc-samples", type=int, default=512)
    ap.add_argument("--mpc-p-reverse", type=float, default=0.25)
    args = ap.parse_args()
    args.field_steps_set = {int(x) for x in args.field_steps.split(",") if x.strip()}

    torch.set_num_threads(max(1, int(args.threads)))
    os.makedirs(args.outdir, exist_ok=True)
    model = load_model(args.ckpt, args.H, args.W, args.device)
    limit_cfg = AckermannLimitConfig(args.v_min, args.v_max, args.steering_max, args.accel_max, args.steering_rate_max, args.dt)
    sens_cfg = SensitivityUpdateConfig(
        eta=args.sens_eta, horizon=args.sens_horizon, dt=args.dt, grad_clip=args.sens_grad_clip,
        w_goal=args.sens_w_goal, w_path=args.sens_w_path, w_barrier=args.sens_w_barrier,
        w_align=args.sens_w_align, w_act=args.sens_w_act, w_stall=args.sens_w_stall,
        force_floor=args.force_floor,
        integrator=args.energy_integrator,
        damping=args.integrator_damping,
        momentum_clip=args.momentum_clip,
        signed_speed=bool(args.signed_speed),
    )
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    controllers = {
        m: DualWeightOnlineController(model, limit_cfg=limit_cfg, sensitivity_cfg=sens_cfg, signed_speed=bool(args.signed_speed))
        for m in modes if m != "mpc"
    }
    rng = np.random.default_rng(args.seed)
    if args.scene_bank:
        scenarios = load_scene_bank(args.scene_bank, args.case_filter)
        if not scenarios:
            raise RuntimeError(f"No scenarios loaded from scene bank {args.scene_bank!r}")
        if args.episodes > 0:
            scenarios = scenarios[: min(int(args.episodes), len(scenarios))]
    else:
        scenarios = [sample_scenario(args.scenario, rng) for _ in range(int(args.episodes))]
    all_records: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    for ep, scenario in enumerate(scenarios):
        by_method: Dict[str, Dict[str, Any]] = {}
        for m in modes:
            if m == "mpc":
                ro = rollout_mpc_scenario(scenario, args, np.random.default_rng(args.seed + 10000 + ep))
            else:
                ro = rollout_dual(controllers[m], scenario, m, args, np.random.default_rng(args.seed + 20000 + ep))
            by_method[m] = ro
            all_records[m].append({k: v for k, v in ro.items() if k not in ("traj", "cmds", "lambda", "field_snaps")})
        if ep < args.plot_count:
            save_scenario_plot(os.path.join(args.outdir, f"scenario_{ep:03d}.png"), scenario, by_method, args)
            save_lambda_plot(os.path.join(args.outdir, f"lambda_{ep:03d}.png"), by_method)
            save_energy_snapshots(os.path.join(args.outdir, f"energy_{ep:03d}"), by_method)
            if not args.no_global_energy:
                save_global_energy_snapshots(os.path.join(args.outdir, f"global_energy_{ep:03d}"), scenario, by_method, args)
    summary = {m: summarize_runs(recs) for m, recs in all_records.items()}
    summary["_eval"] = {"scene_bank": args.scene_bank, "scenario": args.scenario, "episodes": len(scenarios)}
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, "records.json"), "w") as f:
        json.dump(all_records, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
