#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List

import numpy as np
import torch

from eval_dual_weight_sensitivity import load_model
from eval_dual_weight_scenario_mpc import (
    Scenario,
    load_scene_bank,
    sample_scenario,
    build_local_batch,
    local_to_world,
    min_clearance,
    vehicle_step,
    wrap_angle,
)
from scripts.dual_weight_energy_nav import (
    AckermannLimitConfig,
    DualWeightOnlineController,
    SensitivityUpdateConfig,
)


# -----------------------------------------------------------------------------
# Self-contained plotting helpers
# -----------------------------------------------------------------------------

def draw_obstacles(
    ax,
    C: np.ndarray,
    R: np.ndarray,
    *,
    color: str = "k",
    alpha: float = 0.35,
    lw: float = 1.6,
    fill: bool = False,
    label: str | None = None,
) -> None:
    """Draw circular obstacles on a matplotlib axis.

    This local copy intentionally avoids depending on helper functions from
    eval_dual_weight_scenario_mpc.py, because those helpers may not exist in
    older/evolving evaluator versions.
    """
    import matplotlib.patches as patches

    C_arr = np.asarray(C, dtype=np.float32)
    R_arr = np.asarray(R, dtype=np.float32)
    C_arr = C_arr.reshape(-1, 2) if C_arr.size else np.zeros((0, 2), dtype=np.float32)
    R_arr = R_arr.reshape(-1) if R_arr.size else np.zeros((0,), dtype=np.float32)

    used_label = False
    for (cx, cy), rr in zip(C_arr, R_arr):
        patch = patches.Circle(
            (float(cx), float(cy)),
            float(rr),
            fill=fill,
            edgecolor=color,
            facecolor=color if fill else "none",
            linewidth=lw,
            alpha=alpha,
            label=(label if (label and not used_label) else None),
        )
        ax.add_patch(patch)
        used_label = True


def draw_vehicle_box(
    ax,
    pose: np.ndarray,
    wheelbase: float,
    *,
    color: str = "tab:blue",
    alpha: float = 0.25,
    lw: float = 1.2,
    body_length: float | None = None,
    body_width: float | None = None,
) -> None:
    """Draw a simple Ackermann vehicle footprint.

    pose is [x, y, theta] in world coordinates.  The box is only a visualization
    proxy; collision metrics still use the configured circular robot radius.
    """
    import matplotlib.patches as patches

    pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    x, y, th = float(pose_arr[0]), float(pose_arr[1]), float(pose_arr[2])
    L = float(body_length if body_length is not None else max(0.60, 1.8 * float(wheelbase)))
    W = float(body_width if body_width is not None else max(0.30, 0.85 * float(wheelbase)))

    corners = np.array(
        [
            [-0.5 * L, -0.5 * W],
            [ 0.5 * L, -0.5 * W],
            [ 0.5 * L,  0.5 * W],
            [-0.5 * L,  0.5 * W],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s], [s, c]], dtype=np.float32)
    world = corners @ Rm.T + np.array([x, y], dtype=np.float32)

    poly = patches.Polygon(
        world,
        closed=True,
        fill=True,
        facecolor=color,
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(poly)

    # Heading indicator.
    rear = np.array([x, y], dtype=np.float32)
    front = rear + Rm @ np.array([0.5 * L, 0.0], dtype=np.float32)
    ax.plot([rear[0], front[0]], [rear[1], front[1]], color=color, lw=lw, alpha=min(1.0, alpha + 0.35))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Visualize learned short-horizon planner goals along closed-loop trajectory.')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--scenario', type=str, default='parallel_parking', choices=['front_trap', 'narrow_gate', 's_turn', 'parallel_parking', 'random'])
    ap.add_argument('--scene-bank', type=str, default='')
    ap.add_argument('--scene-bank-case-filter', type=str, default='')
    ap.add_argument('--episode-index', type=int, default=0)
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--mode', type=str, default='forward', choices=['fixed', 'forward', 'sensitivity'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', type=str, default='eval_plan_viz')

    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--H', type=int, default=64)
    ap.add_argument('--W', type=int, default=64)
    ap.add_argument('--hat-d', dest='hat_d', type=float, default=3.5)
    ap.add_argument('--dt', type=float, default=0.10)
    ap.add_argument('--max-steps', type=int, default=220)
    ap.add_argument('--goal-tol', type=float, default=0.35)
    ap.add_argument('--goal-heading-tol', type=float, default=0.55)
    ap.add_argument('--wheelbase', type=float, default=0.32)
    ap.add_argument('--robot-radius', type=float, default=0.22)

    ap.add_argument('--v-min', type=float, default=-0.40)
    ap.add_argument('--v-max', type=float, default=0.40)
    ap.add_argument('--steering-max', type=float, default=0.40)
    ap.add_argument('--accel-max', type=float, default=0.20)
    ap.add_argument('--steering-rate-max', type=float, default=0.35)
    ap.add_argument('--ackermann-integrator', type=str, default='semi_implicit')
    ap.add_argument('--signed-speed', action='store_true')

    ap.add_argument('--step-stride', type=int, default=10, help='Plot every Nth local plan snapshot.')
    ap.add_argument('--quiver-scale', type=float, default=1.0)
    ap.add_argument('--save-frames', action='store_true')
    return ap.parse_args()


def rollout_with_plan(
    controller: DualWeightOnlineController,
    scenario: Scenario,
    mode: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    theta0 = float(scenario.start_theta) if scenario.start_theta is not None else math.atan2(float(scenario.goal[1] - scenario.start[1]), float(scenario.goal[0] - scenario.start[0]))
    pose = np.array([scenario.start[0], scenario.start[1], theta0], dtype=np.float32)
    controller.reset()
    traj = [pose.copy()]
    cmds: List[np.ndarray] = []
    lambdas: List[np.ndarray] = []
    plan_goal_world: List[np.ndarray] = []
    plan_seq_world: List[np.ndarray] = []
    local_goal_seq: List[np.ndarray] = []
    clearances: List[float] = [min_clearance(pose[:2], scenario.C, scenario.R, args.robot_radius)]

    for t in range(args.max_steps):
        dist = float(np.linalg.norm(scenario.goal - pose[:2]))
        heading_err_now = 0.0 if scenario.goal_theta is None else abs(wrap_angle(float(pose[2]) - float(scenario.goal_theta)))
        if dist <= args.goal_tol and (scenario.goal_theta is None or heading_err_now <= args.goal_heading_tol):
            break
        batch = build_local_batch(pose, scenario.goal, scenario.C, scenario.R, args.H, args.W, args.hat_d, args.device)
        out = controller.step(batch, adaptation=mode)
        u = out['u'][0].detach().cpu().numpy().astype(np.float32)
        lam = out['lambda'][0].detach().cpu().numpy().astype(np.float32)
        pg = out.get('plan_goal', torch.empty(0))
        ps = out.get('plan_seq', torch.empty(0))
        if isinstance(pg, torch.Tensor) and pg.numel() >= 2:
            pg_l = pg[0].detach().cpu().numpy().astype(np.float32).reshape(1, 2)
            pg_w = local_to_world(pg_l, pose)[0]
            plan_goal_world.append(pg_w.copy())
            local_goal_seq.append(pg_l[0].copy())
        else:
            plan_goal_world.append(np.array([np.nan, np.nan], dtype=np.float32))
            local_goal_seq.append(np.array([np.nan, np.nan], dtype=np.float32))
        if isinstance(ps, torch.Tensor) and ps.numel() >= 2:
            ps_l = ps[0].detach().cpu().numpy().astype(np.float32)
            ps_w = local_to_world(ps_l, pose)
            plan_seq_world.append(ps_w.copy())
        else:
            plan_seq_world.append(np.zeros((0, 2), dtype=np.float32))
        pose = vehicle_step(pose, u, controller.limit_cfg, args.wheelbase, scheme=args.ackermann_integrator)
        traj.append(pose.copy())
        cmds.append(u)
        lambdas.append(lam)
        clearances.append(min_clearance(pose[:2], scenario.C, scenario.R, args.robot_radius))

    traj_arr = np.asarray(traj, dtype=np.float32)
    cmds_arr = np.asarray(cmds, dtype=np.float32) if cmds else np.zeros((0, 2), dtype=np.float32)
    lam_arr = np.asarray(lambdas, dtype=np.float32) if lambdas else np.zeros((0, 8), dtype=np.float32)
    plan_goal_arr = np.asarray(plan_goal_world, dtype=np.float32) if plan_goal_world else np.zeros((0, 2), dtype=np.float32)
    local_goal_arr = np.asarray(local_goal_seq, dtype=np.float32) if local_goal_seq else np.zeros((0, 2), dtype=np.float32)
    return {
        'traj': traj_arr,
        'cmds': cmds_arr,
        'lambda': lam_arr,
        'plan_goal_world': plan_goal_arr,
        'plan_goal_local': local_goal_arr,
        'plan_seq_world': plan_seq_world,
        'clearance': np.asarray(clearances, dtype=np.float32),
    }


def save_plan_figure(path: str, scenario: Scenario, result: Dict[str, Any], args: argparse.Namespace, title: str = '') -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tr = result['traj']
    pgoal = result['plan_goal_world']
    pseq = result['plan_seq_world']
    cmds = result['cmds']
    clearance = result['clearance']
    local_goal = result['plan_goal_local']

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 2, 1)
    draw_obstacles(ax, scenario.C, scenario.R)
    ax.plot(tr[:, 0], tr[:, 1], lw=2.2, color='tab:blue', label='executed trajectory')
    ax.plot([scenario.start[0]], [scenario.start[1]], 'o', ms=7, label='start')
    ax.plot([scenario.goal[0]], [scenario.goal[1]], '*', ms=12, label='global goal')
    if len(pgoal):
        mask = np.isfinite(pgoal[:, 0])
        sc = ax.scatter(pgoal[mask, 0], pgoal[mask, 1], c=np.arange(mask.sum()), s=18, cmap='viridis', label='predicted local goals')
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='step')
    for i in range(0, len(pseq), max(1, int(args.step_stride))):
        seq = pseq[i]
        if seq.shape[0] == 0:
            continue
        ax.plot(seq[:, 0], seq[:, 1], color='tab:orange', alpha=0.35, lw=1.2)
        ax.scatter([seq[0, 0]], [seq[0, 1]], color='tab:orange', s=10, alpha=0.35)
        draw_vehicle_box(ax, tr[i], args.wheelbase, color='tab:blue', alpha=0.10)
    if len(tr) > 1:
        for i in range(0, len(tr), max(1, int(args.step_stride))):
            draw_vehicle_box(ax, tr[i], args.wheelbase, color='tab:blue', alpha=0.15)
    ax.set_title(title or 'Global trajectory with short-horizon plan goals')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    ax2 = fig.add_subplot(2, 2, 2)
    if len(pgoal):
        d_global = np.linalg.norm(tr[:-1, :2] - scenario.goal[None, :], axis=1)
        d_local = np.linalg.norm(pgoal - tr[:-1, :2], axis=1)
        upd = np.zeros((len(pgoal),), dtype=np.float32)
        if len(pgoal) > 1:
            upd[1:] = np.linalg.norm(pgoal[1:] - pgoal[:-1], axis=1)
        ax2.plot(d_global, label='dist to global goal')
        ax2.plot(d_local, label='dist to predicted local goal')
        ax2.plot(upd, label='planner goal update size')
        ax2.set_title('Planning distances and updates')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

    ax3 = fig.add_subplot(2, 2, 3)
    if len(local_goal):
        ax3.plot(local_goal[:, 0], label='plan goal local x')
        ax3.plot(local_goal[:, 1], label='plan goal local y')
    ax3.axhline(0.0, color='k', lw=0.8, alpha=0.5)
    ax3.set_title('Predicted local goal in body frame')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')

    ax4 = fig.add_subplot(2, 2, 4)
    if len(cmds):
        ax4.plot(cmds[:, 0], label='v')
        ax4.plot(cmds[:, 1], '--', label='delta')
    ax4.plot(clearance[:-1], label='signed clearance minus robot radius')
    ax4.axhline(0.0, color='k', lw=0.8, alpha=0.5)
    ax4.set_title('Command and clearance')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_plan_frames(path_prefix: str, scenario: Scenario, result: Dict[str, Any], args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tr = result['traj']
    pseq = result['plan_seq_world']
    pgoal = result['plan_goal_world']
    for t in range(min(len(pseq), len(tr) - 1)):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1, 1, 1)
        draw_obstacles(ax, scenario.C, scenario.R)
        ax.plot(tr[: t + 1, 0], tr[: t + 1, 1], lw=2.0, color='tab:blue', label='trajectory')
        ax.plot([scenario.start[0]], [scenario.start[1]], 'o', ms=6, label='start')
        ax.plot([scenario.goal[0]], [scenario.goal[1]], '*', ms=10, label='global goal')
        seq = pseq[t]
        if seq.shape[0] > 0:
            ax.plot(seq[:, 0], seq[:, 1], '-o', color='tab:orange', alpha=0.8, label='predicted plan')
        if t < len(pgoal) and np.isfinite(pgoal[t, 0]):
            ax.scatter([pgoal[t, 0]], [pgoal[t, 1]], s=40, color='tab:red', label='selected local goal')
        draw_vehicle_box(ax, tr[t], args.wheelbase, color='tab:blue', alpha=0.25)
        ax.set_title(f'step {t}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(f'{path_prefix}_{t:04d}.png', bbox_inches='tight')
        plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    model = load_model(args.ckpt, args.device, H=args.H, W=args.W)
    limit_cfg = AckermannLimitConfig(args.v_min, args.v_max, args.steering_max, args.accel_max, args.steering_rate_max, args.dt)
    sens_cfg = SensitivityUpdateConfig()
    controller = DualWeightOnlineController(model, limit_cfg=limit_cfg, sensitivity_cfg=sens_cfg, signed_speed=args.signed_speed)

    if args.scene_bank:
        scenarios = load_scene_bank(args.scene_bank, case_filter=args.scene_bank_case_filter)
    else:
        rng = np.random.default_rng(args.seed)
        scenarios = [sample_scenario(args.scenario, rng) for _ in range(max(1, args.episodes))]

    ep0 = max(0, int(args.episode_index))
    ep1 = min(len(scenarios), ep0 + max(1, int(args.episodes)))
    for ep in range(ep0, ep1):
        scenario = scenarios[ep]
        result = rollout_with_plan(controller, scenario, args.mode, args)
        title = f'{scenario.name}: {args.mode} | plan_goal_norm(mean)={np.nanmean(np.linalg.norm(result["plan_goal_local"], axis=1)):.3f}' if len(result['plan_goal_local']) else f'{scenario.name}: {args.mode}'
        save_plan_figure(os.path.join(args.outdir, f'plan_viz_{ep:03d}.png'), scenario, result, args, title=title)
        if args.save_frames:
            save_plan_frames(os.path.join(args.outdir, f'plan_viz_frames_{ep:03d}'), scenario, result, args)


if __name__ == '__main__':
    main()
