#!/usr/bin/env python3
"""Compare the pH/Hamiltonian coefficient policy with the sampling MPC baseline."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch

from eval_stage1_ph_energy import load_model, make_ph_policy_fn
from scripts.stage1_ackermann_scan import (
    Stage1Cfg,
    min_clearance_to_obstacles,
    rollout_policy,
    sample_scene,
    stage1_cfg_to_dict,
    _plot_vehicle_box_single,
    _plot_vehicle_boxes,
)
from scripts.stage1_mpc import MPCConfig, mpc_cfg_to_dict, rollout_mpc


def _metric_without_arrays(ro: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in ro.items() if k not in ("traj", "cmds", "mpc_info")}


def summarize(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ["success", "collision", "stopped_by_safety"]:
        out[k + "_rate"] = float(np.mean([1.0 if m[k] else 0.0 for m in metrics]))
    for k in ["min_clearance", "path_length", "final_dist", "steps", "stall_steps", "reverse_steps"]:
        vals = np.asarray([m[k] for m in metrics], dtype=np.float32)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


def _distance_curve(traj: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.linalg.norm(traj[:, :2] - goal[None, :], axis=1)


def _clearance_curve(traj: np.ndarray, scene: Dict[str, np.ndarray], cfg: Stage1Cfg) -> np.ndarray:
    vals = [min_clearance_to_obstacles(p[:2], scene["C"], scene["R"], cfg.scene.robot_radius) for p in traj]
    return np.asarray(vals, dtype=np.float32)


def save_comparison_plot(path: str, scene: Dict[str, np.ndarray], policy_ro: Dict[str, Any],
                         mpc_ro: Dict[str, Any], cfg: Stage1Cfg, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(11.5, 8.0), dpi=150)

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.set_aspect("equal", "box")
    for c, r in zip(scene["C"], scene["R"]):
        ax0.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), fill=False, lw=1.4))
        ax0.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r + cfg.scene.robot_radius), fill=False, lw=0.7, alpha=0.35))
    p_tr = policy_ro["traj"]
    m_tr = mpc_ro["traj"]
    ax0.plot(p_tr[:, 0], p_tr[:, 1], lw=2.0, label="pH energy policy")
    ax0.plot(m_tr[:, 0], m_tr[:, 1], lw=2.0, linestyle="--", label="MPC")
    _plot_vehicle_boxes(ax0, p_tr, cfg.ackermann, color="C0", stride=max(1, len(p_tr)//10), alpha=0.10, linestyle='-')
    _plot_vehicle_boxes(ax0, m_tr, cfg.ackermann, color="C1", stride=max(1, len(m_tr)//10), alpha=0.08, linestyle='--')
    _plot_vehicle_box_single(ax0, p_tr[0, :3], cfg.ackermann, color="C0", alpha=0.16, linestyle='-')
    _plot_vehicle_box_single(ax0, p_tr[-1, :3], cfg.ackermann, color="C0", alpha=0.22, linestyle='-')
    _plot_vehicle_box_single(ax0, m_tr[-1, :3], cfg.ackermann, color="C1", alpha=0.18, linestyle='--')
    ax0.plot([scene["start"][0]], [scene["start"][1]], "o", ms=6, label="start")
    ax0.plot([scene["goal"][0]], [scene["goal"][1]], "*", ms=10, label="goal")
    ax0.set_xlim(cfg.scene.world_x[0], cfg.scene.world_x[1])
    ax0.set_ylim(cfg.scene.world_y[0], cfg.scene.world_y[1])
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8, loc="upper left")
    ax0.set_title("closed-loop trajectories + car boxes")

    ax1 = fig.add_subplot(2, 2, 2)
    p_dist = _distance_curve(p_tr, scene["goal"])
    m_dist = _distance_curve(m_tr, scene["goal"])
    ax1.plot(np.arange(len(p_dist)), p_dist, label="pH energy policy")
    ax1.plot(np.arange(len(m_dist)), m_dist, linestyle="--", label="MPC")
    ax1.set_xlabel("step")
    ax1.set_ylabel("distance to goal [m]")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    ax1.set_title("goal convergence")

    ax2 = fig.add_subplot(2, 2, 3)
    p_clr = _clearance_curve(p_tr, scene, cfg)
    m_clr = _clearance_curve(m_tr, scene, cfg)
    ax2.plot(np.arange(len(p_clr)), p_clr, label="pH energy policy")
    ax2.plot(np.arange(len(m_clr)), m_clr, linestyle="--", label="MPC")
    ax2.axhline(0.0, lw=1.0, linestyle=":")
    ax2.set_xlabel("step")
    ax2.set_ylabel("clearance minus robot radius [m]")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)
    ax2.set_title("safety margin")

    ax3 = fig.add_subplot(2, 2, 4)
    p_cmd = policy_ro["cmds"]
    m_cmd = mpc_ro["cmds"]
    if len(p_cmd):
        ax3.plot(np.arange(len(p_cmd)), p_cmd[:, 0], label="pH v")
        ax3.plot(np.arange(len(p_cmd)), p_cmd[:, 2], label="pH steering")
    if len(m_cmd):
        ax3.plot(np.arange(len(m_cmd)), m_cmd[:, 0], linestyle="--", label="MPC v")
        ax3.plot(np.arange(len(m_cmd)), m_cmd[:, 2], linestyle="--", label="MPC steering")
    ax3.axhline(cfg.ackermann.max_steering_angle, lw=0.8, linestyle=":")
    ax3.axhline(-cfg.ackermann.max_steering_angle, lw=0.8, linestyle=":")
    ax3.set_xlabel("step")
    ax3.set_ylabel("command")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=7, ncol=2)
    ax3.set_title("speed and steering")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Compare Stage-1 pH energy policy against sampling MPC")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--goal-tol", type=float, default=0.35)
    ap.add_argument("--outdir", type=str, default="eval_ph_vs_mpc")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--plot-count", type=int, default=6)
    ap.add_argument("--mpc-horizon", type=int, default=12)
    ap.add_argument("--mpc-samples", type=int, default=512)
    ap.add_argument("--mpc-safe-margin", type=float, default=0.20)
    ap.add_argument("--mpc-v-nominal", type=float, default=0.55)
    ap.add_argument("--mpc-p-reverse", type=float, default=0.25)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    os.makedirs(args.outdir, exist_ok=True)
    model, cfg, _ = load_model(args.ckpt, args.device)
    policy_fn = make_ph_policy_fn(model, cfg, args.device)
    mpc_cfg = MPCConfig(
        horizon=args.mpc_horizon,
        n_samples=args.mpc_samples,
        safe_margin=args.mpc_safe_margin,
        v_nominal=args.mpc_v_nominal,
        p_reverse=args.mpc_p_reverse,
        seed=args.seed + 7,
    )
    rng_scene = np.random.default_rng(args.seed)
    policy_metrics: List[Dict[str, Any]] = []
    mpc_metrics: List[Dict[str, Any]] = []
    for i in range(args.episodes):
        scene = sample_scene(cfg.scene, rng_scene)
        rng_policy = np.random.default_rng(args.seed + 1000 + i)
        rng_mpc = np.random.default_rng(args.seed + 2000 + i)
        pro = rollout_policy(policy_fn, scene, cfg, max_steps=args.max_steps, goal_tol=args.goal_tol, rng=rng_policy)
        mro = rollout_mpc(scene, cfg, mpc_cfg, max_steps=args.max_steps, goal_tol=args.goal_tol, rng=rng_mpc)
        policy_metrics.append(_metric_without_arrays(pro))
        mpc_metrics.append(_metric_without_arrays(mro))
        if i < args.plot_count:
            title = (
                f"ep={i} | pH success={pro['success']} final={pro['final_dist']:.2f} "
                f"clear={pro['min_clearance']:.2f} | MPC success={mro['success']} "
                f"final={mro['final_dist']:.2f} clear={mro['min_clearance']:.2f}"
            )
            save_comparison_plot(os.path.join(args.outdir, f"compare_ph_ep_{i:03d}.png"), scene, pro, mro, cfg, title)
    summary = {
        "policy": summarize(policy_metrics),
        "mpc": summarize(mpc_metrics),
        "sim_cfg": stage1_cfg_to_dict(cfg),
        "mpc_cfg": mpc_cfg_to_dict(mpc_cfg),
        "note": "pH policy predicts Hamiltonian/ICP coefficients; MPC remains a geometry-aware upper-bound baseline.",
    }
    with open(os.path.join(args.outdir, "summary_ph_vs_mpc.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
