#!/usr/bin/env python3
"""
Closed-loop evaluation for the Stage-1 Ackermann/LaserScan policy.

Usage
-----
python -m eval_stage1_ackermann \
  --ckpt checkpoints/stage1_ackermann/best.pt \
  --episodes 50 --outdir eval_stage1

This uses the same synthetic LaserScan renderer and scan adapter as training,
but evaluates closed-loop motion with Ackermann projection, acceleration limits,
steering-rate limits, and collision/safety checks.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch

from scripts.stage1_ackermann_scan import (
    AckermannCfg,
    AdapterCfg,
    LaserScanCfg,
    SceneCfg,
    RecoveryCfg,
    Stage1Cfg,
    expert_from_scan_features,
    project_twist_np,
    rollout_policy,
    sample_scene,
    save_rollout_plot,
    stage1_cfg_to_dict,
)
from train_stage1_ackermann import Stage1PolicyNet


def _tuple2(x):
    return (float(x[0]), float(x[1]))


def cfg_from_checkpoint(ckpt: Dict[str, Any]) -> Stage1Cfg:
    raw = ckpt.get("sim_cfg", {})
    if not raw:
        return Stage1Cfg()
    scene_raw = raw.get("scene", {})
    scan_raw = raw.get("scan", {})
    adapter_raw = raw.get("adapter", {})
    ack_raw = raw.get("ackermann", {})
    recovery_raw = raw.get("recovery", {})

    # JSON stores tuples as lists; dataclasses expect tuples for range fields.
    for key in ["world_x", "world_y", "n_obs_range", "radius_range", "start", "goal_x_range", "goal_y_range"]:
        if key in scene_raw and isinstance(scene_raw[key], list):
            scene_raw[key] = tuple(scene_raw[key])
    return Stage1Cfg(
        scene=SceneCfg(**scene_raw),
        scan=LaserScanCfg(**scan_raw),
        adapter=AdapterCfg(**adapter_raw),
        ackermann=AckermannCfg(**ack_raw),
        recovery=RecoveryCfg(**recovery_raw),
        seed=int(raw.get("seed", 2026)),
    )


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = cfg_from_checkpoint(ckpt)
    model_cfg = ckpt.get("model_cfg", {})
    model = Stage1PolicyNet(
        horizon=int(model_cfg.get("horizon", getattr(cfg.recovery, "train_horizon", 12))),
        n_modes=int(model_cfg.get("n_modes", 6)),
        d_tok=int(model_cfg.get("d_tok", 96)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt


def make_policy_fn(model: Stage1PolicyNet, device: str):
    @torch.no_grad()
    def policy(obs_feats: np.ndarray, goal_feats: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs_feats, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones(1, obs.shape[1], dtype=torch.bool, device=device)
        goal = torch.tensor(goal_feats, dtype=torch.float32, device=device).unsqueeze(0)
        raw = model(obs, mask, goal).squeeze(0).detach().cpu().numpy()
        return raw.astype(np.float32)
    return policy


def make_expert_policy_fn(cfg: Stage1Cfg):
    def policy(obs_feats: np.ndarray, goal_feats: np.ndarray) -> np.ndarray:
        return expert_from_scan_features(obs_feats, goal_feats, cfg.ackermann, cfg.scene.robot_radius)
    return policy


def summarize(metrics):
    out = {}
    for k in ["success", "collision", "stopped_by_safety"]:
        out[k + "_rate"] = float(np.mean([1.0 if m[k] else 0.0 for m in metrics]))
    for k in ["min_clearance", "path_length", "final_dist", "steps", "stall_steps", "reverse_steps"]:
        vals = np.asarray([m[k] for m in metrics], dtype=np.float32)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


def main():
    ap = argparse.ArgumentParser("Evaluate Stage-1 Ackermann/LaserScan policy")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--goal-tol", type=float, default=0.35)
    ap.add_argument("--outdir", type=str, default="eval_stage1_ackermann")
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; keep small for local evaluation")
    ap.add_argument("--plot-count", type=int, default=8)
    ap.add_argument("--baseline-expert", action="store_true", help="also evaluate the scan-only expert baseline")
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    os.makedirs(args.outdir, exist_ok=True)
    model, cfg, ckpt = load_model(args.ckpt, args.device)
    model_policy = make_policy_fn(model, args.device)
    expert_policy = make_expert_policy_fn(cfg)

    rng = np.random.default_rng(args.seed)
    model_metrics = []
    expert_metrics = []

    for i in range(args.episodes):
        scene = sample_scene(cfg.scene, rng)
        ro = rollout_policy(model_policy, scene, cfg, max_steps=args.max_steps, goal_tol=args.goal_tol, rng=rng)
        model_metrics.append({k: v for k, v in ro.items() if k not in ("traj", "cmds")})
        if i < args.plot_count:
            title = f"Stage-1 policy ep={i}, success={ro['success']}, min_clear={ro['min_clearance']:.3f}"
            save_rollout_plot(os.path.join(args.outdir, f"policy_ep_{i:03d}.png"), scene, ro, title=title, cfg=cfg)

        if args.baseline_expert:
            er = rollout_policy(expert_policy, scene, cfg, max_steps=args.max_steps, goal_tol=args.goal_tol, rng=rng)
            expert_metrics.append({k: v for k, v in er.items() if k not in ("traj", "cmds")})
            if i < min(args.plot_count, 4):
                title = f"Scan expert ep={i}, success={er['success']}, min_clear={er['min_clearance']:.3f}"
                save_rollout_plot(os.path.join(args.outdir, f"expert_ep_{i:03d}.png"), scene, er, title=title, cfg=cfg)

    summary = {"policy": summarize(model_metrics), "config": stage1_cfg_to_dict(cfg)}
    if expert_metrics:
        summary["expert"] = summarize(expert_metrics)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, "episodes_policy.json"), "w") as f:
        json.dump(model_metrics, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
