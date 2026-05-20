#!/usr/bin/env python3
"""Closed-loop evaluation for the Stage-1 pH/Hamiltonian coefficient model."""
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
    rollout_policy,
    sample_scene,
    save_rollout_plot,
    stage1_cfg_to_dict,
)
from train_stage1_ph_energy import StagePHCoefNet, command_from_ph


def cfg_from_checkpoint(ckpt: Dict[str, Any]) -> Stage1Cfg:
    raw = ckpt.get("sim_cfg", {})
    if not raw:
        return Stage1Cfg()
    scene_raw = raw.get("scene", {})
    for key in ["world_x", "world_y", "n_obs_range", "radius_range", "start", "goal_x_range", "goal_y_range"]:
        if key in scene_raw and isinstance(scene_raw[key], list):
            scene_raw[key] = tuple(scene_raw[key])
    return Stage1Cfg(
        scene=SceneCfg(**scene_raw),
        scan=LaserScanCfg(**raw.get("scan", {})),
        adapter=AdapterCfg(**raw.get("adapter", {})),
        ackermann=AckermannCfg(**raw.get("ackermann", {})),
        recovery=RecoveryCfg(**raw.get("recovery", {})),
        seed=int(raw.get("seed", 2026)),
    )


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = cfg_from_checkpoint(ckpt)
    d_tok = int(ckpt.get("model_cfg", {}).get("d_tok", 96))
    model = StagePHCoefNet(d_tok=d_tok)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt


def make_ph_policy_fn(model: StagePHCoefNet, cfg: Stage1Cfg, device: str):
    def policy(obs_feats: np.ndarray, goal_feats: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs_feats, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones(1, obs.shape[1], dtype=torch.bool, device=device)
        goal = torch.tensor(goal_feats, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.enable_grad():
            coef = model(obs, mask, goal)
            cmd = command_from_ph(obs, mask, goal, coef, cfg.ackermann, cfg.scene.robot_radius)
        return cmd.detach().cpu().numpy()[0].astype(np.float32)
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
    ap = argparse.ArgumentParser("Evaluate Stage-1 pH Hamiltonian/ICP coefficient policy")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--goal-tol", type=float, default=0.35)
    ap.add_argument("--outdir", type=str, default="eval_stage1_ph_energy")
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--plot-count", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    os.makedirs(args.outdir, exist_ok=True)
    model, cfg, ckpt = load_model(args.ckpt, args.device)
    policy = make_ph_policy_fn(model, cfg, args.device)
    rng = np.random.default_rng(args.seed)
    metrics = []
    for i in range(args.episodes):
        scene = sample_scene(cfg.scene, rng)
        ro = rollout_policy(policy, scene, cfg, max_steps=args.max_steps, goal_tol=args.goal_tol, rng=rng)
        metrics.append({k: v for k, v in ro.items() if k not in ("traj", "cmds")})
        if i < args.plot_count:
            title = f"Stage-1 pH energy ep={i}, success={ro['success']}, min_clear={ro['min_clearance']:.3f}"
            save_rollout_plot(os.path.join(args.outdir, f"ph_policy_ep_{i:03d}.png"), scene, ro, title=title, cfg=cfg)
    summary = {"policy": summarize(metrics), "config": stage1_cfg_to_dict(cfg), "model_cfg": ckpt.get("model_cfg", {})}
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, "episodes_policy.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
