#!/usr/bin/env python3
"""Generate reproducible Stage-1 samples from closed-loop Ackermann MPC rollouts.

This script is designed for two uses:

1. Offline policy training: save fixed train/val/test tensors generated from the
   same scenario distribution and MPC teacher.
2. Reproducible evaluation: save exact scene geometry and per-episode seeds so a
   learned policy can be rolled out later on the identical scenarios.

The original generic ``sample_scene`` path is preserved as ``--case random``.
Named cases such as ``parallel_parking`` and ``terminal_bay`` provide controlled
stress tests for energy switching, reverse motion, terminal clearance, and
multi-round maneuvers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from scripts.stage1_ackermann_scan import (
    AckermannCfg,
    AdapterCfg,
    LaserScanCfg,
    RecoveryCfg,
    SceneCfg,
    Stage1Cfg,
    ackermann_step_np,
    classify_maneuver_mode,
    local_goal_features,
    min_clearance_to_obstacles,
    render_laserscan,
    sample_scene,
    scan_feature_clearances,
    scan_to_obstacle_features,
    stage1_cfg_to_dict,
)
from scripts.stage1_mpc import MPCConfig, mpc_action, mpc_cfg_to_dict
from scripts.stage1_rollout_dataset import (
    ROLLOUT_DATASET_FORMAT,
    ROLLOUT_DATASET_VERSION,
    save_rollout_payload,
)

CASE_TO_ID = {
    "random": 0,
    "easy_open": 1,
    "dense_clutter": 2,
    "narrow_gate": 3,
    "s_turn": 4,
    "front_trap": 5,
    "parallel_parking": 6,
    "terminal_bay": 7,
    "noisy_scan": 8,
    "long_horizon_mixed": 9,
}


def build_cfg(args: argparse.Namespace) -> Stage1Cfg:
    return Stage1Cfg(
        scene=SceneCfg(
            robot_radius=args.robot_radius,
            n_obs_range=(args.n_obs_min, args.n_obs_max),
            corridor_half_width=args.corridor_half_width,
        ),
        scan=LaserScanCfg(
            n_beams=args.n_beams,
            fov=args.fov_deg * math.pi / 180.0,
            range_max=args.range_max,
            noise_std=args.scan_noise,
            dropout_prob=args.scan_dropout,
        ),
        adapter=AdapterCfg(max_tokens=args.max_tokens),
        ackermann=AckermannCfg(
            wheelbase=args.wheelbase,
            v_max=args.v_max,
            v_reverse_max=args.v_reverse_max,
            max_steering_angle=args.max_steering_angle,
            min_turning_radius=args.min_turning_radius,
            dt=args.dt,
        ),
        recovery=RecoveryCfg(
            stall_steps_trigger=args.stall_steps_trigger,
            train_horizon=args.horizon,
        ),
        seed=args.seed,
    )


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def beta_target(min_clearance: float, front_clearance: float) -> float:
    return float(np.clip((0.90 - min(front_clearance, min_clearance)) / 0.90, 0.0, 1.0))


def _scene(start, goal, centers, radii, *, start_heading: Optional[float] = None,
           goal_heading: Optional[float] = None, case: str = "random", scene_seed: int = 0) -> Dict[str, np.ndarray]:
    C = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    R = np.asarray(radii, dtype=np.float32).reshape(-1)
    out: Dict[str, Any] = {
        "start": np.asarray(start, dtype=np.float32),
        "goal": np.asarray(goal, dtype=np.float32),
        "C": C,
        "R": R,
        "case": str(case),
        "scene_seed": int(scene_seed),
    }
    if start_heading is not None:
        out["start_heading"] = float(start_heading)
    if goal_heading is not None:
        out["goal_heading"] = float(goal_heading)
    return out


def scene_to_record(scene: Dict[str, Any], *, episode_id: int, split: str, case: str, scene_seed: int) -> Dict[str, Any]:
    return {
        "episode_id": int(episode_id),
        "split": str(split),
        "case": str(case),
        "case_id": int(CASE_TO_ID.get(case, -1)),
        "scene_seed": int(scene_seed),
        "start": np.asarray(scene["start"], dtype=float).tolist(),
        "goal": np.asarray(scene["goal"], dtype=float).tolist(),
        "C": np.asarray(scene["C"], dtype=float).tolist(),
        "R": np.asarray(scene["R"], dtype=float).tolist(),
        "start_heading": None if "start_heading" not in scene else float(scene["start_heading"]),
        "goal_heading": None if "goal_heading" not in scene else float(scene["goal_heading"]),
    }


def scene_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return _scene(
        record["start"],
        record["goal"],
        record.get("C", []),
        record.get("R", []),
        start_heading=record.get("start_heading", None),
        goal_heading=record.get("goal_heading", None),
        case=record.get("case", "random"),
        scene_seed=int(record.get("scene_seed", 0)),
    )


def _random_disks(
    rng: np.random.Generator,
    n: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    rlim: Tuple[float, float],
    forbidden_points: Iterable[np.ndarray],
    forbidden_radius: float,
    corridor_y_abs: Optional[float] = None,
    max_tries: int = 4000,
) -> Tuple[List[List[float]], List[float]]:
    centers: List[List[float]] = []
    radii: List[float] = []
    fp = [np.asarray(p, dtype=float) for p in forbidden_points]
    tries = 0
    while len(centers) < int(n) and tries < max_tries:
        tries += 1
        c = np.array([rng.uniform(*xlim), rng.uniform(*ylim)], dtype=float)
        r = float(rng.uniform(*rlim))
        if corridor_y_abs is not None and abs(c[1]) < corridor_y_abs + r:
            continue
        if any(np.linalg.norm(c - p) < (forbidden_radius + r) for p in fp):
            continue
        if centers:
            C = np.asarray(centers)
            R = np.asarray(radii)
            if np.any(np.linalg.norm(C - c[None, :], axis=1) < (R + r + 0.08)):
                continue
        centers.append(c.tolist())
        radii.append(r)
    return centers, radii


def sample_case_scene(case: str, scene_cfg: SceneCfg, rng: np.random.Generator, *, scene_seed: int = 0) -> Dict[str, Any]:
    """Sample a named scenario with the same schema used by ``sample_scene``.

    The geometry is deliberately simple and deterministic up to ``scene_seed``.
    That makes the generated train/val/test split reproducible and replayable.
    """
    case = str(case)
    rr = float(scene_cfg.robot_radius)

    if case in ("random", "noisy_scan"):
        scene = sample_scene(scene_cfg, rng)
        scene["case"] = case
        scene["scene_seed"] = int(scene_seed)
        return scene

    if case == "easy_open":
        start, goal = np.array([-2.5, 0.0]), np.array([3.5, 0.0])
        C, R = _random_disks(rng, 4, (-2.5, 3.5), (-1.8, 1.8), (0.18, 0.32), [start, goal], 0.9, corridor_y_abs=0.75)
        return _scene(start, goal, C, R, case=case, scene_seed=scene_seed)

    if case == "dense_clutter":
        start, goal = np.array([-3.0, -0.7]), np.array([4.0, 0.6])
        C, R = _random_disks(rng, int(rng.integers(18, 31)), (-3.2, 4.2), (-2.0, 2.0), (0.16, 0.38), [start, goal], 0.8)
        return _scene(start, goal, C, R, case=case, scene_seed=scene_seed)

    if case == "narrow_gate":
        start, goal = np.array([-3.0, 0.0]), np.array([3.2, 0.0])
        gate_half = rr + rng.uniform(0.08, 0.16)
        r_gate = rng.uniform(0.34, 0.48)
        C = [[-0.15, gate_half + r_gate], [0.15, -gate_half - r_gate]]
        R = [r_gate, r_gate]
        extra_C, extra_R = _random_disks(rng, 8, (-3.0, 3.2), (-1.8, 1.8), (0.14, 0.30), [start, goal, np.zeros(2)], 0.8, corridor_y_abs=0.35)
        return _scene(start, goal, C + extra_C, R + extra_R, case=case, scene_seed=scene_seed)

    if case == "s_turn":
        start, goal = np.array([-4.0, -0.85]), np.array([4.0, 0.85])
        C = [[-2.2, 0.05], [-0.4, -0.65], [1.5, 0.55], [2.8, -0.55]]
        R = [0.55, 0.50, 0.55, 0.42]
        extra_C, extra_R = _random_disks(rng, 6, (-4.0, 4.0), (-2.0, 2.0), (0.12, 0.26), [start, goal], 0.7)
        return _scene(start, goal, C + extra_C, R + extra_R, case=case, scene_seed=scene_seed)

    if case == "front_trap":
        # A near-field obstacle blocks the straight attractive path.  Reverse or
        # strong tangent/flow switching is useful before committing forward.
        start, goal = np.array([-1.2, 0.0]), np.array([3.0, 0.0])
        C = [[0.20, 0.0], [0.85, 0.55], [0.85, -0.55], [1.75, 0.0]]
        R = [0.36, 0.30, 0.30, 0.26]
        extra_C, extra_R = _random_disks(rng, 4, (-1.5, 3.2), (-1.5, 1.5), (0.12, 0.24), [start, goal], 0.7)
        return _scene(start, goal, C + extra_C, R + extra_R, start_heading=0.0, case=case, scene_seed=scene_seed)

    if case == "parallel_parking":
        # Road direction is +x.  The car starts ahead of the slot and faces +x,
        # while the slot center lies behind/left in the local frame, forcing a
        # reverse-entry maneuver.  Obstacles approximate front/rear parked cars
        # and curb posts.
        start = np.array([2.6 + rng.uniform(-0.2, 0.2), -0.85 + rng.uniform(-0.08, 0.08)])
        goal = np.array([0.0, 0.0])
        C = [
            [1.25, 0.05], [1.65, 0.05],       # front parked vehicle
            [-1.25, 0.05], [-1.65, 0.05],     # rear parked vehicle
            [-0.3, 0.72], [0.3, 0.72],        # curb/upper boundary cues
            [2.2, 0.55],                      # front bumper constraint
        ]
        R = [0.30, 0.30, 0.30, 0.30, 0.18, 0.18, 0.22]
        return _scene(start, goal, C, R, start_heading=0.0, goal_heading=0.0, case=case, scene_seed=scene_seed)

    if case == "terminal_bay":
        start, goal = np.array([-3.0, -0.15]), np.array([2.4, 0.0])
        C = [[2.25, 0.62], [2.25, -0.62], [2.95, 0.0], [1.55, 0.72], [1.55, -0.72]]
        R = [0.28, 0.28, 0.32, 0.22, 0.22]
        extra_C, extra_R = _random_disks(rng, 6, (-3.0, 1.5), (-1.7, 1.7), (0.12, 0.28), [start, goal], 0.8)
        return _scene(start, goal, C + extra_C, R + extra_R, case=case, scene_seed=scene_seed)

    if case == "long_horizon_mixed":
        start, goal = np.array([-4.5, -0.8]), np.array([5.0, 0.3])
        C = [[-2.7, 0.1], [-1.4, -0.7], [0.2, 0.55], [1.4, -0.55], [3.4, 0.58], [3.4, -0.58], [4.2, 0.0]]
        R = [0.45, 0.38, 0.42, 0.42, 0.32, 0.32, 0.28]
        extra_C, extra_R = _random_disks(rng, 8, (-4.5, 5.0), (-2.0, 2.0), (0.12, 0.26), [start, goal], 0.8)
        return _scene(start, goal, C + extra_C, R + extra_R, case=case, scene_seed=scene_seed)

    raise ValueError(f"Unknown case {case!r}. Available cases: {sorted(CASE_TO_ID)}")


def rollout_episode(
    scene: Dict[str, np.ndarray],
    cfg: Stage1Cfg,
    mpc_cfg: MPCConfig,
    max_steps: int,
    goal_tol: float,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one MPC episode and keep pre-action policy observations."""
    start = np.asarray(scene["start"], dtype=np.float32)
    goal = np.asarray(scene["goal"], dtype=np.float32)
    d = goal - start
    start_heading = float(scene.get("start_heading", math.atan2(float(d[1]), float(d[0]))))
    pose = np.array([start[0], start[1], start_heading], dtype=np.float32)
    current_speed = 0.0
    current_delta = 0.0
    stall_count = 0
    prev_dist_goal = float(np.linalg.norm(goal - pose[:2]))

    frames: List[Dict[str, Any]] = []
    traj = [pose.copy()]
    commands: List[List[float]] = []
    min_clearance = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
    collision = min_clearance < 0.0
    stopped_by_safety = False
    reverse_steps = 0
    stall_steps = 0
    mode_switches = 0
    prev_mode: Optional[int] = None

    for _ in range(int(max_steps)):
        dist_goal = float(np.linalg.norm(goal - pose[:2]))
        true_clearance = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
        if dist_goal <= goal_tol:
            break
        if true_clearance < cfg.recovery.imminent_stop_clearance:
            stopped_by_safety = True
            break

        ranges = render_laserscan(pose, scene["C"], scene["R"], cfg.scan, rng)
        obs_feats = scan_to_obstacle_features(ranges, cfg.scan, cfg.adapter)
        min_clear_feat, front_clear_feat = scan_feature_clearances(obs_feats, cfg.scene.robot_radius)
        goal_feats = local_goal_features(
            goal,
            pose,
            current_speed=current_speed,
            min_clearance=min_clear_feat,
            front_clearance=front_clear_feat,
            stall_count=stall_count,
            stall_trigger=cfg.recovery.stall_steps_trigger,
        )
        v, omega, delta, info = mpc_action(pose, scene, current_speed, current_delta, cfg, mpc_cfg, rng)
        mode = classify_maneuver_mode(float(v), float(delta), cfg.ackermann)
        if prev_mode is not None and int(mode) != int(prev_mode):
            mode_switches += 1
        prev_mode = int(mode)
        frames.append(
            {
                "pose": pose.copy(),
                "ranges": ranges.copy(),
                "obs_feats": obs_feats.copy(),
                "goal_feats": goal_feats.copy(),
                "beta": beta_target(min_clear_feat, front_clear_feat),
                "clearance": float(true_clearance),
                "stall_count": int(stall_count),
                "mpc_cost": float(info["best_cost"]),
                "mode": int(mode),
            }
        )

        pose = ackermann_step_np(pose, v, omega, cfg.ackermann.dt)
        current_speed = v
        current_delta = delta
        traj.append(pose.copy())
        commands.append([v, omega, delta])
        if v < -cfg.ackermann.v_min:
            reverse_steps += 1

        clearance = min_clearance_to_obstacles(pose[:2], scene["C"], scene["R"], cfg.scene.robot_radius)
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            collision = True
            break
        new_dist_goal = float(np.linalg.norm(goal - pose[:2]))
        progress = prev_dist_goal - new_dist_goal
        stalled_now = (
            abs(v) < cfg.recovery.stall_speed
            and progress < cfg.recovery.stall_progress
            and clearance < cfg.recovery.stall_clearance
        )
        if stalled_now:
            stall_count += 1
            stall_steps += 1
        else:
            stall_count = max(0, stall_count - 1)
        prev_dist_goal = new_dist_goal

    traj_arr = np.asarray(traj, dtype=np.float32)
    cmd_arr = np.asarray(commands, dtype=np.float32) if commands else np.zeros((0, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        frame["pose_after"] = traj_arr[i + 1].copy()
        frame["cmd"] = cmd_arr[i].copy()

    final_dist = float(np.linalg.norm(goal - traj_arr[-1, :2]))
    final_heading_err = None
    if "goal_heading" in scene:
        final_heading_err = abs(wrap_angle(float(traj_arr[-1, 2]) - float(scene["goal_heading"])))
    metrics = {
        "success": bool(final_dist <= goal_tol and not collision and not stopped_by_safety),
        "collision": bool(collision),
        "stopped_by_safety": bool(stopped_by_safety),
        "steps": int(len(cmd_arr)),
        "min_clearance": float(min_clearance),
        "final_dist": final_dist,
        "final_heading_err": None if final_heading_err is None else float(final_heading_err),
        "reverse_steps": int(reverse_steps),
        "reverse_episode": bool(reverse_steps > 0),
        "stall_steps": int(stall_steps),
        "mode_switches": int(mode_switches),
    }
    return frames, {"traj": traj_arr, "cmds": cmd_arr, "metrics": metrics}


def window_samples(
    frames: List[Dict[str, Any]],
    episode_id: int,
    horizon: int,
    stride: int,
    cfg: Stage1Cfg,
    *,
    case: str = "random",
    scene_seed: int = 0,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    usable = len(frames) - int(horizon) + 1
    for i in range(0, max(0, usable), max(1, int(stride))):
        win = frames[i : i + horizon]
        first = win[0]
        cmd_seq = np.asarray([frame["cmd"][:2] for frame in win], dtype=np.float32)
        pose_seq = np.asarray([frame["pose_after"] for frame in win], dtype=np.float32)
        beta_seq = np.asarray([frame["beta"] for frame in win], dtype=np.float32)
        first_delta = float(first["cmd"][2])
        samples.append(
            {
                "obs_feats": torch.tensor(first["obs_feats"], dtype=torch.float32),
                "goal_feats": torch.tensor(first["goal_feats"], dtype=torch.float32),
                "cmd": torch.tensor(cmd_seq[0], dtype=torch.float32),
                "cmd_seq": torch.tensor(cmd_seq, dtype=torch.float32),
                "pose": torch.tensor(first["pose"], dtype=torch.float32),
                "next_pose": torch.tensor(pose_seq[0], dtype=torch.float32),
                "pose_seq": torch.tensor(pose_seq, dtype=torch.float32),
                "beta_seq": torch.tensor(beta_seq, dtype=torch.float32),
                "mode": torch.tensor(
                    classify_maneuver_mode(float(cmd_seq[0, 0]), first_delta, cfg.ackermann),
                    dtype=torch.long,
                ),
                "ranges": torch.tensor(first["ranges"], dtype=torch.float32),
                "recovery_sample": torch.tensor(int(float(first["goal_feats"][7]) > 0.5), dtype=torch.long),
                "episode_id": int(episode_id),
                "step": int(i),
                "case_id": torch.tensor(CASE_TO_ID.get(case, -1), dtype=torch.long),
                "scene_seed": int(scene_seed),
                "true_clearance": float(first["clearance"]),
                "mpc_cost": float(first["mpc_cost"]),
            }
        )
    return samples


def _load_scene_bank(path: Optional[str], split: str) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and split in obj:
        return list(obj[split])
    if isinstance(obj, dict) and "scenes" in obj:
        return list(obj["scenes"])
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Could not interpret scene bank {path!r}")


def generate_split(
    name: str,
    episodes: int,
    seed: int,
    cfg: Stage1Cfg,
    mpc_cfg: MPCConfig,
    args: argparse.Namespace,
    *,
    scene_bank: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    samples: List[Dict[str, Any]] = []
    episode_metrics: List[Dict[str, Any]] = []
    scene_records: List[Dict[str, Any]] = []
    total = len(scene_bank) if scene_bank is not None else int(episodes)
    progress = tqdm(range(total), desc=f"generate {name}/{args.case}", dynamic_ncols=True)
    for episode_id in progress:
        scene_seed = int(seed + 9973 * episode_id)
        if scene_bank is not None:
            rec = deepcopy(scene_bank[episode_id])
            scene = scene_from_record(rec)
            case = str(rec.get("case", args.case))
            scene_seed = int(rec.get("scene_seed", scene_seed))
        else:
            rng_scene = np.random.default_rng(scene_seed)
            case = str(args.case)
            scene = sample_case_scene(case, cfg.scene, rng_scene, scene_seed=scene_seed)
            rec = scene_to_record(scene, episode_id=episode_id, split=name, case=case, scene_seed=scene_seed)
        rollout_rng = np.random.default_rng(seed + 10000 + episode_id)
        frames, rollout = rollout_episode(
            scene,
            cfg,
            mpc_cfg,
            max_steps=args.max_steps,
            goal_tol=args.goal_tol,
            rng=rollout_rng,
        )
        split_samples = window_samples(
            frames,
            episode_id,
            args.horizon,
            args.sample_stride,
            cfg,
            case=case,
            scene_seed=scene_seed,
        )
        metric = dict(rollout["metrics"])
        metric.update(
            {
                "episode_id": int(episode_id),
                "case": case,
                "case_id": int(CASE_TO_ID.get(case, -1)),
                "scene_seed": int(scene_seed),
                "samples": int(len(split_samples)),
            }
        )
        if args.embed_scene_in_summary:
            metric["scene"] = rec
        episode_metrics.append(metric)
        scene_records.append(rec)
        samples.extend(split_samples)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(samples=len(samples), success=sum(1 for m in episode_metrics if m["success"]))
    print(
        json.dumps(
            {
                "split": name,
                "case": args.case,
                "episodes": int(total),
                "samples": int(len(samples)),
                "successes": int(sum(1 for metric in episode_metrics if metric["success"])),
            }
        )
    )
    return samples, episode_metrics, scene_records


def payload(
    split: str,
    samples: List[Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    cfg: Stage1Cfg,
    mpc_cfg: MPCConfig,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "format": ROLLOUT_DATASET_FORMAT,
        "version": ROLLOUT_DATASET_VERSION,
        "split": split,
        "case": str(args.case),
        "samples": samples,
        "episodes": episodes,
        "scenes": scenes,
        "case_to_id": CASE_TO_ID,
        "sim_cfg": stage1_cfg_to_dict(cfg),
        "mpc_cfg": mpc_cfg_to_dict(mpc_cfg),
        "generation": {
            "horizon": int(args.horizon),
            "sample_stride": int(args.sample_stride),
            "max_steps": int(args.max_steps),
            "goal_tol": float(args.goal_tol),
            "seed": int(args.seed),
            "case": str(args.case),
            "note": "Targets come from geometry-aware closed-loop Ackermann MPC rollouts. Exact scene geometry is saved for reproducible offline training and later policy evaluation.",
        },
    }


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _summarize_split(name: str, episodes: List[Dict[str, Any]], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(episodes))
    def mean(key: str) -> float:
        vals = [e[key] for e in episodes if e.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "episodes": int(len(episodes)),
        "samples": int(len(samples)),
        "success_rate": float(sum(bool(e.get("success", False)) for e in episodes) / n),
        "collision_rate": float(sum(bool(e.get("collision", False)) for e in episodes) / n),
        "safety_stop_rate": float(sum(bool(e.get("stopped_by_safety", False)) for e in episodes) / n),
        "reverse_episode_rate": float(sum(bool(e.get("reverse_episode", False)) for e in episodes) / n),
        "mean_reverse_steps": mean("reverse_steps"),
        "mean_stall_steps": mean("stall_steps"),
        "mean_mode_switches": mean("mode_switches"),
        "mean_min_clearance": mean("min_clearance"),
        "p05_min_clearance": float(np.percentile([e["min_clearance"] for e in episodes], 5)) if episodes else float("nan"),
        "mean_final_dist": mean("final_dist"),
        "mean_final_heading_err": mean("final_heading_err"),
    }


def main() -> None:
    ap = argparse.ArgumentParser("Generate reproducible Stage-1 Ackermann rollout datasets from MPC")
    ap.add_argument("--root", type=str, default="data/stage1_ackermann_mpc_rollouts")
    ap.add_argument("--case", type=str, default="random", choices=sorted(CASE_TO_ID.keys()))
    ap.add_argument("--train-episodes", type=int, default=200)
    ap.add_argument("--val-episodes", type=int, default=40)
    ap.add_argument("--test-episodes", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--goal-tol", type=float, default=0.35)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--sample-stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--embed-scene-in-summary", action="store_true", help="Also embed full scene geometry inside each episode metric entry. The separate scenes_*.json files are always written.")
    ap.add_argument("--scene-bank-train", type=str, default=None, help="Optional JSON scene bank to replay for train split.")
    ap.add_argument("--scene-bank-val", type=str, default=None, help="Optional JSON scene bank to replay for val split.")
    ap.add_argument("--scene-bank-test", type=str, default=None, help="Optional JSON scene bank to replay for test split.")

    ap.add_argument("--wheelbase", type=float, default=0.324)
    ap.add_argument("--v-max", type=float, default=0.80)
    ap.add_argument("--v-reverse-max", type=float, default=0.25)
    ap.add_argument("--max-steering-angle", type=float, default=0.396)
    ap.add_argument("--min-turning-radius", type=float, default=0.80)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--robot-radius", type=float, default=0.23)
    ap.add_argument("--n-beams", type=int, default=181)
    ap.add_argument("--fov-deg", type=float, default=270.0)
    ap.add_argument("--range-max", type=float, default=6.0)
    ap.add_argument("--scan-noise", type=float, default=0.015)
    ap.add_argument("--scan-dropout", type=float, default=0.01)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--n-obs-min", type=int, default=8)
    ap.add_argument("--n-obs-max", type=int, default=20)
    ap.add_argument("--corridor-half-width", type=float, default=0.70)
    ap.add_argument("--stall-steps-trigger", type=int, default=8)

    ap.add_argument("--mpc-horizon", type=int, default=12)
    ap.add_argument("--mpc-samples", type=int, default=512)
    ap.add_argument("--mpc-safe-margin", type=float, default=0.20)
    ap.add_argument("--mpc-v-nominal", type=float, default=0.55)
    ap.add_argument("--mpc-p-reverse", type=float, default=0.25)
    args = ap.parse_args()

    cfg = build_cfg(args)
    mpc_cfg = MPCConfig(
        horizon=args.mpc_horizon,
        n_samples=args.mpc_samples,
        safe_margin=args.mpc_safe_margin,
        v_nominal=args.mpc_v_nominal,
        p_reverse=args.mpc_p_reverse,
        seed=args.seed + 7,
    )

    split_specs = [
        ("train", args.train_episodes, args.seed, _load_scene_bank(args.scene_bank_train, "train")),
        ("val", args.val_episodes, args.seed + 100000, _load_scene_bank(args.scene_bank_val, "val")),
        ("test", args.test_episodes, args.seed + 200000, _load_scene_bank(args.scene_bank_test, "test")),
    ]
    all_summary: Dict[str, Any] = {
        "case": str(args.case),
        "case_id": int(CASE_TO_ID.get(args.case, -1)),
        "case_to_id": CASE_TO_ID,
        "sim_cfg": stage1_cfg_to_dict(cfg),
        "mpc_cfg": mpc_cfg_to_dict(mpc_cfg),
        "generation_args": vars(args),
        "splits": {},
    }

    for split_name, n_ep, split_seed, bank in split_specs:
        if int(n_ep) <= 0 and bank is None:
            continue
        samples, episodes, scenes = generate_split(split_name, n_ep, split_seed, cfg, mpc_cfg, args, scene_bank=bank)
        if not samples:
            raise RuntimeError(f"{split_name} generation produced no samples")
        save_rollout_payload(f"{args.root}/{split_name}.pt", payload(split_name, samples, episodes, scenes, cfg, mpc_cfg, args))
        _save_json(f"{args.root}/scenes_{split_name}.json", {"split": split_name, "case": args.case, "scenes": scenes})
        _save_json(f"{args.root}/episodes_{split_name}.json", {"split": split_name, "case": args.case, "episodes": episodes})
        all_summary[f"{split_name}_samples"] = len(samples)
        all_summary[f"{split_name}_episodes"] = episodes
        all_summary["splits"][split_name] = _summarize_split(split_name, episodes, samples)

    _save_json(f"{args.root}/summary.json", all_summary)


if __name__ == "__main__":
    main()
