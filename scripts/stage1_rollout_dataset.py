"""Saved Stage-1 Ackermann rollout datasets.

The on-the-fly ``Stage1AckermannDataset`` is useful for cheap synthetic
snapshots.  Saved rollout datasets keep the same sample contract, but their
targets come from closed-loop Ackermann expert trajectories that can later be
replaced by ROS sim or rosbag recordings.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import Dataset

from scripts.stage1_ackermann_scan import (
    AckermannCfg,
    AdapterCfg,
    LaserScanCfg,
    RecoveryCfg,
    SceneCfg,
    Stage1Cfg,
)

ROLLOUT_DATASET_FORMAT = "stage1_ackermann_rollouts"
ROLLOUT_DATASET_VERSION = 1
_SEQUENCE_KEYS = ("cmd_seq", "pose_seq", "beta_seq", "true_clearance_seq")


def stage1_cfg_from_dict(raw: Mapping[str, Any]) -> Stage1Cfg:
    """Restore ``Stage1Cfg`` from ``stage1_cfg_to_dict`` output."""
    cfg = copy.deepcopy(dict(raw))
    scene_raw = dict(cfg.get("scene", {}))
    for key in ("world_x", "world_y", "n_obs_range", "radius_range", "start", "goal_x_range", "goal_y_range"):
        if key in scene_raw and isinstance(scene_raw[key], list):
            scene_raw[key] = tuple(scene_raw[key])
    return Stage1Cfg(
        scene=SceneCfg(**scene_raw),
        scan=LaserScanCfg(**dict(cfg.get("scan", {}))),
        adapter=AdapterCfg(**dict(cfg.get("adapter", {}))),
        ackermann=AckermannCfg(**dict(cfg.get("ackermann", {}))),
        recovery=RecoveryCfg(**dict(cfg.get("recovery", {}))),
        seed=int(cfg.get("seed", 2026)),
    )


def load_rollout_payload(path: str | Path) -> Dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if payload.get("format") != ROLLOUT_DATASET_FORMAT:
        raise ValueError(f"{path} is not a {ROLLOUT_DATASET_FORMAT!r} dataset")
    if int(payload.get("version", -1)) != ROLLOUT_DATASET_VERSION:
        raise ValueError(
            f"{path} has rollout dataset version {payload.get('version')!r}; "
            f"expected {ROLLOUT_DATASET_VERSION}"
        )
    if "samples" not in payload or "sim_cfg" not in payload:
        raise ValueError(f"{path} is missing rollout samples or Stage-1 configuration")
    return payload


def save_rollout_payload(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), target)


class SavedStage1RolloutDataset(Dataset):
    """Dataset wrapper for generator output compatible with ``collate_stage1``."""

    def __init__(self, path: str | Path, horizon: int | None = None):
        self.path = str(path)
        self.payload = load_rollout_payload(path)
        self.samples = list(self.payload["samples"])
        if not self.samples:
            raise ValueError(f"{path} contains no rollout samples")
        self.sim_cfg = stage1_cfg_from_dict(self.payload["sim_cfg"])
        self.horizon = int(horizon) if horizon is not None else None
        if self.horizon is not None:
            shortest = min(int(sample["pose_seq"].shape[0]) for sample in self.samples)
            if self.horizon > shortest:
                raise ValueError(
                    f"requested horizon {self.horizon} exceeds saved rollout horizon {shortest} in {path}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.samples[int(idx)])
        if self.horizon is not None:
            for key in _SEQUENCE_KEYS:
                if key in sample:
                    sample[key] = sample[key][: self.horizon]
        return sample

    def sampling_weights(self, hard_clearance: float, hard_gain: float) -> torch.Tensor:
        """Weight low-clearance MPC windows for training-time resampling."""
        threshold = max(float(hard_clearance), 1e-6)
        gain = max(float(hard_gain), 0.0)
        clearances = torch.tensor(
            [float(sample.get("true_clearance", threshold)) for sample in self.samples],
            dtype=torch.float32,
        )
        hard = torch.relu((threshold - clearances) / threshold)
        return 1.0 + gain * hard
