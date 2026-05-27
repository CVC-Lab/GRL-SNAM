"""Adapters from reproducible Stage-1 Ackermann rollout datasets to local energy patches.

The Stage-1 generator saves MPC-teacher rollout tensors (train/val/test.pt) and
exact scene banks (scenes_train/val/test.json).  The dual-weight energy trainer,
however, expects NavMaximin-style local energy patches.  This module bridges the
schemas by reconstructing a local obstacle/goal grid around each saved pose.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # normal package import
    from scripts.energy_data_navmax import collate_navmax as _base_collate_navmax
except Exception:  # pragma: no cover
    from energy_data_navmax import collate_navmax as _base_collate_navmax


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


def _as_np(x: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _as_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(float(theta)), math.sin(float(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def world_to_local(P_w: np.ndarray, pose: np.ndarray) -> np.ndarray:
    P_w = np.asarray(P_w, dtype=np.float32)
    if P_w.ndim == 1:
        P_w = P_w[None, :]
    R = rot2(float(pose[2]))
    return (P_w - pose[None, :2]) @ R


def make_coord_grid(H: int, W: int, half: float, dtype=torch.float32) -> torch.Tensor:
    ys = torch.linspace(-half, half, int(H), dtype=dtype)
    xs = torch.linspace(-half, half, int(W), dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([X, Y], dim=-1)


def barrier_map_from_sd(sd: torch.Tensor, d_hat: float, kind: str = "icp") -> torch.Tensor:
    # Same bounded local barrier convention used by the energy-patch trainer.
    d_pos = torch.clamp(sd, min=0.0)
    outside = torch.clamp(float(d_hat) - d_pos, min=0.0)
    inside = torch.clamp(-sd, min=0.0)
    if kind == "exp":
        return torch.exp(torch.clamp(float(d_hat) - sd, min=0.0)) - 1.0
    return outside.pow(2) + 5.0 * inside.pow(2)


def quadratic_goal(grid_xy: torch.Tensor, goal_l: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((grid_xy - goal_l.view(1, 1, 2)) ** 2).sum(dim=-1)


def _load_payload(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def _load_scenes(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "scenes" in obj:
        scenes = obj["scenes"]
    elif isinstance(obj, list):
        scenes = obj
    else:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for i, rec in enumerate(scenes):
        eid = int(rec.get("episode_id", i))
        out[eid] = rec
    return out


def _find_case_dirs(root: Path, cases: Optional[Sequence[str]]) -> List[Path]:
    if (root / "train.pt").exists() or root.suffix == ".pt":
        return [root]
    if cases:
        return [root / c for c in cases]
    manifest = root / "manifest.json"
    if manifest.exists():
        with open(manifest, "r") as f:
            obj = json.load(f)
        dirs = []
        for item in obj.get("cases", []):
            p = Path(item.get("root", root / item.get("case", "")))
            if not p.is_absolute():
                p = root / p
            dirs.append(p)
        if dirs:
            return dirs
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "train.pt").exists()])


def _scene_from_sample_fallback(sample: Dict[str, Any]) -> Dict[str, Any]:
    # If exact scene JSON is missing, build a local pseudo-scene from scan tokens.
    # This is less reproducible than scenes_*.json, but keeps old payloads loadable.
    obs = _as_np(sample.get("obs_feats", np.zeros((0, 6), np.float32)))
    if obs.ndim != 2 or obs.shape[1] < 3:
        obs = np.zeros((0, 6), np.float32)
    C = obs[:, :2].astype(np.float32, copy=False)
    R = np.clip(obs[:, 2], 0.05, 1.5).astype(np.float32, copy=False)
    pose = _as_np(sample.get("pose", [0.0, 0.0, 0.0]))
    goal_feats = _as_np(sample.get("goal_feats", [1.0, 0.0, 1.0, 0.0]))
    if goal_feats.shape[0] >= 2:
        g_l = goal_feats[:2]
    else:
        g_l = np.array([1.0, 0.0], np.float32)
    Rw = rot2(float(pose[2]))
    goal = pose[:2] + g_l @ Rw.T
    return {"start": pose[:2].tolist(), "goal": goal.tolist(), "C": C.tolist(), "R": R.tolist(), "case": "unknown"}


class Stage1EnergyPatchDataset(Dataset):
    """View Stage-1 rollout samples as NavMaximin-style local energy patches.

    Parameters
    ----------
    root_or_file:
        Either a case-suite root, one case directory, or a single split file.
    split:
        train/val/test. Ignored when ``root_or_file`` is a .pt file.
    cases:
        Optional subset of case names when ``root_or_file`` is a suite root.
    H, W, hat_d:
        Local energy-grid resolution and half-width.
    direction_from:
        ``cmd`` uses the MPC action [v, omega] as local direction; ``goal`` uses
        the local goal vector only.  ``cmd`` is recommended for reverse maneuvers.
    """

    def __init__(
        self,
        root_or_file: str,
        split: str = "train",
        cases: Optional[Sequence[str]] = None,
        H: int = 64,
        W: int = 64,
        hat_d: float = 3.5,
        wheelbase: float = 0.324,
        robot_radius: float = 0.23,
        barrier_kind: str = "icp",
        direction_from: str = "cmd",
        max_samples_per_case: Optional[int] = None,
    ):
        self.root_or_file = str(root_or_file)
        self.split = str(split)
        self.H, self.W = int(H), int(W)
        self.hat_d = float(hat_d)
        self.wheelbase = float(wheelbase)
        self.robot_radius = float(robot_radius)
        self.barrier_kind = str(barrier_kind)
        self.direction_from = str(direction_from)
        self.records: List[Tuple[Dict[str, Any], Dict[int, Dict[str, Any]], int, int, str]] = []

        root = Path(root_or_file)
        if root.suffix == ".pt":
            sources = [root]
        else:
            case_dirs = _find_case_dirs(root, cases)
            sources = [p / f"{split}.pt" for p in case_dirs]
        episode_offset = 0
        for case_idx, pt_path in enumerate(sources):
            if not pt_path.exists():
                continue
            payload = _load_payload(pt_path)
            samples = list(payload.get("samples", [])) if isinstance(payload, dict) else []
            if max_samples_per_case is not None:
                samples = samples[: int(max_samples_per_case)]
            case_dir = pt_path.parent
            split_name = str(payload.get("split", split)) if isinstance(payload, dict) else split
            scenes = _load_scenes(case_dir / f"scenes_{split_name}.json")
            case_name = str(payload.get("case", case_dir.name)) if isinstance(payload, dict) else case_dir.name
            for s in samples:
                self.records.append((s, scenes, episode_offset, case_idx, case_name))
            # Large offset so recurrent state never accidentally carries across cases.
            max_ep = 0
            for s in samples:
                max_ep = max(max_ep, int(_as_float(s.get("episode_id", 0))))
            episode_offset += max_ep + 10000
        if not self.records:
            raise RuntimeError(f"No Stage-1 samples found under {root_or_file!r} split={split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def _scene_for(self, sample: Dict[str, Any], scenes: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        eid = int(_as_float(sample.get("episode_id", 0)))
        return scenes.get(eid, _scene_from_sample_fallback(sample))

    def _local_obstacles(self, scene: Dict[str, Any], pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C_w = _as_np(scene.get("C", []))
        R_w = _as_np(scene.get("R", []))
        if C_w.size == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
        C_w = C_w.reshape(-1, 2)
        R_w = R_w.reshape(-1).astype(np.float32, copy=False) + self.robot_radius
        C_l = world_to_local(C_w, pose)
        keep = np.max(np.abs(C_l), axis=1) <= (self.hat_d + R_w)
        return C_l[keep].astype(np.float32, copy=False), R_w[keep].astype(np.float32, copy=False)

    def _direction(self, sample: Dict[str, Any], goal_l: np.ndarray) -> torch.Tensor:
        if self.direction_from == "goal":
            d = goal_l.astype(np.float32, copy=False)
        else:
            cmd = _as_np(sample.get("cmd", [0.0, 0.0]))
            v = float(cmd[0]) if cmd.shape[0] > 0 else 0.0
            omega = float(cmd[1]) if cmd.shape[0] > 1 else 0.0
            # Encode speed sign in x and curvature/turning demand in y.  This is
            # intentionally not a physical lateral velocity; it is the training
            # target for energy-gradient direction and steering extraction.
            d = np.array([v, omega * self.wheelbase], dtype=np.float32)
            if np.linalg.norm(d) < 1e-6:
                d = goal_l.astype(np.float32, copy=False)
        n = float(np.linalg.norm(d))
        if n < 1e-8:
            d = np.array([1.0, 0.0], dtype=np.float32)
        else:
            d = d / n
        return torch.tensor(d, dtype=torch.float32)


    def _pose_seq_local_xy(self, sample: Dict[str, Any], pose: np.ndarray) -> torch.Tensor:
        pose_seq = _as_np(sample.get("pose_seq", np.zeros((0, 3), np.float32)))
        if pose_seq.ndim != 2 or pose_seq.shape[0] == 0:
            return torch.zeros(0, 2, dtype=torch.float32)
        return torch.tensor(world_to_local(pose_seq[:, :2], pose), dtype=torch.float32)

    def _cmd_seq(self, sample: Dict[str, Any]) -> torch.Tensor:
        cmd_seq = _as_np(sample.get("cmd_seq", np.zeros((0, 2), np.float32)))
        if cmd_seq.ndim != 2 or cmd_seq.shape[0] == 0:
            cmd = _as_np(sample.get("cmd", [0.0, 0.0])).reshape(-1)
            if cmd.shape[0] < 2:
                cmd = np.pad(cmd, (0, 2 - cmd.shape[0]))
            cmd_seq = cmd[:2].reshape(1, 2)
        return torch.tensor(cmd_seq[:, :2], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample, scenes, episode_offset, case_idx, case_name = self.records[idx]
        pose = _as_np(sample.get("pose", [0.0, 0.0, 0.0])).reshape(-1)[:3]
        if pose.shape[0] < 3:
            pose = np.pad(pose, (0, 3 - pose.shape[0]))
        scene = self._scene_for(sample, scenes)
        goal_w = _as_np(scene.get("goal", [1.0, 0.0])).reshape(-1)[:2]
        goal_l = world_to_local(goal_w.reshape(1, 2), pose)[0]
        C_l, R_l = self._local_obstacles(scene, pose)
        pose_seq_xy = self._pose_seq_local_xy(sample, pose)
        cmd_seq = self._cmd_seq(sample)
        waypoint_idx = min(max(0, pose_seq_xy.shape[0] - 1), 5) if pose_seq_xy.shape[0] else 0
        waypoint_xy = pose_seq_xy[waypoint_idx] if pose_seq_xy.shape[0] else torch.tensor(goal_l, dtype=torch.float32)

        grid_xy = make_coord_grid(self.H, self.W, self.hat_d)
        barrier_maps: List[torch.Tensor] = []
        obs_feats: List[List[float]] = []
        for c, r in zip(C_l, R_l):
            c_t = torch.tensor(c, dtype=torch.float32)
            sd = torch.linalg.norm(grid_xy - c_t.view(1, 1, 2), dim=-1) - float(r)
            barrier_maps.append(barrier_map_from_sd(sd, self.hat_d, kind=self.barrier_kind))
            norm = float(np.linalg.norm(c))
            ang = float(math.atan2(float(c[1]), float(c[0])))
            prox = float((self.hat_d - max(norm - float(r), 0.0)) / max(self.hat_d, 1e-6))
            obs_feats.append([float(c[0]), float(c[1]), float(r), norm, ang, float(np.clip(prox, 0.0, 1.0))])
        if barrier_maps:
            barrier_stack = torch.stack(barrier_maps, dim=0)
            obs = torch.tensor(obs_feats, dtype=torch.float32)
            obs_weights = torch.ones(obs.shape[0], dtype=torch.float32)
        else:
            barrier_stack = torch.zeros(0, self.H, self.W, dtype=torch.float32)
            obs = torch.zeros(0, 6, dtype=torch.float32)
            obs_weights = torch.zeros(0, dtype=torch.float32)

        goal_l_t = torch.tensor(goal_l, dtype=torch.float32)
        gl_norm = torch.linalg.norm(goal_l_t)
        gl_ang = torch.atan2(goal_l_t[1], goal_l_t[0])
        goal_feats = torch.stack([goal_l_t[0], goal_l_t[1], gl_norm, gl_ang])
        goal_map = quadratic_goal(grid_xy, goal_l_t).unsqueeze(0)
        coord_map = torch.stack([grid_xy[..., 0] / self.hat_d, grid_xy[..., 1] / self.hat_d], dim=0)
        eid = int(_as_float(sample.get("episode_id", 0))) + int(episode_offset)
        step = int(_as_float(sample.get("step", idx)))
        mode = int(_as_float(sample.get("mode", 0)))
        if isinstance(sample.get("mode", None), torch.Tensor) and sample["mode"].numel() > 0:
            mode = int(sample["mode"].detach().cpu().item())
        case_id = int(_as_float(sample.get("case_id", CASE_TO_ID.get(case_name, case_idx))))
        stage = int(100 * case_id + mode)
        success = int(not bool(sample.get("collision", False)))
        return {
            "grid_xy": grid_xy,
            "goal_map": goal_map,
            "coord_map": coord_map,
            "barrier_stack": barrier_stack,
            "obs_feats": obs,
            "obs_weights": obs_weights,
            "goal_feats": goal_feats,
            "pos_xy": torch.zeros(1, 2, dtype=torch.float32),
            "dir_xy": self._direction(sample, goal_l).view(1, 2),
            "cmd": cmd_seq[0].clone(),
            "cmd_seq": cmd_seq,
            "pose_seq_xy": pose_seq_xy,
            "waypoint_xy": waypoint_xy.clone(),
            "mode": torch.tensor(mode, dtype=torch.long),
            "true_clearance": torch.tensor(float(_as_float(sample.get("true_clearance", 0.0))), dtype=torch.float32),
            "meta": {
                "episode": eid,
                "snap": step,
                "stage_idx": stage,
                "hat_d": float(self.hat_d),
                "success": bool(success),
                "case_id": case_id,
                "scene_seed": int(_as_float(sample.get("scene_seed", 0))),
                "bounds": None,
            },
        }



def _pad_seq_2d(items: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length [T,2] sequences to [B,Tmax,2]."""
    if not items:
        return torch.zeros(0, 0, 2), torch.zeros(0, 0, dtype=torch.bool)
    Tmax = max(int(x.shape[0]) for x in items)
    B = len(items)
    out = torch.zeros(B, Tmax, 2, dtype=items[0].dtype)
    mask = torch.zeros(B, Tmax, dtype=torch.bool)
    for i, x in enumerate(items):
        T = int(x.shape[0])
        if T:
            out[i, :T] = x[:, :2]
            mask[i, :T] = True
    return out, mask


def collate_stage1_energy(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate Stage-1 energy patches and preserve MPC teacher sequences."""
    out = _base_collate_navmax(batch)
    if "cmd" in batch[0]:
        out["cmd"] = torch.stack([b["cmd"] for b in batch], dim=0)
    if "cmd_seq" in batch[0]:
        out["cmd_seq"], out["cmd_seq_mask"] = _pad_seq_2d([b["cmd_seq"] for b in batch])
    if "pose_seq_xy" in batch[0]:
        out["pose_seq_xy"], out["pose_seq_mask"] = _pad_seq_2d([b["pose_seq_xy"] for b in batch])
    if "waypoint_xy" in batch[0]:
        out["waypoint_xy"] = torch.stack([b["waypoint_xy"] for b in batch], dim=0)
    if "mode" in batch[0]:
        out["mode"] = torch.stack([b["mode"] for b in batch], dim=0)
    if "true_clearance" in batch[0]:
        out["true_clearance"] = torch.stack([b["true_clearance"] for b in batch], dim=0)
    return out


# Backward-compatible public name used by older imports.
collate_navmax = _base_collate_navmax


def is_stage1_dataset_path(path: str) -> bool:
    p = Path(path)
    if p.suffix == ".pt":
        try:
            obj = _load_payload(p)
            return isinstance(obj, dict) and "samples" in obj and ("sim_cfg" in obj or "mpc_cfg" in obj)
        except Exception:
            return False
    return (p / "train.pt").exists() or (p / "manifest.json").exists() and any((p / c / "train.pt").exists() for c in CASE_TO_ID)


__all__ = ["Stage1EnergyPatchDataset", "collate_navmax", "collate_stage1_energy", "is_stage1_dataset_path"]
