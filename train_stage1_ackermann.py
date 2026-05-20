#!/usr/bin/env python3
"""
Train the Stage-1 hybrid horizon Ackermann/LaserScan policy.

This version upgrades the previous one-step reactive policy into a small
sequence/option policy:

  scan tokens + local goal/history context
      -> shared token encoder
      -> maneuver mode head
      -> soft barrier beta schedule
      -> H-step raw (v, omega) command proposal
      -> Ackermann projection + sequence rollout loss

The first command of the predicted sequence is still what the closed-loop
controller executes, but training now supervises the full short-horizon sequence.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.stage1_ackermann_scan import (
    AckermannCfg,
    AdapterCfg,
    LaserScanCfg,
    SceneCfg,
    RecoveryCfg,
    Stage1AckermannDataset,
    Stage1Cfg,
    ackermann_step_torch,
    collate_stage1,
    project_twist_torch,
    stage1_cfg_to_dict,
)


class Stage1PolicyNet(nn.Module):
    """Hybrid horizon policy for local pseudo-obstacles + goal/recovery context.

    Inputs:
      obs_feats:  [B,N,6] = [cx,cy,r,||c||,angle,proximity] in robot frame
      obs_mask:   [B,N]
      goal_feats: [B,8] = [gx,gy,||g||,angle,current_speed,min_clear,front_clear,stall]

    Outputs when ``return_all=True``:
      raw_seq:     [B,H,2] command sequence proposal (v, omega), pre-projection
      mode_logits: [B,6]   maneuver option logits
      beta_seq:    [B,H]   soft planning barrier schedule in [0, +)
      value:       [B]     scalar cost-to-go diagnostic

    For backward compatibility, ``forward(..., return_all=False)`` returns the
    first raw command [B,2], so existing evaluation wrappers still work.
    """
    def __init__(
        self,
        d_obs: int = 6,
        d_goal: int = 8,
        d_tok: int = 96,
        n_layers: int = 2,
        horizon: int = 12,
        n_modes: int = 6,
    ):
        super().__init__()
        self.d_tok = int(d_tok)
        self.horizon = int(horizon)
        self.n_modes = int(n_modes)
        self.obs_enc = nn.Sequential(
            nn.Linear(d_obs, 128), nn.ReLU(),
            nn.Linear(128, d_tok), nn.ReLU(),
        )
        self.goal_enc = nn.Sequential(
            nn.Linear(d_goal, 128), nn.ReLU(),
            nn.Linear(128, d_tok), nn.ReLU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_tok, nhead=4, dim_feedforward=192, batch_first=True
        )
        self.fuser = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.mode_head = nn.Sequential(
            nn.Linear(d_tok, 128), nn.ReLU(), nn.Linear(128, n_modes)
        )
        self.beta_head = nn.Sequential(
            nn.Linear(d_tok, 128), nn.ReLU(), nn.Linear(128, horizon)
        )
        self.seq_head = nn.Sequential(
            nn.Linear(d_tok, 192), nn.ReLU(),
            nn.Linear(192, 128), nn.ReLU(),
            nn.Linear(128, horizon * 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_tok, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def encode(self, obs_feats: torch.Tensor, obs_mask: torch.Tensor, goal_feats: torch.Tensor) -> torch.Tensor:
        B, N = obs_feats.shape[:2]
        z_goal = self.goal_enc(goal_feats).unsqueeze(1)
        if N == 0:
            tokens = z_goal
            pad = torch.zeros(B, 1, dtype=torch.bool, device=goal_feats.device)
        else:
            z_obs = self.obs_enc(obs_feats.reshape(B * N, -1)).reshape(B, N, -1)
            tokens = torch.cat([z_goal, z_obs], dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=obs_mask.device), ~obs_mask], dim=1)
        z = self.fuser(tokens, src_key_padding_mask=pad)
        return z[:, 0]

    def forward(self, obs_feats: torch.Tensor, obs_mask: torch.Tensor, goal_feats: torch.Tensor,
                return_all: bool = False):
        ctx = self.encode(obs_feats, obs_mask, goal_feats)
        raw_seq = self.seq_head(ctx).reshape(ctx.shape[0], self.horizon, 2)
        out = {
            "raw_seq": raw_seq,
            "mode_logits": self.mode_head(ctx),
            "beta_seq": F.softplus(self.beta_head(ctx)),
            "value": self.value_head(ctx).squeeze(-1),
        }
        if return_all:
            return out
        return raw_seq[:, 0, :]


@dataclass
class TrainStage1Cfg:
    epochs: int = 20
    samples: int = 40000
    val_samples: int = 4096
    bs: int = 256
    lr: float = 3e-4
    workers: int = 0
    horizon: int = 12
    w_cmd: float = 1.0
    w_seq: float = 0.75
    w_dyn: float = 0.35
    w_final: float = 0.25
    w_mode: float = 0.20
    w_beta: float = 0.05
    w_turn: float = 1e-3
    w_move: float = 0.05
    w_reverse: float = 0.50
    move_speed_floor: float = 0.10
    seed: int = 2026
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_stage1_cfg(args: argparse.Namespace) -> Stage1Cfg:
    return Stage1Cfg(
        scene=SceneCfg(
            robot_radius=args.robot_radius,
            n_obs_range=(args.n_obs_min, args.n_obs_max),
            corridor_half_width=args.corridor_half_width,
        ),
        scan=LaserScanCfg(
            n_beams=args.n_beams,
            fov=args.fov_deg * 3.141592653589793 / 180.0,
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
            p_recovery_sample=args.p_recovery,
            move_speed_floor=args.move_speed_floor,
            stall_steps_trigger=args.stall_steps_trigger,
            train_horizon=args.horizon,
        ),
        seed=args.seed,
    )


def project_sequence(raw_seq: torch.Tensor, ack_cfg: AckermannCfg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, _ = raw_seq.shape
    flat = raw_seq.reshape(B * H, 2)
    v, omega, delta = project_twist_torch(flat, ack_cfg)
    return v.reshape(B, H), omega.reshape(B, H), delta.reshape(B, H)


def unroll_ackermann_torch(pose0: torch.Tensor, v_seq: torch.Tensor, omega_seq: torch.Tensor, dt: float) -> torch.Tensor:
    poses = []
    pose = pose0
    H = v_seq.shape[1]
    for k in range(H):
        pose = ackermann_step_torch(pose, v_seq[:, k], omega_seq[:, k], dt)
        poses.append(pose)
    return torch.stack(poses, dim=1)


def pose_sequence_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    xy = F.mse_loss(pred[..., :2], target[..., :2])
    dth = torch.atan2(torch.sin(pred[..., 2] - target[..., 2]), torch.cos(pred[..., 2] - target[..., 2]))
    return xy + 0.05 * (dth ** 2).mean()


def run_epoch(model: Stage1PolicyNet, loader: DataLoader, cfg: TrainStage1Cfg, ack_cfg: AckermannCfg,
              optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    logs = {
        "loss": 0.0, "cmd": 0.0, "seq": 0.0, "dyn": 0.0, "final": 0.0,
        "mode": 0.0, "beta": 0.0, "turn": 0.0, "move": 0.0, "reverse": 0.0,
        "v_mae": 0.0, "omega_mae": 0.0, "reverse_tgt_rate": 0.0, "mode_acc": 0.0,
    }
    n = 0
    for batch in loader:
        obs = batch["obs_feats"].to(cfg.device)
        mask = batch["obs_mask"].to(cfg.device)
        goal = batch["goal_feats"].to(cfg.device)
        cmd_tgt = batch["cmd"].to(cfg.device)
        cmd_seq_tgt = batch["cmd_seq"].to(cfg.device)
        pose = batch["pose"].to(cfg.device)
        next_pose = batch["next_pose"].to(cfg.device)
        pose_seq_tgt = batch["pose_seq"].to(cfg.device)
        beta_seq_tgt = batch["beta_seq"].to(cfg.device)
        mode_tgt = batch["mode"].to(cfg.device)

        out = model(obs, mask, goal, return_all=True)
        raw_seq = out["raw_seq"]
        v_seq, omega_seq, delta_seq = project_sequence(raw_seq, ack_cfg)
        cmd_seq = torch.stack([v_seq, omega_seq], dim=-1)
        cmd = cmd_seq[:, 0, :]
        pred_next = ackermann_step_torch(pose, v_seq[:, 0], omega_seq[:, 0], ack_cfg.dt)
        pred_pose_seq = unroll_ackermann_torch(pose, v_seq, omega_seq, ack_cfg.dt)

        L_cmd = F.mse_loss(cmd, cmd_tgt)
        L_seq = F.mse_loss(cmd_seq, cmd_seq_tgt)
        L_dyn_first = F.mse_loss(pred_next[:, :2], next_pose[:, :2])
        dth1 = torch.atan2(torch.sin(pred_next[:, 2] - next_pose[:, 2]), torch.cos(pred_next[:, 2] - next_pose[:, 2]))
        L_dyn = L_dyn_first + 0.05 * (dth1 ** 2).mean() + pose_sequence_loss(pred_pose_seq, pose_seq_tgt)
        L_final = F.mse_loss(pred_pose_seq[:, -1, :2], pose_seq_tgt[:, -1, :2])
        L_mode = F.cross_entropy(out["mode_logits"], mode_tgt)
        beta_norm = torch.clamp(out["beta_seq"], 0.0, 4.0) / 4.0
        L_beta = F.mse_loss(beta_norm, beta_seq_tgt)
        L_turn = (delta_seq ** 2).mean()

        gdist = goal[:, 2]
        min_clear = goal[:, 5]
        safe_to_move = ((gdist > 0.50) & (min_clear > 0.10)).to(v_seq.dtype)
        speed_floor_err = torch.relu(cfg.move_speed_floor - torch.abs(v_seq[:, 0])) ** 2
        L_move = (safe_to_move * speed_floor_err).sum() / safe_to_move.sum().clamp_min(1.0)

        rev_mask = (cmd_seq_tgt[:, :, 0] < -ack_cfg.v_min).to(v_seq.dtype)
        if rev_mask.sum() > 0:
            L_reverse = (rev_mask * (v_seq - cmd_seq_tgt[:, :, 0]) ** 2).sum() / rev_mask.sum().clamp_min(1.0)
        else:
            L_reverse = torch.zeros((), device=v_seq.device, dtype=v_seq.dtype)

        L = (cfg.w_cmd * L_cmd + cfg.w_seq * L_seq + cfg.w_dyn * L_dyn + cfg.w_final * L_final
             + cfg.w_mode * L_mode + cfg.w_beta * L_beta + cfg.w_turn * L_turn
             + cfg.w_move * L_move + cfg.w_reverse * L_reverse)

        if train:
            optimizer.zero_grad(set_to_none=True)
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        with torch.no_grad():
            err = torch.abs(cmd - cmd_tgt)
            mode_acc = (out["mode_logits"].argmax(dim=-1) == mode_tgt).float().mean()
            logs["loss"] += float(L.item())
            logs["cmd"] += float(L_cmd.item())
            logs["seq"] += float(L_seq.item())
            logs["dyn"] += float(L_dyn.item())
            logs["final"] += float(L_final.item())
            logs["mode"] += float(L_mode.item())
            logs["beta"] += float(L_beta.item())
            logs["turn"] += float(L_turn.item())
            logs["move"] += float(L_move.item())
            logs["reverse"] += float(L_reverse.item())
            logs["reverse_tgt_rate"] += float((cmd_seq_tgt[:, :, 0] < -ack_cfg.v_min).float().mean().item())
            logs["v_mae"] += float(err[:, 0].mean().item())
            logs["omega_mae"] += float(err[:, 1].mean().item())
            logs["mode_acc"] += float(mode_acc.item())
        n += 1

    for k in logs:
        logs[k] /= max(1, n)
    return logs


def main():
    ap = argparse.ArgumentParser("Train Stage-1 hybrid horizon Ackermann + synthetic LaserScan policy")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--val-samples", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; keep small for local smoke tests")
    ap.add_argument("--outdir", type=str, default="checkpoints/stage1_ackermann_horizon")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=12)

    # Calibration / car envelope. Replace these with measured UT car values.
    ap.add_argument("--wheelbase", type=float, default=0.324)
    ap.add_argument("--v-max", type=float, default=0.80)
    ap.add_argument("--v-reverse-max", type=float, default=0.25)
    ap.add_argument("--max-steering-angle", type=float, default=0.396)
    ap.add_argument("--min-turning-radius", type=float, default=0.80)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--robot-radius", type=float, default=0.23)

    # Sensor / scene randomization.
    ap.add_argument("--n-beams", type=int, default=181)
    ap.add_argument("--fov-deg", type=float, default=270.0)
    ap.add_argument("--range-max", type=float, default=6.0)
    ap.add_argument("--scan-noise", type=float, default=0.015)
    ap.add_argument("--scan-dropout", type=float, default=0.01)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--n-obs-min", type=int, default=8)
    ap.add_argument("--n-obs-max", type=int, default=20)
    ap.add_argument("--corridor-half-width", type=float, default=0.70)
    ap.add_argument("--p-recovery", type=float, default=0.30, help="fraction of training samples biased to near-contact recovery")
    ap.add_argument("--move-speed-floor", type=float, default=0.10)
    ap.add_argument("--stall-steps-trigger", type=int, default=8)

    # Loss weights.
    ap.add_argument("--w-cmd", type=float, default=1.0)
    ap.add_argument("--w-seq", type=float, default=0.75)
    ap.add_argument("--w-dyn", type=float, default=0.35)
    ap.add_argument("--w-final", type=float, default=0.25)
    ap.add_argument("--w-mode", type=float, default=0.20)
    ap.add_argument("--w-beta", type=float, default=0.05)
    ap.add_argument("--w-turn", type=float, default=1e-3)
    ap.add_argument("--w-move", type=float, default=0.05)
    ap.add_argument("--w-reverse", type=float, default=0.75)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    sim_cfg = build_stage1_cfg(args)
    train_cfg = TrainStage1Cfg(
        epochs=args.epochs, samples=args.samples, val_samples=args.val_samples, bs=args.bs,
        lr=args.lr, workers=args.workers, horizon=args.horizon,
        w_cmd=args.w_cmd, w_seq=args.w_seq, w_dyn=args.w_dyn, w_final=args.w_final,
        w_mode=args.w_mode, w_beta=args.w_beta, w_turn=args.w_turn,
        w_move=args.w_move, w_reverse=args.w_reverse,
        move_speed_floor=args.move_speed_floor, seed=args.seed,
    )

    train_ds = Stage1AckermannDataset(n_samples=args.samples, cfg=sim_cfg, seed=args.seed)
    val_ds = Stage1AckermannDataset(n_samples=args.val_samples, cfg=sim_cfg, seed=args.seed + 100000)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                          collate_fn=collate_stage1, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_stage1, drop_last=False)

    model = Stage1PolicyNet(horizon=sim_cfg.recovery.train_horizon)
    model.to(train_cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model_cfg = {"horizon": model.horizon, "n_modes": model.n_modes, "d_tok": model.d_tok}
    with open(os.path.join(args.outdir, "stage1_config.json"), "w") as f:
        json.dump({"sim_cfg": stage1_cfg_to_dict(sim_cfg), "train_cfg": train_cfg.__dict__, "model_cfg": model_cfg}, f, indent=2)

    best = float("inf")
    for ep in range(args.epochs):
        tr = run_epoch(model, train_dl, train_cfg, sim_cfg.ackermann, opt)
        with torch.no_grad():
            va = run_epoch(model, val_dl, train_cfg, sim_cfg.ackermann, None)
        print({f"ep{ep}/train_{k}": round(v, 6) for k, v in tr.items()} |
              {f"ep{ep}/val_{k}": round(v, 6) for k, v in va.items()})

        ckpt = {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "sim_cfg": stage1_cfg_to_dict(sim_cfg),
            "train_cfg": train_cfg.__dict__,
            "model_cfg": model_cfg,
            "train_logs": tr,
            "val_logs": va,
        }
        torch.save(ckpt, os.path.join(args.outdir, "latest.pt"))
        if va["loss"] < best:
            best = va["loss"]
            torch.save(ckpt, os.path.join(args.outdir, "best.pt"))
        if ep % 5 == 0 or ep == args.epochs - 1:
            torch.save(ckpt, os.path.join(args.outdir, f"epoch_{ep:03d}.pt"))


if __name__ == "__main__":
    main()
