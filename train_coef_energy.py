#!/usr/bin/env python3
"""
Surrogate-coefficient training pipeline + short-rollout validation loss.

What this does
--------------
• Replaces grid U prediction with a compact model that predicts:
    - per-obstacle barrier weights α_j ≥ 0 (for IPC-like barrier)
    - a goal attraction strength β_goal ≥ 0
    - a linear damping (friction) scale γ ≥ 0 applied to the rigid velocity v_o

• Adds a dataset of short, random rollouts drawn from the stagewise episodes
  produced by stagewise_dataset.py. Each sample contains:
    - initial center o0 and approximate velocity v0
    - the local stage goal g, effective obstacles (inflated radii) at t0
    - a random enlarged step size dt' and a short horizon H (2..6 steps)
    - the target center/velocity after K = round(H*dt'/dt_base) stagewise steps

• The training loss integrates the *surrogate dynamics* from (o0, v0) for H
  steps with step dt', using the predicted (α, β_goal, γ), and matches the
  final (o, v) to the stagewise trajectory. Differentiable end-to-end.

Why a surrogate integrator?
---------------------------
Directly backpropagating through the full hyperelastic ring integration can be
heavy. We validate the learned coefficients on a light rigid-point proxy at the
center o(t) while still using the *same IPC-style barrier law* (with d_hat) and
local obstacles (inflated) that the stagewise integrator used on that frame.

This keeps supervision close to the stagewise policy but makes learning fast and
stable. You can increase fidelity later (e.g., multi-sample ring points).

Usage
-----
python -m train_coef_energy \
  --root ./nav_stagewise_hyperring \
  --epochs 50 --bs 64 --lr 3e-4 --workers 4

Assumptions
-----------
• You already generated episodes with stagewise_dataset.py (manifest.json exists).
• Each episode dir has logs/stagewise_checkpoints.jsonl written by the generator.

Notes
-----
• Handles N=0 obstacles cleanly (no-ops with correct shapes).

• If you want to make damping relative to the episode's nominal gamma_o, you can
  flip the flag --gamma_rel True, which makes gamma = gamma_rel * gamma_o(ep).
"""
from __future__ import annotations
import os, json, math, random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from surrogate_robust import integrate_surrogate_v2, multi_start_penalty


import numpy as np
import builtins
try:
    from torch.serialization import add_safe_globals, safe_globals
except Exception:
    # older torch doesn't have these; fall back to default load later
    add_safe_globals = None
    safe_globals = None


def safe_torch_load(path, map_location="cpu"):
    if safe_globals is None:
        # older torch: just do a normal load
        return torch.load(path, map_location=map_location)
    # allow-list the minimal set commonly present in .pt files that carry numpy arrays
    allow = [
        np.dtype,
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
        np.ndarray,
        np.generic,
        np._core.multiarray._reconstruct,  # TODO: remove unsafe pytorch save scheme
        np._core.multiarray.scalar,
        builtins.set,
        builtins.slice,
        builtins.range,
    ]
    with safe_globals(allow):
        return torch.load(path, map_location=map_location, weights_only=True)

# -----------------------------
# Small math helpers (IPC piecewise)
# -----------------------------


def ipc_piecewise(d: torch.Tensor, d_hat: torch.Tensor | float, vp: float = -5e2, eps: float = 1e-9,
                  max_grad: float = 200.0, max_b: float = 200.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (b(d), db/dd) for IPC-like barrier. Accepts scalar or batched d_hat and broadcasts."""
    # Normalize d_hat and broadcast to d
    if not torch.is_tensor(d_hat):
        dh = d.new_tensor(float(d_hat))
    else:
        dh = d_hat.to(device=d.device, dtype=d.dtype)
    while dh.ndim < d.ndim:
        dh = dh.unsqueeze(-1)
    dh = torch.broadcast_to(dh, d.shape)

    safe = torch.clamp(d, min=float(eps))
    b_in = -(d - dh) ** 2 * torch.log(safe / dh)
    dbdd_in = (dh - d) * (2.0 * torch.log(safe / dh) - dh / safe) + 1.0
    b = torch.where(d <= eps, d.new_tensor(vp), torch.where(d < dh, b_in, torch.zeros_like(d)))
    dbdd = torch.where(d <= eps, d.new_tensor(vp), torch.where(d < dh, dbdd_in, torch.zeros_like(d)))
    b = torch.clamp(b, 0.0, float(max_b))
    dbdd = torch.clamp(dbdd, -float(max_grad), float(max_grad))
    return b, dbdd
# -----------------------------
# Dataset: short random rollouts from stagewise episodes
# -----------------------------

@dataclass
class ShortRolloutCfg:
    min_h: int = 2
    max_h: int = 6
    dt_mult_range: Tuple[float, float] = (1.0, 3.0)  # enlarge step for robustness
    max_skip_to_end: int = 6  # if t1+1 would exceed episode, clamp within this

class ShortRollouts(Dataset):
    """Builds random short rollouts from the stagewise checkpoints JSONL.

    Each __getitem__ returns a dict with:
      - o0: float32[2]
      - v0: float32[2]  (FD from centers at t0 and t0+1)
      - goal: float32[2] (stage_exit at t0)
      - C, R, W: obstacles at t0 (effective radii R already inflated)
      - d_hat: float32 scalar (barrier activation)
      - dt_base: float32, dt' (sampled), H (int)
      - o_tgt, v_tgt: target state after K steps of base integrator
    """
    def __init__(self, root: str, cfg: Optional[ShortRolloutCfg] = None):
        super().__init__()
        self.root = root
        self.cfg = cfg or ShortRolloutCfg()
        man = os.path.join(root, "manifest.json")
        with open(man, "r") as f:
            self.records = json.load(f)
        # Preload checkpoint paths and run header (for base dt & gamma_o)
        self.items: List[Dict[str, Any]] = []
        for rec in self.records:
            ep = safe_torch_load(rec["path"], map_location="cpu")
            logs = ep["logs"]
            ck_path = logs["checkpoints_jsonl"]
            with open(ck_path, "r") as f:
                cks = [json.loads(line) for line in f]
            # Drop first/last if we need v estimates
            if len(cks) < 3:
                continue
            # dt is constant, set in run_episode
            dt_base = float(cks[0].get("dt", ep["meta"].get("dt", 0.03)))
            # optional gamma_o
            gamma_o = float(ep.get("params", {}).get("gamma_o", 4.0))
            # store compact episode cache
            epi = {
                "dt": dt_base,
                "gamma_o": gamma_o,
                "cks": cks,
            }
            self.items.append(epi)

    def __len__(self) -> int:
        return max(1, sum(len(epi["cks"]) for epi in self.items) // 6)

    @staticmethod
    def _vel_fd(c0: List[float], c1: List[float], dt: float) -> torch.Tensor:
        c0t = torch.tensor(c0, dtype=torch.float32)
        c1t = torch.tensor(c1, dtype=torch.float32)
        return (c1t - c0t) / float(dt)

    def __getitem__(self, _idx: int) -> Dict[str, Any]:
        # pick a random episode, then a valid t0
        epi = random.choice(self.items)
        cks = epi["cks"]
        dt = epi["dt"]
        T = len(cks)
        # random horizon and enlarged step
        H = random.randint(self.cfg.min_h, self.cfg.max_h)
        dt_mult = random.uniform(*self.cfg.dt_mult_range)
        K = max(1, int(round(H * dt_mult)))
        t0 = random.randint(0, max(1, T - K - 2))
        t1 = min(T - 2, t0 + K)

        c0 = cks[t0]
        c1 = cks[t1]
        c_next = cks[min(T - 1, t1 + 1)]

        # states
        o0 = torch.tensor(c0["center"], dtype=torch.float32)
        o_tgt = torch.tensor(c1["center"], dtype=torch.float32)
        v0 = self._vel_fd(c0["center"], cks[t0 + 1]["center"], dt)
        v_tgt = self._vel_fd(c1["center"], c_next["center"], dt)

        # local goal = current stage exit (goal of this frame)
        goal = torch.tensor(c0["stage_exit"], dtype=torch.float32)

        # obstacles effective at t0 (already inflated radii)
        obs = c0["obstacles_effective"]
        C = torch.tensor(obs.get("C", []), dtype=torch.float32)
        R = torch.tensor(obs.get("R_eff", []), dtype=torch.float32)
        W = torch.tensor(obs.get("W", []), dtype=torch.float32)
        if C.ndim == 1:
            C = C.reshape(0, 2)
        d_hat = float(c0["barrier"]["barrier_d_hat"]) if "barrier" in c0 else 1.0

        return {
            "o0": o0, "v0": v0, "goal": goal,
            "C": C, "R": R, "W": W,
            "d_hat": torch.tensor(d_hat, dtype=torch.float32),
            "dt_base": torch.tensor(dt, dtype=torch.float32),
            "dt_prime": torch.tensor(dt_mult * dt, dtype=torch.float32),
            "H": torch.tensor(H, dtype=torch.int64),
            "o_tgt": o_tgt, "v_tgt": v_tgt,
            "gamma_o": torch.tensor(epi["gamma_o"], dtype=torch.float32),
        }

# -------- collate: pad variable-N obstacles safely (N may be 0) --------

def collate_short(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    maxN = max((item["C"].shape[0] for item in batch), default=0)
    # tensors (B, ...)
    o0 = torch.stack([b["o0"] for b in batch], 0)
    v0 = torch.stack([b["v0"] for b in batch], 0)
    goal = torch.stack([b["goal"] for b in batch], 0)
    d_hat = torch.stack([b["d_hat"] for b in batch], 0)
    dt_base = torch.stack([b["dt_base"] for b in batch], 0)
    dt_prime = torch.stack([b["dt_prime"] for b in batch], 0)
    H = torch.stack([b["H"] for b in batch], 0)
    o_tgt = torch.stack([b["o_tgt"] for b in batch], 0)
    v_tgt = torch.stack([b["v_tgt"] for b in batch], 0)
    gamma_o = torch.stack([b["gamma_o"] for b in batch], 0)

    # obstacles padded
    C = torch.zeros(B, maxN, 2, dtype=torch.float32)
    R = torch.zeros(B, maxN, dtype=torch.float32)
    W = torch.zeros(B, maxN, dtype=torch.float32)
    mask = torch.zeros(B, maxN, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["C"].shape[0]
        if n:
            C[i, :n] = b["C"]
            R[i, :n] = b["R"]
            W[i, :n] = b["W"]
            mask[i, :n] = True
    
    # features per obstacle: [cx, cy, r, w, dx_goal, dy_goal]
    dxdy = (goal.unsqueeze(1) - C)  # (B,N,2)
    obs_feats = torch.cat([C, R.unsqueeze(-1), W.unsqueeze(-1), dxdy], dim=-1) if maxN>0 else torch.zeros(B,0,6)

    # goal feats (center -> goal offset and norms)
    dg = (goal - o0)
    gdist = torch.linalg.norm(dg, dim=-1, keepdim=True)
    goal_feats = torch.cat([dg, gdist, torch.ones_like(gdist)], dim=-1)  # [B,4]

    return {
        "o0": o0, "v0": v0, "goal": goal,
        "C": C, "R": R, "W": W, "obs_mask": mask,
        "obs_feats": obs_feats, "goal_feats": goal_feats,
        "d_hat": d_hat, "dt_base": dt_base, "dt_prime": dt_prime, "H": H,
        "o_tgt": o_tgt, "v_tgt": v_tgt, "gamma_o": gamma_o,
    }

# -----------------------------
# Model: coefficient heads over tokens
# -----------------------------

class ObstacleEncoder(nn.Module):
    def __init__(self, d_in=6, d_tok=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, d_tok)
        )
    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # [B,N,d_in]
        B, N = feats.shape[0], feats.shape[1]
        if N == 0:
            return feats.new_zeros(B, 0, 64)
        x = feats.reshape(B * N, feats.shape[-1])
        z = self.mlp(x).reshape(B, N, -1)
        return z

class CoefEnergyNet(nn.Module):
    """Predicts {α_j}_j, β_goal, γ from local obstacle & goal context.

    α_j ≥ 0 via softplus. β_goal ≥ 0 via softplus. γ ≥ 0 via softplus.
    If --gamma_rel, interpret γ as a *multiplier* on episode gamma_o.
    """
    def __init__(self, d_obs=6, d_goal=4, d_tok=64, d_ctx=64):
        super().__init__()
        self.obs_enc = ObstacleEncoder(d_in=d_obs, d_tok=d_tok)
        self.goal_enc = nn.Sequential(nn.Linear(d_goal, 64), nn.ReLU(), nn.Linear(64, d_tok))
        enc = nn.TransformerEncoderLayer(d_model=d_tok, nhead=4, dim_feedforward=128, batch_first=True)
        self.fuser = nn.TransformerEncoder(enc, num_layers=2)
        # heads
        self.alpha_head = nn.Sequential(nn.Linear(d_tok, 64), nn.ReLU(), nn.Linear(64, 1))
        self.beta_head  = nn.Sequential(nn.Linear(d_tok, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gamma_head = nn.Sequential(nn.Linear(d_tok, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs_feats: torch.Tensor, obs_mask: torch.Tensor, goal_feats: torch.Tensor):
        B, N = obs_feats.shape[0], obs_feats.shape[1]
        z_goal = self.goal_enc(goal_feats).unsqueeze(1)  # [B,1,d]
        if N == 0:
            tokens = z_goal
            pad = torch.zeros(B, 1, dtype=torch.bool, device=obs_feats.device)
            z_all = self.fuser(tokens, src_key_padding_mask=pad)
            ctx = z_all[:, 0]
            alphas = obs_feats.new_zeros(B, 0)
        else:
            z_obs = self.obs_enc(obs_feats)
            tokens = torch.cat([z_goal, z_obs], dim=1)  # [B,1+N,d]
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=obs_mask.device), ~obs_mask], dim=1)
            z_all = self.fuser(tokens, src_key_padding_mask=pad)
            ctx = z_all[:, 0]
            # α per-obstacle from its token (z_all[:,1:])
            a = self.alpha_head(z_all[:, 1:]).squeeze(-1)
            a = F.softplus(a)  # ≥ 0
            alphas = torch.where(obs_mask, a, torch.zeros_like(a))
        beta  = F.softplus(self.beta_head(ctx)).squeeze(-1)   # [B]
        gamma = F.softplus(self.gamma_head(ctx)).squeeze(-1)  # [B]
        return alphas, beta, gamma

# -----------------------------
# Surrogate integrator (rigid center only)
# -----------------------------

def integrate_surrogate(o0: torch.Tensor, v0: torch.Tensor, goal: torch.Tensor,
                        C: torch.Tensor, R: torch.Tensor, mask: torch.Tensor,
                        alphas: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                        d_hat: torch.Tensor, dt: torch.Tensor, H: torch.Tensor,
                        mass: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate B independent samples for H_i steps with step dt_i.
    All inputs are batched (B, ...). Returns (oT, vT, min_clear_along) for penalty.

    Force model: F = F_barrier(α, d_hat) + F_goal(β) - γ * v
      • F_goal = -β * (o - goal)
      • F_barrier = -Σ_j α_j * db/dd(d_j) * n_j, with d_j = ||o - C_j|| - R_j
    """
    B, N = C.shape[0], C.shape[1]
    o = o0.clone()
    v = v0.clone()
    min_clear = torch.full((B,), float("inf"), dtype=o.dtype, device=o.device)
    for _ in range(int(H.max().item())):
        active = (_ < H).to(o.dtype).unsqueeze(-1)  # shape (B,1)
        # goal force
        F_goal = -beta.unsqueeze(-1) * (o - goal)
        # barrier force
        if N == 0:
            F_bar = torch.zeros_like(o)
            dmin = torch.full_like(min_clear, float("inf"))
        else:
            diff = o.unsqueeze(1) - C    # (B,N,2)
            r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)  # (B,N)
            n_hat = diff / r.unsqueeze(-1)
            d = r - R
            # mask out padded obstacles
            d = torch.where(mask, d, torch.full_like(d, 1e6))
            b, dbdd = ipc_piecewise(d, d_hat.view(-1, 1))  # broadcast d_hat per-B
            # F_bar = - Σ α_j * db/dd * n_hat
            F_bar = -(alphas * dbdd).unsqueeze(-1) * n_hat
            F_bar = F_bar.sum(dim=1)
            dmin = torch.where(mask, d, torch.full_like(d, float("inf"))).min(dim=1).values
        min_clear = torch.minimum(min_clear, dmin)
        # total force & update
        F_tot = F_bar + F_goal - gamma.unsqueeze(-1) * v
        a = F_tot / float(mass)
        o = o + active * dt.unsqueeze(-1) * v
        v = v + active * dt.unsqueeze(-1) * a
    return o, v, min_clear

# -----------------------------
# Trainer
# -----------------------------

@dataclass
class TrainCfg:
    epochs: int = 50
    bs: int = 128
    lr: float = 1e-4
    workers: int = 4
    w_traj: float = 1.0
    w_vel: float = 1.0
    w_friction: float = 0.1
    w_clear: float = 5e-3  # penalty if predicted path penetrates
    gamma_rel: bool = False
    margin_factor: float = 0.5   # minimal squeeze margin = 0.5 * radius
    w_multi: float = 0.5        # weight for multi-start penalty
    ms_count: int = 20            # # of aux starts per sample
    ms_h: int = 3                # short horizon for each aux rollout
    ms_dt_mult: float = 4.0      # enlarge dt for robustness
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self, model: CoefEnergyNet, cfg: TrainCfg):
        self.model = model.to(cfg.device)
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.cfg = cfg
        self.version = 2


    def step_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        dev = self.cfg.device
        o0 = batch["o0"].to(dev)
        v0 = batch["v0"].to(dev)
        goal = batch["goal"].to(dev)
        C = batch["C"].to(dev)
        R = batch["R"].to(dev)
        W = batch["W"].to(dev)
        mask = batch["obs_mask"].to(dev)
        obs_feats = batch["obs_feats"].to(dev)
        goal_feats = batch["goal_feats"].to(dev)
        d_hat = batch["d_hat"].to(dev)
        dt_prime = batch["dt_prime"].to(dev)
        H = batch["H"].to(dev)
        o_tgt = batch["o_tgt"].to(dev)
        v_tgt = batch["v_tgt"].to(dev)
        gamma_o = batch["gamma_o"].to(dev)

        self.model.train()
        alphas, beta, gamma = self.model(obs_feats, mask, goal_feats)
        if self.cfg.gamma_rel:
            gamma = gamma * gamma_o  # interpret as multiplier on ep gamma_o

        
        # integrate surrogate
        if self.version == 2:
            
            # ... after you’ve pulled batch tensors, and predicted alphas, beta, gamma:

            # main rollout (radius-aware)
            oT, vT, clr = integrate_surrogate_v2(
                o0, v0, goal, C, R, mask,
                alphas, beta, gamma,
                d_hat, dt_prime, H,
                robot_radius=batch.get("radius", torch.zeros(o0.shape[0], device=o0.device)),
                margin_factor=self.cfg.margin_factor  # set to 0.5 in cfg
            )
            # losses
            L_traj = F.mse_loss(oT, o_tgt)
            L_vel  = F.mse_loss(vT, v_tgt)
            L_alpha = (alphas.mean() if alphas.numel() else torch.tensor(0.0, device=dev))
            

            # multi-start robustness penalty near obstacles
            L_multi = multi_start_penalty(
                o0, v0, goal, C, R, mask,
                alphas, beta, gamma,
                d_hat, dt_prime, H,
                robot_radius=batch.get("radius", 0.0),
                margin_factor=self.cfg.margin_factor,
                ms_count=self.cfg.ms_count,   # e.g., 3
                ms_h=self.cfg.ms_h,           # e.g., 2
                ms_dt_mult=self.cfg.ms_dt_mult  # e.g., 1.5
            )

            
            L_friction = F.mse_loss(gamma, gamma_o)

            # add to loss
            L = (
                self.cfg.w_traj * L_traj
            + self.cfg.w_vel  * L_vel
            + self.cfg.w_friction * L_friction
            + self.cfg.w_multi * L_multi
            #+ self.cfg.w_stage * L_stage     # if you attached the real stagewise hook
            )

            # barrier penalty if min clearance < 0 (penetration)
            m = self.cfg.margin_factor * batch.get("radius", 0.0)  # per-sample margin
            tau = getattr(self.cfg, "prox_tau", 0.05)
            pen = torch.nn.functional.softplus(((m - clr) / tau)).mean()

            # add weight
            # L = L + getattr(self.cfg, "w_prox", 0.1) * pen
        else:

            oT, vT, clr = integrate_surrogate(o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt_prime, H)
            # losses
            L_traj = F.mse_loss(oT, o_tgt)
            L_vel  = F.mse_loss(vT, v_tgt)
            L_alpha = (alphas.mean() if alphas.numel() else torch.tensor(0.0, device=dev))
            # barrier penalty if min clearance < 0 (penetration)
            pen = F.relu(-clr).mean()
            L = self.cfg.w_traj * L_traj + self.cfg.w_vel * L_vel + self.cfg.w_clear * pen

        self.opt.zero_grad()
        L.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()

        return {
            "loss": float(L.item()),
            "traj": float(L_traj.item()),
            "vel": float(L_vel.item()),
            "friction": float(L_friction.item()),
            "alpha": float(L_alpha.item() if alphas.numel() else 0.0),
            "multi_val": float(L_multi.item()),
            "pen": float(pen.item())
        }

# -----------------------------
# CLI
# -----------------------------

def main():
    # python -m train_coef_energy --root ./nav_stagewise_hyperring --epochs 50 --bs 128 --lr 1e-4 --workers 4 --outdir checkpoints/coef_energy_radius --save-every 10
    # default 0.1 friction + 0.5 multi-check (for alpha)
    import argparse
    ap = argparse.ArgumentParser("Wrapper trainer with checkpointing")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gamma_rel", action="store_true")
    ap.add_argument("--outdir", type=str, default=os.environ.get("CKPT_DIR", "checkpoints/coef_energy"))
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--w_friction", type=float, default=0.1)
    ap.add_argument("--w_multi", type=float, default=0.5)
    
    args = ap.parse_args()

    # data
    ds = ShortRollouts(args.root)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    collate_fn=collate_short, drop_last=False)

    # model + trainer
    model = CoefEnergyNet()
    tcfg = TrainCfg(epochs=args.epochs, bs=args.bs, lr=args.lr, workers=args.workers, gamma_rel=args.gamma_rel, w_friction=args.w_friction, w_multi=args.w_multi)
    trainer = Trainer(model, tcfg)

    os.makedirs(args.outdir, exist_ok=True)
    best = float('inf')

    for ep in range(tcfg.epochs):
        logs_acc = {"loss":0.0,"traj":0.0,"vel":0.0,"alpha":0.0,"friction":0.0,"multi_val":0.0, "pen":0.0}
        n = 0
        for batch in dl:
            logs = trainer.step_batch(batch)
            for k,v in logs.items():
                logs_acc[k] += v
            n += 1
        for k in logs_acc:
            logs_acc[k] /= max(1, n)
        print({f"ep{ep}/{k}": round(v, 6) for k, v in logs_acc.items()})

        # --- checkpointing ---
        ckpt = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.opt.state_dict(),
            'avg_logs': logs_acc,
            'train_cfg': tcfg.__dict__,
        }
        if ep % args.save_every == 0:
            torch.save(ckpt, os.path.join(args.outdir, f"epoch_{ep:03d}.pt"))
        torch.save(ckpt, os.path.join(args.outdir, "latest.pt"))
        if logs_acc.get('traj', 1e9) < best:
            best = logs_acc['traj']
            torch.save(ckpt, os.path.join(args.outdir, "best.pt"))
        

if __name__ == "__main__":
    main()
