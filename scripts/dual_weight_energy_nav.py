"""Dual-weight pH/energy reshaping architecture for stagewise navigation.

Core idea
---------
The network no longer predicts one static potential or one direct action.  It
maintains slow dual weights

    lambda = [goal, barrier, tangent, flow, boundary, memory, stage, damping]

and updates them online from the local environment context.  The fast inner
system then descends the reshaped Hamiltonian/potential and follows the induced
field.  This is intended to handle quasi-static traps by increasing obstacle
barrier weight together with tangential/flow/stage weights instead of merely
slowing down.

This module is intentionally compatible with ``NavMaximinLocalPatches`` from
``energy_data_navmax.py``.  It consumes local barrier stacks, obstacle tokens,
local goal features, and stage indices.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .integrators import EnergyIntegratorConfig, integrate_local_energy_field


LAMBDA_NAMES = (
    "goal",      # scales quadratic goal potential
    "barrier",   # scales fixed ICP/barrier basis
    "tangent",   # scales non-conservative obstacle-boundary sliding field
    "flow",      # scales goal-biased flow field
    "boundary",  # scales stage/window boundary potential
    "memory",    # scales local anti-stall bump
    "stage",     # scales stage transition/forward potential
    "damping",   # scales virtual momentum damping in rollout users
)
LAMBDA_DIM = len(LAMBDA_NAMES)


@dataclass
class DualWeightConfig:
    d_obs: int = 6
    d_goal: int = 4
    d_tok: int = 80
    nhead: int = 4
    transformer_layers: int = 2
    ctx_channels: int = 32
    lambda_step: float = 0.12
    lambda_min: float = 0.0
    lambda_max: float = 6.0
    max_alpha: float = 8.0
    tangent_influence: float = 1.25
    memory_sigma: float = 0.35
    memory_center_x: float = -0.25
    boundary_margin: float = 0.78
    anti_static_force_floor: float = 0.05

    # General short-horizon planner head.  The planner predicts a local
    # feasible motion segment from the same context used by the energy model.
    # During offline training it is supervised by MPC/teacher pose_seq, but
    # online it is used without MPC to create an intermediate tracking target.
    use_plan_goal: bool = False
    plan_horizon: int = 6
    plan_goal_index: int = 3
    plan_radius_frac: float = 0.85

    def to_dict(self) -> Dict[str, float | int | bool]:
        return asdict(self)


class ObstacleEncoder(nn.Module):
    def __init__(self, d_in: int = 6, d_tok: int = 80):
        super().__init__()
        self.d_tok = d_tok
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.SiLU(),
            nn.Linear(128, d_tok), nn.SiLU(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B, N = feats.shape[:2]
        if N == 0:
            return feats.new_zeros(B, 0, self.d_tok)
        return self.net(feats.reshape(B * N, -1)).reshape(B, N, self.d_tok)


class DualWeightEnergyNet(nn.Module):
    """Context encoder + recurrent dual-weight update + energy/field decoder.

    The persistent state is the dual vector ``lambda_prev``.  In training, the
    trainer feeds the previous batch state and resets it on episode boundaries.
    At deployment, ``DualWeightOnlineController`` keeps the same state across
    time steps.
    """

    def __init__(self, H: int, W: int, cfg: Optional[DualWeightConfig] = None):
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.cfg = cfg or DualWeightConfig()
        c = self.cfg

        self.obs_enc = ObstacleEncoder(c.d_obs, c.d_tok)
        self.goal_enc = nn.Sequential(
            nn.Linear(c.d_goal, 128), nn.SiLU(),
            nn.Linear(128, c.d_tok), nn.SiLU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=c.d_tok, nhead=c.nhead, dim_feedforward=2 * c.d_tok,
            batch_first=True, activation="gelu",
        )
        self.fuser = nn.TransformerEncoder(enc_layer, num_layers=c.transformer_layers)
        self.ctx_head = nn.Sequential(nn.Linear(c.d_tok, c.ctx_channels), nn.SiLU())
        self.alpha_head = nn.Sequential(nn.Linear(c.d_tok, 64), nn.SiLU(), nn.Linear(64, 1))

        # Dual update law: Delta lambda = f(context, lambda_prev, local progress/stall features).
        self.init_lambda = nn.Sequential(
            nn.Linear(c.ctx_channels + 8, 96), nn.SiLU(), nn.Linear(96, LAMBDA_DIM), nn.Softplus()
        )
        self.dual_update = nn.Sequential(
            nn.Linear(c.ctx_channels + LAMBDA_DIM + 8, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, LAMBDA_DIM), nn.Tanh(),
        )

        # Short-horizon local planner.  Output is K local waypoints in meters.
        # It is intentionally not an MPC dependency at deployment: MPC/teacher
        # trajectories are used only as offline supervision for this head.
        self.plan_head = nn.Sequential(
            nn.Linear(c.ctx_channels + 8, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, max(1, int(c.plan_horizon)) * 2),
        )

        # Small residual energy.  The residual is deliberately bounded and scaled
        # by the memory/stage weights so it cannot erase the fixed barrier basis.
        self.res_dec = nn.Sequential(
            nn.Conv2d(c.ctx_channels + 4, 48, 3, padding=1), nn.SiLU(),
            nn.Conv2d(48, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 1, 3, padding=1), nn.Tanh(),
        )

    @property
    def lambda_names(self) -> Tuple[str, ...]:
        return LAMBDA_NAMES

    def default_lambda(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Nonzero safe prior: goal and barrier active; tangent/flow can be raised online.
        base = torch.tensor([1.0, 1.0, 0.15, 0.20, 0.20, 0.00, 0.20, 0.30], device=device, dtype=dtype)
        return base.unsqueeze(0).expand(B, -1).clone()

    def encode_context(self, obs_feats: torch.Tensor, obs_mask: torch.Tensor, goal_feats: torch.Tensor):
        B, N = obs_feats.shape[:2]
        z_goal = self.goal_enc(goal_feats).unsqueeze(1)
        z_obs = self.obs_enc(obs_feats)
        if N == 0:
            tokens = z_goal
            pad = torch.zeros(B, 1, dtype=torch.bool, device=goal_feats.device)
        else:
            tokens = torch.cat([z_goal, z_obs], dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=obs_mask.device), ~obs_mask], dim=1)
        z_all = self.fuser(tokens, src_key_padding_mask=pad)
        ctx = self.ctx_head(z_all[:, 0])
        if N == 0:
            alpha = obs_mask.new_zeros(B, 0, dtype=goal_feats.dtype)
        else:
            alpha = F.softplus(self.alpha_head(z_all[:, 1:]).squeeze(-1))
            alpha = self.cfg.max_alpha * torch.tanh(alpha / self.cfg.max_alpha)
            alpha = torch.where(obs_mask, alpha, torch.zeros_like(alpha))
        return ctx, alpha

    def make_aux_features(
        self,
        obs_feats: torch.Tensor,
        obs_mask: torch.Tensor,
        goal_feats: torch.Tensor,
        stage_changed: Optional[torch.Tensor] = None,
        progress: Optional[torch.Tensor] = None,
        stall: Optional[torch.Tensor] = None,
        prev_align: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = goal_feats.shape[0]
        dtype, device = goal_feats.dtype, goal_feats.device
        if obs_feats.shape[1] == 0:
            max_prox = torch.zeros(B, device=device, dtype=dtype)
            mean_prox = torch.zeros(B, device=device, dtype=dtype)
        else:
            prox = obs_feats[..., 5].clamp(0.0, 1.0)
            prox_masked = torch.where(obs_mask, prox, torch.zeros_like(prox))
            max_prox = prox_masked.max(dim=1).values
            denom = obs_mask.float().sum(dim=1).clamp_min(1.0)
            mean_prox = prox_masked.sum(dim=1) / denom
        goal_norm = goal_feats[:, 2] if goal_feats.shape[1] > 2 else torch.linalg.norm(goal_feats[:, :2], dim=-1)
        goal_angle = goal_feats[:, 3] if goal_feats.shape[1] > 3 else torch.atan2(goal_feats[:, 1], goal_feats[:, 0])
        stage_changed = torch.zeros(B, device=device, dtype=dtype) if stage_changed is None else stage_changed.to(device=device, dtype=dtype)
        progress = torch.zeros(B, device=device, dtype=dtype) if progress is None else progress.to(device=device, dtype=dtype)
        stall = torch.zeros(B, device=device, dtype=dtype) if stall is None else stall.to(device=device, dtype=dtype)
        prev_align = torch.zeros(B, device=device, dtype=dtype) if prev_align is None else prev_align.to(device=device, dtype=dtype)
        return torch.stack([max_prox, mean_prox, goal_norm, torch.sin(goal_angle), torch.cos(goal_angle), stage_changed, progress, stall + prev_align], dim=-1)

    def update_lambda(self, ctx: torch.Tensor, lambda_prev: Optional[torch.Tensor], aux: torch.Tensor) -> torch.Tensor:
        B = ctx.shape[0]
        if lambda_prev is None:
            lam = self.default_lambda(B, ctx.device, ctx.dtype) + 0.10 * self.init_lambda(torch.cat([ctx, aux], dim=-1))
        else:
            lam = lambda_prev.to(device=ctx.device, dtype=ctx.dtype)
            if lam.shape[0] != B:
                lam = self.default_lambda(B, ctx.device, ctx.dtype)

        dlam = self.cfg.lambda_step * self.dual_update(torch.cat([ctx, lam, aux], dim=-1))

        # Analytic primal-dual trigger: proximity increases barrier; proximity+stall
        # increases tangent/flow; stage changes increase stage blend.  This is a
        # safety-biased scaffold, while the net learns context-dependent corrections.
        max_prox, mean_prox, _, _, _, stage_changed, _, stall_like = aux.unbind(dim=-1)
        trigger = torch.relu(max_prox - 0.35)
        dlam_analytic = torch.zeros_like(lam)
        name_to_i = {n: i for i, n in enumerate(LAMBDA_NAMES)}
        dlam_analytic[:, name_to_i["barrier"]] += 0.08 * trigger
        dlam_analytic[:, name_to_i["tangent"]] += 0.10 * trigger * (1.0 + stall_like.clamp(0, 1))
        dlam_analytic[:, name_to_i["flow"]] += 0.06 * trigger
        dlam_analytic[:, name_to_i["stage"]] += 0.15 * stage_changed.clamp(0, 1)
        dlam_analytic[:, name_to_i["memory"]] += 0.08 * stall_like.clamp(0, 1)
        dlam_analytic[:, name_to_i["damping"]] -= 0.04 * stall_like.clamp(0, 1)

        lam_next = lam + dlam + dlam_analytic
        return torch.clamp(lam_next, self.cfg.lambda_min, self.cfg.lambda_max)

    def predict_plan(self, ctx: torch.Tensor, aux: torch.Tensor, hat_d_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict a local short-horizon motion plan from context.

        The plan lives in the vehicle-local frame and is bounded by the local
        grid radius.  Training can supervise it with saved teacher pose_seq;
        online deployment uses the predicted plan directly, so no MPC teacher is
        needed at test time.
        """
        B = ctx.shape[0]
        K = max(1, int(self.cfg.plan_horizon))
        raw = self.plan_head(torch.cat([ctx, aux], dim=-1)).view(B, K, 2)
        radius = hat_d_vec.to(ctx.device, ctx.dtype).view(B, 1, 1) * float(self.cfg.plan_radius_frac)
        plan_seq = radius * torch.tanh(raw)
        idx = min(max(0, int(self.cfg.plan_goal_index)), K - 1)
        plan_goal = plan_seq[:, idx, :]
        return {"plan_seq": plan_seq, "plan_goal": plan_goal}

    @staticmethod
    def goal_feats_from_xy(goal_xy: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(goal_xy, dim=-1)
        ang = torch.atan2(goal_xy[:, 1], goal_xy[:, 0])
        return torch.stack([goal_xy[:, 0], goal_xy[:, 1], norm, ang], dim=-1)

    @staticmethod
    def goal_map_from_xy(grid_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        # grid_xy: [B,H,W,2], goal_xy: [B,2] -> [B,1,H,W]
        return 0.5 * ((grid_xy - goal_xy.view(-1, 1, 1, 2)) ** 2).sum(dim=-1, keepdim=False).unsqueeze(1)

    def compose_potential(
        self,
        barrier_stack: torch.Tensor,
        goal_map: torch.Tensor,
        coord_map: torch.Tensor,
        alpha: torch.Tensor,
        lambdas: torch.Tensor,
        ctx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = goal_map.shape
        if barrier_stack.shape[1] == 0:
            U_bar = goal_map.new_zeros(B, 1, H, W)
        else:
            U_bar = torch.einsum("bn,bnhw->bhw", alpha, barrier_stack).unsqueeze(1)

        x = coord_map[:, 0:1]
        y = coord_map[:, 1:2]
        # Soft boundary potential inside local/stage window.
        bnd = torch.relu(torch.abs(x) - self.cfg.boundary_margin) ** 2 + torch.relu(torch.abs(y) - self.cfg.boundary_margin) ** 2
        # Forward/stage potential: descending it gives a persistent +x flow in local frame.
        U_stage = -x
        # Anti-stall memory bump shifted slightly behind the vehicle, so -grad pushes out.
        mem_center_x = goal_map.new_tensor(self.cfg.memory_center_x)
        U_mem = torch.exp(-((x - mem_center_x) ** 2 + y ** 2) / (2.0 * self.cfg.memory_sigma ** 2))

        ctx_map = ctx.view(B, -1, 1, 1).expand(B, -1, H, W)
        res_in = torch.cat([ctx_map, U_bar, goal_map, coord_map], dim=1)
        U_res = self.res_dec(res_in)
        li = {n: i for i, n in enumerate(LAMBDA_NAMES)}
        lam = lambdas
        U = (
            lam[:, li["goal"]].view(B, 1, 1, 1) * goal_map
            + lam[:, li["barrier"]].view(B, 1, 1, 1) * U_bar
            + lam[:, li["boundary"]].view(B, 1, 1, 1) * bnd
            + lam[:, li["stage"]].view(B, 1, 1, 1) * U_stage
            + lam[:, li["memory"]].view(B, 1, 1, 1) * U_mem
            + 0.20 * (1.0 + lam[:, li["memory"]].view(B, 1, 1, 1)) * U_res
        )
        return {
            "U": U,
            "U_barrier": U_bar,
            "U_boundary": bnd,
            "U_stage": U_stage,
            "U_memory": U_mem,
            "U_residual": U_res,
        }

    @staticmethod
    def central_gradients(U: torch.Tensor, dx_vec: torch.Tensor) -> torch.Tensor:
        # Returns [B,2,H,W] gradient in local x/y coordinates.
        B, _, H, W = U.shape
        Ux1 = F.pad(U, (1, 1, 0, 0), mode="replicate")
        Uy1 = F.pad(U, (0, 0, 1, 1), mode="replicate")
        gx = (Ux1[:, :, :, 2:] - Ux1[:, :, :, :-2]) / (2.0 * dx_vec.view(B, 1, 1, 1).clamp_min(1e-8))
        gy = (Uy1[:, :, 2:, :] - Uy1[:, :, :-2, :]) / (2.0 * dx_vec.view(B, 1, 1, 1).clamp_min(1e-8))
        return torch.cat([gx, gy], dim=1)

    def tangent_and_flow_field(
        self,
        grid_xy: torch.Tensor,
        obs_feats: torch.Tensor,
        obs_mask: torch.Tensor,
        goal_feats: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns tangent and flow maps [B,2,H,W].
        B, H, W, _ = grid_xy.shape
        dtype, device = grid_xy.dtype, grid_xy.device
        g = goal_feats[:, :2]
        ghat = g / torch.linalg.norm(g, dim=-1, keepdim=True).clamp_min(1e-8)
        goal_field = ghat.view(B, 2, 1, 1).expand(B, 2, H, W)
        if obs_feats.shape[1] == 0:
            return torch.zeros_like(goal_field), goal_field
        C = obs_feats[..., :2]
        R = obs_feats[..., 2]
        rel = grid_xy[:, None, :, :, :] - C[:, :, None, None, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-8)
        n = rel / dist.unsqueeze(-1)
        t = torch.stack([-n[..., 1], n[..., 0]], dim=-1)
        g_for_dot = ghat[:, None, None, None, :]
        sign = torch.sign((t * g_for_dot).sum(dim=-1, keepdim=True) + 1e-6)
        clear = dist - R[:, :, None, None]
        gate = torch.relu(self.cfg.tangent_influence - clear) / self.cfg.tangent_influence
        w = gate * alpha[:, :, None, None] * obs_mask[:, :, None, None].float()
        tang = (w[..., None] * sign * t).sum(dim=1)  # [B,H,W,2]
        tang = tang.permute(0, 3, 1, 2).contiguous()
        tang = tang / torch.linalg.norm(tang, dim=1, keepdim=True).clamp_min(1.0)
        flow = 0.65 * goal_field + 0.35 * tang
        return tang, flow

    def vector_field(
        self,
        potentials: Dict[str, torch.Tensor],
        grid_xy: torch.Tensor,
        obs_feats: torch.Tensor,
        obs_mask: torch.Tensor,
        goal_feats: torch.Tensor,
        alpha: torch.Tensor,
        lambdas: torch.Tensor,
        hat_d_vec: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = potentials["U"].shape
        dx_vec = (2.0 * hat_d_vec.to(potentials["U"].device, potentials["U"].dtype)) / max(W - 1, 1)
        grad = self.central_gradients(potentials["U"], dx_vec)
        F_cons = -grad
        F_tan, F_flow = self.tangent_and_flow_field(grid_xy, obs_feats, obs_mask, goal_feats, alpha)
        li = {n: i for i, n in enumerate(LAMBDA_NAMES)}
        F = (
            F_cons
            + lambdas[:, li["tangent"]].view(B, 1, 1, 1) * F_tan
            + lambdas[:, li["flow"]].view(B, 1, 1, 1) * F_flow
        )
        return {"F": F, "F_cons": F_cons, "F_tangent": F_tan, "F_flow": F_flow, "gradU": grad}

    def forward(
        self,
        grid_xy: torch.Tensor,
        barrier_stack: torch.Tensor,
        goal_map: torch.Tensor,
        coord_map: torch.Tensor,
        obs_feats: torch.Tensor,
        obs_mask: torch.Tensor,
        goal_feats: torch.Tensor,
        hat_d_vec: torch.Tensor,
        lambda_prev: Optional[torch.Tensor] = None,
        stage_changed: Optional[torch.Tensor] = None,
        progress: Optional[torch.Tensor] = None,
        stall: Optional[torch.Tensor] = None,
        prev_align: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ctx, alpha = self.encode_context(obs_feats, obs_mask, goal_feats)
        aux = self.make_aux_features(obs_feats, obs_mask, goal_feats, stage_changed, progress, stall, prev_align)
        lambdas = self.update_lambda(ctx, lambda_prev, aux)
        plan = self.predict_plan(ctx, aux, hat_d_vec)
        if bool(self.cfg.use_plan_goal):
            effective_goal_map = self.goal_map_from_xy(grid_xy, plan["plan_goal"])
            effective_goal_feats = self.goal_feats_from_xy(plan["plan_goal"])
        else:
            effective_goal_map = goal_map
            effective_goal_feats = goal_feats
        potentials = self.compose_potential(barrier_stack, effective_goal_map, coord_map, alpha, lambdas, ctx)
        fields = self.vector_field(potentials, grid_xy, obs_feats, obs_mask, effective_goal_feats, alpha, lambdas, hat_d_vec)
        return {**potentials, **fields, **plan, "goal_map_effective": effective_goal_map, "goal_feats_effective": effective_goal_feats, "alpha": alpha, "lambda": lambdas, "ctx": ctx, "aux": aux}


# ----------------------------- losses and utilities -----------------------------

def sample_map_at_pos(map_bchw: torch.Tensor, pos_xy: torch.Tensor, hat_d_vec: torch.Tensor) -> torch.Tensor:
    """Bilinear sample a scalar/vector map at local positions.

    map_bchw: [B,C,H,W], pos_xy: [B,2], hat_d_vec: [B].
    Returns [B,C].
    """
    B, C, H, W = map_bchw.shape
    x = (pos_xy[:, 0] / hat_d_vec).clamp(-1.0, 1.0)
    y = (pos_xy[:, 1] / hat_d_vec).clamp(-1.0, 1.0)
    grid = torch.stack([x, y], dim=-1).view(B, 1, 1, 2)
    return F.grid_sample(map_bchw, grid, mode="bilinear", align_corners=True).view(B, C)


def cosine_alignment_loss(vec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    v = vec / torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-8)
    t = target / torch.linalg.norm(target, dim=-1, keepdim=True).clamp_min(1e-8)
    return (1.0 - (v * t).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def laplacian_smoothness(U: torch.Tensor) -> torch.Tensor:
    up = F.pad(U, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
    dn = F.pad(U, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
    lf = F.pad(U, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
    rt = F.pad(U, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    return ((up + dn + lf + rt - 4.0 * U) ** 2).mean()


def lambda_trigger_targets(aux: torch.Tensor) -> torch.Tensor:
    """Heuristic target lambdas for dual regularization, not hard labels."""
    max_prox, mean_prox, goal_norm, _, _, stage_changed, _, stall_like = aux.unbind(dim=-1)
    B = aux.shape[0]
    tgt = aux.new_zeros(B, LAMBDA_DIM)
    li = {n: i for i, n in enumerate(LAMBDA_NAMES)}
    tgt[:, li["goal"]] = 1.0 + 0.2 * torch.tanh(goal_norm)
    tgt[:, li["barrier"]] = 0.5 + 2.5 * max_prox
    tgt[:, li["tangent"]] = 0.2 + 3.0 * torch.relu(max_prox - 0.25) + 1.5 * stall_like.clamp(0, 1)
    tgt[:, li["flow"]] = 0.3 + 1.5 * torch.relu(mean_prox - 0.15)
    tgt[:, li["boundary"]] = 0.2
    tgt[:, li["memory"]] = 1.5 * stall_like.clamp(0, 1)
    tgt[:, li["stage"]] = 0.3 + 2.0 * stage_changed.clamp(0, 1)
    tgt[:, li["damping"]] = 0.3 * (1.0 - stall_like.clamp(0, 1))
    return tgt


def dual_energy_losses(
    out: Dict[str, torch.Tensor],
    pos_xy: torch.Tensor,
    dir_xy: torch.Tensor,
    hat_d_vec: torch.Tensor,
    lambda_prev: Optional[torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    weights = weights or {}
    F_at = sample_map_at_pos(out["F"], pos_xy, hat_d_vec)
    U = out["U"]
    lam = out["lambda"]
    aux = out["aux"]
    L_align = cosine_alignment_loss(F_at, dir_xy)
    L_smooth = laplacian_smoothness(U)
    L_alpha = out["alpha"].mean() if out["alpha"].numel() else U.new_zeros(())
    L_lambda = lam.mean()
    target_lam = lambda_trigger_targets(aux).to(lam.device, lam.dtype)
    L_dual_target = F.mse_loss(lam, target_lam)
    if lambda_prev is None or lambda_prev.shape != lam.shape:
        L_dual_tv = lam.new_zeros(())
    else:
        L_dual_tv = ((lam - lambda_prev.to(lam.device, lam.dtype)) ** 2).mean()
    force_norm = torch.linalg.norm(F_at, dim=-1)
    L_antistatic = (torch.relu(weights.get("force_floor", 0.05) - force_norm) ** 2).mean()
    loss = (
        weights.get("align", 1.0) * L_align
        + weights.get("smooth", 1e-3) * L_smooth
        + weights.get("alpha", 1e-4) * L_alpha
        + weights.get("lambda", 1e-4) * L_lambda
        + weights.get("dual_target", 0.05) * L_dual_target
        + weights.get("dual_tv", 0.01) * L_dual_tv
        + weights.get("anti_static", 0.1) * L_antistatic
    )
    return {
        "loss": loss,
        "align": L_align,
        "smooth": L_smooth,
        "alpha": L_alpha,
        "lambda": L_lambda,
        "dual_target": L_dual_target,
        "dual_tv": L_dual_tv,
        "anti_static": L_antistatic,
        "force_norm": force_norm.mean(),
    }


@dataclass
class AckermannLimitConfig:
    v_min: float = -0.25
    v_max: float = 0.80
    steering_max: float = 0.396
    accel_max: float = 1.5
    steering_rate_max: float = 1.5
    dt: float = 0.10


def project_ackermann_box_rate(u_raw: torch.Tensor, u_prev: Optional[torch.Tensor], cfg: AckermannLimitConfig) -> torch.Tensor:
    """Hard projection for speed/steering box and rate limits.

    u_raw is [B,2] = (v, steering_angle).  This should be used as the final
    deployment shield.  Differentiable soft barriers can be added in training,
    but this projection is the actual hard feasibility layer.
    """
    if u_prev is None:
        v_lo = u_raw.new_full((u_raw.shape[0],), cfg.v_min)
        v_hi = u_raw.new_full((u_raw.shape[0],), cfg.v_max)
        d_lo = u_raw.new_full((u_raw.shape[0],), -cfg.steering_max)
        d_hi = u_raw.new_full((u_raw.shape[0],), cfg.steering_max)
    else:
        up = u_prev.to(u_raw.device, u_raw.dtype)
        v_lo = torch.clamp(up[:, 0] - cfg.accel_max * cfg.dt, min=cfg.v_min)
        v_hi = torch.clamp(up[:, 0] + cfg.accel_max * cfg.dt, max=cfg.v_max)
        d_lo = torch.clamp(up[:, 1] - cfg.steering_rate_max * cfg.dt, min=-cfg.steering_max)
        d_hi = torch.clamp(up[:, 1] + cfg.steering_rate_max * cfg.dt, max=cfg.steering_max)
    v = torch.minimum(torch.maximum(u_raw[:, 0], v_lo), v_hi)
    delta = torch.minimum(torch.maximum(u_raw[:, 1], d_lo), d_hi)
    return torch.stack([v, delta], dim=-1)


def input_barrier_loss(u_raw: torch.Tensor, cfg: AckermannLimitConfig, eps: float = 1e-4) -> torch.Tensor:
    """Soft interior barrier for training; does not replace hard projection."""
    v, d = u_raw[:, 0], u_raw[:, 1]
    # Softplus penalties outside a slightly shrunken safe box are numerically safer than log barriers.
    return (
        F.softplus(v - (cfg.v_max - eps)).mean()
        + F.softplus((cfg.v_min + eps) - v).mean()
        + F.softplus(torch.abs(d) - (cfg.steering_max - eps)).mean()
    )



@dataclass
class SensitivityUpdateConfig:
    """Online projected-gradient update for dual weights.

    The model parameters are frozen; only the controller's lambda_state is
    corrected using a short-horizon differentiable rollout on the current local
    energy field.
    """
    enabled: bool = True
    eta: float = 0.08
    horizon: int = 8
    dt: float = 0.10
    grad_clip: float = 1.0
    w_goal: float = 1.0
    w_path: float = 0.15
    w_barrier: float = 0.25
    w_align: float = 0.25
    w_act: float = 0.05
    w_stall: float = 0.25
    force_floor: float = 0.06
    integrator: str = "semi_implicit"
    damping: float = 0.20
    momentum_clip: float = 1.50
    signed_speed: bool = False
    v_nominal: float = 0.55
    steering_gain: float = 1.4


def recompute_with_lambda(
    model: DualWeightEnergyNet,
    grid_xy: torch.Tensor,
    barrier_stack: torch.Tensor,
    goal_map: torch.Tensor,
    coord_map: torch.Tensor,
    obs_feats: torch.Tensor,
    obs_mask: torch.Tensor,
    goal_feats: torch.Tensor,
    hat_d_vec: torch.Tensor,
    lambdas: torch.Tensor,
    ctx: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    aux: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Recompose U and F for a prescribed dual vector lambda.

    This is the key hook for sensitivity updates: the learned context encoder is
    evaluated once, then lambda is treated as the optimization variable.
    """
    if ctx is None or alpha is None:
        ctx, alpha = model.encode_context(obs_feats, obs_mask, goal_feats)
    if aux is None:
        aux = model.make_aux_features(obs_feats, obs_mask, goal_feats)
    plan = model.predict_plan(ctx, aux, hat_d_vec)
    if bool(model.cfg.use_plan_goal):
        effective_goal_map = model.goal_map_from_xy(grid_xy, plan["plan_goal"])
        effective_goal_feats = model.goal_feats_from_xy(plan["plan_goal"])
    else:
        effective_goal_map = goal_map
        effective_goal_feats = goal_feats
    potentials = model.compose_potential(barrier_stack, effective_goal_map, coord_map, alpha, lambdas, ctx)
    fields = model.vector_field(potentials, grid_xy, obs_feats, obs_mask, effective_goal_feats, alpha, lambdas, hat_d_vec)
    return {**potentials, **fields, **plan, "goal_map_effective": effective_goal_map, "goal_feats_effective": effective_goal_feats, "alpha": alpha, "lambda": lambdas, "ctx": ctx, "aux": aux}


def rollout_field_loss_for_lambda(
    model: DualWeightEnergyNet,
    out: Dict[str, torch.Tensor],
    batch_t: Dict[str, torch.Tensor],
    u_prev: Optional[torch.Tensor],
    limit_cfg: AckermannLimitConfig,
    sens_cfg: SensitivityUpdateConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Short-horizon local feedback objective used for ∂J/∂lambda.

    The previous implementation used explicit Euler on q.  This version routes
    the rollout through a pluggable pH-style integrator.  The default is
    semi-implicit Euler: p_{k+1}=(p_k+dt F(q_k))/(1+dt R), q_{k+1}=q_k+dt p_{k+1}.
    """
    hat_d = batch_t["hat_d_vec"]
    pos = batch_t["pos_xy"]
    goal = out.get("plan_goal", batch_t["goal_feats"][:, :2])
    B = pos.shape[0]
    barrier_map = batch_t["barrier_stack"].amax(dim=1, keepdim=True) if batch_t["barrier_stack"].shape[1] else out["U"].new_zeros(B, 1, out["U"].shape[-2], out["U"].shape[-1])

    if u_prev is not None:
        up = u_prev.to(pos.device, pos.dtype)
        p0 = torch.stack([up[:, 0], torch.zeros_like(up[:, 0])], dim=-1)
    else:
        p0 = pos.new_zeros(B, 2)

    integ_cfg = EnergyIntegratorConfig(
        scheme=sens_cfg.integrator,
        dt=sens_cfg.dt,
        damping=sens_cfg.damping,
        momentum_clip=sens_cfg.momentum_clip,
        clamp_to_grid=True,
    )
    force_fn = lambda q: sample_map_at_pos(out["F"], q, hat_d)
    roll = integrate_local_energy_field(pos, force_fn, hat_d, integ_cfg, int(sens_cfg.horizon), p0=p0)
    q_seq, p_seq, F_seq = roll["q_seq"], roll["p_seq"], roll["F_seq"]

    if q_seq.shape[1] == 0:
        z = pos.new_zeros(())
        diag = {
            "sens_loss": z.detach(), "sens_terminal": z.detach(), "sens_barrier": z.detach(),
            "sens_act_defect": z.detach(), "q_seq": q_seq.detach(), "p_seq": p_seq.detach(),
            "u_raw_seq": pos.new_zeros(B, 0, 2), "u_proj_seq": pos.new_zeros(B, 0, 2), "F_seq": F_seq.detach(),
        }
        return z, diag

    Bq_seq = []
    u_raw_seq = []
    u_proj_seq = []
    act_defects = []
    align_losses = []
    stall_losses = []
    path_losses = []
    u_last = u_prev
    for k in range(q_seq.shape[1]):
        qk = q_seq[:, k, :]
        Fk = F_seq[:, k, :]
        Bq = sample_map_at_pos(barrier_map, qk, hat_d)[:, 0]
        u_raw = DualWeightOnlineController.force_to_ackermann(
            Fk, v_nominal=sens_cfg.v_nominal, steering_gain=sens_cfg.steering_gain, signed_speed=sens_cfg.signed_speed
        )
        u_proj = project_ackermann_box_rate(u_raw, u_last if k == 0 else u_proj_seq[-1], limit_cfg)
        goal_vec = goal - qk
        align_loss = 1.0 - (
            (Fk / torch.linalg.norm(Fk, dim=-1, keepdim=True).clamp_min(1e-8))
            * (goal_vec / torch.linalg.norm(goal_vec, dim=-1, keepdim=True).clamp_min(1e-8))
        ).sum(dim=-1).clamp(-1.0, 1.0)
        act_defect = ((u_raw - u_proj) ** 2).sum(dim=-1)
        stall = torch.relu(Fk.new_tensor(sens_cfg.force_floor) - torch.linalg.norm(Fk, dim=-1)) ** 2
        path = (goal_vec ** 2).sum(dim=-1)
        Bq_seq.append(Bq)
        u_raw_seq.append(u_raw)
        u_proj_seq.append(u_proj)
        act_defects.append(act_defect)
        align_losses.append(align_loss)
        stall_losses.append(stall)
        path_losses.append(path)

    terminal = sens_cfg.w_goal * ((q_seq[:, -1, :] - goal) ** 2).sum(dim=-1)
    path_term = sens_cfg.w_path * torch.stack(path_losses, dim=1).mean(dim=1)
    barrier_term = sens_cfg.w_barrier * torch.stack(Bq_seq, dim=1).mean(dim=1)
    align_term = sens_cfg.w_align * torch.stack(align_losses, dim=1).mean(dim=1)
    act_term = sens_cfg.w_act * torch.stack(act_defects, dim=1).mean(dim=1)
    stall_term = sens_cfg.w_stall * torch.stack(stall_losses, dim=1).mean(dim=1)
    J = (terminal + path_term + barrier_term + align_term + act_term + stall_term).mean()
    diag = {
        "sens_loss": J.detach(),
        "sens_terminal": terminal.mean().detach(),
        "sens_barrier": torch.stack(Bq_seq, dim=1).mean().detach(),
        "sens_act_defect": torch.stack(act_defects, dim=1).mean().detach(),
        "q_seq": q_seq.detach(),
        "p_seq": p_seq.detach(),
        "u_raw_seq": torch.stack(u_raw_seq, dim=1).detach(),
        "u_proj_seq": torch.stack(u_proj_seq, dim=1).detach(),
        "F_seq": F_seq.detach(),
    }
    return J, diag


def sensitivity_update_lambda(
    model: DualWeightEnergyNet,
    batch_t: Dict[str, torch.Tensor],
    lambda_init: torch.Tensor,
    ctx: torch.Tensor,
    alpha: torch.Tensor,
    aux: torch.Tensor,
    u_prev: Optional[torch.Tensor],
    limit_cfg: AckermannLimitConfig,
    sens_cfg: SensitivityUpdateConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute lambda <- Proj(lambda - eta ∂J/∂lambda), then recompose U,F."""
    lam_var = lambda_init.detach().clone().requires_grad_(True)
    out_var = recompute_with_lambda(
        model,
        batch_t["grid_xy"], batch_t["barrier_stack"], batch_t["goal_map"], batch_t["coord_map"],
        batch_t["obs_feats"], batch_t["obs_mask"], batch_t["goal_feats"], batch_t["hat_d_vec"],
        lam_var, ctx=ctx.detach(), alpha=alpha.detach(), aux=aux.detach(),
    )
    J, diag = rollout_field_loss_for_lambda(model, out_var, batch_t, u_prev, limit_cfg, sens_cfg)
    grad = torch.autograd.grad(J, lam_var, retain_graph=False, create_graph=False)[0]
    grad_clipped = torch.clamp(grad, -sens_cfg.grad_clip, sens_cfg.grad_clip)
    lam_next = lam_var - sens_cfg.eta * grad_clipped
    lam_next = torch.clamp(lam_next, model.cfg.lambda_min, model.cfg.lambda_max).detach()
    out_next = recompute_with_lambda(
        model,
        batch_t["grid_xy"], batch_t["barrier_stack"], batch_t["goal_map"], batch_t["coord_map"],
        batch_t["obs_feats"], batch_t["obs_mask"], batch_t["goal_feats"], batch_t["hat_d_vec"],
        lam_next, ctx=ctx.detach(), alpha=alpha.detach(), aux=aux.detach(),
    )
    diag.update({
        "sens_grad": grad.detach(),
        "sens_grad_norm": torch.linalg.norm(grad.detach(), dim=-1),
        "lambda_before_sens": lambda_init.detach(),
        "lambda_after_sens": lam_next.detach(),
    })
    return lam_next, out_next, diag


class DualWeightOnlineController:
    """Stateful online wrapper for deployment/evaluation.

    adaptation modes:
      - ``fixed``: keep a fixed/default lambda and only recompose U,F.
      - ``forward``: learned + analytic recurrent dual update.
      - ``sensitivity``: forward update, then projected gradient correction
        lambda <- lambda - eta ∂J/∂lambda from a short local rollout.
    """
    def __init__(
        self,
        model: DualWeightEnergyNet,
        limit_cfg: Optional[AckermannLimitConfig] = None,
        sensitivity_cfg: Optional[SensitivityUpdateConfig] = None,
        signed_speed: bool = False,
        v_nominal: float = 0.55,
        steering_gain: float = 1.4,
    ):
        self.model = model
        self.limit_cfg = limit_cfg or AckermannLimitConfig()
        self.sensitivity_cfg = sensitivity_cfg or SensitivityUpdateConfig()
        self.signed_speed = bool(signed_speed)
        self.v_nominal = float(v_nominal)
        self.steering_gain = float(steering_gain)
        self.lambda_state: Optional[torch.Tensor] = None
        self.u_prev: Optional[torch.Tensor] = None
        self.prev_goal_norm: Optional[torch.Tensor] = None
        self.prev_align: Optional[torch.Tensor] = None

    def reset(self):
        self.lambda_state = None
        self.u_prev = None
        self.prev_goal_norm = None
        self.prev_align = None

    @staticmethod
    def force_to_ackermann(
        F0: torch.Tensor,
        v_nominal: float = 0.55,
        steering_gain: float = 1.4,
        signed_speed: bool = False,
    ) -> torch.Tensor:
        fx, fy = F0[:, 0], F0[:, 1]
        if signed_speed:
            speed = v_nominal * torch.tanh(fx)
            heading_err = torch.atan2(fy, torch.abs(fx).clamp_min(1e-6))
        else:
            heading_err = torch.atan2(fy, fx)
            speed = v_nominal * torch.tanh(torch.linalg.norm(F0, dim=-1))
        delta = steering_gain * heading_err
        return torch.stack([speed, delta], dim=-1)

    def _prepare_tensors(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dev = next(self.model.parameters()).device
        return {
            "grid_xy": batch["grid_xy"].to(dev),
            "barrier_stack": batch["barrier_stack"].to(dev),
            "goal_map": batch["goal_map"].to(dev),
            "coord_map": batch["coord_map"].to(dev),
            "obs_feats": batch["obs_feats"].to(dev),
            "obs_mask": batch["obs_mask"].to(dev),
            "goal_feats": batch["goal_feats"].to(dev),
            "hat_d_vec": batch["meta"]["hat_d"].to(dev),
            "pos_xy": batch["pos_xy"].to(dev),
            "dir_xy": batch["dir_xy"].to(dev),
        }

    def _progress_stall(self, goal_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_norm = goal_feats[:, 2] if goal_feats.shape[1] > 2 else torch.linalg.norm(goal_feats[:, :2], dim=-1)
        progress = goal_norm.new_zeros(goal_norm.shape) if self.prev_goal_norm is None else (self.prev_goal_norm.to(goal_norm.device) - goal_norm)
        stall = (progress < 1e-3).to(goal_feats.dtype)
        return goal_norm, progress, stall

    def step(self, batch: Dict[str, torch.Tensor], stage_changed: Optional[torch.Tensor] = None,
             adaptation: str = "forward", sensitivity_cfg: Optional[SensitivityUpdateConfig] = None) -> Dict[str, torch.Tensor]:
        adaptation = adaptation.lower()
        if adaptation not in {"fixed", "forward", "sensitivity"}:
            raise ValueError(f"unknown adaptation mode {adaptation!r}; expected fixed, forward, or sensitivity")
        sens_cfg = sensitivity_cfg or self.sensitivity_cfg
        data = self._prepare_tensors(batch)
        goal_norm, progress, stall = self._progress_stall(data["goal_feats"])
        diag: Dict[str, torch.Tensor] = {}

        if adaptation == "fixed":
            with torch.no_grad():
                ctx, alpha = self.model.encode_context(data["obs_feats"], data["obs_mask"], data["goal_feats"])
                aux = self.model.make_aux_features(data["obs_feats"], data["obs_mask"], data["goal_feats"], stage_changed, progress, stall, self.prev_align)
                if self.lambda_state is None or self.lambda_state.shape[0] != data["goal_feats"].shape[0]:
                    self.lambda_state = self.model.default_lambda(data["goal_feats"].shape[0], data["goal_feats"].device, data["goal_feats"].dtype)
                out = recompute_with_lambda(
                    self.model, data["grid_xy"], data["barrier_stack"], data["goal_map"], data["coord_map"],
                    data["obs_feats"], data["obs_mask"], data["goal_feats"], data["hat_d_vec"],
                    self.lambda_state, ctx=ctx, alpha=alpha, aux=aux,
                )
        elif adaptation == "forward":
            with torch.no_grad():
                out = self.model(
                    grid_xy=data["grid_xy"], barrier_stack=data["barrier_stack"], goal_map=data["goal_map"], coord_map=data["coord_map"],
                    obs_feats=data["obs_feats"], obs_mask=data["obs_mask"], goal_feats=data["goal_feats"], hat_d_vec=data["hat_d_vec"],
                    lambda_prev=self.lambda_state, stage_changed=stage_changed, progress=progress, stall=stall, prev_align=self.prev_align,
                )
        else:
            # First apply the learned/analytic forward update without building a
            # graph through model parameters, then correct only lambda by sensitivity.
            with torch.no_grad():
                out_fwd = self.model(
                    grid_xy=data["grid_xy"], barrier_stack=data["barrier_stack"], goal_map=data["goal_map"], coord_map=data["coord_map"],
                    obs_feats=data["obs_feats"], obs_mask=data["obs_mask"], goal_feats=data["goal_feats"], hat_d_vec=data["hat_d_vec"],
                    lambda_prev=self.lambda_state, stage_changed=stage_changed, progress=progress, stall=stall, prev_align=self.prev_align,
                )
                ctx = out_fwd["ctx"].detach()
                alpha = out_fwd["alpha"].detach()
                aux = out_fwd["aux"].detach()
                lam_init = out_fwd["lambda"].detach()
            lam_next, out, diag = sensitivity_update_lambda(
                self.model, data, lam_init, ctx, alpha, aux, self.u_prev, self.limit_cfg, sens_cfg
            )
            self.lambda_state = lam_next.detach()

        F0 = sample_map_at_pos(out["F"], data["pos_xy"], data["hat_d_vec"])
        u_raw = self.force_to_ackermann(
            F0, v_nominal=self.v_nominal, steering_gain=self.steering_gain, signed_speed=self.signed_speed
        )
        u = project_ackermann_box_rate(u_raw, self.u_prev, self.limit_cfg)
        if adaptation != "sensitivity":
            self.lambda_state = out["lambda"].detach()
        self.u_prev = u.detach()
        self.prev_goal_norm = goal_norm.detach()
        d = data["dir_xy"]
        v = F0 / torch.linalg.norm(F0, dim=-1, keepdim=True).clamp_min(1e-8)
        dhat = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-8)
        self.prev_align = (1.0 - (v * dhat).sum(dim=-1).clamp(-1, 1)).detach()
        ret = {
            "F0": F0.detach(), "u_raw": u_raw.detach(), "u": u.detach(),
            "lambda": self.lambda_state.detach(), "U": out["U"].detach(),
            "F": out["F"].detach(), "alpha": out["alpha"].detach(),
            "plan_seq": out.get("plan_seq", torch.empty(0, device=F0.device)).detach(),
            "plan_goal": out.get("plan_goal", torch.empty(0, device=F0.device)).detach(),
            "adaptation": adaptation,
        }
        for k, val in diag.items():
            ret[k] = val.detach() if torch.is_tensor(val) else val
        return ret
