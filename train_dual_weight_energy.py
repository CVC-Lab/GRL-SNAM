#!/usr/bin/env python3
"""Train the dual-weight energy reshaping model on NavMaximinLocalPatches."""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from scripts.energy_data_navmax import NavMaximinLocalPatches, collate_navmax
    from scripts.dual_weight_energy_nav import (
        AckermannLimitConfig, DualWeightConfig, DualWeightEnergyNet, DualWeightOnlineController,
        dual_energy_losses, lambda_trigger_targets, laplacian_smoothness, recompute_with_lambda,
        sample_map_at_pos, project_ackermann_box_rate, cosine_alignment_loss, LAMBDA_DIM,
    )
    from scripts.integrators import EnergyIntegratorConfig, integrate_local_energy_field
    from scripts.failure_replay import FailureReplayBuffer
    from scripts.parking_aug import sample_parallel_parking_batch
    from scripts.stage1_energy_adapter import Stage1EnergyPatchDataset, collate_stage1_energy, is_stage1_dataset_path
except Exception:  # pragma: no cover
    from energy_data_navmax import NavMaximinLocalPatches, collate_navmax
    from dual_weight_energy_nav import (
        AckermannLimitConfig, DualWeightConfig, DualWeightEnergyNet, DualWeightOnlineController,
        dual_energy_losses, lambda_trigger_targets, laplacian_smoothness, recompute_with_lambda,
        sample_map_at_pos, project_ackermann_box_rate, cosine_alignment_loss, LAMBDA_DIM,
    )
    from scripts.integrators import EnergyIntegratorConfig, integrate_local_energy_field
    from scripts.failure_replay import FailureReplayBuffer
    from scripts.parking_aug import sample_parallel_parking_batch
    from scripts.stage1_energy_adapter import Stage1EnergyPatchDataset, collate_stage1_energy, is_stage1_dataset_path


def force_to_ackermann_train(
    F0: torch.Tensor,
    v_nominal: float = 0.55,
    steering_gain: float = 1.4,
    signed_speed: bool = False,
) -> torch.Tensor:
    """Map local force to an Ackermann command used during training.

    Default behavior preserves the original nonnegative-speed map.  When
    ``signed_speed`` is enabled, the speed is determined by the force component
    along the vehicle x-axis, so a backward-pointing energy gradient can produce
    reverse commands.  Steering is computed against ``abs(fx)`` in signed mode so
    reverse arcs do not collapse to a saturated pi-heading command.
    """
    fx, fy = F0[:, 0], F0[:, 1]
    if signed_speed:
        speed = v_nominal * torch.tanh(fx)
        heading_err = torch.atan2(fy, torch.abs(fx).clamp_min(1e-6))
    else:
        heading_err = torch.atan2(fy, fx)
        speed = v_nominal * torch.tanh(torch.linalg.norm(F0, dim=-1))
    delta = steering_gain * heading_err
    return torch.stack([speed, delta], dim=-1)


def per_sample_cosine_error(vec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    v = vec / torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-8)
    t = target / torch.linalg.norm(target, dim=-1, keepdim=True).clamp_min(1e-8)
    return 1.0 - (v * t).sum(dim=-1).clamp(-1.0, 1.0)


class DualWeightTrainer:
    """Trainer with two modes.

    ``weighted`` preserves the original fixed weighted loss.
    ``auglag`` uses a task objective plus adaptive augmented-Lagrangian
    constraints and optional lambda-space teacher distillation.
    """
    def __init__(
        self,
        model: DualWeightEnergyNet,
        device: str,
        lr: float,
        loss_weights: Dict[str, float],
        training_mode: str = "weighted",
        constraint_cfg: Optional[Dict[str, float]] = None,
        teacher_cfg: Optional[Dict[str, float]] = None,
        limit_cfg: Optional[AckermannLimitConfig] = None,
        replay_cfg: Optional[Dict[str, float]] = None,
        parking_cfg: Optional[Dict[str, float]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_weights = loss_weights
        self.training_mode = training_mode
        self.constraint_cfg = constraint_cfg or {}
        self.teacher_cfg = teacher_cfg or {}
        self.limit_cfg = limit_cfg or AckermannLimitConfig()
        self.replay_cfg = replay_cfg or {}
        self.parking_cfg = parking_cfg or {}
        self.replay = FailureReplayBuffer(
            capacity=int(self.replay_cfg.get("capacity", 0)),
            seed=int(self.replay_cfg.get("seed", 0)),
        )
        self.rng = random.Random(int(self.replay_cfg.get("seed", 0)))
        self.lambda_state: Optional[torch.Tensor] = None
        self.prev_episode: Optional[torch.Tensor] = None
        self.prev_stage: Optional[torch.Tensor] = None
        self.mu = {
            "clear": 0.0,
            "act": 0.0,
            "stall": 0.0,
            "tv": 0.0,
            "smooth": 0.0,
        }

    def _state_features(self, meta: Dict[str, torch.Tensor], B: int):
        episode = meta["episode"].to(self.device)
        stage = meta["stage"].to(self.device)
        if self.lambda_state is None or self.lambda_state.shape[0] != B:
            self.lambda_state = None
            self.prev_episode = episode
            self.prev_stage = stage
            stage_changed = torch.zeros(B, device=self.device)
            return self.lambda_state, stage_changed
        new_episode = episode != self.prev_episode
        stage_changed_bool = stage != self.prev_stage
        if new_episode.any():
            self.lambda_state[new_episode] = self.model.default_lambda(int(new_episode.sum()), self.device, self.lambda_state.dtype)
        self.prev_episode = episode
        self.prev_stage = stage
        return self.lambda_state, stage_changed_bool.float()

    def _data_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data = {
            "grid_xy": batch["grid_xy"].to(self.device),
            "barrier_stack": batch["barrier_stack"].to(self.device),
            "goal_map": batch["goal_map"].to(self.device),
            "coord_map": batch["coord_map"].to(self.device),
            "obs_feats": batch["obs_feats"].to(self.device),
            "obs_mask": batch["obs_mask"].to(self.device),
            "goal_feats": batch["goal_feats"].to(self.device),
            "hat_d_vec": batch["meta"]["hat_d"].to(self.device),
            "pos_xy": batch["pos_xy"].to(self.device),
            "dir_xy": batch["dir_xy"].to(self.device),
        }
        for key in [
            "cmd", "cmd_seq", "cmd_seq_mask", "pose_seq_xy", "pose_seq_mask",
            "waypoint_xy", "mode", "true_clearance",
        ]:
            if key in batch:
                data[key] = batch[key].to(self.device)
        return data

    def _barrier_exposure(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Diagnostic barrier exposure at the current local position.

        In NavMaximinLocalPatches, ``pos_xy`` is the local origin.  This metric
        is useful for logging dataset difficulty, but it should not be used as
        the augmented-Lagrangian clearance constraint because it is independent
        of the learned energy field.
        """
        B = data["pos_xy"].shape[0]
        if data["barrier_stack"].shape[1] == 0:
            return data["pos_xy"].new_zeros(())
        barrier_map = data["barrier_stack"].amax(dim=1, keepdim=True)
        return sample_map_at_pos(barrier_map, data["pos_xy"], data["hat_d_vec"])[:, 0].mean()

    def _rollout_barrier_exposure_vector(self, data: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample model-dependent barrier exposure along a predicted local rollout."""
        B = data["pos_xy"].shape[0]
        if data["barrier_stack"].shape[1] == 0:
            return out["U"].new_zeros(B)
        steps = max(1, int(self.constraint_cfg.get("clear_rollout_steps", 1)))
        barrier_map = data["barrier_stack"].amax(dim=1, keepdim=True)
        integ_cfg = EnergyIntegratorConfig(
            scheme=str(self.constraint_cfg.get("integrator", self.teacher_cfg.get("integrator", "semi_implicit"))),
            dt=float(self.constraint_cfg.get("dt", self.teacher_cfg.get("dt", 0.10))),
            damping=float(self.constraint_cfg.get("integrator_damping", self.teacher_cfg.get("integrator_damping", 0.20))),
            momentum_clip=float(self.constraint_cfg.get("momentum_clip", self.teacher_cfg.get("momentum_clip", 1.50))),
            clamp_to_grid=True,
        )
        roll = integrate_local_energy_field(
            data["pos_xy"],
            lambda q: sample_map_at_pos(out["F"], q, data["hat_d_vec"]),
            data["hat_d_vec"],
            integ_cfg,
            steps=steps,
            p0=None,
        )
        q_seq = roll["q_seq"]
        if q_seq.shape[1] == 0:
            return out["U"].new_zeros(B)
        flat_q = q_seq.reshape(B * q_seq.shape[1], 2)
        flat_h = data["hat_d_vec"].repeat_interleave(q_seq.shape[1])
        flat_barrier = barrier_map.repeat_interleave(q_seq.shape[1], dim=0)
        b_seq = sample_map_at_pos(flat_barrier, flat_q, flat_h)[:, 0].view(B, q_seq.shape[1])
        return b_seq.max(dim=1).values

    def _rollout_barrier_exposure(self, data: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Raw differentiable barrier-field exposure along a predicted local rollout.

        This remains useful as a diagnostic, but it is not a calibrated safety
        constraint for Stage-1 parking scenes: tight parking slots naturally have
        large local barrier values even when the commanded motion is feasible.
        """
        return self._rollout_barrier_exposure_vector(data, out).mean()

    def _clearance_mode(self) -> str:
        mode = str(self.constraint_cfg.get("clearance_mode", "auto")).lower()
        if mode == "auto":
            return "signed" if str(self.constraint_cfg.get("data_format", "navmax")) == "stage1" else "barrier"
        return mode

    def _signed_clearance_violation_at(self, data: Dict[str, torch.Tensor], q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample signed-clearance violation and minimum clearance.

        ``q`` is local position with shape [B,2] or [B,T,2].  Obstacle centers
        and radii come from ``obs_feats``; for Stage-1 these radii are already
        expanded by the robot radius in the adapter.  The output cost is bounded
        and dimensionless, unlike the raw rasterized barrier energy.
        """
        obs_feats = data.get("obs_feats")
        obs_mask = data.get("obs_mask")
        B = q.shape[0]
        if obs_feats is None or obs_feats.shape[1] == 0:
            return q.new_zeros(B), q.new_full((B,), float("inf"))
        if q.dim() == 2:
            q_eval = q[:, None, :]
        elif q.dim() == 3:
            q_eval = q
        else:
            raise ValueError(f"q must have shape [B,2] or [B,T,2], got {tuple(q.shape)}")
        centers = obs_feats[:, :, :2].to(q_eval.device, q_eval.dtype)
        radii = obs_feats[:, :, 2].to(q_eval.device, q_eval.dtype).clamp_min(0.0)
        diff = q_eval[:, :, None, :] - centers[:, None, :, :]
        clear = torch.linalg.norm(diff, dim=-1) - radii[:, None, :]
        if obs_mask is not None:
            mask = obs_mask.to(q_eval.device).bool()[:, None, :]
            clear = torch.where(mask, clear, clear.new_full(clear.shape, float("inf")))
        min_clear = clear.amin(dim=-1).amin(dim=-1)
        safe_margin = float(self.constraint_cfg.get("clear_safe_margin", 0.05))
        tau = max(float(self.constraint_cfg.get("clear_tau", 0.10)), 1e-6)
        # ReLU gives zero cost outside the safety margin and a calibrated quadratic
        # violation inside it.  Clipping keeps extremely bad synthetic/replay
        # states from dominating the scalar loss.
        violation = torch.relu((safe_margin - min_clear) / tau).pow(2)
        clip = float(self.constraint_cfg.get("clear_cost_clip", 5.0))
        if clip > 0:
            violation = violation.clamp(max=clip)
        return violation, min_clear

    def _rollout_signed_clearance_vector(self, data: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Signed-clearance cost and min clearance along the learned rollout."""
        B = data["pos_xy"].shape[0]
        if data["obs_feats"].shape[1] == 0:
            return out["U"].new_zeros(B), out["U"].new_full((B,), float("inf"))
        steps = max(1, int(self.constraint_cfg.get("clear_rollout_steps", 1)))
        integ_cfg = EnergyIntegratorConfig(
            scheme=str(self.constraint_cfg.get("integrator", self.teacher_cfg.get("integrator", "semi_implicit"))),
            dt=float(self.constraint_cfg.get("dt", self.teacher_cfg.get("dt", 0.10))),
            damping=float(self.constraint_cfg.get("integrator_damping", self.teacher_cfg.get("integrator_damping", 0.20))),
            momentum_clip=float(self.constraint_cfg.get("momentum_clip", self.teacher_cfg.get("momentum_clip", 1.50))),
            clamp_to_grid=True,
        )
        roll = integrate_local_energy_field(
            data["pos_xy"],
            lambda q: sample_map_at_pos(out["F"], q, data["hat_d_vec"]),
            data["hat_d_vec"],
            integ_cfg,
            steps=steps,
            p0=None,
        )
        q_seq = roll["q_seq"]
        if q_seq.shape[1] == 0:
            return out["U"].new_zeros(B), out["U"].new_full((B,), float("inf"))
        return self._signed_clearance_violation_at(data, q_seq)

    def _rollout_clearance_constraint_vector(self, data: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample clearance constraint, raw barrier diagnostic, signed clearance."""
        raw = self._rollout_barrier_exposure_vector(data, out)
        signed_cost, signed_min = self._rollout_signed_clearance_vector(data, out)
        if self._clearance_mode() == "signed":
            return signed_cost, raw, signed_min
        return raw, raw, signed_min.detach()

    def _rollout_clearance_constraint(self, data: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C, raw, signed_min = self._rollout_clearance_constraint_vector(data, out)
        return C.mean(), raw.mean().detach(), signed_min.mean().detach()

    def _lambda_teacher(
        self,
        data: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        lambda_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step lambda-space random shooting teacher.

        Candidates are scored by local force/goal alignment, obstacle exposure,
        actuator projection defect, anti-static force floor, and lambda TV.  This
        gives a concrete target in dual-weight space instead of manually balancing
        many weak scalar losses.
        """
        C = int(self.teacher_cfg.get("candidates", 0))
        if C <= 0:
            return out["lambda"].detach(), out["lambda"].new_zeros(())
        B, L = out["lambda"].shape
        base = out["lambda"].detach()
        target = lambda_trigger_targets(out["aux"].detach()).to(base.device, base.dtype)
        cand = [base, target]
        noise_scale = float(self.teacher_cfg.get("noise", 0.50))
        for _ in range(max(0, C - 2)):
            cand.append(base + noise_scale * torch.randn_like(base))
        cands = torch.stack(cand, dim=1).clamp(self.model.cfg.lambda_min, self.model.cfg.lambda_max)  # [B,C,L]
        score = []
        barrier_map = data["barrier_stack"].amax(dim=1, keepdim=True) if data["barrier_stack"].shape[1] else out["U"].new_zeros(B, 1, out["U"].shape[-2], out["U"].shape[-1])
        pos = data["pos_xy"]
        for j in range(cands.shape[1]):
            oj = recompute_with_lambda(
                self.model, data["grid_xy"], data["barrier_stack"], data["goal_map"], data["coord_map"],
                data["obs_feats"], data["obs_mask"], data["goal_feats"], data["hat_d_vec"],
                cands[:, j], ctx=out["ctx"].detach(), alpha=out["alpha"].detach(), aux=out["aux"].detach(),
            )
            F0 = sample_map_at_pos(oj["F"], pos, data["hat_d_vec"])
            align = per_sample_cosine_error(F0, data["dir_xy"])
            # Roll one local step through the selected integrator and query barrier exposure there.
            integ_cfg = EnergyIntegratorConfig(
                scheme=str(self.teacher_cfg.get("integrator", "semi_implicit")),
                dt=float(self.teacher_cfg.get("dt", 0.10)),
                damping=float(self.teacher_cfg.get("integrator_damping", 0.20)),
                momentum_clip=float(self.teacher_cfg.get("momentum_clip", 1.50)),
                clamp_to_grid=True,
            )
            roll = integrate_local_energy_field(
                pos, lambda q, Fmap=oj["F"]: sample_map_at_pos(Fmap, q, data["hat_d_vec"]),
                data["hat_d_vec"], integ_cfg, steps=1, p0=None,
            )
            q1 = roll["q_seq"][:, -1, :]
            if self._clearance_mode() == "signed":
                b1, _min_clear = self._signed_clearance_violation_at(data, q1)
            else:
                b1 = sample_map_at_pos(barrier_map, q1, data["hat_d_vec"])[:, 0]
            u_raw = force_to_ackermann_train(
                F0,
                v_nominal=float(self.teacher_cfg.get("v_nominal", 0.55)),
                steering_gain=float(self.teacher_cfg.get("steering_gain", 1.4)),
                signed_speed=bool(self.teacher_cfg.get("signed_speed", False)),
            )
            u_proj = project_ackermann_box_rate(u_raw, None, self.limit_cfg)
            act = ((u_raw - u_proj) ** 2).sum(dim=-1)
            stall = torch.relu(float(self.loss_weights.get("force_floor", 0.05)) - torch.linalg.norm(F0, dim=-1)) ** 2
            tv = ((cands[:, j] - (base if lambda_prev is None else lambda_prev.to(base.device, base.dtype))) ** 2).mean(dim=-1)
            sc = (
                align
                + float(self.teacher_cfg.get("w_barrier", 0.10)) * b1
                + float(self.teacher_cfg.get("w_act", 0.05)) * act
                + float(self.teacher_cfg.get("w_stall", 0.25)) * stall
                + float(self.teacher_cfg.get("w_tv", 0.02)) * tv
            )
            score.append(sc)
        scores = torch.stack(score, dim=1)  # [B,C]
        best_idx = scores.argmin(dim=1)
        best = cands[torch.arange(B, device=base.device), best_idx]
        return best.detach(), scores.gather(1, best_idx[:, None]).mean().detach()

    def _failure_scores(
        self,
        data: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        lambda_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Per-sample failure score used to mine replay states.

        Tags separate safety failures, actuator infeasibility, stall, and reverse
        mismatch.  Reverse mismatch is important for parallel parking: a local
        target behind the car should produce a negative speed under signed-speed
        training.
        """
        F_at = sample_map_at_pos(out["F"], data["pos_xy"], data["hat_d_vec"])
        align = per_sample_cosine_error(F_at, data["dir_xy"])
        force_norm = torch.linalg.norm(F_at, dim=-1)
        clear, clear_raw, min_clear_signed = self._rollout_clearance_constraint_vector(data, out)
        u_raw = force_to_ackermann_train(
            F_at,
            v_nominal=float(self.constraint_cfg.get("v_nominal", 0.55)),
            steering_gain=float(self.constraint_cfg.get("steering_gain", 1.4)),
            signed_speed=bool(self.constraint_cfg.get("signed_speed", False)),
        )
        u_proj = project_ackermann_box_rate(u_raw, None, self.limit_cfg)
        act = ((u_raw - u_proj) ** 2).sum(dim=-1)
        goal_norm = data["goal_feats"][:, 2] if data["goal_feats"].shape[1] > 2 else torch.linalg.norm(data["goal_feats"][:, :2], dim=-1)
        stall = (goal_norm > 0.35).float() * (torch.relu(float(self.loss_weights.get("force_floor", 0.05)) - force_norm) ** 2)
        reverse_needed = (data["dir_xy"][:, 0] < float(self.replay_cfg.get("reverse_dir_threshold", -0.20))).float()
        reverse_mismatch = reverse_needed * torch.relu(u_raw[:, 0] + float(self.replay_cfg.get("reverse_speed_margin", 0.03))) ** 2
        tv = torch.zeros_like(force_norm) if lambda_prev is None or lambda_prev.shape != out["lambda"].shape else ((out["lambda"] - lambda_prev.to(out["lambda"].device, out["lambda"].dtype)) ** 2).mean(dim=-1)
        score = (
            float(self.replay_cfg.get("w_align", 0.50)) * align
            + float(self.replay_cfg.get("w_clear", 2.00)) * clear
            + float(self.replay_cfg.get("w_act", 1.00)) * act
            + float(self.replay_cfg.get("w_stall", 2.00)) * stall
            + float(self.replay_cfg.get("w_reverse", 3.00)) * reverse_mismatch
            + float(self.replay_cfg.get("w_tv", 0.20)) * tv
        )
        tags = {
            "clear": clear > float(self.constraint_cfg.get("eps_clear", 0.05)),
            "act": act > float(self.constraint_cfg.get("eps_act", 0.01)),
            "stall": stall > float(self.constraint_cfg.get("eps_stall", 0.001)),
            "reverse": reverse_needed > 0,
            "reverse_mismatch": reverse_mismatch > 1e-6,
        }
        metrics = {
            "replay_score": score.detach().mean(),
            "reverse_needed": reverse_needed.detach().mean(),
            "reverse_mismatch": reverse_mismatch.detach().mean(),
            "clear_raw": clear_raw.detach().mean(),
            "min_clear_signed": min_clear_signed.detach().mean(),
        }
        return score.detach(), tags, metrics


    def _planner_supervision_loss(self, out: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Supervise the generic short-horizon planner with offline teacher pose_seq.

        This is the offline-only part of the method: the planner learns a local
        feasible motion segment from MPC/teacher rollouts.  At deployment, the
        same plan head runs from observations only; no MPC is queried.
        """
        if "pose_seq_xy" not in data or "plan_seq" not in out:
            return out["F"].new_zeros(())
        K = min(out["plan_seq"].shape[1], data["pose_seq_xy"].shape[1])
        K = min(K, int(self.loss_weights.get("plan_steps", K)))
        if K <= 0:
            return out["F"].new_zeros(())
        pred = out["plan_seq"][:, :K, :]
        tgt = data["pose_seq_xy"][:, :K, :].to(pred.device, pred.dtype)
        mask = data.get("pose_seq_mask", torch.ones(pred.shape[:2], device=pred.device, dtype=torch.bool))[:, :K]
        if not mask.any():
            return pred.new_zeros(())
        scale = data["hat_d_vec"].view(-1, 1, 1).clamp_min(1e-6)
        err = ((pred - tgt) / scale).pow(2).sum(dim=-1)
        return err[mask].mean()

    def _plan_tracking_rollout_loss(self, data: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Train the energy/force field to track its own predicted plan.

        The target plan is detached so this term shapes the pH/energy controller
        rather than letting the planner collapse toward the current rollout.
        """
        if "plan_seq" not in out:
            return out["F"].new_zeros(())
        steps = int(self.loss_weights.get("plan_track_steps", self.loss_weights.get("rollout_steps", 0)))
        if steps <= 0:
            return out["F"].new_zeros(())
        steps = min(steps, int(out["plan_seq"].shape[1]))
        if steps <= 0:
            return out["F"].new_zeros(())
        integ_cfg = EnergyIntegratorConfig(
            scheme=str(self.constraint_cfg.get("integrator", self.teacher_cfg.get("integrator", "semi_implicit"))),
            dt=float(self.constraint_cfg.get("dt", self.teacher_cfg.get("dt", 0.10))),
            damping=float(self.constraint_cfg.get("integrator_damping", self.teacher_cfg.get("integrator_damping", 0.20))),
            momentum_clip=float(self.constraint_cfg.get("momentum_clip", self.teacher_cfg.get("momentum_clip", 1.50))),
        )
        roll = integrate_local_energy_field(
            data["pos_xy"],
            lambda q: sample_map_at_pos(out["F"], q, data["hat_d_vec"]),
            data["hat_d_vec"],
            integ_cfg,
            steps=steps,
            p0=None,
        )
        pred = roll["q_seq"][:, :steps]
        tgt = out["plan_seq"][:, :steps, :].detach()
        scale = data["hat_d_vec"].view(-1, 1, 1).clamp_min(1e-6)
        return (((pred - tgt) / scale).pow(2).sum(dim=-1)).mean()

    def _teacher_command_loss(self, F_at: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Imitate the MPC teacher's local Ackermann command [v, omega].

        The energy model emits a force.  We map it to [v, steering] using the
        same deployment surrogate, convert steering to yaw rate, and compare
        against the saved MPC [v, omega] target.  This anchors reverse/turning
        behavior beyond pure force-direction alignment.
        """
        if "cmd" not in data:
            return F_at.new_zeros(())
        u = force_to_ackermann_train(
            F_at,
            v_nominal=float(self.constraint_cfg.get("v_nominal", 0.55)),
            steering_gain=float(self.constraint_cfg.get("steering_gain", 1.4)),
            signed_speed=bool(self.constraint_cfg.get("signed_speed", False)),
        )
        v = u[:, 0]
        delta = u[:, 1].clamp(-float(self.limit_cfg.steering_max), float(self.limit_cfg.steering_max))
        wheelbase = float(self.constraint_cfg.get("wheelbase", 0.324))
        omega = v * torch.tan(delta) / max(wheelbase, 1e-6)
        pred = torch.stack([v, omega], dim=-1)
        target = data["cmd"].to(pred.device, pred.dtype)
        # Normalize yaw-rate relative to a typical parking-scale value so it
        # does not dominate signed-speed imitation.
        scale = pred.new_tensor([max(float(self.limit_cfg.v_max), abs(float(self.limit_cfg.v_min)), 1e-3), 1.0])
        return F.smooth_l1_loss(pred / scale, target / scale)

    def _waypoint_alignment_loss(self, F_at: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "waypoint_xy" not in data:
            return F_at.new_zeros(())
        target = data["waypoint_xy"] - data["pos_xy"]
        valid = torch.linalg.norm(target, dim=-1) > 1e-4
        if not valid.any():
            return F_at.new_zeros(())
        return per_sample_cosine_error(F_at[valid], target[valid]).mean()

    def _teacher_rollout_loss(self, data: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Match the model-induced local rollout to the MPC teacher pose sequence."""
        if "pose_seq_xy" not in data:
            return out["F"].new_zeros(())
        steps = int(self.loss_weights.get("rollout_steps", 0))
        if steps <= 0:
            return out["F"].new_zeros(())
        T = int(data["pose_seq_xy"].shape[1])
        steps = min(steps, T)
        if steps <= 0:
            return out["F"].new_zeros(())
        integ_cfg = EnergyIntegratorConfig(
            scheme=str(self.constraint_cfg.get("integrator", self.teacher_cfg.get("integrator", "semi_implicit"))),
            dt=float(self.constraint_cfg.get("dt", self.teacher_cfg.get("dt", 0.10))),
            damping=float(self.constraint_cfg.get("integrator_damping", self.teacher_cfg.get("integrator_damping", 0.20))),
            momentum_clip=float(self.constraint_cfg.get("momentum_clip", self.teacher_cfg.get("momentum_clip", 1.50))),
        )
        roll = integrate_local_energy_field(
            data["pos_xy"],
            lambda q: sample_map_at_pos(out["F"], q, data["hat_d_vec"]),
            data["hat_d_vec"],
            integ_cfg,
            steps=steps,
            p0=None,
        )
        pred = roll["q_seq"][:, :steps]
        tgt = data["pose_seq_xy"][:, :steps].to(pred.device, pred.dtype)
        mask = data.get("pose_seq_mask", torch.ones(pred.shape[:2], device=pred.device, dtype=torch.bool))[:, :steps]
        if not mask.any():
            return pred.new_zeros(())
        # Use hat_d normalization so the same weight works across local windows.
        scale = data["hat_d_vec"].view(-1, 1, 1).clamp_min(1e-6)
        err = ((pred - tgt) / scale).pow(2).sum(dim=-1)
        return err[mask].mean()

    def _contact_sliding_loss(self, F_at: torch.Tensor, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encourage safe tangential motion near obstacles instead of bouncing away.

        In a parking slot, nearby obstacles are not only hazards; they define a
        contact corridor.  When the robot is in a safe near-contact band, the
        useful teacher motion is often tangential to the nearest obstacle.  This
        loss aligns the tangential component of the energy force with the teacher
        direction and caps excessive outward normal force, which otherwise causes
        the observed bounce-away behavior.
        """
        if data["obs_feats"].shape[1] == 0:
            z = F_at.new_zeros(())
            return z, z, z
        C = data["obs_feats"][..., :2]
        R = data["obs_feats"][..., 2].clamp_min(1e-6)
        mask = data["obs_mask"].bool()
        q = data["pos_xy"][:, None, :]
        rel = q - C
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
        signed = dist - R
        signed = signed.masked_fill(~mask, 1e6)
        idx = signed.argmin(dim=1)
        B = F_at.shape[0]
        ar = torch.arange(B, device=F_at.device)
        dmin = signed[ar, idx]
        n = rel[ar, idx] / dist[ar, idx].unsqueeze(-1)
        band = float(self.loss_weights.get("contact_band", 0.35))
        near = (dmin > -0.02) & (dmin < band) & mask.any(dim=1)
        if not near.any():
            z = F_at.new_zeros(())
            return z, near.float().mean().detach(), dmin.mean().detach()
        fhat = F_at / torch.linalg.norm(F_at, dim=-1, keepdim=True).clamp_min(1e-8)
        that = data["dir_xy"] / torch.linalg.norm(data["dir_xy"], dim=-1, keepdim=True).clamp_min(1e-8)
        f_tan = fhat - (fhat * n).sum(dim=-1, keepdim=True) * n
        t_tan = that - (that * n).sum(dim=-1, keepdim=True) * n
        tan_valid = (torch.linalg.norm(f_tan, dim=-1) > 1e-5) & (torch.linalg.norm(t_tan, dim=-1) > 1e-5) & near
        if tan_valid.any():
            L_tan = per_sample_cosine_error(f_tan[tan_valid], t_tan[tan_valid]).mean()
        else:
            L_tan = F_at.new_zeros(())
        normal_cap = float(self.loss_weights.get("contact_normal_cap", 0.35))
        normal_out = (fhat * n).sum(dim=-1)
        # Penalize forces dominated by outward obstacle normal; keep some outward
        # component available for safety, but do not let it swamp the tangent.
        L_bounce = (torch.relu(normal_out[near] - normal_cap) ** 2).mean()
        return L_tan + L_bounce, near.float().mean().detach(), dmin.mean().detach()

    def _auglag_losses(self, out: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor], lambda_prev: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        F_at = sample_map_at_pos(out["F"], data["pos_xy"], data["hat_d_vec"])
        U = out["U"]
        lam = out["lambda"]
        # Primary task: make the force point to the desired local direction and maintain useful progress force.
        L_task = per_sample_cosine_error(F_at, data["dir_xy"]).mean()
        force_norm = torch.linalg.norm(F_at, dim=-1)
        goal_norm = data["goal_feats"][:, 2] if data["goal_feats"].shape[1] > 2 else torch.linalg.norm(data["goal_feats"][:, :2], dim=-1)
        progress_reward = -0.05 * (F_at * data["dir_xy"]).sum(dim=-1).mean()
        L_task = L_task + progress_reward
        # Constraints as positive costs.
        # Log current-origin exposure separately; use rollout exposure for AL.
        C_clear_current = self._barrier_exposure(data).detach()
        C_clear, C_clear_raw, min_clear_signed = self._rollout_clearance_constraint(data, out)
        u_raw = force_to_ackermann_train(
            F_at,
            v_nominal=float(self.constraint_cfg.get("v_nominal", 0.55)),
            steering_gain=float(self.constraint_cfg.get("steering_gain", 1.4)),
            signed_speed=bool(self.constraint_cfg.get("signed_speed", False)),
        )
        u_proj = project_ackermann_box_rate(u_raw, None, self.limit_cfg)
        C_act = ((u_raw - u_proj) ** 2).sum(dim=-1).mean()
        C_stall = ((goal_norm > 0.35).float() * (torch.relu(float(self.loss_weights.get("force_floor", 0.05)) - force_norm) ** 2)).mean()
        reverse_needed = (data["dir_xy"][:, 0] < float(self.replay_cfg.get("reverse_dir_threshold", -0.20))).float()
        C_reverse = (reverse_needed * torch.relu(u_raw[:, 0] + float(self.replay_cfg.get("reverse_speed_margin", 0.03))) ** 2).mean()
        C_tv = lam.new_zeros(()) if lambda_prev is None or lambda_prev.shape != lam.shape else ((lam - lambda_prev.to(lam.device, lam.dtype)) ** 2).mean()
        C_smooth = laplacian_smoothness(U)
        L_cmd = self._teacher_command_loss(F_at, data)
        L_waypoint = self._waypoint_alignment_loss(F_at, data)
        L_rollout = self._teacher_rollout_loss(data, out)
        L_plan = self._planner_supervision_loss(out, data)
        L_plan_track = self._plan_tracking_rollout_loss(data, out)
        L_contact, contact_rate, contact_min_clear = self._contact_sliding_loss(F_at, data)
        constraints = {"clear": C_clear, "act": C_act, "stall": C_stall, "tv": C_tv, "smooth": C_smooth}
        al = lam.new_zeros(())
        if not bool(self.constraint_cfg.get("disable_auglag", False)):
            for name, Cval in constraints.items():
                eps = float(self.constraint_cfg.get(f"eps_{name}", 0.0))
                rho = float(self.constraint_cfg.get(f"rho_{name}", 1.0))
                viol = torch.relu(Cval - eps)
                al = al + self.mu[name] * viol + 0.5 * rho * viol * viol
        teacher_lam, teacher_score = self._lambda_teacher(data, out, lambda_prev)
        L_teacher = F.mse_loss(lam, teacher_lam)
        L_alpha = out["alpha"].mean() if out["alpha"].numel() else U.new_zeros(())
        L_lambda_mag = lam.mean()
        loss = (
            float(self.loss_weights.get("task", 1.0)) * L_task
            + float(self.loss_weights.get("teacher", 0.25)) * L_teacher
            + al
            + float(self.loss_weights.get("reverse", 0.0)) * C_reverse
            + float(self.loss_weights.get("cmd", 0.0)) * L_cmd
            + float(self.loss_weights.get("waypoint", 0.0)) * L_waypoint
            + float(self.loss_weights.get("rollout", 0.0)) * L_rollout
            + float(self.loss_weights.get("plan", 0.0)) * L_plan
            + float(self.loss_weights.get("plan_track", 0.0)) * L_plan_track
            + float(self.loss_weights.get("contact", 0.0)) * L_contact
            + float(self.loss_weights.get("alpha", 1e-4)) * L_alpha
            + float(self.loss_weights.get("lambda", 1e-4)) * L_lambda_mag
        )
        out_losses = {
            "loss": loss,
            "task": L_task.detach(),
            "teacher": L_teacher.detach(),
            "teacher_score": teacher_score,
            "alpha": L_alpha.detach(),
            "lambda": L_lambda_mag.detach(),
            "force_norm": force_norm.mean().detach(),
            "C_reverse": C_reverse.detach(),
            "reverse_needed": reverse_needed.mean().detach(),
            "L_cmd": L_cmd.detach(),
            "L_waypoint": L_waypoint.detach(),
            "L_rollout": L_rollout.detach(),
            "L_plan": L_plan.detach(),
            "L_plan_track": L_plan_track.detach(),
            "plan_goal_norm": torch.linalg.norm(out.get("plan_goal", F_at.new_zeros(F_at.shape)), dim=-1).mean().detach(),
            "L_contact": L_contact.detach(),
            "contact_rate": contact_rate,
            "contact_min_clear": contact_min_clear,
        }
        for name, Cval in constraints.items():
            out_losses[f"C_{name}"] = Cval.detach()
            out_losses[f"mu_{name}"] = lam.new_tensor(self.mu[name])
        out_losses["C_clear_current"] = C_clear_current
        out_losses["C_clear_raw"] = C_clear_raw
        out_losses["min_clear_signed"] = min_clear_signed
        out_losses["clearance_mode_signed"] = lam.new_tensor(1.0 if self._clearance_mode() == "signed" else 0.0)
        return out_losses

    def _update_multipliers(self, logs: Dict[str, float]) -> None:
        if self.training_mode != "auglag":
            return
        for name in self.mu:
            eps = float(self.constraint_cfg.get(f"eps_{name}", 0.0))
            rho = float(self.constraint_cfg.get(f"rho_{name}", 1.0))
            mu_max = float(self.constraint_cfg.get("mu_max", 50.0))
            val = float(logs.get(f"C_{name}", 0.0))
            delta = rho * (val - eps)
            step_clip = float(self.constraint_cfg.get("mu_step_clip", 0.0))
            if step_clip > 0:
                delta = max(-step_clip, min(step_clip, delta))
            self.mu[name] = min(mu_max, max(0.0, self.mu[name] + delta))

    def _forward_loss(
        self,
        batch: Dict[str, torch.Tensor],
        lambda_prev: Optional[torch.Tensor],
        stage_changed: Optional[torch.Tensor],
        train: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        data = self._data_to_device(batch)
        with torch.set_grad_enabled(train):
            out = self.model(
                **{k: data[k] for k in ["grid_xy", "barrier_stack", "goal_map", "coord_map", "obs_feats", "obs_mask", "goal_feats", "hat_d_vec"]},
                lambda_prev=lambda_prev,
                stage_changed=stage_changed,
            )
            if self.training_mode == "auglag":
                loss_dict = self._auglag_losses(out, data, lambda_prev)
            else:
                loss_dict = dual_energy_losses(
                    out=out,
                    pos_xy=data["pos_xy"],
                    dir_xy=data["dir_xy"],
                    hat_d_vec=data["hat_d_vec"],
                    lambda_prev=lambda_prev,
                    weights=self.loss_weights,
                )
        return data, out, loss_dict

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()

    @staticmethod
    def _accum(logs: Dict[str, float], key: str, value: float) -> None:
        logs[key] = logs.get(key, 0.0) + float(value)

    def _log_loss_dict(self, logs: Dict[str, float], loss_dict: Dict[str, torch.Tensor], prefix: str = "") -> None:
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self._accum(logs, prefix + k, float(v.detach().item()))

    def _log_lambda_means(self, logs: Dict[str, float], lam: torch.Tensor, prefix: str = "") -> None:
        lam = lam.detach()
        for i, name in enumerate(self.model.lambda_names):
            self._accum(logs, prefix + f"lam_{name}", float(lam[:, i].mean().item()))

    def _maybe_add_failure_replay(
        self,
        batch: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        lambda_prev: Optional[torch.Tensor],
        logs: Dict[str, float],
    ) -> None:
        if int(self.replay_cfg.get("capacity", 0)) <= 0:
            return
        with torch.no_grad():
            score, tags, metrics = self._failure_scores(data, out, lambda_prev)
        added = self.replay.add_from_scores(
            batch,
            score,
            tags,
            min_score=float(self.replay_cfg.get("min_score", 0.05)),
            topk=int(self.replay_cfg.get("topk", 4)),
        )
        self._accum(logs, "replay_added", float(added))
        self._accum(logs, "replay_size", float(len(self.replay)))
        for k, v in metrics.items():
            self._accum(logs, "failure/" + k, float(v.detach().item() if torch.is_tensor(v) else v))

    def _train_one_replay_batch(self, batch: Dict[str, torch.Tensor], logs: Dict[str, float], prefix: str) -> None:
        B = int(batch["goal_map"].shape[0])
        stage_changed = torch.ones(B, device=self.device)
        old_disable_al = self.constraint_cfg.get("disable_auglag", False)
        if prefix.startswith("parking/") and not bool(self.parking_cfg.get("use_auglag", False)):
            self.constraint_cfg["disable_auglag"] = True
        data, out, loss_dict = self._forward_loss(batch, lambda_prev=None, stage_changed=stage_changed, train=True)
        self.constraint_cfg["disable_auglag"] = old_disable_al
        if prefix.startswith("parking/"):
            scale = float(self.parking_cfg.get("loss_scale", self.replay_cfg.get("loss_scale", 1.0)))
        else:
            scale = float(self.replay_cfg.get("loss_scale", 1.0))
        self._optimizer_step(scale * loss_dict["loss"])
        self._log_loss_dict(logs, loss_dict, prefix=prefix)
        self._log_lambda_means(logs, out["lambda"], prefix=prefix)
        with torch.no_grad():
            score, _tags, metrics = self._failure_scores(data, out, None)
        self._accum(logs, prefix + "score", float(score.mean().item()))
        for k, v in metrics.items():
            self._accum(logs, prefix + k, float(v.detach().item() if torch.is_tensor(v) else v))

    def run_epoch(self, loader: DataLoader, train: bool = True):
        self.model.train(train)
        logs: Dict[str, float] = {}
        n = 0
        n_replay = 0
        n_parking = 0
        current_epoch = int(self.parking_cfg.get("epoch", 0))
        if not train:
            self.lambda_state = None
            self.prev_episode = None
            self.prev_stage = None
        for batch in loader:
            B = batch["goal_map"].shape[0]
            lambda_prev, stage_changed = self._state_features(batch["meta"], B)
            data, out, loss_dict = self._forward_loss(batch, lambda_prev, stage_changed, train=train)
            if train:
                self._optimizer_step(loss_dict["loss"])
            self.lambda_state = out["lambda"].detach()
            self._log_loss_dict(logs, loss_dict)
            self._log_lambda_means(logs, out["lambda"])
            if train and current_epoch >= int(self.replay_cfg.get("start_epoch", 0)):
                self._maybe_add_failure_replay(batch, data, out, lambda_prev, logs)
                if len(self.replay) >= int(self.replay_cfg.get("warmup", 4)) and self.rng.random() < float(self.replay_cfg.get("prob", 0.0)):
                    rb = self.replay.sample()
                    if rb is not None:
                        self._train_one_replay_batch(rb, logs, prefix="replay/")
                        n_replay += 1
            n += 1

        if train and current_epoch >= int(self.parking_cfg.get("start_epoch", 0)) and int(self.parking_cfg.get("batches_per_epoch", 0)) > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.parking_cfg.get("seed", 12345)) + int(self.parking_cfg.get("epoch", 0)))
            for _ in range(int(self.parking_cfg.get("batches_per_epoch", 0))):
                pb = sample_parallel_parking_batch(
                    batch_size=int(self.parking_cfg.get("batch_size", 16)),
                    H=int(self.parking_cfg.get("H", 64)),
                    W=int(self.parking_cfg.get("W", 64)),
                    hat_d=float(self.parking_cfg.get("hat_d", 3.0)),
                    device="cpu",
                    generator=gen,
                )
                self._train_one_replay_batch(pb, logs, prefix="parking/")
                n_parking += 1
            self.parking_cfg["epoch"] = int(self.parking_cfg.get("epoch", 0)) + 1

        denom = max(1, n)
        logs = {k: v / denom for k, v in logs.items()}
        if n_replay > 0:
            for k in list(logs.keys()):
                if k.startswith("replay/"):
                    logs[k] = logs[k] * denom / max(1, n_replay)
        if n_parking > 0:
            for k in list(logs.keys()):
                if k.startswith("parking/"):
                    logs[k] = logs[k] * denom / max(1, n_parking)
        if train:
            self._update_multipliers(logs)
            for k, v in self.mu.items():
                logs[f"mu_{k}"] = float(v)
            logs["replay_buffer_size"] = float(len(self.replay))
            logs["replay_steps"] = float(n_replay)
            logs["parking_steps"] = float(n_parking)
            for k, v in self.replay.tag_summary().items():
                logs[k] = v
        return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="NavMax dataset root/manifest, Stage-1 case-suite root, case directory, or split .pt")
    ap.add_argument("--data-format", type=str, default="auto", choices=["auto", "navmax", "stage1"],
                    help="auto detects Stage-1 rollout payloads; navmax preserves the original local-patch loader.")
    ap.add_argument("--stage1-split", type=str, default="train", choices=["train", "val", "test"],
                    help="Split to load when --data points to a Stage-1 case-suite root or case directory.")
    ap.add_argument("--stage1-cases", type=str, default="",
                    help="Comma-separated subset of Stage-1 cases when --data points to a suite root.")
    ap.add_argument("--stage1-direction-from", type=str, default="cmd", choices=["cmd", "goal"],
                    help="Use MPC command direction or local goal direction as dir_xy for Stage-1 adapter.")
    ap.add_argument("--stage1-wheelbase", type=float, default=0.324)
    ap.add_argument("--stage1-robot-radius", type=float, default=0.23)
    ap.add_argument("--stage1-max-samples-per-case", type=int, default=None,
                    help="Optional cap for quick debugging on Stage-1 case-suite datasets.")
    ap.add_argument("--outdir", type=str, default="checkpoints/dual_weight_energy")
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--hat_d", type=float, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--training-mode", type=str, default="weighted", choices=["weighted", "auglag"],
                    help="weighted keeps the original loss; auglag uses task objective + adaptive constraints.")
    # Loss weights
    ap.add_argument("--w-align", type=float, default=1.0)
    ap.add_argument("--w-smooth", type=float, default=1e-3)
    ap.add_argument("--w-alpha", type=float, default=1e-4)
    ap.add_argument("--w-lambda", type=float, default=1e-4)
    ap.add_argument("--w-dual-target", type=float, default=0.05)
    ap.add_argument("--w-dual-tv", type=float, default=0.01)
    ap.add_argument("--w-anti-static", type=float, default=0.10)
    ap.add_argument("--force-floor", type=float, default=0.05)
    ap.add_argument("--w-reverse", type=float, default=0.0,
                    help="Penalty for failing to issue reverse speed when local target is behind the car.")
    ap.add_argument("--w-cmd", type=float, default=0.0,
                    help="Behavior-cloning loss on Stage-1 MPC [v, omega] commands.")
    ap.add_argument("--w-waypoint", type=float, default=0.0,
                    help="Align force to a short-horizon teacher waypoint from pose_seq.")
    ap.add_argument("--w-rollout", type=float, default=0.0,
                    help="Match local energy rollout to the saved MPC pose_seq.")
    ap.add_argument("--rollout-supervision-steps", type=int, default=4,
                    help="Number of teacher pose_seq steps used by --w-rollout.")
    ap.add_argument("--w-plan", type=float, default=0.0,
                    help="Supervise the generic short-horizon planner with saved pose_seq. Offline only; no MPC at deployment.")
    ap.add_argument("--w-plan-track", type=float, default=0.0,
                    help="Train the energy controller to track its own predicted plan.")
    ap.add_argument("--plan-supervision-steps", type=int, default=6,
                    help="Number of pose_seq steps used by --w-plan.")
    ap.add_argument("--plan-track-steps", type=int, default=4,
                    help="Number of predicted plan steps tracked by the local energy rollout.")
    ap.add_argument("--w-contact", type=float, default=0.0,
                    help="Near-contact tangential sliding loss to avoid barrier bounce-away.")
    ap.add_argument("--contact-band", type=float, default=0.35,
                    help="Signed-clearance band in meters where contact/tangent supervision is active.")
    ap.add_argument("--contact-normal-cap", type=float, default=0.35,
                    help="Maximum normalized outward obstacle-normal component before bounce penalty.")
    # Energy-controller / augmented-Lagrangian knobs
    ap.add_argument("--w-task", type=float, default=1.0)
    ap.add_argument("--w-teacher", type=float, default=0.25)
    ap.add_argument("--teacher-candidates", type=int, default=16)
    ap.add_argument("--teacher-noise", type=float, default=0.50)
    ap.add_argument("--teacher-w-barrier", type=float, default=0.10)
    ap.add_argument("--teacher-w-act", type=float, default=0.05)
    ap.add_argument("--teacher-w-stall", type=float, default=0.30)
    ap.add_argument("--teacher-w-tv", type=float, default=0.02)
    ap.add_argument("--eps-clear", type=float, default=0.05)
    ap.add_argument("--eps-act", type=float, default=0.01)
    ap.add_argument("--eps-stall", type=float, default=0.001)
    ap.add_argument("--eps-tv", type=float, default=0.08)
    ap.add_argument("--eps-smooth", type=float, default=0.30)
    ap.add_argument("--rho-clear", type=float, default=2.0)
    ap.add_argument("--rho-act", type=float, default=1.0)
    ap.add_argument("--rho-stall", type=float, default=4.0)
    ap.add_argument("--rho-tv", type=float, default=0.5)
    ap.add_argument("--rho-smooth", type=float, default=0.1)
    ap.add_argument("--clearance-mode", type=str, default="auto", choices=["auto", "barrier", "signed"],
                    help="AL clearance constraint. auto uses signed clearance for Stage-1 and raw barrier for NavMax patches.")
    ap.add_argument("--clear-safe-margin", type=float, default=0.05,
                    help="Extra signed-distance safety margin in meters for --clearance-mode signed.")
    ap.add_argument("--clear-tau", type=float, default=0.10,
                    help="Signed-clearance normalization scale in meters.")
    ap.add_argument("--clear-cost-clip", type=float, default=5.0,
                    help="Clip per-sample signed-clearance violation cost before averaging; <=0 disables clipping.")
    ap.add_argument("--clear-rollout-steps", type=int, default=1,
                    help="Number of predicted local energy steps used for differentiable C_clear.")
    ap.add_argument("--mu-max", type=float, default=50.0,
                    help="Safety cap for training-only augmented-Lagrangian multipliers.")
    ap.add_argument("--mu-step-clip", type=float, default=0.0,
                    help="Optional cap on each epoch's AL multiplier update. 0 disables clipping.")
    # Actuation box used for projection-defect constraint
    ap.add_argument("--v-min", type=float, default=-0.25)
    ap.add_argument("--v-max", type=float, default=0.80)
    ap.add_argument("--steering-max", type=float, default=0.396)
    ap.add_argument("--accel-max", type=float, default=1.5)
    ap.add_argument("--steering-rate-max", type=float, default=1.5)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--energy-integrator", type=str, default="semi_implicit",
                    choices=["explicit", "semi_implicit", "midpoint", "velocity_verlet"],
                    help="Integrator used by lambda-space teacher rollouts.")
    ap.add_argument("--integrator-damping", type=float, default=0.20)
    ap.add_argument("--momentum-clip", type=float, default=1.50)
    ap.add_argument("--signed-speed", action="store_true",
                    help="Allow energy fields with negative local x-force to command reverse Ackermann speed.")
    # Failure replay / medium-level curriculum
    ap.add_argument("--failure-replay-capacity", type=int, default=0)
    ap.add_argument("--failure-replay-prob", type=float, default=0.0,
                    help="Probability of an extra replay update after each normal mini-batch.")
    ap.add_argument("--failure-replay-start-epoch", type=int, default=0,
                    help="Do not mine/use replay before this epoch.")
    ap.add_argument("--failure-replay-warmup", type=int, default=4)
    ap.add_argument("--failure-replay-topk", type=int, default=4)
    ap.add_argument("--failure-replay-min-score", type=float, default=0.05)
    ap.add_argument("--failure-replay-loss-scale", type=float, default=1.0)
    ap.add_argument("--replay-w-clear", type=float, default=2.0)
    ap.add_argument("--replay-w-act", type=float, default=1.0)
    ap.add_argument("--replay-w-stall", type=float, default=2.0)
    ap.add_argument("--replay-w-reverse", type=float, default=3.0)
    ap.add_argument("--reverse-dir-threshold", type=float, default=-0.20)
    ap.add_argument("--reverse-speed-margin", type=float, default=0.03)
    ap.add_argument("--parking-augment-batches", type=int, default=0,
                    help="Synthetic parallel-parking reverse/straighten mini-batches per epoch.")
    ap.add_argument("--parking-augment-bs", type=int, default=None)
    ap.add_argument("--parking-hat-d", type=float, default=3.0)
    ap.add_argument("--parking-loss-scale", type=float, default=0.25)
    ap.add_argument("--parking-augment-start-epoch", type=int, default=0,
                    help="Do not apply synthetic parking augmentation before this epoch.")
    ap.add_argument("--parking-use-auglag", action="store_true",
                    help="Apply AL penalties to synthetic parking augmentation. Default keeps augmentation supervised-only.")
    # Model knobs
    ap.add_argument("--lambda-step", type=float, default=0.12)
    ap.add_argument("--lambda-max", type=float, default=6.0)
    ap.add_argument("--d-tok", type=int, default=80)
    ap.add_argument("--ctx-channels", type=int, default=32)
    ap.add_argument("--use-plan-goal", action="store_true",
                    help="Use the learned short-horizon planner's waypoint as the local energy target at training and deployment.")
    ap.add_argument("--plan-horizon", type=int, default=6)
    ap.add_argument("--plan-goal-index", type=int, default=3)
    ap.add_argument("--plan-radius-frac", type=float, default=0.85)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    os.makedirs(args.outdir, exist_ok=True)

    data_format = args.data_format
    if data_format == "auto":
        data_format = "stage1" if is_stage1_dataset_path(args.data) else "navmax"
    if data_format == "stage1":
        stage1_cases = [c.strip() for c in args.stage1_cases.split(",") if c.strip()] or None
        ds = Stage1EnergyPatchDataset(
            root_or_file=args.data,
            split=args.stage1_split,
            cases=stage1_cases,
            H=args.H,
            W=args.W,
            hat_d=args.hat_d if args.hat_d is not None else 3.5,
            wheelbase=args.stage1_wheelbase,
            robot_radius=args.stage1_robot_radius,
            direction_from=args.stage1_direction_from,
            max_samples_per_case=args.stage1_max_samples_per_case,
        )
    else:
        ds = NavMaximinLocalPatches(root_or_manifest=args.data, hat_d=args.hat_d, H=args.H, W=args.W)
    # IMPORTANT: shuffle=False preserves episode/snapshot order so the dual state is meaningful.
    collate_fn = collate_stage1_energy if data_format == "stage1" else collate_navmax
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, drop_last=False)
    print(json.dumps({"data_format": data_format, "num_samples": len(ds), "data": args.data}, indent=2))

    cfg = DualWeightConfig(
        d_tok=args.d_tok, ctx_channels=args.ctx_channels,
        lambda_step=args.lambda_step, lambda_max=args.lambda_max,
        use_plan_goal=bool(args.use_plan_goal),
        plan_horizon=args.plan_horizon,
        plan_goal_index=args.plan_goal_index,
        plan_radius_frac=args.plan_radius_frac,
    )
    model = DualWeightEnergyNet(H=args.H, W=args.W, cfg=cfg)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            print(json.dumps({"resume_missing_keys": missing, "resume_unexpected_keys": unexpected}, indent=2))

    weights = {
        "align": args.w_align,
        "smooth": args.w_smooth,
        "alpha": args.w_alpha,
        "lambda": args.w_lambda,
        "dual_target": args.w_dual_target,
        "dual_tv": args.w_dual_tv,
        "anti_static": args.w_anti_static,
        "force_floor": args.force_floor,
        "reverse": args.w_reverse,
        "task": args.w_task,
        "teacher": args.w_teacher,
        "cmd": args.w_cmd,
        "waypoint": args.w_waypoint,
        "rollout": args.w_rollout,
        "rollout_steps": args.rollout_supervision_steps,
        "plan": args.w_plan,
        "plan_track": args.w_plan_track,
        "plan_steps": args.plan_supervision_steps,
        "plan_track_steps": args.plan_track_steps,
        "contact": args.w_contact,
        "contact_band": args.contact_band,
        "contact_normal_cap": args.contact_normal_cap,
    }
    constraint_cfg = {
        "eps_clear": args.eps_clear, "eps_act": args.eps_act, "eps_stall": args.eps_stall,
        "eps_tv": args.eps_tv, "eps_smooth": args.eps_smooth,
        "rho_clear": args.rho_clear, "rho_act": args.rho_act, "rho_stall": args.rho_stall,
        "rho_tv": args.rho_tv, "rho_smooth": args.rho_smooth,
        "clearance_mode": args.clearance_mode,
        "data_format": data_format,
        "clear_safe_margin": args.clear_safe_margin,
        "clear_tau": args.clear_tau,
        "clear_cost_clip": args.clear_cost_clip,
        "clear_rollout_steps": args.clear_rollout_steps,
        "mu_max": args.mu_max,
        "mu_step_clip": args.mu_step_clip,
        "dt": args.dt,
        "integrator": args.energy_integrator,
        "integrator_damping": args.integrator_damping,
        "momentum_clip": args.momentum_clip,
        "signed_speed": bool(args.signed_speed),
        "wheelbase": args.stage1_wheelbase,
        "reverse_dir_threshold": args.reverse_dir_threshold,
        "reverse_speed_margin": args.reverse_speed_margin,
    }
    teacher_cfg = {
        "candidates": args.teacher_candidates,
        "noise": args.teacher_noise,
        "w_barrier": args.teacher_w_barrier,
        "w_act": args.teacher_w_act,
        "w_stall": args.teacher_w_stall,
        "w_tv": args.teacher_w_tv,
        "dt": args.dt,
        "integrator": args.energy_integrator,
        "integrator_damping": args.integrator_damping,
        "momentum_clip": args.momentum_clip,
        "signed_speed": bool(args.signed_speed),
    }
    replay_cfg = {
        "capacity": args.failure_replay_capacity,
        "prob": args.failure_replay_prob,
        "start_epoch": args.failure_replay_start_epoch,
        "warmup": args.failure_replay_warmup,
        "topk": args.failure_replay_topk,
        "min_score": args.failure_replay_min_score,
        "loss_scale": args.failure_replay_loss_scale,
        "w_clear": args.replay_w_clear,
        "w_act": args.replay_w_act,
        "w_stall": args.replay_w_stall,
        "w_reverse": args.replay_w_reverse,
        "reverse_dir_threshold": args.reverse_dir_threshold,
        "reverse_speed_margin": args.reverse_speed_margin,
        "seed": 2026,
    }
    parking_cfg = {
        "batches_per_epoch": args.parking_augment_batches,
        "batch_size": args.parking_augment_bs if args.parking_augment_bs is not None else args.bs,
        "H": args.H,
        "W": args.W,
        "hat_d": args.parking_hat_d,
        "seed": 4242,
        "loss_scale": args.parking_loss_scale,
        "start_epoch": args.parking_augment_start_epoch,
        "use_auglag": bool(args.parking_use_auglag),
    }
    limit_cfg = AckermannLimitConfig(args.v_min, args.v_max, args.steering_max, args.accel_max, args.steering_rate_max, args.dt)
    trainer = DualWeightTrainer(
        model, args.device, args.lr, weights, training_mode=args.training_mode,
        constraint_cfg=constraint_cfg, teacher_cfg=teacher_cfg, limit_cfg=limit_cfg,
        replay_cfg=replay_cfg, parking_cfg=parking_cfg,
    )
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({
            "model": cfg.to_dict(), "loss_weights": weights, "constraint_cfg": constraint_cfg,
            "teacher_cfg": teacher_cfg, "replay_cfg": replay_cfg, "parking_cfg": parking_cfg,
            "training_mode": args.training_mode, "data_format": data_format, "args": vars(args)
        }, f, indent=2)

    best = float("inf")
    for ep in range(args.epochs):
        logs = trainer.run_epoch(dl, train=True)
        print({f"ep{ep}/{k}": round(v, 6) for k, v in logs.items()})
        ckpt = {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.opt.state_dict(),
            "model_cfg": cfg.to_dict(),
            "loss_weights": weights,
            "constraint_cfg": constraint_cfg,
            "teacher_cfg": teacher_cfg,
            "replay_cfg": replay_cfg,
            "parking_cfg": parking_cfg,
            "training_mode": args.training_mode,
            "multiplier_state": trainer.mu,
            "logs": logs,
            "lambda_names": model.lambda_names,
        }
        torch.save(ckpt, os.path.join(args.outdir, "latest.pt"))
        if logs["loss"] < best:
            best = logs["loss"]
            torch.save(ckpt, os.path.join(args.outdir, "best.pt"))
        if ep % 10 == 0 or ep == args.epochs - 1:
            torch.save(ckpt, os.path.join(args.outdir, f"epoch_{ep:03d}.pt"))


if __name__ == "__main__":
    main()
