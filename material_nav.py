"""Material-aware navigation research core (faithful port).

This module ports the material-aware extension of GRL-SNAM published by the
core researcher (github.com/SetasAditya/material-aware-grl-snam) into the main
repo, verbatim in math and semantics.  It is the research-track twin of
``train_coef_energy.py`` / ``surrogate_robust.py``: torch, pixel/metre units,
checkpoint-compatible module names.  The packaged simulator integration (the
normalized-frame, default-on feature) lives in ``grl_snam/material.py`` and
uses this module's gate as its numerical oracle.

The executed field is

    F = F_geom + g(context) * lam_soft(context) * f_material
               + lam_hard(context) * f_hazard

with
    f_material = -grad r~(o)              (smoothed material risk, [0, 1])
    f_hazard   = -(db/dphi) * grad phi(o) (softplus barrier on the unsigned
                                           distance-to-hard-hazard field)
    g          = frame-wise primitive feasibility gate (a local activation
                 witness; it never chooses the executed action)

Provenance notes (load-bearing, verified against the source repo):
  * ``integrate_surrogate_material`` uses SEMI-IMPLICIT Euler (v then o with
    the new v) — the same ordering as upstream ``surrogate_robust.py``.  The
    source repo's own stale copy of ``surrogate_robust.py`` is explicit-Euler;
    that copy is NOT ported.
  * The rollout patch is SIX channels (r~, phi, dr~/dx, dr~/dy, dphi/dx,
    dphi/dy); a stale docstring in the source says four.  The hazard force
    uses the true oracle grad phi (channels 4-5), not a -grad r~ proxy.
  * The gate multiplies lam_soft ONLY; lam_hard is never gated.
  * ``mu_lat`` is the highway lateral channel: produced by the model (bias
    init -5 keeps it ~0), discarded by every off-highway harness.  It is kept
    for checkpoint compatibility; callers unpack six outputs.
  * The frozen "v7" repair controller from the source repo (stateful
    hysteresis, velocity tracking) failed its preregistered efficacy gate and
    is intentionally NOT ported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_coef_energy import ipc_piecewise

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RiskPatchEncoder(nn.Module):
    """Small CNN over a (2, P, P) risk patch: ch0 = smoothed r~, ch1 = hard mask.

    Output: (B, d_out) risk context vector.
    """

    def __init__(self, patch_size: int = 32, d_out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, d_out),
            nn.ReLU(),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.net(patch)


class CoefEnergyNetMaterial(nn.Module):
    """CoefEnergyNet backbone plus material-risk heads.

    Inputs
        obs_feats  : (B, N, 6) geometric obstacle features
        obs_mask   : (B, N) bool, True = valid obstacle
        goal_feats : (B, 4) [dx, dy, dist, 1.0]
        risk_patch : (B, 2, P, P) [r~, hard_mask]

    Outputs (six — callers must unpack all of them)
        alphas   : (B, N) per-obstacle barrier weights   (softplus)
        beta     : (B,)   goal attraction                (softplus)
        gamma    : (B,)   damping                        (softplus)
        lam_soft : (B,)   soft risk weight,  sigmoid * lam_soft_max
        lam_hard : (B,)   hard hazard weight, sigmoid * lam_hard_max
        mu_lat   : (B,)   highway lateral weight, sigmoid * mu_lat_max

    Attribute names match the source checkpoints' state-dict keys exactly.
    """

    def __init__(
        self,
        d_obs: int = 6,
        d_goal: int = 4,
        d_tok: int = 64,
        patch_size: int = 32,
        d_risk: int = 64,
        lam_soft_max: float = 5.0,
        lam_hard_max: float = 10.0,
        mu_lat_max: float = 5.0,
    ):
        super().__init__()
        self.lam_soft_max = lam_soft_max
        self.lam_hard_max = lam_hard_max
        self.mu_lat_max = mu_lat_max

        # Geometry backbone (identical to CoefEnergyNet).
        self.obs_enc = nn.Sequential(nn.Linear(d_obs, 128), nn.ReLU(), nn.Linear(128, d_tok))
        self.goal_enc = nn.Sequential(nn.Linear(d_goal, 64), nn.ReLU(), nn.Linear(64, d_tok))
        enc = nn.TransformerEncoderLayer(
            d_model=d_tok, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.fuser = nn.TransformerEncoder(enc, num_layers=2)
        self.alpha_head = nn.Sequential(nn.Linear(d_tok, 64), nn.ReLU(), nn.Linear(64, 1))
        self.beta_head = nn.Sequential(nn.Linear(d_tok, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gamma_head = nn.Sequential(nn.Linear(d_tok, 64), nn.ReLU(), nn.Linear(64, 1))

        # Material risk branch.
        self.risk_enc = RiskPatchEncoder(patch_size=patch_size, d_out=d_risk)
        self.lam_soft_head = nn.Sequential(
            nn.Linear(d_risk + d_tok, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.lam_hard_head = nn.Sequential(
            nn.Linear(d_risk + d_tok, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.mu_lat_head = nn.Sequential(nn.Linear(d_risk + d_tok, 64), nn.ReLU(), nn.Linear(64, 1))
        with torch.no_grad():
            self.mu_lat_head[-1].bias.fill_(-5.0)

    def forward(
        self,
        obs_feats: torch.Tensor,
        obs_mask: torch.Tensor,
        goal_feats: torch.Tensor,
        risk_patch: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        B, N = obs_feats.shape[:2]
        z_goal = self.goal_enc(goal_feats).unsqueeze(1)

        if N == 0:
            tokens = z_goal
            pad = torch.zeros(B, 1, dtype=torch.bool, device=obs_feats.device)
            z_all = self.fuser(tokens, src_key_padding_mask=pad)
            ctx = z_all[:, 0]
            alphas = obs_feats.new_zeros(B, 0)
        else:
            z_obs = self.obs_enc(obs_feats.reshape(B * N, -1)).reshape(B, N, -1)
            tokens = torch.cat([z_goal, z_obs], dim=1)
            pad = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=obs_mask.device), ~obs_mask], dim=1
            )
            z_all = self.fuser(tokens, src_key_padding_mask=pad)
            ctx = z_all[:, 0]
            a = F.softplus(self.alpha_head(z_all[:, 1:]).squeeze(-1))
            alphas = torch.where(obs_mask, a, torch.zeros_like(a))

        beta = F.softplus(self.beta_head(ctx)).squeeze(-1)
        gamma = F.softplus(self.gamma_head(ctx)).squeeze(-1)

        risk_ctx = self.risk_enc(risk_patch)
        mat_feats = torch.cat([risk_ctx, ctx], dim=-1)

        lam_soft = self.lam_soft_max * torch.sigmoid(self.lam_soft_head(mat_feats).squeeze(-1))
        lam_hard = self.lam_hard_max * torch.sigmoid(self.lam_hard_head(mat_feats).squeeze(-1))
        mu_lat = self.mu_lat_max * torch.sigmoid(self.mu_lat_head(mat_feats).squeeze(-1))

        return alphas, beta, gamma, lam_soft, lam_hard, mu_lat


def load_geometry_weights(
    material_model: CoefEnergyNetMaterial, geom_ckpt_path: str, device: str = "cpu"
) -> int:
    """Copy geometry-backbone weights from a Stage-1 CoefEnergyNet checkpoint.

    Risk heads are left randomly initialised.  Returns the number of matched
    tensors; raises if nothing matched (wrong checkpoint format).
    """
    ck = torch.load(geom_ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
    elif isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    else:
        sd = ck
    own_sd = material_model.state_dict()
    matched = {k: v for k, v in sd.items() if k in own_sd and own_sd[k].shape == v.shape}
    own_sd.update(matched)
    material_model.load_state_dict(own_sd)
    if not matched:
        raise RuntimeError(
            f"No compatible geometry weights found in {geom_ckpt_path}. "
            "Check that the Stage 1 checkpoint format matches the model."
        )
    return len(matched)


# ---------------------------------------------------------------------------
# Surrogate dynamics with material forces
# ---------------------------------------------------------------------------


def sdf_barrier_grad(
    sdf_val: torch.Tensor, d_hat_sdf: float = 3.0, k_sharp: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softplus hazard barrier b(phi) and its derivative db/dphi.

        b(phi)  = (1/k) * log(1 + exp(k * (d_hat - phi)))    active when phi < d_hat
        db/dphi = -sigmoid(k * (d_hat - phi))                 negative

    The force uses -lam_hard * (db/dphi) * grad phi, i.e. a push along
    +grad phi (toward open space) that fades sigmoidally past ``d_hat_sdf``.
    """
    inner = k_sharp * (d_hat_sdf - sdf_val)
    b_val = F.softplus(inner) / k_sharp
    db_dphi = -torch.sigmoid(inner)
    return b_val, db_dphi


def bilinear_sample_patch(patch: torch.Tensor, o: torch.Tensor, o0: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a local raster patch centred (in global px coords) at o0.

    patch : (B, C, Hp, Wp);  o, o0 : (B, 2) in (x=col, y=row) global pixels.
    Returns (B, C).  Out-of-bounds clamps to border (grid_sample border mode,
    align_corners=True); normalization is by the half-extent (P-1)/2 + 1e-8.
    """
    B, C, Hp, Wp = patch.shape
    offset = o - o0
    half_w = (Wp - 1) / 2.0
    half_h = (Hp - 1) / 2.0
    gx = offset[:, 0] / (half_w + 1e-8)
    gy = offset[:, 1] / (half_h + 1e-8)
    grid = torch.stack([gx, gy], dim=-1).view(B, 1, 1, 2)
    sampled = F.grid_sample(patch, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.view(B, C)


def integrate_surrogate_material(
    o0: torch.Tensor,
    v0: torch.Tensor,
    goal: torch.Tensor,
    C: torch.Tensor,
    R: torch.Tensor,
    mask: torch.Tensor,
    alphas: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    lam_soft: torch.Tensor,
    lam_hard: torch.Tensor,
    rollout_patch: torch.Tensor,
    d_hat: torch.Tensor,
    dt: torch.Tensor,
    H: torch.Tensor,
    robot_radius: float | torch.Tensor = 0.0,
    margin_factor: float = 0.5,
    mass: float = 1.0,
    d_hat_sdf: float = 3.0,
) -> Tuple[torch.Tensor, ...]:
    """Surrogate integrator with material forces (faithful port).

    Shapes: o0/v0/goal (B,2) in (x=col, y=row) global pixels; C (B,N,2);
    R/alphas (B,N); mask (B,N) bool; beta/gamma/lam_soft/lam_hard/d_hat/dt (B,);
    H (B,) int horizon; rollout_patch (B,6,Hp,Wp) centred at o0 with channels
    [r~, phi, dr~/dx, dr~/dy, dphi/dx, dphi/dy] (phi in metres, gradients in
    the raster's own units — the caller owns the unit convention, as in the
    source repo).

    Per step, fields are bilinearly resampled at the CURRENT position, so the
    forces and the accumulated costs reflect the path actually taken.

    Returns (oT, vT, min_clear_geom, cum_risk, hard_count, arc_length).
    """
    B, N = C.shape[:2]

    if not torch.is_tensor(robot_radius):
        rr = o0.new_tensor(float(robot_radius))
    else:
        rr = robot_radius.to(o0.device, o0.dtype)
    R_eff = R + margin_factor * rr[:, None] if rr.ndim >= 1 else R + margin_factor * rr

    o = o0.clone()
    v = v0.clone()
    min_clear = torch.full((B,), float("inf"), dtype=o.dtype, device=o.device)
    cum_risk = torch.zeros(B, dtype=o.dtype, device=o.device)
    hard_count = torch.zeros(B, dtype=o.dtype, device=o.device)
    arc_length = torch.zeros(B, dtype=o.dtype, device=o.device)

    for s in range(int(H.max().item())):
        active = (s < H).to(o.dtype).unsqueeze(-1)

        sem = bilinear_sample_patch(rollout_patch, o, o0)
        risk_val = sem[:, 0].clamp(0.0, 1.0)
        sdf_val = sem[:, 1].clamp(0.0, 50.0)
        risk_grad = torch.stack([sem[:, 2], sem[:, 3]], dim=-1)
        sdf_grad = torch.stack([sem[:, 4], sem[:, 5]], dim=-1)

        F_goal = -beta.unsqueeze(-1) * (o - goal)

        if N == 0:
            F_geom = torch.zeros_like(o)
            dmin = torch.full((B,), float("inf"), device=o.device)
        else:
            diff = o.unsqueeze(1) - C
            r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
            n_hat = diff / r.unsqueeze(-1)
            d = r - R_eff
            d = torch.where(mask, d, torch.full_like(d, 1e6))
            _, dbdd = ipc_piecewise(d, d_hat.view(-1, 1))
            F_geom = (-(alphas * dbdd).unsqueeze(-1) * n_hat).sum(dim=1)
            dmin = torch.where(mask, d, torch.full_like(d, float("inf"))).min(dim=1).values

        min_clear = torch.minimum(min_clear, dmin)

        F_mat_soft = -lam_soft.unsqueeze(-1) * risk_grad
        _, db_dphi = sdf_barrier_grad(sdf_val, d_hat_sdf=d_hat_sdf)
        F_mat_hard = -lam_hard.unsqueeze(-1) * db_dphi.unsqueeze(-1) * sdf_grad

        F_tot = F_goal + F_geom + F_mat_soft + F_mat_hard - gamma.unsqueeze(-1) * v
        a = F_tot / mass
        # Semi-implicit: the position step uses the NEW velocity (upstream ordering).
        v_new = v + active * dt.unsqueeze(-1) * a
        o_new = o + active * dt.unsqueeze(-1) * v_new

        step_disp = torch.linalg.norm(o_new - o, dim=-1)
        act_1d = active.squeeze(-1)
        arc_length = arc_length + act_1d * step_disp
        cum_risk = cum_risk + act_1d * risk_val * step_disp
        hard_count = hard_count + act_1d * (sdf_val < 1.0).to(o.dtype)

        v = v_new
        o = o_new

    return o, v, min_clear, cum_risk, hard_count, arc_length


# ---------------------------------------------------------------------------
# Feasibility-witness gate (frame-wise, as evaluated in the paper's ablations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateDecision:
    """Outcome of one witness-gate evaluation.

    ``active`` gates lam_soft only.  The selected primitive is an activation
    WITNESS — evidence that a feasible, progress-making, lower-risk direction
    exists — never the executed command.
    """

    active: bool
    nominal_risk: float
    best_risk: float
    feasible_count: int
    selected_direction_rc: Tuple[float, float]
    selected_endpoint_rc: Tuple[float, float]
    selected_min_clearance_m: float


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def _clip_rc(point_rc: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int]:
    # round() is round-half-to-even; the C++ twin uses std::rint to match.
    return (
        int(np.clip(round(float(point_rc[0])), 0, shape[0] - 1)),
        int(np.clip(round(float(point_rc[1])), 0, shape[1] - 1)),
    )


def _ray_cost(
    maps: Mapping[str, np.ndarray],
    position_rc: np.ndarray,
    direction_rc: np.ndarray,
    *,
    horizon_cells: int,
    hard_margin_m: float,
) -> Tuple[float, bool, float]:
    """Mean sampled risk, feasibility, and min hazard clearance along one ray.

    Samples at integer distances 1..horizon; the float point is bounds-checked
    BEFORE rounding; the cell's risk is recorded BEFORE the hard/clearance
    feasibility check breaks the walk (both order-sensitive — preserved).
    """
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    values: List[float] = []
    min_clearance = float("inf")
    feasible = True
    for distance in range(1, horizon_cells + 1):
        query = position_rc + float(distance) * direction_rc
        if not (0.0 <= query[0] < risk.shape[0] and 0.0 <= query[1] < risk.shape[1]):
            feasible = False
            break
        cell = _clip_rc(query, risk.shape)
        values.append(float(risk[cell]))
        clearance = float(sdf[cell])
        min_clearance = min(min_clearance, clearance)
        if bool(hard[cell]) or clearance < hard_margin_m:
            feasible = False
            break
    mean_risk = float(np.mean(values)) if values else float("inf")
    return mean_risk, feasible, min_clearance


def primitive_feasibility_gate(
    maps: Mapping[str, np.ndarray],
    position_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    primitive_count: int = 16,
    horizon_cells: int = 12,
    hard_margin_m: float = 1.0,
    improvement_margin: float = 0.05,
    material_trigger: float = 0.45,
) -> GateDecision:
    """Test whether a feasible, progress-making ray improves on the nominal ray.

    Candidate rays are uniform over 360 degrees, ``[sin t, cos t]`` in (row,
    col).  A ray is eligible only if its endpoint makes at least half a cell
    of progress toward the goal and every sample clears hard terrain.  The
    gate activates iff a feasible ray exists, the nominal (straight-to-goal)
    ray's mean risk reaches ``material_trigger``, and the best ray improves on
    it by ``improvement_margin`` (mean-ray-risk units).

    ``maps`` needs: risk_map float [0,1], hard_mask, sdf_hard (metres).
    ``position_xy``/``goal_xy`` are (x=col, y=row) — reversed internally.
    """
    position_rc = position_xy[::-1].astype(np.float32)
    goal_rc = goal_xy[::-1].astype(np.float32)
    nominal_direction = _unit(goal_rc - position_rc)
    nominal_risk, _, _ = _ray_cost(
        maps,
        position_rc,
        nominal_direction,
        horizon_cells=horizon_cells,
        hard_margin_m=hard_margin_m,
    )

    best_risk = float("inf")
    best_direction = np.zeros(2, dtype=np.float32)
    best_min_clearance = float("nan")
    feasible_count = 0
    current_goal_distance = float(np.linalg.norm(goal_rc - position_rc))
    for index in range(primitive_count):
        angle = 2.0 * math.pi * float(index) / float(primitive_count)
        direction = np.asarray([math.sin(angle), math.cos(angle)], dtype=np.float32)
        endpoint = position_rc + float(horizon_cells) * direction
        if float(np.linalg.norm(goal_rc - endpoint)) >= current_goal_distance - 0.5:
            continue
        candidate_risk, feasible, candidate_min_clearance = _ray_cost(
            maps,
            position_rc,
            direction,
            horizon_cells=horizon_cells,
            hard_margin_m=hard_margin_m,
        )
        if not feasible:
            continue
        feasible_count += 1
        if candidate_risk < best_risk:
            best_risk = candidate_risk
            best_direction = direction
            best_min_clearance = candidate_min_clearance

    active = (
        feasible_count > 0
        and nominal_risk >= material_trigger
        and nominal_risk - best_risk >= improvement_margin
    )
    return GateDecision(
        active=bool(active),
        nominal_risk=nominal_risk,
        best_risk=best_risk,
        feasible_count=feasible_count,
        selected_direction_rc=(float(best_direction[0]), float(best_direction[1])),
        selected_endpoint_rc=(
            float(position_rc[0] + horizon_cells * best_direction[0]),
            float(position_rc[1] + horizon_cells * best_direction[1]),
        ),
        selected_min_clearance_m=best_min_clearance,
    )


__all__ = [
    "RiskPatchEncoder",
    "CoefEnergyNetMaterial",
    "load_geometry_weights",
    "sdf_barrier_grad",
    "bilinear_sample_patch",
    "integrate_surrogate_material",
    "GateDecision",
    "primitive_feasibility_gate",
]
