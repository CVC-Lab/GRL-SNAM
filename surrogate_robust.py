#!/usr/bin/env python3
"""
Radius-aware integrate_surrogate (adds robot radius margin) and a multi-start
robustness penalty that samples feasible starts near obstacles.

Drop-in usage in train_coef_energy.py (pseudo):
------------------------------------------------
from surrogate_robust import integrate_surrogate_v2, multi_start_penalty

# in Trainer.step_batch(...):
oT, vT, clr = integrate_surrogate_v2(o0, v0, goal, C, R, mask,
                                     alphas, beta, gamma,
                                     d_hat, dt_prime, H,
                                     robot_radius=robot_radius,
                                     margin_factor=self.cfg.margin_factor)
L_multi = multi_start_penalty(o0, v0, goal, C, R, mask,
                              alphas, beta, gamma,
                              d_hat, dt_prime, H,
                              robot_radius=robot_radius,
                              margin_factor=self.cfg.margin_factor,
                              ms_count=self.cfg.ms_count,
                              ms_h=self.cfg.ms_h,
                              ms_dt_mult=self.cfg.ms_dt_mult)

# add to loss: L += self.cfg.w_multi * L_multi
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

# You can import from train_coef_energy if available; otherwise include a local copy
try:
    from train_coef_energy import ipc_piecewise
except Exception:
    def ipc_piecewise(d: torch.Tensor, d_hat: torch.Tensor | float, vp: float = -5e2, eps: float = 1e-9,
                      max_grad: float = 200.0, max_b: float = 200.0):
        # Minimal compatible implementation (broadcasts d_hat)
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

@torch.no_grad()
def _nearest_obstacle(
    o: torch.Tensor,            # (B,2)
    C: torch.Tensor,            # (B,N,2)
    R_eff: torch.Tensor,        # (B,N)
    mask: torch.Tensor,         # (B,N) bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (dmin[B], nmin[B,2], jmin[B]) at o. Uses safe advanced indexing.
    """
    B, N = C.shape[:2]
    if N == 0:
        return (
            torch.full((B,), float('inf'), device=o.device, dtype=o.dtype),
            torch.zeros(B, 2, device=o.device, dtype=o.dtype),
            torch.zeros(B, dtype=torch.long, device=o.device),
        )

    diff = o.unsqueeze(1) - C                         # (B,N,2)
    r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)  # (B,N)
    d = r - R_eff                                     # (B,N)
    d = torch.where(mask, d, torch.full_like(d, 1e6))

    jmin = d.argmin(dim=1)                            # (B,)
    idx  = torch.arange(B, device=o.device)           # (B,)

    rmin = r[idx, jmin]                               # (B,)
    nmin = diff[idx, jmin, :] / rmin.unsqueeze(-1)    # (B,2)
    dmin = d[idx, jmin]                               # (B,)

    return dmin, nmin, jmin

def integrate_surrogate_v2(o0: torch.Tensor, v0: torch.Tensor, goal: torch.Tensor,
                           C: torch.Tensor, R: torch.Tensor, mask: torch.Tensor,
                           alphas: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                           d_hat: torch.Tensor, dt: torch.Tensor, H: torch.Tensor,
                           robot_radius: torch.Tensor | float = 0.0, margin_factor: float = 0.5,
                           mass: float = 1.0):
    """Radius-aware version of integrate_surrogate. R_eff = R + margin_factor * robot_radius."""
    B, N = C.shape[:2]
    if not torch.is_tensor(robot_radius):
        rr = o0.new_tensor(float(robot_radius))
    else:
        rr = robot_radius.to(device=o0.device, dtype=o0.dtype)
    R_eff = R + margin_factor * rr[:, None]

    o = o0.clone(); v = v0.clone()
    min_clear = torch.full((B,), float("inf"), dtype=o.dtype, device=o.device)
    for s in range(int(H.max().item())):
        active = (s < H).to(o.dtype).unsqueeze(-1)
        F_goal = -beta.unsqueeze(-1) * (o - goal)
        if N == 0:
            F_bar = torch.zeros_like(o); dmin = torch.full_like(min_clear, float('inf'))
        else:
            diff = o.unsqueeze(1) - C
            r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
            n_hat = diff / r.unsqueeze(-1)
            d = r - R_eff
            d = torch.where(mask, d, torch.full_like(d, 1e6))
            _, dbdd = ipc_piecewise(d, d_hat.view(-1,1))
            F_bar = -(alphas * dbdd).unsqueeze(-1) * n_hat
            F_bar = F_bar.sum(dim=1)
            dmin = torch.where(mask, d, torch.full_like(d, float('inf'))).min(dim=1).values
        min_clear = torch.minimum(min_clear, dmin)
        a = (F_bar + F_goal - gamma.unsqueeze(-1) * v) / float(mass)
        o = o + active * dt.unsqueeze(-1) * v
        v = v + active * dt.unsqueeze(-1) * a
    return o, v, min_clear

def multi_start_penalty(o0: torch.Tensor, v0: torch.Tensor, goal: torch.Tensor,
                        C: torch.Tensor, R: torch.Tensor, mask: torch.Tensor,
                        alphas: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
                        d_hat: torch.Tensor, dt_prime: torch.Tensor, H: torch.Tensor,
                        robot_radius: torch.Tensor | float = 0.0, margin_factor: float = 0.5,
                        ms_count: int = 3, ms_h: int = 2, ms_dt_mult: float = 1.5,
                        tau: float = 0.05) -> torch.Tensor:
    """Sample feasible starts near nearest obstacle and penalize penetrations after short rollouts."""
    dev = o0.device; B, N = C.shape[:2]
    if ms_count <= 0 or N == 0:
        return o0.new_tensor(0.0)
    # effective radius
    if not torch.is_tensor(robot_radius):
        rr = o0.new_tensor(float(robot_radius))
    else:
        rr = robot_radius.to(device=o0.device, dtype=o0.dtype)
    while rr.ndim < 2:
        rr = rr.unsqueeze(-1)
    R_eff = R + margin_factor * rr.squeeze(-1)

    L_acc = o0.new_tensor(0.0)
    dmin0, nmin0, _ = _nearest_obstacle(o0, C, R_eff, mask)

    for _ in range(ms_count):
        frac = 0.9
        step = (frac * dmin0).unsqueeze(-1) * nmin0
        o_ms = o0 - step  # move toward obstacle
        # ensure feasibility
        diff_ms = o_ms.unsqueeze(1) - C
        r_ms = torch.linalg.norm(diff_ms, dim=-1).clamp_min(1e-9)
        d_ms = torch.where(mask, r_ms - R_eff, torch.full_like(r_ms, 1e6))
        ok = (d_ms.min(dim=1).values >= 0)
        o_ms = torch.where(ok.unsqueeze(-1), o_ms, o0 + 0.5 * step)  # fallback if infeasible
        # short rollout
        H_ms = torch.full_like(H, ms_h)
        dt_ms = ms_dt_mult * dt_prime
        _, _, clr_ms = integrate_surrogate_v2(o_ms, v0, goal, C, R, mask, alphas, beta, gamma, d_hat,
                                              dt_ms, H_ms, robot_radius=rr.squeeze(-1), margin_factor=margin_factor)
        L_acc = L_acc + torch.nn.functional.softplus(((- clr_ms) / tau)).mean()
        
    return L_acc / float(ms_count)
