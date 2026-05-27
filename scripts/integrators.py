"""Reusable numerical integrators for energy-field and Ackermann rollouts.

The project has two related integration layers:

1. Local pH / energy-field surrogate in local coordinates.  This integrates
   (q, p) using a sampled force map F(q).  The default is semi-implicit Euler,
   but the template also exposes explicit Euler, midpoint, and velocity Verlet.

2. Ackermann kinematic realization in global coordinates.  This uses a hard
   projected actuator state (v, steering) and then advances the vehicle pose.

The API is intentionally small so higher-order schemes can be added by extending
``EnergyIntegratorConfig.scheme`` / ``ackermann_step_np`` without changing the
training or evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math
import numpy as np
import torch


Tensor = torch.Tensor


@dataclass
class EnergyIntegratorConfig:
    scheme: str = "semi_implicit"  # explicit | semi_implicit | midpoint | velocity_verlet
    dt: float = 0.10
    damping: float = 0.20
    momentum_clip: float = 1.50
    clamp_to_grid: bool = True

    def normalized_scheme(self) -> str:
        s = self.scheme.lower().replace("-", "_")
        aliases = {
            "symplectic_euler": "semi_implicit",
            "semiimplicit": "semi_implicit",
            "semi_implicit_euler": "semi_implicit",
            "verlet": "velocity_verlet",
            "vv": "velocity_verlet",
            "mid_point": "midpoint",
            "rk2": "midpoint",
        }
        s = aliases.get(s, s)
        if s not in {"explicit", "semi_implicit", "midpoint", "velocity_verlet"}:
            raise ValueError(f"unknown energy integrator {self.scheme!r}")
        return s


def clamp_local_q(q: Tensor, hat_d_vec: Tensor) -> Tensor:
    """Smoothly keep local q inside the sampled square grid."""
    h = hat_d_vec.to(q.device, q.dtype).view(-1, 1).clamp_min(1e-6)
    return h * torch.tanh(q / h)


def clip_momentum(p: Tensor, clip: float) -> Tensor:
    if clip is None or clip <= 0:
        return p
    n = torch.linalg.norm(p, dim=-1, keepdim=True).clamp_min(1e-8)
    return p * torch.clamp(float(clip) / n, max=1.0)


def integrate_local_energy_field(
    q0: Tensor,
    force_fn: Callable[[Tensor], Tensor],
    hat_d_vec: Tensor,
    cfg: EnergyIntegratorConfig,
    steps: int,
    p0: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Integrate a local pH/energy surrogate.

    Parameters
    ----------
    q0:
        Initial local coordinate, shape [B,2].
    force_fn:
        Differentiable callable returning F(q), shape [B,2].
    hat_d_vec:
        Per-sample local grid half-width, shape [B].
    cfg:
        Scheme/damping/step-size settings.
    steps:
        Number of integration steps.
    p0:
        Optional initial virtual momentum. If omitted, zero momentum is used.

    Returns
    -------
    dict with q_seq, p_seq, F_seq, each [B,steps,2].
    """
    K = int(steps)
    q = q0
    p = torch.zeros_like(q) if p0 is None else p0.to(q.device, q.dtype)
    dt = float(cfg.dt)
    damping = float(cfg.damping)
    scheme = cfg.normalized_scheme()

    qs, ps, Fs = [], [], []
    for _ in range(K):
        if scheme == "explicit":
            F0 = force_fn(q)
            # First-order gradient-flow style step. p is diagnostic only.
            p = (1.0 - damping * dt) * p + dt * F0
            p = clip_momentum(p, cfg.momentum_clip)
            q = q + dt * F0
            F_used = F0

        elif scheme == "semi_implicit":
            F0 = force_fn(q)
            # Kick then drift with implicit scalar damping:
            #   p_{k+1} = (p_k + dt F(q_k)) / (1 + dt D)
            #   q_{k+1} = q_k + dt p_{k+1}
            # This is the dissipative semi-implicit pH update used by the
            # training memo.  The previous (1 - dt*D)*p + dt*F formula used
            # explicit damping and can behave poorly when force norms grow.
            p = (p + dt * F0) / (1.0 + damping * dt)
            p = clip_momentum(p, cfg.momentum_clip)
            q = q + dt * p
            F_used = F0

        elif scheme == "midpoint":
            F0 = force_fn(q)
            p_mid = clip_momentum((p + 0.5 * dt * F0) / (1.0 + 0.5 * damping * dt), cfg.momentum_clip)
            q_mid = q + 0.5 * dt * p_mid
            if cfg.clamp_to_grid:
                q_mid = clamp_local_q(q_mid, hat_d_vec)
            F_mid = force_fn(q_mid)
            p = clip_momentum((p + dt * F_mid) / (1.0 + damping * dt), cfg.momentum_clip)
            q = q + dt * p
            F_used = F_mid

        else:  # velocity_verlet
            F0 = force_fn(q)
            p_half = clip_momentum((p + 0.5 * dt * F0) / (1.0 + 0.5 * damping * dt), cfg.momentum_clip)
            q_new = q + dt * p_half
            if cfg.clamp_to_grid:
                q_new = clamp_local_q(q_new, hat_d_vec)
            F1 = force_fn(q_new)
            p = clip_momentum((p_half + 0.5 * dt * F1) / (1.0 + 0.5 * damping * dt), cfg.momentum_clip)
            q = q_new
            F_used = F1

        if cfg.clamp_to_grid:
            q = clamp_local_q(q, hat_d_vec)
        qs.append(q)
        ps.append(p)
        Fs.append(F_used)

    B = q0.shape[0]
    empty = q0.new_zeros(B, 0, 2)
    return {
        "q_seq": torch.stack(qs, dim=1) if qs else empty,
        "p_seq": torch.stack(ps, dim=1) if ps else empty,
        "F_seq": torch.stack(Fs, dim=1) if Fs else empty,
    }


@dataclass
class AckermannIntegratorConfig:
    scheme: str = "semi_implicit"  # explicit | semi_implicit | midpoint
    wheelbase: float = 0.324
    dt: float = 0.10

    def normalized_scheme(self) -> str:
        s = self.scheme.lower().replace("-", "_")
        aliases = {
            "semiimplicit": "semi_implicit",
            "semi_implicit_euler": "semi_implicit",
            "symplectic_euler": "semi_implicit",
            "mid_point": "midpoint",
            "rk2": "midpoint",
        }
        s = aliases.get(s, s)
        if s not in {"explicit", "semi_implicit", "midpoint"}:
            raise ValueError(f"unknown Ackermann integrator {self.scheme!r}")
        return s


def wrap_angle_np(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))


def ackermann_pose_step_np(pose: np.ndarray, u: np.ndarray, cfg: AckermannIntegratorConfig) -> np.ndarray:
    """Advance a single Ackermann pose with an already-feasible actuator state."""
    v, delta = float(u[0]), float(u[1])
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    dt = float(cfg.dt)
    L = max(float(cfg.wheelbase), 1e-6)
    yaw = v / L * math.tan(delta)
    scheme = cfg.normalized_scheme()
    if scheme == "midpoint":
        th_eval = th + 0.5 * dt * yaw
        x = x + dt * v * math.cos(th_eval)
        y = y + dt * v * math.sin(th_eval)
        th = wrap_angle_np(th + dt * yaw)
    else:
        # With projected actuator state this is the semi-implicit actuator
        # analogue: update/use the new feasible u, then drift pose.
        x = x + dt * v * math.cos(th)
        y = y + dt * v * math.sin(th)
        th = wrap_angle_np(th + dt * yaw)
    return np.array([x, y, th], dtype=np.float32)


def ackermann_pose_step_np_vec(x, y, th, v, delta, cfg: AckermannIntegratorConfig):
    """Vectorized Ackermann step for MPC shooting arrays."""
    dt = float(cfg.dt)
    L = max(float(cfg.wheelbase), 1e-6)
    yaw = v / L * np.tan(delta)
    if cfg.normalized_scheme() == "midpoint":
        th_eval = th + 0.5 * dt * yaw
        x = x + dt * v * np.cos(th_eval)
        y = y + dt * v * np.sin(th_eval)
        th = wrap_angle_np(th + dt * yaw)
    else:
        x = x + dt * v * np.cos(th)
        y = y + dt * v * np.sin(th)
        th = wrap_angle_np(th + dt * yaw)
    return x, y, th
