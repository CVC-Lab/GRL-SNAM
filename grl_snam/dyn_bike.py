"""The dynamic single-track vehicle, differentiable in torch.

The torch twin of libcvc's ``cvc::nav::diff::dyn_plant`` (``inc/cvc/nav/detail/dyn_bike.h``,
P8a of the vehicle-physics roadmap): the plant the trained controller drives in place of the
kinematic bicycle. libcvc differentiates it with forward-mode duals; here autograd does, so the
two gradients are independent derivations of one model, the parity gate
(``tests/test_dyn_bike_parity.py``, against libcvc's ``dyn_bike_golden.json``).

State per agent, in nav's normalized frame (lengths x S, time in s, mass in kg), the last axis
of a ``(..., 8)`` tensor in this order:

- ``x, y``: the rear axle; ``psi``: the heading;
- ``vx, vy``: the body-frame velocity at the centre of gravity; ``r``: the yaw rate;
- ``delta``: the road-wheel steering angle; ``a``: the drive or brake force per unit mass, both
  behind their actuators.

One step of ``hdt`` from the commands ``(delta_cmd, a_cmd)`` and the grade along the heading:

1. actuators: exact first-order lags, the steering's change limited smoothly to
   ``steer_rate * hdt``;
2. loads: quasi-static longitudinal load transfer, floored at ``fz_min * m * g``;
3. the speed, predicted from the drive or brake force (each axle saturated smoothly at
   ``mu Fz``), drag, rolling resistance and grade;
4. ``(vy, r)`` by implicit Euler, ``iters`` steps, each tire linearized by its secant through its
   root in lateral velocity (Pacejka forces within the friction ellipse's remainder);
5. the speed corrected with the front lateral force the implicit step applied;
6. below ``v_blend``, ``(vy, r)`` blended toward the kinematic bicycle's;
7. the pose: the rear axle moves with ``(vx, vy - lr r)`` at the new heading.

Every branch is a ``torch.where`` on values, as the C++ branches on ``s_val``; the division in
the unselected branch is guarded so it never turns a gradient into NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

ORDER = ("x", "y", "psi", "vx", "vy", "r", "delta", "a")


@dataclass(frozen=True)
class DynBikeParams:
    """``cvc::nav::diff::dyn_bike_params``, in the normalized frame."""

    m: float = 1.0
    Iz: float = 1.0
    lf: float = 0.5
    lr: float = 0.5
    h: float = 0.0
    g: float = 9.81
    mu: float = 0.9
    B_f: float = 12.0
    B_r: float = 12.0
    C: float = 1.2
    front_drive: float = 0.0
    front_brake: float = 0.6
    drag: float = 0.0
    c_rr: float = 0.0
    tau_steer: float = 0.0
    steer_rate: float = 0.0
    tau_drive: float = 0.0
    v_blend: float = 0.0
    v_width: float = 1.0
    eps: float = 0.1
    fz_min: float = 0.05
    iters: int = 3


@dataclass(frozen=True)
class PlantLimits:
    """The kinematic limits the plant shares with the controller (``bike_veh``)."""

    L: float
    vmax: float
    hdt: float
    allow_reverse: bool = False


def _sat(F: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    """F / (1 + (F / cap)^8)^(1/8)."""
    q = (F * F) / (cap * cap)
    q4 = (q * q) * (q * q)
    return F / torch.sqrt(torch.sqrt(torch.sqrt(1.0 + q4)))


def _lateral(alpha, B, C, D):
    """Pacejka's lateral force, D sin(C atan(B alpha))."""
    return D * torch.sin(C * torch.atan(B * alpha))


def _secant_w(F, alpha, w, w0, B, C, D, u):
    """-F / (w - w0), the tire's secant through its root; its slope there near zero slip."""
    ba = B * alpha
    small = ba * ba < 1e-8
    den = torch.where(small, torch.ones_like(w), w - w0)
    return torch.where(small, D * C * B * u / (u * u + w * w), -F / den)


def _clamp(x, lo: float, hi: float):
    return torch.where(
        x < lo, torch.full_like(x, lo), torch.where(x > hi, torch.full_like(x, hi), x)
    )


def dyn_plant(
    p: DynBikeParams,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    grade: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """One step of the plant: ``state`` ``(..., 8)`` to the next ``(..., 8)``."""
    x, y, psi, vx, vy, r, delta, a = state.unbind(-1)
    grade = torch.as_tensor(grade, dtype=state.dtype, device=state.device)
    hdt = lim.hdt
    m, Iz, lf, lr, h, g, mu, Cp = p.m, p.Iz, p.lf, p.lr, p.h, p.g, p.mu, p.C
    L = lf + lr

    # 1. Actuators.
    k_steer = 1.0 - math.exp(-hdt / p.tau_steer) if p.tau_steer > 0.0 else 1.0
    step = (delta_cmd - delta) * k_steer
    if p.steer_rate > 0.0:
        R = p.steer_rate * hdt
        step = R * torch.tanh(step / R)
    delta1 = delta + step
    k_drive = 1.0 - math.exp(-hdt / p.tau_drive) if p.tau_drive > 0.0 else 1.0
    a1 = a + (a_cmd - a) * k_drive

    # 2. Loads.
    fz_floor = p.fz_min * m * g
    Fzf = m * (g * lr - a1 * h) / L
    Fzr = m * (g * lf + a1 * h) / L
    Fzf = torch.where(Fzf < fz_floor, torch.full_like(Fzf, fz_floor), Fzf)
    Fzr = torch.where(Fzr < fz_floor, torch.full_like(Fzr, fz_floor), Fzr)
    cap_f, cap_r = mu * Fzf, mu * Fzr
    F_drive = m * a1
    share = torch.where(
        a1 >= 0.0, torch.full_like(a1, p.front_drive), torch.full_like(a1, p.front_brake)
    )
    Fxf = _sat(F_drive * share, cap_f)
    Fxr = _sat(F_drive * (1.0 - share), cap_r)
    Dyf = torch.sqrt(cap_f * cap_f - 0.9975 * Fxf * Fxf)
    Dyr = torch.sqrt(cap_r * cap_r - 0.9975 * Fxr * Fxr)
    cd, sd = torch.cos(delta1), torch.sin(delta1)
    eps2 = p.eps * p.eps
    sp_min = -0.25 * lim.vmax if lim.allow_reverse else 0.0

    # 3. The speed, predicted without lateral forces.
    u0 = torch.sqrt(vx * vx + eps2)
    sg0 = vx / u0
    F_res = -p.drag * vx * u0 - p.c_rr * m * g * sg0 - m * g * grade
    vx_p = _clamp(vx + hdt * ((Fxf * cd + Fxr + F_res) / m), sp_min, lim.vmax)

    # 4. Lateral and yaw.
    u = torch.sqrt(vx_p * vx_p + eps2)
    sg = vx_p / u
    w0f = u * torch.tan(sg * delta1)
    vy1, r1 = vy, r
    Fyf_step = torch.zeros_like(vy)
    for _ in range(max(1, p.iters)):
        wf = vy1 + lf * r1
        wr = vy1 - lr * r1
        af = sg * delta1 - torch.atan2(wf, u)
        ar = -torch.atan2(wr, u)
        Fyf = _lateral(af, p.B_f, Cp, Dyf)
        Fyr = _lateral(ar, p.B_r, Cp, Dyr)
        FY = Fxf * sd + Fyf * cd + Fyr
        MZ = lf * (Fxf * sd + Fyf * cd) - lr * Fyr
        f_vy = FY / m - r1 * vx_p
        f_r = MZ / Iz
        Sf_t = _secant_w(Fyf, af, wf, w0f, p.B_f, Cp, Dyf, u)
        Sr = _secant_w(Fyr, ar, wr, torch.zeros_like(wr), p.B_r, Cp, Dyr, u)
        Sf = Sf_t * cd
        J11 = (-Sf - Sr) / m
        J12 = (-Sf * lf + Sr * lr) / m - vx_p
        J21 = (-Sf * lf + Sr * lr) / Iz
        J22 = (-Sf * lf * lf - Sr * lr * lr) / Iz
        A11, A12, A21, A22 = 1.0 - hdt * J11, -hdt * J12, -hdt * J21, 1.0 - hdt * J22
        det = A11 * A22 - A12 * A21
        det = torch.where(det < 0.05, torch.full_like(det, 0.05), det)
        b1 = vy + hdt * f_vy - vy1
        b2 = r + hdt * f_r - r1
        dvy = (A22 * b1 - A12 * b2) / det
        dr = (A11 * b2 - A21 * b1) / det
        vy1 = vy1 + dvy
        r1 = r1 + dr
        Fyf_step = Fyf - Sf_t * (dvy + lf * dr)

    # 5. The speed, corrected.
    FX = Fxf * cd - Fyf_step * sd + Fxr + F_res
    vx1 = _clamp(vx + hdt * (FX / m + r1 * vy1), sp_min, lim.vmax)

    # 6. Low speed.
    if p.v_blend > 0.0:
        w = 1.0 / (1.0 + torch.exp((torch.sqrt(vx1 * vx1 + eps2) - p.v_blend) / p.v_width))
        r_k = vx1 * torch.tan(delta1) / L
        vy_k = lr * r_k
        vy1 = vy1 + w * (vy_k - vy1)
        r1 = r1 + w * (r_k - r1)

    # 7. Pose.
    psi1 = psi + hdt * r1
    v_side = vy1 - lr * r1
    c1, s1 = torch.cos(psi1), torch.sin(psi1)
    x1 = x + hdt * (vx1 * c1 - v_side * s1)
    y1 = y + hdt * (vx1 * s1 + v_side * c1)
    return torch.stack((x1, y1, psi1, vx1, vy1, r1, delta1, a1), dim=-1)
