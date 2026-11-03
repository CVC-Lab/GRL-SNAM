"""The four-wheel vehicle with suspension, differentiable in torch.

The torch twin of libcvc's ``cvc::nav::diff::dyn4_plant`` (``inc/cvc/nav/detail/dyn_4w.h``,
P8d of the vehicle-physics roadmap): P8a's single-track plant (:mod:`grl_snam.dyn_bike`) on four
contact patches, each with its own grip, its own terrain height and its own load from a
spring-damper corner. libcvc differentiates it with forward-mode duals; here autograd does, so
the two gradients are independent derivations of one model, the parity gate
(``tests/test_dyn4_parity.py``, against libcvc's ``dyn4_golden.json``).

State per agent, in nav's normalized frame (lengths x S, time in s, mass in kg), the last axis
of a ``(..., 14)`` tensor in this order:

- ``x, y``: the rear axle's midpoint; ``psi``: the heading;
- ``vx, vy``: the body-frame velocity at the centre of gravity; ``r``: the yaw rate;
- ``z, zd``: the body's heave from its rest height on flat ground, and its rate;
- ``phi, phid``: its roll (+ lifts the left side), and rate; ``th, thd``: its pitch (+ lowers
  the nose), and rate;
- ``delta``: the virtual centre wheel's steering angle; ``a``: the drive or brake force per unit
  mass, both behind their actuators.

The wheels sit at body-frame ``(x_i, y_i)`` from the centre of gravity: front left
``(lf, tf)``, front right ``(lf, -tf)``, rear left ``(-lr, tr)``, rear right ``(-lr, -tr)``.

The ground is a callable ``ground(wx, wy) -> (mu, height)`` on tensors of world positions, so
its gradients flow through autograd as libcvc's sampler's flow through its duals.
:class:`FlatGround` is the default.

One step of ``hdt`` from the commands ``(delta_cmd, a_cmd)``:

1. actuators, as P8a (the drive's target ``a_cmd`` at the default ``drive_gain``,
   ``brake_gain``, ``resist_comp``);
2. loads: each corner's spring-damper from the current vertical state, floored smoothly at zero
   (softplus of width ``fz_min * m * g``);
3. the drive or brake force split between the axles, then evenly left and right, each wheel
   saturated smoothly at ``mu_i Fz_i``; the speed predicted from these and the resistances;
4. ``(vy, r)`` by implicit Euler on four tires, the front pair at their Ackermann angles, each
   linearized by its secant through its root in its own lateral velocity;
5. the speed corrected; ``(z, phi, th)`` by implicit Euler on the corners' springs and dampers
   (a 3x3 solve), driven by the ground and the tire forces' moments about the centre of gravity;
6. below ``v_blend``, ``(vy, r)`` blended toward the kinematic bicycle's;
7. the pose: the rear axle moves with ``(vx, vy - lr r)`` at the new heading.

The corners' rates follow libcvc's float arithmetic (``k = 0`` is a stiff spring of natural
frequency ``1 / hdt``, ``c = 0`` critical damping), so the two agree to the last bits. Every
branch is a ``torch.where`` on values, as the C++ branches on ``s_val``. libcvc's opt-in
``grade_force`` (gravity along the ground's slope) is not twinned: a plant that sets it raises.

The wheel-spin plant (libcvc's ``dyn4_spin_plant``, opt-in there): :mod:`grl_snam.dyn_bike`'s
wheel spin on four wheels, the state ``(..., 18)`` with each wheel's speed (front left, front
right, rear left, rear right) after the fourteen above. Each wheel's drive or brake torque is its
axle's share halved between its two wheels, and its tire's longitudinal force Pacejka's of its
slip ``r_w om - T_i^T xi``, ``T_i^T xi`` the contact patch's speed along its wheel, roll and pitch
rates included. Step 3 solves the speed and the four wheels' speeds together by implicit Euler on
the forces' secants (:func:`dyn4_spin_plant`). In port-Hamiltonian form the wheels' angular
momenta join the state ``(q, M xi, J om)``, the longitudinal secants join ``R`` and the wheel
torques are the port, so

    dH/dt = supplied - braked - dissipated - slip + lever + grade (+ the ground's),

each term returned by :func:`dyn4_spin_ph_rates_at`, and :func:`dyn4_spin_plant_books` books
one step: the trapezoid of those powers between the step's start and the state it reached before
its limiters, on the ground it stood on, the energy the limiters and the ground's new heights
moved, and the integrator's residual. The parity gate is ``tests/test_dyn4_spin_ph_parity.py``,
against libcvc's ``dyn4_spin_ph_golden.json``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from grl_snam.dyn_bike import (
    PlantLimits,
    _clamp,
    _clamp_cap,
    _lateral,
    _lateral_cap,
    _longitudinal,
    _sat,
    _secant_w,
    _secant_x,
    _spin_secant,
)

__all__ = [
    "ORDER",
    "SPIN_ORDER",
    "Dyn4Params",
    "Dyn4SpinPHBooks",
    "Dyn4SpinPHParts",
    "Dyn4SpinPHRates",
    "Dyn4Wheels",
    "FlatGround",
    "PlantLimits",
    "ackermann",
    "dyn4_ph_hamiltonian",
    "dyn4_plant",
    "dyn4_spin_ph_hamiltonian",
    "dyn4_spin_ph_parts_at",
    "dyn4_spin_ph_rates_at",
    "dyn4_spin_plant",
    "dyn4_spin_plant_books",
]

ORDER = (
    "x",
    "y",
    "psi",
    "vx",
    "vy",
    "r",
    "z",
    "zd",
    "phi",
    "phid",
    "th",
    "thd",
    "delta",
    "a",
)
SPIN_ORDER = ORDER + ("om_fl", "om_fr", "om_rl", "om_rr")

Ground = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class Dyn4Params:
    """``cvc::nav::diff::dyn4_params``, in the normalized frame."""

    m: float = 1.0
    Iz: float = 1.0
    Ix: float = 1.0
    Iy: float = 1.0
    lf: float = 0.5
    lr: float = 0.5
    tf: float = 0.0
    tr: float = 0.0
    h: float = 0.0
    g: float = 9.81
    k_f: float = 0.0
    k_r: float = 0.0
    c_f: float = 0.0
    c_r: float = 0.0
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
    fz_min: float = 0.02
    iters: int = 3
    grade_force: int = 0  # 1: gravity along the ground's slope (libcvc's dyn4_grade); not twinned
    # The drive's answer to the command (step 1): its gain driving and braking, and the share of
    # the drag and rolling resistance it makes up. The defaults are the command as it is.
    drive_gain: float = 1.0
    brake_gain: float = 1.0
    resist_comp: float = 0.0
    # The wheels, for dyn4_spin_plant (dyn4_plant reads none of these): the rolling radius, each
    # front and each rear wheel's spin inertia, and the longitudinal tire (DynBikeParams').
    r_w: float = 0.4
    J_f: float = 1.0
    J_r: float = 1.0
    B_x: float = 12.0
    C_x: float = 1.2


@dataclass(frozen=True)
class FlatGround:
    """Flat ground of one friction (libcvc's ``flat_ground``, whose ``mu`` is a float)."""

    mu: float = 0.9

    def __call__(self, wx: torch.Tensor, wy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(wx, self.mu), torch.zeros_like(wx)


@dataclass(frozen=True)
class Dyn4Wheels:
    """``dyn4_wheels``: the four wheels at a state, front left, front right, rear left, rear
    right."""

    xi: tuple[float, ...]  # body-frame offsets from the centre of gravity
    yi: tuple[float, ...]
    Fz0: tuple[float, ...]  # static loads
    k: tuple[float, ...]  # the corners' springs and dampers, in libcvc's float arithmetic
    c: tuple[float, ...]
    mu: list[torch.Tensor]  # the ground under each wheel: its friction and height
    hg: list[torch.Tensor]
    Fz: list[torch.Tensor]  # the normal loads


def ackermann(L, y, tan_delta):
    """A front wheel's angle at body-frame ``y``: tan delta_i = L tan delta / (L - y tan delta)."""
    return torch.atan(L * tan_delta / (L - y * tan_delta))


def _corner_rates(p: Dyn4Params, hdt: float) -> tuple[list[float], list[float]]:
    """Each corner's spring and damper, in libcvc's float arithmetic."""
    f32 = np.float32
    m_corner = f32(0.25) * f32(p.m)
    k, c = [], []
    for kc, cc in ((p.k_f, p.c_f), (p.k_f, p.c_f), (p.k_r, p.c_r), (p.k_r, p.c_r)):
        if f32(kc) > 0.0:
            kk = f32(kc)
        else:
            w = f32(1.0) / f32(hdt)
            kk = m_corner * w * w
        cd = f32(cc) if f32(cc) > 0.0 else f32(2.0) * np.sqrt(kk * m_corner)
        k.append(float(kk))
        c.append(float(cd))
    return k, c


def _soft_pos(F: torch.Tensor, e: float) -> torch.Tensor:
    """e log(1 + exp(F / e)), and F itself above 30 e."""
    q = F / e
    big = q > 30.0
    q_safe = torch.where(big, torch.zeros_like(q), q)
    return torch.where(big, F, e * torch.log(1.0 + torch.exp(q_safe)))


def _det3(A):
    return (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )


def _no_grade_force(p: Dyn4Params) -> None:
    if p.grade_force:
        raise NotImplementedError("grade_force (libcvc's dyn4_grade) has no torch twin")


def _wheel_state(p: Dyn4Params, hdt: float, state: torch.Tensor, ground: Ground) -> Dyn4Wheels:
    """``dyn4_wheel_state``: the wheels at a state ``(..., 14)``, their loads from its vertical
    state and the ground under them at its pose."""
    x, y, psi = state[..., 0], state[..., 1], state[..., 2]
    z, zd, phi, phid, th, thd = (state[..., k] for k in range(6, 12))
    m, g, lf, lr = p.m, p.g, p.lf, p.lr
    L = lf + lr
    xi = (lf, lf, -lr, -lr)
    yi = (p.tf, -p.tf, p.tr, -p.tr)
    Fz0 = tuple(0.5 * m * g * (lr if i < 2 else lf) / L for i in range(4))
    k, c = _corner_rates(p, hdt)
    cps, sps = torch.cos(psi), torch.sin(psi)
    cgx, cgy = x + lr * cps, y + lr * sps
    mu_i, hg = [], []
    for i in range(4):
        mu, height = ground(cgx + xi[i] * cps - yi[i] * sps, cgy + xi[i] * sps + yi[i] * cps)
        mu_i.append(mu)
        hg.append(height)
    e = p.fz_min * m * g
    Fz = []
    for i in range(4):
        defl = hg[i] - (z + yi[i] * phi - xi[i] * th)
        rate = zd + yi[i] * phid - xi[i] * thd
        Fz.append(_soft_pos(Fz0[i] + k[i] * defl - c[i] * rate, e))
    return Dyn4Wheels(xi=xi, yi=yi, Fz0=Fz0, k=tuple(k), c=tuple(c), mu=mu_i, hg=hg, Fz=Fz)


def _drive_target(p: Dyn4Params, vx: torch.Tensor, a_cmd: torch.Tensor) -> torch.Tensor:
    """``dyn4_drive_target``: the command at drive_gain (brake_gain braking) and resist_comp of
    the drag and rolling resistance per unit mass; ``a_cmd`` itself at the defaults."""
    target = a_cmd
    if p.drive_gain != 1.0 or p.brake_gain != 1.0:
        target = torch.where(a_cmd >= 0.0, p.drive_gain * a_cmd, p.brake_gain * a_cmd)
    if p.resist_comp != 0.0:
        u0 = torch.sqrt(vx * vx + p.eps * p.eps)
        target = target + p.resist_comp * (p.drag * vx * u0 / p.m + p.c_rr * p.g * (vx / u0))
    return target


# ── Wheel spin (dyn4_spin_plant) ────────────────────────────────────────────────────────────


def _spin_torques(p: Dyn4Params, a: torch.Tensor):
    """``dyn4_spin_torques``: each wheel's drive torque and brake torque magnitude for the
    actuator's ``a``, its axle's share of the single track's halved between its two wheels."""
    drive = a >= 0.0
    share = torch.where(drive, torch.full_like(a, p.front_drive), torch.full_like(a, p.front_brake))
    T = 0.5 * p.r_w * p.m * a
    zero = torch.zeros_like(a)
    Td, Tb = [], []
    for i in range(4):
        Ti = T * (share if i < 2 else 1.0 - share)
        Td.append(torch.where(drive, Ti, zero))
        Tb.append(torch.where(drive, zero, -Ti))
    return Td, Tb


def _spin_long_step(p: Dyn4Params, hdt, state, om0, a1, xi, yi, cdw, sdw, cap, F_ext):
    """``dyn4_spin_long_step``: (vx, om_1..4) by implicit Euler on the tires' longitudinal forces
    and the wheels' torques, as the single track's on four wheels; each wheel couples to vx alone,
    so the solve is an arrowhead's. Each patch's speed along its wheel takes everything but vx
    from the step's start. Gives the speed (unclamped), the wheels' new speeds ``(..., 4)`` and the
    forces the last linearization applied."""
    m, h, rw = p.m, p.h, p.r_w
    eps2 = p.eps * p.eps
    oe2 = eps2 / (rw * rw)
    Jw = (p.J_f, p.J_f, p.J_r, p.J_r)
    Td, Tb = _spin_torques(p, a1)
    vx0, vy, r, phid, thd = (state[..., k] for k in (3, 4, 5, 9, 11))
    om0 = om0.unbind(-1)
    side = [
        -r * yi[i] * cdw[i] + (vy + r * xi[i]) * sdw[i] + h * (phid * sdw[i] - thd * cdw[i])
        for i in range(4)
    ]
    om = list(om0)
    Fx = [torch.zeros_like(vx0)] * 4
    vx = vx0
    for _ in range(max(1, p.iters)):
        b0 = m * (vx0 - vx) + hdt * F_ext
        A00, num, den = m, 0.0, 0.0
        F, K, A0, Ai, bi = [], [], [], [], []
        for i in range(4):
            gi = vx * cdw[i] + side[i]
            u = torch.sqrt(gi * gi + eps2)
            sig = rw * om[i] - gi
            F.append(_longitudinal(sig, u, p.B_x, p.C_x, cap[i]))
            K.append(_secant_x(F[i], sig, p.B_x, p.C_x, cap[i], u))
            Kb = Tb[i] / torch.sqrt(om[i] * om[i] + oe2)
            b0 = b0 + hdt * F[i] * cdw[i]
            bi.append(Jw[i] * (om0[i] - om[i]) + hdt * (Td[i] - Kb * om[i] - rw * F[i]))
            A00 = A00 + hdt * K[i] * cdw[i] * cdw[i]
            A0.append(-hdt * K[i] * cdw[i] * rw)
            Ai.append(Jw[i] + hdt * (K[i] * rw * rw + Kb))
            num = num + A0[i] * bi[i] / Ai[i]
            den = den + A0[i] * A0[i] / Ai[i]
        d0 = (b0 - num) / (A00 - den)
        for i in range(4):
            di = (bi[i] - A0[i] * d0) / Ai[i]
            om[i] = om[i] + di
            Fx[i] = F[i] + K[i] * (rw * di - cdw[i] * d0)
        vx = vx + d0
    return vx, torch.stack(om, dim=-1), Fx


def _plant(p: Dyn4Params, lim: PlantLimits, state, delta_cmd, a_cmd, ground, om0):
    """``dyn4_plant``: one step of the plant (``om0`` None) or of the wheel-spin plant (``om0``
    the wheels' speeds ``(..., 4)``). Gives the next ``(..., 14)``, the twist the step reached
    before its limiters ``(..., 3)``, and the wheels' new speeds (None without them)."""
    _no_grade_force(p)
    if ground is None:
        ground = FlatGround()
    x, y, psi, vx, vy, r, z, zd, phi, phid, th, thd, delta, a = state.unbind(-1)
    hdt = lim.hdt
    m, Iz, Ix, Iy, lf, lr = p.m, p.Iz, p.Ix, p.Iy, p.lf, p.lr
    h, g, Cp = p.h, p.g, p.C
    L = lf + lr

    # 1. Actuators.
    k_steer = 1.0 - math.exp(-hdt / p.tau_steer) if p.tau_steer > 0.0 else 1.0
    step = (delta_cmd - delta) * k_steer
    if p.steer_rate > 0.0:
        R = p.steer_rate * hdt
        step = R * torch.tanh(step / R)
    delta1 = delta + step
    k_drive = 1.0 - math.exp(-hdt / p.tau_drive) if p.tau_drive > 0.0 else 1.0
    a1 = a + (_drive_target(p, vx, a_cmd) - a) * k_drive

    # 2. The wheels, the ground under them and their loads.
    W = _wheel_state(p, hdt, state, ground)
    xi, yi, Fz0, k, c, hg, Fz = W.xi, W.yi, W.Fz0, W.k, W.c, W.hg, W.Fz

    # 3. Longitudinal. With the wheels spinning, the tires' longitudinal forces from their slip,
    # solved with the speed and the wheels'.
    F_drive = m * a1
    share = torch.where(
        a1 >= 0.0, torch.full_like(a1, p.front_drive), torch.full_like(a1, p.front_brake)
    )
    td = torch.tan(delta1)
    Fx, Dy, cdw, sdw, cap = [], [], [], [], []
    for i in range(4):
        axle_share = share if i < 2 else 1.0 - share
        cap.append(W.mu[i] * Fz[i])
        if om0 is None:
            Fx.append(_sat(0.5 * F_drive * axle_share, cap[i]))
            Dy.append(torch.sqrt(cap[i] * cap[i] - 0.9975 * Fx[i] * Fx[i]))
        dw = ackermann(L, yi[i], td) if i < 2 else torch.zeros_like(td)
        cdw.append(torch.cos(dw))
        sdw.append(torch.sin(dw))
    eps2 = p.eps * p.eps
    sp_min = float(np.float32(-0.25) * np.float32(lim.vmax)) if lim.allow_reverse else 0.0
    u0 = torch.sqrt(vx * vx + eps2)
    sg0 = vx / u0
    F_res = -p.drag * vx * u0 - p.c_rr * m * g * sg0
    om1 = None
    if om0 is None:
        FXlong = torch.zeros_like(vx)
        for i in range(4):
            FXlong = FXlong + Fx[i] * cdw[i]
        vx_p = _clamp(vx + hdt * ((FXlong + F_res) / m), sp_min, lim.vmax)
    else:
        vx_s, om1, Fx = _spin_long_step(p, hdt, state, om0, a1, xi, yi, cdw, sdw, cap, F_res)
        vx_p = _clamp(vx_s, sp_min, lim.vmax)
        # A linearization's force may pass the grip by its error: the ellipse reads it within.
        Fx = [_clamp_cap(Fx[i], cap[i]) for i in range(4)]
        Dy = [_lateral_cap(Fx[i], cap[i]) for i in range(4)]

    # 4. Lateral and yaw: implicit Euler for (vy, r), four tires.
    Bw = [p.B_f, p.B_f, p.B_r, p.B_r]
    vy1, r1 = vy, r
    Fy_applied = [torch.zeros_like(vy)] * 4
    zero = torch.zeros_like(vy)
    for _ in range(max(1, p.iters)):
        FY, MZ = zero, zero
        J11, J12, J21, J22 = zero, zero, zero, zero
        Fy, St, dwdvy, dwdr = [], [], [], []
        for i in range(4):
            vxi = vx_p - r1 * yi[i]
            vyi = vy1 + r1 * xi[i]
            wi = -vxi * sdw[i] + vyi * cdw[i]
            fwd = vxi * cdw[i] + vyi * sdw[i]
            ui = torch.sqrt(fwd * fwd + eps2)
            alpha = -torch.atan2(wi, ui)
            Fy.append(_lateral(alpha, Bw[i], Cp, Dy[i]))
            St.append(_secant_w(Fy[i], alpha, wi, torch.zeros_like(wi), Bw[i], Cp, Dy[i], ui))
            dwdvy.append(cdw[i])
            dwdr.append(yi[i] * sdw[i] + xi[i] * cdw[i])
            fby = Fy[i] * cdw[i] + Fx[i] * sdw[i]
            fbx = -Fy[i] * sdw[i] + Fx[i] * cdw[i]
            FY = FY + fby
            MZ = MZ + xi[i] * fby - yi[i] * fbx
            dfy_y = -St[i] * cdw[i]
            dfx_y = St[i] * sdw[i]
            J11 = J11 + dfy_y * dwdvy[i]
            J12 = J12 + dfy_y * dwdr[i]
            J21 = J21 + (xi[i] * dfy_y - yi[i] * dfx_y) * dwdvy[i]
            J22 = J22 + (xi[i] * dfy_y - yi[i] * dfx_y) * dwdr[i]
        f_vy = FY / m - r1 * vx_p
        f_r = MZ / Iz
        A11 = 1.0 - hdt * (J11 / m)
        A12 = -hdt * (J12 / m - vx_p)
        A21 = -hdt * (J21 / Iz)
        A22 = 1.0 - hdt * (J22 / Iz)
        det = A11 * A22 - A12 * A21
        det = torch.where(det < 0.05, torch.full_like(det, 0.05), det)
        b1 = vy + hdt * f_vy - vy1
        b2 = r + hdt * f_r - r1
        dvy = (A22 * b1 - A12 * b2) / det
        dr = (A11 * b2 - A21 * b1) / det
        vy1 = vy1 + dvy
        r1 = r1 + dr
        Fy_applied = [Fy[i] - St[i] * (dwdvy[i] * dvy + dwdr[i] * dr) for i in range(4)]

    # 3, corrected: the speed with the lateral forces the implicit step applied.
    FXb, FYb = F_res, zero
    for i in range(4):
        FXb = FXb + Fx[i] * cdw[i] - Fy_applied[i] * sdw[i]
        FYb = FYb + Fy_applied[i] * cdw[i] + Fx[i] * sdw[i]
    vx1_free = vx + hdt * (FXb / m + r1 * vy1)
    vx1 = _clamp(vx1_free, sp_min, lim.vmax)
    free = torch.stack((vx1_free, vy1, r1), dim=-1)

    # 5. Vertical: implicit Euler on (z, phi, th) with the corners' linear springs and dampers.
    Kq = [[0.0] * 3 for _ in range(3)]
    Dq = [[0.0] * 3 for _ in range(3)]
    Gv = [-m * g + zero, h * FYb, -h * (FXb - F_res)]
    for i in range(4):
        ci = (1.0, yi[i], -xi[i])
        for aa in range(3):
            Gv[aa] = Gv[aa] + ci[aa] * (Fz0[i] + k[i] * hg[i])
            for bb in range(3):
                Kq[aa][bb] = Kq[aa][bb] + k[i] * ci[aa] * ci[bb]
                Dq[aa][bb] = Dq[aa][bb] + c[i] * ci[aa] * ci[bb]
    Mv = (m, Ix, Iy)
    q0, qd0 = (z, phi, th), (zd, phid, thd)
    A = [[0.0] * 3 for _ in range(3)]
    rhs = [None] * 3
    for aa in range(3):
        rhs[aa] = Mv[aa] * qd0[aa] + hdt * Gv[aa]
        for bb in range(3):
            A[aa][bb] = hdt * Dq[aa][bb] + hdt * hdt * Kq[aa][bb] + (Mv[aa] if aa == bb else 0.0)
            rhs[aa] = rhs[aa] - hdt * Kq[aa][bb] * q0[bb]
    det3 = _det3(A)
    qd1 = []
    for col in range(3):
        Ac = [[rhs[aa] if bb == col else A[aa][bb] for bb in range(3)] for aa in range(3)]
        qd1.append(_det3(Ac) / det3)

    # 6. Low speed: toward the kinematic bicycle.
    if p.v_blend > 0.0:
        w = torch.sigmoid(-(torch.sqrt(vx1 * vx1 + eps2) - p.v_blend) / p.v_width)
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
    nxt = torch.stack(
        (
            x1,
            y1,
            psi1,
            vx1,
            vy1,
            r1,
            z + hdt * qd1[0],
            qd1[0],
            phi + hdt * qd1[1],
            qd1[1],
            th + hdt * qd1[2],
            qd1[2],
            delta1,
            a1,
        ),
        dim=-1,
    )
    return nxt, free, om1


def dyn4_plant(
    p: Dyn4Params,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    ground: Ground | None = None,
) -> torch.Tensor:
    """One step of the plant: ``state`` ``(..., 14)`` to the next ``(..., 14)``."""
    nxt, _, _ = _plant(p, lim, state, delta_cmd, a_cmd, ground, None)
    return nxt


def dyn4_spin_plant(
    p: Dyn4Params,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    ground: Ground | None = None,
    *,
    with_free: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One step of the wheel-spin plant: ``state`` ``(..., 18)`` (:data:`SPIN_ORDER`) to the next
    ``(..., 18)``, the wheels' speeds advanced with the speed. The low-speed blend and the speed
    clamp act on the body alone. With ``with_free``, also the twist ``(vx, vy, r)`` ``(..., 3)``
    the step reached before its limiters (libcvc's ``dyn_step_free``).
    """
    nxt, free, om1 = _plant(p, lim, state[..., :14], delta_cmd, a_cmd, ground, state[..., 14:])
    nxt = torch.cat((nxt, om1), dim=-1)
    return (nxt, free) if with_free else nxt


# ── The wheel-spin plant in port-Hamiltonian form ───────────────────────────────────────────
# libcvc's dyn4_spin_ph_*: x = (q, M xi, J om), thirteen coordinates, H = dyn4_ph_hamiltonian +
# sum_i 1/2 J_i om_i^2; R = dyn4_ph_R (each tire's lateral grip what its longitudinal force leaves)
# + sum_i K_i d_i d_i^T, d_i^T dH/dx = r_w om_i - T_i^T xi; the port the wheel torques at the
# wheels' speeds; f_lever the lateral forces' moments on roll and pitch. Each function mirrors its
# namesake's arithmetic, sums in the same order, so the double tier holds to rounding.


@dataclass(frozen=True)
class Dyn4SpinPHParts:
    """``dyn4_spin_ph_parts``: everything the four-wheel wheel-spin form is built from, at a
    state; per wheel ``(..., 4)``, in :class:`Dyn4Wheels`' order."""

    W: Dyn4Wheels  # the wheels: offsets, static loads, springs, dampers, the ground, the loads
    cx: torch.Tensor  # drag and rolling resistance on vx
    Fx: torch.Tensor  # each tire's longitudinal force, from its slip
    Fy: torch.Tensor  # each tire's lateral force
    w: torch.Tensor  # each wheel's lateral velocity in its own frame
    ui: torch.Tensor  # and its floored forward speed
    Sw: torch.Tensor  # the lateral secants through w = 0
    N: torch.Tensor  # (..., 4, 6): each tire's lateral force on (vx, vy, r, zd, phid, thd)
    T: torch.Tensor  # (..., 4, 6): each wheel's longitudinal force on the twist
    g: torch.Tensor  # each patch's ground speed along its wheel, T_i^T xi
    sig: torch.Tensor  # its slip velocity, r_w om - T_i^T xi
    K: torch.Tensor  # the longitudinal secants through sigma = 0
    Td: torch.Tensor  # the drive's torques
    Tb: torch.Tensor  # the brakes' magnitudes
    sg: torch.Tensor  # the brakes' smoothed signs of the wheels' speeds
    u: torch.Tensor  # the inputs: T_d - T_b sg(om)


@dataclass(frozen=True)
class Dyn4SpinPHRates:
    """``dyn4_spin_ph_rates``, the power balance at a state: dH/dt = supplied - braked -
    dissipated - slip + lever + grade (+ the ground's)."""

    dissipated: torch.Tensor  # the tires' lateral slip, drag, rolling resistance, the dampers
    slip: torch.Tensor  # sum_i K_i sigma_i^2 >= 0: the tires' longitudinal slip
    supplied: torch.Tensor  # om^T T_d: the drive's shaft power
    braked: torch.Tensor  # sum_i T_b,i om_i sg(om_i) >= 0: the brakes'
    lever: torch.Tensor  # xi^T f_lever: the tires' lateral forces on roll and pitch
    grade: torch.Tensor  # xi^T f_grade: zero, grade_force not twinned


@dataclass(frozen=True)
class Dyn4SpinPHBooks:
    """``dyn4_spin_ph_books``, one step's books in J: dH = supplied - braked - dissipated - slip +
    lever + grade + ground + limiter + residual."""

    dH: torch.Tensor
    dissipated: torch.Tensor
    slip: torch.Tensor
    supplied: torch.Tensor
    braked: torch.Tensor
    lever: torch.Tensor
    grade: torch.Tensor
    ground: torch.Tensor  # the springs' energy the new ground under the wheels added
    limiter: torch.Tensor  # the kinetic energy the speed clamp and the low-speed blend added
    residual: torch.Tensor  # the integrator's error, second order in hdt per step


def _ph_secant(F, w, B, C, D, u):
    """``dyn4_ph_secant``: -F / w, a tire's lateral secant through its root, exactly; its limit
    D C B / u at w = 0."""
    zero = w == 0.0
    den = torch.where(zero, torch.ones_like(w), w)
    return torch.where(zero, D * C * B / u, -F / den)


def dyn4_ph_hamiltonian(
    p: Dyn4Params, lim: PlantLimits, state: torch.Tensor, ground: Ground | None = None
) -> torch.Tensor:
    """``dyn4_ph_hamiltonian``: the kinetic energy, the corners' springs and gravity on heave, at
    a state ``(..., 14)`` on the ground under it."""
    if ground is None:
        ground = FlatGround()
    W = _wheel_state(p, lim.hdt, state, ground)
    vx, vy, r, z, zd, phi, phid, th, thd = (state[..., k] for k in range(3, 12))
    H = 0.5 * (
        p.m * (vx * vx + vy * vy + zd * zd) + p.Iz * r * r + p.Ix * phid * phid + p.Iy * thd * thd
    )
    for i in range(4):
        l = W.Fz0[i] / W.k[i] + W.hg[i] - (z + W.yi[i] * phi - W.xi[i] * th)
        H = H + 0.5 * W.k[i] * l * l
    return H + p.m * p.g * z


def dyn4_spin_ph_hamiltonian(
    p: Dyn4Params, lim: PlantLimits, state: torch.Tensor, ground: Ground | None = None
) -> torch.Tensor:
    """The energy at a state ``(..., 18)``: :func:`dyn4_ph_hamiltonian`'s and the wheels' spin."""
    H = dyn4_ph_hamiltonian(p, lim, state, ground)
    for i in range(4):
        om = state[..., 14 + i]
        H = H + 0.5 * (p.J_f if i < 2 else p.J_r) * om * om
    return H


def dyn4_spin_ph_parts_at(
    p: Dyn4Params, lim: PlantLimits, state: torch.Tensor, ground: Ground | None = None
) -> Dyn4SpinPHParts:
    """The parts at a state ``(..., 18)``, at its own twist, ``delta`` and ``a``: each tire's
    longitudinal force from its slip (``dyn4_ph_parts_at`` with the wheels' speeds), the lateral
    grip that leaves, and the wheel torques."""
    if ground is None:
        ground = FlatGround()
    _no_grade_force(p)
    b = state[..., :14]
    vx, vy, r, zd, phid, thd, delta, a = (b[..., k] for k in (3, 4, 5, 7, 9, 11, 12, 13))
    om = state[..., 14:].unbind(-1)
    h, Cp, L = p.h, p.C, p.lf + p.lr
    W = _wheel_state(p, lim.hdt, b, ground)
    eps2 = p.eps * p.eps
    u0 = torch.sqrt(vx * vx + eps2)
    cx = p.drag * u0 + p.c_rr * p.m * p.g / u0
    td = torch.tan(delta)
    Bw = (p.B_f, p.B_f, p.B_r, p.B_r)
    twist = (vx, vy, r, zd, phid, thd)
    zero = torch.zeros_like(vx)
    Fx, Fy, w, ui, Sw, N, T, gs, sig, K = ([] for _ in range(10))
    for i in range(4):
        xi, yi = W.xi[i], W.yi[i]
        cap = W.mu[i] * W.Fz[i]
        dw = ackermann(L, yi, td) if i < 2 else zero
        cw, sw = torch.cos(dw), torch.sin(dw)
        n = (-sw, cw, xi * cw + yi * sw, zero, h * cw, h * sw)
        t = (cw, sw, xi * sw - yi * cw, zero, h * sw, -h * cw)
        gi = zero
        for k in range(6):
            gi = gi + t[k] * twist[k]
        ug = torch.sqrt(gi * gi + eps2)
        s_i = p.r_w * om[i] - gi
        Fx.append(_longitudinal(s_i, ug, p.B_x, p.C_x, cap))
        gs.append(gi)
        sig.append(s_i)
        K.append(_spin_secant(Fx[i], s_i, p.B_x, p.C_x, cap, ug))
        Dy = _lateral_cap(Fx[i], cap)
        vxi = vx - r * yi
        vyi = vy + r * xi
        w.append(-vxi * sw + vyi * cw)
        ul = vxi * cw + vyi * sw
        ui.append(torch.sqrt(ul * ul + eps2))
        Fy.append(_lateral(-torch.atan2(w[i], ui[i]), Bw[i], Cp, Dy))
        Sw.append(_ph_secant(Fy[i], w[i], Bw[i], Cp, Dy, ui[i]))
        N.append(torch.stack(n, dim=-1))
        T.append(torch.stack(t, dim=-1))
    Td, Tb = _spin_torques(p, a)
    oe2 = p.eps * p.eps / (p.r_w * p.r_w)
    sg = [om[i] / torch.sqrt(om[i] * om[i] + oe2) for i in range(4)]
    u = [Td[i] - Tb[i] * sg[i] for i in range(4)]

    def wheels(v):
        return torch.stack(v, dim=-1)

    return Dyn4SpinPHParts(
        W=W,
        cx=cx,
        Fx=wheels(Fx),
        Fy=wheels(Fy),
        w=wheels(w),
        ui=wheels(ui),
        Sw=wheels(Sw),
        N=torch.stack(N, dim=-2),
        T=torch.stack(T, dim=-2),
        g=wheels(gs),
        sig=wheels(sig),
        K=wheels(K),
        Td=wheels(Td),
        Tb=wheels(Tb),
        sg=wheels(sg),
        u=wheels(u),
    )


def dyn4_spin_ph_rates_at(
    p: Dyn4Params, lim: PlantLimits, state: torch.Tensor, ground: Ground | None = None
) -> Dyn4SpinPHRates:
    """The power balance at a state ``(..., 18)``, on the ground under it."""
    q = dyn4_spin_ph_parts_at(p, lim, state, ground)
    vx, vy, r, zd, phid, thd = (state[..., k] for k in (3, 4, 5, 7, 9, 11))
    xi6 = (vx, vy, r, zd, phid, thd)
    om = state[..., 14:]
    dissipated = q.cx * vx * vx
    slip = supplied = braked = lever = torch.zeros_like(vx)
    for i in range(4):
        w = lev = torch.zeros_like(vx)
        for k in range(6):
            w = w + q.N[..., i, k] * xi6[k]
            if k >= 3:
                lev = lev + q.N[..., i, k] * xi6[k]
        # The lateral slip's w_i = N_i^T xi has no vertical part (the slips do not see it): the
        # secant reads the horizontal one; the lever, the vertical.
        wh = w - lev
        rate = zd + q.W.yi[i] * phid - q.W.xi[i] * thd
        dissipated = dissipated + q.Sw[..., i] * wh * wh + q.W.c[i] * rate * rate
        slip = slip + q.K[..., i] * q.sig[..., i] * q.sig[..., i]
        supplied = supplied + om[..., i] * q.Td[..., i]
        braked = braked + q.Tb[..., i] * om[..., i] * q.sg[..., i]
        lever = lever + q.Fy[..., i] * lev
    return Dyn4SpinPHRates(
        dissipated=dissipated,
        slip=slip,
        supplied=supplied,
        braked=braked,
        lever=lever,
        grade=torch.zeros_like(vx),
    )


def dyn4_spin_plant_books(
    p: Dyn4Params,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    ground: Ground | None = None,
) -> tuple[torch.Tensor, Dyn4SpinPHBooks]:
    """:func:`dyn4_spin_plant`'s step, unchanged, and its books.

    The powers are the trapezoid between the step's start, with the inputs it applied (the
    actuators' outputs), and the state it reached before its limiters, both at the start's pose
    (the ground the step stood on); ``limiter`` is dH across the clamp and the blend, ``ground``
    the springs' energy the new ground under the wheels added; ``residual`` is the rest.
    """
    nxt, free = dyn4_spin_plant(p, lim, state, delta_cmd, a_cmd, ground, with_free=True)
    s0 = torch.cat((state[..., :12], nxt[..., 12:14], state[..., 14:]), dim=-1)
    f = torch.cat((state[..., :3], nxt[..., 3:]), dim=-1)
    e = torch.cat((state[..., :3], free, nxt[..., 6:]), dim=-1)
    r0 = dyn4_spin_ph_rates_at(p, lim, s0, ground)
    r1 = dyn4_spin_ph_rates_at(p, lim, e, ground)
    half = 0.5 * lim.hdt
    dissipated = half * (r0.dissipated + r1.dissipated)
    slip = half * (r0.slip + r1.slip)
    supplied = half * (r0.supplied + r1.supplied)
    braked = half * (r0.braked + r1.braked)
    lever = half * (r0.lever + r1.lever)
    grade = half * (r0.grade + r1.grade)
    h0 = dyn4_spin_ph_hamiltonian(p, lim, state, ground)
    he = dyn4_spin_ph_hamiltonian(p, lim, e, ground)
    hf = dyn4_spin_ph_hamiltonian(p, lim, f, ground)
    h1 = dyn4_spin_ph_hamiltonian(p, lim, nxt, ground)
    dH = h1 - h0
    limiter = hf - he
    ground_w = h1 - hf
    residual = dH - (supplied - braked - dissipated - slip + lever + grade + ground_w + limiter)
    return nxt, Dyn4SpinPHBooks(
        dH=dH,
        dissipated=dissipated,
        slip=slip,
        supplied=supplied,
        braked=braked,
        lever=lever,
        grade=grade,
        ground=ground_w,
        limiter=limiter,
        residual=residual,
    )
