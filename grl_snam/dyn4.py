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

1. actuators, as P8a;
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
branch is a ``torch.where`` on values, as the C++ branches on ``s_val``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from grl_snam.dyn_bike import PlantLimits, _clamp, _lateral, _sat, _secant_w

__all__ = ["ORDER", "Dyn4Params", "FlatGround", "PlantLimits", "ackermann", "dyn4_plant"]

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


@dataclass(frozen=True)
class FlatGround:
    """Flat ground of one friction (libcvc's ``flat_ground``, whose ``mu`` is a float)."""

    mu: float = 0.9

    def __call__(self, wx: torch.Tensor, wy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(wx, self.mu), torch.zeros_like(wx)


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


def dyn4_plant(
    p: Dyn4Params,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    ground: Ground | None = None,
) -> torch.Tensor:
    """One step of the plant: ``state`` ``(..., 14)`` to the next ``(..., 14)``."""
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
    a1 = a + (a_cmd - a) * k_drive

    # 2. The wheels, the ground under them and their loads.
    xi = [lf, lf, -lr, -lr]
    yi = [p.tf, -p.tf, p.tr, -p.tr]
    Fz0 = [0.5 * m * g * (lr if i < 2 else lf) / L for i in range(4)]
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

    # 3. Longitudinal.
    F_drive = m * a1
    share = torch.where(
        a1 >= 0.0, torch.full_like(a1, p.front_drive), torch.full_like(a1, p.front_brake)
    )
    td = torch.tan(delta1)
    Fx, Dy, cdw, sdw = [], [], [], []
    for i in range(4):
        axle_share = share if i < 2 else 1.0 - share
        cap = mu_i[i] * Fz[i]
        Fx.append(_sat(0.5 * F_drive * axle_share, cap))
        Dy.append(torch.sqrt(cap * cap - 0.9975 * Fx[i] * Fx[i]))
        dw = ackermann(L, yi[i], td) if i < 2 else torch.zeros_like(td)
        cdw.append(torch.cos(dw))
        sdw.append(torch.sin(dw))
    eps2 = p.eps * p.eps
    sp_min = float(np.float32(-0.25) * np.float32(lim.vmax)) if lim.allow_reverse else 0.0
    u0 = torch.sqrt(vx * vx + eps2)
    sg0 = vx / u0
    F_res = -p.drag * vx * u0 - p.c_rr * m * g * sg0
    FXlong = torch.zeros_like(vx)
    for i in range(4):
        FXlong = FXlong + Fx[i] * cdw[i]
    vx_p = _clamp(vx + hdt * ((FXlong + F_res) / m), sp_min, lim.vmax)

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
    vx1 = _clamp(vx + hdt * (FXb / m + r1 * vy1), sp_min, lim.vmax)

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
    return torch.stack(
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
