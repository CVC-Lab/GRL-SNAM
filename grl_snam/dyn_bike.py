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

The plant in port-Hamiltonian form (libcvc's ``dyn_ph_*``): with the twist ``v = (vx, vy, r)``,
the momenta ``p = M v``, ``M = diag(m, m, Iz)`` and ``H = 1/2 v^T M v``, the plant's vector field
is ``p' = (J(p) - R) v + G u + f_eps - m g grade e_x`` with the axle forces ``u = (F_xf, F_xr)``
as the port and ``y = G^T v``. ``R`` is built from the tires' secants, so it is positive
semi-definite at every slip; ``f_eps`` is what the slips' speed floor leaves. So

    dH/dt = supplied - dissipated + floor + grade,

each term returned by :func:`dyn_ph_rates_at`, and :func:`dyn_plant_books` books one step:
the trapezoid of those powers between the step's start (with the actuators' outputs) and the
twist it reached before its limiters, the kinetic energy the limiters then moved, and the
integrator's residual. The parity gate is ``tests/test_dyn_bike_ph_parity.py``, against
libcvc's ``dyn_bike_ph_golden.json``.
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


def _axles(p: DynBikeParams, a: torch.Tensor):
    """Each axle's drive or brake force within its grip, and the lateral grip it leaves, at the
    actuator's ``a`` (``dyn_ph_axles``, the plant's step 2): Fxf, Fxr, Dyf, Dyr."""
    m, g, h, L = p.m, p.g, p.h, p.lf + p.lr
    fz_floor = p.fz_min * m * g
    Fzf = m * (g * p.lr - a * h) / L
    Fzr = m * (g * p.lf + a * h) / L
    Fzf = torch.where(Fzf < fz_floor, torch.full_like(Fzf, fz_floor), Fzf)
    Fzr = torch.where(Fzr < fz_floor, torch.full_like(Fzr, fz_floor), Fzr)
    cap_f, cap_r = p.mu * Fzf, p.mu * Fzr
    F_drive = m * a
    share = torch.where(
        a >= 0.0, torch.full_like(a, p.front_drive), torch.full_like(a, p.front_brake)
    )
    Fxf = _sat(F_drive * share, cap_f)
    Fxr = _sat(F_drive * (1.0 - share), cap_r)
    Dyf = torch.sqrt(cap_f * cap_f - 0.9975 * Fxf * Fxf)
    Dyr = torch.sqrt(cap_r * cap_r - 0.9975 * Fxr * Fxr)
    return Fxf, Fxr, Dyf, Dyr


def dyn_plant(
    p: DynBikeParams,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    grade: torch.Tensor | float = 0.0,
    *,
    with_free: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One step of the plant: ``state`` ``(..., 8)`` to the next ``(..., 8)``.

    With ``with_free``, also the twist ``(vx, vy, r)`` ``(..., 3)`` the step reached before its
    limiters (the speed clamp, the low-speed blend): libcvc's ``dyn_step_free``.
    """
    x, y, psi, vx, vy, r, delta, a = state.unbind(-1)
    grade = torch.as_tensor(grade, dtype=state.dtype, device=state.device)
    hdt = lim.hdt
    m, Iz, lf, lr, g, Cp = p.m, p.Iz, p.lf, p.lr, p.g, p.C
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
    Fxf, Fxr, Dyf, Dyr = _axles(p, a1)
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
    vx1_free = vx + hdt * (FX / m + r1 * vy1)
    vx1 = _clamp(vx1_free, sp_min, lim.vmax)
    free = torch.stack((vx1_free, vy1, r1), dim=-1) if with_free else None

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
    nxt = torch.stack((x1, y1, psi1, vx1, vy1, r1, delta1, a1), dim=-1)
    return (nxt, free) if with_free else nxt


# ── The plant in port-Hamiltonian form ──────────────────────────────────────────────────────
# Each function mirrors its libcvc namesake's arithmetic, sums in the same order, so the double
# tier holds to rounding. They are evaluations: dyn_plant does not depend on them.


@dataclass(frozen=True)
class DynPHParts:
    """``dyn_ph_parts``: everything the form is built from, at a state (each ``(...)``, the
    vectors ``(..., 3)`` on ``(vx, vy, r)``)."""

    u: torch.Tensor  # the slips' floored speed sqrt(vx^2 + eps^2)
    sg: torch.Tensor  # vx / u
    Fxf: torch.Tensor  # the axles' drive or brake forces: the inputs
    Fxr: torch.Tensor
    Fyf: torch.Tensor  # the tires' lateral forces
    Fyr: torch.Tensor
    wf: torch.Tensor  # the axles' lateral velocities
    wr: torch.Tensor
    w0f: torch.Tensor  # the front force's root
    Sf: torch.Tensor  # the secant stiffnesses through the roots
    Sr: torch.Tensor
    Dx: torch.Tensor  # drag and rolling resistance on vx
    cd: torch.Tensor  # cos delta
    nf: torch.Tensor  # each tire's lateral force on (vx, vy, r)
    nr: torch.Tensor
    tf: torch.Tensor  # each axle's drive force on (vx, vy, r)
    tr: torch.Tensor


@dataclass(frozen=True)
class DynPHRates:
    """``dyn_ph_rates``, the power balance at a state: dH/dt = supplied - dissipated + floor +
    grade."""

    dissipated: torch.Tensor  # v^T R v >= 0: the tires' slip, drag and rolling resistance
    supplied: torch.Tensor  # y^T u: the axles' forces at their wheels' ground speed
    floor: torch.Tensor  # v^T f_eps: the slips' speed floor
    grade: torch.Tensor  # -m g grade vx: gravity along the heading


@dataclass(frozen=True)
class DynPHBooks:
    """``dyn_ph_books``, one step's books in J: dH = supplied - dissipated + floor + grade +
    limiter + residual."""

    dH: torch.Tensor
    dissipated: torch.Tensor
    supplied: torch.Tensor
    floor: torch.Tensor
    grade: torch.Tensor
    limiter: torch.Tensor  # the kinetic energy the speed clamp and the low-speed blend added
    residual: torch.Tensor  # the integrator's error, second order in hdt per step


def dyn_ph_hamiltonian(p: DynBikeParams, state: torch.Tensor) -> torch.Tensor:
    """The kinetic energy 1/2 v^T M v."""
    vx, vy, r = state[..., 3], state[..., 4], state[..., 5]
    return 0.5 * (p.m * (vx * vx + vy * vy) + p.Iz * r * r)


def dyn_ph_parts_at(p: DynBikeParams, state: torch.Tensor) -> DynPHParts:
    """The parts at a state ``(..., 8)``, at its own ``delta`` and ``a`` (the actuators'
    outputs)."""
    _, _, _, vx, vy, r, delta, a = state.unbind(-1)
    lf, lr = p.lf, p.lr
    Fxf, Fxr, Dyf, Dyr = _axles(p, a)
    u = torch.sqrt(vx * vx + p.eps * p.eps)
    sg = vx / u
    cd, sd = torch.cos(delta), torch.sin(delta)
    wf = vy + lf * r
    wr = vy - lr * r
    w0f = u * torch.tan(sg * delta)
    af = sg * delta - torch.atan2(wf, u)
    ar = -torch.atan2(wr, u)
    Fyf = _lateral(af, p.B_f, p.C, Dyf)
    Fyr = _lateral(ar, p.B_r, p.C, Dyr)
    zero, one = torch.zeros_like(cd), torch.ones_like(cd)
    return DynPHParts(
        u=u,
        sg=sg,
        Fxf=Fxf,
        Fxr=Fxr,
        Fyf=Fyf,
        Fyr=Fyr,
        wf=wf,
        wr=wr,
        w0f=w0f,
        Sf=_secant_w(Fyf, af, wf, w0f, p.B_f, p.C, Dyf, u),
        Sr=_secant_w(Fyr, ar, wr, zero, p.B_r, p.C, Dyr, u),
        Dx=p.drag * u + p.c_rr * p.m * p.g / u,
        cd=cd,
        nf=torch.stack((-sd, cd, lf * cd), dim=-1),
        nr=torch.stack((zero, one, torch.full_like(cd, -lr)), dim=-1),
        tf=torch.stack((cd, sd, lf * sd), dim=-1),
        tr=torch.stack((one, zero, zero), dim=-1),
    )


def dyn_ph_rates_at(
    p: DynBikeParams, state: torch.Tensor, grade: torch.Tensor | float = 0.0
) -> DynPHRates:
    """The power balance at a state ``(..., 8)``: R, G, u and f_eps evaluated there."""
    q = dyn_ph_parts_at(p, state)
    v = (state[..., 3], state[..., 4], state[..., 5])
    nf = q.nf.unbind(-1)
    nr = q.nr.unbind(-1)
    tf = q.tf.unbind(-1)
    tr = q.tr.unbind(-1)
    # R = Dx e_x e_x^T + (Sf / cos delta) nf nf^T + Sr nr nr^T.
    kf = q.Sf / q.cd
    R = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            R[i][j] = kf * (nf[i] * nf[j]) + q.Sr * (nr[i] * nr[j])
            R[j][i] = R[i][j]
    R[0][0] = R[0][0] + q.Dx
    # f_eps: what the secants leave of the tires' forces.
    nfv = torch.zeros_like(q.u)
    nrv = torch.zeros_like(q.u)
    for i in range(3):
        nfv = nfv + nf[i] * v[i]
        nrv = nrv + nr[i] * v[i]
    ef = q.Fyf + q.Sf / q.cd * nfv
    er = q.Fyr + q.Sr * nrv
    f = [nf[i] * ef + nr[i] * er for i in range(3)]
    dissipated = torch.zeros_like(q.u)
    supplied = torch.zeros_like(q.u)
    floor = torch.zeros_like(q.u)
    for i in range(3):
        for j in range(3):
            dissipated = dissipated + v[i] * R[i][j] * v[j]
        supplied = supplied + v[i] * (tf[i] * q.Fxf + tr[i] * q.Fxr)
        floor = floor + v[i] * f[i]
    return DynPHRates(
        dissipated=dissipated, supplied=supplied, floor=floor, grade=-p.m * p.g * grade * v[0]
    )


def dyn_plant_books(
    p: DynBikeParams,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    grade: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, DynPHBooks]:
    """:func:`dyn_plant`'s step, unchanged, and its books.

    The powers are the trapezoid between the step's start, with the inputs it applied (the
    actuators' outputs ``delta1``, ``a1``), and the twist it reached before its limiters;
    ``limiter`` is dH across the clamp and the blend; ``residual`` is the rest.
    """
    nxt, free = dyn_plant(p, lim, state, delta_cmd, a_cmd, grade, with_free=True)
    s0 = torch.cat((state[..., :6], nxt[..., 6:]), dim=-1)
    e = torch.cat((nxt[..., :3], free, nxt[..., 6:]), dim=-1)
    r0, r1 = dyn_ph_rates_at(p, s0, grade), dyn_ph_rates_at(p, e, grade)
    half = 0.5 * lim.hdt
    dissipated = half * (r0.dissipated + r1.dissipated)
    supplied = half * (r0.supplied + r1.supplied)
    floor = half * (r0.floor + r1.floor)
    grade_w = half * (r0.grade + r1.grade)
    h0 = dyn_ph_hamiltonian(p, state)
    he = dyn_ph_hamiltonian(p, e)
    h1 = dyn_ph_hamiltonian(p, nxt)
    dH = h1 - h0
    limiter = h1 - he
    residual = dH - (supplied - dissipated + floor + grade_w + limiter)
    return nxt, DynPHBooks(
        dH=dH,
        dissipated=dissipated,
        supplied=supplied,
        floor=floor,
        grade=grade_w,
        limiter=limiter,
        residual=residual,
    )
