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
4. ``(vy, r)`` by implicit Euler, ``iters`` steps, each tire's slip in its wheel's frame (as
   :mod:`grl_snam.dyn4`'s), linearized by its secant through its root, zero sideways velocity
   there (Pacejka forces within the friction ellipse's remainder);
5. the speed corrected with the front lateral force the implicit step applied;
6. below ``v_blend``, ``(vy, r)`` blended toward the kinematic bicycle's;
7. the pose: the rear axle moves with ``(vx, vy - lr r)`` at the new heading.

Every branch is a ``torch.where`` on values, as the C++ branches on ``s_val``; the division in
the unselected branch is guarded so it never turns a gradient into NaN.

The plant in port-Hamiltonian form (libcvc's ``dyn_ph_*``): with the twist ``v = (vx, vy, r)``,
the momenta ``p = M v``, ``M = diag(m, m, Iz)`` and ``H = 1/2 v^T M v``, the plant's vector field
is ``p' = (J(p) - R) v + G u + f_eps - m g grade e_x`` with the axle forces ``u = (F_xf, F_xr)``
as the port and ``y = G^T v``. ``R`` is built from the tires' secants, so it is positive
semi-definite at every slip; ``f_eps``, what the secants leave of the tires' forces, is
round-off, since each slip's root is zero sideways velocity in its wheel's frame. So

    dH/dt = supplied - dissipated + floor + grade,

each term returned by :func:`dyn_ph_rates_at`, and :func:`dyn_plant_books` books one step:
the trapezoid of those powers between the step's start (with the actuators' outputs) and the
twist it reached before its limiters, the kinetic energy the limiters then moved, and the
integrator's residual. The parity gate is ``tests/test_dyn_bike_ph_parity.py``, against
libcvc's ``dyn_bike_ph_golden.json``.

The wheel-spin plant (libcvc's ``dyn_spin_plant``, opt-in there): each axle's wheels spin at
their own speed, the state ``(..., 10)`` with ``om_f, om_r`` (rad/s, + rolling forward) after the
eight above. The actuator's ``a`` becomes drive and brake torques on the wheels, split between
the axles as the drive and brake forces are; each tire pushes the body along its wheel with
Pacejka's longitudinal force of its slip ratio ``(r_w om - t^T v) / u``, and its wheel feels
``- r_w F_x``. Step 3 then solves the speed and the wheels' speeds together by implicit Euler on
the forces' secants through their roots (:func:`dyn_spin_plant`); the rest is the plant's. In
port-Hamiltonian form the wheels' angular momenta join the state, the longitudinal secants join
``R`` and the wheel torques are the port, so

    dH/dt = supplied - braked - dissipated - slip + floor + grade,

each term returned by :func:`dyn_spin_ph_rates_at`, and :func:`dyn_spin_plant_books` books one
step as :func:`dyn_plant_books` does. The parity gate is ``tests/test_dyn_spin_ph_parity.py``,
against libcvc's ``dyn_spin_ph_golden.json``.

Both plants take an optional external ``wrench`` ``(F_x, F_y, M_z)`` on the body, held over the
step: the pH form's external port, such as the compliant contact's (:mod:`grl_snam.dyn_contact`).
Their books then gain its energy, ``port`` = the trapezoid of ``v^T W`` (:func:`dyn_ph_port_work`),
and the residual is what that leaves.

The full (q, p) form (libcvc's ``dyn_ph_rhs_full``): with the pose in the state and the ground's
height under the centre of gravity in H, ``q' = T(q) v`` and ``p' = -T(q)^T dV/dq + (J(p) - R) v +
G u + f_eps`` (:func:`dyn_ph_T`, :func:`dyn_ph_rhs_full`). On a ground, :func:`dyn_plant_ground`
applies gravity along and across the heading (the ``side`` slope, opt-in on :func:`dyn_plant`), and
:func:`dyn_plant_books_ground` books it with the grade line ``-dV``. The parity gate is
``tests/test_dyn_ph_full_parity.py``, against libcvc's ``dyn_ph_full_golden.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

ORDER = ("x", "y", "psi", "vx", "vy", "r", "delta", "a")
SPIN_ORDER = ORDER + ("om_f", "om_r")


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
    # The wheels, for dyn_spin_plant (dyn_plant reads none of these).
    r_w: float = 0.4  # the wheels' rolling radius
    J_f: float = 1.0  # the front axle's wheels' spin inertia, both together
    J_r: float = 1.0  # the rear axle's
    B_x: float = 12.0  # Pacejka stiffness factor of the longitudinal force, per unit slip ratio
    C_x: float = 1.2  # its shape factor


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


def _clamp_cap(x, cap):
    """x clamped to [-cap, cap], cap a tensor (its gradient flows where it binds)."""
    return torch.where(x < -cap, -cap, torch.where(x > cap, cap, x))


def _lateral_cap(Fx, cap):
    """The lateral peak an axle has left beside Fx: sqrt(cap^2 - 0.9975 Fx^2)."""
    return torch.sqrt(cap * cap - 0.9975 * Fx * Fx)


def _caps(p: DynBikeParams, a: torch.Tensor):
    """Each axle's grip, mu Fz, its load from the actuator's ``a`` (``dyn_ph_caps``, the plant's
    step 2): cap_f, cap_r."""
    m, g, h, L = p.m, p.g, p.h, p.lf + p.lr
    fz_floor = p.fz_min * m * g
    Fzf = m * (g * p.lr - a * h) / L
    Fzr = m * (g * p.lf + a * h) / L
    Fzf = torch.where(Fzf < fz_floor, torch.full_like(Fzf, fz_floor), Fzf)
    Fzr = torch.where(Fzr < fz_floor, torch.full_like(Fzr, fz_floor), Fzr)
    return p.mu * Fzf, p.mu * Fzr


def _axles(p: DynBikeParams, a: torch.Tensor):
    """Each axle's drive or brake force within its grip, and the lateral grip it leaves, at the
    actuator's ``a`` (``dyn_ph_axles``, the plant's step 2): Fxf, Fxr, Dyf, Dyr."""
    cap_f, cap_r = _caps(p, a)
    F_drive = p.m * a
    share = torch.where(
        a >= 0.0, torch.full_like(a, p.front_drive), torch.full_like(a, p.front_brake)
    )
    Fxf = _sat(F_drive * share, cap_f)
    Fxr = _sat(F_drive * (1.0 - share), cap_r)
    return Fxf, Fxr, _lateral_cap(Fxf, cap_f), _lateral_cap(Fxr, cap_r)


# ── Wheel spin (dyn_spin_plant) ─────────────────────────────────────────────────────────────
# libcvc's "Wheel spin" section: a >= 0 drives each axle's wheels with r_w x its share
# (front_drive) of m a; a < 0 brakes them with r_w x its share (front_brake) of m |a| against
# their rolling, - T_b sg(om), sg the sign smoothed over the slip's speed floor at the rim. Each
# tire pushes the body along its wheel t (t_f = (cos delta, sin delta, lf sin delta), t_r =
# (1, 0, 0) on (vx, vy, r)) with D sin(C_x atan(B_x sigma / u)), sigma = r_w om - t^T v its contact
# patch's slip velocity, u the ground's speed along the wheel floored as the slip angles' are, and
# D = mu Fz the axle's whole grip; the wheel feels - r_w F_x.


def _spin_torques(p: DynBikeParams, a: torch.Tensor):
    """``dyn_spin_torques``: the axles' drive torques Td and brake torque magnitudes Tb for the
    actuator's ``a``, each a (front, rear) pair."""
    drive = a >= 0.0
    share = torch.where(drive, torch.full_like(a, p.front_drive), torch.full_like(a, p.front_brake))
    T = p.r_w * p.m * a
    Tf = T * share
    Tr = T * (1.0 - share)
    zero = torch.zeros_like(a)
    Td = (torch.where(drive, Tf, zero), torch.where(drive, Tr, zero))
    Tb = (torch.where(drive, zero, -Tf), torch.where(drive, zero, -Tr))
    return Td, Tb


def _longitudinal(sigma, u, B, C, D):
    """A tire's longitudinal force at slip velocity sigma over the floored ground speed u."""
    return D * torch.sin(C * torch.atan(B * (sigma / u)))


def _secant_x(F, sigma, B, C, D, u):
    """``dyn_secant_x``: F / sigma, the force's secant through its root; its slope D C B / u
    where |B sigma / u| < 1e-4."""
    bk = B * (sigma / u)
    small = bk * bk < 1e-8
    den = torch.where(small, torch.ones_like(sigma), sigma)
    return torch.where(small, D * C * B / u, F / den)


def _spin_secant(F, sigma, B, C, D, u):
    """``dyn_spin_ph_secant``: F / sigma exactly, and its limit D C B / u at sigma = 0."""
    zero = sigma == 0.0
    den = torch.where(zero, torch.ones_like(sigma), sigma)
    return torch.where(zero, D * C * B / u, F / den)


def _spin_long_step(p: DynBikeParams, hdt, vx0, vy, r, om0, a1, cd, sd, cap_f, cap_r, F_res):
    """``dyn_spin_long_step``: (vx, om_f, om_r) by implicit Euler on the tires' longitudinal
    forces and the wheels' torques, each force and brake linearized by its secant through its
    root, ``iters`` steps from the current state. Gives the speed (unclamped), the wheels' new
    speeds ``(..., 2)`` and the forces the last linearization applied."""
    m, rw, Jf, Jr = p.m, p.r_w, p.J_f, p.J_r
    eps2 = p.eps * p.eps
    oe2 = eps2 / (rw * rw)
    Td, Tb = _spin_torques(p, a1)
    side_f = (vy + p.lf * r) * sd  # the front's ground speed along its wheel, less vx's
    om_f0, om_r0 = om0.unbind(-1)
    vx, of, orr = vx0, om_f0, om_r0
    Fxf = Fxr = torch.zeros_like(vx0)
    for _ in range(max(1, p.iters)):
        gf = vx * cd + side_f
        gr = vx
        uf = torch.sqrt(gf * gf + eps2)
        ur = torch.sqrt(gr * gr + eps2)
        sf = rw * of - gf
        sr = rw * orr - gr
        Ff = _longitudinal(sf, uf, p.B_x, p.C_x, cap_f)
        Fr = _longitudinal(sr, ur, p.B_x, p.C_x, cap_r)
        Kf = _secant_x(Ff, sf, p.B_x, p.C_x, cap_f, uf)
        Kr = _secant_x(Fr, sr, p.B_x, p.C_x, cap_r, ur)
        Kbf = Tb[0] / torch.sqrt(of * of + oe2)
        Kbr = Tb[1] / torch.sqrt(orr * orr + oe2)
        # M (z0 + hdt f(z) - z).
        b0 = m * (vx0 - vx) + hdt * (Ff * cd + Fr + F_res)
        b1 = Jf * (om_f0 - of) + hdt * (Td[0] - Kbf * of - rw * Ff)
        b2 = Jr * (om_r0 - orr) + hdt * (Td[1] - Kbr * orr - rw * Fr)
        # M + hdt R_x: the wheels' rows couple only through vx.
        A00 = m + hdt * (Kf * cd * cd + Kr)
        A01 = -hdt * Kf * cd * rw
        A02 = -hdt * Kr * rw
        A11 = Jf + hdt * (Kf * rw * rw + Kbf)
        A22 = Jr + hdt * (Kr * rw * rw + Kbr)
        d0 = (b0 - A01 * b1 / A11 - A02 * b2 / A22) / (A00 - A01 * A01 / A11 - A02 * A02 / A22)
        d1 = (b1 - A01 * d0) / A11
        d2 = (b2 - A02 * d0) / A22
        vx = vx + d0
        of = of + d1
        orr = orr + d2
        # The forces this linearization applies.
        Fxf = Ff + Kf * (rw * d1 - cd * d0)
        Fxr = Fr + Kr * (rw * d2 - d0)
    return vx, torch.stack((of, orr), dim=-1), Fxf, Fxr


def _plant(
    p: DynBikeParams, lim: PlantLimits, state, om0, delta_cmd, a_cmd, grade, wrench=None, side=None
):
    """``dyn_plant_core``: one step of the plant (``om0`` None) or of the wheel-spin plant
    (``om0`` the wheels' speeds ``(..., 2)``), under the external ``wrench`` ``(..., 3)`` when
    given, and with gravity across the heading when ``side`` (the ground's slope across it, +
    rising to the left) is given. Gives the next ``(..., 8)``, the twist the step reached before
    its limiters ``(..., 3)``, and the wheels' new speeds (None without them)."""
    x, y, psi, vx, vy, r, delta, a = state.unbind(-1)
    grade = torch.as_tensor(grade, dtype=state.dtype, device=state.device)
    if side is not None:
        side = torch.as_tensor(side, dtype=state.dtype, device=state.device)
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
    cd, sd = torch.cos(delta1), torch.sin(delta1)
    eps2 = p.eps * p.eps
    sp_min = -0.25 * lim.vmax if lim.allow_reverse else 0.0

    # 3. The speed, predicted without lateral forces. With the wheels spinning, the tires'
    # longitudinal forces from their slip, solved with the speed and the wheels'.
    u0 = torch.sqrt(vx * vx + eps2)
    sg0 = vx / u0
    F_res = -p.drag * vx * u0 - p.c_rr * m * g * sg0 - m * g * grade
    # An external wrench's force along the body (the external port), held over the step.
    if wrench is not None:
        F_res = F_res + wrench[..., 0]
    if om0 is None:
        Fxf, Fxr, Dyf, Dyr = _axles(p, a1)
        vx_p = _clamp(vx + hdt * ((Fxf * cd + Fxr + F_res) / m), sp_min, lim.vmax)
        om1 = None
    else:
        cap_f, cap_r = _caps(p, a1)
        vx_s, om1, Fxf, Fxr = _spin_long_step(
            p, hdt, vx, vy, r, om0, a1, cd, sd, cap_f, cap_r, F_res
        )
        vx_p = _clamp(vx_s, sp_min, lim.vmax)
        # A linearization's force may pass the grip by its error: the ellipse reads it within.
        Fxf = _clamp_cap(Fxf, cap_f)
        Fxr = _clamp_cap(Fxr, cap_r)
        Dyf, Dyr = _lateral_cap(Fxf, cap_f), _lateral_cap(Fxr, cap_r)

    # 4. Lateral and yaw, each slip in its wheel's frame.
    u = torch.sqrt(vx_p * vx_p + eps2)
    vy1, r1 = vy, r
    Fyf_step = torch.zeros_like(vy)
    for _ in range(max(1, p.iters)):
        vf = vy1 + lf * r1
        wf = -vx_p * sd + vf * cd
        wr = vy1 - lr * r1
        uf = torch.sqrt((vx_p * cd + vf * sd) * (vx_p * cd + vf * sd) + eps2)
        af = -torch.atan2(wf, uf)
        ar = -torch.atan2(wr, u)
        Fyf = _lateral(af, p.B_f, Cp, Dyf)
        Fyr = _lateral(ar, p.B_r, Cp, Dyr)
        FY = Fxf * sd + Fyf * cd + Fyr
        MZ = lf * (Fxf * sd + Fyf * cd) - lr * Fyr
        if wrench is not None:
            FY = FY + wrench[..., 1]
            MZ = MZ + wrench[..., 2]
        if side is not None:
            FY = FY - m * g * side
        f_vy = FY / m - r1 * vx_p
        f_r = MZ / Iz
        Sf_t = _secant_w(Fyf, af, wf, torch.zeros_like(wf), p.B_f, Cp, Dyf, uf)
        Sr = _secant_w(Fyr, ar, wr, torch.zeros_like(wr), p.B_r, Cp, Dyr, u)
        # dwf/d(vy, r) = cos(delta) (1, lf): Sf on the tire, Sfy on the body's y.
        Sf = Sf_t * cd
        Sfy = Sf * cd
        J11 = (-Sfy - Sr) / m
        J12 = (-Sfy * lf + Sr * lr) / m - vx_p
        J21 = (-Sfy * lf + Sr * lr) / Iz
        J22 = (-Sfy * lf * lf - Sr * lr * lr) / Iz
        A11, A12, A21, A22 = 1.0 - hdt * J11, -hdt * J12, -hdt * J21, 1.0 - hdt * J22
        det = A11 * A22 - A12 * A21
        det = torch.where(det < 0.05, torch.full_like(det, 0.05), det)
        b1 = vy + hdt * f_vy - vy1
        b2 = r + hdt * f_r - r1
        dvy = (A22 * b1 - A12 * b2) / det
        dr = (A11 * b2 - A21 * b1) / det
        vy1 = vy1 + dvy
        r1 = r1 + dr
        Fyf_step = Fyf - Sf * (dvy + lf * dr)

    # 5. The speed, corrected.
    FX = Fxf * cd - Fyf_step * sd + Fxr + F_res
    vx1_free = vx + hdt * (FX / m + r1 * vy1)
    vx1 = _clamp(vx1_free, sp_min, lim.vmax)
    free = torch.stack((vx1_free, vy1, r1), dim=-1)

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
    return nxt, free, om1


def dyn_plant(
    p: DynBikeParams,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    grade: torch.Tensor | float = 0.0,
    *,
    with_free: bool = False,
    wrench: torch.Tensor | None = None,
    side: torch.Tensor | float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One step of the plant: ``state`` ``(..., 8)`` to the next ``(..., 8)``.

    With ``with_free``, also the twist ``(vx, vy, r)`` ``(..., 3)`` the step reached before its
    limiters (the speed clamp, the low-speed blend): libcvc's ``dyn_step_free``. ``wrench``, when
    given, is an external wrench ``(..., 3)`` on the body (F_x, F_y, M_z: body axes, at the centre
    of gravity) held over the step, the pH form's external port, such as the contact's
    (:mod:`grl_snam.dyn_contact`): it joins the resistances along the body and the lateral
    solve's force and moment. ``side``, when given, is the ground's slope across the heading:
    gravity's body-y component, ``- m g side``, joins the lateral solve's force
    (:func:`dyn_plant_ground`).
    """
    nxt, free, _ = _plant(p, lim, state, None, delta_cmd, a_cmd, grade, wrench, side)
    return (nxt, free) if with_free else nxt


def dyn_spin_plant(
    p: DynBikeParams,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    grade: torch.Tensor | float = 0.0,
    *,
    with_free: bool = False,
    wrench: torch.Tensor | None = None,
    side: torch.Tensor | float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One step of the wheel-spin plant: ``state`` ``(..., 10)`` (:data:`SPIN_ORDER`) to the
    next ``(..., 10)``, the wheels' speeds advanced with the speed. The low-speed blend and the
    speed clamp act on the body alone. ``with_free``, ``wrench`` and ``side`` as
    :func:`dyn_plant`'s.
    """
    nxt, free, om1 = _plant(
        p, lim, state[..., :8], state[..., 8:], delta_cmd, a_cmd, grade, wrench, side
    )
    nxt = torch.cat((nxt, om1), dim=-1)
    return (nxt, free) if with_free else nxt


# ── The plant in port-Hamiltonian form ──────────────────────────────────────────────────────
# Each function mirrors its libcvc namesake's arithmetic, sums in the same order, so the double
# tier holds to rounding. They are evaluations: dyn_plant does not depend on them.


@dataclass(frozen=True)
class DynPHParts:
    """``dyn_ph_parts``: everything the form is built from, at a state (each ``(...)``, the
    vectors ``(..., 3)`` on ``(vx, vy, r)``)."""

    u: torch.Tensor  # the floored speed sqrt(vx^2 + eps^2) (the rear slip's)
    sg: torch.Tensor  # vx / u
    uf: torch.Tensor  # the front wheel's floored forward speed, in its frame
    Fxf: torch.Tensor  # the axles' drive or brake forces: the inputs
    Fxr: torch.Tensor
    Fyf: torch.Tensor  # the tires' lateral forces
    Fyr: torch.Tensor
    wf: torch.Tensor  # the axles' sideways velocities in their wheels' frames: nf.v, nr.v
    wr: torch.Tensor
    Sf: torch.Tensor  # the secant stiffnesses through wf, wr = 0
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
    floor: torch.Tensor  # v^T f_eps: what the secants leave, round-off
    grade: torch.Tensor  # -m g grade vx: gravity along the heading


@dataclass(frozen=True)
class DynPHBooks:
    """``dyn_ph_books``, one step's books in J: dH = supplied - dissipated + floor + grade +
    limiter (+ port) + residual."""

    dH: torch.Tensor
    dissipated: torch.Tensor
    supplied: torch.Tensor
    floor: torch.Tensor
    grade: torch.Tensor
    limiter: torch.Tensor  # the kinetic energy the speed clamp and the low-speed blend added
    residual: torch.Tensor  # the integrator's error, second order in hdt per step
    port: torch.Tensor | None = None  # an external wrench's energy, v^T W over the step


def dyn_ph_hamiltonian(p: DynBikeParams, state: torch.Tensor) -> torch.Tensor:
    """The kinetic energy 1/2 v^T M v."""
    vx, vy, r = state[..., 3], state[..., 4], state[..., 5]
    return 0.5 * (p.m * (vx * vx + vy * vy) + p.Iz * r * r)


def _ph_parts_lateral(p: DynBikeParams, state, Fxf, Fxr, Dyf, Dyr) -> DynPHParts:
    """``dyn_ph_parts_lateral``: the parts at a state with the axles' longitudinal forces Fxf,
    Fxr and the lateral grip Dyf, Dyr they leave."""
    _, _, _, vx, vy, r, delta, _ = state[..., :8].unbind(-1)
    lf, lr = p.lf, p.lr
    u = torch.sqrt(vx * vx + p.eps * p.eps)
    sg = vx / u
    cd, sd = torch.cos(delta), torch.sin(delta)
    vf = vy + lf * r
    wf = -vx * sd + vf * cd
    wr = vy - lr * r
    uf = torch.sqrt((vx * cd + vf * sd) * (vx * cd + vf * sd) + p.eps * p.eps)
    af = -torch.atan2(wf, uf)
    ar = -torch.atan2(wr, u)
    Fyf = _lateral(af, p.B_f, p.C, Dyf)
    Fyr = _lateral(ar, p.B_r, p.C, Dyr)
    zero, one = torch.zeros_like(cd), torch.ones_like(cd)
    return DynPHParts(
        u=u,
        sg=sg,
        uf=uf,
        Fxf=Fxf,
        Fxr=Fxr,
        Fyf=Fyf,
        Fyr=Fyr,
        wf=wf,
        wr=wr,
        Sf=_secant_w(Fyf, af, wf, zero, p.B_f, p.C, Dyf, uf),
        Sr=_secant_w(Fyr, ar, wr, zero, p.B_r, p.C, Dyr, u),
        Dx=p.drag * u + p.c_rr * p.m * p.g / u,
        cd=cd,
        nf=torch.stack((-sd, cd, lf * cd), dim=-1),
        nr=torch.stack((zero, one, torch.full_like(cd, -lr)), dim=-1),
        tf=torch.stack((cd, sd, lf * sd), dim=-1),
        tr=torch.stack((one, zero, zero), dim=-1),
    )


def dyn_ph_parts_at(p: DynBikeParams, state: torch.Tensor) -> DynPHParts:
    """The parts at a state ``(..., 8)``, at its own ``delta`` and ``a`` (the actuators'
    outputs)."""
    Fxf, Fxr, Dyf, Dyr = _axles(p, state[..., 7])
    return _ph_parts_lateral(p, state, Fxf, Fxr, Dyf, Dyr)


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
    # R = Dx e_x e_x^T + Sf nf nf^T + Sr nr nr^T.
    R = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            R[i][j] = q.Sf * (nf[i] * nf[j]) + q.Sr * (nr[i] * nr[j])
            R[j][i] = R[i][j]
    R[0][0] = R[0][0] + q.Dx
    # f_eps: what the secants leave of the tires' forces (round-off).
    nfv = torch.zeros_like(q.u)
    nrv = torch.zeros_like(q.u)
    for i in range(3):
        nfv = nfv + nf[i] * v[i]
        nrv = nrv + nr[i] * v[i]
    ef = q.Fyf + q.Sf * nfv
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


def dyn_ph_port_work(hdt: float, vx0, vy0, r0, vx1, vy1, r1, W: torch.Tensor) -> torch.Tensor:
    """The external port's energy over a step, ``W`` ``(..., 3)`` held: the trapezoid of its
    power v^T W between the step's start and the twist it reached before its limiters."""
    return 0.5 * hdt * ((vx0 + vx1) * W[..., 0] + (vy0 + vy1) * W[..., 1] + (r0 + r1) * W[..., 2])


def dyn_plant_books(
    p: DynBikeParams,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    grade: torch.Tensor | float = 0.0,
    *,
    wrench: torch.Tensor | None = None,
    side: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, DynPHBooks]:
    """:func:`dyn_plant`'s step, unchanged, and its books.

    The powers are the trapezoid between the step's start, with the inputs it applied (the
    actuators' outputs ``delta1``, ``a1``), and the twist it reached before its limiters;
    ``limiter`` is dH across the clamp and the blend; ``residual`` is the rest. With ``wrench``
    (the external port, held over the step) the books gain its energy, ``port``
    (:func:`dyn_ph_port_work`), and the residual is what that leaves. ``side`` is
    :func:`dyn_plant`'s; the grade line books only the slope along the heading (gravity across it
    is booked as a potential by :func:`dyn_plant_books_ground`).
    """
    nxt, free = dyn_plant(
        p, lim, state, delta_cmd, a_cmd, grade, with_free=True, wrench=wrench, side=side
    )
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
    port = None
    if wrench is not None:
        port = dyn_ph_port_work(
            lim.hdt, state[..., 3], state[..., 4], state[..., 5], *free.unbind(-1), wrench
        )
        residual = residual - port
    return nxt, DynPHBooks(
        dH=dH,
        dissipated=dissipated,
        supplied=supplied,
        floor=floor,
        grade=grade_w,
        limiter=limiter,
        residual=residual,
        port=port,
    )


# ── The full (q, p) form: the pose, and gravity as a potential ──────────────────────────────
# libcvc's dyn_ph_T, dyn_ph_rhs_full and their ground overloads: with the pose q = (x, y, psi)
# (the rear axle) in the state and the ground's height h under the centre of gravity in H,
#
#   H(q, p) = 1/2 p^T M^-1 p + m g h(x + lr cos psi, y + lr sin psi)
#   q' = T(q) v,   p' = -T(q)^T dV/dq + (J(p) - R) v + G u + f_eps + f_g,
#
# T(q) = [[c, -s, lr s], [s, c, -lr c], [0, 0, 1]], and the structure [[0, T], [-T^T, J(p)]] skew
# and Poisson. -T^T dV/dq = -m g (along, across, 0): the ground's gradient at the centre of
# gravity along and across the heading, the slopes' tangents (the force whose work over the planar
# pose is -dV), by central differences over 1e-3 (lf + lr), so any ground serves. Without a
# ground, f_g = -m g (grade, side, 0) carries gravity as data. A ground here is a callable
# ``ground(x, y) -> height``. Each function mirrors its namesake's arithmetic.


def dyn_ph_T(p: DynBikeParams, psi: torch.Tensor) -> torch.Tensor:
    """``T(q)`` ``(..., 3, 3)``: the twist at the centre of gravity to the rear axle's pose
    rates, ``q' = T(q) v``. :func:`dyn_plant`'s pose update is ``q1 = q0 + hdt T(q1) v1``."""
    c, sn = torch.cos(psi), torch.sin(psi)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    lr = p.lr
    return torch.stack(
        (
            torch.stack((c, -sn, lr * sn), dim=-1),
            torch.stack((sn, c, -lr * c), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )


def dyn_ph_ground(p: DynBikeParams, ground, state: torch.Tensor):
    """The ground's height at the centre of gravity and its gradient there by central
    differences: ``(h, gx, gy)``."""
    x, y, psi = state[..., 0], state[..., 1], state[..., 2]
    lr = p.lr
    e = 1e-3 * (p.lf + lr)
    cx = x + lr * torch.cos(psi)
    cy = y + lr * torch.sin(psi)
    h = ground(cx, cy)
    gx = (ground(cx + e, cy) - ground(cx - e, cy)) / (2.0 * e)
    gy = (ground(cx, cy + e) - ground(cx, cy - e)) / (2.0 * e)
    return h, gx, gy


def dyn_ph_slope(p: DynBikeParams, ground, state: torch.Tensor):
    """The ground's slopes at the centre of gravity: along the heading (+ rising ahead) and
    across it (+ rising to the left)."""
    _, gx, gy = dyn_ph_ground(p, ground, state)
    c, sn = torch.cos(state[..., 2]), torch.sin(state[..., 2])
    return c * gx + sn * gy, c * gy - sn * gx


def dyn_ph_potential(p: DynBikeParams, ground, state: torch.Tensor) -> torch.Tensor:
    """``V(q) = m g h`` at the centre of gravity."""
    x, y, psi = state[..., 0], state[..., 1], state[..., 2]
    return p.m * p.g * ground(x + p.lr * torch.cos(psi), y + p.lr * torch.sin(psi))


def dyn_ph_hamiltonian_ground(p: DynBikeParams, ground, state: torch.Tensor) -> torch.Tensor:
    """``H(q, p) = 1/2 p^T M^-1 p + V(q)`` (libcvc's ground overload of ``dyn_ph_hamiltonian``)."""
    return dyn_ph_hamiltonian(p, state) + dyn_ph_potential(p, ground, state)


def dyn_ph_grad_q(p: DynBikeParams, ground, state: torch.Tensor) -> torch.Tensor:
    """``dV/dq = m g (dh/dx, dh/dy, lr (dh/dy cos psi - dh/dx sin psi))`` ``(..., 3)``."""
    _, gx, gy = dyn_ph_ground(p, ground, state)
    mg = p.m * p.g
    psi = state[..., 2]
    return torch.stack(
        (mg * gx, mg * gy, mg * p.lr * (torch.cos(psi) * gy - torch.sin(psi) * gx)), dim=-1
    )


def dyn_ph_rhs(p: DynBikeParams, state: torch.Tensor, grade=0.0) -> torch.Tensor:
    """The momentum-only form's vector field ``(..., 3)``: ``p' = (J - R) v + G u + f_eps -
    m g grade e_x`` at the state."""
    q = dyn_ph_parts_at(p, state)
    v = (state[..., 3], state[..., 4], state[..., 5])
    pm = (p.m * v[0], p.m * v[1], p.Iz * v[2])
    zero = torch.zeros_like(q.u)
    J = ((zero, zero, pm[1]), (zero, zero, -pm[0]), (-pm[1], pm[0], zero))
    nf, nr = q.nf.unbind(-1), q.nr.unbind(-1)
    tf, tr = q.tf.unbind(-1), q.tr.unbind(-1)
    R = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            R[i][j] = q.Sf * (nf[i] * nf[j]) + q.Sr * (nr[i] * nr[j])
            R[j][i] = R[i][j]
    R[0][0] = R[0][0] + q.Dx
    nfv = torch.zeros_like(q.u)
    nrv = torch.zeros_like(q.u)
    for i in range(3):
        nfv = nfv + nf[i] * v[i]
        nrv = nrv + nr[i] * v[i]
    ef = q.Fyf + q.Sf * nfv
    er = q.Fyr + q.Sr * nrv
    out = []
    for i in range(3):
        acc = tf[i] * q.Fxf + tr[i] * q.Fxr + (nf[i] * ef + nr[i] * er)
        for j in range(3):
            acc = acc + (J[i][j] - R[i][j]) * v[j]
        out.append(acc)
    grade = torch.as_tensor(grade, dtype=state.dtype, device=state.device)
    out[0] = out[0] - p.m * p.g * grade
    return torch.stack(out, dim=-1)


def dyn_ph_rhs_full(
    p: DynBikeParams, state: torch.Tensor, grade=0.0, side=0.0, *, ground=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The full form's vector field ``(q', p')``, each ``(..., 3)``.

    Without ``ground``, gravity as data, ``f_g = -m g (grade, side, 0)``: ``q' = T(q) v`` and
    :func:`dyn_ph_rhs`'s ``p'`` with the slope across the heading on ``p_y``. With a ground,
    gravity is in H: ``p' = -T(q)^T dV/dq + (J(p) - R) v + G u + f_eps`` (``grade`` and ``side``
    are not read).
    """
    if ground is not None:
        qd, pd = dyn_ph_rhs_full(p, state, 0.0, 0.0)
        T = dyn_ph_T(p, state[..., 2])
        dV = dyn_ph_grad_q(p, ground, state)
        pd = pd - (
            T[..., 0, :] * dV[..., 0:1] + T[..., 1, :] * dV[..., 1:2] + T[..., 2, :] * dV[..., 2:3]
        )
        return qd, pd
    T = dyn_ph_T(p, state[..., 2])
    v = state[..., 3:6]
    qd = T[..., :, 0] * v[..., 0:1] + T[..., :, 1] * v[..., 1:2] + T[..., :, 2] * v[..., 2:3]
    pd = dyn_ph_rhs(p, state, grade)
    side = torch.as_tensor(side, dtype=state.dtype, device=state.device)
    pd = torch.stack((pd[..., 0], pd[..., 1] - p.m * p.g * side, pd[..., 2]), dim=-1)
    return qd, pd


def dyn_ph_rates_ground(p: DynBikeParams, ground, state: torch.Tensor) -> DynPHRates:
    """The power balance on a ground (libcvc's ground overload of ``dyn_ph_rates_at``): the grade
    line is gravity's power along and across the heading, ``-m g (along vx + across vy) =
    -dV/dt``."""
    along, across = dyn_ph_slope(p, ground, state)
    r = dyn_ph_rates_at(p, state, along)
    return DynPHRates(
        dissipated=r.dissipated,
        supplied=r.supplied,
        floor=r.floor,
        grade=-p.m * p.g * (along * state[..., 3] + across * state[..., 4]),
    )


def dyn_plant_ground(
    p: DynBikeParams,
    lim: PlantLimits,
    ground,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    *,
    with_free: bool = False,
    wrench: torch.Tensor | None = None,
):
    """One step of :func:`dyn_plant` on a ground: gravity along and across the heading, from
    the slopes under the centre of gravity at the step's start (:func:`dyn_ph_slope`)."""
    along, across = dyn_ph_slope(p, ground, state)
    return dyn_plant(
        p, lim, state, delta_cmd, a_cmd, along, with_free=with_free, wrench=wrench, side=across
    )


def dyn_plant_books_ground(
    p: DynBikeParams,
    lim: PlantLimits,
    ground,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    *,
    wrench: torch.Tensor | None = None,
) -> tuple[torch.Tensor, DynPHBooks]:
    """:func:`dyn_plant_ground`'s step and its books: :func:`dyn_plant_books`', with gravity in
    H, so the grade line is ``-dV``, the potential's change over the step (``dH`` stays the
    kinetic energy's: ``dH - grade`` is the change of :func:`dyn_ph_hamiltonian_ground`)."""
    along, across = dyn_ph_slope(p, ground, state)
    nxt, b = dyn_plant_books(p, lim, state, delta_cmd, a_cmd, along, wrench=wrench, side=across)
    dV = dyn_ph_potential(p, ground, nxt) - dyn_ph_potential(p, ground, state)
    return nxt, DynPHBooks(
        dH=b.dH,
        dissipated=b.dissipated,
        supplied=b.supplied,
        floor=b.floor,
        grade=-dV,
        limiter=b.limiter,
        residual=b.residual + b.grade + dV,
        port=b.port,
    )


# ── The wheel-spin plant in port-Hamiltonian form ───────────────────────────────────────────
# libcvc's dyn_spin_ph_*: x = (M v, J_f om_f, J_r om_r), H = 1/2 v^T M v + 1/2 (J_f om_f^2 +
# J_r om_r^2), R = blockdiag(R_body, 0) + K_f d_f d_f^T + K_r d_r d_r^T with d^T dH/dx = sigma, the
# port the wheel torques u = T_d - T_b sg(om) at the wheels' speeds. Each function mirrors its
# namesake's arithmetic, as the dyn_ph_* twins do.


@dataclass(frozen=True)
class DynSpinPHParts:
    """``dyn_spin_ph_parts``: everything the wheel-spin form is built from, at a state."""

    b: DynPHParts  # dyn_ph_parts_at's, with Fxf, Fxr the tires' longitudinal forces
    gf: torch.Tensor  # each wheel's ground speed along it, t^T v
    gr: torch.Tensor
    sf: torch.Tensor  # the patches' slip velocities, r_w om - t^T v
    sr: torch.Tensor
    Kf: torch.Tensor  # the longitudinal secants through sigma = 0
    Kr: torch.Tensor
    Td: torch.Tensor  # (..., 2): the drive's torques, front and rear
    Tb: torch.Tensor  # (..., 2): the brakes' magnitudes
    sgf: torch.Tensor  # the brakes' smoothed signs of the wheels' speeds
    sgr: torch.Tensor
    Tf: torch.Tensor  # the inputs: T_d - T_b sg(om)
    Tr: torch.Tensor


@dataclass(frozen=True)
class DynSpinPHRates:
    """``dyn_spin_ph_rates``, the power balance at a state: dH/dt = supplied - braked -
    dissipated - slip + floor + grade."""

    dissipated: torch.Tensor  # v^T R_body v >= 0: the tires' lateral slip, drag, rolling resistance
    slip: torch.Tensor  # sum K sigma^2 >= 0: the tires' longitudinal slip
    supplied: torch.Tensor  # om^T T_d: the drive's shaft power
    braked: torch.Tensor  # sum T_b om sg(om) >= 0: the brakes'
    floor: torch.Tensor  # v^T f_eps: what the lateral secants leave, round-off
    grade: torch.Tensor  # -m g grade vx


@dataclass(frozen=True)
class DynSpinPHBooks:
    """``dyn_spin_ph_books``, one step's books in J: dH = supplied - braked - dissipated - slip +
    floor + grade + limiter (+ port) + residual."""

    dH: torch.Tensor
    dissipated: torch.Tensor
    slip: torch.Tensor
    supplied: torch.Tensor
    braked: torch.Tensor
    floor: torch.Tensor
    grade: torch.Tensor
    limiter: torch.Tensor  # the kinetic energy the speed clamp and the low-speed blend added
    residual: torch.Tensor  # the integrator's error, second order in hdt per step
    port: torch.Tensor | None = None  # an external wrench's energy, v^T W over the step


def dyn_spin_ph_hamiltonian(p: DynBikeParams, state: torch.Tensor) -> torch.Tensor:
    """The energy: the body's kinetic energy and the wheels' spin, at a state ``(..., 10)``."""
    om_f, om_r = state[..., 8], state[..., 9]
    return dyn_ph_hamiltonian(p, state) + 0.5 * (p.J_f * om_f * om_f + p.J_r * om_r * om_r)


def dyn_spin_ph_parts_at(p: DynBikeParams, state: torch.Tensor) -> DynSpinPHParts:
    """The parts at a state ``(..., 10)``, at its own ``delta`` and ``a``: each tire's
    longitudinal force from its slip, and the lateral grip that leaves."""
    _, _, _, vx, vy, r, delta, a, om_f, om_r = state.unbind(-1)
    rw = p.r_w
    eps2 = p.eps * p.eps
    cap_f, cap_r = _caps(p, a)
    cd, sd = torch.cos(delta), torch.sin(delta)
    gf = vx * cd + (vy + p.lf * r) * sd
    gr = vx
    uf = torch.sqrt(gf * gf + eps2)
    ur = torch.sqrt(gr * gr + eps2)
    sf = rw * om_f - gf
    sr = rw * om_r - gr
    Fxf = _longitudinal(sf, uf, p.B_x, p.C_x, cap_f)
    Fxr = _longitudinal(sr, ur, p.B_x, p.C_x, cap_r)
    Kf = _spin_secant(Fxf, sf, p.B_x, p.C_x, cap_f, uf)
    Kr = _spin_secant(Fxr, sr, p.B_x, p.C_x, cap_r, ur)
    b = _ph_parts_lateral(p, state, Fxf, Fxr, _lateral_cap(Fxf, cap_f), _lateral_cap(Fxr, cap_r))
    Td, Tb = _spin_torques(p, a)
    oe2 = eps2 / (rw * rw)
    sgf = om_f / torch.sqrt(om_f * om_f + oe2)
    sgr = om_r / torch.sqrt(om_r * om_r + oe2)
    return DynSpinPHParts(
        b=b,
        gf=gf,
        gr=gr,
        sf=sf,
        sr=sr,
        Kf=Kf,
        Kr=Kr,
        Td=torch.stack(Td, dim=-1),
        Tb=torch.stack(Tb, dim=-1),
        sgf=sgf,
        sgr=sgr,
        Tf=Td[0] - Tb[0] * sgf,
        Tr=Td[1] - Tb[1] * sgr,
    )


def dyn_spin_ph_rates_at(
    p: DynBikeParams, state: torch.Tensor, grade: torch.Tensor | float = 0.0
) -> DynSpinPHRates:
    """The power balance at a state ``(..., 10)``."""
    q = dyn_spin_ph_parts_at(p, state)
    b = q.b
    vx, om_f, om_r = state[..., 3], state[..., 8], state[..., 9]
    return DynSpinPHRates(
        dissipated=b.Sf * b.wf * b.wf + b.Sr * b.wr * b.wr + b.Dx * vx * vx,
        slip=q.Kf * q.sf * q.sf + q.Kr * q.sr * q.sr,
        supplied=om_f * q.Td[..., 0] + om_r * q.Td[..., 1],
        braked=q.Tb[..., 0] * om_f * q.sgf + q.Tb[..., 1] * om_r * q.sgr,
        # v^T f_eps: nf^T v = wf, nr^T v = wr.
        floor=b.wf * (b.Fyf + b.Sf * b.wf) + b.wr * (b.Fyr + b.Sr * b.wr),
        grade=-p.m * p.g * grade * vx,
    )


def dyn_spin_plant_books(
    p: DynBikeParams,
    lim: PlantLimits,
    state: torch.Tensor,
    delta_cmd: torch.Tensor,
    a_cmd: torch.Tensor,
    grade: torch.Tensor | float = 0.0,
    *,
    wrench: torch.Tensor | None = None,
) -> tuple[torch.Tensor, DynSpinPHBooks]:
    """:func:`dyn_spin_plant`'s step, unchanged, and its books: as :func:`dyn_plant_books`, the
    trapezoid running to the state the step reached before its limiters, the wheels' new speeds
    included (the limiters act on the body alone). ``wrench`` as :func:`dyn_plant_books`'."""
    nxt, free = dyn_spin_plant(
        p, lim, state, delta_cmd, a_cmd, grade, with_free=True, wrench=wrench
    )
    s0 = torch.cat((state[..., :6], nxt[..., 6:8], state[..., 8:]), dim=-1)
    e = torch.cat((nxt[..., :3], free, nxt[..., 6:]), dim=-1)
    r0, r1 = dyn_spin_ph_rates_at(p, s0, grade), dyn_spin_ph_rates_at(p, e, grade)
    half = 0.5 * lim.hdt
    dissipated = half * (r0.dissipated + r1.dissipated)
    slip = half * (r0.slip + r1.slip)
    supplied = half * (r0.supplied + r1.supplied)
    braked = half * (r0.braked + r1.braked)
    floor = half * (r0.floor + r1.floor)
    grade_w = half * (r0.grade + r1.grade)
    h0 = dyn_spin_ph_hamiltonian(p, state)
    he = dyn_spin_ph_hamiltonian(p, e)
    h1 = dyn_spin_ph_hamiltonian(p, nxt)
    dH = h1 - h0
    limiter = h1 - he
    residual = dH - (supplied - braked - dissipated - slip + floor + grade_w + limiter)
    port = None
    if wrench is not None:
        port = dyn_ph_port_work(
            lim.hdt, state[..., 3], state[..., 4], state[..., 5], *free.unbind(-1), wrench
        )
        residual = residual - port
    return nxt, DynSpinPHBooks(
        dH=dH,
        dissipated=dissipated,
        slip=slip,
        supplied=supplied,
        braked=braked,
        floor=floor,
        grade=grade_w,
        limiter=limiter,
        residual=residual,
        port=port,
    )
