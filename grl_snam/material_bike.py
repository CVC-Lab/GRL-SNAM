"""The material rollout on the kinematic bicycle, differentiable in torch.

The torch twin of libcvc's ``cvc::nav::integrate_bicycle_material`` (P8b of the
vehicle-physics roadmap). The force law is the material surrogate's
(:func:`grl_snam.material_nav.integrate_surrogate_material`):

- per-obstacle IPC barriers on the discs ``(C, R)``;
- the goal spring and damping;
- the soft and hard material forces from the 6-channel patch.

Instead of integrating a point mass, it drives the deployed kinematic bicycle
(``drive.cpp``'s ``bicycle_rollout_material``, one substep per step):

- the longitudinal command ``(F . heading - gamma * sp) / mass``, clamped;
- pure pursuit toward the goal, plus the steering bias
  ``k_steer * tanh((F_rep + F_mat) . left)``;
- the corner, stop, creep and reverse governors;
- the ``d_cap`` steering limit and the kinematic update.

The governor reads the nearest disc's clearance and normal.

Every branch is a ``torch.where``, so autograd follows the branch each step took,
which is what the C++ hand-written adjoint does. Square roots and norms are
guarded so an unselected zero never turns a gradient into NaN. The C++ forward
and gradient are checked against this one in
``tests/test_material_bike_parity.py``.

State per agent: position ``o`` (B,2), heading ``th`` (B,) and signed speed ``sp``
(B,). Lengths are in the batch's units; the vehicle limits use the same units.

:func:`integrate_dynamic_material` is libcvc's ``integrate_dynamic_material``: the same controller,
its commands ``(delta, a)`` driving the dynamic single-track vehicle
(:func:`grl_snam.dyn_bike.dyn_plant`) in place of the kinematic update, so the material policy
learns what the vehicle can do. Its state adds ``x`` (B,4) = (vy, r, delta, a); arc length and
cumulative risk follow the dynamic path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grl_snam.material_nav import bilinear_sample_patch, sdf_barrier_grad
from grl_snam.train_coef_energy import ipc_piecewise


@dataclass(frozen=True)
class BicycleParams:
    """Vehicle limits (``cvc::nav::bicycle_material_params``)."""

    L: float = 0.035  # wheelbase
    delta_max: float = 0.6  # steering lock, rad
    a_max: float = 1.5  # longitudinal acceleration limit
    a_lat_max: float = 1.0  # lateral acceleration limit
    k_steer: float = 0.8  # steering-bias gain
    vmax: float = 0.9  # speed limit
    allow_reverse: bool = True


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt(x) for x >= 0 whose gradient is 0, not NaN, at x == 0."""
    pos = x > 0
    return torch.where(
        pos, torch.sqrt(torch.where(pos, x, torch.ones_like(x))), torch.zeros_like(x)
    )


def _clamp(x: torch.Tensor, lo, hi) -> torch.Tensor:
    """clamp as the C++ writes it: below lo -> lo, above hi -> hi, else x."""
    lo_t = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi_t = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return torch.where(x < lo_t, lo_t.expand_as(x), torch.where(x > hi_t, hi_t.expand_as(x), x))


def _material_rollout(
    o0,
    th0,
    sp0,
    x0,
    goal,
    C,
    R,
    mask,
    alphas,
    beta,
    gamma,
    lam_soft,
    lam_hard,
    rollout_patch,
    rr,
    d_hat,
    dt,
    H,
    vehicle,
    margin_factor,
    mass,
    d_hat_sdf,
    k_sharp,
    ext,
    dyn,
):
    """The material controller on the kinematic bicycle (``dyn`` None) or on the dynamic
    vehicle (``dyn`` a :class:`grl_snam.dyn_bike.DynBikeParams`, ``x0`` its further state)."""
    B, N = C.shape[:2]
    dtype, dev = o0.dtype, o0.device
    v = vehicle
    L = float(v.L)
    dmax = float(v.delta_max)
    tan_dmax = torch.tan(torch.tensor(dmax, dtype=dtype, device=dev))
    a_max, a_lat, vmax = float(v.a_max), float(v.a_lat_max), float(v.vmax)
    sp_min = -0.25 * vmax if v.allow_reverse else 0.0
    v_creep_cap = 0.5 * torch.sqrt(torch.tensor(a_lat * L, dtype=dtype, device=dev) / tan_dmax)
    kfloor = tan_dmax / (L * 400.0)
    rr_eff = margin_factor * rr
    R_eff = R + rr_eff[:, None]
    mask = mask.to(torch.bool)
    rows = torch.arange(B, device=dev)
    hdt = dt

    o, th, sp = o0.clone(), th0.clone(), sp0.clone()
    x = None
    if dyn is not None:
        x = torch.zeros(B, 4, dtype=dtype, device=dev) if x0 is None else x0.clone()
    min_clear = torch.full((B,), float("inf"), dtype=dtype, device=dev)
    cum_risk = torch.zeros(B, dtype=dtype, device=dev)
    hard_count = torch.zeros(B, dtype=dtype, device=dev)
    arc_length = torch.zeros(B, dtype=dtype, device=dev)
    zero, one = torch.zeros(B, dtype=dtype, device=dev), torch.ones(B, dtype=dtype, device=dev)
    if ext is not None:
        from grl_snam.material_ext import ExtForce

        ext_force = ExtForce(ext, o0)

    for s in range(int(H.max().item()) if B else 0):
        active = s < H
        ch, sh = torch.cos(th), torch.sin(th)

        sem = bilinear_sample_patch(rollout_patch, o, o0)
        risk = sem[:, 0].clamp(0.0, 1.0)
        sdf = sem[:, 1].clamp(0.0, 50.0)
        grx, gry, gpx, gpy = sem[:, 2], sem[:, 3], sem[:, 4], sem[:, 5]

        # Geometry: the barrier summed over discs; the governor reads the nearest.
        if N:
            diff = o.unsqueeze(1) - C
            r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
            n = diff / r.unsqueeze(-1)
            d = r - R_eff
            d_m = torch.where(mask, d, torch.full_like(d, 1e6))
            _, dbdd = ipc_piecewise(d_m, d_hat.view(-1, 1))
            dbdd = torch.where(mask, dbdd, torch.zeros_like(dbdd))
            Fg = (-(alphas * dbdd).unsqueeze(-1) * n).sum(1)
            brep = dbdd.clamp(max=0.0)
            Fr = (-(alphas * brep).unsqueeze(-1) * n).sum(1)
            d_inf = torch.where(mask, d, torch.full_like(d, float("inf")))
            dmin, jmin = d_inf.min(dim=1)
            has = torch.isfinite(dmin)
            d_gov = torch.where(has, dmin, torch.full_like(dmin, 1e30))
            n_near = n[rows, jmin]
            ngx = torch.where(has, n_near[:, 0], zero)
            ngy = torch.where(has, n_near[:, 1], zero)
        else:
            Fg = torch.zeros_like(o)
            Fr = torch.zeros_like(o)
            dmin = torch.full((B,), float("inf"), dtype=dtype, device=dev)
            d_gov = torch.full((B,), 1e30, dtype=dtype, device=dev)
            ngx = ngy = zero
        min_clear = torch.minimum(min_clear, dmin)

        Fgoal = -beta.unsqueeze(-1) * (o - goal)
        _, db = sdf_barrier_grad(sdf, d_hat_sdf=d_hat_sdf, k_sharp=k_sharp)
        Fmx = -lam_soft * grx + (-lam_hard * db) * gpx
        Fmy = -lam_soft * gry + (-lam_hard * db) * gpy
        Fx = (Fg[:, 0] + Fgoal[:, 0]) + Fmx
        Fy = (Fg[:, 1] + Fgoal[:, 1]) + Fmy
        Fe = ext_force.at(s, o) if ext is not None else None
        if Fe is not None:  # libcvc adds the external force last
            Fx = Fx + Fe[:, 0]
            Fy = Fy + Fe[:, 1]

        a0c = _clamp(((Fx * ch + Fy * sh) - gamma * sp) / mass, -a_max, a_max)

        # Pure pursuit toward the goal.
        tgx, tgy = goal[:, 0] - o[:, 0], goal[:, 1] - o[:, 1]
        Ld_raw = _safe_sqrt(tgx * tgx + tgy * tgy)
        Ld = torch.where(Ld_raw < 1e-6, torch.full_like(Ld_raw, 1e-6), Ld_raw)
        ang = torch.atan2(tgy, tgx) - th
        sa = torch.sin(ang)
        behind = torch.cos(ang) < 0.0
        turn_sign = torch.where(sa >= 0.0, one, -one)
        cap_raw = 1.2 * sp + 4.0 * L
        cap = torch.where(cap_raw < 4.0 * L, torch.full_like(cap_raw, 4.0 * L), cap_raw)
        Ldeff = torch.where(cap < Ld, cap, Ld)
        delta0 = torch.where(behind, turn_sign * dmax, torch.atan2(2.0 * L * sa, Ldeff))
        bsx, bsy = Fr[:, 0] + Fmx, Fr[:, 1] + Fmy
        if Fe is not None and ext.steer:
            bsx = bsx + Fe[:, 0]
            bsy = bsy + Fe[:, 1]
        S = bsx * (-sh) + bsy * ch
        delta2 = _clamp(delta0 + v.k_steer * torch.tanh(S), -dmax, dmax)

        # Governors.
        kappa_raw = torch.abs(torch.tan(delta2)) / L
        kappa = torch.where(kappa_raw < kfloor, kfloor.expand_as(kappa_raw), kappa_raw)
        v_corner = torch.sqrt(a_lat / kappa)
        ds_raw = d_gov - 0.5 * rr_eff
        ds = torch.where(ds_raw < 0.0, zero, ds_raw)
        v_stop = _safe_sqrt(2.0 * a_max * ds)
        msign = torch.where(sp >= 0.0, one, -one)
        appr = _clamp(-(ngx * ch + ngy * sh) * msign, 0.0, 1.0)
        v_stop_dir = torch.where(
            appr > 0.05, v_stop / appr.clamp_min(0.05), torch.full_like(appr, vmax)
        )
        m = torch.where(v_stop_dir < v_corner, v_stop_dir, v_corner)
        vlim0 = torch.where(m < vmax, m, torch.full_like(m, vmax))
        hard_steer = torch.abs(delta2) >= 0.7 * dmax
        can_move = d_gov > 0.25 * rr_eff
        v_floor = torch.where(hard_steer & can_move, torch.full_like(m, 0.08), zero)
        v_lim = torch.where(vlim0 < v_floor, v_floor, vlim0)
        half_vc = 0.5 * v_corner
        v_creep = torch.where(v_creep_cap < half_vc, v_creep_cap.expand_as(half_vc), half_vc)

        fbh_raw = Fg[:, 0] * ch + Fg[:, 1] * sh
        fbh = torch.where(fbh_raw > 0.0, zero, fbh_raw)
        a_bh = torch.where(behind, (v_creep - sp) / hdt + fbh, a0c)
        stuck = hard_steer & can_move & (torch.abs(sp) < 0.06)
        floor_a = (0.08 - sp) / hdt
        a_st = torch.where(stuck & (a_bh < floor_a), floor_a, a_bh)
        if v.allow_reverse:
            head_on = _clamp(-(ngx * ch + ngy * sh), 0.0, 1.0)
            rev = behind & (head_on > 0.6) & (d_gov < 0.5 * rr_eff + 0.02)
        else:
            rev = torch.zeros(B, dtype=torch.bool, device=dev)
        a1 = torch.where(rev, (-0.10 - sp) / hdt, a_st)
        delta3 = torch.where(rev, -delta2, delta2)
        a1c = _clamp(a1, -a_max, a_max)
        lim = (v_lim - sp) / hdt
        a3 = _clamp(torch.where(lim < a1c, lim, a1c), -a_max, a_max)

        sp1 = _clamp(sp + hdt * a3, sp_min, vmax)
        sp2 = sp1 * sp1
        sp2 = torch.where(sp2 < 1e-9, torch.full_like(sp2, 1e-9), sp2)
        d_cap = torch.atan(a_lat * L / sp2)
        delta4 = torch.where(delta3 < -d_cap, -d_cap, torch.where(delta3 > d_cap, d_cap, delta3))
        if dyn is None:
            th1 = th + hdt * (sp1 / L) * torch.tan(delta4)
            o1 = torch.stack(
                [o[:, 0] + hdt * sp1 * torch.cos(th1), o[:, 1] + hdt * sp1 * torch.sin(th1)], -1
            )
        else:
            # The commands (delta4, a3) drive the dynamic vehicle in place of the kinematic
            # update (libcvc's step_fwd_dyn); sp1 above was only the lateral cap's speed.
            nxt = _dyn_step(
                dyn, v, torch.cat([o, th[:, None], sp[:, None], x], -1), delta4, a3, hdt
            )
            o1, th1, sp1, x1 = nxt[:, :2], nxt[:, 2], nxt[:, 3], nxt[:, 4:]

        sdx, sdy = o1[:, 0] - o[:, 0], o1[:, 1] - o[:, 1]
        disp = torch.where(active, _safe_sqrt(sdx * sdx + sdy * sdy), zero)
        arc_length = arc_length + disp
        cum_risk = cum_risk + risk * disp
        hard_count = hard_count + torch.where(active & (sdf < 1.0), one, zero)

        o = torch.where(active.unsqueeze(-1), o1, o)
        th = torch.where(active, th1, th)
        sp = torch.where(active, sp1, sp)
        if dyn is not None:
            x = torch.where(active.unsqueeze(-1), x1, x)

    if dyn is not None:
        return o, th, sp, x, min_clear, cum_risk, hard_count, arc_length
    return o, th, sp, min_clear, cum_risk, hard_count, arc_length


def _dyn_step(dyn, v: BicycleParams, state, delta_cmd, a_cmd, hdt):
    """One step of the dynamic vehicle, each agent at its own step ``hdt`` (B,)."""
    from grl_snam.dyn_bike import PlantLimits, dyn_plant

    out = None
    for u in torch.unique(hdt).tolist():
        lim = PlantLimits(L=float(v.L), vmax=float(v.vmax), hdt=u, allow_reverse=v.allow_reverse)
        r = dyn_plant(dyn, lim, state, delta_cmd, a_cmd)
        out = r if out is None else torch.where((hdt == u).unsqueeze(-1), r, out)
    return out


def integrate_bicycle_material(
    o0: torch.Tensor,
    th0: torch.Tensor,
    sp0: torch.Tensor,
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
    rr: torch.Tensor,
    d_hat: torch.Tensor,
    dt: torch.Tensor,
    H: torch.Tensor,
    vehicle: BicycleParams = BicycleParams(),
    margin_factor: float = 0.5,
    mass: float = 1.0,
    d_hat_sdf: float = 3.0,
    k_sharp: float = 5.0,
    ext=None,
) -> tuple[torch.Tensor, ...]:
    """Roll the bicycle out; returns (oT, thT, spT, min_clear, cum_risk, hard_count, arc_length).

    Shapes as :func:`grl_snam.material_nav.integrate_surrogate_material`, with the start
    heading ``th0`` and signed speed ``sp0`` (B,) in place of ``v0``. ``mask`` (B,N) bool.
    ``ext`` (a :class:`grl_snam.material_ext.MaterialExt`) adds an external force to F and,
    with its ``steer``, to the steering bias, as libcvc's ``material_ext`` does.
    """
    return _material_rollout(
        o0,
        th0,
        sp0,
        None,
        goal,
        C,
        R,
        mask,
        alphas,
        beta,
        gamma,
        lam_soft,
        lam_hard,
        rollout_patch,
        rr,
        d_hat,
        dt,
        H,
        vehicle,
        margin_factor,
        mass,
        d_hat_sdf,
        k_sharp,
        ext,
        None,
    )


def integrate_dynamic_material(
    o0: torch.Tensor,
    th0: torch.Tensor,
    vx0: torch.Tensor,
    x0: torch.Tensor | None,
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
    rr: torch.Tensor,
    d_hat: torch.Tensor,
    dt: torch.Tensor,
    H: torch.Tensor,
    vehicle: BicycleParams,
    dyn,
    margin_factor: float = 0.5,
    mass: float = 1.0,
    d_hat_sdf: float = 3.0,
    k_sharp: float = 5.0,
    ext=None,
) -> tuple[torch.Tensor, ...]:
    """The material controller on the dynamic vehicle (libcvc's ``integrate_dynamic_material``).

    As :func:`integrate_bicycle_material`, with the rear axle ``o0``, heading ``th0`` and
    longitudinal speed ``vx0``, and ``x0`` (B,4) = (vy, r, delta, a) (None: at rest). ``dyn`` is a
    :class:`grl_snam.dyn_bike.DynBikeParams` in the batch's units whose wheelbase ``lf + lr`` is
    ``vehicle.L``. Returns (oT, thT, vxT, xT, min_clear, cum_risk, hard_count, arc_length).
    """
    if abs(float(dyn.lf) + float(dyn.lr) - float(vehicle.L)) > 1e-3 * float(vehicle.L):
        raise ValueError("integrate_dynamic_material: the vehicle's lf + lr must be vehicle.L")
    return _material_rollout(
        o0,
        th0,
        vx0,
        x0,
        goal,
        C,
        R,
        mask,
        alphas,
        beta,
        gamma,
        lam_soft,
        lam_hard,
        rollout_patch,
        rr,
        d_hat,
        dt,
        H,
        vehicle,
        margin_factor,
        mass,
        d_hat_sdf,
        k_sharp,
        ext,
        dyn,
    )
