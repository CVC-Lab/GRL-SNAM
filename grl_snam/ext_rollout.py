"""The torch twin of libcvc's ``diff::bike_rollout_ext`` (``inc/cvc/nav/ext_rollout.h``).

A batch of agents' kinematic-bicycle trajectories with an external force, differentiable by
autograd. It is the trainer-side form of the deployed drive's ``ext_force`` port: cvcdbg's
comm-network trainer learns what produces the force through the real controller and vehicle.

* **The step** is :func:`grl_snam.sdf_nav.bicycle_rollout` (the deployed drive's twin) with one
  substep, so one step here is one substep of the drive.
* **Goals and coefficients** may change at every step: ``goals`` is ``[T, B, 2]`` and ``coefs``
  ``[T, B, 3]`` (alpha, beta, gamma). They are exogenous, recorded from the carrot FSM and
  coef_mlp, so nothing flows from them back into the pose.
* **The force**: ``sample_fn(t, o) -> [B, 2]`` in the normalized drive frame, called at the
  pose of step ``t`` every ``hold`` steps and held in between, once per sampled step in step
  order. Its dependence on ``o`` is the position Jacobian; autograd carries it, as ``J`` does
  in C++. ``steer`` is ``ext_force::steer``.
* **dL/dF_ext per step**, which the C++ reverse returns, is the gradient with respect to
  ``f_offsets`` (``[T, B, 2]``, added to the force at every step; pass zeros that require
  grad).
* **Precision and device**: pass float64 tensors and :func:`field_as` (``torch.float64``) for
  float64; put the tensors and the field on CUDA to run there.

Returns the states ``[T + 1, B, 4]``: ``(ox, oy, th, sp)`` at steps ``0..T``.

:func:`dyn4_rollout_ext` is the same on the four-wheel vehicle (libcvc's ``dyn4_rollout_ext``):
the controller's commands (:func:`grl_snam.sdf_nav.bike_control`) drive
:func:`grl_snam.dyn4.dyn4_plant`, and the state is its 14 entries.
"""

from __future__ import annotations

import copy

import torch

from grl_snam import sdf_nav

# The vehicle keys bike_rollout_ext reads: libcvc's bike_veh (hdt is the step).
VEH_KEYS = ("rr", "d_hat", "hdt", "vmax", "L", "delta_max", "a_max", "a_lat_max", "k_steer")


def field_as(field: sdf_nav.SDFField, dtype=None, device=None) -> sdf_nav.SDFField:
    """A copy of ``field`` whose grid is ``dtype`` on ``device`` (SDFField stores float32)."""
    out = copy.copy(field)
    out.field = field.field.to(
        dtype=dtype or field.field.dtype, device=device or field.field.device
    )
    if device is not None:
        out.dev = torch.device(device)
    return out


def bike_rollout_ext(
    field,
    o: torch.Tensor,
    th: torch.Tensor,
    sp: torch.Tensor,
    goals: torch.Tensor,
    coefs: torch.Tensor,
    sample_fn=None,
    *,
    veh: dict,
    hold: int = 1,
    steer: bool = True,
    allow_reverse: bool = True,
    f_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """``o [B,2]``, ``th [B]``, ``sp [B]``: the start. See the module docstring."""
    if hold < 1:
        raise ValueError("bike_rollout_ext: hold must be >= 1")
    T = goals.shape[0]
    if coefs.shape[0] != T or (f_offsets is not None and f_offsets.shape[0] != T):
        raise ValueError("bike_rollout_ext: goals, coefs and f_offsets need T steps each")
    kw = dict(
        rr=veh["rr"],
        d_hat=veh["d_hat"],
        dt=veh["hdt"],
        nsub=1,
        vmax=veh["vmax"],
        L=veh["L"],
        delta_max=veh["delta_max"],
        a_max=veh["a_max"],
        a_lat_max=veh["a_lat_max"],
        k_steer=veh["k_steer"],
        allow_reverse=allow_reverse,
        ext_steer=steer,
    )
    states = [torch.cat([o, th[:, None], sp[:, None]], -1)]
    F = None
    for t in range(T):
        if sample_fn is not None and t % hold == 0:
            F = sample_fn(t, o)
        Ft = F
        if f_offsets is not None:
            Ft = f_offsets[t] if Ft is None else Ft + f_offsets[t]
        o, th, sp, _ = sdf_nav.bicycle_rollout(
            field,
            o,
            th,
            sp,
            goals[t],
            coefs[t, :, 0],
            coefs[t, :, 1],
            coefs[t, :, 2],
            1,
            ext_force_fn=None if Ft is None else (lambda _o, _f=Ft: _f),
            **kw,
        )
        states.append(torch.cat([o, th[:, None], sp[:, None]], -1))
    return torch.stack(states, 0)


def dyn4_rollout_ext(
    field,
    ground,
    state: torch.Tensor,
    goals: torch.Tensor,
    coefs: torch.Tensor,
    sample_fn=None,
    *,
    veh: dict,
    params,
    hold: int = 1,
    steer: bool = True,
    allow_reverse: bool = True,
    f_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """The torch twin of libcvc's ``diff::dyn4_rollout_ext``: the four-wheel vehicle's trajectory.

    Each step is libcvc's ``dyn4_bike_step``: the controller (:func:`grl_snam.sdf_nav.bike_control`,
    reading the rear axle's pose and the speed ``vx``) gives ``(delta_cmd, a_cmd)`` to the plant
    (:func:`grl_snam.dyn4.dyn4_plant`) on ``ground``, a callable ``(wx, wy) -> (mu, height)``.
    ``state`` is ``[B, 14]`` in :data:`grl_snam.dyn4.ORDER`, ``params`` a
    :class:`grl_snam.dyn4.Dyn4Params`. Goals, coefficients, the force (``sample_fn``, ``hold``,
    ``steer``) and ``f_offsets`` are as :func:`bike_rollout_ext`'s, the force sampled at the rear
    axle's position. Returns the states ``[T + 1, B, 14]``.
    """
    from grl_snam.dyn4 import PlantLimits, dyn4_plant

    if hold < 1:
        raise ValueError("dyn4_rollout_ext: hold must be >= 1")
    T = goals.shape[0]
    if coefs.shape[0] != T or (f_offsets is not None and f_offsets.shape[0] != T):
        raise ValueError("dyn4_rollout_ext: goals, coefs and f_offsets need T steps each")
    lim = PlantLimits(L=veh["L"], vmax=veh["vmax"], hdt=veh["hdt"], allow_reverse=allow_reverse)
    kw = dict(
        rr=veh["rr"],
        d_hat=veh["d_hat"],
        dt=veh["hdt"],
        vmax=veh["vmax"],
        L=veh["L"],
        delta_max=veh["delta_max"],
        a_max=veh["a_max"],
        a_lat_max=veh["a_lat_max"],
        k_steer=veh["k_steer"],
        allow_reverse=allow_reverse,
        ext_steer=steer,
    )
    states = [state]
    F = None
    for t in range(T):
        o = state[:, :2]
        if sample_fn is not None and t % hold == 0:
            F = sample_fn(t, o)
        Ft = F
        if f_offsets is not None:
            Ft = f_offsets[t] if Ft is None else Ft + f_offsets[t]
        delta_cmd, a_cmd, _, _ = sdf_nav.bike_control(
            field,
            o,
            state[:, 2],
            state[:, 3],
            goals[t],
            coefs[t, :, 0],
            coefs[t, :, 1],
            coefs[t, :, 2],
            ext_force_fn=None if Ft is None else (lambda _o, _f=Ft: _f),
            **kw,
        )
        state = dyn4_plant(params, lim, state, delta_cmd, a_cmd, ground)
        states.append(state)
    return torch.stack(states, 0)
