"""U_nav, the navigator's potential, evaluated as a number (torch twin of libcvc's
``cvc/nav/detail/nav_potential.h``).

The force laws here and in libcvc are written as closed-form derivatives. These are the potentials
whose derivatives they are, with the coefficients held fixed::

    U_nav = beta/2 |o - g|^2 + sum_j alpha_j b_IPC(d_j) + lam_soft r~(o) + lam_hard b_sp(phi(o))
            (+ the terrain's grade cost and slope barrier, the same two forms on its own plane)

Damping (``-gamma v``) is dissipation and the external force is a port; neither is part of U. Each
barrier potential is the one whose derivative the matching force primitive returns, its clamps
included, so ``F = -grad U`` holds wherever the sampled planes are consistent. libcvc's
``nav_potential_test`` gradchecks that at every call site and writes
``src/cvc/tests/data/nav_potential_golden.json``, which ``tests/test_nav_potential_parity.py``
holds these functions to.

- :func:`ipc_potential`: the barrier of ``sdf_nav._ipc_dbdd`` (the deployed drive and the coef
  trainer's rollouts), continued linearly below its 1e-6 floor.
- :func:`ipc_potential_pw`: the barrier of ``train_coef_energy.ipc_piecewise``'s clamped
  derivative (the material rollouts). Its ``b`` output is clamped separately from ``dbdd``, so it
  is not the potential of the force inside the clamp band; this is.
- :func:`sdf_barrier_potential`: the softplus hazard barrier (``sdf_barrier_grad``'s ``b``).
- :func:`material_potential`, :func:`surrogate_potential`, :func:`drive_potential`: U at a pose
  for the material rollouts, the coef trainer's surrogate and bicycle, and the deployed drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

#: The material rollouts clamp the barrier derivative to [-MAX_GRAD, MAX_GRAD].
MAX_GRAD = 200.0


@dataclass
class PotentialTerms:
    """The terms of U_nav at a batch of poses (each ``[B]``); ``terrain`` is the deployed drive's."""

    goal: torch.Tensor
    barrier: torch.Tensor
    soft: torch.Tensor
    hard: torch.Tensor
    terrain: torch.Tensor

    def total(self) -> torch.Tensor:
        return self.goal + self.barrier + self.soft + self.hard + self.terrain

    def stacked(self) -> torch.Tensor:
        """``[B, 5]`` in the order goal, barrier, soft, hard, terrain."""
        return torch.stack([self.goal, self.barrier, self.soft, self.hard, self.terrain], -1)


def _as(x, like: torch.Tensor) -> torch.Tensor:
    return x.to(like) if torch.is_tensor(x) else like.new_tensor(float(x))


def _ipc_core(d: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    return -((d - d_hat) ** 2) * torch.log(d / d_hat)


def ipc_potential(d: torch.Tensor, d_hat) -> torch.Tensor:
    """The barrier whose derivative is ``sdf_nav._ipc_dbdd``: ``b(d) = -(d - dh)^2 ln(d/dh)`` on
    ``[1e-6, dh)``, zero from ``dh`` up, and below 1e-6 (where the force holds ``b'(1e-6)``)
    continued linearly with that slope. libcvc ``detail::ipc_b``."""
    dh = _as(d_hat, d)
    lo = 1e-6
    dc = d.clamp_min(lo)
    A = lo - dh
    slope = -(2.0 * A * torch.log(lo / dh) + A * A / lo)
    below = _ipc_core(d.new_tensor(lo), dh) + slope * (d - lo)
    out = torch.where(d >= lo, _ipc_core(dc, dh), below)
    return torch.where(dc < dh, out, torch.zeros_like(d))


def ipc_pw_knee(d_hat, max_grad: float = MAX_GRAD) -> torch.Tensor:
    """Where the clamped barrier derivative starts to clamp: ``b'(d) = -max_grad`` on ``(0, dh)``
    (``b'`` rises monotonically there). Bisection in float64, libcvc ``detail::ipc_pw_knee``."""
    dh = torch.as_tensor(d_hat, dtype=torch.float64)
    lo = torch.zeros_like(dh)
    hi = dh.clone()
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        A = mid - dh
        bp = -(2.0 * A * torch.log(mid / dh) + A * A / mid)
        below = bp < -max_grad
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return hi


def ipc_potential_pw(d: torch.Tensor, d_hat, max_grad: float = MAX_GRAD) -> torch.Tensor:
    """The barrier whose derivative is ``ipc_piecewise``'s clamped ``dbdd`` (the material
    rollouts): ``b`` where the derivative is inside ``[-max_grad, max_grad]``, continued linearly
    with slope ``-max_grad`` below the knee (and through ``d <= 1e-9``, whose -500 the clamp also
    makes ``-max_grad``). Zero from ``dh`` up. libcvc ``detail::ipc_b_pw``."""
    dh = _as(d_hat, d)
    eps = 1e-9
    safe = d.clamp_min(eps)
    bp = -(2.0 * (d - dh) * torch.log(safe / dh) + (d - dh) ** 2 / safe)
    in_band = (d > eps) & ~(bp < -max_grad)
    knee = ipc_pw_knee(dh.detach(), max_grad).to(d)
    linear = _ipc_core(knee, dh) + max_grad * (knee - d)
    out = torch.where(in_band, _ipc_core(safe, dh), linear)
    return torch.where(d < dh, out, torch.zeros_like(d))


def sdf_barrier_potential(phi: torch.Tensor, k_sharp, d_hat) -> torch.Tensor:
    """``b_sp(phi) = softplus(k (d_hat - phi)) / k``, whose derivative is
    ``-sigmoid(k (d_hat - phi))`` (``material_nav.sdf_barrier_grad``). libcvc
    ``detail::sdf_barrier_b``."""
    k = _as(k_sharp, phi)
    return F.softplus(k * (_as(d_hat, phi) - phi)) / k


def material_potential(
    o: torch.Tensor,
    goal: torch.Tensor,
    C: torch.Tensor,
    R: torch.Tensor,
    mask: torch.Tensor,
    alphas: torch.Tensor,
    beta: torch.Tensor,
    lam_soft: torch.Tensor,
    lam_hard: torch.Tensor,
    rollout_patch: torch.Tensor,
    o0: torch.Tensor,
    d_hat: torch.Tensor,
    robot_radius: float | torch.Tensor = 0.0,
    margin_factor: float = 0.5,
    d_hat_sdf: float = 3.0,
    k_sharp: float = 5.0,
) -> PotentialTerms:
    """U at ``o`` for :func:`grl_snam.material_nav.integrate_surrogate_material`'s force (shapes as
    there): the discs' barriers (:func:`ipc_potential_pw`), the goal spring, and the patch's soft and
    hard terms with the patch values clamped as the forces read them. The soft and hard terms are
    the potentials of the forces when the patch's channels 2-5 are the gradients of channels 0-1
    in the batch's units (the caller's convention). libcvc ``detail::material_potential``."""
    from grl_snam.material_nav import bilinear_sample_patch

    if not torch.is_tensor(robot_radius):
        rr = o.new_tensor(float(robot_radius))
    else:
        rr = robot_radius.to(o)
    R_eff = R + margin_factor * rr[:, None] if rr.ndim >= 1 else R + margin_factor * rr
    goal_t = 0.5 * beta * ((o - goal) ** 2).sum(-1)
    if C.shape[1] == 0:
        barrier = torch.zeros_like(goal_t)
    else:
        r = torch.linalg.norm(o.unsqueeze(1) - C, dim=-1).clamp_min(1e-9)
        b = ipc_potential_pw(r - R_eff, d_hat.view(-1, 1).expand_as(r))
        barrier = torch.where(mask, alphas * b, torch.zeros_like(b)).sum(-1)
    sem = bilinear_sample_patch(rollout_patch, o, o0)
    risk = sem[:, 0].clamp(0.0, 1.0)
    sdf = sem[:, 1].clamp(0.0, 50.0)
    return PotentialTerms(
        goal=goal_t,
        barrier=barrier,
        soft=lam_soft * risk,
        hard=lam_hard * sdf_barrier_potential(sdf, k_sharp, d_hat_sdf),
        terrain=torch.zeros_like(goal_t),
    )


def surrogate_potential(field, o, goal, alpha, beta, rr, d_hat) -> PotentialTerms:
    """U at normalized ``o`` (``[B, 2]``) for the force of ``sdf_nav.sdf_rollout`` and the
    bicycle's controller: ``beta/2 |o - g|^2 + alpha b(phi(o) - rr)``. ``field`` is an
    :class:`grl_snam.sdf_nav.SDFField`. libcvc ``diff::surr_potential``."""
    phi, _ = field.sample(o)
    goal_t = 0.5 * _as(beta, o) * ((o - goal) ** 2).sum(-1)
    barrier = _as(alpha, o) * ipc_potential(phi - rr, d_hat)
    z = torch.zeros_like(goal_t)
    return PotentialTerms(goal=goal_t, barrier=barrier, soft=z, hard=z.clone(), terrain=z.clone())


def drive_potential(
    field,
    o: torch.Tensor,
    th: torch.Tensor,
    goal: torch.Tensor,
    alpha,
    beta,
    rr: float,
    d_hat: float,
    *,
    material=None,
    lam_soft=None,
    lam_hard=None,
    k_sharp: float = 1.25,
    d_hat_m: float = 12.0,
    body_offsets=None,
    body_rr: float = 0.0,
    body_gain: float = 1.0,
    terrain=None,
    terrain_lam_soft: float = 0.0,
    terrain_lam_hard: float = 0.0,
    terrain_k_sharp: float = 1.25,
    terrain_d_hat_m: float = 12.0,
    hard_scale=None,
) -> PotentialTerms:
    """U at normalized ``o`` for the deployed drive's conservative force (libcvc
    ``cvc::nav::drive_potential``; ``sdf_nav.bicycle_rollout`` with ``_material_force``).

    - The barrier is ``alpha b(phi - rr)``, or with a footprint (``body_offsets`` along the
      heading ``th``) ``alpha * body_gain * sum b(phi_k - body_rr)``.
    - ``material`` is a sampler with ``sample(on) -> (risk, phi_m, grad_r, grad_phi)`` and the
      world-to-normalized scale ``S`` (:class:`grl_snam.material.MaterialField`). Its hard force
      reads ``grad phi`` per metre while positions are normalized, so in the drive frame the hard
      term is ``lam_hard * S * b_sp(phi_m)``.
    - ``terrain`` is the same on the terrain's plane, its hard weight times ``hard_scale`` (the
      plant arm's tipping modulator) when given.
    """
    goal_t = 0.5 * _as(beta, o) * ((o - goal) ** 2).sum(-1)
    if body_offsets is None:
        phi, _ = field.sample(o)
        barrier = _as(alpha, o) * ipc_potential(phi - rr, d_hat)
    else:
        head = torch.stack([torch.cos(th), torch.sin(th)], -1)
        barrier = torch.zeros_like(goal_t)
        for off in body_offsets:
            phi, _ = field.sample(o + float(off) * head)
            barrier = barrier + ipc_potential(phi - body_rr, d_hat)
        barrier = _as(alpha, o) * body_gain * barrier
    z = torch.zeros_like(goal_t)
    soft, hard, ter = z, z.clone(), z.clone()
    if material is not None:
        risk, phi_m, _, _ = material.sample(o)
        soft = _as(lam_soft, o) * risk
        hard = _as(lam_hard, o) * float(material.S) * sdf_barrier_potential(phi_m, k_sharp, d_hat_m)
    if terrain is not None and (terrain_lam_soft != 0.0 or terrain_lam_hard != 0.0):
        risk, phi_m, _, _ = terrain.sample(o)
        lh = o.new_tensor(terrain_lam_hard)
        if hard_scale is not None:
            lh = lh * _as(hard_scale, o)
        ter = terrain_lam_soft * risk + lh * float(terrain.S) * sdf_barrier_potential(
            phi_m, terrain_k_sharp, terrain_d_hat_m
        )
    return PotentialTerms(goal=goal_t, barrier=barrier, soft=soft, hard=hard, terrain=ter)
