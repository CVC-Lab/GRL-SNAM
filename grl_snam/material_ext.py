"""An external force in the material rollouts.

The torch twin of libcvc's ``cvc::nav::material_ext`` (``inc/cvc/nav/material.h``): the
deployed drive takes an external force alongside the material one
(``drive_step_material_ext`` with an ``ext_force``, cvcdbg's RF/comm force), and the material
training rollouts take the same force, so the material heads learn in the presence of the force
they face at deploy.

It joins the rollouts as the deployed drive adds it:

- :func:`grl_snam.material_nav.integrate_surrogate_material` adds it to the force sum (a point
  mass: ``steer`` has nothing to act on);
- :func:`grl_snam.material_bike.integrate_bicycle_material` adds it last to
  ``F = F_bar + F_goal + F_mat``, and with ``steer`` to the steering bias
  ``(F_rep + F_mat + F_ext) . left``.

Positions are the batch's: the frame and units of ``o``, ``C`` and the rollout patch. Exactly one
source:

- ``patch``: ``K`` force patches per agent, (B,K,2,Hp,Wp) on the rollout patch's grid (centred on
  the start, sampled bilinearly at the current position as its channels are), with an optional
  schedule ``patch_of_step`` (B,steps) picking agent b's patch at step s. Autograd takes the
  patch's own gradient, as libcvc's reverse does.
- ``F``: per-step values (B,steps,2), and optionally ``J`` (B,steps,4), dF/do row-major. The
  forward uses F as given; J reaches the gradient only, through a zero-valued term
  ``J (o - o.detach())``, as libcvc's reverse chains it.
- ``sample``: a callable ``sample(s, o)`` at the sampled steps (every ``hold``), returning F
  (B,2), which autograd differentiates, or ``(F, J)`` for a force autograd cannot see through. The
  sample is held between, its gradient reaching the pose it was sampled at.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from grl_snam.material_nav import bilinear_sample_patch


@dataclass
class MaterialExt:
    """``cvc::nav::material_ext``: see the module docstring."""

    patch: torch.Tensor | None = None
    patch_of_step: torch.Tensor | None = None
    F: torch.Tensor | None = None
    J: torch.Tensor | None = None
    sample: Callable | None = None
    hold: int = 1
    steer: bool = True

    def __post_init__(self) -> None:
        sources = sum(x is not None for x in (self.patch, self.F, self.sample))
        if sources != 1:
            raise ValueError("MaterialExt needs exactly one source: patch, F or sample")
        if self.J is not None and self.F is None:
            raise ValueError("MaterialExt gives J without F")
        if self.patch_of_step is not None and self.patch is None:
            raise ValueError("MaterialExt gives patch_of_step without patch")
        if self.hold < 1:
            raise ValueError("MaterialExt needs hold >= 1")
        if self.patch is not None and self.patch.dim() != 5:
            raise ValueError("MaterialExt.patch must be (B,K,2,Hp,Wp)")


def _through_J(F: torch.Tensor, J: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """F in value, with J (B,4) as its gradient with respect to o."""
    d = o - o.detach()
    gx = J[:, 0] * d[:, 0] + J[:, 1] * d[:, 1]
    gy = J[:, 2] * d[:, 0] + J[:, 3] * d[:, 1]
    return F + torch.stack((gx, gy), dim=-1)


class ExtForce:
    """One rollout's external force: ``at(s, o)`` must visit the steps in order 0, 1, ...
    (the callback's held sample carries from one step to the next)."""

    def __init__(self, ext: MaterialExt, o0: torch.Tensor):
        self.ext = ext
        self.o0 = o0
        self.held: torch.Tensor | None = None

    def at(self, s: int, o: torch.Tensor) -> torch.Tensor:
        e = self.ext
        if e.patch is not None:
            B = o.shape[0]
            if e.patch_of_step is not None:
                k = e.patch_of_step[:, s].long()
            else:
                k = torch.zeros(B, dtype=torch.long, device=o.device)
            p = e.patch[torch.arange(B, device=o.device), k].to(o.dtype)
            return bilinear_sample_patch(p, o, self.o0)
        if e.F is not None:
            F = e.F[:, s].to(o.dtype)
            return _through_J(F, e.J[:, s].to(o.dtype), o) if e.J is not None else F
        if s % e.hold == 0:
            r = e.sample(s, o)
            if isinstance(r, tuple):
                F, J = r
                r = _through_J(F.to(o.dtype), J.to(o.dtype), o)
            self.held = r.to(o.dtype)
        return self.held
