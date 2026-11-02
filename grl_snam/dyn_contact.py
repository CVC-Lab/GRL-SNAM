"""Compliant contact for the dynamic single-track plant, differentiable in torch.

The torch twin of libcvc's ``cvc::nav::diff`` contact (``inc/cvc/nav/detail/dyn_contact.h``): the
vehicle's outline sampled at points about its centre of gravity, each point's contact against a
field (the scene's signed distance) and between two vehicles, in port-Hamiltonian form. In SI.

The footprint is a rectangle about the centre of gravity, ``front`` ahead, ``rear`` behind and
``half_width`` to each side, sampled at ``per_side`` points along each long side and ``per_end``
along each end, the corners shared (:func:`dyn_footprint_points`).

Against a field. A point k at world position ``P_k = c + R(psi) l_k`` with ``d_k = phi(P_k)``
within the skin ``s`` of an obstacle stores the barrier energy ``1/2 k (s - d_k)^2``, damps its
normal velocity with Hunt-Crossley's ``c (s - d_k) / s`` (no damping at first touch, so a point
leaving is never held back), and slides along the obstacle, ``t = (-dphi/dy, dphi/dx)``, with
Coulomb friction on the barrier's push, its direction smoothed over ``slide``. Its world force is

    F_k = k (s - d_k) grad phi - c_k (grad phi . v_Pk) grad phi - f_k (t . v_Pk) t,

and its body wrench (body axes, at the centre of gravity) ``(R^T F_k, l_k x R^T F_k)``. The
barrier's part ``Wc`` is minus the energy's gradient in the pose: it lives in H, through
T*SE(2)'s canonical structure. The damper's and the friction's, ``Wd``, are ``-R_c v`` with
``R_c`` positive semi-definite, their power ``-damping``. The plant takes ``Wc + Wd`` as its
external port, held over the step (:func:`grl_snam.dyn_bike.dyn_plant`'s ``wrench``), and its
books book the port's energy (:func:`grl_snam.dyn_bike.dyn_plant_books`' ``port``).

Between vehicles (:func:`dyn_contact_pair`). Each vehicle's points meet the other's outline as a
rounded rectangle's signed distance (:func:`dyn_rounded_box_sdf`), both ways. Each point-outline
force acts on both bodies, equal and opposite at the point, so the pair keeps its momentum; the
barrier's energy and the damper's power are shared, half to each.

A body is a ``(..., 6)`` tensor in :data:`BODY_ORDER`: its centre of gravity's world position, its
heading, and its twist there (body axes). A field is a callable ``field(x, y) -> (phi, gx, gy)``
on tensors of world positions: the signed distance (> 0 free) and its gradient. A point beyond
the skin adds nothing, as libcvc skips it; every other branch is a ``torch.where`` on values.

libcvc's contact is double and is not differentiated there; here autograd differentiates it, so
the barrier can be held to minus its energy's gradient natively. The parity gate is
``tests/test_dyn_contact_parity.py``, against libcvc's ``dyn_contact_golden.json``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

BODY_ORDER = ("x", "y", "psi", "vx", "vy", "r")

Field = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class DynContactParams:
    """``cvc::nav::diff::dyn_contact_params``, in SI."""

    k: float = 3.0e5  # N/m: each point's barrier stiffness
    c: float = 1.5e4  # N s/m: each point's damping, at the full skin's depth
    friction: float = 0.2  # the outline's friction along what it meets
    slide: float = 0.05  # m/s: the friction's smoothing of its sliding direction
    skin: float = 0.05  # m: the barrier's reach outside the outline
    per_side: int = 7  # points along each long side, the corners included
    per_end: int = 4  # along each end, the corners included
    rounding: float = 0.3  # m: the outline's corner radius, as another vehicle meets it


@dataclass(frozen=True)
class DynFootprint:
    """``dyn_footprint``: the outline about the centre of gravity (m)."""

    front: float = 2.0
    rear: float = 2.0
    half_width: float = 1.0


@dataclass(frozen=True)
class DynContact:
    """``dyn_contact``: the contact on one body."""

    Wc: torch.Tensor  # (..., 3): the barrier's wrench (F_x, F_y, M_z), body axes, at the cg
    Wd: torch.Tensor  # (..., 3): the damper's and the friction's
    energy: torch.Tensor  # the barrier's energy (J)
    damping: torch.Tensor  # the damper's and the friction's dissipation (W, >= 0) at the twist
    touching: torch.Tensor  # how many points are within the skin

    def wrench(self) -> torch.Tensor:
        """The total wrench ``(..., 3)``, the plant's port input."""
        return self.Wc + self.Wd


@dataclass(frozen=True)
class WallField:
    """A wall whose west face is at x = x0, free to its west (libcvc's test ``wall_field``)."""

    x0: float = 0.0

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return self.x0 - x, torch.full_like(x, -1.0), torch.zeros_like(x)


@dataclass(frozen=True)
class PillarField:
    """A round pillar of radius ``rad`` at (cx, cy) (libcvc's test ``pillar_field``)."""

    cx: float = 0.0
    cy: float = 0.0
    rad: float = 1.0

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        dx, dy = x - self.cx, y - self.cy
        d = torch.sqrt(dx * dx + dy * dy)
        return d - self.rad, dx / d, dy / d


def dyn_footprint_points(f: DynFootprint, p: DynContactParams) -> list[tuple[float, float]]:
    """The footprint's points l_k, body axes, from the centre of gravity."""
    pts = []
    ns, ne = max(2, p.per_side), max(2, p.per_end)
    for i in range(ns):
        x = -f.rear + (f.front + f.rear) * i / (ns - 1)
        pts.append((x, f.half_width))
        pts.append((x, -f.half_width))
    for j in range(1, ne - 1):
        y = -f.half_width + 2.0 * f.half_width * j / (ne - 1)
        pts.append((f.front, y))
        pts.append((-f.rear, y))
    return pts


def dyn_contact_add(body: torch.Tensor, rx, ry, fx, fy, W: torch.Tensor) -> torch.Tensor:
    """``W`` ``(..., 3)`` with a world force (fx, fy) at the world offset (rx, ry) from the body's
    centre of gravity added as a body wrench."""
    c, s = torch.cos(body[..., 2]), torch.sin(body[..., 2])
    return W + torch.stack((c * fx + s * fy, -s * fx + c * fy, rx * fy - ry * fx), dim=-1)


def dyn_contact_velocity(body: torch.Tensor, rx, ry):
    """A body point's world velocity, at world offset (rx, ry) from its centre of gravity."""
    psi, vx, vy, r = body[..., 2], body[..., 3], body[..., 4], body[..., 5]
    c, s = torch.cos(psi), torch.sin(psi)
    return c * vx - s * vy - r * ry, s * vx + c * vy + r * rx


class _Sum:
    """A :class:`DynContact` being summed, point by point in libcvc's order."""

    def __init__(self, like: torch.Tensor):
        zero = torch.zeros_like(like)
        self.Wc = self.Wd = torch.stack((zero, zero, zero), dim=-1)
        self.energy = self.damping = zero
        self.touching = torch.zeros_like(like, dtype=torch.int64)

    def contact(self) -> DynContact:
        return DynContact(
            Wc=self.Wc, Wd=self.Wd, energy=self.energy, damping=self.damping, touching=self.touching
        )


def dyn_contact_field(
    p: DynContactParams, pts: list[tuple[float, float]], field: Field, body: torch.Tensor
) -> DynContact:
    """The contact of a body's points against a field."""
    x, y, psi = body[..., 0], body[..., 1], body[..., 2]
    c, s = torch.cos(psi), torch.sin(psi)
    out = _Sum(x)
    for lx, ly in pts:
        rx = c * lx - s * ly
        ry = s * lx + c * ly
        d, gx, gy = field(x + rx, y + ry)
        touch = d < p.skin
        out.touching = out.touching + touch.long()
        pen = torch.where(touch, p.skin - d, torch.zeros_like(d))
        fn = p.k * pen
        cd = p.c * pen / p.skin
        vpx, vpy = dyn_contact_velocity(body, rx, ry)
        vn = gx * vpx + gy * vpy
        # Friction along the obstacle: Coulomb on the barrier's push, its sliding direction
        # smoothed over `slide` (a secant coefficient, so a dissipation in R too).
        tx, ty = -gy, gx
        vt = tx * vpx + ty * vpy
        ft = p.friction * fn / torch.sqrt(vt * vt + p.slide * p.slide)
        out.energy = out.energy + 0.5 * p.k * pen * pen
        out.damping = out.damping + (cd * vn * vn + ft * vt * vt)
        out.Wc = dyn_contact_add(body, rx, ry, fn * gx, fn * gy, out.Wc)
        out.Wd = dyn_contact_add(
            body, rx, ry, -cd * vn * gx - ft * vt * tx, -cd * vn * gy - ft * vt * ty, out.Wd
        )
    return out.contact()


def dyn_rounded_box_sdf(f: DynFootprint, rho: float, lx: torch.Tensor, ly: torch.Tensor):
    """A rounded rectangle's signed distance at (lx, ly) in its body's axes (the footprint's
    outline with corners of radius rho, at most the half width), and its gradient there (body
    axes): (d, gx, gy)."""
    cx0 = 0.5 * (f.front - f.rear)
    a = 0.5 * (f.front + f.rear) - rho
    bb = f.half_width - rho
    qx0, qy0 = lx - cx0, ly
    one, zero = torch.ones_like(qx0), torch.zeros_like(qx0)
    sx = torch.where(qx0 >= 0.0, one, -one)
    sy = torch.where(qy0 >= 0.0, one, -one)
    qx = torch.abs(qx0) - a
    qy = torch.abs(qy0) - bb
    corner = (qx > 0.0) & (qy > 0.0)
    along_x = qx > qy
    length = torch.sqrt(torch.where(corner, qx * qx + qy * qy, one))
    d = torch.where(corner, length - rho, torch.where(along_x, qx - rho, qy - rho))
    gx = torch.where(corner, sx * qx / length, torch.where(along_x, sx, zero))
    gy = torch.where(corner, sy * qy / length, torch.where(along_x, zero, sy))
    return d, gx, gy


def _one_way(
    p: DynContactParams,
    ptsX,
    X: torch.Tensor,
    fY: DynFootprint,
    Y: torch.Tensor,
    cx: _Sum,
    cy: _Sum,
) -> None:
    """``dyn_contact_one_way``: X's points against Y's rounded outline. Each point's force acts
    on X at the point and its opposite on Y; the energy and the dissipation go half to each."""
    c, s = torch.cos(X[..., 2]), torch.sin(X[..., 2])
    cY, sY = torch.cos(Y[..., 2]), torch.sin(Y[..., 2])
    for lx, ly in ptsX:
        rx = c * lx - s * ly
        ry = s * lx + c * ly
        px, py = X[..., 0] + rx, X[..., 1] + ry
        # The point in Y's axes, and Y's outline's distance and gradient there (world).
        ox, oy = px - Y[..., 0], py - Y[..., 1]
        d, glx, gly = dyn_rounded_box_sdf(fY, p.rounding, cY * ox + sY * oy, -sY * ox + cY * oy)
        touch = d < p.skin
        gx = cY * glx - sY * gly
        gy = sY * glx + cY * gly
        cx.touching = cx.touching + touch.long()
        cy.touching = cy.touching + touch.long()
        pen = torch.where(touch, p.skin - d, torch.zeros_like(d))
        fn = p.k * pen
        cd = p.c * pen / p.skin
        vxx, vxy = dyn_contact_velocity(X, rx, ry)
        vyx, vyy = dyn_contact_velocity(Y, ox, oy)
        vn = gx * (vxx - vyx) + gy * (vxy - vyy)
        tx, ty = -gy, gx
        vt = tx * (vxx - vyx) + ty * (vxy - vyy)
        ft = p.friction * fn / torch.sqrt(vt * vt + p.slide * p.slide)
        e = 0.5 * p.k * pen * pen
        w = cd * vn * vn + ft * vt * vt
        cx.energy = cx.energy + 0.5 * e
        cy.energy = cy.energy + 0.5 * e
        cx.damping = cx.damping + 0.5 * w
        cy.damping = cy.damping + 0.5 * w
        dx = -cd * vn * gx - ft * vt * tx
        dy = -cd * vn * gy - ft * vt * ty
        cx.Wc = dyn_contact_add(X, rx, ry, fn * gx, fn * gy, cx.Wc)
        cy.Wc = dyn_contact_add(Y, ox, oy, -fn * gx, -fn * gy, cy.Wc)
        cx.Wd = dyn_contact_add(X, rx, ry, dx, dy, cx.Wd)
        cy.Wd = dyn_contact_add(Y, ox, oy, -dx, -dy, cy.Wd)


def dyn_contact_pair(
    p: DynContactParams,
    pa: list[tuple[float, float]],
    fa: DynFootprint,
    a: torch.Tensor,
    pb: list[tuple[float, float]],
    fb: DynFootprint,
    b: torch.Tensor,
) -> tuple[DynContact, DynContact]:
    """The contact between two vehicles, both ways: (on a, on b)."""
    ca, cb = _Sum(a[..., 0]), _Sum(b[..., 0])
    _one_way(p, pa, a, fb, b, ca, cb)
    _one_way(p, pb, b, fa, a, cb, ca)
    return ca.contact(), cb.contact()
