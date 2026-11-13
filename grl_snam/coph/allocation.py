"""The Ackermann allocation: CoPH's planar force as a car's steer and acceleration.

CoPH commands a force ``u`` on a fully actuated point. A car takes only an acceleration along
its heading ``h`` and a steer that sets the lateral acceleration ``v^2 tan(delta) / L``. This
module is libcvc's ``NAV-COPH-ROADMAP.md`` §3.3.2, which the cars controller (T4) and libcvc's
fleet driver (C5) share; GRL-SNAM's golden pins it (case G11). With ``n`` the left normal:

    a      = clamp(u . h, -a_max, a_max)
    kappa* = (u . n) / max(v^2, v_floor^2)
    delta  = clamp(atan(L kappa*), -delta_max, delta_max)
    u_used = a h + (v^2 tan(delta) / L) n
    e      = u - u_used                      (the force the car did not take)

The steer is the bicycle angle (rad, + left) and the acceleration is along the heading, as
libcvc's ``sim_world::override_command`` takes them. ``v`` is the signed speed along ``h``, so a
car backing up has ``v < 0``; the lateral acceleration ``v^2 tan(delta) / L`` keeps its sign in
reverse. ``v_floor`` keeps ``kappa*`` finite at rest, where the car takes no lateral force at all
and ``e`` holds all of ``u . n``. §3.3.2 leaves ``v_floor`` open; ``SPEED_FLOOR`` is this
module's default.

GRL-SNAM code, not vendored from CoPH-Stagewise (``VENDORED.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SPEED_FLOOR = 0.5  # m/s


@dataclass(frozen=True)
class Allocation:
    accel: float  # along the heading
    steer: float  # the bicycle angle, + left
    kappa: float  # the curvature asked for, before the lock: (u . n) / max(v^2, v_floor^2)
    used: tuple[float, float]  # the force the car took
    unallocated: tuple[float, float]  # u - used


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def allocate(
    u,
    heading: float,
    speed: float,
    wheelbase: float,
    steer_max: float,
    accel_max: float,
    speed_floor: float = SPEED_FLOOR,
) -> Allocation:
    """Allocate the planar force ``u`` to a car heading ``heading`` (rad) at signed ``speed``."""
    hx, hy = math.cos(heading), math.sin(heading)
    nx, ny = -hy, hx
    along = u[0] * hx + u[1] * hy
    across = u[0] * nx + u[1] * ny
    accel = _clamp(along, -accel_max, accel_max)
    kappa = across / max(speed * speed, speed_floor * speed_floor)
    steer = _clamp(math.atan(wheelbase * kappa), -steer_max, steer_max)
    lateral = speed * speed * math.tan(steer) / wheelbase
    used = (accel * hx + lateral * nx, accel * hy + lateral * ny)
    return Allocation(accel, steer, kappa, used, (u[0] - used[0], u[1] - used[1]))
