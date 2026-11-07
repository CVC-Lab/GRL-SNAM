"""What the fleet and live ports take from libcvc's multi-car template, ``pycvc_gl.cars``.

The template (libcvc's docs/CARS.md) gives every navigator, vehicle and plant setting a key in the
state tree, a choice of world, panels around a chosen car and a headless log. The fleet and live
ports reuse those pieces rather than copying them:

- **Settings.** libcvc's own keys, ``nav.*`` and ``nav.vehicle.*`` (metres; ``sim_world::pull``)
  and ``nav_plant.*`` (``plant_host``; ``nav_plant.vehicle`` picks the plant). A port on libcvc's
  sim_world reads them as libcvc does. A port on GRL-SNAM's torch navigator
  (:class:`~grl_snam.nav.SdfNavigator`, normalized units) maps them with :class:`TorchSettings`
  where they mean the same thing (:data:`TORCH_KEYS`).
- **World, panels, log.** ``pycvc_gl.cars``'s ``make_world`` / ``show_world``, its panels
  (``cars_panels.ari``, imported by ``cvc://`` with :func:`component_paths` on the host),
  ``state_from`` / ``raster_arrays`` / ``raster_pictures`` / ``plot_coefficients``, ``CarsLog``,
  ``save_log`` and ``exit_code``.

They need pycvc-gl 3.7.12+. :data:`HAS_CARS` says whether the installed pycvc_gl has them; without
them the ports run as they did, and an option that needs them says what to install
(:func:`require`).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# On Windows torch brings Intel's OpenMP runtime (libiomp5md.dll) and libcvc LLVM's (libomp.dll);
# the second to start (libcvc's terrain generator, after the torch squad has run) refuses to with
# "OMP: Error #15" unless told the other is there. They are separate DLLs, each with its own
# threads, so letting both run is safe there. A value you set yourself is kept.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:  # pragma: no cover - depends on the installed bindings
    from pycvc_gl import cars as C

    HAS_CARS = bool(C.has_cars()) and all(
        hasattr(C, k) for k in ("COMPONENTS", "state_from", "apply_settings", "raster_pictures")
    )
except Exception:  # noqa: BLE001 - an older pycvc_gl, or none
    C = None
    HAS_CARS = False

INSTALL = (
    "pycvc-gl 3.7.12+, whose pycvc_gl.cars has the multi-car template's pieces (cvcpkg install "
    "pycvc-gl-cp312, or your Python's column)"
)


def require(what: str):
    """``pycvc_gl.cars``, or a RuntimeError saying ``what`` needs it and what to install."""
    if not HAS_CARS:
        raise RuntimeError(f"{what} needs {INSTALL}")
    return C


def component_paths() -> list[str]:
    """The component paths a port's Host takes, so its document imports cvc://cars_panels.ari."""
    return [C.COMPONENTS] if HAS_CARS else []


# ── libcvc's keys on GRL-SNAM's torch navigator ───────────────────────────────────────────────

#: libcvc key -> (attribute on an SdfNavigator holding it ("" = the navigator itself), the field,
#: how it converts): "m" a length or speed (metres in the tree, x the navigator's scale on it),
#: "rad" an angle (the same on both), "flag" a 0/1.
TORCH_KEYS = {
    "nav.vehicle.vmax": ("kw", "vmax", "m"),
    "nav.vehicle.rr": ("kw", "rr", "m"),
    "nav.vehicle.d_hat": ("kw", "d_hat", "m"),
    "nav.reach_tol": ("", "reach_tol", "m"),
    "nav.vehicle.a_max": ("_veh", "a_max", "m"),
    "nav.vehicle.a_lat_max": ("_veh", "a_lat_max", "m"),
    "nav.vehicle.wheelbase": ("_veh", "L", "m"),
    "nav.vehicle.delta_max": ("_veh", "delta_max", "rad"),
    "nav.vehicle.k_steer": ("_veh", "k_steer", "rad"),
    "nav.vehicle.allow_reverse": ("_veh", "allow_reverse", "flag"),
}
#: The keys a point navigator takes (``dynamics="point"``, as austin-freedrive drives): its top
#: speed, its footprint and barrier, its arrival radius. The rest are the bicycle's.
POINT_KEYS = ("nav.vehicle.vmax", "nav.vehicle.rr", "nav.vehicle.d_hat", "nav.reach_tol")


def torch_keys(nav) -> tuple[str, ...]:
    """The libcvc keys ``nav`` (an SdfNavigator) takes: :data:`POINT_KEYS` for a point navigator,
    all of :data:`TORCH_KEYS` for a bicycle."""
    return POINT_KEYS if getattr(nav, "_dyn", "point") == "point" else tuple(TORCH_KEYS)


def torch_value(nav, key) -> float:
    """``key``'s value on ``nav``, in the tree's units (metres, radians, 0/1)."""
    where, field, unit = TORCH_KEYS[key]
    v = getattr(nav, field) if not where else getattr(nav, where)[field]
    if unit == "flag":
        return float(bool(v))
    return float(v) / float(nav.S) if unit == "m" else float(v)


def set_torch_value(nav, key, value) -> None:
    """Put ``value`` (the tree's units) on ``nav`` for ``key``."""
    where, field, unit = TORCH_KEYS[key]
    if unit == "flag":
        v = bool(float(value))
    else:
        v = float(value) * float(nav.S) if unit == "m" else float(value)
    if not where:
        setattr(nav, field, v)
    else:
        getattr(nav, where)[field] = v
    if key == "nav.vehicle.rr":
        nav.rr = v  # SdfNavigator keeps its footprint radius twice


def _fmt(v: float) -> str:
    return f"{v:.6g}"


class TorchSettings:
    """libcvc's navigator keys on GRL-SNAM's torch navigators.

    ``navs`` are :class:`~grl_snam.nav.SdfNavigator` s (a squad's, or a demo's one); ``keys`` the
    libcvc keys they take (default :func:`torch_keys` of the first). :meth:`seed` writes each key
    the tree does not have yet from the first navigator, in metres; :meth:`apply` (each tick)
    puts the tree's values on every navigator, in its normalized units, when they change. So a
    key set in the document's ``init:``, on a slider, with ``--set`` / ``--params`` or from
    Python reaches the torch navigator as it reaches libcvc's sim_world."""

    def __init__(self, host, navs, keys=None):
        self.host = host
        self.navs = list(navs)
        self.keys = tuple(keys if keys is not None else torch_keys(self.navs[0]))
        self._last = {}

    def seed(self) -> None:
        for key in self.keys:
            if self.host.get(key) in (None, ""):
                self.host.set(key, _fmt(torch_value(self.navs[0], key)))
        self.apply()

    def apply(self) -> None:
        for key in self.keys:
            v = self.host.get(key)
            if v in (None, "") or self._last.get(key) == v:
                continue
            try:
                x = float(v)
            except ValueError:
                continue
            for nav in self.navs:
                set_torch_value(nav, key, x)
            self._last[key] = v


# ── the world as a demo's City ────────────────────────────────────────────────────────────────


def city_of(world):
    """A :class:`~grl_snam.demos._common.City` over a ``pycvc_gl.cars`` World: its terrain
    height, its bounds, and occupancy from its geometry (rasterized as the demos rasterize the
    document's city), or from its grid for a world with none (the flat square)."""
    from grl_snam.demos._common import City, world_occupancy

    if world.geometry is not None:
        geometry = world.geometry

        def occupancy(bounds, nx, ny, inflate_m=0.0):
            return world_occupancy(geometry, bounds, nx, ny, inflate_m=inflate_m)

    else:

        def occupancy(bounds, nx, ny, inflate_m=0.0):
            mnx, mny, mxx, mxy = bounds
            X, Y = np.meshgrid(np.linspace(mnx, mxx, nx), np.linspace(mny, mxy, ny))
            occ = world.occupied(X, Y)
            if inflate_m > 0.0 and occ.any():
                import pycvc

                cell = (mxx - mnx) / max(nx - 1, 1)
                d2 = np.asarray(pycvc.nav_edt2_squared(occ.astype(np.uint8)), float)
                occ = d2 * cell * cell <= inflate_m * inflate_m
            return occ

    return City(world.z, world.bounds, occupancy)


def humvee_sizes():
    """The Humvee preset's length, width and wheelbase (m), as the template's cars have them."""
    return C._BICYCLE["length"], C._BICYCLE["width"], C._BICYCLE["wheelbase"]
