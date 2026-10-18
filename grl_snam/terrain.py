"""Terrain in the navigator's energy: the Python twin of libcvc's ``cvc/nav/terrain.h``.

libcvc's P5 (docs/roadmap/NAV-VEHICLE-PHYSICS-ROADMAP.md, doc 31 D9 H1) turns a height raster
and a vehicle's limits into three planes:

- **the slope**, dz/dx and dz/dy (m/m): gravity along a heading;
- **the grade cost** r~_g in [0, 1]: sin(slope) over the sine of the steepest feasible slope,
  so 1 is the vehicle's limit;
- **the infeasible cells**: steeper than the vehicle climbs (``max_grade``), or, with a tip
  ratio, than it can cross without rolling (``tip_margin`` x the static stability factor).

Through the material pipeline the grade cost is a soft force, ``F = -lam_grade grad r~_g``,
leaning the drive toward gentler ground, and the infeasible cells are its hard barrier.
:func:`terrain_material` builds that :class:`~grl_snam.material.MaterialGrid`, as
``sim_world::set_terrain`` builds it in C++.

:func:`terrain_planes` is bit-identical to ``cvc::nav::terrain_build``: the slope is
``np.gradient`` of the float32 heights per metre, and the grade cost is computed in double and
stored as float32 (``tests/test_terrain.py`` checks it against ``pycvc.nav_terrain_build``).
Everything is per unit mass, so trained coefficients keep their meaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TerrainPlanes:
    """The planes of :func:`terrain_planes` (row 0 at min y, like every grid here)."""

    slope_x: np.ndarray  # (rows, cols) float32, dz/dx (m/m)
    slope_y: np.ndarray  # (rows, cols) float32, dz/dy (m/m)
    grade_cost: np.ndarray  # (rows, cols) float32, [0, 1]
    infeasible: np.ndarray  # (rows, cols) bool
    limit: float  # tan of the steepest feasible slope


def terrain_limit(max_grade: float = 0.6, tip_ratio: float = 0.0, tip_margin: float = 0.8) -> float:
    """The steepest feasible slope's tangent: ``max_grade``, or ``tip_margin x tip_ratio`` when
    a tip ratio (the static stability factor, track / (2 x CG height)) is lower."""
    lim = float(max_grade)
    if tip_ratio > 0.0:
        lim = min(lim, float(tip_margin) * float(tip_ratio))
    return lim


def terrain_planes(
    height_m: np.ndarray,
    cell_w: float,
    *,
    max_grade: float = 0.6,
    tip_ratio: float = 0.0,
    tip_margin: float = 0.8,
) -> TerrainPlanes:
    """The slope, grade cost and infeasible cells of ``height_m`` (rows, cols metres, row 0 at
    min y) on cells ``cell_w`` metres square, for a vehicle that climbs ``max_grade`` and, with
    a ``tip_ratio``, rolls past ``tip_margin x tip_ratio``. Bit-identical to libcvc's
    ``terrain_build``."""
    if not (cell_w > 0.0):
        raise ValueError("terrain_planes: cell_w must be > 0")
    if not (max_grade > 0.0) or tip_ratio < 0.0 or not (tip_margin > 0.0):
        raise ValueError("terrain_planes: max_grade and tip_margin must be > 0, tip_ratio >= 0")
    h = np.asarray(height_m, np.float32)
    if h.ndim != 2:
        raise ValueError("terrain_planes: heights are a (rows, cols) grid")
    gy, gx = np.gradient(h)  # f32 in, f32 out: the C++ grad1d
    inv_w = np.float32(1.0 / float(cell_w))
    gx = gx * inv_w
    gy = gy * inv_w
    lim = terrain_limit(max_grade, tip_ratio, tip_margin)
    sin_lim = lim / math.sqrt(1.0 + lim * lim)
    t = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2)
    s = t / np.sqrt(1.0 + t * t)
    cost = np.minimum(1.0, s / sin_lim).astype(np.float32)
    return TerrainPlanes(gx, gy, cost, t > lim, lim)


def terrain_material(
    height_m: np.ndarray,
    bounds,
    *,
    center=(0.0, 0.0),
    scale: float = 1.0,
    sigma: float = 1.0,
    params=None,
    max_grade: float = 0.6,
    tip_ratio: float = 0.0,
    tip_margin: float = 0.8,
):
    """The grade cost and the infeasible cells as a :class:`~grl_snam.material.MaterialGrid`
    over ``bounds`` (min_x, min_y, max_x, max_y; cell centres at the grid points): its
    ``field()`` gives the soft grade force and the hard barrier to the material force law, as
    ``sim_world::set_terrain`` builds them. ``center`` and ``scale`` are the navigator's
    normalized frame."""
    from .material import MaterialGrid

    h = np.asarray(height_m, np.float32)
    rows, cols = h.shape
    cell_w = (float(bounds[2]) - float(bounds[0])) / (cols - 1)
    p = terrain_planes(h, cell_w, max_grade=max_grade, tip_ratio=tip_ratio, tip_margin=tip_margin)
    return (
        MaterialGrid(p.grade_cost, p.infeasible, bounds, center, scale, sigma=sigma, params=params),
        p,
    )


def climb_weight(c_rr: float, cell_w: float) -> float:
    """Grid steps per metre of rise that make a metre of climb cost what rolling ``1 / c_rr``
    metres on the flat does: the energy at the wheels (libcvc's ``climb_weight``, H5)."""
    return 1.0 / (float(c_rr) * float(cell_w))
