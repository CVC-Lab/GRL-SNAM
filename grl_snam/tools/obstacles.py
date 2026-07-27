"""Geometry bundle -> circular-obstacle set for the CoefEnergyNet (circle-surrogate)
training track. See :func:`extract`. (The SDF track uses :mod:`grl_snam.tools.sdf`
instead.)"""

from __future__ import annotations

import os

import numpy as np


def extract(
    bundle_dir: str,
    grid: int = 512,
    block: int = 2,
    radius_frac: float = 0.6,
    robot_radius_world: float = 3.0,
) -> dict:
    """Bundle -> (obstacle centers, radius, free-point pool), all in WORLD units.

    The city footprint (``buildings.glb``) is rasterized over the ``terrain.json``
    bounds to a ``grid x grid`` solid occupancy mask, coarsened by ``block`` (each
    obstacle stands in for a ``block x block`` wall patch), and every occupied coarse
    cell becomes one circular obstacle at the cell centre with radius
    ``radius_frac * cell_size``. Free cells become the drivable pool. Needs the
    ``pycvc_gl`` scene helpers (VTK rasterization); imported lazily."""
    from pycvc_gl.scenes import building_occupancy, terrain_grid

    terrain = os.path.join(bundle_dir, "terrain.json")
    glb = os.path.join(bundle_dir, "buildings.glb")
    _, bounds, _, _ = terrain_grid(terrain)
    mnx, mny, mxx, mxy = bounds

    occ = building_occupancy(glb, bounds, grid, grid, inflate_m=0.0)
    ny, nx = occ.shape

    cny, cnx = ny // block, nx // block
    cocc = occ[: cny * block, : cnx * block].reshape(cny, block, cnx, block).any(axis=(1, 3))
    csx = (mxx - mnx) / cnx
    csy = (mxy - mny) / cny
    ys, xs = np.where(cocc)
    centers = np.stack([mnx + (xs + 0.5) * csx, mny + (ys + 0.5) * csy], 1).astype(np.float32)
    radius_world = float(radius_frac * 0.5 * (csx + csy))

    gx = np.linspace(mnx, mxx, nx, dtype=np.float32)
    gy = np.linspace(mny, mxy, ny, dtype=np.float32)
    free_r, free_c = np.where(~occ)
    free_pool = np.stack([gx[free_c], gy[free_r]], 1).astype(np.float32)

    return {
        "centers": centers,
        "radius_world": np.float32(radius_world),
        "robot_radius_world": np.float32(robot_radius_world),
        "bounds": np.asarray(bounds, np.float32),
        "free_pool": free_pool,
        "cell_size": np.float32(0.5 * (csx + csy)),
    }


def extract_to_npz(bundle_dir: str, out: str | None = None, grid: int = 512, block: int = 2) -> str:
    """Run :func:`extract` and save an ``obstacles.npz`` (default ``<bundle>/obstacles.npz``)."""
    data = extract(bundle_dir, grid=grid, block=block)
    out = out or os.path.join(bundle_dir, "obstacles.npz")
    np.savez_compressed(out, **data)
    print(
        "extracted %d obstacles (r=%.1fm) + %d free points -> %s"
        % (len(data["centers"]), float(data["radius_world"]), len(data["free_pool"]), out)
    )
    return out
