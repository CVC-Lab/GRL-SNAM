"""Ingest a real-world geometry bundle into an obstacle set for navigation training.

Turns a ``geometry_bundle`` (a ``terrain.json`` heightfield + a ``buildings.glb``
city mesh) into a compact ``obstacles.npz`` the trainer consumes: the building
footprints become a set of CIRCULAR obstacles (the shape the GRL-SNAM surrogate
repels from), plus a pool of free (drivable) world points to sample start/goal
positions from. Run this ONCE per scene; the trainer then reads the ``.npz`` with
no graphics dependency, and a pre-generated obstacle set can be shipped as-is.

This step needs the ``pycvc_gl`` scene helpers (they rasterize the city mesh with
VTK to a solid occupancy grid), so run it in an environment where volrover3's
Python / cvcGL bindings are importable. The rasterized occupancy is cached next to
the ``.glb`` by ``pycvc_gl.scenes.building_occupancy``, so re-runs are instant.

Usage:
    python scripts/extract_obstacles.py <bundle_dir> [-o obstacles.npz] [--grid 512]

``<bundle_dir>`` holds ``terrain.json`` and ``buildings.glb`` (e.g. a scene export).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from pycvc_gl.scenes import building_occupancy, terrain_grid


def extract(bundle_dir: str, grid: int = 512, block: int = 4, radius_frac: float = 0.6,
            robot_radius_world: float = 3.0):
    """Bundle -> (obstacle centers, radius, free-point pool), all in WORLD units.

    The city footprint is rasterized to a ``grid x grid`` solid occupancy mask,
    coarsened by ``block`` (so each obstacle stands in for a ``block x block``
    patch of wall), and every occupied coarse cell becomes one circular obstacle
    at the cell center with radius ``radius_frac * cell_size``. Free cells become
    the drivable pool. Nothing here is scaled to the surrogate's ~10-unit regime
    yet — the trainer normalizes per working-region so one bundle can be trained
    at several zooms.
    """
    terrain = os.path.join(bundle_dir, "terrain.json")
    glb = os.path.join(bundle_dir, "buildings.glb")
    _, bounds, _, _ = terrain_grid(terrain)  # (min_x, min_y, max_x, max_y)
    mnx, mny, mxx, mxy = bounds

    # Solid top-down occupancy (True = inside a building). inflate_m=0: raw
    # footprints; the trainer's clearance margin keeps the robot off the walls.
    occ = building_occupancy(glb, bounds, grid, grid, inflate_m=0.0)
    ny, nx = occ.shape

    cny, cnx = ny // block, nx // block
    cocc = occ[: cny * block, : cnx * block].reshape(cny, block, cnx, block).any(axis=(1, 3))
    csx = (mxx - mnx) / cnx  # coarse cell size (world), x
    csy = (mxy - mny) / cny  # ... y
    ys, xs = np.where(cocc)
    centers = np.stack([mnx + (xs + 0.5) * csx, mny + (ys + 0.5) * csy], 1).astype(np.float32)
    radius_world = float(radius_frac * 0.5 * (csx + csy))

    # Free-point pool over the SAME frame the terrain sampler uses (row 0 = min_y).
    gx = np.linspace(mnx, mxx, nx, dtype=np.float32)
    gy = np.linspace(mny, mxy, ny, dtype=np.float32)
    freeR, freeC = np.where(~occ)
    free_pool = np.stack([gx[freeC], gy[freeR]], 1).astype(np.float32)

    return {
        "centers": centers,                       # [M,2] obstacle centers (world)
        "radius_world": np.float32(radius_world), # scalar obstacle radius (world)
        "robot_radius_world": np.float32(robot_radius_world),
        "bounds": np.asarray(bounds, np.float32), # (min_x,min_y,max_x,max_y)
        "free_pool": free_pool,                   # [K,2] drivable points (world)
        "cell_size": np.float32(0.5 * (csx + csy)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle_dir", help="dir with terrain.json + buildings.glb")
    ap.add_argument("-o", "--out", default=None, help="output .npz (default <bundle>/obstacles.npz)")
    ap.add_argument("--grid", type=int, default=512, help="occupancy raster resolution")
    ap.add_argument("--block", type=int, default=2,
                    help="coarsen factor (obstacle granularity). Finer (2) gives ~7 m circles that "
                         "leave streets navigable; coarser (4) gives ~14 m circles that crowd them.")
    args = ap.parse_args()

    data = extract(args.bundle_dir, grid=args.grid, block=args.block)
    out = args.out or os.path.join(args.bundle_dir, "obstacles.npz")
    np.savez_compressed(out, **data)
    print(
        "extracted %d obstacles (r=%.1fm) + %d free points over %s -> %s"
        % (len(data["centers"]), float(data["radius_world"]),
           len(data["free_pool"]), tuple(data["bounds"].tolist()), out)
    )


if __name__ == "__main__":
    main()
