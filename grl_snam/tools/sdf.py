"""Geometry bundle -> navigation SDF (``nav_sdf.npz``) — the primary obstacle model
for the learned-nav demo. Two interchangeable sources:

* ``edt`` (default) — grl-snam's own exact 2-D footprint distance transform.
* ``cvc`` — CVC's mesh-exact 3-D SDF via ``pycvc.sdf`` (SDF_V2), sliced to the ground
  plane; leans on the CVC compute layer and is the substrate for 3-D GRL-SNAM later.

Both emit the same normalized ``phi`` + unit-normal grids + meta that
:mod:`grl_snam.tools.train` and the navigator consume source-agnostically.
"""

from __future__ import annotations

import os

import numpy as np

import sdf_nav

TARGET_EXTENT = 10.0  # the surrogate's normalized working scale (~10 units across the region)


def _gltf_mesh(glb_path: str):
    """Flat ``(verts[x,y,z,...], tris[i,j,k,...])`` from a glTF/GLB via VTK (lazy import)."""
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  (register factories)
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
    from vtkmodules.vtkIOGeometry import vtkGLTFReader

    reader = vtkGLTFReader()
    reader.SetFileName(glb_path)
    reader.Update()
    geom = vtkCompositeDataGeometryFilter()
    geom.SetInputConnection(reader.GetOutputPort())
    geom.Update()
    pd = geom.GetOutput()
    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    polys = vtk_to_numpy(pd.GetPolys().GetData())  # [n0,i,j,k, n1,...]; triangulated -> n0==3
    tris: list[int] = []
    i = 0
    while i < len(polys):
        n = int(polys[i])
        if n == 3:
            tris += [int(polys[i + 1]), int(polys[i + 2]), int(polys[i + 3])]
        i += n + 1
    return pts.reshape(-1).tolist(), tris


def build(
    bundle_dir: str,
    source: str = "edt",
    region: float = 430.0,
    grid: int = 512,
    cvc_dim=(512, 512, 48),
    out: str | None = None,
) -> str:
    """Build the navigation SDF for a scene bundle and save ``nav_sdf.npz``. Returns the
    output path. Needs the ``pycvc_gl`` scene helpers (imported lazily)."""
    from pycvc_gl.scenes import building_occupancy, terrain_grid

    terrain = os.path.join(bundle_dir, "terrain.json")
    glb = os.path.join(bundle_dir, "buildings.glb")
    _, bounds, _, _ = terrain_grid(terrain)
    mnx, mny, mxx, mxy = bounds
    cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    scale = TARGET_EXTENT / (2.0 * region)

    if source == "edt":
        occ = building_occupancy(glb, bounds, grid, grid, inflate_m=0.0)
        phi, nxg, nyg = sdf_nav.build_sdf(occ, bounds, scale)
    elif source == "cvc":
        verts, tris = _gltf_mesh(glb)
        phi, nxg, nyg = sdf_nav.build_sdf_cvc(verts, tris, bounds, scale, dim=tuple(cvc_dim))
    else:
        raise ValueError(f"unknown SDF source {source!r} (expected 'edt' or 'cvc')")

    out = out or os.path.join(bundle_dir, "nav_sdf.npz")
    np.savez_compressed(
        out,
        phi=phi,
        normal_x=nxg,
        normal_y=nyg,
        bounds=np.asarray(bounds, np.float32),
        center=np.asarray([cx, cy], np.float32),
        scale=np.float32(scale),
        region=np.float32(region),
        source=source,
    )
    print(
        "built %s SDF %s (phi[%.2f,%.2f]) -> %s"
        % (source, phi.shape, float(phi.min()), float(phi.max()), out)
    )
    return out
