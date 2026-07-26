"""Build a navigation SDF for a scene, from either source (configurable):

  --source edt  (default)  grl-snam's own exact footprint distance transform
                           (2-D, top-down, no extra deps beyond the occupancy)
  --source cvc             CVC's mesh-exact 3-D SDF via pycvc.sdf (SDF_V2), sliced
                           to the ground plane — leans on the CVC compute layer and
                           is the substrate for extending GRL-SNAM to 3-D later

Both emit the same ``<bundle>/nav_sdf.npz`` (normalized ``phi`` + unit ``normal_x/y``
grids + meta), which ``train_sdf.py`` and the demo consume source-agnostically.

Usage:
    python scripts/build_sdf.py <bundle_dir> [--source edt|cvc] [--region 430] [--grid 512]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from pycvc_gl.scenes import building_occupancy, terrain_grid

import sdf_nav

TARGET_EXTENT = 10.0


def _gltf_mesh(glb_path):
    """Flat (verts[x,y,z,...], tris[i,j,k,...]) from a glTF/GLB via VTK."""
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  (register factories)
    from vtkmodules.vtkIOGeometry import vtkGLTFReader
    from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
    from vtkmodules.util.numpy_support import vtk_to_numpy

    reader = vtkGLTFReader(); reader.SetFileName(glb_path); reader.Update()
    geom = vtkCompositeDataGeometryFilter(); geom.SetInputConnection(reader.GetOutputPort()); geom.Update()
    pd = geom.GetOutput()
    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    polys = vtk_to_numpy(pd.GetPolys().GetData())  # [n0,i,j,k, n1,...]; triangulated -> n0==3
    tris = []
    i = 0
    while i < len(polys):
        n = int(polys[i])
        if n == 3:
            tris += [int(polys[i + 1]), int(polys[i + 2]), int(polys[i + 3])]
        i += n + 1
    return pts.reshape(-1).tolist(), tris


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir")
    ap.add_argument("--source", choices=["edt", "cvc"], default="edt")
    ap.add_argument("--region", type=float, default=430.0, help="working-region half-extent (world)")
    ap.add_argument("--grid", type=int, default=512, help="2-D field resolution")
    ap.add_argument("--cvc-dim", type=int, nargs=3, default=(512, 512, 48), help="cvc 3-D SDF dims")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    terrain = os.path.join(args.bundle_dir, "terrain.json")
    glb = os.path.join(args.bundle_dir, "buildings.glb")
    _, bounds, _, _ = terrain_grid(terrain)
    mnx, mny, mxx, mxy = bounds
    cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    S = TARGET_EXTENT / (2.0 * args.region)

    if args.source == "edt":
        occ = building_occupancy(glb, bounds, args.grid, args.grid, inflate_m=0.0)
        phi, nxg, nyg = sdf_nav.build_sdf(occ, bounds, S)
    else:
        verts, tris = _gltf_mesh(glb)
        phi, nxg, nyg = sdf_nav.build_sdf_cvc(verts, tris, bounds, S, dim=tuple(args.cvc_dim))

    out = args.out or os.path.join(args.bundle_dir, "nav_sdf.npz")
    np.savez_compressed(out, phi=phi, normal_x=nxg, normal_y=nyg,
                        bounds=np.asarray(bounds, np.float32), center=np.asarray([cx, cy], np.float32),
                        scale=np.float32(S), region=np.float32(args.region), source=args.source)
    print("built %s SDF %s (phi[%.2f,%.2f]) -> %s"
          % (args.source, phi.shape, float(phi.min()), float(phi.max()), out))


if __name__ == "__main__":
    main()
