"""Load real-world scene geometry (terrain heightfields + glTF city meshes) into
a :class:`grl_snam_lab.lab.Lab`.

These are generic loaders for a ``geometry_bundle`` export — a ``terrain.json``
heightfield plus a ``buildings.glb`` (glTF 2.0) city mesh, as produced by the
CVC-DBG ``geometry-scene-gen`` tool (e.g. the Austin bundle). The glTF is read
with VTK's ``vtkGLTFReader`` (no trimesh/pygltflib needed) and added as a single
VTK prop; the terrain becomes a draped surface mesh; a bilinear ``sampler`` lets
you drape an agent onto the terrain.

ATTRIBUTION: bundles generated from OpenStreetMap are © OpenStreetMap
contributors and licensed under the Open Database License (ODbL,
https://openstreetmap.org/copyright); SRTM terrain is US public domain. If you
publish renders, credit OpenStreetMap. Do NOT ship the Esri ``satellite.png``
overlay some bundles carry — it is proprietary; these loaders never touch it.
"""

from __future__ import annotations

import json
import os


def terrain_grid(path: str):
    """Read a ``terrain.json`` heightfield: returns ``(grid, bounds2d, rows,
    cols)`` where ``grid[row][col]`` is height (row -> y, col -> x) and
    ``bounds2d`` = ``(min_x, min_y, max_x, max_y)``."""
    d = json.load(open(path))
    b = d["bounds"]
    return d["grid"], (b["min_x"], b["min_y"], b["max_x"], b["max_y"]), d["rows"], d["cols"]


def terrain_sampler(path: str):
    """A bilinear height function ``h(x, y)`` over a ``terrain.json`` grid — use
    it to DRAPE an agent/path onto the real terrain (clamped at the edges)."""
    grid, (min_x, min_y, max_x, max_y), rows, cols = terrain_grid(path)
    sx = (cols - 1) / (max_x - min_x) if max_x > min_x else 0.0
    sy = (rows - 1) / (max_y - min_y) if max_y > min_y else 0.0

    def h(x: float, y: float) -> float:
        fx = (x - min_x) * sx
        fy = (y - min_y) * sy
        fx = 0.0 if fx < 0 else (cols - 1 if fx > cols - 1 else fx)
        fy = 0.0 if fy < 0 else (rows - 1 if fy > rows - 1 else fy)
        c0, r0 = int(fx), int(fy)
        c1 = c0 + 1 if c0 < cols - 1 else c0
        r1 = r0 + 1 if r0 < rows - 1 else r0
        tx, ty = fx - c0, fy - r0
        top = grid[r0][c0] * (1 - tx) + grid[r0][c1] * tx
        bot = grid[r1][c0] * (1 - tx) + grid[r1][c1] * tx
        return top * (1 - ty) + bot * ty

    return h


def add_terrain_json(lab, path: str, name: str = "terrain", color=(0.34, 0.40, 0.28)):
    """Add a ``terrain.json`` heightfield to ``lab`` as a draped surface; returns
    a ``sampler`` ``h(x, y)`` for the SAME field (so agents can drape onto it).

    NOTE: uses the Lab's terrain mesh, which names the node ``"terrain"``.
    """
    grid, bounds2d, _rows, _cols = terrain_grid(path)
    lab.add_terrain(grid, bounds=bounds2d, color=color)
    return terrain_sampler(path)


def add_gltf(lab, path: str, name: str, color=(0.74, 0.74, 0.78), opacity: float = 1.0):
    """Load a glTF/GLB mesh with VTK and add it to ``lab`` as one named prop node.
    Returns the ``vtkActor``. Needs the vtk-python wrappers (vtkmodules)."""
    from vtkmodules.vtkIOGeometry import vtkGLTFReader
    from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

    reader = vtkGLTFReader()
    reader.SetFileName(path)
    reader.Update()
    # glTF comes back as a multiblock; flatten to one polydata.
    geom = vtkCompositeDataGeometryFilter()
    geom.SetInputConnection(reader.GetOutputPort())
    geom.Update()
    pd = geom.GetOutput()

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.ScalarVisibilityOff()  # use the single material color, not any glTF scalars
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)

    b = pd.GetBounds()  # (xmin,xmax, ymin,ymax, zmin,zmax)
    lab.add_prop(name, actor, (b[0], b[2], b[4], b[1], b[3], b[5]))
    return actor


def load_geometry_bundle(
    lab,
    bundle_dir: str,
    terrain_color=(0.34, 0.40, 0.28),
    building_color=(0.74, 0.74, 0.78),
    buildings: bool = True,
):
    """Load a ``geometry_bundle`` directory (``terrain.json`` + ``buildings.glb``)
    into ``lab``. Returns the terrain ``sampler`` ``h(x, y)`` for draping agents.

    ``buildings=False`` loads only the terrain (much lighter). The Esri
    ``satellite.png`` some bundles carry is intentionally NOT loaded (proprietary).
    """
    terrain_path = os.path.join(bundle_dir, "terrain.json")
    sampler = add_terrain_json(lab, terrain_path, color=terrain_color)
    if buildings:
        glb = os.path.join(bundle_dir, "buildings.glb")
        if os.path.exists(glb):
            add_gltf(lab, glb, "buildings", color=building_color)
    return sampler
