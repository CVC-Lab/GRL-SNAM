"""The Lab class and pure-Python geometry helpers.

The geometry math (terrain triangulation, polyline indexing) is kept pure
Python so it is testable without the compiled ``pycvc`` bindings; the ``Lab``
methods build ``pycvc`` objects and add them to a ``pycvc_gl`` scene.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# A flat, row-major XYZ vertex buffer and a flat triangle-index buffer.
FlatVerts = list
FlatIndices = list

Color = tuple  # (r, g, b) in 0..1
Bounds2D = tuple  # (min_x, min_y, max_x, max_y)
Bounds3D = tuple  # (min_x, min_y, min_z, max_x, max_y, max_z)


# ── pure-Python geometry helpers (no pycvc) ─────────────────────────────


def terrain_mesh(
    heights: Sequence[Sequence[float]], bounds: Bounds2D
) -> tuple[FlatVerts, FlatIndices]:
    """Triangulate a ``rows x cols`` heightmap over an XY box into a surface.

    Returns ``(vertices_flat, triangles_flat)``: ``vertices_flat`` is
    ``[x,y,z, ...]`` row-major over the grid (z is the height), and
    ``triangles_flat`` is ``[i,j,k, ...]`` (two triangles per grid cell).
    """
    rows = len(heights)
    cols = len(heights[0]) if rows else 0
    if rows < 2 or cols < 2:
        raise ValueError("terrain_mesh needs a grid of at least 2x2")
    min_x, min_y, max_x, max_y = bounds
    dx = (max_x - min_x) / (cols - 1)
    dy = (max_y - min_y) / (rows - 1)

    verts: FlatVerts = []
    for r in range(rows):
        y = min_y + r * dy
        row = heights[r]
        for c in range(cols):
            verts += [min_x + c * dx, y, float(row[c])]

    tris: FlatIndices = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v = r * cols + c
            tris += [v, v + 1, v + cols]
            tris += [v + 1, v + cols + 1, v + cols]
    return verts, tris


def _flatten_points(points: Iterable[Sequence[float]]) -> tuple[FlatVerts, int]:
    """Flatten an iterable of (x, y, z) into a flat buffer; return (buf, n)."""
    verts: FlatVerts = []
    n = 0
    for p in points:
        verts += [float(p[0]), float(p[1]), float(p[2])]
        n += 1
    return verts, n


def polyline_indices(n: int) -> FlatIndices:
    """Consecutive-segment line indices for a path of ``n`` vertices:
    ``[0,1, 1,2, ..., n-2,n-1]``."""
    idx: FlatIndices = []
    for i in range(n - 1):
        idx += [i, i + 1]
    return idx


# ── the Lab ─────────────────────────────────────────────────────────────


def _require_pycvc():
    try:
        import pycvc
        import pycvc_gl
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "grl_snam_lab needs the pycvc / pycvc_gl bindings (from libcvc). "
            "Install libcvc with its Python bindings into your prefix "
            "(e.g. via cvcpkg) and put them on PYTHONPATH."
        ) from exc
    return pycvc, pycvc_gl


class Lab:
    """A live 3D GRL-SNAM scene: add terrain, obstacles, agent paths, agents,
    and scalar fields, then show it. Domain-general — no dataset knowledge."""

    def __init__(self):
        self._pycvc, self._gl = _require_pycvc()
        # One app handle owns this Lab's whole graphics context. Every pycvc
        # object built below (geometry/volume) and the scene co-own it via
        # shared_ptr, so it outlives them — there is no global singleton.
        self._app = self._pycvc.make_app()
        self._scene = self._gl.Scene(self._app)

    # -- meshes --------------------------------------------------------------

    def add_mesh(
        self,
        name: str,
        vertices: Sequence[float],
        triangles: Sequence[int],
        color: Color | None = None,
    ):
        """Add a triangle mesh (obstacle / building / surface) from flat
        row-major ``vertices`` (``[x,y,z,...]``) and ``triangles``
        (``[i,j,k,...]``)."""
        g = self._pycvc.geometry(self._app)
        g.add_vertices(list(vertices))
        g.add_triangles(list(triangles))
        if color is not None:
            g.set_colors(list(color) * g.num_vertices())
        self._scene.add_geometry(name, g)
        return self

    def add_terrain(
        self,
        heights: Sequence[Sequence[float]],
        bounds: Bounds2D,
        color: Color | None = None,
    ):
        """Add a terrain surface from a heightmap over an XY box."""
        verts, tris = terrain_mesh(heights, bounds)
        return self.add_mesh("terrain", verts, tris, color=color)

    # -- agents & paths ------------------------------------------------------

    def add_path(
        self, name: str, points: Iterable[Sequence[float]], color: Color | None = None
    ):
        """Add an agent trajectory as a polyline through ``points``."""
        verts, n = _flatten_points(points)
        if n < 2:
            raise ValueError("add_path needs at least 2 points")
        g = self._pycvc.geometry(self._app)
        g.add_vertices(verts)
        g.add_lines(polyline_indices(n))
        if color is not None:
            g.set_colors(list(color) * g.num_vertices())
        self._scene.add_geometry(name, g)
        return self

    def add_markers(
        self, name: str, positions: Iterable[Sequence[float]], color: Color | None = None
    ):
        """Add agents as points (markers) at ``positions``."""
        verts, n = _flatten_points(positions)
        if n < 1:
            raise ValueError("add_markers needs at least 1 position")
        g = self._pycvc.geometry(self._app)
        g.add_vertices(verts)
        if color is not None:
            g.set_colors(list(color) * g.num_vertices())
        self._scene.add_geometry(name, g)
        return self

    # -- scalar fields -------------------------------------------------------

    def add_field(
        self,
        name: str,
        values: Sequence[float],
        dims: tuple[int, int, int],
        bounds: Bounds3D,
    ):
        """Add a scalar field (risk / SDF / path-loss) as a volume node.

        ``values`` is a flat, row-major (x fastest) grid of ``nx*ny*nz``
        scalars; ``dims`` = (nx, ny, nz); ``bounds`` = the object-space box.
        """
        nx, ny, nz = dims
        v = self._pycvc.volume(self._app)
        v.set_float_grid(list(values), nx, ny, nz, *bounds)
        self._scene.add_volume(name, v)
        return self

    # -- lifecycle -----------------------------------------------------------

    def pump(self):
        self._scene.pump()
        return self

    def num_nodes(self) -> int:
        return self._scene.num_graphics()

    def has(self, name: str) -> bool:
        return self._scene.has(name)

    def show(self, title: str = "GRL-SNAM lab", width: int = 1024, height: int = 768):
        """Open an interactive window (blocks until closed; needs a display)."""
        self._scene.show(title, width, height)

    def render_png(self, path: str, width: int = 1024, height: int = 768):
        """Render one frame to a PNG (offscreen; needs a GL context)."""
        self._scene.render_png(path, width, height)
