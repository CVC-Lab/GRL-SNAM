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

    def __init__(self, app=None, scene=None):
        """Build a Lab.

        Standalone (default): ``Lab()`` creates its own pycvc app + a fresh
        ``pycvc_gl.SceneGraph`` and renders through ``show()`` / ``render_png()``.

        Embedded: pass an existing ``app`` and/or ``scene`` to drive a HOST's
        live scene instead — e.g. inside volrover3::

            import vrhost
            lab = Lab(app=vrhost.app(), scene=vrhost.scene())
            lab.add_terrain(...)      # appears in the running volrover3 window

        ``scene`` is the host's live ``pycvc_gl.SceneGraph`` (adopted via
        ``vrhost.scene()``); every ``add_*`` mutates the running scene. Don't call
        ``show()`` in that mode — the host owns the render loop. If only ``app`` is
        given, a new SceneGraph is built on it; if only ``scene`` is given, its app
        is reused.
        """
        self._pycvc, self._gl = _require_pycvc()
        # One app handle owns this Lab's whole graphics context. Every pycvc
        # object built below (geometry/volume) and the scene co-own it via
        # shared_ptr, so it outlives them — there is no global singleton.
        self._app = app if app is not None else self._pycvc.make_app()
        # The REAL cvcGL SceneGraph (directly wrapped — not a facade): add_*
        # return the live node; move()/recolor() mutate it in place.
        self._scene = scene if scene is not None else self._gl.SceneGraph(self._app)

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
        self._scene.addGraphics(name, g)
        if color is not None:
            self.recolor(name, color)  # single-color material (faithful; see recolor)
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

    def add_path(self, name: str, points: Iterable[Sequence[float]], color: Color | None = None):
        """Add an agent trajectory as a polyline through ``points``."""
        verts, n = _flatten_points(points)
        if n < 2:
            raise ValueError("add_path needs at least 2 points")
        g = self._pycvc.geometry(self._app)
        g.add_vertices(verts)
        g.add_lines(polyline_indices(n))
        self._scene.addGraphics(name, g)
        if color is not None:
            self.recolor(name, color)  # single-color material (faithful; see recolor)
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
        self._scene.addGraphics(name, g)
        if color is not None:
            self.recolor(name, color)  # single-color material (faithful; see recolor)
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
        self._scene.addGraphics(name, v)
        return self

    # -- external / VTK props ------------------------------------------------

    def add_prop(self, name: str, prop, bounds: Bounds3D, parent: str = ""):
        """Add a pre-built VTK prop (e.g. a ``vtkActor`` loaded from glTF/OBJ) as a
        named scene node. ``bounds`` = ``(min_x, min_y, min_z, max_x, max_y,
        max_z)`` (used for framing/clipping). ``parent`` (default the graphics
        root) makes it a CHILD of that node so it inherits the parent's transform
        (e.g. a building mesh under the terrain node). Lets a whole city mesh join
        the scene without going through cvc::geometry. See ``grl_snam_lab.scenes``."""
        self._gl.add_prop(self._scene, name, prop, *[float(c) for c in bounds], parent)
        return self

    # -- in-place UPDATE (the animation path) --------------------------------

    def node(self, name: str):
        """The live scene node named ``name`` (a ``pycvc_gl.GraphicsNode``), or
        ``None`` if absent. Call ``.setPosition/.setScale/.setRotation/
        .setTransform/.resetTransform`` on it to move it without a rebuild."""
        return self._scene.getGraphics(name)

    def move(self, name: str, x: float, y: float, z: float):
        """Move node ``name`` to ``(x, y, z)`` by mutating its transform — the
        geometry + actor are reused, nothing is destroyed or recreated. This is
        the per-frame animation primitive (build once, ``move`` each frame)."""
        n = self._scene.getGraphics(name)
        if n is None:
            raise KeyError(f"grl_snam_lab.Lab.move: no node named {name!r}")
        n.setPosition(float(x), float(y), float(z))
        return self

    def recolor(self, name: str, color: Color):
        """Recolor node ``name`` in place with a single flat material color.

        Uses the node's material color (setUseSingleColor + setColor), NOT
        per-vertex colors: cvcGL renders per-vertex float colors through VTK's
        lookup table (a channel-mangling bug — a red mesh comes out blue), so a
        uniform color must go through the actor property to be faithful.
        """
        gn = self._scene.geometry_node(name)
        if gn is None:
            raise KeyError(f"grl_snam_lab.Lab.recolor: no mesh node named {name!r}")
        gn.setUseSingleColor(True)
        gn.setColor(*[float(c) for c in color])
        return self

    # -- lifecycle -----------------------------------------------------------

    def pump(self):
        self._scene.processEvents()
        return self

    def num_nodes(self) -> int:
        return self._scene.num_graphics()

    def has(self, name: str) -> bool:
        return self._scene.hasGraphics(name)

    def show(self, title: str = "GRL-SNAM lab", width: int = 1024, height: int = 768):
        """Open an interactive window (blocks until closed; needs a display)."""
        self._gl.show(self._scene, title, width, height)

    def render_png(self, path: str, width: int = 1024, height: int = 768):
        """Render one frame to a PNG (offscreen; needs a GL context)."""
        self._gl.render_png(self._scene, path, width, height)
