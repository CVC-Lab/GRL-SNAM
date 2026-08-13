"""The fog demo's scene — the only code that turns a story + trace into pictures.

Both renderers call exactly this: the live VolRover3 demo and the offscreen
capture. One scene builder, one measured bundle, two hosts — which is what
makes "the video and the live window show the same thing" a structural fact
rather than a promise.

Nothing here simulates. Every position, heading, speed, route and belief
snapshot is read from the trace; see :mod:`grl_snam.fog_trace`.

Hard-won details about the graphics layer, all verified on hardware:

* ``add_path``/``add_markers`` build geometry but leave the render mode at the
  default TRIS, under which a polyline and a point cloud draw **nothing**. The
  route and the markers need ``setRenderMode(LINES/POINTS)`` plus a line width
  or point size, or they are silently invisible.
* the render-mode/material setters live on ``scene.geometry_node(name)``
  (a ``GeometryNode``); ``scene.getGraphics(name)`` returns a ``GraphicsNode``
  which has transforms and visibility but no material.
* bounding boxes and extent labels must be turned off per node **and** on
  ``getGraphicsRoot()``, and the grid via ``scene.setGridVisible(False)`` —
  otherwise every frame carries a yellow box and "Max: (178.00, ...)" text.
* ``add_mesh`` with an existing name replaces the node and resets its material,
  so anything re-meshed must have its colour/mode/flags re-applied.
"""

from __future__ import annotations

import math

import numpy as np

# Palette. Deliberately few, high-contrast colours: at map scale a viewer has
# to decode the frame in about a second.
# The ground plate IS the fog: unseen is the default state of the world, and
# knowledge is painted on top of it.
GROUND = (0.055, 0.06, 0.07)
GHOST = (0.95, 0.72, 0.25)  # believed-but-absent (amber)
WALL = (0.85, 0.25, 0.22)  # confirmed obstacle (red)
UNIT = (0.90, 0.35, 0.75)  # transient dynamic mark (magenta)
ROUTE = (0.35, 0.65, 1.00)  # planned spine (blue)
TRACK = (0.98, 0.85, 0.30)  # driven path (yellow)
CAR = (0.97, 0.97, 0.97)
GOAL = (1.00, 0.45, 0.20)
START = (0.35, 0.85, 0.45)
# Fog of war is three-tier and the tiers must be visually ordered: never seen
# is darkest, remembered is dimmer than lit, and what the sensor can see right
# now gets no overlay at all.
FOG_REMEMBERED = (0.13, 0.14, 0.16)  # mapped once, not visible now
FOG_VISIBLE = (0.20, 0.22, 0.26)  # inside the sensor's reach right now
# Real geometry the agent has NOT discovered. Drawn as an outline so a viewer
# can see something is there while the agent's route walks straight past it --
# the single most important thing to show, because otherwise the audience has
# no way to know the map is wrong.
SILHOUETTE = (0.42, 0.46, 0.55)
FOV = (0.30, 0.72, 0.95)

# The vehicle is ~4.5 m long in world units, which is a speck on a 200 m map.
# Draw it larger than life so the heading is legible; the DYNAMICS are honest,
# the icon is not to scale (and the run-book says so).
CAR_L, CAR_W = 12.0, 6.0

# A 16:9 ground plate keeps ResetCamera's framing stable across stories and
# gives the capture a fixed rectangle to calibrate its crop against.
PLATE_HALF_Y = 100.0
PLATE_HALF_X = PLATE_HALF_Y * 16.0 / 9.0


def quad(x0, y0, x1, y1, z):
    """Axis-aligned quad as (vertices, triangles)."""
    return (
        [x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z],
        [0, 1, 2, 0, 2, 3],
    )


def cells_mesh(mask, story, z, *, inset=0.12):
    """One quad per set cell — the belief/unit overlay.

    Returned as a single mesh rather than a node per cell: story 3 re-meshes
    this 61 times, and 61 x (hundreds of nodes) of scene-graph churn in a live
    host is not something to find out about on stage.
    """
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None, None
    mnx, mny, mxx, mxy = story.bounds
    cw = (mxx - mnx) / (story.n - 1)
    ch = (mxy - mny) / (story.n - 1)
    dx, dy = cw * (0.5 - inset), ch * (0.5 - inset)
    verts: list[float] = []
    tris: list[int] = []
    for r, c in zip(rows, cols):
        x, y = story.cell_to_world(float(r), float(c))
        base = len(verts) // 3
        verts += [x - dx, y - dy, z, x + dx, y - dy, z, x + dx, y + dy, z, x - dx, y + dy, z]
        tris += [base, base + 1, base + 2, base, base + 2, base + 3]
    return verts, tris


def outline_segments(mask, story, z):
    """Boundary edges of a cell mask, as disjoint line segments.

    Emits an edge only where a set cell meets an unset one, so a solid block
    yields its outline rather than a grid of every cell. Vectorised: the four
    shifted comparisons are the whole algorithm, which keeps this affordable
    even at Austin grid sizes.

    Returns ``(vertices, indices)`` for ``geometry.add_lines`` -- index PAIRS,
    not a polyline, because the boundary is generally several disjoint loops.
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return None, None
    mnx, mny, mxx, mxy = story.bounds
    cw = (mxx - mnx) / (story.n - 1)
    ch = (mxy - mny) / (story.n - 1)
    verts: list[float] = []
    idx: list[int] = []

    def edge(r0, c0, r1, c1):
        # cell corner coordinates (cell centre offset by half a cell)
        x0 = mnx + (c0 - 0.5) * cw
        y0 = mny + (r0 - 0.5) * ch
        x1 = mnx + (c1 - 0.5) * cw
        y1 = mny + (r1 - 0.5) * ch
        base = len(verts) // 3
        verts.extend([x0, y0, z, x1, y1, z])
        idx.extend([base, base + 1])

    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), bool)
    pad[1:-1, 1:-1] = m
    core = pad[1:-1, 1:-1]
    up, down = pad[:-2, 1:-1], pad[2:, 1:-1]
    left, right = pad[1:-1, :-2], pad[1:-1, 2:]
    for rr, cc in zip(*np.nonzero(core & ~up)):
        edge(rr, cc, rr, cc + 1)
    for rr, cc in zip(*np.nonzero(core & ~down)):
        edge(rr + 1, cc, rr + 1, cc + 1)
    for rr, cc in zip(*np.nonzero(core & ~left)):
        edge(rr, cc, rr + 1, cc)
    for rr, cc in zip(*np.nonzero(core & ~right)):
        edge(rr, cc + 1, rr + 1, cc + 1)
    return verts, idx


def add_segments(lab, name, verts, idx, color):
    """Disjoint line segments. Lab.add_path would connect them into one
    polyline, which for a boundary means spurious edges across the scene."""
    g = lab._pycvc.geometry(lab._app)
    g.add_vertices(verts)
    g.add_lines(idx)
    lab._scene.addGraphics(name, g)
    return lab


def ring_points(cx, cy, radius, z, segments=72):
    """A closed circle as a polyline -- the sensor's range at a glance."""
    ang = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    return [(float(cx + radius * np.cos(a)), float(cy + radius * np.sin(a)), z) for a in ang]


def _style(scene, name, *, color=None, mode=None, line_width=None, point_size=None, opacity=None):
    """Apply material after every (re)mesh — add_mesh resets it."""
    import pycvc_gl

    g = scene.geometry_node(name)
    if color is not None:
        g.setUseSingleColor(True)
        g.setColor(*color)
    if mode == "lines":
        g.setRenderMode(pycvc_gl.GeometryRenderMode_LINES)
    elif mode == "points":
        g.setRenderMode(pycvc_gl.GeometryRenderMode_POINTS)
    if line_width is not None:
        g.setLineWidth(line_width)
    if point_size is not None:
        g.setPointSize(point_size)
    if opacity is not None:
        g.setOpacity(opacity)
    g.setShowBBox(False)
    g.setShowExtentLabels(False)
    g.setShowLabel(False)
    return g


def _hide_chrome(lab):
    scene = lab._scene
    for n in list(scene.graphics_names()):
        g = scene.geometry_node(n)
        g.setShowBBox(False)
        g.setShowExtentLabels(False)
        g.setShowLabel(False)
    root = scene.getGraphicsRoot()
    root.setShowBBox(False)
    root.setShowExtentLabels(False)
    root.setShowLabel(False)
    scene.setGridVisible(False)
    lab.set_axis_visible(False)


def build(lab, story, trace) -> dict:
    """Create every node once. Returns the mutable render state."""
    scene = lab._scene

    v, t = quad(-PLATE_HALF_X, -PLATE_HALF_Y, PLATE_HALF_X, PLATE_HALF_Y, 0.0)
    lab.add_mesh("ground", v, t, GROUND)

    # Ground truth is NOT drawn directly. It reaches the frame only through
    # what the agent believes (red), what it believes wrongly (amber), and
    # what it has not found (outline) -- see apply(). Drawing truth_rects here
    # painted every real building as known the moment the scene loaded, which
    # silently revealed the answer and made the fog decorative. The three
    # original stories had no truth_rects, so the bug stayed invisible until a
    # scene with real geometry was added.

    sx, sy = story.start
    gx, gy = story.waypoints[-1]
    lab.add_markers("start", [(sx, sy, 1.2)], START)
    lab.add_markers("goal", [(gx, gy, 1.2)], GOAL)
    _style(scene, "start", mode="points", point_size=16)
    _style(scene, "goal", mode="points", point_size=22)

    # Placeholders so every node exists before the first frame; a node created
    # mid-playback in a live host is the least-exercised path there is.
    lab.add_path("route", [(sx, sy, 1.0), (gx, gy, 1.0)], ROUTE)
    _style(scene, "route", mode="lines", line_width=5)
    lab.add_path("track", [(sx, sy, 1.1), (sx + 0.1, sy, 1.1)], TRACK)
    _style(scene, "track", mode="lines", line_width=4)

    v, t = quad(-CAR_L / 2, -CAR_W / 2, CAR_L / 2, CAR_W / 2, 1.4)
    lab.add_mesh("car", v, t, CAR)
    # A nose wedge so heading is unambiguous even when the car is stationary.
    v, t = quad(CAR_L / 2, -CAR_W / 6, CAR_L / 2 + 4.0, CAR_W / 6, 1.5)
    lab.add_mesh("nose", v, t, GOAL)

    # The sensor's reach, drawn as a ring that follows the vehicle. Created
    # here so the node exists before the first frame; moved, not re-meshed.
    if trace.sensor_range_m > 0:
        lab.add_path("fov", ring_points(sx, sy, trace.sensor_range_m, 1.6), FOV)
        _style(scene, "fov", mode="lines", line_width=2)

    _hide_chrome(lab)
    return {
        "snap": -2,
        "fov_snap": -2,
        "belief_nodes": set(),
        "track_every": 3,
        "frame": 0,
        "fov_origin": (sx, sy),
    }


def apply(lab, story, trace, t, state):
    """Advance the scene to world time ``t``. Returns the frame's NavMetrics."""
    scene = lab._scene
    pose = trace.pose_at(t)

    # Pose comes from the RECORDED heading — never re-derived from the
    # position stream, which is what made the older capture disagree with its
    # own simulation.
    c, s = math.cos(pose.heading_rad), math.sin(pose.heading_rad)
    m = [c, -s, 0.0, pose.x, s, c, 0.0, pose.y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    scene.geometry_node("car").setTransform(m)
    scene.geometry_node("nose").setTransform(m)

    # Belief is a step function: re-mesh only when the sensor changed its mind.
    snap = trace.snapshot_index_at(t)
    fov_k = trace.fov_index_at(t)
    if snap != state["snap"] or fov_k != state["fov_snap"]:
        state["snap"] = snap
        occ, dyn, _ = trace.belief_at(t)

        # Believed-occupied cells that reality does NOT have are ghosts; ones
        # it does have are confirmed. Colouring them differently is the entire
        # point of story 1 and it is a measurement, not a renderer heuristic.
        truth = story.truth_grid()
        for key, mask, color, z in (
            ("ghost", occ & ~truth, GHOST, 0.5),
            ("wall", occ & truth, WALL, 0.6),
            ("unit", dyn, UNIT, 0.7),
        ):
            v, tri = cells_mesh(mask, story, z)
            if v is None:
                if key in state["belief_nodes"]:
                    scene.geometry_node(key).setVisible(False)
                continue
            lab.add_mesh(key, v, tri, color)
            _style(scene, key, color=color)  # add_mesh resets material
            scene.geometry_node(key).setVisible(True)
            state["belief_nodes"].add(key)

        route = trace.route_at(t)
        if len(route) >= 2:
            lab.add_path("route", [(float(x), float(y), 1.0) for x, y in route], ROUTE)
            _style(scene, "route", color=ROUTE, mode="lines", line_width=5)

    # ── fog of war: three tiers + the undiscovered silhouette ────────────────
    # Re-meshed only when the sensor sweep actually changed (every sense tick,
    # not every frame): at 0.5x playback that is roughly one frame in six.
    if fov_k != state["fov_snap"]:
        state["fov_snap"] = fov_k
        fov = trace.fov_at(t)
        if fov is not None:
            visible, seen = fov
            truth = story.truth_grid()
            for key, mask, color, z in (
                ("fog_seen", seen & ~visible, FOG_REMEMBERED, 0.28),
                ("fog_now", visible, FOG_VISIBLE, 0.30),
            ):
                v, tri = cells_mesh(mask, story, z, inset=0.0)
                if v is None:
                    if key in state["belief_nodes"]:
                        scene.geometry_node(key).setVisible(False)
                    continue
                lab.add_mesh(key, v, tri, color)
                _style(scene, key, color=color)
                scene.geometry_node(key).setVisible(True)
                state["belief_nodes"].add(key)

            # Real geometry the agent has not found. The viewer sees the
            # outline; the agent's route does not know it exists.
            occ, _dyn, _k = trace.belief_at(t)
            undiscovered = truth & ~occ
            v, idx = outline_segments(undiscovered, story, 0.9)
            if v is None:
                if "silhouette" in state["belief_nodes"]:
                    scene.geometry_node("silhouette").setVisible(False)
            else:
                add_segments(lab, "silhouette", v, idx, SILHOUETTE)
                _style(scene, "silhouette", color=SILHOUETTE, mode="lines", line_width=2)
                scene.geometry_node("silhouette").setVisible(True)
                state["belief_nodes"].add("silhouette")

    # The FOV ring rides along with the vehicle rather than being rebuilt.
    if trace.sensor_range_m > 0 and scene.hasGraphics("fov"):
        ox, oy = state["fov_origin"]
        n = scene.geometry_node("fov")
        n.setTransform([1, 0, 0, pose.x - ox, 0, 1, 0, pose.y - oy, 0, 0, 1, 0, 0, 0, 0, 1])

    state["frame"] += 1
    if state["frame"] % state["track_every"] == 0:
        pts = trace.track_upto(t)
        if len(pts) >= 2:
            step = max(1, len(pts) // 400)  # cap the polyline; it only grows
            lab.add_path("track", [(float(x), float(y), 1.1) for x, y in pts[::step]], TRACK)
            _style(scene, "track", color=TRACK, mode="lines", line_width=4)

    return trace.to_metrics(t)
