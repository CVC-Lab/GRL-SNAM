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
GROUND = (0.13, 0.14, 0.16)
GHOST = (0.95, 0.72, 0.25)  # believed-but-absent (amber)
WALL = (0.85, 0.25, 0.22)  # confirmed obstacle (red)
UNIT = (0.90, 0.35, 0.75)  # transient dynamic mark (magenta)
ROUTE = (0.35, 0.65, 1.00)  # planned spine (blue)
TRACK = (0.98, 0.85, 0.30)  # driven path (yellow)
CAR = (0.97, 0.97, 0.97)
GOAL = (1.00, 0.45, 0.20)
START = (0.35, 0.85, 0.45)

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

    # Truth obstacles that exist from the start (none in these three stories,
    # but a story with a pre-existing wall renders it for free).
    for i, rect in enumerate(story.truth_rects):
        x0, y0, x1, y1 = story.rect_world(rect)
        v, t = quad(x0, y0, x1, y1, 0.6)
        lab.add_mesh(f"truth{i}", v, t, WALL)

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

    _hide_chrome(lab)
    return {"snap": -2, "belief_nodes": set(), "track_every": 3, "frame": 0}


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
    if snap != state["snap"]:
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

    state["frame"] += 1
    if state["frame"] % state["track_every"] == 0:
        pts = trace.track_upto(t)
        if len(pts) >= 2:
            step = max(1, len(pts) // 400)  # cap the polyline; it only grows
            lab.add_path("track", [(float(x), float(y), 1.1) for x, y in pts[::step]], TRACK)
            _style(scene, "track", color=TRACK, mode="lines", line_width=4)

    return trace.to_metrics(t)
