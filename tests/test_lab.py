"""Tests for grl_snam_lab.

The geometry helpers are pure Python and always run; the end-to-end Lab test
is skipped where the compiled pycvc / pycvc_gl bindings aren't importable.
"""

import pytest

from grl_snam_lab import terrain_mesh
from grl_snam_lab.lab import _flatten_points, polyline_indices


def test_terrain_mesh_counts_and_placement():
    heights = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]  # 2 rows x 3 cols
    verts, tris = terrain_mesh(heights, bounds=(-10, -20, 10, 20))
    assert len(verts) == 2 * 3 * 3  # 6 vertices * xyz
    assert len(tris) == (2 - 1) * (3 - 1) * 6  # 2 cells * 2 tris * 3 idx
    # first vertex at (min_x, min_y, height[0][0]); last at (max_x, max_y, h[-1][-1])
    assert verts[0:3] == [-10.0, -20.0, 0.0]
    assert verts[-3:] == [10.0, 20.0, 5.0]


def test_terrain_mesh_rejects_degenerate():
    with pytest.raises(ValueError):
        terrain_mesh([[1.0]], bounds=(0, 0, 1, 1))


def test_polyline_indices():
    assert polyline_indices(4) == [0, 1, 1, 2, 2, 3]
    assert polyline_indices(1) == []


def test_flatten_points():
    buf, n = _flatten_points([(1, 2, 3), (4, 5, 6)])
    assert n == 2 and buf == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# ── end-to-end (requires the compiled bindings) ─────────────────────────

pytest.importorskip("pycvc", reason="pycvc bindings not installed")
pytest.importorskip("pycvc_gl", reason="pycvc_gl bindings not installed")


def test_lab_builds_a_scene():
    from grl_snam_lab import Lab

    lab = Lab()
    lab.add_terrain([[0, 0, 0], [0, 1, 0], [0, 0, 0]], bounds=(-5, -5, 5, 5))
    lab.add_mesh(
        "bldg",
        [0, 0, 0, 2, 0, 0, 0, 2, 0],
        [0, 1, 2],
        color=(0.6, 0.6, 0.6),
    )
    lab.add_path("agent0", [(-4, 0, 1), (0, 0, 1), (4, 2, 1)], color=(1, 0, 0))
    lab.add_markers("agents", [(-4, 0, 1), (4, 2, 1)])
    lab.add_field(
        "risk",
        [float(i % 5) for i in range(3 * 3 * 3)],
        dims=(3, 3, 3),
        bounds=(-5, -5, 0, 5, 5, 4),
    )
    lab.pump()
    for name in ("terrain", "bldg", "agent0", "agents", "risk"):
        assert lab.has(name), f"missing node {name}"
    assert lab.num_nodes() == 5
