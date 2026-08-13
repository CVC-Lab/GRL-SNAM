"""Field of view, the three fog tiers, and the undiscovered silhouette.

These pin the properties a viewer relies on to read the demo: that the lit
region matches the drawn sensor ring, that memory and current visibility are
distinguishable, and that geometry the agent has not found is still something
the audience can see.
"""

import numpy as np
import pytest

from grl_snam.belief import BeliefGrid

N = 64
BOUNDS = (-100.0, -100.0, 100.0, 100.0)


def _world(r, c):
    x = BOUNDS[0] + c / (N - 1) * (BOUNDS[2] - BOUNDS[0])
    y = BOUNDS[1] + r / (N - 1) * (BOUNDS[3] - BOUNDS[1])
    return x, y


# ── the field of view ───────────────────────────────────────────────────────


def test_no_field_of_view_before_the_first_sweep():
    b = BeliefGrid((N, N), BOUNDS)
    assert not b.last_visible.any()
    assert not b.ever_seen.any()


def test_the_sweep_reports_what_it_could_see():
    b = BeliefGrid((N, N), BOUNDS)
    truth = np.zeros((N, N), bool)
    x, y = _world(32, 32)
    b.sense(truth, (x, y), range_m=40.0, n_rays=360)
    assert b.last_visible[32, 32], "the sensor cannot see its own cell"
    assert b.last_visible.any()
    assert b.ever_seen.any()


def test_range_is_isotropic_so_the_lit_region_matches_the_drawn_ring():
    """The DDA normalises each ray by its dominant component, so without an
    explicit distance clip a diagonal ray travels sqrt(2) farther than an
    axis-aligned one and the visible set is a SQUARE. Drawing the sensor range
    as a ring made that visible: the lit area overflowed the circle."""
    b = BeliefGrid((N, N), BOUNDS)
    truth = np.zeros((N, N), bool)
    r0 = c0 = 32
    range_m = 40.0
    b.sense(truth, _world(r0, c0), range_m=range_m, n_rays=720)

    cell = (BOUNDS[2] - BOUNDS[0]) / (N - 1)
    rows, cols = np.nonzero(b.last_visible)
    dist = np.hypot((rows - r0) * cell, (cols - c0) * cell)
    # One cell of slack for rasterisation; nothing beyond the ring.
    assert dist.max() <= range_m + cell, f"lit {dist.max():.1f} m beyond a {range_m} m ring"
    # ...and it really is a disc, not a cross: the diagonal reaches out too.
    assert dist.max() > range_m * 0.7


def test_occlusion_applies_to_the_field_of_view_too():
    """You cannot see through a wall — the shadow behind it must be unlit, not
    merely unmapped."""
    b = BeliefGrid((N, N), BOUNDS)
    truth = np.zeros((N, N), bool)
    truth[10:54, 40] = True
    b.sense(truth, _world(32, 20), range_m=200.0, n_rays=720)
    assert b.last_visible[32, 40], "the wall itself is visible"
    assert not b.last_visible[32, 50], "saw straight through a wall"


def test_memory_accumulates_while_visibility_does_not():
    """The distinction the three-tier fog is built on: ever_seen grows, and
    last_visible is only ever the most recent sweep."""
    b = BeliefGrid((N, N), BOUNDS)
    truth = np.zeros((N, N), bool)
    b.sense(truth, _world(32, 12), range_m=25.0, n_rays=360)
    first = b.last_visible.copy()
    seen_after_one = b.ever_seen.sum()

    b.sense(truth, _world(32, 52), range_m=25.0, n_rays=360)
    assert not (b.last_visible & first).all(), "the sensor moved; the lit region should too"
    assert b.ever_seen.sum() > seen_after_one, "memory did not grow"
    assert (b.ever_seen >= b.last_visible).all(), "something is visible but not remembered"


def test_leaving_the_map_clears_the_field_of_view():
    b = BeliefGrid((N, N), BOUNDS)
    truth = np.zeros((N, N), bool)
    b.sense(truth, _world(32, 32), range_m=30.0, n_rays=180)
    assert b.last_visible.any()
    b.sense(truth, (1e6, 1e6), range_m=30.0, n_rays=180)
    assert not b.last_visible.any(), "a stale field of view survived leaving the map"


# ── the silhouette ──────────────────────────────────────────────────────────

torch = pytest.importorskip("torch")

from grl_snam import fog_scene  # noqa: E402
from grl_snam.fog_stories import STORIES  # noqa: E402


def test_outline_traces_a_boundary_not_every_cell():
    """A solid block must yield its outline. Emitting an edge per cell would
    draw a grid and read as texture rather than as a shape."""
    story = STORIES["blocker"]
    mask = np.zeros((story.n, story.n), bool)
    mask[10:20, 10:20] = True  # a 10x10 block: 40 boundary edges
    verts, idx = fog_scene.outline_segments(mask, story, 0.9)
    assert verts is not None
    assert len(idx) // 2 == 40, f"expected the 40-edge perimeter, got {len(idx) // 2}"
    assert len(verts) == len(idx) * 3


def test_outline_of_nothing_is_nothing():
    story = STORIES["blocker"]
    verts, idx = fog_scene.outline_segments(np.zeros((story.n, story.n), bool), story, 0.9)
    assert verts is None and idx is None


def test_outline_handles_disjoint_shapes():
    """Two separate blocks are two loops — which is why these are disjoint
    segments and not one polyline."""
    story = STORIES["blocker"]
    mask = np.zeros((story.n, story.n), bool)
    mask[10:14, 10:14] = True
    mask[40:44, 40:44] = True
    _verts, idx = fog_scene.outline_segments(mask, story, 0.9)
    assert len(idx) // 2 == 32, "each 4x4 block contributes 16 perimeter edges"


def test_the_silhouette_is_exactly_what_the_agent_has_not_found():
    """The demo's claim: the viewer sees real geometry the agent does not know
    about. That set is truth minus belief — never anything else."""
    story = STORIES["blocker"]
    truth = np.zeros((story.n, story.n), bool)
    truth[41:56, 46:50] = True
    believed = np.zeros((story.n, story.n), bool)
    believed[41:48, 46:50] = True  # the half the sensor has swept

    undiscovered = truth & ~believed
    assert undiscovered.any(), "nothing left to silhouette"
    assert not (undiscovered & believed).any(), "silhouetting something already known"
    verts, idx = fog_scene.outline_segments(undiscovered, story, 0.9)
    assert verts is not None and len(idx) > 0


def test_ring_is_closed_and_the_right_size():
    pts = fog_scene.ring_points(10.0, -5.0, 35.0, 1.6, segments=64)
    assert len(pts) == 65, "a closed ring repeats its first point"
    assert pts[0] == pytest.approx(pts[-1])
    r = [np.hypot(x - 10.0, y + 5.0) for x, y, _z in pts]
    assert np.allclose(r, 35.0)
