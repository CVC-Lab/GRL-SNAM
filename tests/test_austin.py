"""City-scale pieces: routable endpoint selection, plate sizing, no-fog mode.

These are the parts that can be tested without the scene bundle — the bundle
is a local artifact and its path is never committed, so the rasterizer itself
is exercised by actually running the demo rather than in CI.
"""

import dataclasses

import numpy as np
import pytest

from grl_snam.planner import far_pair_in_free_space, free_components, inflate

torch = pytest.importorskip("torch")

from grl_snam import fog_scene  # noqa: E402
from grl_snam.fog_stories import STORIES  # noqa: E402

# ── endpoints must come from the space the planner will search ──────────────


def test_components_are_labelled_under_inflation():
    """Free space and INFLATED free space are different maps. A cell can be
    free and still unreachable once the route is inflated for clearance."""
    occ = np.zeros((40, 40), bool)
    occ[:, 20] = True  # a wall splitting the map, with no door
    labels, sizes = free_components(occ, 1)
    assert len(sizes) == 2, "a solid wall makes exactly two components"
    assert labels[5, 5] != labels[5, 35]


def test_a_gap_narrower_than_the_inflation_is_not_a_route():
    """The Austin failure in miniature: a doorway that looks open on the raw
    map is closed once the planner inflates, so endpoints chosen from raw free
    space are unreachable."""
    occ = np.zeros((40, 40), bool)
    occ[:, 20] = True
    occ[19:22, 20] = False  # a 3-cell doorway

    _lbl, sizes_open = free_components(occ, 1)
    _lbl, sizes_shut = free_components(occ, 3)
    assert len(sizes_open) == 1, "at inflation 1 the doorway connects both sides"
    assert len(sizes_shut) >= 2, "at inflation 3 the doorway should be closed"


def test_the_chosen_pair_is_in_one_component_and_far_apart():
    occ = np.zeros((60, 60), bool)
    occ[:, 30] = True
    occ[29:32, 30] = False
    bounds = (-100.0, -100.0, 100.0, 100.0)
    a, b, size = far_pair_in_free_space(occ, bounds, 1)
    assert size > 0
    sep = np.hypot(a[0] - b[0], a[1] - b[1])
    assert sep > 100.0, f"endpoints only {sep:.0f} m apart on a 200 m map"

    labels, _ = free_components(occ, 1)
    ny, nx = occ.shape

    def to_cell(p):
        c = int(round((p[0] - bounds[0]) / (bounds[2] - bounds[0]) * (nx - 1)))
        r = int(round((p[1] - bounds[1]) / (bounds[3] - bounds[1]) * (ny - 1)))
        return r, c

    ra, ca = to_cell(a)
    rb, cb = to_cell(b)
    assert labels[ra, ca] == labels[rb, cb] != 0, "endpoints landed in different components"


def test_a_fully_blocked_map_is_reported_not_guessed():
    with pytest.raises(ValueError):
        far_pair_in_free_space(np.ones((20, 20), bool), (-1.0, -1.0, 1.0, 1.0), 1)


def test_inflation_grows_obstacles_monotonically():
    occ = np.zeros((20, 20), bool)
    occ[10, 10] = True
    assert inflate(occ, 0).sum() == 1
    assert inflate(occ, 1).sum() == 5  # 4-connected cross
    assert inflate(occ, 2).sum() > inflate(occ, 1).sum()


# ── the plate has to cover the world ────────────────────────────────────────


def test_the_plate_is_16_9_and_covers_the_world():
    """The plate was a constant tuned for the +-100 m stories. On a 1200 m city
    it was smaller than the world, so the capture's crop calibrated onto a
    patch in the middle and the clip showed a fraction of the map."""
    for half in (100.0, 600.0, 1500.0):
        story = dataclasses.replace(STORIES["city"], bounds=(-half, -half, half, half))
        hx, hy = fog_scene.plate_half(story)
        assert hx >= half and hy >= half, "the plate does not cover the world"
        assert hx / hy == pytest.approx(16.0 / 9.0)


def test_the_plate_still_matches_the_original_stories():
    hx, hy = fog_scene.plate_half(STORIES["ghost"])
    assert (hx, hy) == pytest.approx((fog_scene.PLATE_HALF_X, fog_scene.PLATE_HALF_Y))


# ── full knowledge renders lit ──────────────────────────────────────────────


def test_no_fog_is_a_story_property():
    """'Without fog of war' is expressed as the agent starting with the true
    map -- same sensor, same planner, different initial knowledge -- plus a
    lit render, because drawing fog over a map the agent fully knows is a lie."""
    assert STORIES["city"].no_fog is False
    lit = dataclasses.replace(STORIES["city"], no_fog=True)
    assert lit.no_fog is True
