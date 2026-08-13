"""The distance transform, and the route inflation that depends on cell size.

Both are places where something correct-looking scales badly: an algorithm
whose Python-level loop count grows with the raster, and a tolerance expressed
in cells rather than metres.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sdf_nav  # noqa: E402


def _reference_edt2(mask):
    """The original scalar implementation, kept here as the oracle."""
    f = np.where(mask, 0.0, 1e20)
    return np.apply_along_axis(sdf_nav._edt1d, 1, np.apply_along_axis(sdf_nav._edt1d, 0, f))


@pytest.mark.parametrize("n", [1, 2, 7, 16, 33, 64])
@pytest.mark.parametrize("density", [0.0, 0.03, 0.5, 1.0])
def test_vectorized_edt_is_exactly_the_scalar_one(n, density):
    """Bit-identical, not merely close: the SDF feeds a learned policy, and a
    field that differs in the last ulp makes a run stop reproducing."""
    rng = np.random.default_rng(n * 100 + int(density * 10))
    if density == 0.0:
        mask = np.zeros((n, n), bool)
    elif density == 1.0:
        mask = np.ones((n, n), bool)
    else:
        mask = rng.random((n, n)) < density
    assert np.array_equal(sdf_nav._edt2(mask), _reference_edt2(mask))


def test_edt_handles_a_single_seed_and_an_empty_grid():
    m = np.zeros((12, 12), bool)
    m[5, 7] = True
    d = sdf_nav._edt2(m)
    assert d[5, 7] == 0.0
    assert d[5, 8] == 1.0
    assert d[7, 7] == 4.0  # squared distance, grid units
    empty = sdf_nav._edt2(np.zeros((8, 8), bool))
    assert (empty >= 1e19).all(), "with no seed every cell is infinitely far"


def test_edt_is_separable_and_symmetric():
    m = np.zeros((20, 20), bool)
    m[10, 10] = True
    d = sdf_nav._edt2(m)
    assert d[10, 4] == d[10, 16] == d[4, 10] == d[16, 10]


# ── inflation is a distance, not a cell count ───────────────────────────────


def test_inflation_is_expressed_in_metres_across_rasters():
    """A tolerance in CELLS silently changes meaning when the raster does. The
    fog stories are ~2 m/cell, so a hard-coded 3 cells was 6 m; on a 1200 m
    city raster at 256 the same 3 cells is 14 m -- wide enough to close every
    street, and measured there it finds no route at all."""
    import dataclasses

    from grl_snam.fog_stories import STORIES, build_scenario

    story = STORIES["city"]
    small = build_scenario(story)
    assert small.inflate_cells == round(story.inflate_m / small.cell_m)

    coarse = dataclasses.replace(story, n=256, bounds=(-600.0, -600.0, 600.0, 600.0))
    big = build_scenario(coarse)
    # the same metric clearance, far fewer cells
    assert big.cell_m > small.cell_m
    assert big.inflate_cells * big.cell_m == pytest.approx(story.inflate_m, abs=big.cell_m)
    assert big.inflate_cells < small.inflate_cells


def test_inflation_never_collapses_to_zero():
    """Rounding a small clearance on a coarse grid must still keep one cell of
    margin, or the route hugs walls."""
    import dataclasses

    from grl_snam.fog_stories import STORIES, build_scenario

    story = dataclasses.replace(STORIES["city"], inflate_m=0.01)
    sc = build_scenario(story)
    assert sc.inflate_cells >= 1
