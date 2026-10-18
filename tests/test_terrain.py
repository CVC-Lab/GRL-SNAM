"""The terrain planes' Python twin (grl_snam.terrain) against libcvc's terrain_build.

With a pycvc carrying ``nav_terrain_build`` (libcvc's P5 terrain plane as numpy arrays), the
twin's slope, grade cost and infeasible cells equal the C++ planes bit for bit. Without it,
the analytic checks still run.
"""

import math

import numpy as np
import pytest

from grl_snam.terrain import climb_weight, terrain_limit, terrain_material, terrain_planes


def _hill(n=61, peak=12.0, width=9.0):
    x = np.arange(n, dtype=np.float64) - n // 2
    return (peak * np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (2 * width**2))).astype(np.float32)


def test_a_plane_has_its_slope_and_a_cost_below_one():
    x = np.arange(21, dtype=np.float64)
    z = np.tile(0.2 * x, (11, 1))  # a 20% grade rising toward +x, 1 m cells
    p = terrain_planes(z, 1.0, max_grade=0.6)
    assert np.allclose(p.slope_x, 0.2) and np.allclose(p.slope_y, 0.0)
    expect = (0.2 / math.sqrt(1.04)) / (0.6 / math.sqrt(1.36))
    assert np.allclose(p.grade_cost, expect)
    assert not p.infeasible.any() and p.limit == 0.6


def test_a_tip_ratio_lowers_the_limit_and_steep_cells_are_infeasible():
    assert terrain_limit(0.6, 1.23, 0.8) == 0.6  # the tip limit, 0.984, is above the climb
    assert terrain_limit(1.2, 1.23, 0.8) == pytest.approx(0.984)
    assert terrain_limit(0.6, 0.5, 0.8) == pytest.approx(0.4)
    x = np.arange(21, dtype=np.float64)
    z = np.tile(0.5 * x, (11, 1))  # 50%: fine under 0.6, past a 0.4 tip limit
    assert not terrain_planes(z, 1.0, max_grade=0.6).infeasible.any()
    tip = terrain_planes(z, 1.0, max_grade=0.6, tip_ratio=0.5, tip_margin=0.8)
    assert tip.infeasible.all() and np.all(tip.grade_cost == 1.0)
    with pytest.raises(ValueError):
        terrain_planes(z, 0.0)


def test_the_planes_are_libcvcs_bit_for_bit():
    pycvc = pytest.importorskip("pycvc")
    if not hasattr(pycvc, "nav_terrain_build"):
        pytest.skip("pycvc lacks nav_terrain_build (libcvc P5's terrain plane)")
    z = _hill()
    for cfg in ({"max_grade": 0.6}, {"max_grade": 0.6, "tip_ratio": 1.23, "tip_margin": 0.8},
                {"max_grade": 0.3}):  # fmt: skip
        c = pycvc.nav_terrain_build(z, 2.0, dict(cfg))
        p = terrain_planes(z, 2.0, **cfg)
        assert np.array_equal(p.slope_x, c["slope"][0]), cfg
        assert np.array_equal(p.slope_y, c["slope"][1]), cfg
        assert np.array_equal(p.grade_cost, c["grade_cost"]), cfg
        assert np.array_equal(p.infeasible, c["infeasible"].astype(bool)), cfg
        assert p.limit == c["limit"], cfg


def test_the_material_grid_pushes_off_the_hill():
    pytest.importorskip("torch")
    import torch

    z = _hill()
    bounds = (-30.0, -30.0, 30.0, 30.0)
    grid, planes = terrain_material(z, bounds, max_grade=0.6)
    assert grid.hard.sum() == planes.infeasible.sum()
    field = grid.field()
    # On the hill's east flank the grade cost rises toward the peak (west): its gradient
    # points west, so the soft force -grad r~_g pushes east, downhill.
    on = torch.tensor([[12.0, 0.0]])
    r, _phi, grad_r, _grad_phi = field.sample(on)
    assert float(r[0]) > 0.0
    assert float(grad_r[0, 0]) < 0.0, "the cost rises toward the peak"


def test_climb_weight():
    assert climb_weight(0.015, 1.0) == pytest.approx(1.0 / 0.015)
    assert climb_weight(0.015, 2.0) == pytest.approx(0.5 / 0.015)
