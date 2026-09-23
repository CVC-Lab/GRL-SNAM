"""Scorecard-driven curriculum: the RegionCurriculum sampler + its wiring into
train_bicycle. Deterministic invariants (weights, floor, cap, cold start, difficulty
attribution); whether it improves a full run is a statistical question for the study.
"""

from __future__ import annotations

import numpy as np
import pytest

from grl_snam.curriculum import RegionCurriculum, build_city_curriculum
from grl_snam.tools import coef_train


def _grid_curriculum(bins=4, eps=0.15, cap=6.0):
    # a fully-free 40x40 scene so every bin is populated
    ny = nx = 40
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return RegionCurriculum(
        rows.ravel(),
        cols.ravel(),
        (ny, nx),
        bounds=(0.0, 0.0, float(nx - 1), float(ny - 1)),
        center=(19.5, 19.5),
        scale=1.0,
        bins=bins,
        eps=eps,
        cap=cap,
    )


def test_cold_start_is_uniform_and_weights_normalize():
    c = _grid_curriculum(bins=4)
    assert c.B == 16
    assert np.allclose(c.weights, 1.0 / c.B)  # uniform before any update
    assert abs(c.weights.sum() - 1.0) < 1e-12


def test_sample_starts_returns_valid_coords_and_bins():
    c = _grid_curriculum(bins=4)
    rng = np.random.default_rng(0)
    starts, bins = c.sample_starts(64, rng)
    assert starts.shape == (64, 2) and starts.dtype == np.float32
    assert bins.shape == (64,)
    assert (bins >= 0).all() and (bins < c.B).all()


def test_update_shifts_weight_toward_hard_bins_with_floor_and_cap():
    c = _grid_curriculum(bins=4, eps=0.1, cap=6.0)
    uniform = 1.0 / c.B
    # bin 0 is very hard, everything else easy
    bin_ids = np.arange(c.B).repeat(4)
    difficulty = np.where(bin_ids == 0, 5.0, 0.01)
    for _ in range(10):  # let the EMA settle
        c.update(bin_ids, difficulty)
    assert c.weights[0] > uniform * 1.5  # hard bin up-weighted
    assert (c.weights > 0).all()  # uniform floor -> nothing starves
    assert c.weights.max() <= c.cap * uniform + 1e-9  # cap respected
    assert abs(c.weights.sum() - 1.0) < 1e-9


def test_weights_bias_sampling_frequency():
    c = _grid_curriculum(bins=3, eps=0.05)
    bin_ids = np.arange(c.B).repeat(8)
    difficulty = np.where(bin_ids == 5, 10.0, 0.01)  # bin 5 dominates
    for _ in range(15):
        c.update(bin_ids, difficulty)
    rng = np.random.default_rng(1)
    _, bins = c.sample_starts(4000, rng)
    freq = np.bincount(bins, minlength=c.B) / 4000.0
    assert freq[5] == freq.max()  # the hard bin is sampled most
    assert freq[5] > 1.5 / c.B  # ... noticeably above uniform


def test_empty_bins_are_dropped():
    # only the bottom-left quadrant is free -> bins over empty regions must not exist
    ny = nx = 40
    rr, cc = np.meshgrid(np.arange(0, 20), np.arange(0, 20), indexing="ij")
    c = RegionCurriculum(
        rr.ravel(),
        cc.ravel(),
        (ny, nx),
        bounds=(0.0, 0.0, 39.0, 39.0),
        center=(19.5, 19.5),
        scale=1.0,
        bins=4,
    )
    assert 0 < c.B < 16  # some bins empty -> compacted
    rng = np.random.default_rng(0)
    starts, bins = c.sample_starts(50, rng)
    assert (bins < c.B).all()


def test_build_city_curriculum_aligns_with_trainer_scene():
    c = build_city_curriculum(64, bins=5)
    assert c.B > 0
    rng = np.random.default_rng(0)
    starts, _ = c.sample_starts(32, rng)
    assert starts.shape == (32, 2)


def test_train_bicycle_curriculum_runs_and_adapts():
    c = build_city_curriculum(64, bins=5)
    before = c.weights.copy()
    m = coef_train.train_bicycle(steps=6, horizon=12, n=48, grid=64, seed=1, curriculum=c)
    assert m is not None
    assert not np.allclose(c.weights, before)  # the curriculum learned from the batches
    assert abs(c.weights.sum() - 1.0) < 1e-9


def test_curriculum_none_is_the_default_uniform_path():
    # sanity: the default path (no curriculum) still trains and reports the 3-tuple
    m = coef_train.train_bicycle(steps=2, horizon=8, n=24, grid=64, seed=1)
    assert len(m.last_loss_terms) == 3


def test_eps_out_of_range_rejected():
    with pytest.raises(ValueError, match="eps"):
        _grid_curriculum(eps=1.5)
