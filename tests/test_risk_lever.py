"""The terrain-risk steering lever: the risk-lookahead feature, the add_risk_feature
widener, and the opt-in w_risk exposure loss in train_bicycle.

These pin WIRING and INVARIANTS (deterministic). Whether a w_risk run actually LOWERS
risk exposure is a statistical question answered by the measurement study (see
docs/NAV_STATS.md), not a unit test — the same way the w_coll operating point was
established by an offline table, not an assertion.
"""

from __future__ import annotations

import pytest
import torch

import sdf_nav
from grl_snam.fog_stories import STORIES, shrunk
from grl_snam.material import city_material_grid
from grl_snam.tools import coef_train


def _scene():
    story = shrunk(STORIES["city"], n=64, max_steps=100)
    truth = story.truth_grid()
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(truth, story.bounds, meta["scale"])
    field = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    grid, idr = city_material_grid(truth, story.bounds, meta["center"], meta["scale"], seed=0)
    return field, grid


def test_risk_feature_appends_and_default_is_byte_identical():
    field, grid = _scene()
    mf = grid.field()
    o = torch.tensor([[0.1, 0.0], [0.3, -0.2]], dtype=torch.float32)
    goal = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)

    base = sdf_nav.coef_feats(field, o, goal)
    withrisk = sdf_nav.coef_feats(field, o, goal, material=mf)
    assert base.shape[1] == 5 and withrisk.shape[1] == 6
    assert torch.equal(base, withrisk[:, :5])  # leading columns bit-identical
    assert (withrisk[:, 5] >= 0).all() and (withrisk[:, 5] <= 1).all()


def test_risk_feature_is_max_ahead_and_differentiable():
    field, grid = _scene()
    mf = grid.field()
    # a point whose carrot points into higher risk should read the risk AHEAD (max),
    # not the (possibly clear) risk underfoot.
    o = torch.zeros(1, 2, requires_grad=True)
    # sample risk along +x vs -x; the feature is the MAX along the carrot ray
    fwd = sdf_nav.coef_feats(field, o, torch.tensor([[2.0, 0.0]]), material=mf)[0, 5]
    here = float(mf.sample(o.detach())[0][0])
    assert fwd.detach().item() >= here - 1e-6  # never below underfoot (max includes o)
    fwd.backward()
    assert o.grad is not None and bool((o.grad.abs() > 0).any())  # differentiable wrt pose


def test_add_risk_feature_output_identical_and_flags():
    field, grid = _scene()
    mf = grid.field()
    o = torch.tensor([[0.1, 0.0], [0.3, -0.2]])
    goal = torch.tensor([[1.0, 0.0], [0.5, 0.5]])

    m5 = sdf_nav.CoefMLP()
    mr = sdf_nav.add_risk_feature(m5)
    assert mr.in_dim == 6 and mr.use_risk and not mr.use_mu
    a5, b5, g5 = m5(sdf_nav.coef_feats(field, o, goal))
    ar, br, gr = mr(sdf_nav.coef_feats(field, o, goal, material=mf))
    assert torch.allclose(a5, ar, atol=1e-6)
    assert torch.allclose(b5, br, atol=1e-6)
    assert torch.allclose(g5, gr, atol=1e-6)

    # stacks on top of the grip widener -> 7 features, both flags set
    mmr = sdf_nav.add_risk_feature(sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP()))
    assert mmr.in_dim == 7 and mmr.use_mu and mmr.use_risk
    with pytest.raises(ValueError, match="already has the risk feature"):
        sdf_nav.add_risk_feature(mmr)


def test_train_bicycle_risk_term_wiring():
    _, grid = _scene()
    mk = lambda: sdf_nav.add_risk_feature(sdf_nav.CoefMLP())  # noqa: E731

    m0 = coef_train.train_bicycle(
        steps=3,
        horizon=10,
        n=32,
        grid=64,
        seed=1,
        model=mk(),
        material=grid,
        w_risk=0.0,
        lam_soft=0.4,
    )
    assert m0.last_loss_terms[2] == 0.0  # w_risk=0 -> the risk summand is inert

    m1 = coef_train.train_bicycle(
        steps=3,
        horizon=10,
        n=32,
        grid=64,
        seed=1,
        model=mk(),
        material=grid,
        w_risk=5.0,
        lam_soft=0.4,
    )
    assert m1.last_loss_terms[2] > 0.0  # w_risk>0 + material -> real, nonzero contribution


def test_no_material_default_is_byte_identical_three_tuple():
    m = coef_train.train_bicycle(steps=2, horizon=8, n=24, grid=64, seed=1)
    assert len(m.last_loss_terms) == 3
    assert m.last_loss_terms[2] == 0.0  # no material -> risk inert, geometry-only objective


def test_risk_model_requires_material():
    with pytest.raises(ValueError, match="risk model needs material"):
        coef_train.train_bicycle(
            steps=1, horizon=2, n=4, grid=32, model=sdf_nav.add_risk_feature(sdf_nav.CoefMLP())
        )
