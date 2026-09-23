"""Learned lam head: CoefMLP can OUTPUT the material-reroute strength lam_soft (the
deployable replacement for the fixed --lam-soft dial), added identity-preserving by
add_lam_head, trained through train_bicycle, driven by the Swarm, and exported to a
4-output .cvcnav. Wiring + invariants; the fixed-vs-learned comparison is in the study.
"""

from __future__ import annotations

import os
import struct
import tempfile

import pytest
import torch

import sdf_nav
from grl_snam.fog_stories import STORIES, shrunk
from grl_snam.material import city_material_grid
from grl_snam.tools import coef_train
from grl_snam.tools.coef_export import write_coef_mlp


def _scene():
    story = shrunk(STORIES["city"], n=64, max_steps=100)
    truth = story.truth_grid()
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(truth, story.bounds, meta["scale"])
    field = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    grid, _ = city_material_grid(truth, story.bounds, meta["center"], meta["scale"], seed=0)
    return field, grid


def test_add_lam_head_identity_and_flags():
    field, grid = _scene()
    mf = grid.field()
    o = torch.tensor([[0.1, 0.0], [0.3, -0.2]])
    goal = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    feat = sdf_nav.coef_feats(field, o, goal, material=mf)

    risk = sdf_nav.add_risk_feature(sdf_nav.CoefMLP())
    lam = sdf_nav.add_lam_head(risk, lam_init=0.4)
    assert lam.use_lam and lam.out_dim == 4 and lam.use_risk and lam.in_dim == risk.in_dim

    a0, b0, g0 = risk(feat)
    a1, b1, g1, ls = lam.coeffs_and_lam(feat)
    assert torch.allclose(a0, a1, atol=1e-6)  # abg bit-identical after adding the head
    assert torch.allclose(b0, b1, atol=1e-6)
    assert torch.allclose(g0, g1, atol=1e-6)
    assert torch.allclose(ls, torch.tensor(0.4), atol=1e-5)  # lam starts at lam_init
    assert len(lam(feat)) == 3  # forward() still returns abg (backward compatible)

    with pytest.raises(ValueError, match="already has a lam head"):
        sdf_nav.add_lam_head(lam)


def test_nonpositive_lam_init_is_rejected():
    # lam_init <= 0 would fold to softplus bias -inf (dead, zero-gradient lam) or NaN — guard it.
    with pytest.raises(ValueError, match="lam_init > 0"):
        sdf_nav.CoefMLP(use_lam=True, lam_init=0.0)
    with pytest.raises(ValueError, match="lam_init > 0"):
        sdf_nav.CoefMLP(use_lam=True, lam_init=-0.1)
    with pytest.raises(ValueError, match="lam_init > 0"):
        sdf_nav.add_lam_head(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()), lam_init=0.0)


def test_cli_rejects_learned_lam_with_nonpositive_init():
    with pytest.raises(SystemExit, match="lam-soft > 0"):
        coef_train.main(
            [
                "--rollout",
                "bicycle",
                "--w-risk",
                "1",
                "--learned-lam",
                "--lam-soft",
                "0",
                "--steps",
                "1",
            ]
        )


def test_coeffs_and_lam_requires_head():
    field, grid = _scene()
    feat = sdf_nav.coef_feats(field, torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(RuntimeError, match="no lam head"):
        sdf_nav.CoefMLP().coeffs_and_lam(feat)


def test_full_out_bias_and_export_has_four_outputs():
    lam = sdf_nav.add_lam_head(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()), lam_init=0.5)
    ob = lam.full_out_bias()
    assert ob.shape[0] == 4
    assert float(ob[-1]) == pytest.approx(0.5)  # raw lam_init appended

    p = tempfile.mktemp(suffix=".cvcnav")
    try:
        write_coef_mlp(lam, p)
        with open(p, "rb") as f:
            assert f.read(4) == b"CVNV"
            _ver, _flags, in_f, out_f, _nl = struct.unpack("<IIIII", f.read(20))
        assert out_f == 4  # the format carries the lam output; C++ folds log(expm1) uniformly
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_train_bicycle_learns_lam():
    _, grid = _scene()
    model = sdf_nav.add_lam_head(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()), lam_init=0.4)
    m = coef_train.train_bicycle(
        steps=12, horizon=12, n=48, grid=64, seed=1, model=model, material=grid, w_risk=6.0
    )
    field, g2 = _scene()
    feat = sdf_nav.coef_feats(field, torch.zeros(4, 2), torch.ones(4, 2), material=g2.field())
    _, _, _, lam = m.coeffs_and_lam(feat)
    # after training lam has moved off the constant init (learned a position-dependent value)
    assert lam.std().item() > 1e-5 or abs(lam.mean().item() - 0.4) > 1e-3


def test_lam_net_requires_material():
    # a lam-ONLY net (no risk feature) exercises the lam-specific guard; a risk+lam net
    # would trip the risk guard first (both correctly demand material).
    with pytest.raises(ValueError, match="lam-head model needs material"):
        coef_train.train_bicycle(
            steps=1, horizon=2, n=4, grid=32, model=sdf_nav.add_lam_head(sdf_nav.CoefMLP())
        )


def test_lam_net_scores_via_swarm():
    from grl_snam.scorecard import NavScorecard
    from grl_snam.tools.scorecard_eval import evaluate

    lam = sdf_nav.add_lam_head(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()))
    card = evaluate(lam, scenes=2, agents=16, steps=80, material=True)
    assert isinstance(card, NavScorecard)
    assert 0.0 <= card.arrival_rate <= 1.0
    with pytest.raises(ValueError, match="use_risk/use_lam net needs material"):
        evaluate(lam, scenes=1, agents=8, steps=20, material=False)
