"""Learned lam head: CoefMLP can OUTPUT the material-reroute strength lam_soft (the
deployable replacement for the fixed --lam-soft dial), added identity-preserving by
add_lam_head, trained through train_bicycle, driven by the Swarm, and exported to a
4-output .cvcnav. Wiring + invariants; the fixed-vs-learned comparison is in the study.
"""

from __future__ import annotations

import os
import struct
import tempfile

import numpy as np
import pytest
import torch

import grl_snam.sdf_nav as sdf_nav
from grl_snam.fog_stories import STORIES, shrunk
from grl_snam.material import city_material_grid
from grl_snam.tools import coef_train
from grl_snam.tools.coef_export import (
    FLAG_LAM_SIGMOID,
    FLAG_SOFTPLUS_LOG_EXPM1,
    write_coef_mlp,
)


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


# ── two-head sigmoid (lam_soft, lam_hard) — .cvcnav format v2 ────────────────────────────────
# The paper's two-head reroute (App A.4, material_nav.py:175-176): lam_soft = lam_soft_max*
# sigmoid(head), lam_hard = lam_hard_max*sigmoid(head). add_lam_heads lifts a trained abg net
# into it basin-preserving; the exporter writes it as format v2, byte-compatible with the C++
# cvc::nav::coef_mlp v2 loader (nav_material_deploy_test.cpp).


def _decode_cvcnav(path):
    """Pure-numpy .cvcnav decoder — an INDEPENDENT reference for the C++ loader/forward, so the
    exporter's bytes can be checked without a pycvc build."""
    with open(path, "rb") as f:
        b = f.read()
    assert b[:4] == b"CVNV"
    off = 4
    ver, flags, in_f, out_f, nl = struct.unpack_from("<IIIII", b, off)
    off += 20
    (ah,) = struct.unpack_from("<Q", b, off)
    off += 8
    layers = []
    for _ in range(nl):
        rows, cols, act = struct.unpack_from("<III", b, off)
        off += 12
        w = np.frombuffer(b, np.float32, rows * cols, off).reshape(rows, cols).copy()
        off += rows * cols * 4
        bb = np.frombuffer(b, np.float32, rows, off).copy()
        off += rows * 4
        layers.append((w, bb, act))
    (obn,) = struct.unpack_from("<I", b, off)
    off += 4
    out_bias = np.frombuffer(b, np.float32, obn, off).copy()
    off += obn * 4
    lam_soft_max = lam_hard_max = None
    if flags & FLAG_LAM_SIGMOID:
        lam_soft_max, lam_hard_max = struct.unpack_from("<ff", b, off)
        off += 8
    (meta_len,) = struct.unpack_from("<I", b, off)
    off += 4 + meta_len
    return dict(
        ver=ver,
        flags=flags,
        in_f=in_f,
        out_f=out_f,
        layers=layers,
        out_bias=out_bias,
        lam_soft_max=lam_soft_max,
        lam_hard_max=lam_hard_max,
        end=off,
        size=len(b),
    )


def _forward_numpy(dec, feats):
    """Run the decoded .cvcnav forward in numpy — the reference the torch CoefMLP must match."""
    a = feats.astype(np.float32)
    for w, bb, act in dec["layers"]:
        a = a @ w.T + bb
        if act == 1:  # SiLU
            a = a * (1.0 / (1.0 + np.exp(-a)))

    def softplus(x):
        return np.where(x > 20.0, x, np.log1p(np.exp(x)))

    fold = np.log(np.expm1(dec["out_bias"]))
    if dec["flags"] & FLAG_LAM_SIGMOID:
        res = np.empty_like(a)
        res[:, :3] = softplus(a[:, :3] + fold[None, :])  # abg keep the softplus fold
        res[:, 3] = dec["lam_soft_max"] * (1.0 / (1.0 + np.exp(-a[:, 3])))
        if dec["out_f"] >= 5:
            res[:, 4] = dec["lam_hard_max"] * (1.0 / (1.0 + np.exp(-a[:, 4])))
        return res
    return softplus(a + fold[None, :])


def test_add_lam_heads_identity_flags_and_init():
    field, grid = _scene()
    mf = grid.field()
    o = torch.tensor([[0.1, 0.0], [0.3, -0.2]])
    goal = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    feat = sdf_nav.coef_feats(field, o, goal, material=mf)

    risk = sdf_nav.add_risk_feature(sdf_nav.CoefMLP())
    two = sdf_nav.add_lam_heads(risk, soft_init=0.4, hard_init=1.0)
    assert two.use_lam and two.use_lam_hard and two.lam_sigmoid
    assert two.out_dim == 5 and two.use_risk and two.in_dim == risk.in_dim
    assert two.lam_soft_max == 5.0 and two.lam_hard_max == 10.0

    a0, b0, g0 = risk(feat)
    a1, b1, g1, ls, lh = two.coeffs_and_lam(feat)
    assert torch.allclose(a0, a1, atol=1e-6)  # abg bit-identical after lifting
    assert torch.allclose(b0, b1, atol=1e-6)
    assert torch.allclose(g0, g1, atol=1e-6)
    assert torch.allclose(ls, torch.tensor(0.4), atol=1e-5)  # lam_soft starts at soft_init
    assert torch.allclose(lh, torch.tensor(1.0), atol=1e-5)  # lam_hard starts at hard_init
    assert len(two(feat)) == 3  # forward() still returns abg

    with pytest.raises(ValueError, match="already has a lam head"):
        sdf_nav.add_lam_heads(two)


def test_two_head_sigmoid_bounds_and_out_of_range_init():
    # sigmoid bounds: lam_soft in [0,5], lam_hard in [0,10] for any input.
    two = sdf_nav.add_lam_heads(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()))
    with torch.no_grad():  # random extreme final-layer weights on the lam rows
        two.net[-1].weight[3:].uniform_(-50, 50)
    feat = torch.randn(500, two.in_dim)
    _, _, _, ls, lh = two.coeffs_and_lam(feat)
    assert (ls >= 0).all() and (ls <= 5.0 + 1e-4).all()
    assert (lh >= 0).all() and (lh <= 10.0 + 1e-4).all()
    # init strictly inside (0, max) — logit is +/-inf at the ends.
    with pytest.raises(ValueError, match="0 < .* < its max"):
        sdf_nav.CoefMLP(use_lam=True, use_lam_hard=True, lam_hard_init=0.0)
    with pytest.raises(ValueError, match="0 < .* < its max"):
        sdf_nav.CoefMLP(use_lam=True, use_lam_hard=True, lam_hard_init=10.0)
    with pytest.raises(ValueError, match="requires use_lam"):
        sdf_nav.CoefMLP(use_lam=False, use_lam_hard=True)


def test_two_head_export_v2_layout_and_forward_parity(tmp_path):
    torch.manual_seed(1)
    two = sdf_nav.add_lam_heads(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()), 2.0, 4.0)
    with torch.no_grad():  # make the lam heads position-dependent so parity is non-trivial
        two.net[-1].weight[3:].normal_(0, 0.5)
    p = str(tmp_path / "two_head.cvcnav")
    write_coef_mlp(two, p)

    dec = _decode_cvcnav(p)
    assert dec["ver"] == 2
    assert dec["flags"] & FLAG_LAM_SIGMOID
    assert dec["flags"] & FLAG_SOFTPLUS_LOG_EXPM1
    assert dec["in_f"] == two.in_dim and dec["out_f"] == 5
    assert dec["out_bias"].shape[0] == 3  # abg only
    assert dec["lam_soft_max"] == pytest.approx(5.0)
    assert dec["lam_hard_max"] == pytest.approx(10.0)
    assert dec["end"] == dec["size"]  # trailer + meta consume the file exactly

    # byte+numeric parity: the numpy decode of the exported blob reproduces the torch forward.
    feats = np.random.default_rng(0).standard_normal((256, two.in_dim)).astype(np.float32)
    with torch.no_grad():
        out = two._outputs(torch.from_numpy(feats)).numpy()
    ref = _forward_numpy(dec, feats)
    assert np.allclose(out, ref, rtol=1e-4, atol=1e-5), np.abs(out - ref).max()
    # and the exported lam columns are within the sigmoid ceilings.
    assert (ref[:, 3] >= 0).all() and (ref[:, 3] <= 5.0 + 1e-4).all()
    assert (ref[:, 4] >= 0).all() and (ref[:, 4] <= 10.0 + 1e-4).all()


def test_single_softplus_lam_still_exports_v1(tmp_path):
    # BACK-COMPAT: a single-softplus lam net (add_lam_head) stays format v1, no sigmoid flag,
    # out_bias_len == out_features — byte-unchanged from before the v2 bump.
    lam = sdf_nav.add_lam_head(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()), lam_init=0.5)
    assert not getattr(lam, "lam_sigmoid", False) and not lam.use_lam_hard
    p = str(tmp_path / "v1.cvcnav")
    write_coef_mlp(lam, p)
    dec = _decode_cvcnav(p)
    assert dec["ver"] == 1
    assert not (dec["flags"] & FLAG_LAM_SIGMOID)
    assert dec["out_f"] == 4 and dec["out_bias"].shape[0] == 4  # all-softplus, one bias per output
    assert dec["lam_soft_max"] is None and dec["end"] == dec["size"]


def test_train_bicycle_learns_two_head_lam():
    _, grid = _scene()
    model = sdf_nav.add_lam_heads(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()), 0.4, 1.0)
    m = coef_train.train_bicycle(
        steps=12, horizon=12, n=48, grid=64, seed=1, model=model, material=grid, w_risk=6.0
    )
    field, g2 = _scene()
    feat = sdf_nav.coef_feats(field, torch.zeros(4, 2), torch.ones(4, 2), material=g2.field())
    _, _, _, ls, lh = m.coeffs_and_lam(feat)
    # both heads stay within their sigmoid ceilings after training
    assert (ls >= 0).all() and (ls <= 5.0).all()
    assert (lh >= 0).all() and (lh <= 10.0).all()


def test_cli_rejects_learned_lam_hard_with_nonpositive_init():
    with pytest.raises(SystemExit, match="lam-hard-init > 0"):
        coef_train.main(
            [
                "--rollout",
                "bicycle",
                "--w-risk",
                "1",
                "--learned-lam-hard",
                "--lam-soft",
                "0.4",
                "--lam-hard-init",
                "0",
                "--steps",
                "1",
            ]
        )
