"""coef_export CLI unwraps every checkpoint format grl-snam writes.

``grl-snam train`` (tools/train.py) saves ``{"model_state_dict", "meta"}``; older
checkpoints saved ``{"model": ...}``; a bare state_dict has neither key. The
export CLI must handle all three. It used to know only ``"model"``, so it fed the
whole ``{"model_state_dict", "meta"}`` dict to ``load_state_dict`` and crashed on
every ``grl-snam train`` output — the standard train->export->deploy path.
"""

from __future__ import annotations

import torch

import sdf_nav
from grl_snam.tools import coef_export


def _export_and_check(tmp_path, ckpt_obj, name):
    ckpt = tmp_path / f"{name}.pt"
    torch.save(ckpt_obj, ckpt)
    out = tmp_path / f"{name}.cvcnav"
    coef_export.main([str(ckpt), str(out)])
    assert out.read_bytes()[:4] == b"CVNV"


def test_export_from_train_checkpoint(tmp_path):
    """The format `grl-snam train` actually writes (the regression)."""
    m = sdf_nav.CoefMLP()
    _export_and_check(tmp_path, {"model_state_dict": m.state_dict(), "meta": {"steps": 1}}, "train")


def test_export_from_legacy_model_key(tmp_path):
    m = sdf_nav.CoefMLP()
    _export_and_check(tmp_path, {"model": m.state_dict()}, "legacy")


def test_export_from_bare_state_dict(tmp_path):
    m = sdf_nav.CoefMLP()
    _export_and_check(tmp_path, m.state_dict(), "bare")


def _read_header(path):
    import struct

    with open(path, "rb") as f:
        assert f.read(4) == b"CVNV"
        ver, flags, in_f, out_f, nl = struct.unpack("<IIIII", f.read(20))
    return flags, in_f, out_f


def test_exported_flags_encode_the_feature_layout(tmp_path):
    # The .cvcnav flags byte must tell the C++ drive which feature columns to build (a bare
    # 6-in net is grip; the risk flag disambiguates), matching cvc::nav coef_mlp kFlagFeat*.
    F = coef_export

    def flags_of(model):
        p = tmp_path / "m.cvcnav"
        coef_export.write_coef_mlp(model, str(p))
        fl, in_f, out_f = _read_header(str(p))
        return fl, in_f, out_f

    base = flags_of(sdf_nav.CoefMLP())
    assert base[0] == F.FLAG_SOFTPLUS_LOG_EXPM1  # softplus only, no feature flags
    assert base[1] == 5

    mu = flags_of(sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP()))
    assert mu[0] & F.FLAG_FEAT_MU and not (mu[0] & F.FLAG_FEAT_RISK)
    assert mu[1] == 6

    risk = flags_of(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()))
    assert risk[0] & F.FLAG_FEAT_RISK and not (risk[0] & F.FLAG_FEAT_MU)
    assert risk[1] == 6

    lam = flags_of(sdf_nav.add_lam_head(sdf_nav.add_risk_feature(sdf_nav.CoefMLP())))
    assert lam[0] & F.FLAG_FEAT_RISK  # risk feature flagged
    assert lam[2] == 4  # lam head signalled by out_features, no flag needed

    mu_risk = flags_of(sdf_nav.add_risk_feature(sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP())))
    assert mu_risk[0] & F.FLAG_FEAT_MU and mu_risk[0] & F.FLAG_FEAT_RISK
    assert mu_risk[1] == 7
