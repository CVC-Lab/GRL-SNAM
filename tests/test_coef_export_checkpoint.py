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
