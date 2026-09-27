"""Deployable lam-net integration: CLI-train a risk/lam coef_mlp (the ``coef-train --rollout
bicycle --w-risk --learned-lam`` path) to a ``.cvcnav``, confirm the shipped 6-in-risk-only+lam
layout survives export, and reload it torch-free — the Python train -> export -> reload half of the
deployable path. The C++ twin (libcvc ``NavMaterialDeploy.ShippedRiskOnlyLamNetRoundTripsAndDrives``
+ ``NavCoefTrain.TrainThenSaveReloadIntegration``) covers the load+DRIVE contract; this pins that the
CLI actually produces a loadable risk/lam artifact. Mirrors libcvc's train->save->reload integration
test on the grl-snam side.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the CLI training path is torch

import grl_snam.tools.coef_export as CE  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.tools import coef_train  # noqa: E402


def _read_header(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"CVNV", "not a CVNV coef_mlp blob"
        _ver, flags, in_f, out_f, _nl = struct.unpack("<IIIII", f.read(20))
    return flags, in_f, out_f


def test_cli_lam_net_trains_exports_and_reloads(tmp_path):
    out = tmp_path / "lam.cvcnav"
    # The deployable CLI path verbatim (coef_train.py --learned-lam branch). Small/fast fixture
    # matching test_lam_head's proven lam sizes (steps=12, horizon=12, n=48, grid=64, w_risk=6).
    coef_train.main(
        [
            "--rollout",
            "bicycle",
            "--w-risk",
            "6",
            "--learned-lam",
            "--lam-soft",
            "0.4",
            "--steps",
            "12",
            "--horizon",
            "12",
            "--n",
            "48",
            "--grid",
            "64",
            "--seed",
            "1",
            "--out",
            str(out),
        ]
    )
    assert out.exists() and out.stat().st_size > 0

    # (1) The SHIPPED risk-only+lam layout survives export: 6-in (base5 + risk, NO grip), 4-out
    # (learned lam head), risk flag set + mu flag clear. Same contract libcvc's shipped-blob deploy
    # test pins on the C++ side.
    flags, in_f, out_f = _read_header(out)
    assert in_f == 6, f"expected 6-in risk net, got {in_f}"
    assert out_f == 4, f"expected 4-out lam head, got {out_f}"
    assert flags & CE.FLAG_FEAT_RISK, "risk feature flag not set"
    assert not (flags & CE.FLAG_FEAT_MU), "grip (mu) flag set — expected risk-only"

    # (2) Torch-free reload: the trained 4-output net loads via the C++ coef_mlp and forwards to
    # (N,4) with a finite learned-lam (4th) column — the reload half of the deployable contract,
    # on the actual CLI-produced artifact (not a hand-built net).
    if not nav_native.HAS_COEF_MLP:
        pytest.skip("pycvc lacks nav_coef_mlp_forward — cannot reload .cvcnav torch-free")
    feats = np.random.default_rng(0).uniform(-1.0, 1.0, (16, 6)).astype(np.float32)
    coef = nav_native.coef_mlp_forward(str(out), feats)
    assert coef.shape == (16, 4), f"reloaded lam net forwarded to {coef.shape}, expected (16, 4)"
    assert np.all(np.isfinite(coef)), "reloaded net produced non-finite coefficients"
    assert np.all(coef[:, 3] >= 0.0), "learned lam (4th output, softplus) must be >= 0"
