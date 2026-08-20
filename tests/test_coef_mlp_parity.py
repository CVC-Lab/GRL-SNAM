"""The torch-free C++ coefficient policy (cvc::nav::coef_mlp) vs torch CoefMLP.

`coef_export.write_coef_mlp` serializes a trained `sdf_nav.CoefMLP` to the
versioned `.cvcnav` file; `nav_native.coef_mlp_forward` runs the C++ forward from
it. This checks float-equivalence to `CoefMLP.forward` (the roadmap P2 contract,
rtol 1e-4) including a row that drives the softplus past its linear-tail
threshold, plus the format round-trip and the arch_hash guard.
"""

import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import sdf_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.tools import coef_export  # noqa: E402

pytestmark = pytest.mark.skipif(
    not nav_native.HAS_COEF_MLP, reason="pycvc build lacks nav_coef_mlp_forward"
)


def _model(seed=0):
    torch.manual_seed(seed)
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _torch_forward(m, feats):
    with torch.no_grad():
        a, b, g = m(torch.from_numpy(feats))
    return np.stack([a.numpy(), b.numpy(), g.numpy()], axis=1)  # (N,3)


def test_coef_mlp_matches_torch(tmp_path):
    m = _model()
    path = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(path))
    rng = np.random.default_rng(0)
    # ordinary features + a few extreme rows that push net output large (softplus
    # linear tail) and very negative (softplus -> ~0).
    feats = rng.standard_normal((4000, 5)).astype(np.float32)
    feats[0] = [50.0, 50.0, 50.0, 50.0, 50.0]
    feats[1] = [-50.0, -50.0, -50.0, -50.0, -50.0]
    ref = _torch_forward(m, feats)
    got = nav_native.coef_mlp_forward(str(path), feats)
    assert got.shape == (4000, 3)
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5), np.abs(got - ref).max()
    # the extreme rows still hold (softplus tail + floor).
    assert np.all(np.isfinite(got))


def test_roundtrip_header_and_arch_hash(tmp_path):
    m = _model()
    path = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(path))
    raw = path.read_bytes()
    assert raw[:4] == b"CVNV"
    fmt, flags, in_f, out_f, nlayers = struct.unpack("<IIIII", raw[4:24])
    (arch,) = struct.unpack("<Q", raw[24:32])
    assert fmt == coef_export.FORMAT_VERSION
    assert (in_f, out_f) == (5, 3)
    assert nlayers == 3  # Linear(5,64), Linear(64,64), Linear(64,3)
    # arch_hash recomputed from the shapes matches what was written.
    shape_act = [64, 5, 1, 64, 64, 1, 3, 64, 0]
    assert coef_export.arch_hash(5, 3, shape_act) == arch


def test_stale_arch_hash_is_rejected(tmp_path):
    m = _model()
    path = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(path))
    raw = bytearray(path.read_bytes())
    raw[24] ^= 0xFF  # corrupt the arch_hash
    bad = tmp_path / "bad.cvcnav"
    bad.write_bytes(raw)
    with pytest.raises(Exception):
        nav_native.coef_mlp_forward(str(bad), np.zeros((1, 5), np.float32))
