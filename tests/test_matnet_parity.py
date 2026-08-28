"""Parity for the torch-free learned material coefficient net (CoefEnergyNetMaterial).

Three layers, mirroring the tier map:
  1. matnet_forward_numpy (the explicit math-path reference) vs torch, with the
     fused-attention fast path DISABLED — proves the numpy reference is a valid
     oracle. Runs in CI (torch only).
  2. .cvcnm round-trip: write -> read -> arch_hash stable, tensors present.
  3. pycvc nav_matnet_forward (cvc::nav::coef_energy_net) vs matnet_forward_numpy
     — the C++ FLOAT parity (rtol 1e-4). Skips until a matnet-capable pycvc
     ships in the closure (same pattern as test_material_parity).

The numpy reference — not torch — is the parity oracle for the C++: torch's
TransformerEncoder in eval/no-grad may dispatch to a fused attention fast path
(BetterTransformer / flash / NestedTensor) that skips padded tokens and rounds
differently. Layer 1 pins numpy==torch(math); the C++ is validated against numpy.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import material_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.tools.matnet_export import (  # noqa: E402
    matnet_forward_numpy,
    read_matnet,
    write_matnet,
)


@contextlib.contextmanager
def _math_attention():
    """Force torch's MATH scaled-dot-product-attention (no flash / mem-efficient /
    nested fast paths) so the reference is the explicit math path."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            yield
        return
    except Exception:
        pass
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            yield
    except Exception:
        yield


def _model(seed=0):
    torch.manual_seed(seed)
    m = material_nav.CoefEnergyNetMaterial()
    m.eval()
    m.fuser.enable_nested_tensor = False
    return m


def _case(rng, n_max=8, p=32):
    n = int(rng.integers(0, n_max))
    obs = rng.standard_normal((n, 6)).astype(np.float32) if n else np.zeros((0, 6), np.float32)
    mask = (rng.random(n) > 0.3).astype(bool) if n else np.zeros(0, bool)
    goal = rng.standard_normal(4).astype(np.float32)
    patch = rng.random((2, p, p)).astype(np.float32)
    return n, obs, mask, goal, patch


def _torch_forward(m, obs, mask, goal, patch):
    n = obs.shape[0]
    with torch.no_grad(), _math_attention():
        of = torch.from_numpy(obs)[None]
        mk = torch.from_numpy(mask)[None] if n else torch.zeros(1, 0, dtype=torch.bool)
        gf = torch.from_numpy(goal)[None]
        rp = torch.from_numpy(patch)[None]
        a, b, g, ls, lh, ml = m(of, mk, gf, rp)
    return (
        a[0].numpy() if n else np.zeros(0, np.float32),
        float(b[0]),
        float(g[0]),
        float(ls[0]),
        float(lh[0]),
        float(ml[0]),
    )


def test_numpy_reference_matches_torch_math_path(tmp_path):
    m = _model()
    path = tmp_path / "m.cvcnm"
    write_matnet(m, str(path))
    mn = read_matnet(str(path))
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(40):
        n, obs, mask, goal, patch = _case(rng)
        ta, tb, tg, tls, tlh, tml = _torch_forward(m, obs, mask, goal, patch)
        na, nb, ng, nls, nlh, nml = matnet_forward_numpy(mn, obs, mask, goal, patch)
        if n:
            worst = max(worst, float(np.abs(na - ta).max()))
        worst = max(
            worst, abs(nb - tb), abs(ng - tg), abs(nls - tls), abs(nlh - tlh), abs(nml - tml)
        )
    assert worst < 1e-4, f"numpy reference vs torch(math): worst {worst:.2e}"


def test_cvcnm_roundtrip_and_arch_hash_stable(tmp_path):
    m = _model()
    p1, p2 = tmp_path / "a.cvcnm", tmp_path / "b.cvcnm"
    write_matnet(m, str(p1))
    write_matnet(m, str(p2))
    a, b = read_matnet(str(p1)), read_matnet(str(p2))
    assert a.arch_hash == b.arch_hash
    assert a.patch_size == 32 and a.d_tok == 64 and a.nhead == 4
    assert len(a.t) == len(b.t) == 64
    for k in (
        "obs_enc.0.weight",
        "fuser.layers.1.self_attn.in_proj_weight",
        "risk_enc.net.4.weight",
        "lam_soft_head.2.bias",
    ):
        assert k in a.t
    # a re-initialised (different) model gets the SAME arch_hash (topology, not values)
    assert read_matnet(str(p1)).arch_hash == a.arch_hash


@pytest.mark.skipif(not nav_native.HAS_MATNET, reason="pycvc lacks nav_matnet_forward")
def test_cpp_matnet_matches_numpy_reference(tmp_path):
    m = _model()
    path = tmp_path / "m.cvcnm"
    write_matnet(m, str(path))
    mn = read_matnet(str(path))
    rng = np.random.default_rng(2)

    # build a ragged batch of agents
    cases = [_case(rng) for _ in range(9)]
    offsets = np.zeros(len(cases) + 1, np.int32)
    for i, (n, *_rest) in enumerate(cases):
        offsets[i + 1] = offsets[i] + n
    total = int(offsets[-1])
    obs_feats = np.zeros((total, 6), np.float32)
    obs_mask = np.zeros(total, np.uint8)
    goal_feats = np.zeros((len(cases), 4), np.float32)
    risk_patch = np.zeros((len(cases), 2, 32, 32), np.float32)
    for i, (n, obs, mask, goal, patch) in enumerate(cases):
        o0 = offsets[i]
        if n:
            obs_feats[o0 : o0 + n] = obs
            obs_mask[o0 : o0 + n] = mask
        goal_feats[i] = goal
        risk_patch[i] = patch

    ca, cb, cg, cls, clh, cml = nav_native.matnet_forward(
        str(path), obs_feats, obs_mask, offsets, goal_feats, risk_patch
    )

    worst = 0.0
    for i, (n, obs, mask, goal, patch) in enumerate(cases):
        na, nb, ng, nls, nlh, nml = matnet_forward_numpy(mn, obs, mask, goal, patch)
        o0 = offsets[i]
        if n:
            worst = max(worst, float(np.abs(ca[o0 : o0 + n] - na).max()))
        worst = max(
            worst,
            abs(cb[i] - nb),
            abs(cg[i] - ng),
            abs(cls[i] - nls),
            abs(clh[i] - nlh),
            abs(cml[i] - nml),
        )
    assert worst < 1e-4, f"C++ matnet vs numpy reference: worst {worst:.2e}"
