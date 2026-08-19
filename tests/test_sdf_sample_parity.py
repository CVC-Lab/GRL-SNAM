"""The torch-free C++ SDF sampler (cvc::nav::sdf_sample) vs the torch reference.

`nav_native.sdf_sample` is a libtorch-free transcription of
`sdf_nav.SDFField.sample` / `BatchedSDFField.sample` — torch `grid_sample`
bilinear (align_corners=True, padding_mode="border") plus the unit-normal renorm
— the first numeric piece of the pure-C++ drive (docs/CVCNAV_CPP_PORT_ROADMAP.md
P1). Its contract is FLOAT-EQUIVALENT (<=~1 ULP), asserted below at tolerance;
in practice, matching torch's float32 op order makes it bit-exact on this
platform, which the test also records.

The `map_id` gather is what makes shared/clustered/private one sampler: agent i
reads plane `map_id[i]` (None => plane 0 for all).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import sdf_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402

pytestmark = pytest.mark.skipif(
    not nav_native.HAS_SDF_SAMPLE, reason="pycvc build lacks nav_sdf_sample"
)

BOUNDS = (-200.0, -150.0, 260.0, 190.0)
CENTER = (7.0, -3.0)
SCALE = 0.5
H, W = 96, 128


def _field(rng, m):
    return (rng.standard_normal((m, 3, H, W)) * 3.0).astype(np.float32)


def _positions(rng, n):
    # normalized (centered) positions spanning in- and out-of-bounds (border).
    half = (BOUNDS[2] - BOUNDS[0]) / 2
    return (rng.uniform(-1.6, 1.6, (n, 2)) / (2 * SCALE) * half).astype(np.float32)


def _ref(field, on, map_id):
    if map_id is None:
        sf = sdf_nav.SDFField(field[0, 0], field[0, 1], field[0, 2], BOUNDS, CENTER, SCALE)
        phi, nrm = sf.sample(torch.from_numpy(on))
        return phi.numpy(), nrm.numpy()
    fields = [
        sdf_nav.SDFField(field[g, 0], field[g, 1], field[g, 2], BOUNDS, CENTER, SCALE)
        for g in range(field.shape[0])
    ]
    bf = sdf_nav.BatchedSDFField.stack([fields[m] for m in map_id])  # agent i -> plane map_id[i]
    phi, nrm = bf.sample(torch.from_numpy(on))
    return phi.numpy(), nrm.numpy()


def _sample(field, on, map_id):
    return nav_native.sdf_sample(
        field, on, bounds=BOUNDS, center=CENTER, scale=SCALE, map_id=map_id
    )


@pytest.mark.parametrize(
    "m,use_map",
    [(1, False), (64, True), (8, True)],
    ids=["shared", "private-gather", "clustered"],
)
def test_sdf_sample_matches_torch(m, use_map):
    rng = np.random.default_rng(0)
    field = _field(rng, m)
    on = _positions(rng, 2000)
    map_id = rng.integers(0, m, 2000).astype(np.int32) if use_map else None
    rphi, rnrm = _ref(field, on, map_id)
    cphi, cnrm = _sample(field, on, map_id)
    # float-equivalence contract (holds bit-exact on this platform).
    assert np.allclose(cphi, rphi, rtol=1e-5, atol=1e-6), np.abs(cphi - rphi).max()
    assert np.allclose(cnrm, rnrm, rtol=1e-4, atol=1e-6), np.abs(cnrm - rnrm).max()


def test_shared_is_bit_exact_here():
    """Not contractual, but a regression tripwire: on this platform the sampler
    reproduces torch's float32 grid_sample exactly, so a change that widens the
    residual is worth noticing."""
    rng = np.random.default_rng(1)
    field = _field(rng, 1)
    on = _positions(rng, 1000)
    rphi, rnrm = _ref(field, on, None)
    cphi, cnrm = _sample(field, on, None)
    assert np.array_equal(cphi, rphi)
    assert np.array_equal(cnrm, rnrm)


def test_out_of_bounds_border_clamp():
    """Positions far outside the grid clamp to the border, matching torch."""
    rng = np.random.default_rng(2)
    field = _field(rng, 1)
    big = (BOUNDS[2] - BOUNDS[0]) * 10
    on = np.array([[big, big], [-big, -big], [big, -big], [0.0, 0.0]], np.float32)
    rphi, rnrm = _ref(field, on, None)
    cphi, cnrm = _sample(field, on, None)
    assert np.allclose(cphi, rphi, rtol=1e-5, atol=1e-6)
    assert np.allclose(cnrm, rnrm, rtol=1e-4, atol=1e-6)


def test_single_agent_and_empty():
    rng = np.random.default_rng(3)
    field = _field(rng, 1)
    phi, nrm = _sample(field, _positions(rng, 1), None)
    assert phi.shape == (1,) and nrm.shape == (1, 2)
    phi0, nrm0 = _sample(field, np.zeros((0, 2), np.float32), None)
    assert phi0.shape == (0,) and nrm0.shape == (0, 2)
