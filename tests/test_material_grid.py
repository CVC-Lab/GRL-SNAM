"""MaterialGrid derived-plane pipeline + the pinned no-scipy Gaussian blur.

These pin the exact numeric conventions the C++ ``material_build`` twin must
reproduce BIT-identically: kernel tap op order, sequential normalization,
symmetric (edge-repeat) padding, f64 accumulation with a single f32 store,
gradients of the f32-stored planes, f32-cast scalar divides, and the
metres-end-to-end hazard distance field.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sdf_nav  # noqa: E402
from grl_snam.material import (  # noqa: E402
    MaterialGrid,
    MaterialParams,
    gaussian_blur,
    gaussian_kernel,
)

BOUNDS = (-100.0, -100.0, 100.0, 100.0)
CENTER = (0.0, 0.0)
SCALE = 0.05


def _grid(n=32, risk=None, hard=None, **kw):
    r = np.zeros((n, n), np.float32) if risk is None else risk
    h = np.zeros((n, n), bool) if hard is None else hard
    return MaterialGrid(r, h, BOUNDS, CENTER, SCALE, **kw)


# ---------------------------------------------------------------------------
# Kernel + blur
# ---------------------------------------------------------------------------


def test_gaussian_kernel_matches_pinned_formula():
    w = gaussian_kernel(1.0)
    assert len(w) == 9  # radius = int(4*sigma + 0.5) = 4
    taps = [math.exp(-0.5 * k * k) for k in range(-4, 5)]
    total = 0.0
    for t in taps:
        total += t
    for a, b in zip(w, [t / total for t in taps]):
        assert a == b  # exact — same op order
    assert w[4] == max(w)


def test_blur_preserves_constant_field():
    a = np.full((20, 20), 0.7, np.float32)
    out = gaussian_blur(a, 1.0)
    assert out.dtype == np.float32
    assert np.allclose(out, 0.7, atol=2e-7)


def test_blur_corner_impulse_pins_symmetric_padding():
    """scipy's default 'reflect' is edge-REPEATING (symmetric): index -1 maps
    back to index 0. A mirror ('np.pad reflect') implementation gives a
    measurably different corner value — this is the test that catches it."""
    n = 16
    a = np.zeros((n, n), np.float32)
    a[0, 0] = 1.0
    out = gaussian_blur(a, 1.0)
    w = gaussian_kernel(1.0)
    r = len(w) // 2
    # per axis, taps landing on index 0 from position 0: k=0 (idx 0) and
    # k=-1 (idx -1 -> 0 under symmetric). Mirror would give only k=0.
    edge = w[r] + w[r - 1]
    assert out[0, 0] == pytest.approx(edge * edge, rel=1e-6)
    # (1,1) receives the impulse via taps k=-1 (direct) and k=-2 (reflected
    # through the edge) on each axis; mirror padding would give w[r-1] alone.
    off = w[r - 1] + w[r - 2]
    assert out[1, 1] == pytest.approx(off * off, rel=1e-6)


def test_blur_rejects_grids_smaller_than_kernel():
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((3, 40), np.float32), 1.0)


def test_blur_sigma_zero_is_identity():
    a = np.random.default_rng(0).random((12, 12)).astype(np.float32)
    assert np.array_equal(gaussian_blur(a, 0.0), a)


# ---------------------------------------------------------------------------
# Derived planes
# ---------------------------------------------------------------------------


def test_phi_hard_is_metres_edt_of_hard_cells():
    n = 32
    hard = np.zeros((n, n), bool)
    hard[10, 10] = True
    g = _grid(n, hard=hard)
    cell_w = (BOUNDS[2] - BOUNDS[0]) / (n - 1)
    assert g.phi_hard_m.dtype == np.float32
    assert g.phi_hard_m[10, 10] == 0.0
    assert g.phi_hard_m[10, 13] == pytest.approx(3.0 * cell_w, rel=1e-6)
    assert g.phi_hard_m[13, 14] == pytest.approx(5.0 * cell_w, rel=1e-6)  # 3-4-5
    # matches the exact house EDT chain with one f32 store
    ref = (np.sqrt(sdf_nav._edt2(hard)) * cell_w).astype(np.float32)
    assert np.array_equal(g.phi_hard_m, ref)


def test_risk_is_blurred_and_clipped():
    n = 32
    risk = np.zeros((n, n), np.float32)
    risk[16, 16] = 1.0
    g = _grid(n, risk=risk)
    assert g.risk.max() < 1.0  # blurred peak
    assert g.risk[16, 16] == g.risk.max()
    assert g.risk.min() >= 0.0
    ref = np.clip(gaussian_blur(risk, 1.0), 0.0, 1.0).astype(np.float32)
    assert np.array_equal(g.risk, ref)


def test_gradients_are_of_the_f32_stored_planes_with_pinned_divides():
    n = 32
    rng = np.random.default_rng(7)
    risk = rng.random((n, n)).astype(np.float32)
    hard = rng.random((n, n)) < 0.1
    g = _grid(n, risk=risk, hard=hard)
    cell_w = g.cell_w
    gy, gx = np.gradient(g.risk)  # f32 in, f32 out
    assert np.array_equal(g.grad_rx, gx / np.float32(cell_w * SCALE))
    assert np.array_equal(g.grad_ry, gy / np.float32(cell_w * SCALE))
    pgy, pgx = np.gradient(g.phi_hard_m)
    assert np.array_equal(g.grad_px, pgx / np.float32(cell_w))
    assert np.array_equal(g.grad_py, pgy / np.float32(cell_w))
    for plane in (g.grad_rx, g.grad_ry, g.grad_px, g.grad_py):
        assert plane.dtype == np.float32


def test_hazard_gradient_is_near_unit_away_from_the_hazard():
    n = 48
    hard = np.zeros((n, n), bool)
    hard[:, 0] = True  # a wall along the left edge
    g = _grid(n, hard=hard)
    # away from the wall the EDT slope is exactly one metre per metre
    assert g.grad_px[20, 20] == pytest.approx(1.0, rel=1e-5)
    assert g.grad_py[20, 20] == pytest.approx(0.0, abs=1e-6)


def test_stamp_bumps_version_and_rederives():
    g = _grid(32)
    assert g.version == 0
    f0 = g.field()
    g.stamp_risk(10, 14, 10, 14, 0.9)
    assert g.version == 1
    assert g.risk[11, 11] > 0.3
    assert g.field() is not f0  # lazy field invalidated
    g.stamp_hard(20, 22, 20, 22)
    assert g.version == 2
    assert g.phi_hard_m[21, 21] == 0.0


def test_cost_raster_soft_plus_finite_hard():
    n = 32
    risk = np.full((n, n), 0.5, np.float32)
    hard = np.zeros((n, n), bool)
    hard[5, 5] = True
    g = _grid(n, risk=risk, hard=hard, params=MaterialParams())
    c = g.cost_raster()
    assert c.dtype == np.float64
    # blur keeps the constant 0.5 (within f32 noise); hard adds 25
    assert c[20, 20] == pytest.approx(10.0 * 0.5, rel=1e-5)
    assert c[5, 5] == pytest.approx(10.0 * float(g.risk[5, 5]) + 25.0, rel=1e-6)
    assert np.all(c >= 0.0)


def test_material_field_channels_and_sampling():
    n = 32
    risk = np.zeros((n, n), np.float32)
    risk[:, 16:] = 0.8  # risk on the +x half
    hard = np.zeros((n, n), bool)
    hard[0, :] = True  # hazard along -y edge
    g = _grid(n, risk=risk, hard=hard)
    f = g.field()
    assert tuple(f.field.shape) == (1, 6, n, n)
    on = torch.zeros(3, 2)  # the world centre, normalized
    r, phi_m, grad_r, grad_phi = f.sample(on)
    assert r.shape == (3,) and phi_m.shape == (3,) and grad_r.shape == (3, 2)
    # centre sits at the blurred risk boundary; phi_m ~ half the world height
    assert 0.0 < float(r[0]) < 0.8
    assert float(phi_m[0]) == pytest.approx(100.0, rel=0.05)
    assert float(grad_r[0, 0]) > 0.0  # risk increases with +x
    assert float(grad_phi[0, 1]) > 0.0  # clearance increases away from the -y wall


def test_barrier_scale_oracle_phi_in_metres():
    """The k/S rescale trap: phi stays in metres so at phi == d_hat_sdf_m the
    barrier factor is EXACTLY -sigmoid(0) = -0.5, and one metre to either side
    is sigmoid(+-k_sharp)."""
    from sdf_nav import _material_force

    class _Flat:
        def __init__(self, phi_m):
            self.phi = phi_m

        def sample(self, o):
            b = o.shape[0]
            return (
                torch.zeros(b),
                torch.full((b,), self.phi),
                torch.zeros(b, 2),
                torch.tensor([[1.0, 0.0]]).expand(b, 2),
            )

    lam_h = torch.tensor([2.0])
    zero = torch.tensor([0.0])
    o = torch.zeros(1, 2)
    f_at = _material_force(_Flat(3.0), o, zero, lam_h, 5.0, 3.0)
    assert float(f_at[0, 0]) == pytest.approx(2.0 * 0.5)  # -lam * (-0.5) * 1
    f_in = _material_force(_Flat(2.0), o, zero, lam_h, 5.0, 3.0)
    assert float(f_in[0, 0]) == pytest.approx(2.0 / (1.0 + math.exp(-5.0)), rel=1e-6)
    f_out = _material_force(_Flat(4.0), o, zero, lam_h, 5.0, 3.0)
    assert float(f_out[0, 0]) == pytest.approx(2.0 / (1.0 + math.exp(5.0)), rel=1e-5)
    far = _material_force(_Flat(50.0), o, zero, lam_h, 5.0, 3.0)
    assert float(far[0, 0]) == 0.0  # sigmoid underflows to exact zero in f32
