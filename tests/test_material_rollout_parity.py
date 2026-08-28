"""Parity for the torch-free obstacle-list material rollout.

cvc::nav::integrate_surrogate_material vs the torch reference
material_nav.integrate_surrogate_material (itself a bit-identical port of the
source method's differentiable integrator). This is the rollout the learned
coef_energy_net feeds and the forward the P5 training backward differentiates.

Tier: FLOAT contract (rtol 1e-4). The C++ matches torch's float32 op order
closely (bit-exact against some torch builds), but the residual is
torch-version dependent through grid_sample's rounding, so the contract is
float-equivalent, not bit — a bit tripwire here would be fragile across the
heterogeneous closure. Skips until a rollout-capable pycvc ships.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import material_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402

pytestmark = pytest.mark.skipif(
    not nav_native.HAS_MATERIAL_ROLLOUT_INTEGRATOR,
    reason="pycvc lacks nav_integrate_surrogate_material",
)


def _batch(rng, B, N, Hp=13, Wp=13):
    o0 = (rng.random((B, 2)) * 8 - 4).astype(np.float32)
    v0 = (rng.random((B, 2)) * 0.4 - 0.2).astype(np.float32)
    goal = (rng.random((B, 2)) * 8 - 4).astype(np.float32)
    C = (
        (rng.random((B, N, 2)) * 10 - 5).astype(np.float32)
        if N
        else np.zeros((B, 0, 2), np.float32)
    )
    R = (rng.random((B, N)) * 2 + 0.5).astype(np.float32) if N else np.zeros((B, 0), np.float32)
    mask = (rng.random((B, N)) > 0.3) if N else np.zeros((B, 0), bool)
    alphas = (rng.random((B, N)) * 2).astype(np.float32) if N else np.zeros((B, 0), np.float32)
    beta = (rng.random(B) * 2).astype(np.float32)
    gamma = rng.random(B).astype(np.float32)
    lam_soft = (rng.random(B) * 5).astype(np.float32)
    lam_hard = (rng.random(B) * 10).astype(np.float32)
    patch = rng.standard_normal((B, 6, Hp, Wp)).astype(np.float32)
    patch[:, 0] = rng.random((B, Hp, Wp))
    patch[:, 1] = rng.random((B, Hp, Wp)) * 10
    d_hat = (rng.random(B) * 2 + 1).astype(np.float32)
    dt = np.full(B, 0.01, np.float32)
    H = rng.integers(1, 7, B).astype(np.int32)
    rr = (rng.random(B) * 2).astype(np.float32)
    return dict(
        o0=o0,
        v0=v0,
        goal=goal,
        C=C,
        R=R,
        mask=mask,
        alphas=alphas,
        beta=beta,
        gamma=gamma,
        lam_soft=lam_soft,
        lam_hard=lam_hard,
        patch=patch,
        d_hat=d_hat,
        dt=dt,
        H=H,
        rr=rr,
    )


def _torch_ref(b):
    o, v, mc, cr, hc, al = material_nav.integrate_surrogate_material(
        torch.from_numpy(b["o0"].copy()),
        torch.from_numpy(b["v0"].copy()),
        torch.from_numpy(b["goal"]),
        torch.from_numpy(b["C"]),
        torch.from_numpy(b["R"]),
        torch.from_numpy(b["mask"]),
        torch.from_numpy(b["alphas"]),
        torch.from_numpy(b["beta"]),
        torch.from_numpy(b["gamma"]),
        torch.from_numpy(b["lam_soft"]),
        torch.from_numpy(b["lam_hard"]),
        torch.from_numpy(b["patch"]),
        torch.from_numpy(b["d_hat"]),
        torch.from_numpy(b["dt"]),
        torch.from_numpy(b["H"]),
        robot_radius=torch.from_numpy(b["rr"]),
        margin_factor=0.5,
        d_hat_sdf=3.0,
    )
    return o.numpy(), v.numpy(), mc.numpy(), cr.numpy(), hc.numpy(), al.numpy()


def _cpp(b):
    return nav_native.integrate_surrogate_material(
        b["o0"],
        b["v0"],
        b["goal"],
        b["C"],
        b["R"],
        b["mask"],
        b["alphas"],
        b["beta"],
        b["gamma"],
        b["lam_soft"],
        b["lam_hard"],
        b["patch"],
        b["rr"],
        b["d_hat"],
        b["dt"],
        b["H"],
        margin_factor=0.5,
        mass=1.0,
        d_hat_sdf=3.0,
        k_sharp=5.0,
    )


def test_rollout_float_parity_vs_torch():
    rng = np.random.default_rng(21)
    for N in (0, 3, 5, 8):
        b = _batch(rng, B=7, N=N)
        ref = _torch_ref(b)
        cpp = _cpp(b)
        for name, r, c in zip(("o", "v", "min_clear", "cum_risk", "hard_count", "arc"), ref, cpp):
            assert np.allclose(c, r, rtol=1e-4, atol=1e-5), f"N={N} {name}"


def test_rollout_thread_determinism():
    rng = np.random.default_rng(23)
    b = _batch(rng, B=20, N=4)
    a = nav_native.integrate_surrogate_material(
        b["o0"],
        b["v0"],
        b["goal"],
        b["C"],
        b["R"],
        b["mask"],
        b["alphas"],
        b["beta"],
        b["gamma"],
        b["lam_soft"],
        b["lam_hard"],
        b["patch"],
        b["rr"],
        b["d_hat"],
        b["dt"],
        b["H"],
        num_threads=1,
    )
    c = nav_native.integrate_surrogate_material(
        b["o0"],
        b["v0"],
        b["goal"],
        b["C"],
        b["R"],
        b["mask"],
        b["alphas"],
        b["beta"],
        b["gamma"],
        b["lam_soft"],
        b["lam_hard"],
        b["patch"],
        b["rr"],
        b["d_hat"],
        b["dt"],
        b["H"],
        num_threads=8,
    )
    for x, y in zip(a, c):
        assert np.array_equal(x, y)
