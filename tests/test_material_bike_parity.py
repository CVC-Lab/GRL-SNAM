"""Parity for the material rollout on the kinematic bicycle (libcvc P8b).

``cvc::nav::integrate_bicycle_material`` and its hand-written adjoint against the torch
twin :func:`grl_snam.material_bike.integrate_bicycle_material`, whose gradient comes from
autograd. Two independent derivations of the same controller agreeing is the check.

Tier: FLOAT contract, as for the surrogate's parity. The forward is compared at
rtol 1e-4, and the gradient at rtol 1e-3, because the grid_sample rounding differs
by torch build. The batches sit at interior points: agents head roughly at the goal
below the speed limit, and the forces stay inside their clamps. Then no branch
flips between the two float paths. The behaviour test needs only torch.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam.material_bike import BicycleParams, integrate_bicycle_material  # noqa: E402

VEHICLE = BicycleParams(L=0.3, delta_max=0.6, a_max=4.0, a_lat_max=1.5, k_steer=0.8, vmax=1.5)


def _batch(seed, B=6, N=3, P=13, H=(2, 3)):
    rng = np.random.default_rng(seed)
    o0 = rng.uniform(-0.5, 0.5, (B, 2)).astype(np.float32)
    goal = rng.uniform(1.5, 2.5, (B, 2)).astype(np.float32)
    th0 = (
        np.arctan2(goal[:, 1] - o0[:, 1], goal[:, 0] - o0[:, 0]) + rng.uniform(-0.2, 0.2, B)
    ).astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi, (B, N))
    dist = rng.uniform(1.5, 1.9, (B, N))
    C = (o0[:, None, :] + dist[..., None] * np.stack([np.cos(ang), np.sin(ang)], -1)).astype(
        np.float32
    )
    fy, fx = np.meshgrid(np.linspace(0, 1, P), np.linspace(0, 1, P), indexing="ij")
    one = np.stack(
        [
            0.35 + 0.25 * np.sin(2.0 * fx + 1.3 * fy),
            2.0 + 1.2 * np.cos(1.7 * fx - 0.9 * fy),
            0.20 * np.cos(2.1 * fx),
            0.20 * np.sin(1.5 * fy),
            0.15 * np.sin(1.1 * fx + fy),
            0.15 * np.cos(0.8 * fx - fy),
        ]
    ).astype(np.float32)
    return {
        "o0": o0,
        "th0": th0,
        "sp0": rng.uniform(0.3, 0.6, B).astype(np.float32),
        "goal": goal,
        "C": C,
        "R": rng.uniform(0.5, 0.7, (B, N)).astype(np.float32),
        "mask": np.ones((B, N), np.uint8),
        "alphas": rng.uniform(0.03, 0.08, (B, N)).astype(np.float32),
        "beta": rng.uniform(0.8, 1.2, B).astype(np.float32),
        "gamma": rng.uniform(0.2, 0.5, B).astype(np.float32),
        "lam_soft": rng.uniform(0.7, 1.3, B).astype(np.float32),
        "lam_hard": rng.uniform(0.7, 1.3, B).astype(np.float32),
        "rollout_patch": np.broadcast_to(one, (B, 6, P, P)).copy(),
        "rr": np.full(B, 0.5, np.float32),
        "d_hat": np.full(B, 3.0, np.float32),
        "dt": np.full(B, 0.1, np.float32),
        "H": np.array([H[i % len(H)] for i in range(B)], np.int32),
    }


_ORDER = (
    "o0",
    "th0",
    "sp0",
    "goal",
    "C",
    "R",
    "mask",
    "alphas",
    "beta",
    "gamma",
    "lam_soft",
    "lam_hard",
    "rollout_patch",
    "rr",
    "d_hat",
    "dt",
    "H",
)
_COEFS = ("alphas", "beta", "gamma", "lam_soft", "lam_hard")


def _torch_rollout(b, requires_grad=False):
    t = {k: torch.from_numpy(np.asarray(b[k])) for k in _ORDER}
    t["mask"] = t["mask"].to(torch.bool)
    t["H"] = t["H"].to(torch.int64)
    if requires_grad:
        for k in _COEFS:
            t[k] = t[k].clone().requires_grad_(True)
    out = integrate_bicycle_material(*(t[k] for k in _ORDER), vehicle=VEHICLE)
    return t, out


def test_torch_twin_drives_to_the_goal():
    b = _batch(21, B=8, N=1, H=(80,))
    b["mask"][:] = 0
    b["lam_soft"][:] = 0.0
    b["lam_hard"][:] = 0.0
    b["th0"] = b["th0"] + np.where(np.arange(8) % 2, 1.2, -1.2).astype(np.float32)
    _, (oT, _thT, spT, *_rest) = _torch_rollout(b)
    d0 = np.linalg.norm(b["goal"] - b["o0"], axis=1)
    d1 = np.linalg.norm(b["goal"] - oT.numpy(), axis=1)
    assert np.all(d1 < 0.25 * d0), f"did not close on the goal: {d0} -> {d1}"
    assert np.all(spT.numpy() <= VEHICLE.vmax + 1e-6)


nav_native = pytest.importorskip("grl_snam.nav_native")
needs_native = pytest.mark.skipif(
    not getattr(nav_native, "HAS_BICYCLE_MATERIAL_ROLLOUT", False),
    reason="pycvc lacks nav_integrate_bicycle_material",
)


@needs_native
@pytest.mark.parametrize("seed,H", [(7, (2, 3)), (11, (4, 5))])
def test_forward_matches_the_cpp_rollout(seed, H):
    b = _batch(seed, H=H)
    _, ref = _torch_rollout(b)
    got = nav_native.integrate_bicycle_material(*(b[k] for k in _ORDER), VEHICLE)
    names = ("oT", "thT", "spT", "min_clear", "cum_risk", "hard_count", "arc_length")
    for name, r, g in zip(names, ref, got):
        np.testing.assert_allclose(g, r.detach().numpy(), rtol=1e-4, atol=1e-5, err_msg=name)


@needs_native
@pytest.mark.parametrize("seed,H", [(7, (2, 3)), (11, (4, 5))])
def test_gradient_matches_the_cpp_adjoint(seed, H):
    b = _batch(seed, H=H)
    B = b["o0"].shape[0]
    rng = np.random.default_rng(99)
    seeds = {
        "oT": rng.uniform(-1, 1, (B, 2)).astype(np.float32),
        "thT": rng.uniform(-1, 1, B).astype(np.float32),
        "spT": rng.uniform(-1, 1, B).astype(np.float32),
        "min_clear": rng.uniform(-1, 1, B).astype(np.float32),
        "cum_risk": rng.uniform(-1, 1, B).astype(np.float32),
        "arc_length": rng.uniform(-1, 1, B).astype(np.float32),
    }
    t, (oT, thT, spT, mc, cr, _hc, arc) = _torch_rollout(b, requires_grad=True)
    loss = (
        (torch.from_numpy(seeds["oT"]) * oT).sum()
        + (torch.from_numpy(seeds["thT"]) * thT).sum()
        + (torch.from_numpy(seeds["spT"]) * spT).sum()
        + (torch.from_numpy(seeds["min_clear"]) * mc).sum()
        + (torch.from_numpy(seeds["cum_risk"]) * cr).sum()
        + (torch.from_numpy(seeds["arc_length"]) * arc).sum()
    )
    loss.backward()
    got = nav_native.integrate_bicycle_material_vjp(
        *(b[k] for k in _ORDER),
        VEHICLE,
        seeds["oT"],
        seeds["thT"],
        seeds["spT"],
        seeds["min_clear"],
        seeds["cum_risk"],
        seeds["arc_length"],
    )
    for name, g in zip(_COEFS, got):
        ref = t[name].grad.numpy()
        np.testing.assert_allclose(g, ref, rtol=1e-3, atol=1e-4, err_msg=name)
