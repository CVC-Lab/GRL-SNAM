"""U_nav, the navigator's potential: the torch twin against libcvc, and F = -grad U by autograd.

The reference is libcvc's ``src/cvc/tests/data/nav_potential_golden.json`` (copied to
``tests/fixtures``; written by libcvc's ``NavPotential.Golden`` with
``CVC_NAV_POTENTIAL_GOLDEN_WRITE=1``):

- the barrier potentials on a sweep (``ipc_b``, ``ipc_b_pw``, ``sdf_barrier_b``);
- U along a material surrogate rollout on a curved patch: the potential, its terms and the
  conservative force at each step, and the end state;
- ``drive_potential`` on a small curved field with a material stack, with and without a
  footprint.

Tier: FLOAT. libcvc evaluates in float32 with its own op order; the twin runs float32 through
torch kernels, so values agree to about 1e-5 relative (the rollout's end state a little looser).

The autograd half needs no golden: on planes whose gradient channels are the gradients of their
value channels, the twin's own forces equal ``-grad U`` from autograd, which is libcvc's
finite-difference gradcheck (``nav_potential_test``) derived a second way.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam.material import MaterialField  # noqa: E402
from grl_snam.material_nav import integrate_surrogate_material  # noqa: E402
from grl_snam.nav_potential import (  # noqa: E402
    drive_potential,
    ipc_potential,
    ipc_potential_pw,
    material_potential,
    sdf_barrier_potential,
    surrogate_potential,
)
from grl_snam.sdf_nav import SDFField, _ipc_dbdd, _material_force  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "nav_potential_golden.json"
RTOL = 2e-5


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-nav-potential-golden-1"
    return data


def _t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


def _close(got, want, rtol=RTOL, atol=1e-6, what=""):
    got = torch.as_tensor(got, dtype=torch.float64).flatten()
    want = torch.as_tensor(want, dtype=torch.float64).flatten()
    assert got.shape == want.shape, f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    err = (got - want).abs()
    tol = atol + rtol * torch.maximum(got.abs(), want.abs())
    bad = (err > tol).nonzero().flatten()
    assert bad.numel() == 0, (
        f"{what}: {bad.numel()} of {got.numel()} differ; first at {bad[0].item()}: "
        f"torch {got[bad[0]].item()!r}, libcvc {want[bad[0]].item()!r}"
    )


# ── parity with libcvc ──────────────────────────────────────────────────────────────


def test_barrier_potentials_match_libcvc():
    b = _golden()["barriers"]
    d, dh = _t(b["d"]), _t(b["d_hat"])
    _close(ipc_potential(d, dh), b["ipc_b"], what="ipc_b")
    _close(ipc_potential_pw(d, dh), b["ipc_b_pw"], what="ipc_b_pw")
    phi, k, dhs = _t(b["phi"]), _t(b["k_sharp"]), _t(b["d_hat_sdf"])
    _close(sdf_barrier_potential(phi, k, dhs), b["sdf_barrier_b"], what="sdf_barrier_b")


def _material_case(m):
    B, N, Hp, Wp = m["B"], m["N"], m["Hp"], m["Wp"]
    return dict(
        o0=_t(m["o0"]).view(B, 2),
        v0=_t(m["v0"]).view(B, 2),
        goal=_t(m["goal"]).view(B, 2),
        C=_t(m["C"]).view(B, N, 2),
        R=_t(m["R"]).view(B, N),
        mask=torch.tensor(m["mask"], dtype=torch.bool).view(B, N),
        alphas=_t(m["alphas"]).view(B, N),
        beta=_t(m["beta"]),
        gamma=_t(m["gamma"]),
        lam_soft=_t(m["lam_soft"]),
        lam_hard=_t(m["lam_hard"]),
        rollout_patch=_t(m["patch"]).view(B, 6, Hp, Wp),
        d_hat=_t(m["d_hat"]),
        dt=_t(m["dt"]),
        H=torch.tensor(m["H"], dtype=torch.int64),
        robot_radius=_t(m["rr"]),
        margin_factor=m["margin_factor"],
        mass=m["mass"],
        d_hat_sdf=m["d_hat_sdf"],
        k_sharp=m["k_sharp"],
    )


def test_material_rollout_energy_matches_libcvc():
    m = _golden()["material_rollout"]
    B = m["B"]
    energy: dict = {}
    oT, vT, *_ = integrate_surrogate_material(**_material_case(m), energy=energy)
    H = max(m["H"])
    assert energy["potential"].shape == (B, H + 1)
    assert energy["terms"].shape == (B, H + 1, 4)
    assert energy["force"].shape == (B, H, 2)
    _close(energy["potential"], m["potential"], rtol=5e-5, what="potential")
    _close(energy["terms"], m["terms"], rtol=5e-5, what="terms")
    _close(energy["force"], m["force"], rtol=5e-5, atol=1e-5, what="force")
    _close(oT, m["oT"], rtol=5e-5, atol=1e-5, what="oT")
    _close(vT, m["vT"], rtol=5e-5, atol=1e-5, what="vT")


def test_the_energy_record_leaves_the_rollout_unchanged():
    case = _material_case(_golden()["material_rollout"])
    plain = integrate_surrogate_material(**case)
    logged = integrate_surrogate_material(**case, energy={})
    for a, b in zip(plain, logged):
        assert torch.equal(a, b)


def _drive_fields(dr):
    Hf, Wf = dr["H"], dr["W"]
    f = np.asarray(dr["field"], np.float32).reshape(3, Hf, Wf)
    bounds, center, S = dr["bounds"], dr["center"], dr["S"]
    field = SDFField(f[0], f[1], f[2], bounds, center, S)
    planes = np.asarray(dr["material"], np.float32).reshape(6, Hf, Wf)
    return field, MaterialField.from_planes(planes, bounds, center, S)


@pytest.mark.parametrize("footprint", [False, True])
def test_drive_potential_matches_libcvc(footprint):
    dr = _golden()["drive"]
    field, mat = _drive_fields(dr)
    kw = {}
    if footprint:
        kw = dict(body_offsets=dr["body_offsets"], body_rr=dr["body_rr"], body_gain=dr["body_gain"])
    u = drive_potential(
        field,
        _t(dr["o"]).view(-1, 2),
        _t(dr["th"]),
        _t(dr["goal"]).view(-1, 2),
        _t(dr["alpha"]),
        _t(dr["beta"]),
        dr["rr"],
        dr["d_hat"],
        material=mat,
        lam_soft=_t(dr["lam_soft"]),
        lam_hard=_t(dr["lam_hard"]),
        k_sharp=dr["k_sharp"],
        d_hat_m=dr["d_hat_m"],
        **kw,
    )
    want = dr["terms_footprint" if footprint else "terms"]
    _close(u.stacked(), want, what="drive terms" + (" (footprint)" if footprint else ""))


# ── F = -grad U, by autograd ───────────────────────────────────────────────────────


def _linear_planes(Hg=41, lo=-10.0, hi=10.0, S=0.1):
    """Exact planes on a world grid: an SDF wall (normalized), linear risk (its gradient per
    normalized unit) and linear phi_m in metres (its gradient per metre)."""
    xs = np.linspace(lo, hi, Hg, dtype=np.float64)
    wx, wy = np.meshgrid(xs, xs)  # rows are y
    n = np.array([0.6, -0.8])
    phi = S * (n[0] * wx + n[1] * wy + 3.0)
    ones = np.ones_like(wx)
    field = SDFField(
        phi.astype(np.float32),
        (n[0] * ones).astype(np.float32),
        (n[1] * ones).astype(np.float32),
        (lo, lo, hi, hi),
        (0.0, 0.0),
        S,
    )
    r = 0.5 + 0.02 * wx - 0.015 * wy
    pm = 6.0 + 0.3 * wx + 0.2 * wy
    planes = np.stack([r, pm, 0.02 / S * ones, -0.015 / S * ones, 0.3 * ones, 0.2 * ones], 0)
    return field, MaterialField.from_planes(planes, (lo, lo, hi, hi), (0.0, 0.0), S)


def test_drive_force_is_minus_grad_u():
    """sdf_nav's drive force (barrier + goal + _material_force) against -grad drive_potential."""
    field, mat = _linear_planes()
    g = torch.Generator().manual_seed(3)
    o = (torch.rand(256, 2, generator=g) * 1.6 - 0.8).double()
    goal = (torch.rand(256, 2, generator=g) * 1.6 - 0.8).double()
    al = 0.5 + torch.rand(256, generator=g).double()
    be = 0.5 + torch.rand(256, generator=g).double()
    ls = 0.3 + torch.rand(256, generator=g).double()
    lh = 0.5 + torch.rand(256, generator=g).double()
    rr, d_hat = 0.15, 0.35
    phi, _ = field.sample(o.float())
    keep = (phi - rr) > 0.02
    o, goal, al, be, ls, lh = (x[keep] for x in (o, goal, al, be, ls, lh))
    assert o.shape[0] > 100
    o32 = o.float().requires_grad_(True)
    u = drive_potential(
        field,
        o32,
        torch.zeros(o.shape[0]),
        goal.float(),
        al.float(),
        be.float(),
        rr,
        d_hat,
        material=mat,
        lam_soft=ls.float(),
        lam_hard=lh.float(),
    )
    (grad,) = torch.autograd.grad(u.total().sum(), o32)
    with torch.no_grad():
        ph, nrm = field.sample(o32)
        ipc = _ipc_dbdd(ph - rr, d_hat)
        f = -(al.float() * ipc).unsqueeze(-1) * nrm - be.float().unsqueeze(-1) * (
            o32 - goal.float()
        )
        f = f + _material_force(mat, o32, ls.float(), lh.float(), 1.25, 12.0)
    err = (f + grad).norm(dim=-1) / f.norm(dim=-1).clamp_min(1e-2)
    assert (u.barrier > 0).sum() > 30, "the barrier should be engaged at some poses"
    assert err.max().item() < 2e-3, f"worst |F + grad U| / |F| = {err.max().item():.3e}"


def test_material_rollout_force_is_minus_grad_u():
    """The torch material rollout's logged force against -grad material_potential, on a linear
    patch (its gradient channels are its value channels' gradients in pixels)."""
    B, N, Hp, Wp = 16, 3, 33, 33
    g = torch.Generator().manual_seed(5)
    ys, xs = torch.meshgrid(
        torch.arange(Hp) - 0.5 * (Hp - 1), torch.arange(Wp) - 0.5 * (Wp - 1), indexing="ij"
    )
    one = torch.ones(Hp, Wp)
    patch = torch.stack(
        [
            0.45 + 0.012 * xs - 0.009 * ys,
            4.0 + 0.06 * xs + 0.04 * ys,
            0.012 * one,
            -0.009 * one,
            0.06 * one,
            0.04 * one,
        ],
        0,
    ).float()
    o0 = torch.rand(B, 2, generator=g) * 8 - 4
    ang = torch.rand(B, N, generator=g) * 2 * math.pi
    C = o0.unsqueeze(1) + 3.6 * torch.stack([ang.cos(), ang.sin()], -1)
    case = dict(
        o0=o0,
        v0=torch.zeros(B, 2),
        goal=o0 + torch.rand(B, 2, generator=g) * 16 - 8,
        C=C,
        R=torch.full((B, N), 1.0),
        mask=torch.ones(B, N, dtype=torch.bool),
        alphas=0.3 + torch.rand(B, N, generator=g) * 1.7,
        beta=0.05 + 0.1 * torch.rand(B, generator=g),
        gamma=torch.ones(B),
        lam_soft=1.0 + torch.rand(B, generator=g),
        lam_hard=1.0 + torch.rand(B, generator=g),
        rollout_patch=patch.expand(B, -1, -1, -1).contiguous(),
        d_hat=torch.full((B,), 3.0),
        dt=torch.full((B,), 0.05),
        H=torch.full((B,), 1, dtype=torch.int64),
        robot_radius=torch.ones(B),
        d_hat_sdf=5.0,
    )
    energy: dict = {}
    integrate_surrogate_material(**case, energy=energy)
    o = o0.clone().requires_grad_(True)
    u = material_potential(
        o,
        case["goal"],
        C,
        case["R"],
        case["mask"],
        case["alphas"],
        case["beta"],
        case["lam_soft"],
        case["lam_hard"],
        case["rollout_patch"],
        o0,
        case["d_hat"],
        robot_radius=case["robot_radius"],
        d_hat_sdf=5.0,
    )
    (grad,) = torch.autograd.grad(u.total().sum(), o)
    f = energy["force"][:, 0]
    err = (f + grad).norm(dim=-1) / f.norm(dim=-1).clamp_min(1e-2)
    assert (u.barrier > 0).sum() > B // 2
    assert err.max().item() < 1e-4, f"worst |F + grad U| / |F| = {err.max().item():.3e}"


def test_surrogate_force_is_minus_grad_u():
    """sdf_nav's point-mass force (barrier + goal) against -grad surrogate_potential."""
    field, _ = _linear_planes()
    g = torch.Generator().manual_seed(9)
    o = torch.rand(200, 2, generator=g) * 1.6 - 0.8
    goal = torch.rand(200, 2, generator=g) * 1.6 - 0.8
    phi, _ = field.sample(o)
    o, goal = o[(phi - 0.15) > 0.02], goal[(phi - 0.15) > 0.02]
    al, be = 1.3, 0.7
    oo = o.clone().requires_grad_(True)
    u = surrogate_potential(field, oo, goal, al, be, 0.15, 0.35)
    (grad,) = torch.autograd.grad(u.total().sum(), oo)
    with torch.no_grad():
        ph, nrm = field.sample(oo)
        f = -(al * _ipc_dbdd(ph - 0.15, 0.35)).unsqueeze(-1) * nrm - be * (oo - goal)
    err = (f + grad).norm(dim=-1) / f.norm(dim=-1).clamp_min(1e-2)
    assert err.max().item() < 2e-3, f"worst |F + grad U| / |F| = {err.max().item():.3e}"
