"""The per-step energy books (the white paper's Eq. 38): the torch twin against libcvc.

The reference is libcvc's ``src/cvc/tests/data/nav_energy_books_golden.json`` (copied to
``tests/fixtures``; written by ``NavPotential.BooksGolden`` with
``CVC_NAV_ENERGY_BOOKS_GOLDEN_WRITE=1``):

- ``bound``: ``energy_residual_bound`` on a sweep;
- ``clamp``: a ``diff::surr_step`` rollout (``sdf_nav.sdf_rollout`` here) whose vmax clamp cuts
  every step, booked per step: ``book_step``'s arguments and its books;
- ``sharp``: ``integrate_surrogate_material``'s books with a port force and the sharp check;
- ``unlogged``: the same at dt 0.002 under a strong port force, which fails the bound at every
  step when left out of the books.

Tiers:

- DOUBLE for :func:`book_step` and :func:`energy_residual_bound` on libcvc's own arguments: the
  same float64 arithmetic, to 1e-12 (measured: bit for bit).
- FLOAT for the rollouts: libcvc's float32 rollout against torch's, each booked in float64 from its
  own states. States, potentials and kinetic energies to the material parity's 5e-5. Each book is
  compared relative to the larger of itself and the step's bound, its natural scale (r_E nearly
  cancels): to 1e-5 on the material surrogate, whose books are float64 throughout (measured
  8.3e-8), and to 2e-5 on the clamped rollout, whose U and forces are float32, as libcvc books
  them there (measured 1.1e-6).

Two checks need no golden: on the twin, r_E is second order in dt, and on a quadratic potential
(a constant Hessian) the sharp check meets r_E exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam.material_ext import MaterialExt  # noqa: E402
from grl_snam.material_nav import integrate_surrogate_material  # noqa: E402
from grl_snam.nav_potential import (  # noqa: E402
    book_step,
    energy_residual_bound,
    surrogate_potential,
)
from grl_snam.sdf_nav import SDFField, _ipc_dbdd, sdf_rollout  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "nav_energy_books_golden.json"
BOOKS = ("w_damp", "w_port", "w_clamp", "w_cross", "residual", "bound")


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-nav-energy-books-golden-1"
    return data


def _t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


def _d(x):
    return torch.tensor(x, dtype=torch.float64)


def _close(got, want, rtol, atol=0.0, scale=None, what=""):
    """|got - want| <= atol + rtol max(|got|, |want|, scale)."""
    got = torch.as_tensor(got, dtype=torch.float64).flatten()
    want = torch.as_tensor(want, dtype=torch.float64).flatten()
    assert got.shape == want.shape, f"{what}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    ref = torch.maximum(got.abs(), want.abs())
    if scale is not None:
        ref = torch.maximum(ref, torch.as_tensor(scale, dtype=torch.float64).flatten())
    err = (got - want).abs()
    bad = (err > atol + rtol * ref).nonzero().flatten()
    assert bad.numel() == 0, (
        f"{what}: {bad.numel()} of {got.numel()} differ; first at {bad[0].item()}: "
        f"torch {got[bad[0]].item()!r}, libcvc {want[bad[0]].item()!r}"
    )


def _f32(x: float) -> float:
    """A float parameter as libcvc holds it."""
    return float(np.float32(x))


# ── DOUBLE: the books' arithmetic ───────────────────────────────────────────────────


def test_energy_residual_bound_matches_libcvc():
    b = _golden()["bound"]
    got = energy_residual_bound(
        _d(b["dt"]),
        _d(b["g"]),
        _d(b["gamma"]),
        _d(b["v"]),
        _d(b["F"]),
        _d(b["L_hat"]),
        _d(b["vt"]),
        _d(b["mass"]),
        _d(b["kappa_H"]),
    )
    _close(got, b["bound"], rtol=1e-14, what="bound")


def _clamp_args(c):
    a = c["args"]
    vec = {k: _d(a[k]).view(-1, 2) for k in ("v", "vp", "vt", "dx", "g", "g1")}
    sc = {k: _d(a[k]) for k in ("U0", "U1", "vHv")}
    assert sc["U0"].numel() == c["B"] * c["steps"]
    gamma = _t(c["gamma"]).double().repeat_interleave(c["steps"])  # float, as libcvc holds it
    return vec, sc, gamma


def test_book_step_matches_libcvc_on_the_clamped_steps():
    c = _golden()["clamp"]
    vec, sc, gamma = _clamp_args(c)
    k = book_step(
        _f32(c["hdt"]),
        vec["v"],
        vec["vp"],
        vec["vt"],
        vec["dx"],
        vec["g"],
        vec["g1"],
        torch.zeros_like(vec["v"]),
        sc["U0"],
        sc["U1"],
        gamma=gamma,
        vHv=sc["vHv"],
    )
    want = c["books"]
    scale = _d(want["bound"])
    for name in ("delta_H", "residual_sharp") + BOOKS:
        _close(getattr(k, name), want[name], rtol=1e-12, scale=scale, what=name)
    _close(k.kinetic, want["kinetic"], rtol=1e-14, what="kinetic")
    assert (k.w_clamp > 0).all(), "the clamp cuts every step"
    assert (k.residual.abs() <= k.bound).all()


# ── FLOAT: the clamped surrogate rollout ───────────────────────────────────────────


def _clamp_field(c):
    f = np.asarray(c["field"], np.float32).reshape(3, c["H"], c["W"])
    return SDFField(f[0], f[1], f[2], c["bounds"], c["center"], c["S"])


def _surrogate_step_books(field, o, v, goal, al, be, ga, rr, d_hat, vmax, hdt):
    """One diff::surr_step (sdf_rollout, one step) and its books as libcvc's golden books it:
    the forces and U in float32 at float32 positions, the books in float64."""

    def force(at):
        phi, nrm = field.sample(at)
        return -(al * _ipc_dbdd(phi - rr, d_hat)).unsqueeze(-1) * nrm - be.unsqueeze(-1) * (
            at - goal
        )

    def U(at):
        return surrogate_potential(field, at, goal, al, be, rr, d_hat).total().double()

    o1, v1, _ = sdf_rollout(field, o, v, goal, al, be, ga, 1, rr=rr, d_hat=d_hat, dt=hdt, vmax=vmax)
    dt = _f32(hdt)
    o64, v64, vt = o.double(), v.double(), v1.double()
    f0, f1 = force(o).double(), force(o1).double()
    vp = v64 + dt * (f0 - ga.double().unsqueeze(-1) * v64)
    h = 0.5 * dt
    df = force((o64 + h * vt).float()).double() - force((o64 - h * vt).float()).double()
    vHv = -(vt[:, 0] * df[:, 0] + vt[:, 1] * df[:, 1]) / (2.0 * h)
    k = book_step(
        dt,
        v64,
        vp,
        vt,
        o1.double() - o64,
        -f0,
        -f1,
        torch.zeros_like(v64),
        U(o),
        U(o1),
        gamma=ga.double(),
        vHv=vHv,
    )
    return o1, v1, k


def test_clamped_rollout_and_books_match_libcvc():
    c = _golden()["clamp"]
    B, S = c["B"], c["steps"]
    field = _clamp_field(c)
    o, v = _t(c["o0"]).view(B, 2), _t(c["v0"]).view(B, 2)
    goal = _t(c["goal"]).view(B, 2)
    al, be, ga = _t(c["alpha"]), _t(c["beta"]), _t(c["gamma"])
    os, vs, books = [o], [v], []
    for _ in range(S):
        o, v, k = _surrogate_step_books(
            field, o, v, goal, al, be, ga, c["rr"], c["d_hat"], c["vmax"], c["hdt"]
        )
        os.append(o)
        vs.append(v)
        books.append(k)
    _close(torch.stack(os, 1), c["o"], rtol=5e-5, atol=1e-6, what="o")
    _close(torch.stack(vs, 1), c["v"], rtol=5e-5, atol=1e-6, what="v")
    want = c["books"]
    scale = _d(want["bound"])
    for name in ("delta_H", "residual_sharp") + BOOKS:
        got = torch.stack([getattr(k, name) for k in books], 1)
        _close(got, want[name], rtol=2e-5, scale=scale, what=name)
    w_clamp = torch.stack([k.w_clamp for k in books], 1)
    r = torch.stack([k.residual for k in books], 1)
    bound = torch.stack([k.bound for k in books], 1)
    assert (w_clamp > 0).all(), "the clamp cuts every step"
    assert (r.abs() <= bound).all(), "with W_clamp and W_cross booked every step closes"


# ── FLOAT: the material surrogate's books ──────────────────────────────────────────


def _material_case(m):
    B, N, Hp, Wp = m["B"], m["N"], m["Hp"], m["Wp"]
    S = max(m["H"])
    return dict(
        o0=_t(m["o0"]).view(B, 2),
        v0=_t(m["v0"]).view(B, 2),
        goal=_t(m["goal"]).view(B, 2),
        C=_t(m["C"]).view(B, N, 2),
        R=_t(m["R"]).view(B, N),
        mask=torch.tensor(m["mask"]).view(B, N) != 0,
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
        ext=MaterialExt(F=_t(m["F_ext"]).view(B, S, 2)),
    )


@pytest.mark.parametrize("sect", ["sharp", "unlogged"])
def test_material_books_match_libcvc(sect):
    m = _golden()[sect]
    B, S = m["B"], max(m["H"])
    sharp = "residual_sharp" in m
    energy: dict = {"books": True, "sharp": sharp}
    oT, vT, *_ = integrate_surrogate_material(**_material_case(m), energy=energy)
    _close(oT, m["oT"], rtol=5e-5, atol=1e-6, what="oT")
    _close(vT, m["vT"], rtol=5e-5, atol=1e-6, what="vT")
    _close(energy["potential"], m["potential"], rtol=5e-5, atol=1e-6, what="potential")
    assert energy["kinetic"].shape == (B, S + 1)
    _close(energy["kinetic"], m["kinetic"], rtol=5e-5, atol=1e-6, what="kinetic")
    scale = _d(m["bound"])
    for name in BOOKS + (("residual_sharp",) if sharp else ()):
        assert energy[name].shape == (B, S) and energy[name].dtype == torch.float64, name
        _close(energy[name], m[name], rtol=1e-5, scale=scale, what=name)
    assert ("residual_sharp" in energy) == sharp
    active = torch.arange(S) < torch.tensor(m["H"]).unsqueeze(-1)
    r, bound = energy["residual"], energy["bound"]
    assert (r.abs() <= bound).all(), "the booked port closes every step"
    assert (energy["w_clamp"] == 0).all() and (energy["w_cross"] == 0).all()
    for name in BOOKS + (("residual_sharp",) if sharp else ()):
        assert (energy[name][~active] == 0).all(), f"{name}: a frozen step books zero"
    assert (energy["w_port"][active] != 0).all()


def test_an_unlogged_port_raises_the_alarm():
    """Booked, F_ext's work closes each step within the bound; left out (r_E + W_port), it is
    work of order dt against a bound of order dt^2, and every step alarms, as in libcvc."""
    m = _golden()["unlogged"]
    energy: dict = {"books": True}
    integrate_surrogate_material(**_material_case(m), energy=energy)
    r, wp, bound = energy["residual"], energy["w_port"], energy["bound"]
    alarm = ((r + wp).abs() > bound).flatten().to(torch.int64)
    assert alarm.tolist() == m["alarm"]
    assert alarm.all()
    assert (r.abs() <= bound).all()


def test_the_books_leave_the_rollout_unchanged():
    case = _material_case(_golden()["sharp"])
    plain = integrate_surrogate_material(**case)
    booked = integrate_surrogate_material(**case, energy={"books": True, "sharp": True})
    for a, b in zip(plain, booked):
        assert torch.equal(a, b)


def test_the_twins_residual_is_second_order():
    """No golden: one step from the sharp case's start as dt halves 0.02 -> 0.0025, each agent's
    r_E / dt^2 holds its constant (libcvc's NavPotential.SurrogateEnergyIdentityPerStep on the
    twin; measured within 1% per halving)."""
    case = _material_case(_golden()["sharp"])
    B = case["o0"].shape[0]
    case["H"] = torch.ones(B, dtype=torch.int64)
    case["ext"] = MaterialExt(F=case["ext"].F[:, :1].clone())
    c2 = []
    for dt in (0.02, 0.01, 0.005, 0.0025):
        case["dt"] = torch.full((B,), dt)
        energy: dict = {"books": True}
        integrate_surrogate_material(**case, energy=energy)
        r = energy["residual"][:, 0]
        assert (r.abs() <= energy["bound"][:, 0]).all()
        c2.append(r / _f32(dt) ** 2)
    for a, b in zip(c2, c2[1:]):
        q = a / b
        assert ((q > 0.95) & (q < 1.05)).all(), f"r_E / dt^2 per halving: {q.tolist()}"


def test_the_sharp_check_is_exact_on_a_quadratic_potential():
    """No golden: with only the goal spring (no discs, no patch forces) U is quadratic, its
    Hessian is constant, and the semi-implicit step meets the sharp check to round-off, where r_E
    itself reaches a sizeable part of the bound."""
    case = _material_case(_golden()["sharp"])
    case["mask"] = torch.zeros_like(case["mask"])
    case["lam_soft"] = torch.zeros_like(case["lam_soft"])
    case["lam_hard"] = torch.zeros_like(case["lam_hard"])
    energy: dict = {"sharp": True}
    integrate_surrogate_material(**case, energy=energy)
    active = torch.arange(energy["bound"].shape[1]) < case["H"].unsqueeze(-1)
    r, rs, bound = (energy[k][active] for k in ("residual", "residual_sharp", "bound"))
    assert ((r - rs).abs() <= 1e-9 * bound).all(), f"{((r - rs).abs() / bound).max().item():.2e}"
    assert (r.abs() / bound).max() > 0.2
