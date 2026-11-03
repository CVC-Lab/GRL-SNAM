"""Parity for the dynamic single-track vehicle's full (q, p) port-Hamiltonian form on a ground.

libcvc's ``dyn_ph_T``, ``dyn_ph_rhs_full``, the ground overloads (the slopes, the potential,
``dV/dq``, the rates) and ``dyn_plant_books_ground``, differentiated with forward-mode duals
through the ground, against the torch twins in :mod:`grl_snam.dyn_bike`, differentiated by
autograd. The reference is libcvc's ``src/cvc/tests/data/dyn_ph_full_golden.json`` (copied to
``tests/fixtures``; written by libcvc's ``DynPHFull.Golden`` with
``CVC_DYN_PH_FULL_GOLDEN_WRITE=1``): the Humvee of ``dyn_bike_golden.json`` on a hillside (its
formula and coefficients in the file) at six states across, up and down the slope, sliding,
creeping and reversing. Each case records the slopes, ``V`` and ``H`` at its state, ``dV/dq`` and
``T(q)``, the full field on the ground and with the case's grade and side slope as data, the
rates on the ground; one step's next state, the twist it reached before its limiters, and its
books (the grade line ``-dV``); and the gradient of a fixed weighting of the books on the state
and the commands.

Tier: DOUBLE contract, as ``test_dyn_bike_ph_parity.py``: both sides run in double from the same
float parameters, so every value agrees to 1e-9 relative.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from grl_snam.dyn_bike import (  # noqa: E402
    ORDER,
    DynBikeParams,
    PlantLimits,
    dyn_ph_grad_q,
    dyn_ph_hamiltonian_ground,
    dyn_ph_potential,
    dyn_ph_rates_ground,
    dyn_ph_rhs,
    dyn_ph_rhs_full,
    dyn_ph_slope,
    dyn_ph_T,
    dyn_plant,
    dyn_plant_books,
    dyn_plant_books_ground,
    dyn_plant_ground,
)

GOLDEN = Path(__file__).parent / "fixtures" / "dyn_ph_full_golden.json"
RTOL = 1e-9
BOOKS = ("dH", "dissipated", "supplied", "floor", "grade", "limiter", "residual")
RATES = ("dissipated", "supplied", "floor", "grade")


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-dyn-ph-full-golden-1"
    assert tuple(data["order"]) == ORDER
    return data


class Hillside:
    """The golden's ground, ``sx (x - x0) + sy (y - y0) + A sin(kx (x - x0)) cos(ky (y - y0))``,
    as libcvc's test samples it."""

    def __init__(self, g):
        assert g["kind"] == "hillside"
        self.sx, self.sy, self.A = g["sx"], g["sy"], g["A"]
        self.kx, self.ky, self.x0, self.y0 = g["kx"], g["ky"], g["x0"], g["y0"]

    def __call__(self, x, y):
        dx = x - self.x0
        dy = y - self.y0
        return (
            self.sx * dx + self.sy * dy + self.A * torch.sin(self.kx * dx) * torch.cos(self.ky * dy)
        )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


def _limits(c) -> PlantLimits:
    lim = c["limits"]
    return PlantLimits(
        L=lim["L"], vmax=lim["vmax"], hdt=lim["hdt"], allow_reverse=bool(lim["allow_reverse"])
    )


def _check(label: str, got, want) -> None:
    got = [float(v) for v in torch.as_tensor(got).reshape(-1)]
    assert len(got) == len(want), (label, len(got), len(want))
    for k, (g, w) in enumerate(zip(got, want)):
        assert _close(g, w), f"{label}[{k}]: torch {g!r}, libcvc {w!r}"


CASES = range(len(_golden()["cases"]))


@pytest.mark.parametrize("case", CASES)
def test_the_form_on_the_ground_matches_libcvc(case):
    data = _golden()
    p = DynBikeParams(**data["params"])
    ground = Hillside(data["ground"])
    c = data["cases"][case]
    name = c["name"]
    state = torch.tensor(c["state"], dtype=torch.float64)
    along, across = dyn_ph_slope(p, ground, state)
    got = {
        "along": along,
        "across": across,
        "V": dyn_ph_potential(p, ground, state),
        "H": dyn_ph_hamiltonian_ground(p, ground, state),
    }
    for key, want in c["ground"].items():
        _check(f"{name} ground.{key}", got[key], [want])
    _check(f"{name} grad_q", dyn_ph_grad_q(p, ground, state), c["grad_q"])
    _check(f"{name} T", dyn_ph_T(p, state[2]), c["T"])
    qd, pd = dyn_ph_rhs_full(p, state, ground=ground)
    _check(f"{name} qdot", qd, c["qdot"])
    _check(f"{name} pdot", pd, c["pdot"])
    _, pdd = dyn_ph_rhs_full(p, state, c["grade"], c["side"])
    _check(f"{name} pdot_data", pdd, c["pdot_data"])
    rates = dyn_ph_rates_ground(p, ground, state)
    for key in RATES:
        _check(f"{name} rates.{key}", getattr(rates, key), [c["rates"][key]])


@pytest.mark.parametrize("case", CASES)
def test_the_step_its_books_and_their_gradient_match_libcvc(case):
    data = _golden()
    p = DynBikeParams(**data["params"])
    ground = Hillside(data["ground"])
    c = data["cases"][case]
    name = c["name"]
    limits = _limits(c)
    state = torch.tensor(c["state"], dtype=torch.float64, requires_grad=True)
    dcmd = torch.tensor(c["delta_cmd"], dtype=torch.float64, requires_grad=True)
    acmd = torch.tensor(c["a_cmd"], dtype=torch.float64, requires_grad=True)
    nxt, books = dyn_plant_books_ground(p, limits, ground, state, dcmd, acmd)
    _check(f"{name} next", nxt.detach(), c["next"])
    _, free = dyn_plant_ground(p, limits, ground, state, dcmd, acmd, with_free=True)
    _check(f"{name} free", free.detach(), [c["free"][k] for k in ("vx", "vy", "r")])
    for key in BOOKS:
        _check(f"{name} books.{key}", getattr(books, key).detach(), [c["books"][key]])
    weighted = sum(w * getattr(books, key) for w, key in zip(data["g_books"], BOOKS))
    weighted.backward()
    _check(f"{name} g_state", state.grad, c["g_state"])
    _check(f"{name} g_delta_cmd", dcmd.grad, [c["g_delta_cmd"]])
    _check(f"{name} g_a_cmd", acmd.grad, [c["g_a_cmd"]])


def _structure(p: DynBikeParams, x: torch.Tensor) -> torch.Tensor:
    """The full structure ``[[0, T(q)], [-T(q)^T, J(p)]]`` ``(6, 6)`` at x = (q, p)."""
    T = dyn_ph_T(p, x[2])
    px, py = x[3], x[4]
    z = torch.zeros_like(px)
    Jp = torch.stack((torch.stack((z, z, py)), torch.stack((z, z, -px)), torch.stack((-py, px, z))))
    top = torch.cat((torch.zeros(3, 3, dtype=x.dtype), T), dim=1)
    bottom = torch.cat((-T.transpose(0, 1), Jp), dim=1)
    return torch.cat((top, bottom), dim=0)


def test_the_full_structure_is_skew_and_poisson():
    # Torch alone: the structure is skew, moves no energy, and its Jacobi identity holds (by
    # autograd), as libcvc's DynPHFull.TheStructureIsSkewAndPoisson checks by differences.
    data = _golden()
    p = DynBikeParams(**data["params"])
    ground = Hillside(data["ground"])
    gen = torch.Generator().manual_seed(3)
    for _ in range(20):
        u = 2.0 * torch.rand(6, generator=gen, dtype=torch.float64) - 1.0
        state = torch.tensor(
            [30 * u[0], 30 * u[1], 3 * u[2], 15 + 10 * u[3], 3 * u[4], u[5], 0.2, 1.0],
            dtype=torch.float64,
        )
        x = torch.stack(
            (state[0], state[1], state[2], p.m * state[3], p.m * state[4], p.Iz * state[5])
        )
        J = _structure(p, x)
        assert torch.equal(J, -J.T)
        e = torch.cat((dyn_ph_grad_q(p, ground, state), state[3:6]))
        assert abs(float(e @ J @ e)) <= 1e-12 * float(e.abs() @ J.abs() @ e.abs())
        dJ = torch.autograd.functional.jacobian(lambda xx: _structure(p, xx), x)  # (6, 6, 6)
        jac = (
            torch.einsum("il,jkl->ijk", J, dJ)
            + torch.einsum("jl,kil->ijk", J, dJ)
            + torch.einsum("kl,ijl->ijk", J, dJ)
        )
        size = (
            torch.einsum("il,jkl->ijk", J.abs(), dJ.abs())
            + torch.einsum("jl,kil->ijk", J.abs(), dJ.abs())
            + torch.einsum("kl,ijl->ijk", J.abs(), dJ.abs())
        )
        assert float((jac.abs() / size.clamp_min(1e-300)).max()) < 1e-12


def test_the_pose_update_is_T_v():
    # dyn_plant's pose step is q1 = q0 + hdt T(q1) v1 at nav's step, on the ground and flat.
    data = _golden()
    p = DynBikeParams(**data["params"])
    ground = Hillside(data["ground"])
    limits = PlantLimits(L=p.lf + p.lr, vmax=40.0, hdt=0.06, allow_reverse=True)
    state = torch.tensor([c["state"] for c in data["cases"]], dtype=torch.float64)
    dcmd = torch.tensor([c["delta_cmd"] for c in data["cases"]], dtype=torch.float64)
    acmd = torch.tensor([c["a_cmd"] for c in data["cases"]], dtype=torch.float64)
    for nxt in (
        dyn_plant_ground(p, limits, ground, state, dcmd, acmd),
        dyn_plant(p, limits, state, dcmd, acmd),
    ):
        T = dyn_ph_T(p, nxt[..., 2])
        want = limits.hdt * (T @ nxt[..., 3:6].unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(nxt[..., :3] - state[..., :3], want, rtol=0.0, atol=1e-13)


def test_the_data_overload_is_the_momentum_form_without_a_side_slope():
    data = _golden()
    p = DynBikeParams(**data["params"])
    state = torch.tensor([c["state"] for c in data["cases"]], dtype=torch.float64)
    _, pd = dyn_ph_rhs_full(p, state, 0.03, 0.0)
    assert torch.equal(pd, dyn_ph_rhs(p, state, 0.03))


def test_side_slope_books_close_with_gravity_in_h():
    # Torch alone, as libcvc's DynPHFull.SideSlopeBooksCloseWithGravityInH: a driver holds a line
    # across the hillside, the body sliding; with gravity in H every step's books close but for the
    # integrator's error (about 4 smaller per halving of the step), while the same steps booked in
    # the momentum-only form leave the slope across the heading's work unbooked.
    data = _golden()
    p = DynBikeParams(**{**data["params"], "mu": 0.6})
    ground = Hillside(data["ground"])
    seconds = 2.0
    means = []
    for hdt in (0.01, 0.005):
        limits = PlantLimits(L=p.lf + p.lr, vmax=40.0, hdt=hdt)
        s = torch.tensor([-40.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        n = round(seconds / hdt)
        abs_residual = residual = flow = momentum = 0.0
        for t in range(n):
            time = t * hdt
            dcmd = -0.04 * s[1] - 0.6 * s[2] + 0.12 * math.sin(2.0 * math.pi * 0.4 * time)
            dcmd = dcmd.clamp(-0.5, 0.5)
            acmd = (0.8 * (12.0 - s[3])).clamp(-4.0, 3.0)
            nxt, b = dyn_plant_books_ground(p, limits, ground, s, dcmd, acmd)
            along, across = dyn_ph_slope(p, ground, s)
            _, bm = dyn_plant_books(p, limits, s, dcmd, acmd, along, side=across)
            abs_residual += abs(b.residual.item())
            residual += b.residual.item()
            momentum += bm.residual.item()
            flow += sum(abs(getattr(b, name).item()) for name in BOOKS[1:6])
            s = nxt
        assert abs(residual) / flow < 1e-3, (hdt, residual, flow)
        assert abs(momentum) > 30.0 * abs(residual), (hdt, momentum, residual)
        means.append(abs_residual / n)
    ratio = means[0] / means[1]
    assert 3.0 < ratio < 5.5, means
