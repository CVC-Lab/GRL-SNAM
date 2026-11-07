"""Parity for the dynamic single-track vehicle in port-Hamiltonian form, and its step books.

libcvc's ``dyn_ph_parts_at``, ``dyn_ph_rates_at`` and ``dyn_plant_books``, differentiated with
forward-mode duals, against the torch twins in :mod:`grl_snam.dyn_bike`, differentiated by
autograd. The reference is libcvc's ``src/cvc/tests/data/dyn_bike_ph_golden.json`` (copied to
``tests/fixtures``; written by libcvc's ``DynBikePH.Golden`` with
``CVC_DYN_BIKE_PH_GOLDEN_WRITE=1``): the Humvee of ``dyn_bike_golden.json`` at its seven states
(cruising, a hard corner, trail braking, past the tire's peak, backing up, creeping, pulling
away from rest) and braking to a stop at the speed clamp. Each case records the parts and the
rates at its state; one step's next state, the twist it reached before its limiters, and its
books; and the gradient of a fixed weighting of the books on the state and the commands.

Tier: DOUBLE contract, as ``test_dyn_bike_parity.py``: both sides run in double from the same
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
    dyn_ph_hamiltonian,
    dyn_ph_parts_at,
    dyn_ph_rates_at,
    dyn_plant,
    dyn_plant_books,
)

GOLDEN = Path(__file__).parent / "fixtures" / "dyn_bike_ph_golden.json"
RTOL = 1e-9
BOOKS = ("dH", "dissipated", "supplied", "floor", "grade", "limiter", "residual")


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-dyn-bike-ph-golden-1"
    assert tuple(data["order"]) == ORDER
    return data


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


def _limits(c) -> PlantLimits:
    lim = c["limits"]
    return PlantLimits(
        L=lim["L"], vmax=lim["vmax"], hdt=lim["hdt"], allow_reverse=bool(lim["allow_reverse"])
    )


CASES = range(len(_golden()["cases"]))


@pytest.mark.parametrize("case", CASES)
def test_parts_and_rates_match_libcvc(case):
    data = _golden()
    p = DynBikeParams(**data["params"])
    c = data["cases"][case]
    state = torch.tensor(c["state"], dtype=torch.float64)
    q = dyn_ph_parts_at(p, state)
    for name, want in c["parts"].items():
        got = getattr(q, name).item()
        assert _close(got, want), f"{c['name']} parts.{name}: torch {got!r}, libcvc {want!r}"
    rates = dyn_ph_rates_at(p, state, c["grade"])
    for name, want in c["rates"].items():
        got = float(getattr(rates, name))
        assert _close(got, want), f"{c['name']} rates.{name}: torch {got!r}, libcvc {want!r}"


@pytest.mark.parametrize("case", CASES)
def test_books_and_their_gradient_match_libcvc(case):
    data = _golden()
    p = DynBikeParams(**data["params"])
    c = data["cases"][case]
    limits = _limits(c)
    state = torch.tensor(c["state"], dtype=torch.float64, requires_grad=True)
    dcmd = torch.tensor(c["delta_cmd"], dtype=torch.float64, requires_grad=True)
    acmd = torch.tensor(c["a_cmd"], dtype=torch.float64, requires_grad=True)
    nxt, books = dyn_plant_books(p, limits, state, dcmd, acmd, c["grade"])
    for k, name in enumerate(ORDER):
        assert _close(
            nxt[k].item(), c["next"][k]
        ), f"{c['name']} next {name}: torch {nxt[k].item()!r}, libcvc {c['next'][k]!r}"
    _, free = dyn_plant(p, limits, state, dcmd, acmd, c["grade"], with_free=True)
    for k, name in enumerate(("vx", "vy", "r")):
        got, want = free[k].item(), c["free"][name]
        assert _close(got, want), f"{c['name']} free.{name}: torch {got!r}, libcvc {want!r}"
    for name in BOOKS:
        got, want = getattr(books, name).item(), c["books"][name]
        assert _close(got, want), f"{c['name']} books.{name}: torch {got!r}, libcvc {want!r}"
    weighted = sum(w * getattr(books, name) for w, name in zip(data["g_books"], BOOKS))
    weighted.backward()
    for k, name in enumerate(ORDER):
        got, want = state.grad[k].item(), c["g_state"][k]
        assert _close(got, want), f"{c['name']} d/d{name}: torch {got!r}, libcvc {want!r}"
    assert _close(dcmd.grad.item(), c["g_delta_cmd"]), (dcmd.grad.item(), c["g_delta_cmd"])
    assert _close(acmd.grad.item(), c["g_a_cmd"]), (acmd.grad.item(), c["g_a_cmd"])


def test_a_batch_books_like_its_rows():
    data = _golden()
    p = DynBikeParams(**data["params"])
    rows = [c for c in data["cases"] if not c["limits"]["allow_reverse"]]
    limits = _limits(rows[0])
    state = torch.tensor([c["state"] for c in rows], dtype=torch.float64)
    dcmd = torch.tensor([c["delta_cmd"] for c in rows], dtype=torch.float64)
    acmd = torch.tensor([c["a_cmd"] for c in rows], dtype=torch.float64)
    grade = torch.tensor([c["grade"] for c in rows], dtype=torch.float64)
    _, books = dyn_plant_books(p, limits, state, dcmd, acmd, grade)
    rates = dyn_ph_rates_at(p, state, grade)
    for i, c in enumerate(rows):
        for name in BOOKS:
            assert _close(getattr(books, name)[i].item(), c["books"][name]), (c["name"], name)
        for name, want in c["rates"].items():
            assert _close(getattr(rates, name)[i].item(), want), (c["name"], name)


def test_books_close_to_second_order():
    # The behaviour gate needs only torch, as libcvc's DynBikePH.StepBooksCloseToSecondOrder: on
    # a swerve with the throttle working, every step's books close but for the integrator's
    # error, which falls by about 4 per halving of the step, and the run's books close to a
    # small fraction of the energy that flowed.
    data = _golden()
    p = DynBikeParams(**data["params"])
    seconds = 2.0
    means = []
    for hdt in (0.01, 0.005):
        limits = PlantLimits(L=p.lf + p.lr, vmax=40.0, hdt=hdt)
        s = torch.tensor([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        h0 = dyn_ph_hamiltonian(p, s).item()
        n = round(seconds / hdt)
        abs_residual = residual = flow = 0.0
        for t in range(n):
            time = t * hdt
            dcmd = torch.tensor(0.3 * math.sin(2.0 * math.pi * 0.7 * time), dtype=torch.float64)
            acmd = torch.tensor(1.5 * math.sin(2.0 * math.pi * time / 3.0), dtype=torch.float64)
            s, b = dyn_plant_books(p, limits, s, dcmd, acmd)
            abs_residual += abs(b.residual.item())
            residual += b.residual.item()
            flow += sum(abs(getattr(b, name).item()) for name in BOOKS[1:6])
        dH = dyn_ph_hamiltonian(p, s).item() - h0
        assert abs(residual) / flow < 5e-3, (hdt, residual, flow, dH)
        means.append(abs_residual / n)
    ratio = means[0] / means[1]
    assert 3.0 < ratio < 5.5, means
