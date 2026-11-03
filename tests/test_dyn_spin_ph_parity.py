"""Parity for the wheel-spin single-track vehicle, its port-Hamiltonian form and its step books.

libcvc's ``dyn_spin_ph_parts_at``, ``dyn_spin_ph_rates_at``, ``dyn_spin_plant`` and
``dyn_spin_plant_books``, differentiated with forward-mode duals, against the torch twins in
:mod:`grl_snam.dyn_bike`, differentiated by autograd. The reference is libcvc's
``src/cvc/tests/data/dyn_spin_ph_golden.json`` (copied to ``tests/fixtures``; written by libcvc's
``DynSpinPH.Golden`` with ``CVC_DYN_SPIN_PH_GOLDEN_WRITE=1``): the Humvee of
``dyn_bike_ph_golden.json`` with its wheels, at nine states (cruising, a hard corner on a grade,
trail braking with the front locking, a wheelspin launch, past the tire's peak, a locked skid,
creeping, pulling away from rest, braking while backing up). Each case records the parts and the
rates at its state; one step's next state, the twist it reached before its limiters, and its
books; and the gradient of a fixed weighting of the books on the state and the commands.

Tier: DOUBLE contract, as ``test_dyn_bike_ph_parity.py``: both sides run in double from the same
float parameters, so every value agrees to 1e-9 relative.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from grl_snam.dyn_bike import (  # noqa: E402
    SPIN_ORDER,
    DynBikeParams,
    PlantLimits,
    dyn_spin_ph_parts_at,
    dyn_spin_ph_rates_at,
    dyn_spin_plant,
    dyn_spin_plant_books,
)

GOLDEN = Path(__file__).parent / "fixtures" / "dyn_spin_ph_golden.json"
RTOL = 1e-9
BOOKS = ("dH", "dissipated", "slip", "supplied", "braked", "floor", "grade", "limiter", "residual")
GROUPS = ("parts", "rates", "next", "free", "books", "grads")


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-dyn-spin-ph-golden-1"
    assert tuple(data["order"]) == SPIN_ORDER
    return data


def _rel(a: float, b: float) -> float:
    """|a - b| relative to the larger magnitude (at least 1)."""
    return abs(a - b) / max(1.0, abs(a), abs(b))


def _limits(c) -> PlantLimits:
    lim = c["limits"]
    return PlantLimits(
        L=lim["L"], vmax=lim["vmax"], hdt=lim["hdt"], allow_reverse=bool(lim["allow_reverse"])
    )


def _part(q, name: str) -> float:
    """A golden part by its key: dyn_ph_parts' in ``q.b``; Td_f .. Tb_r, the torque pairs."""
    if name[:3] in ("Td_", "Tb_"):
        return getattr(q, name[:2])[..., "fr".index(name[3])].item()
    return getattr(q.b if hasattr(q.b, name) else q, name).item()


def _compare(data, c):
    """Torch's value beside libcvc's for everything the case records, by group, and whether the
    books' step is the plant's to the last bit."""
    p = DynBikeParams(**data["params"])
    limits = _limits(c)
    out = {g: [] for g in GROUPS}
    s = torch.tensor(c["state"], dtype=torch.float64)
    q = dyn_spin_ph_parts_at(p, s)
    out["parts"] = [(k, _part(q, k), want) for k, want in c["parts"].items()]
    rates = dyn_spin_ph_rates_at(p, s, c["grade"])
    out["rates"] = [(k, float(getattr(rates, k)), want) for k, want in c["rates"].items()]
    dc = torch.tensor(c["delta_cmd"], dtype=torch.float64)
    ac = torch.tensor(c["a_cmd"], dtype=torch.float64)
    nxt, free = dyn_spin_plant(p, limits, s, dc, ac, c["grade"], with_free=True)
    out["next"] = [(k, nxt[i].item(), c["next"][i]) for i, k in enumerate(SPIN_ORDER)]
    out["free"] = [(k, free[i].item(), c["free"][k]) for i, k in enumerate(("vx", "vy", "r"))]
    state = s.clone().requires_grad_(True)
    dcmd = dc.clone().requires_grad_(True)
    acmd = ac.clone().requires_grad_(True)
    nxt_b, books = dyn_spin_plant_books(p, limits, state, dcmd, acmd, c["grade"])
    out["books"] = [(k, getattr(books, k).item(), c["books"][k]) for k in BOOKS]
    weighted = sum(w * getattr(books, k) for w, k in zip(data["g_books"], BOOKS))
    weighted.backward()
    out["grads"] = [
        (f"d/d{k}", state.grad[i].item(), c["g_state"][i]) for i, k in enumerate(SPIN_ORDER)
    ]
    out["grads"].append(("d/ddelta_cmd", dcmd.grad.item(), c["g_delta_cmd"]))
    out["grads"].append(("d/da_cmd", acmd.grad.item(), c["g_a_cmd"]))
    return out, torch.equal(nxt_b.detach(), nxt)


CASES = range(len(_golden()["cases"]))


@pytest.mark.parametrize("case", CASES)
def test_parts_and_rates_match_libcvc(case):
    data = _golden()
    c = data["cases"][case]
    out, _ = _compare(data, c)
    for group in ("parts", "rates"):
        for name, got, want in out[group]:
            assert (
                _rel(got, want) <= RTOL
            ), f"{c['name']} {group}.{name}: torch {got!r}, libcvc {want!r}"


@pytest.mark.parametrize("case", CASES)
def test_step_books_and_their_gradient_match_libcvc(case):
    data = _golden()
    c = data["cases"][case]
    out, same_step = _compare(data, c)
    assert same_step, f"{c['name']}: the books' step is not dyn_spin_plant's"
    for group in ("next", "free", "books", "grads"):
        for name, got, want in out[group]:
            assert (
                _rel(got, want) <= RTOL
            ), f"{c['name']} {group} {name}: torch {got!r}, libcvc {want!r}"


def test_worst_difference_per_group():
    data = _golden()
    worst = {g: (0.0, "") for g in GROUPS}
    for c in data["cases"]:
        out, _ = _compare(data, c)
        for g in GROUPS:
            for name, got, want in out[g]:
                if _rel(got, want) > worst[g][0]:
                    worst[g] = (_rel(got, want), f"{c['name']} {name}")
    print("torch twin against libcvc's duals (max relative, where):")
    for g in GROUPS:
        print(f"  {g:6s} {worst[g][0]:.2e}  {worst[g][1]}")
    for g in GROUPS:
        assert worst[g][0] <= RTOL, (g, worst[g])


def test_a_batch_books_like_its_rows():
    data = _golden()
    p = DynBikeParams(**data["params"])
    rows = [c for c in data["cases"] if not c["limits"]["allow_reverse"]]
    limits = _limits(rows[0])
    state = torch.tensor([c["state"] for c in rows], dtype=torch.float64)
    dcmd = torch.tensor([c["delta_cmd"] for c in rows], dtype=torch.float64)
    acmd = torch.tensor([c["a_cmd"] for c in rows], dtype=torch.float64)
    grade = torch.tensor([c["grade"] for c in rows], dtype=torch.float64)
    nxt, books = dyn_spin_plant_books(p, limits, state, dcmd, acmd, grade)
    rates = dyn_spin_ph_rates_at(p, state, grade)
    for i, c in enumerate(rows):
        for k in range(len(SPIN_ORDER)):
            assert _rel(nxt[i, k].item(), c["next"][k]) <= RTOL, (c["name"], SPIN_ORDER[k])
        for name in BOOKS:
            assert _rel(getattr(books, name)[i].item(), c["books"][name]) <= RTOL, (c["name"], name)
        for name, want in c["rates"].items():
            assert _rel(getattr(rates, name)[i].item(), want) <= RTOL, (c["name"], name)
