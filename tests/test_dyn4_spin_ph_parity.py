"""Parity for the four-wheel vehicle with its wheels spinning, its port-Hamiltonian form and its
step books.

libcvc's ``dyn4_spin_ph_parts_at``, ``dyn4_spin_ph_rates_at``, ``dyn4_spin_plant`` and
``dyn4_spin_plant_books``, differentiated with forward-mode duals, against the torch twins in
:mod:`grl_snam.dyn4`, differentiated by autograd. The reference is libcvc's
``src/cvc/tests/data/dyn4_spin_ph_golden.json`` (copied to ``tests/fixtures``; written by libcvc's
``Dyn4SpinPH.Golden`` with ``CVC_DYN4_SPIN_PH_GOLDEN_WRITE=1``): the Humvee with its springs on flat
ground (mu 0.9), with ``dyn4_golden.json``'s lags, at seven states (cruising, a hard corner, trail
braking with the fronts locking, a wheelspin launch, a locked skid, pulling away from rest,
braking while backing up). Each case records each wheel's longitudinal and lateral force, its
patch's ground speed, slip velocity, longitudinal secant and torque, and the rates at its state;
one step's next state, the twist it reached before its limiters, and its books; and the gradient
of a fixed weighting of the books on the state and the commands.

Tier: DOUBLE contract, as ``test_dyn4_parity.py``: both sides run in double from the same float
parameters, so every value agrees to 1e-9 relative.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from grl_snam.dyn4 import (  # noqa: E402
    SPIN_ORDER,
    Dyn4Params,
    FlatGround,
    PlantLimits,
    dyn4_spin_ph_parts_at,
    dyn4_spin_ph_rates_at,
    dyn4_spin_plant,
    dyn4_spin_plant_books,
)

GOLDEN = Path(__file__).parent / "fixtures" / "dyn4_spin_ph_golden.json"
RTOL = 1e-9
BOOKS = (
    "dH",
    "dissipated",
    "slip",
    "supplied",
    "braked",
    "lever",
    "grade",
    "ground",
    "limiter",
    "residual",
)
# The golden's per-wheel keys, and the parts' fields they are.
WHEELS = {"Fx": "Fx", "Fy": "Fy", "ground": "g", "sigma": "sig", "K": "K", "u": "u"}
GROUPS = ("wheels", "rates", "next", "free", "books", "grads")


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-dyn4-spin-ph-golden-1"
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


def _compare(data, c):
    """Torch's value beside libcvc's for everything the case records, by group, and whether the
    books' step is the plant's to the last bit."""
    p = Dyn4Params(**data["params"])
    ground = FlatGround(data["ground"]["mu"])
    limits = _limits(c)
    out = {g: [] for g in GROUPS}
    s = torch.tensor(c["state"], dtype=torch.float64)
    q = dyn4_spin_ph_parts_at(p, limits, s, ground)
    for key, field in WHEELS.items():
        got = getattr(q, field)
        out["wheels"] += [(f"{key}[{i}]", got[i].item(), c["wheels"][key][i]) for i in range(4)]
    rates = dyn4_spin_ph_rates_at(p, limits, s, ground)
    out["rates"] = [(k, float(getattr(rates, k)), want) for k, want in c["rates"].items()]
    dc = torch.tensor(c["delta_cmd"], dtype=torch.float64)
    ac = torch.tensor(c["a_cmd"], dtype=torch.float64)
    nxt, free = dyn4_spin_plant(p, limits, s, dc, ac, ground, with_free=True)
    out["next"] = [(k, nxt[i].item(), c["next"][i]) for i, k in enumerate(SPIN_ORDER)]
    out["free"] = [(k, free[i].item(), c["free"][k]) for i, k in enumerate(("vx", "vy", "r"))]
    state = s.clone().requires_grad_(True)
    dcmd = dc.clone().requires_grad_(True)
    acmd = ac.clone().requires_grad_(True)
    nxt_b, books = dyn4_spin_plant_books(p, limits, state, dcmd, acmd, ground)
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
def test_wheels_and_rates_match_libcvc(case):
    data = _golden()
    c = data["cases"][case]
    out, _ = _compare(data, c)
    for group in ("wheels", "rates"):
        for name, got, want in out[group]:
            assert (
                _rel(got, want) <= RTOL
            ), f"{c['name']} {group}.{name}: torch {got!r}, libcvc {want!r}"


@pytest.mark.parametrize("case", CASES)
def test_step_books_and_their_gradient_match_libcvc(case):
    data = _golden()
    c = data["cases"][case]
    out, same_step = _compare(data, c)
    assert same_step, f"{c['name']}: the books' step is not dyn4_spin_plant's"
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
    p = Dyn4Params(**data["params"])
    ground = FlatGround(data["ground"]["mu"])
    rows = [c for c in data["cases"] if not c["limits"]["allow_reverse"]]
    limits = _limits(rows[0])
    state = torch.tensor([c["state"] for c in rows], dtype=torch.float64)
    dcmd = torch.tensor([c["delta_cmd"] for c in rows], dtype=torch.float64)
    acmd = torch.tensor([c["a_cmd"] for c in rows], dtype=torch.float64)
    nxt, books = dyn4_spin_plant_books(p, limits, state, dcmd, acmd, ground)
    rates = dyn4_spin_ph_rates_at(p, limits, state, ground)
    for i, c in enumerate(rows):
        for k in range(len(SPIN_ORDER)):
            assert _rel(nxt[i, k].item(), c["next"][k]) <= RTOL, (c["name"], SPIN_ORDER[k])
        for name in BOOKS:
            assert _rel(getattr(books, name)[i].item(), c["books"][name]) <= RTOL, (c["name"], name)
        for name, want in c["rates"].items():
            assert _rel(getattr(rates, name)[i].item(), want) <= RTOL, (c["name"], name)


def test_grade_force_is_refused():
    # libcvc's opt-in grade_force (gravity along the ground's slope, dyn4_grade) has no twin yet:
    # a plant that asks for it must not silently step without it.
    data = _golden()
    p = Dyn4Params(**{**data["params"], "grade_force": 1})
    c = data["cases"][0]
    s = torch.tensor(c["state"], dtype=torch.float64)
    cmd = torch.tensor(0.0, dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        dyn4_spin_plant(p, _limits(c), s, cmd, cmd)
