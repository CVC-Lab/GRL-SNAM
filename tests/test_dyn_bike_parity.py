"""Parity for the dynamic single-track vehicle (libcvc P8a).

libcvc's ``cvc::nav::diff::dyn_plant``, differentiated with forward-mode duals, against the torch
twin :func:`grl_snam.dyn_bike.dyn_plant`, differentiated by autograd. The reference is libcvc's
``src/cvc/tests/data/dyn_bike_golden.json`` (copied to ``tests/fixtures``; written by libcvc's
``DynBike.Golden`` with ``CVC_DYN_BIKE_GOLDEN_WRITE=1``): seven cases of a Humvee with steering
and drive lags and the low-speed blend: cruising, a hard corner, braking in a turn, past the
tire's peak, backing up, creeping, and pulling away from rest. Each case records the next
state and the VJP of a fixed output gradient.

Tier: DOUBLE contract. Both sides run in double from the same float parameters, so the forward
and the gradient agree to 1e-9 relative. A libm or torch kernel differs only in the last bits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from grl_snam.dyn_bike import ORDER, DynBikeParams, PlantLimits, dyn_plant  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "dyn_bike_golden.json"
RTOL = 1e-9


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-dyn-bike-golden-1"
    assert tuple(data["order"]) == ORDER
    return data


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


@pytest.mark.parametrize("case", range(len(_golden()["cases"])))
def test_next_state_and_gradient_match_libcvc(case):
    data = _golden()
    p = DynBikeParams(**data["params"])
    c = data["cases"][case]
    lim = c["limits"]
    limits = PlantLimits(
        L=lim["L"], vmax=lim["vmax"], hdt=lim["hdt"], allow_reverse=bool(lim["allow_reverse"])
    )
    state = torch.tensor(c["state"], dtype=torch.float64, requires_grad=True)
    dcmd = torch.tensor(c["delta_cmd"], dtype=torch.float64, requires_grad=True)
    acmd = torch.tensor(c["a_cmd"], dtype=torch.float64, requires_grad=True)
    nxt = dyn_plant(p, limits, state, dcmd, acmd, c["grade"])
    for k, name in enumerate(ORDER):
        assert _close(nxt[k].item(), c["next"][k]), (
            f"case {case} next {name}: torch {nxt[k].item()!r}, libcvc {c['next'][k]!r}"
        )
    g_out = torch.tensor(data["g_out"], dtype=torch.float64)
    (nxt * g_out).sum().backward()
    for k, name in enumerate(ORDER):
        assert _close(state.grad[k].item(), c["g_state"][k]), (
            f"case {case} d/d{name}: torch {state.grad[k].item()!r}, libcvc {c['g_state'][k]!r}"
        )
    assert _close(dcmd.grad.item(), c["g_delta_cmd"])
    assert _close(acmd.grad.item(), c["g_a_cmd"])


def test_a_batch_steps_like_its_rows():
    data = _golden()
    p = DynBikeParams(**data["params"])
    rows = [c for c in data["cases"] if not c["limits"]["allow_reverse"]]
    lim = rows[0]["limits"]
    limits = PlantLimits(L=lim["L"], vmax=lim["vmax"], hdt=lim["hdt"])
    state = torch.tensor([c["state"] for c in rows], dtype=torch.float64)
    dcmd = torch.tensor([c["delta_cmd"] for c in rows], dtype=torch.float64)
    acmd = torch.tensor([c["a_cmd"] for c in rows], dtype=torch.float64)
    grade = torch.tensor([c["grade"] for c in rows], dtype=torch.float64)
    nxt = dyn_plant(p, limits, state, dcmd, acmd, grade)
    for i, c in enumerate(rows):
        for k in range(len(ORDER)):
            assert _close(nxt[i, k].item(), c["next"][k])


def test_braking_from_speed_stops_near_the_friction_limit():
    # The behaviour gate needs only torch: from 20 m/s, braking far past the tires' grip, the
    # Humvee stops within 10% of v^2 / (2 mu g), as libcvc's DynBike gate requires.
    data = _golden()
    p = DynBikeParams(**data["params"])
    limits = PlantLimits(L=p.lf + p.lr, vmax=40.0, hdt=0.06)
    s = torch.tensor([0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    dist = 0.0
    for _ in range(2000):
        if s[3].item() <= 1e-3:
            break
        n = dyn_plant(p, limits, s, torch.tensor(0.0, dtype=torch.float64),
                      torch.tensor(-2.0 * p.mu * p.g, dtype=torch.float64))
        dist += torch.hypot(n[0] - s[0], n[1] - s[1]).item()
        s = n
    ideal = 20.0 ** 2 / (2.0 * p.mu * p.g)
    assert abs(dist / ideal - 1.0) < 0.10, (dist, ideal)
