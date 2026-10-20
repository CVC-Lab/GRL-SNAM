"""Parity for the four-wheel vehicle with suspension (libcvc P8d).

libcvc's ``cvc::nav::diff::dyn4_plant``, differentiated with forward-mode duals, against the torch
twin :func:`grl_snam.dyn4.dyn4_plant`, differentiated by autograd. The reference is libcvc's
``src/cvc/tests/data/dyn4_golden.json`` (copied to ``tests/fixtures``; written by libcvc's
``Dyn4.Golden`` with ``CVC_DYN4_GOLDEN_WRITE=1``): eleven cases of the Humvee, with its springs and
again with rigid corners, on a friction split with a round bump. The cases cover cruising, a hard
corner, braking in a turn, a wheel on the bump, braking across the split, creeping, backing up,
past the tires' peak, and pulling away from rest. Each case records the next state and the VJP of
a fixed output gradient, so the ground's gradients are held too.

Tier: DOUBLE contract. Both sides run in double from the same float parameters, so the forward
and the gradient agree to 1e-9 relative. A libm or torch kernel differs only in the last bits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from grl_snam.dyn4 import (  # noqa: E402
    ORDER,
    Dyn4Params,
    FlatGround,
    PlantLimits,
    ackermann,
    dyn4_plant,
)

GOLDEN = Path(__file__).parent / "fixtures" / "dyn4_golden.json"
RTOL = 1e-9


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-dyn4-golden-1"
    assert tuple(data["order"]) == ORDER
    return data


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


def _ground(gd):
    """libcvc's golden_ground: the friction split along y = 0 and a round bump."""

    def at(wx, wy):
        t = 0.5 + 0.5 * torch.tanh(wy / gd["split_width"])
        mu = gd["mu_right"] + t * (gd["mu_left"] - gd["mu_right"])
        dx, dy = wx - gd["cx"], wy - gd["cy"]
        return mu, gd["A"] * torch.exp(-(dx * dx + dy * dy) / (gd["r"] * gd["r"]))

    return at


def _limits(c) -> PlantLimits:
    lim = c["limits"]
    return PlantLimits(
        L=lim["L"], vmax=lim["vmax"], hdt=lim["hdt"], allow_reverse=bool(lim["allow_reverse"])
    )


@pytest.mark.parametrize("case", range(len(_golden()["cases"])))
def test_next_state_and_gradient_match_libcvc(case):
    data = _golden()
    c = data["cases"][case]
    p = Dyn4Params(**data["params"][c["params"]])
    state = torch.tensor(c["state"], dtype=torch.float64, requires_grad=True)
    dcmd = torch.tensor(c["delta_cmd"], dtype=torch.float64, requires_grad=True)
    acmd = torch.tensor(c["a_cmd"], dtype=torch.float64, requires_grad=True)
    nxt = dyn4_plant(p, _limits(c), state, dcmd, acmd, _ground(data["ground"]))
    for k, name in enumerate(ORDER):
        assert _close(
            nxt[k].item(), c["next"][k]
        ), f"case {case} next {name}: torch {nxt[k].item()!r}, libcvc {c['next'][k]!r}"
    g_out = torch.tensor(data["g_out"], dtype=torch.float64)
    (nxt * g_out).sum().backward()
    for k, name in enumerate(ORDER):
        assert _close(
            state.grad[k].item(), c["g_state"][k]
        ), f"case {case} d/d{name}: torch {state.grad[k].item()!r}, libcvc {c['g_state'][k]!r}"
    assert _close(dcmd.grad.item(), c["g_delta_cmd"])
    assert _close(acmd.grad.item(), c["g_a_cmd"])


def test_a_batch_steps_like_its_rows():
    data = _golden()
    rows = [c for c in data["cases"] if c["params"] == 0 and not c["limits"]["allow_reverse"]]
    p = Dyn4Params(**data["params"][0])
    state = torch.tensor([c["state"] for c in rows], dtype=torch.float64)
    dcmd = torch.tensor([c["delta_cmd"] for c in rows], dtype=torch.float64)
    acmd = torch.tensor([c["a_cmd"] for c in rows], dtype=torch.float64)
    nxt = dyn4_plant(p, _limits(rows[0]), state, dcmd, acmd, _ground(data["ground"]))
    for i, c in enumerate(rows):
        for k in range(len(ORDER)):
            assert _close(nxt[i, k].item(), c["next"][k])


def test_the_front_wheels_follow_ackermann():
    # libcvc's P2 gate, as Jolt's Humvee meets it: cot(right) - cot(left) = track / wheelbase.
    p = Dyn4Params(**_golden()["params"][0])
    L = p.lf + p.lr
    for delta in (0.05, 0.2, 0.45, -0.3):
        t = torch.tan(torch.tensor(delta, dtype=torch.float64))
        left, right = ackermann(L, p.tf, t), ackermann(L, -p.tf, t)
        cot = 1.0 / torch.tan(right) - 1.0 / torch.tan(left)
        assert abs(cot.item() - 2.0 * p.tf / L) < 1e-12


def test_braking_from_speed_stops_near_the_friction_limit():
    # The behaviour gate needs only torch: from 20 m/s, braking far past the tires' grip on flat
    # ground, the sprung Humvee stops within 10% of v^2 / (2 mu g), as libcvc's Dyn4 gate requires.
    p = Dyn4Params(**_golden()["params"][0])
    mu = 0.8
    limits = PlantLimits(L=p.lf + p.lr, vmax=40.0, hdt=0.06)
    s = torch.zeros(14, dtype=torch.float64)
    s[3] = 20.0
    dist = 0.0
    for _ in range(2000):
        if s[3].item() <= 0.01:
            break
        n = dyn4_plant(
            p,
            limits,
            s,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(-30.0, dtype=torch.float64),
            FlatGround(mu),
        )
        dist += torch.hypot(n[0] - s[0], n[1] - s[1]).item()
        s = n
    ideal = 20.0**2 / (2.0 * mu * p.g)
    assert abs(dist / ideal - 1.0) < 0.10, (dist, ideal)
