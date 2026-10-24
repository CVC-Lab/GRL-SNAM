"""grl_snam.ext_rollout.dyn4_rollout_ext against libcvc's diff::dyn4_rollout_ext.

The reference is libcvc's ``src/cvc/tests/data/dyn4_ext_rollout_golden.json`` (copied to
``tests/fixtures``; written by ``Dyn4ExtRollout.Golden`` with ``CVC_DYN4_EXT_GOLDEN_WRITE=1``):
six agents' 10-step trajectories of the four-wheel vehicle on a friction split with a bump, with
per-step goals and coefficients and an affine force each (F = A o + b) held for 1 or 3 steps, one
that does not steer. Four saturate the controller (their coefficient gradients are zero); two run
inside its clamps, so the coefficients and the force move its commands (libcvc#798). It records the field they sample and the C++ reverse for
L = sum_t w_t . state_t: dL/dF_ext per step, dL/dcoef per step and dL/d(initial state). The C++
rollout is float; float64 autograd here is the exact gradient of the same math.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam import sdf_nav  # noqa: E402
from grl_snam.dyn4 import ORDER, Dyn4Params  # noqa: E402
from grl_snam.ext_rollout import dyn4_rollout_ext, field_as  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "dyn4_ext_rollout_golden.json"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def golden():
    g = json.loads(GOLDEN.read_text())
    assert g["format"] == "cvc-dyn4-ext-rollout-golden-1"
    assert tuple(g["order"]) == ORDER
    return g


def _field(g, dtype, device):
    f = g["field"]
    H, W = f["H"], f["W"]
    planes = [np.asarray(f[k], np.float32).reshape(H, W) for k in ("phi", "nx", "ny")]
    fld = sdf_nav.SDFField(*planes, f["bounds"], f["center"], f["scale"], device=device)
    return field_as(fld, dtype=dtype)


def _ground(gd):
    """libcvc's ext_ground: the friction split along y = 0 and a round bump."""

    def at(wx, wy):
        t = 0.5 + 0.5 * torch.tanh(wy / gd["split_width"])
        mu = gd["mu_right"] + t * (gd["mu_left"] - gd["mu_right"])
        dx, dy = wx - gd["cx"], wy - gd["cy"]
        return mu, gd["A"] * torch.exp(-(dx * dx + dy * dy) / (gd["r"] * gd["r"]))

    return at


def _run(g, case, dtype, device):
    """One golden case as a batch of 1: (states [T+1,14], dL/dF [T,2], dL/dcoef [T,3],
    dL/dinit [14])."""
    T = g["T"]
    s0 = torch.zeros(1, 14, dtype=dtype, device=device)
    s0[0, :4] = torch.tensor(case["state0"], dtype=dtype, device=device)
    s0.requires_grad_(True)
    goals = torch.tensor(case["goals"], dtype=dtype, device=device).view(T, 1, 2)
    coefs = torch.tensor(case["coefs"], dtype=dtype, device=device).view(T, 1, 3)
    coefs.requires_grad_(True)
    A = torch.tensor(case["A"], dtype=dtype, device=device).view(2, 2)
    b = torch.tensor(case["b"], dtype=dtype, device=device)
    off = torch.zeros(T, 1, 2, dtype=dtype, device=device, requires_grad=True)
    veh = {
        k: g["veh"][k]
        for k in ("rr", "d_hat", "hdt", "vmax", "L", "delta_max", "a_max", "a_lat_max", "k_steer")
    }
    states = dyn4_rollout_ext(
        _field(g, dtype, device),
        _ground(g["ground"]),
        s0,
        goals,
        coefs,
        lambda t, p: p @ A.T + b,
        veh=veh,
        params=Dyn4Params(**g["params"]),
        hold=case["hold"],
        steer=bool(case["steer"]),
        allow_reverse=bool(g["veh"]["allow_reverse"]),
        f_offsets=off,
    )[:, 0, :]
    w = torch.tensor(g["w"], dtype=dtype, device=device).view(T + 1, 14)
    (w * states).sum().backward()
    return states.detach(), off.grad[:, 0, :], coefs.grad[:, 0, :], s0.grad[0]


def _rel(a, b):
    """max |a - b| relative to the larger magnitude (at least 1)."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("case", range(6))
def test_trajectory_and_reverse_match_libcvc(golden, case, dev):
    c = golden["cases"][case]
    states, g_f, g_c, g_i = _run(golden, c, torch.float64, dev)
    r = {
        "states": _rel(states.cpu(), c["states"]),
        "dL/dF": _rel(g_f.cpu(), c["g_fext"]),
        "dL/dcoef": _rel(g_c.cpu(), c["g_coef"]),
        "dL/dinit": _rel(g_i.cpu(), c["g_init"]),
    }
    print(f"case {case} ({dev}): " + ", ".join(f"{k} {v:.2e}" for k, v in r.items()))
    assert r["states"] < 1e-4, r
    for k in ("dL/dF", "dL/dcoef", "dL/dinit"):
        assert r[k] < 1e-3, (k, r)


def test_a_batch_steps_like_its_rows(golden):
    # The cases with hold 1 that steer share a rollout's settings: batch them.
    rows = [c for c in golden["cases"] if c["hold"] == 1 and c["steer"] == 1]
    T = golden["T"]
    s0 = torch.zeros(len(rows), 14, dtype=torch.float64)
    for i, c in enumerate(rows):
        s0[i, :4] = torch.tensor(c["state0"], dtype=torch.float64)
    goals = torch.stack([torch.tensor(c["goals"], dtype=torch.float64).view(T, 2) for c in rows], 1)
    coefs = torch.stack([torch.tensor(c["coefs"], dtype=torch.float64).view(T, 3) for c in rows], 1)
    A = torch.stack([torch.tensor(c["A"], dtype=torch.float64).view(2, 2) for c in rows])
    b = torch.stack([torch.tensor(c["b"], dtype=torch.float64) for c in rows])
    veh = {
        k: golden["veh"][k]
        for k in ("rr", "d_hat", "hdt", "vmax", "L", "delta_max", "a_max", "a_lat_max", "k_steer")
    }
    states = dyn4_rollout_ext(
        _field(golden, torch.float64, "cpu"),
        _ground(golden["ground"]),
        s0,
        goals,
        coefs,
        lambda t, p: torch.einsum("bij,bj->bi", A, p) + b,
        veh=veh,
        params=Dyn4Params(**golden["params"]),
        allow_reverse=bool(golden["veh"]["allow_reverse"]),
    )
    for i, c in enumerate(rows):
        assert _rel(states[:, i, :], c["states"]) < 1e-4
