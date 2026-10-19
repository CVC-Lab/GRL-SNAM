"""grl_snam.ext_rollout against libcvc's diff::bike_rollout_ext (tests/fixtures/ext_rollout_golden.json,
written by libcvc's DiffExtRollout.Golden).

The golden holds six agents' 12-step trajectories with per-step goals and coefficients, an affine
force each (F = A o + b) held for 1 or 3 steps, one that does not steer, the field they sample,
and the C++ reverse for L = sum_t w_t . state_t: dL/dF_ext per step, dL/dcoef per step and
dL/d(initial state). The C++ controller is float, so the trajectories agree to float precision;
float64 autograd here is the exact gradient of the same math, which holds the C++ float reverse
to far tighter than finite differences can.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam import sdf_nav  # noqa: E402
from grl_snam.ext_rollout import bike_rollout_ext, field_as  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "ext_rollout_golden.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def _field(g, dtype, device="cpu"):
    f = g["field"]
    H, W = f["H"], f["W"]
    planes = [np.asarray(f[k], np.float32).reshape(H, W) for k in ("phi", "nx", "ny")]
    fld = sdf_nav.SDFField(*planes, f["bounds"], f["center"], f["scale"], device=device)
    return field_as(fld, dtype=dtype)


def _veh(g):
    return {
        k: g["veh"][k]
        for k in ("rr", "d_hat", "hdt", "vmax", "L", "delta_max", "a_max", "a_lat_max", "k_steer")
    }


def _run(g, case, dtype, device="cpu", grads=True):
    """One golden case as a batch of 1: (states [T+1,4], dL/dF [T,2], dL/dcoef [T,3], dL/dinit [4])."""
    T = g["T"]
    fld = _field(g, dtype, device)
    s0 = torch.tensor(case["state0"], dtype=dtype, device=device)
    o = s0[None, :2].clone().requires_grad_(grads)
    th = s0[None, 2].clone().requires_grad_(grads)
    sp = s0[None, 3].clone().requires_grad_(grads)
    goals = torch.tensor(case["goals"], dtype=dtype, device=device).view(T, 1, 2)
    coefs = (
        torch.tensor(case["coefs"], dtype=dtype, device=device).view(T, 1, 3).requires_grad_(grads)
    )
    A = torch.tensor(case["A"], dtype=dtype, device=device).view(2, 2)
    b = torch.tensor(case["b"], dtype=dtype, device=device)
    off = torch.zeros(T, 1, 2, dtype=dtype, device=device, requires_grad=grads)
    states = bike_rollout_ext(
        fld,
        o,
        th,
        sp,
        goals,
        coefs,
        lambda t, p: p @ A.T + b,
        veh=_veh(g),
        hold=case["hold"],
        steer=bool(case["steer"]),
        allow_reverse=bool(g["veh"]["allow_reverse"]),
        f_offsets=off,
    )[:, 0, :]
    if not grads:
        return states.detach(), None, None, None
    w = torch.tensor(g["w"], dtype=dtype, device=device).view(T + 1, 4)
    (w * states).sum().backward()
    g_init = torch.cat([o.grad[0], th.grad, sp.grad])
    return states.detach(), off.grad[:, 0, :], coefs.grad[:, 0, :], g_init


def _rel(a, b):
    """max |a - b| relative to the larger magnitude (at least 1)."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))


def test_golden_shape(golden):
    assert golden["format"] == "cvc-ext-rollout-golden-1"
    assert len(golden["cases"]) == 6
    assert {c["hold"] for c in golden["cases"]} == {1, 3}
    assert {c["steer"] for c in golden["cases"]} == {0, 1}


def test_float64_replays_the_trajectory_and_the_reverse(golden):
    T = golden["T"]
    worst = {"states": 0.0, "g_fext": 0.0, "g_coef": 0.0, "g_init": 0.0}
    for case in golden["cases"]:
        states, g_fext, g_coef, g_init = _run(golden, case, torch.float64)
        got = {
            "states": states.cpu().numpy(),
            "g_fext": g_fext.cpu().numpy(),
            "g_coef": g_coef.cpu().numpy(),
            "g_init": g_init.detach().cpu().numpy(),
        }
        want = {
            "states": np.asarray(case["states"]).reshape(T + 1, 4),
            "g_fext": np.asarray(case["g_fext"]).reshape(T, 2),
            "g_coef": np.asarray(case["g_coef"]).reshape(T, 3),
            "g_init": np.asarray(case["g_init"]),
        }
        for k in worst:
            worst[k] = max(worst[k], _rel(got[k], want[k]))
    print("float64 twin against the C++ float rollout (max relative):", worst)
    # The C++ controller runs in float32, so its trajectory and its hand-written reverse carry
    # float rounding through 12 steps of clamps, tanh and atan2: measured 2e-7 at worst (2026-10-19,
    # Windows). An adjoint mistake shows at O(1e-2) or worse. 1e-5 leaves room for another
    # platform's torch and libm.
    for k in worst:
        assert worst[k] < 1e-5, (k, worst[k])


def test_float32_matches_the_cpp_trajectory(golden):
    T = golden["T"]
    worst = 0.0
    for case in golden["cases"]:
        states, *_ = _run(golden, case, torch.float32, grads=False)
        worst = max(worst, _rel(states.numpy(), np.asarray(case["states"]).reshape(T + 1, 4)))
    print("float32 twin against the C++ float rollout:", worst)
    assert worst < 1e-5  # measured 3e-8


def test_a_batch_is_its_agents(golden):
    """The six cases of one (hold, steer) as a batch give each case's own trajectory."""
    g = golden
    T = g["T"]
    cases = [c for c in g["cases"] if c["hold"] == 1 and c["steer"] == 1]
    assert len(cases) >= 2
    dt = torch.float64
    fld = _field(g, dt)
    s0 = torch.tensor([c["state0"] for c in cases], dtype=dt)
    goals = (
        torch.tensor([c["goals"] for c in cases], dtype=dt).view(len(cases), T, 2).transpose(0, 1)
    )
    coefs = (
        torch.tensor([c["coefs"] for c in cases], dtype=dt).view(len(cases), T, 3).transpose(0, 1)
    )
    A = torch.tensor([c["A"] for c in cases], dtype=dt).view(-1, 2, 2)
    b = torch.tensor([c["b"] for c in cases], dtype=dt)
    states = bike_rollout_ext(
        fld,
        s0[:, :2],
        s0[:, 2],
        s0[:, 3],
        goals,
        coefs,
        lambda t, p: torch.einsum("bij,bj->bi", A, p) + b,
        veh=_veh(g),
        hold=1,
        allow_reverse=bool(g["veh"]["allow_reverse"]),
    )
    for i, c in enumerate(cases):
        one, *_ = _run(g, c, dt, grads=False)
        assert torch.allclose(states[:, i, :], one, atol=1e-12, rtol=0)


def test_the_force_is_sampled_once_per_held_window_in_order(golden):
    g = golden
    case = next(c for c in g["cases"] if c["hold"] == 3)
    T = g["T"]
    calls = []
    s0 = torch.tensor(case["state0"], dtype=torch.float64)
    bike_rollout_ext(
        _field(g, torch.float64),
        s0[None, :2],
        s0[None, 2],
        s0[None, 3],
        torch.tensor(case["goals"], dtype=torch.float64).view(T, 1, 2),
        torch.tensor(case["coefs"], dtype=torch.float64).view(T, 1, 3),
        lambda t, p: (calls.append(t), torch.zeros_like(p))[1],
        veh=_veh(g),
        hold=3,
    )
    assert calls == list(range(0, T, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu(golden):
    for case in golden["cases"]:
        cpu = _run(golden, case, torch.float64)
        gpu = _run(golden, case, torch.float64, device="cuda")
        for a, b in zip(cpu, gpu):
            assert torch.allclose(a.cpu(), b.detach().cpu(), atol=1e-9, rtol=1e-9)
