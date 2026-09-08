"""ext_force_fn parity — the generic external-force channel in sdf_rollout and
bicycle_rollout (the Python twin of cvc::nav's ext_force port). A ``None`` hook
(the default) and a zero force are byte-identical to the plain rollout; a live
force bends the trace. This is what carries the DBG comm force once it moves off
the carrot bias: F_comm/F_jam sum with F_bar/F_goal in the integrator.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sdf_nav  # noqa: E402

BOUNDS = (-100.0, -100.0, 100.0, 100.0)
CENTER = (0.0, 0.0)
SCALE = 0.05


def _world():
    rng = np.random.default_rng(7)
    n_grid = 64
    occ = np.zeros((n_grid, n_grid), bool)
    occ[20:30, 30:34] = True
    phi, nx_g, ny_g = sdf_nav.build_sdf(occ, BOUNDS, SCALE)
    field = sdf_nav.SDFField(phi, nx_g, ny_g, BOUNDS, CENTER, SCALE)
    N = 48
    o = ((rng.random((N, 2)) * 8.0) - 4.0).astype(np.float32)
    th = (rng.random(N).astype(np.float32) * 6.0) - 3.0
    sp = rng.random(N).astype(np.float32) * 0.8
    goal = ((rng.random((N, 2)) * 8.0) - 4.0).astype(np.float32)
    al = rng.random(N).astype(np.float32) * 2
    be = rng.random(N).astype(np.float32) * 4
    ga = rng.random(N).astype(np.float32) * 4
    return field, o, th, sp, goal, al, be, ga


def _bike(w, **kw):
    field, o, th, sp, goal, al, be, ga = w
    return sdf_nav.bicycle_rollout(
        field,
        torch.from_numpy(o.copy()),
        torch.from_numpy(th.copy()),
        torch.from_numpy(sp.copy()),
        torch.from_numpy(goal),
        torch.from_numpy(al),
        torch.from_numpy(be),
        torch.from_numpy(ga),
        3,
        rr=0.15,
        d_hat=0.35,
        dt=0.06,
        vmax=0.9,
        allow_reverse=True,
        **kw,
    )


def _sdf(w, **kw):
    field, o, th, sp, goal, al, be, ga = w
    v = torch.zeros((o.shape[0], 2), dtype=torch.float32)
    return sdf_nav.sdf_rollout(
        field,
        torch.from_numpy(o.copy()),
        v,
        torch.from_numpy(goal),
        torch.from_numpy(al),
        torch.from_numpy(be),
        torch.from_numpy(ga),
        3,
        rr=0.15,
        d_hat=0.35,
        dt=0.06,
        **kw,
    )


def _zero(o):
    return torch.zeros_like(o)


def _west(o):
    f = torch.zeros_like(o)
    f[:, 0] = -2.0
    return f


def test_bicycle_ext_force_none_and_zero_are_byte_identical():
    w = _world()
    base = _bike(w)
    for hook in (None, _zero):
        got = _bike(w, ext_force_fn=hook)
        for a, b in zip(base[:3], got[:3]):
            assert torch.equal(a, b), f"hook={hook}"


def test_bicycle_ext_force_changes_trajectory():
    w = _world()
    assert not torch.equal(_bike(w)[0], _bike(w, ext_force_fn=_west)[0])


def test_sdf_ext_force_none_and_zero_identical_and_force_changes():
    w = _world()
    base = _sdf(w)
    assert torch.equal(base[0], _sdf(w, ext_force_fn=None)[0])
    assert torch.equal(base[0], _sdf(w, ext_force_fn=_zero)[0])
    assert not torch.equal(base[0], _sdf(w, ext_force_fn=_west)[0])
