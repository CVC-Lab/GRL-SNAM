"""SdfNavigator.ext_force_fn plumbing — byte-identical when inert, live when set.

#66 added the generic ext_force_fn hook to sdf_rollout / bicycle_rollout; this
pins that SdfNavigator threads it through step() correctly: a None or zero-force
hook leaves every trace bit-for-bit unchanged (the byte-identical-off contract
the DBG comm-force cutover relies on), and a non-zero hook actually bends the
path.
"""

import numpy as np
import torch

import sdf_nav
from grl_snam.nav import SdfNavigator


def _open_field():
    n = 64
    phi = np.full((n, n), 5.0, np.float32)
    zeros = np.zeros((n, n), np.float32)
    bounds = (-100.0, -100.0, 100.0, 100.0)
    scale = 0.05
    field = sdf_nav.SDFField(phi, zeros, zeros, bounds, (0.0, 0.0), scale)
    meta = dict(
        scale=scale,
        center=(0.0, 0.0),
        region=100.0,
        rr=0.15,
        d_hat=0.35,
        dt=0.06,
        nsub=2,
        vmax=0.9,
        bounds=list(bounds),
    )
    return field, meta


def _trace(dynamics, ext_force_fn, steps=25):
    field, meta = _open_field()
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP()
    model.eval()
    nav = SdfNavigator(field, model, meta, reach_tol=0.05, dynamics=dynamics)
    nav.ext_force_fn = ext_force_fn
    nav.start((-50.0, 0.0), (50.0, 0.0))
    out = []
    for _ in range(steps):
        nav.step()
        out.append(nav.o.clone())
    return torch.stack(out)


def test_ext_force_none_and_zero_are_byte_identical():
    for dyn in ("point", "bicycle"):
        none_trace = _trace(dyn, None)
        zero_trace = _trace(dyn, lambda o: torch.zeros_like(o))
        assert torch.equal(none_trace, zero_trace), f"{dyn}: zero-force hook must be inert"


def test_ext_force_changes_the_trajectory():
    for dyn in ("point", "bicycle"):
        none_trace = _trace(dyn, None)
        # A constant northward push (rollout/normalized frame) must bend the path.
        push = _trace(dyn, lambda o: torch.tensor([[0.0, 0.3]]).expand_as(o).contiguous())
        assert not torch.equal(none_trace, push), f"{dyn}: a non-zero force must change the trace"
        # And specifically push the final y positive relative to the unforced run.
        assert push[-1][0, 1] > none_trace[-1][0, 1], f"{dyn}: northward force should raise final y"
