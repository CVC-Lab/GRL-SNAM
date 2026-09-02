"""SdfNavigator native drive dispatch (``GRL_SNAM_NAV_DRIVE=native``).

The per-agent navigator the DBG convoy drives (Squad -> Scenario ->
``SdfNavigator``) can optionally run the torch-free C++ fused drive
(``nav_native.drive_step``) instead of the torch coef-net + bicycle rollout,
keeping the carrot FSM in Python. This is the SdfNavigator assembly of the
drive_step parity (test_drive_step_parity.py) and the same wiring the Swarm uses
(test_swarm_native_drive.py) -- so a native navigator must trace the torch
navigator to float32 tolerance over a long roll. Torch stays the default and the
reference; ``native`` is the opt-in a real-time host uses.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import sdf_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402
from grl_snam.nav import SdfNavigator  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (nav_native.HAS_DRIVE and hasattr(nav_native._pycvc, "nav_drive_step")),
    reason="pycvc build lacks nav_drive_step",
)


def _scene():
    story = shrunk(STORIES["city"], n=96, max_steps=100)
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(story.truth_grid(), story.bounds, meta["scale"])
    sf = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    return sf, story.bounds, meta


def _model():
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _nav(native, sf, meta, start, goal):
    prev = os.environ.get("GRL_SNAM_NAV_DRIVE")
    os.environ["GRL_SNAM_NAV_DRIVE"] = "native" if native else "torch"
    try:
        nav = SdfNavigator(sf, _model(), meta, dynamics="bicycle")
    finally:
        if prev is None:
            os.environ.pop("GRL_SNAM_NAV_DRIVE", None)
        else:
            os.environ["GRL_SNAM_NAV_DRIVE"] = prev
    nav.start(start, goal)
    return nav


def _endpoints(bounds):
    mnx, mny, mxx, mxy = bounds
    start = (mnx + 0.2 * (mxx - mnx), mny + 0.2 * (mxy - mny))
    goal = (mnx + 0.8 * (mxx - mnx), mny + 0.8 * (mxy - mny))
    return start, goal


def test_flag_off_by_default():
    """No env => torch drive; the native path must never engage implicitly."""
    sf, bounds, meta = _scene()
    start, goal = _endpoints(bounds)
    nav = _nav(False, sf, meta, start, goal)
    assert nav._native_drive is False
    assert nav._native_weights_path is None


def test_native_navigator_traces_torch():
    """The native SdfNavigator drive stays locked to the torch drive tick-for-tick.

    This is a WIRING check -- it proves ``__init__``'s weight export and ``step()``'s
    bounds/center/scale/params/nsub threading hand the C++ fused drive what the torch
    path computes. The tight per-step kernel parity is test_drive_step_parity's job
    (rtol=1e-4 over 3000 random states); here we only need the SdfNavigator assembly
    to track torch across a real navigating trajectory.

    We re-sync the native navigator to the torch one *before every tick* and bound the
    single-step position divergence, rather than racing two free 120-tick
    trajectories. The toolchain-dependent ~1-ULP ``grid_sample`` residual
    (test_sdf_sample_parity: bit-exact on the dev x86-64 build, ~1 ULP on the CI
    runner) is amplified near obstacle boundaries -- exactly where a navigating agent
    spends its time, and unlike the random-in-box states the kernel gate samples -- so
    a free rollout would compound it into a chaotic split (a lone corner-to-corner
    agent WILL cross a steering junction). Re-synced, each tick is an independent
    wiring check. The 2e-2 bound clears that amplified residual with margin; a
    mis-threaded param (frame, scale, nsub) diverges by an order of magnitude or more.
    """
    sf, bounds, meta = _scene()
    start, goal = _endpoints(bounds)
    a = _nav(False, sf, meta, start, goal)  # torch reference (free-running)
    b = _nav(True, sf, meta, start, goal)  # native, same scene / carrot / policy
    assert b._native_drive is True

    start_o = a.o.clone()
    maxerr = 0.0
    for _ in range(120):
        # start each tick from identical state so the only difference is the drive
        # kernel (native C++ vs torch), never divergent carrot feedback
        b.o, b.th, b.sp = a.o.clone(), a.th.clone(), a.sp.clone()
        a.step()
        b.step()
        maxerr = max(maxerr, (a.o - b.o).abs().max().item())

    assert (a.o - start_o).abs().max().item() > 0.1  # the run actually navigated
    assert maxerr < 2e-2, f"native drive diverged from torch tick-for-tick: {maxerr:.3e}"
