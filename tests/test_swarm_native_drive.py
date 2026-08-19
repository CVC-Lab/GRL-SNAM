"""In-Swarm native drive dispatch (``GRL_SNAM_NAV_DRIVE=native``).

The Swarm can optionally drive through the torch-free C++ path (``cvc::nav``)
instead of the torch coef-net + bicycle rollout, keeping the carrot FSM in
Python. This is the whole-Swarm assembly of the drive_step parity
(test_drive_step_parity.py) — same field, same carrot, same policy — so a native
Swarm must trace the torch Swarm to float32 tolerance and reach the exact same
agents (docs/CVCNAV_CPP_PORT_ROADMAP.md P8). Torch stays the default and the
reference; ``native`` is the opt-in a pure-C++ host would use.

The two swarms share a *frozen* field (sense disabled) so the only moving part is
the drive: any divergence is the drive, not a belief/rebuild difference.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import sdf_nav  # noqa: E402
from grl_snam import nav_native, planner  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402
from grl_snam.squad import AgentSpec  # noqa: E402
from grl_snam.swarm import Swarm  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (nav_native.HAS_DRIVE and hasattr(nav_native._pycvc, "nav_drive_step")),
    reason="pycvc build lacks nav_drive_step",
)


def _story(n=96):
    return shrunk(STORIES["city"], n=n, max_steps=10_000_000)


def _specs(story, truth, n, seed=0):
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    rng = np.random.default_rng(seed)
    s = rng.integers(0, len(rows), n)
    g = rng.integers(0, len(rows), n)
    return [
        AgentSpec(f"a{i}", w(rows[s[i]], cols[s[i]]), w(rows[g[i]], cols[g[i]])) for i in range(n)
    ]


def _model():
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _build(native, *, belief_mode="shared", clusters=None, n=64, seed=0):
    story = _story()
    truth = story.truth_grid()
    specs = _specs(story, truth, n, seed=seed)
    prev = os.environ.get("GRL_SNAM_NAV_DRIVE")
    os.environ["GRL_SNAM_NAV_DRIVE"] = "native" if native else "torch"
    try:
        sw = Swarm(
            story,
            specs,
            model=_model(),
            truth_occ=truth,
            prior_occ=truth,
            belief_mode=belief_mode,
            clusters=clusters,
        )
    finally:
        if prev is None:
            os.environ.pop("GRL_SNAM_NAV_DRIVE", None)
        else:
            os.environ["GRL_SNAM_NAV_DRIVE"] = prev
    sw._sense_shared = lambda: None  # freeze the field: isolate the drive
    return sw


def test_flag_off_by_default():
    """No env => torch drive; the native path must never engage implicitly."""
    sw = _build(native=False)
    assert sw._native_drive is False
    assert sw._native_weights_path is None


def test_native_drive_traces_torch_shared():
    """Shared belief: a native Swarm follows the torch Swarm to f32 tolerance and
    reaches the exact same agents over a long roll."""
    a = _build(native=False)
    b = _build(native=True)
    assert b._native_drive is True

    ticks = 120
    maxerr = 0.0
    for _ in range(ticks):
        a.step()
        b.step()
        maxerr = max(maxerr, (a.o - b.o).abs().max().item())

    assert maxerr < 1e-3, f"native drive diverged from torch: {maxerr:.3e}"
    assert torch.equal(
        a.reached, b.reached
    ), f"reach-set differs: {int((a.reached != b.reached).sum())} agents"
    assert int(a.reached.sum()) > 0  # the scene is actually solvable (not a trivial pass)


def test_native_drive_runs_grouped():
    """The map_id path (clustered belief, M>1) drives without error and moves
    agents — a smoke test that per-plane field selection works natively."""
    b = _build(native=True, belief_mode="clustered", clusters=4)
    assert b._native_drive is True and b.M > 1
    start = b.o.clone()
    for _ in range(20):
        b.step()
    assert (b.o - start).abs().max().item() > 0  # agents moved
