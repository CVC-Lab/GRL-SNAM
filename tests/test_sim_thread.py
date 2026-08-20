"""The pure-C++ threading layer (cvc::nav::sim_thread) — run the swarm off the
render thread (roadmap P7).

The C++ worker never holds the GIL (it runs only cvc::nav kernels), so it
advances genuinely concurrently with Python — even a tight Python read loop can't
starve it (unlike the Python SimThread). Verified: concurrent stepping, lock-free
non-tearing reads, a live retarget the agent reacts to, pause/resume, clean stop.
"""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import sdf_nav  # noqa: E402
from grl_snam import nav_native, planner  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402
from grl_snam.squad import AgentSpec  # noqa: E402
from grl_snam.swarm import Swarm  # noqa: E402
from grl_snam.tools import coef_export  # noqa: E402

pytestmark = pytest.mark.skipif(
    not nav_native.HAS_SIM_THREAD, reason="pycvc build lacks nav_sim_thread_create"
)


def _native_world(tmp_path, n=800, grid=96):
    story = shrunk(STORIES["city"], n=grid, max_steps=10_000_000)
    truth = story.truth_grid()
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    rng = np.random.default_rng(1)
    s, g = rng.integers(0, len(rows), n), rng.integers(0, len(rows), n)
    specs = [
        AgentSpec(f"a{i}", w(rows[s[i]], cols[s[i]]), w(rows[g[i]], cols[g[i]])) for i in range(n)
    ]
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP().eval()
    sw = Swarm(
        story, specs, model=m, truth_occ=truth, prior_occ=truth, belief_mode="shared", nsub=2
    )
    wp = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(wp))
    cw = nav_native.sim_world_from_swarm(sw, wp, truth=truth, freeze_sense=True)
    return cw, sw


def test_sim_thread_runs_concurrently_and_never_tears(tmp_path):
    cw, sw = _native_world(tmp_path)
    sim = nav_native.NativeSimThread(cw, hz=120.0)
    sim.start()
    try:
        time.sleep(0.1)
        torn, reads, last_tick = 0, 0, -1
        t_end = time.perf_counter() + 1.0
        while time.perf_counter() < t_end:
            f = sim.read()
            if f is not None:
                pos, hd, sp, md, rc, tick = f
                if len(pos) != cw.n or len(hd) != cw.n or tick < last_tick:
                    torn += 1
                last_tick = tick
            reads += 1
            time.sleep(0.002)
        assert torn == 0, "frames must never tear or go backwards"
        assert sim.ticks > 20, f"the C++ sim must advance concurrently (got {sim.ticks} ticks)"
    finally:
        sim.stop()
    assert sim.ticks > 0


def test_sim_thread_live_retarget_reacts(tmp_path):
    cw, sw = _native_world(tmp_path, n=400)
    sim = nav_native.NativeSimThread(cw, hz=200.0)
    sim.start()
    try:
        time.sleep(0.1)
        new_gn = -sw.goal[0].detach().cpu().numpy()
        new_gw = np.array([new_gn[0] / sw.S + sw.cx, new_gn[1] / sw.S + sw.cy])
        f0 = sim.read()
        d0 = float(np.hypot(*(f0[0][0] - new_gw)))
        sim.retarget(0, float(new_gn[0]), float(new_gn[1]))
        time.sleep(1.0)
        f1 = sim.read()
        d1 = float(np.hypot(*(f1[0][0] - new_gw)))
        assert d1 < d0 - 1.0, (d0, d1)
    finally:
        sim.stop()


def test_sim_thread_pause_resume(tmp_path):
    cw, _ = _native_world(tmp_path, n=200)
    sim = nav_native.NativeSimThread(cw, hz=200.0)
    sim.start()
    try:
        time.sleep(0.1)
        sim.set_paused(True)
        time.sleep(0.05)
        a = sim.ticks
        time.sleep(0.15)
        assert sim.ticks == a, "paused sim must not advance"
        sim.set_paused(False)
        time.sleep(0.15)
        assert sim.ticks > a, "resumed sim must advance"
    finally:
        sim.stop()
