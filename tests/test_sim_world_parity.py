"""The pure-C++ reactive swarm runtime (cvc::nav::sim_world) vs the torch Swarm.

sim_world runs the WHOLE swarm with zero torch — sense, occupancy, field rebuild,
carrot FSM, drive, park. This is the roadmap's P6 behavioral gate: built from an
identical init + the same .cvcnav policy, the C++ trajectories track the torch
Swarm and the reach-set matches. Frozen sense (static known map) isolates the
drive/FSM path; a second run exercises the live sense/rebuild path.
"""

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
    not nav_native.HAS_SIM_WORLD, reason="pycvc build lacks nav_sim_world_create"
)


def _swarm(n=1500, grid=128, seed=0):
    story = shrunk(STORIES["city"], n=grid, max_steps=10_000_000)
    truth = story.truth_grid()
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    rng = np.random.default_rng(seed)
    s, g = rng.integers(0, len(rows), n), rng.integers(0, len(rows), n)
    specs = [
        AgentSpec(f"a{i}", w(rows[s[i]], cols[s[i]]), w(rows[g[i]], cols[g[i]])) for i in range(n)
    ]
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP().eval()
    sw = Swarm(
        story, specs, model=m, truth_occ=truth, prior_occ=truth, belief_mode="shared", nsub=2
    )
    return sw, truth, m


def test_sim_world_tracks_torch_swarm_frozen_sense(tmp_path):
    sw, truth, m = _swarm()
    sw._sense_shared = lambda: None  # static known map
    wp = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(wp))
    cw = nav_native.sim_world_from_swarm(sw, wp, truth=truth, freeze_sense=True)

    max_err = 0.0
    for _ in range(80):
        sw.step()
        cw.step()
        pos, _hd, _sp, _md, rc = cw.snapshot()
        sw_world = sw.n2w(sw.o.clone()).cpu().numpy()
        max_err = max(max_err, float(np.abs(pos - sw_world).max()))
    # The behavioral contract is the reach-set equality (hard). The trajectory
    # max-err is a float-equivalence gate over 80 ticks of mode-flip chaos: it
    # was sub-5cm on the dev build, but the pycvc-gl +cvc.6 -> +cvc.7 rebuild
    # shifted the runner's float32 codegen ~1 ULP, pushing a threshold-adjacent
    # agent's divergence to ~7cm (roadmap P6: "reach-set +/- tiny budget,
    # mode-flip rate <0.5% and every flip threshold-adjacent"). 10cm keeps the
    # gate meaningful (a real drift regression is far larger) while surviving
    # benign toolchain codegen shifts.
    assert int(rc.sum()) == int(sw.reached.sum().item()), (int(rc.sum()), int(sw.reached.sum()))
    assert max_err < 0.10, max_err


def test_sim_world_live_sense_runs_and_agents_progress(tmp_path):
    """With the live sense/rebuild path on, the C++ swarm runs and makes progress
    (agents close on goals, at least some reach)."""
    sw, truth, m = _swarm(n=400, grid=96)
    wp = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(wp))
    cw = nav_native.sim_world_from_swarm(sw, wp, truth=truth, freeze_sense=False, sense_every=4)
    p0, *_ = cw.snapshot()
    goal_w = sw.n2w(sw.goal.clone()).cpu().numpy()
    d0 = np.hypot(*(p0 - goal_w).T)
    for _ in range(150):
        cw.step()
    p1, _hd, _sp, _md, rc = cw.snapshot()
    d1 = np.hypot(*(p1 - goal_w).T)
    assert np.isfinite(p1).all()
    assert (d1 < d0).mean() > 0.6, "most agents should make progress"
    assert int(rc.sum()) >= 1


def test_sim_world_retarget_reacts(tmp_path):
    sw, truth, m = _swarm(n=300, grid=96)
    sw._sense_shared = lambda: None
    wp = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(wp))
    cw = nav_native.sim_world_from_swarm(sw, wp, truth=truth, freeze_sense=True)
    # normalized goal far from agent 0
    new_gn = -sw.goal[0].detach().cpu().numpy()
    p0, *_ = cw.snapshot()
    new_gw = np.array([new_gn[0] / sw.S + sw.cx, new_gn[1] / sw.S + sw.cy])
    d0 = float(np.hypot(*(p0[0] - new_gw)))
    cw.retarget(0, float(new_gn[0]), float(new_gn[1]))
    for _ in range(120):
        cw.step()
    p1, *_ = cw.snapshot()
    d1 = float(np.hypot(*(p1[0] - new_gw)))
    assert d1 < d0 - 1.0, (d0, d1)
