"""The vectorized, shared-belief swarm.

Two things are asserted here. First, correctness by reduction: a single agent in
a :class:`~grl_snam.swarm.Swarm` must drive exactly as it does as a serial
:class:`~grl_snam.nav.SdfNavigator` over the same field — the SoA carrot FSM and
batched rollout are a faithful vectorization, matching N serial navigators to
float32 tolerance (the same tier ``Squad(batched_drive=True)`` already accepts).
Second, the sim/render threading contract: lock-free immutable snapshots that
never tear, and live commands (retarget, pause) taking effect.

Unlike :class:`~grl_snam.squad.Squad`, belief here is SHARED — the swarm is the
thousands-of-agents render path, a different simulator from the fog-of-war twin.
"""

import math
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sdf_nav  # noqa: E402
from grl_snam import planner  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402
from grl_snam.nav import SdfNavigator  # noqa: E402
from grl_snam.sim_thread import Pause, RetargetGoal, SimThread  # noqa: E402
from grl_snam.squad import AgentSpec  # noqa: E402
from grl_snam.swarm import Swarm  # noqa: E402


def _story(n=96):
    return shrunk(STORIES["city"], n=n, max_steps=10_000_000)


def _free_specs(story, truth, n, seed=0):
    """`n` agents with starts+goals drawn from the largest free component."""
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


def test_swarm_needs_agents():
    story = _story()
    with pytest.raises(ValueError):
        Swarm(story, [], model=_model())


def test_belief_and_field_are_shared():
    """The whole point of the swarm: ONE belief, ONE field for all N agents
    (the O(1) map that Squad's per-agent belief is not)."""
    story = _story()
    truth = story.truth_grid()
    sw = Swarm(story, _free_specs(story, truth, 8), model=_model(), truth_occ=truth)
    assert sw.field.field.shape[0] == 1, "shared field must be a single [1,3,H,W] texture"
    assert sw.o.shape == (8, 2) and sw.th.shape == (8,)
    # every agent samples that one texture
    phi, nrm = sw.field.sample(sw.o)
    assert phi.shape == (8,) and nrm.shape == (8, 2)


def test_swarm_matches_serial_navigators_to_float32():
    """A faithful vectorization: Swarm (sensing frozen, static known map) vs N
    serial SdfNavigators over the SAME field + model + fixed goals."""
    story = _story(96)
    truth = story.truth_grid()
    specs = _free_specs(story, truth, 20, seed=3)
    m = _model()

    sw = Swarm(story, specs, model=m, truth_occ=truth, prior_occ=truth, sense_every=1)
    sw._sense_shared = lambda: None  # freeze the field -> pure FSM + drive parity
    field = sw.field

    navs = []
    for sp in specs:
        nv = SdfNavigator(field, m, sw.meta, reach_tol=sw.reach_tol, dynamics="bicycle")
        nv.start(sp.start, sp.goal)
        navs.append(nv)

    max_pos_err = max_th_err = 0.0
    for _ in range(150):
        for nv in navs:
            nv.step()
            if nv.reached:
                nv.park()  # mirror the swarm's reached -> park
        sw.step()
        sw_world = sw.n2w(sw.o.clone()).cpu().numpy()
        for i, nv in enumerate(navs):
            pw = nv.pos_world()
            max_pos_err = max(
                max_pos_err, float(np.hypot(sw_world[i, 0] - pw[0], sw_world[i, 1] - pw[1]))
            )
            dth = float(sw.th[i]) - float(nv.th[0])
            max_th_err = max(max_th_err, abs(math.atan2(math.sin(dth), math.cos(dth))))

    # float32 batched-reduction drift, exactly the Squad(batched_drive) tier.
    assert max_pos_err < 5e-3, f"position drift {max_pos_err:.2e} m exceeds float32 tolerance"
    assert max_th_err < 5e-3, f"heading drift {max_th_err:.2e} rad exceeds float32 tolerance"


def test_agents_drive_toward_goals():
    """End-to-end: on a known map, agents close on their goals and stop."""
    story = _story(96)
    truth = story.truth_grid()
    specs = _free_specs(story, truth, 16, seed=5)
    sw = Swarm(story, specs, model=_model(), truth_occ=truth, prior_occ=truth, sense_every=1)
    sw._sense_shared = lambda: None
    d0 = (sw.goal - sw.o).norm(dim=1).clone()
    for _ in range(400):
        sw.step()
    d1 = (sw.goal - sw.o).norm(dim=1)
    assert bool((d1 < d0).float().mean() > 0.7), "most agents should make progress"
    assert int(sw.reached.sum()) >= 1, "at least one agent should reach its goal"


def test_snapshot_is_immutable_and_consistent():
    story = _story()
    truth = story.truth_grid()
    sw = Swarm(story, _free_specs(story, truth, 12), model=_model(), truth_occ=truth)
    snap = sw.snapshot()
    N = sw.N
    assert snap.pos.shape == (N, 2) and snap.heading.shape == (N,)
    assert snap.mode.shape == (N,) and snap.goal.shape == (N, 2)
    # advancing the sim must not mutate an already-taken snapshot (fresh columns)
    pos0 = snap.pos.copy()
    sw.step()
    sw.step()
    assert np.array_equal(snap.pos, pos0), "a published snapshot must never change under the writer"
    assert sw.snapshot().gen > snap.gen


def test_sim_thread_runs_concurrently_and_reacts_to_commands():
    """The threading contract: the sim advances on its own thread, snapshots
    never tear, and a live retarget + pause take effect."""
    story = _story(96)
    truth = story.truth_grid()
    specs = _free_specs(story, truth, 32, seed=7)
    sw = Swarm(story, specs, model=_model(), truth_occ=truth, prior_occ=truth, sense_every=4)
    sw._sense_shared = lambda: None  # isolate the GIL-released drive path

    sim = SimThread(sw, hz=120.0)
    sim.start()
    try:
        time.sleep(0.05)
        # renderer reads at a realistic cadence (yields the GIL between frames)
        torn = 0
        last_gen = -1
        for _ in range(40):
            f = sim.buffer.read()  # lock-free
            if len(f.pos) != sw.N or len(f.heading) != sw.N or f.gen < last_gen:
                torn += 1
            last_gen = f.gen
            time.sleep(0.005)
        assert torn == 0, "snapshots must never tear or go out of bounds"
        assert sim.ticks > 0, "the sim thread must advance concurrently"

        # live retarget: goal column updates on the next drained tick
        target = specs[0].goal
        sim.send(RetargetGoal(0, (target[0], target[1])))
        time.sleep(0.1)
        f = sim.buffer.read()
        err = float(np.hypot(f.goal[0, 0] - target[0], f.goal[0, 1] - target[1]))
        assert err < 1e-3, f"retarget command not applied (goal error {err:.3f} m)"

        # pause freezes the tick counter
        sim.send(Pause(True))
        time.sleep(0.05)
        a = sim.buffer.read().tick
        time.sleep(0.1)
        assert sim.buffer.read().tick == a, "paused sim must not advance"
    finally:
        sim.stop()
    assert not sim.is_alive(), "sim thread must shut down cleanly"
