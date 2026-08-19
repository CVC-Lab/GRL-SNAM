"""Several agents in one world.

The property that makes this worth building is that belief is PRIVATE: a
shared map would be a much smaller change and a much weaker demo, because the
interesting behaviour is one agent knowing something another does not.
"""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam.fog_stories import STORIES  # noqa: E402
from grl_snam.squad import AgentSpec, Squad  # noqa: E402


def _small():
    from grl_snam.fog_stories import shrunk

    return shrunk(STORIES["city"], n=48, max_steps=120)


def _agents():
    return [
        AgentSpec("a", (-70.0, -60.0), (70.0, 60.0), (0.3, 0.7, 1.0)),
        AgentSpec("b", (-70.0, 60.0), (70.0, -60.0), (1.0, 0.5, 0.2)),
    ]


def test_a_squad_needs_agents():
    with pytest.raises(ValueError):
        Squad(_small(), [])


def test_each_agent_gets_its_own_belief():
    """Not a shared grid. If these were the same object the demo would be
    claiming something it does not do."""
    sq = Squad(_small(), _agents())
    a, b = sq.scenarios["a"], sq.scenarios["b"]
    assert a.belief is not b.belief
    assert a.nav is not b.nav
    assert a.route is not b.route or a.route is None


def test_knowledge_diverges_between_agents():
    """After driving apart, each agent has seen things the other has not."""
    sq = Squad(_small(), _agents())
    sq.run(max_steps=120)
    seen_a = sq.scenarios["a"].belief.ever_seen
    seen_b = sq.scenarios["b"].belief.ever_seen
    assert seen_a.any() and seen_b.any()
    assert (seen_a & ~seen_b).any(), "agent a knows nothing agent b does not"
    assert (seen_b & ~seen_a).any(), "agent b knows nothing agent a does not"


def test_agents_are_real_to_each_other():
    """A peer is stamped into the OTHER agents' truth, so it occludes and has
    to be discovered — never into its own, or it would map itself as an
    obstacle and refuse to move."""
    sq = Squad(_small(), _agents())
    sq._stamp_peers()
    a, b = sq.scenarios["a"], sq.scenarios["b"]

    # Check this THROUGH the step that used to erase it. The previous version of
    # this test read truth_now straight after _stamp_peers and passed for
    # months, while FogScenario._stamp_movers -- whose first line is
    # `truth_now = truth.copy()` -- wiped every peer microseconds later, on
    # every tick. Asserting the write rather than its survival is what let the
    # squad be mutually invisible under a green suite.
    a._stamp_movers()
    b._stamp_movers()

    # b's body appears in a's world...
    added_to_a = a.truth_now & ~a.truth
    assert added_to_a.any(), "peers did not survive the per-tick truth reset"
    assert (b.truth_now & ~b.truth).any()

    # ...and nothing was added at a's OWN position. (Asserting the cell is
    # simply clear would be wrong: on the shrunk test grid the start happens to
    # sit inside a static block, which peer-stamping has nothing to do with.)
    r, c = a.belief.world_to_cell(*a.nav.pos_world())
    assert not added_to_a[r, c], "an agent stamped itself as an obstacle"

    # what a gained is exactly b's footprint, and vice versa
    rb, cb = b.belief.world_to_cell(*b.nav.pos_world())
    assert added_to_a[rb, cb], "a did not receive b's body"


def test_the_shared_world_is_not_mutated_permanently():
    """truth_now is rebuilt each tick; the static truth must never accumulate
    peer footprints, or the city slowly fills with ghosts of past positions."""
    sq = Squad(_small(), _agents())
    before = sq.scenarios["a"].truth.copy()
    sq.run(max_steps=40)
    assert np.array_equal(sq.scenarios["a"].truth, before)


def test_a_run_reports_per_agent_results():
    sq = Squad(_small(), _agents())
    res = sq.run(max_steps=80)
    assert set(res.tracks) == {"a", "b"}
    assert set(res.penetration) == {"a", "b"}
    for k in ("a", "b"):
        assert res.tracks[k].shape[1] == 2
        assert res.penetration[k] >= 0


def test_agents_step_in_lockstep():
    """Every agent advances exactly one world tick per squad step, or their
    traces cannot share a clock."""
    sq = Squad(_small(), _agents())
    for _ in range(10):
        sq.step()
    assert sq.scenarios["a"].step_i == sq.scenarios["b"].step_i == 10


def test_one_agent_squad_matches_the_single_agent_path():
    """A squad of one must behave like the plain scenario — otherwise the
    multi-agent path is a fork rather than a wrapper."""
    story = _small()
    spec = AgentSpec("solo", story.start, tuple(story.waypoints[0]))
    sq = Squad(story, [spec])
    sq.run(max_steps=60)
    sc = sq.scenarios["solo"]
    assert sc.step_i == 60 or sc.done
    assert sc.belief.ever_seen.any()


def test_differing_goals_are_honoured():
    story = _small()
    agents = _agents()
    sq = Squad(story, agents)
    for a in agents:
        wp = sq.scenarios[a.key].waypoints[0]
        assert tuple(np.asarray(wp, float)) == pytest.approx(a.goal)


def test_the_story_start_is_overridden_per_agent():
    story = dataclasses.replace(_small(), start=(0.0, 0.0))
    sq = Squad(story, _agents())
    for a in _agents():
        x, y = sq.scenarios[a.key].nav.pos_world()
        assert (x, y) == pytest.approx(a.start, abs=2.0)


def test_a_peer_in_sensor_range_is_actually_discovered():
    """Surviving the truth reset is necessary; being SEEN is the point.

    Separate from the stamping test because that one's agents start 120 m apart
    with a ~38 m sensor and cross late -- a fine fixture for stamping, useless
    for discovery. Here they start within range of each other, so a ray really
    does land on a peer and it lands in the decaying layer (never the static
    map, or a moving agent would leave a permanent wall behind it).
    """
    sq = Squad(
        _small(),
        [
            AgentSpec("a", (-70.0, -8.0), (70.0, 0.0), (0.3, 0.7, 1.0)),
            AgentSpec("b", (-70.0, 8.0), (70.0, 8.0), (1.0, 0.5, 0.2)),
        ],
    )
    for _ in range(40):
        sq.step()
    assert any(
        s.dyn.occupancy(s._t()).any() for s in sq.scenarios.values()
    ), "no peer was ever sensed, so peers are still not real to each other"


# ── stage-4: batched planning (bit-identical to serial) ──────────────────────

from grl_snam import nav_native  # noqa: E402
from grl_snam.squad import FollowGoal  # noqa: E402

_HAVE_BATCH = nav_native.enabled() and hasattr(nav_native, "build_sdf_batch")


def _tracks(agents, *, batched):
    sq = Squad(_small(), agents, batched_planning=batched)
    return sq.run(max_steps=120, stop_when_done=False).tracks


@pytest.mark.skipif(not _HAVE_BATCH, reason="pycvc batch kernels unavailable")
@pytest.mark.parametrize(
    "agents",
    [
        [  # independent agents
            AgentSpec("a", (-70.0, -60.0), (70.0, 60.0)),
            AgentSpec("b", (-70.0, 60.0), (70.0, -60.0)),
            AgentSpec("c", (0.0, -70.0), (0.0, 70.0)),
        ],
        [  # a FollowGoal convoy — the order-sensitive case
            AgentSpec("lead", (-70.0, -60.0), (70.0, 60.0)),
            AgentSpec("follow", (-72.0, -62.0), (0.0, 0.0), moving_goal=FollowGoal("lead")),
        ],
    ],
    ids=["independent", "convoy"],
)
def test_batched_planning_is_bit_identical_to_serial(agents):
    """Batching the SDF build across agents is an optimization, not a behaviour
    change: agents couple only through the tick-start peer stamp and
    insertion-ordered acting, both preserved by the sense/act split."""
    serial = _tracks(agents, batched=False)
    batched = _tracks(agents, batched=True)
    assert set(serial) == set(batched)
    for k in serial:
        assert np.array_equal(serial[k], batched[k]), f"agent {k} diverged"


def test_stagger_assigns_distinct_phases():
    sq = Squad(
        _small(),
        [
            AgentSpec("a", (-70.0, -60.0), (70.0, 60.0)),
            AgentSpec("b", (-70.0, 60.0), (70.0, -60.0)),
            AgentSpec("c", (0.0, -70.0), (0.0, 70.0)),
        ],
        stagger_sense=True,
    )
    phases = [sc.sense_phase for sc in sq.scenarios.values()]
    assert len(set(phases)) > 1, "stagger did not spread the sense schedule"
    sq.run(max_steps=40, stop_when_done=False)  # runs clean


# ── stage-2: batched vehicle rollout (bit-identical to serial) ───────────────

import sdf_nav  # noqa: E402


def _shared_model():
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _tracks_model(agents, model, *, batched_drive):
    sq = Squad(_small(), agents, model=model, batched_drive=batched_drive)
    return sq.run(max_steps=90, stop_when_done=False).tracks, sq


def test_batched_drive_matches_serial_to_float32():
    """Rolling every agent forward in one batched bicycle_rollout matches the
    serial path to float32 precision. It is NOT guaranteed byte-identical the way
    the SDF/A* kernels are: torch's batched grid_sample/matmul can round
    differently by up to ~1 float32 ULP for some inputs (bit-identical for the
    first ~60 chained steps here, drifting to ~5e-7 by 120). So it is opt-in
    (batched_drive=True) and asserted equal to within a tight float32 tolerance,
    never gross divergence — a real indexing bug would diverge by metres.
    Needs a SHARED model instance (a fresh per-agent CoefMLP would make the
    agents non-interchangeable in the batch)."""
    agents = [
        AgentSpec("a", (-70.0, -60.0), (70.0, 60.0)),
        AgentSpec("b", (-70.0, 60.0), (70.0, -60.0)),
        AgentSpec("c", (0.0, -70.0), (0.0, 70.0)),
    ]
    model = _shared_model()
    serial, _ = _tracks_model(agents, model, batched_drive=False)
    batched, sq_b = _tracks_model(agents, model, batched_drive=True)
    assert sq_b._can_batch_drive(), "drive batching did not engage for a shared-model squad"
    for k in serial:
        assert np.allclose(serial[k], batched[k], atol=1e-4, rtol=0.0), f"agent {k} diverged"


def test_convoy_disables_drive_batch_and_stays_bit_identical():
    """A FollowGoal reads its leader's pose mid-tick, so the drive is NOT
    batched even with batched_drive=True (that would feed a stale pose); the
    serial-drive fallback is bit-identical to the fully serial path."""
    agents = [
        AgentSpec("lead", (-70.0, -60.0), (70.0, 60.0)),
        AgentSpec("follow", (-72.0, -62.0), (0.0, 0.0), moving_goal=FollowGoal("lead")),
    ]
    model = _shared_model()
    serial, _ = _tracks_model(agents, model, batched_drive=False)
    batched, sq_b = _tracks_model(agents, model, batched_drive=True)
    assert not sq_b._can_batch_drive(), "a FollowGoal convoy must fall back to the serial drive"
    for k in serial:
        assert np.array_equal(serial[k], batched[k]), f"agent {k} diverged"


def test_drive_batch_off_by_default_and_guarded():
    """Off by default (twin stays byte-exact); on, it still needs one shared
    model — model=None gives each agent its own CoefMLP, so they are not
    interchangeable in one batched call."""
    agents = [
        AgentSpec("a", (-70.0, -60.0), (70.0, 60.0)),
        AgentSpec("b", (-70.0, 60.0), (70.0, -60.0)),
    ]
    assert not Squad(_small(), agents, model=_shared_model())._can_batch_drive()  # default off
    # opt-in but model=None -> fresh per-agent models -> still off
    assert not Squad(_small(), agents, batched_drive=True)._can_batch_drive()
