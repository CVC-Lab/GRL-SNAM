"""The fog demo's testable surface: clock, story specs, and trace round-trip.

Deliberately no GL and no host. Rendering is exercised by running the capture,
not by the test suite — ``tests/test_lab.py`` already documents that headless
``render_png`` segfaults, and CI has no display.
"""

import math

import numpy as np
import pytest

from grl_snam.clock import WorldClock

torch = pytest.importorskip("torch")

from grl_snam.fog_stories import STORIES, build_scenario, shrunk, unit_track  # noqa: E402
from grl_snam.fog_trace import Trace  # noqa: E402
from grl_snam.tools.fog_record import is_stale, record, story_hash  # noqa: E402

# ── the world clock ─────────────────────────────────────────────────────────


def test_world_time_is_derived_and_does_not_drift():
    """The property the C++ clock exists to guarantee: t() is tick*fixed_dt,
    never a running sum. Demonstrate the drift we avoid, then assert we
    avoided it."""
    c = WorldClock(fixed_dt=0.06)
    n = 200_000
    for _ in range(n):
        c.step_once()
    assert c.tick() == n
    assert c.t() == n * 0.06

    accumulated = 0.0
    for _ in range(n):
        accumulated += 0.06
    assert accumulated != c.t(), "the drift being avoided is not real at this scale"
    assert abs(accumulated - c.t()) < 1e-3


def test_alpha_stays_in_range_and_steps_are_whole_quanta():
    c = WorldClock(fixed_dt=0.06)
    for dt in (0.0, 1e-12, 0.03, 0.06, 0.09, 1.0, -1.0, float("nan"), float("inf"), 1e9):
        r = c.advance(dt)
        assert 0.0 <= r.alpha < 1.0, dt
        assert not math.isnan(r.alpha), dt
        assert r.steps >= 0 and r.dropped_steps >= 0


def test_a_stall_is_clamped_and_the_drop_is_reported():
    c = WorldClock(fixed_dt=0.06, max_steps_per_advance=4)
    r = c.advance(1.2)  # 20 quanta demanded, 4 allowed
    assert r.steps == 4
    assert r.steps + r.dropped_steps == 20
    assert c.total_dropped() == r.dropped_steps
    assert c.pending_seconds() < c.fixed_dt


def test_non_live_modes_bank_no_wall_time():
    for mode in ("paused", "replay", "stepping"):
        c = WorldClock(fixed_dt=0.06, mode=mode)
        c.advance(10.0)
        assert c.tick() == 0
        assert c.pending_seconds() == 0.0


def test_returning_to_live_clears_the_bank():
    c = WorldClock(fixed_dt=0.06)
    c.advance(0.05)  # bank most of a quantum
    assert c.pending_seconds() > 0.0
    c.set_mode("live")  # already live: must NOT clear
    assert c.pending_seconds() > 0.0
    c.set_mode("paused")
    c.set_mode("live")  # a real transition: clears
    assert c.pending_seconds() == 0.0


def test_seek_time_lands_on_the_containing_quantum():
    c = WorldClock(fixed_dt=0.06, mode="replay")
    c.seek_time(1.0)
    assert c.tick() == 16  # floor(1.0 / 0.06)
    assert c.t() == pytest.approx(0.96)


# ── the story specs ─────────────────────────────────────────────────────────


def test_the_three_stories_exist_and_are_self_consistent():
    assert set(STORIES) == {"ghost", "blocker", "unit"}
    for key, st in STORIES.items():
        assert st.key == key
        assert st.title and st.subtitle
        assert st.dt > 0 and st.n > 0
        # reach_tol is NORMALIZED: reach_tol/scale metres. The navigator
        # default of 0.8 is 16 m here and visibly stops short of the marker.
        assert st.reach_tol / st.scale < 6.0, f"{key}: goal tolerance too loose to look right"
        for r0, r1, c0, c1 in st.truth_rects + st.prior_rects:
            assert 0 <= r0 < r1 <= st.n and 0 <= c0 < c1 <= st.n, key
        for x, y in st.waypoints:
            assert st.bounds[0] < x < st.bounds[2] and st.bounds[1] < y < st.bounds[3]


def test_each_story_carries_its_own_distinguishing_mechanism():
    """If these collapse, the demo tells the same story three times."""
    assert STORIES["ghost"].prior_rects and not STORIES["ghost"].truth_rects
    assert any(e.kind == "add_rect" for e in STORIES["blocker"].events)
    assert any(e.kind == "unit_at" for e in STORIES["unit"].events)


def test_unit_track_sweeps_and_is_not_a_one_cell_blip():
    ev = unit_track(step0=10, every=3, count=12, r0=60, c0=30, r1=20, c1=50)
    assert len(ev) == 12
    assert ev[0].args == (60, 30) and ev[-1].args == (20, 50)
    assert [e.step for e in ev] == list(range(10, 10 + 12 * 3, 3))


def test_captions_are_ordered_and_cover_the_opening():
    for key, st in STORIES.items():
        assert st.captions, key
        assert st.captions[0][0] == 0.0, f"{key}: opening second has no caption"
        for (a0, a1, _), (b0, _, _) in zip(st.captions, st.captions[1:]):
            assert a0 < a1 and a1 <= b0, f"{key}: captions overlap or run backwards"


# ── record / replay ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_trace(tmp_path_factory):
    story = shrunk(STORIES["ghost"], n=48, max_steps=90)
    out = record(story.key, tmp_path_factory.mktemp("trace"), story=story)
    return story, Trace.load(out)


def test_trace_round_trips(small_trace):
    story, tr = small_trace
    assert tr.n_ticks > 0
    assert tr.fixed_dt == story.dt
    assert tr.shape == (story.n, story.n)
    assert tr.duration_s == pytest.approx(tr.n_ticks * story.dt)
    ticks = tr.rows["tick"]
    assert np.array_equal(ticks, np.arange(1, tr.n_ticks + 1)), "ticks must be dense and monotone"


def test_pose_at_a_tick_boundary_is_the_recorded_row_exactly(small_trace):
    """No interpolation drift where it matters: at t = k*fixed_dt the renderer
    must show precisely what the simulator recorded."""
    _, tr = small_trace
    for i in (0, tr.n_ticks // 3, tr.n_ticks - 1):
        p = tr.pose_at(i * tr.fixed_dt)
        assert p.x == tr.rows["x"][i]
        assert p.y == tr.rows["y"][i]
        assert p.heading_rad == tr.rows["heading_rad"][i]
        assert p.alpha == 0.0


def test_pose_interpolates_between_ticks(small_trace):
    _, tr = small_trace
    i = tr.n_ticks // 3
    a, b = tr.pose_at(i * tr.fixed_dt), tr.pose_at((i + 1) * tr.fixed_dt)
    mid = tr.pose_at((i + 0.5) * tr.fixed_dt)
    assert min(a.x, b.x) <= mid.x <= max(a.x, b.x)
    assert 0.0 < mid.alpha < 1.0


def test_heading_interpolation_takes_the_short_way_round():
    """A vehicle crossing +-pi must not spin the long way for one frame."""
    from grl_snam.fog_trace import _lerp_angle

    got = _lerp_angle(3.0, -3.0, 0.5)
    assert abs(abs(got) - math.pi) < 0.15, f"took the long way: {got}"


def test_belief_snapshots_decode_and_are_a_step_function(small_trace):
    story, tr = small_trace
    occ, dyn, k = tr.belief_at(0.0)
    assert occ.shape == (story.n, story.n) and dyn.shape == occ.shape
    assert k == 0, "the prior map must be snapshot 0, before any ray is cast"
    assert occ.any(), "the ghost story's prior wall is missing from the first snapshot"
    # the index only ever moves forward
    idx = [tr.snapshot_index_at(t) for t in np.linspace(0, tr.duration_s, 40)]
    assert idx == sorted(idx)


def test_track_grows_and_route_is_available(small_trace):
    _, tr = small_trace
    early = tr.track_upto(tr.duration_s * 0.25)
    late = tr.track_upto(tr.duration_s * 0.75)
    assert len(early) < len(late) <= tr.n_ticks
    assert late.shape[1] == 2


def test_manifest_carries_the_numbers_quoted_on_stage(small_trace):
    _, tr = small_trace
    s = tr.summary
    for k in ("steps", "world_seconds", "map_updates", "penetration_steps", "detour_peak_m"):
        assert k in s, k
    assert s["straight_line_m"] > 0
    assert tr.manifest["seed"] == 0 and tr.manifest["torch"]


def test_metrics_bridge_feeds_the_existing_hud(small_trace):
    """The HUD must go through grl_snam.metrics, not a second implementation."""
    from grl_snam.metrics import hud_lines

    _, tr = small_trace
    m = tr.to_metrics(tr.duration_s * 0.5)
    lines = hud_lines(m)
    assert lines and all(isinstance(x, str) for x in lines)


def test_a_changed_story_makes_an_existing_trace_stale(small_trace, tmp_path):
    story, _ = small_trace
    assert is_stale(tmp_path, story), "a missing trace must read as stale"
    import dataclasses

    assert story_hash(story) != story_hash(dataclasses.replace(story, sense_every=99))


# ── the scenario builder ────────────────────────────────────────────────────


def test_build_scenario_is_deterministic_for_a_seed():
    story = shrunk(STORIES["blocker"], n=48, max_steps=40)

    def run():
        sc = build_scenario(story, seed=7)
        return np.asarray([(r.x, r.y) for r in sc.run(max_steps=40).records])

    assert np.array_equal(run(), run())
