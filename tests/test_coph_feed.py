"""grl_snam.coph.feed: a live run is T1's run, and its messages equal its replay's.

The gate of ``grl-snam ariadne coph`` (roadmap §3.2): the viewer's episodes equal T1's, because
it is the same code and the viewer only reads it. These tests need no pycvc_gl.
"""

from __future__ import annotations

import json

import pytest

from grl_snam.coph import feed
from grl_snam.coph.evaluation import load

STEPS = 50  # the first door warnings already reach the receivers (tests/test_coph.py)
# The wall-clock fields (the episode's, each audit's and each update diagnostic's), which differ
# between any two runs. active_seconds and start_seconds are simulated time and stay.
CLOCKS = {"wall_seconds", "audit_seconds", "seconds"}


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in CLOCKS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _canonical(messages):
    return sorted(json.dumps(m) for m in messages)


@pytest.fixture(scope="module")
def crossing(tmp_path_factory):
    """reciprocal_fixed at 50 steps: live through the feed, and through crossing_demo.run."""
    from grl_snam.coph import crossing_demo

    root = tmp_path_factory.mktemp("coph-feed")
    live = []
    feed.run("crossing", "reciprocal_fixed", 76000, STEPS, sink=live.extend, out=root / "feed")
    (root / "t1" / "episodes").mkdir(parents=True)  # as crossing_demo.main does
    crossing_demo.run((76000, "reciprocal_fixed", str(root / "t1"), STEPS))
    return dict(
        live=live,
        feed=load(root / "feed" / "episodes" / "76000_reciprocal_fixed.json.gz"),
        t1=load(root / "t1" / "episodes" / "76000_reciprocal_fixed.json.gz"),
    )


def test_live_run_is_t1s_run(crossing):
    assert _strip(crossing["feed"]) == _strip(crossing["t1"])


def test_live_messages_equal_the_replay(crossing):
    # A live run sends messages as they happen and a replay agent by agent; the set is the same.
    assert _canonical(crossing["live"]) == _canonical(feed.messages(crossing["t1"]))


def test_messages_cover_the_episode(crossing):
    r, live = crossing["t1"], crossing["live"]
    kinds = [m[0] for m in live]
    assert kinds.count("pose") == sum(len(s) for s in r["samples"])
    assert kinds.count("step") == sum(len(t) for t in r["trajectories"])
    assert kinds.count("update") == sum(len(u) for u in r["updates"])
    delivered = [e for e in r["events"] if e["kind"] == "delivered"]
    packets = [m for m in live if m[0] == "packet"]
    assert len(packets) == len(delivered) == 14
    sent = {e["packet"]: e["bytes"] for e in r["events"] if e["kind"] == "sent"}
    assert all(m[5] == sent[m[1]] for m in packets)  # the runner's own byte counts
    # A delivered event's acceptance is the update that took the packet in.
    accepted = {e["packet"]: e["accepted"] for e in delivered}
    for m in live:
        if m[0] == "update":
            assert all(accepted[p] == m[4] for p in m[7])
    geometry = [(m[1], m[3]) for m in live if m[0] == "geometry"]
    assert sorted(geometry) == [(0, "received_geometry"), (1, "locally_sensed")]
    assert kinds[-2:] == ["outcome", "outcome"]


def test_job_refuses_unknown_arms_and_scenes():
    with pytest.raises(ValueError, match="no arm"):
        feed.job("crossing", "field", 76000, 10)
    with pytest.raises(ValueError, match="no scene"):
        feed.job("hazard", "field", 79000, 10, scene="tidal")
    with pytest.raises(ValueError, match="no demo"):
        feed.job("maze", "disabled", 1, 10)


def test_hazard_job_matches_hazard_demo():
    from grl_snam.coph import hazard_demo

    spec, channel, comm, kwargs = feed.job("hazard", "raw", 79000, 30, scene="mild")
    assert spec == hazard_demo.specification("mild", 79000, 30)
    assert channel == "reciprocal_fixed" and comm == hazard_demo.communication("raw")
    assert comm.anchors == hazard_demo.RAW_ANCHORS
    assert kwargs["config"].steps == 30


# ── the timeline the viewer draws from ─────────────────────────────────────────────────────────


def test_timeline_reads_live_and_replay_alike(crossing):
    from grl_snam.coph.timeline import Timeline

    spec = crossing["t1"]["spec"]
    live = Timeline("crossing", spec, "reciprocal_fixed").ingest(crossing["live"])
    replay = Timeline.from_episode("crossing", crossing["t1"])
    assert live.finished and replay.finished and live.duration == replay.duration == 6.0
    for t in (0.0, 0.5, 1.0, 2.9, 4.33, 6.0):
        for i in range(2):
            assert live.state(i, t, track=True) == replay.state(i, t, track=True)


def test_timeline_state_is_the_episodes(crossing):
    import numpy as np

    from grl_snam.coph.results import path_length
    from grl_snam.coph.timeline import Timeline

    r = crossing["t1"]
    tl = Timeline.from_episode("crossing", r)
    for i in range(2):
        s = tl.state(i, tl.duration)
        steps = r["trajectories"][i]
        assert s.ledger["work"] == pytest.approx(sum(x["energy"]["work"] for x in steps))
        assert s.ledger["heat"] == pytest.approx(sum(x["energy"]["heat"] for x in steps))
        assert abs(s.ledger["residual"]) < 1e-12 and s.max_residual < 1e-12  # the ledger closes
        assert s.path == pytest.approx(path_length(r["samples"][i]))
        assert s.version == r["updates"][i][-1]["post"]["version"]
        assert list(s.policy) == r["updates"][i][-1]["post"]["policy"]
        assert s.accepts + s.rejects == len(r["updates"][i])
        assert len(s.stages) == len(r["updates"][i][-1]["post"]["stages"])
        assert s.outcome["success"] == r["agents"][i]["success"]
    # Packets: received by one agent, sent by the other, at the runner's byte counts.
    sent = sum(e["bytes"] for e in r["events"] if e["kind"] == "sent")
    end = [tl.state(i, tl.duration) for i in range(2)]
    assert end[0].bytes_sent + end[1].bytes_sent == sent
    assert end[0].bytes_received == end[1].bytes_sent
    # Before its first boundary an agent is at its start, inactive, on the initial policy.
    first = tl.state(0, -1.0)
    assert not first.active and first.q == (0.0, 4.0) and first.policy == (1.0, 2.0)
    # Agent 0 learns of the middle door from agent 1's packet; agent 1 sensed it itself.
    door = tuple(r["spec"]["doors"]["middle"])
    assert door in end[0].known and door in end[1].known
    assert np.isclose(
        tl.deliveries[0][0],
        r["events"][[e["kind"] for e in r["events"]].index("delivered")]["time"],
    )


def test_floor_draws_the_corridors_and_fields(crossing):
    import numpy as np

    from grl_snam.coph.timeline import Floor, Timeline

    r = crossing["t1"]
    floor = Floor(r["spec"], feed.world("crossing", r["spec"]), px_per_m=8)
    assert (floor.w, floor.h) == (14 * 8, round(9.5 * 8))

    # The middle row is a corridor; the block between the rows is not.
    def at(x, y):
        return int((y - floor.y0) * floor.px), int((x - floor.x0) * floor.px)

    assert floor.corridor[at(6.0, 4.0)] and not floor.corridor[at(6.0, 2.0)]
    assert np.allclose(floor.truth, 0.02)  # crossing has no risk patches
    s = Timeline.from_episode("crossing", r).state(0, 6.0)
    field = floor.agent_risk(s.stages)
    inside = ~np.isnan(field)
    assert inside.any() and not inside.all() and np.all(field[inside] > 0)
    img = floor.image(field, [(floor.ring(s.q, 1.25), (1.0, 0.0, 0.0), 1.0)])
    assert img.shape == (floor.h, floor.w, 3) and img.dtype == np.uint8
    assert (img == [255, 0, 0]).all(axis=-1).any()  # the sensor ring


def test_log_has_the_cars_shape_and_the_ledger(crossing):
    import json

    import numpy as np

    from grl_snam.coph.timeline import COEF_FIELDS, LOG_FIELDS, Timeline, comparison, log_arrays

    a = Timeline.from_episode("crossing", crossing["t1"], label="left")
    b = Timeline.from_episode("crossing", crossing["feed"], label="right")
    out = log_arrays([a, b], dict(demo="crossing"))
    n = len(a.times())
    for k in LOG_FIELDS:
        assert out[k].shape == (n, 4)
    for k in COEF_FIELDS:
        assert out[f"coef_{k}"].shape == (n, 4)
    assert np.isnan(out["steer"]).all() and np.isnan(out["cmd_accel"]).all()
    assert out["t"][0] == 0.0 and out["t"][-1] == 6.0
    meta = json.loads(str(out["meta"]))
    assert [c["label"] for c in meta["columns"]] == ["left", "left", "right", "right"]
    assert np.array_equal(out["x"][:, :2], out["x"][:, 2:])  # the same episode twice
    assert comparison("crossing", a, b)["saving"] == [0.0, 0.0]


# ── what the learners measured ─────────────────────────────────────────────────────────────────


def test_updates_carry_what_they_measured(crossing):
    r = crossing["t1"]
    updates = [m for m in feed.messages(r) if m[0] == "update"]
    expected = [(i, u["diagnostic"]) for i in range(2) for u in r["updates"][i]]
    assert len(updates) == len(expected)
    for m, (i, d) in zip(updates, expected):
        assert m[1] == i and list(m[9]) == [k for k, _ in feed.STATS]
        for k, src in feed.STATS:
            assert m[9][k] == (None if d.get(src) is None else float(d[src])), k


def test_learning_is_what_each_update_measured(crossing):
    import numpy as np

    from grl_snam.coph.timeline import LEARNING, Floor, Timeline, field_error, learning

    r = crossing["t1"]
    floor = Floor(r["spec"], feed.world("crossing", r["spec"]), px_per_m=4)
    replay = Timeline.from_episode("crossing", r)
    live = Timeline("crossing", r["spec"], "reciprocal_fixed").ingest(crossing["live"])
    for i in range(2):
        got = learning(replay, i, floor)
        assert list(got) == list(LEARNING) and len(got["t"]) == len(r["updates"][i])
        for k, u in enumerate(r["updates"][i]):
            d, post = u["diagnostic"], u["post"]
            before, after = d["field_validation_before"], d["field_validation_after"]
            assert got["t"][k] == u["time"] and got["version"][k] == post["version"]
            assert got["field_before"][k] == before and got["field_proposed"][k] == after
            assert got["field_heldout"][k] == (after if d["accepted"] else before)
            plant = d["plant_validation_after" if d["accepted"] else "plant_validation_before"]
            assert got["plant_heldout"][k] == plant
            assert [got["pi_risk"][k], got["pi_barrier"][k]] == post["policy"]
            assert got["damping"][k] == post["damping"]
        # The plant fit finds the world's damping (0.85) from 0.6, 0.15 an update.
        assert got["damping"][0] == 0.6
        assert got["damping"][-1] == pytest.approx(r["spec"]["damping"], abs=1e-3)
        # The field error is |r - r_true| over the covered corridor, split by where it sensed.
        last = r["updates"][i][-1]
        seen = np.zeros(floor.points.shape[:2], bool)
        for s in r["samples"][i]:
            if s["time"] <= last["time"] + 1e-9:
                seen |= ((floor.points - s["x"][:2]) ** 2).sum(-1) <= 1.25**2
        want = field_error(floor, last["post"]["stages"], seen)
        keys = ("error_seen", "error_seen_lo", "error_seen_hi", "error_unseen")
        assert [got[k][-1] for k in keys] == pytest.approx(want, nan_ok=True)
        assert np.isfinite(got["error_seen"]).all()
        assert (got["error_seen_lo"] <= got["error_seen_hi"]).all()
        # A live run measures the same, and until= stops at that time.
        for k in LEARNING:
            np.testing.assert_array_equal(learning(live, i, floor)[k], got[k])
        early = learning(replay, i, floor, until=3.0)
        n = int((got["t"] <= 3.0).sum())
        assert 0 < n < len(got["t"]) and len(early["t"]) == n
        np.testing.assert_array_equal(early["field_heldout"], got["field_heldout"][:n])
