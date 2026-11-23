"""The Laboratory's acceptance tests (the engagement docs' Laboratory, §9), on live's arms.

- **Test 5:** one Retarget moves every arm's goal on the same tick, and the Stats rows tell the
  arms apart.
- **Test 6, exact replay:** a session recorded with ``--record-trace`` (a Retarget pressed and a
  setting changed mid-run) replays with ``--replay-trace`` to byte-identical poses and the same
  final hash. A one-arm session's trace replays in libcvc's own headless replay too.
- **Test 6, two policy builds:** replayed with another build of a policy, the run matches the
  recording until the first tick where the two builds' coefficients differ.

They run headless on the flat square (no download) on nav's kinematic plant, with arms that need
no grl-snam-weights, and skip without pycvc_gl's Python host and its multi-car template. Test 6
needs libcvc's input traces (transfix/libcvc#1168, a pycvc after 3.7.14) and skips without them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from grl_snam.ariadne import _trace

ARMS = "untrained,constant,brisk=const:1,6,2"
GOAL = "-20, -5"
#: The recorded session's frames, at 30 a second.
FRAMES = 450


def _ticks(frames: int) -> int:
    """The navigator's ticks in ``frames`` frames: one every ``pycvc_gl.cars.DT``, libcvc's tick
    (0.05 s before 4.0.2, then ``pycvc.SIM_TICK``: 0.06 s in 4.0.2, 7/120 s since)."""
    from grl_snam.ariadne import _cars

    return int(frames / 30.0 / _cars.C.DT + 1e-9)


def _ready() -> bool:
    from grl_snam.ariadne import _cars, _host

    return _host.HAS_HOST and _cars.HAS_CARS


needs_host = pytest.mark.skipif(
    not _ready(), reason="needs pycvc_gl's Python host and pycvc_gl.cars (pycvc-gl 3.7.12+)"
)
needs_trace = pytest.mark.skipif(not _trace.HAS_TRACE, reason=f"needs {_trace.INSTALL}")


def _main(args):
    from grl_snam.ariadne.__main__ import main

    return main(args)


def _run(args, actions=None):
    """``main(args)``, keeping the hosts it opens and running ``actions`` ({frame: fn(host)})
    before the port's own frame."""
    from grl_snam.ariadne import _host

    hosts = []
    real = _host.open_host

    def open_host(*a, **kw):
        host = real(*a, **kw)
        hosts.append(host)
        frames = [0]

        @host.on_frame
        def act(dt):
            frames[0] += 1
            if frames[0] in (actions or {}):
                actions[frames[0]](host)

        return host

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_host, "open_host", open_host)
        return _main(args), hosts


def _live(*extra):
    return ["live", "austin-freedrive", "--headless", "--world", "flat", "--plant", "kinematic",
            *extra]  # fmt: skip


def _log(path):
    from grl_snam.learning.runlog import RunLog

    return RunLog.load(path)


# ── Test 5 ─────────────────────────────────────────────────────────────────────────────────────


@needs_host
def test_5_one_retarget_moves_every_arm_and_the_stats_tell_them_apart(tmp_path):
    pytest.importorskip("torch")
    code, hosts = _run(
        _live("--frames", "450", "--arms", ARMS, "--set", f"arms.goal={GOAL}",
              "--log", str(tmp_path / "run.npz")),
        {120: lambda h: h.emit("arms_retarget")},
    )  # fmt: skip
    assert code == 0
    a = _log(tmp_path / "run.npz").arrays
    tick = int(a["leg_tick"][1])
    for j in range(3):
        moved = np.flatnonzero(a["goal_x"][:, j] != a["goal_x"][0, j])
        assert moved[0] == tick
    rows = hosts[0].get_tree("arms.stats")
    assert len({(r["clear_mean"], r["path"], r["elapsed"]) for r in rows}) == 3


# ── Test 6 ─────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """A recorded session: three arms, the Retarget pressed at frame 120, the top speed set to
    6 m/s at frame 200 (a slider's write)."""
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("lab")
    code, _ = _run(
        _live("--frames", str(FRAMES), "--arms", ARMS, "--seed", "3", "--set", f"arms.goal={GOAL}",
              "--record-trace", str(root / "s.trace"), "--log", str(root / "rec.npz")),
        {120: lambda h: h.emit("arms_retarget"), 200: lambda h: h.set("nav.vehicle.vmax", "6")},
    )  # fmt: skip
    return dict(root=root, code=code)


@needs_host
@needs_trace
def test_6_the_trace_holds_the_sessions_inputs(session):
    import pycvc

    assert session["code"] == 0
    t = pycvc.input_trace.read(str(session["root"] / "s.trace"))
    recs = [t.records[k] for k in range(t.record_count())]
    events = [r.name for r in recs if r.kind_name() == "event"]
    sets = [r.path for r in recs if r.kind_name() == "set"]
    assert events == ["nav.retarget"]
    assert sets == ["live.nav.vehicle.vmax"] or sets == ["nav.vehicle.vmax"], sets
    assert t.end.present and t.end.hash.startswith("sha256:")
    header = json.loads(t.header.to_json())
    meta = header["meta"] if isinstance(header["meta"], dict) else json.loads(header["meta"])
    assert [a["name"] for a in meta["arms"]] == ["untrained", "constant", "brisk"]
    assert meta["seed"] == 3 and meta["plant"] == "kinematic"


@needs_host
@needs_trace
def test_6_exact_replay(session, capsys):
    """The replay gives byte-identical poses and the same final hash."""
    root = session["root"]
    capsys.readouterr()
    code, hosts = _run(_live("--replay-trace", str(root / "s.trace"),
                             "--log", str(root / "rep.npz")))  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "the replay matches the trace" in out, out
    rec, rep = _log(root / "rec.npz").arrays, _log(root / "rep.npz").arrays
    n = len(rec["t"])
    assert len(rep["t"]) == n >= _ticks(FRAMES) - 1  # every frame's ticks, recorded and replayed
    for k in ("x", "y", "heading", "speed", "goal_x", "goal_y", "tel_alpha", "book_context"):
        assert rec[k].tobytes() == rep[k].tobytes(), k
    assert rec["leg_tick"].tolist() == rep["leg_tick"].tolist()
    assert "all matched" in hosts[0].get("trace.status")


@needs_host
@needs_trace
def test_6_a_one_arm_trace_replays_in_libcvc(tmp_path):
    """A one-arm session's trace is libcvc's own: its headless replay (cvc trace verify's) matches
    it, the Retarget included."""
    import pycvc

    pytest.importorskip("torch")
    path = tmp_path / "one.trace"
    code, _ = _run(_live("--frames", "300", "--arms", "untrained", "--set", f"arms.goal={GOAL}",
                         "--record-trace", str(path)),
                   {90: lambda h: h.emit("arms_retarget")})  # fmt: skip
    assert code == 0
    r = pycvc.nav_trace_replay(pycvc.App(), pycvc.input_trace.read(str(path)))
    assert r["matched"] and r["compared"] > 100, r["report"]


@needs_host
@needs_trace
def test_6_seeking_back_runs_again_from_tick_0(session):
    """A seek back rebuilds the worlds and runs from tick 0; forward or back, the replay is the
    recording at that tick, and it still matches to the end."""
    root = session["root"]
    rec = _log(root / "rec.npz").arrays
    seen = {}

    def keep(name):
        def act(h):
            seen[name] = (h.get("trace.status"), h.get_tree("arms.stats"))

        return act

    code, hosts = _run(
        _live("--replay-trace", str(root / "s.trace"), "--set", "view.paused=1", "--frames", "12",
              "--log", str(root / "seek.npz")),
        {2: lambda h: h.set("play.pos", "0.6"), 4: keep("ahead"),
         5: lambda h: h.set("play.pos", "0.2"), 7: keep("back"),
         8: lambda h: h.set("play.pos", "1"), 10: keep("end")},
    )  # fmt: skip
    assert code == 0
    assert "all matched" in seen["ahead"][0] and "all matched" in seen["end"][0]
    seek = _log(root / "seek.npz").arrays  # the last pass: from tick 0 (the seek back) to the end
    n = len(rec["t"])
    assert len(seek["t"]) == n
    assert seek["x"].tobytes() == rec["x"].tobytes()


def _gated_nets(tmp_path):
    """Two builds of one policy: a seeded CoefMLP with one hidden unit gated on the distance to the
    goal (the unit is exactly 0 until the goal is near: SiLU of a large negative number), the
    second build's weight out of that unit changed. Until the gate opens they are bit for bit the
    same net."""
    torch = pytest.importorskip("torch")
    import grl_snam.sdf_nav as sdf_nav
    from grl_snam.tools.coef_export import write_coef_mlp

    torch.manual_seed(7)
    net = sdf_nav.CoefMLP()
    with torch.no_grad():
        first = net.net[0]
        first.weight[0].zero_()
        first.weight[0, 1] = -2000.0  # feature 1: the distance to the goal (normalized)
        first.bias[0] = 2000.0 * 0.25 - 100.0  # open below 0.25 (75 m at 1/300)
    write_coef_mlp(net, str(tmp_path / "a.cvcnav"))
    with torch.no_grad():
        net.net[2].weight[:, 0] += 0.5
    write_coef_mlp(net, str(tmp_path / "b.cvcnav"))
    return tmp_path / "a.cvcnav", tmp_path / "b.cvcnav"


@needs_host
@needs_trace
def test_6_two_policy_builds(tmp_path, capsys):
    """Replayed with the other build, the run is the recording's, byte for byte, until the first
    tick where the two builds' coefficients differ; after it, the trajectory departs."""
    a, b = _gated_nets(tmp_path)
    code, _ = _run(_live("--frames", "450", "--arms", f"gated={a}", "--record-trace",
                         str(tmp_path / "g.trace"), "--log", str(tmp_path / "a.npz")))  # fmt: skip
    assert code == 0
    capsys.readouterr()
    code, _ = _run(_live("--replay-trace", str(tmp_path / "g.trace"), "--arms", f"gated={b}",
                         "--log", str(tmp_path / "b.npz")))  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "departs from the trace at tick" in out, out
    ra, rb = _log(tmp_path / "a.npz").arrays, _log(tmp_path / "b.npz").arrays
    coef = np.stack([ra["tel_alpha"][:, 0] != rb["tel_alpha"][:, 0],
                     ra["tel_beta"][:, 0] != rb["tel_beta"][:, 0],
                     ra["tel_gamma"][:, 0] != rb["tel_gamma"][:, 0]]).any(axis=0)  # fmt: skip
    k0 = int(np.argmax(coef))
    assert coef.any() and k0 > 20, k0  # the gate holds the builds alike for a while
    for f in ("x", "y", "heading", "speed"):
        assert ra[f][:k0].tobytes() == rb[f][:k0].tobytes(), f
    assert not np.array_equal(ra["x"][k0:], rb["x"][k0:])


# ── refusals ───────────────────────────────────────────────────────────────────────────────────


@needs_host
def test_traces_refusals(tmp_path, capsys):
    if not _trace.HAS_TRACE:
        assert _main(_live("--record-trace", str(tmp_path / "x.trace"))) == 2
        assert "traces need" in capsys.readouterr().err
        return
    assert _main(["live", "austin-freedrive", "--headless", "--world", "flat",
                  "--record-trace", str(tmp_path / "x.trace")]) == 2  # the torch path  # fmt: skip
    assert _main(_live("--record-trace", "a", "--replay-trace", "b")) == 2
    (tmp_path / "bad.trace").write_text("not a trace\n")
    assert _main(_live("--replay-trace", str(tmp_path / "bad.trace"))) == 2
    err = capsys.readouterr().err
    assert "module-level" in err and "not both" in err and "bad.trace" in err


def test_the_session_hash_is_each_arms_or_theirs_together(monkeypatch):
    """One arm: its world's own hash (so libcvc replays a one-arm trace by itself); several: the
    sha256 of theirs, in order."""

    class _Fake:
        @staticmethod
        def nav_trace_state_hash(sim):
            return f"sha256:{sim}"

    class _C:
        def __init__(self, sim):
            self.sim = sim

    monkeypatch.setattr(_trace, "pycvc", _Fake)
    assert _trace.session_hash([_C("aa")]) == "sha256:aa"
    both = _trace.session_hash([_C("aa"), _C("bb")])
    assert both.startswith("sha256:") and both != _trace.session_hash([_C("bb"), _C("aa")])


def test_tracing_is_documented_where_it_is_refused():
    """The torch path's refusal and the trace flags say what they need."""
    from grl_snam.ariadne import live

    assert "module-level" in live.TORCH_ARMS
    assert "input_trace_recorder" in _trace.INSTALL
