"""grl-snam ariadne scrub: every checkpoint of a training run, deployed on one scenario.

The first tests need no host: the checkpoints a history names, the cache's key, the table and the
metric chart, and side-by-side run logs. The rest deploy a copy of the coef-train fixture history
(tests/fixtures/learning_history/coef_train_history.jsonl), whose four checkpoints' blobs the test
writes, on the flat square on nav's kinematic plant (no download, no grl-snam-weights), and skip
without pycvc_gl's Python host and its multi-car template. They check that a checkpoint deployed
twice gives the same run, that checkpoint 0 deployed is the untrained arm of ``live --arms``, that
the cache is read back, and the view's controls.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from grl_snam.ariadne import scrub
from grl_snam.learning import load

FIXTURE = Path(__file__).parent / "fixtures" / "learning_history" / "coef_train_history.jsonl"
#: Each checkpoint's policy: step 0 a seeded CoefMLP (the untrained net), then constants.
CONSTANTS = {2: (1.0, 4.0, 3.0), 4: (1.0, 6.0, 2.0), 6: (0.5, 8.0, 1.5)}
TICKS = 300  # past the approach to the goal, where the policies part


def _blobs(root: Path) -> Path:
    """A copy of the fixture history with its four blobs written beside it."""
    torch = pytest.importorskip("torch")
    import grl_snam.sdf_nav as sdf_nav
    from grl_snam.learning.arms import constant_coef_mlp
    from grl_snam.tools.coef_export import write_coef_mlp

    (root / "run").mkdir(parents=True, exist_ok=True)
    hist = root / FIXTURE.name
    shutil.copy(FIXTURE, hist)
    torch.manual_seed(5)
    write_coef_mlp(sdf_nav.CoefMLP(), str(root / "run" / "coef.step000000.cvcnav"))
    for step, values in CONSTANTS.items():
        write_coef_mlp(constant_coef_mlp(values), str(root / "run" / f"coef.step{step:06d}.cvcnav"))
    return hist


# ── no host ────────────────────────────────────────────────────────────────────────────────────


def test_the_checkpoints_a_history_names(tmp_path, monkeypatch):
    h = load(FIXTURE)[-1]
    rows, missing = scrub.checkpoints(FIXTURE, h)
    assert rows == [] and missing == [f"run/coef.step{s:06d}.cvcnav" for s in (0, 2, 4, 6)]
    hist = _blobs(tmp_path)
    rows, missing = scrub.checkpoints(hist, load(hist)[-1])
    assert missing == [] and [r["step"] for r in rows] == [0, 2, 4, 6]
    assert [r["selected"] for r in rows] == [False, False, True, False]  # the end record's pick
    assert rows[0]["blob"] == str((tmp_path / "run" / "coef.step000000.cvcnav").resolve())
    # A blob is found beside the history, from the working directory, or by its name beside it.
    monkeypatch.chdir(tmp_path)
    assert scrub.resolve_blob("elsewhere/h.jsonl", "run/coef.step000002.cvcnav") is not None
    assert scrub.resolve_blob(hist, "/no/where/coef.step000004.cvcnav") is None
    assert scrub.resolve_blob(hist, "other/coef.step000004.cvcnav") is None
    (tmp_path / "coef.step000004.cvcnav").write_bytes(b"x")
    assert scrub.resolve_blob(hist, "other/coef.step000004.cvcnav").name == "coef.step000004.cvcnav"


def test_the_cache_key():
    a = {"start": [0, 1], "ticks": 200, "plant": "kinematic"}
    assert scrub.scenario_hash(a) == scrub.scenario_hash(dict(reversed(list(a.items()))))
    assert scrub.scenario_hash(a) != scrub.scenario_hash({**a, "ticks": 201})
    assert scrub.cache_path("r/h.jsonl", "ab" * 32, "c" * 16).as_posix() == (
        "r/h.jsonl.scrub/" + "ab" * 8 + "-" + "c" * 16 + ".npz"
    )


def _metrics(path_m, reached=1.0):
    return {"reached": reached, "time_s": 5.0, "path_m": path_m, "pen_steps": 0.0,
            "clear_min_m": 1.25, "clear_mean_m": 3.5, "goal_m": 2.0}  # fmt: skip


def test_the_table_and_the_metric_chart():
    rows = [{"step": s, "selected": s == 4, "best": s in (0, 2)} for s in (0, 2, 4)]
    metrics = [_metrics(10.0), _metrics(12.5), _metrics(9.0, reached=0.0)]
    table = scrub.table_rows(rows, metrics, 1)
    assert [r["at"] for r in table] == ["", ">", ""]
    assert [r["flag"] for r in table] == ["best", "best", "pick"]
    assert table[2]["reached"] == "no" and table[1]["path_m"] == "12.5"
    assert table[0]["pen_steps"] == "0" and table[0]["clear_min_m"] == "1.25"
    series, markers, text = scrub.metric_chart(rows, metrics, "path_m")
    assert [s["label"] for s in series] == ["path (m)", "step 0"]
    assert series[1]["y"].tolist() == [10.0, 10.0, 10.0] and markers == [(4, "pick", "best")]
    assert text == "path (m): step 0 10.0; lowest 9.0 (step 4), highest 12.5 (step 2)"


def test_two_runs_side_by_side():
    from grl_snam.learning.runlog import RunLog

    def one(x0):
        arrays = {"t": np.array([0.05, 0.1]), "x": np.array([[x0], [x0 + 1]]),
                  "leg_tick": np.array([0]), "leg_goal": np.array([[3.0, 4.0]])}  # fmt: skip
        return RunLog(arrays, {"arms": [{"name": "a", "slot": 0}], "dt": 0.05})

    both = scrub.stack([one(0.0), one(10.0)], ["step 0", "step 9"])
    assert both.arrays["x"].tolist() == [[0, 10], [1, 11]]
    assert both.arrays["leg_goal"].tolist() == [[3.0, 4.0]]
    assert [(a["name"], a["slot"]) for a in both.meta["arms"]] == [("step 0", 0), ("step 9", 1)]


# ── deployed ───────────────────────────────────────────────────────────────────────────────────


def _host_ready() -> bool:
    from grl_snam.ariadne import _cars, _host

    return _host.HAS_HOST and _cars.HAS_CARS


needs_host = pytest.mark.skipif(
    not _host_ready(), reason="needs pycvc_gl's Python host and pycvc_gl.cars (pycvc-gl 3.7.12+)"
)


def _main(args):
    from grl_snam.ariadne.__main__ import main

    return main(args)


def _run(args, actions=None):
    """``main(args)``, keeping the hosts it opens and running ``actions`` ({frame: fn(host)})."""
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


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    root = tmp_path_factory.mktemp("scrub")
    hist = _blobs(root)
    args = ["scrub", str(hist), "--headless", "--world", "flat", "--ticks", str(TICKS),
            "--frames", "3"]  # fmt: skip
    code, hosts = _run([*args, "--report", str(root / "report.json")])
    return dict(root=root, hist=hist, args=args, code=code, host=hosts[0],
                report=json.loads((root / "report.json").read_text()))  # fmt: skip


@needs_host
def test_every_checkpoint_deployed_and_cached(deployed, capsys):
    assert deployed["code"] == 0
    rep = deployed["report"]
    assert rep["schema"] == scrub.REPORT_SCHEMA and rep["scenario"]["ticks"] == TICKS
    cks = rep["checkpoints"]
    assert [c["step"] for c in cks] == [0, 2, 4, 6] and all(Path(c["cache"]).is_file() for c in cks)
    assert len({c["metrics"]["path_m"] for c in cks}) == 4  # each policy drives its own path
    stamps = [Path(c["cache"]).stat().st_mtime_ns for c in cks]
    capsys.readouterr()
    code, _ = _run(deployed["args"])
    assert code == 0 and "cached" in capsys.readouterr().out
    assert [Path(c["cache"]).stat().st_mtime_ns for c in cks] == stamps  # read, not run again


@needs_host
def test_one_checkpoint_deployed_twice_is_the_same_run(deployed):
    from grl_snam.learning.runlog import RunLog

    path = Path(deployed["report"]["checkpoints"][2]["cache"])
    first = RunLog.load(path)
    path.unlink()
    assert _run(deployed["args"])[0] == 0
    again = RunLog.load(path)
    assert sorted(first.arrays) == sorted(again.arrays)
    for k, v in first.arrays.items():
        if k not in ("ms", "stats_ms_step"):  # the wall time is the machine's
            assert np.array_equal(v, again.arrays[k], equal_nan=True), k


@needs_host
def test_checkpoint_0_is_the_untrained_arm(deployed, tmp_path):
    """Step 0's blob deployed by the scrubber is ``live --arms untrained=BLOB``, tick for tick."""
    from grl_snam.learning.runlog import RunLog

    blob = deployed["report"]["checkpoints"][0]["blob"]
    frames = str(int(TICKS * 1.5))  # 30 frames a second, 20 ticks a second
    assert _main(["live", "austin-freedrive", "--headless", "--frames", frames, "--world", "flat",
                  "--plant", "kinematic", "--arms", f"untrained={blob}",
                  "--log", str(tmp_path / "live.npz")]) == 0  # fmt: skip
    live = RunLog.load(tmp_path / "live.npz")
    k0 = RunLog.load(deployed["report"]["checkpoints"][0]["cache"])
    n = min(len(live), len(k0))
    assert n >= TICKS - 1
    for k in (
        "x",
        "y",
        "heading",
        "speed",
        "tel_alpha",
        "tel_potential",
        "book_dU",
        "stats_path_m",
    ):
        assert np.array_equal(live.arrays[k][:n], k0.arrays[k][:n]), k
    assert live.meta["arms"][0]["sha256"] == k0.meta["arms"][0]["sha256"]


@needs_host
def test_the_view_picks_a_checkpoint(deployed):
    host = deployed["host"]
    # It opens on the history's pick, step 4, beside step 0.
    assert [r["at"] for r in host.get_tree("scrub.table")] == ["", "", ">", ""]
    assert host.get("scrub.now").startswith("checkpoint 3 of 4: step 4, the pick")
    assert [r["arm"] for r in host.get_tree("arms.stats")] == ["step 0", "step 4"]
    coef = host.get_tree("ledger.coef")
    assert (coef[1]["beta"], coef[1]["gamma"]) == ("6.000", "2.000")  # step 4's constants

    def click(step):
        def act(h):
            h.set("scrub.pick", str(step))
            h.emit("scrub_pick")

        return act

    seen = {}
    _, hosts = _run([*deployed["args"][:-1], "12"], {
        2: lambda h: h.emit("scrub_next"), 4: lambda h: seen.__setitem__("next", h.get("scrub.step")),
        5: click(1.7), 7: lambda h: seen.__setitem__("click", h.get("scrub.step")),
        8: lambda h: h.set("scrub.pos", "0"), 10: lambda h: seen.__setitem__("pos", h.get("scrub.step")),
    })  # fmt: skip
    assert seen == {"next": "6", "click": "2", "pos": "0"}
    assert [r["arm"] for r in hosts[0].get_tree("arms.stats")] == ["step 0", "step 0 "]


@needs_host
def test_the_charts(deployed):
    """On a host with chart: the metric against the step (step 0's dashed), the held-out curve,
    each with the cursor at the checkpoint."""
    host = deployed["host"]
    if not hasattr(host, "set_plot"):
        pytest.skip("needs pycvc_gl's chart: (libcvc 3.7.14, host.set_plot)")
    metric = host.get_tree("scrub.metric")
    assert [s["label"] for s in metric["s"]] == ["path (m)", "step 0"]
    assert [m["label"] for m in metric["m"]] == ["pick"] and host.get("scrub.step") == "4"
    curve = host.get_tree("scrub.curve")
    assert curve["s"] and host.get("scrub.curve_name") == "checkpoint.fitness"


@needs_host
@pytest.mark.skipif(
    os.environ.get("GRL_SNAM_ARIADNE_GL", "0") in ("", "0"),
    reason="GRL_SNAM_ARIADNE_GL=1 runs the offscreen checks (needs a GL context)",
)
def test_the_scrubber_offscreen(deployed, tmp_path):
    png = tmp_path / "scrub.png"
    args = [a if a != "--headless" else "--offscreen" for a in deployed["args"][:-1]]
    assert _main([*args, "30", "--png", str(png)]) == 0
    assert png.stat().st_size > 10_000


@needs_host
def test_refusals(tmp_path, capsys):
    assert _main(["scrub", str(FIXTURE), "--headless", "--world", "flat"]) == 2  # no blob there
    hist = _blobs(tmp_path)
    assert _main(["scrub", str(hist), "--headless", "--world", "flat", "--plant", "torch"]) == 2
    assert _main(["scrub", str(tmp_path / "none.jsonl"), "--headless"]) == 2
    err = capsys.readouterr().err
    assert "names a saved policy" in err and "sim_world" in err and "none.jsonl" in err
