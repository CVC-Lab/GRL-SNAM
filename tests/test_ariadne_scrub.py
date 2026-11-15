"""grl-snam ariadne scrub: every checkpoint of a training run, deployed on one scenario.

The first tests need no host: the checkpoints a history names, the cache's key, the table and the
metric chart, and side-by-side run logs. The rest deploy a copy of the coef-train fixture history
(tests/fixtures/learning_history/coef_train_history.jsonl), whose four checkpoints' blobs the test
writes, on the flat square on nav's kinematic plant (no download, no grl-snam-weights), and skip
without pycvc_gl's Python host and its multi-car template. They check that a checkpoint deployed
twice gives the same run, that checkpoint 0 deployed is the untrained arm of ``live --arms``, that
the cache is read back, and the view's controls. The view-only mode (``--view``) reads the
deploy's report back, moved elsewhere and without torch, and pages/assemble.py deploys a scrub
page.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grl_snam.ariadne import scrub
from grl_snam.learning import load

REPO = Path(__file__).resolve().parent.parent
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


def test_a_report_finds_its_files(tmp_path, monkeypatch):
    """As given, relative to the report, from the working directory, else by name beside it."""
    (tmp_path / "rep" / "logs").mkdir(parents=True)
    report = tmp_path / "rep" / "scrub.json"
    for f in ("logs/a.npz", "b.npz"):
        (tmp_path / "rep" / f).write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert scrub.find_file(report, "logs/a.npz") == tmp_path / "rep" / "logs" / "a.npz"
    assert scrub.find_file(report, str(tmp_path / "rep" / "b.npz")) == tmp_path / "rep" / "b.npz"
    assert scrub.find_file(report, "/gone/elsewhere/b.npz") == tmp_path / "rep" / "b.npz"
    assert scrub.find_file(report, "rep/b.npz") == tmp_path / "rep" / "b.npz"  # the cwd's
    assert scrub.find_file(report, "c.npz") is None and scrub.find_file(report, None) is None


def _log(path, x0):
    from grl_snam.learning.runlog import SCHEMA, RunLog

    t = np.arange(1, 4) * 0.05
    RunLog({"t": t, "x": np.full((3, 1), float(x0))}, {"schema": SCHEMA, "arms": []}).save(path)
    return path


def test_a_deployed_run_reads_each_log_when_shown(tmp_path, monkeypatch):
    monkeypatch.setattr(scrub, "KEEP_LOGS", 2)
    logs = [_log(tmp_path / f"{i}.npz", i) for i in range(4)]
    rows = [{"step": float(i), "selected": i == 2} for i in range(4)]
    d = scrub.Deployed(tmp_path / "seed3.history.jsonl.gz", -1, rows, [{}] * 4, {}, logs)
    assert d.name == "seed3" and d.pick == 2 and d._read == {}
    assert float(d.log(1).arrays["x"][0, 0]) == 1.0
    d.log(0), d.log(2), d.log(3)
    assert sorted(d._read) == [0, 3]  # at most KEEP_LOGS, step 0's always
    assert d.log(0) is d.log(0)


def test_load_report_refuses_what_it_cannot_show(tmp_path):
    bad = tmp_path / "arms.json"
    bad.write_text(json.dumps({"schema": "grl_snam.arms/v1"}))
    with pytest.raises(ValueError, match="not a scrub report"):
        scrub.load_report(bad)
    rep = tmp_path / "scrub.json"
    rep.write_text(json.dumps({"schema": scrub.REPORT_SCHEMA, "history": "gone.jsonl"}))
    with pytest.raises(ValueError, match="history 'gone.jsonl' is not there"):
        scrub.load_report(rep)
    shutil.copy(FIXTURE, tmp_path / FIXTURE.name)
    rep.write_text(json.dumps({"schema": scrub.REPORT_SCHEMA, "history": FIXTURE.name,
                               "checkpoints": [{"step": 0, "cache": "gone.npz"}]}))  # fmt: skip
    with pytest.raises(ValueError, match="step 0's run log 'gone.npz' is not there"):
        scrub.load_report(rep)


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


def _moved(deployed, dest: Path, history_name=None) -> Path:
    """The deploy's report, its history and run logs copied to ``dest``, the report naming them
    by paths that are not there (so it finds them beside itself)."""
    dest.mkdir(parents=True)
    rep = json.loads(json.dumps(deployed["report"]))
    hist = dest / (history_name or Path(rep["history"]).name)
    shutil.copy(rep["history"], hist)
    rep["history"] = f"/gone/{hist.name}"
    for c in rep["checkpoints"]:
        shutil.copy(c["cache"], dest / Path(c["cache"]).name)
        c["cache"] = f"/gone/{Path(c['cache']).name}"
    out = dest / "scrub.json"
    out.write_text(json.dumps(rep))
    return out


@needs_host
def test_view_only_shows_what_the_deploy_showed(deployed, tmp_path):
    """The view of the report, moved, is the deploy's view: the table, the pick, the Stats."""
    report = _moved(deployed, tmp_path / "page")
    code, hosts = _run(["scrub", "--view", str(report), "--headless", "--frames", "3"])
    assert code == 0
    host, first = hosts[0], deployed["host"]

    def tree(h, key):  # but the wall time: another test deploys a checkpoint again
        return [{k: v for k, v in r.items() if k != "ms"} for r in h.get_tree(key)]

    for key in ("scrub.table", "arms.stats", "ledger.coef"):
        assert tree(host, key) == tree(first, key), key
    assert host.get("scrub.now") == first.get("scrub.now")
    assert host.get("scrub.title").startswith("coef_train_history: 4 checkpoints, steps 0 to 6")
    assert host.get("scrub.made").startswith("deployed with GRL-SNAM ")


@needs_host
def test_view_only_needs_no_torch(deployed, tmp_path):
    report = _moved(deployed, tmp_path / "page")
    code = ("import sys; sys.modules['torch'] = None\n"  # import torch fails
            "from grl_snam.ariadne.__main__ import main\n"
            f"sys.exit(main(['scrub', '--view', {str(report)!r}, '--headless', '--frames', '2']))")  # fmt: skip
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]),
    }
    r = subprocess.run([sys.executable, "-s", "-c", code], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
    assert "checkpoint 3 of 4: step 4, the pick" in r.stderr


@needs_host
def test_view_only_runs_side_by_side(deployed, tmp_path):
    """Two reports: a Run combo picks one; the view opens on the first's pick."""
    a = _moved(deployed, tmp_path / "a", "seed1.history.jsonl")
    b = _moved(deployed, tmp_path / "b", "seed2.history.jsonl")
    seen = {}
    code, hosts = _run(["scrub", "--view", str(a), str(b), "--headless", "--frames", "6",
                        "--windows", "scrub,episode"], {
        2: lambda h: seen.__setitem__("first", h.get("scrub.title")),
        3: lambda h: h.set("scrub.run_choice", "seed2"),
        5: lambda h: seen.__setitem__("second", h.get("scrub.title")),
    })  # fmt: skip
    assert code == 0
    host = hosts[0]
    assert host.get("scrub.run_count") == "2" and host.get("scrub.run_labels") == "seed1\nseed2"
    assert seen["first"].startswith("seed1:") and seen["second"].startswith("seed2:")
    assert host.get("scrub.now").startswith("checkpoint 3 of 4: step 4, the pick")
    assert (host.get("tree.scrub.open"), host.get("tree.training.open")) == ("1", "0")


@needs_host
def test_view_only_data_sets_side_by_side(deployed, tmp_path):
    """Two data sets with a run of the same name: a Set combo; picking the other set shows its
    run of that name at the same training step."""
    a = _moved(deployed, tmp_path / "a", "seed1.history.jsonl")
    b = _moved(deployed, tmp_path / "b", "seed1.history.jsonl")
    for path, name in ((a, "retrain"), (b, "fixed/retrain")):
        rep = json.loads(path.read_text())
        rep["set"] = name
        path.write_text(json.dumps(rep))
    seen = {}
    code, hosts = _run(["scrub", "--view", str(a), str(b), "--headless", "--frames", "8"], {
        2: lambda h: h.emit("scrub_next"),
        4: lambda h: seen.update(first=h.get("scrub.title"), step=h.get("scrub.step")),
        5: lambda h: h.set("scrub.set_choice", "fixed/retrain"),
        7: lambda h: seen.update(second=h.get("scrub.title"), step2=h.get("scrub.step")),
    })  # fmt: skip
    assert code == 0
    host = hosts[0]
    assert host.get("scrub.set_count") == "2" and host.get("scrub.set_labels") == (
        "retrain\nfixed/retrain"
    )
    assert host.get("scrub.run_labels") == "seed1" and host.get("scrub.run_count") == "1"
    assert seen["first"].startswith("retrain/seed1:") and seen["second"].startswith(
        "fixed/retrain/seed1:"
    )
    assert seen["step"] == seen["step2"] == "6"  # Next from the pick (4), kept across the sets


@needs_host
def test_view_only_refusals(deployed, tmp_path, capsys):
    report = _moved(deployed, tmp_path / "page")
    assert _main(["scrub", str(deployed["hist"]), "--view", str(report), "--headless"]) == 2
    assert _main(["scrub", "--headless"]) == 2
    assert _main(["scrub", "--view", str(tmp_path / "none.json"), "--headless"]) == 2
    other = json.loads(report.read_text())
    other["scenario"]["run"]["world"] = "generated"
    (tmp_path / "page" / "other.json").write_text(json.dumps(other))
    assert _main(["scrub", "--view", str(report), str(tmp_path / "page" / "other.json"),
                  "--headless"]) == 2  # fmt: skip
    err = capsys.readouterr().err
    assert "not both" in err and "none.json" in err and "different worlds" in err


def _assemble():
    path = REPO / "pages" / "assemble.py"
    spec = importlib.util.spec_from_file_location("pages_assemble", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_host
def test_assemble_deploys_a_scrub_page(deployed, tmp_path):
    """pages/assemble.py --scrub-only deploys a history's checkpoints (found in
    --checkpoints/<run>/) with this Python, and the page's own report views them."""
    assemble = _assemble()
    host = tmp_path / "ariglx-web"
    host.mkdir()
    for name in assemble.HOST_FILES:
        (host / name).write_text(f"stand-in {name}\n")
    blobs = tmp_path / "checkpoints" / "coef_train_history"
    shutil.copytree(deployed["root"] / "run", blobs)
    page = tmp_path / "site" / "scrub"
    assert assemble.main(["--host", str(host), "--out", str(page), "--scrub-only",
                          "--scrub-histories", str(FIXTURE), "--checkpoints", str(blobs.parent),
                          "--python", sys.executable, "--scrub-world", "flat",
                          "--scrub-ticks", "40"]) == 0  # fmt: skip
    config = json.loads((page / "ariadne.config.json").read_text())
    bundle = page / config["args"][2]
    assert config["args"][:3] == ["scrub", "--view", "data/coef_train_history.scrub.zip"]
    import zipfile

    with zipfile.ZipFile(bundle) as z:
        rep, names = json.loads(z.read("scrub.json")), set(z.namelist())
    assert rep["scenario"]["ticks"] == 40 and len(rep["checkpoints"]) == 4
    assert all(c["cache"] in names for c in rep["checkpoints"])
    assert all(c["cache"].endswith(".lean.npz") for c in rep["checkpoints"])  # the page's are lean
    # The view reads the bundle as it is (the logs from inside it), and shows what the deploy did.
    code, hosts = _run(["scrub", "--view", str(bundle), "--headless", "--frames", "3"])
    assert code == 0
    assert hosts[0].get("scrub.now").startswith("checkpoint 3 of 4: step 4, the pick")
    assert [r["arm"] for r in hosts[0].get_tree("arms.stats")] == ["step 0", "step 4"]
