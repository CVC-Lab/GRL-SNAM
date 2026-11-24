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
import math
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


def test_how_far_a_checkpoint_drives_from_step_0():
    """apart_m: how far the car gets from step 0's at the same tick, at most; the texts say how far,
    and when it is under a car's width that the two drive as one."""
    from grl_snam.learning.runlog import RunLog

    def one(ys):
        arrays = {"t": np.array([0.05, 0.1, 0.15]), "x": np.array([[0.0], [1.0], [2.0]]),
                  "y": np.asarray(ys, float)[:, None]}  # fmt: skip
        return RunLog(arrays, {"arms": [{"name": "a", "slot": 0}], "dt": 0.05})

    first, other = one([0, 0, 0]), one([0, 5, 4])
    assert scrub.apart(other, first) == (5.0, 4.0)
    assert scrub.metrics_of(other, first)["apart_m"] == 5.0
    assert math.isnan(scrub.metrics_of(other)["apart_m"])  # no step 0 to measure from
    rows = [{"step": s} for s in (0, 100, 200)]
    metrics = [{"apart_m": 0.0}, {"apart_m": 0.3}, {"apart_m": 1.5}]
    assert scrub.apart_text(rows, metrics, 0, 0.0, 0.0, 2.2) == (
        "this is step 0 itself: the two cars are one run"
    )
    near = scrub.apart_text(rows, metrics, 1, 0.3, 0.1, 2.2)
    assert near.startswith("step 100 gets 0.3 m from step 0 at most, 0.1 m at the end:")
    assert near.endswith("step 100 drives as step 0 does")
    assert scrub.apart_text(rows, metrics, 2, 5.0, 4.0, 2.2) == (
        "step 200 gets 5.0 m from step 0 at most, 4.0 m at the end"
    )
    whole = scrub.run_apart_text(rows, metrics, 2.2)
    assert whole.startswith("its checkpoints get 1.5 m from step 0 at most (step 200),")
    assert whole.endswith("on this scenario, training has not changed how it drives")
    assert "training" not in scrub.run_apart_text(rows, metrics, 1.0)  # one parts by a car's width
    assert scrub.run_apart_text(rows, [{}, {"apart_m": 0.3}, {}], 2.2) == ""  # not all known
    table = scrub.table_rows([{**r, "selected": False, "best": False} for r in rows],
                             [{**_metrics(9.0), **m} for m in metrics], 2)  # fmt: skip
    assert [r["apart_m"] for r in table] == ["0.0", "0.3", "1.5"]
    old = scrub.table_rows([{"step": 0, "selected": False, "best": False}], [_metrics(9.0)], 0)
    assert old[0]["apart_m"] == "-"  # a report from before the distance was kept


def test_the_run_a_view_opens_on():
    from types import SimpleNamespace as Run

    runs = [Run(set="retrain", plant="humvee"), Run(set="fixed/retrain", plant="humvee"),
            Run(set="fixed/retrain", plant="kinematic")]  # fmt: skip
    assert scrub.first_run(runs) == 0
    assert scrub.first_run(runs, "fixed/retrain") == 1
    assert scrub.first_run(runs, "fixed/retrain", "kinematic") == 2
    assert scrub.first_run(runs, plant="kinematic") == 2
    assert scrub.first_run(runs, "world-units/retrain") is None


def test_a_table_rewrites_only_the_cells_that_changed():
    """live.set_rows: with what it wrote before, a table of the same rows and columns changes only
    the cells that differ; another shape is written whole."""
    from grl_snam.ariadne.live import set_rows

    class Host:
        def __init__(self):
            self.calls = []

        def set_table(self, key, rows):
            self.calls.append(("table", key, len(rows)))

        def set(self, key, value):
            self.calls.append(("set", key, value))

    host, last = Host(), {}
    set_rows(host, "t", [{"a": "1", "b": "2"}], last)
    set_rows(host, "t", [{"a": "1", "b": "3"}], last)
    set_rows(host, "t", [{"a": "1", "b": "3"}], last)
    set_rows(host, "t", [{"a": "1", "b": "3"}, {"a": "4", "b": "5"}], last)
    set_rows(host, "t", [{"a": "1", "c": "3"}, {"a": "4", "c": "5"}], last)
    set_rows(host, "u", [{"a": "1"}])
    set_rows(host, "u", [{"a": "1"}])  # without last: whole each time
    assert host.calls == [("table", "t", 1), ("set", "t.0.b", "3"), ("table", "t", 2),
                          ("table", "t", 2), ("table", "u", 1), ("table", "u", 1)]  # fmt: skip


def test_the_world_a_report_drove_on_against_the_one_rebuilt():
    """world_check: field by field; what one side lacks is not compared (nor the city's state
    path); an older generated record's flat keys read as its settings."""
    from grl_snam.ariadne._world import grid_fingerprint

    occ = np.zeros((5, 6), np.uint8)
    occ[1, 2] = 1
    grid = grid_fingerprint(occ, (-10.0, -12.0, 10.0, 12.0))
    assert grid["shape"] == [5, 6] and grid["blocked"] == round(1 / 30, 5)
    assert grid == grid_fingerprint(occ.copy(), (-10.0000001, -12.0, 10.0, 12.0))  # mm-rounded
    other = occ.copy()
    other[3, 3] = 1
    assert grid_fingerprint(other, (-10.0, -12.0, 10.0, 12.0))["sha256"] != grid["sha256"]

    gen = {"source": "generated", "settings": {"seed": "7", "half": "500"},
           "stats": {"buildings": "160", "water": "12 ponds"}, "grid": grid}  # fmt: skip
    assert scrub.world_check(gen, dict(gen)) == ([], ["source", "settings", "stats", "grid"], [])
    moved = {**gen, "grid": grid_fingerprint(other, (-10.0, -12.0, 10.0, 12.0))}
    diffs, _, _ = scrub.world_check(gen, moved)
    assert diffs == ["the navigator's grid differs (6.67% of its cells blocked here, 3.33% "
                     "deployed)"]  # fmt: skip
    diffs, _, alike = scrub.world_check(gen, moved, changed=1)  # the cells known: 1 in 30
    assert diffs == ["the navigator's grid differs in 1 of its 30 cells (3.33%)"] and not alike
    big = {**gen, "grid": {**grid, "shape": [100, 100]}}  # 1 of 10,000: the same world
    diffs, _, alike = scrub.world_check(big, {**big, "grid": {**big["grid"], "sha256": "0" * 16}},
                                        changed=1)  # fmt: skip
    assert not diffs and alike == ["its navigator's grid differs in 1 of 10,000 cells (0.01%)"]
    diffs, _, _ = scrub.world_check(gen, {**gen, "settings": {"seed": "8", "half": "500"}})
    assert diffs == ["the generator's settings differ: seed 8 here, 7 deployed"]
    # A report from before the grid was kept, its generated world recorded flat (GRL-SNAM before
    # #196): the same settings, no grid to compare, so nothing differs.
    flat = {"source": "generated", "seed": 7, "half_m": 500.0, "building_count": 160}
    assert scrub.world_check(flat, gen) == ([], ["source", "settings", "stats"], [])

    data = {"source": "data", "uri": "cvcpkg://cy-pca/scene-austin-south-small",
            "canonical": "cvcpkg://cy-pca/scene-austin-south-small@1.0.0+cvc.2",
            "sha256": "ab" * 32, "path": "scrub.data.city", "grid": grid}  # fmt: skip
    page = {**data, "path": "ui.docs.ari.data.city"}
    assert scrub.world_check(data, page) == ([], ["source", "canonical", "sha256", "grid"], [])
    diffs, _, _ = scrub.world_check(data, {**page, "canonical": data["canonical"][:-1] + "3"})
    assert diffs[0].startswith(
        "another dataset: cvcpkg://cy-pca/scene-austin-south-small@1.0.0+cvc.3"
    )
    diffs, _, _ = scrub.world_check(data, {**page, "sha256": "cd" * 32})
    assert diffs == ["the dataset's archive differs (another build of the same version?)"]
    diffs, _, _ = scrub.world_check(data, gen)
    assert diffs == ["it drove on the data world, this is the generated one"]


def test_a_report_s_grid_and_where_another_differs():
    """pack_grid / unpack_grid round-trip a grid's blocked cells; grid_changes says where another
    grid differs (cell centres, m); nearest and near_text say how near the runs come to them."""
    from grl_snam.ariadne._world import grid_changes, pack_grid, unpack_grid

    occ = np.zeros((11, 21), np.uint8)
    occ[2, 3] = occ[5, 7] = 1
    packed = pack_grid(occ)
    assert packed["shape"] == [11, 21] and np.array_equal(unpack_grid(packed), occ != 0)
    other = occ.copy()
    other[5, 7], other[9, 20] = 0, 1
    bounds = (-10.0, -5.0, 10.0, 5.0)  # 1 m cells, row 0 at the south edge
    assert grid_changes(packed, other, bounds).tolist() == [[-3.0, 0.0], [10.0, 4.0]]
    assert grid_changes(packed, np.zeros((3, 3)), bounds) is None  # another shape
    path = np.array([[-3.0, 30.0], [-3.0, 3.0]])  # 3 m north of the cell at (-3, 0)
    assert scrub.nearest(path, grid_changes(packed, other, bounds)) == pytest.approx(3.0)
    assert math.isinf(scrub.nearest(path, np.zeros((0, 2))))
    assert scrub.near_text(3.0, 4.7).startswith("these two runs pass 3.0 m from one of those cells")
    assert scrub.near_text(80.0, 4.7) == ("these two runs come no nearer than 80 m to those cells:"
                                          "\nwhat they drove past is the same here")  # fmt: skip


def test_the_scene_s_legend():
    """live.ghost_legend: a line per colour slot in use, naming the arm (and its note) whose cube,
    car and path are that colour; the other slots empty; what stacked cubes and the post mean."""
    from grl_snam.ariadne._arms import Arm
    from grl_snam.ariadne.live import ghost_legend

    arms = [Arm(name="step 0", slot=0), Arm(name="step 2800", slot=1)]
    legend = ghost_legend(arms, {"step 2800": "the checkpoint picked"}, [0.45, 1.0])
    assert legend["slot0"] == "blue cube: step 0\n  see-through car, blue path"
    assert legend["slot1"] == (
        "orange cube: step 2800 (the checkpoint picked)\n  solid car, orange path"
    )
    assert all(legend[f"slot{j}"] == "" for j in range(2, 8))
    assert "cubes stacked over one car" in legend["notes"] and "yellow post" in legend["notes"]
    names = ("trained", "untrained", "constant")
    trio = ghost_legend([Arm(name=n, slot=j) for j, n in enumerate(names)])
    assert trio["slot2"] == "green cube: constant\n  see-through car, green path"


def test_a_lean_log_cached_before_packing_is_packed_when_read(tmp_path):
    """A lean log an older scrubber cached (grl_snam.run_log/v1) is read, not deployed again, and
    packed in place (v2): the same run, smaller. Read again, it is left as it is."""
    from grl_snam.learning.runlog import PACKED_SCHEMA, SCHEMA, RunLog

    blob, hist, scenario = tmp_path / "a.step0.pt", tmp_path / "h.jsonl", {"seed": 0, "ticks": 4}
    blob.write_bytes(b"a policy")
    path = scrub.cache_path(hist, scrub._arms._sha256(blob), scrub.scenario_hash(scenario),
                            lean=True)  # fmt: skip
    col = np.linspace(-3.0, 300.0, 4, dtype=np.float32)[:, None]
    pose = ("x", "y", "heading", "speed", "goal_x", "goal_y", "stats_path_m")
    arrays = {"t": 0.05 * np.arange(1, 5), "leg_tick": np.array([0]),
              "leg_goal": np.array([[3.0, 4.0]], np.float32),
              **{k: col * (i + 1) for i, k in enumerate(pose)}}  # fmt: skip
    path.parent.mkdir(parents=True)
    meta = {"schema": SCHEMA, "lean": True, "arms": [{"name": "a"}]}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    said, rows = [], [{"step": 0.0, "blob": str(blob)}]
    (lg,) = scrub.deploy_all(None, None, "kinematic", rows, scenario, hist, lean=True,
                             log=said.append)  # fmt: skip
    assert "cached, packed" in said[0] and rows[0]["cache"] == str(path)
    back = RunLog.load(path)
    assert back.meta["schema"] == PACKED_SCHEMA
    for k, v in arrays.items():
        assert back.arrays[k].dtype == v.dtype and back.arrays[k].tobytes() == v.tobytes(), k
    stamp = path.stat().st_mtime_ns
    said.clear()
    scrub.deploy_all(None, None, "kinematic", rows, scenario, hist, lean=True, log=said.append)
    assert ", cached:" in said[0] and path.stat().st_mtime_ns == stamp


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
    # nav's kinematic plant, as live --arms drives the untrained arm below (and fast).
    args = ["scrub", str(hist), "--headless", "--world", "flat", "--ticks", str(TICKS),
            "--plant", "kinematic", "--frames", "3"]  # fmt: skip
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
    import math

    from grl_snam.ariadne import _cars
    from grl_snam.learning.runlog import RunLog

    blob = deployed["report"]["checkpoints"][0]["blob"]
    # 30 frames a second, a tick every pycvc_gl.cars.DT: libcvc's tick, 0.05 s before 4.0.2, then
    # pycvc.SIM_TICK (0.06 s in 4.0.2, 7/120 s since).
    frames = str(math.ceil(TICKS * _cars.C.DT * 30))
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

    # How far each checkpoint's car gets from step 0's: in the report, the table and the texts.
    far = [c["metrics"]["apart_m"] for c in deployed["report"]["checkpoints"]]
    assert far[0] == 0.0 and all(v > 0 for v in far[1:])
    assert [r["apart_m"] for r in host.get_tree("scrub.table")] == [f"{v:.1f}" for v in far]
    assert host.get("scrub.apart").startswith(f"step 4 gets {far[2]:.1f} m from step 0 at most")
    assert host.get("scrub.apart_run").startswith(f"its checkpoints get {max(far):.1f} m")
    farthest = deployed["report"]["checkpoints"][int(np.argmax(far))]["step"]

    seen = {}
    _, hosts = _run([*deployed["args"][:-1], "14"], {
        2: lambda h: h.emit("scrub_next"), 4: lambda h: seen.__setitem__("next", h.get("scrub.step")),
        5: click(1.7), 7: lambda h: seen.__setitem__("click", h.get("scrub.step")),
        8: lambda h: h.set("scrub.pos", "0"),
        10: lambda h: seen.update(pos=h.get("scrub.step"), stats=[r["arm"] for r in h.get_tree("arms.stats")],
                                  apart=h.get("scrub.apart")),
        11: lambda h: h.emit("scrub_far"), 13: lambda h: seen.__setitem__("far", h.get("scrub.step")),
    })  # fmt: skip
    assert seen.pop("stats") == ["step 0", "step 0 "]
    assert seen.pop("apart") == "this is step 0 itself: the two cars are one run"
    assert seen == {"next": "6", "click": "2", "pos": "0", "far": f"{farthest:g}"}
    assert "the farthest from step 0" in hosts[0].get("scrub.now")


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
def test_view_only_opens_where_it_is_told(deployed, tmp_path, capsys):
    """--data-set (and --plant) pick the run a view opens on, --at far the checkpoint whose car
    gets farthest from step 0's; a set no report has, or an --at that is no checkpoint, is
    refused."""
    a = _moved(deployed, tmp_path / "a", "seed1.history.jsonl")
    b = _moved(deployed, tmp_path / "b", "seed1.history.jsonl")
    for path, name in ((a, "retrain"), (b, "fixed/retrain")):
        rep = json.loads(path.read_text())
        rep["set"] = name
        path.write_text(json.dumps(rep))
    far = [c["metrics"]["apart_m"] for c in deployed["report"]["checkpoints"]]
    code, hosts = _run(["scrub", "--view", str(a), str(b), "--headless", "--frames", "2",
                        "--data-set", "fixed/retrain", "--plant", "kinematic", "--at", "far"])  # fmt: skip
    assert code == 0
    host = hosts[0]
    assert host.get("scrub.title").startswith("fixed/retrain/seed1:")
    assert host.get("scrub.now").startswith(f"checkpoint {int(np.argmax(far)) + 1} of 4: ")
    assert "the farthest from step 0" in host.get("scrub.now")
    assert _main(["scrub", "--view", str(a), "--headless", "--data-set", "nope"]) == 2
    assert _main(["scrub", "--view", str(a), "--headless", "--at", "first"]) == 2
    err = capsys.readouterr().err
    assert "no report is of data set 'nope'" in err and "--at first" in err


@needs_host
@pytest.mark.skipif(
    os.environ.get("GRL_SNAM_ARIADNE_GL", "0") in ("", "0"),
    reason="GRL_SNAM_ARIADNE_GL=1 runs the offscreen checks (needs a GL context)",
)
def test_view_only_says_when_its_world_is_another(deployed, tmp_path, capsys):
    """A drawn view rebuilds the report's world and compares it field by field: the deploy's own
    world says so, with the grid among the fields compared; a report whose grid was another says
    what differs."""
    report = _moved(deployed, tmp_path / "page")
    rep = json.loads(report.read_text())
    assert rep["scenario"]["world"]["grid"]["shape"]  # the deploy kept the grid's fingerprint
    assert _main(["scrub", "--view", str(report), "--offscreen", "--frames", "2"]) == 0
    err = capsys.readouterr().err
    assert "scrub: the world is the one deployed on (source, grid compared)" in err
    # The deploy's grid with one cell flipped, as another libcvc's might be.
    from grl_snam.ariadne._world import pack_grid, unpack_grid

    grid = unpack_grid(rep["world_grid"])
    grid[1, 1] = not grid[1, 1]
    rep["world_grid"] = pack_grid(grid)
    rep["scenario"]["world"]["grid"]["sha256"] = "0" * 16
    report.write_text(json.dumps(rep))
    seen = {}
    code, _ = _run(["scrub", "--view", str(report), "--offscreen", "--frames", "3"],
                   {3: lambda h: seen.update(note=h.get("scrub.world_note"),
                                             near=h.get("scrub.world_near"))})  # fmt: skip
    assert code == 0
    cells = int(np.prod(rep["scenario"]["world"]["grid"]["shape"]))
    assert seen["note"].startswith(f"the world deployed on, but its navigator's grid differs in 1 "
                                   f"of {cells:,} cells")  # fmt: skip
    assert seen["near"].startswith("these two runs ")


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


def test_the_page_opens_on_the_set_whose_checkpoints_part_most():
    assemble = _assemble()

    def rep(name, plant, far):
        cks = [{"metrics": {"apart_m": 0.0}}] + [{"metrics": {"apart_m": v}} for v in far]
        return {"set": name, "scenario": {"plant": plant}, "checkpoints": cks}

    reps = [rep("retrain", "humvee", [0.3, 0.4, 0.2]), rep("fixed/retrain", "humvee", [4.0, 80.0, 3.0]),
            rep("world-units/retrain", "humvee", [1.0, 1.2, 1.1]),
            rep("world-units/retrain", "kinematic", [90.0, 99.0, 95.0])]  # fmt: skip
    assert assemble.open_set(reps, "humvee") == "fixed/retrain"  # its median, 4.0 m
    assert assemble.open_set(reps, "kinematic") == "world-units/retrain"
    unknown = [{**r, "checkpoints": [{"metrics": {}}, {"metrics": {}}]} for r in reps]
    assert assemble.open_set(unknown, "humvee") == "retrain"  # no distances: the first set


def _glb(path, extras):
    """A glTF binary with only its JSON chunk: ``asset.extras`` as given."""
    import struct

    js = json.dumps({"asset": {"version": "2.0", "extras": extras}}).encode()
    js += b" " * (-len(js) % 4)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(js))
                     + struct.pack("<II", len(js), 0x4E4F534A) + js)  # fmt: skip
    return path


def test_assemble_takes_a_city_only_for_a_data_world(tmp_path):
    """--scrub-city names a data world's city, as a URI the page can fetch."""
    assemble = _assemble()
    host = tmp_path / "ariglx-web"
    host.mkdir()
    for name in assemble.HOST_FILES:
        (host / name).write_text(f"stand-in {name}\n")
    base = ["--host", str(host), "--out", str(tmp_path / "out"), "--scrub-only"]
    for extra in (["--scrub-city", "cvcpkg://cy-pca/scene-austin-south-small"],
                  ["--scrub-world", "data", "--scrub-city", str(tmp_path)]):  # fmt: skip
        with pytest.raises(SystemExit, match="--scrub-city is a data world's city"):
            assemble.main(base + extra)


def test_the_humvee_model_s_credit(tmp_path):
    assemble = _assemble()
    glb = _glb(tmp_path / "Humvee.glb", {"title": "A Humvee", "author": "Someone",
                                         "license": "CC-BY-4.0", "source": "https://example.org/h"})  # fmt: skip
    credit = assemble.model_credit(glb)
    assert credit.startswith(f"## Humvee.glb ({assemble.VEHICLE_MODEL})")
    assert '"A Humvee" by Someone, https://example.org/h, licensed CC-BY-4.0' in credit
    assert "licensed" not in assemble.model_credit(_glb(tmp_path / "plain.glb", {}))


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
    glb = _glb(tmp_path / "Humvee.glb", {"license": "CC-BY-4.0", "author": "Someone"})
    assert assemble.main(["--host", str(host), "--out", str(page), "--scrub-only",
                          "--scrub-histories", str(FIXTURE), "--checkpoints", str(blobs.parent),
                          "--python", sys.executable, "--scrub-world", "flat",
                          "--scrub-ticks", "40", "--scrub-plants", "kinematic",
                          "--vehicle-model", str(glb)]) == 0  # fmt: skip
    config = json.loads((page / "ariadne.config.json").read_text())
    bundle = page / config["args"][2]
    assert config["args"][:3] == ["scrub", "--view", "data/coef_train_history.scrub.zip"]
    assert config["args"][3:] == assemble.SCRUB_VIEW  # one data set: no --data-set
    # The Humvee's model ships where pycvc_gl looks in a prefix (main.py: the page), credited.
    assert assemble.VEHICLE_MODEL in config["preload"]
    assert (page / assemble.VEHICLE_MODEL).read_bytes() == glb.read_bytes()
    assert "by Someone" in (page / "THIRD_PARTY_NOTICES.md").read_text()
    import zipfile

    with zipfile.ZipFile(bundle) as z:
        rep, names = json.loads(z.read("scrub.json")), set(z.namelist())
    assert rep["scenario"]["ticks"] == 40 and len(rep["checkpoints"]) == 4
    assert [c["metrics"]["apart_m"] for c in rep["checkpoints"]][0] == 0.0
    assert all(c["cache"] in names for c in rep["checkpoints"])
    assert all(c["cache"].endswith(".lean.npz") for c in rep["checkpoints"])  # the page's are lean
    # The view reads the bundle as it is (the logs from inside it), and shows what the deploy did.
    code, hosts = _run(["scrub", "--view", str(bundle), "--headless", "--frames", "3"])
    assert code == 0
    assert hosts[0].get("scrub.now").startswith("checkpoint 3 of 4: step 4, the pick")
    assert [r["arm"] for r in hosts[0].get_tree("arms.stats")] == ["step 0", "step 4"]


@needs_host
def test_the_humvee_is_the_default_plant(deployed, tmp_path):
    """With no --plant the checkpoints drive the Jolt Humvee, which collides with the world."""
    report = tmp_path / "humvee.json"
    code, _ = _run(["scrub", str(deployed["hist"]), "--headless", "--world", "flat",
                    "--ticks", "40", "--cache", str(tmp_path / "cache"), "--report", str(report)])  # fmt: skip
    assert code == 0
    assert json.loads(report.read_text())["scenario"]["plant"] == "humvee"


@needs_host
def test_several_plants_and_the_plant_combo(deployed, tmp_path):
    """--plant twice deploys on both; the Plant combo shows the same run and step on the other."""
    seen = {}
    code, hosts = _run(["scrub", str(deployed["hist"]), "--headless", "--world", "flat",
                        "--ticks", "40", "--plant", "kinematic", "--plant", "humvee",
                        "--cache", str(tmp_path / "cache"), "--frames", "8"], {
        2: lambda h: h.emit("scrub_next"),
        4: lambda h: seen.update(first=h.get("scrub.scenario"), step=h.get("scrub.step")),
        5: lambda h: h.set("scrub.plant_choice", "humvee"),
        7: lambda h: seen.update(second=h.get("scrub.scenario"), step2=h.get("scrub.step")),
    })  # fmt: skip
    assert code == 0
    host = hosts[0]
    assert host.get("scrub.plant_count") == "2"
    assert host.get("scrub.plant_labels") == "kinematic\nhumvee"
    assert " on kinematic," in seen["first"] and " on humvee," in seen["second"]
    assert seen["step"] == seen["step2"]  # the same checkpoint, on the other plant
    assert host.get("scrub.plant_choice") == "humvee"
    # One report holds one plant's run.
    assert _main(["scrub", str(deployed["hist"]), "--headless", "--world", "flat", "--plant",
                  "kinematic", "--plant", "humvee", "--report", str(tmp_path / "r.json")]) == 2  # fmt: skip
