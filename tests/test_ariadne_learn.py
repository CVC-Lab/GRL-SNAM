"""grl-snam ariadne learn: the learning instruments as an Ariadne document.

The charts' series and markers and the tables' rows are built by pure functions, tested here on
the learning-history fixtures with no host. The port itself needs pycvc_gl's Python host, and its
charts pycvc_gl's chart: (libcvc 3.7.14, host.set_plot)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from grl_snam.ariadne import PORTS, _host, learn
from grl_snam.learning import History, load
from grl_snam.learning.arms import arms_table
from grl_snam.learning.report import report

FIX = Path(__file__).resolve().parent / "fixtures" / "learning_history"
SDF = FIX / "sdf_train_history.jsonl"
EXT = FIX / "ext_trainer_history.jsonl"
PLANTED = FIX / "coef_energy_history_planted.jsonl"

needs_host = pytest.mark.skipif(
    not _host.HAS_HOST, reason="needs pycvc_gl's Python host (pycvc_gl.ariadne)"
)
needs_chart = pytest.mark.skipif(
    not (_host.HAS_HOST and hasattr(_host.Host, "set_plot")),
    reason="needs pycvc_gl's chart: (libcvc 3.7.14, host.set_plot)",
)


def _seeds(n=3):
    """n runs of one synthetic trainer: a held-out loss term that falls, its twin, a metric."""
    runs = []
    for k in range(n):
        rng = np.random.default_rng(k)
        recs = [{"kind": "header", "schema": "test/v1", "heads": ["alpha", "trunk"]}]
        for i in range(30):
            v = 1.0 * math.exp(-i / 6) + 0.2 + rng.normal(0, 0.005)
            recs.append({"kind": "step", "step": 10 * i, "terms": {"goal": v * 0.9}})
            recs.append(
                {
                    "kind": "checkpoint",
                    "step": 10 * i,
                    "heldout": {"obj_goal": v, "reach": 0.2 + 0.01 * i},
                    "best": i == 20,
                    "grad_norm": 2.0 - 0.05 * i,
                    "heads": {
                        "alpha": {"mean": 1 + 0.01 * i, "spread": 0.1, "dtheta_rel": 0.01},
                        "trunk": {"dtheta_rel": 0.02, "dtheta_rel_init": 0.1},
                    },
                    "blob": f"run{k}/c.step{10 * i:06d}.cvcnav",
                }
            )
        runs.append(History.from_records(recs, path=f"/x/seed{k}.history.jsonl"))
    return runs


def test_the_port_is_registered():
    assert PORTS["learn"][0] == "grl_snam.ariadne.learn"
    assert (_host.DOCS / "learn.ari").is_file()


def test_names():
    runs = _seeds(2) + [History.from_records([{"kind": "x", "step": 1}])]
    assert learn.run_labels(runs) == ["seed0", "seed1", "run 2"]
    two = load([EXT])
    assert learn.run_labels(two) == ["ext_trainer_history", "ext_trainer_history#1"]
    dup = [History.from_records([], path="/a/h.jsonl"), History.from_records([], path="/b/h.jsonl")]
    assert learn.run_labels(dup) == ["h (0)", "h (1)"]
    names = learn.heldout_names(load(SDF))
    # The loss terms (a training twin, or fitness) first, then the held-out metrics.
    assert names[0] == "checkpoint.fitness" and "checkpoint.heldout.obj_goal" in names[:5]
    assert names.index("checkpoint.heldout.reach") > names.index("checkpoint.heldout.obj_reg")
    assert learn.head_names(load(SDF)) == ["alpha", "beta", "gamma", "trunk"]


def test_loss_chart_band_and_each():
    runs = _seeds(3)
    series, markers, log_y, text = learn.loss_chart(runs, "checkpoint.heldout.obj_goal")
    assert [s["label"] for s in series] == ["held-out, 3 runs", "train"]
    assert series[0]["style"] == "c0" and series[1]["style"] == "c1 dashed"
    assert np.all(series[0]["lo"] <= series[0]["y"]) and np.all(series[0]["y"] <= series[0]["hi"])
    kinds = {k for _, _, k in markers}
    assert {"best", "elbow"} <= kinds and not any("pick" in label for _, label, _ in markers)
    assert "trained on step.terms.goal" in text and "picked at seed0 200" in text
    assert log_y is False  # 0.2 .. 1.2 spans less than a decade
    series, markers, _, text = learn.loss_chart(runs, "checkpoint.heldout.obj_goal", "each")
    assert [s["label"] for s in series][:2] == ["seed0 held-out", "seed0 train"]
    assert series[2]["style"] == "c1" and series[3]["style"] == "c1 dashed"
    assert "pick (seed2)" in [label for _, label, _ in markers]
    reach = learn.loss_chart(runs, "checkpoint.heldout.reach")[3]
    assert "higher is better" in reach and "(a metric)" in reach
    assert learn.loss_chart(runs, "nope") == ([], [], False, "no run has nope")
    one = learn.loss_chart(runs[:1], "checkpoint.heldout.obj_goal")
    assert [s["label"] for s in one[0]] == ["held-out", "train"] and ("pick" in one[3])


def test_log_y_for_a_curve_over_decades():
    recs = [{"kind": "e", "epoch": i, "heldout": {"t": 10.0**-i}, "train": {"t": 10.0**-i}}
            for i in range(6)]  # fmt: skip
    assert learn.loss_chart([History.from_records(recs)], "e.heldout.t")[2] is True


def test_heads_grad_and_dtheta():
    runs = _seeds(2)
    series, markers, _, text = learn.heads_chart(runs, "alpha")
    assert len(series) == 2 and series[0]["lo"][0] == pytest.approx(0.9)
    assert "seed0: mean 1 -> 1.29, spread 0.1 -> 0.1" in text
    assert learn.heads_chart(runs, "nope")[3] == "no run records a mean for nope"
    series, _, log_y, text = learn.grad_chart(runs[:1])
    assert [s["label"] for s in series] == ["seed0"] and "grad norm 2 -> 0.55" in text
    assert learn.grad_chart([History()])[3] == "no run records a gradient norm"
    sdf = load(SDF)
    series, _, _, text = learn.grad_chart(sdf)
    assert [s["label"] for s in series] == ["sdf_train_history", "max"]
    assert "clipped on 100% of steps" in text
    series, _, log_y, text = learn.dtheta_chart(runs[0])
    assert [s["label"] for s in series] == ["alpha", "trunk"] and log_y is True
    assert "trunk 0.1" in text
    assert learn.dtheta_chart(None)[3] == "no run"


def test_checkpoint_rows():
    (h,) = load(SDF)
    rows = learn.checkpoint_rows(h, "checkpoint.heldout.obj_goal")
    assert [r["x"] for r in rows] == ["0", "100", "1000", "2100", "4000"]
    assert rows[3]["mark"] == "* pick" and rows[0]["mark"] == "best"
    assert rows[3]["blob"] == "coef_sdf.step002100.pt"
    assert rows[0]["value"] == "0.9637"
    assert learn.checkpoint_rows(h)[3]["value"] == "0.08008"  # the fitness
    assert learn.checkpoint_rows(None) == []
    big = History.from_records(
        [{"kind": "checkpoint", "step": i, "fitness": i, "best": i == 7} for i in range(1000)]
    )
    rows = learn.checkpoint_rows(big)
    assert len(rows) <= learn.MAX_ROWS + 2 and rows[-1]["x"] == "999"
    assert any(r["mark"] for r in rows)


def _table():
    seeds = list(range(10))
    base = np.linspace(0.2, 0.6, 10)
    values = {
        "coef": {
            "reach": {
                "trained": dict(zip(seeds, base + 0.1, strict=True)),
                "untrained": dict(zip(seeds, base, strict=True)),
                "constant": dict(
                    zip(seeds, base + 0.1 + 0.3 * (-1.0) ** np.arange(10), strict=True)
                ),
            },
            "pen": {
                "trained": dict(zip(seeds, base + 0.05, strict=True)),
                "untrained": dict(zip(seeds, base, strict=True)),
            },
        }
    }
    return arms_table(
        values,
        ["trained", "untrained", "constant"],
        direction={"reach": "up", "pen": "down"},
        primary={"coef": ["pen"]},
        n_boot=300,
    )


def test_arms_rows():
    rows = learn.arms_rows(_table(), n_boot=300)
    assert [(r["metric"], r["arm"]) for r in rows][:2] == [
        ("pen (primary)", "trained"),
        ("pen (primary)", "untrained"),
    ]
    by = {(r["metric"], r["arm"]): r for r in rows}
    assert by[("pen (primary)", "trained")]["diff"].endswith("* worse")  # +0.05 pen
    assert by[("reach", "trained")]["diff"].endswith("* better")
    assert by[("reach", "untrained")]["diff"] == ""  # the reference
    assert "*" not in by[("reach", "constant")]["diff"]
    assert by[("reach", "trained")]["value"].startswith("0.5 [")
    assert learn.arms_rows(None) == []
    harness = {"arms": [{"name": "a"}, {"name": "b"}], "results": [{"case": "x", "metrics": {
        "m": {"a": {"mean": 1.0, "ci95": [0.9, 1.1], "n": 5},
              "b": {"mean": 1.2, "ci95": [1.0, 1.4], "n": 5,
                    "paired_diff": {"mean": 0.2, "ci95": [0.1, 0.3], "n": 5}}}}}]}  # fmt: skip
    rows = learn.arms_rows(harness)
    assert rows[1]["diff"] == "+0.2 [+0.1, +0.3] *"  # no direction: no better / worse


def test_verdict_and_planted_rows():
    runs = load([PLANTED])
    rep = report(runs, _table(), n_boot=200)
    rows = learn.verdict_rows(rep)
    assert len(rows) == len(rep["claims"]) and {r["status"] for r in rows} <= {"PASS", "FAIL"}
    assert any("too short" in r["detail"] for r in rows)
    assert any(r["detail"].startswith("picks at") for r in rows)
    assert any(r["detail"].startswith("picked ") for r in rows if r["family"] == "planted")
    task = next(r for r in rows if r["family"] == "task")
    assert task["detail"].endswith("n=10")
    summary = learn.verdict_summary(rep)
    assert summary.startswith(f"{rep['summary']['pass']} of {len(rep['claims'])} claims pass")
    assert learn.verdict_summary(None).startswith("No claims yet")
    planted = {r["region"]: r for r in learn.planted_rows(rep)}
    assert set(planted) == {"lo", "hi", "ratio"}
    assert planted["lo"]["recovered"] == "FAIL" and planted["lo"]["untrained"] == "PASS"
    assert planted["lo"]["picked"] == "0.573" and planted["lo"]["start"] == "0.647"
    assert learn.planted_rows(None) == [] and learn.verdict_rows(None) == []
    one_run = {"evidence": {"gain": {"mean": 0.3, "n": 1}, "margin": 0.1}}
    assert learn.claim_detail(one_run) == "+0.3 (1 run; margin 0.1)"
    assert learn.claim_detail({"evidence": {}}) == ""


def test_marks_text():
    from grl_snam.learning import marks

    m = marks(np.arange(5.0), [3.0, 2.0, 1.5, 1.4, 1.39])
    assert learn.marks_text(m).startswith("best 4 (")
    assert learn.marks_text(marks([], [])) == "no points"


# ── the port ───────────────────────────────────────────────────────────────────────────────────


@needs_host
def test_headless_run_writes_the_report(tmp_path):
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps(_table()))
    out = tmp_path / "r.json"
    argv = ["--headless", str(SDF), str(PLANTED), "--arms", str(arms), "--report", str(out)]
    assert learn.main(argv + ["--boot", "200"]) == 0
    rep = json.loads(out.read_text())
    assert rep["schema"] == "grl_snam.learn_report/v1" and rep["sources"]["arms"] == str(arms)
    assert learn.main(argv + ["--boot", "200", "--strict"]) == 1  # some claims fail


@needs_host
def test_refusals(tmp_path, capsys):
    assert learn.main(["--headless"]) == 2
    assert learn.main(["--headless", str(tmp_path / "missing.jsonl")]) == 2
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps(_table()))
    assert learn.main(["--headless", str(SDF), "--arms", str(arms), "--trained", "best"]) == 2
    assert "no 'best' arm" in capsys.readouterr().err


def _open(argv):
    args = _host.parse(["--headless", *argv], learn._args)
    host = _host.open_host("learn.ari", "learn", args)
    return host, args


@needs_host
def test_the_state_tree(tmp_path):
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps(_table()))
    host, args = _open([str(SDF), str(PLANTED), "--arms", str(arms), "--boot", "200"])
    try:
        port = learn.LearnPort(host, load(args.hist), _table(), args)
        port.make_report()
        port.publish_all()
        host.run(fps=30, frames=2)
        assert host.get("learn.term") == "checkpoint.fitness"
        assert host.get("learn.run") == "sdf_train_history"
        assert host.get_int("learn.claims") == len(port.rep["claims"])
        assert host.get("learn.claims.0.status") in ("PASS", "FAIL")
        assert host.get_int("learn.checkpoints") == 5
        assert host.get("learn.checkpoints.3.mark") == "* pick"
        assert host.get_int("learn.arms") == 5 and host.get_int("learn.planted") == 3
        assert "higher is better" in host.get("learn.loss_text")
        # A choice in a window redraws its panels on the next frame.
        host.set("learn.run", "coef_energy_history_planted")
        host.set("learn.term", "epoch.heldout.traj")
        port.frame(0.0)
        assert host.get_int("learn.checkpoints") == 3
        assert (
            host.get("learn.checkpoints.0.x") == "-1"
            and host.get("learn.checkpoints.0.kind") == "init"
        )
        assert host.get("learn.ck_caption").startswith("coef_energy_history_planted: 3 checkpoints")
    finally:
        host.close()


@needs_host
def test_follow_redraws_a_growing_history(tmp_path):
    live = tmp_path / "live.jsonl"
    host, args = _open([str(live), "--follow", "--every", "0"])
    try:
        port = learn.LearnPort(host, load(args.hist, follow=True), None, args)
        port.make_report()
        port.publish_all()
        assert host.get("learn.caption") == "0 run(s): none yet"
        lines = [json.dumps({"kind": "header", "schema": "t/v1"})]
        for i in range(6):
            lines.append(json.dumps({"kind": "e", "epoch": i, "heldout": {"t": 1 / (1 + i)},
                                     "train": {"t": 1 / (1 + i)}}))  # fmt: skip
        live.write_text("\n".join(lines) + "\n")
        port.frame(0.0)
        assert host.get("learn.caption") == "1 run(s): t/v1"
        assert host.get("learn.term") == "e.heldout.t"
        assert host.get_int("learn.claims") >= 1
    finally:
        host.close()


@needs_chart
def test_the_charts(tmp_path):
    host, args = _open([str(SDF), "--boot", "200"])
    try:
        port = learn.LearnPort(host, load(args.hist), None, args)
        port.make_report()
        port.publish_all()
        assert host.get("learn.loss.s.0.label") == "held-out"
        assert host.get("learn.loss.m.0.kind") in ("best", "elbow", "plateau", "note")
        assert host.get("learn.head_chart.s.0.label") == "sdf_train_history alpha"
        assert host.get("learn.grad.s.0.label") == "sdf_train_history"
        assert host.get("learn.dtheta.s.0.label") == "alpha"
    finally:
        host.close()
