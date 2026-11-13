"""grl_snam.learning.report: the claims learn-report makes, each pass or fail."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from grl_snam.learning import report as R
from grl_snam.learning.arms import arms_table
from grl_snam.learning.history import History, load

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "learning_history"


def _run(heldout, *, seed=0, train=None, best_at=None, planted=None):
    """A synthetic history: per epoch a held-out term, a training twin, a pick flag."""
    recs = [{"kind": "header", "schema": "test.history/v1", "seed": seed}]
    best = math.inf
    for ep, v in enumerate(heldout):
        rec = {"kind": "epoch", "epoch": ep, "heldout": {"loss": v}}
        if train is not None:
            rec["train"] = {"loss": train[ep]}
        flag = v < best if best_at is None else ep == best_at
        best = min(best, v)
        rec["best_heldout"] = flag
        if planted is not None:
            rec["heldout"]["planted"] = {
                r: {"rel_err": errs[ep], "truth": 1.0} for r, errs in planted.items()
            }
        recs.append(rec)
    return History.from_records(recs, path=f"seed{seed}.jsonl")


def _decay(seed, n=30, gain=1.0):
    rng = np.random.default_rng(seed)
    return list(1.0 - gain * (1 - np.exp(-np.arange(n) / 5)) + rng.normal(0, 0.01, n))


def _by_id(rep):
    return {c["id"]: c for c in rep["claims"]}


def test_loss_claim_across_seeds_passes_when_every_seed_learned():
    runs = [_run(_decay(s), seed=s, train=_decay(s + 10)) for s in range(3)]
    rep = R.report(runs, n_boot=500)
    c = _by_id(rep)["loss:epoch.heldout.loss"]
    assert c["status"] == "pass" and c["family"] == "loss"
    ev = c["evidence"]
    assert ev["gain"]["n"] == 3 and ev["gain"]["ci95"][0] > 0
    assert ev["runs"] == ["seed0.jsonl", "seed1.jsonl", "seed2.jsonl"]
    assert "bootstrap" in ev["basis"] and len(ev["picked"]) == 3


def test_loss_claim_fails_on_noise():
    runs = [_run(_decay(s, gain=0.0), seed=s) for s in range(3)]
    c = _by_id(R.report(runs, n_boot=500))["loss:epoch.heldout.loss"]
    assert c["status"] == "fail"


def test_one_run_uses_the_curves_own_noise_scale():
    h = load(FIX / "coef_energy_history.jsonl")
    c = _by_id(R.report(h, n_boot=200))["loss:epoch.heldout.traj"]
    assert c["status"] == "pass"  # 0.9 -> 0.55
    assert c["evidence"]["gain"] == {"mean": pytest.approx(0.35), "n": 1}
    assert c["evidence"]["basis"].startswith("1 run")
    flat = _run([1.0, 1.0, 1.01, 0.99, 1.0])
    assert _by_id(R.report([flat]))["loss:epoch.heldout.loss"]["status"] == "fail"
    short = _run([1.0])
    got = _by_id(R.report([short]))["loss:epoch.heldout.loss"]
    assert got["status"] == "fail" and "fewer than 2" in got["evidence"]["reason"]


def test_fitness_is_maximized():
    h = load(FIX / "coef_train_history.jsonl")
    c = _by_id(R.report(h))["loss:checkpoint.fitness"]
    assert c["evidence"]["mode"] == "max" and c["status"] == "pass"  # 0.40 -> 0.60
    # --maximize turns a term the other way.
    c2 = _by_id(R.report([_run([0.2, 0.5, 0.9])], maximize=["loss"]))["loss:epoch.heldout.loss"]
    assert c2["evidence"]["mode"] == "max"


def test_selection_claims():
    good = load(FIX / "coef_train_history.jsonl")
    ids = _by_id(R.report(good))
    assert ids["selection:heldout"]["status"] == "pass"
    assert ids["selection:not-untrained"]["status"] == "pass"
    assert ids["selection:heldout"]["evidence"]["runs"][0]["picked"] == 4
    stuck = _run([0.5, 0.6, 0.7], best_at=0)
    ids = _by_id(R.report([stuck]))
    assert ids["selection:heldout"]["status"] == "pass"
    assert ids["selection:not-untrained"]["status"] == "fail"
    nopick = History.from_records([{"kind": "step", "step": 1, "train": {"loss": 1.0}}])
    ids = _by_id(R.report([nopick]))
    assert ids["selection:heldout"]["status"] == "fail"
    assert R.selection_claims([]) == []


def _table():
    seeds = list(range(8))
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.8, 8)
    values = {
        "coef": {
            "reach": {
                "trained": dict(zip(seeds, base + 0.1, strict=True)),
                "untrained": dict(zip(seeds, base, strict=True)),
                # trained - constant alternates -0.3, +0.3: no difference on the average.
                "constant": dict(
                    zip(seeds, base + 0.1 + 0.3 * (-1.0) ** np.arange(8), strict=True)
                ),
            },
            "pen": {
                "trained": dict(zip(seeds, base, strict=True)),
                "untrained": dict(zip(seeds, base + 0.05, strict=True)),
                "constant": dict(zip(seeds, base + 0.05, strict=True)),
            },
            "other": {"trained": {0: 1.0}, "untrained": {0: 1.0}},
        }
    }
    return arms_table(
        values,
        ["trained", "untrained", "constant"],
        direction={"reach": "up", "pen": "down"},
        primary={"coef": ["reach", "pen"]},
        n_boot=500,
    )


def test_task_claims_from_per_seed_values():
    claims, skipped = R.task_claims(_table(), n_boot=500)
    ids = {c.id: c for c in claims}
    assert ids["task:coef:reach:trained-vs-untrained"].passed  # +0.1 on every seed
    assert not ids["task:coef:reach:trained-vs-constant"].passed  # +0 +- 0.3
    assert ids["task:coef:pen:trained-vs-untrained"].passed  # lower is better
    ev = ids["task:coef:reach:trained-vs-untrained"].evidence
    assert ev["diff"]["mean"] == pytest.approx(0.1) and ev["from"] == "per-seed values"
    assert set(ev) >= {"trained", "untrained", "direction"}
    assert skipped == []
    # An explicit metric with no direction is reported as untested.
    claims, skipped = R.task_claims(_table(), metrics=["other"], n_boot=100)
    assert claims == [] and skipped[0]["reason"] == "direction unknown"
    claims, _ = R.task_claims(_table(), metrics=["other"], better={"other": "up"}, n_boot=100)
    assert len(claims) == 1 and not claims[0].passed
    with pytest.raises(ValueError, match="no 'best' arm"):
        R.task_claims(_table(), trained="best")
    with pytest.raises(ValueError):
        R.task_claims(_table(), against=["nope"])


def test_task_claims_from_a_harness_table_without_values():
    """The A/B harness's JSON: no per-seed values, the paired difference to the first arm."""
    table = {
        "arms": [{"name": "untrained", "flags": []}, {"name": "trained", "flags": []}],
        "results": [
            {
                "case": "case_a",
                "metrics": {
                    "progress": {
                        "untrained": {"mean": 0.4, "ci95": [0.3, 0.5], "n": 20},
                        "trained": {
                            "mean": 0.5,
                            "ci95": [0.4, 0.6],
                            "n": 20,
                            "paired_diff": {"mean": 0.1, "ci95": [0.02, 0.18], "n": 20},
                        },
                    }
                },
            }
        ],
    }
    claims, _ = R.task_claims(table, better={"progress": "up"})
    assert len(claims) == 1 and claims[0].passed
    assert claims[0].evidence["from"] == "the table's paired difference"
    # Trained as the table's reference: the other arm's difference, negated.
    table["arms"].reverse()
    table["results"][0]["metrics"]["progress"] = {
        "trained": {"mean": 0.5, "ci95": [0.4, 0.6], "n": 20},
        "untrained": {
            "mean": 0.4,
            "ci95": [0.3, 0.5],
            "n": 20,
            "paired_diff": {"mean": -0.1, "ci95": [-0.18, -0.02], "n": 20},
        },
    }
    claims, _ = R.task_claims(table, better={"progress": "up"})
    assert claims[0].passed and claims[0].evidence["diff"]["ci95"] == [0.02, 0.18]
    # Neither arm is the reference and there are no values: untested.
    table["arms"].insert(0, {"name": "shipped", "flags": []})
    table["results"][0]["metrics"]["progress"]["shipped"] = {"mean": 0.4, "ci95": [0, 1], "n": 1}
    claims, skipped = R.task_claims(table, against=["untrained"], better={"progress": "up"})
    assert claims == [] and "neither arm" in skipped[0]["reason"]
    # The reference's own row has no difference to itself.
    claims, skipped = R.task_claims(table, against=["shipped"], better={"progress": "up"})
    assert claims == [] and "no paired difference" in skipped[0]["reason"]


def test_planted_claims():
    good = _run(
        [1.0, 0.7, 0.5, 0.45],
        planted={"lo": [0.9, 0.4, 0.15, 0.12], "hi": [0.8, 0.5, 0.3, 0.35]},
    )
    ids = _by_id(R.report([good], planted_bound=0.2))
    assert ids["planted:lo:recovered"]["status"] == "pass"
    assert ids["planted:lo:untrained-fails"]["status"] == "pass"
    assert ids["planted:hi:recovered"]["status"] == "fail"
    assert ids["planted:lo:recovered"]["evidence"]["runs"][0]["at"] == 3.0
    assert ids["planted:lo:recovered"]["evidence"]["runs"][0]["picked"] == pytest.approx(0.12)


def test_report_document_and_marks():
    runs = load([FIX / "coef_energy_history.jsonl", FIX / "ext_trainer_history.jsonl"])
    rep = R.report(runs, _table(), n_boot=200, sources={"histories": ["a"]})
    assert rep["schema"] == R.REPORT_SCHEMA and rep["sources"] == {"histories": ["a"]}
    assert rep["summary"]["pass"] + rep["summary"]["fail"] == len(rep["claims"])
    assert set(rep["summary"]["families"]) == {"loss", "selection", "task"}
    first = rep["runs"][0]
    assert first["schema"] == "grl_snam.coef_energy_history/v1"
    # The marks are on the smoothed curve (0.9 0.6 0.5 0.52 0.55 smooths to its minimum at the
    # end); the pick is the writer's, on the raw values.
    assert first["curves"]["epoch.heldout.traj"]["best"]["x"] == 4.0
    assert set(first["heads"]) == {"alpha", "beta", "gamma"}
    assert first["selected"]["x"] == 2.0 and len(first["checkpoints"]) == 5
    json.dumps(rep)  # serializable
    text = R.format_claims(rep)
    assert "PASS" in text and "pass," in text


def test_cli(tmp_path, capsys):
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps(_table()))
    out = tmp_path / "r.json"
    hist = str(FIX / "coef_train_history.jsonl")
    assert R.main([hist, "--arms", str(arms), "--json", str(out), "--boot", "200"]) == 0
    rep = json.loads(out.read_text())
    assert rep["sources"]["arms"] == str(arms)
    assert any(c["id"].startswith("task:") for c in rep["claims"])
    # --strict: a failing claim (trained does not beat constant on reach) exits 1.
    assert R.main([hist, "--arms", str(arms), "--strict", "--boot", "200"]) == 1
    assert R.main([hist, "--arms", str(arms), "--trained", "nope"]) == 2
    assert R.main([hist, "--arms", str(arms), "--against", "untrained", "--metric", "reach"]) == 0
    with pytest.raises(SystemExit):
        R.main([hist, "--better", "reach"])
    assert "untested" not in capsys.readouterr().err


def test_grl_snam_subcommands_pass_through(tmp_path):
    from click.testing import CliRunner

    from grl_snam.cli import main

    out = tmp_path / "r.json"
    hist = str(FIX / "coef_energy_history.jsonl")
    got = CliRunner().invoke(main, ["learn-report", hist, "--json", str(out), "--boot", "100"])
    assert got.exit_code == 0, got.output
    assert json.loads(out.read_text())["schema"] == R.REPORT_SCHEMA
    got = CliRunner().invoke(main, ["arms-eval", "--help"])
    assert got.exit_code == 0 and "--arm" in got.output
