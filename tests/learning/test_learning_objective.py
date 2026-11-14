"""grl_snam.learning.objective, arms-eval's loss cases, and learn-report's loss claims on them."""

from __future__ import annotations

import gzip
import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import grl_snam.sdf_nav as sdf_nav  # noqa: E402
from grl_snam.learning import arms as A  # noqa: E402
from grl_snam.learning import objective as O  # noqa: E402
from grl_snam.learning import report as R  # noqa: E402
from grl_snam.learning.history import load  # noqa: E402
from grl_snam.learning_history import CoefTrainMonitor, CoefTrainMonitorCfg  # noqa: E402
from grl_snam.tools import arms_eval, coef_train, sdf_eval  # noqa: E402
from grl_snam.tools.train import train_sdf  # noqa: E402

GRID = 48


@pytest.fixture(scope="module")
def sdf_npz(tmp_path_factory):
    """A 96 x 96 field over +-300 m (1 unit = 100 m) with two buildings."""
    n, half = 96, 300.0
    occ = np.zeros((n, n), bool)
    occ[30:50, 20:40] = True
    occ[55:75, 60:80] = True
    scale = 1.0 / 100.0
    bounds = (-half, -half, half, half)
    phi, nx, ny = sdf_nav.build_sdf(occ, bounds, scale)
    path = tmp_path_factory.mktemp("sdf") / "nav_sdf.npz"
    np.savez(
        path, phi=phi, normal_x=nx, normal_y=ny, bounds=np.array(bounds, np.float32),
        center=np.zeros(2, np.float32), scale=np.float32(scale), region=np.float32(280.0),
    )  # fmt: skip
    return str(path)


@pytest.fixture(scope="module")
def sdf_run(sdf_npz, tmp_path_factory):
    """A 4-step ``grl-snam train`` run with a held-out split: its checkpoints and history."""
    d = tmp_path_factory.mktemp("run")
    train_sdf(
        sdf_npz, str(d / "coef_sdf.pt"), steps=4, batch=16, horizon=8, threads=1, device="cpu",
        log_every=100, seed=0, history=str(d / "history.jsonl"), checkpoint_every=2,
        val_frac=0.25, val_blocks=4, val_pairs=16, val_ticks=10,
    )  # fmt: skip
    return d


@pytest.fixture(scope="module")
def coef_scene():
    return O.coef_scene(GRID)


def _monitor_terms(fn, seed, **kw):
    """The step-1 terms a coef-train loop logs (its history's step record)."""
    got = []
    mon = CoefTrainMonitor(CoefTrainMonitorCfg(history=None), save=lambda *a: None, meta={})
    mon._write = got.append
    fn(1, 10, 12, 0.0, seed, grid=GRID, monitor=mon, **kw)
    return next(r for r in got if r.get("kind") == "step")["terms"]


@pytest.mark.parametrize(
    "rollout,fn", [("surrogate", coef_train.train), ("bicycle", coef_train.train_bicycle)]
)
def test_coef_terms_are_the_trainers_own_at_step_zero(rollout, fn, coef_scene):
    """At lr 0 the loop's windows score a fixed net, so its logged step terms are the mean of
    the held-out scorer's per-start terms on the same draw."""
    torch.manual_seed(0)
    logged = _monitor_terms(fn, 5, model=sdf_nav.CoefMLP())
    torch.manual_seed(0)
    mine = O.coef_terms(
        sdf_nav.CoefMLP(), grid=GRID, horizon=10, n=12, seed=5, rollout=rollout, scene=coef_scene
    )
    assert set(mine) == set(O.COEF_TERMS) and all(v.shape == (12,) for v in mine.values())
    for t in O.COEF_TERMS:
        assert float(mine[t].mean()) == pytest.approx(logged[t], rel=1e-5, abs=1e-7), t


def test_sdf_terms_mean_is_the_histories_objective(sdf_npz):
    scene = sdf_eval.load_scene(sdf_npz)
    m = O.sdf_meta(scene)
    assert m["rr"] == pytest.approx(0.03) and m["horizon"] == 28
    blocks, held = O.sdf_split(val_frac=0.25, val_seed=0, val_blocks=4)
    s, g = O.sdf_heldout_pairs(scene, blocks, held, 24, 7, rr=m["rr"])
    torch.manual_seed(1)
    model = sdf_nav.CoefMLP()
    per = O.sdf_terms(model, scene, s, g, **m)
    want = sdf_eval.objective_terms(model, scene, s, g, **m)
    for t in O.SDF_TERMS:
        assert float(per[t].mean()) == pytest.approx(want[t], rel=1e-6), t
    assert np.allclose(per["loss"], per["goal"] + 3.0 * per["col"] + 0.1 * per["reg"], atol=1e-5)


def test_scoring_takes_no_gradient_keeps_the_mode_and_the_rng():
    model = sdf_nav.CoefMLP().train()
    torch.manual_seed(3)
    with O.scoring(model) as m:
        assert not torch.is_grad_enabled() and not m.training
        torch.rand(5)  # a draw inside does not advance the caller's generator
    after = torch.rand(1)
    torch.manual_seed(3)
    assert torch.equal(after, torch.rand(1)) and model.training and torch.is_grad_enabled()


def test_coef_terms_refuse_what_they_cannot_score(coef_scene):
    with pytest.raises(ValueError, match="rollout"):
        O.coef_terms(sdf_nav.CoefMLP(), rollout="jolt", scene=coef_scene)
    with pytest.raises(ValueError, match="geometry-only"):
        O.coef_terms(sdf_nav.add_risk_feature(sdf_nav.CoefMLP()), scene=coef_scene)
    with pytest.raises(ValueError, match=">= 1"):
        O.coef_terms(sdf_nav.CoefMLP(), horizon=0, scene=coef_scene)


def test_the_split_comes_from_a_history_header_or_the_runs_draw(sdf_run, tmp_path):
    hist = sdf_run / "history.jsonl"
    blocks, held = O.sdf_split(hist)
    assert (blocks, held) == (4, sdf_eval.block_split(4, 0.25, 0))
    assert O.sdf_split(val_frac=0.25, val_blocks=4) == (blocks, held)
    gz = tmp_path / "h.jsonl.gz"
    gz.write_bytes(gzip.compress(hist.read_bytes()))
    assert O.sdf_split(gz) == (blocks, held)
    bare = tmp_path / "bare.jsonl"
    bare.write_text(json.dumps({"kind": "header"}) + "\n")
    with pytest.raises(ValueError, match="no held-out blocks"):
        O.sdf_split(bare)
    with pytest.raises(ValueError, match="needs a history or val_frac"):
        O.sdf_split()
    (tmp_path / "empty.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        O.read_header(tmp_path / "empty.jsonl")


def test_arms_eval_objective_sdf_and_the_loss_claims(sdf_npz, sdf_run, tmp_path, capsys):
    out = tmp_path / "objective.json"
    hist = str(sdf_run / "history.jsonl")
    argv = [
        "--cases", "objective-sdf", "--sdf", sdf_npz, "--split-history", hist,
        "--pairs", "24", "--pair-seed", "3", "--boot", "200", "--label", "seed0",
        "--arm", f"trained={sdf_run / 'coef_sdf.pt'}",
        "--arm", f"untrained={sdf_run / 'coef_sdf.step000000.pt'}",
        "--arm", f"constant=const:hist={hist}",
        "--out", str(out),
    ]  # fmt: skip
    assert arms_eval.main(argv) == 0
    assert "24 pairs, paired" in capsys.readouterr().out
    t = json.loads(out.read_text())
    (res,) = t["results"]
    assert (res["case"], res["family"], res["unit"], res["episodes"]) == (
        "objective-sdf",
        "loss",
        "pair",
        24,
    )
    assert res["primary"] == list(O.SDF_TERMS) and t["meta"]["label"] == "seed0"
    assert all(t["direction"][m] == "down" for m in O.SDF_TERMS)
    assert "seed 3" in res["flags"][0] and "rollout rr 0.03" in res["flags"][2]
    # The untrained arm's terms are the step-0 net's on the same pairs.
    scene = sdf_eval.load_scene(sdf_npz)
    m = O.sdf_meta(scene, torch.load(sdf_run / "coef_sdf.pt")["meta"])
    s, g = O.sdf_heldout_pairs(scene, *O.sdf_split(hist), 24, 3, rr=m["rr"])
    step0 = A.load_arm_model(A.parse_arm(f"u={sdf_run / 'coef_sdf.step000000.pt'}"))
    want = O.sdf_terms(step0, scene, s, g, **m)["goal"]
    assert np.allclose(res["metrics"]["goal"]["untrained"]["values"], want)

    rep = R.report(load(hist), t, n_boot=200)
    ids = {c["id"]: c for c in rep["claims"]}
    for other in ("untrained", "constant"):
        for term in O.SDF_TERMS:
            c = ids[f"loss:objective-sdf:{term}:trained-vs-{other}"]
            assert c["family"] == "loss" and c["evidence"]["unit"] == "pair"
            assert c["statement"].startswith(f"held-out {term} (objective-sdf) is lower")
            assert c["evidence"]["diff"]["n"] == 24
    fams = [c["family"] for c in rep["claims"]]
    assert fams == sorted(fams, key=["loss", "selection", "task", "planted"].index)
    assert rep["summary"]["families"]["loss"]["pass"] + rep["summary"]["families"]["loss"][
        "fail"
    ] == sum(f == "loss" for f in fams)
    assert "task" not in rep["summary"]["families"]


def test_arms_eval_objective_coef(tmp_path, capsys):
    out = tmp_path / "objective.json"
    argv = [
        "--cases", "objective-coef", "--grid", str(GRID), "--heldout-agents", "6",
        "--heldout-seed", "9", "--horizon", "6", "--rollout", "bicycle", "--boot", "100",
        "--arm", "bias=const:bias", "--arm", "slow=const:1,6,1", "--out", str(out),
    ]  # fmt: skip
    assert arms_eval.main(argv) == 0
    capsys.readouterr()
    (res,) = json.loads(out.read_text())["results"]
    assert (res["family"], res["unit"], res["seeds"]) == ("loss", "start", list(range(6)))
    assert set(res["metrics"]) == set(O.COEF_TERMS)
    assert "bicycle rollout, horizon 6, window 6, w_coll 3" in res["flags"][1]
    want = O.coef_terms(
        A.constant_coef_mlp({"alpha": 1, "beta": 6, "gamma": 1}),
        grid=GRID, horizon=6, n=6, seed=9, rollout="bicycle",
    )  # fmt: skip
    assert np.allclose(res["metrics"]["loss"]["slow"]["values"], want["loss"])


def test_arms_eval_refuses_a_loss_case_without_its_inputs(sdf_run, capsys):
    with pytest.raises(SystemExit):
        arms_eval.main(["--arm", "a=const:", "--cases", "objective-sdf"])
    with pytest.raises(SystemExit):
        arms_eval.main(["--arm", "a=const:", "--cases", "objective-sdf", "--sdf", "x.npz"])
    capsys.readouterr()
    with pytest.raises(ValueError, match="needs sdf"):
        arms_eval.evaluate_arms([A.parse_arm("a=const:")], [0], cases=["objective-sdf"])
    # Two checkpoints trained under other rollout settings are not one objective.
    other = sdf_run / "other.pt"
    ck = torch.load(sdf_run / "coef_sdf.pt")
    ck["meta"] = dict(ck["meta"], horizon=5)
    torch.save(ck, other)
    arms = [A.parse_arm(f"a={sdf_run / 'coef_sdf.pt'}"), A.parse_arm(f"b={other}")]
    with pytest.raises(ValueError, match="other settings"):
        arms_eval.checkpoint_meta(arms)
    assert arms_eval.checkpoint_meta([A.parse_arm("c=const:")]) is None


# --------------------------------------------------------------------------- pooled runs


def test_runs_paired_diff_pools_runs_and_units():
    ref = [{0: 1.0, 1: 2.0, 2: 3.0}, {0: 2.0, 1: 2.0, 2: None}]
    arm = [{0: 0.5, 1: 1.5, 2: 2.5}, {0: 1.0, 1: 1.0, 2: 0.0}]
    d = A.runs_paired_diff(ref, arm, n_boot=300, rng=1)
    assert d["n"] == 5 and d["runs"] == 2
    assert d["mean"] == pytest.approx((-0.5 * 3 - 1.0 * 2) / 5)
    assert d["per_run"] == pytest.approx([-0.5, -1.0])
    assert d["ci95"][0] <= d["mean"] <= d["ci95"][1] < 0
    assert A.runs_paired_diff(ref, arm, n_boot=300, rng=1) == d  # seeded
    # A constant shift on every cell: an interval of width zero.
    shift = A.runs_paired_diff(ref[:1], [{u: v + 0.25 for u, v in ref[0].items()}], 200, 2)
    assert shift["ci95"] == pytest.approx([0.25, 0.25])
    assert A.runs_paired_diff([{0: None}], [{0: 1.0}]) is None
    with pytest.raises(ValueError, match="one reference per run"):
        A.runs_paired_diff(ref, arm[:1])
    level = A.runs_summary(arm, 200, random.Random(0))
    assert level["mean"] == pytest.approx(np.mean([0.5, 1.5, 2.5, 1.0, 1.0, 0.0]))


def _loss_table(label, shift, *, units=12, seed=0):
    """A loss case where trained = untrained - shift (plus noise) and constant = trained."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(1.0, 2.0, units)
    noise = rng.normal(0.0, 0.01, units)
    vals = {
        "trained": dict(enumerate(base - shift + noise)),
        "untrained": dict(enumerate(base)),
        "constant": dict(enumerate(base - shift + noise + rng.normal(0.0, 0.02, units))),
    }
    return A.arms_table(
        {"objective-x": {"loss": vals}},
        ["trained", "untrained", "constant"],
        direction={"loss": "down"},
        primary={"objective-x": ["loss"]},
        family={"objective-x": "loss"},
        unit={"objective-x": "pair"},
        n_boot=200,
        meta={"label": label},
    )


def test_loss_claims_on_one_table_and_pooled_over_runs():
    one = _loss_table("seed0", 0.2)
    claims, skipped = R.arms_claims(one, n_boot=300)
    ids = {c.id: c for c in claims}
    assert skipped == [] and {c.family for c in claims} == {"loss"}
    assert ids["loss:objective-x:loss:trained-vs-untrained"].passed
    assert not ids["loss:objective-x:loss:trained-vs-constant"].passed
    assert ids["loss:objective-x:loss:trained-vs-untrained"].evidence["from"] == "per-seed values"
    # The compatibility name returns the same claims.
    assert [c.id for c in R.task_claims(one, n_boot=300)[0]] == list(ids)

    runs = [_loss_table(f"seed{k}", 0.2, seed=k) for k in range(3)]
    claims, _ = R.arms_claims(runs, n_boot=300)
    c = {c.id: c for c in claims}["loss:objective-x:loss:trained-vs-untrained"]
    assert c.passed and c.evidence["diff"]["runs"] == 3 and c.evidence["diff"]["n"] == 36
    assert c.evidence["diff"]["labels"] == ["seed0", "seed1", "seed2"]
    assert c.evidence["trained"]["runs"] == 3 and "runs x units" in c.evidence["from"]
    # One run that did not learn widens the pooled interval across 0.
    mixed = runs[:2] + [_loss_table("seed2", -0.6, seed=2)]
    claims, _ = R.arms_claims(mixed, n_boot=300)
    assert not {c.id: c for c in claims}["loss:objective-x:loss:trained-vs-untrained"].passed


def test_pooled_claims_skip_a_table_without_values_and_refuse_a_missing_arm():
    runs = [_loss_table("seed0", 0.2), _loss_table("seed1", 0.2, seed=1)]
    for e in runs[1]["results"][0]["metrics"]["loss"].values():
        e.pop("values")
    claims, skipped = R.arms_claims(runs, against=["untrained"], n_boot=100)
    assert claims == [] and "seed1 has no per-unit values" in skipped[0]["reason"]
    runs[1]["arms"] = [a for a in runs[1]["arms"] if a["name"] != "trained"]
    with pytest.raises(ValueError, match="'seed1' has no 'trained' arm"):
        R.arms_claims(runs, n_boot=100)
    assert R.arms_claims(None) == ([], [])


def test_learn_report_cli_takes_one_table_per_run(tmp_path, capsys):
    paths = []
    for k in range(2):
        p = tmp_path / f"seed{k}.json"
        p.write_text(json.dumps(_loss_table(f"seed{k}", 0.2, seed=k)))
        paths.append(str(p))
    hist = str(tmp_path / "h.jsonl")
    (tmp_path / "h.jsonl").write_text(json.dumps({"kind": "header"}) + "\n")
    out = tmp_path / "r.json"
    argv = [hist, "--arms", paths[0], "--arms", paths[1], "--json", str(out), "--boot", "100"]
    assert R.main(argv) == 0
    rep = json.loads(out.read_text())
    assert rep["sources"]["arms"] == paths[0] and rep["sources"]["arms_tables"] == paths
    c = next(c for c in rep["claims"] if c["id"] == "loss:objective-x:loss:trained-vs-untrained")
    assert c["evidence"]["diff"]["runs"] == 2
    text = capsys.readouterr().out
    assert "PASS  loss:objective-x:loss:trained-vs-untrained" in text and "over 2 runs" in text
    assert R.main([hist, "--json", str(out)]) == 0
    assert json.loads(out.read_text())["sources"] == {
        "histories": [hist],
        "arms": None,
        "arms_tables": [],
    }
