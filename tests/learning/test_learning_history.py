"""grl_snam.learning.history: one reader for every trainer's learning history.

Five writers, one set of conventions: the two GRL-SNAM trainers (round trips through the real
writers, plus a fixture each), libcvc's ``coef_trainer`` and two downstream trainers (fixtures in
their exact record layouts, with neutral names).
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pytest

from grl_snam.learning import history as H

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "learning_history"


def _one(name):
    runs = H.load(FIX / name)
    assert len(runs) == 1
    return runs[0]


def test_grl_snam_coef_train_history():
    h = _one("coef_train_history.jsonl")
    assert h.schema == "grl_snam.coef_train_history/v1"
    assert h.meta["heldout_seed"] == 123 and "kind" not in h.meta
    assert h["step.terms.goal"].axis == "step"
    assert h["step.terms.goal"].x.tolist() == [1, 2, 3, 4, 5, 6]
    assert h["checkpoint.fitness"].y.tolist() == [0.40, 0.55, 0.62, 0.60]
    pairs = {(p.term, p.train and p.train.name, p.heldout and p.heldout.name) for p in h.pairs()}
    assert ("fitness", None, "checkpoint.fitness") in pairs
    assert ("goal", "step.terms.goal", None) in pairs
    assert set(h.heads()) == {"alpha", "beta", "gamma", "trunk"}
    assert set(h.heads()["alpha"]) == {"mean", "spread", "dtheta_rel", "dtheta_rel_init"}
    assert set(h.grad_norms()) == {"mean", "max", "clipped_frac"}
    rows = h.checkpoints()
    assert [r.x for r in rows] == [0, 2, 4, 6]
    assert rows[0].blob == "run/coef.step000000.cvcnav"
    sel = h.selected()
    assert sel.x == 4 and sel.fitness == 0.62 and sel.heldout == {"fitness": 0.62}
    assert h.extras["end"]["best_step"] == 4
    assert "checkpoint.blob" not in h  # paths are not series


def test_the_checkpoint_table_is_built_once_per_run():
    h = _one("coef_train_history.jsonl")
    rows = h.checkpoints()
    rows.clear()  # the caller's list, not the run's
    again = h.checkpoints()
    assert [r.x for r in again] == [0, 2, 4, 6]
    assert again[2] is h.checkpoints()[2] and h.selected() is again[2]
    grown = H.History.from_records(h.records + [{"kind": "checkpoint", "step": 8}], path=h.path)
    assert [r.x for r in grown.checkpoints()] == [0, 2, 4, 6, 8]


def test_grl_snam_coef_energy_history():
    h = _one("coef_energy_history.jsonl")
    assert h.schema == "grl_snam.coef_energy_history/v1"
    traj = next(p for p in h.pairs() if p.term == "traj")
    assert traj.train.name == "epoch.train.traj" and traj.heldout.name == "epoch.heldout.traj"
    assert traj.heldout.axis == "epoch" and traj.mode == "min"
    assert traj.heldout.y.tolist() == [0.9, 0.6, 0.5, 0.52, 0.55]
    assert h["epoch.heads.alpha.within"].y[0] == 0.03
    # No end record: the pick is the last checkpoint the writer flagged a new best.
    assert h.selected().x == 2 and h.selected().record["best_heldout"] is True
    assert len(h.checkpoints()) == 5 and h.checkpoints()[0].heldout["traj"] == 0.9


def test_libcvc_coef_train_history():
    h = _one("cvc_nav_coef_train_history.jsonl")
    assert h.schema == "cvc.nav.coef_train_history/v1"
    lw = h["step.last_window_loss"]
    assert math.isnan(lw.y[2]) and len(lw.finite()) == 5  # a bare NaN is a gap
    assert {(p.term, p.train and p.train.name) for p in h.pairs()} >= {("loss", "step.loss")}
    assert h.grad_norms()["clipped_frac"].y.tolist() == [0.0] * 6
    # The end record's selected_step wins over the per-record flags (a rollback came after it).
    assert h.selected().x == 3 and h.extras["end"]["rollbacks"] == [6]
    assert [r.x for r in h.checkpoints()] == [0, 3, 6]
    assert h.checkpoints()[2].record["rolled_back"] is True


def test_downstream_trainer_history_with_two_runs():
    runs = H.load(FIX / "ext_trainer_history.jsonl")
    assert len(runs) == 2 and [h.run for h in runs] == [0, 1]
    assert runs[1].label == "ext_trainer_history.jsonl#1"
    h = runs[0]
    # step and epoch records both carry a step: x is the step for both.
    assert h["epoch.heldout.nav"].x.tolist() == [4, 8] and h["epoch.heldout.nav"].axis == "step"
    # Twins come from the same kind of record when there is one.
    nav = [p for p in h.pairs() if p.term == "nav"]
    assert {(p.train.name, p.heldout.name) for p in nav} == {
        ("step.train.nav", "step.heldout.nav"),
        ("epoch.train.nav", "epoch.heldout.nav"),
    }
    assert set(h.heads()) == {"alpha", "beta", "gamma", "head_a", "trunk"}
    assert set(h.grad_norms()) >= {"mean", "last", "clip"}
    # The pick is nested (heldout_pick.best); the epoch rows name a checkpoint path.
    assert h.selected().x == 8 and h.selected().kind == "step"
    assert [r.blob for r in h.checkpoints() if r.kind == "epoch"] == [
        "ckpt/epoch000.pt",
        "ckpt/epoch001.pt",
    ]


def test_downstream_native_history():
    h = _one("ext_native_history.jsonl")
    reach = next(p for p in h.pairs() if p.term == "reach")
    assert reach.heldout.name == "heldout.terms.reach" and reach.train.name == "step.terms.reach"
    loss = next(p for p in h.pairs() if p.term == "loss")
    assert loss.heldout.name == "heldout.loss" and loss.train.name == "step.loss"
    assert h.grad_norms() == {"value": h["step.grad_norm"]}
    rows = h.checkpoints()
    assert [r.blob for r in rows] == [
        None,
        "run/native.step000002.bin",
        "run/native.step000004.bin",
    ]
    assert h.selected().x == 2
    assert set(h.heads()) == {"head_a", "head_b", "trunk"}


def test_classify():
    assert H.classify("epoch.heldout.traj") == ("heldout", "traj")
    assert H.classify("checkpoint.heldout_terms.goal") == ("heldout", "goal")
    assert H.classify("heldout.terms.reach") == ("heldout", "reach")
    assert H.classify("heldout.loss") == ("heldout", "loss")
    assert H.classify("step.loss") == ("train", "loss")
    assert H.classify("checkpoint.fitness") == ("heldout", "fitness")
    assert H.classify("step.terms") is None
    assert H.classify("step.heads.alpha.mean") is None
    assert H.classify("step.grad_norm.mean") is None
    assert H.classify("step.lr") is None
    assert H.mode_of("fitness") == "max" and H.mode_of("traj") == "min"


def test_band_across_seeds():
    def run(seed, xs):
        recs = [{"kind": "header", "schema": "t/v1"}]
        recs += [{"kind": "e", "epoch": x, "heldout": {"t": 1.0 / (x + 1) + seed}} for x in xs]
        return H.History.from_records(recs)

    runs = [run(0.0, [0, 1, 2]), run(0.1, [0, 1, 2]), run(0.2, [0, 1])]
    b = H.band(runs, "e.heldout.t")
    assert b.x.tolist() == [0, 1, 2] and b.n.tolist() == [3, 3, 2]
    assert b.lo[0] == pytest.approx(1.0) and b.hi[0] == pytest.approx(1.2)
    assert b.mean[0] == pytest.approx(1.1) and b.median[1] == pytest.approx(0.6)
    q = H.band(runs, "e.heldout.t", q=(0.25, 0.75))
    assert q.lo[0] == pytest.approx(1.05) and q.hi[0] == pytest.approx(1.15)
    assert H.band(runs, "nope") is None
    assert b.to_dict()["n"] == [3, 3, 2]


def test_parse_rules():
    lines = [
        json.dumps({"kind": "step", "step": 1, "train": {"a": 1}}),  # no header: one run, no meta
        "",
        json.dumps({"kind": "step", "step": True, "epoch": 2, "train": {"a": 2}}),  # bool is no x
        json.dumps({"kind": "note", "msg": "hello", "train": {"a": 9}}),  # no x: an extra
    ]
    (h,) = H.parse_lines(lines)
    assert h.meta == {} and h["step.train.a"].x.tolist() == [1, 2]
    assert h["step.train.a"].at(2) == 2.0 and math.isnan(h["step.train.a"].at(7))
    assert h.extras["note"]["msg"] == "hello"
    assert h.selected() is None and h.checkpoints() == []
    assert h["step.train.a"].to_dict()["y"] == [1.0, 2.0]
    with pytest.raises(ValueError, match="not a JSON record"):
        H.parse_lines(["{bad", json.dumps({"kind": "x"})])
    with pytest.raises(ValueError, match="JSON object"):
        H.parse_lines(["[1, 2]"])
    # Tolerant: a half-written LAST line is skipped.
    (t,) = H.parse_lines(
        [json.dumps({"kind": "s", "step": 1, "loss": 3}), '{"kind": "s", "st'], tolerant=True
    )
    assert t["s.loss"].y.tolist() == [3.0]


def test_follow_rereads_a_growing_file(tmp_path):
    p = tmp_path / "live.jsonl"
    runs = H.load(p, follow=True)  # not there yet: empty, not an error
    assert runs.follow and len(runs) == 0 and runs.refresh() is False
    with p.open("w") as f:
        f.write(json.dumps({"kind": "header", "schema": "t/v1"}) + "\n")
        f.write(json.dumps({"kind": "step", "step": 1, "train": {"a": 1.0}}) + "\n")
        f.write('{"kind": "step", "step": 2, "tr')  # the trainer is mid-line
    assert runs.refresh() is True and len(runs) == 1
    assert runs[0]["step.train.a"].y.tolist() == [1.0]
    assert runs.refresh() is False  # nothing changed
    with p.open("a") as f:
        f.write('ain": {"a": 0.5}}\n')
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert runs.refresh() is True
    assert runs[0]["step.train.a"].y.tolist() == [1.0, 0.5]
    with pytest.raises(FileNotFoundError):
        H.load(tmp_path / "missing.jsonl")


def test_round_trip_coef_train_monitor(tmp_path):
    """The real coef-train writer, read back by the conventions."""
    pytest.importorskip("torch")
    from grl_snam import learning_history as lh
    from grl_snam.tools import coef_train

    fits = iter([0.2, 0.5, 0.4])
    cfg = lh.CoefTrainMonitorCfg(
        history=str(tmp_path / "h.jsonl"),
        checkpoint_every=2,
        checkpoint_prefix=str(tmp_path / "ck"),
        best_path=str(tmp_path / "best.cvcnav"),
    )
    mon = lh.CoefTrainMonitor(cfg, save=coef_train.write_coef_mlp, fitness=lambda m: next(fits))
    coef_train.train_bicycle(steps=4, horizon=8, n=16, grid=64, seed=1, monitor=mon)
    (h,) = H.load(tmp_path / "h.jsonl")
    assert h.schema == lh.HISTORY_SCHEMA
    assert h["checkpoint.fitness"].y.tolist() == [0.2, 0.5, 0.4]
    assert h.selected().x == 2 and Path(h.selected().blob).exists()
    assert set(h.heads()) >= {"alpha", "beta", "gamma", "trunk"}
    assert {"loss", "goal", "coll"} <= {p.term for p in h.pairs() if p.train is not None}
    assert len(h["step.grad_norm.mean"]) == 4


def _episode(root: Path, k: int, T: int = 14) -> str:
    import torch

    rng = random.Random(k)
    gx, gy = 4.0 + rng.uniform(-1, 1), 2.0 + rng.uniform(-1, 1)
    cks = []
    for t in range(T):
        f = t / (T - 1)
        cks.append(
            {
                "center": [f * gx + 0.1 * rng.uniform(-1, 1), f * gy],
                "stage_exit": [gx, gy],
                "obstacles_effective": {
                    "C": [[1.5 + 0.2 * k, 0.4], [2.5, 1.8 - 0.1 * k]],
                    "R_eff": [0.35, 0.3],
                    "W": [1.0, 1.0],
                },
                "barrier": {"barrier_d_hat": 0.5},
                "dt": 0.03,
            }
        )
    jl = root / f"ep{k}.jsonl"
    jl.write_text("\n".join(json.dumps(c) for c in cks) + "\n")
    pt = root / f"ep{k}.pt"
    torch.save(
        {"logs": {"checkpoints_jsonl": str(jl)}, "meta": {"dt": 0.03}, "params": {"gamma_o": 4.0}},
        pt,
    )
    return str(pt)


def test_round_trip_train_coef_energy(tmp_path):
    """The real train_coef_energy writer (an epoch history with a held-out split), read back."""
    torch = pytest.importorskip("torch")
    import grl_snam.train_coef_energy as tce

    root = tmp_path / "data"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps([{"path": _episode(root, k)} for k in range(6)]))
    hist = tmp_path / "h.jsonl"
    random.seed(0)
    torch.manual_seed(0)
    tce.main(
        ["--root", str(root), "--epochs", "3", "--bs", "8", "--workers", "0"]
        + ["--outdir", str(tmp_path / "out"), "--val-frac", "0.34", "--val-samples", "32"]
        + ["--history", str(hist)]
    )
    (h,) = H.load(hist)
    assert h.schema == "grl_snam.coef_energy_history/v1"
    traj = next(p for p in h.pairs() if p.term == "traj")
    assert traj.train.name == "epoch.train.traj" and traj.heldout.name == "epoch.heldout.traj"
    # Epochs 0..2, after an init record (epoch -1) where the writer has one.
    x = traj.heldout.x.tolist()
    assert x[-3:] == [0, 1, 2] and x[:-3] in ([], [-1.0])
    epochs = traj.heldout.x >= 0
    best = traj.heldout.x[epochs][np.argmin(traj.heldout.y[epochs])]
    assert h.selected() is not None and h.selected().x == float(best)
    assert {"alpha", "beta", "gamma"} <= set(h.heads())


def test_prefixed_heldout_terms_pair_and_metrics_take_their_direction():
    h = _one("sdf_train_history.jsonl")
    pairs = {p.term: p for p in h.pairs() if p.heldout is not None}
    assert pairs["obj_goal"].train.name == "step.terms.goal" and pairs["obj_goal"].is_loss
    assert pairs["obj_loss"].train.name == "step.terms.loss"
    assert pairs["reach"].train is None and not pairs["reach"].is_loss
    assert pairs["reach"].mode == "max" and pairs["min_clear_m"].mode == "max"
    assert pairs["gap_m"].mode == "min" and pairs["obj_goal"].mode == "min"
    assert pairs["fitness"].is_loss and pairs["fitness"].mode == "max"
    assert h.mode_of("checkpoint.heldout.reach") == "max"
    assert h.mode_of("checkpoint.nope") == "min"
    assert h.meta["fitness"] == "heldout seek-drive reach"  # a string here: just meta
    assert h.selected().x == 2100


def test_a_header_can_declare_what_to_maximize():
    recs = [{"kind": "header", "maximize": ["score_x"]}]
    recs += [{"kind": "e", "epoch": i, "heldout": {"score_x": i, "other": i}} for i in range(3)]
    h = H.History.from_records(recs)
    modes = {p.term: p.mode for p in h.pairs()}
    assert modes == {"score_x": "max", "other": "min"}
    assert H.History.from_records([{"kind": "header", "maximize": "x"}]).maximize == ()


def test_init_record_joins_the_first_evaluated_series():
    h = _one("coef_energy_history_planted.jsonl")
    traj = h["epoch.heldout.traj"]
    assert traj.x.tolist() == [-1.0, 0.0, 1.0] and traj.axis == "epoch"
    assert "init.heldout.traj" not in h
    assert h["epoch.heads.alpha.mean"].x[0] == -1.0
    rows = h.checkpoints()
    assert rows[0].kind == "init" and rows[0].x == -1.0
    assert rows[0].heldout["traj"] == pytest.approx(0.007456472463672981)
    # planted values are recorded as series but are not loss terms.
    assert "epoch.heldout.planted.lo.rel_err" in h
    assert not any("planted" in p.term for p in h.pairs())
    # An init record with nothing else, or with its own x.
    (only,) = H.parse_lines([json.dumps({"kind": "init", "heldout": {"t": 1.0}})])
    assert only["init.heldout.t"].x.tolist() == [0.0]
    (own,) = H.parse_lines(
        [
            json.dumps({"kind": "init", "step": 5, "heldout": {"t": 2.0}}),
            json.dumps({"kind": "c", "step": 10, "heldout": {"t": 1.0}}),
        ]
    )
    assert own["c.heldout.t"].x.tolist() == [5.0, 10.0]
