"""The coefficient trainers' learning instruments (grl_snam.learning_history).

The history (per step the loss terms and the pre-clip grad norm; per checkpoint each head's mean
and spread over a probe batch and its ||dtheta||/||theta||), checkpoints every N steps and the
best-on-held-out pick. Off by default, and observers when on: the trained weights are the same.
"""

from __future__ import annotations

import math

import pytest
import torch

import grl_snam.sdf_nav as sdf_nav
from grl_snam import learning_history as lh
from grl_snam.tools import coef_train

SMALL = dict(horizon=8, n=16, grid=64, seed=1)


def _monitor(tmp_path, *, every=2, fitness=None, best=True, **kw):
    fit = fitness if fitness is not None else (lambda model: 0.0)
    cfg = lh.CoefTrainMonitorCfg(
        history=str(tmp_path / "h.jsonl"),
        checkpoint_every=every,
        checkpoint_prefix=str(tmp_path / "ck"),
        best_path=str(tmp_path / "best.cvcnav") if best else None,
        **kw,
    )
    return lh.CoefTrainMonitor(cfg, save=coef_train.write_coef_mlp, fitness=fit)


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("trainer", ["bicycle", "point"])
def test_instruments_do_not_change_training(tmp_path, trainer):
    """The same run with the history, checkpoints and a held-out pick trains the same weights."""
    run = coef_train.train_bicycle if trainer == "bicycle" else coef_train.train
    plain = run(steps=4, **SMALL)
    instrumented = run(steps=4, monitor=_monitor(tmp_path), **SMALL)
    a, b = _state(plain), _state(instrumented)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_history_records_steps_checkpoints_and_the_end(tmp_path):
    coef_train.train_bicycle(steps=5, monitor=_monitor(tmp_path, every=2), **SMALL)
    recs = lh.read_history(tmp_path / "h.jsonl")
    kinds = [r["kind"] for r in recs]
    assert kinds[0] == "header" and kinds[-1] == "end"
    assert recs[0]["schema"] == lh.HISTORY_SCHEMA
    assert [r["step"] for r in recs if r["kind"] == "step"] == [1, 2, 3, 4, 5]
    assert [r["step"] for r in recs if r["kind"] == "checkpoint"] == [0, 2, 4, 5]
    step = next(r for r in recs if r["kind"] == "step")
    assert set(step["terms"]) == {"loss", "goal", "coll", "risk"}
    assert set(step["grad_norm"]) == {"mean", "max", "clipped_frac"}
    assert step["windows"] == 2  # horizon 8 / window 6
    last = [r for r in recs if r["kind"] == "checkpoint"][-1]
    assert list(last["heads"]) == ["alpha", "beta", "gamma", "trunk"]
    for head in ("alpha", "beta", "gamma"):
        assert {"mean", "spread", "dtheta_rel", "dtheta_rel_init"} <= set(last["heads"][head])
        assert last["heads"][head]["spread"] > 0.0  # the heads vary over the probe batch
        assert last["heads"][head]["dtheta_rel"] > 0.0
    for r in recs:
        if r["kind"] == "checkpoint":
            assert r["blob"] == lh.checkpoint_path(str(tmp_path / "ck"), r["step"])
            assert (tmp_path / f"ck.step{r['step']:06d}.cvcnav").exists()


def test_history_grad_norm_is_what_clip_grad_norm_returns(tmp_path, monkeypatch):
    seen: list[float] = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(*a, **k):
        n = real(*a, **k)
        seen.append(float(n))
        return n

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    coef_train.train(steps=3, monitor=_monitor(tmp_path, every=0, best=False), **SMALL)
    steps = [r for r in lh.read_history(tmp_path / "h.jsonl") if r["kind"] == "step"]
    per_step = len(seen) // len(steps)
    for i, r in enumerate(steps):
        window = seen[i * per_step : (i + 1) * per_step]
        assert r["grad_norm"]["max"] == pytest.approx(max(window), rel=1e-12)
        assert r["grad_norm"]["mean"] == pytest.approx(sum(window) / len(window), rel=1e-12)


def test_a_frozen_run_moves_no_head(tmp_path):
    coef_train.train_bicycle(steps=3, lr=0.0, monitor=_monitor(tmp_path, every=1), **SMALL)
    cks = [r for r in lh.read_history(tmp_path / "h.jsonl") if r["kind"] == "checkpoint"]
    for r in cks:
        for head, hr in r["heads"].items():
            assert hr["dtheta_rel"] == 0.0 and hr["dtheta_rel_init"] == 0.0, head
    assert cks[0]["heads"] == cks[-1]["heads"]


def test_the_pick_is_the_best_held_out_checkpoint(tmp_path):
    """A fitness peaking at the second checkpoint: best.cvcnav is that checkpoint's blob."""
    scores = iter([0.1, 0.9, 0.4, 0.3])
    mon = _monitor(tmp_path, every=2, fitness=lambda model: next(scores))
    coef_train.train_bicycle(steps=6, monitor=mon, **SMALL)
    assert [c["step"] for c in mon.checkpoints] == [0, 2, 4, 6]
    assert mon.best_step == 2 and mon.best == 0.9
    assert (tmp_path / "best.cvcnav").read_bytes() == (
        tmp_path / "ck.step000002.cvcnav"
    ).read_bytes()
    end = lh.read_history(tmp_path / "h.jsonl")[-1]
    assert end == {"kind": "end", "best_step": 2, "best_fitness": 0.9}


def test_a_best_at_step_zero_is_the_untrained_policy(tmp_path):
    mon = _monitor(tmp_path, every=1, fitness=lambda model: -mon._last_ckpt)
    coef_train.train(steps=2, monitor=mon, **SMALL)
    assert mon.best_step == 0


def test_best_path_needs_a_fitness(tmp_path):
    cfg = lh.CoefTrainMonitorCfg(best_path=str(tmp_path / "b.cvcnav"))
    with pytest.raises(ValueError, match="fitness"):
        lh.CoefTrainMonitor(cfg, save=coef_train.write_coef_mlp)
    assert not lh.CoefTrainMonitorCfg().enabled


def test_param_groups_cover_every_parameter_once():
    m = sdf_nav.CoefMLP(use_lam=True, use_lam_hard=True)
    groups = lh.coef_param_groups(m)
    assert list(groups) == ["alpha", "beta", "gamma", "lam_soft", "lam_hard", "trunk"]
    n = sum(t.numel() for ps in groups.values() for t in ps)
    assert n == sum(p.numel() for p in m.parameters())


def test_relative_change():
    a = torch.tensor([3.0, 4.0])
    assert lh.relative_change(a, a) == 0.0
    assert lh.relative_change(torch.tensor([3.0, 0.0]), a) == pytest.approx(0.8)
    assert lh.relative_change(torch.zeros(2), torch.zeros(2)) == 0.0
    assert math.isinf(lh.relative_change(torch.ones(2), torch.zeros(2)))


def test_cli_wires_the_instruments(tmp_path):
    out = tmp_path / "net.cvcnav"
    argv = ["--rollout", "bicycle", "--steps", "2", "--horizon", "8", "--n", "16", "--grid", "64"]
    argv += ["--out", str(out), "--history", str(tmp_path / "h.jsonl"), "--checkpoint-every", "1"]
    argv += ["--keep-best", "--heldout-agents", "8", "--heldout-ticks", "10"]
    coef_train.main(argv)
    assert out.exists() and (tmp_path / "net.best.cvcnav").exists()
    assert (tmp_path / "net.step000001.cvcnav").exists()
    cks = [r for r in lh.read_history(tmp_path / "h.jsonl") if r["kind"] == "checkpoint"]
    assert [c["step"] for c in cks] == [0, 1, 2]
    assert all(0.0 <= c["fitness"] <= 1.0 for c in cks)


def test_cli_refuses_a_pick_without_checkpoints_and_the_native_backend(tmp_path):
    with pytest.raises(SystemExit):
        coef_train.main(["--keep-best", "--out", str(tmp_path / "a.cvcnav")])
    with pytest.raises(SystemExit):
        coef_train.main(["--backend", "native", "--history", str(tmp_path / "h.jsonl")])
