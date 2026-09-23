"""The base-policy scorecard eval CLI — the Swarm collector -> NavScorecard training
bridge. A tiny corpus must reduce to a well-formed, in-range fitness row."""

import json

import pytest

torch = pytest.importorskip("torch")

import sdf_nav  # noqa: E402
from grl_snam.scorecard import NavScorecard  # noqa: E402
from grl_snam.tools import scorecard_eval  # noqa: E402


def test_scorecard_eval_produces_wellformed_card():
    model = sdf_nav.CoefMLP().eval()
    card = scorecard_eval.evaluate(
        model, scenes=2, agents=12, steps=40, grid=96, contact_r=5.0, checkpoint_label="test"
    )
    assert isinstance(card, NavScorecard)
    assert card.checkpoint == "test"
    assert card.n_episodes == 2
    assert card.n_vehicle_runs == 2 * 12
    assert 0.0 <= card.success_rate <= 1.0
    assert 0.0 <= card.arrival_rate <= 1.0
    assert card.mean_penetration_pct >= 0.0
    assert card.mean_path_ratio > 0.0
    d = json.loads(card.to_json())
    assert d["checkpoint"] == "test"
    assert "material_time_share" in d and len(d["material_time_share"]) == 13


def test_scorecard_eval_load_model_default_is_the_bias_net():
    m = scorecard_eval.load_model(None)
    assert m is not None  # no checkpoint -> a fresh CoefMLP, no crash


def test_scene_specs_are_distinct_per_seed():
    _, _, a = scorecard_eval.scene_specs(96, 0, 8)
    _, _, b = scorecard_eval.scene_specs(96, 1, 8)
    assert len(a) == len(b) == 8
    assert [s.start for s in a] != [s.start for s in b]  # different seeds -> different corpus


def test_coef_train_score_reports_base_scorecard(monkeypatch, capsys):
    """`coef_train --score` routes the trained model through the base scorecard
    (tools.scorecard_eval.evaluate) and reports the fitness row, not just reach."""
    import types

    from grl_snam.scorecard import NavScorecard
    from grl_snam.tools import coef_train

    called = {}

    def fake_eval(model, *, scenes, checkpoint_label):
        called["scenes"] = scenes
        called["label"] = checkpoint_label
        return NavScorecard(checkpoint=checkpoint_label, n_vehicle_runs=10, arrival_rate=0.5)

    monkeypatch.setattr("grl_snam.tools.scorecard_eval.evaluate", fake_eval)
    args = types.SimpleNamespace(out="ckpt.cvcnav", score_scenes=3, score_json="")
    coef_train._report_scorecard(object(), args)

    out = capsys.readouterr().out
    assert "base scorecard" in out and "arrival=0.500" in out
    assert called == {"scenes": 3, "label": "ckpt.cvcnav"}
