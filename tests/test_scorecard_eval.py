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
