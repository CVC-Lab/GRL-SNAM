"""Material-aware Swarm drive: a risk-widened net (sdf_nav.add_risk_feature) can now be
DRIVEN and SCORED through the Swarm / scorecard_eval — it builds the risk-lookahead column
and applies the reroute force it was trained with. Base nets stay stats-only (drive
unchanged), and a grip (use_mu) net is still rejected (no friction field on the Swarm).
"""

from __future__ import annotations

import numpy as np
import pytest

import sdf_nav
from grl_snam.fog_stories import STORIES, shrunk
from grl_snam.material import city_material_grid
from grl_snam.scorecard import NavScorecard
from grl_snam.squad import AgentSpec
from grl_snam.swarm import Swarm
from grl_snam.tools.scorecard_eval import evaluate


def _specs(story, truth, n, seed):
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape
    fr, fc = np.nonzero(~truth)
    rng = np.random.default_rng(seed)

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    return [
        AgentSpec(
            f"a{i}",
            w(fr[rng.integers(0, len(fr))], fc[rng.integers(0, len(fc))]),
            w(fr[rng.integers(0, len(fr))], fc[rng.integers(0, len(fc))]),
        )
        for i in range(n)
    ]


def test_swarm_coef_feats_builds_risk_column_for_a_risk_net():
    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    truth = story.truth_grid()
    meta = story.meta()
    grid, idr = city_material_grid(truth, story.bounds, meta["center"], meta["scale"], seed=0)
    specs = _specs(story, truth, 6, seed=0)

    risk = sdf_nav.add_risk_feature(sdf_nav.CoefMLP())
    sw = Swarm(story, specs, model=risk, truth_occ=truth, prior_occ=truth, material=grid)
    # a step must not raise (the drive builds a 6-feature vector the 6-in net accepts)
    sw.step()
    # the feature builder yields the extra risk column
    import torch

    phi, nrm = sw.field.sample(sw.o)
    feat = sw._coef_feats(phi, nrm, sw.o + torch.ones_like(sw.o))
    assert feat.shape[1] == 6


def test_risk_net_scored_end_to_end():
    risk = sdf_nav.add_risk_feature(sdf_nav.CoefMLP())
    card = evaluate(risk, scenes=2, agents=16, steps=80, material=True)
    assert isinstance(card, NavScorecard)
    assert 0.0 <= card.arrival_rate <= 1.0
    assert len(card.material_time_share) == 13


def test_base_net_drive_unchanged_by_material_stats():
    # the PR1 invariant survives: a BASE net gets the id raster (stats) but NOT the force,
    # so its fitness is identical whether or not material is attached.
    base = sdf_nav.CoefMLP()
    on = evaluate(base, scenes=2, agents=16, steps=80, material=True)
    off = evaluate(base, scenes=2, agents=16, steps=80, material=False)
    assert on.arrival_rate == off.arrival_rate
    assert on.mean_path_ratio == off.mean_path_ratio


def test_risk_net_requires_material_to_score():
    risk = sdf_nav.add_risk_feature(sdf_nav.CoefMLP())
    with pytest.raises(ValueError, match="use_risk net needs material"):
        evaluate(risk, scenes=1, agents=8, steps=20, material=False)


def test_swarm_rejects_grip_net_without_friction():
    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    truth = story.truth_grid()
    meta = story.meta()
    grid, _ = city_material_grid(truth, story.bounds, meta["center"], meta["scale"], seed=0)
    specs = _specs(story, truth, 4, seed=0)
    mu_net = sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP())  # use_mu, no friction source on the Swarm
    sw = Swarm(story, specs, model=mu_net, truth_occ=truth, prior_occ=truth, material=grid)
    with pytest.raises(RuntimeError, match="grip .use_mu. model"):
        sw.step()
