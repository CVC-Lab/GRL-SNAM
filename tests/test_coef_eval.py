"""The safety-for-reach evaluator: it must measure the BODY when given one."""

import numpy as np
import torch

import sdf_nav
from grl_snam.tools import coef_eval


def _world():
    return coef_eval.ice_band_world(grid=64)


def test_the_ice_band_is_a_band_not_a_constant():
    """A uniform grip field makes the mu feature a constant, so it could not
    test anticipation even in principle. Routes have to cross a transition."""
    _, _, rand_on, friction = _world()
    o = torch.from_numpy(rand_on(2048, np.random.default_rng(0)))
    mu = friction.sample(o)
    assert float(mu.min()) < 0.5, "no ice reachable from the free component"
    assert float(mu.max()) > 0.9, "no dry ground reachable from the free component"


def test_body_clearance_is_never_greater_than_the_point_clearance():
    """The body is a MIN over discs, so it can only be tighter than the point it
    is centred on. If this inverts, the metric is reporting the wrong extremum."""
    field, _, rand_on, _ = _world()
    o = torch.from_numpy(rand_on(512, np.random.default_rng(1)))
    th = torch.from_numpy(np.random.default_rng(2).uniform(-np.pi, np.pi, 512).astype(np.float32))
    point = coef_eval._clearance(field, o, th, None)
    body = coef_eval._clearance(field, o, th, coef_eval.FOOTPRINT)
    assert torch.all(body <= point + 1e-6)
    assert float((body < point - 1e-6).float().mean()) > 0.5, "the body is not binding anywhere"


def test_evaluate_reports_both_axes():
    """A reach number alone cannot say whether a change was good, so the
    evaluator must always return the safety side with it."""
    field, meta, rand_on, friction = _world()
    torch.manual_seed(0)
    out = coef_eval.evaluate(
        sdf_nav.CoefMLP(), field, meta, rand_on, friction=friction, n=48, ticks=20
    )
    assert set(out) == {"reach", "pen", "clear", "gap"}
    assert 0.0 <= out["reach"] <= 1.0
    assert out["pen"] >= 0.0


def test_evaluating_with_a_footprint_is_not_the_same_as_without():
    """Scoring a footprint vehicle on the reference point is the mistake this
    harness exists to prevent; the two must be distinguishable."""
    field, meta, rand_on, friction = _world()
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP()
    kw = dict(field=field, meta=meta, rand_on=rand_on, friction=friction, n=64, ticks=25)
    point = coef_eval.evaluate(model, **kw)
    body = coef_eval.evaluate(model, veh=coef_eval.FOOTPRINT, **kw)
    assert body["pen"] > point["pen"], "a body must find contacts the centre point does not"
