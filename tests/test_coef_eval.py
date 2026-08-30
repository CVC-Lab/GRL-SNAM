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
    assert body["clear"] < point["clear"], "a body must come closer than its centre point"


def test_penetration_threshold_is_overlap_not_the_stopping_margin():
    """Two thresholds were published under one name once already: penetration is
    clearance below -0.5*rr (overlapping by more than half the radius), while
    +0.5*rr is the stopping margin and fires about ten times as often. The sign
    is the whole distinction, so pin it."""
    assert coef_eval.pen_threshold(0.15) == -0.075
    assert coef_eval.pen_threshold(0.15) < 0.0, "a positive threshold is the near-miss bug"


def test_penetration_is_far_rarer_than_a_margin_violation():
    """Why the sign matters, measured: along driven trajectories the loose
    threshold fires many times more often, which is the factor by which the two
    tables disagreed."""
    field, meta, rand_on, friction = coef_eval.ice_band_world(grid=96)
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP()
    rr = meta["rr"]
    rng = np.random.default_rng(5)
    o = torch.from_numpy(rand_on(128, rng))
    goal = torch.from_numpy(rand_on(128, rng))
    th = torch.from_numpy(rng.uniform(-np.pi, np.pi, 128).astype(np.float32))
    sp = torch.zeros(128)
    overlap = margin = 0
    with torch.no_grad():
        for _ in range(60):
            al, be, ga = model(sdf_nav.coef_feats(field, o, goal))
            o, th, sp, _ = sdf_nav.bicycle_rollout(
                field,
                o,
                th,
                sp,
                goal,
                al,
                be,
                ga,
                1,
                rr=rr,
                d_hat=meta["d_hat"],
                dt=meta["dt"],
                vmax=meta["vmax"],
                **coef_eval.VEH,
            )
            phi, _ = field.sample(o)
            overlap += int((phi < 0.5 * rr).sum())
            margin += int((phi < 1.5 * rr).sum())
    assert margin > 3 * overlap, f"expected a wide gap, got margin={margin} overlap={overlap}"


def test_this_modules_penetration_is_looser_than_the_shipped_metric():
    """FogScenario.body_penetration fires on ANY overlap; this module requires
    overlap past half the radius. The direction of that inequality is the part
    worth pinning -- if it ever inverts, one of the two moved."""
    assert coef_eval.pen_threshold(0.15) < 0.0, "scenario's threshold is 0.0; ours must be below"
