"""SdfNavigator + live metrics on a synthetic OPEN field (no compiled bindings needed)."""

import numpy as np

import sdf_nav
from grl_snam.metrics import NavMetrics, NavStats, hud_lines
from grl_snam.nav import SdfNavigator


def _open_field():
    """A flat, obstacle-free SDF (clearance everywhere) + matching physics meta."""
    n = 64
    phi = np.full((n, n), 5.0, np.float32)  # 5 normalized units of clearance everywhere
    zeros = np.zeros((n, n), np.float32)
    bounds = (-100.0, -100.0, 100.0, 100.0)
    scale = 0.05  # 200 world units -> 10 normalized
    field = sdf_nav.SDFField(phi, zeros, zeros, bounds, (0.0, 0.0), scale)
    meta = dict(
        scale=scale,
        center=(0.0, 0.0),
        region=100.0,
        rr=0.15,
        d_hat=0.35,
        dt=0.06,
        nsub=1,
        vmax=0.9,
        bounds=list(bounds),
    )
    return field, meta


def test_navigator_makes_progress_and_emits_metrics():
    field, meta = _open_field()
    model = sdf_nav.CoefMLP()
    model.eval()
    nav = SdfNavigator(field, model, meta, reach_tol=0.8)
    nav.start((-50.0, 0.0), (50.0, 0.0))
    metrics, best_i, _o, _v = nav.drive_to_goal(max_steps=2000)
    assert metrics, "navigator produced no steps"
    assert isinstance(metrics[best_i], NavMetrics)
    assert metrics[best_i].goal_dist_m < metrics[0].goal_dist_m  # closed distance to the goal
    last = metrics[-1]
    assert last.alpha > 0 and last.beta > 0 and last.gamma > 0  # network emitted coefficients
    assert last.clearance_m > 0  # open field


def test_hud_lines_and_stats():
    m = NavMetrics(
        step=3,
        goal_dist_m=42.0,
        clearance_m=8.0,
        alpha=1.0,
        beta=3.0,
        gamma=4.0,
        speed_mps=6.5,
        mode="seek",
    )
    stats = NavStats()
    stats.update(m)
    lines = hud_lines(m, stats)
    assert any("coeffs" in ln for ln in lines)
    assert any("reached" in ln for ln in lines)
    assert stats.steps == 1


def test_goals_reached_counts_distinct_not_per_frame():
    stats = NavStats()
    # goal 0 reported reached over 3 frames, then goal 1 over 2 frames -> 2 distinct, not 5
    for _ in range(3):
        stats.update(NavMetrics(goal_index=0, reached=True))
    for _ in range(2):
        stats.update(NavMetrics(goal_index=1, reached=True))
    assert stats.goals_reached == 2
    assert stats.steps == 5
