"""The stagewise generator's planted dissipation (scripts/spline_stagewise6.py): ``gamma_o_regions``
sets the translational damping by the centre's x, ``fric_mu`` is the contact friction (it was a
hard-coded 0.1, still the default), and every checkpoint logs the damping it integrated with
(``gamma_o_true``)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

import scripts.spline_stagewise6 as she  # noqa: E402
import scripts.stagewise_dataset as sd  # noqa: E402


def test_gamma_o_at_reads_the_first_matching_region():
    sys = she.HyperelasticRingSystem(
        gamma_o=4.0, gamma_o_regions=[(-1.0, 1.0, 2.5), (1.0, 5.0, 6.0)]
    )
    assert sys.gamma_o_at(0.0) == 2.5
    assert sys.gamma_o_at(1.0) == 6.0
    assert sys.gamma_o_at(7.0) == 4.0
    assert she.HyperelasticRingSystem().gamma_o_at(0.0) == 4.0
    assert she.HyperelasticRingSystem().fric_mu == 0.1


def _planner(params):
    cfg = sd.GenCfg()
    sd.set_all_seeds(3)
    C, R, W = sd.sample_obstacles_case2_harder(cfg)
    world = she.WorldObstacles(C, R, W, d_hat=cfg.d_hat)
    planner = sd.planner_from_cfg(
        cfg, world, cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius,
        fric_mu=params.get("fric_mu"), gamma_o_regions=params.get("gamma_o_regions"),
    )  # fmt: skip
    return cfg, world, planner


def _episode(params, steps=40):
    cfg, world, planner = _planner(params)
    rows = []
    for t in range(steps):
        info, i0, _i1, st0, _st1, rt = planner.step(cfg.dt, world)
        ck = sd.make_checkpoint(
            t, i0, st0, info, planner.sys, planner.sys.B, rt["C"], rt["R"], rt["W"], cfg.inflate,
            rt["barrier"],
        )  # fmt: skip
        ck["norm_f_fric"] = info["dbg"]["norm_f_fric"]
        rows.append(ck)
    return rows


def test_checkpoints_log_the_damping_in_force():
    rows = _episode({"gamma_o_regions": [(-1e9, 0.05, 2.5), (0.05, 1e9, 6.0)]})
    x = np.array([r["center"][0] for r in rows])
    g = np.array([r["gamma_o_true"] for r in rows])
    assert set(g) <= {2.5, 6.0}
    # the damping is looked up at the centre BEFORE the step, the logged centre is after it
    prev_x = np.concatenate([[0.0], x[:-1]])
    assert np.array_equal(g, np.where(prev_x < 0.05, 2.5, 6.0))
    assert all(r["gamma_o_true"] == 4.0 for r in _episode({}, steps=3))


def _contact_step(**kw):
    """One step of a ring moving past an obstacle inside the barrier band."""
    sys = she.HyperelasticRingSystem(d_hat=1.0, **kw)
    sys.v_o = torch.tensor([0.5, 0.2])
    sys.omega = torch.tensor(0.3)
    obs = she.ObstacleProviderTorch(np.array([[0.9, 0.0]]), np.array([0.2]), np.array([1.0]))
    bar = she.IPCBarrier(obs, d_hat=1.0)
    return sys.step(0.03, obs, bar, she.StageForceFieldTorch(), (-1.0, 3.0, -2.0, 2.0), (2.0, 0.0))


def test_the_friction_knob_is_live():
    """The contact friction scales with fric_mu (it was a hard-coded 0.1, still the default) and
    is exactly zero at 0."""
    assert _planner({})[2].sys.fric_mu == 0.1
    f = {mu: _contact_step(fric_mu=mu)["dbg"]["norm_f_fric"] for mu in (0.0, 0.1, 0.3)}
    assert f[0.0] == 0.0 and f[0.1] > 0.0
    assert f[0.3] == pytest.approx(3.0 * f[0.1], rel=1e-5)
    assert _contact_step()["dbg"]["norm_f_fric"] == pytest.approx(f[0.1], rel=1e-6)
    g = _contact_step(gamma_o=4.0, gamma_o_regions=[(-1.0, 1.0, 2.5)])
    assert g["gamma_o_true"] == 2.5
