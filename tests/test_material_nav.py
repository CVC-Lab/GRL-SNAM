"""Oracles for the faithful material-aware research core (material_nav.py).

Three layers of evidence:
  1. Golden gate decisions generated from the source implementation
     (SetasAditya/material-aware-grl-snam rebuttal_experiments/exp1) — pure
     numpy scalar arithmetic, platform-stable, committed as literals here.
  2. Hand-derived closed-form and float64-recompute oracles for the barrier,
     the patch sampler, and the integrator (semi-implicit order, force signs,
     accumulators) — independent of torch version quirks.
  3. tests/test_material_fork_xcheck.py additionally re-proves bit-identity
     against a live fork checkout when GRL_SNAM_MATERIAL_FORK is set.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import material_nav as mnav

# ---------------------------------------------------------------------------
# Witness gate
# ---------------------------------------------------------------------------

GATE_KW = dict(
    primitive_count=16,
    horizon_cells=12,
    hard_margin_m=1.0,
    improvement_margin=0.05,
    material_trigger=0.45,
)

# Generated from the source repo's primitive_feasibility_gate on the exact
# inputs rebuilt below from default_rng(20260827) (bit-stream is stable per
# numpy's RNG compatibility policy).  Fields: active, nominal_risk, best_risk,
# feasible_count, direction_rc, endpoint_rc, min_clearance_m.
GATE_GOLDEN = [
    (
        False,
        0.44541666470468044,
        0.31491666721800965,
        4,
        (-0.9238795042037964, -0.3826834261417389),
        (9.913446426391602, 16.467798233032227),
        1.2170000076293945,
    ),
    (
        False,
        0.46391667146235704,
        0.4927499988116324,
        4,
        (0.3826834261417389, -0.9238795042037964),
        (11.062200546264648, 0.3434467315673828),
        1.215000033378601,
    ),
    (
        True,
        0.6466666633884112,
        0.5214166585355997,
        1,
        (-0.7071067690849304, 0.7071067690849304),
        (3.4447193145751953, 25.865280151367188),
        1.3370000123977661,
    ),
    (
        True,
        0.49258333413551253,
        0.42158333615710336,
        4,
        (-0.9238795042037964, 0.3826834261417389),
        (9.063446044921875, 10.012201309204102),
        1.875,
    ),
    (
        True,
        0.6585833306113879,
        0.5550833319624265,
        2,
        (-0.9238795042037964, -0.3826834261417389),
        (0.1734466552734375, 6.297799110412598),
        1.2740000486373901,
    ),
    (
        False,
        0.5266666660706202,
        float("inf"),
        0,
        (0.0, 0.0),
        (6.840000152587891, 2.9200000762939453),
        float("nan"),
    ),
    (
        False,
        0.6335000041872263,
        float("inf"),
        0,
        (0.0, 0.0),
        (13.34000015258789, 20.479999542236328),
        float("nan"),
    ),
    (
        False,
        0.3895000144839287,
        float("inf"),
        0,
        (0.0, 0.0),
        (16.920000076293945, 21.170000076293945),
        float("nan"),
    ),
]


def _golden_inputs():
    rng = np.random.default_rng(20260827)
    for i in range(8):
        h, w = 24, 26
        risk = np.round(rng.random((h, w)), 3).astype(np.float32)
        hard = (rng.random((h, w)) < (0.02 if i % 2 else 0.0)).astype(np.uint8)
        lo = 1.2 if i < 5 else 0.0
        sdf = np.round(lo + rng.random((h, w)) * 3, 3).astype(np.float32)
        pos = np.round(np.array([rng.uniform(2, w - 3), rng.uniform(2, h - 3)]), 2).astype(
            np.float32
        )
        goal = np.round(np.array([rng.uniform(2, w - 3), rng.uniform(2, h - 3)]), 2).astype(
            np.float32
        )
        yield {"risk_map": risk, "hard_mask": hard, "sdf_hard": sdf}, pos, goal


def test_gate_matches_source_goldens_exactly():
    for (maps, pos, goal), exp in zip(_golden_inputs(), GATE_GOLDEN):
        g = mnav.primitive_feasibility_gate(maps, pos, goal, **GATE_KW)
        assert g.active == exp[0]
        assert g.nominal_risk == exp[1]
        assert g.best_risk == exp[2]
        assert g.feasible_count == exp[3]
        assert g.selected_direction_rc == exp[4]
        assert g.selected_endpoint_rc == exp[5]
        if math.isnan(exp[6]):
            assert math.isnan(g.selected_min_clearance_m)
        else:
            assert g.selected_min_clearance_m == exp[6]


def _open_maps(h=40, w=40, risk=0.0):
    return {
        "risk_map": np.full((h, w), risk, np.float32),
        "hard_mask": np.zeros((h, w), np.uint8),
        "sdf_hard": np.full((h, w), 10.0, np.float32),
    }


def test_gate_uniform_risk_never_activates():
    # No ray can improve on the nominal by the margin when risk is constant.
    maps = _open_maps(risk=0.9)
    g = mnav.primitive_feasibility_gate(
        maps, np.array([20.0, 20.0], np.float32), np.array([35.0, 20.0], np.float32), **GATE_KW
    )
    assert g.feasible_count > 0
    assert not g.active
    assert g.nominal_risk == pytest.approx(0.9)
    assert g.nominal_risk - g.best_risk < GATE_KW["improvement_margin"]


def test_gate_activates_on_cheaper_lateral_corridor():
    # High-risk band straight ahead, clean corridor below: gate must fire.
    maps = _open_maps()
    maps["risk_map"][:, :] = 0.05
    maps["risk_map"][15:26, 22:40] = 0.9  # band covering the nominal ray
    pos = np.array([20.0, 20.0], np.float32)  # (x=col, y=row)
    goal = np.array([35.0, 20.0], np.float32)
    g = mnav.primitive_feasibility_gate(maps, pos, goal, **GATE_KW)
    assert g.active
    assert g.nominal_risk >= GATE_KW["material_trigger"]
    assert g.nominal_risk - g.best_risk >= GATE_KW["improvement_margin"]


def test_gate_endpoint_progress_filter():
    # Goal one cell away: every 12-cell ray endpoint overshoots => no candidate
    # can make progress, feasible_count == 0, inactive.
    maps = _open_maps(risk=0.9)
    g = mnav.primitive_feasibility_gate(
        maps, np.array([20.0, 20.0], np.float32), np.array([21.0, 20.0], np.float32), **GATE_KW
    )
    assert g.feasible_count == 0
    assert not g.active


def test_gate_hard_wall_blocks_rays_and_lambda_hard_is_never_gated():
    # A hard ring around the agent: all rays infeasible.  (The lam_hard channel
    # is by-contract independent of the gate; asserted at the integrator level.)
    maps = _open_maps(risk=0.9)
    maps["hard_mask"][17:24, 17:24] = 1
    maps["hard_mask"][19:22, 19:22] = 0
    g = mnav.primitive_feasibility_gate(
        maps, np.array([20.0, 20.0], np.float32), np.array([35.0, 20.0], np.float32), **GATE_KW
    )
    assert g.feasible_count == 0
    assert not g.active
    assert g.best_risk == float("inf")


def test_ray_cost_out_of_bounds_is_infeasible_and_keeps_sampled_prefix():
    maps = _open_maps(risk=0.9)
    # Walking up from row 1: sample 1 lands at row 0 (in bounds), sample 2 at
    # row -1 (OOB float check, pre-rounding) => infeasible, mean over prefix.
    risk_mean, feasible, _ = mnav._ray_cost(
        maps,
        np.array([1.0, 5.0], np.float32),
        np.array([-1.0, 0.0], np.float32),
        horizon_cells=12,
        hard_margin_m=1.0,
    )
    assert not feasible
    assert risk_mean == pytest.approx(0.9)


def test_gate_low_risk_trigger_suppresses():
    maps = _open_maps()
    maps["risk_map"][:, :] = 0.1
    maps["risk_map"][18:23, 24:40] = 0.3  # below the 0.45 trigger
    g = mnav.primitive_feasibility_gate(
        maps, np.array([20.0, 20.0], np.float32), np.array([35.0, 20.0], np.float32), **GATE_KW
    )
    assert not g.active
    assert g.nominal_risk < GATE_KW["material_trigger"]


def test_gate_cell_rounding_is_half_even():
    # risk differs between cells 10 and 12 on the row axis; a sample landing
    # exactly at row 10.5/11.5 must round half-to-even (10 and 12).
    maps = _open_maps()
    maps["risk_map"][10, :] = 0.25
    maps["risk_map"][11, :] = 0.5
    maps["risk_map"][12, :] = 0.75
    r, _, _ = mnav._ray_cost(
        maps,
        np.array([9.5, 5.0], np.float32),
        np.array([1.0, 0.0], np.float32),  # walk down rows: samples 10.5, 11.5, ...
        horizon_cells=2,
        hard_margin_m=1.0,
    )
    # samples at rows 10.5 -> 10 (0.25) and 11.5 -> 12 (0.75); mean 0.5
    assert r == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Hazard barrier
# ---------------------------------------------------------------------------


def test_sdf_barrier_grad_closed_form():
    phi = torch.tensor([3.0, 0.0, 50.0], dtype=torch.float32)
    b, db = mnav.sdf_barrier_grad(phi, d_hat_sdf=3.0, k_sharp=5.0)
    # at phi == d_hat: db/dphi = -sigmoid(0) = -0.5 exactly; b = softplus(0)/k
    assert db[0].item() == pytest.approx(-0.5)
    assert b[0].item() == pytest.approx(math.log(2.0) / 5.0, rel=1e-6)
    # deep inside hazard: full push
    assert db[1].item() == pytest.approx(-1.0, abs=1e-6)
    # far away: force fades to zero
    assert db[2].item() == pytest.approx(0.0, abs=1e-6)
    assert b[2].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Patch sampler
# ---------------------------------------------------------------------------


def test_bilinear_sample_patch_center_and_integer_offsets():
    p = torch.arange(5 * 5, dtype=torch.float32).reshape(1, 1, 5, 5)
    o0 = torch.tensor([[100.0, 200.0]])
    # centre => centre pixel (2,2) = 12
    s = mnav.bilinear_sample_patch(p, o0.clone(), o0)
    assert s.item() == pytest.approx(12.0)
    # +1 col => (2,3) = 13 ; +1 row => (3,2) = 17
    s = mnav.bilinear_sample_patch(p, o0 + torch.tensor([[1.0, 0.0]]), o0)
    assert s.item() == pytest.approx(13.0)
    s = mnav.bilinear_sample_patch(p, o0 + torch.tensor([[0.0, 1.0]]), o0)
    assert s.item() == pytest.approx(17.0)
    # halfway => bilinear average
    s = mnav.bilinear_sample_patch(p, o0 + torch.tensor([[0.5, 0.0]]), o0)
    assert s.item() == pytest.approx(12.5)


def test_bilinear_sample_patch_border_clamp():
    p = torch.arange(3 * 3, dtype=torch.float32).reshape(1, 1, 3, 3)
    o0 = torch.tensor([[0.0, 0.0]])
    far = o0 + torch.tensor([[100.0, 0.0]])  # way past the right border
    s = mnav.bilinear_sample_patch(p, far, o0)
    assert s.item() == pytest.approx(5.0)  # (1,2) — right edge of centre row


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------


def _one_agent_case(lam_soft=2.0, lam_hard=4.0):
    o0 = torch.tensor([[10.0, 12.0]])
    v0 = torch.tensor([[0.3, -0.2]])
    goal = torch.tensor([[20.0, 12.0]])
    C = torch.tensor([[[13.0, 12.5]]])
    R = torch.tensor([[1.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    alphas = torch.tensor([[0.8]])
    beta = torch.tensor([1.1])
    gamma = torch.tensor([0.4])
    patch = torch.zeros(1, 6, 9, 9)
    patch[0, 0] = 0.6  # constant risk
    patch[0, 1] = 2.0  # phi metres (inside the d_hat_sdf=3 activation band)
    patch[0, 2] = 0.05  # dr/dx
    patch[0, 3] = -0.02  # dr/dy
    patch[0, 4] = 0.7  # dphi/dx
    patch[0, 5] = 0.1  # dphi/dy
    return dict(
        o0=o0,
        v0=v0,
        goal=goal,
        C=C,
        R=R,
        mask=mask,
        alphas=alphas,
        beta=beta,
        gamma=gamma,
        lam_soft=torch.tensor([lam_soft]),
        lam_hard=torch.tensor([lam_hard]),
        rollout_patch=patch,
        d_hat=torch.tensor([2.75]),
        dt=torch.tensor([0.01]),
        H=torch.tensor([2]),
        robot_radius=1.5,
        margin_factor=0.5,
        d_hat_sdf=3.0,
    )


def _manual_step(o, v, kw, dt=0.01):
    """Float64 recompute of one integrator step (constant patch => no sampling)."""
    o = o.astype(np.float64)
    v = v.astype(np.float64)
    beta, gamma = kw["beta"].item(), kw["gamma"].item()
    lam_s, lam_h = kw["lam_soft"].item(), kw["lam_hard"].item()
    goal = kw["goal"].numpy().astype(np.float64)[0]
    c = kw["C"].numpy().astype(np.float64)[0, 0]
    r_eff = kw["R"].item() + 0.5 * 1.5
    d_hat = kw["d_hat"].item()

    f_goal = -beta * (o - goal)
    diff = o - c
    r = max(np.linalg.norm(diff), 1e-9)
    n_hat = diff / r
    d = r - r_eff
    if d < d_hat:
        dbdd = (d_hat - d) * (2.0 * math.log(max(d, 1e-9) / d_hat) - d_hat / d) + 1.0
        dbdd = min(max(dbdd, -200.0), 200.0)
    else:
        dbdd = 0.0
    f_geom = -(kw["alphas"].item() * dbdd) * n_hat
    f_soft = -lam_s * np.array([0.05, -0.02])
    db_dphi = -1.0 / (1.0 + math.exp(-5.0 * (3.0 - 2.0)))
    f_hard = -lam_h * db_dphi * np.array([0.7, 0.1])
    f_tot = f_goal + f_geom + f_soft + f_hard - gamma * v
    v_new = v + dt * f_tot  # velocity first ...
    o_new = o + dt * v_new  # ... position uses NEW velocity (semi-implicit)
    return o_new, v_new


def test_integrator_matches_float64_recompute_two_steps():
    kw = _one_agent_case()
    oT, vT, min_clear, cum_risk, hard_count, arc = mnav.integrate_surrogate_material(**kw)
    o = kw["o0"].numpy()[0].copy()
    v = kw["v0"].numpy()[0].copy()
    exp_arc = 0.0
    for _ in range(2):
        o2, v2 = _manual_step(o, v, kw)
        exp_arc += np.linalg.norm(o2 - o)
        o, v = o2, v2
    assert np.allclose(oT.numpy()[0], o, atol=1e-5)
    assert np.allclose(vT.numpy()[0], v, atol=1e-5)
    # the integrator accumulates in float32; the recompute is float64
    assert arc.item() == pytest.approx(exp_arc, rel=1e-4)
    assert cum_risk.item() == pytest.approx(0.6 * exp_arc, rel=1e-4)
    # phi = 2.0 < 1.0 is false => hard_count counts nothing here
    assert hard_count.item() == 0.0
    assert min_clear.item() > 0.0


def test_integrator_semi_implicit_order_is_load_bearing():
    # An explicit-Euler recompute (position uses the OLD velocity) must NOT match.
    kw = _one_agent_case()
    oT, *_ = mnav.integrate_surrogate_material(**kw)
    o = kw["o0"].numpy()[0].astype(np.float64)
    v = kw["v0"].numpy()[0].astype(np.float64)
    for _ in range(2):
        o_new_explicit = o + 0.01 * v  # wrong (the source repo's stale surrogate copy)
        _, v = _manual_step(o, v, kw)
        o = o_new_explicit
    assert not np.allclose(oT.numpy()[0], o, atol=1e-7)


def test_integrator_zero_lambdas_reduce_to_geometry_only():
    kw_mat = _one_agent_case(lam_soft=3.0, lam_hard=5.0)
    kw_geo = _one_agent_case(lam_soft=0.0, lam_hard=0.0)
    # zero the material channels in the "material" run too => identical fields
    kw_mat["rollout_patch"][:, 2:] = 0.0
    out_m = mnav.integrate_surrogate_material(**kw_mat)
    out_g = mnav.integrate_surrogate_material(**kw_geo)
    for a, b in zip(out_m, out_g):
        assert torch.equal(a, b)


def test_integrator_hard_count_and_clamps():
    kw = _one_agent_case()
    kw["rollout_patch"][0, 1] = 0.5  # phi < 1.0 everywhere
    kw["rollout_patch"][0, 0] = 1.7  # risk raw > 1 => clamped to 1.0 in cum_risk
    _, _, _, cum_risk, hard_count, arc = mnav.integrate_surrogate_material(**kw)
    assert hard_count.item() == 2.0  # both steps near-hazard
    assert cum_risk.item() == pytest.approx(arc.item(), rel=1e-6)  # risk clamped to 1


def test_integrator_horizon_mask_freezes_finished_agents():
    kw = _one_agent_case()
    o0 = torch.cat([kw["o0"], kw["o0"] + 1.0])
    for key in ("v0", "goal", "beta", "gamma", "lam_soft", "lam_hard", "d_hat", "dt"):
        kw[key] = torch.cat([kw[key], kw[key]])
    for key in ("C", "R", "mask", "alphas", "rollout_patch"):
        kw[key] = torch.cat([kw[key], kw[key]])
    kw["o0"] = o0
    kw["H"] = torch.tensor([2, 0])
    oT, vT, _, _, _, arc = mnav.integrate_surrogate_material(**kw)
    assert torch.equal(oT[1], o0[1])  # H=0 agent never moves
    assert torch.equal(vT[1], kw["v0"][1])
    assert arc[1].item() == 0.0
    assert arc[0].item() > 0.0


def test_integrator_empty_obstacles():
    kw = _one_agent_case()
    kw["C"] = torch.zeros(1, 0, 2)
    kw["R"] = torch.zeros(1, 0)
    kw["mask"] = torch.zeros(1, 0, dtype=torch.bool)
    kw["alphas"] = torch.zeros(1, 0)
    oT, vT, min_clear, *_ = mnav.integrate_surrogate_material(**kw)
    assert torch.isinf(min_clear).all()
    assert torch.isfinite(oT).all() and torch.isfinite(vT).all()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_model_six_outputs_caps_and_mu_lat_bias():
    m = mnav.CoefEnergyNetMaterial()
    assert m.mu_lat_head[-1].bias.item() == pytest.approx(-5.0)
    B, N = 2, 3
    outs = m(
        torch.randn(B, N, 6),
        torch.ones(B, N, dtype=torch.bool),
        torch.randn(B, 4),
        torch.rand(B, 2, 32, 32),
    )
    assert len(outs) == 6
    alphas, beta, gamma, lam_soft, lam_hard, mu_lat = outs
    assert alphas.shape == (B, N)
    for t in (alphas, beta, gamma):
        assert (t >= 0).all()
    assert ((lam_soft > 0) & (lam_soft < 5.0)).all()
    assert ((lam_hard > 0) & (lam_hard < 10.0)).all()
    assert ((mu_lat > 0) & (mu_lat < 5.0)).all()


def test_model_masked_obstacles_get_zero_alpha_and_n0_works():
    m = mnav.CoefEnergyNetMaterial()
    B, N = 1, 4
    mask = torch.tensor([[True, False, True, False]])
    alphas, *_ = m(torch.randn(B, N, 6), mask, torch.randn(B, 4), torch.rand(B, 2, 32, 32))
    assert (alphas[~mask] == 0).all()
    outs = m(
        torch.zeros(1, 0, 6),
        torch.zeros(1, 0, dtype=torch.bool),
        torch.randn(1, 4),
        torch.rand(1, 2, 32, 32),
    )
    assert outs[0].shape == (1, 0)


def test_model_checkpoint_key_contract():
    # The names the researcher's checkpoints address; renaming any of these
    # breaks load_geometry_weights / checkpoint loading.
    m = mnav.CoefEnergyNetMaterial()
    keys = set(m.state_dict().keys())
    for prefix in (
        "obs_enc.0",
        "goal_enc.0",
        "fuser.layers.0",
        "alpha_head.0",
        "beta_head.0",
        "gamma_head.0",
        "risk_enc.net.0",
        "lam_soft_head.0",
        "lam_hard_head.0",
        "mu_lat_head.0",
    ):
        assert any(k.startswith(prefix) for k in keys), prefix
