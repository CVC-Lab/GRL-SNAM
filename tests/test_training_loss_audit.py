"""Regression tests for the 2026-10 training-loss audit of the CoefEnergyNet trainer.

Each test pins one defect that was found by inspection and confirmed by a failing test:

* ``multi_start_penalty`` drew no randomness, so its ``ms_count`` rollouts were identical
  copies of one rollout (and a per-sample ``[B]`` robot radius crashed it when ``B != N``).
* ``Trainer.step_batch`` computed the penetration penalty ``pen`` but never added it to the loss.
* ``grl_snam.integrate_surrogate`` exported the legacy EXPLICIT-Euler integrator while the trainer
  integrates with the semi-implicit ``integrate_surrogate_v2`` (which itself crashed on its own
  default scalar ``robot_radius``).
"""

from __future__ import annotations

import pytest
import torch

import grl_snam
import grl_snam.surrogate_robust as sr
import grl_snam.train_coef_energy as tce


def _scene(B: int = 3, N: int = 2):
    o0 = torch.tensor([[0.0, 0.0], [1.0, 0.5], [-1.0, 2.0]])[:B]
    v0 = torch.tensor([[0.5, 0.1], [0.2, -0.3], [0.0, 0.4]])[:B]
    goal = torch.tensor([[5.0, 0.0], [5.0, 1.0], [3.0, 3.0]])[:B]
    C = torch.tensor(
        [[[1.0, 0.2], [3.0, -1.0]], [[2.0, 0.5], [0.0, 3.0]], [[0.0, 2.0], [-3.0, 0.0]]]
    )[:B, :N]
    R = torch.full((B, N), 0.4)
    mask = torch.ones(B, N, dtype=torch.bool)
    alphas = torch.ones(B, N, requires_grad=True)
    beta = torch.ones(B, requires_grad=True)
    gamma = torch.full((B,), 0.5, requires_grad=True)
    d_hat = torch.full((B,), 0.5)
    dt = torch.full((B,), 0.05)
    H = torch.full((B,), 4, dtype=torch.long)
    return o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H


def _spy_starts(monkeypatch) -> list[torch.Tensor]:
    """Record the start state of every auxiliary rollout multi_start_penalty runs."""
    starts: list[torch.Tensor] = []
    real = sr.integrate_surrogate_v2

    def spy(o0, *args, **kwargs):
        starts.append(o0.detach().clone())
        return real(o0, *args, **kwargs)

    monkeypatch.setattr(sr, "integrate_surrogate_v2", spy)
    return starts


# --------------------------------------------------------------------------- multi-start


def test_multi_start_rollouts_are_distinct(monkeypatch):
    """The ms_count auxiliary starts must be distinct samples, not one start repeated."""
    starts = _spy_starts(monkeypatch)
    torch.manual_seed(0)
    sr.multi_start_penalty(*_scene(), ms_count=20)
    assert len(starts) == 20
    distinct = {tuple(s.flatten().tolist()) for s in starts}
    assert len(distinct) == 20, f"only {len(distinct)} distinct starts out of 20"


def test_multi_start_starts_stay_between_the_agent_and_the_obstacle(monkeypatch):
    """Every sampled start sits on the agent->nearest-obstacle segment, inside frac_range."""
    o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H = _scene()
    starts = _spy_starts(monkeypatch)
    torch.manual_seed(1)
    sr.multi_start_penalty(
        o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H,
        ms_count=50, frac_range=(0.8, 0.98),
    )  # fmt: skip
    dmin0, nmin0, _ = sr._nearest_obstacle(o0, C, R, mask)
    for s in starts:
        frac = ((o0 - s) * nmin0).sum(-1) / dmin0  # how far toward the obstacle we moved
        assert torch.all(frac >= 0.8 - 1e-5) and torch.all(frac <= 0.98 + 1e-5)


def test_multi_start_degenerate_range_reproduces_the_legacy_single_start():
    """frac_range=(0.9, 0.9) is the pre-fix behaviour (and libcvc's native port) exactly,
    and draws no random numbers."""
    o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H = _scene()
    state = torch.random.get_rng_state()
    got = sr.multi_start_penalty(
        o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H,
        ms_count=1, ms_h=3, ms_dt_mult=4.0, frac_range=(0.9, 0.9),
    )  # fmt: skip
    assert torch.equal(torch.random.get_rng_state(), state), "degenerate range consumed RNG"

    # The legacy draw, written out: 90% of the clearance toward the nearest obstacle.
    dmin0, nmin0, _ = sr._nearest_obstacle(o0, C, R, mask)
    o_ms = o0 - (0.9 * dmin0).unsqueeze(-1) * nmin0
    _, _, clr = sr.integrate_surrogate_v2(
        o_ms, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, 4.0 * dt,
        torch.full_like(H, 3), robot_radius=torch.zeros(3),
    )  # fmt: skip
    want = torch.nn.functional.softplus(-clr / 0.05).mean()
    assert torch.allclose(got, want, rtol=0, atol=1e-7)


def test_multi_start_is_reproducible_with_a_generator():
    scene = _scene()
    a = sr.multi_start_penalty(*scene, ms_count=8, generator=torch.Generator().manual_seed(7))
    b = sr.multi_start_penalty(*scene, ms_count=8, generator=torch.Generator().manual_seed(7))
    c = sr.multi_start_penalty(*scene, ms_count=8, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_multi_start_penalty_carries_gradient_to_the_coefficients():
    o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H = _scene()
    L = sr.multi_start_penalty(
        o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H, ms_count=4
    )
    L.backward()
    for t in (alphas, beta, gamma):
        assert t.grad is not None and torch.isfinite(t.grad).all()
    assert alphas.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [(-0.1, 0.5), (0.5, 1.0), (0.9, 0.8)])
def test_multi_start_rejects_a_bad_frac_range(bad):
    with pytest.raises(ValueError):
        sr.multi_start_penalty(*_scene(), ms_count=2, frac_range=bad)


def test_multi_start_accepts_a_per_sample_robot_radius():
    """A [B] robot radius with B != N used to broadcast against the obstacle axis and crash."""
    o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H = _scene(B=3, N=2)
    L = sr.multi_start_penalty(
        o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H,
        robot_radius=torch.full((3,), 0.3), ms_count=2,
    )  # fmt: skip
    assert torch.isfinite(L)


# --------------------------------------------------------------------------- integrators


def test_integrate_surrogate_v2_works_with_its_default_scalar_radius():
    o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H = _scene()
    o_def, v_def, c_def = sr.integrate_surrogate_v2(
        o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H
    )
    o_vec, v_vec, c_vec = sr.integrate_surrogate_v2(
        o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, H, robot_radius=torch.zeros(3)
    )
    assert torch.equal(o_def, o_vec) and torch.equal(v_def, v_vec) and torch.equal(c_def, c_vec)


def _one_step(integrator):
    o0, v0, goal, C, R, mask, alphas, beta, gamma, d_hat, dt, _ = _scene()
    # No obstacles in range: the only forces are the goal spring and damping.
    far = torch.full_like(C, 1e3)
    H1 = torch.ones(3, dtype=torch.long)
    o1, v1, _ = integrator(o0, v0, goal, far, R, mask, alphas, beta, gamma, d_hat, dt, H1)
    a0 = -beta.unsqueeze(-1) * (o0 - goal) - gamma.unsqueeze(-1) * v0
    return o0, v0, dt.unsqueeze(-1), a0, o1, v1


def test_top_level_integrate_surrogate_is_the_trainers_semi_implicit_integrator():
    assert grl_snam.integrate_surrogate is sr.integrate_surrogate_v2
    o0, v0, dt, a0, o1, v1 = _one_step(grl_snam.integrate_surrogate)
    assert torch.allclose(v1, v0 + dt * a0, atol=1e-6)
    assert torch.allclose(o1, o0 + dt * (v0 + dt * a0), atol=1e-6)  # uses v_{n+1}


def test_legacy_explicit_integrator_is_still_available_under_an_explicit_name():
    o0, v0, dt, a0, o1, v1 = _one_step(tce.integrate_surrogate_explicit)
    assert torch.allclose(o1, o0 + dt * v0, atol=1e-6)  # uses v_n
    assert torch.allclose(v1, v0 + dt * a0, atol=1e-6)


# --------------------------------------------------------------------------- trainer loss


def _penetrating_batch():
    """One agent driving fast into a disc obstacle, so the ROLLOUT (not the data start) penetrates
    and its minimum clearance depends on the predicted coefficients."""
    return {
        "o0": torch.tensor([[-0.7, 0.0]]),
        "v0": torch.tensor([[10.0, 0.0]]),
        "goal": torch.tensor([[3.0, 0.0]]),
        "C": torch.tensor([[[0.0, 0.0]]]),
        "R": torch.tensor([[0.5]]),
        "W": torch.ones(1, 1),
        "obs_mask": torch.ones(1, 1, dtype=torch.bool),
        "obs_feats": torch.tensor([[[0.0, 0.0, 0.5, 1.0, 3.0, 0.0]]]),
        "goal_feats": torch.tensor([[3.7, 0.0, 3.7, 1.0]]),
        "d_hat": torch.tensor([0.5]),
        "dt_base": torch.tensor([0.03]),
        "dt_prime": torch.tensor([0.05]),
        "H": torch.tensor([3]),
        "o_tgt": torch.tensor([[-0.4, 0.0]]),
        "v_tgt": torch.tensor([[1.0, 0.0]]),
        "gamma_o": torch.tensor([4.0]),
    }


def _step(**cfg_kw):
    torch.manual_seed(0)
    cfg = tce.TrainCfg(device="cpu", ms_count=2, **cfg_kw)
    logs = tce.Trainer(tce.CoefEnergyNet(), cfg).step_batch(_penetrating_batch())
    without_pen = (
        cfg.w_traj * logs["traj"]
        + cfg.w_vel * logs["vel"]
        + cfg.w_friction * logs["friction"]
        + cfg.w_multi * logs["multi_val"]
    )
    return cfg, logs, without_pen


def test_penetration_penalty_is_added_to_the_loss():
    cfg, logs, without_pen = _step()
    assert logs["pen"] > 1.0, "fixture must actually penetrate"
    assert cfg.w_prox > 0.0
    assert logs["loss"] == pytest.approx(without_pen + cfg.w_prox * logs["pen"], rel=1e-5)


def test_penetration_penalty_weight_zero_restores_the_old_objective():
    _cfg, logs, without_pen = _step(w_prox=0.0)
    assert logs["loss"] == pytest.approx(without_pen, rel=1e-6)


def test_penetration_penalty_reaches_the_gradient():
    """pen must reach the parameters, not just the logged scalar."""

    def grads_after(w_prox):
        torch.manual_seed(0)
        model = tce.CoefEnergyNet()
        tce.Trainer(model, tce.TrainCfg(device="cpu", ms_count=2, w_prox=w_prox)).step_batch(
            _penetrating_batch()
        )
        # step_batch zeroes grads BEFORE backward, so .grad still holds this step's gradient.
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    assert (grads_after(0.0) - grads_after(0.1)).abs().max() > 1e-5
