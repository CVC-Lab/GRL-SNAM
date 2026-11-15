"""The gym twin (:mod:`grl_snam.gym_twin`) against libcvc's gym (``pycvc.WorldGym``).

Gates (docs/gym-twin.md):

1. **Replay.** On ``dynamic4w``, away from a collision, one tick of the twin's plant from the
   gym's state, with the command the gym's plant took in each substep, is the gym's next state to
   1e-9 (relative): one model, run in double on both sides from the same float parameters.
2. **Residuals.** The twin's own tick (its controller, then its plant with the acceleration raw,
   no driver) against the gym's for the same commands stays within a few centimetres a tick on
   flat, open ground. What remains is the gym's driver.
3. **Gradients.** A rollout's loss has finite gradients in the coefficients, and they match finite
   differences.

Skipped without a pycvc whose WorldGym exports ``fields()`` and ``plants()`` (libcvc's gym twin
exports), or without Jolt (the gym needs a physics world).
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pycvc = pytest.importorskip("pycvc")

if not hasattr(pycvc, "WorldGym") or not hasattr(pycvc.WorldGym, "plants"):
    pytest.skip("this pycvc's WorldGym exports no fields() / plants()", allow_module_level=True)
if not pycvc.AriApp.have_yaml():
    pytest.skip("this libcvc reads no YAML documents", allow_module_level=True)
if "jolt" not in pycvc.physics_backends() or not pycvc.physics_backend_info("jolt")["available"]:
    pytest.skip("no Jolt in this build", allow_module_level=True)

from grl_snam.gym_twin import (  # noqa: E402
    GymTwin,
    arrival_loss,
    clearance_loss,
    constant_policy,
    drive_gym,
    progress_loss,
    tick_residuals,
)

DOC = """requires: [scene.physics]
world:
  vehicles:
    archetypes:
      humvee: { vehicle: humvee, controllers: [nav@1, none] }
scene:
  physics: {}
  world: { source: square, cell: 2 }
  nodes:
    - node: square
      world: square
      source: { box: { size: [160, 160, 1] } }
      transform: { position: [0, 0, -0.5] }
      body: static
"""

DEFAULT = (1.0, 3.0, 4.0)  # CoefMLP's bias: the known-good regime


def _doc() -> str:
    d = tempfile.mkdtemp(prefix="gym_twin_")
    p = os.path.join(d, "gym.ari")
    with open(p, "w", encoding="utf-8") as f:
        f.write(DOC)
    return p


def _gym(plant="dynamic4w", n=2):
    agents = [
        {"plant": plant, "controller": "none", "x": -40.0, "y": -20.0 + 40.0 * k, "yaw": 0.0}
        for k in range(n)
    ]
    g = pycvc.WorldGym(_doc(), agents=agents, max_steps=2000)
    o = g.reset(0)
    for k in range(n):
        g.set_goal(int(o["id"][k]), 40.0, -20.0 + 40.0 * k + 10.0)
    return g


def test_the_twin_reads_the_gym():
    g = _gym()
    t = GymTwin(g)
    assert t.n == 2
    assert t.exact == [True, True]
    assert t.substeps >= 1
    assert t.S == 0.05
    s = t.state(g)
    assert s.shape == (2, 14) and s.dtype == torch.float64
    o = g.observe()
    np.testing.assert_allclose(s[:, 2].numpy(), o["yaw"], atol=1e-6)
    # The ground and the field at the agents: flat, open, the grip the plant drives on.
    mu, h = t.ground(s[:, 0], s[:, 1])
    np.testing.assert_allclose(mu.numpy(), 0.9, atol=1e-6)
    np.testing.assert_allclose(h.numpy(), 0.0, atol=1e-6)
    assert bool((t.clearance(s[:, :2]) > 10.0).all())


def test_gate1_replay_is_the_gyms_tick():
    g = _gym()
    t = GymTwin(g)
    for i in range(30):
        s0 = t.state(g)
        steer = np.full(2, 0.05 * math.sin(0.2 * i))
        accel = np.full(2, 2.0 if i < 20 else -1.0)
        g.step(steer, accel)
        s1 = t.state(g)
        r = t.replay(s0, g)
        hit = [int(p["collisions_tick"]) for p in g.plants()]
        assert hit == [0, 0]
        np.testing.assert_allclose(r.numpy(), s1.numpy(), rtol=1e-9, atol=1e-9, err_msg=f"tick {i}")


def test_gate2_the_twins_own_tick_is_close_off_contact():
    g = _gym()
    t = GymTwin(g)
    res = tick_residuals(g, t, constant_policy(DEFAULT), 60)
    assert not res["collided"].any()
    # What remains is the gym's driver, which the twin does not have.
    assert float(np.median(res["position_m"])) < 0.05, res["position_m"].max()
    assert float(res["heading_rad"].max()) < 0.01


def test_gate3_gradients_are_finite_and_match_finite_differences():
    # The coefficients act where the drive is not at its limits: closing on a goal at speed
    # (beta's pull, gamma's damping), or near a wall (alpha's barrier). In open ground far from the
    # goal the drive accelerates at its limit and steers at the goal whatever they are, so their
    # gradient there is zero, truly. Here: at speed, a goal 15 m ahead; no wall in reach.
    g = _gym()
    t = GymTwin(g)
    drive_gym(g, t, constant_policy(DEFAULT), 80)
    o = g.observe()
    for k in range(2):
        g.set_goal(int(o["id"][k]), float(o["x"][k]) + 15.0, float(o["y"][k]) + 3.0)
    t = GymTwin(g)
    s0 = t.state(g)
    assert bool((s0[:, 3] > 5.0).all())

    def loss(c):
        states, _ = t.rollout(lambda f: c.expand(f.shape[0], 3), s0, 12)
        return progress_loss(t, states) + clearance_loss(t, states)

    c = torch.tensor(DEFAULT, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(loss(c), c)
    assert torch.isfinite(grad).all()
    assert grad[1] != 0 and grad[2] != 0, grad  # beta, gamma
    eps = 1e-5
    for k in range(3):
        e = torch.zeros(3, dtype=torch.float64)
        e[k] = eps
        fd = (loss(c.detach() + e) - loss(c.detach() - e)) / (2 * eps)
        assert abs(fd.item() - grad[k].item()) <= 1e-6 * max(1e-3, abs(fd.item())), (k, fd, grad)


def test_the_gym_judges_a_policy():
    g = _gym()
    t = GymTwin(g)
    d0 = np.hypot(
        g.observe()["goal_x"] - g.observe()["x"], g.observe()["goal_y"] - g.observe()["y"]
    )
    out, log = drive_gym(g, t, constant_policy(DEFAULT), 200)
    o = out[0]
    d1 = np.hypot(o["goal_x"] - o["x"], o["goal_y"] - o["y"])
    assert len(log) > 0
    assert bool((d1 < d0 - 20.0).all()), (d0, d1)


def test_arrival_pays_for_stopping_inside_the_reach():
    # Two hand-made rollouts to the same goal: one ends inside the reach and stopped, the other
    # drives through it at speed. Only the first has arrived.
    g = _gym()
    t = GymTwin(g)
    goal = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    T = 10
    stop = torch.zeros(T + 1, 1, 14, dtype=torch.float64)
    through = torch.zeros(T + 1, 1, 14, dtype=torch.float64)
    for k in range(T + 1):
        stop[k, 0, 0] = -20.0 + 20.0 * min(1.0, k / 8)  # reaches the goal and stays, vx -> 0
        stop[k, 0, 3] = max(0.0, 10.0 - 1.25 * k)
        through[k, 0, 0] = -20.0 + 3.0 * k  # passes the goal at 10 m/s
        through[k, 0, 3] = 10.0
    assert arrival_loss(t, stop, goal) < arrival_loss(t, through, goal)
    x = through.clone().requires_grad_(True)
    (gx,) = torch.autograd.grad(arrival_loss(t, x, goal), x)
    assert torch.isfinite(gx).all()


def test_other_models_are_refused():
    g = _gym(plant="kinematic")
    with pytest.raises(ValueError, match="four-wheel"):
        GymTwin(g)
