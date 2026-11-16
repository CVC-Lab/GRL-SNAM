"""The gym twin (:mod:`grl_snam.gym_twin`) against libcvc's gym (``pycvc.WorldGym``).

Gates (docs/gym-twin.md):

1. **Replay.** On ``dynamic4w``, away from a collision, one tick of the twin's plant from the
   gym's state, with the command the gym's plant took in each substep, is the gym's next state to
   1e-9 (relative): one model, run in double on both sides from the same float parameters.
2. **Residuals.** The twin's own tick (its controller, then its plant with the acceleration raw,
   no driver) against the gym's for the same commands stays within a few centimetres a tick on
   flat, open ground. What remains is the gym's driver.
3. **Gradients.** A rollout's loss has finite gradients in the coefficients, and they match finite
   differences, with the plant arm and without.
4. **Training** runs from ``experiments/gym_twin/``, not here.
5. **The plant arm.** With nav@1 driving, the twin's command from the gym's state and the drive the
   gym reports (its coefficients and carrot) is the command the gym's plant took, en route and
   arriving, parked at the goal included: with ``plant_arm=True`` on all but a few agent-ticks,
   and without it not on the approach.

Skipped without a pycvc whose WorldGym exports ``fields()`` and ``plants()`` (libcvc's gym twin
exports), or without Jolt (the gym needs a physics world); gate 5 without ``plants()``' ``drive``
and ``plan``.
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


@pytest.mark.parametrize("plant_arm", [False, True])
def test_gate3_gradients_are_finite_and_match_finite_differences(plant_arm):
    # The coefficients act where the drive is not at its limits: closing on a goal at speed
    # (beta's pull, gamma's damping), or near a wall (alpha's barrier). In open ground far from the
    # goal the drive accelerates at its limit and steers at the goal whatever they are, so their
    # gradient there is zero, truly. Here: at speed, a goal 15 m ahead; no wall in reach.
    # With the plant arm, through its governors and the steering's slew too. Where the gym's nav@1
    # stops at its goals (stop_at_goal), that stopping speed, not the goal's pull, brakes an
    # approach above ~5 m/s, so beta and gamma act only below it: there, a goal 6 m ahead of an
    # agent at ~3.5 m/s (the rollout may park it, which the gradient goes through as a brake).
    g = _gym()
    if plant_arm:
        _needs_plan(g)
    t = GymTwin(g, plant_arm=plant_arm)
    ticks, ahead, v_min = (30, 6.0, 2.0) if plant_arm else (80, 15.0, 5.0)
    drive_gym(g, t, constant_policy(DEFAULT), ticks)
    o = g.observe()
    for k in range(2):
        g.set_goal(int(o["id"][k]), float(o["x"][k]) + ahead, float(o["y"][k]) + 0.2 * ahead)
    t = GymTwin(g, plant_arm=plant_arm)
    s0 = t.state(g)
    assert bool((s0[:, 3] > v_min).all())

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


def _needs_plan(g):
    """Skip without ``plants()``' ``drive`` and ``plan`` (libcvc's entity_drive, entity_plan)."""
    p = g.plants()[0]
    if "plan" not in p or "drive" not in p:
        pytest.skip("this pycvc's plants() reports no drive or plan (libcvc's entity_plan)")


def _nav_gym(goals_x=(15.0, 30.0, 45.0), start_x=-70.0):
    """Three Humvees on dynamic4w that nav@1 drives, from x = start_x to goals 85-115 m east,
    toward the square's east edge (its wall for the stopping governor). They arrive one by one and
    park while the others drive on: at rest at the goal where nav@1 stops at its goals
    (``stop_at_goal``), and some metres past it, braking from speed, where it does not."""
    n = len(goals_x)
    agents = [
        {
            "plant": "dynamic4w",
            "controller": "nav@1",
            "x": start_x,
            "y": -30.0 + 30.0 * k,
            "yaw": 0.0,
        }
        for k in range(n)
    ]
    g = pycvc.WorldGym(_doc(), agents=agents, max_steps=4000)
    o = g.reset(0)
    for k in range(n):
        g.set_goal(int(o["id"][k]), goals_x[k], -30.0 + 30.0 * k + 5.0)
    return g


def plant_arm_departures(
    g, twins, ticks=1500, en_route_m=60.0, steer_tol=1e-4, accel_tol=1e-2, hold_ticks=20
):
    """Gate 5: each tick nav@1 drives the gym; from the gym's state before the tick, each twin's
    controller is given the drive the gym reports (its coefficients and carrot) and whether the
    tick parks the agent, and its command is compared with the one the gym's plant took
    (``plants()``' ``steer_command`` and ``accel_command``). An agent-tick departs when its
    steering differs by more than ``steer_tol`` rad or its acceleration by more than ``accel_tol``
    m/s^2. Per twin, the fraction that depart:

    * ``en_route``: more than ``en_route_m`` from the goal, not parked;
    * ``arriving``: within it, not parked;
    * ``parked``: ticks that park the agent, on the acceleration (the park brake). nav@1 holds a
      parked agent's carrot 2 cm ahead of it, where pure pursuit's angle is float rounding times
      2 L / 2 cm (~330): the gym's float steering and the twin's double one differ there by up to
      the actuator's slew, so their steering is ``parked_steer``, apart.

    Also the agent-ticks' counts and the largest departures outside parking. It ends
    ``hold_ticks`` after every agent is parked and stopped, or after ``ticks``: past the gym's
    own end of the episode, and on a libcvc whose nav@1 does not stop at its goals (which parks
    one as it passes its goal at speed, so the gym's arrival is never met)."""
    nan = np.full(twins[0].n, np.nan)
    t0 = twins[0]
    rows = {k: {"steer": [], "accel": []} for k in range(len(twins))}
    where, park = [], []
    held = 0
    with torch.no_grad():
        for _ in range(ticks):
            parked = t0.parked(g)
            s0 = t0.state(g)
            held = held + 1 if bool(parked.all()) and float(s0[:, 3].abs().max()) < 0.01 else 0
            if held > hold_ticks:
                break
            dist = (s0[:, :2] - t0.goal).norm(dim=-1)
            out = g.step(nan, nan)
            pl = g.plants()
            assert all(p["drive"]["known"] for p in pl)
            coefs = torch.tensor(
                [[p["drive"][c] for c in ("alpha", "beta", "gamma")] for p in pl],
                dtype=torch.float64,
            )
            carrot = torch.tensor([[p["drive"]["goal_x"], p["drive"]["goal_y"]] for p in pl])
            gs = torch.tensor([p["steer_command"] for p in pl], dtype=torch.float64)
            ga = torch.tensor([p["accel_command"] for p in pl], dtype=torch.float64)
            for k, t in enumerate(twins):
                d, a = t.control(s0, coefs, t.to_norm(carrot.double()), parked=parked)
                rows[k]["steer"].append((d - gs).abs().numpy())
                rows[k]["accel"].append((a - ga).abs().numpy())
            where.append((dist > en_route_m).numpy())
            park.append(parked.numpy())
            if out[3]:  # truncated; a terminated episode steps on while its agents come to rest
                break
    far, pk = np.stack(where, 0), np.stack(park, 0)
    en, arr = far & ~pk, ~far & ~pk
    res = {
        "ticks": len(where),
        "en_route_n": int(en.sum()),
        "arriving_n": int(arr.sum()),
        "parked_n": int(pk.sum()),
    }
    for k in range(len(twins)):
        ds, da = np.stack(rows[k]["steer"], 0), np.stack(rows[k]["accel"], 0)
        dep = (ds > steer_tol) | (da > accel_tol)
        res[k] = {
            "en_route": float(dep[en].mean()),
            "arriving": float(dep[arr].mean()),
            "parked": float((da > accel_tol)[pk].mean()),
            "parked_steer": float((ds > steer_tol)[pk].mean()),
            "steer_max": float(ds[~pk].max()),
            "accel_max": float(da[~pk].max()),
            "parked_accel_max": float(da[pk].max()),
        }
    return res


def test_gate5_the_plant_arm_is_the_command_the_plant_took():
    g = _nav_gym()
    _needs_plan(g)
    before, after = GymTwin(g), GymTwin(g, plant_arm=True)
    assert after.arm.a_brake > after.veh["a_max"]  # a Humvee brakes harder than it accelerates
    res = plant_arm_departures(g, [before, after])
    assert min(res["en_route_n"], res["arriving_n"], res["parked_n"]) > 100, res
    # Without the plant arm the approach departs: braking past a_max, the stopping governor's
    # latency and footprint, the steering's slew, and the park brake.
    assert res[0]["arriving"] > 0.05 and res[0]["parked"] > 0.5, res
    # With it, the command is the plant's on all but a few agent-ticks.
    assert res[1]["en_route"] <= 0.01, res
    assert res[1]["arriving"] <= 0.02, res
    assert res[1]["parked"] <= 0.01, res


def test_the_plant_arm_parks_a_rollout_at_its_goal():
    # A twin rollout that arrives stops within the reach and holds, as nav@1 parks an agent: the
    # park brake's speed is zero after it, the control law alone may not stop there.
    g = _nav_gym(goals_x=(-35.0, -35.0, -35.0))
    _needs_plan(g)
    t = GymTwin(g, plant_arm=True)
    states, cmds = t.rollout(constant_policy(DEFAULT), t.state(g), 200)
    d = (states[:, :, :2] - t.goal[None]).norm(dim=-1)
    arrived = (d < t.reach).any(0)
    assert bool(arrived.all()), d[-1]
    assert float(states[-1, :, 3].abs().max()) < 0.05, states[-1, :, 3]
    assert float(cmds[:, :, 1].min()) >= -t.arm.a_brake / t.S - 1e-9


def test_other_models_are_refused():
    g = _gym(plant="kinematic")
    with pytest.raises(ValueError, match="four-wheel"):
        GymTwin(g)
