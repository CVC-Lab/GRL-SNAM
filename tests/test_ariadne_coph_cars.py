"""grl-snam ariadne coph-cars (T4, demo 2b): CoPH's learner driving the cars template's cars.

Most of it runs on a stand-in CarsState and a kinematic bicycle, so it needs torch and not the
template. The template runs (an episode on pycvc_gl.cars, the arms and their report) skip without
pycvc_gl.cars (pycvc-gl 3.7.12+).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam.ariadne import PORTS  # noqa: E402
from grl_snam.ariadne import coph_cars as cc  # noqa: E402
from grl_snam.coph.allocation import allocate  # noqa: E402
from grl_snam.coph.model import flow, potential, tensor  # noqa: E402

needs_template = pytest.mark.skipif(cc._template() is None, reason="needs pycvc_gl.cars")


def test_the_port_is_registered():
    assert PORTS["coph-cars"][0] == "grl_snam.ariadne.coph_cars"


# ── scale, goal, patches ──────────────────────────────────────────────────────────────────────


def test_froude_scales_lengths_times_and_speeds_not_accelerations():
    f = cc.Froude(25.0, (100.0, -50.0))
    p = np.array([125.0, 0.0])
    assert np.allclose(f.q(p), [1.0, 2.0]) and np.allclose(f.p(f.q(p)), p)
    assert np.allclose(f.v([5.0, 0.0]), [1.0, 0.0])  # 5 m/s at s = 25 is CoPH's 1
    assert f.length(31.25) == pytest.approx(1.25)  # the Humvee's sensor radius, §3.3.1
    # D6: CoPH's 0.12 s is 0.6 s of world time at s = 25, ten 0.06 s ticks; two at s = 1.
    assert f.ticks_per_step(0.06) == 10 and cc.Froude(1.0).ticks_per_step(0.06) == 2


def test_the_local_goal_is_30_m_along_the_line():
    assert np.allclose(cc.local_goal([0, 0], [100, 0]), [30, 0])
    assert np.allclose(cc.local_goal([0, 0], [3, 4]), [3, 4])


def test_patches_are_paired_by_seed_and_lie_across_each_line():
    starts, goals = np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([[10.0, 0.0], [5.0, -5.0]])
    a = cc.risk_patches(starts, goals, seed=7)
    assert a == cc.risk_patches(starts, goals, seed=7) != cc.risk_patches(starts, goals, seed=8)
    for (x, y, *_), s, g in zip(a, starts, goals):
        d = g - s
        t = np.dot([x, y] - s, d) / np.dot(d, d)
        assert 0.35 <= t <= 0.65


def test_the_material_raster_samples_its_risk_plane(monkeypatch):
    patch = (2.0, 1.0, 1.0, 1.0, 0.9)
    m = cc.MaterialRaster([patch], (-5.0, -5.0, 5.0, 5.0), 0.25)
    peak = float(m.sample([[2.0, 1.0]])[0])
    assert 0.5 < peak <= 1.0 and float(m.sample([[-4.5, -4.5]])[0]) == pytest.approx(0.02, abs=1e-3)
    assert float(m.sample([[50.0, 50.0]])[0]) == m.background  # off the grid
    monkeypatch.setattr(cc, "_pycvc", lambda: None)  # no libcvc: the raw risk, unblurred
    raw = cc.MaterialRaster([patch], (-5.0, -5.0, 5.0, 5.0), 0.25)
    assert raw.source == "raw" and float(raw.sample([[2.0, 1.0]])[0]) == pytest.approx(
        0.92, abs=1e-6
    )


# ── the learner, the ledger, V ────────────────────────────────────────────────────────────────


def test_stages_follow_the_car_along_either_axis_either_way():
    a = cc.make_learner("coph", [0.0, 0.0, 0.0, 0.0], (0.0, -20.0), 0)
    assert a.stage_mode == "spatial"
    assert not a.advance(np.array([0.5, -1.0, 0.0, -1.0]))
    assert a.advance(np.array([0.0, -1.5, 0.0, -1.0]))  # left the inner box downwards
    assert a.stages[-1].center == pytest.approx((0.0, -2.8))  # 4 m x (1 - 0.3)
    assert a.advance(np.array([1.2, -2.8, 1.0, 0.0]))
    assert a.stages[-1].center == pytest.approx((2.1, -2.8))  # 3 m x (1 - 0.3)
    assert [j["stage"] for j in a.context_jumps] == [1, 2]


def test_the_ledger_closes_on_cophs_own_plant():
    led = cc.Ledger()
    x = np.array([0.0, 0.0, 0.3, -0.1])
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = rng.uniform(-2.0, 2.0, 2)
        y = flow(tensor(x), tensor(u), 0.7, cc.COPH_DT).numpy()
        led.add_step(x, y, u, np.zeros(2), 0.7)
        x = y
    assert abs(led.residual) < 1e-12 * max(1.0, abs(led.work))
    assert led.max_step_residual < 1e-13 and led.unallocated_work == 0.0


def test_the_ledger_books_what_a_car_did_not_take():
    led = cc.Ledger()
    x0, x1 = np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.12, 0.0, 1.0, 0.0])
    led.add_step(x0, x1, np.zeros(2), np.array([2.0, 0.5]), 0.6)
    assert led.unallocated_work == pytest.approx(2.0 * 0.12)


def test_potential_grid_is_cophs_potential():
    a = cc.make_learner("coph", [0.0, 0.0, 0.0, 0.0], (5.0, 1.0), 0)
    a.geometry_store = ((1.0, 0.5, 0.2), (2.0, -0.6, 0.3))
    s = a.stages[-1]
    s.weights = tensor(np.linspace(-0.4, 0.3, len(s.centers)))
    pts = np.array([[0.2, 0.1], [1.0, 0.0], [-0.7, 1.4], [1.4, -0.9]])
    V = cc.potential_grid(a, pts)
    for q, v in zip(pts, V):
        want = potential(tensor(q), tensor(a.control_target(s)), s.weights, s.centers, a.cfg.rbf_width,
                         a.policy, a.control_obstacles(s), a.cfg)  # fmt: skip
        assert v == pytest.approx(float(want), rel=1e-12, abs=1e-12)


def test_raw_readings_carry_values_only():
    a = cc.make_learner("raw", [0.0, 0.0, 0.0, 0.0], (5.0, 0.0), 0)
    assert a.communication.anchors == cc.RAW_ANCHORS
    m = cc.make_learner("coph", [0.0, 0.0, 0.0, 0.0], (5.0, 0.0), 1)
    assert m.communication.anchors == 4


# ── the controller on a stand-in for the template ─────────────────────────────────────────────


@dataclass
class State:
    """The CarsState fields the controller reads."""

    t: float
    dt: float
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    goal: np.ndarray
    wheelbase: np.ndarray
    max_steer: np.ndarray
    max_accel: np.ndarray
    reach: float = 0.5
    ranges: np.ndarray = None
    ray_angles: np.ndarray = field(
        default_factory=lambda: np.linspace(-np.pi, np.pi, 16, endpoint=False)
    )
    range_max: float = 20.0
    collided: np.ndarray = None

    @property
    def n(self):
        return len(self.x)

    @property
    def pos(self):
        return np.stack([self.x, self.y], 1)

    @property
    def reached(self):
        return np.hypot(*(self.pos - self.goal).T) <= self.reach


def _cars(goals):
    n = len(goals)
    return State(0.0, 0.06, np.zeros(n), np.linspace(-1.0, 1.0, n), np.zeros(n), np.zeros(n),
                 np.asarray(goals, float), np.full(n, 0.324), np.full(n, 0.5), np.full(n, 3.0),
                 ranges=np.full((n, 16), 20.0), collided=np.zeros(n, bool))  # fmt: skip


def _step(s, steer, accel):
    """nav's kinematic bicycle, one tick."""
    out = State(**{k: getattr(s, k) for k in s.__dataclass_fields__})
    out.speed = s.speed + accel * s.dt
    out.heading = s.heading + s.speed / s.wheelbase * np.tan(steer) * s.dt
    out.x = s.x + s.speed * np.cos(s.heading) * s.dt
    out.y = s.y + s.speed * np.sin(s.heading) * s.dt
    out.t = s.t + s.dt
    return out


def test_the_controller_holds_u_and_allocates_it_every_tick():
    ctl = cc.CophCars(arm="off", seed=3)
    s = _cars([[4.0, -1.0], [4.0, 1.0]])
    held = []
    for k in range(6):
        steer, accel, info = ctl(s, dict(cc.PARAMS))
        held.append(ctl.run.u.copy())
        for i in range(s.n):  # every tick: the held u through allocation.allocate
            al = allocate(ctl.run.u[i], s.heading[i], s.speed[i], s.wheelbase[i], s.max_steer[i],
                          s.max_accel[i])  # fmt: skip
            assert (steer[i], accel[i]) == (al.steer, al.accel)
        s = _step(s, steer, accel)
    assert ctl.run.k == 2 and ctl.run.steps == 3
    assert np.array_equal(held[0], held[1]) and np.array_equal(held[2], held[3])  # D6
    assert list(info["coeffs"])[:4] == ["pi_risk", "pi_barrier", "damping", "version"]
    assert set(info["coeffs"]) == set(cc.COEFFS)
    assert all(v.shape == (2,) for v in info["coeffs"].values())


def test_the_learner_observes_the_force_the_car_took():
    ctl = cc.CophCars(arm="off", seed=3)
    s = _cars([[0.0, 4.0]])  # the goal is to the left: at rest, most of u is lateral
    for _ in range(4):
        steer, accel, _ = ctl(s, dict(cc.PARAMS))
        s = _step(s, steer, accel)
    tr = ctl.run.learners[0].stages[-1].transitions[0]
    u = np.asarray(tr.u)
    # At rest the car takes no lateral force (v < v_floor), so u_used lies along the heading.
    assert (
        abs(u[1]) < 0.05 * np.hypot(*ctl.run.u[0]) and ctl.run.ledgers[0].allocation_residual > 0.5
    )


def test_a_car_stops_learning_once_it_arrives():
    ctl = cc.CophCars(arm="off", seed=3)
    s = _cars([[0.0, -1.0]])  # already within reach (the one car starts at (0, -1))
    steer, accel, _ = ctl(s, dict(cc.PARAMS))
    assert ctl.run.done[0] and ctl.run.records[0].arrived_at == 0.0 and accel[0] == 0.0


def test_cost_draws_v_or_r_for_the_chosen_car():
    ctl = cc.CophCars(arm="off", seed=3)
    s = _cars([[4.0, -1.0], [4.0, 1.0]])
    ctl(s, dict(cc.PARAMS))
    X, Y = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 5))
    V = ctl.cost(s, {"view": 0.0}, 1, X, Y)
    r = ctl.cost(s, {"view": 1.0}, 1, X, Y)
    assert V.shape == r.shape == X.shape and np.all(np.isfinite(V)) and np.all(r > 0)
    assert not np.allclose(V, r)


def test_packets_flow_between_cars_on_the_coph_arm():
    ctl = cc.CophCars(arm="coph", seed=3)
    s = _cars([[4.0, -0.2], [4.0, 0.2]])
    for _ in range(40):
        steer, accel, _ = ctl(s, dict(cc.PARAMS))
        s = _step(s, steer, accel)
    recs = ctl.run.records
    assert all(r.bytes_sent > 0 for r in recs)
    assert sum(sum(r.reasons.values()) for r in recs) > 0


def rollout(ctl, s, steps):
    """Stands in for pycvc_gl.cars.rollout: the controller on imagined states, time running on."""
    q, out = s, []
    for _ in range(steps):
        steer, accel, _ = ctl(q, dict(cc.PARAMS))
        out.append((steer.copy(), accel.copy()))
        q = _step(q, steer, accel)
    return out


def test_the_templates_predicted_paths_teach_the_learners_nothing():
    ctl = cc.CophCars(arm="off", seed=3)
    s = _cars([[4.0, -1.0], [4.0, 1.0]])
    for _ in range(3):
        steer, accel, _ = ctl(s, dict(cc.PARAMS))
        s = _step(s, steer, accel)
    run, steps, ticks = ctl.run, ctl.run.steps, ctl.run.ticks
    versions = [len(a.stages[-1].transitions) for a in run.learners]
    plan = rollout(ctl, s, 20)  # 1.2 s ahead: past the next CoPH step
    assert ctl.run is run and (run.steps, run.ticks) == (steps, ticks)
    assert [len(a.stages[-1].transitions) for a in run.learners] == versions
    al = allocate(
        run.u[0], s.heading[0], s.speed[0], s.wheelbase[0], s.max_steer[0], s.max_accel[0]
    )
    assert (plan[0][0][0], plan[0][1][0]) == (al.steer, al.accel)  # the held u, allocated
    steer, accel, info = ctl(s, dict(cc.PARAMS))  # the real tick goes on where it was
    assert ctl.run is run and run.ticks == ticks + 1 and "coeffs" in info


def test_a_new_run_rebuilds_the_learners():
    ctl = cc.CophCars(seed=3)  # the arm comes from ctl.arm
    s = _cars([[4.0, 0.0]])
    ctl(s, dict(cc.PARAMS, arm=0.0))
    first = ctl.run
    assert first.arm == "off"
    ctl(s, dict(cc.PARAMS, arm=1.0))  # a different arm is a different run
    assert ctl.run is not first and ctl.run.arm == "coph"


# ── the template ──────────────────────────────────────────────────────────────────────────────


@needs_template
def test_an_episode_on_the_template_logs_the_ledger(tmp_path):
    log = tmp_path / "ep.npz"
    ep = cc.episode("coph", 1, cars=2, frames=12, log=log)
    assert ep["ticks_per_step"] == 2 and len(ep["cars"]) == 2 and ep["material"]
    z = np.load(log)
    assert z["coef_pi_risk"].shape == (12, 2) and "coef_unallocated_work" in z
    meta = json.loads(str(z["meta"]))
    assert meta["arm"] == "coph" and meta["controller"] == "coph-cars"


@needs_template
def test_the_templates_own_rollout_leaves_the_learners_alone():
    C = cc._template()
    world = C.flat_world(cc.roboracer_length())
    fleet = C.Cars(world, 2, "kinematic", seed=1)
    ctl = cc.CophCars(world, arm="off", seed=1)
    s = fleet.state()
    for _ in range(3):
        steer, accel, _ = ctl(s, dict(cc.PARAMS))
        s = fleet.step(steer, accel)
    run, ticks = ctl.run, ctl.run.ticks
    paths = C.rollout(ctl, s, dict(cc.PARAMS), world, steps=20)
    assert paths.shape == (2, 21, 2) and ctl.run is run and run.ticks == ticks


@needs_template
def test_the_arms_write_their_report(tmp_path, capsys):
    out = tmp_path / "arms"
    args = ["--arms", "off", "coph", "--seeds", "1", "--cars", "2", "--frames", "8"]
    assert cc.main([*args, "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["arms"]) == {"off", "coph"} and summary["seeds"] == [1]
    text = (out / "REPORT.md").read_text()
    assert text.index("allocation residual") < text.index("## Outcomes")
    assert summary["arms"]["off"]["bytes"] == 0.0


def test_without_the_template_it_says_what_to_install(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pycvc_gl.cars", None)
    monkeypatch.setattr(cc, "_template", lambda: None)  # once imported, the package has it
    assert cc.main(["--headless", "--frames", "1"]) == 2
    assert "pycvc-gl 3.7.12" in capsys.readouterr().err


def test_the_summary_prices_excess_risk_against_the_best_arm():
    def ep(arm, risk):
        led = cc.Ledger(risk_cost=risk).as_dict()
        car = dict(car=0, arrived=True, arrived_at=10.0, path_m=5.0, collided=False,
                   collided_ticks=0, bytes=0, accepted=1, rejected=0, geometry_overflow=False,
                   ledger=led)  # fmt: skip
        return dict(arm=arm, seed=1, cars=[car])

    s = cc.summarize([ep("off", 3.0), ep("coph", 1.0), ep("raw", 2.0)])
    assert s["arms"]["coph"]["excess_risk_max"] == 0.0
    assert s["arms"]["off"]["excess_risk_max"] == pytest.approx(2.0)
    assert math.isclose(s["arms"]["raw"]["excess_risk_mean"], 1.0)
