"""grl_snam.cars (N Ackermann cars that a controller of your own drives) and its Ariadne template,
the ``cars`` port (grl_snam/ariadne/template/cars.py + cars.ari).

The library needs pycvc with cvc::nav; Jolt and the dynamic plant skip without them. The sim_world
backend (your commands in the navigator's place) needs a pycvc with nav_sim_world_override
(libcvc after 3.7.11) and skips without it; the direct backend (numpy bicycle,
physics_world.drive) runs on any. The port's runs need pycvc_gl's Python host and run headless.
No torch.
"""

import math

import numpy as np
import pytest

cars_mod = pytest.importorskip("grl_snam.cars")
if not cars_mod.HAS_PYCVC:
    pytest.skip("pycvc has no cvc::nav", allow_module_level=True)

from grl_snam.cars import (  # noqa: E402
    DT,
    LOG_FIELDS,
    Cars,
    CarsLog,
    parse_plant,
    vehicle,
)

JOLT = cars_mod.has_jolt()
needs_jolt = pytest.mark.skipif(not JOLT, reason="this pycvc has no Jolt physics")
needs_override = pytest.mark.skipif(
    not cars_mod.HAS_OVERRIDE, reason="this pycvc has no nav_sim_world_override"
)


def seek(s, speed=0.5):
    """A plain goal-seeker: steer at the goal, cruise at `speed` x top speed, slow into it."""
    d = s.goal - s.pos
    err = (np.arctan2(d[:, 1], d[:, 0]) - s.heading + np.pi) % (2 * np.pi) - np.pi
    v = np.minimum(speed * s.max_speed, 0.8 * s.goal_dist)
    return 1.5 * err, 1.5 * (np.where(s.reached, 0.0, v) - s.speed)


# ── the parts ──────────────────────────────────────────────────────────────────────────────────


def test_plants_parse():
    assert parse_plant("kinematic") == ("kinematic", ["roboracer"])
    assert parse_plant("humvee") == ("jolt", ["humvee"])
    assert parse_plant("jolt:roboracer, humvee") == ("jolt", ["roboracer", "humvee"])
    assert parse_plant("dynamic:humvee") == ("dynamic", ["humvee"])
    assert parse_plant("kinematic:") == ("kinematic", ["roboracer"])
    for bad in ("warp", "jolt:tank"):
        with pytest.raises(ValueError):
            parse_plant(bad)


def test_vehicle_presets():
    r, h = vehicle("roboracer"), vehicle("humvee")
    assert r.length < 1.0 < h.length
    assert r.wheelbase == pytest.approx(0.324, abs=1e-3)
    assert h.wheelbase == pytest.approx(3.30, abs=1e-3)
    assert r.rear_x < 0.0 and h.rear_x < 0.0
    assert 0.3 < r.max_steer < 0.6 and r.max_speed == pytest.approx(10.0)


@pytest.mark.parametrize("world", ["flat", "blocks"])
def test_worlds_scale_to_the_cars(world):
    small = Cars(3, "kinematic:roboracer", world, seed=1, backend="direct")
    big = Cars(3, "kinematic:humvee", world, seed=1, backend="direct")
    span = lambda c: c.world.bounds[2] - c.world.bounds[0]  # noqa: E731
    assert span(big) / span(small) == pytest.approx(4.84 / 0.535, rel=0.02)
    for c in (small, big):
        s = c.state()
        assert not c.world.occupied(s.x, s.y).any(), "the cars start on free ground"
        assert not c.world.occupied(s.goal[:, 0], s.goal[:, 1]).any()
    if world == "blocks":
        assert len(small.world.boxes) > 4 and small.world.occ.mean() > 0.2


def test_reset_is_repeatable():
    a = Cars(5, "kinematic", "blocks", seed=7, backend="direct")
    b = Cars(5, "kinematic", "blocks", seed=7, backend="direct")
    assert np.allclose(a.starts, b.starts) and np.allclose(a.goals, b.goals)
    c = Cars(5, "kinematic", "blocks", seed=8, backend="direct")
    assert not np.allclose(a.starts, c.starts)


# ── the kinematic bicycle ──────────────────────────────────────────────────────────────────────


def test_the_bicycle_drives_straight_then_turns():
    cars = Cars(1, "kinematic", "flat", seed=0, backend="direct", headings=0.0)
    x0, y0 = cars.state().x[0], cars.state().y[0]
    for _ in range(20):  # 1 s at 1 m/s^2 from rest
        s = cars.step([0.0], [1.0])
    assert s.speed[0] == pytest.approx(1.0, abs=1e-9)
    assert s.x[0] - x0 == pytest.approx(0.5, abs=0.03) and s.y[0] == pytest.approx(y0)
    for _ in range(20):  # a left turn at constant speed: heading rate v / L tan(steer)
        s = cars.step([0.3], [0.0])
    assert s.heading[0] == pytest.approx(math.tan(0.3) / s.wheelbase[0], rel=0.01)
    assert s.steer[0] == pytest.approx(0.3)


def test_commands_are_held_to_the_limits():
    cars = Cars(1, "kinematic", "flat", seed=0, backend="direct", headings=0.0)
    s = cars.step([5.0], [100.0])
    assert cars.last_command[0][0] == pytest.approx(s.max_steer[0])
    assert cars.last_command[1][0] == pytest.approx(s.max_accel[0])
    for _ in range(200):
        s = cars.step([0.0], [100.0])
    assert s.speed[0] == pytest.approx(s.max_speed[0])
    s = cars.step([float("nan")], [float("nan")])  # a NaN is zero
    assert cars.last_command[0][0] == 0.0


@needs_override
def test_the_numpy_bicycle_is_navs():
    """The direct backend's bicycle and libcvc's sim_world (nav_sim_world_override) agree."""
    a = Cars(4, "kinematic", "flat", seed=3, backend="direct")
    b = Cars(4, "kinematic", "flat", seed=3, backend="sim")
    rng = np.random.default_rng(0)
    for _ in range(60):
        steer, accel = rng.uniform(-0.4, 0.4, 4), rng.uniform(-3.0, 3.0, 4)
        sa, sb = a.step(steer, accel), b.step(steer, accel)
    for k in ("x", "y", "heading", "speed"):
        assert np.allclose(getattr(sa, k), getattr(sb, k), atol=2e-3), k


# ── what the controller reads ──────────────────────────────────────────────────────────────────


def test_ranges_see_the_wall():
    cars = Cars(1, "kinematic", "flat", seed=0, backend="direct", headings=0.0)
    s = cars.state()
    edge = cars.world.bounds[2] - 0.5 * cars.unit  # the wall's inner face
    cx = s.center[0, 0]
    ahead = s.ranges[0, np.argmin(np.abs(s.ray_angles))]
    want = min(edge - cx, s.range_max)
    assert ahead == pytest.approx(want, abs=1.0 * cars.world.cell)


def test_collisions_count_cars_and_walls():
    cars = Cars(2, "kinematic", "flat", seed=0, backend="direct", headings=0.0)
    b = cars._b
    b.x[:] = [0.0, 0.1]
    b.y[:] = [0.0, 0.05]
    assert cars._read().collided.all(), "two cars on top of each other"
    b.x[:] = [0.0, 5.0 * cars.unit]
    assert not cars._read().collided.any()
    b.x[0] = cars.world.bounds[2] - 0.2 * cars.unit
    assert cars._read().collided[0], "a car in the wall"


def test_new_goals_count_arrivals():
    cars = Cars(3, "kinematic", "flat", seed=0, backend="direct")
    s = cars.state()
    for _ in range(600):
        s = cars.step(*seek(s))
        if s.reached.any():
            before = s.goal.copy()
            s = cars.new_goals(s.reached)
            assert not np.allclose(before, s.goal)
    assert cars.arrivals >= 3


# ── the plants ─────────────────────────────────────────────────────────────────────────────────


def drive(cars, ticks=400):
    s = cars.state()
    d0 = s.goal_dist.copy()
    for _ in range(ticks):
        s = cars.step(*seek(s))
    return d0, s


def test_kinematic_cars_reach_their_goals():
    d0, s = drive(Cars(6, "kinematic", "flat", seed=1, backend="direct"))
    assert s.reached.all(), s.goal_dist


@needs_jolt
def test_jolt_cars_reach_their_goals_without_override():
    cars = Cars(4, "jolt:roboracer", "flat", seed=1, backend="direct")
    d0, s = drive(cars)
    assert s.reached.sum() >= 3, s.goal_dist
    assert np.abs(s.steer).max() <= s.max_steer.max() + 0.02


@needs_jolt
def test_jolt_mixed_presets_cycle():
    cars = Cars(3, "jolt:roboracer,humvee", "flat", seed=2, backend="direct")
    assert [v.name for v in cars.vehicles] == ["roboracer", "humvee", "roboracer"]
    s = cars.step([0.0] * 3, [1.0] * 3)
    assert s.length[1] > 4.0 > s.length[0]


@needs_jolt
@needs_override
def test_jolt_backs_up_on_the_sim_backend():
    cars = Cars(1, "jolt:roboracer", "flat", seed=0, backend="sim", headings=0.0)
    x0 = cars.state().x[0]
    for _ in range(30):
        s = cars.step([0.0], [-1.0])
    assert s.speed[0] < -0.1 and s.x[0] < x0 - 0.1


@needs_override
@pytest.mark.skipif(not cars_mod.HAS_DYN, reason="this pycvc has no dyn_bike_plant")
def test_dynamic_cars_move():
    d0, s = drive(Cars(4, "dynamic:roboracer", "flat", seed=1), ticks=300)
    assert (s.goal_dist < d0).all()


def test_dynamic_needs_the_override(monkeypatch):
    monkeypatch.setattr(cars_mod, "HAS_OVERRIDE", False)
    with pytest.raises(RuntimeError, match="override"):
        Cars(2, "dynamic:roboracer", "flat")
    assert Cars(2, "kinematic", "flat").backend_name == "direct"


def test_the_navigator_baseline():
    cars = Cars(4, "kinematic", "flat", seed=1, navigator=True)
    with pytest.raises(RuntimeError):
        cars.step([0.0] * 4, [0.0] * 4)
    s = cars.state()
    d0 = s.goal_dist.copy()
    for _ in range(300):
        s = cars.step_navigator()
    assert (s.goal_dist < d0).sum() >= 3
    assert np.isnan(cars.last_command[0]).all()


# ── logging ────────────────────────────────────────────────────────────────────────────────────


def test_the_log_saves_npz_and_csv(tmp_path):
    cars = Cars(3, "kinematic", "flat", seed=0, backend="direct")
    log = CarsLog()
    s = cars.state()
    for _ in range(10):
        s = cars.step(*seek(s))
        log.record(s, cars.last_command)
    z = np.load(log.save(tmp_path / "run.npz"))
    assert z["x"].shape == (10, 3) and z["t"][-1] == pytest.approx(10 * DT)
    assert set(LOG_FIELDS) <= set(z.files) and str(z["meta_plant"]) == "kinematic"
    rows = open(log.save(tmp_path / "run.csv")).read().splitlines()
    assert rows[0].startswith("tick,t,car,x,y") and len(rows) == 1 + 10 * 3


# ── the template and its port ──────────────────────────────────────────────────────────────────


def test_the_default_controller_drives_the_cars():
    from grl_snam.ariadne.template import cars as template

    cars = Cars(6, "kinematic", "flat", seed=4, backend="direct")
    s = cars.state()
    for _ in range(300):
        s = cars.step(*template.controller(s, dict(template.PARAMS)))
    assert s.reached.sum() >= 5, s.goal_dist
    assert cars.collisions.sum() <= 10


def test_copy_writes_both_files(tmp_path):
    from grl_snam.ariadne.template import cars as template

    out = tmp_path / "mine"
    assert template.main(["--copy", str(out)]) == 0
    assert (out / "cars.py").read_bytes() == (template.HERE / "cars.py").read_bytes()
    assert (out / "cars.ari").is_file()
    assert template.main(["--copy", str(out)]) == 1, "refuses to overwrite"
    assert template.main(["--copy", str(out), "--force"]) == 0


@pytest.fixture
def ports_main():
    pytest.importorskip("pycvc_gl.ariadne", reason="pycvc_gl has no Python host (libcvc 3.7.9+)")
    from grl_snam.ariadne.__main__ import main

    return main


def test_the_port_runs_headless_and_logs(ports_main, tmp_path):
    log = tmp_path / "run.npz"
    args = ["cars", "--headless", "--frames", "40", "--plant", "kinematic", "--log", str(log)]
    assert ports_main(args) == 0
    z = np.load(log)
    assert z["x"].shape == (40, 6) and not np.isnan(z["cmd_accel"]).any()


@needs_jolt
def test_the_port_runs_jolt_headless(ports_main, tmp_path):
    log = tmp_path / "run.csv"
    args = ["cars", "--headless", "--frames", "20", "--cars", "3", "--plant", "jolt:roboracer"]
    assert ports_main([*args, "--log", str(log)]) == 0
    assert len(log.read_text().splitlines()) == 1 + 20 * 3


def _austin_bundle():
    import os

    b = os.environ.get("GRL_SNAM_SCENE_BUNDLE")
    if not b:
        try:
            from pycvc_gl.scenes import find_scene

            b = find_scene()
        except Exception:  # noqa: BLE001
            b = None
    return b if b and os.path.isfile(os.path.join(str(b), "terrain.json")) else None


def test_the_port_drives_on_austin_from_the_data_store(ports_main, tmp_path):
    """Austin loads once into the document's data store (what its city node draws, aerial image
    and all) and the cars drive on the same objects. Needs an installed Austin bundle (no
    download in tests)."""
    bundle = _austin_bundle()
    if bundle is None:
        pytest.skip("no Austin scene bundle (scene-austin-south or -small)")
    log = tmp_path / "austin.npz"
    args = ["cars", "--headless", "--frames", "10", "--cars", "3", "--plant", "kinematic"]
    assert ports_main([*args, "--world", "austin", "--bundle", str(bundle), "--log", str(log)]) == 0
    z = np.load(log)
    assert str(z["meta_world"]) == "austin" and z["x"].shape == (10, 3)
