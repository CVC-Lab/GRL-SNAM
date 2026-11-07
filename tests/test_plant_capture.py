"""The plant choice in ``grl-snam capture`` and its HUD: the CLI's options, the plant names,
the HUD lines and the run summary (pure Python), and with a pycvc that carries libcvc's plants
:class:`grl_snam.plant_nav.PlantNavigator` driving the policy on nav's kinematic bicycle, the
dynamic single-track vehicle and a Jolt vehicle (those tests skip on any other build)."""

import math

import numpy as np
import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")

from grl_snam import nav_native  # noqa: E402
from grl_snam.cli import main  # noqa: E402
from grl_snam.metrics import NavMetrics, hud_lines, plant_hud_lines  # noqa: E402
from grl_snam.tools import capture  # noqa: E402


def test_capture_takes_a_plant():
    for cmd in ("drive", "multigoal"):
        out = CliRunner().invoke(main, ["capture", cmd, "--help"]).output
        for opt in ("--plant", "--vehicle", "--mu", "--routes", "--fog", "--weights", "--reach-m"):
            assert opt in out, (cmd, opt)
        assert "point|kinematic|dynamic|jolt" in out
    out = CliRunner().invoke(main, ["pipeline", "--help"]).output
    assert "--plant" in out and "--mu" in out
    assert "--plant" in CliRunner().invoke(main, ["finale", "--help"]).output


def test_plant_names():
    assert nav_native.make_plant(None) is None
    assert nav_native.make_plant("kinematic") is None
    for bad in ("hovercraft", "dynamic4w", "jolt:tank"):
        with pytest.raises(ValueError):
            nav_native.make_plant(bad)


def test_the_hud_reads_the_plant():
    jolt = {
        "name": "jolt:humvee",
        "mu": 0.3,
        "slip_deg": 34.0,
        "tire_use": 2.5,
        "lateral_g": 0.31,
        "speed_mps": 9.0,
        "touching": True,
        "books": {"source": 5e5, "tire": 4e4, "brake": 1e4, "aero": 2e3, "rolling": 3e3,
                  "kinetic": 4e4},
    }  # fmt: skip
    lines = plant_hud_lines(jolt)
    assert lines[0].startswith("plant jolt:humvee") and "0.30" in lines[0]
    assert "[PAST PEAK]" in lines[1] and "[SPIN]" in lines[1] and "[CONTACT]" in lines[1]
    assert lines[2].startswith("energy kJ") and "500.0" in lines[2]
    dyn = dict(jolt, name="dynamic:humvee", tire_use=0.5, slip_deg=3.0, touching=False)
    dyn["books"] = {"drive": 6e4, "brake": 1e4, "dissipated": 5e3, "contact": 0.0,
                    "residual": 1.0, "kinetic": 4e4}  # fmt: skip
    lines = plant_hud_lines(dyn)
    assert "[" not in lines[1], "within the grip, no flags"
    assert lines[2].startswith("books kJ") and "drive   60.0" in lines[2]
    assert plant_hud_lines({"name": "kinematic"}) == [
        "plant kinematic: kinematic bicycle, no tires"
    ]
    m = NavMetrics(plant=jolt)
    assert hud_lines(m)[-3:] == plant_hud_lines(jolt)
    assert all(not ln.startswith("plant") for ln in hud_lines(NavMetrics()))


def test_the_run_summary_says_how_each_leg_ended():
    def leg(n, reached, stall, slip=0.0):
        ms = []
        for i in range(n):
            ms.append(
                NavMetrics(
                    step=i + 1, goal_dist_m=100.0 - i, clearance_m=5.0, speed_mps=3.0,
                    reached=reached and i == n - 1, stall=stall if i == n - 1 else 0,
                    plant={"slip_deg": slip, "tire_use": 1.5 if i % 2 else 0.5,
                           "lateral_g": 0.2, "touching": i == 3, "books": {"drive": 10.0 * i}},
                )
            )  # fmt: skip
        return ms

    legs = [(0, leg(10, True, 0)), (1, leg(5, False, capture.STALL_TICKS + 1, 40.0)),
            (2, leg(4, False, 2))]  # fmt: skip
    s = capture._summary(legs, [(1, 2), (3, 4), (5, 6)], "jolt", "humvee", 0.3, 0.06)
    assert [lg["ended"] for lg in s["legs"]] == ["reached", "stalled", "out of ticks"]
    first = s["legs"][0]
    assert first["ticks"] == 10 and math.isclose(first["world_s"], 0.6)
    assert first["ticks_past_peak"] == 5 and first["contact_ticks"] == 1
    assert s["legs"][1]["max_abs_slip_deg"] == 40.0
    assert first["books_J"] == {"drive": 90.0}


def test_the_jolt_rollout_trains_natively_only():
    from grl_snam.tools import coef_train

    with pytest.raises(SystemExit, match="--backend native"):
        coef_train.main(["--rollout", "jolt", "--steps", "1"])


plants = pytest.mark.skipif(
    not (nav_native.HAS_DYN_PLANT and nav_native.HAS_SIM_WORLD),
    reason="pycvc lacks libcvc's plant bindings (nav_plant, dyn_bike_plant)",
)


def _city_drive():
    """The city story's field, a fresh policy and a start and goal across its free space."""
    import grl_snam.sdf_nav as sdf_nav
    from grl_snam import planner
    from grl_snam.fog_stories import STORIES, shrunk

    story = shrunk(STORIES["city"], n=96, max_steps=600)
    truth = story.truth_grid()
    meta = dict(story.meta(), bounds=story.bounds)
    S = float(meta["scale"])
    phi, nx, ny = sdf_nav.build_sdf(truth, story.bounds, S)
    field = sdf_nav.SDFField(phi, nx, ny, story.bounds, meta["center"], S)
    labels, sizes = planner.free_components(truth, 2)
    rows, cols = np.nonzero(labels == max(sizes, key=sizes.get))
    mnx, mny, mxx, mxy = story.bounds
    pts = np.stack([mnx + cols / 95 * (mxx - mnx), mny + rows / 95 * (mxy - mny)], 1)

    def free(x, y):
        return tuple(float(c) for c in pts[int(np.argmin(((pts - (x, y)) ** 2).sum(1)))])

    torch.manual_seed(0)
    return field, sdf_nav.CoefMLP().eval(), meta, truth, free(-70, -60), free(70, 60)


@plants
@pytest.mark.parametrize("plant", ["kinematic", "dynamic:roboracer", "jolt:roboracer"])
def test_the_policy_drives_each_plant(plant):
    from grl_snam.plant_nav import PlantNavigator

    field, model, meta, truth, start, goal = _city_drive()
    nav = PlantNavigator(field, model, meta, truth, plant, reach_m=4.0)
    nav.start(start, goal)
    d0 = math.dist(start, goal)
    ms, best_i, o, v = nav.drive_to_goal(max_steps=300)
    assert o is None and v is None, "a vehicle is not rewound"
    assert ms[best_i].goal_dist_m < d0 - 20.0, (plant, d0, ms[best_i].goal_dist_m)
    p = ms[-1].plant
    assert p["name"] == plant
    if plant == "kinematic":
        assert set(p) == {"name"}
        return
    assert {"slip_deg", "tire_use", "lateral_g", "speed_mps", "books", "touching"} <= set(p)
    if plant.startswith("dynamic"):
        b = nav.plant.books(0)
        gap = b["kinetic"] - nav.plant.plant.kinetic_at_attach(0) - (
            b["drive"] - b["brake"] - b["dissipated"] + b["floor"] + b["grade"] + b["limiter"]
            + b["contact"] + b["residual"])  # fmt: skip
        assert abs(gap) <= 1e-9 * b["flow"], "the books close"
    else:
        assert p["books"]["source"] > 0.0, "Jolt's ledger says what the battery gave"
    st = nav.nav_stats()
    if st is not None:
        # (building_contacts counts every chassis contact with a static body, the ground too.)
        assert st["penetration_steps"] == 0 and st["veh_contacts"] == 0, st


@plants
def test_low_grip_slides_past_the_peak():
    """On mu 0.15 the dynamic RoboRacer cannot corner as nav plans (at 0.5 g): its tires spend
    far longer past their peak than on mu 0.9 (134 ticks against 38 when measured; the city
    policy pushes a small car past the peak in its sharpest turns even on dry ground)."""
    from grl_snam.plant_nav import PlantNavigator

    field, model, meta, truth, start, goal = _city_drive()
    past = {}
    for mu in (0.9, 0.15):
        nav = PlantNavigator(field, model, meta, truth, "dynamic:roboracer", ground_mu=mu)
        nav.start(start, goal)
        ms, _, _, _ = nav.drive_to_goal(max_steps=300)
        past[mu] = sum(1 for m in ms if m.plant["tire_use"] > 1.0)
    assert past[0.15] > 2 * past[0.9], past


@plants
def test_a_squad_drives_the_dynamic_plant():
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.squad import AgentSpec, Squad

    _field, _model, _meta, truth, a, b = _city_drive()
    story = shrunk(STORIES["city"], n=96, max_steps=1500)
    agents = [AgentSpec("a", a, b), AgentSpec("b", b, a)]
    torch.manual_seed(0)
    sq = Squad(story, agents, truth_occ=truth, prior_occ=truth, plant="dynamic")
    assert sq.native.plant_kind == "dynamic" and isinstance(sq.plant, nav_native.DynamicPlant)
    r = sq.run(max_steps=1500)
    assert all(e > 0.0 for e in r.energy_J.values()), r.energy_J
    assert r.penetration == {"a": 0, "b": 0}, r.penetration
    assert np.allclose(sq.goal_of("a"), b) and sq.reached_of("a") == r.reached["a"]


@plants
def test_record_squad_on_a_plant(tmp_path):
    import json

    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.squad import AgentSpec
    from grl_snam.tools.squad_record import record_squad

    _field, _model, _meta, truth, a, b = _city_drive()
    story = shrunk(STORIES["city"], n=96, max_steps=40)
    out = record_squad(
        story, [AgentSpec("a", a, b), AgentSpec("b", b, a)], tmp_path, max_steps=40,
        truth_occ=truth, prior_occ=truth, plant="jolt",
    )  # fmt: skip
    squad = json.loads((out / "squad.json").read_text())
    assert squad["plant"] == "jolt" and squad["ticks"] == 40
    tr = np.load(out / "a" / "trace.npz")
    assert len(tr["row_x"]) == 40 and not tr["row_sensed"].any(), "the native world senses alone"
    assert np.isfinite(tr["row_clearance_m"]).all()
    assert np.allclose(tr["row_goal_x"], b[0]) and np.allclose(tr["row_goal_y"], b[1])
