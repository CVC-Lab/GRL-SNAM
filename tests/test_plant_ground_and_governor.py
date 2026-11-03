"""A plant's ground is no building, and libcvc's risk governor under the material on a plant.

``JoltPlant`` names its ground to ``pycvc.nav_plant`` (``ground_bodies``), so a chassis scraping
it is not a building contact in nav_stats, and falls back on a pycvc without the argument.
``material-demo --plant`` and ``PlantNavigator(risk_governor=...)`` switch on the governor
(``NativeSimWorld.set_risk_governor``): the material's forces alone leave the dynamic plant's run
in the scenario story as it is without material; the governor holds it out of the water.
"""

import numpy as np
import pytest
from click.testing import CliRunner

pytest.importorskip("torch")

from grl_snam import nav_native  # noqa: E402
from grl_snam.cli import main  # noqa: E402
from grl_snam.tools import material_demo  # noqa: E402


def test_the_material_demo_takes_the_governor():
    out = CliRunner().invoke(main, ["material-demo", "--help"]).output
    assert "--governor" in out and "--no-governor" in out


def test_the_governor_is_on_for_the_physics_plants_by_default():
    assert material_demo._governed(None, "kinematic") is False
    for plant in ("dynamic", "dynamic:roboracer", "jolt", "jolt:humvee"):
        assert material_demo._governed(None, plant) is nav_native.HAS_RISK_GOVERNOR
    assert material_demo._governed(True, "kinematic") is True
    assert material_demo._governed(False, "jolt") is False


@pytest.mark.skipif(not nav_native.HAS_PLANT, reason="pycvc lacks the physics bindings")
def test_jolt_names_its_ground(monkeypatch):
    real = nav_native._pycvc.nav_plant
    seen = []

    def spy(world, descs, **kw):
        seen.append(kw.get("ground_bodies"))
        return real(world, descs, **kw)

    monkeypatch.setattr(nav_native._pycvc, "nav_plant", spy)
    try:
        pl = nav_native.JoltPlant("roboracer", bounds=(-20.0, -20.0, 20.0, 20.0))
    except TypeError:
        pytest.skip("this pycvc's nav_plant takes no ground_bodies")
    assert seen[-1] == [pl.ground]

    def old(world, descs, **kw):  # a pycvc before ground_bodies
        if "ground_bodies" in kw:
            raise TypeError("nav_plant() got an unexpected keyword argument 'ground_bodies'")
        seen.append("fallback")
        return real(world, descs, **kw)

    monkeypatch.setattr(nav_native._pycvc, "nav_plant", old)
    nav_native.JoltPlant("roboracer", bounds=(-20.0, -20.0, 20.0, 20.0))
    assert seen[-1] == "fallback"


@pytest.mark.skipif(
    not (nav_native.HAS_RISK_GOVERNOR and nav_native.HAS_DYN_PLANT),
    reason="pycvc lacks nav_sim_world_set_risk_governor or dyn_bike_plant",
)
def test_the_governor_holds_the_dynamic_plant_out_of_the_water():
    _risk, _hard, plain, forces = material_demo.scenario_runs("dynamic", governor=False)
    assert plain[3] > 0, "without material it drives through the water"
    k = min(len(plain[0]), len(forces[0]))
    assert (
        len(plain[0]) == len(forces[0]) and np.abs(plain[0][:k] - forces[0][:k]).max() < 1e-3
    ), "the forces alone leave the dynamic plant's run as it is without material"
    _risk, _hard, plain_g, governed = material_demo.scenario_runs("dynamic", governor=True)
    assert np.array_equal(plain_g[0], plain[0]), "without material the governor has no risk to read"
    assert governed[3] == 0, "the governor stops it short of the water"
