"""The fog-of-war stories and the material demo on a vehicle plant.

``fog record --plant`` runs a story's one agent on libcvc's sim_world and keeps that world honest
to the story (a wall raised, decaying marks, moving vehicles, a moving goal), writing the same
trace the Python scenario writes. ``material-demo --plant`` and ``Squad(plant=..., material=...)``
put the material's forces and witness gate on the native world. The CLI tests are pure Python;
the rest need a pycvc with libcvc's fog bindings (``sim_world::set_truth``) and skip without.
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")

from grl_snam import nav_native  # noqa: E402
from grl_snam.cli import main  # noqa: E402


def test_fog_and_the_material_demo_take_a_plant():
    for args in (
        ["fog", "record", "--help"],
        ["fog", "all", "--help"],
        ["material-demo", "--help"],
    ):
        out = CliRunner().invoke(main, args).output
        assert "--plant" in out and "--mu" in out, args
    out = CliRunner().invoke(main, ["fog", "record", "--help"]).output
    for story in ("ghost", "blocker", "unit", "city", "traffic", "pursuit"):
        assert story in out


fog = pytest.mark.skipif(
    not (nav_native.HAS_FOG and nav_native.HAS_PLANT),
    reason="pycvc lacks libcvc's sim_world fog bindings (nav_sim_world_set_truth)",
)


def _record(tmp_path, story, plant, max_steps):
    from grl_snam.fog_trace import Trace
    from grl_snam.tools import fog_record

    d = fog_record.record(story, tmp_path / story, plant=plant, max_steps=max_steps)
    return Trace.load(d), json.loads((d / "trace.json").read_text()), np.load(d / "trace.npz")


def _unpack(packed, n=96):
    return np.unpackbits(packed)[: n * n].reshape(n, n).astype(bool)


@fog
def test_a_wall_raised_mid_run_is_in_the_truth_and_the_belief(tmp_path):
    """blocker: at step 40 a wall rises across the route. The trace's truth gains it, the
    agent's belief learns it by sensing, and the Jolt RoboRacer still arrives, never inside it."""
    tr, manifest, z = _record(tmp_path, "blocker", "jolt", 1500)
    wall = np.zeros((96, 96), bool)
    wall[41:56, 46:50] = True
    assert not (_unpack(z["truth_snap"][0]) & wall).any(), "the field starts open"
    assert _unpack(z["truth_snap"][-1])[wall].all(), "and the wall is there at the end"
    assert any((_unpack(o) & wall).any() for o in z["snap_occ"]), "the belief learned it"
    s = manifest["summary"]
    assert s["reached_goal"] and s["penetration_steps"] == 0, s
    assert manifest["plant"] == "jolt" and manifest["nav"].startswith("sim_world")
    assert tr.n_ticks == s["steps"] and tr.fov_at(0.5 * tr.duration_s) is not None


@fog
def test_unit_marks_decay(tmp_path):
    """unit: the marks are believed as the decaying layer, and they go."""
    _tr, _m, z = _record(tmp_path, "unit", "kinematic", 400)
    dyn = [int(_unpack(d).sum()) for d in z["snap_dyn"]]
    assert max(dyn) > 0, "the unit's marks were believed"
    assert dyn[-1] < max(dyn), "and decayed"


@fog
def test_traffic_moves_in_the_truth(tmp_path):
    """traffic: the movers' footprints move through the recorded truth."""
    _tr, manifest, z = _record(tmp_path, "traffic", "dynamic", 120)
    first, last = _unpack(z["truth_snap"][0]), _unpack(z["truth_snap"][-1])
    assert (first != last).any(), "the truth changes as the vehicles drive"
    assert manifest["summary"]["steps"] == 120


@fog
def test_the_goal_moves_in_pursuit(tmp_path):
    _tr, _m, z = _record(tmp_path, "pursuit", "jolt", 200)
    gx, gy = z["row_goal_x"], z["row_goal_y"]
    assert np.hypot(gx[-1] - gx[0], gy[-1] - gy[0]) > 10.0, "the target ran"


@fog
def test_a_squad_on_a_plant_takes_a_material_grid():
    from grl_snam import planner
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.material import MaterialGrid
    from grl_snam.squad import AgentSpec, Squad

    story = shrunk(STORIES["city"], n=96, max_steps=200)
    truth = story.truth_grid()
    labels, sizes = planner.free_components(truth, 2)
    rows, cols = np.nonzero(labels == max(sizes, key=sizes.get))
    mnx, mny, mxx, mxy = story.bounds
    pts = np.stack([mnx + cols / 95 * (mxx - mnx), mny + rows / 95 * (mxy - mny)], 1)

    def free(x, y):
        return tuple(float(c) for c in pts[int(np.argmin(((pts - (x, y)) ** 2).sum(1)))])

    risk = np.zeros((96, 96), np.float32)
    risk[40:56, 40:56] = 0.9
    grid = MaterialGrid(risk, np.zeros((96, 96), bool), story.bounds, (0, 0), story.scale)
    torch.manual_seed(0)
    sq = Squad(
        story, [AgentSpec("a", free(-70, -60), free(70, 60))], truth_occ=truth, prior_occ=truth,
        plant="jolt", material=grid,
    )  # fmt: skip
    assert sq.native.has_material
    sq.run(max_steps=60, stop_when_done=False)
    assert sq.native.material_gate_active().shape == (1,)


@fog
def test_the_material_demo_runs_on_a_plant(tmp_path):
    pytest.importorskip("matplotlib")
    from grl_snam.tools import material_demo

    material_demo.main(str(tmp_path), plant="kinematic")
    assert (tmp_path / "material_scenario.png").exists()
    assert (tmp_path / "material_swarm.png").exists()
