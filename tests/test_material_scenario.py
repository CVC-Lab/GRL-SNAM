"""Material-aware navigation end-to-end: scenario, squad, and swarm.

The contract under test: attaching a MaterialGrid turns the whole feature on
(planner cost + forces + witness gate) in pure Python; NOT attaching one — and
even attaching an all-zero one — leaves trajectories exactly as before.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sdf_nav  # noqa: E402
from grl_snam.material import MaterialGrid, MaterialParams  # noqa: E402
from grl_snam.scenario import FogScenario  # noqa: E402

N = 96
BOUNDS = (-100.0, -100.0, 100.0, 100.0)
SCALE = 0.05


def _meta():
    return dict(
        scale=SCALE,
        center=(0.0, 0.0),
        region=100.0,
        rr=0.15,
        d_hat=0.35,
        dt=0.06,
        nsub=2,
        vmax=0.9,
        bounds=list(BOUNDS),
    )


def _model():
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _cell_of(x, y):
    c = (x - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]) * (N - 1)
    r = (y - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1]) * (N - 1)
    return int(round(r)), int(round(c))


def _mud_band():
    """High risk band across the direct route, with a clean corridor below."""
    risk = np.zeros((N, N), np.float32)
    risk[40:56, 40:70] = 0.95  # mud straddling the straight line start->goal
    return risk


def _scenario(material=None, **kw):
    truth = np.zeros((N, N), bool)
    return FogScenario(
        truth,
        BOUNDS,
        SCALE,
        _model(),
        _meta(),
        waypoints=[(60.0, 0.0)],
        material=material,
        **kw,
    ).start((-60.0, 0.0))


def _risk_exposure(res, risk):
    total = 0.0
    for rec in res.records:
        r, c = _cell_of(rec.x, rec.y)
        if 0 <= r < N and 0 <= c < N:
            total += float(risk[r, c])
    return total


def test_zero_material_grid_leaves_trajectory_value_identical():
    """The safe-default-on demonstration: an all-zero MaterialGrid produces
    exactly-zero forces and an inert gate, so the drive is value-identical to
    no material at all (the planner cost raster is all zeros too)."""
    zero = MaterialGrid(np.zeros((N, N), np.float32), np.zeros((N, N), bool), BOUNDS, (0, 0), SCALE)
    a = _scenario().run(max_steps=600, stop_when_done=False)
    b = _scenario(material=zero).run(max_steps=600, stop_when_done=False)
    pa, pb = a.positions, b.positions
    assert pa.shape == pb.shape
    assert np.array_equal(pa, pb)


def test_material_scenario_detours_around_mud_and_still_arrives():
    risk = _mud_band()
    grid = MaterialGrid(risk, np.zeros((N, N), bool), BOUNDS, (0, 0), SCALE)
    plain = _scenario().run(max_steps=3000)
    mat = _scenario(material=grid).run(max_steps=3000)
    assert mat.waypoints_reached == 1, "material-aware run failed to arrive"
    exp_plain = _risk_exposure(plain, risk)
    exp_mat = _risk_exposure(mat, risk)
    assert exp_plain > 0, "baseline never crossed the mud — test setup broken"
    assert (
        exp_mat < 0.25 * exp_plain
    ), f"material-aware exposure {exp_mat:.1f} not clearly below baseline {exp_plain:.1f}"


def test_hard_material_cells_are_avoided_without_being_walls():
    """A hard-hazard strip (water: lethal, not geometry) with a gap: the
    planner surcharge + hazard barrier must route through the gap; truth
    occupancy is EMPTY so nothing physically blocks the straight line."""
    risk = np.zeros((N, N), np.float32)
    hard = np.zeros((N, N), bool)
    hard[38:58, 46:50] = True
    hard[64:80, 46:50] = True  # gap between rows 58..64 and below 38
    risk[hard] = 1.0
    grid = MaterialGrid(risk, hard, BOUNDS, (0, 0), SCALE)
    res = _scenario(material=grid).run(max_steps=3500)
    assert res.waypoints_reached == 1
    hard_hits = 0
    for rec in res.records:
        r, c = _cell_of(rec.x, rec.y)
        if 0 <= r < N and 0 <= c < N and hard[r, c]:
            hard_hits += 1
    assert hard_hits == 0, f"drove through hard hazard cells {hard_hits} steps"


def test_stamp_risk_mid_run_forces_a_replan():
    grid = MaterialGrid(np.zeros((N, N), np.float32), np.zeros((N, N), bool), BOUNDS, (0, 0), SCALE)
    sc = _scenario(material=grid)
    for _ in range(20):
        sc.step()
    route_before = list(sc.route or [])
    # mud onset squarely on the committed route
    grid.stamp_risk(44, 52, 44, 60, 0.95)
    sc.step()
    assert sc.route is not None
    assert list(sc.route) != route_before, "stamped mud did not change the committed route"


def test_material_metrics_are_populated():
    risk = _mud_band()
    grid = MaterialGrid(risk, np.zeros((N, N), bool), BOUNDS, (0, 0), SCALE)
    sc = _scenario(material=grid)
    m = sc.nav.step()
    assert hasattr(m, "material_risk") and hasattr(m, "material_gate")
    assert m.material_risk >= 0.0
    plain = _scenario()
    m2 = plain.nav.step()
    assert m2.material_risk == 0.0 and m2.material_gate is False


def test_gate_disabled_param_keeps_soft_force_always_on():
    risk = _mud_band()
    p = MaterialParams(gate_enabled=False)
    grid = MaterialGrid(risk, np.zeros((N, N), bool), BOUNDS, (0, 0), SCALE, params=p)
    sc = _scenario(material=grid)
    lam_s, lam_h = sc.material.lambdas((0.0, 0.0), (60.0, 0.0))
    assert lam_s == p.lam_soft  # no gate multiplier
    assert lam_h == p.lam_hard
    assert sc.material.last_gate is None


def test_route_cost_fn_seam_composes_instead_of_clobbering():
    risk = _mud_band()
    grid = MaterialGrid(risk, np.zeros((N, N), bool), BOUNDS, (0, 0), SCALE)
    sc = _scenario(material=grid)
    user = np.full((N, N), 2.0)
    sc.route_cost_fn = lambda: user
    cost = sc._route_cost()
    assert np.array_equal(cost, user + grid.cost_raster())
    sc.route_cost_fn = None
    assert np.array_equal(sc._route_cost(), grid.cost_raster())


def test_squad_with_material_falls_back_to_serial_drive():
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.squad import AgentSpec, Squad

    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    specs = [
        AgentSpec("a0", (-40.0, -40.0), (40.0, 40.0)),
        AgentSpec("a1", (-40.0, 40.0), (40.0, -40.0)),
    ]
    risk = np.zeros((64, 64), np.float32)
    risk[28:36, 20:44] = 0.9
    grid = MaterialGrid(risk, np.zeros((64, 64), bool), story.bounds, (0, 0), story.scale)
    squad = Squad(story, specs, _model(), batched_drive=True, material=grid)
    assert not squad._can_batch_drive(), "material squad must fall back to the serial act"
    for _ in range(6):
        squad.step()
    for sc in squad.scenarios.values():
        assert sc.material is not None
        assert sc.nav.material is sc.material


def _swarm_run(story, material, risk, hard, steps=500):
    """One agent -45 -> +45 across a 64-cell story world; returns
    (risk_exposure, hard_hits, reached, steps_taken)."""
    from grl_snam.squad import AgentSpec
    from grl_snam.swarm import Swarm

    torch.manual_seed(0)
    s = Swarm(
        story,
        [AgentSpec("a0", (-45.0, 0.0), (45.0, 0.0))],
        _model(),
        seed=0,
        truth_occ=np.zeros((64, 64), bool),
        material=material,
    )
    exposure = 0.0
    hits = 0
    k = 0
    for k in range(steps):
        s.step()
        w = s.n2w(s.o).cpu().numpy()
        c = int(
            np.clip(
                round((w[0, 0] - story.bounds[0]) / (story.bounds[2] - story.bounds[0]) * 63), 0, 63
            )
        )
        r = int(
            np.clip(
                round((w[0, 1] - story.bounds[1]) / (story.bounds[3] - story.bounds[1]) * 63), 0, 63
            )
        )
        exposure += float(risk[r, c])
        hits += int(hard[r, c])
        if bool(s.reached[0]):
            break
    return exposure, hits, bool(s.reached[0]), k + 1


def test_swarm_avoids_soft_risk_blob_and_still_reaches():
    """A locally-avoidable mud blob on the straight line: the reactive swarm
    (no planner — forces + gate are its only material channel) must skirt it
    entirely and still arrive."""
    from grl_snam.fog_stories import STORIES, shrunk

    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    rr, cc = np.mgrid[0:64, 0:64]
    risk = np.where((rr - 30) ** 2 + (cc - 32) ** 2 <= 49, 0.95, 0.0).astype(np.float32)
    hard = np.zeros((64, 64), bool)
    grid = MaterialGrid(risk, hard, story.bounds, (0, 0), story.scale)
    e_plain, _, reached_plain, _ = _swarm_run(story, None, risk, hard)
    e_mat, _, reached_mat, _ = _swarm_run(story, grid, risk, hard)
    assert reached_plain and e_plain > 10.0, "baseline setup broken"
    assert reached_mat, "material swarm failed to arrive"
    assert e_mat < 0.1 * e_plain, f"swarm exposure {e_mat:.1f} vs plain {e_plain:.1f}"


def test_swarm_never_enters_hard_hazard():
    """Hazard squarely on the goal line: the no-entry guarantee holds. The
    reactive field may dead-end against it (a potential-field minimum — the
    source method always ran under a planner scaffold; see MaterialParams):
    reach is NOT asserted here, non-entry is."""
    from grl_snam.fog_stories import STORIES, shrunk

    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    rr, cc = np.mgrid[0:64, 0:64]
    hard = (rr - 30) ** 2 + (cc - 32) ** 2 <= 36
    risk = np.where(hard, 1.0, 0.0).astype(np.float32)
    grid = MaterialGrid(risk, hard, story.bounds, (0, 0), story.scale)
    _, hits_plain, _, _ = _swarm_run(story, None, risk, hard)
    _, hits_mat, _, _ = _swarm_run(story, grid, risk, hard)
    assert hits_plain > 0, "baseline never touched the hazard — setup broken"
    assert hits_mat == 0, f"material swarm entered the hazard {hits_mat} steps"


def test_swarm_rounds_offset_hazard_and_reaches():
    """A hazard grazing (not blocking) the path: no entry AND arrival."""
    from grl_snam.fog_stories import STORIES, shrunk

    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    rr, cc = np.mgrid[0:64, 0:64]
    hard = (rr - 38) ** 2 + (cc - 32) ** 2 <= 36
    risk = np.where(hard, 1.0, 0.0).astype(np.float32)
    grid = MaterialGrid(risk, hard, story.bounds, (0, 0), story.scale)
    _, hits_plain, reached_plain, _ = _swarm_run(story, None, risk, hard)
    _, hits_mat, reached_mat, _ = _swarm_run(story, grid, risk, hard)
    assert reached_plain and hits_plain > 0, "baseline setup broken"
    assert reached_mat and hits_mat == 0


def test_swarm_material_run_is_deterministic():
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.squad import AgentSpec
    from grl_snam.swarm import Swarm

    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    specs = [AgentSpec("a0", (-40.0, 0.0), (40.0, 0.0))]
    rr, cc = np.mgrid[0:64, 0:64]
    risk = np.where((rr - 30) ** 2 + (cc - 32) ** 2 <= 49, 0.9, 0.0).astype(np.float32)
    grid = MaterialGrid(risk, np.zeros((64, 64), bool), story.bounds, (0, 0), story.scale)

    def trace():
        torch.manual_seed(0)
        s = Swarm(story, specs, _model(), seed=0, truth_occ=np.zeros((64, 64), bool), material=grid)
        out = []
        for _ in range(120):
            s.step()
            out.append(s.o.numpy().copy())
        return np.stack(out)

    a, b = trace(), trace()
    assert np.array_equal(a, b)
