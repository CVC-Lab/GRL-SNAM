"""The sim-frame witness gate — the NORMATIVE bit-spec the C++ twin copies.

Everything here is exact-comparison: the float64 sequential-accumulation
serial gate, the vectorized batch gate (must be byte-identical to N serial
calls), the shared direction table, half-even cell rounding, and the
occupancy-in-feasibility rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grl_snam.material import (
    _DIRS_16,
    GateResult,
    gate_directions,
    witness_gate,
    witness_gate_batch,
)

KW = dict(
    horizon_cells=12,
    hard_margin_m=1.0,
    primitive_count=16,
    improvement_margin=0.05,
    material_trigger=0.45,
    progress_slack_cells=0.5,
)


def _open(h=40, w=40, risk=0.0, clear=10.0):
    return (
        np.full((h, w), risk, np.float32),
        np.zeros((h, w), bool),
        np.full((h, w), clear, np.float32),
    )


def test_direction_table_is_exact_and_matches_math():
    assert len(_DIRS_16) == 16
    assert _DIRS_16[0] == (0.0, 1.0)
    for k, (dr, dc) in enumerate(_DIRS_16):
        a = 2.0 * math.pi * k / 16.0
        assert dr == math.sin(a) and dc == math.cos(a)  # exact, this host
        assert math.hypot(dr, dc) == pytest.approx(1.0, abs=1e-15)
    assert gate_directions(16) is _DIRS_16
    assert len(gate_directions(8)) == 8


def test_gate_activates_on_cheaper_corridor_and_reports_fields():
    risk, hard, clear = _open()
    risk[:, :] = 0.05
    risk[15:26, 22:40] = 0.9  # band across the nominal ray
    res = witness_gate(risk, hard, clear, (20.0, 20.0), (20.0, 35.0), **KW)
    assert isinstance(res, GateResult)
    assert res.active
    assert res.feasible_count > 0
    assert res.nominal_risk >= 0.45
    assert res.nominal_risk - res.best_risk >= 0.05
    assert math.hypot(*res.direction_rc) == pytest.approx(1.0, abs=1e-12)


def test_gate_uniform_risk_and_low_trigger_stay_off():
    risk, hard, clear = _open(risk=0.9)
    res = witness_gate(risk, hard, clear, (20.0, 20.0), (20.0, 35.0), **KW)
    assert not res.active and res.feasible_count > 0
    risk2, hard2, clear2 = _open(risk=0.1)
    risk2[18:23, 24:40] = 0.3
    res2 = witness_gate(risk2, hard2, clear2, (20.0, 20.0), (20.0, 35.0), **KW)
    assert not res2.active and res2.nominal_risk < 0.45


def test_occupancy_participates_in_feasibility():
    """A ray through a BUILDING is not a detour: gate_hard includes occupancy,
    so walling the low-risk corridor off with occ must kill the activation."""
    risk, hard, clear = _open()
    risk[:, :] = 0.05
    risk[15:26, 22:40] = 0.9
    res_open = witness_gate(risk, hard, clear, (20.0, 20.0), (20.0, 35.0), **KW)
    assert res_open.active
    # wall off everything except the risky band — as the caller's gate_hard
    # (material.hard | occ) and its clearance plane would encode it
    gate_hard = np.ones_like(hard)
    gate_hard[15:26, :] = False
    clear_blocked = np.where(gate_hard, 0.0, 10.0).astype(np.float32)
    res_blocked = witness_gate(risk, gate_hard, clear_blocked, (20.0, 20.0), (20.0, 35.0), **KW)
    # The corridor rays are gone; only in-band rays (which share the nominal
    # ray's risk) survive, so no ray improves on nominal and the gate is off.
    assert not res_blocked.active
    assert res_blocked.best_risk >= res_blocked.nominal_risk - KW["improvement_margin"]
    # ... and fully walling the grid kills feasibility outright.
    all_hard = np.ones_like(hard)
    all_clear = np.zeros_like(clear)
    res_all = witness_gate(risk, all_hard, all_clear, (20.0, 20.0), (20.0, 35.0), **KW)
    assert res_all.feasible_count == 0 and not res_all.active


def test_clearance_margin_blocks_infeasible_rays():
    risk, hard, clear = _open(risk=0.9)
    clear[:, :] = 0.5  # everywhere under the 1.0 m margin
    res = witness_gate(risk, hard, clear, (20.0, 20.0), (20.0, 35.0), **KW)
    assert res.feasible_count == 0 and not res.active
    assert res.best_risk == math.inf


def test_risk_recorded_before_feasibility_break():
    """The source appends the tripping cell's risk BEFORE breaking — the mean
    includes it. Order-sensitive; pinned here for the C++ twin."""
    risk, hard, clear = _open(risk=0.0)
    risk[20, 22] = 0.9  # second sample along the +col ray from (20, 20)
    hard[20, 22] = True  # ... which is also the tripping cell
    from grl_snam.material import _ray_risk

    mean, feasible, _ = _ray_risk(risk, hard, clear, 20.0, 20.0, 0.0, 1.0, 12, 1.0)
    assert not feasible
    assert mean == pytest.approx((0.0 + 0.9) / 2.0)  # two samples, then break


def test_oob_breaks_before_sampling():
    risk, hard, clear = _open(risk=0.9)
    from grl_snam.material import _ray_risk

    # from row 1 walking -row: t=1 hits row 0 (sampled), t=2 is -1 (OOB, float
    # test before rounding) -> infeasible with exactly one sample
    mean, feasible, _ = _ray_risk(risk, hard, clear, 1.0, 5.0, -1.0, 0.0, 12, 1.0)
    assert not feasible
    assert mean == pytest.approx(0.9)


def test_progress_filter_goal_one_cell_away():
    risk, hard, clear = _open(risk=0.9)
    res = witness_gate(risk, hard, clear, (20.0, 20.0), (20.0, 21.0), **KW)
    assert res.feasible_count == 0 and not res.active


def test_cell_rounding_is_half_even_on_the_axis_ray():
    """Direction (0, 1) is exactly representable, so samples from col 4.5 land
    on 5.5, 6.5, ... — round-half-even = 6, 6, 8, 8, ..."""
    risk, hard, clear = _open(h=20, w=30)
    risk[10, :] = 0.0
    risk[10, 6] = 0.6
    risk[10, 8] = 0.2
    from grl_snam.material import _ray_risk

    mean, feasible, _ = _ray_risk(risk, hard, clear, 10.0, 4.5, 0.0, 1.0, 4, 1.0)
    # samples at cols 5.5->6, 6.5->6, 7.5->8, 8.5->8: (0.6+0.6+0.2+0.2)/4
    assert feasible
    assert mean == pytest.approx(0.4)


def test_gate_zero_goal_distance_degenerates_safely():
    risk, hard, clear = _open(risk=0.9)
    res = witness_gate(risk, hard, clear, (20.0, 20.0), (20.0, 20.0), **KW)
    assert not res.active
    assert res.feasible_count == 0  # no endpoint can beat a zero goal distance


def test_batch_gate_is_byte_identical_to_serial():
    rng = np.random.default_rng(42)
    for _ in range(40):
        h, w = int(rng.integers(20, 70)), int(rng.integers(20, 70))
        risk = rng.random((h, w)).astype(np.float32)
        hard = rng.random((h, w)) < 0.05
        clear = (rng.random((h, w)) * 4).astype(np.float32)
        n = 17
        pos = np.stack([rng.uniform(0, h - 1, n), rng.uniform(0, w - 1, n)], axis=1)
        goal = np.stack([rng.uniform(0, h - 1, n), rng.uniform(0, w - 1, n)], axis=1)
        active, nominal, best, count = witness_gate_batch(risk, hard, clear, pos, goal, **KW)
        for i in range(n):
            ref = witness_gate(risk, hard, clear, tuple(pos[i]), tuple(goal[i]), **KW)
            assert bool(active[i]) == ref.active
            assert nominal[i] == ref.nominal_risk  # byte-equal f64
            assert best[i] == ref.best_risk
            assert int(count[i]) == ref.feasible_count


def test_batch_gate_handles_agents_on_and_off_grid_edges():
    risk, hard, clear = _open(risk=0.9)
    pos = np.array([[0.0, 0.0], [39.0, 39.0], [20.0, 20.0]])
    goal = np.array([[39.0, 39.0], [0.0, 0.0], [20.0, 35.0]])
    active, nominal, best, count = witness_gate_batch(risk, hard, clear, pos, goal, **KW)
    for i in range(3):
        ref = witness_gate(risk, hard, clear, tuple(pos[i]), tuple(goal[i]), **KW)
        assert (bool(active[i]), nominal[i], best[i], int(count[i])) == (
            ref.active,
            ref.nominal_risk,
            ref.best_risk,
            ref.feasible_count,
        )
