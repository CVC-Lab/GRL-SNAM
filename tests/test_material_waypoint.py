import math

import numpy as np
import pytest

from grl_snam.material import MaterialGrid, MaterialRuntime
from grl_snam.material_waypoint import MaterialWaypointParams, choose_material_local_waypoint


def _runtime(hard_cells=(), risk_cells=(), shape=(21, 21), bounds=(0.0, 0.0, 20.0, 20.0)):
    """A 1 m/cell raster: unit risk planes, no smoothing, nothing occupied."""
    risk = np.zeros(shape, np.float32)
    for (row, col), value in risk_cells:
        risk[row, col] = value
    hard = np.zeros(shape, bool)
    for row, col in hard_cells:
        hard[row, col] = True
    risk[hard] = 1.0
    runtime = MaterialRuntime(MaterialGrid(risk, hard, bounds, (0.0, 0.0), 1.0, sigma=0.0))
    runtime.update_occ(np.zeros(shape, bool))
    return runtime


def _no_material(ux, uy, distance):
    return 0.0, None


def test_ray_profile_stops_at_the_first_hard_cell_and_counts_its_risk():
    runtime = _runtime(hard_cells=[(10, 15)], risk_cells=[((10, 12), 0.5)])
    mean, stop = runtime.ray_profile((10.0, 10.0), 1.0, 0.0, 9.0)
    assert stop == pytest.approx(5.0)
    # Cells 11..15 along the ray: 0, 0.5, 0, 0, 1.0 (the blocking cell is included).
    assert mean == pytest.approx(1.5 / 5.0)


def test_ray_profile_is_feasible_past_a_hazard_beside_the_ray_unless_clearance_is_asked_for():
    runtime = _runtime(hard_cells=[(11, 14)])  # one cell off the ray
    assert runtime.ray_profile((10.0, 10.0), 1.0, 0.0, 8.0) == (0.0, None)
    _mean, stop = runtime.ray_profile((10.0, 10.0), 1.0, 0.0, 8.0, clearance_m=1.5)
    assert stop is not None and stop <= 4.0


def test_ray_profile_steps_one_cell_on_an_anisotropic_raster():
    # 2 m per column, 1 m per row: a hard cell 3 columns east is 6 m away.
    runtime = _runtime(hard_cells=[(10, 13)], shape=(21, 21), bounds=(0.0, 0.0, 40.0, 20.0))
    _mean, stop = runtime.ray_profile((20.0, 10.0), 1.0, 0.0, 15.0)
    assert stop == pytest.approx(6.0)


def test_ray_profile_stops_at_the_raster_edge():
    runtime = _runtime()
    _mean, stop = runtime.ray_profile((18.0, 10.0), 1.0, 0.0, 10.0)
    assert stop is not None and stop <= 3.0


def test_waypoint_returns_the_goal_when_it_is_closer_than_a_usable_ray():
    waypoint, info = choose_material_local_waypoint(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.5, 0.0),
        np.full(8, 60.0),
        sensor_radius_m=60.0,
        material_profile=_no_material,
    )
    assert waypoint == (0.5, 0.0)
    assert info["usable_m"] == pytest.approx(0.5)


def test_waypoint_prefers_the_low_risk_ray_when_material_weight_is_on():
    # 8 rays; ray 0 (+x) points at the goal but crosses risk 0.9; rays 1 and 7 are clean.
    def profile(ux, uy, distance):
        return (0.9, None) if ux > 0.99 else (0.0, None)

    for weight, expect_straight in ((0.0, True), (2.0, False)):
        waypoint, _info = choose_material_local_waypoint(
            (0.0, 0.0),
            (0.0, 0.0),
            (100.0, 0.0),
            np.full(8, 60.0),
            sensor_radius_m=60.0,
            material_profile=profile,
            params=MaterialWaypointParams(risk_weight=weight),
        )
        straight = abs(waypoint[1]) < 1e-9
        assert straight == expect_straight


def test_waypoint_stops_short_of_material_and_reports_it():
    def profile(ux, uy, distance):
        return 0.2, 10.0  # every ray meets a hazard 10 m out

    waypoint, info = choose_material_local_waypoint(
        (0.0, 0.0),
        (1.0, 0.0),
        (100.0, 0.0),
        np.full(8, 60.0),
        sensor_radius_m=60.0,
        material_profile=profile,
    )
    assert info["material_limited"] is True
    assert math.hypot(*waypoint) == pytest.approx(10.0 - MaterialWaypointParams().wall_margin_m)


def test_waypoint_fallback_creeps_toward_the_goal_and_flags_material_blocking():
    def profile(ux, uy, distance):
        return 1.0, 0.0  # material blocks every ray at the start

    waypoint, info = choose_material_local_waypoint(
        (0.0, 0.0),
        (0.0, 0.0),
        (10.0, 0.0),
        np.full(8, 60.0),
        sensor_radius_m=60.0,
        material_profile=profile,
    )
    assert waypoint == pytest.approx((MaterialWaypointParams().min_usable_m, 0.0))
    assert info["material_limited"] is True
