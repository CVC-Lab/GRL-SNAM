import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from grl_snam.belief_material import (
    BeliefMaterialParams,
    belief_material_runtime,
    material_patch_belief,
)
from grl_snam.belief_material_controller import (
    BeliefMaterialCoefficients,
    BeliefMaterialControlCommand,
    BeliefMaterialController,
    BeliefMaterialRollout,
    BeliefMaterialSensor,
    belief_material_ray_count,
    build_belief_material_controller_inputs,
)
from grl_snam.material import GateResult
from grl_snam.material_nav import CoefEnergyNetMaterial
from grl_snam.tools.matnet_export import BeliefHeader, write_matnet


def _runtime():
    risk = np.arange(25, dtype=np.float32).reshape(5, 5) / np.float32(24.0)
    hard = np.zeros((5, 5), dtype=bool)
    hard[3, 2] = True
    seen = np.zeros((5, 5), dtype=bool)
    seen[1:4, 1:4] = True
    runtime = belief_material_runtime(
        risk,
        hard,
        seen,
        np.zeros((5, 5), dtype=bool),
        (0.0, 0.0, 4.0, 4.0),
        (2.0, 2.0),
        1.0,
        sigma=0.0,
    )
    return runtime, seen


def _model(risk_channels=3, patch_size=25):
    torch.manual_seed(7)
    model = CoefEnergyNetMaterial(
        d_tok=8,
        d_risk=8,
        nhead=2,
        num_layers=1,
        patch_size=patch_size,
        risk_channels=risk_channels,
    )
    model.patch_size = patch_size
    model.fuser.enable_nested_tensor = False
    model.fuser.use_nested_tensor = False
    return model


def test_builds_training_compatible_geometry_and_belief_patch():
    runtime, seen = _runtime()
    position = np.array([1.0, 2.0], dtype=np.float32)
    goal = np.array([4.0, 6.0], dtype=np.float32)
    centers = np.array([[3.0, 5.0], [7.0, 11.0]], dtype=np.float32)

    obs_feats, obs_mask, goal_feats, risk_patch = build_belief_material_controller_inputs(
        position,
        goal,
        centers,
        np.array([0.5, 1.5], dtype=np.float32),
        np.array([2.0, 4.0], dtype=np.float32),
        runtime,
        seen,
        patch_extent_m=4.0,
        patch_size=5,
        feature_scale_m=1.0,
    )

    np.testing.assert_array_equal(
        obs_feats,
        np.array(
            [[2.0, 3.0, 0.5, 2.0, 1.0, 1.0], [6.0, 9.0, 1.5, 4.0, -3.0, -5.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(obs_mask, np.array([True, True]))
    np.testing.assert_array_equal(goal_feats, np.array([3.0, 4.0, 5.0, 1.0], np.float32))
    np.testing.assert_array_equal(
        risk_patch,
        material_patch_belief(
            runtime.grid.risk,
            runtime.grid.hard,
            seen,
            runtime.grid.bounds,
            position,
            4.0,
            5,
        ),
    )
    assert obs_feats.dtype == goal_feats.dtype == risk_patch.dtype == np.float32
    assert obs_mask.dtype == bool


def test_geometry_features_are_robot_relative_and_scaled_like_coefenergy():
    runtime, seen = _runtime()

    obs_feats, _obs_mask, goal_feats, _risk_patch = build_belief_material_controller_inputs(
        (10.0, 20.0),
        (30.0, 60.0),
        [[14.0, 26.0]],
        [2.0],
        [1.5],
        runtime,
        seen,
        feature_scale_m=20.0,
    )

    np.testing.assert_allclose(
        obs_feats,
        np.array([[0.2, 0.3, 0.1, 1.5, 0.8, 1.7]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        goal_feats,
        np.array([1.0, 2.0, math.sqrt(5.0), 1.0], dtype=np.float32),
    )


def test_geometry_feature_scale_must_be_positive():
    runtime, seen = _runtime()
    with pytest.raises(ValueError, match="feature_scale_m"):
        build_belief_material_controller_inputs(
            (2.0, 2.0), (2.0, 3.0), [], [], [], runtime, seen, feature_scale_m=0.0
        )


def test_empty_obstacles_keep_model_ready_shapes_and_default_patch_size():
    runtime, seen = _runtime()
    obs_feats, obs_mask, goal_feats, risk_patch = build_belief_material_controller_inputs(
        (2.0, 2.0),
        (2.0, 3.0),
        [],
        [],
        [],
        runtime,
        seen,
    )

    assert obs_feats.shape == (0, 6)
    assert obs_mask.shape == (0,)
    np.testing.assert_array_equal(goal_feats, np.array([0.0, 0.05, 0.05, 1.0], np.float32))
    assert risk_patch.shape == (3, 25, 25)


def test_rejects_obstacle_and_seen_shape_mismatches():
    runtime, seen = _runtime()
    with pytest.raises(ValueError, match="obstacle_radii"):
        build_belief_material_controller_inputs(
            (0.0, 0.0), (1.0, 1.0), [[2.0, 2.0]], [], [1.0], runtime, seen
        )
    with pytest.raises(ValueError, match="seen"):
        build_belief_material_controller_inputs(
            (0.0, 0.0),
            (1.0, 1.0),
            [],
            [],
            [],
            runtime,
            np.zeros((2, 2), dtype=bool),
        )


def test_controller_matches_direct_three_channel_model_inference():
    runtime, seen = _runtime()
    inputs = build_belief_material_controller_inputs(
        (2.0, 2.0),
        (3.0, 4.0),
        [[1.0, 1.0], [3.0, 2.0]],
        [0.25, 0.5],
        [1.5, 2.0],
        runtime,
        seen,
    )
    model = _model()
    controller = BeliefMaterialController(model)

    got = controller.predict(*inputs)

    obs_feats, obs_mask, goal_feats, risk_patch = inputs
    with torch.inference_mode():
        expected = model(
            torch.as_tensor(obs_feats).unsqueeze(0),
            torch.as_tensor(obs_mask).unsqueeze(0),
            torch.as_tensor(goal_feats).unsqueeze(0),
            torch.as_tensor(risk_patch).unsqueeze(0),
        )
    np.testing.assert_allclose(got.alphas, expected[0][0].numpy())
    for value, tensor in zip(
        (got.beta, got.gamma, got.lam_soft, got.lam_hard, got.mu_lat), expected[1:]
    ):
        assert value == pytest.approx(float(tensor[0]))
    assert not model.training


def test_controller_handles_zero_obstacles():
    runtime, seen = _runtime()
    inputs = build_belief_material_controller_inputs(
        (2.0, 2.0), (2.0, 3.0), [], [], [], runtime, seen
    )

    got = BeliefMaterialController(_model()).predict(*inputs)

    assert got.alphas.shape == (0,)
    assert np.isfinite([got.beta, got.gamma, got.lam_soft, got.lam_hard, got.mu_lat]).all()


def test_controller_requires_three_channels_and_expected_patch_shape():
    with pytest.raises(ValueError, match="3-channel"):
        BeliefMaterialController(_model(risk_channels=2))

    runtime, seen = _runtime()
    inputs = list(
        build_belief_material_controller_inputs((2.0, 2.0), (2.0, 3.0), [], [], [], runtime, seen)
    )
    inputs[-1] = np.zeros((3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="risk_patch"):
        BeliefMaterialController(_model()).predict(*inputs)


def test_controller_loads_validated_belief_cvcnm_v2(tmp_path):
    checkpoint = tmp_path / "belief.cvcnm"
    header = BeliefHeader(patch_extent_m=100.0)
    write_matnet(_model(), str(checkpoint), belief=header)

    controller = BeliefMaterialController.from_cvcnm(checkpoint)

    assert controller.model.risk_channels == 3
    assert controller.model.belief_header == header
    assert controller.patch_size == 25
    assert controller.patch_extent_m == 100.0
    assert controller.belief == BeliefMaterialParams()
    assert not controller.model.training


def test_controller_refuses_non_belief_cvcnm_v1(tmp_path):
    checkpoint = tmp_path / "world-truth.cvcnm"
    write_matnet(_model(risk_channels=2), str(checkpoint))

    with pytest.raises(ValueError, match="3-channel .cvcnm version 2"):
        BeliefMaterialController.from_cvcnm(checkpoint)


@pytest.mark.parametrize(
    ("header", "field"),
    [
        (replace(BeliefHeader(), seen_rule=1, patch_extent_m=100.0), "seen_rule"),
        (replace(BeliefHeader(), unknown_risk=0.25, patch_extent_m=100.0), "unknown_risk"),
        (replace(BeliefHeader(), unknown_hard=1, patch_extent_m=100.0), "unknown_hard"),
        (
            replace(BeliefHeader(), unknown_gate_hard=0, patch_extent_m=100.0),
            "unknown_gate_hard",
        ),
        (
            replace(BeliefHeader(), material_range_m=60.0, patch_extent_m=100.0),
            "material_range_m",
        ),
        (replace(BeliefHeader(), patch_extent_m=80.0), "patch_extent_m"),
    ],
)
def test_controller_refuses_incompatible_belief_header(tmp_path, header, field):
    checkpoint = tmp_path / f"bad-{field}.cvcnm"
    write_matnet(_model(), str(checkpoint), belief=header)

    with pytest.raises(ValueError, match=field):
        BeliefMaterialController.from_cvcnm(checkpoint)


def test_controller_refuses_checkpoint_patch_size_mismatch(tmp_path):
    checkpoint = tmp_path / "wrong-patch-size.cvcnm"
    write_matnet(
        _model(patch_size=16),
        str(checkpoint),
        belief=BeliefHeader(patch_extent_m=100.0),
    )

    with pytest.raises(ValueError, match="patch_size=16"):
        BeliefMaterialController.from_cvcnm(checkpoint)


@pytest.mark.parametrize("gate_active", [False, True])
def test_controller_decision_gates_only_soft_coefficient(monkeypatch, gate_active):
    runtime, seen = _runtime()
    gate = GateResult(
        active=gate_active,
        nominal_risk=0.8,
        best_risk=0.2,
        feasible_count=1,
        direction_rc=(0.0, 1.0),
        endpoint_rc=(2.0, 3.0),
        min_clearance_m=2.0,
    )
    monkeypatch.setattr(runtime, "gate", lambda *_: gate)
    planes_before = [
        plane.copy()
        for plane in (
            runtime.grid.risk,
            runtime.grid.phi_hard_m,
            runtime.grid.grad_rx,
            runtime.grid.grad_ry,
            runtime.grid.grad_px,
            runtime.grid.grad_py,
        )
    ]

    command = BeliefMaterialController(_model()).decide(
        (2.0, 2.0),
        (3.0, 4.0),
        [[1.0, 1.0], [3.0, 2.0]],
        [0.25, 0.5],
        [1.5, 2.0],
        runtime,
        seen,
    )

    assert isinstance(command, BeliefMaterialControlCommand)
    assert command.gate is gate
    assert command.effective.lam_soft == (command.predicted.lam_soft if gate_active else 0.0)
    assert command.effective.lam_hard == command.predicted.lam_hard
    np.testing.assert_array_equal(command.effective.alphas, command.predicted.alphas)
    assert command.effective.beta == command.predicted.beta
    assert command.effective.gamma == command.predicted.gamma
    assert command.effective.mu_lat == command.predicted.mu_lat
    for before, after in zip(
        planes_before,
        (
            runtime.grid.risk,
            runtime.grid.phi_hard_m,
            runtime.grid.grad_rx,
            runtime.grid.grad_ry,
            runtime.grid.grad_px,
            runtime.grid.grad_py,
        ),
    ):
        np.testing.assert_array_equal(after, before)


def _command(*, beta=1.0, lam_soft=1.0):
    coefficients = BeliefMaterialCoefficients(
        alphas=np.empty((0,), dtype=np.float32),
        beta=beta,
        gamma=0.0,
        lam_soft=lam_soft,
        lam_hard=0.0,
        mu_lat=0.0,
    )
    return BeliefMaterialControlCommand(
        predicted=coefficients,
        effective=coefficients,
        gate=None,
    )


def _rollout(controller, runtime, command):
    patch = np.zeros((6, 5, 5), dtype=np.float32)
    patch[1] = 50.0
    patch[3] = 1.0
    return controller.rollout(
        command,
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        goal=(1.0, 0.0),
        obstacle_centers=[],
        obstacle_radii=[],
        rollout_patch=patch,
        runtime=runtime,
        d_hat=0.5,
        dt=0.1,
        steps=1,
    )


def test_rollout_uses_shared_geometry_and_material_dynamics():
    runtime, _ = _runtime()
    controller = BeliefMaterialController(_model())

    full = _rollout(controller, runtime, _command())
    no_geometry = _rollout(controller, runtime, _command(beta=0.0))
    no_material = _rollout(controller, runtime, _command(lam_soft=0.0))

    assert isinstance(full, BeliefMaterialRollout)
    assert full.position[0] > no_geometry.position[0]
    assert full.position[1] < no_material.position[1]
    assert full.arc_length > 0.0


def test_rollout_reads_material_barrier_constants_from_runtime(monkeypatch):
    import grl_snam.belief_material_controller as controller_module

    runtime, _ = _runtime()
    runtime.params.k_sharp = 2.25
    runtime.params.d_hat_sdf_m = 4.5
    captured = {}
    original = controller_module.integrate_surrogate_material

    def record_constants(*args, **kwargs):
        captured["k_sharp"] = kwargs["k_sharp"]
        captured["d_hat_sdf"] = kwargs["d_hat_sdf"]
        return original(*args, **kwargs)

    monkeypatch.setattr(controller_module, "integrate_surrogate_material", record_constants)
    _rollout(BeliefMaterialController(_model()), runtime, _command())

    assert captured == {"k_sharp": 2.25, "d_hat_sdf": 4.5}


def test_live_material_kwargs_use_effective_coefficients_and_existing_field():
    runtime, _ = _runtime()
    runtime.params.k_sharp = 2.25
    runtime.params.d_hat_sdf_m = 4.5
    controller = BeliefMaterialController(_model())
    predicted = BeliefMaterialCoefficients(
        alphas=np.array([3.0], dtype=np.float32),
        beta=4.0,
        gamma=5.0,
        lam_soft=6.0,
        lam_hard=7.0,
        mu_lat=8.0,
    )
    effective = BeliefMaterialCoefficients(
        alphas=predicted.alphas,
        beta=predicted.beta,
        gamma=predicted.gamma,
        lam_soft=0.0,
        lam_hard=predicted.lam_hard,
        mu_lat=predicted.mu_lat,
    )

    kwargs = controller.live_material_kwargs(
        BeliefMaterialControlCommand(predicted=predicted, effective=effective, gate=None),
        runtime,
    )

    assert set(kwargs) == {
        "material",
        "lam_soft",
        "lam_hard",
        "mat_k_sharp",
        "mat_d_hat_m",
    }
    assert kwargs["material"] is runtime.field
    torch.testing.assert_close(kwargs["lam_soft"], torch.tensor([0.0]))
    torch.testing.assert_close(kwargs["lam_hard"], torch.tensor([7.0]))
    assert kwargs["mat_k_sharp"] == 2.25
    assert kwargs["mat_d_hat_m"] == 4.5


def test_live_material_kwargs_run_through_shared_sdf_rollout():
    from grl_snam import sdf_nav

    class OpenField:
        @staticmethod
        def sample(position):
            batch = position.shape[0]
            return (
                torch.full((batch,), 50.0, device=position.device),
                torch.zeros((batch, 2), device=position.device),
            )

    runtime, _ = _runtime()
    controller = BeliefMaterialController(_model())

    def run(lam_soft):
        command = _command(beta=0.0, lam_soft=lam_soft)
        position = torch.zeros((1, 2), dtype=torch.float32)
        position_out, _, _ = sdf_nav.sdf_rollout(
            OpenField(),
            position,
            torch.zeros_like(position),
            position.clone(),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            1,
            rr=0.0,
            d_hat=0.5,
            dt=0.1,
            **controller.live_material_kwargs(command, runtime),
        )
        return position_out

    torch.testing.assert_close(run(0.0), torch.zeros((1, 2)))
    assert torch.linalg.vector_norm(run(1.0)).item() > 0.0


def test_material_ray_count_is_raised_to_grid_coverage():
    assert belief_material_ray_count(2.0 * math.pi, 60.0, 2.0) == 189
    assert belief_material_ray_count(math.pi, 0.0, 2.0) == 1
    with pytest.raises(ValueError, match="cell_size_m"):
        belief_material_ray_count(math.pi, 60.0, 0.0)


def test_sensor_rebuilds_only_when_seen_grows_and_uses_belief_planes():
    from grl_snam.belief import BeliefGrid

    shape = (9, 9)
    bounds = (0.0, 0.0, 8.0, 8.0)
    truth = np.zeros(shape, dtype=bool)
    truth[:, 7] = True
    risk = np.arange(81, dtype=np.float32).reshape(shape) / np.float32(80.0)
    hard = np.zeros(shape, dtype=bool)
    hard[4, 6] = True
    belief = BeliefGrid(shape, bounds)
    sensor = BeliefMaterialSensor(
        belief,
        truth,
        risk,
        hard,
        center=(4.0, 4.0),
        scale=1.0,
        sigma=0.0,
    )
    initial_runtime = sensor.runtime

    first = sensor.sense((4.0, 4.0), range_m=3.0)

    assert first.n_rays == math.ceil(2.0 * math.pi * 3.0)
    assert first.material_grew and first.runtime_rebuilt and first.gate_updated
    assert sensor.runtime is not initial_runtime
    seen = belief.material_seen
    assert seen is not None
    np.testing.assert_array_equal(sensor.runtime.grid.risk_raw, np.where(seen, risk, 0.0))
    np.testing.assert_array_equal(sensor.runtime.grid.hard, np.where(seen, hard, False))

    rebuilt_runtime = sensor.runtime
    second = sensor.sense((4.0, 4.0), range_m=3.0)
    assert not second.material_grew
    assert not second.runtime_rebuilt
    assert not second.gate_updated
    assert sensor.runtime is rebuilt_runtime


def test_sensor_updates_gate_without_rebuilding_material_planes():
    from grl_snam.belief import BeliefGrid

    shape = (9, 9)
    sensor = BeliefMaterialSensor(
        BeliefGrid(shape, (0.0, 0.0, 8.0, 8.0)),
        np.zeros(shape, dtype=bool),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=bool),
        center=(4.0, 4.0),
        scale=1.0,
        sigma=0.0,
    )
    sensor.sense((4.0, 4.0), range_m=3.0)
    runtime = sensor.runtime
    dynamic = np.zeros(shape, dtype=bool)
    dynamic[4, 5] = True

    update = sensor.sense(
        (4.0, 4.0),
        range_m=3.0,
        dynamic_occupancy=dynamic,
    )

    assert not update.material_grew
    assert not update.runtime_rebuilt
    assert update.gate_updated
    assert sensor.runtime is runtime
