from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grl_snam.belief_material import belief_material_runtime, material_patch_belief
from grl_snam.belief_material_training import (
    BeliefMaterialTrainingSample,
    build_belief_material_training_batch,
)
from grl_snam.material_nav import CoefEnergyNetMaterial
from grl_snam.material_train_native import BATCH_KEYS, pack_batch
from grl_snam.material_train_torch import MaterialTrainerTorch
from grl_snam.tools.matnet_export import BeliefHeader, write_matnet


def _runtime():
    shape = (7, 7)
    risk = np.arange(49, dtype=np.float32).reshape(shape) / np.float32(48.0)
    hard = np.zeros(shape, dtype=bool)
    hard[5, 3] = True
    seen = np.zeros(shape, dtype=bool)
    seen[1:6, 1:6] = True
    runtime = belief_material_runtime(
        risk,
        hard,
        seen,
        np.zeros(shape, dtype=bool),
        (0.0, 0.0, 6.0, 6.0),
        (3.0, 3.0),
        1.0,
        sigma=0.0,
    )
    return runtime, seen


def _sample(*, obstacles=1, gate_active=True):
    runtime, seen = _runtime()
    centers = np.array([[2.0 + i, 2.0] for i in range(obstacles)], dtype=np.float32)
    rollout_patch = np.zeros((6, 9, 9), dtype=np.float32)
    rollout_patch[0] = 0.25
    rollout_patch[1] = 5.0
    rollout_patch[2] = 0.1
    return BeliefMaterialTrainingSample(
        position=np.array([3.0, 3.0], dtype=np.float32),
        velocity=np.array([0.1, 0.0], dtype=np.float32),
        goal=np.array([5.0, 3.0], dtype=np.float32),
        obstacle_centers=centers,
        obstacle_radii=np.full(obstacles, 0.4, dtype=np.float32),
        obstacle_weights=np.arange(1, obstacles + 1, dtype=np.float32),
        runtime=runtime,
        seen=seen,
        rollout_patch=rollout_patch,
        robot_radius=0.25,
        d_hat=1.0,
        dt=0.1,
        horizon=2,
        target_position=np.array([3.2, 3.0], dtype=np.float32),
        target_velocity=np.array([0.2, 0.0], dtype=np.float32),
        target_gamma=4.0,
        gate_active=gate_active,
    )


def _model():
    torch.manual_seed(11)
    model = CoefEnergyNetMaterial(
        d_tok=8,
        d_risk=8,
        nhead=2,
        num_layers=1,
        patch_size=25,
        risk_channels=3,
    )
    model.patch_size = 25
    model.fuser.enable_nested_tensor = False
    model.fuser.use_nested_tensor = False
    return model


def test_training_batch_pads_obstacles_and_uses_belief_inputs():
    first = _sample(obstacles=1, gate_active=False)
    second = _sample(obstacles=2, gate_active=True)

    batch = build_belief_material_training_batch([first, second])

    assert set(batch) == {*BATCH_KEYS, "gate_active"}
    assert batch["obs_feats"].shape == (2, 2, 6)
    np.testing.assert_array_equal(batch["obs_mask"], [[1, 0], [1, 1]])
    np.testing.assert_array_equal(batch["C"][0, 1], [0.0, 0.0])
    assert batch["R"][0, 1] == 0.0
    np.testing.assert_array_equal(batch["gate_active"], [0, 1])
    np.testing.assert_array_equal(
        batch["risk_patch"][0],
        material_patch_belief(
            first.runtime.grid.risk,
            first.runtime.grid.hard,
            first.seen,
            first.runtime.grid.bounds,
            first.position,
            100.0,
            25,
        ),
    )
    assert batch["risk_patch"].shape == (2, 3, 25, 25)
    assert batch["rollout_patch"].shape == (2, 6, 9, 9)


def test_training_batch_matches_native_dtype_and_contiguity_contract():
    batch = build_belief_material_training_batch([_sample()])

    packed = pack_batch(batch)

    for name, value in zip(BATCH_KEYS, packed):
        expected = np.uint8 if name == "obs_mask" else np.int32 if name == "H" else np.float32
        assert value.dtype == expected
        assert value.flags.c_contiguous


def test_training_batch_runs_through_shared_torch_trainer():
    batch = build_belief_material_training_batch(
        [_sample(gate_active=False), _sample(obstacles=2, gate_active=True)]
    )
    trainer = MaterialTrainerTorch(_model(), w_multi=0.0, dtype=torch.float64)

    parts = trainer.loss_parts(batch)

    assert np.isfinite(list(parts.values())).all()


def test_training_batch_derives_deployment_gate_when_not_overridden(monkeypatch):
    sample = replace(_sample(), gate_active=None)
    calls = []

    def gate(position, goal):
        calls.append((tuple(position), tuple(goal)))
        return SimpleNamespace(active=False)

    monkeypatch.setattr(sample.runtime, "gate", gate)

    batch = build_belief_material_training_batch([sample])

    assert calls == [((3.0, 3.0), (5.0, 3.0))]
    np.testing.assert_array_equal(batch["gate_active"], [0])


def test_training_batch_rejects_mixed_rollout_patch_shapes():
    first = _sample()
    second = replace(_sample(), rollout_patch=np.zeros((6, 7, 7), dtype=np.float32))

    with pytest.raises(ValueError, match="same shape"):
        build_belief_material_training_batch([first, second])


def test_training_batch_runs_through_shared_native_trainer_when_available(tmp_path):
    from grl_snam import material_train_native as native

    if not (native.HAS_MATERIAL_TRAINER_GATE and native.HAS_MATERIAL_TRAINER_LOSS):
        pytest.skip("pycvc lacks the belief MaterialTrainer loss surface")

    checkpoint = tmp_path / "belief.cvcnm"
    write_matnet(
        _model(),
        str(checkpoint),
        belief=BeliefHeader(patch_extent_m=100.0),
    )
    trainer = native.MaterialTrainer(checkpoint, w_multi=0.0)
    batch = build_belief_material_training_batch([_sample(), _sample(obstacles=2)])

    assert trainer.risk_channels == 3
    assert np.isfinite(trainer.loss(batch))
