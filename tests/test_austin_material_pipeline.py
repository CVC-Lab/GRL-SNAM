"""End to end on a synthetic world: routes -> train -> ``.cvcnm`` -> four-arm evaluation."""

import json

import numpy as np
import torch

from grl_snam.belief_material_controller import BeliefMaterialContract, BeliefMaterialController
from grl_snam.material_evaluation import MATERIAL_EVALUATION_ARMS
from grl_snam.material_routes import MaterialRouteGenerationParams, generate_material_route_manifest
from grl_snam.tools.austin_material_evaluation import (
    AustinMaterialEvaluationParams,
    AustinMaterialEvaluationScene,
    evaluate_austin_material_model,
)
from grl_snam.tools.austin_material_training import (
    AustinMaterialTrainingParams,
    train_austin_material_model,
)
from grl_snam.train_coef_energy import CoefEnergyNet


def _scene():
    n = 61
    occupancy = np.zeros((n, n), bool)
    occupancy[8:11, 22:38] = True
    risk = np.zeros((n, n), np.float32)
    risk[26:34, 8:52] = 0.30
    hard = np.zeros((n, n), bool)
    hard[42:47, 15:45] = True
    risk[hard] = 1.0
    return AustinMaterialEvaluationScene(occupancy, risk, hard, (0.0, 0.0, 120.0, 120.0))


def test_train_then_evaluate_every_arm(tmp_path):
    scene = _scene()
    manifest = generate_material_route_manifest(
        scene.occupancy,
        scene.risk,
        scene.hard,
        scene.bounds,
        counts={"train": 2, "validation": 1, "test": 2},
        hard_counts={"train": 1, "validation": 0, "test": 1},
        seed=3,
        params=MaterialRouteGenerationParams(
            min_distance_m=40.0,
            max_distance_m=110.0,
            min_endpoint_separation_m=6.0,
            min_split_corridor_separation_m=4.0,
            max_attempts_per_route=20_000,
        ),
    )
    torch.manual_seed(0)
    geometry = tmp_path / "geometry.pt"
    torch.save(CoefEnergyNet().state_dict(), geometry)
    params = AustinMaterialTrainingParams(updates=6, batch_size=8, samples_per_route=4)

    report = train_austin_material_model(
        scene,
        manifest,
        geometry_checkpoint=geometry,
        output_dir=tmp_path / "run",
        seed=1,
        device="cpu",
        params=params,
    )

    assert report["trainer"] == "MaterialTrainerTorch"
    assert report["checkpoint_contract"]["material"] == {
        "feature_frame": "robot_relative",
        "feature_scale_m": 20.0,
        "k_sharp": 5.0,
        "d_hat_sdf_m": 3.0,
        "material_gate_trigger": 0.30,
        "geometry_d_hat_m": 3.0,
    }
    lambdas = report["learned_lambdas"]
    assert lambdas["lambda_soft"]["fixed"] == params.fixed_lambda_soft
    assert abs(lambdas["lambda_hard"]["mean"] - params.fixed_lambda_hard) < 0.5
    assert json.loads((tmp_path / "run" / "training.json").read_text())["seed"] == 1

    controller = BeliefMaterialController.from_cvcnm(report["checkpoint"])
    assert controller.contract == BeliefMaterialContract()
    evaluation = evaluate_austin_material_model(
        controller,
        scene,
        manifest,
        seed=1,
        params=AustinMaterialEvaluationParams(
            max_steps=25, waypoint_risk_candidates=(0.5, 1.0), bootstrap_samples=50
        ),
    )

    assert set(evaluation["summary"]) == set(MATERIAL_EVALUATION_ARMS)
    assert len(evaluation["rows"]) == len(MATERIAL_EVALUATION_ARMS) * len(manifest["test"])
    assert evaluation["selected_waypoint_risk_weight"] in (0.5, 1.0)
    for row in evaluation["rows"]:
        assert row["steps"] >= 1
        assert np.isfinite(row["path_length_m"])
