import json

import imageio.v3 as iio
import numpy as np

from grl_snam.material_evaluation import MaterialRouteMetrics
from grl_snam.tools.austin_material_evaluation import (
    AustinMaterialEvaluationParams,
    evaluate_austin_material_model,
    load_austin_material_evaluation_scene,
)


def _rgba_mask(rows, cols, cells):
    mask = np.zeros((rows, cols, 4), dtype=np.uint8)
    for row, col in cells:
        mask[row, col] = (1, 1, 1, 255)
    return mask


def _bundle(tmp_path):
    (tmp_path / "terrain.json").write_text(
        json.dumps(
            {
                "rows": 2,
                "cols": 3,
                "bounds": {"min_x": 0, "min_y": 0, "max_x": 3, "max_y": 2},
            }
        )
    )
    navmask = np.zeros((6, 9), np.uint8)
    navmask[0, 0] = 255
    iio.imwrite(tmp_path / "navmask.png", navmask)
    iio.imwrite(tmp_path / "foliage_mask.png", np.array([[1, 0, 0], [0, 0, 0]], np.uint8))
    iio.imwrite(tmp_path / "open_fields.png", _rgba_mask(2, 3, []))
    iio.imwrite(tmp_path / "water.png", _rgba_mask(2, 3, [(1, 2)]))
    iio.imwrite(tmp_path / "roads.png", _rgba_mask(2, 3, []))


def test_evaluation_scene_uses_audited_route_grid(tmp_path):
    _bundle(tmp_path)

    scene = load_austin_material_evaluation_scene(tmp_path)

    assert scene.occupancy.shape == scene.risk.shape == scene.hard.shape == (2, 3)
    assert scene.occupancy[1, 0]
    assert scene.risk[1, 0] == 0.30
    assert scene.hard[0, 2]


def test_evaluator_selects_on_validation_then_runs_every_test_arm():
    manifest = {
        "schema": "grl-snam-material-routes/1",
        "train": [],
        "validation": [
            {
                "id": 1,
                "start_xy": [0.0, 0.0],
                "goal_xy": [1.0, 0.0],
                "direct_distance_m": 1.0,
                "challenge_kind": "soft",
                "material_challenge": "foliage",
                "split_locked_before_training": True,
            }
        ],
        "test": [
            {
                "id": 2,
                "start_xy": [0.0, 0.0],
                "goal_xy": [1.0, 0.0],
                "direct_distance_m": 1.0,
                "challenge_kind": "hard",
                "material_challenge": "water",
                "split_locked_before_training": True,
            }
        ],
        "protocol": {
            "seed": 1,
            "test_locked_before_training": True,
            "counts": {"train": 0, "validation": 1, "test": 1},
            "hard_counts": {"train": 0, "validation": 0, "test": 1},
            "minimum_endpoint_separation_m": 0.0,
        },
    }
    calls = []

    def route_runner(_controller, _scene, route, *, arm, split, seed, waypoint_risk_weight, params):
        calls.append((split, arm, waypoint_risk_weight))
        risk = abs(waypoint_risk_weight - 1.0)
        return MaterialRouteMetrics(
            seed=seed,
            route_id=route["id"],
            split=split,
            challenge_kind=route["challenge_kind"],
            arm=arm,
            success=True,
            collision=False,
            timeout=False,
            direct_distance_m=1.0,
            path_length_m=1.0,
            risk_distance=risk,
            hard_hazard_distance_m=0.0,
            steps=1,
            hard_hazard_steps=0,
            gate_active_steps=0,
            prediction_latency_ms=0.1,
        )

    report = evaluate_austin_material_model(
        None,
        None,
        manifest,
        seed=7,
        params=AustinMaterialEvaluationParams(
            waypoint_risk_candidates=(0.5, 1.0), bootstrap_samples=20
        ),
        route_runner=route_runner,
    )

    assert report["selected_waypoint_risk_weight"] == 1.0
    assert report["params"]["material_gate_trigger"] == 0.30
    assert calls[:2] == [("validation", "learned", 0.5), ("validation", "learned", 1.0)]
    assert {call[1] for call in calls[2:]} == {
        "fixed_lambda",
        "force_only",
        "waypoint",
        "learned",
    }
