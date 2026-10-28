import math

import numpy as np

from grl_snam.tools.austin_material_evaluation import AustinMaterialEvaluationScene
from grl_snam.tools.austin_material_training import (
    AustinMaterialTrainingParams,
    build_austin_material_training_samples,
)


def _manifest():
    route = {
        "id": 1,
        "start_xy": [2.0, 10.0],
        "goal_xy": [18.0, 10.0],
        "direct_distance_m": 16.0,
        "challenge_kind": "soft",
        "material_challenge": "foliage",
        "split_locked_before_training": True,
    }
    return {
        "schema": "grl-snam-material-routes/1",
        "train": [route],
        "validation": [],
        "test": [],
        "protocol": {
            "seed": 1,
            "test_locked_before_training": True,
            "counts": {"train": 1, "validation": 0, "test": 0},
            "hard_counts": {"train": 0, "validation": 0, "test": 0},
            "minimum_endpoint_separation_m": 0.0,
        },
    }


def test_training_samples_use_only_train_routes_and_observed_material():
    occupancy = np.zeros((21, 21), dtype=bool)
    occupancy[7:14, 12] = True
    risk = np.zeros((21, 21), dtype=np.float32)
    risk[7:14, 8:12] = 0.7
    scene = AustinMaterialEvaluationScene(
        occupancy=occupancy,
        risk=risk,
        hard=np.zeros_like(occupancy),
        bounds=(0.0, 0.0, 20.0, 20.0),
    )

    samples, diagnostics = build_austin_material_training_samples(
        scene,
        _manifest(),
        params=AustinMaterialTrainingParams(
            sensor_range_m=6.0,
            samples_per_route=3,
            teacher_lookahead_m=5.0,
        ),
    )

    assert len(samples) == 3
    assert diagnostics["routes"] == 1
    assert diagnostics["challenge_counts"] == {"soft": 1, "hard": 0}
    assert diagnostics["mean_visible_fraction"] < 1.0
    assert all(sample.seen.dtype == np.bool_ for sample in samples)
    assert all(sample.rollout_patch.shape == (6, 3, 3) for sample in samples)
    assert all(math.isfinite(float(sample.runtime.grid.risk.mean())) for sample in samples)
