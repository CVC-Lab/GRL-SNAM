import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from grl_snam.material_routes import (
    MaterialRouteGenerationParams,
    generate_material_route_manifest,
    segment_distance_m,
    validate_material_route_manifest,
)

COMMITTED_ROUTES = Path(__file__).resolve().parents[1] / "experiments/austin_material/routes.json"


def _world():
    """A 60x60 raster at 10 m/cell: open ground with foliage stripes and two water ponds."""
    shape = (60, 60)
    occupancy = np.zeros(shape, bool)
    occupancy[:, :2] = occupancy[:, -2:] = occupancy[:2, :] = occupancy[-2:, :] = True
    risk = np.zeros(shape, np.float32)
    risk[:, 10::12] = 0.30
    hard = np.zeros(shape, bool)
    hard[20:24, 20:24] = True
    hard[40:44, 36:40] = True
    risk[hard] = 1.0
    bounds = (0.0, 0.0, 590.0, 590.0)
    return occupancy, risk, hard, bounds


def _generate(seed=3, **overrides):
    occupancy, risk, hard, bounds = _world()
    params = MaterialRouteGenerationParams(
        min_distance_m=80.0,
        max_distance_m=200.0,
        min_endpoint_separation_m=20.0,
        min_split_corridor_separation_m=30.0,
        **overrides,
    )
    return generate_material_route_manifest(
        occupancy,
        risk,
        hard,
        bounds,
        counts={"train": 4, "validation": 2, "test": 4},
        hard_counts={"train": 1, "validation": 1, "test": 1},
        seed=seed,
        params=params,
    )


def test_segment_distance():
    assert segment_distance_m((0, 0), (10, 0), (5, -5), (5, 5)) == 0.0
    assert segment_distance_m((0, 0), (10, 0), (0, 3), (10, 3)) == 3.0
    assert math.isclose(segment_distance_m((0, 0), (1, 0), (4, 4), (5, 5)), 5.0)
    assert segment_distance_m((0, 0), (0, 0), (3, 4), (3, 4)) == 5.0


def test_generator_is_deterministic_and_records_its_protocol():
    first, second = _generate(), _generate()
    assert first == second
    assert _generate(seed=4) != first
    assert first["protocol"]["minimum_split_corridor_separation_m"] == 30.0
    assert [route["challenge_kind"] for route in first["test"]].count("hard") == 1
    assert first["train"][0]["id"] == 500


def test_generated_splits_keep_their_corridors_apart():
    manifest = _generate()
    routes = [
        (split, route) for split in ("train", "validation", "test") for route in manifest[split]
    ]
    for index, (split, a) in enumerate(routes):
        for other_split, b in routes[index + 1 :]:
            if other_split != split:
                distance = segment_distance_m(
                    a["start_xy"], a["goal_xy"], b["start_xy"], b["goal_xy"]
                )
                assert distance >= 30.0


def test_validator_rejects_a_test_corridor_crossing_a_training_corridor():
    manifest = _generate()
    leaked = copy.deepcopy(manifest)
    train = leaked["train"][0]
    test = leaked["test"][0]
    # Put the test route's goal on the training route's midpoint: endpoints stay apart, corridors meet.
    midpoint = [(a + b) / 2.0 for a, b in zip(train["start_xy"], train["goal_xy"])]
    test["goal_xy"] = midpoint
    test["direct_distance_m"] = math.dist(test["start_xy"], midpoint)
    leaked["protocol"]["minimum_endpoint_separation_m"] = 0.0
    with pytest.raises(ValueError, match="corridors violate"):
        validate_material_route_manifest(leaked)


def test_committed_austin_manifest_is_valid_and_pinned():
    manifest = json.loads(COMMITTED_ROUTES.read_text(encoding="utf-8"))
    validate_material_route_manifest(manifest)
    protocol = manifest["protocol"]
    assert protocol["minimum_split_corridor_separation_m"] >= 60.0
    assert protocol["source"]["version"] == "1.0.0+cvc.3"
    assert set(protocol["source"]["files_sha256"]) >= {"navmask.png", "water.png"}
    counts = {split: len(manifest[split]) for split in ("train", "validation", "test")}
    assert counts == {"train": 12, "validation": 6, "test": 30}
