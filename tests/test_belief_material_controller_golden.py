"""libcvc's belief-material controller golden, regenerated and compared byte for byte.

``grl_snam.tools.belief_material_controller_golden`` writes the golden libcvc commits in
``src/cvc/tests/data`` (its ``NAV-BELIEF-MATERIAL-CONTROLLER-ROADMAP.md`` §6). This test
regenerates it and compares bytes with the copy in ``tests/fixtures``. A change to the
controller, the contract, the token builder or ``scan_visible_geometry`` fails here until the
golden is regenerated (``--out tests/fixtures/belief_material_controller_golden``) and the new
files are committed in libcvc too.
"""

import json
from pathlib import Path

import numpy as np

from grl_snam.belief_material_controller import BeliefMaterialContract, BeliefMaterialController
from grl_snam.tools import belief_material_controller_golden as golden

FIXTURE = Path(__file__).parent / "fixtures" / "belief_material_controller_golden"


def test_the_golden_regenerates_byte_for_byte(tmp_path):
    json_path, cvcnm_path = golden.generate(tmp_path)
    for path in (json_path, cvcnm_path):
        assert path.read_bytes() == (FIXTURE / path.name).read_bytes(), path.name


def test_the_golden_net_loads_with_the_default_contract():
    controller = BeliefMaterialController.from_cvcnm(
        FIXTURE / golden.CVCNM_NAME,
        patch_size=golden.NET["patch_size"],
        patch_extent_m=golden.PATCH_EXTENT_M,
    )
    assert controller.contract == BeliefMaterialContract()


def test_the_golden_covers_both_verdicts_and_a_cutoff():
    g = json.loads((FIXTURE / golden.JSON_NAME).read_text())
    accepted = [c["accepted"] for c in g["contract"]]
    assert any(accepted) and not all(accepted)
    assert any(c["mismatches"] for c in g["header_check"])
    assert any(not c["mismatches"] for c in g["header_check"])
    assert any(len(c["mismatches"]) > 1 for c in g["header_check"])
    # max_obstacles 5 cuts every agent's visible set, and the tie scene cuts rings of equal
    # distances (decision E5).
    assert all(len(o["cases"][1]["radii"]) == 5 for o in g["obstacles"])
    assert any(len(o["cases"][0]["radii"]) > 5 for o in g["obstacles"])
    tie = g["obstacle_ties"][0]
    centers = np.array(tie["centers"], np.uint32).view(np.float32).reshape(-1, 2)
    assert [(int(y), int(x)) for x, y in centers] == [
        (4, 4),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
        (3, 3),
        (3, 5),
    ]
    assert {t["variant"] for t in g["tokens"]} == {"main", "scale_7_5", "no_obstacles"}
