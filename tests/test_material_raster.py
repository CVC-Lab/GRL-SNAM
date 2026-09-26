"""grl-snam material-raster: satellite -> material palette segmentation + grip/risk (nav-stats item 2)."""

import json

import numpy as np

from grl_snam import material_palette as mp
from grl_snam.tools import material_raster as mr


def test_classifier_tags_and_flips_to_sim_orientation():
    # 2x2 orthophoto (one pixel per grid cell), north-up. Top row = north (max_y), bottom = south.
    #   top:    [green vegetation, neutral gray]      bottom: [warm tan soil, dark blue water]
    img = np.array(
        [
            [[50, 160, 50], [150, 150, 150]],
            [[175, 130, 90], [40, 90, 200]],
        ],
        dtype=np.uint8,
    )
    mid = mr.classify(img, rows=2, cols=2)
    assert mid.shape == (2, 2)
    # classify() flips rows so material_id[0] == world min_y (the image's BOTTOM row). So grid row 0
    # is the tan/blue (soil/water) south row, grid row 1 is the green/gray north row.
    assert mid[0, 0] == mp.MATERIAL_ID["soil"]
    assert mid[0, 1] == mp.MATERIAL_ID["water"]
    assert mid[1, 0] == mp.MATERIAL_ID["foliage"]
    assert mid[1, 1] == mp.OPEN_AIR_ID  # neutral gray -> the road/hard-surface default


def test_material_json_pulls_grip_and_risk_from_the_palette():
    mid = np.array(
        [
            [mp.MATERIAL_ID["foliage"], mp.OPEN_AIR_ID],
            [mp.MATERIAL_ID["soil"], mp.MATERIAL_ID["water"]],
        ],
        dtype=np.int32,
    )
    doc = mr.to_material_json(mid, {"min_x": -100, "min_y": -50, "max_x": 100, "max_y": 50})
    assert doc["schema"] == "cvc-scene-material/1"
    assert doc["rows"] == 2 and doc["cols"] == 2
    assert doc["bounds"] == {"min_x": -100.0, "min_y": -50.0, "max_x": 100.0, "max_y": 50.0}
    assert doc["material_id"] == mid.tolist()
    # mu/risk come from grl_snam.material_palette, keyed by the ids present
    assert doc["mu"][str(mp.MATERIAL_ID["foliage"])] == mp.GRIP_MU[mp.MATERIAL_ID["foliage"]]
    assert doc["risk"][str(mp.MATERIAL_ID["water"])] == mp.TERRAIN_RISK[mp.MATERIAL_ID["water"]]
    assert doc["mu"][str(mp.OPEN_AIR_ID)] == 1.0 and doc["risk"][str(mp.OPEN_AIR_ID)] == 0.0


def test_palette_grip_risk_defaults_and_risk_set_consistency():
    # unlisted ids (building materials, unknowns) default to full grip / no risk
    assert mp.grip_mu(mp.MATERIAL_ID["reinforced_concrete"]) == 1.0
    assert mp.terrain_risk(mp.MATERIAL_ID["brick"]) == 0.0
    assert mp.grip_mu(999) == 1.0 and mp.terrain_risk(999) == 0.0
    # open_air is the safe surface; the risk-bearing ids (mu<1 AND risk>0) are exactly RISK_MATERIAL_IDS
    assert mp.grip_mu(mp.OPEN_AIR_ID) == 1.0 and mp.terrain_risk(mp.OPEN_AIR_ID) == 0.0
    risky = {i for i in range(mp.NUM_MATERIALS) if mp.grip_mu(i) < 1.0}
    assert risky == set(mp.RISK_MATERIAL_IDS)
    assert all(mp.terrain_risk(i) > 0.0 for i in mp.RISK_MATERIAL_IDS)


def test_generate_writes_material_json(tmp_path):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :] = (150, 150, 150)  # all neutral -> all open_air
    img[0, 0] = (50, 170, 50)  # one green cell
    import imageio.v3 as iio

    sat = tmp_path / "sat.png"
    iio.imwrite(sat, img)
    out = tmp_path / "material.json"
    preview = tmp_path / "tags.png"
    dist = mr.generate(str(sat), 4, 4, (-8, -8, 8, 8), str(out), preview=str(preview))
    doc = json.loads(out.read_text())
    assert doc["rows"] == 4 and doc["cols"] == 4
    assert np.array(doc["material_id"]).shape == (4, 4)
    assert "open_air" in dist and dist["open_air"] > 0.5
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert preview.exists()  # write_preview ran


def test_from_bundle_reads_terrain_and_satellite(tmp_path):
    import imageio.v3 as iio

    (tmp_path / "terrain.json").write_text(
        json.dumps(
            {"rows": 4, "cols": 4, "bounds": {"min_x": -8, "min_y": -8, "max_x": 8, "max_y": 8}}
        )
    )
    img = np.full((4, 4, 3), (150, 150, 150), dtype=np.uint8)
    img[2, 2] = (40, 90, 200)  # a water cell
    iio.imwrite(tmp_path / "satellite.png", img)

    out_path, dist = mr.from_bundle(str(tmp_path))
    assert out_path == str(tmp_path / "material.json")
    doc = json.loads((tmp_path / "material.json").read_text())
    assert doc["rows"] == 4 and doc["bounds"]["max_y"] == 8.0
    assert "water" in dist
