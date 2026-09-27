"""Segment a satellite orthophoto into the shared 13-class material palette and write a scene
``material.json`` raster (the material-id grid + per-material grip/risk tables).

The nav-stats scorecard buckets time/distance by material id (``cvc::nav`` ``nav_samplers.material_id``
→ ``material_time_share``), and the drive reads grip ``mu`` (a friction field) + material risk to slow
on soft ground and reroute around hazard. A scene that carries no material data leaves all of those at
their neutral defaults (mu=1, risk=0) and the buckets empty. This turns a scene's satellite imagery —
which the terrain mesh already drapes as its texture — into that material input: one deterministic
land-cover classification feeds all three consumers.

Reusable beyond the cvc Austin bundle: point it at any north-up orthophoto plus the world bounds +
grid it covers (a scene bundle's ``terrain.json`` supplies those, or pass them explicitly). The
classifier is a documented color/index heuristic (excess-green vegetation, warm-tan soil, dark-blue
water, bright warm-gray rock, everything else the neutral ``open_air`` hard/free surface), not a
learned model — reproducible and reviewable; tune :func:`classify` + the palette's ``GRIP_MU`` /
``TERRAIN_RISK`` for other biomes.

NOTE on the palette: it is an RF/building-material table (``reinforced_concrete`` etc. carry
penetration-loss dB) plus a few natural-terrain classes, with NO asphalt/road class. ``reinforced_
concrete`` means building walls, not pavement; buildings are obstacles the convoy never drives
(tagged from the scene's building metadata), so this GROUND classifier only ever emits the drivable
classes ``open_air`` / ``foliage`` / ``soil`` / ``water`` / ``rock``.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..material_palette import (
    MATERIAL_ID,
    MATERIALS,
    OPEN_AIR_ID,
    grip_mu,
    terrain_risk,
)

FOLIAGE = MATERIAL_ID["foliage"]
SOIL = MATERIAL_ID["soil"]
WATER = MATERIAL_ID["water"]
ROCK = MATERIAL_ID["rock"]

# color-coded display colors for the review preview (id -> RGB)
_PREVIEW_COL = {
    OPEN_AIR_ID: (170, 170, 175),
    FOLIAGE: (60, 150, 60),
    SOIL: (170, 130, 80),
    ROCK: (120, 110, 95),
    WATER: (40, 90, 200),
}


def classify(sat_rgb: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean-pool a north-up orthophoto to ``(rows, cols)`` and tag each cell into the drivable palette.

    Returns an ``int32`` ``[rows, cols]`` grid oriented to the SIM convention — row 0 == world
    ``min_y``, col 0 == ``min_x`` — so a consumer sampling ``material_id[r][c]`` with
    ``r=(y-min_y)/(max_y-min_y)*rows`` / ``c=(x-min_x)/(max_x-min_x)*cols`` (the ``nav_samplers`` /
    ``sim_world`` occupancy convention) is a straight index. The orthophoto is north-up (image row 0 ==
    ``max_y``) and the mesh drapes it with a lower-left UV origin, so the pooled grid is flipped in y.
    """
    im = np.asarray(sat_rgb)[..., :3].astype(np.float32)
    H, W = im.shape[0], im.shape[1]
    ph, pw = H // rows, W // cols
    if ph < 1 or pw < 1:
        raise ValueError(f"image {W}x{H} smaller than the {cols}x{rows} grid")
    im = im[: rows * ph, : cols * pw].reshape(rows, ph, cols, pw, 3).mean(axis=(1, 3))

    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    bright = (r + g + b) / 3.0
    exg = 2.0 * g - r - b  # excess green (positive over foliage incl. muted olive canopy)
    warm = r - b

    # DEFAULT open_air = the neutral hard/free surface (roads, pavement, developed ground); carve out
    # only the terrain-RISK classes on top. Order matters (later rules win on overlap).
    mid = np.full((rows, cols), OPEN_AIR_ID, dtype=np.int32)
    mid[(warm > 14) & (r > g) & (g >= b) & (exg < 6)] = (
        SOIL  # warm tan/brown = bare / natural ground
    )
    mid[(exg > 14) & (g >= b)] = FOLIAGE  # excess-green = grass / tree canopy
    mid[(bright > 170) & (warm > 6) & (exg < 8) & (r > g)] = ROCK  # bright warm-gray = exposed rock
    mid[(b > r + 6) & (b > g) & (bright < 125)] = WATER  # dark blue = water
    return mid[::-1].copy()  # flip to row 0 == min_y (see docstring)


def _bounds_dict(bounds) -> dict:
    if isinstance(bounds, dict):
        return {k: float(bounds[k]) for k in ("min_x", "min_y", "max_x", "max_y")}
    min_x, min_y, max_x, max_y = bounds
    return {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
    }


def to_material_json(material_id: np.ndarray, bounds) -> dict:
    """Build the ``cvc-scene-material/1`` document from a material-id grid + world bounds."""
    rows, cols = int(material_id.shape[0]), int(material_id.shape[1])
    ids = sorted(int(v) for v in np.unique(material_id))
    return {
        "schema": "cvc-scene-material/1",
        "provenance": "scene land cover (grl-snam material-raster: masks if present, else satellite); ids = cvc::dbg MATERIAL_TABLE",
        "rows": rows,
        "cols": cols,
        "bounds": _bounds_dict(bounds),
        "palette": list(MATERIALS),
        # per-material grip + risk from the shared palette (grl_snam.material_palette); a loader
        # defaults any absent id to mu=1 / risk=0.
        "mu": {str(i): grip_mu(i) for i in ids},
        "risk": {str(i): terrain_risk(i) for i in ids},
        "material_id": material_id.tolist(),  # [rows][cols] row-major; row 0 == min_y, col 0 == min_x
    }


def write_preview(material_id: np.ndarray, path: str, scale: int = 3) -> None:
    """Write a color-coded tag map (nearest-neighbor upscaled) for visual review."""
    import imageio.v3 as iio

    rows, cols = material_id.shape
    vis = np.zeros((rows, cols, 3), dtype=np.uint8)
    for mid, col in _PREVIEW_COL.items():
        vis[material_id == mid] = col
    vis = np.repeat(np.repeat(vis, scale, axis=0), scale, axis=1)
    iio.imwrite(path, vis)


def distribution(material_id: np.ndarray) -> dict:
    """{material name: fraction} over the grid, most-common first (for CLI reporting)."""
    tot = material_id.size
    out = {}
    for i in sorted(
        (int(v) for v in np.unique(material_id)), key=lambda i: -int((material_id == i).sum())
    ):
        out[MATERIALS[i]] = float((material_id == i).sum()) / tot
    return out


def _write(material_id: np.ndarray, bounds, out: str, preview: str | None) -> dict:
    doc = to_material_json(material_id, bounds)
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    if preview:
        write_preview(material_id, preview)
    return distribution(material_id)


def generate(
    satellite: str, rows: int, cols: int, bounds, out: str, preview: str | None = None
) -> dict:
    """Segment ``satellite`` over an ``rows x cols`` grid spanning ``bounds`` and write ``out``
    (``material.json``). ``bounds`` is a dict (min_x/min_y/max_x/max_y) or a 4-tuple. Returns the
    class distribution."""
    import imageio.v3 as iio

    return _write(classify(iio.imread(satellite), rows, cols), bounds, out, preview)


# foliage_mask.png land classes (the scene pipeline's unet-efficientnet segmentation) -> palette id.
# 0 = none (defer to the other masks); 1 tree / 2 grass / 4 shrub -> foliage; 3 rock; 5 bare -> soil.
_FOLIAGE_CLASS = {1: FOLIAGE, 2: FOLIAGE, 4: FOLIAGE, 3: ROCK, 5: SOIL}


def segment_from_masks(bundle: str, rows: int, cols: int):
    """Build the material_id grid from a full bundle's AUTHORITATIVE land-cover masks — the scene
    pipeline's OSM + ML (unet) segmentation, higher quality than the satellite color heuristic:
    ``foliage_mask.png`` (per-pixel land class 0..5), ``roads.png`` / ``water.png`` /
    ``open_fields.png`` (alpha masks). Returns None if the masks are absent (a lean bundle) so the
    caller falls back to :func:`classify`. Priority, later wins: open_air default -> open_fields=soil
    -> foliage classes -> water -> roads=open_air (you drive on roads, even through vegetation/fields).
    Downsampled to the grid by MAJORITY class, then flipped to row 0 == world min_y."""
    import os

    import imageio.v3 as iio

    fol_p = os.path.join(bundle, "foliage_mask.png")
    if not os.path.exists(fol_p):
        return None

    def alpha_mask(name: str):
        p = os.path.join(bundle, name)
        if not os.path.exists(p):
            return None
        im = np.asarray(iio.imread(p))
        return (im[..., 3] > 0) if (im.ndim == 3 and im.shape[2] == 4) else (im > 0)

    fol = np.asarray(iio.imread(fol_p))
    if fol.ndim == 3:
        fol = fol[..., 0]
    h, w = fol.shape
    pix = np.full((h, w), OPEN_AIR_ID, dtype=np.int32)
    of = alpha_mask("open_fields.png")
    if of is not None:
        pix[of] = SOIL
    for v, mid_id in _FOLIAGE_CLASS.items():
        pix[fol == v] = mid_id
    wa = alpha_mask("water.png")
    if wa is not None:
        pix[wa] = WATER
    rd = alpha_mask("roads.png")
    if rd is not None:
        pix[rd] = OPEN_AIR_ID

    ph, pw = h // rows, w // cols
    if ph < 1 or pw < 1:
        raise ValueError(f"masks {w}x{h} smaller than the {cols}x{rows} grid")
    pix = pix[: rows * ph, : cols * pw]
    best = np.zeros((rows, cols))
    mid = np.full((rows, cols), OPEN_AIR_ID, dtype=np.int32)
    for m in (OPEN_AIR_ID, SOIL, FOLIAGE, WATER, ROCK):
        frac = (pix == m).reshape(rows, ph, cols, pw).mean((1, 3))
        take = frac > best
        mid[take] = m
        best[take] = frac[take]
    return mid[::-1].copy()


def from_bundle(
    bundle: str, out: str | None = None, preview: str | None = None
) -> tuple[str, dict]:
    """Read a cvc scene bundle's ``terrain.json`` (rows/cols/bounds), build the material raster from
    its authoritative land-cover MASKS if present (:func:`segment_from_masks`) and otherwise by
    classifying ``satellite.png`` (:func:`classify`), and write ``<bundle>/material.json`` (or
    ``out``). Returns (out_path, distribution)."""
    import imageio.v3 as iio

    terr = json.load(open(os.path.join(bundle, "terrain.json")))
    rows, cols, bounds = int(terr["rows"]), int(terr["cols"]), terr["bounds"]
    out = out or os.path.join(bundle, "material.json")
    material_id = segment_from_masks(bundle, rows, cols)
    if material_id is None:
        material_id = classify(iio.imread(os.path.join(bundle, "satellite.png")), rows, cols)
    return out, _write(material_id, bounds, out, preview)
