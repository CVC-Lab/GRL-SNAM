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
        "provenance": "satellite-derived land cover (grl-snam material-raster); ids = cvc::dbg MATERIAL_TABLE",
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


def generate(
    satellite: str, rows: int, cols: int, bounds, out: str, preview: str | None = None
) -> dict:
    """Segment ``satellite`` over an ``rows x cols`` grid spanning ``bounds`` and write ``out``
    (``material.json``). ``bounds`` is a dict (min_x/min_y/max_x/max_y) or a 4-tuple. Returns the
    class distribution."""
    import imageio.v3 as iio

    material_id = classify(iio.imread(satellite), rows, cols)
    doc = to_material_json(material_id, bounds)
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    if preview:
        write_preview(material_id, preview)
    return distribution(material_id)


def from_bundle(
    bundle: str, out: str | None = None, preview: str | None = None
) -> tuple[str, dict]:
    """Convenience: read a cvc scene bundle's ``terrain.json`` (rows/cols/bounds) + ``satellite.png``,
    segment, and write ``<bundle>/material.json`` (or ``out``). Returns (out_path, distribution)."""
    terr = json.load(open(os.path.join(bundle, "terrain.json")))
    out = out or os.path.join(bundle, "material.json")
    dist = generate(
        os.path.join(bundle, "satellite.png"),
        int(terr["rows"]),
        int(terr["cols"]),
        terr["bounds"],
        out,
        preview,
    )
    return out, dist
