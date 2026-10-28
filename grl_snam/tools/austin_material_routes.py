"""Austin adapter for the generic public material-route generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..material_palette import MATERIAL_ID, NUM_MATERIALS, terrain_risk
from ..material_routes import MaterialRouteGenerationParams, generate_material_route_manifest
from .material_raster import (
    AUSTIN_BUNDLE_PACKAGE,
    AUSTIN_BUNDLE_SHA256,
    AUSTIN_BUNDLE_VERSION,
    load_austin_material_raster,
    require_austin_bundle,
)

__all__ = [
    "AustinMaterialRouteInputs",
    "generate_austin_material_route_manifest",
    "load_austin_navmask",
    "load_austin_material_route_inputs",
]


@dataclass(frozen=True)
class AustinMaterialRouteInputs:
    """Public Austin bundle rasters aligned on the terrain grid."""

    occupancy: np.ndarray
    risk: np.ndarray
    hard: np.ndarray
    material_id: np.ndarray
    bounds: tuple[float, float, float, float]


def load_austin_navmask(bundle: str | Path) -> np.ndarray:
    """Load the pinned bundle's full-resolution occupancy in simulator orientation."""
    bundle_path = Path(bundle)
    require_austin_bundle(str(bundle_path), extra_files=("navmask.png",))

    import imageio.v3 as iio

    navmask = np.asarray(iio.imread(bundle_path / "navmask.png"))
    if navmask.ndim != 2:
        raise ValueError(f"Austin navmask must be a 2-D grayscale image, got shape {navmask.shape}")
    if not np.isin(navmask, (0, 255)).all():
        raise ValueError("Austin navmask must contain only 0 (free) and 255 (blocked)")
    return (navmask == 255)[::-1].copy()


def load_austin_material_route_inputs(bundle: str | Path) -> AustinMaterialRouteInputs:
    """Load route-generation inputs from the pinned full Austin package.

    Trees, grass, and shrubs remain shared-palette foliage risk; only water is
    a hard material hazard. Terrain slope is intentionally absent because it
    belongs to the independent grade channel.
    """
    bundle_path = Path(bundle)
    navmask_occupancy = load_austin_navmask(bundle_path)
    material_id, bounds_dict = load_austin_material_raster(str(bundle_path))
    bounds = tuple(bounds_dict[key] for key in ("min_x", "min_y", "max_x", "max_y"))
    rows, cols = material_id.shape

    nav_rows, nav_cols = navmask_occupancy.shape
    if nav_rows % rows or nav_cols % cols:
        raise ValueError(
            f"Austin navmask shape {navmask_occupancy.shape} is not divisible by terrain grid "
            f"{material_id.shape}"
        )
    row_scale, col_scale = nav_rows // rows, nav_cols // cols
    occupancy = navmask_occupancy.reshape(rows, row_scale, cols, col_scale).any(axis=(1, 3))

    risk_values = np.asarray([terrain_risk(index) for index in range(NUM_MATERIALS)], np.float32)
    risk = risk_values[material_id]
    hard = material_id == MATERIAL_ID["water"]
    return AustinMaterialRouteInputs(
        occupancy=occupancy,
        risk=risk,
        hard=hard,
        material_id=material_id,
        bounds=bounds,
    )


def generate_austin_material_route_manifest(
    bundle: str | Path,
    *,
    counts,
    hard_counts,
    seed: int,
    params: MaterialRouteGenerationParams = MaterialRouteGenerationParams(),
) -> dict:
    """Generate routes while recording only public package provenance."""
    inputs = load_austin_material_route_inputs(bundle)
    return generate_material_route_manifest(
        inputs.occupancy,
        inputs.risk,
        inputs.hard,
        inputs.bounds,
        counts=counts,
        hard_counts=hard_counts,
        seed=seed,
        material_id=inputs.material_id,
        params=params,
        source={
            "package": AUSTIN_BUNDLE_PACKAGE,
            "version": AUSTIN_BUNDLE_VERSION,
            "archive_sha256": AUSTIN_BUNDLE_SHA256,
        },
    )
