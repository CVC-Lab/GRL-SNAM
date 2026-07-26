"""Scene loaders + grounded routing — moved to :mod:`pycvc_gl.scenes`. Re-exported
here for backward compatibility."""
from pycvc_gl.scenes import *  # noqa: F401,F403
from pycvc_gl.scenes import (  # noqa: F401
    terrain_grid, terrain_sampler, add_terrain_json, add_gltf, load_geometry_bundle,
    building_occupancy, plan_ground_route, resample_polyline,
)
