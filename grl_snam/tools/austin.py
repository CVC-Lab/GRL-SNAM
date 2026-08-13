"""The fog demo on real city geometry.

Same simulator, same renderer, a real world: an OpenStreetMap + SRTM scene
bundle (buildings + terrain) rasterized into the occupancy grid the belief
machinery already consumes. Two variants, which is the whole point of putting
them side by side:

``nofog``
    the agent starts with a CORRECT map of the city. Its route is planned once
    and holds; the sensor confirms what it already knows.
``fog``
    the agent starts knowing nothing. It drives into unknown space under the
    optimistic policy, discovers the city block by block, and replans every
    time the map changes underneath it.

The bundle path is always supplied at runtime and never defaulted: the DATA is
OpenStreetMap (ODbL) and SRTM (US public domain), so renders are publishable,
but the bundle itself is a local artifact of a private project and its path
does not belong in this repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grl_snam.fog_stories import Story, build_scenario
from grl_snam.planner import far_pair_in_free_space

# 1200 m of city at 384 cells is 3.13 m/cell -- fine enough for a 4.5 m
# vehicle and coarse enough that pure-python A* over n^2 stays interactive.
DEFAULT_HALF_M = 600.0
DEFAULT_N = 384


def occupancy(bundle_dir: str | Path, *, half_m: float = DEFAULT_HALF_M, n: int = DEFAULT_N):
    """Rasterize the bundle's buildings into a boolean occupancy grid."""
    from pycvc_gl.scenes import building_occupancy

    glb = str(Path(bundle_dir) / "buildings.glb")
    bounds = (-half_m, -half_m, half_m, half_m)
    return building_occupancy(glb, bounds, n, n, inflate_m=0.0).astype(bool), bounds


def austin_story(
    occ: np.ndarray,
    bounds,
    *,
    key: str = "austin",
    fog: bool = True,
    sensor_range_m: float = 120.0,
    inflate_m: float = 6.0,
    max_steps: int = 6000,
) -> tuple[Story, tuple, tuple]:
    """Build the Story plus a start/goal pair that is guaranteed routable.

    Endpoints come from the largest connected component of free space UNDER
    THE PLANNER'S OWN INFLATION. Picking them from raw free space is the trap:
    a cell can be free and still unreachable once the route is inflated, so a
    run drives most of the way and then reports no route for the last stretch.
    """
    cell_m = (bounds[2] - bounds[0]) / (occ.shape[1] - 1)
    start, goal, _ = far_pair_in_free_space(occ, bounds, max(1, round(inflate_m / cell_m)))
    # nudge off the exact border so the vehicle body starts inside the map
    start = (start[0] * 0.97, start[1] * 0.97)
    goal = (goal[0] * 0.97, goal[1] * 0.97)

    caps = (
        (
            (0.0, 6.0, "Real Austin geometry. The agent knows nothing."),
            (6.0, 16.0, "Every street is discovered by looking at it."),
            (16.0, 30.0, "The route is replanned as the map fills in."),
            (30.0, 9999.0, "Across the city on a map it built while driving."),
        )
        if fog
        else (
            (0.0, 6.0, "Real Austin geometry. The map is CORRECT and complete."),
            (6.0, 16.0, "One route, planned once, and it holds."),
            (16.0, 9999.0, "No surprises - this is the baseline to compare against."),
        )
    )
    story = Story(
        key=key,
        title="Austin" + (" under fog" if fog else " with a full map"),
        subtitle="OpenStreetMap + SRTM city geometry",
        n=occ.shape[0],
        bounds=tuple(float(b) for b in bounds),
        scale=0.05,  # keep the normalized regime identical to the fog stories
        dt=0.06,
        nsub=2,
        vmax=0.9,
        rr=0.15,
        d_hat=0.35,
        start=start,
        waypoints=(goal,),
        sensor=dict(range_m=sensor_range_m, n_rays=360),
        sense_every=4,
        inflate_m=inflate_m,
        reach_tol=0.35,
        max_steps=max_steps,
        captions=caps,
        no_fog=not fog,
    )
    return story, start, goal


def build(occ: np.ndarray, story: Story, *, fog: bool, model=None, seed: int = 0):
    """A scenario over the real city. Without fog the agent simply starts with
    the true map, which is the honest way to express 'no fog of war': same
    sensor, same planner, different initial knowledge."""
    prior = None if fog else occ
    return build_scenario(story, model, seed=seed, truth_occ=occ, prior_occ=prior)
