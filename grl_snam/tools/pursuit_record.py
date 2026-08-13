"""A pursuit where the quarry is a vehicle, not a scripted path.

The first version of this used :class:`~grl_snam.fog_stories.MovingGoal`, which
walks a polyline decided before the run starts. A polyline knows nothing about
the city, so the targets drove through buildings -- fine for a synthetic scene
with a handful of rectangles, indefensible on real geometry.

Here the runners are ordinary agents. They plan, they sense, they route around
what they find, and they are peers, so a pursuer has to *see* its target rather
than being told where it is. The chase geometry stops being decoration and
starts being the result.

Three things are arranged deliberately:

* **runners start far away** -- across the map, not beside the pursuers, so the
  clip opens with a gap to close instead of an immediate intercept;
* **runners are faster** than the pursuers, so closing requires cutting corners
  rather than out-accelerating, and the chase lasts;
* **the two pairs cross** -- the runners' escape lines are chosen to intersect,
  so the packs have to thread through each other and around each other's
  vehicles, which is only meaningful now that peers are real to each other.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from grl_snam.planner import free_components
from grl_snam.squad import AgentSpec, FollowGoal
from grl_snam.tools.austin import austin_story, occupancy
from grl_snam.tools.finale_record import COLORS
from grl_snam.tools.squad_record import record_squad

N_RUNNERS = 2
PURSUERS_PER_RUNNER = 3


def _free_snapper(occ, bounds, inflate_m: float = 6.0):
    cell = (bounds[2] - bounds[0]) / (occ.shape[1] - 1)
    labels, sizes = free_components(occ, max(1, round(inflate_m / cell)))
    big = max(sizes, key=lambda k: sizes[k])
    rows, cols = np.nonzero(labels == big)
    xs = bounds[0] + cols / (occ.shape[1] - 1) * (bounds[2] - bounds[0])
    ys = bounds[1] + rows / (occ.shape[0] - 1) * (bounds[3] - bounds[1])

    def snap(tx, ty):
        i = int(np.argmin((xs - tx) ** 2 + (ys - ty) ** 2))
        return float(xs[i]), float(ys[i])

    return snap


def build_pursuit(occ, bounds, *, runner_vmax=None):
    """Runners fleeing on crossing lines, each hunted by a pack of pursuers."""
    snap = _free_snapper(occ, bounds)

    # Crossing escape lines: runner A flees NW->SE, runner B flees SW->NE, so
    # the two packs must pass through each other around the middle of the map.
    runners = [
        ("r0", snap(-430.0, 380.0), snap(430.0, -300.0)),
        ("r1", snap(-430.0, -380.0), snap(430.0, 300.0)),
    ]
    # Pursuers start on the FAR side from each runner, so the opening frame has
    # a gap of most of the map rather than an immediate intercept.
    packs = [
        [snap(300.0, -260.0 + 70.0 * i) for i in range(PURSUERS_PER_RUNNER)],
        [snap(300.0, 240.0 - 70.0 * i) for i in range(PURSUERS_PER_RUNNER)],
    ]

    agents = []
    for j, (key, start, goal) in enumerate(runners):
        agents.append(AgentSpec(key, start, goal, COLORS[j], vmax=runner_vmax))
    for j, (key, _s, _g) in enumerate(runners):
        for i, start in enumerate(packs[j]):
            agents.append(
                AgentSpec(
                    f"p{j}{i}",
                    start,
                    # Placeholder; FollowGoal overwrites it every tick, but the
                    # first plan happens before the follow target is ever read.
                    runners[j][1],
                    COLORS[2 + j * PURSUERS_PER_RUNNER + i],
                    moving_goal=FollowGoal(key),
                )
            )
    return agents


def record_pursuit_v2(
    bundle_dir: str, out_dir, *, max_steps: int = 3000, runner_vmax_scale: float = 1.25,
    use_planner: bool = True, progress=None,
):  # fmt: skip
    """Record the chase.

    ``runner_vmax_scale`` is why this is not simply a convoy: the quarry is
    quicker than its hunters, so the pack closes by cutting corners through the
    city rather than by out-running it in a straight line.
    """
    occ, bounds = occupancy(bundle_dir)
    story0, _s, _g = austin_story(occ, bounds, key="pursuit2", fog=True)
    agents = build_pursuit(occ, bounds, runner_vmax=story0.vmax * runner_vmax_scale)

    nav = "route + SDF" if use_planner else "SDF only"
    story = dataclasses.replace(
        story0,
        key="pursuit2",
        max_steps=max_steps,
        use_planner=use_planner,
        # Same lookahead the traffic scene and the convoy needed; the story
        # default of 14 m makes a vehicle chase a carrot closer than two
        # seconds of driving and oscillate.
        route_lookahead_m=22.0,
        captions=(
            (0.0, 9.0, f"Two runners, six hunters. The quarry is a vehicle too.  [nav: {nav}]"),
            (9.0, 20.0, "No scripted paths - the runners plan and route around what they find."),
            (20.0, 32.0, "A hunter has to SEE its target. Nobody is told where anybody is."),
            (
                32.0,
                9999.0,
                "The escape lines cross, so the two packs must thread through each other.",
            ),
        ),
    )
    return record_squad(
        story, agents, out_dir, max_steps=max_steps, truth_occ=occ, prior_occ=None,
        progress=progress,
    )  # fmt: skip
