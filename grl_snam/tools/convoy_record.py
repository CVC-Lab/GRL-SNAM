"""A convoy: each vehicle's target is the vehicle in front of it.

Only the leader has a destination. Everyone behind is chasing a target that is
itself reacting to a world it is still discovering, which is the point --
:class:`~grl_snam.fog_stories.MovingGoal` replays a path decided before the run
starts, and a convoy cannot, because where the vehicle in front will be is not
knowable in advance.

Two behaviours fall out of the existing machinery rather than being scripted,
and they are the reason this is worth showing:

* a follower's target is also a peer, so it is stamped into the follower's
  ground truth and has to be *discovered* by a sensor ray like any obstacle;
* because the target occupies its own cell, the follower's goal snaps to the
  nearest free cell beside it, so the convoy closes to a standoff rather than
  driving into the vehicle it is chasing.

Nobody is told the formation. Each vehicle knows one thing: where the vehicle
in front of it was, the last time it looked.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from grl_snam.planner import free_components
from grl_snam.squad import AgentSpec, FollowGoal
from grl_snam.tools.austin import austin_story, occupancy
from grl_snam.tools.finale_record import COLORS
from grl_snam.tools.squad_record import record_squad

N_VEHICLES = 5
SPACING_M = 55.0  # gap between vehicles at the start


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


def build_convoy(occ, bounds, *, n: int = N_VEHICLES):
    """Leader plus ``n-1`` followers, each targeting the one ahead."""
    snap = _free_snapper(occ, bounds)
    # The leader starts well inside the west edge, not on it: the tail is laid
    # out BEHIND it, and at 55 m spacing a five-vehicle convoy needs 220 m of
    # room back there. Starting at -470 ran the last two off the map, where the
    # free-space snapper collapsed both onto the same boundary cell.
    lead_start = snap(-300.0, -120.0)
    lead_goal = snap(470.0, 90.0)

    agents = [AgentSpec("v0", lead_start, lead_goal, COLORS[0])]
    for i in range(1, n):
        # Strung out BEHIND the leader along the line back to the start, so the
        # convoy begins as a convoy instead of forming one.
        back = math.hypot(lead_goal[0] - lead_start[0], lead_goal[1] - lead_start[1])
        ux = (lead_start[0] - lead_goal[0]) / back
        uy = (lead_start[1] - lead_goal[1]) / back
        start = snap(lead_start[0] + ux * SPACING_M * i, lead_start[1] + uy * SPACING_M * i)
        if any(math.dist(start, a.start) < 12.0 for a in agents):
            raise ValueError(
                f"convoy vehicle {i} snapped onto an existing vehicle at {start} -- "
                "the tail has run out of free space; move lead_start inward or "
                "reduce SPACING_M"
            )
        agents.append(
            AgentSpec(
                f"v{i}",
                start,
                # A placeholder: FollowGoal overwrites the waypoint every tick.
                # It has to be somewhere routable or the first plan fails before
                # the follow target has ever been read.
                lead_start,
                COLORS[i % len(COLORS)],
                moving_goal=FollowGoal(f"v{i - 1}"),
            )
        )
    return agents, lead_goal


def record_convoy(
    bundle_dir: str, out_dir, *, n: int = N_VEHICLES, max_steps: int = 3200,
    use_planner: bool = True, progress=None,
):  # fmt: skip
    occ, bounds = occupancy(bundle_dir)
    story0, _s, _g = austin_story(occ, bounds, key="convoy", fog=True)
    agents, _goal = build_convoy(occ, bounds, n=n)

    nav = "route + SDF" if use_planner else "SDF only"
    story = dataclasses.replace(
        story0,
        key="convoy",
        max_steps=max_steps,
        use_planner=use_planner,
        # Same fix as the traffic scene. At the story default of 14 m the leader
        # chases a carrot closer than two seconds of driving and oscillates:
        # measured over the first 36 s it closed only 166 m of an 810 m run at
        # 14 m against 335 m at 22 m, having driven 334 m to do it instead of
        # 503 m. The followers were never the problem -- the leader does this
        # alone.
        route_lookahead_m=22.0,
        captions=(
            (0.0, 9.0, f"A convoy. Only the leader has a destination.  [nav: {nav}]"),
            (9.0, 22.0, "Every other vehicle is chasing the one in front of it."),
            (
                22.0,
                34.0,
                "The target is a peer, so it has to be SEEN - nobody is told where it is.",
            ),
            (34.0, 9999.0, "No formation is scripted. The line is what falls out."),
        ),
    )
    return record_squad(
        story, agents, out_dir, max_steps=max_steps, truth_occ=occ, prior_occ=None,
        progress=progress,
    )  # fmt: skip
