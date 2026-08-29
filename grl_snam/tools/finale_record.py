"""The two acts of the finale, defined once so they can be re-recorded.

Act one is a rendezvous: eight vehicles enter from the west edge of the map
with no prior knowledge and converge on a staging line in the east. Act two is
a pursuit: the same eight break north after four targets that do not wait to be
caught.

Both are recorded, never simulated at render time. The renderer replays what
was measured — :mod:`grl_snam.tools.finale_capture` reads these traces and
draws them, and if it re-derived anything it would be showing you a second
simulation rather than this one.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from grl_snam.fog_stories import MovingGoal
from grl_snam.planner import free_components
from grl_snam.squad import AgentSpec
from grl_snam.tools.austin import austin_story, occupancy
from grl_snam.tools.squad_record import record_squad

# Eight readable hues that stay distinct at inset scale, where a vehicle is a
# dozen pixels of arrowhead.
COLORS = [
    (0.35, 0.75, 1.00), (1.00, 0.55, 0.25), (0.45, 0.90, 0.50), (0.90, 0.50, 0.90),
    (0.98, 0.85, 0.30), (0.40, 0.95, 0.90), (1.00, 0.42, 0.42), (0.70, 0.60, 1.00),
]  # fmt: skip

N_AGENTS = 8
WEST_X = -540.0
LINE_X = 470.0  # the staging line the squad forms on
LINE_HALF = 160.0


def _free_snapper(occ, bounds, inflate_m: float = 6.0):
    """Snap a point to the nearest cell that is free UNDER PLANNER INFLATION.

    Raw free space is the trap: a cell can be free and still unreachable once
    the route is inflated, which yields a goal the planner can see and never
    arrive at.
    """
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


def staging_line(snap):
    """Where the squad forms up, ordered south to north.

    A LINE, not a ring. A ring puts three of its eight slots on the far side,
    and since every agent approaches from the west, the first five to arrive
    park in a wall between the stragglers and their goals: measured, the three
    agents holding the easternmost slots spent the last minute of the run
    orbiting the rendezvous at 10 m/s, 115 m / 121 m / 329 m short of it.

    Slot order matches start order, so no agent ever crosses another's path and
    none can be walled out by one that arrived first.
    """
    return [snap(LINE_X, y) for y in np.linspace(-LINE_HALF, LINE_HALF, N_AGENTS)]


def record_rendezvous(bundle_dir: str, out_dir, *, max_steps: int = 3000, progress=None):
    """Act one: enter blind from the west, converge on the staging line."""
    occ, bounds = occupancy(bundle_dir)
    story0, _s, _g = austin_story(occ, bounds, key="finale", fog=True)
    snap = _free_snapper(occ, bounds)

    starts = [snap(WEST_X, y) for y in np.linspace(-430.0, 430.0, N_AGENTS)]
    goals = staging_line(snap)
    agents = [AgentSpec(f"v{i}", starts[i], goals[i], COLORS[i]) for i in range(N_AGENTS)]

    story = dataclasses.replace(
        story0,
        key="finale_rendezvous",
        max_steps=max_steps,
        captions=(
            (0.0, 8.0, "Eight vehicles. Real Austin. None of them has a map."),
            (8.0, 22.0, "Each builds its own as it drives - coverage shared, knowledge private."),
            (22.0, 9999.0, "Converging on a rendezvous across the city."),
        ),
    )
    return record_squad(
        story, agents, out_dir, max_steps=max_steps, truth_occ=occ, prior_occ=None,
        progress=progress, route_clearance=(6.0, 1.5),
    )  # fmt: skip


def record_pursuit(bundle_dir: str, out_dir, *, max_steps: int = 2400, progress=None):
    """Act two: break north from the staging line after four moving targets."""
    occ, bounds = occupancy(bundle_dir)
    story0, _s, _g = austin_story(occ, bounds, key="finale2", fog=True)
    snap = _free_snapper(occ, bounds)

    starts = staging_line(snap)  # exactly where act one leaves them

    # Four targets, two pursuers each: a pair converging on one target from
    # different streets reads as coordination, where eight separate chases
    # reads as noise. Targets run at 4 m/s against the vehicles' ~10, so the
    # chase actually closes rather than becoming a parade.
    targets = []
    for ty in (470.0, 330.0, 400.0, 250.0):
        leg = [
            snap(x, ty + 40.0 * math.sin(x / 220.0)) for x in (420.0, 180.0, -80.0, -340.0, -540.0)
        ]
        targets.append(MovingGoal(path=tuple(leg), speed_mps=4.0, start_s=0.0, loop=True))

    agents = [
        AgentSpec(
            f"v{i}", starts[i], targets[i // 2].path[0], COLORS[i], moving_goal=targets[i // 2]
        )
        for i in range(N_AGENTS)
    ]

    story = dataclasses.replace(
        story0,
        key="finale_pursuit",
        max_steps=max_steps,
        captions=(
            (0.0, 8.0, "Rendezvous complete. Four targets, moving, in the north quarter."),
            (8.0, 20.0, "Two vehicles per target - each still planning on its own private map."),
            (20.0, 9999.0, "The route retargets in place; it is never restarted."),
        ),
    )
    return record_squad(
        story, agents, out_dir, max_steps=max_steps, stall_ticks=260,
        truth_occ=occ, prior_occ=None, progress=progress,
    )  # fmt: skip


def record_both(bundle_dir: str, out_root, *, progress=None) -> dict[str, Path]:
    out = Path(out_root)
    return {
        "rendezvous": record_rendezvous(bundle_dir, out / "finale_rendezvous", progress=progress),
        "pursuit": record_pursuit(bundle_dir, out / "finale_pursuit", progress=progress),
    }
