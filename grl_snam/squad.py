"""Several agents in one world, each with its own private belief.

The design decision that matters here is that belief is **per agent**. A shared
map would be a much smaller change and a much weaker demo: the interesting
behaviour is that agent A can know about a wall agent B has never seen, so the
two route differently through the same city and only converge once their
knowledge does. Sharing a belief grid would erase precisely that.

The second decision is that agents are **real to each other**. Each one's
sensor reads a truth grid that includes the others' current footprints, so they
occlude, get discovered, and are routed around — the same treatment
:class:`~grl_snam.fog_stories.Mover` gets, rather than a special case. Nobody is
told where anybody is.

Implementation is deliberately thin: one :class:`~grl_snam.scenario.FogScenario`
per agent over a shared static truth, stepped in lockstep. That reuses the whole
tested single-agent path — sensing, the route spine, the vehicle model, the
collision scoring — instead of forking it, and it keeps each agent's trace in
exactly the format the renderer already reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fog_stories import Story, build_scenario


@dataclass
class AgentSpec:
    """One agent's start, goal and colour.

    ``moving_goal`` makes the target a path rather than a point -- each agent
    can chase its own, which is the difference between a rendezvous and a
    pursuit.
    """

    key: str
    start: tuple[float, float]
    goal: tuple[float, float]
    color: tuple[float, float, float] = (0.97, 0.97, 0.97)
    moving_goal: object | None = None


@dataclass
class SquadResult:
    ticks: int = 0
    reached: dict = field(default_factory=dict)
    penetration: dict = field(default_factory=dict)
    tracks: dict = field(default_factory=dict)


class Squad:
    """N independent agents over one shared world."""

    def __init__(
        self,
        story: Story,
        agents: list[AgentSpec],
        model=None,
        *,
        seed: int = 0,
        truth_occ=None,
        prior_occ=None,
    ):
        if not agents:
            raise ValueError("a squad needs at least one agent")
        self.story = story
        self.agents = list(agents)
        self.scenarios = {}
        for a in self.agents:
            # Each agent gets its own scenario -- and therefore its own
            # BeliefGrid, its own route, its own SDF. Nothing is shared but
            # ground truth.
            import dataclasses

            s = dataclasses.replace(
                story,
                start=a.start,
                waypoints=(a.goal,),
                moving_goal=a.moving_goal if a.moving_goal is not None else story.moving_goal,
            )
            self.scenarios[a.key] = build_scenario(
                s, model, seed=seed, truth_occ=truth_occ, prior_occ=prior_occ
            )
        self.step_i = 0

    # ── the shared world ────────────────────────────────────────────────────
    def _footprint(self, sc, half_m: float) -> tuple[int, int, int, int] | None:
        x, y = sc.nav.pos_world()
        r, c = sc.belief.world_to_cell(x, y)
        if not sc.belief.in_bounds(r, c):
            return None
        rad = max(1, int(round(half_m / sc.cell_m)))
        ny, nx = sc.truth.shape
        return (max(0, r - rad), min(ny, r + rad + 1), max(0, c - rad), min(nx, c + rad + 1))

    def _stamp_peers(self, half_m: float = 4.0) -> None:
        """Write every agent's body into every OTHER agent's truth.

        Done before sensing, so a peer is discovered the same way a wall is:
        by a ray landing on it. An agent never sees its own footprint, or it
        would map itself as an obstacle and refuse to move.

        Handed over as a MASK rather than written straight into ``truth_now``.
        Writing it directly did not survive: the first thing ``FogScenario.step``
        does is ``_stamp_movers``, whose opening line is
        ``truth_now = truth.copy()`` -- so every peer footprint was erased
        microseconds after being stamped, before anything sensed against it.
        Measured with eight agents 17 m apart: 63 cells stamped, 0 surviving.
        The squad was mutually invisible, which also made its zero-collision
        result vacuous, peers being absent from the grid that penetration is
        scored against. A mask the scenario ORs in is order-independent and
        cannot be clobbered.
        """
        boxes = {k: self._footprint(sc, half_m) for k, sc in self.scenarios.items()}
        for k, sc in self.scenarios.items():
            mask = np.zeros(sc.truth.shape, dtype=bool)
            for j, box in boxes.items():
                if j == k or box is None:
                    continue
                r0, r1, c0, c1 = box
                mask[r0:r1, c0:c1] = True
            sc.peer_occ = mask

    def _demote_peers(self, half_m: float = 4.0) -> None:
        """A peer the sensor just saw belongs in the DECAYING layer.

        Same reasoning as a Mover: baked into the static map, a moving agent
        leaves a permanent wall along its path and every later route detours
        around a corridor nobody is in.
        """
        boxes = {k: self._footprint(sc, half_m) for k, sc in self.scenarios.items()}
        for k, sc in self.scenarios.items():
            t = sc._t()
            for j, box in boxes.items():
                if j == k or box is None:
                    continue
                r0, r1, c0, c1 = box
                if not sc.belief.last_visible[r0:r1, c0:c1].any():
                    continue
                sc.belief.logodds[r0:r1, c0:c1] = np.minimum(sc.belief.logodds[r0:r1, c0:c1], 0.0)
                sc.dyn.mark((r0 + r1) // 2, (c0 + c1) // 2, t, radius_cells=(r1 - r0) // 2 or 1)

    # ── the loop ────────────────────────────────────────────────────────────
    def step(self) -> dict:
        self._stamp_peers()
        out = {}
        for k, sc in self.scenarios.items():
            out[k] = sc.step()
        self._demote_peers()
        self.step_i += 1
        return out

    @property
    def done(self) -> bool:
        return all(sc.done for sc in self.scenarios.values())

    def run(
        self,
        max_steps: int = 4000,
        *,
        stop_when_done: bool = True,
        stall_ticks: int = 0,
    ) -> SquadResult:
        """Step until everyone arrives, or until nobody is making progress.

        ``stall_ticks`` ends the run when no agent has improved on its best
        distance-to-goal for that many ticks. Without it a single agent that
        cannot close the last stretch holds the whole run open to max_steps,
        and the recording ends with a long stretch of nothing moving -- dead
        air in the clip, and minutes of wasted render.
        """
        res = SquadResult()
        tracks = {k: [] for k in self.scenarios}
        pen = {k: 0 for k in self.scenarios}
        best = {k: float("inf") for k in self.scenarios}
        since = 0
        for _ in range(max_steps):
            recs = self.step()
            improved = False
            for k, r in recs.items():
                tracks[k].append((r.x, r.y))
                pen[k] += int(r.truth_penetration)
                if r.goal_dist_m < best[k] - 0.5:
                    best[k] = r.goal_dist_m
                    improved = True
            since = 0 if improved else since + 1
            res.ticks += 1
            if stop_when_done and self.done:
                break
            if stall_ticks and since >= stall_ticks:
                break
        res.tracks = {k: np.asarray(v, np.float32) for k, v in tracks.items()}
        res.penetration = pen
        res.reached = {k: sc.done for k, sc in self.scenarios.items()}
        return res
