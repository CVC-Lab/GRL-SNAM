"""G12 of libcvc's CoPH golden: an episode of the crossing demo with communication on.

``grl_snam.tools.coph_golden`` writes this section into ``coph_golden.json`` (libcvc's
``NAV-COPH-ROADMAP.md`` §4.5, §5; C5). Two ``CrossingLearner`` agents run the crossing for 60
ticks under the demos' own runner (``staggered_demo.execute``, arm ``reciprocal_fixed``): each
observes, advances, exports every ``update_every`` ticks, takes the other's packets at its next
boundary, updates and acts with the routed controller, on the crossing world with its two doors.

The world is the demos' (``CrossingWorld``) with one change: its sensor noise is a fixed rational
pattern instead of numpy's generator, so the C++ can sense the same readings at its own positions.
The plant is the world's exact flow, in torch, so positions differ in their last bits between
platforms; they are FLOAT64 at atol 1e-6. Every decision is BIT, and generation checks that none
of them changes when the world's damping is nudged by 1e-9 relative each way.
"""

from __future__ import annotations

import numpy as np

from grl_snam.coph.cooperative import CommunicationConfig
from grl_snam.coph.crossing_demo import (
    EDGES,
    NODES,
    SEEDS,
    CrossingLearner,
    CrossingWorld,
    configuration,
    specification,
)
from grl_snam.coph.records import Observation, Reading
from grl_snam.coph.shapes import is_slab, slab_distance
from grl_snam.coph.staggered_demo import execute
from grl_snam.tools.coph_golden_packets import recorded_draws

SEED = SEEDS[0]
TICKS = 60
ARM = "reciprocal_fixed"
NUDGE = 1e-9  # the decisions also hold at 1e-6, where positions move by 2.5e-7
TIER = dict(rtol=0.0, atol=1e-6)


def noise(agent: int, tick: int, j: int) -> float:
    """The golden world's sensor noise: a fixed pattern in [-0.003, 0.003], exact in doubles."""
    return 0.003 * (((7 * tick + 13 * j + 5 * agent) % 11) - 5) / 5.0


class GoldenWorld(CrossingWorld):
    """``CrossingWorld`` with ``noise`` in place of numpy's normal draws (``World.sense``)."""

    def sense(self, x, tick):
        agent = self.seed - 2 * self.spec["seed"]
        grid = np.linspace(-1.0, 1.0, 5)
        offsets = np.array([(a, b) for a in grid for b in grid if a * a + b * b <= 1.0])
        points = x[:2] + offsets * self.cfg.sensor_radius
        truth = self.risk_truth(points)
        values = [max(0.001, float(t) + noise(agent, tick, j)) for j, t in enumerate(truth)]
        readings = tuple(
            Reading(f"{self.seed}:{tick}:{j}", tick * self.cfg.dt, tuple(q), v)
            for j, (q, v) in enumerate(zip(points, values))
        )
        visible = tuple(
            o
            for o in self._obstacles
            if (
                float(slab_distance(x[:2], o))
                if is_slab(o)
                else np.linalg.norm(x[:2] - o[:2]) - o[2]
            )
            <= self.cfg.sensor_radius
        )
        return Observation(
            tick * self.cfg.dt, readings, visible, tuple(x[:2]), self.cfg.sensor_radius
        )


class GoldenLearner(CrossingLearner):
    """``CrossingLearner`` that keeps each export's bootstrap draws (both generators), by packet
    identity, so the C++ can inject them (D8)."""

    draws: dict = {}

    def export(self, x, recipient):
        generators: list = []
        with recorded_draws(generators):
            packet = super().export(x, recipient)
        if packet is not None:
            GoldenLearner.draws.setdefault(packet.identity, [g["draws"] for g in generators])
        return packet


def _world(damping_factor: float):
    class World(GoldenWorld):
        def __init__(self, spec, cfg):
            super().__init__(spec, cfg)
            self._damping = self._damping * damping_factor

    return World


def episode(damping_factor: float = 1.0) -> dict:
    """``execute`` on the golden world: its result."""
    spec = specification(SEED, steps=TICKS)
    comm = CommunicationConfig(byte_budget=4000000, ttl=300.0)
    GoldenLearner.draws = {}
    r = execute(
        spec, ARM, comm, audit=False, learner_factory=GoldenLearner,
        world_factory=_world(damping_factor), config=configuration(TICKS),
    )  # fmt: skip
    r["draws"] = dict(GoldenLearner.draws)
    return r


def _waypoint(meta) -> list:
    from grl_snam.tools.coph_golden import _hexes

    return _hexes(meta.get("waypoint") or ())


def decisions(r: dict) -> dict:
    """What must match bit for bit: per tick each agent's stage, version and routed control; per
    update its packets, attempts and verdict; every send and delivery; the geometry events."""
    from grl_snam.tools.coph_golden_learner import OBSTACLES

    agents = []
    for i in range(2):
        nav = [t["navigation"] for t in r["trajectories"][i]]
        agents.append(dict(
            samples=[[s["stage"], s["version"], bool(s["done"])] for s in r["samples"][i]],
            controls=[[t["tick"], m.get("reason"), m.get("route_reason"),
                       [int(n) for n in m.get("route_nodes") or []], _waypoint(m),
                       int(m.get("guard_steps", 0)), bool(m.get("route_changed", False))]
                      for t, m in zip(r["trajectories"][i], nav)],
            updates=[[u["tick"], list(u["packets"]), u["diagnostic"]["accepted"],
                      u["diagnostic"]["reason"], u["diagnostic"]["version_after"],
                      [[a["power"], a["reason"], a["accepted"]]
                       for a in (u["diagnostic"].get("policy_trace") or {}).get("attempts", [])]]
                     for u in r["updates"][i]],
            geometry=[[e["kind"], OBSTACLES.index(tuple(e["obstacle"])), e.get("source", e.get("packet"))]
                      for e in r["geometry_events"][i]],
            outcome=[bool(r["agents"][i]["success"]), bool(r["agents"][i]["collision"])],
        ))  # fmt: skip
    events = []
    for e in r["events"]:
        if e["kind"] == "sent":
            events.append(["sent", e["sender"], e["receiver"], e["packet"]])
        elif e["kind"] == "delivered":
            events.append(["delivered", e["sender"], e["receiver"], e["packet"], e["reason"],
                           e.get("version_after"), e.get("accepted")])  # fmt: skip
        elif e["kind"] == "activation":
            events.append(["activation", e["agent"]])
        else:
            events.append([e["kind"]])
    return dict(agents=agents, events=events)


def episode_case(check: bool = True) -> dict:
    """G12: the episode's decisions and its positions and forces."""
    from grl_snam.tools.coph_golden import _hexes, hexbits

    r = episode()
    mine = decisions(r)
    if check:
        for sign in (1.0, -1.0):
            if decisions(episode(1.0 + sign * NUDGE)) != mine:
                raise AssertionError(
                    f"G12: a {NUDGE} nudge of the world's damping changes a decision"
                )
    spec = r["spec"]
    out = [
        dict(x=[_hexes(s["x"]) for s in r["samples"][i]],
             u=[_hexes(t["u"]) for t in r["trajectories"][i]],
             nominal=[_hexes(t["nominal"]) for t in r["trajectories"][i]])
        for i in range(2)
    ]  # fmt: skip
    return dict(
        tier=TIER,
        seed=SEED,
        ticks=TICKS,
        arm=ARM,
        world=dict(
            initials=[_hexes(x) for x in spec["initials"]], goals=[_hexes(g) for g in spec["goals"]],
            damping=hexbits(spec["damping"]), obstacles=[_hexes(o) for o in spec["obstacles"]],
            nodes=[_hexes(n) for n in NODES], edges=[list(e) for e in EDGES],
            world_seeds=[spec["seed"] * 2 + i for i in range(2)],
        ),
        draws=[[e[3], r["draws"][e[3]]] for e in mine["events"] if e[0] == "sent"],
        **mine,
        out=out,
    )  # fmt: skip


EPISODE_RULES = """\
G12 pins the demos' runner (staggered_demo.execute, arm reciprocal_fixed) with two
CrossingLearner agents (MazeLearner on TerrainLearner on CooperativeLearner on Learner) for 60
ticks of the crossing: agent 0 from (0, 4) to (12, 4), agent 1 from (12, 4) to (0, 4), the
crossing's network, its middle door at x = 9.5 and its north door at x = 2.5 (world.obstacles,
slabs), damping 0.85, the profile with steps 60, and CommunicationConfig(byte_budget=4e6, ttl=300).

The world (CrossingWorld): the plant is the exact flow at the world's damping (G2), a step hits when
the motion comes within 0.12 of a door (shapes.slab_sweep) or leaves the corridors
(maze_geometry.corridor_clear, margin 0.12). Sensing at tick t for agent i (world seed
2 seed + i): the 21 lattice points x + sensor_radius (a, b), a, b in linspace(-1, 1, 5),
a^2 + b^2 <= 1, in that order; value_j = max(0.001, 0.02 + noise(i, t, j)), noise = 0.003
(((7 t + 13 j + 5 i) % 11) - 5) / 5; uid '<world seed>:<t>:<j>', which the learner prefixes with
'<agent>:'; the doors within sensor_radius of x (slab_distance) are visible; sensor centre x,
radius sensor_radius. Times are the runner's clock, (120000 t) / 1e6.

The schedule (execute), per boundary t, each agent in turn: complete the held step (the world
steps with the held force, the learner observes with the transition (t - 1, now, x, u, after,
version), the episode ends for it on a hit, within 0.25 of its goal, or at t = 60, else it
advances); at t = 0 it only observes. Then exports: at t % update_every == 0 each active agent
exports to the other (not when either is done). Each packet is delivered to its receiver at the
same boundary (delay 0), in order of (timestamp, identity), and assimilated. Each agent not done
then updates when t % update_every == 0 or it received a packet, and acts (the routed control,
held to the next boundary). The status replies and byte counts never bind at this budget, and the
runner's audits are off.

draws: each sent packet's bootstrap resample indices, [identity, [CooperativeLearner's,
TerrainLearner's]] (G9's draws), recorded from the learner's own generators; the C++ injects them.

Decisions (BIT), per agent: samples [stage, version, done] per boundary; controls per step [tick,
reason, route_reason, route_nodes, waypoint, guard_steps, route_changed]; updates [tick, packets,
accepted, reason, version_after, attempts]; geometry events [kind, obstacle index, source or
packet]; outcome [success, collision]. events: the runner's, in order. out (FLOAT64, atol 1e-6):
per agent the state x at each boundary, the applied force u and the nominal force per step.
"""
