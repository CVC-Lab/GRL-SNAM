"""G13 of libcvc's CoPH golden: two car learners on an open field with communication on.

``grl_snam.tools.coph_golden`` writes this section into ``coph_golden.json`` (libcvc's
``NAV-COPH-ROADMAP.md`` §4.5, §5; C5 part 2). The learner is the cars controller's
(``grl_snam.ariadne.coph_cars.CarLearner``, demo 2b): a ``CooperativeLearner`` whose stages follow
it in any direction and which routes with the profile's own planner (``navigation.route``: A*
through its sensor footprints on a 0.1 m grid), since a car's world has no surveyed network. Its
packets are ``CooperativeLearner``'s, with no maze or terrain layers.

Two of them cross an open field under the demos' runner (``staggered_demo.execute``, arm
``reciprocal_fixed``), as in G12, on ``CooperativeWorld``: the exact flow, two disks and a risk
patch between the agents. The sensor noise is G12's rational pattern. Every decision is BIT, and
generation checks that none of them changes when the world's damping is nudged by 1e-9 relative
each way; positions and forces are FLOAT64 at atol 1e-6.
"""

from __future__ import annotations

import numpy as np

from grl_snam.coph.cooperative import CommunicationConfig
from grl_snam.coph.cooperative_evaluation import CooperativeWorld
from grl_snam.coph.crossing_demo import configuration
from grl_snam.coph.records import Observation, Reading
from grl_snam.coph.staggered_demo import execute
from grl_snam.tools.coph_golden_episode import noise
from grl_snam.tools.coph_golden_packets import recorded_draws

SEED = 91000
TICKS = 120
ARM = "reciprocal_fixed"
NUDGE = 1e-9
TIER = dict(rtol=0.0, atol=1e-6)
GRID = 0.1  # navigation.route's spacing
SPEC = dict(
    name="cars",
    seed=SEED,
    steps=TICKS,
    initials=[(0.0, 0.0, 0.0, 0.0), (4.0, 0.4, 0.0, 0.0)],
    goals=[(4.0, 0.4), (0.0, 0.0)],
    damping=0.85,
    obstacles=[(1.3, 0.3, 0.3), (2.8, 0.05, 0.25)],
    patches=[(2.0, 0.2, 0.7, 0.5, 0.9)],
)


class GoldenCarWorld(CooperativeWorld):
    """``CooperativeWorld`` with G12's ``noise`` in place of numpy's normal draws."""

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
            if np.linalg.norm(x[:2] - np.asarray(o[:2])) - o[2] <= self.cfg.sensor_radius
        )
        return Observation(
            tick * self.cfg.dt, readings, visible, tuple(x[:2]), self.cfg.sensor_radius
        )


def _learner():
    from grl_snam.ariadne.coph_cars import _learner_classes

    car, _ = _learner_classes()

    class GoldenCarLearner(car):
        """``CarLearner`` that keeps each export's bootstrap draws by packet identity, so the C++
        can inject them (D8), and traces its updates' policy attempts."""

        draws: dict = {}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.decision_tracing = True  # each update's policy attempts (records only)

        def export(self, x, recipient):
            generators: list = []
            with recorded_draws(generators):
                packet = super().export(x, recipient)
            if packet is not None:
                GoldenCarLearner.draws.setdefault(packet.identity, [g["draws"] for g in generators])
            return packet

    return GoldenCarLearner


def _world(damping_factor: float):
    class World(GoldenCarWorld):
        def __init__(self, spec, cfg):
            super().__init__(spec, cfg)
            self._damping = self._damping * damping_factor

    return World


def episode(damping_factor: float = 1.0) -> dict:
    """``execute`` on the golden field: its result."""
    learner = _learner()
    learner.draws = {}
    comm = CommunicationConfig(byte_budget=4000000, ttl=300.0)
    r = execute(
        dict(SPEC), ARM, comm, audit=False, learner_factory=learner,
        world_factory=_world(damping_factor), config=configuration(TICKS),
    )  # fmt: skip
    r["draws"] = dict(learner.draws)
    return r


def _path(route, goal) -> list:
    """A planned route: each grid point as its [i, j] on the 0.1 m grid, the goal as 'goal'."""
    out = []
    for p in route or ():
        p = [float(p[0]), float(p[1])]
        ij = [int(round(p[0] / GRID)), int(round(p[1] / GRID))]
        if p == [ij[0] * GRID, ij[1] * GRID]:
            out.append(ij)
        elif p == [float(goal[0]), float(goal[1])]:
            out.append("goal")
        else:
            raise AssertionError(f"G13: route point {p} is neither on the grid nor the goal")
    return out


def decisions(r: dict) -> dict:
    """What must match bit for bit: per tick each agent's stage, version and routed control (the
    route when it was planned); per update its packets, attempts and verdict; every send and
    delivery."""
    from grl_snam.tools.coph_golden import _hexes

    agents = []
    for i in range(2):
        goal = SPEC["goals"][i]
        controls = []
        for t in r["trajectories"][i]:
            m = t["navigation"]
            changed = bool(m.get("route_changed", False))
            controls.append([
                t["tick"], m.get("reason"), m.get("route_reason"),
                int(m.get("route_expansions", 0)), _path(m.get("route"), goal) if changed else None,
                _hexes(m.get("waypoint") or ()), int(m.get("guard_steps", 0)), changed,
                bool(m.get("stagnation", False)),
            ])  # fmt: skip
        agents.append(dict(
            samples=[[s["stage"], s["version"], bool(s["done"])] for s in r["samples"][i]],
            controls=controls,
            updates=[[u["tick"], list(u["packets"]), u["diagnostic"]["accepted"],
                      u["diagnostic"]["reason"], u["diagnostic"]["version_after"],
                      [[a["power"], a["reason"], a["accepted"]]
                       for a in (u["diagnostic"].get("policy_trace") or {}).get("attempts", [])]]
                     for u in r["updates"][i]],
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


def cars_case(check: bool = True) -> dict:
    """G13: the episode's decisions and its positions and forces."""
    from grl_snam.tools.coph_golden import _hexes, hexbits

    r = episode()
    mine = decisions(r)
    if check:
        for sign in (1.0, -1.0):
            if decisions(episode(1.0 + sign * NUDGE)) != mine:
                raise AssertionError(
                    f"G13: a {NUDGE} nudge of the world's damping changes a decision"
                )
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
            initials=[_hexes(x) for x in SPEC["initials"]],
            goals=[_hexes(g) for g in SPEC["goals"]],
            damping=hexbits(SPEC["damping"]),
            obstacles=[_hexes(o) for o in SPEC["obstacles"]],
            patches=[_hexes(p) for p in SPEC["patches"]],
            world_seeds=[SEED * 2 + i for i in range(2)],
        ),
        draws=[[e[3], r["draws"][e[3]]] for e in mine["events"] if e[0] == "sent"],
        **mine,
        out=out,
    )  # fmt: skip


CARS_RULES = """\
G13 pins the cars controller's learner (grl_snam.ariadne.coph_cars.CarLearner, demo 2b) under the
demos' runner (staggered_demo.execute, arm reciprocal_fixed), two agents for 120 ticks on an open
field: agent 0 from (0, 0) to (4, 0.4), agent 1 from (4, 0.4) to (0, 0); disks (1.3, 0.3, 0.3) and
(2.8, 0.05, 0.25); damping 0.85; the profile with steps 120, and CommunicationConfig(byte_budget=4e6,
ttl=300).

The learner: CooperativeLearner on Learner, with stage_mode 'spatial'. It advances as MazeLearner
does: when x leaves the last stage's inner box (half-sizes (w, h) (1 - overlap) / 2) along either
axis, the new stage's centre moves (w or h) (1 - overlap) along the axis it left by most
(argmax |delta| / size), towards x; it pools the kept stages' readings it contains (by uid, the
last sample_cap) and their footprints (the last footprint_cap), takes the navigation context with
planned_at -10000 as its entry context and the learner's version, keeps the last previous + 1
stages, and re-files the geometry store's shapes into every stage they intersect. Routing is
navigation.route without a maze graph: A* on the 0.1 m grid of the kept stages' boxes through the
sensor footprints (radius less 0.075), clear of the obstacles by 0.12 + safety_margin + 0.075, with
8-neighbour edges (diagonals only between two free orthogonal neighbours) priced length (1 + 4
mean risk) over the stages' supported fields (1 where unsupported), and edge_clear for the
connectors. Packets are CooperativeLearner's: no maze geometry and no terrain layer.

The world (CooperativeWorld): each agent's own plant is the exact flow at the world's damping; a
step hits when the motion comes within 0.12 of a disk's surface. The risk is 0.02 plus each patch
(x, y, sx, sy, amplitude) amplitude exp(-((qx - x)^2 / sx^2 + (qy - y)^2 / sy^2) / 2): patch
(2, 0.2, 0.7, 0.5, 0.9). Sensing as G12's: the 21 lattice points, value_j = max(0.001, risk + noise(i,
t, j)), uid '<world seed>:<t>:<j>' (world seed 2 seed + i); the disks whose surface is within
sensor_radius are visible. Times are the runner's clock, (120000 t) / 1e6.

The schedule is G12's (execute). draws: each sent packet's bootstrap resample indices,
[identity, [CooperativeLearner's]].

Decisions (BIT), per agent: samples [stage, version, done] per boundary; controls per step [tick,
reason, route_reason, route_expansions, route, waypoint, guard_steps, route_changed, stagnation],
where route is the planned path when route_changed (each point [i, j] on the 0.1 m grid, or
'goal') and null otherwise; updates [tick, packets, accepted, reason, version_after, attempts];
outcome [success, collision]. events: the runner's, in order. out (FLOAT64, atol 1e-6): per agent
the state x at each boundary, the applied force u and the nominal force per step.
"""
