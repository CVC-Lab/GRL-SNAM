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

``Squad(..., plant=...)`` is the other runtime: the squad runs in libcvc's pure-C++
``cvc::nav::sim_world``, one private belief plane per agent, and its agents drive a
physics plant's vehicles (``plant="jolt"``, a vehicle preset name, or a
:class:`~grl_snam.nav_native.JoltPlant`) or nav's C++ bicycle (``"kinematic"``).
It keeps the squad's interface — ``step``, ``run``, ``done``, the per-agent records
and :class:`SquadResult` — but not the Python path's planner: the C++ world drives
its own carrot toward each goal, and peers meet as bodies in the physics world
rather than as stamps in each other's truth. The torch path stays the default, and
the one that trains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import grl_snam.sdf_nav as sdf_nav

from . import nav_native as _native
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
    #: Per-agent top speed override. Story.vmax is one number for the whole
    #: world, which is fine until the point of the scene is that one vehicle is
    #: quicker than another -- a chase where quarry and hunter move at the same
    #: speed never resolves either way.
    vmax: float | None = None
    #: Per-agent sensor override, merged over ``Story.sensor``. Same keys
    #: :meth:`grl_snam.belief.BeliefGrid.sense` takes -- ``range_m``,
    #: ``n_rays``, ``fov_rad`` -- so a scout with a long thin cone and a
    #: vehicle with a short 360 sweep can share one scene. ``heading_rad`` is
    #: NOT settable here: the scenario supplies it live from the agent's own
    #: pose every tick, which is what makes a cone point where the agent looks.
    sensor: dict | None = None


@dataclass
class SquadResult:
    ticks: int = 0
    reached: dict = field(default_factory=dict)
    penetration: dict = field(default_factory=dict)
    #: J each agent's vehicle drew from its source (fuel or battery) — filled only on
    #: a physics plant (``Squad(plant=...)``), whose energy ledger measures it.
    energy_J: dict = field(default_factory=dict)
    #: Steps with ANY body point in truth, per agent. `penetration` is a POINT
    #: test at the rear axle and under-reports a body clipping a corner, so a
    #: footprint's whole benefit is invisible in it. Report both when comparing
    #: vehicle configurations, or the safety half of the trade goes unmeasured.
    body_penetration: dict = field(default_factory=dict)
    tracks: dict = field(default_factory=dict)


class FollowGoal:
    """A goal that is another agent, read live rather than replayed.

    A :class:`~grl_snam.fog_stories.MovingGoal` walks a path fixed before the
    run starts. A convoy cannot: the vehicle in front is reacting to a world it
    is discovering, so where it will be is not knowable in advance. This
    resolves to the leader's CURRENT position every time it is asked.

    Order matters and is deliberate. ``Squad.step`` walks its agents in
    insertion order, so a follower declared after its leader reads the
    leader's position *after* the leader has moved this tick (zero lag); one
    declared before reads last tick's (one-tick lag). Build a convoy front to
    back and it behaves like a convoy.
    """

    def __init__(self, leader_key: str):
        self.leader_key = str(leader_key)
        self._squad = None

    def bind(self, squad) -> None:
        self._squad = squad

    def position_at(self, t: float) -> tuple[float, float]:
        if self._squad is None:
            raise RuntimeError("FollowGoal was never bound to a Squad")
        return self._squad.position_of(self.leader_key)


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
        batched_planning: bool = True,
        batched_drive: bool = False,
        stagger_sense: bool = False,
        material=None,
        plant=None,
        vehicle_routes: bool | None = None,
    ):
        if not agents:
            raise ValueError("a squad needs at least one agent")
        self.story = story
        self.agents = list(agents)
        self.scenarios = {}
        # A FollowGoal is a goal that IS another agent, so it can only be
        # resolved once every scenario exists. Bind them to this squad before
        # the first step.
        for a in self.agents:
            if isinstance(a.moving_goal, FollowGoal):
                a.moving_goal.bind(self)
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
                **({} if a.vmax is None else {"vmax": float(a.vmax)}),
                # Merged, not replaced: an agent that overrides only fov_rad
                # keeps the story's range_m and n_rays.
                **({} if a.sensor is None else {"sensor": {**story.sensor, **a.sensor}}),
            )
            # One shared MaterialGrid across the squad: material is world
            # truth (the oracle setting), not per-agent belief; each
            # scenario still gets its own MaterialRuntime (its gate surface
            # composes with its own believed occupancy).
            self.scenarios[a.key] = build_scenario(
                s, model, seed=seed, truth_occ=truth_occ, prior_occ=prior_occ, material=material
            )
        self.step_i = 0

        # Stage-4 batching (PERFORMANCE.md): build every agent's SDF in one
        # threaded, GIL-releasing call between the sense and act halves of a
        # tick, instead of N serial builds. The result is bit-identical to the
        # serial path — agents couple only through the tick-start peer stamp and
        # insertion-ordered acting, both preserved — so it stays on whenever the
        # native kernels are present. All agents share the world's bounds/scale.
        self.batched_planning = bool(batched_planning)
        # Batching the vehicle rollout across agents (one torch call on [N]
        # tensors) is a separate, OPT-IN switch. Unlike the SDF/A* kernels — which
        # are integer/float64-exact and stay bit-identical — torch's batched
        # grid_sample/matmul can round differently from the serial path by up to
        # ~1 float32 ULP for some inputs, which a chaotic navigator amplifies over
        # a long episode. So it defaults OFF (the twin stays byte-exact) and is
        # enabled only when the caller wants the throughput and accepts a
        # trajectory that matches the serial one to float32 precision, not bit.
        self.batched_drive = bool(batched_drive)
        self._drive_fields = None  # persistent (N,3,H,W) stack for _batched_drive
        first = next(iter(self.scenarios.values()))
        self._bounds = first.bounds
        self._scale = first.scale
        # A peer only matters to an agent within its SENSOR range — beyond that
        # it is neither sensed nor collided with. So peer stamping / demotion
        # only ever needs nearby peers, found with a uniform spatial hash: that
        # turns the O(N^2) all-pairs sweep (the scaling wall past ~100 agents)
        # into ~O(N), bit-identically. This is the max world distance at which a
        # peer's body edge can still fall inside some agent's sensor cone.
        self._peer_range = (
            max((sc.sensor.get("range_m", 60.0) for sc in self.scenarios.values()), default=60.0)
            + 2.0 * first.cell_m
        )
        # Stagger the sense/rebuild/replan schedule so the agents do not all pay
        # the sense tick on the same frame (the worst frame, not the mean, is
        # what a 30 Hz loop must fit). Off by default: it shifts each agent's
        # schedule, so it changes trajectories and existing golden traces.
        if stagger_sense:
            for i, sc in enumerate(self.scenarios.values()):
                sc.sense_phase = i % sc.sense_every

        # The native runtime (libcvc's sim_world, optionally on a physics plant).
        self.native = None
        self.plant = None
        self._native_weights = None
        self._routes = None
        if plant is not None:
            self._native_init(plant, prior_occ, material, vehicle_routes)
        elif vehicle_routes:
            raise ValueError("vehicle_routes needs Squad(plant=...)")

    # ── the native runtime (Squad(plant=...)) ───────────────────────────────
    def _native_init(self, plant, prior_occ, material, vehicle_routes=None) -> None:
        """Build the C++ sim_world that runs this squad: the scenarios' truth, starts,
        goals, sensor and vehicle parameters, the shared policy, one belief plane per
        agent; then the plant. On a physics plant the agents with a fixed goal drive vehicle
        routes (``vehicle_routes``: None = whenever pycvc has them, True = required, False =
        nav's carrot); an agent chasing a moving goal keeps the carrot."""
        import os
        import tempfile

        from .tools.coef_export import write_coef_mlp

        if not _native.HAS_SIM_WORLD or not _native.HAS_SIM_WORLD_PLANES:
            raise RuntimeError(
                "Squad(plant=...) needs a pycvc with libcvc's P6 bindings (nav_sim_world_create "
                "with map_id, nav_sim_world_set_plant)"
            )
        if material is not None:
            raise NotImplementedError("Squad(plant=...) with a material grid is a follow-up")
        scs = list(self.scenarios.values())
        sc0 = scs[0]
        nav0 = sc0.nav
        for a, sc in zip(self.agents, scs):
            if sc.nav.kw != nav0.kw or sc.sensor != sc0.sensor:
                raise NotImplementedError(
                    f"Squad(plant=...): agent {a.key!r} overrides vmax or its sensor; the native "
                    "world takes one vehicle and one sensor for the squad"
                )
        fd, path = tempfile.mkstemp(suffix=".cvcnav")
        os.close(fd)
        write_coef_mlp(nav0.model, path)
        self._native_weights = path

        def w2n(p):
            return nav0.w2n(p)

        o = np.stack([w2n(sc.nav.pos_world()) for sc in scs]).astype(np.float32)
        goal = np.stack([w2n(sc.waypoints[0]) for sc in scs]).astype(np.float32)
        color = np.array([a.color for a in self.agents], np.float32)
        truth = np.ascontiguousarray(sc0.truth, np.uint8)
        prior = (
            np.ascontiguousarray(prior_occ, np.uint8)
            if prior_occ is not None
            else np.zeros_like(truth)
        )
        H, W = truth.shape
        mnx, mny, mxx, mxy = sc0.bounds
        k, v = nav0.kw, dict(nav0._veh)
        n = len(scs)
        self.plant = _native.make_plant(plant, bounds=sc0.bounds, obstacles=sc0.truth)
        if self.plant is not None:
            # The navigator's limits in its trained metric are more than any tire holds;
            # on a physics plant it plans with the vehicle's (never more than the story's).
            lim = self.plant.nav_limits(nav0.S)
            v["a_max"] = min(float(v["a_max"]), lim["a_max"])
            v["a_lat_max"] = min(float(v["a_lat_max"]), lim["a_lat_max"])
        h = _native._pycvc.nav_sim_world_create(
            truth,
            prior,
            path,
            o,
            goal,
            color,
            H,
            W,
            mnx,
            mny,
            mxx,
            mxy,
            nav0.cx,
            nav0.cy,
            nav0.S,
            float(sc0.sensor.get("range_m", 60.0)),
            int(sc0.sensor.get("n_rays", 240)),
            float(sc0.sensor.get("fov_rad", 2 * np.pi)),
            k["rr"],
            k["d_hat"],
            k["dt"],
            k["vmax"],
            v["L"],
            v["delta_max"],
            v["a_max"],
            v["a_lat_max"],
            v["k_steer"],
            int(nav0.nsub),
            int(bool(v["allow_reverse"])),
            float(nav0.reach_tol),
            int(sc0.sense_every),
            0,
            2.2,
            -1.4,
            8.0,
            1 if sc0.unknown == "optimistic" else 0,
            0.5,
            0.15,
            4.0,
            np.arange(n, dtype=np.int32),
            n,
        )
        self.native = _native.NativeSimWorld(h, n)
        if self.plant is not None:
            # A vehicle added inside a building's box is wedged in it for good; nav's
            # bicycle would just drive out. Refuse it rather than run a stuck agent.
            for a, sc in zip(self.agents, scs):
                if sc._truth_hit(sc.nav.pos_world()):
                    raise ValueError(
                        f"Squad(plant=...): agent {a.key!r} starts inside an obstacle cell; a "
                        "physics vehicle cannot be added there"
                    )
        self.native.set_plant(self.plant)
        self._native_dt = float(nav0.dt)
        self._native_reached = np.zeros(n, bool)
        self._native_tick = 0
        use = vehicle_routes
        if use is None:
            use = self.plant is not None and _native.HAS_VEHICLE_ROUTES
        if use:
            if self.plant is None:
                raise ValueError("vehicle_routes needs a physics plant, not nav's bicycle")
            fixed = [
                i
                for i, a in enumerate(self.agents)
                if (a.moving_goal if a.moving_goal is not None else self.story.moving_goal) is None
            ]
            if fixed:
                self._routes = _native.VehicleRoutes(self.native, sc0.bounds, fixed)
                pos, head = self.native.snapshot()[:2]
                self._routes.start(pos, head, {i: tuple(scs[i].waypoints[0]) for i in fixed})

    def position_of(self, key: str) -> tuple[float, float]:
        """Agent ``key``'s position (world metres), from whichever runtime steps it."""
        if self.native is not None:
            i = list(self.scenarios).index(key)
            pos = self.native.snapshot()[0]
            return float(pos[i, 0]), float(pos[i, 1])
        x, y = self.scenarios[key].nav.pos_world()
        return float(x), float(y)

    def _native_step(self) -> dict:
        from .scenario import StepRecord

        t = self._native_tick * self._native_dt
        keys = list(self.scenarios)
        # A moving goal is the agent's target where it is NOW (a FollowGoal reads its
        # leader's current position, the one-tick-lag case of the serial path).
        for i, a in enumerate(self.agents):
            mg = a.moving_goal if a.moving_goal is not None else self.story.moving_goal
            if mg is not None:
                gx, gy = mg.position_at(t)
                gn = self.scenarios[a.key].nav.w2n((gx, gy))
                self.native.retarget(i, float(gn[0]), float(gn[1]))
        if self._routes is not None:
            pos0, head0 = self.native.snapshot()[:2]
            self._routes.apply(t, pos0, head0)
        self.native.step()
        self._native_tick += 1
        pos, head, speed, mode, reached = self.native.snapshot()
        if self._routes is not None:
            # A route agent's snapshot `reached` is "at the follower's carrot"; it has arrived
            # when it has finished its route.
            reached = np.asarray(reached, bool).copy()
            done = self._routes.done()
            for i in self._routes.agents:
                reached[i] = bool(done[i])
        out = {}
        for i, k in enumerate(keys):
            sc = self.scenarios[k]
            x, y = float(pos[i, 0]), float(pos[i, 1])
            hit = sc._truth_hit((x, y))
            newly = bool(reached[i]) and not self._native_reached[i]
            gx, gy = sc.waypoints[0]
            out[k] = StepRecord(
                step=self._native_tick,
                x=x,
                y=y,
                heading_rad=float(head[i]),
                speed_mps=float(speed[i]),
                reached_wp=0 if newly else None,
                goal_index=0,
                mode="wall" if int(mode[i]) == 1 else "seek",
                rebuilt=False,
                truth_penetration=hit,
                body_penetration=hit,
                body_clearance_m=float("nan"),
                belief_version=0,
                goal_dist_m=float(np.hypot(gx - x, gy - y)),
            )
        self._native_reached = np.asarray(reached, bool).copy()
        return out

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
        boxes, neigh = self._peer_neighbors(half_m)
        for k, sc in self.scenarios.items():
            near = neigh[k]
            if not near:
                sc.peer_occ = None  # no peer in range: nothing to sense or hit
                continue
            mask = np.zeros(sc.truth.shape, dtype=bool)
            for j in near:
                box = boxes[j]
                if box is None:
                    continue
                r0, r1, c0, c1 = box
                mask[r0:r1, c0:c1] = True
            sc.peer_occ = mask

    def _peer_neighbors(self, half_m: float):
        """Footprint boxes for every agent + the in-range peer keys for each. A
        peer farther than sensor_range + body is never sensed or hit, so
        omitting it is bit-identical while cutting the all-pairs O(N^2) sweep to
        ~O(N). The neighbour query itself is a CGAL Kd_tree fixed-radius search
        (nav_native.neighbors) when the native kernels are present — a proper
        spatial index, robust to clustering — with a pure-Python uniform spatial
        hash as the fallback."""
        items = list(self.scenarios.items())
        keys = [k for k, _ in items]
        navs = [sc.nav for _, sc in items]
        boxes = {k: self._footprint(sc, half_m) for k, sc in items}
        r = self._peer_range + half_m

        if _native.enabled() and hasattr(_native, "neighbors"):
            # World positions in one vectorized pass, then the CGAL Kd_tree.
            o = torch.cat([nav.o for nav in navs]).numpy()  # [N,2] normalized
            n0 = navs[0]
            positions = np.empty((len(navs), 2), np.float64)
            positions[:, 0] = o[:, 0] / n0.S + n0.cx
            positions[:, 1] = o[:, 1] / n0.S + n0.cy
            idx = _native.neighbors(positions, r)
            return boxes, {keys[i]: [keys[int(j)] for j in idx[i]] for i in range(len(keys))}

        # Fallback: pure-Python uniform spatial hash (bucket = the query radius).
        pos = {k: sc.nav.pos_world() for k, sc in items}
        inv = 1.0 / max(r, 1e-9)
        buckets: dict = {}
        for k, (x, y) in pos.items():
            buckets.setdefault((int(np.floor(x * inv)), int(np.floor(y * inv))), []).append(k)
        r2 = r * r
        neigh: dict = {}
        for k, (x, y) in pos.items():
            bx, by = int(np.floor(x * inv)), int(np.floor(y * inv))
            near = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in buckets.get((bx + dx, by + dy), ()):
                        if j != k and (pos[j][0] - x) ** 2 + (pos[j][1] - y) ** 2 <= r2:
                            near.append(j)
            neigh[k] = near
        return boxes, neigh

    def _demote_peers(self, half_m: float = 4.0) -> None:
        """A peer the sensor just saw belongs in the DECAYING layer.

        Same reasoning as a Mover: baked into the static map, a moving agent
        leaves a permanent wall along its path and every later route detours
        around a corridor nobody is in.
        """
        boxes, neigh = self._peer_neighbors(half_m)
        for k, sc in self.scenarios.items():
            t = sc._t()
            for j in neigh[k]:
                box = boxes[j]
                if box is None:
                    continue
                r0, r1, c0, c1 = box
                if not sc.belief.last_visible[r0:r1, c0:c1].any():
                    continue
                sc.belief.logodds[r0:r1, c0:c1] = np.minimum(sc.belief.logodds[r0:r1, c0:c1], 0.0)
                sc.dyn.mark((r0 + r1) // 2, (c0 + c1) // 2, t, radius_cells=(r1 - r0) // 2 or 1)

    # ── the loop ────────────────────────────────────────────────────────────
    def _batch_enabled(self) -> bool:
        """The SDF build can be batched across agents (needs the native kernels)."""
        return self.batched_planning and _native.enabled() and hasattr(_native, "build_sdf_batch")

    def _can_batch_astar(self) -> bool:
        """The sense-tick A* replans can be batched across agents. Bit-identical
        (C++ A*), so it rides the same switch as the SDF batch."""
        return self._batch_enabled() and hasattr(_native, "astar_batch")

    def _can_batch_drive(self) -> bool:
        """Whether this tick will batch the vehicle rollout across agents (one
        torch call on [N] tensors — PERFORMANCE.md stage 2). Requires the opt-in
        ``batched_drive`` flag (it matches the serial path to float32 precision,
        not bit — see __init__) AND that the agents are interchangeable in that
        call: all in bicycle mode, sharing ONE model instance and identical
        vehicle/integration params, and none reading a peer's LIVE pose. A
        FollowGoal reads its leader mid-tick, which the all-pre-drive-then-drive
        order would feed a stale pose — so a convoy falls back to the serial
        act."""
        if not self.batched_drive:
            return False
        if any(isinstance(a.moving_goal, FollowGoal) for a in self.agents):
            return False
        navs = [sc.nav for sc in self.scenarios.values()]
        n0 = navs[0]
        if n0._dyn != "bicycle":
            return False
        # Material-aware agents fall back to the serial act: the batched drive
        # calls bicycle_rollout directly, bypassing SdfNavigator's per-tick
        # gate + material force wiring (batched material is a follow-up).
        if any(nav.material is not None for nav in navs):
            return False
        return all(
            nav._dyn == "bicycle"
            and nav.model is n0.model
            and nav._veh == n0._veh
            and nav.kw == n0.kw
            and nav.nsub == n0.nsub
            for nav in navs
        )

    @torch.no_grad()
    def _batched_drive(self) -> dict:
        """Roll every agent forward one step in a single batched rollout and
        distribute the new pose + metrics. Bit-identical to N serial
        ``SdfNavigator.step`` calls: the coefficient net and ``bicycle_rollout``
        are elementwise across the batch, and each agent's ``BatchedSDFField``
        plane is exactly its own field."""
        items = list(self.scenarios.items())
        navs = [sc.nav for _, sc in items]
        n0 = navs[0]
        gts = torch.cat([nav._plan_carrot() for nav in navs])  # [N,2]
        o = torch.cat([nav.o for nav in navs])  # [N,2]
        th = torch.cat([nav.th for nav in navs])  # [N]
        sp = torch.cat([nav.sp for nav in navs])  # [N]
        # Persistent (N,3,H,W) field: cat once, then overwrite only the planes of
        # the agents that rebuilt this tick (M of N with stagger) rather than
        # re-cat'ing all N every tick (~113MB/tick at 384^2). Same values as the
        # cat, so grid_sample is unchanged. A field only changes on a rebuild
        # (_rebuilt), so the un-copied planes stay in sync by construction.
        if self._drive_fields is None or self._drive_fields.shape[0] != len(navs):
            self._drive_fields = torch.cat([nav.field.field for nav in navs])
        else:
            for i, (_, sc) in enumerate(items):
                if sc._rebuilt:
                    self._drive_fields[i].copy_(navs[i].field.field[0])
        f0 = n0.field  # bounds/center/scale live on the SDFField, shared by all
        bfield = sdf_nav.BatchedSDFField(
            self._drive_fields, f0.mnx, f0.mny, f0.mxx, f0.mxy, f0.cx, f0.cy, f0.S
        )
        al, be, ga = n0.model(sdf_nav.coef_feats(bfield, o, gts))
        no, nth, nsp, _ = sdf_nav.bicycle_rollout(
            bfield, o, th, sp, gts, al, be, ga, 1, nsub=n0.nsub, **n0.kw, **n0._veh
        )
        metrics = {}
        for i, (k, sc) in enumerate(items):
            nav = sc.nav
            nav.o = no[i : i + 1]
            nav.th = nth[i : i + 1]
            nav.sp = nsp[i : i + 1]
            head = torch.stack([torch.cos(nav.th), torch.sin(nav.th)], -1)
            nav.v = nav.sp.unsqueeze(-1) * head
            nav.step_i += 1
            metrics[k] = nav._metrics(al[i : i + 1], be[i : i + 1], ga[i : i + 1])
        return metrics

    def step(self) -> dict:
        if self.native is not None:
            out = self._native_step()
            self.step_i += 1
            return out
        self._stamp_peers()
        out = {}
        sdf_batch = self._batch_enabled()
        drive_batch = self._can_batch_drive()
        if sdf_batch or drive_batch:
            # Phase 1 — sense every agent; _step_sense returns the composited
            # occupancy for those that must rebuild their SDF (else None).
            pending = [(k, sc) for k, sc in self.scenarios.items() if sc._step_sense() is not None]
            # Phase 2 — build all pending SDFs in one threaded call (native only).
            if sdf_batch and pending:
                occs = np.stack([sc._pending_occ for _, sc in pending])
                fields = _native.build_sdf_batch(occs, self._bounds, self._scale)
                for (_, sc), (phi, nx_g, ny_g) in zip(pending, fields):
                    sc._pending_field = sc._finalize_field(phi, nx_g, ny_g)
            # Phase 2.5 — batch every rebuilding agent's sense-tick A* replan into
            # one threaded astar_batch. Bit-identical to the per-agent plan(): the
            # C++ A* is the same, and _replan_commit is the same hysteresis tail.
            # Only the plain (cost-free) case batches; a route_cost_fn OR an
            # attached material grid (whose risk raster is a cost) falls back
            # to the per-agent replan in _act_pre_drive.
            if self._can_batch_astar() and all(
                sc.route_cost_fn is None and sc.material is None for sc in self.scenarios.values()
            ):
                reb = [sc for sc in self.scenarios.values() if sc._sense_rebuild and sc.use_planner]
                if reb:
                    # Batch the dilation too (all agents share inflate_cells),
                    # then hand each its inflated grid into _replan_inputs.
                    occs = np.stack([sc._pending_occ for sc in reb])
                    grids = _native.inflate_batch(occs, reb[0].planner.inflate_cells)
                    inp = [sc._replan_inputs(grids[i]) for i, sc in enumerate(reb)]
                    starts = np.array([s for _, s, _, _ in inp], np.int32)
                    goals = np.array([gc for _, _, gc, _ in inp], np.int32)
                    routes = _native.astar_batch(grids, starts, goals, None)
                    for sc, cells in zip(reb, routes):
                        sc._replan_commit(cells)
                        sc._sense_replanned = True
            if drive_batch:
                # Phases 3-5 — pre-drive all (no FollowGoal, so nobody reads a
                # peer's live pose), roll the whole squad forward in one batched
                # rollout, then post-drive all.
                for _, sc in self.scenarios.items():
                    sc._act_pre_drive()
                metrics = self._batched_drive()
                for k, sc in self.scenarios.items():
                    out[k] = sc._act_post_drive(metrics[k])
            else:
                # Per-agent act, in insertion order (so a FollowGoal convoy reads
                # its leader's post-step pose exactly as in the serial path); each
                # agent's SDF build was already batched into _pending_field.
                for k, sc in self.scenarios.items():
                    out[k] = sc._step_act()
        else:
            for k, sc in self.scenarios.items():
                out[k] = sc.step()
        self._demote_peers()
        self.step_i += 1
        return out

    @property
    def done(self) -> bool:
        if self.native is not None:
            return bool(self._native_reached.all())
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
        bpen = {k: 0 for k in self.scenarios}
        best = {k: float("inf") for k in self.scenarios}
        since = 0
        for _ in range(max_steps):
            recs = self.step()
            improved = False
            for k, r in recs.items():
                tracks[k].append((r.x, r.y))
                pen[k] += int(r.truth_penetration)
                bpen[k] += int(r.body_penetration)
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
        res.body_penetration = bpen
        if self.native is not None:
            res.reached = {k: bool(self._native_reached[i]) for i, k in enumerate(self.scenarios)}
            if self.plant is not None:
                res.energy_J = {k: self.plant.energy_J(i) for i, k in enumerate(self.scenarios)}
        else:
            res.reached = {k: sc.done for k, sc in self.scenarios.items()}
        return res


def attach_clearance_routing(squad: Squad, d_safe: float = 6.0, gamma: float = 1.5) -> None:
    """Bias every agent's global A* route toward standoff from obstacles.

    Sets each scenario's ``route_cost_fn`` (the sanctioned cost-surface seam) to a
    fixed :func:`sdf_nav.clearance_cost` surcharge over its initial planning
    occupancy — so the route the local drive follows keeps clearance instead of
    hugging walls. Call after constructing the squad and before stepping. This is a
    lever for demo route quality: the learned local policy is already near its
    ceiling, and a higher-standoff spine lifts route-guided reach on the procedural
    city squad ~0.80 -> ~0.90 (n=20 x 5 seeds, d_safe=6) AND gives smoother,
    wall-avoiding paths. **Budget-sensitive**: the standoff route is longer, so it
    only helps when the run has tick headroom to finish it — under a tight budget it
    can regress below the shortest path. A fixed surcharge suits a static map; for a
    map that keeps changing, recompute per replan instead. Passing route_cost_fn
    also turns off A* batching (each agent's search is now cost-specific).
    """
    for sc in squad.scenarios.values():
        cg = sdf_nav.clearance_cost(sc._compose_occ(), d_safe, gamma)
        sc.route_cost_fn = (lambda c: (lambda: c))(cg)
