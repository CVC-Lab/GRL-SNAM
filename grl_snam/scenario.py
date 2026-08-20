"""The reproducible fog-of-war scenario: one sim loop, no rendering.

This is the unit the live demo and the offscreen capture are both thin views
over — sense, maybe rebuild, drive, apply events — so the *simulation* can be
tested headlessly and the renderers cannot disagree about what happened.

The shape follows the roadmap's scenario definition (§22.5.2): scene + initial
state + events + termination, deterministic given the same inputs. Events fire
on *step index* (the world-time analog at fixed dt), mutating ground truth or
marking the dynamic layer; the agent only ever learns of them through its own
sensor.

The loop each step:

    1. events due this step mutate TRUTH (a building demolished, a blocker
       raised) or mark the dynamic layer (a unit at a position);
    2. the sensor raycasts TRUTH into BELIEF (every ``sense_every`` steps —
       sensing is the expensive part, and a real sensor has a frame rate);
    3. if any believed cell flipped, the SDF is rebuilt from the composited
       belief + dynamic layer and swapped under the navigator — the rebuild
       *is* the replan;
    4. the navigator advances one step toward the current waypoint (bicycle
       dynamics by default: acceleration, turning radius, corner braking);
    5. collision is scored against TRUTH — penetrating a wall the agent did
       not know about still counts, which is exactly the honesty a fog demo
       needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import sdf_nav

from .belief import BeliefGrid, DynamicLayer, composite_occupancy
from .nav import SdfNavigator
from .planner import BeliefRoutePlanner, inflate


@dataclass
class Event:
    """A ground-reality change the agent does not get told about."""

    step: int
    kind: str  # "remove_rect" | "add_rect" | "unit_at"
    # rect: (r0, r1, c0, c1) in cells; unit: (r, c)
    args: tuple = ()


@dataclass
class StepRecord:
    step: int
    x: float
    y: float
    heading_rad: float
    speed_mps: float
    reached_wp: int | None  # waypoint index reached THIS step (pre-advance), else None
    goal_index: int
    mode: str
    rebuilt: bool
    truth_penetration: bool
    belief_version: int
    goal_dist_m: float


@dataclass
class ScenarioResult:
    records: list[StepRecord] = field(default_factory=list)
    waypoints_reached: int = 0
    rebuilds: int = 0
    truth_penetration_steps: int = 0

    @property
    def positions(self) -> np.ndarray:
        return np.array([[r.x, r.y] for r in self.records], np.float32)


class FogScenario:
    """Drive waypoints under incomplete information, deterministically."""

    def __init__(
        self,
        truth_occ: np.ndarray,
        bounds,
        scale: float,
        model,
        meta: dict,
        waypoints,
        *,
        prior_occ: np.ndarray | None = None,
        prior_confidence: float = 4.0,
        events: list[Event] | None = None,
        unknown: str = "optimistic",
        sense_every: int = 5,
        sense_phase: int = 0,
        sensor: dict | None = None,
        dynamics: str = "bicycle",
        vehicle: dict | None = None,
        unit_ttl_s: float = 4.0,
        reach_tol: float = 0.8,
        use_planner: bool = True,
        route_lookahead_m: float = 14.0,
        inflate_m: float = 6.0,
        movers=(),
        moving_goal=None,
    ):
        # `truth` is the STATIC world. Movers are stamped on top of it each
        # tick into `truth_now`, which is what the sensor and the collision
        # check both use -- a moving vehicle has to be seen to be known.
        self.truth = truth_occ.astype(bool).copy()
        self.movers = tuple(movers or ())
        self.moving_goal = moving_goal
        self.truth_now = self.truth.copy()
        self.bounds = tuple(float(b) for b in bounds)
        self.scale = float(scale)
        self.meta = meta
        self.model = model
        self.events = sorted(events or [], key=lambda e: e.step)
        self._next_event = 0
        self.unknown = unknown
        self.sense_every = max(1, int(sense_every))
        # Phase offset for the sense/rebuild/replan schedule. A Squad staggers
        # agents (agent i by i mod sense_every) so they do not all carry the
        # whole sense-tick cost on the same frame — a real-time correctness
        # requirement (PERFORMANCE.md), even though it saves nothing on average.
        # 0 (the default) is the original synchronized schedule.
        self.sense_phase = int(sense_phase)
        self.sensor = dict(range_m=60.0, n_rays=240)
        self.sensor.update(sensor or {})
        # Set by a caller to return a (ny, nx) float raster of per-cell route
        # surcharges, or None. Called once per re-plan, not per tick.
        self.route_cost_fn = None

        self._world_dt = float(meta["dt"])  # world seconds per step (fixed quantum)
        self.step_i = 0

        self.belief = BeliefGrid(self.truth.shape, self.bounds)
        if prior_occ is not None:
            # A map made earlier: confidently wrong where reality has moved on.
            self.belief.logodds[:] = np.where(
                prior_occ, prior_confidence, -prior_confidence
            ).astype(np.float32)
        self.dyn = DynamicLayer(self.truth.shape, ttl_s=unit_ttl_s)

        self.waypoints = [np.asarray(w, np.float32) for w in waypoints]
        assert self.waypoints, "a scenario needs at least one waypoint"
        self.wp_i = 0
        # True once the FINAL user waypoint is reached. The right "are we
        # there" flag for callers: nav.reached refers to the nav's own goal,
        # which under route tracking is a sub-goal ~lookahead metres ahead —
        # inside reach_tol by construction, i.e. true almost always.
        self.done = False
        # Set by Squad._stamp_peers; None for a single-agent scenario.
        self.peer_occ = None

        # The global spine: A* over BELIEF, replanned whenever belief
        # changes. The reactive navigator alone is a local law -- a fresh wall
        # square across the route is a potential-field minimum it can only
        # wall-follow out of; the route spine is what turns "discovered a
        # blocker" into "computed a new best path" (the measured-on-Austin
        # architecture, here fog-aware).
        self.use_planner = bool(use_planner)
        self.route_lookahead_m = float(route_lookahead_m)
        # Inflation must clear the BARRIER's influence radius, not just the
        # vehicle body: a route hugging a 2 m boundary while the IPC barrier
        # reaches ~7 m leaves tracking and repulsion in a stalemate at every
        # corner.
        #
        # Specified in METRES, not cells. It was hard-coded at 3 cells, which
        # is 6 m at the fog stories' 2.08 m/cell -- but cells are not a fixed
        # size. On a 1200 m city raster at 256 the same 3 cells is 14 m, wide
        # enough to close every street: measured there, inflate 1 routes in
        # 1847 m, 2 in 2006 m, and 3 finds NO ROUTE AT ALL.
        cell_m = (self.bounds[2] - self.bounds[0]) / (self.truth.shape[1] - 1)
        self.cell_m = cell_m
        self.inflate_cells = max(1, round(float(inflate_m) / cell_m))
        self.planner = BeliefRoutePlanner(
            self.bounds, self.truth.shape, inflate_cells=self.inflate_cells
        )
        self.route: list | None = None
        self.no_route = False
        self._last_dyn = None
        # Carried between the two halves of a tick (_step_sense -> _step_act) so
        # a Squad can batch every agent's SDF build in one threaded call.
        self._pending_occ = None
        self._sense_rebuild = False
        self._pending_field = None

        self.nav = SdfNavigator(
            self._build_field(),
            model,
            meta,
            reach_tol=reach_tol,
            dynamics=dynamics,
            vehicle=vehicle,
        )

    # ── internals ────────────────────────────────────────────────────────────
    def _t(self) -> float:
        """World time — derived from the step count, never accumulated."""
        return self.step_i * self._world_dt

    def _compose_occ(self) -> np.ndarray:
        return composite_occupancy(self.belief, self.dyn, self._t(), unknown=self.unknown)

    def _finalize_field(self, phi, nx_g, ny_g) -> sdf_nav.SDFField:
        # An occupancy grid with few/no walls yields astronomically large
        # distances (an EMPTY one gives ~1e9): far outside the +-region regime
        # CoefMLP was trained on, its features saturate the net and the
        # coefficients collapse — the vehicle simply parks. Clip to the
        # normalized working range; beyond it clearance is "far" either way.
        region_n = float(self.meta["region"]) * self.scale
        np.clip(phi, -2.0 * region_n, 2.0 * region_n, out=phi)
        center = self.meta["center"]
        return sdf_nav.SDFField(phi, nx_g, ny_g, self.bounds, center, self.scale)

    def _build_field(self, occ: np.ndarray | None = None) -> sdf_nav.SDFField:
        if occ is None:
            occ = self._compose_occ()
        phi, nx_g, ny_g = sdf_nav.build_sdf(occ, self.bounds, self.scale)
        return self._finalize_field(phi, nx_g, ny_g)

    def _apply_due_events(self) -> bool:
        touched_truth = False
        while (
            self._next_event < len(self.events)
            and self.events[self._next_event].step <= self.step_i
        ):
            ev = self.events[self._next_event]
            self._next_event += 1
            if ev.kind == "remove_rect":
                r0, r1, c0, c1 = ev.args
                self.truth[r0:r1, c0:c1] = False
                touched_truth = True
            elif ev.kind == "add_rect":
                r0, r1, c0, c1 = ev.args
                self.truth[r0:r1, c0:c1] = True
                touched_truth = True
            elif ev.kind == "unit_at":
                r, c = ev.args
                self.dyn.mark(r, c, self._t(), radius_cells=1)
            else:
                raise ValueError(f"unknown event kind {ev.kind!r}")
        return touched_truth

    def _stamp_movers(self) -> None:
        """Rebuild ``truth_now`` for this tick: the static world plus every
        mover's current footprint."""
        self.truth_now = self.truth.copy()
        # Peers are stamped by the Squad BEFORE this runs, so they have to be
        # re-applied after the reset or they vanish (see Squad._stamp_peers).
        if self.peer_occ is not None:
            np.logical_or(self.truth_now, self.peer_occ, out=self.truth_now)
        if not self.movers:
            return
        t = self._t()
        for m in self.movers:
            x, y = m.position_at(t)
            r, c = self.belief.world_to_cell(x, y)
            cw = (self.bounds[2] - self.bounds[0]) / (self.truth.shape[1] - 1)
            rad = max(1, int(round(m.half_m / cw)))
            r0, r1 = max(0, r - rad), min(self.truth.shape[0], r + rad + 1)
            c0, c1 = max(0, c - rad), min(self.truth.shape[1], c + rad + 1)
            self.truth_now[r0:r1, c0:c1] = True

    def _demote_movers_to_dynamic(self) -> None:
        """A mover the sensor just saw belongs in the DECAYING layer, not in
        the static map.

        Without this a moving vehicle smears a permanent wall along its path:
        every sweep bakes its current cells into log-odds and nothing ever
        removes them. Marking the cells dynamic (they expire) and clearing the
        static belief there is what makes a moving obstacle behave like one.
        """
        if not self.movers:
            return
        t = self._t()
        cw = (self.bounds[2] - self.bounds[0]) / (self.truth.shape[1] - 1)
        for m in self.movers:
            x, y = m.position_at(t)
            r, c = self.belief.world_to_cell(x, y)
            if not self.belief.in_bounds(r, c):
                continue
            rad = max(1, int(round(m.half_m / cw)))
            r0, r1 = max(0, r - rad), min(self.truth.shape[0], r + rad + 1)
            c0, c1 = max(0, c - rad), min(self.truth.shape[1], c + rad + 1)
            if not self.belief.last_visible[r0:r1, c0:c1].any():
                continue  # not seen this sweep: the agent still does not know
            self.belief.logodds[r0:r1, c0:c1] = np.minimum(self.belief.logodds[r0:r1, c0:c1], 0.0)
            self.dyn.mark(r, c, t, radius_cells=rad)

    def _truth_hit(self, pos_world) -> bool:
        r, c = self.belief.world_to_cell(pos_world[0], pos_world[1])
        return bool(self.belief.in_bounds(r, c) and self.truth_now[r, c])

    def _replan_route(self, *, force: bool = False, occ: np.ndarray | None = None):
        """Re-plan the belief-space route from HERE to the active waypoint.

        With HYSTERESIS: when two detours cost nearly the same (a blocker
        square across the route), every belief update flips which one looks
        shorter and a naive replanner oscillates between the wall's ends
        forever. Keep the committed route unless belief actually invalidates
        it or the fresh plan is decisively (>20%) shorter."""
        if not self.use_planner:
            return
        if occ is None:
            occ = self._compose_occ()
        here = tuple(self.nav.pos_world())
        goal = tuple(self.waypoints[self.wp_i])
        # Optional per-cell route surcharge, supplied by whoever owns the
        # meaning. Base GRL-SNAM never sets this and never interprets it: it is
        # the seam a downstream package (e.g. an RF-aware planner) plugs a cost
        # surface into without this module growing a dependency on the domain.
        cost = self.route_cost_fn() if self.route_cost_fn is not None else None
        # Inflate ONCE and reuse for both plan() and route_valid() — the
        # dilation is the single biggest per-tick cost at scale, and both would
        # otherwise recompute it from the same occ. Same grid => bit-identical.
        grid = inflate(occ, self.planner.inflate_cells)
        fresh = self.planner.plan(occ, here, goal, cost=cost, inflated=grid)
        if fresh is None:
            self.no_route = True
            self.route = None
            return
        self.no_route = False
        if force or not self.planner.route_valid(occ, self.route, inflated=grid):
            self.route = [here] + fresh[1:] if len(fresh) > 1 else fresh
            return
        keep_len = self.planner.route_length([here] + self.route[1:])
        if self.planner.route_length(fresh) < 0.8 * keep_len:
            self.route = fresh

    def _replan_inputs(self, grid=None):
        """The batchable half of the sense-tick replan: return this agent's A*
        inputs (grid, start cell, goal cell, cost) so a Squad can run one
        `astar_batch` across all rebuilding agents. `grid` may be a
        pre-inflated grid (from a batched `inflate_batch`); otherwise this
        inflates `_pending_occ`. Stashes what `_replan_commit` needs; returns
        None if not planning."""
        if not self.use_planner:
            return None
        here = tuple(self.nav.pos_world())
        goal = tuple(self.waypoints[self.wp_i])
        cost = self.route_cost_fn() if self.route_cost_fn is not None else None
        if grid is None:
            grid = inflate(self._pending_occ, self.planner.inflate_cells)
        self._replan_grid = grid
        self._replan_here = here
        self._replan_cost = cost
        return grid, self.planner._w2c(*here), self.planner._w2c(*goal), cost

    def _replan_commit(self, cells):
        """Finish the sense-tick replan from the batched A* cell path (or None).
        Byte-for-byte the force=False tail of `_replan_route`."""
        grid = self._replan_grid
        fresh = self.planner.route_from_cells(grid, cells, self._replan_cost)
        if fresh is None:
            self.no_route = True
            self.route = None
            return
        self.no_route = False
        here = self._replan_here
        if not self.planner.route_valid(self._pending_occ, self.route, inflated=grid):
            self.route = [here] + fresh[1:] if len(fresh) > 1 else fresh
            return
        keep_len = self.planner.route_length([here] + self.route[1:])
        if self.planner.route_length(fresh) < 0.8 * keep_len:
            self.route = fresh

    def _route_subgoal(self):
        """The point on the route the local controller should chase: project
        the vehicle onto the polyline, then walk the lookahead distance ahead
        of the projection — corners are anticipated, not discovered (the
        Austin spine's index+K idea, but geometric)."""
        if not self.route or len(self.route) < 2:
            return tuple(self.waypoints[self.wp_i])
        pts = np.asarray(self.route, np.float64)
        p = np.asarray(self.nav.pos_world(), np.float64)
        # nearest point on the polyline (segment-wise projection)
        best_d, best_seg, best_t = float("inf"), 0, 0.0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            tproj = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            d = float(np.linalg.norm(a + tproj * ab - p))
            if d < best_d:
                best_d, best_seg, best_t = d, i, tproj
        # walk lookahead metres ahead of the projection
        remain = self.route_lookahead_m
        a = pts[best_seg] + best_t * (pts[best_seg + 1] - pts[best_seg])
        for i in range(best_seg, len(pts) - 1):
            b = pts[i + 1]
            seg = float(np.linalg.norm(b - a))
            if seg >= remain and seg > 1e-9:
                return tuple(a + (remain / seg) * (b - a))
            remain -= seg
            a = b
        return tuple(pts[-1])

    # ── the loop ─────────────────────────────────────────────────────────────
    def start(self, start_world):
        self.nav.start(tuple(start_world), tuple(self.waypoints[0]))
        self._replan_route()
        return self

    def _step_sense(self) -> np.ndarray | None:
        """Pre-plan half of a tick: apply events, stamp movers, sense, and
        decide whether the SDF must be rebuilt. Returns the composited
        occupancy to build the field from when a rebuild is due (so a Squad can
        batch every agent's SDF build in one threaded call between the halves),
        else None. Identical in effect to the sense block of the old step()."""
        self._apply_due_events()
        self._stamp_movers()
        self._pending_occ = None
        self._sense_rebuild = False
        self._pending_field = None
        self._sense_replanned = False  # set by a Squad that batched this tick's A*

        if (self.step_i + self.sense_phase) % self.sense_every == 0:
            v0 = self.belief.version
            self.belief.sense(
                self.truth_now,
                self.nav.pos_world(),
                heading_rad=float(self.nav.th[0]),
                **self.sensor,
            )
            # The dynamic layer changes the planning surface without touching
            # belief.version, so it has to be able to trigger a rebuild too --
            # but only when it actually CHANGED. Rebuilding whenever it merely
            # holds something meant a scene with moving vehicles replanned on
            # every sense tick forever (measured: 177 replans in 739 ticks),
            # and a route that keeps moving under the local controller is what
            # made the vehicle loop.
            self._demote_movers_to_dynamic()
            dyn_now = self.dyn.occupancy(self._t())
            dyn_changed = self._last_dyn is None or not np.array_equal(dyn_now, self._last_dyn)
            if self.belief.version != v0 or dyn_changed:
                self._last_dyn = dyn_now
                # Composited once here and reused for BOTH the field build and
                # the replan below (they are the same occupancy), which also
                # dedupes the array the old code built twice.
                self._pending_occ = self._compose_occ()
                self._sense_rebuild = True
        return self._pending_occ

    def _act_pre_drive(self) -> None:
        """The part of the act half BEFORE the vehicle rollout: build the field
        (unless a Squad batched it into self._pending_field) and replan, retarget
        a moving goal, and aim the local controller at the route sub-goal. Leaves
        self.nav ready to drive; a Squad rolls every agent forward in one batched
        rollout between _act_pre_drive and _act_post_drive."""
        self._rebuilt = False
        if self._sense_rebuild:
            if self._pending_field is not None:
                self.nav.field = self._pending_field  # batched by the Squad
            else:
                self.nav.field = self._build_field(self._pending_occ)
            if not self._sense_replanned:  # else a Squad already batched the A*
                self._replan_route(occ=self._pending_occ)
            self._rebuilt = True
        self._pending_occ = None
        self._pending_field = None
        self._sense_rebuild = False

        if self.moving_goal is not None:
            # The goal moves, so the waypoint the planner routes to has to move
            # with it. track_goal (not set_goal) further down keeps the local
            # controller's escape state across the retarget.
            gx, gy = self.moving_goal.position_at(self._t())
            self.waypoints[self.wp_i] = np.asarray((gx, gy), np.float32)
            if (self.step_i + self.sense_phase) % self.sense_every == 0:
                self._replan_route(force=True)

        if self.use_planner and self.route:
            # The route is the moving target; track_goal (not set_goal) so the
            # wall-follow escape machinery survives per-step retargeting.
            self.nav.track_goal(self._route_subgoal(), goal_index=self.wp_i)

    def _act_post_drive(self, m) -> StepRecord:
        """The part of the act half AFTER the vehicle rollout: waypoint
        bookkeeping and the StepRecord, from the metrics `m` the drive produced."""
        self.step_i += 1

        # "Reached" means the USER waypoint, not the route sub-goal the
        # local controller happens to be chasing.
        wp = self.waypoints[self.wp_i]
        dist_wp_m = float(np.hypot(m.x - wp[0], m.y - wp[1]))
        wp_reached = dist_wp_m < (self.nav.reach_tol / self.scale)
        reached_wp = int(self.wp_i) if wp_reached else None
        if wp_reached and self.wp_i + 1 < len(self.waypoints):
            self.wp_i += 1
            # set_goal resets nav.reached — which is why the reach is recorded
            # BEFORE advancing, or intermediate waypoints are never counted.
            self.nav.set_goal(tuple(self.waypoints[self.wp_i]), goal_index=self.wp_i)
            # force: the committed route targets the OLD waypoint. The keep/
            # fresh hysteresis compares routes to DIFFERENT goals otherwise —
            # and the old route (ending where the vehicle stands) always wins,
            # pinning the sub-goal to the reached waypoint forever.
            self._replan_route(force=True)
        elif wp_reached:
            self.done = True
            # Terminal: brake and hold heading. Without this the vehicle
            # keeps chasing a goal it is sitting on and pirouettes.
            self.nav.park()

        pen = self._truth_hit((m.x, m.y))
        return StepRecord(
            step=self.step_i,
            x=m.x,
            y=m.y,
            heading_rad=m.heading_rad,
            speed_mps=m.speed_mps,
            reached_wp=reached_wp,
            goal_index=self.wp_i,
            mode=m.mode,
            rebuilt=self._rebuilt,
            truth_penetration=pen,
            belief_version=self.belief.version,
            goal_dist_m=m.goal_dist_m,
        )

    def _step_act(self) -> StepRecord:
        """Act half of a tick, run serially (pre-drive, drive, post-drive). A
        Squad calls the three parts separately so it can batch the drive."""
        self._act_pre_drive()
        m = self.nav.step()
        return self._act_post_drive(m)

    def step(self) -> StepRecord:
        """One tick. The two halves run back-to-back here (behaviour identical
        to the pre-split loop); a Squad calls them separately so it can batch
        the SDF build across agents in between."""
        self._step_sense()
        return self._step_act()

    def run(self, max_steps: int = 4000, *, stop_when_done: bool = True) -> ScenarioResult:
        res = ScenarioResult()
        reached = set()
        for _ in range(max_steps):
            rec = self.step()
            res.records.append(rec)
            if rec.rebuilt:
                res.rebuilds += 1
            if rec.truth_penetration:
                res.truth_penetration_steps += 1
            if rec.reached_wp is not None:
                reached.add(rec.reached_wp)
            if stop_when_done and self.done:
                break
            if self.no_route and stop_when_done:
                break  # surfaced, not papered over: the goal is unroutable in belief
        res.waypoints_reached = len(reached)
        return res
