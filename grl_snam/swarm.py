"""A vectorized, shared-belief swarm — the structure-of-arrays end state.

:class:`~grl_snam.squad.Squad` gives every agent its own belief, its own SDF
field and its own scenario object, then batches the *kernels* (SDF build, A*,
inflate, the vehicle rollout) across those objects. That reaches ~256 agents at
30 Hz, and there it hits a wall built of three things the batching can't touch:

  1. the map cost is O(N) — one EDT + one 4.3 MB field *per agent*;
  2. the carrot state machine is a per-agent Python loop (``squad.py`` calls
     ``[nav._plan_carrot() for nav in navs]``);
  3. the sense / act glue walks a dict of ``FogScenario`` objects every tick.

``Swarm`` collapses all three. It is one struct-of-arrays: N agents share ONE
belief, ONE ``SDFField`` (so ``field.sample`` broadcasts all N positions against
a single ``[1,3,H,W]`` texture — the map cost is O(1) and the memory is flat in
N), and the whole tick — carrot FSM, coefficient net, bicycle rollout, metrics,
waypoint bookkeeping — runs as masked ``[N]`` torch ops with no per-agent Python
anywhere on the drive path. On a non-sense tick nothing but the ``[N]`` drive
executes, which is what lets thousands of vehicles fit a 60 Hz frame.

This is a DIFFERENT simulator from the Squad, on purpose: sharing the belief
erases per-agent fog of war (see the module docstring of ``squad.py`` for why
that fog is the whole point of the fidelity twin). ``Swarm`` is the
thousands-of-agents render/reactivity path; ``Squad`` stays the byte-exact,
legible-at-a-handful-of-agents twin. The carrot FSM here is a faithful
vectorization of :meth:`grl_snam.nav.SdfNavigator._plan_carrot`, validated agent
for agent against N serial navigators sharing one field (``tests`` +
``bench_swarm.py``), so a single agent in a ``Swarm`` drives exactly as it does
in a ``Squad`` — only the belief it drives against is shared.

The ``[N]`` columns and the immutable :meth:`snapshot` are what
:class:`grl_snam.sim_thread.SimThread` runs on a background thread and publishes
lock-free to a renderer; see that module for the concurrency model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import sdf_nav

from .belief import BeliefGrid, DynamicLayer, composite_occupancy
from .fog_stories import Story, build_scenario

SEEK, WALL = 0, 1


@dataclass(frozen=True)
class Snapshot:
    """An immutable, renderer-facing view of the swarm at one tick.

    Every array is freshly allocated the moment it is built and never mutated
    afterwards, so a reader on another thread sees a whole consistent frame or
    the previous one — never a torn mix (see :mod:`grl_snam.sim_thread`). Poses
    are in WORLD metres; ``mode`` is the integer FSM state (stringify at the HUD
    only for inspected agents).
    """

    gen: int
    tick: int
    world_t: float
    pos: np.ndarray  # f32[N,2] world
    heading: np.ndarray  # f32[N]
    speed: np.ndarray  # f32[N]  (world m/s)
    color: np.ndarray  # f32[N,3]
    mode: np.ndarray  # i8[N]  0=seek 1=wall
    goal: np.ndarray  # f32[N,2] world
    active: np.ndarray  # bool[N]
    reached: np.ndarray  # bool[N]
    field_ver: int
    field_ref: object  # the shared [1,3,H,W] SDF texture, by REFERENCE
    occ_ref: object  # the shared occupancy raster (bool[H,W]), by reference


class Swarm:
    """N agents over one shared belief, advanced as a single struct-of-arrays.

    ``story`` / ``specs`` / ``model`` mirror :class:`grl_snam.squad.Squad`
    (``specs`` are :class:`~grl_snam.squad.AgentSpec`). Only the shared-belief
    mode is implemented here — the O(1)-map swarm; the private-belief SoA is the
    fidelity twin's job (``Squad``).

    ``sense_every`` gates the shared belief update + SDF rebuild (the only O(N)
    /O(map) work); on the other ticks the swarm does nothing but the vectorized
    drive. ``reach_tol`` is in normalized units (as ``SdfNavigator``).
    """

    def __init__(
        self,
        story: Story,
        specs,
        model=None,
        *,
        seed: int = 0,
        truth_occ=None,
        prior_occ=None,
        sense_every: int = 4,
        reach_tol: float = 0.8,
        device: str = "cpu",
    ):
        if not specs:
            raise ValueError("a swarm needs at least one agent")
        self.story = story
        self.specs = list(specs)
        self.dev = torch.device(device)
        self.sense_every = max(1, int(sense_every))
        self.reach_tol = float(reach_tol)

        # One template FogScenario gives us the whole shared world — meta,
        # bounds/scale, the truth raster, and (the objects we actually share) a
        # BeliefGrid, a DynamicLayer, a route planner and an SDFField built by
        # the exact tested path. We drive N agents against these directly.
        import dataclasses

        s0 = dataclasses.replace(story, start=self.specs[0].start, waypoints=(self.specs[0].goal,))
        tpl = build_scenario(s0, model, seed=seed, truth_occ=truth_occ, prior_occ=prior_occ)
        self._tpl = tpl
        self.model = tpl.model
        self.meta = tpl.meta
        self.bounds = tpl.bounds
        self.scale = float(tpl.scale)
        self.truth = tpl.truth
        self.belief: BeliefGrid = tpl.belief  # SHARED across all agents
        self.dyn: DynamicLayer = tpl.dyn
        self.sensor = dict(tpl.sensor)
        self.unknown = tpl.unknown
        self.cell_m = float(tpl.cell_m)

        # The shared field + its world<->normalized constants (one texture that
        # every agent's position samples against).
        self.field: sdf_nav.SDFField = tpl.nav.field
        self.S = float(self.meta["scale"])
        self.cx, self.cy = (float(c) for c in self.meta["center"])

        # Vehicle + integration params (shared; the drive is one call on [N]).
        n = tpl.nav
        self.kw = dict(n.kw)
        self.veh = dict(n._veh)
        self.nsub = int(n.nsub)
        self.dt = float(self.meta["dt"])
        self.a_max = float(self.veh["a_max"])

        self._build_soa()

        self.gstep = 0
        self.field_ver = 0
        self._gen = 0
        self._last_belief_ver = self.belief.version
        self._occ = self._tpl._compose_occ()  # current planning raster (shared)

    # ── construction ─────────────────────────────────────────────────────────
    def _build_soa(self) -> None:
        """Lay every agent's pose + FSM scalar out as an ``[N]`` column."""
        N = len(self.specs)
        self.N = N
        dev = self.dev
        w2n = self._w2n_np
        o = np.stack([w2n(s.start) for s in self.specs]).astype(np.float32)  # [N,2]
        g = np.stack([w2n(s.goal) for s in self.specs]).astype(np.float32)  # [N,2]
        self.o = torch.from_numpy(o).to(dev)
        self.goal = torch.from_numpy(g).to(dev)
        gd = g - o
        self.th = torch.from_numpy(np.arctan2(gd[:, 1], gd[:, 0]).astype(np.float32)).to(dev)
        self.sp = torch.zeros(N, device=dev)
        self.color = torch.tensor(
            [list(s.color) for s in self.specs], dtype=torch.float32, device=dev
        )

        dist0 = torch.from_numpy(np.linalg.norm(gd, axis=1).astype(np.float32)).to(dev)
        z = torch.zeros(N, device=dev)
        self.stall = torch.zeros(N, dtype=torch.int64, device=dev)
        self.mode = torch.full((N,), SEEK, dtype=torch.int64, device=dev)
        self.turn = torch.ones(N, device=dev)
        self.dhit = z.clone()
        self.best = dist0.clone()
        self.init = dist0.clamp_min(1e-6)
        self.wall_entry = torch.zeros(N, 2, device=dev)
        self.we_valid = torch.zeros(N, dtype=torch.bool, device=dev)
        self.tracking = torch.zeros(N, dtype=torch.bool, device=dev)
        self.pos_hist = torch.zeros(N, 40, 2, device=dev)
        self.hist_count = torch.zeros(N, dtype=torch.int64, device=dev)
        self.parked = torch.zeros(N, dtype=torch.bool, device=dev)
        self.reached = torch.zeros(N, dtype=torch.bool, device=dev)
        self.active = torch.ones(N, dtype=torch.bool, device=dev)
        self._arange = torch.arange(N, device=dev)

    # ── world <-> normalized (vectorized) ────────────────────────────────────
    def _w2n_np(self, p):
        return np.array([(p[0] - self.cx) * self.S, (p[1] - self.cy) * self.S], np.float32)

    def n2w(self, on: torch.Tensor) -> torch.Tensor:
        """``[N,2]`` normalized -> ``[N,2]`` world."""
        out = on / self.S
        out[:, 0] += self.cx
        out[:, 1] += self.cy
        return out

    # ── the tick ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self) -> None:
        """Advance the whole swarm one fixed-dt tick as masked ``[N]`` ops.

        The pipeline mirrors :meth:`grl_snam.nav.SdfNavigator.step` element for
        element (carrot FSM -> coefficient net -> bicycle rollout -> metrics),
        but over the shared field and with the sense/rebuild gated to
        ``sense_every``. Everything is order-independent across agents.
        """
        # 1. SENSE (gated) into the ONE shared belief, then ONE rebuild if the
        #    planning surface actually changed. This is the only O(N)/O(map) work
        #    a tick can do, and it is amortized by sense_every.
        if self.gstep % self.sense_every == 0:
            self._sense_shared()

        # 2. FUSED SAMPLE at the start-of-tick pose — reused by the carrot FSM
        #    (wall normal) AND the coefficient features (clearance), so the whole
        #    tick pays exactly two grid_samples (here + metrics).
        phi, nrm = self.field.sample(self.o)  # [N], [N,2]

        # 3. CARROT — the state machine as masked [N] updates.
        carrot = self._plan_carrot(phi, nrm)  # [N,2]  (may mutate self.sp: parked)

        # 4. DRIVE — coefficient net on the reused sample, then ONE batched
        #    bicycle rollout over the shared field.
        feat = self._coef_feats(phi, nrm, carrot)
        al, be, ga = self.model(feat)
        self.o, self.th, self.sp, _ = sdf_nav.bicycle_rollout(
            self.field,
            self.o,
            self.th,
            self.sp,
            carrot,
            al,
            be,
            ga,
            1,
            nsub=self.nsub,
            **self.kw,
            **self.veh,
        )

        # 5. METRICS + WAYPOINT — a reached agent parks (single-goal swarm).
        dg_new = (self.goal - self.o).norm(dim=1)
        self.reached = dg_new < self.reach_tol
        newly = self.reached & ~self.parked & self.active
        self.parked = self.parked | newly

        self.gstep += 1
        self._gen += 1

    def _sense_shared(self) -> None:
        """All active agents ray-cast into the shared belief, then rebuild the
        one shared SDF field iff the planning surface moved.

        O(N) in the ray-casts (the per-plane C++ ``sense_batch`` kernel is the
        planned replacement — it collapses this to one GIL-released call), but
        gated to every ``sense_every`` ticks and dwarfed by the fact that the
        rebuild is a SINGLE EDT, not N of them.
        """
        world = self.n2w(self.o).cpu().numpy()
        th = self.th.cpu().numpy()
        active = self.active.cpu().numpy()
        for i in range(self.N):
            if active[i]:
                self.belief.sense(
                    self.truth,
                    (float(world[i, 0]), float(world[i, 1])),
                    heading_rad=float(th[i]),
                    **self.sensor,
                )
        self._tpl.step_i = self.gstep  # so dyn TTL uses the right world time
        occ = composite_occupancy(self.belief, self.dyn, self._tpl._t(), unknown=self.unknown)
        changed = self.belief.version != self._last_belief_ver or not np.array_equal(occ, self._occ)
        if changed:
            self._occ = occ
            self.field = self._tpl._build_field(occ)
            self._last_belief_ver = self.belief.version
            self.field_ver += 1

    @torch.no_grad()
    def _plan_carrot(self, phi: torch.Tensor, nrm: torch.Tensor) -> torch.Tensor:
        """The exact vectorization of ``SdfNavigator._plan_carrot`` as six masked
        branches over the ``[N]`` columns (see arch spec §2.1-2.4). ``phi``/``nrm``
        are the shared-field sample at the start-of-tick pose ``self.o``."""
        o = self.o
        p = o
        N = self.N
        idx = self._arange
        goal = self.goal
        dgv = goal - o
        dg = dgv.norm(dim=1)  # [N]
        gdir = dgv / (dg[:, None] + 1e-6)  # [N,2]
        tang = torch.stack([-nrm[:, 1], nrm[:, 0]], -1)  # [N,2] wall tangent
        z_i = torch.zeros_like(self.stall)

        # ── Branch 1: stall accounting (non-tracking vs displacement ring) ────
        closing = dg < self.best - 1e-3
        stall_nt = torch.where(closing, z_i, self.stall + 1)
        best_nt = torch.where(closing, dg, self.best)

        slot = self.hist_count % 40
        tracked = self.tracking & self.active
        cur = self.pos_hist[idx, slot]
        self.pos_hist[idx, slot] = torch.where(tracked[:, None], p, cur)
        oldest = (self.hist_count + 1) % 40
        moved = (p - self.pos_hist[idx, oldest]).norm(dim=1)
        have = self.hist_count >= 40
        frozen = have & (moved < 0.15) & (dg > self.reach_tol)
        stall_tk = torch.where(have, torch.where(frozen, self.stall + 1, z_i), self.stall)
        best_tk = torch.minimum(self.best, dg)
        self.hist_count = torch.where(tracked, self.hist_count + 1, self.hist_count)

        self.stall = torch.where(self.tracking, stall_tk, stall_nt)
        self.best = torch.where(self.tracking, best_tk, best_nt)

        # ── Branch 2: seek -> wall entry ─────────────────────────────────────
        enter = (self.mode == SEEK) & (self.stall > 70)
        turn_new = torch.where(
            (tang * gdir).sum(1) >= 0,
            torch.ones(N, device=self.dev),
            -torch.ones(N, device=self.dev),
        )
        self.dhit = torch.where(enter, dg, self.dhit)
        self.wall_entry = torch.where(enter[:, None], p, self.wall_entry)
        self.we_valid = self.we_valid | enter
        self.turn = torch.where(enter, turn_new, self.turn)
        self.mode = torch.where(enter, torch.full_like(self.mode, WALL), self.mode)
        self.stall = torch.where(enter, z_i, self.stall)

        # ── Branch 3: carrot placement (uses the just-updated mode) ──────────
        tang_w = self.turn[:, None] * tang
        wall_c = p + (0.6 * tang_w + 0.4 * nrm) * 1.6
        seek_c = p + gdir * torch.minimum(torch.full_like(dg, 1.8), dg)[:, None]
        carrot = torch.where((self.mode == WALL)[:, None], wall_c, seek_c)

        # ── Branch 4: wall exit (affects the NEXT tick's mode) ───────────────
        esc_tk = self.we_valid & ((p - self.wall_entry).norm(dim=1) > 2.0)
        exit_tk = esc_tk | (self.stall > 240)
        exit_nt = (dg < self.dhit - 1.2) | (self.stall > 240)
        exiting = (self.mode == WALL) & torch.where(self.tracking, exit_tk, exit_nt)
        self.mode = torch.where(exiting, torch.full_like(self.mode, SEEK), self.mode)
        self.best = torch.where(exiting, dg, self.best)
        self.stall = torch.where(exiting, z_i, self.stall)

        # ── Branch 5: parked (brake; carrot straight ahead so steer error=0) ──
        self.sp = torch.where(
            self.parked, torch.clamp(self.sp - self.a_max * self.dt, min=0.0), self.sp
        )
        ahead = torch.stack([torch.cos(self.th), torch.sin(self.th)], -1)
        park_c = p + ahead * torch.maximum(torch.full_like(self.sp, 1e-3), self.sp * 2.0)[:, None]
        carrot = torch.where(self.parked[:, None], park_c, carrot)

        return carrot

    def _coef_feats(
        self, phi: torch.Tensor, nrm: torch.Tensor, carrot: torch.Tensor
    ) -> torch.Tensor:
        """``sdf_nav.coef_feats`` reusing the already-taken sample (the serial
        path re-samples; the values are identical since it samples at the same
        ``o``). Features are taken toward the CARROT, exactly as the drive."""
        dg = carrot - self.o
        gd = dg.norm(dim=-1, keepdim=True)
        gdir = dg / (gd + 1e-6)
        align = (gdir * nrm).sum(-1, keepdim=True)
        return torch.cat([phi.unsqueeze(-1), gd, gdir, align], -1)

    # ── live-scene commands (used by SimThread; safe to call standalone) ──────
    def retarget(self, i: int, goal_world) -> None:
        """Move agent ``i``'s goal live, keeping its escape machinery (track_goal
        semantics: switch to displacement-stall, do not zero the FSM)."""
        gn = torch.from_numpy(self._w2n_np(goal_world)).to(self.dev)
        self.goal[i] = gn
        dg = float((gn - self.o[i]).norm())
        self.best[i] = min(float(self.best[i]), dg)
        self.init[i] = max(float(self.init[i]), dg, 1e-6)
        self.tracking[i] = True
        self.reached[i] = False
        self.parked[i] = False

    def add_obstacle(self, x0, y0, x1, y1) -> None:
        """Stamp a live rectangular blocker into the shared dynamic layer; it is
        sensed and routed-around on the next sense tick and expires by TTL."""
        r0, c0 = self.belief.world_to_cell(x0, y0)
        r1, c1 = self.belief.world_to_cell(x1, y1)
        rr, cc = (r0 + r1) // 2, (c0 + c1) // 2
        rad = max(1, abs(r1 - r0) // 2, abs(c1 - c0) // 2)
        self.dyn.mark(rr, cc, self._tpl._t(), radius_cells=rad)
        # Force a rebuild next sense tick even if belief.version is unchanged.
        self._occ = None if self._occ is None else self._occ  # occ diff will catch it

    # ── the immutable frame ──────────────────────────────────────────────────
    @torch.no_grad()
    def snapshot(self) -> Snapshot:
        """Freeze the current state into a fresh, immutable :class:`Snapshot`.

        Every column is copied out to a new numpy array so the returned frame is
        safe to hand to another thread; the heavy shared field/occupancy go by
        reference (they are swapped, never mutated in place — a rebuild allocates
        a new ``SDFField``). Poses come out in WORLD metres."""
        world = self.n2w(self.o.clone()).cpu().numpy().astype(np.float32)
        gworld = self.n2w(self.goal.clone()).cpu().numpy().astype(np.float32)
        # world speed: forward speed is in normalized units/s; /S -> m/s
        return Snapshot(
            gen=self._gen,
            tick=self.gstep,
            world_t=self.gstep * self.dt,
            pos=world,
            heading=self.th.detach().cpu().numpy().astype(np.float32),
            speed=(self.sp.detach().cpu().numpy() / self.S).astype(np.float32),
            color=self.color.detach().cpu().numpy().astype(np.float32),
            mode=self.mode.detach().cpu().numpy().astype(np.int8),
            goal=gworld,
            active=self.active.detach().cpu().numpy().copy(),
            reached=self.reached.detach().cpu().numpy().copy(),
            field_ver=self.field_ver,
            field_ref=self.field.field,
            occ_ref=self._occ,
        )

    @property
    def all_reached(self) -> bool:
        return bool(self.reached[self.active].all()) if bool(self.active.any()) else True
