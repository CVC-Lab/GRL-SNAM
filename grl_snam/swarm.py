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

import os
from dataclasses import dataclass

import numpy as np
import torch

import sdf_nav

from . import nav_native as _native
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
    map_id: np.ndarray  # i32[N]  which belief plane each agent senses/samples
    field_ver: int
    field_ref: object  # the [M,3,H,W] SDF texture stack, by REFERENCE
    occ_ref: object  # list of per-plane occupancy rasters (bool[H,W]), by reference


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
        belief_mode: str = "shared",
        clusters=None,
        nsub: int | None = None,
        material=None,
    ):
        if not specs:
            raise ValueError("a swarm needs at least one agent")
        if belief_mode not in ("shared", "clustered", "private"):
            raise ValueError(f"belief_mode must be shared/clustered/private, got {belief_mode!r}")
        self.story = story
        self.specs = list(specs)
        self.dev = torch.device(device)
        self.sense_every = max(1, int(sense_every))
        self.reach_tol = float(reach_tol)
        self.belief_mode = belief_mode

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
        self.sensor = dict(tpl.sensor)
        self.unknown = tpl.unknown
        self.cell_m = float(tpl.cell_m)
        self.S = float(self.meta["scale"])
        self.cx, self.cy = (float(c) for c in self.meta["center"])

        # Vehicle + integration params (shared; the drive is one call on [N]).
        n = tpl.nav
        self.kw = dict(n.kw)
        self.veh = dict(n._veh)
        # Substeps per drive tick: how many times bicycle_rollout integrates the
        # vehicle within one world dt (a thin wall can't be tunnelled at higher
        # nsub). Configurable; None inherits the story's meta value (so the
        # serial-navigator parity test, which reads the same meta, still holds).
        # The deployment/C++-port default is 1 (see docs/CVCNAV_CPP_PORT_ROADMAP).
        self.nsub = int(n.nsub) if nsub is None else max(1, int(nsub))
        self.dt = float(self.meta["dt"])
        self.a_max = float(self.veh["a_max"])

        self._build_soa()

        # The map_id seam: shared -> 1 plane, private -> N planes, clustered -> K.
        self.map_id, self.M = self._resolve_maps(belief_mode, clusters)
        self._map_id_t = torch.as_tensor(self.map_id, dtype=torch.int64, device=self.dev)
        self._groups = (
            None
            if self.M == 1 or belief_mode == "private"
            else [
                torch.from_numpy(np.nonzero(self.map_id == g)[0]).to(self.dev)
                for g in range(self.M)
            ]
        )
        # M belief planes as VIEWS into one contiguous (M,H,W) block, so the C++
        # sense_batch kernel writes belief in place. Every plane starts from the
        # template's prior (identical), then diverges per group as it senses.
        self._build_maps(tpl)
        self.belief: BeliefGrid = self.beliefs[0]  # aliases (back-compat, M==1)
        self.dyn: DynamicLayer = self.dyns[0]

        self.gstep = 0
        self.field_ver = 0
        self._gen = 0
        self._last_ver = [int(v) for v in self._version]
        self._occ = [None] * self.M
        self.fields = self._build_all_fields()  # list[SDFField], one per plane
        self.field = self._make_field()  # SDFField (M==1) | BatchedSDFField

        # Opt-in: drive via the torch-free C++ path (cvc::nav) instead of torch.
        # Off by default — torch stays the reference/twin (GRL_SNAM_NAV_DRIVE, a
        # separate flag from the bit-kernel GRL_SNAM_NAV_BACKEND). The C++ drive
        # is float-equivalent to torch (docs/CVCNAV_CPP_PORT_ROADMAP.md P8).
        # Material-aware navigation (grl_snam.material.MaterialGrid): default
        # ON when attached — one SHARED truth-material plane for the whole
        # swarm (material is world state, not per-agent belief), the witness
        # gate evaluated as one vectorized [N] call each tick, and the material
        # force threaded into the batched rollout. None (the default) leaves
        # every trajectory bit-for-bit unchanged.
        # The gate's feasibility surface here is TRUTH occupancy: the Swarm has
        # no planner (its material channel is forces + gate only) and material
        # is oracle world state — matching the source method's oracle-maps
        # setting. FogScenario, which routes over belief, gates against the
        # belief surface its planner pays costs on instead.
        self._material = None
        if material is not None:
            from .material import MaterialRuntime

            self._material = MaterialRuntime(material)
            self._material.update_occ(np.asarray(self.truth, bool))

        self._native_drive = False
        self._native_weights_path = None
        want_native_drive = os.environ.get("GRL_SNAM_NAV_DRIVE", "torch").lower() == "native"
        if self._material is not None and want_native_drive:
            # The torch-free fused drive (nav_drive_step) does not yet carry the
            # material forces — that native material drive is a documented
            # follow-up. Rather than silently drop the material (wrong) or hard
            # fail (blocks a legitimate combination), fall back to the torch
            # drive, which DOES honor material via _material_kw(); the geometry
            # kernels + witness gate still run in C++ under
            # GRL_SNAM_MATERIAL_BACKEND=native, and a pure torch-free material
            # drive is available today through cvc::nav sim_world::set_material.
            import warnings

            warnings.warn(
                "GRL_SNAM_NAV_DRIVE=native is not yet supported together with a "
                "material grid; using the torch drive for this Swarm (material "
                "forces are still applied). For a torch-free material drive use "
                "the C++ cvc::nav sim_world::set_material path.",
                RuntimeWarning,
                stacklevel=2,
            )
            want_native_drive = False
        if want_native_drive and _native.HAS_DRIVE:
            import tempfile

            from .tools.coef_export import write_coef_mlp

            fd, path = tempfile.mkstemp(suffix=".cvcnav")
            os.close(fd)
            write_coef_mlp(self.model, path)
            self._native_weights_path = path
            self._native_drive = True

    def __del__(self):
        p = getattr(self, "_native_weights_path", None)
        if p:
            try:
                os.remove(p)
            except OSError:
                pass

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

    # ── belief grouping (the map_id seam) ─────────────────────────────────────
    def _resolve_maps(self, mode, clusters):
        """``map_id[N]`` (int32) + plane count ``M``. shared -> all 0; private ->
        arange; clustered -> densified group labels (so ``map_id`` is always a
        gapless ``[0,M)`` — an agent never indexes a plane that does not exist)."""
        N = self.N
        if mode == "shared":
            return np.zeros(N, np.int32), 1
        if mode == "private":
            return np.arange(N, dtype=np.int32), N
        labels = self._cluster_labels(clusters)
        _, dense = np.unique(labels, return_inverse=True)  # gapless 0..K-1
        return dense.astype(np.int32), int(dense.max()) + 1

    def _cluster_labels(self, clusters):
        """Resolve the ``clusters`` argument to an ``[N]`` integer label array:
        an explicit array/list, a ``callable(specs)->labels``, or an ``int`` K
        (k-means-lite on start positions — a spatial partition of the swarm)."""
        N = self.N
        if clusters is None:
            raise ValueError("belief_mode='clustered' needs `clusters` (int K, array, or callable)")
        if callable(clusters):
            return np.asarray(clusters(self.specs), np.int64).reshape(N)
        arr = np.asarray(clusters)
        if arr.ndim >= 1 and arr.size == N:
            return arr.astype(np.int64).reshape(N)
        k = int(clusters)
        if k <= 1:
            return np.zeros(N, np.int64)
        pts = np.stack([self._w2n_np(s.start) for s in self.specs]).astype(np.float64)
        rng = np.random.default_rng(0)
        cen = pts[rng.choice(N, size=min(k, N), replace=False)]
        for _ in range(15):  # a few Lloyd iterations — deterministic, seed-fixed
            lab = np.argmin(((pts[:, None] - cen[None]) ** 2).sum(-1), axis=1)
            for j in range(len(cen)):
                m = lab == j
                if m.any():
                    cen[j] = pts[m].mean(0)
        return lab

    def _build_maps(self, tpl) -> None:
        """Allocate the ``(M,H,W)`` belief block and wrap M ``BeliefGrid`` views
        over it (so the in-place ``sense_batch`` kernel writes belief directly),
        each seeded from the template's prior. M ``DynamicLayer``s alongside."""
        H, W = tpl.belief.ny, tpl.belief.nx
        M = self.M
        self._logodds = np.repeat(tpl.belief.logodds[None], M, axis=0).astype(np.float32)
        self._lastvis = np.repeat(tpl.belief.last_visible[None], M, axis=0).copy()
        self._everseen = np.repeat(tpl.belief.ever_seen[None], M, axis=0).copy()
        self._version = np.zeros(M, np.int32)
        self.beliefs, self.dyns = [], []
        for g in range(M):
            b = BeliefGrid((H, W), self.bounds)
            b.logodds = self._logodds[g]  # views: np.clip(out=)/|= write through
            b.last_visible = self._lastvis[g]  # (rebound by the Python fallback)
            b.ever_seen = self._everseen[g]
            self.beliefs.append(b)
            self.dyns.append(DynamicLayer((H, W), ttl_s=tpl.dyn.ttl_s))

    def _build_all_fields(self):
        """One :class:`sdf_nav.SDFField` per plane, from that plane's composite
        occupancy — batched over planes when the C++ kernel is present."""
        t = self._tpl._t()
        occs = [
            composite_occupancy(self.beliefs[g], self.dyns[g], t, unknown=self.unknown)
            for g in range(self.M)
        ]
        self._occ = occs
        if self.M > 1 and _native.enabled() and hasattr(_native, "build_sdf_batch"):
            tris = _native.build_sdf_batch(np.stack(occs), self.bounds, self.scale)
            return [self._tpl._finalize_field(*tri) for tri in tris]
        return [self._tpl._build_field(occ) for occ in occs]

    def _make_field(self):
        """The sampler the drive calls every substep: an ``SDFField`` for the one
        shared plane, else a map_id-gathering ``BatchedSDFField`` over the stack."""
        f0 = self.fields[0]
        if self.M == 1:
            return f0
        self._field_stack = torch.cat([f.field for f in self.fields], 0)  # [M,3,H,W]
        return sdf_nav.BatchedSDFField(
            self._field_stack,
            f0.mnx,
            f0.mny,
            f0.mxx,
            f0.mxy,
            f0.cx,
            f0.cy,
            f0.S,
            groups=self._groups,
        )

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
        #    bicycle rollout over the shared field. Optionally via the torch-free
        #    C++ drive (float-equivalent; sample->coef_feats->coef_mlp->bicycle in
        #    one GIL-released call). The carrot FSM (step 3) stays in Python.
        if self._native_drive:
            self._native_drive_step(carrot)
        else:
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
                **self._material_kw(),
            )

        # 5. METRICS + WAYPOINT — a reached agent parks (single-goal swarm).
        dg_new = (self.goal - self.o).norm(dim=1)
        self.reached = dg_new < self.reach_tol
        newly = self.reached & ~self.parked & self.active
        self.parked = self.parked | newly

        self.gstep += 1
        self._gen += 1

    def _material_kw(self) -> dict:
        """Per-tick material kwargs for the batched rollout: one vectorized
        witness-gate call over all N agents (against each agent's own goal),
        yielding the effective lambda columns. Empty when no material."""
        if self._material is None:
            return {}
        self._material.consume_version_change()
        p = self._material.params
        if p.gate_enabled:
            pos_w = self.n2w(self.o).cpu().numpy().astype(np.float64)
            goal_w = self.n2w(self.goal).cpu().numpy().astype(np.float64)
            active = self._material.gate_batch(pos_w, goal_w)
            mult = torch.from_numpy(active.astype(np.float32)).to(self.dev)
        else:
            mult = torch.ones(self.N, device=self.dev)
        self.material_gate = mult.bool()  # exposed for renderers/metrics
        return dict(
            material=self._material.field,
            lam_soft=p.lam_soft * mult,
            lam_hard=torch.full((self.N,), float(p.lam_hard), device=self.dev),
            mat_k_sharp=float(p.k_sharp),
            mat_d_hat_m=float(p.d_hat_sdf_m),
        )

    def _native_drive_step(self, carrot: torch.Tensor) -> None:
        """Drive one tick through the torch-free C++ path (:func:`nav_native.drive_step`)
        instead of the torch coef-net + rollout, updating ``self.o/th/sp`` in place.
        Float-equivalent to the torch drive; used only when ``GRL_SNAM_NAV_DRIVE=native``.
        The field stack, poses and carrot cross to numpy (f32) and the fresh poses come
        back to the Swarm's device. ``map_id`` selects each agent's belief plane
        (``None`` for shared M==1)."""
        field_np = self.field.field.detach().cpu().numpy()  # (M,3,H,W) f32
        mid = None if self.M == 1 else self.map_id
        o2, th2, sp2, _ = _native.drive_step(
            field_np,
            self.o.detach().cpu().numpy(),
            self.th.detach().cpu().numpy(),
            self.sp.detach().cpu().numpy(),
            carrot.detach().cpu().numpy(),
            self._native_weights_path,
            bounds=self.bounds,
            center=(self.cx, self.cy),
            scale=self.S,
            params={**self.kw, **self.veh, "nsub": self.nsub},
            map_id=mid,
        )
        self.o = torch.from_numpy(o2).to(self.dev)
        self.th = torch.from_numpy(th2).to(self.dev)
        self.sp = torch.from_numpy(sp2).to(self.dev)

    def _sense_shared(self) -> None:
        """Sense every active agent into its belief plane (``map_id``), then
        rebuild the fields of the planes whose planning surface actually moved.

        Prefers the C++ ``sense_batch`` kernel — one GIL-released call, threaded
        across planes, bit-identical to N serial :meth:`BeliefGrid.sense` — and
        falls back to a pure-Python per-plane loop. Named ``_sense_shared`` for
        the shared-belief history; it now serves every belief mode. The rebuild
        is a single EDT per changed plane (O(1) in N for shared)."""
        world = self.n2w(self.o).cpu().numpy()
        th = self.th.cpu().numpy()
        active = self.active.cpu().numpy()
        m = np.nonzero(active)[0].astype(np.int64)
        if m.size == 0:
            return
        if _native.enabled() and getattr(_native, "HAS_SENSE_BATCH", False):
            _native.sense_batch(
                self.truth,
                world[m],
                th[m],
                self._logodds,
                self._lastvis,
                self._everseen,
                self._version,
                agent_map=self.map_id[m],
                range_m=self.sensor["range_m"],
                n_rays=self.sensor.get("n_rays", 240),
                fov_rad=self.sensor.get("fov_rad", 2.0 * np.pi),
                bounds=self.bounds,
                peer_boxes=None,  # shared/clustered: peers decay in the dyn layer,
                mover_boxes=None,  # never smeared into the static belief (no self-lock)
            )
            for g in range(self.M):
                self.beliefs[g].version = int(self._version[g])
                self.beliefs[g].last_visible = self._lastvis[g]  # re-view (kernel wrote the block)
        else:
            for i in m:  # ascending index == the serial reference order
                g = int(self.map_id[i])
                self.beliefs[g].sense(
                    self.truth,
                    (float(world[i, 0]), float(world[i, 1])),
                    heading_rad=float(th[i]),
                    **self.sensor,
                )
            for g in range(self.M):  # resync the block (last_visible rebinds; version is a py int)
                self._version[g] = self.beliefs[g].version
                self._lastvis[g] = self.beliefs[g].last_visible
                self.beliefs[g].last_visible = self._lastvis[g]
        self._tpl.step_i = self.gstep  # so dyn TTL uses the right world time
        self._rebuild_changed_planes()

    def _rebuild_changed_planes(self) -> None:
        """Rebuild the SDF field of any plane whose belief version bumped or whose
        dynamic layer changed — one ``build_sdf_batch`` over just those planes."""
        t = self._tpl._t()
        occs, reb = [], []
        for g in range(self.M):
            occ_g = composite_occupancy(self.beliefs[g], self.dyns[g], t, unknown=self.unknown)
            if int(self._version[g]) != self._last_ver[g] or not np.array_equal(
                occ_g, self._occ[g]
            ):
                self._occ[g] = occ_g
                occs.append(occ_g)
                reb.append(g)
                self._last_ver[g] = int(self._version[g])
        if not reb:
            return
        if len(reb) > 1 and _native.enabled() and hasattr(_native, "build_sdf_batch"):
            tris = _native.build_sdf_batch(np.stack(occs), self.bounds, self.scale)
            new = [self._tpl._finalize_field(*tri) for tri in tris]
        else:
            new = [self._tpl._build_field(o) for o in occs]
        for g, f in zip(reb, new):
            self.fields[g] = f
        if self.M == 1:
            self.field = self.fields[0]  # swap a fresh SDFField (immutable for readers)
        else:
            # overwrite only the changed planes of the shared stack (in place; a
            # live-rebuild-under-threading path would double-buffer this — the
            # pipeline_edt follow-up). BatchedSDFField holds the stack by ref.
            for g, f in zip(reb, new):
                self._field_stack[g].copy_(f.field[0])
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
        """Stamp a live rectangular blocker into every plane's dynamic layer (a
        real wall all groups should route around); it is sensed and routed-around
        on the next sense tick and expires by TTL. The occupancy diff in
        :meth:`_rebuild_changed_planes` picks it up even with belief unchanged."""
        r0, c0 = self.beliefs[0].world_to_cell(x0, y0)
        r1, c1 = self.beliefs[0].world_to_cell(x1, y1)
        rr, cc = (r0 + r1) // 2, (c0 + c1) // 2
        rad = max(1, abs(r1 - r0) // 2, abs(c1 - c0) // 2)
        t = self._tpl._t()
        for d in self.dyns:
            d.mark(rr, cc, t, radius_cells=rad)

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
            map_id=self.map_id.copy(),
            field_ver=self.field_ver,
            field_ref=self.field.field,
            occ_ref=self._occ,
        )

    @property
    def all_reached(self) -> bool:
        return bool(self.reached[self.active].all()) if bool(self.active.any()) else True
