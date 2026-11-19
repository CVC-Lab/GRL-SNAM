"""CoPH drives the cars (T4, demo 2b): one CoPH learner per car, on libcvc's cars template.

    python -m grl_snam.ariadne coph-cars --arms --seeds 1 2 3 --out runs/coph-cars   # headless arms
    python -m grl_snam.ariadne coph-cars --arms --cars 2 --frames 200 --out /tmp/c   # a short one
    python -m grl_snam.ariadne coph-cars                    # the template's window, CoPH driving
    python -m grl_snam.ariadne coph-cars --headless --frames 600 --log run.npz

libcvc's ``docs/roadmap/NAV-COPH-ROADMAP.md`` §3.3 is the spec. Each car carries one CoPH learner
(:class:`CarLearner`, a ``grl_snam.coph`` ``CooperativeLearner`` in the demos' profile). Every
tick :class:`CophCars` reads the cars' state, steps CoPH at CoPH's own cadence and allocates its
force to a steer and an acceleration:

- **Scale** (§3.3.1, D4). Froude scaling by ``s``: lengths x ``s``, times and speeds x ``sqrt(s)``,
  accelerations unchanged (:class:`Froude`). All cars share one CoPH frame, so a packet's anchors
  mean the same place to every receiver. The first runs are RoboRacers at ``s`` = 1 (D3).
- **Clocks** (D6). CoPH steps every ``round(0.12 sqrt(s) / tick)`` template ticks (2 at ``s`` = 1:
  the template's tick is 0.06 s) and holds its force ``u`` between steps.
- **Allocation** (§3.3.2, D5). ``u`` is allocated every tick with ``grl_snam.coph.allocation``
  (libcvc's ``coph_allocate``; 0.5 m/s speed floor). The learner observes the mean of the force
  the car took, ``u_used``, over the held step, never ``u``. The force it did not take, ``e``, is
  booked as its own ledger term: the unallocated work ``e . dq``.
- **World adapter** (§3.3.3). Obstacles are the occupied cells within the scaled sensor radius, as
  disks, from ``pycvc.nav_visible_obstacles`` on the world's grid. Risk readings sample the
  material raster's risk plane ``r~`` (``pycvc.nav_material_build``) at CoPH's 21 sensor points.
  The template's worlds carry no material layer, so the adapter lays seeded risk patches on one
  (:class:`MaterialRaster`): one across each car's straight line to its goal, the same for every
  arm of a seed. The goal well uses a local goal 30 m along the line (D12).
- **Panels.** The third return value's ``coeffs`` are ``pi_risk``, ``pi_barrier``, ``damping`` and
  ``version`` (the plotted four), then the ledger columns, all logged. ``cost()`` draws the chosen
  car's ``V`` (``ctl.view`` 0) or its ``r`` (1).

**Arms** (§3.3.4): communication off, CoPH packets and raw readings (``--arms``). They run
headless on the template's script API (``pycvc_gl.cars.Cars``) over paired seeds: each seed gives
every arm the same starts, goals and patches. ``summary.json`` and ``REPORT.md`` give arrivals,
path length, collisions, excess risk cost against the best arm, bytes, and the ledger with the
allocation residual. Each episode's log is the template's (``CarsLog``) with the coefficients.

What does not fit the template, and how this port handles it:
- **A template controller sees no world grid.** ``CarsState.world`` is the world's kind, so
  inside ``pycvc_gl.cars.run`` (the window) obstacles come from the cars' rays instead: each ray
  that hits within the sensor radius gives a disk. The arms run on the script API, where the
  controller holds the world (:attr:`CophCars.world`) and uses ``nav_visible_obstacles``.
- **The template's flat world is a Humvee's** (``flat_world(4.84)``, 194 m across) whatever the
  vehicle. The arms use a RoboRacer's (21 m across), as D3 asks.

This is exploratory (§8): the formulation treats a fully actuated point, and the allocation is
new. The report leads with the allocation residual and the ledger; it claims only what the paired
seeds measure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COPH_DT = 0.12  # CoPH's own step, s (in CoPH's frame)
SIM_TICK = 0.06  # the template's tick when pycvc_gl does not say (pycvc.SIM_TICK)
LOCAL_GOAL_M = 30.0  # D12: the goal well's local goal, m along the straight line
ARMS = ("off", "coph", "raw")
ARM_CODES = {name: float(k) for k, name in enumerate(ARMS)}  # ctl.arm, a slider's float
RAW_ANCHORS = 12  # the hazard demo's raw arm (grl_snam.coph.hazard_demo.RAW_ANCHORS)
#: The controller's parameters (``ctl.<name>`` in the template's document).
PARAMS = {
    "arm": ARM_CODES["coph"],  # 0 communication off, 1 CoPH packets, 2 raw readings
    "scale": 1.0,  # Froude s
    "view": 0.0,  # cost(): 0 the chosen car's V, 1 its r
    "local_goal_m": LOCAL_GOAL_M,
}
#: The coefficients the panels plot (the first four) and the ledger columns logged after them.
COEFFS = (
    "pi_risk",
    "pi_barrier",
    "damping",
    "version",
    "work",
    "heat",
    "kinetic_change",
    "residual",
    "unallocated_work",
    "allocation_residual",
    "shaping_jumps",
    "risk_cost",
    "bytes",
    "accepted",
    "rejected",
    "stage",
)

# CoPH's 21 sensor points: a 5 x 5 lattice on [-1, 1]^2 inside the unit disc (World.sense).
SENSOR_OFFSETS = np.array(
    [
        (a, b)
        for a in np.linspace(-1.0, 1.0, 5)
        for b in np.linspace(-1.0, 1.0, 5)
        if a * a + b * b <= 1.0
    ]
)


def _template():
    try:
        from pycvc_gl import cars as C
    except ImportError:
        return None
    return C if C.has_cars() else None


def _pycvc():
    try:
        import pycvc
    except ImportError:
        return None
    return pycvc


# ── scale and frame ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Froude:
    """Froude scaling by ``s`` (D4) about a shared ``origin``: CoPH's ``q = (p - origin) / s``,
    ``v_coph = v / sqrt(s)``, accelerations (forces per unit mass) unchanged."""

    s: float = 1.0
    origin: tuple = (0.0, 0.0)

    def q(self, p):
        return (np.asarray(p, float) - np.asarray(self.origin, float)) / self.s

    def p(self, q):
        return np.asarray(q, float) * self.s + np.asarray(self.origin, float)

    def v(self, v):
        return np.asarray(v, float) / math.sqrt(self.s)

    def length(self, metres):
        return float(metres) / self.s

    def ticks_per_step(self, tick=SIM_TICK) -> int:
        """Template ticks per CoPH step: CoPH's 0.12 s, in world time, over the tick (D6)."""
        return max(1, int(round(COPH_DT * math.sqrt(self.s) / tick)))


def local_goal(pos, goal, reach_m=LOCAL_GOAL_M):
    """The goal, or the point ``reach_m`` along the straight line to it when it is farther (D12)."""
    pos, goal = np.asarray(pos, float), np.asarray(goal, float)
    d = goal - pos
    dist = float(np.hypot(*d))
    return goal.copy() if dist <= reach_m else pos + d * (reach_m / dist)


# ── the learner on a car ──────────────────────────────────────────────────────────────────────


def coph_config():
    """The demos' profile (roadmap §1, D7): routed control, progress shaping, mission replay."""
    from grl_snam.coph.crossing_demo import configuration

    return configuration(10**6)


def communication(arm):
    """The demos' channel (byte budget 4 MB, ttl 300 s), with 12 anchors for raw readings."""
    from grl_snam.coph.cooperative import CommunicationConfig

    return CommunicationConfig(
        byte_budget=4_000_000, ttl=300.0, anchors=RAW_ANCHORS if arm == "raw" else 4
    )


def _learner_classes():
    from grl_snam.coph.cooperative import CooperativeLearner
    from grl_snam.coph.navigation import intersects
    from grl_snam.coph.records import new_stage
    from grl_snam.coph.route_field import sync
    from grl_snam.coph.sharing import GRADIENT_UNKNOWN

    class CarLearner(CooperativeLearner):
        """A ``CooperativeLearner`` whose stages follow the car in any direction.

        Stages advance by the demos' learner's rule (``MazeLearner.advance``, roadmap §1): along
        whichever axis the car left the inner box by, either way. Routing is the profile's own
        (``navigation.route``: A* through sensed footprints), not the maze graph's, since a car's
        world has no surveyed network. Like the demos' learner, it never sets ``Stage.anchors``."""

        stage_mode = "spatial"

        def advance(self, x):
            cfg, s = self.cfg, self.stages[-1]
            sizes = np.array([cfg.stage_width, cfg.stage_height])
            delta = np.abs(np.asarray(x[:2], float) - s.center)
            if not np.any(delta > sizes * (1 - cfg.overlap) / 2):
                return False
            old = float(self.shaping_energy(x, s))
            axis = int(np.argmax(delta / sizes))
            center = np.asarray(s.center, float).copy()
            center[axis] += np.sign(x[axis] - center[axis]) * sizes[axis] * (1 - cfg.overlap)
            n = new_stage(s.id + 1, tuple(center), x, self.goal, cfg)
            pooled = [r for st in self.stages for r in st.readings if n.contains(r.q, cfg)]
            n.readings = list({r.uid: r for r in pooled}.values())[-cfg.sample_cap :]
            n.footprints = list(
                dict.fromkeys(tuple(f) for st in self.stages for f in st.footprints)
            )[-cfg.footprint_cap :]
            n.entry_navigation = copy.deepcopy(self.navigation)
            n.entry_navigation["planned_at"] = -10000
            n.version = self.version
            self.stages = (self.stages + [n])[-(cfg.previous + 1) :]
            for st in self.stages:
                st.obstacles = tuple(
                    o for o in self.geometry_store if intersects(o, st.center, cfg)
                )
            sync(self)
            new = float(self.shaping_energy(x, n))
            self.context_jumps.append(dict(stage=n.id, shaping_jump=new - old, physical_jump=0.0))
            return True

    class RawCarLearner(CarLearner):
        """Sends raw readings, value only, at the most hazardous supported points: the hazard
        demo's ``RawLearner.export`` on a car."""

        def export(self, x, recipient):
            from dataclasses import replace

            m = super().export(x, recipient)
            if m is None:
                return None
            s, picked = self.stages[-1], []
            for r in sorted(s.readings, key=lambda r: (-r.value, r.uid)):
                if int(r.uid.split(":")[-1]) % 5 and all(
                    np.linalg.norm(np.asarray(r.q) - p) > 0.3 for p in picked
                ):
                    picked.append(np.asarray(r.q))
                if len(picked) == self.communication.anchors:
                    break
            values = {tuple(r.q): r.value for r in s.readings}
            anchors = tuple(
                (float(q[0]), float(q[1]), float(values[tuple(q)]), 0.0, 0.0) for q in picked
            )
            if not anchors:
                return m
            cov = np.diag([0.0025, GRADIENT_UNKNOWN, GRADIENT_UNKNOWN] * len(anchors))
            return replace(
                m, anchors=anchors, covariance=tuple(map(tuple, cov)), uncertainty="raw readings"
            )

    return CarLearner, RawCarLearner


def make_learner(arm, x, goal, agent_id, cfg=None):
    """A car's learner for ``arm`` at CoPH state ``x`` (q, v) heading for CoPH point ``goal``."""
    car, raw = _learner_classes()
    cls = raw if arm == "raw" else car
    return cls(
        cfg or coph_config(), np.asarray(x, float), tuple(goal), agent_id=str(agent_id),
        communication=communication(arm),
    )  # fmt: skip


# ── the world adapter ─────────────────────────────────────────────────────────────────────────


def risk_patches(starts, goals, scale=1.0, seed=0, amplitude=0.9, sigma_m=1.0):
    """Seeded risk patches (x, y, sx, sy, amplitude), m: one across each car's straight line to its
    goal, at 35-65 % of the way, shifted up to a metre (x ``scale``) off the line."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, 31337]))
    out = []
    for a, b in zip(np.asarray(starts, float), np.asarray(goals, float)):
        d = b - a
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        nrm = np.array([-d[1], d[0]]) / length
        c = a + rng.uniform(0.35, 0.65) * d + rng.uniform(-1.0, 1.0) * scale * nrm
        sx = sigma_m * scale * rng.uniform(0.8, 1.25)
        sy = sigma_m * scale * rng.uniform(0.8, 1.25)
        out.append((float(c[0]), float(c[1]), float(sx), float(sy), float(amplitude)))
    return out


class MaterialRaster:
    """A material layer on a grid: the raw risk of ``patches`` over a ``background``, and its risk
    plane ``r~`` (``pycvc.nav_material_build``: blurred by ``sigma`` cells, clipped to [0, 1]).

    The grid follows the template's ``World``: row 0 at ``min_y``, cell centres on the grid points.
    ``hard`` (uint8, the grid's shape) marks blocked cells; zeros when None."""

    def __init__(self, patches, bounds, cell, hard=None, background=0.02, sigma=1.0):
        self.patches = [tuple(map(float, p)) for p in patches]
        self.bounds = tuple(map(float, bounds))
        self.cell = float(cell)
        self.background = float(background)
        mnx, mny, mxx, mxy = self.bounds
        cols = int(round((mxx - mnx) / self.cell)) + 1
        rows = int(round((mxy - mny) / self.cell)) + 1
        X, Y = np.meshgrid(mnx + np.arange(cols) * self.cell, mny + np.arange(rows) * self.cell)
        self.raw = self.raw_at(X, Y).astype(np.float32)
        hard = np.zeros((rows, cols), np.uint8) if hard is None else np.asarray(hard, np.uint8)
        if hard.shape != self.raw.shape:
            raise ValueError(f"hard is {hard.shape}, the grid {self.raw.shape}")
        pycvc = _pycvc()
        #: Where r~ came from: libcvc's material build, or the raw risk when pycvc lacks it.
        self.source = "raw"
        self.r = np.clip(self.raw, 0.0, 1.0)
        if pycvc is not None and hasattr(pycvc, "nav_material_build"):
            planes = np.asarray(pycvc.nav_material_build(self.raw, hard, self.cell, 1.0, sigma))
            self.r, self.source = planes[0].astype(float), "nav_material_build"

    def raw_at(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        value = np.full(np.broadcast(X, Y).shape, self.background)
        for x, y, sx, sy, amp in self.patches:
            value = value + amp * np.exp(-0.5 * (((X - x) / sx) ** 2 + ((Y - y) / sy) ** 2))
        return np.clip(value, 0.0, 1.0)

    def sample(self, points):
        """``r~`` at world points (k, 2), bilinear; the background off the grid."""
        pts = np.atleast_2d(np.asarray(points, float))
        mnx, mny, _, _ = self.bounds
        rows, cols = self.r.shape
        fc, fr = (pts[:, 0] - mnx) / self.cell, (pts[:, 1] - mny) / self.cell
        inside = (fc >= 0) & (fc <= cols - 1) & (fr >= 0) & (fr <= rows - 1)
        out = np.full(len(pts), self.background)
        if inside.any():
            c, r = fc[inside], fr[inside]
            c0 = np.minimum(np.floor(c).astype(int), cols - 2)
            r0 = np.minimum(np.floor(r).astype(int), rows - 2)
            tc, tr = c - c0, r - r0
            R = self.r
            top = R[r0, c0] * (1 - tc) + R[r0, c0 + 1] * tc
            bot = R[r0 + 1, c0] * (1 - tc) + R[r0 + 1, c0 + 1] * tc
            out[inside] = top * (1 - tr) + bot * tr
        return out


def visible_obstacles(world, pos, radius_m, cap):
    """The occupied cells of ``world`` within ``radius_m`` of ``pos``, as disks (centres (k, 2),
    radii (k,), m), nearest first: ``pycvc.nav_visible_obstacles`` on the window around the car."""
    occ = np.asarray(world.occ)
    rows, cols = occ.shape
    mnx, mny, mxx, mxy = world.bounds
    cw, ch = (mxx - mnx) / (cols - 1), (mxy - mny) / (rows - 1)
    c0 = max(0, int(math.floor((pos[0] - radius_m - mnx) / cw)))
    c1 = min(cols - 1, int(math.ceil((pos[0] + radius_m - mnx) / cw)))
    r0 = max(0, int(math.floor((pos[1] - radius_m - mny) / ch)))
    r1 = min(rows - 1, int(math.ceil((pos[1] + radius_m - mny) / ch)))
    if c1 - c0 < 1 or r1 - r0 < 1:
        return np.empty((0, 2)), np.empty(0)
    win = np.ascontiguousarray(occ[r0 : r1 + 1, c0 : c1 + 1], np.uint8)
    X, Y = np.meshgrid(mnx + np.arange(c0, c1 + 1) * cw, mny + np.arange(r0, r1 + 1) * ch)
    seen = ((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 <= radius_m * radius_m).astype(np.uint8)
    bounds = (mnx + c0 * cw, mny + r0 * ch, mnx + c1 * cw, mny + r1 * ch)
    pycvc = _pycvc()
    if pycvc is not None and hasattr(pycvc, "nav_visible_obstacles"):
        centers, radii, _ = pycvc.nav_visible_obstacles(
            seen, win, *bounds, float(pos[0]), float(pos[1]), int(cap)
        )
        return np.asarray(centers, float).reshape(-1, 2), np.asarray(radii, float)
    hit = np.argwhere((seen & win) != 0)  # the twin's own order: nearest first, then row-major
    centers = (
        np.stack([X[hit[:, 0], hit[:, 1]], Y[hit[:, 0], hit[:, 1]]], 1)
        if len(hit)
        else np.empty((0, 2))
    )
    order = np.argsort(np.hypot(*(centers - pos).T), kind="stable")[:cap]
    return centers[order], np.full(len(order), 0.5 * min(cw, ch))


def ray_obstacles(s, i, radius_m, disk_m):
    """Car ``i``'s ray hits within ``radius_m`` as disks of radius ``disk_m``: the obstacles a
    template controller can see without the world's grid."""
    r = np.asarray(s.ranges[i], float)
    a = np.asarray(s.ray_angles, float) + float(s.heading[i])
    hit = r < min(float(s.range_max), radius_m)
    pts = np.stack([s.x[i] + r[hit] * np.cos(a[hit]), s.y[i] + r[hit] * np.sin(a[hit])], 1)
    return pts.reshape(-1, 2), np.full(int(hit.sum()), float(disk_m))


# ── the ledger ────────────────────────────────────────────────────────────────────────────────


@dataclass
class Ledger:
    """A car's account in CoPH's frame (per unit mass), summed over its CoPH steps.

    For each step, with ``u_used`` the mean force the car took and ``d`` the learner's damping:
    ``work = u_used . dq`` (the car's own displacement), ``kinetic_change`` from its own
    velocities, ``heat`` the dissipation CoPH's plant books from the same start under ``u_used``
    (``model.balance``'s closed form), and ``residual = kinetic_change - work + heat``. On CoPH's
    own plant the residual is about 1e-15 a step; on a car it measures how far the car's motion
    is from a damped point mass. ``unallocated_work`` is ``e . dq`` with ``e = u - u_used``, the
    force the car did not take (D5). ``allocation_residual`` is ``sqrt(sum |e|^2 / sum |u|^2)``
    over the ticks."""

    work: float = 0.0
    heat: float = 0.0
    kinetic_change: float = 0.0
    unallocated_work: float = 0.0
    shaping_jumps: float = 0.0
    risk_cost: float = 0.0
    steps: int = 0
    e2: float = 0.0
    u2: float = 0.0
    max_step_residual: float = 0.0

    @property
    def residual(self) -> float:
        return self.kinetic_change - self.work + self.heat

    @property
    def allocation_residual(self) -> float:
        return math.sqrt(self.e2 / self.u2) if self.u2 > 0 else 0.0

    def add_step(self, x0, x1, u_used, e_mean, damping, dt=COPH_DT):
        import torch

        from grl_snam.coph.model import balance, tensor

        x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
        u_used, e_mean = np.asarray(u_used, float), np.asarray(e_mean, float)
        dq = x1[:2] - x0[:2]
        with torch.no_grad():
            heat = balance(tensor(x0), tensor(u_used), float(damping), dt)["heat"]
        work = float(u_used @ dq)
        kinetic = 0.5 * (float(x1[2:] @ x1[2:]) - float(x0[2:] @ x0[2:]))
        self.work += work
        self.heat += heat
        self.kinetic_change += kinetic
        self.unallocated_work += float(e_mean @ dq)
        self.max_step_residual = max(self.max_step_residual, abs(kinetic - work + heat))
        self.steps += 1

    def as_dict(self) -> dict:
        return dict(
            work=self.work, heat=self.heat, kinetic_change=self.kinetic_change,
            residual=self.residual, max_step_residual=self.max_step_residual,
            unallocated_work=self.unallocated_work, allocation_residual=self.allocation_residual,
            shaping_jumps=self.shaping_jumps, risk_cost=self.risk_cost, steps=self.steps,
        )  # fmt: skip


# ── V and r on a grid ─────────────────────────────────────────────────────────────────────────


def potential_grid(learner, Q):
    """The learner's ``V`` (``model.potential`` with its control target, policy and obstacles, on
    its current stage) at CoPH points ``Q`` (k, 2), batched. Disk obstacles only."""
    import torch

    from grl_snam.coph.model import risk, shaping_parameters, tensor

    cfg, s = learner.cfg, learner.stages[-1]
    with torch.no_grad():
        q = tensor(np.asarray(Q, float))
        target = tensor(learner.control_target(s))
        squared = (q - target).square().sum(-1)
        weights, centers = tensor(s.weights), tensor(s.centers)
        shaping = learner.policy[0] * risk(q, weights, centers, cfg.rbf_width)
        obstacles = [o for o in learner.control_obstacles(s) if len(o) == 3]
        if obstacles:
            obs = tensor(obstacles)
            clearance = torch.linalg.vector_norm(q[:, None, :] - obs[None, :, :2], dim=-1)
            clearance = clearance - obs[:, 2] - 0.12
            barrier = torch.nn.functional.softplus((0.35 - clearance) * 6).square().sum(-1)
            shaping = shaping + learner.policy[1] * barrier / 36
        if cfg.shaping_mode == "progress":
            pp = shaping_parameters(weights, cfg.rbf_width, obstacles, cfg)
            V = (
                0.5 * pp["k"] * squared
                + pp["alpha"] * squared / (squared + pp["ell"] ** 2) * shaping
            )
        else:
            V = 0.5 * squared + shaping
    return V.numpy()


def risk_grid(learner, Q):
    """The learner's ``r`` (its current stage's field) at CoPH points ``Q`` (k, 2)."""
    import torch

    from grl_snam.coph.model import risk, tensor

    s = learner.stages[-1]
    with torch.no_grad():
        return risk(tensor(np.asarray(Q, float)), tensor(s.weights), tensor(s.centers),
                    learner.cfg.rbf_width).numpy()  # fmt: skip


# ── the controller ────────────────────────────────────────────────────────────────────────────


def _arm_name(value) -> str:
    if isinstance(value, str):
        if value not in ARMS:
            raise ValueError(f"arm {value!r}: one of {ARMS}")
        return value
    k = int(round(float(value)))
    if not 0 <= k < len(ARMS):
        raise ValueError(f"ctl.arm {value}: 0 off, 1 coph, 2 raw")
    return ARMS[k]


@dataclass
class CarRecord:
    """What the summary reads for one car."""

    arrived_at: float | None = None
    path_m: float = 0.0
    collided_ticks: int = 0
    bytes_sent: int = 0
    accepted: int = 0
    rejected: int = 0
    reasons: dict = field(default_factory=dict)


class _Run:
    """One run's cars and their learners (rebuilt when the run restarts)."""

    def __init__(self, ctl, s, p):
        import torch

        torch.set_num_threads(1)
        self.ctl = ctl
        self.n = s.n
        self.arm = _arm_name(ctl.arm if ctl.arm is not None else p.get("arm", PARAMS["arm"]))
        scale = float(ctl.scale if ctl.scale is not None else p.get("scale", 1.0))
        self.reach_m = float(p.get("local_goal_m", LOCAL_GOAL_M))
        self.tick = float(getattr(s, "dt", SIM_TICK) or SIM_TICK)
        starts, goals = np.asarray(s.pos, float), np.asarray(s.goal, float)
        world = ctl.world
        origin = world.center if world is not None else tuple(np.r_[starts, goals].mean(0))
        self.frame = Froude(scale, tuple(map(float, origin)))
        self.k = self.frame.ticks_per_step(self.tick)
        self.cfg = coph_config()
        self.sensor_m = self.cfg.sensor_radius * scale
        self.seed = ctl.seed if ctl.seed is not None else _starts_seed(starts, goals)
        patches = risk_patches(starts, goals, scale, self.seed)
        if world is not None:
            self.material = MaterialRaster(patches, world.bounds, world.cell, hard=world.occ)
        else:
            lo = np.r_[starts, goals].min(0) - 10 * scale
            hi = np.r_[starts, goals].max(0) + 10 * scale
            cell = max(0.1 * scale, float((hi - lo).max()) / 800.0)
            self.material = MaterialRaster(patches, (*lo, *hi), cell)
        self.goals = goals.copy()
        self.local = np.array([local_goal(a, b, self.reach_m) for a, b in zip(starts, goals)])
        x0 = [self.state(s, i) for i in range(self.n)]
        self.learners = [
            make_learner(self.arm, x0[i], self.frame.q(self.local[i]), i, self.cfg)
            for i in range(self.n)
        ]
        self.ledgers = [Ledger() for _ in range(self.n)]
        self.records = [CarRecord() for _ in range(self.n)]
        self.u = np.zeros((self.n, 2))
        self.used_sum = np.zeros((self.n, 2))
        self.e_sum = np.zeros((self.n, 2))
        self.held = np.zeros(self.n, int)
        self.x_prev = [None] * self.n
        self.version_prev = [0] * self.n
        self.jumps_booked = [0] * self.n
        self.done = np.zeros(self.n, bool)
        self.queues = [[] for _ in range(self.n)]
        self.ticks = 0
        self.steps = 0
        self.t_last = float(s.t)
        self.last_pos = starts.copy()
        self.wall = 0.0

    # CoPH's state of car i: q from the rear axle, v = speed along the heading (D5's v h).
    def state(self, s, i):
        h = float(s.heading[i])
        v = self.frame.v(float(s.speed[i]) * np.array([math.cos(h), math.sin(h)]))
        return np.r_[self.frame.q((s.x[i], s.y[i])), v]

    def sense(self, s, i, step, now):
        from grl_snam.coph.records import Observation, Reading

        x = self.state(s, i)
        pts = x[:2] + SENSOR_OFFSETS * self.cfg.sensor_radius
        rng = np.random.default_rng(np.random.SeedSequence([self.seed & 0x7FFFFFFF, i, step, 7919]))
        values = self.material.sample(self.frame.p(pts)) + rng.normal(0.0, 0.003, len(pts))
        values = np.maximum(0.001, values)
        readings = tuple(
            Reading(f"{self.seed}:{step}:{j}", now, (float(q[0]), float(q[1])), float(y))
            for j, (q, y) in enumerate(zip(pts, values))
        )
        pos = np.array([s.x[i], s.y[i]], float)
        if self.ctl.world is not None:
            centers, radii = visible_obstacles(
                self.ctl.world, pos, self.sensor_m, self.cfg.obstacle_cap
            )
        else:
            centers, radii = ray_obstacles(s, i, self.sensor_m, 0.15 * self.frame.s)
        obstacles = tuple(
            (float(c[0]), float(c[1]), float(r))
            for c, r in zip(self.frame.q(centers).reshape(-1, 2), np.asarray(radii) / self.frame.s)
        )
        return Observation(
            now, readings, obstacles, (float(x[0]), float(x[1])), self.cfg.sensor_radius
        )

    def _regoal(self, s, i):
        """Follow a new template goal, and slide the local goal along the line (D12)."""
        g = np.asarray(s.goal[i], float)
        pos = np.array([s.x[i], s.y[i]], float)
        moved = not np.allclose(g, self.goals[i])
        near = np.hypot(*(pos - self.local[i])) < 0.5 * self.reach_m and not np.allclose(
            self.local[i], g
        )
        if moved or near:
            self.goals[i] = g
            self.local[i] = local_goal(pos, g, self.reach_m)
            self.learners[i].goal = tuple(self.frame.q(self.local[i]))
            if moved:
                self.done[i] = False

    def coph_step(self, s):
        """One CoPH step for every car: observe the held step, exchange, update, act."""
        from grl_snam.coph.records import Transition

        now = self.steps * COPH_DT
        x = [self.state(s, i) for i in range(self.n)]
        live = []
        for i, a in enumerate(self.learners):
            self._regoal(s, i)
            if self.done[i]:
                continue
            if self.x_prev[i] is not None and self.held[i]:
                u_used = self.used_sum[i] / self.held[i]
                e_mean = self.e_sum[i] / self.held[i]
                self.ledgers[i].add_step(self.x_prev[i], x[i], u_used, e_mean, a.damping)
                risk = float(self.material.sample(np.array([[s.x[i], s.y[i]]]))[0])
                self.ledgers[i].risk_cost += COPH_DT * 4.0 * risk
                tr = Transition(self.steps - 1, now, tuple(self.x_prev[i]), tuple(u_used),
                                tuple(x[i]), self.version_prev[i])  # fmt: skip
                a.observe(self.sense(s, i, self.steps, now), tr)
            elif self.x_prev[i] is None:
                a.observe(self.sense(s, i, self.steps, now))
            self.used_sum[i] = 0.0
            self.e_sum[i] = 0.0
            self.held[i] = 0
            if bool(s.reached[i]):
                if self.records[i].arrived_at is None:
                    self.records[i].arrived_at = float(s.t)
                self.done[i] = True
                continue
            a.advance(x[i])
            live.append(i)
        cfg = self.cfg
        if self.arm != "off" and self.steps % cfg.update_every == 0:
            for i in live:
                for j in live:
                    if j == i:
                        continue
                    m = self.learners[i].export(x[i], str(j))
                    if m is None:
                        continue
                    size = m.bytes()
                    if (
                        self.records[i].bytes_sent + size
                        > self.learners[i].communication.byte_budget
                    ):
                        continue
                    self.records[i].bytes_sent += size
                    self.queues[j].append(m)
        for i in live:
            a, rec = self.learners[i], self.records[i]
            delivered = 0
            for m in sorted(self.queues[i], key=lambda m: (m.timestamp, m.identity)):
                event = a.assimilate(m)
                delivered += 1
                rec.reasons[event["reason"]] = rec.reasons.get(event["reason"], 0) + 1
            self.queues[i] = []
            if self.steps % cfg.update_every == 0 or delivered:
                d = a.update(x[i])
                # An accepted update reports its jump in V; a stage advance appends its own to
                # context_jumps (booked below).
                self.ledgers[i].shaping_jumps += float(d.get("shaping_jump", 0.0) or 0.0)
                if d.get("accepted"):
                    rec.accepted += 1
                else:
                    rec.rejected += 1
            u, _ = a.action(x[i])
            self.u[i] = np.asarray(u, float)
            self.x_prev[i] = x[i]
            self.version_prev[i] = a.version
        for i, a in enumerate(self.learners):
            new = a.context_jumps[self.jumps_booked[i] :]
            self.ledgers[i].shaping_jumps += sum(float(j.get("shaping_jump", 0.0)) for j in new)
            self.jumps_booked[i] = len(a.context_jumps)
            if self.done[i]:
                self.u[i] = 0.0
                self.x_prev[i] = None
        self.steps += 1

    def __call__(self, s, p):
        from grl_snam.coph.allocation import allocate

        began = time.perf_counter()
        pos = np.asarray(s.pos, float)
        for i in range(self.n):
            self.records[i].path_m += float(np.hypot(*(pos[i] - self.last_pos[i])))
            self.records[i].collided_ticks += int(bool(s.collided[i]))
        self.last_pos = pos.copy()
        if self.ticks % self.k == 0:
            self.coph_step(s)
        steer, accel = np.zeros(self.n), np.zeros(self.n)
        dt_coph = self.tick / math.sqrt(self.frame.s)
        for i in range(self.n):
            if self.done[i]:
                accel[i] = -np.clip(
                    s.speed[i] / max(self.tick, 0.25), -s.max_accel[i], s.max_accel[i]
                )
                continue
            al = allocate(self.u[i], float(s.heading[i]), float(s.speed[i]), float(s.wheelbase[i]),
                          float(s.max_steer[i]), float(s.max_accel[i]))  # fmt: skip
            steer[i], accel[i] = al.steer, al.accel
            used, e = np.asarray(al.used), np.asarray(al.unallocated)
            self.used_sum[i] += used
            self.e_sum[i] += e
            self.held[i] += 1
            self.ledgers[i].e2 += float(e @ e) * dt_coph
            self.ledgers[i].u2 += float(self.u[i] @ self.u[i]) * dt_coph
        self.ticks += 1
        self.t_last = float(s.t)
        self.wall += time.perf_counter() - began
        return steer, accel, {"coeffs": self.coeffs()}

    def predict(self, s):
        """``(steer, accel, {})`` for a state the template imagines (``rollout``): each car's held
        ``u``, allocated to that state. Nothing is learned or booked."""
        from grl_snam.coph.allocation import allocate

        steer, accel = np.zeros(s.n), np.zeros(s.n)
        for i in range(min(s.n, self.n)):
            if self.done[i]:
                accel[i] = -np.clip(
                    s.speed[i] / max(self.tick, 0.25), -s.max_accel[i], s.max_accel[i]
                )
                continue
            al = allocate(self.u[i], float(s.heading[i]), float(s.speed[i]), float(s.wheelbase[i]),
                          float(s.max_steer[i]), float(s.max_accel[i]))  # fmt: skip
            steer[i], accel[i] = al.steer, al.accel
        return steer, accel, {}

    def coeffs(self) -> dict:
        out = {k: np.zeros(self.n) for k in COEFFS}
        for i, (a, led, rec) in enumerate(zip(self.learners, self.ledgers, self.records)):
            pol = np.asarray(a.policy, float)
            row = dict(
                pi_risk=pol[0], pi_barrier=pol[1], damping=a.damping, version=a.version,
                work=led.work, heat=led.heat, kinetic_change=led.kinetic_change,
                residual=led.residual, unallocated_work=led.unallocated_work,
                allocation_residual=led.allocation_residual, shaping_jumps=led.shaping_jumps,
                risk_cost=led.risk_cost, bytes=rec.bytes_sent, accepted=rec.accepted,
                rejected=rec.rejected, stage=a.stages[-1].id,
            )  # fmt: skip
            for k, v in row.items():
                out[k][i] = float(v)
        return out

    def summary(self) -> list[dict]:
        cars = []
        for i, (a, led, rec) in enumerate(zip(self.learners, self.ledgers, self.records)):
            cars.append(dict(
                car=i, arrived=rec.arrived_at is not None, arrived_at=rec.arrived_at,
                path_m=rec.path_m, collided=rec.collided_ticks > 0,
                collided_ticks=rec.collided_ticks, bytes=rec.bytes_sent, accepted=rec.accepted,
                rejected=rec.rejected, packets=rec.reasons, version=a.version,
                pi=list(map(float, a.policy)), damping=float(a.damping),
                geometry_overflow=bool(a.geometry_overflow), ledger=led.as_dict(),
            ))  # fmt: skip
        return cars


def _starts_seed(starts, goals) -> int:
    """A seed from the starts and goals (to the centimetre): the same for every arm of a seed."""
    key = np.round(np.r_[np.ravel(starts), np.ravel(goals)], 2).tobytes()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


class CophCars:
    """The cars template's controller: ``controller(s, p) -> (steer, accel, {"coeffs": ...})``.

    ``world`` (a ``pycvc_gl.cars.World``) gives the adapter the occupancy grid; without it, as in
    the template's own loop, obstacles come from the rays. ``arm``, ``scale`` and ``seed``, when
    given, override ``ctl.arm``, ``ctl.scale`` and the seed taken from the starts."""

    def __init__(self, world=None, *, arm=None, scale=None, seed=None):
        self.world = world
        self.arm = arm
        self.scale = scale
        self.seed = seed
        self.run = None

    def _key(self, p):
        return (
            _arm_name(self.arm if self.arm is not None else p.get("arm", PARAMS["arm"])),
            float(self.scale if self.scale is not None else p.get("scale", 1.0)),
            float(p.get("local_goal_m", LOCAL_GOAL_M)),
        )

    def __call__(self, s, p=None):
        p = dict(PARAMS, **(p or {}))
        run = self.run
        # The template draws its predicted paths by calling the controller on imagined states
        # (pycvc_gl.cars.rollout). The learners must not learn from those: predict instead.
        if sys._getframe(1).f_code.co_name == "rollout":
            if run is None:
                return np.zeros(s.n), np.zeros(s.n), {}
            return run.predict(s)
        if run is None or run.n != s.n or float(s.t) < run.t_last - 1e-9 or run.key != self._key(p):
            run = self.run = _Run(self, s, p)
            run.key = self._key(p)
        return run(s, p)

    def cost(self, s, p, i, X, Y):
        """Car ``i``'s ``V`` (``p["view"]`` 0) or ``r`` (1) at world points ``(X, Y)``."""
        run = self.run
        X = np.asarray(X, float)
        if run is None or not 0 <= i < run.n:
            return np.zeros_like(X)
        Q = run.frame.q(np.stack([X.ravel(), np.asarray(Y, float).ravel()], 1))
        a = run.learners[i]
        field_ = (
            risk_grid(a, Q) if float((p or {}).get("view", 0.0)) >= 0.5 else potential_grid(a, Q)
        )
        return field_.reshape(X.shape)


#: The template's entry points: ``controller`` and ``cost`` on one CophCars.
controller = CophCars()
cost = controller.cost


# ── headless arms over paired seeds ───────────────────────────────────────────────────────────


def roboracer_length() -> float:
    pycvc = _pycvc()
    d = pycvc.vehicle_desc.roboracer()
    return 2.0 * float(d.half_extents()[0])


LAYOUTS = ("random", "convoy")


def convoy(world, n, length, seed=0):
    """A column of ``n`` cars 2.5 car lengths apart on one route across the square, each heading
    for its own goal at the far side (spread a car width or two apart). The seed shifts the column
    up to two car lengths sideways, so the patches move with it."""
    cx, cy = world.center
    half = 0.5 * (world.bounds[2] - world.bounds[0])
    dy = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, 4242])).uniform(
        -2, 2
    )
    y = cy + dy * length
    starts = np.array([(cx - 0.45 * half - 2.5 * length * k, y) for k in range(n)])
    goals = np.array([(cx + 0.75 * half, y + (k - (n - 1) / 2) * 2.0 * length) for k in range(n)])
    return starts, goals


def episode(arm, seed, *, cars=4, frames=1500, vehicle="roboracer", scale=1.0, layout="random",
            log=None):  # fmt: skip
    """One headless run on a RoboRacer-sized flat square (D3): ``arm`` (off, coph, raw), the
    template's ``seed`` (random starts and goals, or the ``convoy`` column's offset; the patches
    follow them). A car arrives within 1.5 car lengths of its goal (the template's own radius is
    at least 3 m, a Humvee's). Stops when every car has arrived. Returns ``{"cars": [...], ...}``;
    ``log`` writes the template's log there."""
    import warnings

    C = _template()
    if C is None:
        raise RuntimeError("coph-cars needs pycvc_gl.cars (pycvc-gl 3.7.12+)")
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r}: one of {LAYOUTS}")
    # The vendored learner's update converts a graph tensor to a float on every acceptance test.
    warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True")
    length = roboracer_length() if "roboracer" in vehicle else 4.84
    world = C.flat_world(length)
    placed = {}
    if layout == "convoy":
        placed = dict(zip(("starts", "goals"), convoy(world, int(cars), length, seed)))
    fleet = C.Cars(world, int(cars), vehicle, seed=int(seed), **placed)
    fleet.reach = 1.5 * float(np.max(fleet.length))
    fleet._state = fleet._read()
    ctl = CophCars(world, arm=arm, scale=scale, seed=int(seed))
    record = C.CarsLog() if log else None
    s = fleet.state()
    began = time.perf_counter()
    for _ in range(int(frames)):
        steer, accel, info = ctl(s, dict(PARAMS))
        s = fleet.step(steer, accel)
        if record is not None:
            record.record(s, fleet.last_command)
            record.coeffs(info["coeffs"], s.n)
        if ctl.run.done.all():
            break
    run = ctl.run
    out = dict(
        arm=arm, seed=int(seed), cars=run.summary(), t=float(s.t), ticks=run.ticks,
        steps=run.steps, wall_s=time.perf_counter() - began, controller_s=run.wall,
        world=world.info, vehicle=vehicle, scale=float(scale), patches=run.material.patches,
        material=run.material.source, ticks_per_step=run.k, layout=layout, reach_m=fleet.reach,
    )  # fmt: skip
    if record is not None:
        record.meta.update(world=world.info, cars=int(cars), seed=int(seed),
                           controller="coph-cars", arm=arm, summary=out)  # fmt: skip
        record.save(str(log))
    return out


def _episode_job(job):
    arm, seed, kw = job
    return episode(arm, seed, **kw)


def summarize(episodes) -> dict:
    """Per arm, over paired seeds: arrivals, path length, collisions, excess risk cost against the
    best arm (per seed and car), bytes, and the ledger."""
    by = {}
    for ep in episodes:
        by.setdefault(ep["arm"], {})[ep["seed"]] = ep
    seeds = sorted(set.intersection(*(set(v) for v in by.values()))) if by else []
    best = {}
    for seed in seeds:
        n = len(next(iter(by.values()))[seed]["cars"])
        for i in range(n):
            best[(seed, i)] = min(by[a][seed]["cars"][i]["ledger"]["risk_cost"] for a in by)
    arms = {}
    for arm, eps in by.items():
        cars = [(seed, c) for seed in seeds for c in eps[seed]["cars"]]
        if not cars:
            continue
        led = [c["ledger"] for _, c in cars]
        excess = [c["ledger"]["risk_cost"] - best[(seed, c["car"])] for seed, c in cars]
        arrived = [c for _, c in cars if c["arrived"]]
        u2 = sum(x["allocation_residual"] ** 2 for x in led)
        packets = {}
        for _, c in cars:
            for reason, k in c.get("packets", {}).items():
                packets[reason] = packets.get(reason, 0) + k
        damping = [c.get("damping", float("nan")) for _, c in cars]
        arms[arm] = dict(
            packets=packets, damping_mean=float(np.mean(damping)),
            damping_at_floor=sum(abs(d - 0.15) < 1e-9 for d in damping),
            pi_risk_mean=float(np.mean([c.get("pi", [np.nan])[0] for _, c in cars])),
            seeds=seeds, cars=len(cars), arrivals=len(arrived),
            arrival_s=float(np.mean([c["arrived_at"] for c in arrived])) if arrived else None,
            path_m=float(np.mean([c["path_m"] for c in arrived])) if arrived else None,
            collided_cars=sum(c["collided"] for _, c in cars),
            collided_ticks=sum(c["collided_ticks"] for _, c in cars),
            excess_risk_mean=float(np.mean(excess)), excess_risk_max=float(np.max(excess)),
            risk_cost=float(np.mean([x["risk_cost"] for x in led])),
            bytes=float(np.mean([c["bytes"] for _, c in cars])),
            accepted=sum(c["accepted"] for _, c in cars),
            rejected=sum(c["rejected"] for _, c in cars),
            allocation_residual=float(math.sqrt(u2 / len(led))),
            work=float(np.sum([x["work"] for x in led])),
            unallocated_work=float(np.sum([x["unallocated_work"] for x in led])),
            heat=float(np.sum([x["heat"] for x in led])),
            kinetic_change=float(np.sum([x["kinetic_change"] for x in led])),
            ledger_residual=float(np.sum([x["residual"] for x in led])),
            max_step_residual=float(np.max([x["max_step_residual"] for x in led])),
            shaping_jumps=float(np.sum([x["shaping_jumps"] for x in led])),
            geometry_overflow=sum(c["geometry_overflow"] for _, c in cars),
        )  # fmt: skip
    return dict(seeds=seeds, arms=arms)


def report(summary, episodes) -> str:
    """REPORT.md: the ledger and the allocation residual first (§8), then the outcomes."""
    arms = summary["arms"]
    order = [a for a in ARMS if a in arms]
    seeds = summary["seeds"]
    ep0 = episodes[0] if episodes else {}
    reach = ep0.get("reach_m")
    lines = [
        "# CoPH drives the cars (T4, demo 2b): exploratory",
        "",
        f"{ep0.get('vehicle', 'roboracer')} at s = {ep0.get('scale', 1.0):g} on a flat square, "
        f"{len(ep0.get('cars', []))} cars, layout {ep0.get('layout', 'random')}, paired seeds "
        f"{seeds}; CoPH steps every {ep0.get('ticks_per_step', '?')} ticks"
        + (f"; a car arrives within {reach:.2f} m" if reach else "")
        + ". Each seed gives every arm the same starts, goals and risk patches.",
        "",
        "## The ledger and the allocation (per unit mass, CoPH's frame, summed over cars and seeds)",
        "",
        "| arm | allocation residual | work (u_used . dq) | unallocated work (e . dq) | heat | "
        "kinetic change | ledger residual | worst step residual | shaping jumps |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in order:
        r = arms[a]
        lines.append(
            f"| {a} | {r['allocation_residual']:.3f} | {r['work']:.3f} | {r['unallocated_work']:.3f} "
            f"| {r['heat']:.3f} | {r['kinetic_change']:.3f} | {r['ledger_residual']:.3f} "
            f"| {r['max_step_residual']:.3g} | {r['shaping_jumps']:.3f} |"
        )
    lines += [
        "",
        "The allocation residual is sqrt(sum |e|^2 / sum |u|^2) over the ticks: the share of CoPH's "
        "force the cars did not take. The ledger residual is kinetic change - work + heat, with heat "
        "booked by CoPH's plant at each learner's damping: zero on CoPH's own plant, so here it "
        "measures how far a car's motion is from a damped point mass under the force it took.",
        "",
        "## Outcomes",
        "",
        "| arm | arrived | mean arrival (s) | mean path (m) | cars touching | excess risk (mean / max) "
        "| bytes per car | updates accepted / rejected |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in order:
        r = arms[a]
        t = "-" if r["arrival_s"] is None else f"{r['arrival_s']:.1f}"
        pth = "-" if r["path_m"] is None else f"{r['path_m']:.1f}"
        lines.append(
            f"| {a} | {r['arrivals']}/{r['cars']} | {t} | {pth} | {r['collided_cars']} "
            f"| {r['excess_risk_mean']:.3f} / {r['excess_risk_max']:.3f} | {r['bytes']:.0f} "
            f"| {r['accepted']} / {r['rejected']} |"
        )
    lines += [
        "",
        "## Learning and packets",
        "",
        "| arm | final damping (mean; at the 0.15 floor) | final pi_risk (mean) | packets by the "
        "receiver's verdict |",
        "|---|---|---|---|",
    ]
    for a in order:
        r = arms[a]
        verdicts = ", ".join(f"{k} {v}" for k, v in sorted(r["packets"].items())) or "none sent"
        lines.append(
            f"| {a} | {r['damping_mean']:.3f}; {r['damping_at_floor']}/{r['cars']} "
            f"| {r['pi_risk_mean']:.3f} | {verdicts} |"
        )
    lines += [
        "",
        "A receiver refuses a packet as `unsupported` when none of its anchors lies in the "
        "receiver's kept stages (`CooperativeLearner.assimilate`).",
        "",
        "Excess risk is each car's risk cost (4 r~ dt, summed) minus the lowest any arm got on the "
        "same seed and car. Nothing here is a navigation claim: the seeds are few, and the "
        "formulation treats a fully actuated point (roadmap §3.3.2, §8).",
        "",
    ]
    return "\n".join(lines)


def run_arms(args) -> int:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arm_names or list(ARMS)
    kw = dict(cars=args.cars, frames=args.frames, vehicle=args.vehicle, scale=args.scale,
              layout=args.layout)  # fmt: skip
    jobs = [(a, s, dict(kw, log=str(out / f"{a}_{s}.npz") if args.logs else None))
            for s in args.seeds for a in arms]  # fmt: skip
    episodes = []
    if args.workers <= 1:
        for job in jobs:
            episodes.append(_episode_job(job))
            print(_line(episodes[-1]), flush=True)
    else:
        # Spawned, as grl-snam coph's runners are: forking a process that runs torch can deadlock.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for ep in pool.map(_episode_job, jobs):
                episodes.append(ep)
                print(_line(ep), flush=True)
    summary = summarize(episodes)
    (out / "episodes.json").write_text(json.dumps(episodes, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    text = report(summary, episodes)
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def _line(ep) -> str:
    cars = ep["cars"]
    return (f"coph-cars: {ep['arm']} seed {ep['seed']}: {sum(c['arrived'] for c in cars)}/{len(cars)} "
            f"arrived, {ep['t']:.1f} s simulated in {ep['wall_s']:.0f} s")  # fmt: skip


def _arms_args(argv):
    p = argparse.ArgumentParser(prog="python -m grl_snam.ariadne coph-cars --arms",
                                description="CoPH's arms on the cars, headless, over paired seeds")  # fmt: skip
    p.add_argument("--arms", dest="arm_names", nargs="*", choices=ARMS, default=None,
                   help="which arms (default all: off coph raw)")  # fmt: skip
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--cars", type=int, default=4)
    p.add_argument("--frames", type=int, default=1500, help="ticks per episode at most (0.06 s)")
    p.add_argument("--vehicle", default="roboracer")
    p.add_argument("--layout", choices=LAYOUTS, default="random",
                   help="random starts and goals (the template's), or a convoy on one route")  # fmt: skip
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--logs", action="store_true", help="also write each episode's template log")
    return p.parse_args(argv)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--arms" in argv:
        return run_arms(_arms_args(argv))
    C = _template()
    if C is None:
        print("grl-snam ariadne coph-cars runs on libcvc's cars template, pycvc_gl.cars, which "
              "needs pycvc-gl 3.7.12+ (cvcpkg install pycvc-gl-cp312, or your Python's column).",
              file=sys.stderr)  # fmt: skip
        return 2
    defaults = {"--world": "flat", "--vehicle": "roboracer", "--cars": "4"}
    for flag, value in defaults.items():
        if flag not in argv:
            argv += [flag, value]
    ctl = CophCars()
    return C.run(ctl, PARAMS, cost=ctl.cost, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
