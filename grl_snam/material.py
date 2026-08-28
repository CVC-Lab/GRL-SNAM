"""Material-aware navigation for the packaged simulator (default-on when attached).

This is the sim-frame integration of the material-aware method whose faithful
research-core port lives in root ``material_nav.py``.  A :class:`MaterialGrid`
carries per-cell terrain data on the SAME raster as the occupancy grid:

* ``risk`` — smoothed material risk r~ in [0, 1] (mud, water skirts, rubble…);
  a *soft* cost: it biases the planner and pushes the drive, never blocks.
* ``hard`` — hard-hazard cells (water, cliffs…): lethal-but-not-geometry.
  Physical blocking stays occupancy's job; hard cells get a large finite
  planner surcharge and an always-on barrier force.

Attaching a MaterialGrid to a scenario/swarm activates the whole feature in
pure Python with no further flags — forces, witness gate, and planner cost.
No MaterialGrid (the default everywhere) leaves every existing trajectory
bit-for-bit unchanged.

The executed field (per substep, inside the rollouts in ``sdf_nav``):

    F_soft = -lam_soft_eff * grad r~        lam_soft_eff = lam_soft * gate
    db     = -sigmoid(k_sharp * (d_hat_sdf_m - phi_m))
    F_hard = -lam_hard * db * grad_phi
    F      = F_bar + F_goal + F_soft + F_hard          (+ steering bias)

Units are pinned to avoid the classic rescale trap: ``phi_m`` (distance to the
nearest hard cell) stays in WORLD METRES end-to-end, so the barrier constants
``k_sharp`` (1/m) and ``d_hat_sdf_m`` (m) are the source method's values with
no conversion anywhere.  Risk gradients are stored per NORMALIZED unit so
F_soft composes directly with the goal spring's force scale; hazard gradients
are metres-per-metre (a near-unit outward direction field).

The witness gate here is the NORMATIVE bit-spec for the C++ twin
(``cvc::nav::witness_gate``): float64 end-to-end, a shared exact direction
table, sequential accumulation, round-half-even cells.  The research-core gate
in ``material_nav`` keeps the source repo's float32 math verbatim; the two are
the same algorithm and may disagree only in the last ulp on adversarial ties.

Native forwarding is OPT-IN via ``GRL_SNAM_MATERIAL_BACKEND=native`` (unlike
the geometry kernels, which default native when present) — deliberate: the
pure-Python path is the feature's default; the C++ path is the accelerator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import sdf_nav

from . import nav_native as _native

# The 16 gate ray directions (row, col) = (sin th, cos th), th = 2*pi*k/16,
# as exact float64 constants shared verbatim with the C++ twin — keeping libm
# sin/cos out of the BIT contract.  Generated once; do not "tidy".
_DIRS_16 = (
    (0.0, 1.0),
    (0.3826834323650898, 0.9238795325112867),
    (0.7071067811865475, 0.7071067811865476),
    (0.9238795325112867, 0.38268343236508984),
    (1.0, 6.123233995736766e-17),
    (0.9238795325112867, -0.3826834323650897),
    (0.7071067811865476, -0.7071067811865475),
    (0.3826834323650899, -0.9238795325112867),
    (1.2246467991473532e-16, -1.0),
    (-0.38268343236508967, -0.9238795325112868),
    (-0.7071067811865475, -0.7071067811865477),
    (-0.9238795325112865, -0.38268343236509034),
    (-1.0, -1.8369701987210297e-16),
    (-0.9238795325112866, 0.38268343236509),
    (-0.7071067811865477, 0.7071067811865474),
    (-0.3826834323650904, 0.9238795325112865),
)


def gate_directions(count: int):
    """The gate's ray direction set. 16 (the default) uses the shared exact
    table; other counts are computed (and are outside the BIT contract)."""
    if count == 16:
        return _DIRS_16
    return tuple(
        (math.sin(2.0 * math.pi * k / count), math.cos(2.0 * math.pi * k / count))
        for k in range(count)
    )


# ---------------------------------------------------------------------------
# Gaussian blur (no scipy dependency; pinned op order for the C++ BIT twin)
# ---------------------------------------------------------------------------


def gaussian_kernel(sigma: float) -> np.ndarray:
    """scipy-compatible 1-D Gaussian taps, computed with a PINNED op order:
    ``exp(-0.5/(sigma*sigma) * k*k)``, radius ``int(4*sigma + 0.5)``, then a
    SEQUENTIAL normalization sum.  Scalar ``math.exp`` (libm), so the C++ twin
    (``std::exp`` on the identical expression) reproduces it bit-for-bit
    on-host."""
    radius = int(4.0 * float(sigma) + 0.5)
    inv = -0.5 / (float(sigma) * float(sigma))
    taps = [math.exp(inv * (k * k)) for k in range(-radius, radius + 1)]
    total = 0.0
    for t in taps:  # sequential, never np.sum (pairwise) — the C++ twin matches
        total += t
    return np.array([t / total for t in taps], np.float64)


def _reflect_idx(idx: np.ndarray, n: int) -> np.ndarray:
    # scipy mode='reflect' is edge-REPEATING (d c b a | a b c d), i.e.
    # np.pad(mode="symmetric") — NOT np.pad(mode="reflect"). Single bounce.
    idx = np.where(idx < 0, -idx - 1, idx)
    return np.where(idx >= n, 2 * n - 1 - idx, idx)


def gaussian_blur(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Separable Gaussian blur, float64 accumulation through BOTH passes
    (rows then cols), symmetric/edge-repeat boundary, one float32 store at the
    end.  Tap loop order is the accumulation order — the C++ twin copies it."""
    if sigma <= 0.0:
        return np.asarray(a, np.float32).copy()
    w = gaussian_kernel(sigma)
    radius = len(w) // 2
    out = np.asarray(a, np.float64)
    for axis in (0, 1):
        n = out.shape[axis]
        if n < radius + 1:
            raise ValueError(f"grid axis {axis} ({n}) smaller than blur radius+1 ({radius + 1})")
        acc = np.zeros_like(out)
        base = np.arange(n)
        for j, k in enumerate(range(-radius, radius + 1)):
            idx = _reflect_idx(base + k, n)
            acc = acc + w[j] * np.take(out, idx, axis=axis)
        out = acc
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class GateParams:
    """Frame-wise feasibility-witness gate parameters (source defaults, with
    the two cell-denominated quantities re-expressed in metres so they survive
    grid-resolution changes; the source BEV was 0.5 m/cell, this sim's default
    story grid is ~2.1 m/cell).

    * ``horizon_m``: witness ray length. 25 m ~= the source's 12 cells at the
      default 96-cell/±100 m story grid (the source's 6 m would be 3 cells
      here — myopic).
    * ``hard_margin_m``: min clearance to hard terrain along a feasible ray.
      ``None`` = 2 grid cells, the source's margin (1.0 m at 0.5 m/cell).
    """

    primitive_count: int = 16
    horizon_m: float = 25.0
    hard_margin_m: float | None = None
    improvement_margin: float = 0.05  # mean-ray-risk units (exp1 convention)
    material_trigger: float = 0.45
    progress_slack_cells: float = 0.5


@dataclass
class MaterialParams:
    """Sim-frame material force/cost parameters.

    All four force constants are SIM-FRAME tuning targets, not ported values
    (the formulas are the ported part).  The source's exp9 fixed-coefficient
    result (lam_soft = 1.5 matches learned) ran in the DFC pixel frame with
    Sobel-scaled gradients; here lam_soft = 1.5 measurably LAUNCHES a vehicle
    off-world (the normalized-frame gradients are ~an order hotter), and the
    source's barrier reach d_hat_sdf = 3 m — six cells on its 0.5 m/cell BEV —
    is under ONE cell on the default story grid (2.1 m/cell), i.e. invisible
    until contact.  Defaults below were validated behaviorally (blob-detour,
    hazard no-entry, still-reaches acceptance tests): lam_soft 0.5, lam_hard
    1.0, and the barrier re-widened to the source's CELL-relative reach
    (~6 cells => 12 m) with the ramp steepness scaled to match (k*d_hat
    preserved: 5*3 == 1.25*12).  The source's metre values live on in
    ``material_nav`` (research core) and remain reachable here by explicitly
    passing k_sharp=5, d_hat_sdf_m=3.

    Scope note: the executed field is a LOCAL layer.  In the source method it
    always ran under a waypoint scaffold from a planner, and the same holds
    here — with a planner (FogScenario/Squad) the risk surcharge routes around
    hazards and the forces polish locally; a planner-less reactive agent
    (Swarm/sim_world) gets the no-entry guarantee but can dead-end against a
    hazard squarely blocking its goal line (a potential-field minimum, exactly
    as in the source's field)."""

    lam_soft: float = 0.5
    lam_hard: float = 1.0
    k_sharp: float = 1.25  # 1/m (source: 5.0)
    d_hat_sdf_m: float = 12.0  # m (source: 3.0)
    risk_weight: float = 10.0  # A* surcharge per unit risk (grid-step units)
    hard_penalty: float = 25.0  # A* surcharge on hard cells (finite: bias, not forbid)
    gate: GateParams | None = None
    gate_enabled: bool = True

    def __post_init__(self):
        if self.gate is None:
            self.gate = GateParams()


# ---------------------------------------------------------------------------
# MaterialGrid — the raster + derived planes
# ---------------------------------------------------------------------------


class MaterialGrid:
    """Per-cell material data on the occupancy raster, plus derived planes.

    Derived pipeline (pinned; the C++ ``material_build`` twin is BIT-identical):
      risk       = gaussian_blur(risk_raw, sigma) clipped to [0,1], f32
      phi_m      = sqrt(edt2(hard)) * cell_w      (f64 chain, one f32 store)
      grad r~    = np.gradient(risk_f32) / float32(cell_w * scale)   [per
                   normalized unit]
      grad phi   = np.gradient(phi_m_f32) / float32(cell_w)          [m per m]
    ``np.gradient`` runs on the f32-STORED planes (the build_sdf precedent —
    and exactly what the C++ f32 ``grad1d`` computes)."""

    def __init__(
        self,
        risk: np.ndarray,
        hard: np.ndarray,
        bounds,
        center,
        scale: float,
        *,
        sigma: float = 1.0,
        params: MaterialParams | None = None,
    ):
        risk = np.asarray(risk, np.float32)
        hard_b = np.asarray(hard).astype(bool)
        if risk.shape != hard_b.shape:
            raise ValueError(f"risk {risk.shape} and hard {hard_b.shape} shapes differ")
        self.risk_raw = risk.copy()
        self.hard = hard_b.copy()
        self.bounds = tuple(float(b) for b in bounds)
        self.center = (float(center[0]), float(center[1]))
        self.scale = float(scale)
        self.sigma = float(sigma)
        self.params = params if params is not None else MaterialParams()
        ny, nx = self.risk_raw.shape
        self.cell_w = (self.bounds[2] - self.bounds[0]) / (nx - 1)
        #: bumped on every stamp_* — the scenario's cue to re-cost and replan.
        self.version = 0
        self._derive()

    # -- derived planes ------------------------------------------------------
    def _derive(self) -> None:
        if _native.material_enabled():
            planes = _native.material_build(
                self.risk_raw, self.hard, self.cell_w, self.scale, self.sigma
            )
            self.risk, self.phi_hard_m, self.grad_rx, self.grad_ry, self.grad_px, self.grad_py = (
                planes
            )
        else:
            self.risk = np.clip(gaussian_blur(self.risk_raw, self.sigma), 0.0, 1.0).astype(
                np.float32
            )
            self.phi_hard_m = (np.sqrt(sdf_nav._edt2(self.hard)) * self.cell_w).astype(np.float32)
            gy, gx = np.gradient(self.risk)  # f32 in, f32 out
            denom_n = np.float32(self.cell_w * self.scale)
            self.grad_rx = gx / denom_n
            self.grad_ry = gy / denom_n
            pgy, pgx = np.gradient(self.phi_hard_m)
            denom_w = np.float32(self.cell_w)
            self.grad_px = pgx / denom_w
            self.grad_py = pgy / denom_w
        self._field = None

    def field(self, device: str = "cpu") -> MaterialField:
        """The torch sampler over the derived planes (lazy, rebuilt on stamp)."""
        if self._field is None or self._field.dev != torch.device(device):
            self._field = MaterialField(self, device=device)
        return self._field

    # -- mutation (mud-onset-style demo events) ------------------------------
    # Deviation from the source's dynamic-event pipeline, documented: events
    # there paint on the already-blurred cache and re-blur at sigma=0.75 every
    # tick; here a stamp edits the RAW plane and re-derives once at this
    # grid's single sigma.
    def stamp_risk(self, r0: int, r1: int, c0: int, c1: int, value: float) -> None:
        self.risk_raw[r0:r1, c0:c1] = np.float32(value)
        self._derive()
        self.version += 1

    def stamp_hard(self, r0: int, r1: int, c0: int, c1: int, value: bool = True) -> None:
        self.hard[r0:r1, c0:c1] = bool(value)
        self._derive()
        self.version += 1

    # -- planner cost --------------------------------------------------------
    def cost_raster(self) -> np.ndarray:
        """Per-cell A* surcharge (float64, grid-step units, >= 0): soft risk
        bias + a large-but-finite penalty on hard cells.  Additive on cell
        entry (the house astar contract) — unlike the source's multiplicative
        step scaling, so diagonal moves undercount by sqrt(2); documented.
        Physically blocked cells are occupancy's job (A* never enters them)."""
        p = self.params
        return p.risk_weight * self.risk.astype(np.float64) + p.hard_penalty * self.hard.astype(
            np.float64
        )


class MaterialField:
    """Torch sampler over a MaterialGrid's [1, 6, H, W] plane stack.

    Channels: [r~, phi_m, dr~/dx_n, dr~/dy_n, dphi/dx_w, dphi/dy_w].
    ``sample`` uses the exact ``SDFField.sample`` coordinate chain (normalized
    -> world -> grid, bilinear, align_corners, border) so the C++
    ``material_sample`` twin can copy the proven op order.  The single shared
    plane broadcasts over any batch size (material is world truth, not
    per-agent belief)."""

    def __init__(self, grid: MaterialGrid, device: str = "cpu"):
        self.dev = torch.device(device)
        planes = np.stack(
            [grid.risk, grid.phi_hard_m, grid.grad_rx, grid.grad_ry, grid.grad_px, grid.grad_py],
            0,
        )[None]
        self.field = torch.from_numpy(planes).float().to(self.dev)
        self.mnx, self.mny, self.mxx, self.mxy = grid.bounds
        self.cx, self.cy = grid.center
        self.S = grid.scale

    def sample(self, on: torch.Tensor):
        """on: ``[B,2]`` normalized -> ``(risk[B], phi_m[B], grad_r[B,2],
        grad_phi[B,2])``."""
        wx = on[:, 0] / self.S + self.cx
        wy = on[:, 1] / self.S + self.cy
        gx = 2 * (wx - self.mnx) / (self.mxx - self.mnx) - 1
        gy = 2 * (wy - self.mny) / (self.mxy - self.mny) - 1
        grid = torch.stack([gx, gy], -1)[None, None]  # [1,1,B,2]
        out = F.grid_sample(
            self.field, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )[
            0, :, 0, :
        ].t()  # [B,6]
        return out[:, 0], out[:, 1], out[:, 2:4], out[:, 4:6]


# ---------------------------------------------------------------------------
# Witness gate — Layer-B normative reference (float64 end-to-end)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """One witness-gate evaluation.  ``active`` gates lam_soft only; the
    selected ray is an activation witness, never the executed command."""

    active: bool
    nominal_risk: float
    best_risk: float
    feasible_count: int
    direction_rc: tuple
    endpoint_rc: tuple
    min_clearance_m: float


def _ray_risk(risk, gate_hard, clear_m, pr, pc, dr, dc, horizon_cells, hard_margin_m):
    """Sequential f64 walk of one ray.  Order pinned to the source: the float
    point is bounds-checked BEFORE rounding; risk is accumulated BEFORE the
    hard/clearance break.  Cells: round-half-even then clip."""
    rows, cols = risk.shape
    acc = 0.0
    count = 0
    min_clear = math.inf
    feasible = True
    for t in range(1, horizon_cells + 1):
        qr = pr + t * dr
        qc = pc + t * dc
        if not (0.0 <= qr < rows and 0.0 <= qc < cols):
            feasible = False
            break
        r = min(max(int(round(qr)), 0), rows - 1)
        c = min(max(int(round(qc)), 0), cols - 1)
        acc += float(risk[r, c])
        count += 1
        clear = float(clear_m[r, c])
        if clear < min_clear:
            min_clear = clear
        if gate_hard[r, c] or clear < hard_margin_m:
            feasible = False
            break
    mean = acc / count if count else math.inf
    return mean, feasible, min_clear


def witness_gate(
    risk: np.ndarray,
    gate_hard: np.ndarray,
    clear_m: np.ndarray,
    pos_rc,
    goal_rc,
    *,
    horizon_cells: int,
    hard_margin_m: float,
    primitive_count: int = 16,
    improvement_margin: float = 0.05,
    material_trigger: float = 0.45,
    progress_slack_cells: float = 0.5,
) -> GateResult:
    """The frame-wise feasibility witness on grid arrays, in CONTINUOUS CELL
    coordinates (row, col float64 — callers convert from world once).

    ``gate_hard`` must already include occupancy (hard | occ): a ray through a
    building is not evidence of a feasible detour.  ``clear_m`` is the metres
    clearance-to-gate_hard plane.  Activation:

        active = feasible_count > 0
               and nominal_risk >= material_trigger
               and nominal_risk - best_risk >= improvement_margin
    """
    pr, pc = float(pos_rc[0]), float(pos_rc[1])
    gr, gc = float(goal_rc[0]), float(goal_rc[1])
    dgr, dgc = gr - pr, gc - pc
    # sqrt(x*x + y*y), NOT math.hypot: CPython's hypot is its own correctly-
    # rounded algorithm and can differ from libm's in the last ulp — this
    # spelling is reproducible verbatim in the C++ twin.
    norm = math.sqrt(dgr * dgr + dgc * dgc)
    if norm < 1e-8:
        ndr, ndc = 0.0, 0.0
    else:
        ndr, ndc = dgr / norm, dgc / norm
    nominal_risk, _, _ = _ray_risk(
        risk, gate_hard, clear_m, pr, pc, ndr, ndc, horizon_cells, hard_margin_m
    )

    best = math.inf
    best_dir = (0.0, 0.0)
    best_clear = math.nan
    feasible_count = 0
    goal_dist = norm
    for dr, dc in gate_directions(primitive_count):
        er = pr + horizon_cells * dr
        ec = pc + horizon_cells * dc
        per, pec = gr - er, gc - ec
        if math.sqrt(per * per + pec * pec) >= goal_dist - progress_slack_cells:
            continue
        cand, feasible, cand_clear = _ray_risk(
            risk, gate_hard, clear_m, pr, pc, dr, dc, horizon_cells, hard_margin_m
        )
        if not feasible:
            continue
        feasible_count += 1
        if cand < best:
            best = cand
            best_dir = (dr, dc)
            best_clear = cand_clear
    active = (
        feasible_count > 0
        and nominal_risk >= material_trigger
        and nominal_risk - best >= improvement_margin
    )
    return GateResult(
        active=bool(active),
        nominal_risk=nominal_risk,
        best_risk=best,
        feasible_count=feasible_count,
        direction_rc=best_dir,
        endpoint_rc=(pr + horizon_cells * best_dir[0], pc + horizon_cells * best_dir[1]),
        min_clearance_m=best_clear,
    )


def witness_gate_batch(
    risk: np.ndarray,
    gate_hard: np.ndarray,
    clear_m: np.ndarray,
    pos_rc: np.ndarray,
    goal_rc: np.ndarray,
    *,
    horizon_cells: int,
    hard_margin_m: float,
    primitive_count: int = 16,
    improvement_margin: float = 0.05,
    material_trigger: float = 0.45,
    progress_slack_cells: float = 0.5,
):
    """Vectorized-over-agents witness gate, byte-identical to N serial
    :func:`witness_gate` calls (tested).  The per-ray step loop stays
    SEQUENTIAL (t ascending, f64 elementwise adds — the same accumulation
    order as the serial walk); only the agent axis is vectorized.

    Returns ``(active[N] bool, nominal[N], best[N], feasible_count[N])``.
    """
    rows, cols = risk.shape
    pos = np.asarray(pos_rc, np.float64)
    goal = np.asarray(goal_rc, np.float64)
    n = pos.shape[0]
    risk64 = risk.astype(np.float64)
    clear64 = clear_m.astype(np.float64)
    hard_b = gate_hard.astype(bool)

    def walk(dr, dc):
        """Walk one direction (per-agent [N] arrays) sequentially in t."""
        acc = np.zeros(n, np.float64)
        count = np.zeros(n, np.int64)
        alive = np.ones(n, bool)
        feasible = np.ones(n, bool)
        for t in range(1, horizon_cells + 1):
            qr = pos[:, 0] + t * dr
            qc = pos[:, 1] + t * dc
            inb = (qr >= 0.0) & (qr < rows) & (qc >= 0.0) & (qc < cols)
            oob = alive & ~inb
            feasible[oob] = False
            alive = alive & inb
            if not alive.any():
                break
            r = np.clip(np.rint(qr).astype(np.int64), 0, rows - 1)
            c = np.clip(np.rint(qc).astype(np.int64), 0, cols - 1)
            rv = risk64[r, c]
            acc = np.where(alive, acc + rv, acc)
            count = np.where(alive, count + 1, count)
            trip = alive & (hard_b[r, c] | (clear64[r, c] < hard_margin_m))
            feasible[trip] = False
            alive = alive & ~trip
        mean = np.where(count > 0, acc / np.maximum(count, 1), np.inf)
        return mean, feasible

    dg = goal - pos
    # sqrt-of-sum-of-squares, matching the serial reference (not np.hypot).
    goal_dist = np.sqrt(dg[:, 0] * dg[:, 0] + dg[:, 1] * dg[:, 1])
    ndir = np.where((goal_dist < 1e-8)[:, None], 0.0, dg / np.maximum(goal_dist, 1e-300)[:, None])
    # nominal: per-agent direction — walk with [N] dr/dc vectors
    nominal, _ = walk(ndir[:, 0], ndir[:, 1])

    best = np.full(n, np.inf)
    feasible_count = np.zeros(n, np.int64)
    for dr, dc in gate_directions(primitive_count):
        er = pos[:, 0] + horizon_cells * dr
        ec = pos[:, 1] + horizon_cells * dc
        per = goal[:, 0] - er
        pec = goal[:, 1] - ec
        progress = np.sqrt(per * per + pec * pec) < goal_dist - progress_slack_cells
        cand, feasible = walk(np.float64(dr), np.float64(dc))
        ok = progress & feasible
        feasible_count += ok.astype(np.int64)
        better = ok & (cand < best)
        best = np.where(better, cand, best)
    with np.errstate(invalid="ignore"):
        # inf - inf (agent off-grid: nominal AND best both infeasible) is nan;
        # nan >= margin is False — exactly the serial reference's semantics.
        active = (
            (feasible_count > 0)
            & (nominal >= material_trigger)
            & (nominal - best >= improvement_margin)
        )
    return active, nominal, best, feasible_count


# ---------------------------------------------------------------------------
# MaterialRuntime — what a scenario/swarm wires into the navigator(s)
# ---------------------------------------------------------------------------


class MaterialRuntime:
    """Owns the (material, occupancy)-derived gate surfaces and hands the
    navigator its per-tick lambdas.

    The gate's feasibility must see PHYSICAL obstacles too (a ray through a
    building is not a detour), so ``gate_hard = material.hard | occupancy`` and
    ``gate_clear_m`` is the metres-EDT of that union — recomputed when the
    planning occupancy is rebuilt or the material grid is stamped (one extra
    EDT per rebuild, the same cost class as the SDF rebuild)."""

    def __init__(self, grid: MaterialGrid):
        self.grid = grid
        self._occ = None
        self._seen_version = grid.version
        self._gate_hard = None
        self._gate_clear_m = None
        self._cost = None
        self._horizon_cells = None
        self._hard_margin_m = None
        self.last_gate: GateResult | None = None
        self.update_occ(np.zeros(grid.risk_raw.shape, bool))

    # -- derived-surface upkeep ---------------------------------------------
    def _recompute(self) -> None:
        g = self.grid
        self._gate_hard = np.logical_or(g.hard, self._occ)
        self._gate_clear_m = (np.sqrt(sdf_nav._edt2(self._gate_hard)) * g.cell_w).astype(np.float32)
        self._cost = None
        gp = g.params.gate
        self._horizon_cells = max(1, int(round(gp.horizon_m / g.cell_w)))
        self._hard_margin_m = (
            2.0 * g.cell_w if gp.hard_margin_m is None else float(gp.hard_margin_m)
        )

    def update_occ(self, occ: np.ndarray) -> None:
        """Called with the composited planning occupancy whenever it rebuilds."""
        occ = np.asarray(occ, bool)
        if self._occ is not None and np.array_equal(occ, self._occ):
            return
        self._occ = occ.copy()
        self._recompute()

    def consume_version_change(self) -> bool:
        """True (once) when the material grid was stamped since last checked;
        refreshes the derived surfaces."""
        if self.grid.version == self._seen_version:
            return False
        self._seen_version = self.grid.version
        self._recompute()
        return True

    # -- planner cost --------------------------------------------------------
    def cost_raster(self) -> np.ndarray:
        if self._cost is None:
            self._cost = self.grid.cost_raster()
        return self._cost

    # -- world -> continuous cell coords (f64, unrounded) -------------------
    def _world_to_cell_f(self, x: float, y: float):
        mnx, mny, mxx, mxy = self.grid.bounds
        ny, nx = self.grid.risk_raw.shape
        cc = (float(x) - mnx) / (mxx - mnx) * (nx - 1)
        cr = (float(y) - mny) / (mxy - mny) * (ny - 1)
        return cr, cc

    # -- the per-tick gate + lambdas ----------------------------------------
    def gate(self, pos_world, goal_world) -> GateResult:
        gp = self.grid.params.gate
        if _native.material_enabled():
            res = _native.witness_gate(
                self.grid.risk,
                self._gate_hard,
                self._gate_clear_m,
                self._world_to_cell_f(*pos_world),
                self._world_to_cell_f(*goal_world),
                horizon_cells=self._horizon_cells,
                hard_margin_m=self._hard_margin_m,
                primitive_count=gp.primitive_count,
                improvement_margin=gp.improvement_margin,
                material_trigger=gp.material_trigger,
                progress_slack_cells=gp.progress_slack_cells,
            )
        else:
            res = witness_gate(
                self.grid.risk,
                self._gate_hard,
                self._gate_clear_m,
                self._world_to_cell_f(*pos_world),
                self._world_to_cell_f(*goal_world),
                horizon_cells=self._horizon_cells,
                hard_margin_m=self._hard_margin_m,
                primitive_count=gp.primitive_count,
                improvement_margin=gp.improvement_margin,
                material_trigger=gp.material_trigger,
                progress_slack_cells=gp.progress_slack_cells,
            )
        self.last_gate = res
        return res

    def lambdas(self, pos_world, goal_world):
        """(lam_soft_eff, lam_hard) for this tick: the gate multiplies
        lam_soft ONLY; lam_hard is never gated."""
        p = self.grid.params
        if p.gate_enabled:
            mult = 1.0 if self.gate(pos_world, goal_world).active else 0.0
        else:
            self.last_gate = None
            mult = 1.0
        return p.lam_soft * mult, p.lam_hard

    def gate_batch(self, pos_world: np.ndarray, goal_world: np.ndarray):
        """Vectorized gate for [N] agents (Swarm). Returns active[N] bool."""
        gp = self.grid.params.gate
        mnx, mny, mxx, mxy = self.grid.bounds
        ny, nx = self.grid.risk_raw.shape
        pos = np.empty((pos_world.shape[0], 2), np.float64)
        goal = np.empty_like(pos)
        pos[:, 0] = (pos_world[:, 1] - mny) / (mxy - mny) * (ny - 1)  # row from y
        pos[:, 1] = (pos_world[:, 0] - mnx) / (mxx - mnx) * (nx - 1)  # col from x
        goal[:, 0] = (goal_world[:, 1] - mny) / (mxy - mny) * (ny - 1)
        goal[:, 1] = (goal_world[:, 0] - mnx) / (mxx - mnx) * (nx - 1)
        active, _, _, _ = witness_gate_batch(
            self.grid.risk,
            self._gate_hard,
            self._gate_clear_m,
            pos,
            goal,
            horizon_cells=self._horizon_cells,
            hard_margin_m=self._hard_margin_m,
            primitive_count=gp.primitive_count,
            improvement_margin=gp.improvement_margin,
            material_trigger=gp.material_trigger,
            progress_slack_cells=gp.progress_slack_cells,
        )
        return active

    # -- what the rollouts consume ------------------------------------------
    @property
    def field(self) -> MaterialField:
        return self.grid.field()

    @property
    def params(self) -> MaterialParams:
        return self.grid.params
