"""Incomplete information: a believed map that differs from ground truth.

Two grids, one sensor. ``truth`` is what is actually there — the simulator
senses (and collides) against it. :class:`BeliefGrid` is what the *agent*
thinks — the planner and the SDF are built from belief, never from truth. The
gap between them is the whole point: a stale map keeps a ghost obstacle until
the agent observes the gap; a new blocker does not exist for the planner until
a ray actually hits it.

Belief is **log-odds occupancy** (standard occupancy-grid mapping), so "fuzzy"
is literal — every cell carries ``p(occupied)`` in [0, 1], not a bit. Unknown
space is a declared policy, not an accident: ``optimistic`` plans through it
(free-space assumption — what most real robots do; replan on discovery),
``pessimistic`` treats it as wall.

Moving obstacles (units, environmental blockers) deliberately live in a
separate :class:`DynamicLayer` with time decay, composited at query time —
baking them into the static belief would accumulate stale ghosts forever.

Everything here is numpy on the same (ny, nx) grid/bounds convention as
``sdf_nav.build_sdf``, and rebuilding the SDF from belief is cheap enough to
do naively (the exact EDT at 512^2 is milliseconds), so there is no
incremental-update cleverness to distrust.
"""

from __future__ import annotations

import numpy as np

# Log-odds increments per observation. Asymmetric on purpose: one hit is strong
# evidence (sensors rarely hallucinate walls), one miss is weaker (a ray can
# skim a corner). Clamped so no cell becomes unrecoverable.
L_OCC = 2.2
L_FREE = -1.4
L_CLAMP = 8.0


class BeliefGrid:
    """Log-odds occupancy belief over the same raster as the truth grid."""

    def __init__(self, shape, bounds, *, prior_logodds: float = 0.0):
        self.ny, self.nx = int(shape[0]), int(shape[1])
        self.mnx, self.mny, self.mxx, self.mxy = (float(b) for b in bounds)
        self.logodds = np.full((self.ny, self.nx), float(prior_logodds), np.float32)
        # Bumped whenever any cell crosses the 0.5 boundary — the caller's cue
        # that the planning surface changed and the SDF is worth rebuilding.
        self.version = 0
        # Cells the LAST sweep could actually see (free cells along a ray plus
        # the cell each ray terminated on). This is the agent's instantaneous
        # field of view, as distinct from `logodds`, which is its memory — and
        # the difference between the two is exactly what a viewer needs to see
        # to understand fog of war. Empty until the first sense().
        self.last_visible = np.zeros((self.ny, self.nx), bool)
        # Every cell ever seen. Remembered-but-not-currently-visible is the
        # middle tier of the classic three-tier fog.
        self.ever_seen = np.zeros((self.ny, self.nx), bool)

    # ── coordinates ─────────────────────────────────────────────────────────
    def world_to_cell(self, x, y):
        cx = (x - self.mnx) / (self.mxx - self.mnx) * (self.nx - 1)
        cy = (y - self.mny) / (self.mxy - self.mny) * (self.ny - 1)
        return int(round(cy)), int(round(cx))  # (row, col)

    def in_bounds(self, r, c):
        return 0 <= r < self.ny and 0 <= c < self.nx

    # ── probabilities ───────────────────────────────────────────────────────
    def p(self) -> np.ndarray:
        """Per-cell ``p(occupied)`` in [0, 1]."""
        return 1.0 / (1.0 + np.exp(-self.logodds))

    def known(self, band: float = 0.15) -> np.ndarray:
        """Cells whose belief has moved out of the ``0.5 +- band`` fuzz."""
        return np.abs(self.p() - 0.5) > band

    def confidence_at(self, x, y) -> float:
        """``|2p - 1|`` at a world point — 0 at fully unknown, 1 at certain.
        The hook for scaling the IPC barrier: multiply alpha by
        ``0.5 + 0.5 * (1 - confidence)`` to give uncertain walls a wide berth,
        or by ``p`` itself to cut close only past confirmed-free space."""
        r, c = self.world_to_cell(x, y)
        if not self.in_bounds(r, c):
            return 1.0
        pv = 1.0 / (1.0 + np.exp(-float(self.logodds[r, c])))
        return abs(2.0 * pv - 1.0)

    # ── the sensor ──────────────────────────────────────────────────────────
    def sense(
        self,
        truth_occ: np.ndarray,
        pos_world,
        *,
        range_m: float,
        n_rays: int = 180,
        fov_rad: float = 2.0 * np.pi,
        heading_rad: float = 0.0,
    ) -> int:
        """Ray-cast the *truth* grid from ``pos_world`` and update belief.

        Standard occupancy-grid sensor model with occlusion: each ray marks
        cells free up to the first truth hit, marks the hit cell occupied, and
        sees nothing beyond it — "you cannot see behind that building" falls
        out of the walk, not out of special-casing.

        Returns the number of cells whose believed state (the p>0.5 bit)
        flipped, and bumps :attr:`version` if any did.
        """
        assert truth_occ.shape == (self.ny, self.nx), "truth/belief raster mismatch"
        cell_w = (self.mxx - self.mnx) / (self.nx - 1)
        cell_h = (self.mxy - self.mny) / (self.ny - 1)

        r0, c0 = self.world_to_cell(pos_world[0], pos_world[1])
        if not self.in_bounds(r0, c0):
            self.last_visible = np.zeros((self.ny, self.nx), bool)
            return 0

        before = self.logodds > 0.0

        # All rays marched in lockstep as [n_rays, n_steps] index arrays — the
        # per-cell Python loop was ~100x slower and dominated the sim step.
        angles = heading_rad + (np.arange(n_rays) / max(n_rays, 1) - 0.5) * fov_rad
        dxy = np.stack([np.cos(angles), np.sin(angles)], -1)  # world dirs
        sr = dxy[:, 1] * (cell_w / cell_h)  # keep world-isotropic rays on an
        sc = dxy[:, 0]  # anisotropic raster
        norm = np.maximum(np.maximum(np.abs(sr), np.abs(sc)), 1e-9)
        sr, sc = sr / norm, sc / norm

        n_steps = int(range_m / cell_w) + 1
        steps = np.arange(n_steps, dtype=np.float64)
        rows = np.rint(r0 + sr[:, None] * steps[None, :]).astype(np.intp)
        cols = np.rint(c0 + sc[:, None] * steps[None, :]).astype(np.intp)

        # True RANGE limit, in world units. The DDA above normalises each ray by
        # its dominant component, so a step is one cell along that axis and the
        # reachable set is a SQUARE: a diagonal ray travels sqrt(2) times
        # farther than an axis-aligned one. Drawing the sensor's range as a ring
        # made that visible -- the lit region overflowed the circle. Clip by
        # actual distance so `range_m` means the same thing in every direction.
        dist = np.hypot(
            (rows - r0) * cell_h,
            (cols - c0) * cell_w,
        )
        inside = (rows >= 0) & (rows < self.ny) & (cols >= 0) & (cols < self.nx) & (dist <= range_m)
        rows_c = np.clip(rows, 0, self.ny - 1)
        cols_c = np.clip(cols, 0, self.nx - 1)
        hit = truth_occ[rows_c, cols_c] & inside  # [R, S]

        # First hit (or first out-of-bounds) truncates each ray; the hit cell
        # itself is marked occupied and nothing beyond it is seen — occlusion.
        stop = hit | ~inside
        any_stop = stop.any(axis=1)
        first_stop = np.where(any_stop, stop.argmax(axis=1), n_steps)
        idx = np.broadcast_to(steps, hit.shape)
        free_mask = inside & (idx < first_stop[:, None])
        occ_mask = hit & (idx == first_stop[:, None])

        # The swept field of view: every cell a ray reached, including the one
        # it stopped on. Free of extra cost -- the masks already exist.
        visible = np.zeros((self.ny, self.nx), bool)
        visible[rows_c[free_mask], cols_c[free_mask]] = True
        visible[rows_c[occ_mask], cols_c[occ_mask]] = True
        self.last_visible = visible
        self.ever_seen |= visible

        # Accumulate per-visit evidence exactly as the scalar loop did (a cell
        # crossed by several rays gets several increments), then clamp.
        delta = np.zeros_like(self.logodds)
        np.add.at(delta, (rows_c[free_mask], cols_c[free_mask]), L_FREE)
        np.add.at(delta, (rows_c[occ_mask], cols_c[occ_mask]), L_OCC)
        np.clip(self.logodds + delta, -L_CLAMP, L_CLAMP, out=self.logodds)

        flips = int(np.count_nonzero((self.logodds > 0.0) != before))
        if flips:
            self.version += 1
        return flips

    # ── the planning surface ────────────────────────────────────────────────
    def to_occupancy(
        self, *, unknown: str = "optimistic", p_thresh: float = 0.5, band: float = 0.15
    ) -> np.ndarray:
        """Boolean occupancy for ``sdf_nav.build_sdf``, under a declared
        unknown-space policy.

        ``optimistic``  — unknown is free: drive in, replan on discovery.
        ``pessimistic`` — unknown is wall: only confirmed-free space is
        traversable. The toggle is itself a good demo.
        """
        pr = self.p()
        if unknown == "optimistic":
            return pr > max(p_thresh, 0.5 + band)
        if unknown == "pessimistic":
            # occupied unless confidently free
            return ~(pr < min(1.0 - p_thresh, 0.5 - band))
        raise ValueError(f"unknown policy {unknown!r}")


class DynamicLayer:
    """Short-lived obstacles (units, transient blockers) with time decay.

    Kept OUT of the static belief on purpose: a moving unit baked into the
    log-odds grid leaves a permanent smear of stale ghosts along its path.
    Here each mark carries a timestamp and simply expires; the layer is
    composited into the planning surface at query time.
    """

    def __init__(self, shape, *, ttl_s: float = 4.0):
        self.ny, self.nx = int(shape[0]), int(shape[1])
        self.ttl_s = float(ttl_s)
        self._stamp = np.full((self.ny, self.nx), -np.inf, np.float64)

    def mark(self, r: int, c: int, t_now: float, *, radius_cells: int = 1):
        r0, r1 = max(0, r - radius_cells), min(self.ny, r + radius_cells + 1)
        c0, c1 = max(0, c - radius_cells), min(self.nx, c + radius_cells + 1)
        self._stamp[r0:r1, c0:c1] = t_now

    def occupancy(self, t_now: float) -> np.ndarray:
        """Cells marked within the last ``ttl_s`` seconds (of *world* time —
        pass the world clock's ``t()``, not wall time)."""
        return (t_now - self._stamp) <= self.ttl_s


def composite_occupancy(
    belief: BeliefGrid, dyn: DynamicLayer | None, t_now: float = 0.0, **to_occ_kw
) -> np.ndarray:
    """The final planning raster: static belief under its unknown-space policy,
    OR-ed with whatever the dynamic layer currently remembers."""
    occ = belief.to_occupancy(**to_occ_kw)
    if dyn is not None:
        occ = occ | dyn.occupancy(t_now)
    return occ
