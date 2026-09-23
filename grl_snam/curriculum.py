"""Scorecard-driven curriculum for coef_train — bias training-start sampling toward the
regions the current policy handles WORST (hard-example / terrain-risk mining).

The trainer otherwise draws starts uniformly over the free space, so easy open corridors
and hard risky/tight regions get equal weight and the gradient is dominated by the easy
majority. :class:`RegionCurriculum` partitions the free space into a coarse grid of bins,
tracks a per-bin difficulty EMA fed straight from the training batch's own outcomes (no
extra eval pass), and reweights start sampling toward the hard bins — with a uniform floor
so no bin starves, a cap so one bin can't dominate, and a uniform cold start so early noisy
scores can't lock the curriculum. Default-off in the trainer, so a plain run is unchanged.

Difficulty is whatever the caller attributes back per agent (:meth:`update`): coef_train
uses goal shortfall + terrain-risk exposure, so the curriculum concentrates on regions that
are both hard to reach AND risky — exactly the scorecard axes a campaign ranks on.
"""

from __future__ import annotations

import numpy as np


class RegionCurriculum:
    """Adaptive start-region sampler. Build it over a scene's free cells; call
    :meth:`sample_starts` for each batch and :meth:`update` with the per-agent difficulty
    that batch produced."""

    def __init__(
        self,
        free_rows: np.ndarray,
        free_cols: np.ndarray,
        grid_shape,
        bounds,
        center,
        scale: float,
        *,
        bins: int = 6,
        eps: float = 0.15,
        momentum: float = 0.85,
        cap: float = 6.0,
    ):
        if not (0.0 <= eps <= 1.0):
            raise ValueError(f"eps must be in [0,1], got {eps}")
        self.ny, self.nx = int(grid_shape[0]), int(grid_shape[1])
        self.mnx, self.mny, self.mxx, self.mxy = (float(b) for b in bounds)
        self.cx, self.cy = float(center[0]), float(center[1])
        self.S = float(scale)
        self.bins = int(bins)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.cap = float(cap)

        fr = np.asarray(free_rows, np.int64)
        fc = np.asarray(free_cols, np.int64)
        if len(fr) == 0:
            raise ValueError("RegionCurriculum needs at least one free cell")
        # bin each free cell by its WORLD position (the frame starts are drawn in)
        wx = self.mnx + fc / (self.nx - 1) * (self.mxx - self.mnx)
        wy = self.mny + fr / (self.ny - 1) * (self.mxy - self.mny)
        gx = np.clip(
            ((wx - self.mnx) / (self.mxx - self.mnx) * self.bins).astype(int), 0, self.bins - 1
        )
        gy = np.clip(
            ((wy - self.mny) / (self.mxy - self.mny) * self.bins).astype(int), 0, self.bins - 1
        )
        bin_of_cell = gy * self.bins + gx
        # keep only non-empty bins; map to a compact 0..B-1 index space
        self._cells = {}  # compact bin idx -> (rows, cols) arrays of that bin's free cells
        self._bin_ids = []  # compact idx -> original bin id (for stability/debug)
        for b in np.unique(bin_of_cell):
            m = bin_of_cell == b
            self._cells[len(self._bin_ids)] = (fr[m], fc[m])
            self._bin_ids.append(int(b))
        self.B = len(self._bin_ids)
        # difficulty EMA (start neutral) + weights (start UNIFORM = cold start)
        self.diff = np.ones(self.B, dtype=np.float64)
        self.weights = np.full(self.B, 1.0 / self.B, dtype=np.float64)
        self._seen = np.zeros(self.B, dtype=bool)

    def _cell_to_norm(self, r, c):
        wx = self.mnx + c / (self.nx - 1) * (self.mxx - self.mnx)
        wy = self.mny + r / (self.ny - 1) * (self.mxy - self.mny)
        return (wx - self.cx) * self.S, (wy - self.cy) * self.S

    def sample_starts(self, n: int, rng: np.random.Generator):
        """Return ``(starts[n,2] float32 normalized, bin_ids[n] int)`` — each start a
        random free cell inside a bin drawn by the current difficulty weights."""
        chosen = rng.choice(self.B, size=n, p=self.weights)
        out = np.empty((n, 2), np.float32)
        for i in range(n):
            rows, cols = self._cells[int(chosen[i])]
            j = int(rng.integers(0, len(rows)))
            out[i, 0], out[i, 1] = self._cell_to_norm(rows[j], cols[j])
        return out, chosen

    def update(self, bin_ids: np.ndarray, difficulty: np.ndarray) -> None:
        """Fold one batch's per-agent ``difficulty`` (>=0) back into the bins the agents
        STARTED in, then recompute the sampling weights (uniform floor + cap)."""
        bin_ids = np.asarray(bin_ids, np.int64)
        difficulty = np.asarray(difficulty, np.float64)
        for b in np.unique(bin_ids):
            m = bin_ids == b
            batch_mean = float(difficulty[m].mean())
            if self._seen[b]:
                self.diff[b] = self.momentum * self.diff[b] + (1.0 - self.momentum) * batch_mean
            else:
                self.diff[b] = batch_mean  # first real signal replaces the neutral prior
                self._seen[b] = True
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        d = np.clip(self.diff, 0.0, None)
        s = d.sum()
        score = d / s if s > 0 else np.full(self.B, 1.0 / self.B)
        uniform = 1.0 / self.B
        w = (1.0 - self.eps) * score + self.eps * uniform  # already sums to 1
        # Cap each bin's FINAL probability at cap x its uniform share, redistributing the
        # excess to the uncapped bins (water-filling) so the cap actually binds — a plain
        # clip+renormalize would re-inflate the capped bin above the ceiling.
        ceil = self.cap * uniform
        if ceil < 1.0:
            for _ in range(self.B):
                over = w > ceil + 1e-12
                if not over.any():
                    break
                excess = float((w[over] - ceil).sum())
                w[over] = ceil
                under = ~over
                us = float(w[under].sum())
                if us <= 0.0:
                    break
                w[under] += excess * (w[under] / us)
        self.weights = w / w.sum()


def build_city_curriculum(grid: int, *, bins: int = 6, eps: float = 0.15, **kw) -> RegionCurriculum:
    """A :class:`RegionCurriculum` over the SAME largest-free-component the coef_train
    ``_scene(grid)`` trainer samples from (``shrunk(STORIES['city'], n=grid)``), so its
    bins align with the training scene's free space."""
    from . import planner
    from .fog_stories import STORIES, shrunk

    story = shrunk(STORIES["city"], n=grid, max_steps=100)
    truth = story.truth_grid()
    meta = story.meta()
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    return RegionCurriculum(
        rows,
        cols,
        truth.shape,
        story.bounds,
        meta["center"],
        meta["scale"],
        bins=bins,
        eps=eps,
        **kw,
    )
