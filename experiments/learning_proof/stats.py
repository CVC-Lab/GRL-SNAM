"""Paired bootstrap for the learning check's arm comparisons.

An arm with several training seeds has a value per (seed, unit), a unit being an evaluation
episode (a start/goal pair, or one agent of a scene). A comparison ``a - b`` pairs units: both arms
drove the same ones. Its CI resamples the training seeds and the units, both with replacement (a
two-level percentile bootstrap), so it carries training-seed variance as well as evaluation noise.
"""

from __future__ import annotations

import numpy as np


def _as2d(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x[None, :] if x.ndim == 1 else x


def paired(a, b, valid=None, *, n_boot=10000, seed=0) -> dict:
    """``a - b`` paired by unit. ``a`` and ``b`` are ``[S, P]`` (seeds x units) or ``[P]``; a
    one-seed side is broadcast. ``valid`` (same shape as the difference) masks units a metric is
    undefined for (time to goal where either arm did not arrive). Returns the mean difference,
    its 95% CI, the per-seed means and the counts."""
    a, b = _as2d(a), _as2d(b)
    d = a - b
    if valid is None:
        m = np.ones(d.shape, bool)
    else:
        v = np.asarray(valid, bool)
        m = np.broadcast_to(v[None, :] if v.ndim == 1 else v, d.shape)
    S, P = d.shape
    per_seed = [float(d[s][m[s]].mean()) if m[s].any() else None for s in range(S)]
    rng = np.random.default_rng(seed)
    dz = np.where(m, d, 0.0)
    mz = m.astype(np.float64)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, S, S)
        pi = rng.integers(0, P, P)
        num = dz[si][:, pi].sum()
        den = mz[si][:, pi].sum()
        boots[i] = num / den if den > 0 else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {
        "mean": float(dz.sum() / mz.sum()) if mz.sum() > 0 else None,
        "lo": float(lo),
        "hi": float(hi),
        "per_seed": per_seed,
        "n_units": int(P),
        "n_valid": int(m.sum()),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def level(a, valid=None, *, n_boot=10000, seed=0) -> dict:
    """An arm's own mean and 95% CI (the same two-level resampling as :func:`paired`)."""
    a = _as2d(a)
    return paired(a, np.zeros_like(a), valid, n_boot=n_boot, seed=seed)
