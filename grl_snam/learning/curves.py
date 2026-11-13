"""Marks on a held-out curve: where it was best, where it bent, where it went flat and whether it
turned back up.

Each mark is read on the curve after a running median (``w`` = 5 points), so a single noisy
evaluation moves none of them. ``mode`` says which way is better: ``"min"`` for a loss,
``"max"`` for a fitness. The definitions are exact so another implementation can reproduce them
(``tests/fixtures/learning_marks_golden.json`` pins this one's output):

- **smooth**: point i is the median of the finite values at indices ``i-h .. i+h`` (``h = w // 2``),
  the window cut at the ends. Non-finite points are dropped from the curve before anything else.
- **IQR**: the 75th minus the 25th percentile of the smoothed curve (linear interpolation, numpy's
  default): the curve's own scale.
- **noise** (sigma): from the second differences of the points, ``d_i = y[i+1] - 2 y[i] +
  y[i-1]`` (by index), ``1.4826 * median(|d - median(d)|) / sqrt(6)``: the robust standard
  deviation of white noise, which a locally straight trend does not inflate. The IQR of a curve
  that never moved IS its noise, so a threshold in IQRs alone fires on noise; every mark below
  that compares a change to a threshold also clears a noise floor.
- **best**: the first index of the smoothed minimum (``max``: maximum).
- **elbow** (Kneedle, Satopaa et al. 2011, offline): x and the smoothed y are scaled to [0, 1]
  over the whole curve (y flipped for ``max``, so the curve falls); the elbow is the point furthest
  below the chord, the first maximum of ``1 - x_n - y_n``. It needs at least 3 points, a maximum
  of at least ``knee_min``, and a fall from the start to best of more than ``z * sigma``; a
  straight or concave curve, or one that only fell by noise, has no elbow.
- **plateau**: window k is the ``window`` points from index k (stride 1; ``window = 0``, the
  default, is ``max(5, n // 6)`` so the test does not depend on how often a trainer evaluates). It
  is flat when ``|Theil-Sen slope| * (x[k+window-1] - x[k]) < max(eps * IQR, k_noise * sigma)``:
  the curve's change across the window is small next to its whole range, or no larger than its
  noise. The plateau is the start of the last run of consecutive flat windows, when that run
  reaches the last window and is at least ``patience`` long: the curve has been flat since there.
  On a head's mean or spread this is its mini elbow, where the head settled.
- **divergence**: the start of the last run of points worse than ``best + max(delta * IQR, z *
  sigma)`` (``max``: below ``best - ...``) that reaches the end, when it starts after best and is at
  least ``min_points`` long: the first x after best where the curve stays above that line.

A zero threshold (a constant, noise-free curve) makes any change count: a window with no change is
flat and one with any change is not.

The defaults were set on synthetic curves of 20, 60 and 200 points, 100 noise draws each. Pure
noise got no elbow and no divergence in any draw. A fall that turns back up (overfitting) got its
divergence in every draw. A decay to a floor with noise at 1% of its fall got its plateau in at
least 99% of draws; a straight descent never did.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MarkParams:
    """The marks' parameters (see the module docstring)."""

    w: int = 5
    #: 0: ``max(5, n // 6)``.
    window: int = 0
    patience: int = 3
    eps: float = 0.2
    k_noise: float = 1.0
    delta: float = 1.0
    z: float = 4.0
    knee_min: float = 0.1
    min_points: int = 3

    def window_for(self, n: int) -> int:
        return self.window if self.window > 0 else max(5, n // 6)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = MarkParams()


@dataclass
class Mark:
    """A point of the smoothed curve: its index (into the finite points), x and smoothed y."""

    index: int
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"index": self.index, "x": self.x, "y": self.y}


@dataclass
class Marks:
    mode: str
    x: np.ndarray
    smooth: np.ndarray
    iqr: float
    noise: float
    best: Mark | None
    elbow: Mark | None
    plateau: Mark | None
    divergence: Mark | None

    def to_dict(self) -> dict:
        def m(v):
            return None if v is None else v.to_dict()

        return {
            "mode": self.mode,
            "n": int(len(self.x)),
            "iqr": self.iqr,
            "noise": self.noise,
            "best": m(self.best),
            "elbow": m(self.elbow),
            "plateau": m(self.plateau),
            "divergence": m(self.divergence),
        }


def median_smooth(y, w: int = 5) -> np.ndarray:
    """The running median of ``y`` over ``w`` points (see the module docstring)."""
    y = np.asarray(y, dtype=float)
    h = int(w) // 2
    out = np.empty(len(y))
    for i in range(len(y)):
        seg = y[max(0, i - h) : i + h + 1]
        seg = seg[np.isfinite(seg)]
        out[i] = np.median(seg) if seg.size else math.nan
    return out


def theil_sen(x, y) -> float:
    """The median of the pairwise slopes (pairs with equal x are left out); 0 with none."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return 0.0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def iqr(y) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return math.nan
    q75, q25 = np.percentile(y, [75, 25])
    return float(q75 - q25)


def noise(y) -> float:
    """The robust standard deviation of the noise, from the second differences (see the module
    docstring); 0 with fewer than 3 points."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0
    d = y[2:] - 2.0 * y[1:-1] + y[:-2]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(6.0))


def best_index(s: np.ndarray, mode: str) -> int:
    return int(np.argmax(s) if mode == "max" else np.argmin(s))


def elbow_index(
    x: np.ndarray, s: np.ndarray, best: int, mode: str, knee_min: float, floor: float = 0.0
) -> int | None:
    """Kneedle on the whole curve: the index furthest below the chord, or None (also when the
    curve fell from its start to ``best`` by no more than ``floor``)."""
    if len(s) < 3 or not abs(s[0] - s[best]) > floor:
        return None
    xs = x
    ys = s if mode != "max" else -s
    xr = xs[-1] - xs[0]
    yr = ys.max() - ys.min()
    if xr <= 0 or yr <= 0:
        return None
    xn = (xs - xs[0]) / xr
    yn = (ys - ys.min()) / yr
    d = 1.0 - xn - yn
    k = int(np.argmax(d))
    return k if d[k] >= knee_min else None


def plateau_index(
    x: np.ndarray, s: np.ndarray, threshold: float, window: int, patience: int
) -> int | None:
    """The start of the final run of flat windows (see the module docstring), or None."""
    n_win = len(s) - window + 1
    if n_win < 1:
        return None
    flat = []
    for k in range(n_win):
        change = abs(theil_sen(x[k : k + window], s[k : k + window]) * (x[k + window - 1] - x[k]))
        flat.append(change < threshold if threshold > 0 else change == 0)
    if not flat[-1]:
        return None
    k = n_win - 1
    while k > 0 and flat[k - 1]:
        k -= 1
    return k if n_win - k >= patience else None


def divergence_index(
    s: np.ndarray, b: int, margin: float, mode: str, min_points: int
) -> int | None:
    """The start of the final run of points past ``best +- margin``, or None."""
    worse = s < s[b] - margin if mode == "max" else s > s[b] + margin
    if not worse[-1]:
        return None
    j = len(s) - 1
    while j > 0 and worse[j - 1]:
        j -= 1
    return j if j > b and len(s) - j >= min_points else None


def _floor(v: float) -> float:
    return v if math.isfinite(v) else 0.0


def marks(x, y, mode: str = "min", params: MarkParams = DEFAULT) -> Marks:
    """Every mark of the curve ``y(x)`` (see the module docstring)."""
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', not {mode!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return Marks(mode, x, y, math.nan, math.nan, None, None, None, None)
    s = median_smooth(y, params.w)
    scale = _floor(iqr(s))
    sigma = _floor(noise(y))

    def mark(i):
        return None if i is None else Mark(int(i), float(x[i]), float(s[i]))

    b = best_index(s, mode)
    flat_at = max(params.eps * scale, params.k_noise * sigma)
    margin = max(params.delta * scale, params.z * sigma)
    return Marks(
        mode=mode,
        x=x,
        smooth=s,
        iqr=scale,
        noise=sigma,
        best=mark(b),
        elbow=mark(elbow_index(x, s, b, mode, params.knee_min, params.z * sigma)),
        plateau=mark(plateau_index(x, s, flat_at, params.window_for(len(s)), params.patience)),
        divergence=mark(divergence_index(s, b, margin, mode, params.min_points)),
    )


def series_marks(series, mode: str = "min", params: MarkParams = DEFAULT) -> Marks:
    """:func:`marks` of a :class:`grl_snam.learning.history.Series`."""
    return marks(series.x, series.y, mode, params)
