"""A seeded percentile bootstrap, and paired differences that keep the pairs together.

The bootstrap is the one the downstream A/B harness uses, step for step, so an interval computed
here and one computed there from the same values and the same seed are the same interval:

- the statistic is the plain mean, ``sum(xs) / len(xs)``;
- ``n_boot`` (5000) resamples, each ``len(xs)`` draws of ``rng.randrange(len(xs))`` from a
  standard-library :class:`random.Random`;
- the interval is the percentile one, the sorted resample means at ``int(alpha / 2 * n_boot)``
  and ``min(n_boot - 1, int((1 - alpha / 2) * n_boot))``;
- fewer than two values give the degenerate interval ``(mean, mean)``.

One generator is meant to be passed through every call of a table in order (one per arm's mean,
then one per arm's paired difference), as that harness does; the order is part of the result.

A paired difference is ``arm - reference`` per unit (a seed, a scene, a seed and scene), so it
resamples units, never the two arms' values separately: two arms that differ by a constant on
every seed have an interval of width zero however noisy the seeds are.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence

N_BOOT = 5000
ALPHA = 0.05


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def as_rng(rng: random.Random | int | None, seed: int = 1) -> random.Random:
    """``rng`` itself, a generator seeded with it (an int), or one seeded with ``seed``."""
    if isinstance(rng, random.Random):
        return rng
    return random.Random(seed if rng is None else int(rng))


def bootstrap_ci(
    xs: Sequence[float],
    n_boot: int = N_BOOT,
    rng: random.Random | int | None = None,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """The percentile bootstrap interval of the mean (see the module docstring)."""
    xs = list(xs)
    if len(xs) < 2:
        m = mean(xs)
        return m, m
    rng = as_rng(rng)
    k = len(xs)
    means = sorted(mean([xs[rng.randrange(k)] for _ in range(k)]) for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def pairs(reference, arm) -> list[tuple[float, float]]:
    """``(reference, arm)`` value pairs, one per unit both have. Mappings pair by key (in the
    reference's order); sequences pair by position. A missing value (``None`` or NaN) on either
    side drops that unit."""
    if isinstance(reference, Mapping) and isinstance(arm, Mapping):
        units = [(reference[k], arm[k]) for k in reference if k in arm]
    elif isinstance(reference, Mapping) or isinstance(arm, Mapping):
        raise TypeError("pair two mappings (by unit) or two sequences (by position)")
    else:
        units = list(zip(reference, arm, strict=False))
    return [(float(a), float(b)) for a, b in units if _present(a) and _present(b)]


def differences(reference, arm) -> list[float]:
    """``arm - reference`` per paired unit."""
    return [b - a for a, b in pairs(reference, arm)]


def summarize(
    xs: Iterable[float], n_boot: int = N_BOOT, rng: random.Random | int | None = None
) -> dict:
    """``{"mean", "ci95": [lo, hi], "n"}`` of the present values."""
    vs = [float(v) for v in xs if _present(v)]
    lo, hi = bootstrap_ci(vs, n_boot, rng)
    return {"mean": mean(vs), "ci95": [lo, hi], "n": len(vs)}


def paired_diff(
    reference, arm, n_boot: int = N_BOOT, rng: random.Random | int | None = None
) -> dict | None:
    """``{"mean", "ci95", "n"}`` of ``arm - reference`` over the paired units; ``None`` when no
    unit pairs."""
    d = differences(reference, arm)
    if not d:
        return None
    lo, hi = bootstrap_ci(d, n_boot, rng)
    return {"mean": mean(d), "ci95": [lo, hi], "n": len(d)}


def excludes_zero(ci: Sequence[float]) -> bool:
    """The interval lies wholly on one side of 0 (the harness marks it ``*``)."""
    lo, hi = ci
    return not (lo <= 0.0 <= hi)


def negate(entry: dict) -> dict:
    """A difference summary for ``reference - arm`` from one for ``arm - reference``."""
    lo, hi = entry["ci95"]
    return dict(entry, mean=-entry["mean"], ci95=[-hi, -lo])
