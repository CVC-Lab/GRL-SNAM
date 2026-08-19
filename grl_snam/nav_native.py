"""C++ fast path for the navigation hot loop, via libcvc's ``cvc::nav`` kernels
(exposed as ``pycvc.nav_*``).

The pure-Python reference in :mod:`grl_snam.planner` and :mod:`sdf_nav` is the
source of truth; these adapters are a **bit-identical** drop-in that moves the
four functions that dominate ``Squad.step`` — the exact Euclidean distance
transform, 8-connected A*, Bresenham line-of-sight and string-pull — into
compiled code. See GRL-SNAM/docs/PERFORMANCE.md: measured 43-77x per kernel,
which is what takes the tick from ~0.7 Hz toward the 30 Hz / 100-1000 agent
target.

Contract: every adapter here returns the SAME Python type as the function it
replaces (a bool ndarray, an ``(r, c)`` tuple or ``None``, a list of ``(r, c)``
tuples), so callers can dispatch with a one-line early return and nothing
downstream can tell which path ran. The libcvc parity test
(tests/test_nav_cpp_parity.py) asserts that byte-for-byte on every release.

If ``pycvc`` is missing or too old to carry ``nav_*`` (every currently published
build), :data:`AVAILABLE` is False and callers stay on the Python path — so
importing this module is always safe.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import pycvc as _pycvc

    AVAILABLE = hasattr(_pycvc, "nav_astar")
except Exception:  # pragma: no cover - pycvc is an optional accelerator
    _pycvc = None
    AVAILABLE = False


def enabled() -> bool:
    """True when the C++ path should be used. Off automatically when pycvc lacks
    the kernels; can be forced off with ``GRL_SNAM_NAV_BACKEND=python`` (the
    parity test uses this to obtain the reference)."""
    if not AVAILABLE:
        return False
    return os.environ.get("GRL_SNAM_NAV_BACKEND", "native").lower() != "python"


# ── planner.py kernels ──────────────────────────────────────────────────────


def inflate(occ: np.ndarray, cells: int) -> np.ndarray:
    return _pycvc.nav_inflate(occ, int(cells)).astype(bool)


def line_of_sight(occ: np.ndarray, a, b) -> bool:
    return bool(_pycvc.nav_line_of_sight(occ, int(a[0]), int(a[1]), int(b[0]), int(b[1])))


def nearest_free(occ: np.ndarray, r: int, c: int, max_radius: int = 12):
    a = _pycvc.nav_nearest_free(occ, int(r), int(c), int(max_radius))
    return None if a.size == 0 else (int(a[0]), int(a[1]))


def astar(occ: np.ndarray, start, goal, cost: np.ndarray | None = None):
    c = None if cost is None else np.ascontiguousarray(cost, np.float64)
    a = _pycvc.nav_astar(occ, int(start[0]), int(start[1]), int(goal[0]), int(goal[1]), c)
    return None if a.shape[0] == 0 else [(int(r), int(cc)) for r, cc in a]


def simplify(occ: np.ndarray, path):
    if not path or len(path) < 3:
        return path
    a = _pycvc.nav_simplify(occ, np.asarray(path, np.int32))
    return [(int(r), int(c)) for r, c in a]


# ── sdf_nav.py kernels ──────────────────────────────────────────────────────


def edt2(mask: np.ndarray) -> np.ndarray:
    return _pycvc.nav_edt2_squared(mask)


def build_sdf(occ: np.ndarray, bounds, scale: float):
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    s = _pycvc.nav_build_sdf(occ, mnx, mny, mxx, mxy, float(scale))  # (3, H, W) float32
    return s[0], s[1], s[2]
