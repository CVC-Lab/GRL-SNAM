"""The C++ (libcvc ``cvc::nav`` / ``pycvc.nav_*``) navigation kernels must be
**bit-identical** to the pure-Python reference in :mod:`grl_snam.planner` and
:mod:`sdf_nav`. A fast digital twin that moves differently is worthless
(GRL-SNAM/docs/PERFORMANCE.md, "Fidelity guardrails"), so this is exact
equality, not closeness.

Skipped automatically when pycvc is absent or predates the nav kernels (every
currently published build), so it is safe to keep in CI while the libcvc side
lands.
"""

import numpy as np
import pytest

pycvc = pytest.importorskip("pycvc")
if not hasattr(pycvc, "nav_astar"):
    pytest.skip("pycvc build has no cvc::nav kernels", allow_module_level=True)

import sdf_nav  # noqa: E402
from grl_snam import nav_native, planner  # noqa: E402

_SIZES = [3, 5, 8, 13, 16, 24, 33, 48, 64, 96, 128]


def _cases(seed, n=150):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        rows = int(rng.choice(_SIZES))
        cols = int(rng.choice(_SIZES))
        occ = rng.random((rows, cols)) < float(rng.uniform(0.08, 0.55))
        occ.reshape(-1)[0] = True  # at least one building
        occ.reshape(-1)[-1] = False  # and one free cell -> finite SDF
        yield rng, occ, rows, cols


def test_edt_and_sdf_bit_identical(monkeypatch):
    monkeypatch.setenv("GRL_SNAM_NAV_BACKEND", "python")
    for rng, occ, rows, cols in _cases(0xED7):
        assert np.array_equal(nav_native.edt2(occ), sdf_nav._edt2(occ))
        bounds = (0.0, 0.0, float(cols - 1), float(rows - 1))
        scale = 0.1
        pr, nxr, nyr = sdf_nav.build_sdf(occ, bounds, scale)
        p, nx, ny = nav_native.build_sdf(occ, bounds, scale)
        assert np.array_equal(p, pr), "phi"
        assert np.array_equal(nx, nxr), "normal_x"
        assert np.array_equal(ny, nyr), "normal_y"


def test_planner_kernels_bit_identical(monkeypatch):
    monkeypatch.setenv("GRL_SNAM_NAV_BACKEND", "python")
    for rng, occ, rows, cols in _cases(0xA57A2):
        cells = int(rng.integers(0, 3))
        inflated = planner.inflate(occ, cells)  # python (backend forced off)
        assert np.array_equal(nav_native.inflate(occ, cells), inflated), "inflate"

        sr, sc = int(rng.integers(0, rows)), int(rng.integers(0, cols))
        gr, gc = int(rng.integers(0, rows)), int(rng.integers(0, cols))
        assert nav_native.line_of_sight(occ, (sr, sc), (gr, gc)) == planner._line_of_sight(
            occ, (sr, sc), (gr, gc)
        ), "line_of_sight"
        assert nav_native.nearest_free(occ, sr, sc) == planner._nearest_free(
            occ, sr, sc
        ), "nearest_free"

        cost = (
            rng.integers(0, 4, size=(rows, cols)).astype(np.float64) if rng.integers(0, 2) else None
        )
        assert nav_native.astar(inflated, (sr, sc), (gr, gc), cost) == planner.astar(
            inflated, (sr, sc), (gr, gc), cost=cost
        ), "astar"

        plain = planner.astar(inflated, (sr, sc), (gr, gc))
        if plain is not None:
            assert nav_native.simplify(inflated, plain) == planner.simplify(
                inflated, plain
            ), "simplify"


def test_dispatch_routes_public_api_through_cpp(monkeypatch):
    """With the backend enabled, the public planner/sdf_nav functions must
    return exactly what the forced-Python path returns — i.e. the dispatch is
    transparent."""
    monkeypatch.delenv("GRL_SNAM_NAV_BACKEND", raising=False)
    assert nav_native.enabled()

    rng = np.random.default_rng(0xD15)
    occ = rng.random((48, 48)) < 0.22
    occ[0, 0] = True
    occ[-1, -1] = False
    bounds = (0.0, 0.0, 47.0, 47.0)

    # native path (backend on)
    inflated_native = planner.inflate(occ, 1)
    route_native = planner.astar(inflated_native, (1, 1), (46, 46))
    phi_native, _, _ = sdf_nav.build_sdf(occ, bounds, 0.1)

    # reference path (backend off)
    monkeypatch.setenv("GRL_SNAM_NAV_BACKEND", "python")
    inflated_py = planner.inflate(occ, 1)
    route_py = planner.astar(inflated_py, (1, 1), (46, 46))
    phi_py, _, _ = sdf_nav.build_sdf(occ, bounds, 0.1)

    assert np.array_equal(inflated_native, inflated_py)
    assert route_native == route_py
    assert np.array_equal(phi_native, phi_py)


@pytest.mark.skipif(not hasattr(pycvc, "nav_astar_batch"), reason="pycvc lacks the batch kernels")
def test_batch_matches_per_agent(monkeypatch):
    """The threaded batch kernels (stage 4) return exactly what N per-agent
    calls return."""
    monkeypatch.setenv("GRL_SNAM_NAV_BACKEND", "python")
    rng = np.random.default_rng(0xBA7C4)
    N, H, W = 24, 40, 40
    # bool, so the Python reference's ~occ is a logical (not bitwise) complement
    occ = np.stack([rng.random((H, W)) < rng.uniform(0.08, 0.35) for _ in range(N)])
    occ[:, 0, 0] = True  # a building
    occ[:, -1, -1] = False  # a free cell
    starts = rng.integers(0, H, size=(N, 2)).astype(np.int32)
    goals = rng.integers(0, H, size=(N, 2)).astype(np.int32)
    bounds = (0.0, 0.0, float(W - 1), float(H - 1))

    routes = nav_native.astar_batch(occ, starts, goals)
    fields = nav_native.build_sdf_batch(occ, bounds, 0.1)
    assert len(routes) == N and len(fields) == N
    for i in range(N):
        ref = planner.astar(occ[i], tuple(starts[i]), tuple(goals[i]))  # python
        assert routes[i] == ref, f"batch route {i}"
        pr, nxr, nyr = sdf_nav.build_sdf(occ[i], bounds, 0.1)  # python
        assert np.array_equal(fields[i][0], pr), f"batch phi {i}"
        assert np.array_equal(fields[i][1], nxr), f"batch nx {i}"
        assert np.array_equal(fields[i][2], nyr), f"batch ny {i}"
