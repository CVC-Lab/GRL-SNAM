"""``pycvc.nav_sense_batch`` (via :func:`grl_snam.nav_native.sense_batch`) must be
**bit-identical** to N sequential :meth:`grl_snam.belief.BeliefGrid.sense` calls
in ascending agent index — the last per-agent Python hole in the tick, ported to
C++ without moving the digital twin. Exact equality, not closeness, in every
belief mode (private / clustered / shared) and across ticks (the running clamp).

Skipped automatically when pycvc is absent or predates ``nav_sense_batch``.
"""

import numpy as np
import pytest

pycvc = pytest.importorskip("pycvc")
if not hasattr(pycvc, "nav_sense_batch"):
    pytest.skip("pycvc build has no nav_sense_batch", allow_module_level=True)

from grl_snam import nav_native  # noqa: E402
from grl_snam.belief import BeliefGrid  # noqa: E402

SENSOR = dict(range_m=9.0, n_rays=120, fov_rad=2.0 * np.pi)


def _world(rows, cols):
    return (0.0, 0.0, float(cols - 1), float(rows - 1))


def _cell_to_world(r, c, rows, cols):
    return (float(c), float(r))  # bounds chosen so world == cell units


def _scene(seed, rows, cols, N, M):
    rng = np.random.default_rng(seed)
    truth = rng.random((rows, cols)) < 0.16  # bool — BeliefGrid.sense requires it
    truth[0, 0] = False
    # agents at random free-ish cells (world == cell coords)
    pos = np.stack([rng.uniform(1, cols - 2, N), rng.uniform(1, rows - 2, N)], axis=1).astype(
        np.float64
    )
    headings = rng.uniform(-np.pi, np.pi, N).astype(np.float64)
    agent_map = (np.arange(N) if M == N else rng.integers(0, M, N)).astype(np.int32)
    return rng, truth, pos, headings, agent_map


def _ref_beliefs(truth, pos, headings, agent_map, M, rows, cols, boxes_of=None):
    """N sequential BeliefGrid.sense into M planes, ascending index. `boxes_of(i)`
    optionally returns a truth_now override for agent i (peers/movers baked in)."""
    beliefs = [BeliefGrid((rows, cols), _world(rows, cols)) for _ in range(M)]
    flips = np.zeros(len(pos), np.int32)
    for i in range(len(pos)):
        t_now = truth if boxes_of is None else boxes_of(i)
        flips[i] = beliefs[agent_map[i]].sense(
            t_now, (float(pos[i, 0]), float(pos[i, 1])), heading_rad=float(headings[i]), **SENSOR
        )
    lo = np.stack([b.logodds for b in beliefs]).astype(np.float32)
    lv = np.stack([b.last_visible for b in beliefs])
    es = np.stack([b.ever_seen for b in beliefs])
    ver = np.array([b.version for b in beliefs], np.int32)
    return lo, lv, es, ver, flips


def _blocks(M, rows, cols, N):
    return (
        np.zeros((M, rows, cols), np.float32),
        np.zeros((M, rows, cols), np.bool_),
        np.zeros((M, rows, cols), np.bool_),
        np.zeros((M,), np.int32),
        None,
    )


def _assert_equal(ref, got, tag):
    lo_r, lv_r, es_r, ver_r, fl_r = ref
    lo, lv, es, ver, fl = got
    assert np.array_equal(lo, lo_r), f"{tag}: logodds"
    assert lo.tobytes() == lo_r.tobytes(), f"{tag}: logodds bits"
    assert np.array_equal(lv, lv_r), f"{tag}: last_visible"
    assert np.array_equal(es, es_r), f"{tag}: ever_seen"
    assert np.array_equal(ver, ver_r), f"{tag}: version"
    assert np.array_equal(fl, fl_r), f"{tag}: flips"


@pytest.mark.parametrize("mode", ["private", "clustered", "shared"])
def test_sense_batch_bit_identical(mode):
    for seed in (1, 7, 42, 100):
        rows, cols, N = 40, 44, 24
        M = {"private": N, "clustered": 5, "shared": 1}[mode]
        _, truth, pos, headings, agent_map = _scene(seed, rows, cols, N, M)
        ref = _ref_beliefs(truth, pos, headings, agent_map, M, rows, cols)
        lo, lv, es, ver, _ = _blocks(M, rows, cols, N)
        fl = nav_native.sense_batch(
            truth,
            pos,
            headings,
            lo,
            lv,
            es,
            ver,
            agent_map,
            bounds=_world(rows, cols),
            num_threads=4,
            **SENSOR,
        )
        _assert_equal(ref, (lo, lv, es, ver, fl), f"{mode} seed {seed}")


def test_sense_batch_multi_tick_running_clamp():
    """The clamp accumulates across ticks — agents move and re-sense onto the
    running belief. Private mode, several ticks, byte-identical each tick."""
    rows, cols, N, M = 48, 48, 16, 16
    rng, truth, pos, headings, agent_map = _scene(3, rows, cols, N, M)
    beliefs = [BeliefGrid((rows, cols), _world(rows, cols)) for _ in range(M)]
    lo, lv, es, ver, _ = _blocks(M, rows, cols, N)
    for tick in range(8):
        pos = np.clip(pos + rng.uniform(-1.5, 1.5, pos.shape), 1, min(rows, cols) - 2)
        headings = (headings + rng.uniform(-0.3, 0.3, N)) % (2 * np.pi)
        fl_ref = np.array(
            [
                beliefs[agent_map[i]].sense(
                    truth,
                    (float(pos[i, 0]), float(pos[i, 1])),
                    heading_rad=float(headings[i]),
                    **SENSOR,
                )
                for i in range(N)
            ],
            np.int32,
        )
        fl = nav_native.sense_batch(
            truth,
            pos,
            headings,
            lo,
            lv,
            es,
            ver,
            agent_map,
            bounds=_world(rows, cols),
            num_threads=4,
            **SENSOR,
        )
        ref = (
            np.stack([b.logodds for b in beliefs]).astype(np.float32),
            np.stack([b.last_visible for b in beliefs]),
            np.stack([b.ever_seen for b in beliefs]),
            np.array([b.version for b in beliefs], np.int32),
            fl_ref,
        )
        _assert_equal(ref, (lo, lv, es, ver, fl), f"tick {tick}")


def test_sense_batch_peer_boxes_occlude_and_deposit():
    """Per-agent peer boxes must sense identically to baking those boxes into the
    agent's truth_now — the private/Squad-twin behaviour (peers are +L_OCC hits)."""
    rows, cols, N, M = 36, 36, 8, 8
    rng, truth, pos, headings, agent_map = _scene(11, rows, cols, N, M)
    kmax = 2
    peer_boxes = np.zeros((N, kmax, 4), np.int32)
    for i in range(N):
        cr, cc = int(pos[i, 1]) + 3, int(pos[i, 0]) + 3
        peer_boxes[i, 0] = (max(0, cr), min(rows, cr + 2), max(0, cc), min(cols, cc + 2))
        # second slot zero-area (padding)

    def truth_now(i):
        t = truth.copy()
        r0, r1, c0, c1 = peer_boxes[i, 0]
        t[r0:r1, c0:c1] = 1
        return t

    ref = _ref_beliefs(truth, pos, headings, agent_map, M, rows, cols, boxes_of=truth_now)
    lo, lv, es, ver, _ = _blocks(M, rows, cols, N)
    fl = nav_native.sense_batch(
        truth,
        pos,
        headings,
        lo,
        lv,
        es,
        ver,
        agent_map,
        bounds=_world(rows, cols),
        peer_boxes=peer_boxes,
        num_threads=4,
        **SENSOR,
    )
    _assert_equal(ref, (lo, lv, es, ver, fl), "peer_boxes")


def test_sense_batch_in_place_not_copied():
    """The planes are mutated in place (same object), and a read-only / wrong
    dtype plane is rejected, not silently copied."""
    rows, cols, N, M = 24, 24, 6, 1
    _, truth, pos, headings, agent_map = _scene(9, rows, cols, N, M)
    lo, lv, es, ver, _ = _blocks(M, rows, cols, N)
    lo_id = id(lo)
    before = lo.copy()
    nav_native.sense_batch(
        truth,
        pos,
        headings,
        lo,
        lv,
        es,
        ver,
        agent_map,
        bounds=_world(rows, cols),
        **SENSOR,
    )
    assert id(lo) == lo_id, "logodds must be the same object (in place)"
    assert not np.array_equal(lo, before), "in-place sense must have changed logodds"

    ro = np.zeros((M, rows, cols), np.float32)
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        nav_native.sense_batch(
            truth,
            pos,
            headings,
            ro,
            lv,
            es,
            ver,
            agent_map,
            bounds=_world(rows, cols),
            **SENSOR,
        )
    f64 = np.zeros((M, rows, cols), np.float64)  # wrong dtype
    with pytest.raises(ValueError):
        nav_native.sense_batch(
            truth,
            pos,
            headings,
            f64,
            lv,
            es,
            ver,
            agent_map,
            bounds=_world(rows, cols),
            **SENSOR,
        )
