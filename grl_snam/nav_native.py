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


# ── batched, threaded per-agent kernels (PERFORMANCE.md stage 4) ─────────────
# The agents are independent, so a sense tick's N replans / SDF rebuilds fan out
# across cores in a single call that releases the GIL. Wire these into
# ``Squad.step`` (one call for the whole squad on a sense tick) to reach the
# 100-1000 agent target; each result is byte-identical to the per-agent path.


def astar_batch(occs, starts, goals, costs=None, num_threads=0):
    """Batched A*. ``occs`` is (N,H,W) uint8/bool (agent i's belief in plane i);
    ``starts``/``goals`` are (N,2); ``costs`` is (N,H,W) or None. Returns a list
    of N routes (each a list of ``(r, c)`` or ``None`` when unreachable)."""
    occ = np.ascontiguousarray(occs, np.uint8)
    st = np.ascontiguousarray(starts, np.int32)
    gl = np.ascontiguousarray(goals, np.int32)
    c = None if costs is None else np.ascontiguousarray(costs, np.float64)
    arrs = _pycvc.nav_astar_batch(occ, st, gl, c, int(num_threads))
    return [None if a.shape[0] == 0 else [(int(r), int(cc)) for r, cc in a] for a in arrs]


def build_sdf_batch(occs, bounds, scale, num_threads=0):
    """Batched SDF build over ``occs`` (N,H,W). Returns a list of N
    ``(phi, normal_x, normal_y)`` triples."""
    occ = np.ascontiguousarray(occs, np.uint8)
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    s = _pycvc.nav_build_sdf_batch(occ, mnx, mny, mxx, mxy, float(scale), int(num_threads))
    return [(s[i, 0], s[i, 1], s[i, 2]) for i in range(s.shape[0])]


def inflate_batch(occs, cells, num_threads=0):
    """Batched 4-connected dilation over ``occs`` (N,H,W). Returns (N,H,W)
    uint8 (0/1). Each plane is byte-identical to the serial :func:`inflate`."""
    occ = np.ascontiguousarray(occs, np.uint8)
    return _pycvc.nav_inflate_batch(occ, int(cells), int(num_threads))


def neighbors(positions, radius):
    """Fixed-radius neighbour query over ``positions`` (N,2) via a CGAL Kd_tree.
    Returns a list of N index arrays: entry i holds the indices of every other
    point within ``radius`` of point i. The specialized structure for the crowd
    N-body query (peers within sensor range), robust to clustering."""
    pos = np.ascontiguousarray(positions, np.float64)
    return _pycvc.nav_neighbors(pos, float(radius))


HAS_SENSE_BATCH = AVAILABLE and hasattr(_pycvc, "nav_sense_batch")


def sense_batch(
    truth,
    positions,
    headings,
    logodds,
    last_visible,
    ever_seen,
    version,
    agent_map,
    *,
    range_m,
    n_rays=240,
    fov_rad=2.0 * np.pi,
    bounds,
    peer_boxes=None,
    mover_boxes=None,
    l_occ=2.2,
    l_free=-1.4,
    l_clamp=8.0,
    num_threads=0,
):
    """Batched, in-place belief sense — a bit-identical port of
    :meth:`grl_snam.belief.BeliefGrid.sense` over N agents into M belief planes.

    ``logodds`` (M,H,W float32), ``last_visible``/``ever_seen`` (M,H,W bool) and
    ``version`` (M,) int32 are **mutated in place** — they are validated and
    written through, NEVER copied or coerced (a silent copy would swallow every
    update). ``agent_map`` (N,) int32 selects each agent's plane:
    ``arange(N)`` = private (fully parallel), ``zeros(N)`` = shared (one plane,
    folded serially in ascending index), arbitrary labels = clustered. Agents
    that share a plane are folded sequentially in ascending index, exactly as N
    serial ``sense`` calls would; distinct planes run on separate threads.

    ``peer_boxes`` (N,Kmax,4) and ``mover_boxes`` (Mv,4) are optional HALF-OPEN
    ``(r0,r1,c0,c1)`` cell rects that, when given, occlude rays AND deposit
    ``+l_occ`` AND enter the field of view — the reference ``truth_now``
    composition. Pass ``None`` (the shared/clustered swarm's choice) to keep
    peers out of the log-odds and route them through the decaying dynamic layer
    instead. Returns ``flips`` (N,) int32."""
    N = int(positions.shape[0])
    for nm, a, dt, nd in (
        ("logodds", logodds, np.float32, 3),
        ("last_visible", last_visible, np.bool_, 3),
        ("ever_seen", ever_seen, np.bool_, 3),
        ("version", version, np.int32, 1),
    ):
        if a.dtype != dt or a.ndim != nd or not a.flags["C_CONTIGUOUS"] or not a.flags["WRITEABLE"]:
            raise ValueError(
                f"sense_batch: {nm} must be a writable C-contiguous {dt.__name__} array of "
                f"ndim {nd} — it is mutated in place, never copied"
            )
    tr = np.ascontiguousarray(truth, np.uint8)  # read-only inputs: coerce freely
    pos = np.ascontiguousarray(positions, np.float64)
    hd = np.ascontiguousarray(headings, np.float64)
    am = np.ascontiguousarray(agent_map, np.int32)
    rng = np.ascontiguousarray(np.broadcast_to(range_m, N), np.float64)  # scalar OR (N,)
    nry = np.ascontiguousarray(np.broadcast_to(n_rays, N), np.int32)
    fov = np.ascontiguousarray(np.broadcast_to(fov_rad, N), np.float64)
    pb = None if peer_boxes is None else np.ascontiguousarray(peer_boxes, np.int32)
    mb = None if mover_boxes is None else np.ascontiguousarray(mover_boxes, np.int32)
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    return _pycvc.nav_sense_batch(
        tr,
        pos,
        hd,
        rng,
        nry,
        fov,
        logodds,
        last_visible,
        ever_seen,
        version,
        am,
        pb,
        mb,
        mnx,
        mny,
        mxx,
        mxy,
        float(l_occ),
        float(l_free),
        float(l_clamp),
        int(num_threads),
    )


HAS_SDF_SAMPLE = AVAILABLE and hasattr(_pycvc, "nav_sdf_sample")


def sdf_sample(field, on, *, bounds, center, scale, map_id=None, num_threads=0):
    """Torch-free bilinear SDF sample (the drive's field read). ``field`` is an
    ``(M,3,H,W)`` float32 stack (channel 0 phi, 1 normal_x, 2 normal_y); ``on`` is
    ``(N,2)`` float32 normalized (centered) positions; ``map_id`` is ``(N,)`` int32
    or None (=> plane 0 for all). ``bounds``/``center``/``scale`` are the SDFField's
    world<->grid constants. Returns ``(phi (N,), normal (N,2))`` float32,
    float-equivalent to :meth:`sdf_nav.SDFField.sample` /
    :meth:`sdf_nav.BatchedSDFField.sample` (torch grid_sample, align_corners=True,
    border) — the first piece of the torch-free C++ drive port."""
    f = np.ascontiguousarray(field, np.float32)
    o = np.ascontiguousarray(on, np.float32)
    mid = None if map_id is None else np.ascontiguousarray(map_id, np.int32)
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    cx, cy = float(center[0]), float(center[1])
    return _pycvc.nav_sdf_sample(
        f, o, mid, mnx, mny, mxx, mxy, cx, cy, float(scale), int(num_threads)
    )


HAS_COEF_MLP = AVAILABLE and hasattr(_pycvc, "nav_coef_mlp_forward")


def coef_mlp_forward(path, feats, num_threads=0):
    """Forward ``feats`` ``(N,in)`` float32 through the torch-free C++ policy in
    the ``.cvcnav`` file at ``path``; returns ``(N,out)`` float32 (alpha, beta,
    gamma). Float-equivalent to :meth:`sdf_nav.CoefMLP.forward`. Export the file
    with :func:`grl_snam.tools.coef_export.write_coef_mlp`."""
    f = np.ascontiguousarray(feats, np.float32)
    return _pycvc.nav_coef_mlp_forward(str(path), f, int(num_threads))


HAS_OCCUPANCY = AVAILABLE and hasattr(_pycvc, "nav_composite_occupancy")


def to_occupancy(logodds, *, unknown="optimistic", p_thresh=0.5, band=0.15):
    """Threshold a log-odds belief into a bool occupancy raster, bit-identical to
    :meth:`grl_snam.belief.BeliefGrid.to_occupancy` (float32 sigmoid)."""
    occ = _pycvc.nav_composite_occupancy(
        np.ascontiguousarray(logodds, np.float32),
        unknown,
        float(p_thresh),
        float(band),
        None,
        0.0,
        4.0,
    )
    return occ.astype(bool)


def composite_occupancy(
    logodds, dyn_stamp, t_now, ttl_s, *, unknown="optimistic", p_thresh=0.5, band=0.15
):
    """to_occupancy OR the decaying dynamic layer, bit-identical to
    :func:`grl_snam.belief.composite_occupancy`. ``dyn_stamp`` is the
    :class:`~grl_snam.belief.DynamicLayer` ``_stamp`` raster (float64) or None."""
    ds = None if dyn_stamp is None else np.ascontiguousarray(dyn_stamp, np.float64)
    occ = _pycvc.nav_composite_occupancy(
        np.ascontiguousarray(logodds, np.float32),
        unknown,
        float(p_thresh),
        float(band),
        ds,
        float(t_now),
        float(ttl_s),
    )
    return occ.astype(bool)


HAS_DRIVE = AVAILABLE and hasattr(_pycvc, "nav_bicycle_rollout")


def coef_feats(field, on, goal, *, bounds, center, scale, map_id=None, num_threads=0):
    """Coefficient-net features ``[phi, |goal-o|, gdir_x, gdir_y, gdir.normal]``,
    float-equivalent to :func:`sdf_nav.coef_feats`. ``field`` ``(M,3,H,W)`` f32,
    ``on``/``goal`` ``(N,2)`` f32 normalized, ``map_id`` ``(N,)`` int32 or None."""
    f = np.ascontiguousarray(field, np.float32)
    o = np.ascontiguousarray(on, np.float32)
    g = np.ascontiguousarray(goal, np.float32)
    mid = None if map_id is None else np.ascontiguousarray(map_id, np.int32)
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    return _pycvc.nav_coef_feats(
        f,
        o,
        g,
        mid,
        mnx,
        mny,
        mxx,
        mxy,
        float(center[0]),
        float(center[1]),
        float(scale),
        int(num_threads),
    )


def bicycle_rollout(
    field, o, th, sp, goal, al, be, ga, *, bounds, center, scale, params, map_id=None, num_threads=0
):
    """One torch-free bicycle drive tick (``params['nsub']`` substeps), float-
    equivalent to :func:`sdf_nav.bicycle_rollout` (steps=1). ``params`` carries
    ``rr,d_hat,dt,vmax,L,delta_max,a_max,a_lat_max,k_steer,nsub,allow_reverse``.
    Returns fresh ``(o, th, sp, minclr)`` f32; inputs are not mutated."""
    f = np.ascontiguousarray(field, np.float32)
    P = params
    return _pycvc.nav_bicycle_rollout(
        f,
        np.ascontiguousarray(o, np.float32),
        np.ascontiguousarray(th, np.float32),
        np.ascontiguousarray(sp, np.float32),
        np.ascontiguousarray(goal, np.float32),
        np.ascontiguousarray(al, np.float32),
        np.ascontiguousarray(be, np.float32),
        np.ascontiguousarray(ga, np.float32),
        None if map_id is None else np.ascontiguousarray(map_id, np.int32),
        *(float(b) for b in bounds),
        float(center[0]),
        float(center[1]),
        float(scale),
        float(P["rr"]),
        float(P["d_hat"]),
        float(P["dt"]),
        float(P["vmax"]),
        float(P["L"]),
        float(P["delta_max"]),
        float(P["a_max"]),
        float(P["a_lat_max"]),
        float(P["k_steer"]),
        int(P["nsub"]),
        int(bool(P["allow_reverse"])),
        int(num_threads),
    )


def drive_step(
    field,
    o,
    th,
    sp,
    carrot,
    weights_path,
    *,
    bounds,
    center,
    scale,
    params,
    map_id=None,
    num_threads=0,
):
    """The fused per-agent drive tick (sample -> coef_feats -> coef_mlp -> bicycle),
    float-equivalent to the torch Swarm drive. ``weights_path`` is a ``.cvcnav``
    policy (see :mod:`grl_snam.tools.coef_export`); ``params`` as
    :func:`bicycle_rollout`. Returns fresh ``(o, th, sp, minclr)`` f32."""
    f = np.ascontiguousarray(field, np.float32)
    P = params
    return _pycvc.nav_drive_step(
        f,
        np.ascontiguousarray(o, np.float32),
        np.ascontiguousarray(th, np.float32),
        np.ascontiguousarray(sp, np.float32),
        np.ascontiguousarray(carrot, np.float32),
        str(weights_path),
        None if map_id is None else np.ascontiguousarray(map_id, np.int32),
        *(float(b) for b in bounds),
        float(center[0]),
        float(center[1]),
        float(scale),
        float(P["rr"]),
        float(P["d_hat"]),
        float(P["dt"]),
        float(P["vmax"]),
        float(P["L"]),
        float(P["delta_max"]),
        float(P["a_max"]),
        float(P["a_lat_max"]),
        float(P["k_steer"]),
        int(P["nsub"]),
        int(bool(P["allow_reverse"])),
        int(num_threads),
    )


HAS_SIM_WORLD = AVAILABLE and hasattr(_pycvc, "nav_sim_world_create")


class NativeSimWorld:
    """A thin handle over the pure-C++ ``cvc::nav::sim_world`` — the torch-free
    reactive swarm runtime. Build one with :func:`sim_world_from_swarm`, then
    ``step()`` and read ``snapshot()`` (pos world, heading, speed, mode,
    reached). This is what a C++ host embeds; the Python handle is for tests and
    orchestration."""

    def __init__(self, handle, n):
        self._h = handle
        self.n = int(n)

    def step(self, num_threads=0):
        _pycvc.nav_sim_world_step(self._h, int(num_threads))

    def snapshot(self):
        """(pos (N,2) world f32, heading f32, speed f32, mode i32, reached bool)."""
        return _pycvc.nav_sim_world_snapshot(self._h)

    def retarget(self, i, gx_n, gy_n):
        _pycvc.nav_sim_world_retarget(self._h, int(i), float(gx_n), float(gy_n))


def sim_world_from_swarm(sw, weights_path, *, truth, freeze_sense=False, sense_every=4):
    """Build a :class:`NativeSimWorld` mirroring a shared-belief :class:`Swarm`'s
    init (same poses, goals, colors, field prior, sensor / vehicle params and the
    same ``.cvcnav`` policy) — the pure-C++ counterpart of the torch swarm."""
    o = np.ascontiguousarray(sw.o.detach().cpu().numpy(), np.float32)
    goal = np.ascontiguousarray(sw.goal.detach().cpu().numpy(), np.float32)
    color = np.ascontiguousarray(sw.color.detach().cpu().numpy(), np.float32)
    tr = np.ascontiguousarray(truth, np.uint8)
    H, W = tr.shape
    v, k = sw.veh, sw.kw
    mnx, mny, mxx, mxy = sw.bounds
    h = _pycvc.nav_sim_world_create(
        tr,
        tr,
        str(weights_path),
        o,
        goal,
        color,
        H,
        W,
        mnx,
        mny,
        mxx,
        mxy,
        sw.cx,
        sw.cy,
        sw.scale,
        sw.sensor.get("range_m", 60.0),
        int(sw.sensor.get("n_rays", 240)),
        float(sw.sensor.get("fov_rad", 2 * np.pi)),
        k["rr"],
        k["d_hat"],
        k["dt"],
        k["vmax"],
        v["L"],
        v["delta_max"],
        v["a_max"],
        v["a_lat_max"],
        v["k_steer"],
        int(sw.nsub),
        int(bool(v["allow_reverse"])),
        float(sw.reach_tol),
        int(sense_every),
        int(bool(freeze_sense)),
        2.2,
        -1.4,
        8.0,
        1,
        0.5,
        0.15,
        4.0,
    )
    return NativeSimWorld(h, sw.N)


HAS_SIM_THREAD = AVAILABLE and hasattr(_pycvc, "nav_sim_thread_create")


class NativeSimThread:
    """Run a :class:`NativeSimWorld` off the render thread — the C++
    ``cvc::nav::sim_thread``. The worker is a real thread that never touches the
    GIL (it runs only cvc::nav kernels), so it advances genuinely concurrently
    with Python; ``read()`` is a lock-free snapshot the renderer never blocks
    the sim to take."""

    def __init__(self, sim_world, hz=60.0):
        self._sw = sim_world  # keep the NativeSimWorld (and its capsule) alive
        self._h = _pycvc.nav_sim_thread_create(sim_world._h, float(hz))

    def start(self):
        _pycvc.nav_sim_thread_start(self._h)

    def stop(self):
        _pycvc.nav_sim_thread_stop(self._h)

    def read(self):
        """(pos (N,2) world, heading, speed, mode, reached, tick) or None."""
        return _pycvc.nav_sim_thread_read(self._h)

    def retarget(self, i, gx_n, gy_n):
        _pycvc.nav_sim_thread_retarget(self._h, int(i), float(gx_n), float(gy_n))

    def set_paused(self, paused):
        _pycvc.nav_sim_thread_set_paused(self._h, int(bool(paused)))

    def set_rate(self, hz):
        _pycvc.nav_sim_thread_set_rate(self._h, float(hz))

    @property
    def ticks(self):
        return _pycvc.nav_sim_thread_ticks(self._h)

    @property
    def behind(self):
        return _pycvc.nav_sim_thread_behind(self._h)


HAS_CUDA_DRIVE = AVAILABLE and hasattr(_pycvc, "nav_drive_step_cuda")


def drive_step_cuda(field, o, th, sp, carrot, weights_path, *, bounds, center, scale, params):
    """The GPU fused drive tick (nav/drive.cu), float-equivalent to
    :func:`drive_step` / the torch drive. Shared field (plane 0). Raises if this
    pycvc was built without CUDA or no CUDA device is available."""
    f = np.ascontiguousarray(field, np.float32)
    P = params
    return _pycvc.nav_drive_step_cuda(
        f,
        np.ascontiguousarray(o, np.float32),
        np.ascontiguousarray(th, np.float32),
        np.ascontiguousarray(sp, np.float32),
        np.ascontiguousarray(carrot, np.float32),
        str(weights_path),
        *(float(b) for b in bounds),
        float(center[0]),
        float(center[1]),
        float(scale),
        float(P["rr"]),
        float(P["d_hat"]),
        float(P["dt"]),
        float(P["vmax"]),
        float(P["L"]),
        float(P["delta_max"]),
        float(P["a_max"]),
        float(P["a_lat_max"]),
        float(P["k_steer"]),
        int(P["nsub"]),
        int(bool(P["allow_reverse"])),
    )
