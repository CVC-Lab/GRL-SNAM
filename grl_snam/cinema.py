"""Cameras that frame a moving group without flying through the city.

Two problems, both geometric:

**Framing.** A shot of eight vehicles spread across a city has to keep all of
them on screen while they converge, without the constant zooming that makes a
tracking shot unwatchable. :func:`frame_group` fits the camera to the group's
bounding sphere and :class:`SmoothCamera` damps the result.

**Occlusion.** A camera placed by geometry alone will happily sit inside a
building or draw its line of sight straight through one, and on a real city
that happens constantly rather than occasionally. :func:`clear_eye` solves for
the smallest lift that puts the whole eye-to-focal segment above the skyline
it crosses.

The height field is rasterized from the building mesh itself, not from the
footprint occupancy: a footprint says a building is *there*, which is enough to
know the camera is inside one but not enough to know how far to rise. The same
grid answers both.
"""

from __future__ import annotations

import numpy as np

# Never let the eye sit exactly on a roof or the terrain: a shot that grazes
# geometry flickers as the depth buffer argues with itself.
DEFAULT_MARGIN_M = 12.0


def building_height_grid(verts, bounds, n: int, occ=None, fill_iters: int = 64) -> np.ndarray:
    """Max mesh height per cell, in world units.

    Buildings arrive as extruded footprints, so the tallest vertex over a cell
    is that cell's roof. Vectorised binning: at ~750k vertices a per-triangle
    rasterisation would dominate the whole setup, and the roof is all the
    camera needs.

    Pass ``occ`` (the footprint occupancy) -- without it the result marks only
    the cells that happen to contain a vertex, i.e. building CORNERS, and the
    interior of every roof reads as zero height.
    """
    v = np.asarray(verts, np.float64).reshape(-1, 3)
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    cols = np.clip(((v[:, 0] - mnx) / (mxx - mnx) * (n - 1)).round().astype(np.intp), 0, n - 1)
    rows = np.clip(((v[:, 1] - mny) / (mxy - mny) * (n - 1)).round().astype(np.intp), 0, n - 1)
    h = np.zeros((n, n), np.float64)
    np.maximum.at(h, (rows, cols), v[:, 2])

    if occ is None:
        return h

    # Binning VERTICES marks a building's corners and leaves its ROOF hollow:
    # measured on Austin, vertex bins covered 2.5% of cells against a 31.3%
    # footprint. A ray crossing the middle of a tower would read height 0 and
    # be judged clear. Flood the corner heights inward across the footprint --
    # a grey dilation masked by occupancy, which for extruded buildings
    # reconstructs the roof exactly and cannot leak outside the footprint.
    occ = np.asarray(occ, bool)
    h = np.where(occ, h, np.where(h > 0, h, 0.0))
    for _ in range(int(fill_iters)):
        g = h
        for shift in (
            np.roll(h, 1, 0),
            np.roll(h, -1, 0),
            np.roll(h, 1, 1),
            np.roll(h, -1, 1),
        ):
            g = np.maximum(g, shift)
        grown = np.where(occ, g, h)
        if np.array_equal(grown, h):
            break
        h = grown

    # A footprint island whose vertices all rounded into neighbouring cells is
    # never seeded, so the flood leaves it at zero -- 4359 such cells on
    # Austin, each one a building the camera would fly through. Fall back to a
    # representative height rather than zero: for clearance, guessing a
    # building is taller than it is costs a little altitude, and guessing it is
    # shorter costs the shot.
    holes = occ & (h <= 0.0)
    if holes.any():
        seeded = h[occ & (h > 0.0)]
        h[holes] = float(np.median(seeded)) if seeded.size else 10.0
    return h


def sample_height(h: np.ndarray, bounds, x, y) -> np.ndarray:
    """Nearest-cell height lookup for world points (arrays welcome)."""
    mnx, mny, mxx, mxy = (float(b) for b in bounds)
    ny, nx = h.shape
    c = np.clip(((np.asarray(x) - mnx) / (mxx - mnx) * (nx - 1)).round().astype(np.intp), 0, nx - 1)
    r = np.clip(((np.asarray(y) - mny) / (mxy - mny) * (ny - 1)).round().astype(np.intp), 0, ny - 1)
    return h[r, c]


def clear_eye(
    eye,
    focal,
    height: np.ndarray,
    bounds,
    *,
    margin_m: float = DEFAULT_MARGIN_M,
    ground_margin_m: float = 2.0,
    samples: int = 64,
    near_cut: float = 0.88,
    tail_m: float | None = None,
    max_lift_m: float | None = None,
):
    """Lift ``eye`` until the whole segment to ``focal`` clears the skyline.

    Raising the eye by ``d`` raises the point at parameter ``t`` by
    ``(1 - t) * d``, so the requirement ``z(t) >= height(t) + margin`` solves
    directly for the smallest sufficient lift::

        d = max over t of  (height(t) + margin - z(t)) / (1 - t)

    Samples past ``near_cut`` are ignored: the far end of the segment IS the
    subject, which stands on the ground, and demanding clearance there would
    push the camera to infinity for no benefit.

    Returns the corrected eye. Only the height changes — the direction of the
    shot is the caller's, and silently sliding it sideways would fight whatever
    framing decision put it there.
    """
    eye = np.asarray(eye, np.float64)
    focal = np.asarray(focal, np.float64)
    if tail_m is not None:
        # A cut expressed as a FRACTION means a wide shot stops checking
        # hundreds of metres short of its subject. Expressed as a distance, the
        # exempt zone is the same few metres of ground the subject stands on
        # whatever the shot size, so wide shots get checked almost all the way
        # in and close shots still get their approach.
        dist = float(np.linalg.norm(focal - eye))
        near_cut = float(np.clip(1.0 - tail_m / max(dist, 1e-6), 0.5, 0.985))
    t = np.linspace(0.0, near_cut, samples)
    p = eye[None, :] + t[:, None] * (focal - eye)[None, :]
    # The margin is about not clipping BUILDINGS. Demanding the same clearance
    # over open ground would forbid every low approach to a subject standing on
    # it -- the camera has to come down somewhere.
    h = sample_height(height, bounds, p[:, 0], p[:, 1])
    need = np.where(h > 0.0, h + margin_m, ground_margin_m)
    deficit = need - p[:, 2]
    denom = np.maximum(1.0 - t, 1e-3)
    lift = float(np.max(deficit / denom))
    if max_lift_m is not None:
        # The solve is exact but unbounded, and on a wide shot that is a trap:
        # `tail_m` puts near_cut at 1 - 30/dist, so at 1000 m the last sample
        # divides by 0.03, and a 40 m building beside the subject demands a
        # 1800 m climb. Refusing to climb that far leaves the subject partly
        # occluded; climbing it abandons the city altogether. Cap the climb and
        # let the bearing search -- which scores actual subject visibility --
        # find a viewpoint that does not need one.
        lift = min(lift, float(max_lift_m))
    if lift <= 0.0:
        return eye
    # A hair extra: the solve is exact, so a sample sitting at equality lands
    # ON the margin and rounds either way.
    return np.array([eye[0], eye[1], eye[2] + lift * 1.001 + 1e-3])


def frame_group(
    points,
    *,
    elevation_deg=38.0,
    azimuth_deg=215.0,
    fill=0.62,
    min_radius=60.0,
    fov_deg=30.0,
    max_height_m=None,
):
    """An eye/focal that frames every point, looking down from a fixed bearing.

    Fits the group's bounding sphere and backs off far enough that it subtends
    ``fill`` of the view. A fixed bearing on purpose: a camera that also chases
    the group's heading swings wildly when the group is still converging.
    """
    p = np.asarray(points, np.float64).reshape(-1, 3)
    focal = p.mean(axis=0)
    radius = max(float(np.linalg.norm(p - focal, axis=1).max()), min_radius)
    # Distance so the sphere fills `fill` of the vertical view angle. The LENS
    # is the main control over how far away the camera has to stand: at 30
    # degrees a group must be framed from 4.66 radii, at 55 degrees from 2.15.
    # For a group spread across a kilometre that is the difference between
    # hanging above the city and standing in it.
    dist = radius / max(np.tan(np.radians(fov_deg * 0.5)) * fill, 1e-6)
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    if max_height_m is not None:
        # Given how far back the group forces us, pick the steepest elevation
        # that still keeps the eye under the ceiling. A fixed elevation turns
        # every metre of required distance into altitude, so the wider the
        # group spreads the higher the camera climbs -- exactly backwards. This
        # trades the climb for a longer, flatter look across the city.
        el = min(el, float(np.arcsin(np.clip(max_height_m / max(dist, 1e-6), 0.0, 1.0))))
    offset = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float64)
    return focal + offset * dist, focal, radius


def clear_shot(
    points,
    height: np.ndarray,
    bounds,
    *,
    elevation_deg: float,
    azimuth_deg: float,
    fill: float = 0.62,
    margin_m: float = DEFAULT_MARGIN_M,
    tail_m: float = 30.0,
    bearings: int = 24,
    fov_deg: float = 30.0,
    max_height_m=None,
    max_lift_m: float | None = None,
    turn_cost_m_per_deg: float = 1.1,
    hidden_cost_m: float = 400.0,
    subject_tail_m: float = 8.0,
    subject_rise_m: float = 0.0,
):
    """Frame the group from the bearing that sees it, not just from over it.

    :func:`clear_eye` can only answer occlusion by climbing, and climbing is
    the wrong answer when the thing in the way is right next to the subject: a
    stadium wall 30 m from the group would push the camera hundreds of metres
    up and flatten the shot into a plan view. A camera operator would step
    sideways instead.

    So try every bearing, cost each by the lift it still needs, and take the
    cheapest — with a penalty for turning away from the bearing the shot
    schedule asked for, so the camera holds its intended angle whenever that
    angle works and only swings out when it genuinely cannot see. The penalty
    is in metres-of-lift per degree, which is what makes the two commensurable.

    Searching elevation as well was tried and dropped: it tripled the cost and
    moved 1 blocked ray in 960. What is left after the bearing search is a
    vehicle driving hard against a wall, where the ray grazes that wall in its
    final few metres -- no camera position fixes that, and visually it does not
    need fixing, because the vehicle is at the building's edge and in plain
    sight.

    The bearing is scored on whether the SUBJECTS are visible, not merely on
    whether the line to their centroid is. Those differ exactly when it matters:
    the centroid of a group straddling a corner sits in the open while half the
    group is behind the wall. Hiding a vehicle is priced far above any lift, so
    clearance wins the argument and lift and bearing only break ties.

    Returns ``(eye, focal, radius, azimuth_used)``.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if subject_rise_m:
        # Score the MARKER, not the chassis. A renderer that stands a beacon on
        # each subject has already solved "can the viewer find it": measured on
        # the eight-agent rendezvous at a 231 m eye, 49.9% of vehicle BODIES are
        # behind a building and 0.8% of beacons are. Scoring bodies makes the
        # search fight an unwinnable battle -- the hidden term saturates, swamps
        # the lift and turn terms, and the bearing choice turns to noise -- while
        # also driving the camera up out of the city to see chassis that the
        # viewer was never tracking anyway.
        pts = pts.copy()
        pts[:, 2] += float(subject_rise_m)
    best = None
    for k in range(int(bearings)):
        az = azimuth_deg + 360.0 * k / int(bearings)
        eye, focal, radius = frame_group(
            points,
            elevation_deg=elevation_deg,
            azimuth_deg=az,
            fill=fill,
            fov_deg=fov_deg,
            max_height_m=max_height_m,
        )
        lifted = clear_eye(
            eye, focal, height, bounds, margin_m=margin_m, tail_m=tail_m, max_lift_m=max_lift_m
        )
        lift = float(lifted[2] - eye[2])
        hidden = _hidden_count(lifted, pts, height, bounds, tail_m=subject_tail_m)
        # Shortest way round: 350 degrees off is 10 degrees off.
        turn = abs((az - azimuth_deg + 180.0) % 360.0 - 180.0)
        cost = hidden_cost_m * hidden + lift + turn_cost_m_per_deg * turn
        if best is None or cost < best[0]:
            best = (cost, lifted, focal, radius, az)
        if cost == 0.0:  # nothing hidden, no lift, no turn -- cannot do better
            break
    _c, eye, focal, radius, az = best
    return eye, focal, radius, az


def _hidden_count(eye, pts, height, bounds, *, tail_m: float = 8.0, samples: int = 48) -> int:
    """How many of ``pts`` the skyline hides from ``eye``."""
    eye = np.asarray(eye, np.float64)
    d = np.linalg.norm(pts - eye[None, :], axis=1)
    cut = np.clip(1.0 - tail_m / np.maximum(d, 1e-6), 0.3, 0.99)
    # One ray per subject, sampled in lockstep: (subjects, samples, 3).
    t = cut[:, None] * np.linspace(0.0, 1.0, samples)[None, :]
    p = eye[None, None, :] + t[:, :, None] * (pts - eye[None, :])[:, None, :]
    h = sample_height(height, bounds, p[:, :, 0], p[:, :, 1])
    return int(np.count_nonzero(np.any(p[:, :, 2] < h + 1.0, axis=1)))


def shot_angles(u: float, *, low_deg: float = 16.0, high_deg: float = 34.0, drift_deg: float = 9.0):
    """Elevation and azimuth for normalised clip progress ``u`` in [0, 1].

    Opens high and wide — an establishing angle that reads the city as a map —
    then eases down to a low angle where the vehicles have the skyline behind
    them and the geometry has depth. The decay is exponential rather than
    linear so the descent is over early and the rest of the clip is steady:
    a camera still descending during the action reads as indecision.

    Azimuth drifts slowly and monotonically. That is what makes a static city
    look three-dimensional — parallax the viewer did not ask for. Slowly is the
    operative word: a bearing that chases the group's heading swings hard every
    time the group re-forms, which is exactly when the viewer needs a stable
    frame of reference.
    """
    u = float(np.clip(u, 0.0, 1.0))
    elevation = low_deg + (high_deg - low_deg) * float(np.exp(-u / 0.12))
    return elevation, 215.0 + drift_deg * u


class SmoothCamera:
    """Critically-damped follow for an eye/focal stream.

    Framing a converging group produces a target that jumps every frame — the
    bounding sphere shrinks as they close. Damping is what turns that into a
    move rather than a twitch; the time constants are separate because a
    focal point that lags its own subject reads as a mistake, while an eye
    that lags reads as weight.
    """

    def __init__(self, eye_tau: float = 1.1, focal_tau: float = 0.55):
        self.eye_tau = float(eye_tau)
        self.focal_tau = float(focal_tau)
        self._eye = None
        self._focal = None

    def update(self, eye, focal, dt: float):
        eye = np.asarray(eye, np.float64)
        focal = np.asarray(focal, np.float64)
        if self._eye is None:
            self._eye, self._focal = eye.copy(), focal.copy()
            return self._eye, self._focal
        ke = 1.0 - np.exp(-max(dt, 1e-6) / max(self.eye_tau, 1e-6))
        kf = 1.0 - np.exp(-max(dt, 1e-6) / max(self.focal_tau, 1e-6))
        self._eye += (eye - self._eye) * ke
        self._focal += (focal - self._focal) * kf
        return self._eye, self._focal
