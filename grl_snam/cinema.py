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
    if lift <= 0.0:
        return eye
    # A hair extra: the solve is exact, so a sample sitting at equality lands
    # ON the margin and rounds either way.
    return np.array([eye[0], eye[1], eye[2] + lift * 1.001 + 1e-3])


def frame_group(points, *, elevation_deg=38.0, azimuth_deg=215.0, fill=0.62, min_radius=60.0):
    """An eye/focal that frames every point, looking down from a fixed bearing.

    Fits the group's bounding sphere and backs off far enough that it subtends
    ``fill`` of the view. A fixed bearing on purpose: a camera that also chases
    the group's heading swings wildly when the group is still converging.
    """
    p = np.asarray(points, np.float64).reshape(-1, 3)
    focal = p.mean(axis=0)
    radius = max(float(np.linalg.norm(p - focal, axis=1).max()), min_radius)
    # distance so the sphere fills `fill` of a 30-degree vertical view angle
    dist = radius / max(np.tan(np.radians(30.0 * 0.5)) * fill, 1e-6)
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    offset = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float64)
    return focal + offset * dist, focal, radius


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
