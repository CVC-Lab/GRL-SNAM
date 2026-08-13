"""Framing a group, and keeping the camera out of the city.

A camera placed by geometry alone flies through buildings constantly on a real
map — measured on Austin, 300 of 300 low-angle shots had their line of sight
inside geometry before this existed.
"""

import numpy as np
import pytest

from grl_snam.cinema import (
    SmoothCamera,
    building_height_grid,
    clear_eye,
    frame_group,
    sample_height,
)

BOUNDS = (-100.0, -100.0, 100.0, 100.0)
N = 64


def _tower(height=40.0, r0=28, r1=36, c0=28, c1=36):
    """One extruded block: occupancy plus the vertices of its corners only —
    which is exactly what a mesh gives you."""
    occ = np.zeros((N, N), bool)
    occ[r0:r1, c0:c1] = True

    def w(r, c):
        return (
            BOUNDS[0] + c / (N - 1) * (BOUNDS[2] - BOUNDS[0]),
            BOUNDS[1] + r / (N - 1) * (BOUNDS[3] - BOUNDS[1]),
        )

    verts = []
    for r, c in ((r0, c0), (r0, c1 - 1), (r1 - 1, c0), (r1 - 1, c1 - 1)):
        x, y = w(r, c)
        verts += [x, y, 0.0, x, y, height]
    return occ, verts


# ── the height field ────────────────────────────────────────────────────────


def test_roofs_are_solid_not_hollow():
    """Binning vertices marks a building's CORNERS. Without filling, a ray over
    the middle of a tower reads height 0 and is judged clear — measured on
    Austin, vertex bins covered 2.5% of cells against a 31.3% footprint."""
    occ, verts = _tower(height=40.0)
    hollow = building_height_grid(verts, BOUNDS, N)
    filled = building_height_grid(verts, BOUNDS, N, occ=occ)

    assert (occ & (hollow <= 0)).any(), "the hollow case should have interior gaps"
    assert not (occ & (filled <= 0)).any(), "every footprint cell needs a height"
    assert filled[32, 32] == pytest.approx(40.0), "the roof middle is at roof height"


def test_height_never_leaks_outside_the_footprint():
    occ, verts = _tower()
    h = building_height_grid(verts, BOUNDS, N, occ=occ)
    assert not (h[~occ] > 0).any(), "the flood escaped the building"


def test_an_unseeded_footprint_still_gets_a_height():
    """A building whose vertices all round into neighbouring cells is never
    seeded, so the flood leaves it at zero — a building the camera would fly
    straight through."""
    occ, verts = _tower(height=30.0)
    occ[5:8, 5:8] = True  # an island with no vertices at all
    h = building_height_grid(verts, BOUNDS, N, occ=occ)
    assert h[6, 6] > 0.0, "an unseeded island must not read as open ground"


# ── clearance ───────────────────────────────────────────────────────────────


def _blocked(h, eye, focal, margin=12.0, ground=2.0, n=64, cut=0.88):
    """Mirrors clear_eye's contract: clearance over BUILDINGS, and merely
    above the ground elsewhere — a camera diving at a subject standing in the
    open is not 'blocked'."""
    t = np.linspace(0.0, cut, n)
    p = np.asarray(eye)[None, :] + t[:, None] * (np.asarray(focal) - np.asarray(eye))[None, :]
    hs = sample_height(h, BOUNDS, p[:, 0], p[:, 1])
    return bool(np.any(np.where(hs > 0.0, hs + margin, ground) > p[:, 2]))


def test_a_line_of_sight_through_a_tower_is_lifted_over_it():
    occ, verts = _tower(height=50.0)
    h = building_height_grid(verts, BOUNDS, N, occ=occ)
    eye = np.array([-90.0, -90.0, 5.0])
    focal = np.array([60.0, 60.0, 0.0])  # the segment crosses the tower
    assert _blocked(h, eye, focal), "the fixture should start blocked"
    fixed = clear_eye(eye, focal, h, BOUNDS)
    assert not _blocked(h, fixed, focal)
    assert fixed[2] > eye[2]
    assert fixed[0] == eye[0] and fixed[1] == eye[1], "only the height may change"


def test_a_clear_shot_is_left_alone():
    """Lifting a camera that does not need it would quietly flatten every
    low-angle shot in the film."""
    occ, verts = _tower(height=50.0)
    h = building_height_grid(verts, BOUNDS, N, occ=occ)
    eye = np.array([-90.0, 90.0, 40.0])
    focal = np.array([-60.0, 60.0, 0.0])  # nowhere near the tower
    assert not _blocked(h, eye, focal)
    assert np.array_equal(clear_eye(eye, focal, h, BOUNDS), eye)


def test_an_eye_buried_in_a_building_is_raised_above_it():
    occ, verts = _tower(height=60.0)
    h = building_height_grid(verts, BOUNDS, N, occ=occ)
    eye = np.array([0.0, 0.0, 3.0])  # inside the tower
    focal = np.array([80.0, 80.0, 0.0])
    fixed = clear_eye(eye, focal, h, BOUNDS)
    assert fixed[2] > 60.0, "still inside the building"


def test_clearance_holds_across_many_random_low_shots():
    occ, verts = _tower(height=45.0)
    h = building_height_grid(verts, BOUNDS, N, occ=occ)
    rng = np.random.default_rng(0)
    for _ in range(200):
        pts = np.stack([rng.uniform(-60, 60, 6), rng.uniform(-60, 60, 6), np.zeros(6)], axis=-1)
        eye, focal, _r = frame_group(pts, elevation_deg=float(rng.uniform(3.0, 20.0)))
        assert not _blocked(h, clear_eye(eye, focal, h, BOUNDS), focal)


# ── framing ─────────────────────────────────────────────────────────────────


def test_every_agent_is_inside_the_framed_sphere():
    pts = np.stack([[0, 100, -80, 40], [0, -60, 90, 20], [0, 0, 0, 0]], axis=-1).astype(float)
    eye, focal, radius = frame_group(pts)
    assert np.linalg.norm(pts - focal, axis=1).max() <= radius + 1e-9
    assert eye[2] > focal[2], "the shot should look down"


def test_a_tighter_group_brings_the_camera_in():
    wide = np.stack([[-200, 200], [0, 0], [0, 0]], axis=-1).astype(float)
    tight = np.stack([[-10, 10], [0, 0], [0, 0]], axis=-1).astype(float)
    e_wide, f_wide, _ = frame_group(wide)
    e_tight, f_tight, _ = frame_group(tight)
    assert np.linalg.norm(e_tight - f_tight) < np.linalg.norm(e_wide - f_wide)


def test_a_single_point_does_not_collapse_the_camera_onto_it():
    eye, focal, radius = frame_group(np.zeros((1, 3)))
    assert radius > 0 and np.linalg.norm(eye - focal) > 1.0


def test_smoothing_lags_toward_the_target_and_converges():
    cam = SmoothCamera(eye_tau=0.5, focal_tau=0.25)
    e0, f0 = cam.update([0, 0, 0], [0, 0, 0], 0.05)
    assert np.allclose(e0, 0) and np.allclose(f0, 0), "the first frame snaps"
    e1, _f1 = cam.update([100, 0, 0], [50, 0, 0], 0.05)
    assert 0.0 < e1[0] < 100.0, "the camera should lag, not jump"
    for _ in range(400):
        e, f = cam.update([100, 0, 0], [50, 0, 0], 0.05)
    assert e[0] == pytest.approx(100.0, abs=0.5)
    assert f[0] == pytest.approx(50.0, abs=0.5)
