"""Framing a group, and keeping the camera out of the city.

A camera placed by geometry alone flies through buildings constantly on a real
map — measured on Austin, 300 of 300 low-angle shots had their line of sight
inside geometry before this existed.
"""

import numpy as np
import pytest

from grl_snam.cinema import (
    SmoothCamera,
    _hidden_count,
    building_height_grid,
    clear_eye,
    clear_shot,
    frame_group,
    sample_height,
    shot_angles,
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


# ── choosing a bearing that can actually see the group ──────────────────────


def test_a_wall_between_camera_and_group_is_walked_around_not_climbed_over():
    """The whole point of the bearing search: step sideways, don't crane up.

    A group standing just east of a tall slab, with the scheduled bearing
    looking through it. :func:`clear_eye` alone can only answer by climbing —
    which flattens the shot into a plan view — so the search should find a
    bearing that simply sees past the slab instead.
    """
    occ = np.zeros((N, N), bool)
    occ[20:44, 30:34] = True  # a north-south slab
    verts = []
    for r in (20, 43):
        for c in (30, 33):
            x = BOUNDS[0] + c / (N - 1) * (BOUNDS[2] - BOUNDS[0])
            y = BOUNDS[1] + r / (N - 1) * (BOUNDS[3] - BOUNDS[1])
            verts += [x, y, 0.0, x, y, 60.0]
    height = building_height_grid(verts, BOUNDS, N, occ=occ)

    pts = [(20.0, -6.0, 1.0), (20.0, 0.0, 1.0), (20.0, 6.0, 1.0)]  # east of the slab
    west = 180.0  # straight through it

    blind, focal, _r = frame_group(pts, elevation_deg=18.0, azimuth_deg=west, fill=0.8)
    lifted = clear_eye(blind, focal, height, BOUNDS, margin_m=10.0, tail_m=20.0)
    assert lifted[2] - blind[2] > 10.0, "the naive shot really is obstructed"

    eye, _f, _r2, used = clear_shot(
        pts, height, BOUNDS, elevation_deg=18.0, azimuth_deg=west, fill=0.8, margin_m=10.0
    )
    assert _hidden_count(eye, np.array(pts), height, BOUNDS) == 0
    # It went around rather than up: the chosen eye is no higher than the
    # unobstructed framing height for its own bearing.
    ref, _f2, _r3 = frame_group(pts, elevation_deg=18.0, azimuth_deg=used, fill=0.8)
    assert eye[2] - ref[2] < 1.0
    assert abs((used - west + 180.0) % 360.0 - 180.0) > 30.0


def test_an_unobstructed_shot_keeps_the_bearing_it_was_given():
    """No obstruction, no deviation — the schedule owns the camera."""
    occ = np.zeros((N, N), bool)
    height = building_height_grid([0.0, 0.0, 0.0], BOUNDS, N, occ=occ)
    pts = [(0.0, -8.0, 1.0), (0.0, 8.0, 1.0)]
    _e, _f, _r, used = clear_shot(
        pts, height, BOUNDS, elevation_deg=30.0, azimuth_deg=215.0, fill=0.8
    )
    assert used == pytest.approx(215.0)


def test_a_group_split_by_a_tower_is_scored_on_the_vehicles_not_the_centroid():
    """The centroid can sit in the open while half the group is behind a wall.

    This is why the search scores visibility of the subjects themselves: a
    bearing chosen on centroid clearance alone passes this case while hiding a
    vehicle.
    """
    occ, verts = _tower(height=50.0)
    height = building_height_grid(verts, BOUNDS, N, occ=occ)
    # Straddle the tower, so the centroid lands on top of it.
    cx = BOUNDS[0] + 32 / (N - 1) * (BOUNDS[2] - BOUNDS[0])
    cy = BOUNDS[1] + 32 / (N - 1) * (BOUNDS[3] - BOUNDS[1])
    pts = [(cx - 34.0, cy, 1.0), (cx + 34.0, cy, 1.0)]
    eye, _f, _r, _u = clear_shot(
        pts, height, BOUNDS, elevation_deg=22.0, azimuth_deg=0.0, fill=0.8, margin_m=10.0
    )
    assert _hidden_count(eye, np.array(pts), height, BOUNDS) == 0


def test_the_tail_exemption_is_a_distance_not_a_fraction():
    """A fractional cut stops checking hundreds of metres short on a wide shot."""
    occ, verts = _tower(height=45.0)
    height = building_height_grid(verts, BOUNDS, N, occ=occ)
    cx = BOUNDS[0] + 32 / (N - 1) * (BOUNDS[2] - BOUNDS[0])
    cy = BOUNDS[1] + 32 / (N - 1) * (BOUNDS[3] - BOUNDS[1])
    focal = np.array([cx + 46.0, cy, 1.0])
    eye = np.array([cx - 600.0, cy, 60.0])  # far out west, tower just short of focal
    # 12% of a 646 m shot is 78 m -- the tower sits inside that and is missed.
    assert clear_eye(eye, focal, height, BOUNDS, margin_m=8.0, near_cut=0.88)[2] == eye[2]
    # As a distance, only the last 20 m are exempt, so the tower is seen.
    assert clear_eye(eye, focal, height, BOUNDS, margin_m=8.0, tail_m=20.0)[2] > eye[2]


def test_shot_opens_high_and_settles_low():
    hi, _a0 = shot_angles(0.0, low_deg=26.0, high_deg=56.0)
    mid, _a1 = shot_angles(0.25, low_deg=26.0, high_deg=56.0)
    lo, az_end = shot_angles(1.0, low_deg=26.0, high_deg=56.0)
    assert hi == pytest.approx(56.0)
    assert lo == pytest.approx(26.0, abs=0.1)
    assert mid < 0.5 * (hi + lo), "the descent front-loads, it is not linear"
    assert az_end > shot_angles(0.0)[1], "bearing drifts monotonically"


# ── handing the camera from one act to the next ─────────────────────────────


def test_a_primed_camera_starts_where_the_last_shot_left_it():
    """Without priming, a second act re-centres on its own first target — a cut.

    An unprimed SmoothCamera snaps to whatever it is first handed, because it
    has no history to lag behind. Primed, it lags from the previous act's
    position instead, which is what turns the join into a move.
    """
    left_eye = np.array([100.0, 50.0, 200.0])
    left_focal = np.array([0.0, 0.0, 0.0])
    new_eye = np.array([-400.0, -400.0, 90.0])
    new_focal = np.array([-300.0, -300.0, 0.0])

    fresh = SmoothCamera(eye_tau=3.2, focal_tau=1.6)
    e_fresh, _f = fresh.update(new_eye, new_focal, 1 / 24)
    assert np.allclose(e_fresh, new_eye), "an unprimed camera snaps — that is the cut"

    primed = SmoothCamera(eye_tau=3.2, focal_tau=1.6)
    primed.prime(left_eye, left_focal)
    e_primed, _f2 = primed.update(new_eye, new_focal, 1 / 24)
    # It has moved off the handover point, but nowhere near the new target.
    moved = float(np.linalg.norm(e_primed - left_eye))
    remaining = float(np.linalg.norm(e_primed - new_eye))
    assert 0.0 < moved < remaining, "a primed camera should ease across, not jump"
    assert moved < 0.05 * float(np.linalg.norm(new_eye - left_eye))


def test_the_handover_state_round_trips():
    cam = SmoothCamera()
    cam.update(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]), 1 / 24)
    st = cam.state(azimuth_deg=241.0)
    nxt = SmoothCamera()
    nxt.prime(st.eye, st.focal)
    assert st.azimuth_deg == pytest.approx(241.0)
    assert np.allclose(nxt.update(st.eye, st.focal, 1 / 24)[0], st.eye)


def test_a_split_schedule_is_continuous_at_the_seam():
    """Act one ending at u and act two starting at u must agree exactly.

    Each act renders a sub-range of ONE schedule. If the second restarted at
    u=0 the elevation would snap back to its opening establishing angle at the
    join, which is the most visible kind of cut.
    """
    seam = 0.654
    end_of_act_one = shot_angles(seam)
    start_of_act_two = shot_angles(seam)
    assert end_of_act_one == start_of_act_two
    # ...and the schedule is still going somewhere: the second act keeps
    # descending and drifting rather than repeating the first act's opening.
    assert shot_angles(1.0)[0] < shot_angles(seam)[0] + 1e-9
    assert shot_angles(1.0)[1] > shot_angles(seam)[1]
