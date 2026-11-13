"""The Ackermann allocation (``grl_snam.coph.allocation``, NAV-COPH-ROADMAP §3.3.2).

The golden (G11, ``tests/test_coph_golden.py``) pins its values; these check what it means.
"""

import math

import pytest

from grl_snam.coph.allocation import SPEED_FLOOR, allocate

ROBORACER = dict(wheelbase=0.33, steer_max=0.4189, accel_max=3.0)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


@pytest.mark.parametrize("speed", [1.0, -1.0, 2.5])
@pytest.mark.parametrize("heading", [0.0, 0.7, -2.3])
def test_a_force_within_the_limits_is_taken_whole(speed, heading):
    """Above the floor, within the lock and the clip, the car takes u: e = 0, in reverse too."""
    h = (math.cos(heading), math.sin(heading))
    n = (-h[1], h[0])
    u = (0.5 * h[0] + 0.3 * n[0], 0.5 * h[1] + 0.3 * n[1])
    a = allocate(u, heading, speed, **ROBORACER)
    assert abs(a.steer) < ROBORACER["steer_max"]
    assert math.hypot(*a.unallocated) < 1e-12
    assert a.accel == pytest.approx(0.5) and a.kappa == pytest.approx(0.3 / speed**2)


def test_at_rest_the_lateral_force_is_unallocated():
    heading = 0.7
    n = (-math.sin(heading), math.cos(heading))
    a = allocate((1.0 * n[0], 1.0 * n[1]), heading, 0.0, **ROBORACER)
    assert a.kappa == pytest.approx(1.0 / SPEED_FLOOR**2)
    assert a.steer == ROBORACER["steer_max"]  # turned to the lock, toward the force
    assert math.hypot(*a.used) < 1e-12 and _dot(a.unallocated, n) == pytest.approx(1.0)


def test_the_lock_and_the_clip_hold_and_e_books_the_rest():
    u = (5.0, -4.0)  # heading 0: 5 along, 4 to the right
    a = allocate(u, 0.0, 1.0, **ROBORACER)
    assert a.accel == ROBORACER["accel_max"] and a.steer == -ROBORACER["steer_max"]
    lateral = math.tan(-ROBORACER["steer_max"]) / ROBORACER["wheelbase"]
    assert a.used == pytest.approx((3.0, lateral))
    assert a.unallocated == pytest.approx((2.0, -4.0 - lateral))
