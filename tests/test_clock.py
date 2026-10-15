"""grl_snam.clock — the pure-Python twin of cvc::world_clock + clock_follower.

Mirrors libcvc's world_clock_test cases for lockstep, batch, configure,
first_tick, realized speed, the rate-scaled cap, followers and pacing
invariance, and, where pycvc is installed, checks the twin and the native clock
agree tick for tick.
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from grl_snam.clock import ClockFollower, WorldClock, steps_due, ticks_per_step


def test_lockstep_runs_exactly_the_configured_quanta():
    c = WorldClock(fixed_dt=1.0 / 120.0, lockstep_steps=3)
    c.set_mode("lockstep")
    for wall in (0.0, 0.001, 0.5, 10.0, -1.0):
        r = c.advance(wall)
        assert r.steps == 3 and r.dropped_steps == 0 and r.alpha == 0.0
    assert c.tick() == 15
    c.set_lockstep_steps(0)
    assert c.lockstep_steps == 1


def test_entering_lockstep_drops_the_bank():
    c = WorldClock()
    c.advance(0.5 / 120.0)
    assert c.pending_seconds() > 0.0
    c.set_mode("lockstep")
    assert c.pending_seconds() == 0.0


def test_advance_ticks_runs_n_in_any_mode_with_no_cap():
    c = WorldClock()
    for mode in ("live", "paused", "stepping", "replay", "lockstep"):
        c.reset()
        c.set_mode(mode)
        r = c.advance_ticks(1000)
        assert r.steps == 1000 and r.first_tick == 0 and c.tick() == 1000
        assert c.total_dropped() == 0
    assert c.advance_ticks(-5).steps == 0


def test_first_tick_names_the_quanta_each_call_ran():
    c = WorldClock()
    expect = 0
    for d in (0.02, 0.0, 0.05, 0.004, 0.1):
        r = c.advance(d)
        assert r.first_tick == expect
        expect += r.steps
        assert c.tick() == expect


def test_configure_replaces_the_quantum_and_restarts_time():
    c = WorldClock(mode="stepping")
    c.advance_ticks(50)
    c.configure(0.015, lockstep_steps=2)
    assert c.tick() == 0 and c.fixed_dt == 0.015 and c.lockstep_steps == 2
    assert c.mode == "stepping"
    with pytest.raises(ValueError):
        c.configure(0.0)
    assert c.fixed_dt == 0.015, "a rejected config changes nothing"


def test_realized_scale_reports_what_ran():
    c = WorldClock(fixed_dt=0.01, scale=2.0, max_steps_per_advance=1000)
    assert c.realized_scale() == 0.0
    for _ in range(100):
        c.advance(0.01)
    assert abs(c.realized_scale() - 2.0) < 0.05
    d = WorldClock(fixed_dt=0.01, scale=2.0, max_steps_per_advance=1)
    for _ in range(100):
        d.advance(0.01)
    assert abs(d.realized_scale() - 1.0) < 0.05 and d.total_dropped() > 0


def test_a_cap_that_scales_with_rate_keeps_two_times():
    fixed = WorldClock(fixed_dt=0.015, scale=2.0)
    scaled = WorldClock(fixed_dt=0.015, scale=2.0, cap_scales_with_rate=True)
    for _ in range(30):
        fixed.advance(0.07)
        scaled.advance(0.07)
    assert fixed.total_dropped() > 0 and scaled.total_dropped() == 0


def test_ticks_per_step_accepts_whole_multiples_only():
    assert ticks_per_step(0.06, 0.01) == 6
    assert ticks_per_step(0.06, 0.015) == 4
    import struct

    f32 = struct.unpack("f", struct.pack("f", 0.06))[0]
    assert ticks_per_step(f32, 0.015) == 4, "a float32 trained dt"
    for dt, q in ((0.06, 1.0 / 120.0), (0.01, 0.015), (0.0, 0.015), (0.06, -1.0)):
        with pytest.raises(ValueError):
            ticks_per_step(dt, q)


def test_steps_due_matches_brute_force():
    for first in range(20):
        for n in range(20):
            for p in range(1, 7):
                for ph in range(9):
                    brute = sum(1 for t in range(first + 1, first + n + 1) if t % p == ph % p)
                    assert steps_due(first, n, p, ph) == brute


def test_followers_at_different_rates_share_one_clock():
    c = WorldClock(fixed_dt=0.01)
    physics = ClockFollower.for_step(c, 0.01)
    nav = ClockFollower.for_step(c, 0.06)
    ps = ns = 0
    for _ in range(50):
        for f in (0.016, 0.033, 0.0, 0.051, 0.017, 0.2, 0.004):
            c.advance(f)
            ps += physics.take(c)
            ns += nav.take(c)
    assert ps == c.tick() and ns == c.tick() // 6
    assert nav.alpha(12, 0.0) == 0.0 and abs(nav.alpha(15, 0.0) - 0.5) < 1e-12


def test_the_trajectory_is_the_same_under_every_pacing_mode():
    ticks = 600

    def simulate(drive):
        c = WorldClock(fixed_dt=0.01, max_steps_per_advance=1000)
        nav = ClockFollower.for_step(c, 0.06)
        x, trace = 1.0, []
        while c.tick() < ticks:
            drive(c)
            for _ in range(nav.take(c)):
                x = x * 1.0001 + 0.5 * math.sin(x)
                trace.append(x)
        return trace

    def fast(c):
        c.set_scale(2.0)
        c.advance(0.0167)

    def slow(c):
        c.set_scale(0.5)
        c.advance(0.033)

    def lock(c):
        c.set_mode("lockstep")
        c.advance(0.0)

    def step(c):
        c.set_mode("stepping")
        c.advance(0.02)
        c.step_once()

    live = simulate(lambda c: c.advance(0.0167))
    n = ticks // 6
    for trace in (
        simulate(fast),
        simulate(slow),
        simulate(lock),
        simulate(lambda c: c.advance_ticks(37)),
        simulate(step),
    ):
        assert trace[:n] == live[:n]


def test_a_follower_thread_wakes_when_the_driver_moves_the_clock():
    c = WorldClock(fixed_dt=0.01)
    assert c.wait_for_tick_change(0, 0.01) == 0, "no driver: times out at the same tick"
    seen = []
    t = threading.Thread(target=lambda: seen.append(c.wait_for_tick_change(0, 5.0)))
    t.start()
    time.sleep(0.02)
    c.advance_ticks(3)
    t.join(5.0)
    assert seen == [3]
    c.reset()
    assert c.wait_for_tick_change(3, 0.0) == 0, "a rewind counts as a change"


def test_parity_with_the_native_clock():
    pycvc = pytest.importorskip("pycvc")
    if not hasattr(pycvc, "clock_follower"):
        pytest.skip("pycvc predates the shared-clock API")
    py = WorldClock(fixed_dt=0.01, scale=1.5, max_steps_per_advance=8)
    nat = pycvc.world_clock(0.01, 1.5, 8)
    deltas = [0.016, 0.031, 0.008, 0.0, 0.021, 0.099, 0.004, 0.017, 0.012, 0.5, 0.001]
    for rep in range(20):
        for d in deltas:
            a, b = py.advance(d), nat.advance(d)
            assert (a.steps, a.dropped_steps, a.first_tick) == (
                b.steps,
                b.dropped_steps,
                b.first_tick,
            )
            assert abs(a.alpha - b.alpha) < 1e-12
        if rep == 10:
            py.set_mode("lockstep")
            nat.set_mode_name("lockstep")
    assert py.tick() == nat.tick() and py.total_dropped() == nat.total_dropped()
    assert ticks_per_step(0.06, 0.01) == pycvc.ticks_per_step(0.06, 0.01)
