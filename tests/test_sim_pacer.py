"""SimPacer: the wall-clock -> fixed-dt pacing that keeps the live demos running
at a fixed sim rate on any display rate (instead of a fixed count of nav steps
per drawn frame, which sped up with the frame rate)."""

from grl_snam.demos._common import SimPacer


def _run_one_wall_second(fps, sim_dt, speed=1.0):
    pacer = SimPacer()
    ticks = 0
    for _ in range(fps):  # `fps` frames == one wall second
        ticks += pacer.ticks(1.0 / fps, sim_dt, speed)
    return ticks


def test_tick_rate_is_framerate_independent():
    sim_dt = 0.06  # one nav.step() advances meta["dt"] seconds of sim time
    counts = {fps: _run_one_wall_second(fps, sim_dt) for fps in (30, 60, 144, 1000)}
    # Every display rate yields the same number of sim ticks per wall second.
    assert len(set(counts.values())) == 1, counts
    # And that count is real time: ticks * sim_dt ≈ 1 wall second (within one tick).
    (ticks,) = set(counts.values())
    assert abs(ticks * sim_dt - 1.0) <= sim_dt


def test_speed_scales_the_rate():
    sim_dt = 0.06
    # One wall second at playback `speed` advances ~speed sim seconds (within one
    # tick of integer quantization), so a faster speed covers proportionally more.
    base = _run_one_wall_second(240, sim_dt, speed=1.0)
    fast = _run_one_wall_second(240, sim_dt, speed=3.0)
    assert abs(base * sim_dt - 1.0) <= sim_dt
    assert abs(fast * sim_dt - 3.0) <= sim_dt


def test_a_stall_drops_time_instead_of_bursting():
    # A long hitch must not fast-forward the sim by a huge tick burst.
    pacer = SimPacer()
    assert pacer.ticks(5.0, 0.06, 1.0, cap=8) == 8
    # The backlog was dropped, so the next normal frame runs 0 ticks (no catch-up).
    assert pacer.ticks(1.0 / 60.0, 0.06, 1.0, cap=8) == 0


def test_nonpositive_dt_is_a_noop():
    pacer = SimPacer()
    assert pacer.ticks(0.0, 0.06) == 0
    assert pacer.ticks(1.0 / 60.0, 0.0) == 0
