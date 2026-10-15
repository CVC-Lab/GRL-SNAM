"""Fixed-quantum world clock — a Python mirror of ``cvc::world_clock``.

Three clocks are easy to conflate and must not be: **wall** time (what a
steady clock reports; the only real input), **world** time (what the scene
believes it is; this class), and **render** cadence (a policy, not a clock).
The simulation advances in whole ``fixed_dt`` quanta and presentation
interpolates between the last two states by ``alpha``.

That decoupling is what makes the fog demo trustworthy: the recorded trace is
comparable to itself on a fast machine and a slow one, per-second quantities
derive from world dt (so rendering faster never makes the vehicle drive
faster), and the live window and the offscreen capture pace by the *same*
clock against the *same* trace — which is the whole renderer-parity claim.

This mirrors ``inc/cvc/core/world_clock.h`` and ``inc/cvc/core/clock_follower.h``
in libcvc (transfix/libcvc#181, then the one-clock-for-every-simulation work in
``docs/roadmap/WORLD-CLOCK-SYNC-ROADMAP.md``). pycvc wraps the C++ clock as
``pycvc.world_clock`` (with ``pycvc.clock_follower`` and
``pycvc.world_clock_binding``); this is the pure-Python twin with the same
contract, so standalone GRL-SNAM keeps working without the native module, and
``tests/test_clock.py`` checks the two agree wherever pycvc is installed.

One clock, many simulations: a process has ONE driver that advances the clock
once per frame (or a batch loop calling :meth:`WorldClock.advance_ticks`); every
simulation follows it with a :class:`ClockFollower` at its own whole-quantum
period, so they share one notion of time and one speed/pause/step control.
Pacing decides *when* quanta run, never *which* quanta run.

The semantics kept deliberately faithful to the C++:

* ``t()`` is **derived** as ``tick * fixed_dt``, never accumulated — summing a
  float quantum a million times drifts, and a clock that drifts is not a clock.
* the bank is split with one exact :func:`math.fmod`, never subtract-in-a-loop,
  so no quantum is lost to rounding at a large accumulator;
* the spiral-of-death clamp **reports** what it drops rather than silently
  swallowing it;
* ``paused``/``replay``/``stepping`` bank no wall time, so resuming never
  releases a burst; returning to live clears the bank.

NOTE ON THE QUANTUM: the fog demo constructs this with the scenario's
``meta["dt"]`` (0.06 s), *not* the C++ default of 1/120. A trace stamps its
``fixed_dt`` in the manifest for exactly this reason — assuming 120 Hz ticks
from a 0.06 s trace is wrong by 7.2x.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

# Ceiling on the world delta one advance() may bank; past this a caller passed
# an absolute timestamp where a delta was meant. Keeps every downstream
# quantity finite (a large-but-finite dt times a large-but-finite scale can
# round to inf, and inf makes fmod NaN, which bricks the clock silently).
_MAX_WORLD_DT = 1.0e15

# Wall seconds per realized-speed window (the C++ kRealizedWindow).
_REALIZED_WINDOW = 0.5

MODES = ("live", "replay", "paused", "stepping", "lockstep")


@dataclass(frozen=True)
class StepResult:
    """What one :meth:`WorldClock.advance` decided."""

    steps: int = 0
    """Whole quanta to simulate now (at most the cap in live mode)."""

    alpha: float = 0.0
    """``[0, 1)`` interpolation into the *next* quantum."""

    dropped_steps: int = 0
    """Quanta discarded by the clamp. Non-zero means this stretch is not a
    faithful simulation of the elapsed wall time — surface it, do not hide
    it."""

    first_tick: int = 0
    """The tick before this call: the quanta to run are ``first_tick + 1``
    through ``first_tick + steps``."""


def _sanitize_dt(dt: float) -> float:
    """Non-finite/negative deltas become 0; absurd ones clamp. Time never runs
    backwards here."""
    if not math.isfinite(dt) or dt < 0.0:
        return 0.0
    return min(dt, _MAX_WORLD_DT)


def _clamp_alpha(a: float) -> float:
    """alpha must land in ``[0, 1)`` whatever the accumulator holds: exactly
    1.0 would present a frame belonging to a state we have not computed, and a
    NaN would propagate into the scene."""
    if not a >= 0.0:  # False for NaN
        return 0.0
    if a >= 1.0:
        return math.nextafter(1.0, 0.0)
    return a


class WorldClock:
    """The authoritative notion of simulation time for a fog-demo playback."""

    def __init__(
        self,
        fixed_dt: float = 1.0 / 120.0,
        *,
        scale: float = 1.0,
        max_steps_per_advance: int = 8,
        mode: str = "live",
        lockstep_steps: int = 1,
        cap_scales_with_rate: bool = False,
    ):
        if mode not in MODES:
            raise ValueError(f"world clock: mode must be one of {MODES}")
        self._mode = mode
        # Notified whenever the tick moves, so a follower thread can sleep until
        # the driver advances the clock (wait_for_tick_change).
        self._tick_changed = threading.Condition()
        self._apply_config(
            fixed_dt, scale, max_steps_per_advance, lockstep_steps, cap_scales_with_rate
        )

    def _apply_config(
        self,
        fixed_dt: float,
        scale: float,
        max_steps_per_advance: int,
        lockstep_steps: int,
        cap_scales_with_rate: bool,
    ) -> None:
        if not (fixed_dt > 0.0) or not math.isfinite(fixed_dt):
            raise ValueError("world clock: fixed_dt must be finite and > 0")
        if not math.isfinite(scale):
            raise ValueError("world clock: scale must be finite")
        if max_steps_per_advance < 1:
            raise ValueError("world clock: max_steps_per_advance must be >= 1")
        if lockstep_steps < 1:
            raise ValueError("world clock: lockstep_steps must be >= 1")
        self.fixed_dt = float(fixed_dt)
        self.max_steps_per_advance = int(max_steps_per_advance)
        self._scale = max(0.0, float(scale))
        self._lockstep_steps = int(lockstep_steps)
        self._cap_scales_with_rate = bool(cap_scales_with_rate)
        self._tick = 0
        self._accumulator = 0.0
        self._dropped = 0
        self._clear_measurement()
        self._ticked()

    def configure(
        self,
        fixed_dt: float,
        *,
        scale: float = 1.0,
        max_steps_per_advance: int = 8,
        lockstep_steps: int = 1,
        cap_scales_with_rate: bool = False,
    ) -> None:
        """Replace the configuration and return to tick 0 (a new quantum changes
        what every tick means). The mode is kept. This is how a host picks a
        quantum that every simulation it runs divides into whole steps, before
        any of them starts. Raises on an invalid config, leaving the clock
        unchanged."""
        # Validate on a scratch clock first so a bad config changes nothing.
        WorldClock(
            fixed_dt,
            scale=scale,
            max_steps_per_advance=max_steps_per_advance,
            lockstep_steps=lockstep_steps,
            cap_scales_with_rate=cap_scales_with_rate,
        )
        self._apply_config(
            fixed_dt, scale, max_steps_per_advance, lockstep_steps, cap_scales_with_rate
        )

    def _clear_measurement(self) -> None:
        self._win_wall = 0.0
        self._win_world = 0.0
        self._realized = 0.0

    def _note_wall(self, wall_dt: float) -> None:
        self._win_wall += wall_dt
        if self._win_wall >= _REALIZED_WINDOW:
            self._realized = self._win_world / self._win_wall
            self._win_wall = 0.0
            self._win_world = 0.0

    def _ticked(self) -> None:
        """Wake every thread in :meth:`wait_for_tick_change`."""
        with self._tick_changed:
            self._tick_changed.notify_all()

    def _run(self, n: int) -> StepResult:
        """Run n quanta now: no cap, no bank."""
        first = self._tick
        self._tick += n
        self._win_world += n * self.fixed_dt
        if n:
            self._ticked()
        return StepResult(
            steps=n, alpha=_clamp_alpha(self._accumulator / self.fixed_dt), first_tick=first
        )

    # ── advancing ───────────────────────────────────────────────────────────
    def advance(self, wall_dt: float) -> StepResult:
        """Advance by a wall-clock delta in seconds."""
        wall = _sanitize_dt(wall_dt)
        if self._mode in ("paused", "replay", "stepping"):
            # paused banks nothing (resuming must not release a flood), replay
            # is driven by seek, stepping only by step_once.
            r = StepResult(
                alpha=_clamp_alpha(self._accumulator / self.fixed_dt), first_tick=self._tick
            )
        elif self._mode == "lockstep":
            # Frame by frame: exactly lockstep_steps quanta, whatever the wall says.
            r = self._run(self._lockstep_steps)
        else:
            r = self._advance_live(_sanitize_dt(wall * self._scale))
        # After the steps are counted, so a window that closes now includes them.
        self._note_wall(wall)
        return r

    def _advance_live(self, world_dt: float) -> StepResult:
        first = self._tick
        # Sanitized on BOTH sides of the multiply (by the caller): two
        # individually finite values can round to +inf, and inf must never
        # reach the accumulator.
        self._accumulator += world_dt

        # One exact split. fmod is exact, so the phase is preserved bit for
        # bit; repeated subtraction is not.
        remainder = math.fmod(self._accumulator, self.fixed_dt)
        whole = (self._accumulator - remainder) / self.fixed_dt
        demanded = int(round(whole)) if whole > 0.0 else 0

        cap = self.max_steps_per_advance
        if self._cap_scales_with_rate and self._scale > 1.0:
            cap = int(min(cap * min(math.ceil(self._scale), 1.0e6), 2147483647.0))
        run = min(demanded, cap)
        dropped = demanded - run

        self._tick += run
        self._accumulator = remainder
        self._dropped += dropped
        self._win_world += run * self.fixed_dt
        if run:
            self._ticked()
        return StepResult(
            steps=run,
            alpha=_clamp_alpha(self._accumulator / self.fixed_dt),
            dropped_steps=dropped,
            first_tick=first,
        )

    def step_once(self) -> StepResult:
        """Advance exactly one quantum regardless of wall time. The banked
        remainder is left alone and reported as alpha, so stepping while time
        is banked does not jerk the presentation backwards."""
        return self._run(1)

    def advance_ticks(self, n: int) -> StepResult:
        """Advance exactly n quanta now, in any mode, with no cap: the batch
        primitive. A headless harness or an episode generator runs as fast as
        the work allows and books exactly the time it simulated."""
        return self._run(max(0, int(n)))

    # ── time ────────────────────────────────────────────────────────────────
    def tick(self) -> int:
        """The authoritative integer; world seconds derive from it."""
        return self._tick

    def t(self) -> float:
        """World seconds — ``tick * fixed_dt``, never accumulated."""
        return self._tick * self.fixed_dt

    def seek_tick(self, tick: int) -> None:
        """Reposition in replay. Does not touch the banked remainder."""
        if tick < 0:
            raise ValueError("world clock: tick must be >= 0")
        self._tick = int(tick)
        self._ticked()

    def wait_for_tick_change(self, tick: int, timeout: float) -> int:
        """Block until the tick differs from ``tick`` or ``timeout`` seconds
        pass, and return the current tick. A follower thread waits here for the
        driver to move the clock, instead of polling."""
        with self._tick_changed:
            self._tick_changed.wait_for(lambda: self._tick != tick, timeout=max(0.0, timeout))
            return self._tick

    def seek_time(self, t: float) -> None:
        """Seek to the quantum containing world time ``t``."""
        if not math.isfinite(t) or t < 0.0:
            raise ValueError("world clock: seek time must be finite and >= 0")
        self.seek_tick(int(t / self.fixed_dt))

    # ── mode and rate ───────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"world clock: mode must be one of {MODES}")
        # Leaving a non-live mode drops banked time rather than releasing it as
        # a burst of steps on the first live frame. Lockstep never banks, so
        # entering it drops the bank too (a stale remainder would show up as a
        # non-zero alpha on every lockstep frame).
        if self._mode != mode and mode in ("live", "lockstep"):
            self._accumulator = 0.0
        self._mode = mode

    @property
    def lockstep_steps(self) -> int:
        """Quanta per :meth:`advance` in lockstep mode."""
        return self._lockstep_steps

    def set_lockstep_steps(self, n: int) -> None:
        self._lockstep_steps = max(1, int(n))

    @property
    def cap_scales_with_rate(self) -> bool:
        """When set, the live cap is ``max_steps_per_advance * ceil(scale)`` for
        ``scale > 1``, so asking for 2x is not quietly capped at 1x's quanta."""
        return self._cap_scales_with_rate

    def set_cap_scales_with_rate(self, on: bool) -> None:
        self._cap_scales_with_rate = bool(on)

    @property
    def scale(self) -> float:
        """World seconds per wall second. 0 pauses, >1 fast-forwards."""
        return self._scale

    def set_scale(self, s: float) -> None:
        # The setter clamps rather than throwing: a speed slider must not
        # crash a renderer. The constructor is stricter (a programming error).
        self._scale = 0.0 if (not math.isfinite(s) or s < 0.0) else float(s)

    # ── observability ───────────────────────────────────────────────────────
    def total_dropped(self) -> int:
        """Total quanta ever discarded by the clamp."""
        return self._dropped

    def pending_seconds(self) -> float:
        """Banked fraction of a quantum not yet simulated, in ``[0, fixed_dt)``."""
        return self._accumulator

    def realized_scale(self) -> float:
        """World seconds actually simulated per wall second over the last
        completed window of at least half a wall second of :meth:`advance`
        calls. The speed achieved, not the speed requested; 0 until the first
        window completes."""
        return self._realized

    def reset(self) -> None:
        """Back to tick 0 with an empty bank. Mode, scale and quantum persist —
        this is a rewind, not a factory reset."""
        self._tick = 0
        self._accumulator = 0.0
        self._dropped = 0
        self._clear_measurement()
        self._ticked()


# ── following one clock at several rates (cvc/core/clock_follower.h) ──────────


def ticks_per_step(dt: float, fixed_dt: float) -> int:
    """Quanta per step for a simulation of step ``dt`` on a clock of quantum
    ``fixed_dt``. Raises unless ``dt`` is a positive whole multiple of
    ``fixed_dt``: a step of 7.2 quanta cannot be scheduled without drifting,
    which is exactly the mismatch that books one time and integrates another.
    The tolerance (relative 1e-6) admits a float32 trained dt."""
    if not (dt > 0.0 and math.isfinite(dt) and fixed_dt > 0.0 and math.isfinite(fixed_dt)):
        raise ValueError("ticks_per_step: dt and fixed_dt must be finite and > 0")
    k = round(dt / fixed_dt)
    if k < 1 or abs(k * fixed_dt - dt) > 1.0e-6 * dt:
        raise ValueError(
            "ticks_per_step: the step is not a whole number of clock quanta; configure "
            "the world clock with a fixed_dt that divides every simulation's step"
        )
    return int(k)


def steps_due(first_tick: int, steps: int, period: int, phase: int = 0) -> int:
    """How many steps of a simulation with this period and phase fall among
    the quanta ``first_tick + 1 .. first_tick + steps``."""
    if period <= 0 or steps <= 0:
        return 0
    r = phase % period

    def upto(x: int) -> int:  # count of t in [0, x] with t % period == r
        return 0 if x < r else (x - r) // period + 1

    return upto(first_tick + steps) - upto(first_tick)


class ClockFollower:
    """One simulation's cursor on a shared clock. :meth:`take` returns how many
    of its steps fell due since the last call; the caller runs that many. A
    follower never advances the clock, so any number of them can share one."""

    def __init__(self, period: int = 1, phase: int = 0, start_tick: int = 0):
        self.period = max(1, int(period))
        self.phase = int(phase) % self.period
        self._last = int(start_tick)

    @classmethod
    def for_step(cls, clock: WorldClock, dt: float, phase: int = 0) -> ClockFollower:
        """A follower of a simulation whose step is ``dt`` seconds on ``clock``,
        starting at the clock's current tick."""
        return cls(ticks_per_step(dt, clock.fixed_dt), phase, clock.tick())

    def take(self, now) -> int:
        """Steps due since the last call. ``now`` is a tick or a clock (anything
        with ``tick()``). A clock that went backwards re-anchors and yields 0."""
        now_tick = int(now.tick()) if hasattr(now, "tick") else int(now)
        if now_tick < self._last:
            self._last = now_tick
            return 0
        n = steps_due(self._last, now_tick - self._last, self.period, self.phase)
        self._last = now_tick
        return n

    def due_at(self, tick: int) -> bool:
        return tick % self.period == self.phase

    def alpha(self, tick: int, clock_alpha: float) -> float:
        """Presentation fraction into this simulation's NEXT step, from the
        clock's tick and the alpha :meth:`WorldClock.advance` returned."""
        since = (tick + self.period - self.phase) % self.period
        return _clamp_alpha((since + clock_alpha) / self.period)

    @property
    def last_tick(self) -> int:
        return self._last

    def reset(self, tick: int) -> None:
        self._last = int(tick)
