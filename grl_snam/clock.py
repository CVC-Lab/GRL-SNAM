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

This mirrors ``inc/cvc/core/world_clock.h`` in libcvc (transfix/libcvc#181).
There is no pycvc binding for it yet; when one lands, this class is the swap
point and its contract is what the binding must satisfy. The semantics kept
deliberately faithful to the C++:

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
from dataclasses import dataclass

# Ceiling on the world delta one advance() may bank; past this a caller passed
# an absolute timestamp where a delta was meant. Keeps every downstream
# quantity finite (a large-but-finite dt times a large-but-finite scale can
# round to inf, and inf makes fmod NaN, which bricks the clock silently).
_MAX_WORLD_DT = 1.0e15

MODES = ("live", "replay", "paused", "stepping")


@dataclass(frozen=True)
class StepResult:
    """What one :meth:`WorldClock.advance` decided."""

    steps: int = 0
    """Whole quanta to simulate now (at most ``max_steps_per_advance``)."""

    alpha: float = 0.0
    """``[0, 1)`` interpolation into the *next* quantum."""

    dropped_steps: int = 0
    """Quanta discarded by the clamp. Non-zero means this stretch is not a
    faithful simulation of the elapsed wall time — surface it, do not hide
    it."""


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
    ):
        if not (fixed_dt > 0.0) or not math.isfinite(fixed_dt):
            raise ValueError("world clock: fixed_dt must be finite and > 0")
        if not math.isfinite(scale):
            raise ValueError("world clock: scale must be finite")
        if max_steps_per_advance < 1:
            raise ValueError("world clock: max_steps_per_advance must be >= 1")
        if mode not in MODES:
            raise ValueError(f"world clock: mode must be one of {MODES}")

        self.fixed_dt = float(fixed_dt)
        self.max_steps_per_advance = int(max_steps_per_advance)
        self._scale = max(0.0, float(scale))
        self._mode = mode
        self._tick = 0
        self._accumulator = 0.0
        self._dropped = 0

    # ── advancing ───────────────────────────────────────────────────────────
    def advance(self, wall_dt: float) -> StepResult:
        """Advance by a wall-clock delta in seconds."""
        # paused banks nothing (resuming must not release a flood), replay is
        # driven by seek, stepping only by step_once.
        if self._mode in ("paused", "replay", "stepping"):
            return StepResult(alpha=_clamp_alpha(self._accumulator / self.fixed_dt))

        # Sanitized on BOTH sides of the multiply: two individually finite
        # values can round to +inf, and inf must never reach the accumulator.
        self._accumulator += _sanitize_dt(_sanitize_dt(wall_dt) * self._scale)

        # One exact split. fmod is exact, so the phase is preserved bit for
        # bit; repeated subtraction is not.
        remainder = math.fmod(self._accumulator, self.fixed_dt)
        whole = (self._accumulator - remainder) / self.fixed_dt
        demanded = int(round(whole)) if whole > 0.0 else 0

        run = min(demanded, self.max_steps_per_advance)
        dropped = demanded - run

        self._tick += run
        self._accumulator = remainder
        self._dropped += dropped
        return StepResult(
            steps=run,
            alpha=_clamp_alpha(self._accumulator / self.fixed_dt),
            dropped_steps=dropped,
        )

    def step_once(self) -> StepResult:
        """Advance exactly one quantum regardless of wall time. The banked
        remainder is left alone and reported as alpha, so stepping while time
        is banked does not jerk the presentation backwards."""
        self._tick += 1
        return StepResult(steps=1, alpha=_clamp_alpha(self._accumulator / self.fixed_dt))

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
        # a burst of steps on the first live frame.
        if self._mode != mode and mode == "live":
            self._accumulator = 0.0
        self._mode = mode

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

    def reset(self) -> None:
        """Back to tick 0 with an empty bank. Mode, scale and quantum persist —
        this is a rewind, not a factory reset."""
        self._tick = 0
        self._accumulator = 0.0
        self._dropped = 0
