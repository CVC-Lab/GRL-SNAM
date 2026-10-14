"""Run the swarm on its own thread; publish frames a renderer reads lock-free.

This is the direct answer to "can nav run in a separate thread so the renderer
isn't blocked, while agents keep updating from — and reacting to — the live
scene?" **Yes**, and this module is how.

    ┌──────────── SimThread (owns the Swarm) ─────────────────┐
    │ 1 drain command queue   (live edits, top-of-tick)       │
    │ 2 world.step()          (GIL released inside kernels)    │
    │ 3 publish snapshot      (one atomic reference store)     │
    │ 4 wait for the clock    (tick wait / sleep: GIL released)│
    └───────────────────────┬─────────────────────────────────┘
                            │ publish (ref swap, no lock)
                   ┌────────▼─────────┐   read latest (ref load)
                   │ TripleBuffer     │◀────── renderer / web / capture threads
                   └──────────────────┘

Three properties make it real concurrency rather than a thread that hogs the
GIL:

* **The publish is a single atomic reference store** of a frozen
  :class:`~grl_snam.swarm.Snapshot`. In CPython a reference store/load is one
  bytecode and the pointed-at object never changes after publish, so a reader
  gets the whole previous frame or the whole next one — never a torn mix. No
  lock is taken on the hot read path. This is the one place we lean on the GIL
  as a correctness primitive, which is exactly what it is for.

* **The sim spends the bulk of each tick with the GIL released** — inside the
  native nav kernels (``build_sdf``/``astar``/``sense``) and inside torch ATen
  ops (``grid_sample``/``matmul``/the bicycle rollout). A C++ renderer (VolRover
  via pycvc_gl) draws on another core during those windows; ``time.sleep`` for
  the pacing releases it too. The only GIL-held slice is the thin snapshot copy
  and the command drain.

* **Live reactivity flows the other way** through a thread-safe command queue
  drained at the *top* of the tick — before sensing — so a moved obstacle or a
  retargeted goal is sensed and reacted to on that very tick.

Time comes from a :class:`~grl_snam.clock.WorldClock`: the thread runs the
steps its clock's quanta make due (following a shared clock the frame loop
drives, or driving a private one at ``hz``), so the steps a run takes never
depend on how its passes were timed.

Fixed-sim / variable-render decoupling: the sim ticks at a fixed dt; a renderer
reads the latest published frame at its own cadence and may interpolate between
the two most-recent frames (``read_pair``). Rendering faster never drives the
vehicles faster; a slow renderer drops frames without ever stalling the sim.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from .clock import ClockFollower, WorldClock, ticks_per_step
from .swarm import Snapshot, Swarm


# ── commands (live-scene edits, applied top-of-tick) ─────────────────────────
@dataclass
class RetargetGoal:
    agent: int
    goal: tuple  # (x, y) world


@dataclass
class MoveObstacle:
    rect: tuple  # (x0, y0, x1, y1) world


@dataclass
class SetRate:
    hz: float


@dataclass
class Pause:
    paused: bool = True


@dataclass
class Nudge:
    steps: int = 1  # advance N ticks while paused


@dataclass
class Stop:
    pass


class TripleBuffer:
    """One writer, many readers, no lock on the read path.

    The writer publishes the ``(prev, curr)`` pair by a single reference store;
    a reader loads it by a single reference load. Both are one CPython bytecode,
    and the frozen :class:`Snapshot` objects never mutate after publish, so a
    reader can only ever observe a complete, self-consistent pair.
    """

    def __init__(self, initial: Snapshot):
        self._pair = (initial, initial)  # (prev, curr)

    def publish(self, prev: Snapshot, curr: Snapshot) -> None:
        self._pair = (prev, curr)  # atomic reference store

    def read(self) -> Snapshot:
        """The latest frame."""
        return self._pair[1]  # atomic reference load

    def read_pair(self) -> tuple[Snapshot, Snapshot]:
        """The two most-recent frames, for sub-tick interpolation."""
        return self._pair


class SimThread(threading.Thread):
    """Advance a :class:`~grl_snam.swarm.Swarm` on a daemon thread, publishing an
    immutable frame each step. Time comes from a :class:`~grl_snam.clock.WorldClock`.

    Usage::

        sim = SimThread(swarm, hz=60.0)          # paces itself (a private clock)
        sim = SimThread(swarm, clock=clock)      # follows a shared clock
        sim.start()
        ...                       # render loop, elsewhere:
        clock.advance(wall_dt)    # (shared clock) the ONE driver, once per frame
        frame = sim.buffer.read() # lock-free, never blocks the sim
        sim.send(RetargetGoal(3, (120.0, 40.0)))   # reacted to next tick
        sim.stop()

    Given a shared ``clock``, the thread FOLLOWS it like ``cvc::nav::sim_thread``:
    one step every ``world.dt / clock.fixed_dt`` quanta, run as the driver
    advances the clock, so speed, pause, single-step and lockstep are the clock's,
    shared with every other simulation on it. ``world.dt`` must be a whole number
    of the clock's quanta. Given only ``hz`` (the original form), the thread
    drives a private clock whose quantum is ``world.dt`` at a speed of
    ``hz * world.dt`` (60 Hz at 0.06 s is 3.6x real time). Either way the steps a
    run takes are fixed by the clock's ticks, not by how this thread's passes
    happen to be timed.
    """

    def __init__(
        self,
        world: Swarm,
        *,
        hz: float = 60.0,
        clock: WorldClock | None = None,
        drive_clock: bool = False,
        phase: int = 0,
        name: str = "sim",
    ):
        super().__init__(name=name, daemon=True)
        self.world = world
        dt = float(world.dt)
        if clock is None:
            # The hz form: a private clock, one quantum per step, driven here. A
            # cap of 8 per advance keeps the old "fell behind -> no spiral of death"
            # behaviour, now with the dropped quanta counted.
            clock = WorldClock(fixed_dt=dt, scale=float(hz) * dt, max_steps_per_advance=8)
            self._private = True
            self._drive = True
        else:
            self._private = False
            self._drive = bool(drive_clock)
        self.clock = clock
        self._follower = ClockFollower(ticks_per_step(dt, clock.fixed_dt), phase, clock.tick())
        self._seen_tick = clock.tick()  # the clock tick this thread has caught up to
        self._run = threading.Event()
        self._run.set()
        self._paused = False
        self.cmdq: queue.SimpleQueue = queue.SimpleQueue()
        self.buffer = TripleBuffer(world.snapshot())
        # Diagnostics (read-only from other threads; plain attributes).
        self.ticks = 0  # steps run
        self.behind = 0  # quanta dropped by the clock this thread drives (0 when following)
        self._step_ms_ewma = 0.0

    def start(self) -> None:
        # Anchor on the caller's thread, before the worker exists: steps that fall
        # due from here on are run (none from before start() is replayed as a
        # burst), and a driver advancing the clock right after start() cannot race
        # the worker's first look at it.
        self._follower.reset(self.clock.tick())
        self._seen_tick = self.clock.tick()
        super().start()

    def wait_caught_up(self, timeout: float = 5.0) -> bool:
        """Block until this thread has run and published every step due up to the
        clock's current tick (True), or ``timeout`` seconds pass (False). A
        capture calls it after driving the clock and before reading, so each
        frame holds exactly the steps its quanta made due."""
        target = self.clock.tick()
        deadline = time.perf_counter() + max(0.0, float(timeout))
        while self._seen_tick < target:
            if not self.is_alive() or time.perf_counter() >= deadline:
                return False
            time.sleep(0.0005)
        return True

    # ── producer API (any thread) ────────────────────────────────────────────
    def send(self, cmd) -> None:
        """Enqueue a live-scene command; applied at the top of the next tick."""
        self.cmdq.put(cmd)

    def stop(self, join: bool = True, timeout: float = 2.0) -> None:
        self.cmdq.put(Stop())
        self._run.clear()
        if join and self.is_alive():
            self.join(timeout)

    # ── the loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        prev = self.world.snapshot()
        k = self._follower.period
        last = time.perf_counter()
        while self._run.is_set():
            self._drain_commands()  # top-of-tick: live edits sensed THIS tick

            if self._drive:
                # The one driver of this clock: advance it by the wall time since
                # the last pass. What runs is decided by the clock, never by this
                # loop's timing, so a slow pass changes how many steps run now,
                # not what they compute.
                now = time.perf_counter()
                r = self.clock.advance(now - last)
                last = now
                self.behind += r.dropped_steps

            now_tick = self.clock.tick()
            due = self._follower.take(now_tick)
            if self._paused:
                due = 0  # this simulation alone is paused: due steps are skipped, not banked
            for _ in range(due):
                if not self._run.is_set():
                    break
                self._drain_commands()
                t0 = time.perf_counter()
                self.world.step()  # GIL released across kernels + torch ATen ops
                step_ms = (time.perf_counter() - t0) * 1000.0
                self._step_ms_ewma = 0.9 * self._step_ms_ewma + 0.1 * step_ms
                curr = self.world.snapshot()
                self.buffer.publish(prev, curr)  # one atomic ref store; no lock
                prev = curr
                self.ticks += 1
            self._seen_tick = now_tick  # everything due up to now_tick is published

            if self._drive:
                # Sleep until about when the next step falls due in wall time
                # (releasing the GIL for the renderer), bounded both ways so a
                # pause or a speed change is noticed promptly.
                wait = 0.002
                scale = self.clock.scale
                if not self._paused and self.clock.mode == "live" and scale > 0.0:
                    since = (self.clock.tick() + k - self._follower.phase) % k
                    world_left = (k - since) * self.clock.fixed_dt - self.clock.pending_seconds()
                    wait = world_left / scale
                time.sleep(min(max(wait, 0.0002), 0.02))
            else:
                # Follow: wake when the driver moves the clock (or after 20 ms, to
                # notice stop() and queued commands).
                self.clock.wait_for_tick_change(now_tick, 0.02)

    # ── command handling ─────────────────────────────────────────────────────
    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.cmdq.get_nowait()
            except queue.Empty:
                return
            if isinstance(cmd, RetargetGoal):
                self.world.retarget(cmd.agent, cmd.goal)
            elif isinstance(cmd, MoveObstacle):
                self.world.add_obstacle(*cmd.rect)
            elif isinstance(cmd, SetRate):
                # The private clock's speed; a shared clock's speed is its own control.
                if self._private:
                    self.clock.set_scale(max(1e-3, float(cmd.hz)) * self.clock.fixed_dt)
            elif isinstance(cmd, Pause):
                if self._private:
                    # Pause the private clock itself, so a Nudge can still step it.
                    self.clock.set_mode("paused" if cmd.paused else "live")
                else:
                    # A shared clock is left alone (pausing everything is the clock's
                    # own control); this simulation's due steps are skipped instead.
                    self._paused = bool(cmd.paused)
            elif isinstance(cmd, Nudge):
                # Steps through the clock, so world time and the steps run agree. On
                # a shared clock the driver owns the steps (world_clock_binding's
                # sim.step), so a nudge is ignored there.
                if self._private:
                    self.clock.advance_ticks(max(0, int(cmd.steps)) * self._follower.period)
            elif isinstance(cmd, Stop):
                self._run.clear()
                return

    @property
    def step_ms(self) -> float:
        """Exponentially-smoothed per-tick compute time (ms)."""
        return self._step_ms_ewma
