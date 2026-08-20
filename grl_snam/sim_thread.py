"""Run the swarm on its own thread; publish frames a renderer reads lock-free.

This is the direct answer to "can nav run in a separate thread so the renderer
isn't blocked, while agents keep updating from — and reacting to — the live
scene?" **Yes**, and this module is how.

    ┌──────────── SimThread (owns the Swarm) ─────────────────┐
    │ 1 drain command queue   (live edits, top-of-tick)       │
    │ 2 world.step()          (GIL released inside kernels)    │
    │ 3 publish snapshot      (one atomic reference store)     │
    │ 4 pace to fixed dt      (time.sleep -> releases the GIL) │
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
    """Advance a :class:`~grl_snam.swarm.Swarm` at a fixed rate on a daemon
    thread, publishing an immutable frame each tick.

    Usage::

        sim = SimThread(swarm, hz=60.0)
        sim.start()
        ...                       # render loop, elsewhere:
        frame = sim.buffer.read() # lock-free, never blocks the sim
        sim.send(RetargetGoal(3, (120.0, 40.0)))   # reacted to next tick
        sim.stop()
    """

    def __init__(self, world: Swarm, *, hz: float = 60.0, name: str = "sim"):
        super().__init__(name=name, daemon=True)
        self.world = world
        self._period = 1.0 / float(hz)
        self._run = threading.Event()
        self._run.set()
        self._paused = False
        self._nudge = 0
        self.cmdq: queue.SimpleQueue = queue.SimpleQueue()
        self.buffer = TripleBuffer(world.snapshot())
        # Diagnostics (read-only from other threads; plain attributes).
        self.ticks = 0
        self.behind = 0  # ticks that overran their dt budget
        self._step_ms_ewma = 0.0

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
        next_t = time.perf_counter()
        while self._run.is_set():
            self._drain_commands()  # top-of-tick: live edits sensed THIS tick
            if self._paused and self._nudge <= 0:
                time.sleep(0.002)  # idle without spinning; GIL free for renderer
                next_t = time.perf_counter()
                continue
            if self._nudge > 0:
                self._nudge -= 1

            t0 = time.perf_counter()
            self.world.step()  # GIL released across kernels + torch ATen ops
            step_ms = (time.perf_counter() - t0) * 1000.0
            self._step_ms_ewma = 0.9 * self._step_ms_ewma + 0.1 * step_ms

            curr = self.world.snapshot()
            self.buffer.publish(prev, curr)  # one atomic ref store; no lock
            prev = curr
            self.ticks += 1

            next_t += self._period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)  # release the GIL; renderer runs free
            else:
                self.behind += 1
                next_t = time.perf_counter()  # fell behind -> no spiral of death

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
                self._period = 1.0 / max(1e-3, float(cmd.hz))
            elif isinstance(cmd, Pause):
                self._paused = bool(cmd.paused)
            elif isinstance(cmd, Nudge):
                self._nudge += int(cmd.steps)
            elif isinstance(cmd, Stop):
                self._run.clear()
                return

    @property
    def step_ms(self) -> float:
        """Exponentially-smoothed per-tick compute time (ms)."""
        return self._step_ms_ewma
