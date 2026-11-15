"""``live --record-trace`` / ``--replay-trace``: a session's inputs, recorded and replayed exactly.

This is R49, the Laboratory's live/replay harness (the engagement docs' Laboratory, §7 and §9,
Phase 6). The trace is libcvc's input trace, ``cvc.input_trace/v1`` (libcvc's
docs/INPUT_TRACE.md, ``pycvc.input_trace_recorder`` / ``input_trace_player``). It holds the
starting state (the world, the starts and goals, the settings, the policies' sha256), every write to
the ``nav``, ``nav_plant``, ``run`` and ``world`` keys stamped by the world clock's tick, nav's own
verbs as events (a Retarget is ``nav.retarget``), and the state's hash after each frame. A run on
the world clock is a function of exactly those, so a replay reproduces the session tick for tick,
and a replay with another policy shows where the trajectory departs.

- **Recording** steps on the app's ``world_clock`` (quantum :data:`QUANTUM`, the navigator every
  ``dt / QUANTUM`` quanta), not live's own pacer, so every input is stamped with the tick that
  consumed it. The UI's writes land between frames, and the Retarget is applied at the start of
  a frame, before the clock moves.
- **The header** describes the first arm's world (``nav_trace_header``), and its ``meta`` holds the
  arms (each one's spec and its ``.cvcnav``'s sha256), the start, the goal, the plant and the
  frame. Every arm's world is that world with its own policy.
- **The hash** is libcvc's ``nav_trace_state_hash`` of each arm's world: the arm's own for one arm,
  so ``cvc trace verify`` replays a one-arm trace by itself; for several, the sha256 of theirs
  together (:func:`session_hash`).
- **Replaying** puts the clock in replay mode. The player writes the recorded sets into the tree
  at their ticks and hands back the events, and every recorded hash is compared. ``sim_world``
  has no save and restore, so seeking backwards rebuilds the worlds and runs again from tick 0.

v1 covers the kinematic plant. libcvc refuses to describe a world on a vehicle plant: Jolt is
deterministic only within one binary.
"""

from __future__ import annotations

import hashlib
import json

try:  # pragma: no cover - depends on the installed bindings
    import pycvc

    HAS_TRACE = all(
        hasattr(pycvc, k)
        for k in ("input_trace_recorder", "input_trace_player", "nav_trace_header",
                  "nav_trace_state_hash", "nav_trace_retarget", "nav_trace_apply_event")
    )  # fmt: skip
except Exception:  # noqa: BLE001 - no pycvc
    pycvc = None
    HAS_TRACE = False

INSTALL = (
    "a pycvc with libcvc's input traces (transfix/libcvc#1168, libcvc 3.7.15+): "
    "pycvc.input_trace_recorder and input_trace_player"
)
#: The world clock's quantum while tracing (s): the suite's, which nav's 0.05 s step divides.
QUANTUM = 0.01
#: The keys a trace records (below the host's prefix), and the ones it leaves out: what sim_world
#: publishes about itself.
WATCH = ("nav", "nav_plant", "run", "world")
IGNORE = ("nav.agents", "nav.settings")


def _under(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def session_hash(fleet) -> str:
    """The state's hash over every arm's world: one arm's own ``nav_trace_state_hash``, else the
    sha256 of theirs, in order."""
    hashes = [pycvc.nav_trace_state_hash(c.sim) for c in fleet]
    if len(hashes) == 1:
        return hashes[0]
    return "sha256:" + hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def agent_id(sim) -> int:
    """The id of a world's one agent (traces name agents by id)."""
    return int(list(pycvc.nav_sim_world_ids(sim))[0])


class Recorder:
    """Records a session of ``fleet`` (``pycvc_gl.cars.Cars``, one per arm) to ``path``.

    Make it before the first step: it pulls the tree's settings into every world, describes the
    first arm's world (its own policy and the arm's) with ``meta`` in the header, and starts the
    app's world clock at :data:`QUANTUM`. Each frame: :meth:`retarget` for a Retarget, then
    :meth:`frame` advances the clock and steps the worlds that fell due; :meth:`finish` at the
    end."""

    def __init__(self, host, fleet, arms, path, *, world_id: str, seed: int, dt: float,
                 meta: dict):  # fmt: skip
        self.host, self.fleet = host, fleet
        self.clock = host.app.world_clock()
        self.clock.set_config(QUANTUM, 1.0, 1000)
        for c in fleet:
            pycvc.nav_sim_world_pull(c.sim, host.app, host.prefix)
        sim = fleet[0].sim
        p = int(pycvc.nav_sim_world_policy_of(sim, 0))
        policies = {0: ""}
        if p:
            policies[p] = arms[0].path
        header = pycvc.nav_trace_header(sim, world_id, seed=int(seed), policies=policies,
                                        prefix=host.prefix, meta=json.dumps(meta))  # fmt: skip
        self.rec = pycvc.input_trace_recorder(
            host.app,
            self.clock,
            [_under(host.prefix, k) for k in WATCH],
            [_under(host.prefix, k) for k in IGNORE],
            True,
            False,
        )  # skip what sim_world publishes about itself (telemetry); stamp writes as they land
        self.rec.start(header, str(path))
        self.follower = pycvc.clock_follower.for_step(self.clock, dt)
        self.path = str(path)

    def retarget(self, gx_n: float, gy_n: float) -> None:
        """Every arm's goal to (gx_n, gy_n) (normalized): nav's ``nav.retarget`` on the first arm's
        world, recorded, and the same on the others."""
        first = self.fleet[0].sim
        pycvc.nav_trace_retarget(first, self.rec, agent_id(first), gx_n, gy_n)
        for c in self.fleet[1:]:
            pycvc.nav_sim_world_retarget(c.sim, 0, gx_n, gy_n)

    def frame(self, wall_dt: float, step) -> int:
        """Advance the clock by ``wall_dt`` (scaled by the sim speed already) and call ``step()``
        (one pull and one tick of every arm) for each navigator step that fell due; then the
        state's hash. Returns the steps run."""
        if wall_dt > 0.0:
            self.clock.advance(wall_dt)
        n = int(self.follower.take(self.clock.tick()))
        for _ in range(n):
            step()
        self.rec.check(session_hash(self.fleet))
        return n

    def finish(self) -> str:
        h = session_hash(self.fleet)
        self.rec.finish(h)
        return h


def read(path):
    """The trace at ``path``, its header (a dict) and its ``meta`` (a dict)."""
    trace = pycvc.input_trace.read(str(path))
    header = json.loads(trace.header.to_json())
    meta = header.get("meta") or {}
    if isinstance(meta, str):
        meta = json.loads(meta or "{}")
    return trace, header, meta


class Player:
    """Plays a trace back on the host's app (``meta``: the header's, the port's own): :meth:`rewind` (the clock in replay mode at the start
    tick, the header's settings written to the tree), then :meth:`quantum` one quantum at a time.
    Each quantum applies the records due (the sets go to the tree; each event goes to
    ``on_event(event, args)``), moves the clock on, runs the navigator steps due (``step()``), and
    compares the recorded hash when there is one (``hash()``)."""

    def __init__(self, host, trace, header, meta=None):
        self.host, self.trace, self.header = host, trace, header
        self.meta = dict(meta or {})
        self.start_tick = int(header.get("start_tick", 0))
        end = trace.end
        last = max((int(trace.records[k].tick) for k in range(trace.record_count())), default=0)
        self.end_tick = int(end.tick) if end.present else max(last, self.start_tick)
        self.clock = host.app.world_clock()
        self.play = pycvc.input_trace_player(host.app, self.clock, trace)
        self.dt = float(header.get("dt", 0.05))
        fixed = float(header.get("fixed_dt", QUANTUM))
        self.period = max(1, round(self.dt / fixed))
        self.phase = int(header.get("phase", 0))
        self.follower = None

    def rewind(self) -> None:
        self.play.rewind()
        self.follower = pycvc.clock_follower(self.period, self.phase, self.clock.tick())

    def done(self) -> bool:
        return bool(self.play.done())

    def tick(self) -> int:
        return int(self.clock.tick())

    def quantum(self, on_event, step, hash_fn) -> None:
        for e in self.play.apply_due():
            on_event(e, json.loads(e.args) if e.args else {})
        self.play.next_quantum()
        for _ in range(int(self.follower.take(self.clock.tick()))):
            step()
        if self.play.expected_hash():
            self.play.verify(hash_fn())

    @property
    def compared(self) -> int:
        """The hashes compared so far."""
        return int(self.play.compared())

    def matched(self) -> bool:
        """Every hash compared matched, the end's among them."""
        return bool(self.play.matched())

    def first_divergence(self) -> int | None:
        """The tick of the first hash that did not match, or None while every hash compared so far
        matched (:meth:`matched` is the whole trace's: the end's hash too)."""
        if not self.play.diverged():
            return None
        return int(self.play.first_divergence())
