"""CoPH-Stagewise's crossing and hazard episodes in 3-D, communication off beside CoPH (demo 2a).

    python -m grl_snam.ariadne coph crossing                  # live, seed 76000, both arms
    python -m grl_snam.ariadne coph hazard --scene severe     # live, none beside field
    python -m grl_snam.ariadne coph crossing --from runs/crossing   # replay a grl-snam coph run
    python -m grl_snam.ariadne coph crossing --headless --steps 200 --log run.npz

The episodes are the vendored demos' own (``grl_snam.coph``, roadmap §3.2). A live run calls the
runner exactly as ``grl-snam coph`` does, one worker process per arm, and keeps the episodes with
``--out`` in ``grl-snam coph``'s layout. ``--from`` replays such a run's episodes. Either way the
viewer reads the episode through :mod:`grl_snam.coph.feed`'s messages and changes nothing, so its
outcomes are T1's (``tests/test_coph_feed.py``).

The two arms run on the same seed and stand side by side, communication off on the left:
- **the floor:** the surveyed corridors, with the risk field as a heat map (the truth, or one
  agent's own ``r``: the View combo), each agent's three kept stages (the newest brightest) and
  its sensor disc;
- **the doors** (crossing): full-width slabs, drawn once an agent of that arm has sensed or been
  told about them;
- **the agents:** RoboRacer-sized cars (0.58 m) at the point's position, heading along its
  velocity, for display only; the Humvee model scaled down when it is installed, else a box;
- **packets:** an arc from sender to receiver at delivery, green when the receiver's gate accepted
  the update that took it in, red when not, as wide as its bytes;
- **the panels:** per agent, its status, version, ``pi`` (risk, barrier), damping, accepts and
  rejects with the last reason, payload bytes sent and received, and its energy ledger (work,
  heat, kinetic change, shaping jumps, the balance residual);
- **the learning panels** (:class:`Learning`, on pycvc_gl's ``chart:``, libcvc 3.7.14): for one
  agent, each arm's held-out field loss before and after each update, with the gate's
  threshold, and a chosen statistic (the field against the truth, damping, ``pi``, the plant's
  held-out loss, replay cost). An older pycvc_gl hides them.

The headline shows the reference's established result with its seed counts, this run's own number
once both arms finish, and what the reference has not established (replay, receiver field
fitting, adaptive-rate sharing), which runs here all the same.

Headless (``--headless``), the run goes to its end with nothing drawn; ``--log run.npz`` writes the
cars template's log shape (``pycvc_gl.cars.CarsLog``) plus the ledger columns
(:func:`grl_snam.coph.timeline.log_arrays`). The exit code is 0, 1 when a run failed, and 3 when
fewer agents arrived than ``--min-goals``.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from grl_snam.ariadne import _host
from grl_snam.coph import feed
from grl_snam.coph.records import Config
from grl_snam.coph.timeline import Floor, Timeline, comparison, learning, log_arrays

TITLES = {
    "crossing": "Crossing traverses: what one agent has seen saves the other's route",
    "hazard": "Hazard: a shared graded field, priced by each agent's own risk weight",
}
REFERENCE = {
    "crossing": "Reference (CoPH-Sense c1f1ff3, 2 seeds x 4 arms): 16/16 goals, 0 collisions; "
    "team saving 27.1 m both ways, 13.4 m one way.",
    "hazard": "Reference (seed 79000, 4 scenes x 8 arms): the carrier's worst-case excess cost is "
    "4.6 with the shared field and 68.1 with none; in 'severe', 0.0 and 68.1.",
}
UNESTABLISHED = (
    "Not established by the reference, though they run here: any gain from replay, from "
    "receiver field fitting, or from adaptive-rate sharing."
)
ROLES = {"crossing": ("0 west", "1 east"), "hazard": ("scout .25", "carrier 6")}
ARM_LABELS = {
    "disabled": "comms off",
    "none": "comms off",
    "reciprocal_fixed": "CoPH both ways",
    "one_way_01": "CoPH 0 to 1",
    "one_way_10": "CoPH 1 to 0",
    "field": "shared field",
    "raw": "raw readings",
    "no_learning": "no rx fitting",
    "decision": "shared decision",
}
#: The arms' names in the Learning charts' legends and markers.
ARM_SHORT = {"disabled": "off", "none": "off", "reciprocal_fixed": "CoPH",
             "one_way_01": "CoPH 0>1", "one_way_10": "CoPH 1>0", "field": "field",
             "raw": "raw", "no_learning": "no rx fit", "decision": "decision"}  # fmt: skip
COLORS = ((0.93, 0.45, 0.13), (0.16, 0.47, 0.85))  # agent 0, agent 1
ACCEPT, REJECT = (0.15, 0.75, 0.25), (0.86, 0.16, 0.16)
GAP = 3.0  # metres between the two arms' floors
CAR = (0.58, 0.30, 0.25)  # a RoboRacer's length, width and height (m)
ARC_SECONDS = 2.5  # how long a delivered packet's arc stays up (simulated time)
ARCS = 12  # arc nodes per arm


def _args(p):
    p.add_argument("demo", choices=("crossing", "hazard"), help="which vendored demo")
    p.add_argument("--scene", default="severe", choices=feed.HAZARD_SCENES, help="hazard's scene")
    p.add_argument(
        "--arms",
        default=None,
        help="the two arms, comma-separated, left then right (default: communication off, then "
        "crossing's reciprocal_fixed or hazard's field)",
    )
    p.add_argument("--seed", type=int, default=None, help="default 76000 (crossing), 79000")
    p.add_argument(
        "--steps", type=int, default=None, help="the horizon (default the demo's: 2500, 1800)"
    )
    p.add_argument(
        "--from",
        dest="source",
        default=None,
        help="replay the episodes of a `grl-snam coph` run's --out directory instead",
    )
    p.add_argument(
        "--out", default=None, help="keep a live run's episodes here, in grl-snam coph's layout"
    )
    p.add_argument("--log", default=None, help="write the run's .npz log when it ends")
    p.add_argument("--at", type=float, default=0.0, help="start the play head here (s)")
    p.add_argument(
        "--heat",
        default="truth",
        choices=("truth", "agent0", "agent1"),
        help="the floor's heat map to start with: the true risk field or an agent's own r",
    )
    p.add_argument("--min-goals", type=int, default=0, help="exit 3 below this many arrivals")


def label(arm):
    return ARM_LABELS.get(arm, arm)


# ── where the episodes come from ───────────────────────────────────────────────────────────────


class Source:
    """The two arms' timelines: filled by worker processes as a live run goes, or from a
    finished run's episode files."""

    def __init__(self, demo, timelines, *, queue=None, procs=(), origin="", where=None):
        self.demo, self.timelines, self.queue, self.procs = demo, timelines, queue, list(procs)
        self.origin, self.where = origin, where
        self.done = {tl.arm for tl in timelines} if queue is None else set()
        self.error = None

    @classmethod
    def replay(cls, demo, arms, seed, scene, root):
        from grl_snam.coph.evaluation import load

        where = feed.run_dir(demo, root, scene) / "episodes"
        timelines = []
        for arm in arms:
            path = where / f"{seed}_{arm}.json.gz"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing: run `grl-snam coph {demo} --out {root}` first (it runs "
                    "every arm), or drop --from to run these two live"
                )
            timelines.append(Timeline.from_episode(demo, load(path), label=label(arm)))
        return cls(demo, timelines, origin="replay", where=str(where))

    @classmethod
    def live(cls, demo, arms, seed, steps, scene, out=None):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        timelines, procs = [], []
        for arm in arms:
            spec = feed.job(demo, arm, seed, steps, scene)[0]
            timelines.append(Timeline(demo, spec, arm, label=label(arm)))
            where = str(feed.run_dir(demo, out, scene)) if out else None
            p = ctx.Process(
                target=feed.serve,
                args=(queue, demo, arm, seed, steps, scene, where),
                daemon=True,
            )
            p.start()
            procs.append(p)
        return cls(demo, timelines, queue=queue, procs=procs, origin=f"live, {steps} steps",
                   where=str(feed.run_dir(demo, out, scene)) if out else None)  # fmt: skip

    @property
    def live_running(self):
        return self.queue is not None and len(self.done) < len(self.timelines)

    @property
    def finished(self):
        return all(tl.finished for tl in self.timelines)

    def poll(self, budget=400):
        """Take in what the workers sent since the last frame (at most ``budget`` batches);
        the number taken."""
        if self.queue is None:
            return 0
        import queue as queue_

        by_arm = {tl.arm: tl for tl in self.timelines}
        n = 0
        for n in range(1, budget + 1):
            try:
                kind, arm, payload = self.queue.get_nowait()
            except queue_.Empty:
                return n - 1
            if kind == "batch":
                by_arm[arm].ingest(payload)
            elif kind == "done":
                self.done.add(arm)
            else:
                self.error = f"the {arm} arm failed:\n{payload}"
                self.done.add(arm)
        return n

    def head(self):
        """The newest time both arms have reached while a live run goes; the end once done."""
        running = [tl.duration for tl in self.timelines if not tl.finished]
        if running and self.live_running:
            return min(running)
        return max(tl.duration for tl in self.timelines)

    def close(self):
        for p in self.procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)


# ── the transport ──────────────────────────────────────────────────────────────────────────────


class Transport:
    """The play head over a run whose end may still be growing (a live run): with ``play.live``
    on, it sits on the newest boundary; otherwise it plays at ``play.speed``, and dragging the
    position slider scrubs (and leaves Live)."""

    def __init__(self, host, title):
        self.host, self.t, self._pos = host, 0.0, 0.0
        host.set("play.title", title)

    def restart(self):
        self.t = 0.0
        self.host.set("play.playing", True)
        self.host.set("play.live", False)

    def seek(self, t):
        """Put the play head at ``t`` (a click on a learning chart), leaving Live."""
        self.t = max(float(t), 0.0)
        self.host.set("play.live", False)

    def tick(self, dt, head, live):
        host, duration = self.host, max(head, 1e-6)
        pos = host.get_float("play.pos", 0.0)
        if abs(pos - self._pos) > 1e-6:  # dragged: the slider wins
            self.t = min(max(pos, 0.0), 1.0) * duration
            host.set("play.live", False)
        elif live and host.get_bool("play.live"):
            self.t = head
        elif host.get_bool("play.playing"):
            self.t += dt * max(host.get_float("play.speed", 1.0), 0.0)
            if self.t > duration:
                if host.get_bool("play.loop") and not live:
                    self.t = 0.0
                else:
                    self.t = duration
                    if not live:
                        host.set("play.playing", False)
        self.t = min(max(self.t, 0.0), duration)
        self._pos = self.t / duration
        host.set("play.pos", round(self._pos, 6))
        self._pos = host.get_float("play.pos", self._pos)  # as the state text reads it back
        host.set("play.time", f"{self.t:6.1f} / {head:.1f} s" + ("  (live)" if live else ""))
        return self.t


# ── the readouts ───────────────────────────────────────────────────────────────────────────────


def status(state):
    if not state.active:
        return "waiting"
    if state.outcome is not None:
        o = state.outcome
        return "arrived" if o["success"] else ("collided" if o["collision"] else "timed out")
    return "driving"


def rows(demo, timelines, t):
    """The Agents table's rows at time ``t``: one per agent per arm, its learner, its packet
    payload bytes and its energy ledger."""
    out = []
    for tl in timelines:
        for i in range(2):
            s = tl.state(i, t)
            g = s.ledger
            out.append(
                dict(
                    arm=tl.label,
                    agent=ROLES[demo][i],
                    status=status(s),
                    version=str(s.version),
                    pi_risk=f"{s.policy[0]:.3f}",
                    pi_barrier=f"{s.policy[1]:.3f}",
                    damping=f"{s.damping:.3f}",
                    decisions=f"{s.accepts}/{s.rejects}",
                    reason=s.last_reason or "-",
                    sent=str(s.bytes_sent),
                    received=str(s.bytes_received),
                    work=f"{g['work']:.3f}",
                    heat=f"{g['heat']:.3f}",
                    dke=f"{g['kinetic_change']:.3f}",
                    shaping=f"{g['shaping']:.3f}",
                    residual=f"{g['residual']:.1e}",
                )
            )
    return out


def result_line(demo, timelines, seed, scene):
    c = comparison(demo, *timelines)
    if c is None:
        return "This run: its own number appears once both arms finish."
    left, right = (tl.label for tl in timelines)
    if demo == "crossing":
        s = c["saving"]
        return (
            f"This run (seed {seed}): {right} saves agent 0 {s[0]:.1f} m and agent 1 "
            f"{s[1]:.1f} m against {left} (team {s[0] + s[1]:.1f} m)."
        )
    a, b = c["carrier_cost"]
    return (
        f"This run ('{scene}', seed {seed}): the carrier's cost (time + risk) is {a:.1f} with "
        f"{left} and {b:.1f} with {right}."
    )


def goals(timelines):
    return sum(bool(a.outcome and a.outcome["success"]) for tl in timelines for a in tl.agents)


def collisions(timelines):
    return sum(bool(a.outcome and a.outcome["collision"]) for tl in timelines for a in tl.agents)


def publish(host, demo, source, t, seed, scene):
    host.set_table("agents", rows(demo, source.timelines, t))
    left, right = (tl.label for tl in source.timelines)
    host.set("coph.headline", f"{TITLES[demo]}. Left: {left}; right: {right} (same seed).")
    host.set("coph.reference", REFERENCE[demo])
    host.set("coph.result", result_line(demo, source.timelines, seed, scene))
    if source.finished:
        state = (
            f"finished: {goals(source.timelines)}/4 goals, "
            f"{collisions(source.timelines)} collisions ({source.origin})"
        )
    else:
        heads = ", ".join(f"{tl.label} {tl.duration:.1f} s" for tl in source.timelines)
        state = f"running: {heads} ({source.origin})"
    host.set("coph.status", state)
    host.set("coph.unestablished", UNESTABLISHED)


# ── the learning panels ────────────────────────────────────────────────────────────────────────

SLACK = Config().validation_slack
#: The Learning window's second chart (``learn.show``): its caption, whether its y is log, and
#: the series it draws from each arm's :func:`~grl_snam.coph.timeline.learning`, as (key, label
#: suffix, line style, band keys or None).
STATS = {
    "field": (
        "|r - r_true| where it sensed (10-90% band) and not",
        False,
        (("error_seen", "sensed", "solid", ("error_seen_lo", "error_seen_hi")),
         ("error_unseen", "not sensed", "dashed", None)),
    ),
    "damping": (
        "Damping: the plant fit's estimate, and the world's",
        False,
        (("damping", "estimate", "solid", None),),
    ),
    "policy": (
        "pi_risk (solid) and pi_barrier (dashed), from replay",
        False,
        (("pi_risk", "pi_risk", "solid", None), ("pi_barrier", "pi_barrier", "dashed", None)),
    ),
    "plant": (
        "Held-out plant loss after each update",
        True,
        (("plant_heldout", "held-out", "solid", None),),
    ),
    "replay": (
        "Replay cost after each update's line search",
        False,
        (("replay", "replay", "solid", None),),
    ),
}  # fmt: skip


class Learning:
    """The Learning window: what each arm's learner for one agent (``learn.agent``) measured at
    each of its updates, on pycvc_gl's ``chart:`` (libcvc 3.7.14, ``host.set_plot``).

    - ``learn.heldout``: the held-out field loss before and after each update, which the gate
      compares on the learner's validation readings, and the gate's threshold, before + slack.
    - ``learn.stat``: the statistic ``learn.show`` picks (:data:`STATS`).

    Both mark where the agent first knew of a door and where it arrived, put a cursor at the play
    head (``learn.t``), and seek there on a click (``learn.pick``, then ``learn_seek``). Without
    ``host.set_plot`` (an older pycvc_gl) this does nothing, and the document hides the window
    (``(has? "chart")``)."""

    #: How often a growing run's charts are redrawn (wall seconds).
    REFRESH = 0.25

    def __init__(self, host, demo, timelines, transport):
        self.host, self.demo, self.transport = host, demo, transport
        self.enabled = hasattr(host, "set_plot")
        self.floors = [Floor(tl.spec, feed.world(demo, tl.spec), px_per_m=4) for tl in timelines]
        self.key, self._at, self._t = None, -math.inf, None
        if self.enabled:
            host.on("learn_seek", self.seek)

    def seek(self):
        t = self.host.get_float("learn.pick", math.nan)
        if math.isfinite(t):
            self.transport.seek(t)

    def publish(self, timelines, t, finished=False):
        """Move the cursor to ``t``, and redraw the charts when the choice or the data changed
        (a growing run's at most every :attr:`REFRESH` seconds, until ``finished``)."""
        if not self.enabled:
            return
        host = self.host
        if self._t is None or abs(t - self._t) > 1e-3:
            host.set("learn.t", round(t, 3))
            self._t = t
        agent = min(max(host.get_int("learn.agent", 0), 0), 1)
        show = host.get("learn.show", "field")
        show = show if show in STATS else "field"
        data = tuple((len(tl.agents[agent].updates), tl.agents[agent].outcome is not None)
                     for tl in timelines)  # fmt: skip
        key, now = (agent, show, data), time.perf_counter()
        if key == self.key or not any(n for n, _ in data):
            return
        same_choice = self.key is not None and self.key[:2] == key[:2]
        if same_choice and not finished and now - self._at < self.REFRESH:
            return
        self.key, self._at = key, now
        series = [learning(tl, agent, floor) for tl, floor in zip(timelines, self.floors)]
        markers = self.markers(timelines, agent)
        names = [ARM_SHORT.get(tl.arm, tl.label) for tl in timelines]
        heldout = []
        for k, (name, s) in enumerate(zip(names, series)):
            heldout.append(dict(x=s["t"], y=s["field_before"], label=f"{name} before",
                                style=f"c{k} points"))  # fmt: skip
            heldout.append(dict(x=s["t"], y=s["field_heldout"], label=f"{name} after",
                                style=f"c{k}"))  # fmt: skip
        last = series[-1]
        heldout.append(dict(x=last["t"], y=last["field_before"] + SLACK,
                            label=f"gate ({names[-1]})", style="c7 dotted"))  # fmt: skip
        host.set_plot("learn.heldout", heldout, markers=markers)
        caption, log_y, keys = STATS[show]
        stat = []
        for k, (arm, s) in enumerate(zip(names, series)):
            for name, suffix, line, band in keys:
                d = dict(x=s["t"], y=s[name], label=f"{arm} {suffix}", style=f"c{k} {line}")
                if band:
                    d.update(lo=s[band[0]], hi=s[band[1]])
                stat.append(d)
        if show == "damping":
            truth = float(timelines[0].spec.get("damping", math.nan))
            ends = [min(s["t"][0] for s in series if len(s["t"])),
                    max(s["t"][-1] for s in series if len(s["t"]))]  # fmt: skip
            stat.append(dict(x=ends, y=[truth, truth], label="the world's", style="c7 dashed"))
        host.set_plot("learn.stat", stat, markers=markers)
        host.set("learn.log", "1" if log_y else "0")
        arms = "; ".join(f"{n}: {tl.label}" for n, tl in zip(names, timelines))
        host.set("learn.caption", f"Agent {ROLES[self.demo][agent]}. {arms}")
        host.set("learn.note", caption)

    @staticmethod
    def markers(timelines, agent):
        """Where the agent first knew of an obstacle (a door) and where it reached its goal,
        per arm."""
        out = []
        for tl in timelines:
            a = tl.agents[agent]
            a.index()
            name = ARM_SHORT.get(tl.arm, tl.label)
            if a.geometry:
                out.append((a.geometry[0][0], f"door ({name})", "note"))
            if a.outcome and a.outcome["success"]:
                out.append((a.outcome["finish_time"], f"goal ({name})", "note"))
        return sorted(out)


# ── the scene ──────────────────────────────────────────────────────────────────────────────────


def _box(length, width, height):
    return _host.humvee_box(length, width, height)


def _quad(x0, x1, y0, y1, z=0.0):
    v = [x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]
    return v, [0, 1, 2, 0, 2, 3], [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def _pose(x, y, heading, z=0.0, scale=1.0):
    c, s = math.cos(heading) * scale, math.sin(heading) * scale
    return [c, -s, 0, x, s, c, 0, y, 0, 0, scale, z, 0, 0, 0, 1]


HIDDEN = _pose(0.0, 0.0, 0.0, z=-50.0, scale=1e-3)


def _car_model(host):
    """The Humvee's model at a RoboRacer's length, or None (then each car is a box)."""
    try:
        from pycvc_gl.scenes import find_vehicle_model
        from pycvc_gl.vehicle import VehicleModel

        path = find_vehicle_model()
    except Exception:  # noqa: BLE001 - none installed
        return None
    if not path:
        return None
    try:
        return VehicleModel(host.app, path, CAR[0])  # pycvc_gl 3.7.13+
    except TypeError:
        return VehicleModel(path, CAR[0])


class View:
    """Both arms' scenes: floors, doors, cars, goals, tracks and packet arcs."""

    def __init__(self, host, demo, timelines, aspect=4 / 3):
        from grl_snam.fog_scene import _style

        self.host, self.demo, self.timelines, self._style = host, demo, timelines, _style
        self.aspect = max(float(aspect), 0.5)
        self.floors = [Floor(tl.spec, feed.world(demo, tl.spec)) for tl in timelines]
        f0 = self.floors[0]
        self.offsets = [k * (f0.x1 - f0.x0 + GAP) for k in range(len(timelines))]
        self.frame, self.risk_cache, self.arcs_used = 0, {}, {}
        lab = host.lab
        model = _car_model(host)
        for k, (tl, floor, ox) in enumerate(zip(timelines, self.floors, self.offsets)):
            v, t, uv = _quad(floor.x0 + ox, floor.x1 + ox, floor.y0, floor.y1)
            lab.add_mesh(f"floor_{k}", v, t, color=(1.0, 1.0, 1.0), uvs=uv)
            lab.set_texture(f"floor_{k}", floor.image(floor.truth))
            for name, door in tl.doors().items():
                cx, cy, r, hx, hy = door
                length = 2 * math.hypot(hx, hy) + 2 * r
                lab.add_mesh(f"door_{k}_{name}", *_box(length, 2 * r, 0.6), color=(0.55, 0.12, 0.1))
                host.scene.getGraphics(f"door_{k}_{name}").setTransform(HIDDEN)
            for i in range(2):
                colour = COLORS[i]
                if model is not None:
                    model.add(host.scene, f"car_{k}_{i}", colour)
                else:
                    lab.add_mesh(f"car_{k}_{i}", *_box(*CAR), color=colour)
                gx, gy = tl.spec["goals"][i]
                lab.add_mesh(f"goal_{k}_{i}", *_pyramid(0.35), color=colour)
                host.scene.getGraphics(f"goal_{k}_{i}").setTransform(
                    _pose(gx + ox + (0.15 if i else -0.15), gy, 0.0, z=0.45)
                )
        self.overview()

    def overview(self):
        """Both floors from the south at 52 degrees, between the panels above and below.
        frame_bounds puts the camera in orbit mode (azimuth -60, elevation 30); the orbit's
        state keys then set the pose."""
        f0, ox = self.floors[0], self.offsets[-1]
        x0, x1, y0, y1 = f0.x0, f0.x1 + ox, f0.y0, f0.y1
        width = x1 - x0
        self.host.frame_bounds((x0, y0, 0.0), (x1, y1, 1.0))
        self.host.set_many(
            {
                "center.x": 0.5 * (x0 + x1),
                "center.y": 0.5 * (y0 + y1) - 0.1 * (y1 - y0),
                "center.z": 0.0,
                "azimuth": -90.0,
                "elevation": 52.0,
                # Fits both floors across the window: measured at 1600x900 and 1024x768.
                "distance": max(1.2, 2.1 / self.aspect) * width,
            },
            prefix="viewers.main.camera.orbit",
        )

    def _path(self, name, points, colour, width):
        """(Re)draw the polyline ``name``."""
        scene = self.host.scene
        if scene.hasGraphics(name):
            scene.removeGraphics(name)
        self.host.lab.add_path(name, points, colour)
        self._style(scene, name, color=colour, mode="lines", line_width=width)

    def _heat(self, k, states):
        source = self.host.get("view.heat", "truth")
        floor = self.floors[k]
        if source not in ("agent0", "agent1"):
            return floor.truth
        s = states[int(source[-1])]
        key = (k, s.agent, s.version, tuple(st["id"] for st in s.stages))
        if key not in self.risk_cache:
            if len(self.risk_cache) > 64:
                self.risk_cache.clear()
            self.risk_cache[key] = floor.agent_risk(s.stages)
        return self.risk_cache[key]

    def draw(self, t, camera):
        host, lab, frame = self.host, self.host.lab, self.frame
        for k, (tl, floor, ox) in enumerate(zip(self.timelines, self.floors, self.offsets)):
            states = [tl.state(i, t, track=frame % 6 == 0) for i in range(2)]
            if frame % 2 == 0:
                overlays = []
                for s in states:
                    colour = COLORS[s.agent]
                    for age, st in enumerate(sorted(s.stages, key=lambda x: -x["id"])):
                        overlays.append(
                            (floor.outline(st["bounds"]), colour, (0.9, 0.55, 0.3)[age % 3])
                        )
                    if s.active and s.outcome is None:
                        overlays.append((floor.ring(s.q, feed_sensor_radius()), colour, 0.85))
                lab.set_texture(f"floor_{k}", floor.image(self._heat(k, states), overlays))
            known = set().union(*(s.known for s in states))
            for name, door in tl.doors().items():
                cx, cy, r, hx, hy = door
                node = host.scene.getGraphics(f"door_{k}_{name}")
                node.setTransform(
                    _pose(cx + ox, cy, math.atan2(hy, hx)) if door in known else HIDDEN
                )
            for s in states:
                host.scene.getGraphics(f"car_{k}_{s.agent}").setTransform(
                    _pose(s.q[0] + ox, s.q[1], s.heading, z=0.02)
                )
                if frame % 6 == 0 and len(s.track) >= 2:
                    step = max(1, len(s.track) // 400)
                    pts = [(float(x) + ox, float(y), 0.05) for x, y in s.track[::step]]
                    pts.append((s.q[0] + ox, s.q[1], 0.05))
                    self._path(f"track_{k}_{s.agent}", pts, COLORS[s.agent], 2)
            self._arcs(k, tl, t, ox)
        camera.update(self._follow_target(t))
        self.frame += 1

    def _follow_target(self, t):
        s = self.timelines[-1].state(0, t)
        return (s.q[0] + self.offsets[-1], s.q[1], 0.0)

    def _arcs(self, k, tl, t, ox):
        host = self.host
        recent = tl.deliveries_between(t - ARC_SECONDS, t)[-ARCS:]
        for j, (when, sender, receiver, identity, nbytes, accepted, _) in enumerate(recent):
            a = tl.packets[identity][2]
            b = tl.state(receiver, when).q
            span = math.dist(a, b)
            pts = []
            for n in range(17):
                f = n / 16
                pts.append(
                    (
                        a[0] + (b[0] - a[0]) * f + ox,
                        a[1] + (b[1] - a[1]) * f,
                        0.3 + (0.4 + 0.12 * span) * 4 * f * (1 - f),
                    )
                )
            name = f"arc_{k}_{j}"
            colour = ACCEPT if accepted else REJECT
            self._path(name, pts, colour, min(8.0, 1.0 + nbytes / 4000.0))
        for j in range(len(recent), self.arcs_used.get(k, 0)):
            host.scene.getGraphics(f"arc_{k}_{j}").setTransform(HIDDEN)
        self.arcs_used[k] = max(len(recent), self.arcs_used.get(k, 0))


def _pyramid(size):
    s = size
    v = [
        0.0,
        0.0,
        -s,
        -s / 2,
        -s / 2,
        0.0,
        s / 2,
        -s / 2,
        0.0,
        s / 2,
        s / 2,
        0.0,
        -s / 2,
        s / 2,
        0.0,
    ]
    return v, [0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 1, 2, 3, 1, 3, 4]


def feed_sensor_radius():
    from grl_snam.coph.timeline import CFG

    return CFG.sensor_radius


# ── the port ───────────────────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _host.parse(argv, _args)
    if args.mode != "window" and not any(a.split("=")[0] == "--frames" for a in argv):
        args.frames = 0  # headless and offscreen run to the end, not the host's default 30
    demo = args.demo
    arms = tuple(a.strip() for a in args.arms.split(",")) if args.arms else feed.DEFAULT_ARMS[demo]
    if len(arms) != 2 or arms[0] == arms[1]:
        print("grl_snam coph: --arms takes two different arms", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else feed.DEFAULT_SEEDS[demo]
    steps = args.steps or feed.DEFAULT_STEPS[demo]
    try:
        for arm in arms:
            feed.job(demo, arm, seed, steps, args.scene)
        if args.source:
            source = Source.replay(demo, arms, seed, args.scene, args.source)
        else:
            source = Source.live(demo, arms, seed, steps, args.scene, args.out)
    except (ValueError, FileNotFoundError) as e:
        print(f"grl_snam coph: {e}", file=sys.stderr)
        return 2
    title = TITLES[demo] + (f" ('{args.scene}')" if demo == "hazard" else "")
    try:
        host = _host.open_host("coph.ari", "coph", args)
        drawn = host.mode != "headless"
        view = View(host, demo, source.timelines, args.width / args.height) if drawn else None
        camera = _host.CameraSwitch(host, view.overview) if drawn else None
        if drawn:
            _host.track_camera(host, length=2.0)
        transport = Transport(host, title)
        host.on("restart", transport.restart)
        panels = Learning(host, demo, source.timelines, transport)
        host.set("view.heat", args.heat)
        if args.at > 0:
            transport.t = args.at
            host.set("play.live", False)

        @host.on_frame
        def _frame(dt):
            if not source.poll() and not drawn and source.live_running:
                time.sleep(0.02)  # headless frames are not paced: wait for the workers
            if source.error:
                print(f"grl_snam coph: {source.error}", file=sys.stderr, flush=True)
                host.request_close()
                return
            t = transport.tick(dt, source.head(), source.live_running)
            publish(host, demo, source, t, seed, args.scene)
            panels.publish(source.timelines, t, source.finished and not source.live_running)
            if view is not None:
                view.draw(t, camera)
            if not drawn and source.finished and not source.live_running:
                host.request_close()

        code = _host.finish(host, args)
    finally:
        source.close()
    if args.log:
        meta = dict(demo=demo, arms=list(arms), seed=seed, steps=steps, origin=source.origin,
                    episodes=source.where,
                    scene=args.scene if demo == "hazard" else None, reference=REFERENCE[demo],
                    result=result_line(demo, source.timelines, seed, args.scene))  # fmt: skip
        _save(args.log, log_arrays(source.timelines, meta))
    arrived = goals(source.timelines)
    print(
        f"grl_snam coph: {demo} {', '.join(arms)}, seed {seed}: goals {arrived}/4, collisions "
        f"{collisions(source.timelines)}; {result_line(demo, source.timelines, seed, args.scene)}",
        flush=True,
    )
    if source.error:
        return 1
    if code:
        return code
    return 3 if arrived < args.min_goals else 0


def _save(path, arrays):
    import numpy as np

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


if __name__ == "__main__":
    sys.exit(main())
