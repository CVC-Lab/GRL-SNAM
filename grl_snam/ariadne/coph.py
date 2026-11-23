"""CoPH-Stagewise's crossing and hazard episodes in 3-D, communication off beside CoPH (demo 2a).

    python -m grl_snam.ariadne coph crossing                  # live, seed 76000, both arms
    python -m grl_snam.ariadne coph hazard --scene severe     # live, none beside field
    python -m grl_snam.ariadne coph crossing --from runs/crossing   # replay a grl-snam coph run
    python -m grl_snam.ariadne coph crossing --headless --steps 200 --log run.npz
    python -m grl_snam.ariadne coph walkthrough --from runs/coph    # the scripted walkthrough
    python -m grl_snam.ariadne coph walkthrough --from runs/coph --offscreen --record coph.mp4

The episodes are the vendored demos' own (``grl_snam.coph``, roadmap §3.2). A live run calls the
runner exactly as ``grl-snam coph`` does, one worker process per arm, and keeps the episodes with
``--out`` in ``grl-snam coph``'s layout. ``--from`` replays such a run's episodes. Either way the
viewer reads the episode through :mod:`grl_snam.coph.feed`'s messages and changes nothing, so its
outcomes are T1's (``tests/test_coph_feed.py``).

The two arms run on the same seed and stand side by side, communication off on the left:
- **the floor:** the surveyed corridors, with the risk field as a heat map (the truth, or one
  agent's own ``r``: the Heat map combo), each agent's three kept stages (the newest brightest) and
  its sensor disc, and a nameplate north of each floor naming its arm;
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

The headline labels the claims as the roadmap's §8 has them: what the reference established, with
its own seed counts (green); this run's own number once both arms finish; what the reference has
not established (replay, receiver field fitting, adaptive-rate sharing; amber); and that the
validation gate cannot reject an update at its 0.003 slack, with this run's count (amber).

The panels are docked: the CoPH window and Learning down the right, the Headline above and Agents
below, and the camera frames the floors in the rest (:data:`RIGHT`, :data:`TOP`, :data:`BOTTOM`).
The window opens at 1600x900 unless ``--width`` and ``--height`` say otherwise.

``--tour`` plays the scripted walkthrough of the default arms and seed (:func:`tour_steps`): camera
moves, the play head and the heat map, with a caption per step in the CoPH window, whose Next,
Back and Restart buttons drive it; Playing, or a drag of the position slider, hands the run back.
``walkthrough`` runs the crossing tour, then the hazard tour ('severe'), each in its own process.
A tour replays episodes: those under ``--from`` (``<from>/crossing``, ``<from>/hazard`` for
``walkthrough``), computed there first, headless, when missing. ``--shots DIR`` writes a PNG at
the end of each step, and ``--record PATH`` an mp4 of every frame (ffmpeg on PATH, or
``--ffmpeg``); with ``--offscreen`` the frames step at 30 per simulated second whatever the
machine's speed.

Headless (``--headless``), the run goes to its end with nothing drawn; ``--log run.npz`` writes the
cars template's log shape (``pycvc_gl.cars.CarsLog``) plus the ledger columns
(:func:`grl_snam.coph.timeline.log_arrays`). The exit code is 0, 1 when a run failed, and 3 when
fewer agents arrived than ``--min-goals``.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from grl_snam.ariadne import _host
from grl_snam.coph import feed
from grl_snam.coph.records import Config
from grl_snam.coph.timeline import Floor, Timeline, comparison, learning, log_arrays

TITLES = {
    "crossing": "Crossing traverses: what one agent has seen saves the other's route",
    "hazard": "Hazard: a shared graded field, priced by each agent's own risk weight",
}
#: What the reference establishes (roadmap §1 and §8), with its own seed counts: CoPH-Sense's
#: CROSSING_DEMO.md and HAZARD_DEMO.md, reproduced here on 2026-11-09 to 0.1.
REFERENCE = {
    "crossing": "Established by the reference (CoPH-Sense c1f1ff3, its own runs: 2 seeds x 4 arms, "
    "16/16 goals, 0 collisions): sharing what each agent senses saves the team 27.1 m of detour, "
    "on both seeds (28.0 m predicted in advance).",
    "hazard": "Established by the reference (CoPH-Sense c1f1ff3, its own runs: 1 seed x 4 scenes x "
    "8 arms, 64/64 goals): with the scout's field shared, the carrier's worst-case excess cost is "
    "4.6, against 68.1 with no communication ('severe': 0.0 against 68.1).",
}
#: What the reference has not established (roadmap §1, "Not established"), and how each piece runs
#: here. Adaptive-rate sharing is in the vendored code (cooperative.rate_probability) but neither
#: demo calls it: staggered_demo exports on each sender's fixed 8-step (0.96 s) cadence.
UNESTABLISHED = {
    "crossing": "Not established by the reference: any gain from replay (it runs in both arms "
    "here), from the receiver's field fitting (it runs in the CoPH arm), or from adaptive-rate "
    "sharing (not exercised: the arms export every 0.96 s).",
    "hazard": "Not established by the reference: any gain from replay (it runs in both arms "
    "here), from the receiver's field fitting (it runs in the CoPH arm, and the arm without it "
    "ties the shared field at 4.6, as raw readings do at more bytes), or from adaptive-rate "
    "sharing (not exercised: the arms export every 0.96 s).",
}
#: The validation gate over the reference's own 40 episodes (CoPH-Sense's results/*_reproduction,
#: 2 crossing seeds x 4 arms and 4 hazard scenes x 8 arms, counted 2026-11-23): every update was
#: kept, 3,112 of the 5,212 with a worse held-out field loss (by at most 0.0022), and the median
#: held-out field loss was 2.8e-5, about a hundredth of the slack.
GATE_REFERENCE = "the reference's 40 episodes kept all 5,212 updates"
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
#: The arms' names in the Learning charts' legends and markers, and in the Agents table.
ARM_SHORT = {"disabled": "off", "none": "off", "reciprocal_fixed": "CoPH",
             "one_way_01": "CoPH 0>1", "one_way_10": "CoPH 1>0", "field": "field",
             "raw": "raw", "no_learning": "no rx fit", "decision": "decision"}  # fmt: skip
#: Each floor's nameplate.
PLATES = {"disabled": "Communication off", "none": "Communication off",
          "reciprocal_fixed": "CoPH: each agent shares what it senses",
          "one_way_01": "CoPH: agent 0 shares with agent 1",
          "one_way_10": "CoPH: agent 1 shares with agent 0",
          "field": "CoPH: the scout shares its graded field",
          "raw": "CoPH: the scout shares raw readings",
          "no_learning": "CoPH: shared field, no receiver fitting",
          "decision": "CoPH: the scout shares its decision"}  # fmt: skip
COLORS = ((0.93, 0.45, 0.13), (0.16, 0.47, 0.85))  # agent 0, agent 1
ACCEPT, REJECT = (0.15, 0.75, 0.25), (0.86, 0.16, 0.16)
GAP = 3.0  # metres between the two arms' floors
PLATE = 1.6  # metres of nameplate north of each floor
CAR = (0.58, 0.30, 0.25)  # a RoboRacer's length, width and height (m)
ARC_SECONDS = 2.5  # how long a delivered packet's arc stays up (simulated time)
ARCS = 12  # arc nodes per arm

#: The docked panels as fractions of the work area, as coph.ari places them: the right column
#: (the CoPH window above Learning) and, left of it, Headline above and Agents below. The camera
#: frames the floors in what is left (:meth:`View.region`).
RIGHT, TOP, BOTTOM = 0.27, 0.28, 0.22
MENU_PX = 24  # the menu bar's height
DEFAULT_SIZE = (1600, 900)
VFOV = 30.0  # the camera's vertical field of view (degrees)
FPS = 30  # frames per simulated second in a tour or a recording


def _args(p):
    p.add_argument("demo", choices=("crossing", "hazard", "walkthrough"),
                   help="which vendored demo, or the walkthrough of both")  # fmt: skip
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
    p.add_argument("--tour", action="store_true", help="play the scripted walkthrough")
    p.add_argument("--shots", default=None, help="write a PNG here at the end of each tour step")
    p.add_argument("--record", default=None, help="write every frame to this mp4 (ffmpeg)")
    p.add_argument("--ffmpeg", default=None, help="the ffmpeg to record with (default: PATH's)")


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

    @staticmethod
    def episode(demo, arm, seed, scene, root):
        return feed.run_dir(demo, root, scene) / "episodes" / f"{seed}_{arm}.json.gz"

    @classmethod
    def replay(cls, demo, arms, seed, scene, root):
        from grl_snam.coph.evaluation import load

        timelines = []
        for arm in arms:
            path = cls.episode(demo, arm, seed, scene, root)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing: run `grl-snam coph {demo} --out {root}` first (it runs "
                    "every arm), or drop --from to run these two live"
                )
            timelines.append(Timeline.from_episode(demo, load(path), label=label(arm)))
        where = feed.run_dir(demo, root, scene) / "episodes"
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


def _compute(job):
    demo, arm, seed, steps, scene, out = job
    feed.run(demo, arm, seed, steps, scene=scene, out=out)
    return arm


def ensure_episodes(demo, arms, seed, steps, scene, root):
    """Compute, headless, the episodes of ``arms`` that ``root`` lacks (one process per arm),
    in ``grl-snam coph``'s layout, so that a tour can replay them."""
    missing = [a for a in arms if not Source.episode(demo, a, seed, scene, root).exists()]
    if not missing:
        return
    import multiprocessing as mp

    where = str(feed.run_dir(demo, root, scene))
    print(f"grl_snam coph: computing {demo}'s {', '.join(missing)} into {where} first (a crossing "
          "arm takes about 20 minutes, a hazard arm about 6) ...", flush=True)  # fmt: skip
    with mp.get_context("spawn").Pool(len(missing)) as pool:
        for arm in pool.imap_unordered(
            _compute, [(demo, a, seed, steps, scene, where) for a in missing]
        ):
            print(f"grl_snam coph: {demo} {arm} done", flush=True)


# ── the transport ──────────────────────────────────────────────────────────────────────────────


class Transport:
    """The play head over a run whose end may still be growing (a live run): with ``play.live``
    on, it sits on the newest boundary; otherwise it plays at ``play.speed``, and dragging the
    position slider scrubs (and leaves Live)."""

    def __init__(self, host, title):
        self.host, self.t, self._pos = host, 0.0, 0.0
        self.dragged = False
        host.set("play.title", title)

    def restart(self):
        self.t = 0.0
        self.host.set("play.playing", True)
        self.host.set("play.live", False)

    def seek(self, t):
        """Put the play head at ``t`` (a click on a learning chart, or a tour), leaving Live."""
        self.t = max(float(t), 0.0)
        self.host.set("play.live", False)

    def tick(self, dt, head, live):
        host, duration = self.host, max(head, 1e-6)
        pos = host.get_float("play.pos", 0.0)
        self.dragged = abs(pos - self._pos) > 1e-6
        if self.dragged:  # dragged: the slider wins
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
                    arm=ARM_SHORT.get(tl.arm, tl.label),
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


def gate_counts(timelines, t=math.inf):
    """The validation gate's decisions up to ``t`` over both arms' agents: (kept, rejected, kept
    with a worse held-out field loss, the largest rise in that loss)."""
    kept = rejected = worse = 0
    rise = 0.0
    for tl in timelines:
        for a in tl.agents:
            for u in a.updates:
                if u[0] > t + 1e-9:
                    continue
                if u[2]:
                    kept += 1
                    stats = u[7] if len(u) > 7 else {}
                    before, after = stats.get("field_before"), stats.get("field_after")
                    if before is not None and after is not None and after > before:
                        worse += 1
                        rise = max(rise, after - before)
                else:
                    rejected += 1
    return kept, rejected, worse, rise


def gate_line(timelines, t=math.inf):
    kept, rejected, worse, rise = gate_counts(timelines, t)
    return (
        f"The validation gate can't reject: its slack ({SLACK:g}) is about 100x the typical "
        f"held-out field loss, and {GATE_REFERENCE}. This run so far: {kept} kept, {rejected} "
        f"rejected; {worse} kept with a worse held-out field loss (up to +{rise:.4f})."
    )


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
    host.set("coph.unestablished", UNESTABLISHED[demo])
    host.set("coph.gate", gate_line(source.timelines, t))


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
        "pi_risk (solid) and pi_barrier (dashed), from replay: no gain from it is established",
        False,
        (("pi_risk", "pi_risk", "solid", None), ("pi_barrier", "pi_barrier", "dashed", None)),
    ),
    "plant": (
        "Held-out plant loss after each update",
        True,
        (("plant_heldout", "held-out", "solid", None),),
    ),
    "replay": (
        "Replay cost after each update's line search: no gain from replay is established",
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


def _font(size):
    from PIL import ImageFont

    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def plate_image(text, width_m, height_m, px_per_m=80):
    """A nameplate as uint8 RGB, row 0 its north edge (as :func:`texture` hands a floor over):
    light text on a dark plate, as large as fits, or None without Pillow."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    w, h = int(round(width_m * px_per_m)), int(round(height_m * px_per_m))
    img = Image.new("RGB", (w, h), (34, 38, 44))
    draw = ImageDraw.Draw(img)
    size = int(h * 0.6)
    font = _font(size)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    if x1 - x0 > 0.94 * w:
        font = _font(max(int(size * 0.94 * w / (x1 - x0)), 8))
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text(((w - (x1 - x0)) / 2 - x0, (h - (y1 - y0)) / 2 - y0), text,
              fill=(238, 240, 242), font=font)  # fmt: skip
    return np.ascontiguousarray(np.asarray(img))


def texture(image):
    """A :class:`Floor` image (row 0 its south edge) as ``Lab.set_texture`` lays it on a quad
    from :func:`_quad`, which puts an image's row 0 at the quad's north edge."""
    import numpy as np

    return np.ascontiguousarray(image[::-1])


def project(pose, width, height, point):
    """Pixel ``(u, v)`` (v down) of the ground point ``point`` under the orbit ``pose``: the eye
    at ``distance`` from the centre, ``azimuth`` round from +x and ``elevation`` up, z up."""
    az, el = math.radians(pose["azimuth"]), math.radians(pose["elevation"])
    c = (pose["center.x"], pose["center.y"], pose["center.z"])
    d = pose["distance"]
    eye = (c[0] + d * math.cos(el) * math.cos(az), c[1] + d * math.cos(el) * math.sin(az),
           c[2] + d * math.sin(el))  # fmt: skip
    fwd = [(a - b) / d for a, b in zip(c, eye)]
    right = (fwd[1], -fwd[0], 0.0)  # fwd x z
    n = math.hypot(right[0], right[1]) or 1.0
    right = (right[0] / n, right[1] / n, 0.0)
    up = (right[1] * fwd[2], -right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0])
    p = (point[0] - eye[0], point[1] - eye[1], -eye[2])
    z = sum(a * b for a, b in zip(p, fwd))
    f = 0.5 * height / math.tan(math.radians(VFOV) / 2)
    return (0.5 * width + f * sum(a * b for a, b in zip(p, right)) / z,
            0.5 * height - f * sum(a * b for a, b in zip(p, up)) / z)  # fmt: skip


def fit(width, height, box, region, elevation=55.0, azimuth=-90.0, margin=0.96):
    """The orbit (``viewers.main.camera.orbit.*``) that shows the ground rectangle ``box``
    ``(x0, x1, y0, y1)`` inside ``region`` ``(u0, u1, v0, v1)``, pixels of a ``width`` x
    ``height`` view with v down, looking north at ``elevation`` degrees: a first guess at the
    orbit's centre, then a few rounds of projecting the box's corners (:func:`project`) and
    scaling and shifting until they fill ``margin`` of the region, centred."""
    x0, x1, y0, y1 = box
    u0, u1, v0, v1 = region
    e = math.radians(elevation)
    f = 0.5 * height / math.tan(math.radians(VFOV) / 2)
    s = margin * min((u1 - u0) / max(x1 - x0, 1e-6), (v1 - v0) / max((y1 - y0) * math.sin(e), 1e-6))
    pose = {"center.x": 0.5 * (x0 + x1), "center.y": 0.5 * (y0 + y1), "center.z": 0.0,
            "azimuth": azimuth, "elevation": elevation, "distance": f / s}  # fmt: skip
    for _ in range(12):
        pts = [project(pose, width, height, (x, y)) for x in (x0, x1) for y in (y0, y1)]
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        scale = margin * min((u1 - u0) / max(max(us) - min(us), 1e-6),
                             (v1 - v0) / max(max(vs) - min(vs), 1e-6))  # fmt: skip
        s = f / pose["distance"]  # pixels per metre at the centre
        du = 0.5 * (u0 + u1) - 0.5 * (min(us) + max(us))
        dv = 0.5 * (v0 + v1) - 0.5 * (min(vs) + max(vs))
        pose["center.x"] -= du / s
        pose["center.y"] += dv / (s * math.sin(e))
        pose["distance"] /= scale
        if abs(scale - 1.0) < 1e-3 and abs(du) < 0.5 and abs(dv) < 0.5:
            break
    return pose


class View:
    """Both arms' scenes: floors, nameplates, doors, cars, goals, tracks and packet arcs, and
    the camera's poses (:meth:`pose_for`)."""

    def __init__(self, host, demo, timelines, size=DEFAULT_SIZE):
        from grl_snam.fog_scene import _style

        self.host, self.demo, self.timelines, self._style = host, demo, timelines, _style
        self.size = (max(int(size[0]), 64), max(int(size[1]), 64))
        self.floors = [Floor(tl.spec, feed.world(demo, tl.spec)) for tl in timelines]
        f0 = self.floors[0]
        self.offsets = [k * (f0.x1 - f0.x0 + GAP) for k in range(len(timelines))]
        self.frame, self.risk_cache, self.arcs_used = 0, {}, {}
        self.pose = None
        lab = host.lab
        model = _car_model(host)
        for k, (tl, floor, ox) in enumerate(zip(timelines, self.floors, self.offsets)):
            v, t, uv = _quad(floor.x0 + ox, floor.x1 + ox, floor.y0, floor.y1)
            lab.add_mesh(f"floor_{k}", v, t, color=(1.0, 1.0, 1.0), uvs=uv)
            lab.set_texture(f"floor_{k}", texture(floor.image(floor.truth)))
            plate = plate_image(PLATES.get(tl.arm, label(tl.arm)), floor.x1 - floor.x0, PLATE)
            if plate is not None:
                v, t, uv = _quad(floor.x0 + ox, floor.x1 + ox, floor.y1 + 0.1,
                                 floor.y1 + 0.1 + PLATE, z=0.01)  # fmt: skip
                lab.add_mesh(f"plate_{k}", v, t, color=(1.0, 1.0, 1.0), uvs=uv)
                lab.set_texture(f"plate_{k}", plate)
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

    # ── the camera ─────────────────────────────────────────────────────────────────────────

    def region(self):
        """Where the floors go: the window less the docked panels (pixels, v down)."""
        w, h = self.size
        work = h - MENU_PX
        return (8.0, (1.0 - RIGHT) * w - 8.0, MENU_PX + TOP * work + 8.0, h - BOTTOM * work - 8.0)

    def box(self, k=None):
        """The ground rectangle of arm ``k``'s floor and nameplate, or of both arms'."""
        f0 = self.floors[0]
        if k is None:
            return f0.x0, f0.x1 + self.offsets[-1], f0.y0, f0.y1 + 0.1 + PLATE
        ox = self.offsets[k]
        return f0.x0 + ox, f0.x1 + ox, f0.y0, f0.y1 + 0.1 + PLATE

    def pose_for(self, view):
        """The orbit for a tour's view: ``("overview",)``, ``("arm", k)`` or ``("spot", k, x, y,
        half_width)`` (in arm ``k``'s own coordinates), with an optional elevation last."""
        kind, rest = view[0], list(view[1:])
        if kind == "overview":
            box, elev = self.box(), (rest[0] if rest else 55.0)
        elif kind == "arm":
            box, elev = self.box(rest[0]), (rest[1] if len(rest) > 1 else 55.0)
        else:
            k, x, y, r = rest[:4]
            ox = self.offsets[k]
            box, elev = (x - r + ox, x + r + ox, y - 0.6 * r, y + 0.6 * r), (
                rest[4] if len(rest) > 4 else 60.0
            )
        return fit(*self.size, box, self.region(), elevation=elev)

    def set_pose(self, pose):
        self.pose = dict(pose)
        self.host.set_many(self.pose, prefix="viewers.main.camera.orbit")

    def overview(self):
        """Both floors from the south at 55 degrees, in the room the docked panels leave.
        frame_bounds puts the camera in orbit mode (azimuth -60, elevation 30); the orbit's
        state keys then set the pose."""
        x0, x1, y0, y1 = self.box()
        self.host.frame_bounds((x0, y0, 0.0), (x1, y1, 1.0))
        self.set_pose(self.pose_for(("overview",)))

    # ── each frame ─────────────────────────────────────────────────────────────────────────

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
        heat = host.get("view.heat", "truth")
        redraw = frame % 2 == 0 or heat != getattr(self, "_last_heat", heat)
        self._last_heat = heat
        for k, (tl, floor, ox) in enumerate(zip(self.timelines, self.floors, self.offsets)):
            states = [tl.state(i, t, track=frame % 6 == 0) for i in range(2)]
            if redraw:
                overlays = []
                for s in states:
                    colour = COLORS[s.agent]
                    for age, st in enumerate(sorted(s.stages, key=lambda x: -x["id"])):
                        overlays.append(
                            (floor.outline(st["bounds"]), colour, (0.9, 0.55, 0.3)[age % 3])
                        )
                    if s.active and s.outcome is None:
                        overlays.append((floor.ring(s.q, feed_sensor_radius()), colour, 0.85))
                lab.set_texture(f"floor_{k}", texture(floor.image(self._heat(k, states), overlays)))
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


# ── the walkthrough ────────────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """One step of a tour: show ``caption``, move the camera to ``view`` (:meth:`View.pose_for`)
    over ``move`` seconds, set the heat map and the Learning window (``learn`` = (agent, show),
    or None to hide it), play to ``until`` (simulated seconds) at ``speed``, then hold ``hold``
    seconds."""

    caption: str
    view: tuple = ("overview",)
    until: float | None = None
    speed: float = 1.0
    hold: float = 3.0
    heat: str = "truth"
    learn: tuple | None = None
    move: float = 1.5


def _first(agent, kind=None, obstacle=None):
    """When ``agent`` first knew of an obstacle (of ``kind``, or ``obstacle``), else None."""
    agent.index()
    for t, k, o in agent.geometry:
        if (kind is None or k == kind) and (obstacle is None or tuple(o) == tuple(obstacle)):
            return t
    return None


def _finish(agent):
    o = agent.outcome
    return o["finish_time"] if o and o["success"] else None


def _turn_back(agent, start):
    """When ``agent`` first turned back west after driving east from ``start`` (s), else None."""
    agent.index()
    best = None
    for t, x, *_ in agent.poses:
        if t < start:
            continue
        if best is None or x[0] >= best[1]:
            best = (t, x[0])
        elif best[1] - x[0] > 0.3:
            return best[0]
    return None


def tour_steps(demo, timelines, seed, scene):
    """The walkthrough of the default arms (communication off on the left, CoPH on the right),
    with its times read from the episodes. A step never plays back to before the last one's end
    (an episode cut short by ``--steps`` lacks the later events)."""
    steps, end = _tour_steps(demo, timelines, seed, scene), 0.0
    for s in steps:
        if s.until is not None:
            s.until = end = max(s.until, end)
    return steps


def _tour_steps(demo, timelines, seed, scene):
    off, on = timelines
    c = comparison(demo, off, on)
    if demo == "crossing":
        doors = {name: tuple(d) for name, d in on.doors().items()}
        middle, north = doors["middle"], doors["north"]
        t_sense = _first(on.agents[1], obstacle=middle) or 4.0
        t_told = _first(on.agents[0], "received_geometry", middle) or t_sense
        t_north = _first(on.agents[0], obstacle=north) or 21.0
        t_north_told = _first(on.agents[1], "received_geometry", north) or t_north
        t_off_middle = _first(off.agents[0], obstacle=middle) or 25.0
        t_off_north1 = _first(off.agents[1], obstacle=north) or 41.0
        t_off_north0 = _first(off.agents[0], obstacle=north) or 62.0
        on_done = [_finish(a) or on.duration for a in on.agents]
        off_done = [_finish(a) or off.duration for a in off.agents]
        s = c["saving"] if c else (math.nan, math.nan)
        return [
            Step(
                "Crossing. Agent 0 (orange) starts in the west and agent 1 (blue) in the east, "
                "each bound for the other's start. Two doors in the corridor maze are shut, and "
                "neither agent knows where. Left: communication off. Right: CoPH, each agent "
                "sending what it senses to the other. Same seed, same code.",
                hold=7.0,
            ),
            Step(
                f"{t_sense:.1f} s: agent 1 senses the middle door beside its start, in both arms. "
                f"On the right its packet (the green arc) reaches agent 0 at {t_told:.1f} s, and "
                "agent 0 turns north before it has seen the door.",
                view=("arm", 1),
                until=t_told + 1.2,
                speed=0.5,
                hold=4.0,
            ),
            Step(
                f"Right: agent 0 meets the north door at {t_north:.1f} s and says so; agent 1, "
                f"told at {t_north_told:.1f} s, turns back before reaching it. Both now take the "
                "south corridor.",
                view=("arm", 1),
                until=t_north_told + 2.0,
                speed=3.0,
                hold=4.0,
            ),
            Step(
                f"Left, with no communication: agent 0 finds the middle door itself at "
                f"{t_off_middle:.1f} s and turns back, while agent 1 heads for the north door "
                "that agent 0 on the right has already reported.",
                view=("arm", 0),
                until=t_off_middle + 3.0,
                speed=2.0,
                hold=4.0,
            ),
            Step(
                f"Left: agent 1 finds the north door at {t_off_north1:.1f} s, and agent 0 finds "
                f"it too at {t_off_north0:.1f} s. Each pays for every dead end itself.",
                until=t_off_north0 + 1.5,
                speed=4.0,
                hold=3.0,
            ),
            Step(
                f"Right: both arrive, at {on_done[0]:.1f} and {on_done[1]:.1f} s.",
                until=max(on_done) + 0.5,
                speed=4.0,
                hold=2.5,
            ),
            Step(
                f"Left: they arrive at {off_done[0]:.1f} and {off_done[1]:.1f} s.",
                until=max(off_done),
                speed=4.0,
                hold=2.5,
            ),
            Step(
                f"This seed ({seed}): sharing saves agent 0 {s[0]:.1f} m and agent 1 {s[1]:.1f} m "
                f"(team {s[0] + s[1]:.1f} m). The reference's own result, over 2 seeds x 4 arms "
                "with 16/16 goals, is the 27.1 m in the headline. Learning, right: agent 0's "
                "held-out field loss at each update and the gate's threshold, which no update "
                "comes near.",
                learn=(0, "field"),
                hold=10.0,
            ),
        ]
    t_rx = min((d[0] for d in on.deliveries if d[2] == 1), default=26.0)
    start = on.start(1)
    t_back = _turn_back(off.agents[1], start) or 41.0
    on_done, off_done = _finish(on.agents[1]) or on.duration, _finish(off.agents[1]) or off.duration
    scout = _finish(on.agents[0]) or 36.0
    cost = c["carrier_cost"] if c else (math.nan, math.nan)
    return [
        Step(
            f"Hazard ('{scene}'). A graded hazard (red) lies across the middle corridor. The "
            "scout (orange, risk weight 0.25) crosses first; the carrier (blue, risk weight 6) "
            f"leaves the same start {start:.0f} s later for the same goal. Left: communication "
            "off. Right: the scout shares its graded field.",
            hold=7.0,
        ),
        Step(
            "The scout drives straight through: at its risk weight the hazard costs it little. "
            "The floors now show the scout's own fitted field r, which picks the hazard up as "
            "the scout passes. On the right, the scout sends what it has fitted to the carrier.",
            heat="agent0",
            until=start - 1.0,
            speed=4.0,
            hold=3.0,
        ),
        Step(
            f"Right: the scout's packets reach the carrier at {t_rx:.1f} s, before the carrier "
            "has sensed anything. Its router prices the scout's field at the carrier's own risk "
            "weight, 6, and it takes the north corridor.",
            view=("arm", 1),
            until=t_rx + 6.0,
            speed=1.5,
            hold=4.0,
        ),
        Step(
            "Left, with no communication: the carrier knows nothing of the hazard. Its own "
            "field r (now on the floors) picks the hazard up only once it drives in, and it "
            f"turns back at {t_back:.1f} s.",
            view=("arm", 0),
            heat="agent1",
            until=t_back + 3.0,
            speed=2.0,
            hold=4.0,
        ),
        Step(
            f"The scout arrived at {scout:.1f} s in both arms. Right: the carrier arrives at "
            f"{on_done:.1f} s.",
            until=on_done + 0.5,
            speed=4.0,
            hold=2.5,
        ),
        Step(
            f"Left: the carrier arrives at {off_done:.1f} s, after the way back.",
            until=off_done,
            speed=4.0,
            hold=2.5,
        ),
        Step(
            f"This run: the carrier's cost (time + risk) is {cost[0]:.1f} with communication off "
            f"and {cost[1]:.1f} with the shared field. The reference's result over 4 scenes x 8 "
            "arms is the worst case in the headline, 4.6 against 68.1. Its arm without receiver "
            "fitting ties at 4.6: the carrier routes on the scout's field either way, so what its "
            "own fit of that field adds is not established.",
            learn=(1, "field"),
            hold=10.0,
        ),
    ]


def _lerp_pose(a, b, f):
    out = {}
    for k in b:
        x, y = a.get(k, b[k]), b[k]
        if k == "azimuth":
            y = x + ((y - x + 180.0) % 360.0 - 180.0)
        if k == "distance":  # zoom evenly
            out[k] = math.exp(math.log(max(x, 1e-6)) * (1 - f) + math.log(max(y, 1e-6)) * f)
        else:
            out[k] = x + (y - x) * f
    return out


class Tour:
    """Plays :func:`tour_steps` over the transport: ``view.tour`` on, it owns the play head, the
    camera, the heat map and the Learning window; Next, Back and Restart (``tour_next``,
    ``tour_back``, ``tour_restart``) move between steps. Playing, or a drag of the position
    slider, turns it off and hands the run back."""

    def __init__(self, host, view, transport, steps, *, demo, shots=None, close_at_end=False):
        self.host, self.view, self.transport, self.steps = host, view, transport, steps
        self.demo, self.shots, self.close_at_end = demo, shots, close_at_end
        self.want_shot = None
        self.starts, t = [], 0.0
        for s in steps:
            self.starts.append(t)
            t = s.until if s.until is not None else t
        self.index, self.phase, self.phase_t, self.cam_from = -1, "done", 0.0, None
        host.on("tour_next", self.next)
        host.on("tour_back", self.back)
        host.on("tour_restart", self.restart)
        host.set("view.tour", "1")
        if shots:
            Path(shots).mkdir(parents=True, exist_ok=True)
        self.begin(0)

    def begin(self, i, seek=False):
        host, step = self.host, self.steps[i]
        self.index, self.phase, self.phase_t = i, "move", 0.0
        if seek:
            self.transport.seek(self.starts[i])
        host.set("view.tour", "1")
        host.set("play.playing", False)
        host.set("play.live", False)
        host.set("view.camera", "overview")
        host.set("view.heat", step.heat)
        host.set("view.learning", "1" if step.learn else "0")
        if step.learn:
            host.set("learn.agent", str(step.learn[0]))
            host.set("learn.show", step.learn[1])
        host.set("tour.step", f"Step {i + 1} of {len(self.steps)}")
        host.set("tour.caption", step.caption)
        self.cam_from = dict(self.view.pose or self.view.pose_for(("overview",)))
        self.cam_to = self.view.pose_for(step.view)

    def next(self):
        if self.index < 0:
            return
        step = self.steps[self.index]
        if step.until is not None:
            self.transport.seek(step.until)
        if self.index + 1 < len(self.steps):
            self.view.set_pose(self.cam_to)
            self.begin(self.index + 1)
        else:
            self._end()

    def back(self):
        if self.index >= 0:
            self.begin(max(self.index - 1, 0), seek=True)

    def restart(self):
        self.begin(0, seek=True)

    def _end(self):
        self.phase = "done"
        self.host.set("tour.step", "End of the walkthrough")
        if self.close_at_end:
            self.host.request_close()

    @property
    def active(self):
        return self.phase != "done" and self.host.get_bool("view.tour")

    def frame(self, dt, head, live=False):
        """Before the transport's tick: advance the step; False when the tour is off. While a live
        run is short of a step's end the step waits for it; a finished run ends where it ends."""
        host = self.host
        if self.phase == "done":
            return False
        if not host.get_bool("view.tour"):
            return False
        if host.get_bool("play.playing") or self.transport.dragged:
            host.set("view.tour", "0")  # the presenter took the run back
            host.set("tour.step", "Walkthrough off: Restart or Next to resume")
            return False
        step = self.steps[self.index]
        self.phase_t += dt
        if self.phase == "move":
            f = min(self.phase_t / max(step.move, 1e-6), 1.0)
            self.view.set_pose(_lerp_pose(self.cam_from, self.cam_to, f * f * (3 - 2 * f)))
            if f >= 1.0:
                self.phase, self.phase_t = ("play" if step.until is not None else "hold"), 0.0
                if self.phase == "hold":
                    self._shot()
        elif self.phase == "play":
            until = step.until if live else min(step.until, head)
            t = self.transport.t
            if t < until:
                self.transport.seek(min(until, t + dt * step.speed, head))
            if self.transport.t >= until - 1e-9:
                self.transport.seek(until)
                self.phase, self.phase_t = "hold", 0.0
                self._shot()
        elif self.phase == "hold" and self.phase_t >= step.hold:
            if self.index + 1 < len(self.steps):
                self.begin(self.index + 1)
            else:
                self._end()
        return True

    def _shot(self):
        if self.shots:
            self.want_shot = str(Path(self.shots) / f"{self.demo}_{self.index + 1:02d}.png")


class Recorder:
    """Every frame into an H.264 mp4, through ffmpeg's stdin."""

    def __init__(self, path, fps=FPS, ffmpeg=None):
        exe = ffmpeg or os.environ.get("FFMPEG") or shutil.which("ffmpeg")
        if not exe:
            raise FileNotFoundError("--record needs ffmpeg: on PATH, in $FFMPEG, or --ffmpeg")
        self.path, self.fps, self.exe, self.proc, self.frames = str(path), fps, exe, None, 0
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def write(self, rgb):
        rgb = rgb[: rgb.shape[0] // 2 * 2, : rgb.shape[1] // 2 * 2]  # yuv420p wants even sizes
        if self.proc is None:
            h, w = rgb.shape[:2]
            self.proc = subprocess.Popen(
                [self.exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "pipe:0",
                 "-c:v", "libx264", "-preset", "medium",
                 "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        try:
            self.proc.stdin.write(rgb.tobytes())
        except (BrokenPipeError, OSError):
            raise RuntimeError(
                f"ffmpeg ({self.exe}) stopped reading frames for {self.path}: its message is above"
            ) from None
        self.frames += 1

    def close(self):
        if self.proc is None:
            return 0
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self.proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited {code} writing {self.path}")
        return self.frames


def run_frames(host, args, tour=None, recorder=None, fps=FPS):
    """``_host.finish`` with a tour's shots and a recording: each frame steps ``1/fps`` of
    simulated time; a window is paced to it unless it records."""
    dt, pace, n = 1.0 / fps, host.mode == "window" and recorder is None, 0
    try:
        while not host.should_close():
            if args.frames and n >= args.frames:
                break
            t0 = time.perf_counter()
            try:
                host.step(dt)
            except KeyboardInterrupt:
                host.request_close()
                break
            if tour is not None and tour.want_shot:
                host.snapshot(tour.want_shot)
                tour.want_shot = None
            if recorder is not None:
                recorder.write(host.frame_rgb())
            n += 1
            if pace:
                rest = dt - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)
        if args.png and host.mode != "headless":
            host.snapshot(args.png)
        code = host.exit_code()
    finally:
        host.close()
        if recorder is not None:
            frames = recorder.close()
            print(f"grl_snam coph: {frames} frames ({frames / fps:.1f} s) in {recorder.path}",
                  flush=True)  # fmt: skip
    return code


# ── the port ───────────────────────────────────────────────────────────────────────────────────


def walkthrough(argv, args) -> int:
    """The crossing tour, then the hazard tour ('severe'), each in a process of its own, over
    the episodes under ``--from`` (``crossing/``, ``hazard/``); with ``--record``, one mp4."""
    root = Path(args.source or "runs/coph")
    passthrough, skip = [], {"walkthrough"}
    it = iter(argv)
    for a in it:  # drop the demo, --from and --record; pass the rest (window flags, --shots)
        if a in skip:
            continue
        if a in ("--from", "--record", "--scene"):
            next(it, None)
            continue
        if a.split("=")[0] in ("--from", "--record", "--scene"):
            continue
        passthrough.append(a)
    parts, code = [], 0
    with tempfile.TemporaryDirectory(prefix="coph-walkthrough-") as tmp:
        for demo in ("crossing", "hazard"):
            cmd = [sys.executable, "-m", "grl_snam.ariadne", "coph", demo, "--tour",
                   "--from", str(root / demo), *passthrough]  # fmt: skip
            if demo == "hazard":
                cmd += ["--scene", "severe"]
            if args.record:
                parts.append(Path(tmp) / f"{demo}.mp4")
                cmd += ["--record", str(parts[-1])]
            code = subprocess.call(cmd)
            if code:
                return code
        if args.record:
            listing = Path(tmp) / "parts.txt"
            listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
            exe = args.ffmpeg or os.environ.get("FFMPEG") or shutil.which("ffmpeg")
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            code = subprocess.call([exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                                    "-i", str(listing), "-c", "copy", str(args.record)])  # fmt: skip
            if not code:
                print(f"grl_snam coph: the walkthrough is in {args.record}", flush=True)
    return code


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _host.parse(argv, _args)
    sized = any(a.split("=")[0] in ("--width", "--height") for a in argv)
    if not sized:
        args.width, args.height = DEFAULT_SIZE
    if args.demo == "walkthrough":
        return walkthrough(argv, args)
    if args.mode != "window" and not any(a.split("=")[0] == "--frames" for a in argv):
        args.frames = 0  # headless and offscreen run to the end, not the host's default 30
    demo = args.demo
    arms = tuple(a.strip() for a in args.arms.split(",")) if args.arms else feed.DEFAULT_ARMS[demo]
    if len(arms) != 2 or arms[0] == arms[1]:
        print("grl_snam coph: --arms takes two different arms", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else feed.DEFAULT_SEEDS[demo]
    steps = args.steps or feed.DEFAULT_STEPS[demo]
    if args.tour and (arms != feed.DEFAULT_ARMS[demo] or seed != feed.DEFAULT_SEEDS[demo]):
        print("grl_snam coph: --tour follows the default arms and seed", file=sys.stderr)
        return 2
    if args.tour and args.mode == "headless":
        print("grl_snam coph: --tour draws: drop --headless", file=sys.stderr)
        return 2
    try:
        for arm in arms:
            feed.job(demo, arm, seed, steps, args.scene)
        if args.tour:
            root = args.source or args.out or str(Path("runs/coph") / demo)
            ensure_episodes(demo, arms, seed, steps, args.scene, root)
            source = Source.replay(demo, arms, seed, args.scene, root)
        elif args.source:
            source = Source.replay(demo, arms, seed, args.scene, args.source)
        else:
            source = Source.live(demo, arms, seed, steps, args.scene, args.out)
        recorder = Recorder(args.record, ffmpeg=args.ffmpeg) if args.record else None
    except (ValueError, FileNotFoundError) as e:
        print(f"grl_snam coph: {e}", file=sys.stderr)
        return 2
    title = TITLES[demo] + (f" ('{args.scene}')" if demo == "hazard" else "")
    try:
        host = _host.open_host("coph.ari", "coph", args)
        drawn = host.mode != "headless"
        view = View(host, demo, source.timelines, (args.width, args.height)) if drawn else None
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
        tour = None
        if args.tour:
            tour = Tour(
                host,
                view,
                transport,
                tour_steps(demo, source.timelines, seed, args.scene),
                demo=demo,
                shots=args.shots,
                close_at_end=bool(args.record or args.shots or host.mode != "window"),
            )
        else:
            host.set("view.tour", "0")

        @host.on_frame
        def _frame(dt):
            if not source.poll() and not drawn and source.live_running:
                time.sleep(0.02)  # headless frames are not paced: wait for the workers
            if source.error:
                print(f"grl_snam coph: {source.error}", file=sys.stderr, flush=True)
                host.request_close()
                return
            head = source.head()
            if tour is not None:
                tour.frame(dt, head, source.live_running)
            t = transport.tick(dt, head, source.live_running)
            publish(host, demo, source, t, seed, args.scene)
            panels.publish(source.timelines, t, source.finished and not source.live_running)
            if view is not None:
                view.draw(t, camera)
            if not drawn and source.finished and not source.live_running:
                host.request_close()

        if tour is not None or recorder is not None:
            code = run_frames(host, args, tour, recorder)
        else:
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
