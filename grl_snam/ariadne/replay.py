"""grl-snam ariadne replay: a recorded ``live --arms`` run, played back.

    python -m grl_snam.ariadne live austin-freedrive --plant kinematic --arms trained,untrained,constant --log run.npz
    python -m grl_snam.ariadne replay run.npz
    python -m grl_snam.ariadne replay run.npz --at 12 --offscreen --frames 1 --png at12.png

The run log (:mod:`grl_snam.learning.runlog`) holds every arm's pose, drive telemetry, books, Stats
row and plant ledger, tick by tick. The replay rebuilds the recorded world from the log's ``run``
and ``world`` keys, draws each arm's ghost and track where the arm was at the play head, and
writes the Stats and Ledger windows (``docs/stats_panels.ari``) as the live run showed them at
that tick. It writes them with the live run's own code (:func:`grl_snam.ariadne._arms.panel_tables`,
:func:`~grl_snam.ariadne._arms.chart_series`, :func:`grl_snam.ariadne.live.update_ghosts`), so
what a replay shows at a tick is what the live run showed there.

The document (``docs/replay.ari``) has libcvc's sim transport: Paused, Speed, Step (one tick on,
while paused) and Restart. Beside it are a Position slider that scrubs, Loop and the camera. The
charts draw the whole run with a cursor at the play head, and a click on one seeks there.
``--at SECONDS`` starts the play head there (``--paused`` holds it). Headless, nothing is drawn and no world is built:
the play head runs ``--frames`` frames over the panels (the rows only). The exit code is 0, and 2
for a file that is not a run log.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from grl_snam.ariadne import _arms, _cars, _host
from grl_snam.ariadne.live import publish_charts, publish_tables, update_ghosts
from grl_snam.learning.runlog import RunLog


def _args(p):
    p.add_argument("log", help="a run log (grl_snam.run_log/v1): live --arms ... --log run.npz")
    p.add_argument("--at", type=float, default=0.0, help="start the play head here (s)")
    p.add_argument("--loop", action="store_true", help="play it again from the start at the end")
    p.add_argument("--paused", action="store_true", help="start paused (at --at)")
    p.add_argument("--camera", choices=("follow", "orbit"), default=None,
                   help="follow the arm the Stats window picks (the default), or orbit the world")  # fmt: skip


def _refuse(msg) -> int:
    print(f"grl-snam ariadne replay: {msg}", file=sys.stderr)
    return 2


class PlayHead:
    """The play head over a run of ``duration`` s with ticks of ``tick`` s, driven by the
    document's transport: ``sim.paused`` and ``sim.speed``; ``sim.step`` moves it a tick on and
    ``sim.restart`` back to the start; ``play.pos`` (0..1) follows it, and moves it when dragged;
    ``play.loop`` wraps it at the end, else it stops there; a chart's click (``play.seek``, then
    ``arms_seek``) puts it at the time clicked. Each frame :meth:`tick` writes ``play.pos``,
    ``play.t`` (s, the charts' cursor) and ``play.time``, and returns the time."""

    def __init__(self, host, duration: float, tick: float, at: float = 0.0):
        self.host = host
        self.duration = max(float(duration), 1e-6)
        self.dt = float(tick)
        self.t = self._clamp(at)
        self._pos = None  # the position this play head last wrote
        host.on("sim.step", self.step)
        host.on("sim.restart", self.restart)
        host.on("arms_seek", self.seek)

    def _clamp(self, t) -> float:
        return min(max(float(t), 0.0), self.duration)

    def step(self):
        self.t = self._clamp(self.t + self.dt)

    def restart(self):
        self.t = 0.0
        self.host.set("sim.paused", "0")

    def seek(self):
        t = self.host.get_float("play.seek")
        if math.isfinite(t):
            self.t = self._clamp(t)

    def tick(self, dt: float) -> float:
        host = self.host
        pos = host.get_float("play.pos", 0.0)
        if self._pos is not None and abs(pos - self._pos) > 1e-6:  # dragged: the slider wins
            self.t = self._clamp(pos * self.duration)
        elif self._pos is not None and not host.get_bool("sim.paused"):
            self.t += dt * max(host.get_float("sim.speed", 1.0), 0.0)
            if self.t > self.duration:
                if host.get_bool("play.loop"):
                    self.t = 0.0
                else:
                    self.t = self.duration
                    host.set("sim.paused", "1")
        host.set("play.pos", round(self.t / self.duration, 6))
        self._pos = host.get_float("play.pos", 0.0)  # as the state text reads back
        host.set("play.t", f"{self.t:.4f}")
        host.set("play.time", f"{self.t:6.1f} / {self.duration:.1f} s")
        return self.t


def _pick(host, key, names) -> int:
    """The index of the arm the combo at ``key`` names (the first when it names none)."""
    v = host.get(key, "")
    return names.index(v) if v in names else 0


def arms_of(log: RunLog) -> list[_arms.Arm]:
    """The log's arms, as :class:`grl_snam.ariadne._arms.Arm` s."""
    fields = set(_arms.Arm.__dataclass_fields__)
    return [_arms.Arm(**{k: v for k, v in a.items() if k in fields}) for a in log.meta["arms"]]


def tables_at(log: RunLog, arms, k: int) -> dict:
    """The Stats and Ledger windows' tables at tick ``k``, as the live run wrote them there."""
    vals = [log.at(k, j) for j in range(len(arms))]
    stats = [{c: v["stats"].get(c, math.nan) for c in _arms.STATS} for v in vals]
    return _arms.panel_tables(arms, stats, [v["tel"] for v in vals],
                              [v["book_sum"] for v in vals], [v["book"] for v in vals])  # fmt: skip


def charts_of(log: RunLog, arms, agent: int, book: str) -> dict:
    """The charts over the whole run (:func:`grl_snam.ariadne._arms.chart_series`)."""
    scale = float(log.meta.get("scale", _arms.POLICY_SCALE))
    clearance = log.arrays.get("tel_clearance", np.full((len(log), len(arms)), np.nan))
    sums = {b: log.cumulative(f"book_{b}") for b in _arms.BOOKS}
    legs = [float(k) * log.dt for k in log.arrays.get("leg_tick", [0])]
    return _arms.chart_series(arms, log.t, clearance / scale, sums, legs, agent, book)


def poses_at(log: RunLog, t: float):
    """Every arm's (x, y, heading) at time ``t``, between the ticks either side of it."""
    a, n = log.arrays, len(log)
    k = int(np.searchsorted(log.t, t, side="right")) - 1
    if k < 0:
        return [(a["x"][0, j], a["y"][0, j], a["heading"][0, j]) for j in range(a["x"].shape[1])]
    k1 = min(k + 1, n - 1)
    f = 0.0 if k1 == k else min(max((t - log.t[k]) / (log.t[k1] - log.t[k]), 0.0), 1.0)
    out = []
    for j in range(a["x"].shape[1]):
        h0, h1 = float(a["heading"][k, j]), float(a["heading"][k1, j])
        dh = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
        out.append((float(a["x"][k, j]) + f * float(a["x"][k1, j] - a["x"][k, j]),
                    float(a["y"][k, j]) + f * float(a["y"][k1, j] - a["y"][k, j]), h0 + f * dh))  # fmt: skip
    return out


def main(argv=None) -> int:
    args = _host.parse(argv, _args)
    try:
        log = RunLog.load(args.log)
    except (OSError, ValueError) as e:
        return _refuse(str(e))
    if not len(log) or not log.meta.get("arms"):
        return _refuse(f"{args.log}: the run has no ticks")
    drawn = args.mode != "headless"
    if drawn and not _cars.HAS_CARS:
        return _refuse(f"drawing the recorded world needs {_cars.INSTALL}")
    arms = arms_of(log)
    names = [a.name for a in arms]
    meta = log.meta
    host = _host.open_host("replay.ari", "replay", args, load=False)
    _host.load_document(host, "replay.ari", city=drawn, worlds=drawn)
    run_keys = {k: v for k, v in (meta.get("run") or {}).items() if v not in (None, "")}
    host.set_many(run_keys, prefix="run")
    host.set_many({k: v for k, v in (meta.get("world_keys") or {}).items() if v not in (None, "")},
                  prefix="world")  # fmt: skip
    host.set_many({"names": "\n".join(names), "status": ""}, prefix="arms")
    host.set("view.arm", names[0])
    host.set("ledger.agent", names[0])
    if args.loop:
        host.set("play.loop", "1")
    if args.paused:
        host.set("sim.paused", "1")
    if args.camera:
        host.set("view.camera", args.camera)
    prov = meta.get("provenance") or {}
    host.set("replay.title", f"{meta.get('demo', 'live')} on {meta.get('plant', '?')}: "
                             f"{len(arms)} arms, {', '.join(names)}; seed {meta.get('seed')}")  # fmt: skip
    made = prov.get("grl_snam_commit") or ""
    made = f"GRL-SNAM {made[:10]}" if made else f"GRL-SNAM {prov.get('grl_snam', '?')}"
    host.set("replay.made", f"{made}, libcvc {prov.get('libcvc') or '?'}; {len(log)} ticks of "
                            f"{log.dt:g} s, {len(log.arrays.get('leg_tick', [0]))} legs")  # fmt: skip
    plant = (meta.get("settings") or {}).get("nav_plant.driving") or meta.get("plant", "")
    world = None
    st = {"nodes": [], "tracks": [[] for _ in arms], "camera": None, "modeled": None,
          "charts": None}  # fmt: skip
    if drawn:
        C = _cars.C
        kind = run_keys.get("world", "data")
        world = C.make_world(host, kind)
        C.show_world(host, kind, str(_host.DOCS / "replay.ari"))
        C.draw_world(host, {"world": world})
        start = meta.get("start") or (log.arrays["x"][0, 0], log.arrays["y"][0, 0])
        host.lab.add_markers("start", [(start[0], start[1], world.z(*start) + 1.0)],
                             color=(0.15, 0.85, 0.25))  # fmt: skip
    head = PlayHead(host, log.duration, log.dt, args.at)
    charts = hasattr(host, "set_plot")
    a = log.arrays
    every = max(1, len(log) // 600)

    @host.on_frame
    def frame(dt):
        t = head.tick(dt)
        k = log.index(t)
        leg = log.leg(k)
        j = _pick(host, "ledger.agent", names)
        rows = _arms.plant_ledger_rows(log.at(k, j)["plant"])
        publish_tables(host, tables_at(log, arms, k), rows, names[j], plant)
        host.set("arms.time", f"t = {float(log.t[k]):.1f} s, tick {k + 1} of {len(log)}, "
                              f"leg {leg + 1}")  # fmt: skip
        book = host.get("ledger.book", "dU")
        if charts and st["charts"] != (j, book):
            publish_charts(host, charts_of(log, arms, j, book))
            st["charts"] = (j, book)
        if world is not None:
            for i in range(len(arms)):  # each track: the recorded path up to the play head
                st["tracks"][i] = list(zip(a["x"][: k + 1 : every, i].tolist(),
                                           a["y"][: k + 1 : every, i].tolist(), strict=True))  # fmt: skip
            sizes = [(a["length"][k, i], a["width"][k, i], a["wheelbase"][k, i])
                     for i in range(len(arms))]  # fmt: skip
            goal = (float(a["goal_x"][k, 0]), float(a["goal_y"][k, 0]))
            follow = _pick(host, "view.arm", names)
            update_ghosts(host, st, world, arms, poses_at(log, t), sizes, goal, follow)

    host.run(fps=30, frames=args.frames or None)
    if args.png and drawn:
        host.snapshot(args.png)
    code = host.exit_code()
    host.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
