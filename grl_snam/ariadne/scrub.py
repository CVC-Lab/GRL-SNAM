"""grl-snam ariadne scrub: watch it learn. Every checkpoint of a training run, deployed.

    python -m grl_snam.ariadne scrub run/history.jsonl                          # Austin, kinematic
    python -m grl_snam.ariadne scrub run/history.jsonl --world flat --ticks 600
    python -m grl_snam.ariadne scrub run/history.jsonl --headless --report scrub.json
    python -m grl_snam.ariadne scrub --view scrub.json [more.json ...]       # view only

A learning history (:mod:`grl_snam.learning.history`) names the policy each checkpoint saved, its
``blob`` (a ``.cvcnav`` or a ``.pt``; a relative path is the history's directory's). The scrubber
deploys every one of them on one scenario: libcvc's sim_world on ``--plant`` (kinematic by
default), from the scenario's start to its goal, for ``--ticks`` ticks. It drives them as
``live --arms`` does, tick for tick (:func:`grl_snam.ariadne.live.make_fleet`,
:func:`~grl_snam.ariadne.live.tick_arms`), so checkpoint 0 deployed here is the untrained arm
deployed there.

Each run is a run log (:mod:`grl_snam.learning.runlog`), cached at
``<history>.scrub/<blob sha256>-<scenario hash>.npz``. The scenario is the world (its ``run.*``
keys, and a generated world's ``world.*``), the start and goal, the seed, the ticks, the plant, the policy's frame,
the settings set on the command line, and libcvc's version, so any of them changing runs the
checkpoints again.

The view (``docs/scrub.ari``):
- **Checkpoint k** is picked by a combo, a slider, Previous and Next, or a click on a chart. Its
  ghost runs beside step 0's (the first checkpoint) over the episode, with the Stats and Ledger
  windows for the two.
- **A deployed metric** (Metric picks it) against the checkpoint's step, with step 0's as a
  dashed line, and the same as a table.
- **A held-out curve** from the history, with the learn port's marks.

Both charts carry a cursor at checkpoint k. ``--report R.json`` writes every checkpoint's metrics
(``grl_snam.scrub/v1``), its run log's file and the scenario. ``--blobs DIR`` looks for the saved
policies there (by file name) before beside the history, and ``--cache DIR`` caches the run logs
there in place of ``<history>.scrub/``. ``--data-set NAME`` names the run's data set in the report (the
learning check's ``retrain``, ``fixed/retrain``, ...), and ``--lean`` caches lean run logs
(:meth:`grl_snam.learning.runlog.RunLog.save`: single precision, about 40% smaller) for a page.

**View only.** ``--view R.json ...`` shows runs already deployed: it reads the reports, the run log
each checkpoint lists and the history each report names, and deploys nothing, so it needs neither
torch nor the policies, and no sim_world steps. A file a report names is found as given, then
relative to the report, then by its file name beside the report, so a report travels with its
files (the Pages site's scrub page). It does not hash the scenario again: the libcvc in use, the
world and the files' places may differ from the ones that deployed. With several reports a Run
combo picks one, and a Set combo their data set when they come from several: a set picked shows
its run of the same name at the same training step, so the sets compare side by side. Their
worlds must be the same. The world is rebuilt from the report's ``run`` and
``world`` keys (a generated world from its generator settings, which the deploy's libcvc made:
another libcvc's may differ, and the view says so); ``--no-world``, a city that is not on this
machine in a browser, or a pycvc_gl without the multi-car template, shows the windows alone (no
3-D world). A run log is read when its checkpoint is first shown.

The exit code is 0, and 2 when the history names no saved policy or the plant is torch (the
scrubber deploys on libcvc's sim_world), or when a report or a file it names is not there.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from grl_snam.ariadne import _arms, _cars, _host

REPORT_SCHEMA = "grl_snam.scrub/v1"
#: Bumped when the scrubber's runs would change, so older caches are not read.
SCRUB_VERSION = 1
#: docs/scrub.ari's windows, by id.
WINDOWS = ("scrub", "training", "episode", "sim_transport", "arms_stats", "arms_ledger")
#: The run logs a view keeps read at once (step 0's is always kept).
KEEP_LOGS = 8
#: The deployed metrics, from each run's last tick: name -> (label, run log array, digits).
METRICS = {
    "reached": ("reached the goal", "stats_reached", 0),
    "time_s": ("time (s)", "stats_elapsed_s", 1),
    "path_m": ("path (m)", "stats_path_m", 1),
    "pen_steps": ("ticks penetrating", "stats_pen_steps", 0),
    "clear_min_m": ("clearance min (m)", "stats_clear_min_m", 2),
    "clear_mean_m": ("clearance mean (m)", "stats_clear_mean_m", 2),
    "goal_m": ("left to the goal (m)", "goal_dist", 1),
}


def _args(p):
    p.add_argument("history", nargs="?", default=None,
                   help="a learning history (.jsonl or .jsonl.gz) whose checkpoints name their "
                        "saved policy (blob)")  # fmt: skip
    p.add_argument("--view", nargs="+", default=None, metavar="REPORT",
                   help="view only: runs already deployed, as these reports (--report's) list "
                        "them; deploys nothing")  # fmt: skip
    p.add_argument(
        "--blobs",
        default=None,
        help="a directory to find the saved policies in (by file name) first",
    )
    p.add_argument(
        "--cache", default=None, help="cache the run logs here (default <history>.scrub)"
    )
    p.add_argument("--no-world", action="store_true", help="the windows alone: no 3-D world")
    p.add_argument("--data-set", default="", metavar="NAME",
                   help="the data set the run belongs to, in the report (e.g. retrain): a view of "
                        "several sets picks one with a Set combo")  # fmt: skip
    p.add_argument("--lean", action="store_true",
                   help="cache lean run logs (single precision, about 40%% smaller), for a page")  # fmt: skip
    p.add_argument("--camera", choices=("follow", "orbit"), default=None,
                   help="follow checkpoint k's ghost (the default), or orbit the world")  # fmt: skip
    p.add_argument("--windows", default=None, metavar="ID,...",
                   help="the windows open at the start (default all: " + ",".join(WINDOWS)
                        + "); the others are in the Windows menu")  # fmt: skip
    p.add_argument("--run", type=int, default=-1, help="which run in the file (default the last)")
    p.add_argument("--plant", default=None, help="nav_plant.vehicle (default kinematic)")
    p.add_argument("--ticks", type=int, default=1200, help="each checkpoint's ticks (0.05 s each)")
    p.add_argument("--seed", type=int, default=0, help="the scenario's seed")
    p.add_argument("--scale", type=float, default=None,
                   help="the policies' frame (normalized units per metre; default: the first "
                        ".pt checkpoint's, else 1/300)")  # fmt: skip
    p.add_argument("--bundle", default=None, help="a local Austin bundle: the same as --uri DIR")
    p.add_argument("--report", default=None, help="write every checkpoint's metrics (JSON)")
    p.add_argument("--at", type=int, default=None, help="start at checkpoint k (default the pick)")
    if _cars.HAS_CARS:
        _cars.C.add_args(p)


def _refuse(msg) -> int:
    print(f"grl-snam ariadne scrub: {msg}", file=sys.stderr)
    return 2


# ── the checkpoints and the scenario (no host) ─────────────────────────────────────────────────


def resolve_blob(history, blob: str, blobs=None) -> Path | None:
    """The file ``blob`` names: its file name in ``blobs`` (a directory) when given and there,
    else as given when absolute, else beside the history, else from the working directory, else
    its file name beside the history. None when none exists."""
    p = Path(blob)
    here = Path(history).resolve().parent
    first = [Path(blobs) / p.name] if blobs else []
    for c in first + ([p] if p.is_absolute() else [here / p, Path.cwd() / p, here / p.name]):
        if c.is_file():
            return c.resolve()
    return None


def checkpoints(history, h, blobs=None) -> tuple[list[dict], list[str]]:
    """Run ``h``'s checkpoints that saved a policy, in order: ``{step, blob, kind, best,
    selected, fitness}`` (``blob`` the file), and the blobs named that are not there. ``blobs``
    is a directory to look in first (:func:`resolve_blob`)."""
    rows, missing = [], []
    for c in h.checkpoints():
        if not c.blob:
            continue
        path = resolve_blob(history, c.blob, blobs)
        if path is None:
            missing.append(c.blob)
            continue
        if path.suffix not in (".cvcnav", ".pt"):
            continue
        rows.append({"step": float(c.x), "axis": c.axis, "blob": str(path), "kind": c.kind,
                     "best": bool(c.best), "selected": bool(c.selected), "fitness": c.fitness})  # fmt: skip
    return rows, missing


def scenario_hash(scenario: dict) -> str:
    """A short, stable hash of ``scenario`` (a JSON-able dict)."""
    text = json.dumps(scenario, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cache_path(history, blob_sha: str, shash: str, cache=None, lean: bool = False) -> Path:
    """Where a checkpoint's run log is cached: ``cache`` (a directory), else ``<history>.scrub``;
    a lean one (:meth:`grl_snam.learning.runlog.RunLog.save`) as ``.lean.npz``."""
    return (
        Path(cache or f"{history}.scrub") / f"{blob_sha[:16]}-{shash}{'.lean' if lean else ''}.npz"
    )


def metrics_of(log) -> dict:
    """The deployed metrics (:data:`METRICS`) at a run log's last tick (its one arm)."""
    out = {}
    for name, (_, key, _) in METRICS.items():
        a = log.arrays.get(key)
        out[name] = float(a[-1, 0]) if a is not None and len(a) else math.nan
    return out


def _cell(v, digits) -> str:
    if v is None or not math.isfinite(v):
        return "-"
    if digits == 0:
        return str(int(round(v)))
    return f"{v:.{digits}f}"


def table_rows(rows, metrics, k: int) -> list[dict]:
    """The checkpoint table: one row per checkpoint, ``k``'s marked."""
    out = []
    for i, (r, m) in enumerate(zip(rows, metrics, strict=True)):
        flag = ("pick" if r["selected"] else "") or ("best" if r["best"] else "")
        cells = {name: _cell(m[name], d) for name, (_, _, d) in METRICS.items()}
        cells["reached"] = "yes" if m["reached"] >= 0.5 else "no"
        out.append({"at": ">" if i == k else "", "step": f"{r['step']:g}", "flag": flag, **cells})
    return out


def metric_chart(rows, metrics, name: str):
    """``name`` against the checkpoint's step, with step 0's as a dashed line and the pick and the
    held-out bests marked: (series, markers, text)."""
    label, _, digits = METRICS.get(name, METRICS["path_m"])
    x = np.asarray([r["step"] for r in rows], float)
    y = np.asarray([m[name] for m in metrics], float)
    series = [{"x": x, "y": y, "label": label, "style": "c1"},
              {"x": x, "y": np.full_like(y, y[0]), "label": "step 0", "style": "c0 dashed"}]  # fmt: skip
    markers = [(r["step"], "pick", "best") for r in rows if r["selected"]]
    fin = np.isfinite(y)
    text = f"{label}: step 0 {_cell(y[0], digits)}"
    if fin.any():
        yy = np.where(fin, y, np.nan)
        lo, hi = int(np.nanargmin(yy)), int(np.nanargmax(yy))
        text += (f"; lowest {_cell(y[lo], digits)} (step {x[lo]:g}), highest "
                 f"{_cell(y[hi], digits)} (step {x[hi]:g})")  # fmt: skip
    return series, markers, text


def stack(logs, names):
    """One run log of several one-arm logs of the same ticks (the arms side by side, named
    ``names``, in colour slots 0, 1, ...)."""
    from grl_snam.learning.runlog import RunLog

    first = logs[0]
    arrays = {}
    for k, v in first.arrays.items():
        if v.ndim == 2 and v.shape[1] == 1 and k not in ("leg_goal",):
            arrays[k] = np.concatenate([lg.arrays[k] for lg in logs], axis=1)
        else:
            arrays[k] = v
    meta = dict(first.meta)
    meta["arms"] = [{**lg.meta["arms"][0], "name": n, "slot": j}
                    for j, (lg, n) in enumerate(zip(logs, names, strict=True))]  # fmt: skip
    return RunLog(arrays, meta)


# ── deploying a checkpoint ─────────────────────────────────────────────────────────────────────


def run_checkpoint(host, world, plant, row, scenario, shash, history, work):
    """Deploy one checkpoint (``row``) on the scenario: a run log of its one arm."""
    from grl_snam.ariadne import live
    from grl_snam.learning.runlog import RunLog

    C = _cars.C
    arm = _arms.Arm(name=f"step {row['step']:g}", spec=row["blob"],
                    kind="cvcnav" if row["blob"].endswith(".cvcnav") else "pt")  # fmt: skip
    _arms.resolve([arm], work, seed=scenario["seed"])
    start, goal = tuple(scenario["start"]), tuple(scenario["goal"])
    fleet = live.make_fleet(world, plant, [arm], start, goal, seed=scenario["seed"],
                            scale=scenario["scale"], app=host.app, prefix=host.prefix)  # fmt: skip
    live.begin_stats(fleet, [arm], world.kind, scenario["seed"])
    run = _arms.ArmsRun([arm], start=start, goal=goal, seed=scenario["seed"], dt=C.DT,
                        scale=scenario["scale"], world=world.info, plant=plant, demo="scrub")  # fmt: skip
    for _ in range(int(scenario["ticks"])):
        live.tick_arms(fleet, run)
    return RunLog.from_run(run, history=str(history), checkpoint=row, scenario=scenario,
                           scenario_hash=shash)  # fmt: skip


def deploy_all(host, world, plant, rows, scenario, history, *, cache=None, lean=False, log=print):
    """Every checkpoint's run log, from the cache (``cache``, else ``<history>.scrub``) or deployed
    now (and cached; ``lean`` caches lean logs)."""
    from grl_snam.learning.runlog import RunLog

    shash = scenario_hash(scenario)
    work = tempfile.mkdtemp(prefix="grl-snam-scrub-")
    out = []
    for i, row in enumerate(rows):
        sha = _arms._sha256(row["blob"])
        path = cache_path(history, sha, shash, cache, lean)
        try:
            lg = RunLog.load(path)
            how = "cached"
        except (OSError, ValueError):
            lg = run_checkpoint(host, world, plant, row, scenario, shash, history, work)
            lg.save(path, lean=lean)
            how = "deployed"
        row["sha256"], row["cache"] = sha, str(path)
        m = metrics_of(lg)
        log(f"scrub: step {row['step']:g} ({i + 1}/{len(rows)}), {how}: reached "
            f"{'yes' if m['reached'] >= 0.5 else 'no'}, path {m['path_m']:.1f} m, "
            f"{m['pen_steps']:.0f} ticks penetrating, {m['goal_m']:.1f} m to go")  # fmt: skip
        out.append(lg)
    return out


# ── the report, and runs read back ─────────────────────────────────────────────────────────────


def find_file(report, name) -> Path | None:
    """The file a report names: as given when absolute, else relative to the report, else from the
    working directory, else its file name beside the report. None when none exists."""
    if not name:
        return None
    p = Path(name)
    here = Path(report).resolve().parent
    for c in [p] if p.is_absolute() else [here / p, Path.cwd() / p]:
        if c.is_file():
            return c
    return here / p.name if (here / p.name).is_file() else None


def write_report(path, history, run, rows, metrics, scenario, provenance=None, set_name="") -> dict:
    """``--report``'s JSON (``grl_snam.scrub/v1``): the history, its data set (``set``, "" for
    none), the scenario and its hash, what deployed it (``provenance``: the GRL-SNAM commit,
    libcvc's version), and every checkpoint's row (its run log is ``cache``) with its metrics."""
    prov = {k: v for k, v in (provenance or {}).items() if k != "history"}
    rep = {"schema": REPORT_SCHEMA, "history": str(history), "set": set_name, "run": run,
           "scenario": scenario,
           "scenario_hash": scenario_hash(scenario), "provenance": prov,
           "checkpoints": [{**r, "metrics": m} for r, m in zip(rows, metrics, strict=True)]}  # fmt: skip
    Path(path).write_text(json.dumps(rep, indent=1, default=str))
    return rep


class Deployed:
    """One history's checkpoints, deployed on one scenario: their rows and metrics, the scenario,
    the history (read when first asked for) and each checkpoint's run log (a :class:`RunLog`, or
    a file read when first shown; :data:`KEEP_LOGS` are kept read)."""

    def __init__(self, history, run, rows, metrics, scenario, logs, *, h=None, provenance=None,
                 set_name=""):  # fmt: skip
        from grl_snam.ariadne import learn

        self.history, self.run, self.rows, self.metrics = history, run, rows, metrics
        self.set = set_name or ""
        self.scenario, self.provenance, self._h = scenario, provenance or {}, h
        self._files = list(logs)
        self._read: dict[int, object] = {}
        self.name = learn.run_label(self.h, 0) if h is not None else _stem(history)

    @property
    def h(self):
        if self._h is None:
            from grl_snam.learning import load

            self._h = load(str(self.history))[self.run]
        return self._h

    @property
    def pick(self) -> int:
        """The history's pick (its selected checkpoint), else the last."""
        return next((i for i, r in enumerate(self.rows) if r["selected"]), len(self.rows) - 1)

    def log(self, i: int):
        from grl_snam.learning.runlog import RunLog

        if i in self._read:
            self._read[i] = self._read.pop(i)  # the most recent last
            return self._read[i]
        lg = self._files[i]
        lg = lg if isinstance(lg, RunLog) else RunLog.load(lg)
        self._read[i] = lg
        while len(self._read) > KEEP_LOGS:
            del self._read[next(j for j in self._read if j != 0)]
        return lg

    @property
    def label(self) -> str:
        """The run's name, its data set's first (``retrain/seed0``) when it has one."""
        return f"{self.set}/{self.name}" if self.set else self.name

    def title(self) -> str:
        r = self.rows
        return f"{self.label}: {len(r)} checkpoints, steps {r[0]['step']:g} to {r[-1]['step']:g}"

    def scenario_text(self) -> str:
        sc = self.scenario
        kind = (sc.get("run") or {}).get("world") or "data"
        (sx, sy), (gx, gy) = sc["start"], sc["goal"]
        return (f"{kind} on {sc.get('plant', '?')}, {sc.get('ticks')} ticks from ({sx:.0f}, {sy:.0f}) "
                f"to ({gx:.0f}, {gy:.0f}), seed {sc.get('seed')}")  # fmt: skip

    def made_text(self) -> str:
        p = self.provenance
        if not p:
            return ""
        made = p.get("grl_snam_commit") or ""
        made = f"GRL-SNAM {made[:10]}" if made else f"GRL-SNAM {p.get('grl_snam', '?')}"
        return f"deployed with {made}, libcvc {p.get('libcvc') or '?'}"


def _stem(history) -> str:
    base = Path(str(history)).name
    for ext in (".gz", ".jsonl", ".json", ".history"):
        base = base[: -len(ext)] if base.endswith(ext) else base
    return base


def load_report(path) -> Deployed:
    """The runs a report (``--report``'s) lists, to view: its history and every checkpoint's run
    log found (:func:`find_file`), nothing read yet but the report. Raises ValueError when it is
    not a scrub report or a file it names is not there, OSError when it cannot be read."""
    rep = json.loads(Path(path).read_text())
    if not isinstance(rep, dict) or rep.get("schema") != REPORT_SCHEMA:
        schema = rep.get("schema") if isinstance(rep, dict) else None
        raise ValueError(f"{path} is {schema!r}, not a scrub report ({REPORT_SCHEMA})")
    hist = find_file(path, rep.get("history"))
    if hist is None:
        raise ValueError(f"{path}: its history {rep.get('history')!r} is not there")
    cks = rep.get("checkpoints") or []
    if not cks:
        raise ValueError(f"{path}: no checkpoints")
    rows, metrics, logs = [], [], []
    for c in cks:
        f = find_file(path, c.get("cache"))
        if f is None:
            raise ValueError(
                f"{path}: step {c.get('step')}'s run log {c.get('cache')!r} is not there"
            )
        rows.append({k: v for k, v in c.items() if k != "metrics"})
        got = c.get("metrics") or {}
        metrics.append({n: float(got[n]) if got.get(n) is not None else math.nan for n in METRICS})
        logs.append(f)
    return Deployed(hist, int(rep.get("run", -1)), rows, metrics, rep.get("scenario") or {}, logs,
                    provenance=rep.get("provenance"), set_name=str(rep.get("set") or ""))  # fmt: skip


# ── the view ───────────────────────────────────────────────────────────────────────────────────


class ScrubView:
    """Checkpoint k beside step 0: the controls that pick k, the metric and training charts, the
    table, and the two runs over the episode (the replay's play head, tables and ghosts). ``runs``
    are :class:`Deployed` s: a Set combo picks their data set when they have several, and a Run
    combo a run of it. ``world`` is the world to draw them on (None: the windows alone)."""

    def __init__(self, host, runs, world, k=None):
        from grl_snam.ariadne.replay import PlayHead

        self.host, self.runs, self.world = host, list(runs), world
        self.charts = hasattr(host, "set_plot")
        self.st = {"nodes": [], "tracks": [[], []], "camera": None, "modeled": None,
                   "charts": None}  # fmt: skip
        self.sets = list(dict.fromkeys(r.set for r in self.runs))
        self.by_set = {s: [i for i, r in enumerate(self.runs) if r.set == s] for s in self.sets}
        self.run_names = [""] * len(self.runs)  # unique in their set
        for idx in self.by_set.values():
            names = [self.runs[i].name for i in idx]
            for j, i in enumerate(idx):
                self.run_names[i] = f"{names[j]} ({j})" if names.count(names[j]) > 1 else names[j]
        host.set_many({"set_labels": "\n".join(self.sets), "set_count": str(len(self.sets))},
                      prefix="scrub")  # fmt: skip
        named = ", ".join(s or "(none)" for s in self.sets)
        print(f"scrub: {len(self.runs)} runs in {len(self.sets)} data sets ({named})",
              file=sys.stderr, flush=True)  # fmt: skip
        self.head = PlayHead(host, self.runs[0].log(0).duration, self.runs[0].log(0).dt)
        host.on("scrub_prev", lambda: self.pick(self.k - 1))
        host.on("scrub_next", lambda: self.pick(self.k + 1))
        host.on("scrub_pick", self.pick_step)
        self.use(0, k)

    # the runs and the checkpoint

    @property
    def rows(self):
        return self.r.rows

    @property
    def metrics(self):
        return self.r.metrics

    def use(self, i: int, k=None) -> None:
        """Show run ``i``: its checkpoint ``k`` (default its pick) beside its step 0."""
        from grl_snam.ariadne import learn

        host = self.host
        self.i, self.r = i, self.runs[i]
        in_set = [self.run_names[j] for j in self.by_set[self.r.set]]
        self.labels = [f"step {r['step']:g}" + (" (pick)" if r["selected"] else "")
                       for r in self.rows]  # fmt: skip
        self.k, self._shown, self._pos, self._choice = -1, None, None, None
        self._metric = self._curve = None
        names = learn.heldout_names([self.r.h])
        host.set_many({"labels": "\n".join(self.labels), "curve_names": "\n".join(names),
                       "run_labels": "\n".join(in_set), "run_count": str(len(in_set)),
                       "run_choice": self.run_names[i], "set_choice": self.r.set,
                       "title": self.r.title(), "scenario": self.r.scenario_text(),
                       "made": self.r.made_text()}, prefix="scrub")  # fmt: skip
        self._run, self._set = host.get("scrub.run_choice", ""), host.get("scrub.set_choice", "")
        if names and host.get("scrub.curve_name", "") not in names:
            host.set("scrub.curve_name", names[0])
        first = self.r.log(0)
        self.head.duration, self.head.dt = max(float(first.duration), 1e-6), float(first.dt)
        self.head.t = min(self.head.t, self.head.duration)
        self.pick(self.r.pick if k is None else k)

    def use_set(self, name: str) -> None:
        """Show data set ``name``: its run of the same name as this one when it has one (else its
        first), at the checkpoint of the same step when that run has one (else its pick)."""
        idx = self.by_set[name]
        i = next((j for j in idx if self.run_names[j] == self.run_names[self.i]), idx[0])
        step = self.rows[self.k]["step"] if self.k >= 0 else None
        k = next((n for n, r in enumerate(self.runs[i].rows) if r["step"] == step), None)
        self.use(i, k)

    def pick(self, k: int) -> None:
        self.k = min(max(int(k), 0), len(self.rows) - 1)

    def pick_step(self) -> None:
        """A click on a chart: the checkpoint nearest the step clicked."""
        s = self.host.get_float("scrub.pick")
        if math.isfinite(s):
            self.pick(int(np.argmin([abs(r["step"] - s) for r in self.rows])))

    def _controls(self) -> None:
        """Whichever control moved since the last frame picks the run or k; then all of them show
        it."""
        host = self.host
        name = host.get("scrub.set_choice", "")
        if name != self._set and name in self.by_set:
            self.use_set(name)
        run = host.get("scrub.run_choice", "")
        if run != self._run:
            i = next((j for j in self.by_set[self.r.set] if self.run_names[j] == run), None)
            if i is not None:
                self.use(i)
        n = len(self.rows)
        pos, choice = host.get_float("scrub.pos", 0.0), host.get("scrub.choice", "")
        if self._pos is not None and abs(pos - self._pos) > 1e-6:
            self.pick(round(pos * (n - 1)))
        elif self._choice is not None and choice != self._choice and choice in self.labels:
            self.pick(self.labels.index(choice))
        host.set("scrub.pos", round(self.k / max(n - 1, 1), 6))
        self._pos = host.get_float("scrub.pos", 0.0)
        host.set("scrub.choice", self.labels[self.k])
        self._choice = host.get("scrub.choice", "")

    def _show_checkpoint(self) -> None:
        from grl_snam.ariadne.replay import arms_of

        host, r = self.host, self.rows[self.k]
        names = ["step 0" if self.rows[0]["step"] == 0 else f"step {self.rows[0]['step']:g}",
                 f"step {r['step']:g}"]  # fmt: skip
        if names[0] == names[1]:
            names[1] += " "  # the same checkpoint twice: two arms all the same
        self.both = stack([self.r.log(0), self.r.log(self.k)], names)
        self.arms = arms_of(self.both)
        host.set_many({"names": "\n".join(names)}, prefix="arms")
        for key in ("view.arm", "ledger.agent"):
            host.set(key, names[1])
        flags = ", the pick" if r["selected"] else (", a held-out best" if r["best"] else "")
        now = (f"checkpoint {self.k + 1} of {len(self.rows)}: step {r['step']:g}{flags}; "
               f"{Path(r['blob']).name}")  # fmt: skip
        host.set("scrub.now", now)
        host.set("scrub.step", f"{r['step']:g}")
        host.set("play.title", f"{names[0]} (blue) and {names[1]} (orange), {self.both.duration:.1f}"
                               " s each")  # fmt: skip
        host.set_table("scrub.table", table_rows(self.rows, self.metrics, self.k))
        self.st["tracks"] = [[], []]
        self.st["charts"] = None
        self._shown = self.k
        print(f"scrub: {self.r.label}: {now}", file=sys.stderr, flush=True)

    def frame(self, dt) -> None:
        from grl_snam.ariadne.live import publish_charts, publish_tables, update_ghosts
        from grl_snam.ariadne.replay import charts_of, poses_at, tables_at

        host = self.host
        self._controls()
        if self._shown != self.k:
            self._show_checkpoint()
        metric = host.get("scrub.metric_name", "path_m")
        if metric != self._metric:
            series, markers, text = metric_chart(self.rows, self.metrics, metric)
            host.set("scrub.metric_text", text)
            if self.charts:
                host.set_plot("scrub.metric", series, markers=markers)
            self._metric = metric
        curve = host.get("scrub.curve_name", "")
        if curve != self._curve:
            self._training(curve)
        both, arms = self.both, self.arms
        t = self.head.tick(dt)
        k = both.index(t)
        names = [a.name for a in arms]
        j = names.index(host.get("ledger.agent")) if host.get("ledger.agent") in names else 1
        rows = _arms.plant_ledger_rows(both.at(k, j)["plant"])
        publish_tables(host, tables_at(both, arms, k), rows, names[j], both.meta.get("plant", ""))
        host.set("arms.time", f"t = {float(both.t[k]):.1f} s, tick {k + 1} of {len(both)}")
        book = host.get("ledger.book", "dU")
        if self.charts and self.st["charts"] != (j, book):
            publish_charts(host, charts_of(both, arms, j, book))
            self.st["charts"] = (j, book)
        if self.world is not None:
            a = both.arrays
            every = max(1, len(both) // 600)
            for i in range(2):
                self.st["tracks"][i] = list(zip(a["x"][: k + 1 : every, i].tolist(),
                                                a["y"][: k + 1 : every, i].tolist(), strict=True))  # fmt: skip
            sizes = [(a["length"][k, i], a["width"][k, i], a["wheelbase"][k, i]) for i in range(2)]
            goal = (float(a["goal_x"][k, 0]), float(a["goal_y"][k, 0]))
            follow = names.index(host.get("view.arm")) if host.get("view.arm") in names else 1
            update_ghosts(host, self.st, self.world, arms, poses_at(both, t), sizes, goal, follow)

    def _training(self, name: str) -> None:
        from grl_snam.ariadne import learn

        host = self.host
        self._curve = name
        if not name:
            host.set("scrub.curve_text", "the history has no held-out curve")
            return
        series, markers, log_y, text = learn.loss_chart([self.r.h], name)
        host.set("scrub.curve_text", text)
        host.set("scrub.curve_log", "1" if log_y else "0")
        if self.charts:
            host.set_plot("scrub.curve", series, markers=markers)


def _run_view(host, args, runs, world) -> int:
    """The view's frames: until closed, ``--frames`` of them, or one when headless. The windows
    ``--windows`` does not name start closed (``tree.<id>.open``)."""
    if args.camera:
        host.set("view.camera", args.camera)
    if args.windows is not None:
        keep = {w.strip() for w in args.windows.split(",")}
        for w in WINDOWS:
            host.set(f"tree.{w}.open", "1" if w in keep else "0")
    view = ScrubView(host, runs, world, args.at)
    once = host.mode == "headless" and not args.frames

    @host.on_frame
    def _frame(dt):
        view.frame(dt)
        if once:
            host.request_close()

    return _host.finish(host, args)


def view_main(args) -> int:
    """``--view``: the runs the reports list, read back; nothing deployed."""
    try:
        runs = [load_report(p) for p in args.view]
    except (OSError, ValueError) as e:
        return _refuse(str(e))
    worlds = [(r.scenario.get("run"), r.scenario.get("world_keys")) for r in runs]
    if not args.no_world and any(w != worlds[0] for w in worlds[1:]):
        return _refuse("the reports were deployed on different worlds: view them one at a time, "
                       "or with --no-world")  # fmt: skip
    sc = runs[0].scenario
    run_keys = {k: v for k, v in (sc.get("run") or {}).items() if v not in (None, "")}
    kind = run_keys.get("world", "data")
    drawn = args.mode != "headless" and not args.no_world and _cars.HAS_CARS
    note = (
        ""
        if drawn or args.mode == "headless"
        else (
            "the windows alone: --no-world"
            if args.no_world
            else f"no 3-D world: it needs {_cars.INSTALL}"
        )
    )
    if drawn and kind == "data" and _host.BROWSER and not os.path.isdir(str(run_keys.get("uri"))):
        drawn, note = (
            False,
            "no 3-D world: the run's city is not in this page",
        )  # cvcpkg.org: no CORS
    host = _host.open_host("scrub.ari", "scrub", args, load=False)
    city = drawn and kind == "data"
    _host.load_document(host, "scrub.ari", city=city, worlds=city)
    host.set_many(run_keys, prefix="run")
    host.set_many({k: v for k, v in (sc.get("world_keys") or {}).items() if v not in (None, "")},
                  prefix="world")  # fmt: skip
    world = None
    if drawn:
        C = _cars.C
        world = C.make_world(host, kind)
        C.show_world(host, kind, str(_host.DOCS / "scrub.ari"))
        C.draw_world(host, {"world": world})
        sx, sy = sc["start"]
        host.lab.add_markers("start", [(sx, sy, world.z(sx, sy) + 1.0)], color=(0.15, 0.85, 0.25))
        if json.dumps(world.info, sort_keys=True, default=str) != json.dumps(
            sc.get("world"), sort_keys=True, default=str
        ):
            note = ("this world is not quite the one deployed on (another libcvc's generator?): "
                    "the ghosts may cross what is drawn")  # fmt: skip
    elif _cars.HAS_CARS and args.mode != "headless":
        _cars.C.show_world(host, "flat", str(_host.DOCS / "scrub.ari"))  # the city node, emptied
    if note:
        print(f"scrub: {note}", file=sys.stderr)
    host.set("scrub.world_note", note)
    return _run_view(host, args, runs, world)


def deploy_main(args) -> int:
    """A history's checkpoints, deployed (or read from the cache), then the view."""
    if not _cars.HAS_CARS:
        return _refuse(f"needs {_cars.INSTALL}")
    from grl_snam.ariadne import live
    from grl_snam.learning import load
    from grl_snam.learning.runlog import libcvc_version

    try:
        runs = load(args.history)
        h = runs[args.run]
    except (OSError, ValueError, IndexError) as e:
        return _refuse(f"{args.history}: {e}")
    rows, missing = checkpoints(args.history, h, args.blobs)
    if missing:
        print(f"scrub: {len(missing)} saved policies are not there (first: {missing[0]})",
              file=sys.stderr)  # fmt: skip
    if not rows:
        return _refuse(f"{args.history}: no checkpoint names a saved policy (blob) that is there")
    C = _cars.C
    drawn = args.mode != "headless" and not args.no_world
    host = _host.open_host("scrub.ari", "scrub", args, load=False)
    _host.load_document(host, "scrub.ari", worlds=True)
    flags = {"nav_plant.vehicle": args.plant}
    if args.bundle and not args.uri:
        flags.update({"run.uri": str(args.bundle), "run.data_path": ""})
    C.apply_settings(host, args, flags)
    plant = host.get("nav_plant.vehicle", "kinematic") or "kinematic"
    if plant == "torch":
        host.close()
        return _refuse("the scrubber deploys on libcvc's sim_world: --plant kinematic, humvee, ...")
    kind = host.get("run.world", "data") or "data"
    meta = live.world_meta(host)  # before the scene fills in the generator's other keys
    world = C.make_world(host, kind)
    if drawn:
        C.show_world(host, kind, str(_host.DOCS / "scrub.ari"))
        C.draw_world(host, {"world": world})
    start, goal = live.start_and_goal(world, kind, host.get("run.uri"))
    scale = args.scale
    if scale is None:
        pt = next((r["blob"] for r in rows if r["blob"].endswith(".pt")), None)
        scale = (_arms._pt_scale(pt) if pt else None) or _arms.POLICY_SCALE
    world_keys = meta["world_keys"] if kind == "generated" else {}  # only a generated world's
    scenario = {"scrub": SCRUB_VERSION, "run": meta["run"], "world_keys": world_keys,
                "world": world.info, "start": list(start), "goal": list(goal), "seed": args.seed,
                "ticks": int(args.ticks), "plant": plant, "scale": float(scale), "dt": C.DT,
                "overrides": sorted(args.set or []), "params": args.params,
                "libcvc": libcvc_version()}  # fmt: skip
    logs = deploy_all(host, world, plant, rows, scenario, args.history, cache=args.cache,
                      lean=args.lean)  # fmt: skip
    metrics = [metrics_of(lg) for lg in logs]
    prov = logs[0].meta.get("provenance")
    if args.report:
        write_report(args.report, args.history, args.run, rows, metrics, scenario, prov,
                     args.data_set)  # fmt: skip
        print(f"scrub: {len(rows)} checkpoints to {args.report}")
    if drawn:
        host.lab.add_markers("start", [(start[0], start[1], world.z(*start) + 1.0)],
                             color=(0.15, 0.85, 0.25))  # fmt: skip
    deployed = Deployed(args.history, args.run, rows, metrics, scenario, logs, h=h, provenance=prov,
                        set_name=args.data_set)  # fmt: skip
    return _run_view(host, args, [deployed], world if drawn else None)


def main(argv=None) -> int:
    try:
        args = _host.parse(argv, _args)
    except RuntimeError as e:
        return _refuse(str(e))
    if args.view and args.history:
        return _refuse("name a history to deploy, or --view reports, not both")
    if args.view:
        return view_main(args)
    if not args.history:
        return _refuse("name a learning history to deploy, or --view reports deployed before")
    return deploy_main(args)


if __name__ == "__main__":
    sys.exit(main())
