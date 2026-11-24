"""grl-snam ariadne scrub: watch it learn. Every checkpoint of a training run, deployed.

    python -m grl_snam.ariadne scrub run/history.jsonl                          # Austin, Humvee
    python -m grl_snam.ariadne scrub run/history.jsonl --plant humvee --plant kinematic
    python -m grl_snam.ariadne scrub run/history.jsonl --world flat --ticks 600
    python -m grl_snam.ariadne scrub run/history.jsonl --headless --report scrub.json
    python -m grl_snam.ariadne scrub --view scrub.json [more.json ...]       # view only

A learning history (:mod:`grl_snam.learning.history`) names the policy each checkpoint saved, its
``blob`` (a ``.cvcnav`` or a ``.pt``; a relative path is the history's directory's). The scrubber
deploys every one of them on one scenario: libcvc's sim_world on ``--plant`` (the Jolt Humvee by
default, which collides with the world's buildings; nav's kinematic bicycle, ``kinematic``, drives
through them), from the scenario's start to its goal, for ``--ticks`` ticks. ``--plant`` given
again deploys on each, and a Plant combo switches between them. It drives them as
``live --arms`` does, tick for tick (:func:`grl_snam.ariadne.live.make_fleet`,
:func:`~grl_snam.ariadne.live.tick_arms`), so checkpoint 0 deployed here is the untrained arm
deployed there.

Each run is a run log (:mod:`grl_snam.learning.runlog`), cached at
``<history>.scrub/<blob sha256>-<scenario hash>.npz``. The scenario is the world (the document's
world on, its city and its grid's crop, and a generated world's ``world.*``:
:func:`grl_snam.ariadne._world.record`), the start and goal, the seed, the ticks, the plant, the
policy's frame, the settings set on the command line, and libcvc's version, so any of them changing
runs the checkpoints again. The world is the document's world node (``docs/scrub.ari``'s scene),
which libcvc's world runtime runs: Austin by default, ``--world generated`` or ``flat`` for its other
worlds, ``--uri`` for another city (:mod:`grl_snam.ariadne._world`).

The view (``docs/scrub.ari``):
- **Checkpoint k** is picked by a combo, a slider, Previous and Next, Most apart (the checkpoint
  whose car gets farthest from step 0's), or a click on a chart. Its car (opaque, orange) runs
  beside step 0's (the first checkpoint; see-through, blue) over the episode, each with its whole
  path drawn, with the Stats and Ledger windows for the two. The Scrub window says how far k's car
  gets from step 0's (``apart_m``: at most, and at the end), and how far the run's checkpoints do
  all told; under a car's width, that the two drive as one.
- **A deployed metric** (Metric picks it) against the checkpoint's step, with step 0's as a
  dashed line, and every metric in the Checkpoints window's table, one row per checkpoint.
- **A held-out curve** from the history, with the learn port's marks.
- **The episode**: Paused, Speed, Step and Restart, and the play head (the Episode window).

Both charts carry a cursor at checkpoint k. Each frame moves the two cars to the play head; the
Stats and Ledger tables follow it at most every :data:`TABLES_EVERY` s while it plays (a new
checkpoint, run or arm at once), writing only the cells that changed, and the paths are drawn once
per checkpoint shown. ``--at K`` opens on checkpoint K (default the pick), ``--at far`` on the one
whose car gets farthest from step 0's. ``--report R.json`` writes every checkpoint's metrics
(``grl_snam.scrub/v1``), its run log's file and the scenario. ``--blobs DIR`` looks for the saved
policies there (by file name) before beside the history, and ``--cache DIR`` caches the run logs
there in place of ``<history>.scrub/``. ``--data-set NAME`` names the run's data set in the report (the
learning check's ``retrain``, ``fixed/retrain``, ...), and ``--lean`` caches lean run logs
(:meth:`grl_snam.learning.runlog.RunLog.save`: single precision, packed; about 70 KB a checkpoint
of 1200 ticks) for a page.

**View only.** ``--view R.json ...`` shows runs already deployed: it reads the reports, the run log
each checkpoint lists and the history each report names, and deploys nothing, so it needs neither
torch nor the policies, and no sim_world steps. A file a report names is found as given, then
relative to the report, then by its file name beside the report, so a report travels with its
files. A report may also be a run's bundle, one zip with the report, the history and the logs
(the Pages site's scrub page ships those). ``--data-set`` and ``--plant`` pick the run it opens
on (default the first report's). It does not hash the scenario again: the libcvc in use, the
world and the files' places may differ from the ones that deployed. With several reports a Run
combo picks one, and a Set combo their data set when they come from several: a set picked shows
its run of the same name at the same training step, so the sets compare side by side. Their
worlds must be the same. The report's world is chosen in the document (its ``run`` and
``world_keys``: :func:`grl_snam.ariadne._world.from_record`; a generated world from its generator
settings, which the deploy's libcvc made: another libcvc's may differ, and the view says so);
``--no-world``, a city that is not on this
machine in a browser, or a pycvc_gl without the multi-car template, shows the windows alone (no
3-D world). A run log is read when its checkpoint is first shown.

**Is it the world they drove on?** A deploy keeps the world's record (``scenario.world``: a
generated world's settings and stats, a dataset's version and archive sha256, and the navigator's
grid's fingerprint, :func:`grl_snam.ariadne._world.grid_fingerprint`) and the grid's blocked cells
(``world_grid``, a few KB). The view compares the world it built with them field by field
(:func:`world_check`; a field an older report lacks is not compared). Another world is "not quite
the one deployed on", and says what differs. A grid that differs in at most :data:`GRID_ALIKE` of
its cells is the same world, rasterized by another build of libcvc (the page's wasm libcvc 4.0.0
differs from the native deploy in 0.15% of the generated city's cells); the Scrub window says how
many, and how near the two runs shown come to them.

The exit code is 0, and 2 when the history names no saved policy or the plant is torch (the
scrubber deploys on libcvc's sim_world), or when a report or a file it names is not there.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from grl_snam.ariadne import _arms, _cars, _host, _world

REPORT_SCHEMA = "grl_snam.scrub/v1"
#: Bumped when the scrubber's runs would change, so older caches are not read.
SCRUB_VERSION = 1
#: docs/scrub.ari's windows, by id.
WINDOWS = ("scrub", "checkpoints", "training", "episode", "sim_transport", "arms_stats",
           "arms_ledger")  # fmt: skip
#: The run logs a view keeps read at once (step 0's is always kept).
KEEP_LOGS = 8
#: The deployed metrics: name -> (label, run log array, digits). Each is the array's value at the
#: run's last tick, but ``apart_m`` (no array): how far the car got from step 0's, at the same
#: tick, at most over the run (:func:`apart`).
METRICS = {
    "reached": ("reached the goal", "stats_reached", 0),
    "time_s": ("time (s)", "stats_elapsed_s", 1),
    "path_m": ("path (m)", "stats_path_m", 1),
    "pen_steps": ("ticks penetrating", "stats_pen_steps", 0),
    "clear_min_m": ("clearance min (m)", "stats_clear_min_m", 2),
    "clear_mean_m": ("clearance mean (m)", "stats_clear_mean_m", 2),
    "goal_m": ("left to the goal (m)", "goal_dist", 1),
    "apart_m": ("apart from step 0, most (m)", None, 1),
}
#: While the episode plays, the Stats and Ledger tables follow the play head at most this often
#: (s of wall time); a new checkpoint, arm or run shows at once.
TABLES_EVERY = 0.2
#: The two runs' trails are drawn once a checkpoint has been shown this long (s of wall time), so
#: dragging the Scrub slider through the checkpoints draws none on the way.
TRAILS_AFTER = 0.25
#: The most points a trail has (a run's ticks, thinned).
TRAIL_POINTS = 600


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
                        "several sets picks one with a Set combo. With --view, the data set it "
                        "opens on")  # fmt: skip
    p.add_argument("--lean", action="store_true",
                   help="cache lean run logs (single precision, packed), for a page")  # fmt: skip
    p.add_argument("--camera", choices=("follow", "orbit"), default=None,
                   help="follow checkpoint k's ghost (the default), or orbit the world")  # fmt: skip
    p.add_argument("--windows", default=None, metavar="ID,...",
                   help="the windows open at the start (default all: " + ",".join(WINDOWS)
                        + "); the others are in the Windows menu")  # fmt: skip
    p.add_argument("--run", type=int, default=-1, help="which run in the file (default the last)")
    p.add_argument("--plant", action="append", default=None, metavar="PLANT",
                   help="the vehicle the checkpoints drive (nav_plant.vehicle): humvee, the "
                        "default, a Jolt Humvee that collides with the buildings; kinematic, nav's "
                        "bicycle, which drives through them; roboracer, ... Give it again to deploy "
                        "on several, and the Plant combo switches between them. With --view, the "
                        "plant it opens on")  # fmt: skip
    p.add_argument("--ticks", type=int, default=1200,
                   help="each checkpoint's ticks, of libcvc's tick each (pycvc_gl.cars.DT: 0.05 s "
                        "before libcvc 4.0.2, 0.06 s in 4.0.2, 7/120 s since)")  # fmt: skip
    p.add_argument("--seed", type=int, default=0, help="the scenario's seed")
    p.add_argument("--scale", type=float, default=None,
                   help="the policies' frame (normalized units per metre; default: the first "
                        ".pt checkpoint's, else 1/300)")  # fmt: skip
    p.add_argument("--bundle", default=None, help="a local Austin bundle: the same as --uri DIR")
    p.add_argument("--report", default=None, help="write every checkpoint's metrics (JSON)")
    p.add_argument("--at", default=None, metavar="K",
                   help="start at checkpoint K (0 is the first; default the pick), or at 'far': the "
                        "checkpoint whose car gets farthest from step 0's")  # fmt: skip
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


def metrics_of(log, first=None) -> dict:
    """The deployed metrics (:data:`METRICS`) at a run log's last tick (its one arm); ``apart_m``
    from ``first`` (step 0's log; NaN without it)."""
    out = {}
    for name, (_, key, _) in METRICS.items():
        a = log.arrays.get(key) if key else None
        out[name] = float(a[-1, 0]) if a is not None and len(a) else math.nan
    if first is not None:
        out["apart_m"] = apart(log, first)[0]
    return out


def apart(log, first, j: int = 0, i: int = 0) -> tuple[float, float]:
    """How far arm ``j`` of ``log`` is from arm ``i`` of ``first`` at the same tick (m): the most
    over the ticks both have, and at the last of them. NaN when they have none."""
    a, b = log.arrays, first.arrays
    n = min(len(a["x"]), len(b["x"]))
    if not n:
        return math.nan, math.nan
    d = np.hypot(a["x"][:n, j] - b["x"][:n, i], a["y"][:n, j] - b["y"][:n, i])
    return float(np.max(d)), float(d[-1])


def apart_text(rows, metrics, k: int, far: float, end: float, width: float) -> str:
    """What the two cars of checkpoint ``k`` and step 0 do apart: how far k's got from step 0's
    (``far`` at most, ``end`` at the last tick), and when that is less than a car's ``width`` that
    the two drive as one."""
    if k == 0:
        return "this is step 0 itself: the two cars are one run"
    if not math.isfinite(far):
        return ""
    step = f"step {rows[k]['step']:g}"
    text = f"{step} gets {far:.1f} m from step 0 at most, {end:.1f} m at the end"
    if far < width:
        text += (f":\nnever a car's width ({width:.1f} m) apart, the two cars overlap all the way,\n"
                 f"and {step} drives as step 0 does")  # fmt: skip
    return text


def run_apart_text(rows, metrics, width: float) -> str:
    """How far this run's checkpoints get from step 0, all of them (once each one's ``apart_m`` is
    known): the farthest, and when even that is less than a car's ``width``, that training has not
    changed how the policy drives here."""
    far = np.asarray([m.get("apart_m", math.nan) for m in metrics[1:]], float)
    if not len(far) or not np.isfinite(far).all():
        return ""
    i = int(np.argmax(far)) + 1
    text = (f"its checkpoints get {far.max():.1f} m from step 0 at most (step {rows[i]['step']:g}),"
            f"\nhalf of them {np.median(far):.1f} m or less")  # fmt: skip
    if far.max() < width:
        text += (f":\nevery one drives within a car's width ({width:.1f} m) of step 0,\n"
                 "so on this scenario, training has not changed how it drives")  # fmt: skip
    return text


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
        cells = {name: _cell(m.get(name, math.nan), d) for name, (_, _, d) in METRICS.items()}
        cells["reached"] = "yes" if m["reached"] >= 0.5 else "no"
        out.append({"at": ">" if i == k else "", "step": f"{r['step']:g}", "flag": flag, **cells})
    return out


def metric_chart(rows, metrics, name: str):
    """``name`` against the checkpoint's step, with step 0's as a dashed line and the pick and the
    held-out bests marked: (series, markers, text)."""
    label, _, digits = METRICS.get(name, METRICS["path_m"])
    x = np.asarray([r["step"] for r in rows], float)
    y = np.asarray([m.get(name, math.nan) for m in metrics], float)
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
    now (and cached; ``lean`` caches lean logs). A lean log cached before they were packed
    (``grl_snam.run_log/v1``) is packed when read: the same run, smaller."""
    from grl_snam.learning.runlog import PACKED_SCHEMA, RunLog

    shash = scenario_hash(scenario)
    work = tempfile.mkdtemp(prefix="grl-snam-scrub-")
    out = []
    for i, row in enumerate(rows):
        sha = _arms._sha256(row["blob"])
        path = cache_path(history, sha, shash, cache, lean)
        try:
            lg = RunLog.load(path)
            how = "cached"
            if lean and lg.meta.get("schema") != PACKED_SCHEMA:
                lg.save(path, lean=True)
                how = "cached, packed"
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


# ── the world the runs drove on, against the one rebuilt ───────────────────────────────────────

#: A generated world's record before its settings and stats were kept apart (GRL-SNAM before
#: #196): its flat keys -> the settings' names (``world.*``).
_FLAT_SETTINGS = {"seed": "seed", "half_m": "half", "amp_m": "amp", "water_level_m": "water_level",
                  "trees": "trees", "rocks": "rocks", "buildings": "buildings",
                  "tree_gen": "tree_gen", "preset": "preset", "ontology": "ontology",
                  "terrain": "terrain", "city": "city"}  # fmt: skip


#: Up to this share of the navigator's grid's cells may differ, everything else alike, and the
#: world is still the deploy's, rasterized by another build of libcvc: the page's (wasm) and the
#: deploy's (native) libcvc 4.0.0 differ in 365 of the generated city's 251,001 cells (0.15%).
GRID_ALIKE = 0.01


def _record(info) -> dict:
    """A world's record (:attr:`_world.world`'s ``info``) as :func:`world_check` reads it: an
    older generated world's flat keys as its settings (and its building count as a stat)."""
    info = dict(info or {})
    if info.get("source") == "generated" and "settings" not in info:
        flat = info
        info = {"source": "generated", "grid": flat.get("grid"),
                "settings": {new: flat[old] for old, new in _FLAT_SETTINGS.items() if old in flat}}  # fmt: skip
        if "building_count" in flat:
            info["stats"] = {"buildings": flat["building_count"]}
    return info


def _known(v) -> bool:
    return v not in (None, "")


def _same(a, b) -> bool:
    """Two recorded values alike: as numbers when both read as numbers, else as text."""
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return str(a) == str(b)


def world_check(recorded, here, changed=None) -> tuple[list[str], list[str], list[str]]:
    """What differs between the world a report's runs drove on (``recorded``: its scenario's
    ``world``) and the one rebuilt here (``here``: :attr:`_world.world`'s ``info``), field by field:
    the differences that make it another world, the fields compared, and the differences that
    leave it the same world (a grid of which at most :data:`GRID_ALIKE` of the cells differ). It compares the kind of world; a generated world's settings and the
    stats its generator published; a dataset's package version (``canonical``, else its ``uri``)
    and archive (``sha256``); and the navigator's grid (``grid``: its shape, bounds and cells).
    A field one side lacks (a report from before it was kept) is not compared, and neither is
    ``path`` (the state node the city sits at, the document's own). ``changed``, when known, is
    how many of the grid's cells are blocked on one side only (:func:`_world.grid_changes`)."""
    a, b = _record(recorded), _record(here)
    diffs, compared, alike = [], [], []
    if a.get("source") != b.get("source"):
        return [f"it drove on the {a.get('source') or '?'} world, this is the "
                f"{b.get('source') or '?'} one"], ["source"], []  # fmt: skip
    compared.append("source")

    def fields(name, what, x, y):
        both = [k for k in sorted(set(x or {}) & set(y or {})) if _known(x[k]) and _known(y[k])]
        if not both:
            return
        compared.append(name)
        off = [f"{k} {y[k]} here, {x[k]} deployed" for k in both if not _same(x[k], y[k])]
        if off:
            diffs.append(f"{what}: " + "; ".join(off[:4]) + ("; ..." if len(off) > 4 else ""))

    fields("settings", "the generator's settings differ", a.get("settings"), b.get("settings"))
    fields("stats", "the generator made another world (another libcvc's?)", a.get("stats"),
           b.get("stats"))  # fmt: skip
    if a.get("source") == "data":
        key = "canonical" if _known(a.get("canonical")) and _known(b.get("canonical")) else "uri"
        if _known(a.get(key)) and _known(b.get(key)):
            compared.append(key)
            if a[key] != b[key]:
                diffs.append(f"another dataset: {b[key]} here, {a[key]} deployed")
        if _known(a.get("sha256")) and _known(b.get("sha256")):
            compared.append("sha256")
            if a["sha256"] != b["sha256"]:
                diffs.append("the dataset's archive differs (another build of the same version?)")
    ga, gb = a.get("grid") or {}, b.get("grid") or {}
    if _known(ga.get("sha256")) and _known(gb.get("sha256")):
        compared.append("grid")
        if ga.get("shape") != gb.get("shape") or ga.get("bounds") != gb.get("bounds"):
            diffs.append(f"the navigator's grid is {gb.get('shape')} over {gb.get('bounds')} here, "
                         f"{ga.get('shape')} over {ga.get('bounds')} deployed")  # fmt: skip
        elif ga["sha256"] != gb["sha256"]:
            cells = int(np.prod(gb.get("shape") or [0]))
            if changed is not None and changed <= GRID_ALIKE * cells:
                alike.append(f"its navigator's grid differs in {changed:,} of {cells:,} cells "
                             f"({changed / max(cells, 1):.2%})")  # fmt: skip
            elif changed is not None:
                diffs.append(f"the navigator's grid differs in {changed:,} of its {cells:,} cells "
                             f"({changed / max(cells, 1):.2%})")  # fmt: skip
            else:
                diffs.append(f"the navigator's grid differs ({float(gb.get('blocked', 0)):.2%} of "
                             f"its cells blocked here, {float(ga.get('blocked', 0)):.2%} deployed)")  # fmt: skip
    return diffs, compared, alike


def nearest(points, cells, at_most: int = 20000) -> float:
    """How near ``points`` ((n, 2), m) come to ``cells`` ((m, 2), m): the least distance between
    any two (m; inf when either is empty). Over ``at_most`` cells, every k-th is taken."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    c = np.asarray(cells, np.float64).reshape(-1, 2)
    if not len(p) or not len(c):
        return math.inf
    c = c[:: max(1, -(-len(c) // at_most))]
    best = math.inf
    for i in range(0, len(c), 2048):
        d = ((c[i : i + 2048, None, :] - p[None, :, :]) ** 2).sum(-1)
        best = min(best, float(d.min()))
    return math.sqrt(best)


def near_text(d: float, car: float) -> str:
    """How near the two runs shown come to the cells where the world here is not the deploy's."""
    if not math.isfinite(d):
        return ""
    if d > 2 * car:
        return (f"these two runs come no nearer than {d:.0f} m to those cells:\n"
                "what they drove past is the same here")  # fmt: skip
    return f"these two runs pass {d:.1f} m from one of those cells: there, they may cross what is drawn"


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


def write_report(path, history, run, rows, metrics, scenario, provenance=None, set_name="",
                 world_grid=None) -> dict:  # fmt: skip
    """``--report``'s JSON (``grl_snam.scrub/v1``): the history, its data set (``set``, "" for
    none), the scenario and its hash, what deployed it (``provenance``: the GRL-SNAM commit,
    libcvc's version), every checkpoint's row (its run log is ``cache``) with its metrics, and the
    navigator's grid the runs drove on (``world_grid``: :func:`_world.pack_grid`'s, a few KB), so
    a view can say which cells of its own differ."""
    prov = {k: v for k, v in (provenance or {}).items() if k != "history"}
    rep = {"schema": REPORT_SCHEMA, "history": str(history), "set": set_name, "run": run,
           "scenario": scenario,
           "scenario_hash": scenario_hash(scenario), "provenance": prov,
           "checkpoints": [{**r, "metrics": m} for r, m in zip(rows, metrics, strict=True)]}  # fmt: skip
    if world_grid is not None:
        rep["world_grid"] = world_grid
    Path(path).write_text(json.dumps(rep, indent=1, default=str))
    return rep


class Deployed:
    """One history's checkpoints, deployed on one scenario: their rows and metrics, the scenario,
    the history (read when first asked for) and each checkpoint's run log (a :class:`RunLog`, or
    a file read when first shown; :data:`KEEP_LOGS` are kept read)."""

    def __init__(self, history, run, rows, metrics, scenario, logs, *, h=None, provenance=None,
                 set_name="", world_grid=None):  # fmt: skip
        from grl_snam.ariadne import learn

        self.history, self.run, self.rows, self.metrics = history, run, rows, metrics
        self.set = set_name or ""
        #: The plant its checkpoints were deployed on (the scenario's).
        self.plant = str((scenario or {}).get("plant") or "")
        self.scenario, self.provenance, self._h = scenario, provenance or {}, h
        #: The navigator's grid the runs drove on (:func:`_world.pack_grid`'s), or None.
        self.world_grid = world_grid
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
        if isinstance(lg, _Member):
            lg = RunLog.load(io.BytesIO(lg.read()))
        elif not isinstance(lg, RunLog):
            lg = RunLog.load(lg)
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


#: The report inside a run's bundle: one zip with it, the history and the run logs, the report's
#: paths the names inside (pages/assemble.py writes them, since a page's loader takes at most 512
#: files).
BUNDLE_REPORT = "scrub.json"


class _Member:
    """A file inside a zip, read when asked for."""

    def __init__(self, archive, name):
        self.archive, self.name = str(archive), name

    def read(self) -> bytes:
        with zipfile.ZipFile(self.archive) as z:
            return z.read(self.name)


def load_report(path) -> Deployed:
    """The runs a report (``--report``'s) lists, to view: its history and every checkpoint's run
    log found (:func:`find_file`), nothing read yet but the report. ``path`` may also be a run's
    bundle (:data:`BUNDLE_REPORT`), whose logs are read from it when shown. Raises ValueError when
    it is not a scrub report or a file it names is not there, OSError when it cannot be read."""
    bundle = zipfile.is_zipfile(path)
    if bundle:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            text = z.read(BUNDLE_REPORT).decode() if BUNDLE_REPORT in names else "{}"
    else:
        text = Path(path).read_text()
    rep = json.loads(text)
    if not isinstance(rep, dict) or rep.get("schema") != REPORT_SCHEMA:
        schema = rep.get("schema") if isinstance(rep, dict) else None
        raise ValueError(f"{path} is {schema!r}, not a scrub report ({REPORT_SCHEMA})")

    def found(name):
        if not bundle:
            return find_file(path, name)
        return _Member(path, name) if name in names else None

    hist = found(rep.get("history"))
    if hist is None:
        raise ValueError(f"{path}: its history {rep.get('history')!r} is not there")
    if bundle:  # the history reader reads a file: the bundle's, written out
        out = Path(tempfile.mkdtemp(prefix="grl-snam-scrub-")) / Path(rep["history"]).name
        out.write_bytes(hist.read())
        hist = out
    cks = rep.get("checkpoints") or []
    if not cks:
        raise ValueError(f"{path}: no checkpoints")
    rows, metrics, logs = [], [], []
    for c in cks:
        f = found(c.get("cache"))
        if f is None:
            raise ValueError(
                f"{path}: step {c.get('step')}'s run log {c.get('cache')!r} is not there"
            )
        rows.append({k: v for k, v in c.items() if k != "metrics"})
        got = c.get("metrics") or {}
        metrics.append({n: float(got[n]) if got.get(n) is not None else math.nan for n in METRICS})
        logs.append(f)
    return Deployed(hist, int(rep.get("run", -1)), rows, metrics, rep.get("scenario") or {}, logs,
                    provenance=rep.get("provenance"), set_name=str(rep.get("set") or ""),
                    world_grid=rep.get("world_grid"))  # fmt: skip


# ── the view ───────────────────────────────────────────────────────────────────────────────────


def first_run(runs, set_name: str = "", plant: str = "") -> int | None:
    """The first of ``runs`` (:class:`Deployed` s) of data set ``set_name`` deployed on ``plant``
    ("" for any), or None when none is."""
    return next((i for i, r in enumerate(runs)
                 if (not set_name or r.set == set_name) and (not plant or r.plant == plant)), None)  # fmt: skip


class ScrubView:
    """Checkpoint k beside step 0: the controls that pick k, the metric and training charts, the
    table, and the two runs over the episode (the replay's play head, tables and ghosts). ``runs``
    are :class:`Deployed` s: a Set combo picks their data set when they have several, a Plant combo
    the plant they were deployed on when they have several, and a Run combo a run of that set on
    that plant; it opens on run ``first``. ``world`` is the world to draw them on (None: the windows
    alone).

    Each frame moves the two ghosts to the play head. The rest follows less often: the Stats and
    Ledger tables when the checkpoint, the run or the arm changes, and while the episode plays at
    most every :data:`TABLES_EVERY` s; the two runs' whole trails once per checkpoint shown (step
    0's once per run), :data:`TRAILS_AFTER` s after it was picked."""

    def __init__(self, host, runs, world, k=None, first: int = 0):
        from grl_snam.ariadne.replay import PlayHead

        self.host, self.runs, self.world = host, list(runs), world
        self.charts = hasattr(host, "set_plot")
        # No running tracks (st["tracks"] None): the view draws each run's whole path. Checkpoint
        # k's ghost is opaque, step 0's see-through, so k's shows where the two overlap.
        self.st = {"nodes": [], "tracks": None, "camera": None, "modeled": None, "charts": None,
                   "opacity": [0.45, 1.0]}  # fmt: skip
        self._tables_key, self._tables_tick, self._tables_at = None, None, 0.0
        self._published: dict = {}  # the tables' rows as last written (live.set_rows)
        self._trails, self._trail0, self._shown_at = None, None, 0.0
        self._modeled = None  # what the cars were drawn as, said once
        #: Where the world here is not the deploy's: the cells' centres ((n, 2), m), or None.
        self.world_changes = None
        self.sets = list(dict.fromkeys(r.set for r in self.runs))
        self.plants = list(dict.fromkeys(r.plant for r in self.runs))
        self.by_set = {s: [i for i, r in enumerate(self.runs) if r.set == s] for s in self.sets}
        #: The runs of each data set on each plant: the Run combo's choices.
        self.by_group: dict[tuple[str, str], list[int]] = {}
        for i, r in enumerate(self.runs):
            self.by_group.setdefault((r.set, r.plant), []).append(i)
        self.run_names = [""] * len(self.runs)  # unique in their set, on their plant
        for idx in self.by_group.values():
            names = [self.runs[i].name for i in idx]
            for j, i in enumerate(idx):
                self.run_names[i] = f"{names[j]} ({j})" if names.count(names[j]) > 1 else names[j]
        host.set_many({"set_labels": "\n".join(self.sets), "set_count": str(len(self.sets))},
                      prefix="scrub")  # fmt: skip
        named = ", ".join(s or "(none)" for s in self.sets)
        on = ", ".join(p or "?" for p in self.plants)
        print(f"scrub: {len(self.runs)} runs in {len(self.sets)} data sets ({named}) on "
              f"{len(self.plants)} plants ({on})", file=sys.stderr, flush=True)  # fmt: skip
        self.head = PlayHead(host, self.runs[first].log(0).duration, self.runs[first].log(0).dt)
        host.on("scrub_prev", lambda: self.pick(self.k - 1))
        host.on("scrub_next", lambda: self.pick(self.k + 1))
        host.on("scrub_pick", self.pick_step)
        host.on("scrub_far", self.pick_far)
        self.use(first, k)

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
        in_set = [self.run_names[j] for j in self.by_group[(self.r.set, self.r.plant)]]
        plants = [p for p in self.plants if (self.r.set, p) in self.by_group]
        self.labels = [f"step {r['step']:g}" + (" (pick)" if r["selected"] else "")
                       for r in self.rows]  # fmt: skip
        self.k, self._shown, self._pos, self._choice = -1, None, None, None
        self._metric = self._curve = None
        names = learn.heldout_names([self.r.h])
        host.set_many({"labels": "\n".join(self.labels), "curve_names": "\n".join(names),
                       "run_labels": "\n".join(in_set), "run_count": str(len(in_set)),
                       "run_choice": self.run_names[i], "set_choice": self.r.set,
                       "plant_labels": "\n".join(plants), "plant_count": str(len(plants)),
                       "plant_choice": self.r.plant,
                       "title": self.r.title(), "scenario": self.r.scenario_text(),
                       "made": self.r.made_text()}, prefix="scrub")  # fmt: skip
        self._run, self._set = host.get("scrub.run_choice", ""), host.get("scrub.set_choice", "")
        self._plant = host.get("scrub.plant_choice", "")
        if names and host.get("scrub.curve_name", "") not in names:
            host.set("scrub.curve_name", names[0])
        first = self.r.log(0)
        self.head.duration, self.head.dt = max(float(first.duration), 1e-6), float(first.dt)
        self.head.t = min(self.head.t, self.head.duration)
        self.pick(self.r.pick if k is None else k)

    def use_set(self, name: str) -> None:
        """Show data set ``name``: on this plant when it was deployed on it (else its first), its
        run of the same name as this one when it has one (else its first), at the checkpoint of the
        same step when that run has one (else its pick)."""
        plant = self.r.plant if (name, self.r.plant) in self.by_group else self.runs[
            self.by_set[name][0]].plant  # fmt: skip
        self._use_like((name, plant))

    def use_plant(self, plant: str) -> None:
        """Show this data set on ``plant``: the same run and step, deployed there."""
        if (self.r.set, plant) in self.by_group:
            self._use_like((self.r.set, plant))

    def _use_like(self, group) -> None:
        """Show the run of ``group`` (a data set and a plant) named as this one, else its first, at
        this checkpoint's step when it has one, else its pick."""
        idx = self.by_group[group]
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

    def pick_far(self) -> None:
        """Most apart: the checkpoint that gets farthest from step 0 (a checkpoint whose distance
        the report does not have is measured from the two run logs)."""
        far = [-1.0]
        for i in range(1, len(self.rows)):
            far.append(self._apart(i)[0])
        self._shown = None  # the table and the chart again, every distance now in them
        self.pick(int(np.nanargmax(np.asarray(far, float))) if len(far) > 1 else 0)

    def _apart(self, i: int) -> tuple[float, float]:
        """How far checkpoint ``i``'s car gets from step 0's (:func:`apart`), kept in its metrics."""
        far, end = apart(self.r.log(i), self.r.log(0))
        m = self.metrics[i]
        if not math.isfinite(m.get("apart_m", math.nan)) and math.isfinite(far):
            m["apart_m"] = far
            self._metric = None  # the chart again
        return far, end

    def _width(self) -> float:
        """The car's width (m), from step 0's run log."""
        w = self.r.log(0).arrays.get("width")
        return float(w[0, 0]) if w is not None and len(w) else math.nan

    def _controls(self) -> None:
        """Whichever control moved since the last frame picks the run or k; then all of them show
        it."""
        host = self.host
        name = host.get("scrub.set_choice", "")
        if name != self._set and name in self.by_set:
            self.use_set(name)
        plant = host.get("scrub.plant_choice", "")
        if plant != self._plant and plant in self.plants:
            self.use_plant(plant)
        run = host.get("scrub.run_choice", "")
        if run != self._run:
            group = self.by_group[(self.r.set, self.r.plant)]
            i = next((j for j in group if self.run_names[j] == run), None)
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
        self.st["legend_notes"] = {
            names[0]: "untrained" if self.rows[0]["step"] == 0 else "the first checkpoint",
            names[1]: "the checkpoint picked" if self.k else "step 0 again",
        }
        host.set_many({"names": "\n".join(names)}, prefix="arms")
        for key in ("view.arm", "ledger.agent"):
            host.set(key, names[1])
        far, end = self._apart(self.k) if self.k else (0.0, 0.0)
        known = np.asarray([m.get("apart_m", math.nan) for m in self.metrics[1:]], float)
        flags = ["the pick" if r["selected"] else ("a held-out best" if r["best"] else "")]
        if self.k and len(known) and np.isfinite(known).all() and self.k == 1 + np.argmax(known):
            flags.append("the farthest from step 0")
        flags = "".join(f", {f}" for f in flags if f)
        now = (f"checkpoint {self.k + 1} of {len(self.rows)}: step {r['step']:g}{flags}; "
               f"{Path(r['blob']).name}")  # fmt: skip
        host.set("scrub.now", now)
        host.set("scrub.step", f"{r['step']:g}")
        host.set("play.title", f"{names[0]} (blue) and {names[1]} (orange), {self.both.duration:.1f}"
                               " s each")  # fmt: skip
        width = self._width()
        host.set("scrub.apart", apart_text(self.rows, self.metrics, self.k, far, end, width))
        host.set("scrub.apart_run", run_apart_text(self.rows, self.metrics, width))
        host.set_table("scrub.table", table_rows(self.rows, self.metrics, self.k))
        self.st["charts"] = None
        self._shown, self._tables_key = self.k, None
        self._shown_at = time.perf_counter()
        self.world_near()
        print(f"scrub: {self.r.label}: {now}", file=sys.stderr, flush=True)

    def world_near(self) -> None:
        """How near the two runs shown come to the cells where the world here is not the deploy's
        (``scrub.world_near``; empty when none differ, or the deploy kept no grid)."""
        if self.world_changes is None or not len(self.world_changes):
            self.host.set("scrub.world_near", "")
            return
        a, n = self.both.arrays, len(self.both)
        idx = np.arange(0, n, max(1, n // 300))
        pts = np.stack([a["x"][idx].ravel(), a["y"][idx].ravel()], axis=1)
        length = float(a["length"][0, 0]) if "length" in a else 5.0
        self.host.set("scrub.world_near", near_text(nearest(pts, self.world_changes), length))

    def _draw_trails(self) -> None:
        """Both runs' whole paths, each in its car's colour: step 0's once per run, checkpoint k's
        each time one is shown."""
        import pycvc_gl

        host, world, a = self.host, self.world, self.both.arrays
        n = len(self.both)
        idx = np.unique(np.r_[np.arange(0, n, max(1, -(-n // TRAIL_POINTS))), n - 1])
        for j, arm in enumerate(self.arms):
            if j == 0 and self._trail0 == self.i:
                continue
            name = f"arm{j}_track"
            if host.scene.hasGraphics(name):
                host.scene.removeGraphics(name)
            xs, ys = a["x"][idx, j].tolist(), a["y"][idx, j].tolist()
            if len(xs) >= 2:
                host.lab.add_path(name, [(x, y, world.z(x, y) + 0.3) for x, y in zip(xs, ys, strict=True)],
                                  _arms.slot_rgb(arm.slot))  # fmt: skip
                g = host.scene.geometry_node(name)
                g.setRenderMode(pycvc_gl.GeometryRenderMode_LINES)
                g.setLineWidth(2.5)
        self._trail0, self._trails = self.i, (self.i, self._shown)

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
        now = time.perf_counter()
        if self._tables_key != j or (
            k != self._tables_tick and now - self._tables_at >= TABLES_EVERY
        ):
            rows = _arms.plant_ledger_rows(both.at(k, j)["plant"])
            publish_tables(host, tables_at(both, arms, k), rows, names[j], both.meta.get("plant", ""),
                           self._published)  # fmt: skip
            host.set("arms.time", f"t = {float(both.t[k]):.1f} s, tick {k + 1} of {len(both)}")
            self._tables_key, self._tables_tick, self._tables_at = j, k, now
        book = host.get("ledger.book", "dU")
        if self.charts and self.st["charts"] != (j, book):
            publish_charts(host, charts_of(both, arms, j, book))
            self.st["charts"] = (j, book)
        if self.world is not None:
            a = both.arrays
            if self._trails != (self.i, self._shown) and now - self._shown_at >= TRAILS_AFTER:
                self._draw_trails()
            sizes = [(a["length"][k, i], a["width"][k, i], a["wheelbase"][k, i]) for i in range(2)]
            goal = (float(a["goal_x"][k, 0]), float(a["goal_y"][k, 0]))
            follow = names.index(host.get("view.arm")) if host.get("view.arm") in names else 1
            update_ghosts(host, self.st, self.world, arms, poses_at(both, t), sizes, goal, follow)
            modeled = self.st.get("modeled")
            if modeled is not None and modeled != self._modeled:
                self._modeled = list(modeled)
                what = "the Humvee's model" if all(modeled) else "boxes (no Humvee model here)"
                print(f"scrub: the cars are {what}", file=sys.stderr, flush=True)

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


def _run_view(host, args, runs, world, *, later=None, first=0) -> int:
    """The view's frames: until closed, ``--frames`` of them, or one when headless. The windows
    ``--windows`` does not name start closed (``tree.<id>.open``). It opens on run ``first``.
    ``later(view)``, when given, hands the view its world once there is one (``world`` is then
    None)."""
    if args.camera:
        host.set("view.camera", args.camera)
    if args.windows is not None:
        keep = {w.strip() for w in args.windows.split(",")}
        for w in WINDOWS:
            host.set(f"tree.{w}.open", "1" if w in keep else "0")
    view = ScrubView(host, runs, world, None if args.at == "far" else args.at, first)
    if args.at == "far":
        view.pick_far()
    if later is not None:
        later(view)
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
    plant = args.plant[0] if args.plant else ""
    first = first_run(runs, args.data_set, plant)
    if first is None:
        return _refuse(f"no report is of data set {args.data_set or '(any)'!r} on plant "
                       f"{plant or '(any)'!r}")  # fmt: skip
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
    _host.load_document(host, "scrub.ari")
    if note:
        print(f"scrub: {note}", file=sys.stderr)
    host.set("scrub.world_note", note)
    if not drawn:  # the city the document names stays unloaded (the network is shut): no world
        return _run_view(host, args, runs, None, first=first)
    _world.from_record(host, sc.get("run"), sc.get("world_keys"))

    def arrive(world, view):
        """The reports' world, drawn: the start, and a note when it is not the deployed one."""
        if isinstance(world, RuntimeError):  # in a page, where the world comes in a frame
            why = f"no 3-D world: {world}"
            print(f"scrub: {why}", file=sys.stderr)
            host.set("scrub.world_note", why)
            return
        sx, sy = sc["start"]
        host.lab.add_markers("start", [(sx, sy, world.z(sx, sy) + 1.0)], color=(0.15, 0.85, 0.25))
        packed = next((r.world_grid for r in runs if r.world_grid), None)
        changes = _world.grid_changes(packed, world.occ, world.bounds) if packed else None
        diffs, compared, alike = world_check(sc.get("world"), world.info,
                                             None if changes is None else len(changes))  # fmt: skip
        view.world_changes = changes
        if diffs:
            why = ("this world is not quite the one deployed on:\n" + "\n".join(diffs)
                   + ("" if changes is not None else "\nthe ghosts may cross what is drawn"))  # fmt: skip
            print(f"scrub: {why}", file=sys.stderr)
            host.set("scrub.world_note", why)
        else:
            more = "; " + "; ".join(alike) if alike else ""
            print(f"scrub: the world is the one deployed on ({', '.join(compared)} compared){more}",
                  file=sys.stderr)  # fmt: skip
            if alike:  # the same world, rasterized a little otherwise by this build of libcvc
                host.set("scrub.world_note", "the world deployed on, but " + "; ".join(alike)
                         + ":\nthis build of libcvc rasterizes it a little otherwise")  # fmt: skip
        view.world = world
        if getattr(view, "both", None) is not None:
            view.world_near()

    if _world.steps(host):  # the world now, before the first frame
        try:
            world = _world.world(host)
        except RuntimeError as e:
            host.close()
            return _refuse(f"the reports' world: {e}")
        return _run_view(host, args, runs, None, later=lambda view: arrive(world, view),
                         first=first)  # fmt: skip
    # A page makes the world in its frames, after this returns: the view takes it in one.
    return _run_view(host, args, runs, None, first=first,
                     later=lambda view: _world.when_ready(host, lambda w: arrive(w, view)))  # fmt: skip


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
    _host.load_document(host, "scrub.ari")
    _world.apply_settings(host, args, {"nav_plant.vehicle": args.plant[0] if args.plant else None},
                          bundle=args.bundle)  # fmt: skip
    plants = list(
        dict.fromkeys(args.plant or [host.get("nav_plant.vehicle", "humvee") or "humvee"])
    )
    if "torch" in plants:
        host.close()
        return _refuse("the scrubber deploys on libcvc's sim_world: --plant humvee, kinematic, ...")
    if args.report and len(plants) > 1:
        host.close()
        return _refuse("--report writes one plant's run: deploy each --plant to its own report")
    kind = _world.source(host)
    try:
        world = _world.world(host)
    except RuntimeError as e:
        host.close()
        return _refuse(str(e))
    meta = live.world_meta(host)
    start, goal = live.start_and_goal(world, kind, _world.city_uri(host))
    scale = args.scale
    if scale is None:
        pt = next((r["blob"] for r in rows if r["blob"].endswith(".pt")), None)
        scale = (_arms._pt_scale(pt) if pt else None) or _arms.POLICY_SCALE
    world_keys = meta["world_keys"] if kind == "generated" else {}  # only a generated world's
    world_grid = _world.pack_grid(world.occ)  # the grid the runs drive on, for a view to compare
    deployed = []
    for plant in plants:
        # The cars read the plant from the tree (Cars.pull), so it is set there for each one.
        host.set("nav_plant.vehicle", plant)
        scenario = {"scrub": SCRUB_VERSION, "run": meta["run"], "world_keys": world_keys,
                    "world": world.info, "start": list(start), "goal": list(goal),
                    "seed": args.seed, "ticks": int(args.ticks), "plant": plant,
                    "scale": float(scale), "dt": C.DT, "overrides": sorted(args.set or []),
                    "params": args.params, "libcvc": libcvc_version()}  # fmt: skip
        on = copy.deepcopy(rows)  # deploy_all records each one's run log in its row
        print(f"scrub: {len(on)} checkpoints on {plant}", flush=True)
        logs = deploy_all(host, world, plant, on, scenario, args.history, cache=args.cache,
                          lean=args.lean)  # fmt: skip
        metrics = [metrics_of(lg, logs[0]) for lg in logs]
        prov = logs[0].meta.get("provenance")
        if args.report:
            write_report(args.report, args.history, args.run, on, metrics, scenario, prov,
                         args.data_set, world_grid)  # fmt: skip
            print(f"scrub: {len(on)} checkpoints to {args.report}")
        deployed.append(Deployed(args.history, args.run, on, metrics, scenario, logs, h=h,
                                 provenance=prov, set_name=args.data_set, world_grid=world_grid))  # fmt: skip
    host.set("nav_plant.vehicle", plants[0])
    if drawn:
        host.lab.add_markers("start", [(start[0], start[1], world.z(*start) + 1.0)],
                             color=(0.15, 0.85, 0.25))  # fmt: skip
    return _run_view(host, args, deployed, world if drawn else None)


def main(argv=None) -> int:
    try:
        args = _host.parse(argv, _args)
    except RuntimeError as e:
        return _refuse(str(e))
    if args.view and args.history:
        return _refuse("name a history to deploy, or --view reports, not both")
    if args.at not in (None, "far"):
        try:
            args.at = int(args.at)
        except ValueError:
            return _refuse(f"--at {args.at}: a checkpoint's index, or far")
    if args.view:
        return view_main(args)
    if not args.history:
        return _refuse("name a learning history to deploy, or --view reports deployed before")
    return deploy_main(args)


if __name__ == "__main__":
    sys.exit(main())
