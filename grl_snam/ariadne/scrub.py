"""grl-snam ariadne scrub: watch it learn. Every checkpoint of a training run, deployed.

    python -m grl_snam.ariadne scrub run/history.jsonl                          # Austin, kinematic
    python -m grl_snam.ariadne scrub run/history.jsonl --world flat --ticks 600
    python -m grl_snam.ariadne scrub run/history.jsonl --headless --report scrub.json

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
(``grl_snam.scrub/v1``). The exit code is 0, and 2 when the history names no saved policy or the
plant is torch (the scrubber deploys on libcvc's sim_world).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from grl_snam.ariadne import _arms, _cars, _host

REPORT_SCHEMA = "grl_snam.scrub/v1"
#: Bumped when the scrubber's runs would change, so older caches are not read.
SCRUB_VERSION = 1
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
    p.add_argument("history", help="a learning history (.jsonl or .jsonl.gz) whose checkpoints "
                                   "name their saved policy (blob)")  # fmt: skip
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


def resolve_blob(history, blob: str) -> Path | None:
    """The file ``blob`` names: as given when absolute, else beside the history, else from the
    working directory, else its file name beside the history. None when none exists."""
    p = Path(blob)
    here = Path(history).resolve().parent
    for c in ([p] if p.is_absolute() else [here / p, Path.cwd() / p, here / p.name]):
        if c.is_file():
            return c.resolve()
    return None


def checkpoints(history, h) -> tuple[list[dict], list[str]]:
    """Run ``h``'s checkpoints that saved a policy, in order: ``{step, blob, kind, best,
    selected, fitness}`` (``blob`` the file), and the blobs named that are not there."""
    rows, missing = [], []
    for c in h.checkpoints():
        if not c.blob:
            continue
        path = resolve_blob(history, c.blob)
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


def cache_path(history, blob_sha: str, shash: str) -> Path:
    return Path(f"{history}.scrub") / f"{blob_sha[:16]}-{shash}.npz"


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


def deploy_all(host, world, plant, rows, scenario, history, *, log=print):
    """Every checkpoint's run log, from the cache or deployed now (and cached)."""
    from grl_snam.learning.runlog import RunLog

    shash = scenario_hash(scenario)
    work = tempfile.mkdtemp(prefix="grl-snam-scrub-")
    out = []
    for i, row in enumerate(rows):
        sha = _arms._sha256(row["blob"])
        path = cache_path(history, sha, shash)
        try:
            lg = RunLog.load(path)
            how = "cached"
        except (OSError, ValueError):
            lg = run_checkpoint(host, world, plant, row, scenario, shash, history, work)
            lg.save(path)
            how = "deployed"
        row["sha256"], row["cache"] = sha, str(path)
        m = metrics_of(lg)
        log(f"scrub: step {row['step']:g} ({i + 1}/{len(rows)}), {how}: reached "
            f"{'yes' if m['reached'] >= 0.5 else 'no'}, path {m['path_m']:.1f} m, "
            f"{m['pen_steps']:.0f} ticks penetrating, {m['goal_m']:.1f} m to go")  # fmt: skip
        out.append(lg)
    return out


# ── the view ───────────────────────────────────────────────────────────────────────────────────


class ScrubView:
    """Checkpoint k beside step 0: the controls that pick k, the metric and training charts, the
    table, and the two runs over the episode (the replay's play head, tables and ghosts)."""

    def __init__(self, host, h, rows, logs, world, k):
        from grl_snam.ariadne import learn
        from grl_snam.ariadne.replay import PlayHead

        self.host, self.h, self.rows, self.logs, self.world = host, h, rows, logs, world
        self.metrics = [metrics_of(lg) for lg in logs]
        self.charts = hasattr(host, "set_plot")
        self.labels = [f"step {r['step']:g}" + (" (pick)" if r["selected"] else "")
                       for r in rows]  # fmt: skip
        self.k, self._shown, self._pos, self._choice, self._metric = -1, None, None, None, None
        self._curve = None
        self.st = {"nodes": [], "tracks": [[], []], "camera": None, "modeled": None,
                   "charts": None}  # fmt: skip
        names = learn.heldout_names([h])
        host.set_many({"labels": "\n".join(self.labels), "curve_names": "\n".join(names)},
                      prefix="scrub")  # fmt: skip
        if names and host.get("scrub.curve_name", "") not in names:
            host.set("scrub.curve_name", names[0])
        self.head = PlayHead(host, logs[0].duration, logs[0].dt)
        host.on("scrub_prev", lambda: self.pick(self.k - 1))
        host.on("scrub_next", lambda: self.pick(self.k + 1))
        host.on("scrub_pick", self.pick_step)
        self.pick(k)

    def pick(self, k: int) -> None:
        self.k = min(max(int(k), 0), len(self.rows) - 1)

    def pick_step(self) -> None:
        """A click on a chart: the checkpoint nearest the step clicked."""
        s = self.host.get_float("scrub.pick")
        if math.isfinite(s):
            self.pick(int(np.argmin([abs(r["step"] - s) for r in self.rows])))

    def _controls(self) -> None:
        """Whichever control moved since the last frame picks k; then all of them show it."""
        host, n = self.host, len(self.rows)
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
        self.both = stack([self.logs[0], self.logs[self.k]], names)
        self.arms = arms_of(self.both)
        host.set_many({"names": "\n".join(names)}, prefix="arms")
        for key in ("view.arm", "ledger.agent"):
            host.set(key, names[1])
        flags = ", the pick" if r["selected"] else (", a held-out best" if r["best"] else "")
        host.set("scrub.now", f"checkpoint {self.k + 1} of {len(self.rows)}: step {r['step']:g}"
                              f"{flags}; {Path(r['blob']).name}")  # fmt: skip
        host.set("scrub.step", f"{r['step']:g}")
        host.set("play.title", f"{names[0]} (blue) and {names[1]} (orange), {self.both.duration:.1f}"
                               " s each")  # fmt: skip
        host.set_table("scrub.table", table_rows(self.rows, self.metrics, self.k))
        self.st["tracks"] = [[], []]
        self.st["charts"] = None
        self._shown = self.k

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
        series, markers, log_y, text = learn.loss_chart([self.h], name)
        host.set("scrub.curve_text", text)
        host.set("scrub.curve_log", "1" if log_y else "0")
        if self.charts:
            host.set_plot("scrub.curve", series, markers=markers)


def main(argv=None) -> int:
    if not _cars.HAS_CARS:
        return _refuse(f"needs {_cars.INSTALL}")
    from grl_snam.ariadne import live
    from grl_snam.learning import load
    from grl_snam.learning.runlog import libcvc_version

    args = _host.parse(argv, _args)
    try:
        runs = load(args.history)
        h = runs[args.run]
    except (OSError, ValueError, IndexError) as e:
        return _refuse(f"{args.history}: {e}")
    rows, missing = checkpoints(args.history, h)
    if missing:
        print(f"scrub: {len(missing)} saved policies are not there (first: {missing[0]})",
              file=sys.stderr)  # fmt: skip
    if not rows:
        return _refuse(f"{args.history}: no checkpoint names a saved policy (blob) that is there")
    C = _cars.C
    drawn = args.mode != "headless"
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
    logs = deploy_all(host, world, plant, rows, scenario, args.history)
    metrics = [metrics_of(lg) for lg in logs]
    if args.report:
        Path(args.report).write_text(json.dumps({
            "schema": REPORT_SCHEMA, "history": str(args.history), "run": args.run,
            "scenario": scenario, "scenario_hash": scenario_hash(scenario),
            "checkpoints": [{**r, "metrics": m} for r, m in zip(rows, metrics, strict=True)],
        }, indent=1, default=str))  # fmt: skip
        print(f"scrub: {len(rows)} checkpoints to {args.report}")
    k = (
        args.at
        if args.at is not None
        else next((i for i, r in enumerate(rows) if r["selected"]), len(rows) - 1)
    )
    host.set("scrub.title", f"{Path(args.history).name}: {len(rows)} checkpoints, steps "
                            f"{rows[0]['step']:g} to {rows[-1]['step']:g}")  # fmt: skip
    host.set("scrub.scenario", f"{kind} on {plant}, {args.ticks} ticks from ({start[0]:.0f}, "
                               f"{start[1]:.0f}) to ({goal[0]:.0f}, {goal[1]:.0f}), seed {args.seed}")  # fmt: skip
    if drawn:
        host.lab.add_markers("start", [(start[0], start[1], world.z(*start) + 1.0)],
                             color=(0.15, 0.85, 0.25))  # fmt: skip
    view = ScrubView(host, h, rows, logs, world if drawn else None, k)
    host.on_frame(view.frame)
    host.run(fps=30, frames=args.frames or None)
    if args.png and drawn:
        host.snapshot(args.png)
    code = host.exit_code()
    host.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
