"""grl-snam ariadne learn: did it learn? The learning instruments in an Ariadne document.

    python -m grl_snam.ariadne learn run/history.jsonl                   # one run
    python -m grl_snam.ariadne learn seed*/history.jsonl --arms arms.json
    python -m grl_snam.ariadne learn run/history.jsonl --follow          # a run still training
    python -m grl_snam.ariadne learn seed*/history.jsonl --arms arms.json \\
        --headless --report report.json                                  # the verdict as JSON

It reads learning histories with :mod:`grl_snam.learning`: any trainer that follows its
conventions (coef-train, grl-snam train, train_coef_energy, libcvc's coef_trainer, downstream
trainers). It also reads an optional arms table (``grl-snam arms-eval``'s ``grl_snam.arms/v1``, or
the A/B harness's JSON), and makes ``grl-snam learn-report``'s claims on them. The windows
(``docs/learn.ari``):

- **Did it learn?**: the claims, each pass or fail, with their evidence, and the summary.
- **Loss**: the held-out curve Term picks against its training twin, with its marks (best, elbow,
  plateau, divergence) and each run's pick. With several runs (seeds) it is a band across them,
  or each run's own curve.
- **Heads**: a head's mean over the probe or held-out set with its spread as a band, per run,
  marked where it settled (its plateau).
- **Training**: each run's gradient norm, and each head's relative parameter change per
  checkpoint for the run Run picks.
- **Checkpoints**: that run's checkpoint table with Term's value, its pick marked.
- **Arms**: each arm's mean [95% CI] per metric, and its paired difference to the untrained arm
  (else the table's reference), ``*`` when the CI excludes 0.
- **Planted**: each planted region's recovered error at the pick and at the start, against the
  bound.

The charts are pycvc_gl's ``chart:`` (libcvc 3.7.14, ``host.set_plot``). Without it each window
shows its marks as text, and its tables, which every pycvc_gl with the Python host (3.7.9+)
draws. ``--follow`` re-reads a history when it changes, redraws, and re-makes the claims at most
every ``--every`` seconds. The functions above the host's are pure: they build each chart's
series and markers and each table's rows from the histories, the arms table and the report.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from grl_snam.learning import band, load, marks
from grl_snam.learning.curves import DEFAULT as MARK_PARAMS
from grl_snam.learning.history import History, classify
from grl_snam.learning.report import report as make_report
from grl_snam.learning.report import trained_minus
from grl_snam.learning.stats import as_rng, excludes_zero

#: Each mark, as the chart's marker kind and how the text names it.
MARKS = (("best", "best", "best"), ("elbow", "elbow", "elbow"),
         ("plateau", "plateau", "flat from"), ("divergence", "note", "diverges at"))  # fmt: skip
#: More checkpoint rows than this are thinned (the first, the last and the marked ones stay).
MAX_ROWS = 300


# ── naming ─────────────────────────────────────────────────────────────────────────────────────


def run_label(h: History, k: int) -> str:
    """A short name for run ``k``: its file's stem (``seed0`` for ``seed0.history.jsonl``),
    ``#n`` for the n-th run in a file."""
    if not h.path:
        return f"run {k}"
    base = Path(h.path).name
    for ext in (".gz", ".jsonl", ".json", ".history"):
        base = base[: -len(ext)] if base.endswith(ext) else base
    return f"{base}#{h.run}" if h.run else base


def run_labels(runs) -> list[str]:
    """Every run's :func:`run_label`, made unique by its index where two collide."""
    labels = [run_label(h, k) for k, h in enumerate(runs)]
    return [f"{lab} ({k})" if labels.count(lab) > 1 else lab for k, lab in enumerate(labels)]


def heldout_names(runs) -> list[str]:
    """Every held-out curve across the runs: the loss terms first (those with a training twin,
    and fitness), then the held-out metrics."""
    loss: list[str] = []
    other: list[str] = []
    for h in runs:
        for p in h.pairs():
            if p.heldout is None or p.heldout.name in loss or p.heldout.name in other:
                continue
            (loss if p.is_loss else other).append(p.heldout.name)
    return loss + other


def head_names(runs) -> list[str]:
    """The heads any run records a mean or a parameter change for, in order."""
    out: list[str] = []
    for h in runs:
        for g in h.heads():
            if g not in out:
                out.append(g)
    return out


def _pair(h: History, name: str):
    return next((p for p in h.pairs() if p.heldout is not None and p.heldout.name == name), None)


def _style(k: int, line: str = "") -> str:
    return f"c{k % 8} {line}".strip()


def _fmt(v, spec: str = ".4g") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return format(float(v), spec)


# ── marks ──────────────────────────────────────────────────────────────────────────────────────


def mark_list(m, suffix: str = "") -> list[tuple[float, str, str]]:
    """The chart markers of a :class:`grl_snam.learning.curves.Marks`."""
    out = []
    for attr, kind, _word in MARKS:
        got = getattr(m, attr)
        if got is not None:
            out.append((got.x, f"{attr}{suffix}", kind))
    return out


def marks_text(m) -> str:
    """One line: where the curve was best, bent, went flat and turned back up."""
    if m.best is None:
        return "no points"
    parts = [f"best {m.best.x:g} ({m.best.y:.4g})"]
    for attr, _kind, word in MARKS[1:]:
        got = getattr(m, attr)
        parts.append(f"{word} {got.x:g}" if got is not None else f"no {attr}")
    return ", ".join(parts)


def _log_y(series) -> bool:
    """Log y when every value is positive and they span at least a decade."""
    ys = [float(v) for s in series for v in np.asarray(s["y"], float) if np.isfinite(v)]
    return bool(ys) and min(ys) > 0 and max(ys) / min(ys) >= 10.0


# ── charts: (series, markers, log_y, text) ─────────────────────────────────────────────────────


def loss_chart(runs, name: str, seeds: str = "band", params=MARK_PARAMS):
    """The held-out curve ``name`` against its training twin. Several runs: their band (mean,
    min to max) with the marks of the mean curve (``seeds="band"``), or each run's own curves,
    marks and pick (``"each"``). The text lists the marks and every run's pick."""
    have = [(k, h) for k, h in enumerate(runs) if name in h]
    labels = run_labels(runs)
    if not have:
        return [], [], False, f"no run has {name}"
    h0 = have[0][1]
    mode = h0.mode_of(name)
    p0 = _pair(h0, name)
    series: list[dict] = []
    markers: list[tuple] = []
    lines: list[str] = []
    if seeds == "band" and len(have) > 1:
        hs = [h for _, h in have]
        b = band(hs, name)
        series.append(
            dict(x=b.x, y=b.mean, lo=b.lo, hi=b.hi, label=f"held-out, {len(hs)} runs", style="c0")
        )
        if p0 is not None and p0.train is not None:
            tb = band(hs, p0.train.name)
            if tb is not None:
                series.append(
                    dict(x=tb.x, y=tb.mean, lo=tb.lo, hi=tb.hi, label="train", style="c1 dashed")
                )
        m = marks(b.x, b.mean, mode, params)
        markers += mark_list(m)
        lines.append(f"mean of {len(hs)} runs: {marks_text(m)}")
    else:
        for j, (k, h) in enumerate(have):
            s = h[name]
            tag = f"{labels[k]} " if len(have) > 1 else ""
            series.append(dict(x=s.x, y=s.y, label=f"{tag}held-out", style=_style(j)))
            p = _pair(h, name)
            if p is not None and p.train is not None:
                series.append(
                    dict(x=p.train.x, y=p.train.y, label=f"{tag}train", style=_style(j, "dashed"))
                )
            m = marks(s.x, s.y, mode, params)
            markers += mark_list(m, f" ({labels[k]})" if tag else "")
            lines.append(f"{tag}{marks_text(m)}")
    picks = []
    for k, h in have:
        sel = h.selected()
        if sel is None:
            continue
        picks.append(f"{labels[k]} {sel.x:g}" if len(have) > 1 else f"{sel.x:g}")
        if not (seeds == "band" and len(have) > 1):  # a band marks no run's own pick
            markers.append((sel.x, f"pick ({labels[k]})" if len(have) > 1 else "pick", "note"))
    if picks:
        lines.append("picked at " + ", ".join(picks))
    better = "higher" if mode == "max" else "lower"
    twin = f", trained on {p0.train.name}" if p0 is not None and p0.train is not None else ""
    head = f"{name}: {better} is better{twin}" + ("" if p0 is None or p0.is_loss else " (a metric)")
    return series, sorted(markers), _log_y(series), "\n".join([head, *lines])


def heads_chart(runs, head: str, params=MARK_PARAMS):
    """Head ``head``'s mean with mean +- spread as a band, per run, marked where it settled."""
    labels = run_labels(runs)
    series, markers, lines = [], [], []
    for k, h in enumerate(runs):
        stats = h.heads().get(head, {})
        if "mean" not in stats:
            continue
        mean = stats["mean"]
        d = dict(x=mean.x, y=mean.y, label=f"{labels[k]} {head}", style=_style(k))
        sp = stats.get("spread")
        if sp is not None and np.array_equal(sp.x, mean.x):
            d.update(lo=mean.y - sp.y, hi=mean.y + sp.y)
        series.append(d)
        m = marks(mean.x, mean.y, "min", params)
        if m.plateau is not None:
            markers.append((m.plateau.x, f"settles ({labels[k]})", "plateau"))
        f = mean.finite()
        text = f"{labels[k]}: mean {_fmt(f.y[0] if len(f) else None)} -> {_fmt(f.y[-1] if len(f) else None)}"
        if sp is not None and len(sp.finite()):
            text += f", spread {_fmt(sp.finite().y[0])} -> {_fmt(sp.finite().y[-1])}"
        text += f", settles at {m.plateau.x:g}" if m.plateau is not None else ", still moving"
        lines.append(text)
    if not series:
        lines.append(f"no run records a mean for {head}")
    return series, sorted(markers), False, "\n".join(lines)


def grad_chart(runs):
    """Each run's gradient norm (its mean per record; its max dashed when there is one run)."""
    labels = run_labels(runs)
    series, lines = [], []
    for k, h in enumerate(runs):
        gn = h.grad_norms()
        main = gn.get("mean") or gn.get("value")
        if main is None:
            continue
        series.append(dict(x=main.x, y=main.y, label=f"{labels[k]}", style=_style(k)))
        if len(runs) == 1 and "max" in gn:
            series.append(dict(x=gn["max"].x, y=gn["max"].y, label="max", style="c0 dotted"))
        f = main.finite()
        text = f"{labels[k]}: grad norm {_fmt(f.y[0] if len(f) else None)} -> "
        text += _fmt(f.y[-1] if len(f) else None)
        clip = gn.get("clipped_frac")
        if clip is not None and len(clip.finite()):
            text += f", clipped on {100 * float(np.mean(clip.finite().y)):.0f}% of steps"
        lines.append(text)
    if not series:
        lines.append("no run records a gradient norm")
    return series, [], _log_y(series), "\n".join(lines)


def dtheta_chart(h: History | None):
    """Each head's relative parameter change since the last checkpoint, for one run."""
    if h is None:
        return [], [], False, "no run"
    series, parts = [], []
    for j, (g, stats) in enumerate(h.heads().items()):
        d = stats.get("dtheta_rel")
        if d is None:
            continue
        series.append(dict(x=d.x, y=d.y, label=g, style=_style(j)))
        init = stats.get("dtheta_rel_init")
        if init is not None and len(init.finite()):
            parts.append(f"{g} {_fmt(init.finite().y[-1], '.3g')}")
    text = "dtheta/theta since the start, at the last checkpoint: " + (", ".join(parts) or "-")
    return series, [], True, text


# ── tables ─────────────────────────────────────────────────────────────────────────────────────


def checkpoint_rows(h: History | None, name: str | None = None) -> list[dict]:
    """One run's checkpoint table: x, kind, the held-out ``name``'s value there (else the
    fitness), the pick (``*``) or a held-out best, and the saved policy's file."""
    if h is None:
        return []
    term = (classify(name) or ("", ""))[1] if name else ""
    rows = []
    for c in h.checkpoints():
        v = c.heldout.get(term) if term else None
        v = c.fitness if v is None else v
        mark = "* pick" if c.selected else ("best" if c.best else "")
        blob = os.path.basename(c.blob) if c.blob else ""
        rows.append({"x": f"{c.x:g}", "kind": c.kind, "value": _fmt(v), "mark": mark, "blob": blob})
    if len(rows) > MAX_ROWS:
        step = math.ceil(len(rows) / MAX_ROWS)
        rows = [r for i, r in enumerate(rows) if i % step == 0 or r["mark"] or i == len(rows) - 1]
    return rows


def _fmt_ci(e: dict) -> str:
    lo, hi = e["ci95"]
    return f"{e['mean']:.4g} [{lo:.4g}, {hi:.4g}]"


def arms_rows(table: dict | None, reference: str | None = None, n_boot: int = 5000, seed=1):
    """Each (case, metric, arm): the mean [95% CI], and the paired difference to ``reference``
    (default: the untrained arm, else the table's own reference), ``*`` when its CI excludes 0,
    with better / worse where the table gives the metric's direction. Primary metrics first."""
    if not table:
        return []
    names = [a["name"] for a in table.get("arms", [])]
    ref = reference or ("untrained" if "untrained" in names else table.get("reference"))
    ref = ref or (names[0] if names else None)
    direction = table.get("direction", {})
    rng = as_rng(seed)
    rows = []
    for res in table.get("results", []):
        primary = list(res.get("primary") or [])
        metrics = primary + [m for m in res.get("metrics", {}) if m not in primary]
        for metric in metrics:
            per = res["metrics"].get(metric) or {}
            for a in names:
                e = per.get(a)
                if e is None:
                    continue
                diff = ""
                if a != ref and ref in per:
                    d, _how = trained_minus(table, res, metric, a, ref, n_boot, rng)
                    if d is not None:
                        lo, hi = d["ci95"]
                        diff = f"{d['mean']:+.4g} [{lo:+.4g}, {hi:+.4g}]"
                        if excludes_zero(d["ci95"]):
                            diff += " *"
                            up = direction.get(metric)
                            if up in ("up", "down"):
                                good = (lo > 0) == (up == "up")
                                diff += " better" if good else " worse"
                rows.append(
                    {
                        "case": res.get("case", ""),
                        "metric": metric + (" (primary)" if metric in primary else ""),
                        "arm": a,
                        "value": _fmt_ci(e),
                        "diff": diff,
                    }
                )
    return rows


def claim_detail(c: dict) -> str:
    """A claim's evidence in a few words: its interval, or its reason."""
    ev = c.get("evidence", {})
    g = ev.get("diff") or ev.get("gain")
    if isinstance(g, dict) and "ci95" in g:
        lo, hi = g["ci95"]
        return f"{g['mean']:+.4g} [{lo:+.4g}, {hi:+.4g}] n={g['n']}"
    if "reason" in ev:
        return str(ev["reason"])
    if isinstance(g, dict):
        return f"{g['mean']:+.4g} (1 run; margin {_fmt(ev.get('margin'))})"
    runs = ev.get("runs")
    if isinstance(runs, list) and runs and isinstance(runs[0], dict):
        if "picked" in runs[0] and "bound" in ev:
            return "picked " + ", ".join(_fmt(r.get("picked"), ".3g") for r in runs)
        if "on_heldout" in runs[0]:
            return "picks at " + ", ".join(_fmt(r.get("picked"), "g") for r in runs)
    return ""


def verdict_rows(rep: dict | None) -> list[dict]:
    """The report's claims as rows: status, family, the claim and its evidence."""
    if not rep:
        return []
    return [
        {
            "status": c["status"].upper(),
            "family": c["family"],
            "claim": c["statement"],
            "detail": claim_detail(c),
        }
        for c in rep.get("claims", [])
    ]


def verdict_summary(rep: dict | None) -> str:
    if not rep or not rep.get("claims"):
        return "No claims yet: no held-out evaluation, arms table or planted latent to test."
    s = rep["summary"]
    fams = ", ".join(f"{f} {v['pass']}/{v['pass'] + v['fail']}" for f, v in s["families"].items())
    untested = len(rep.get("untested", []))
    tail = (
        f"; {untested} untested (metrics with no training twin, or no direction)"
        if untested
        else ""
    )
    return f"{s['pass']} of {s['pass'] + s['fail']} claims pass ({fams}){tail}."


def planted_rows(rep: dict | None) -> list[dict]:
    """Per planted region: the recovered relative error at each run's pick and at its start (the
    untrained net), the bound, and the two claims' status."""
    if not rep:
        return []
    by: dict[str, dict] = {}
    for c in rep.get("claims", []):
        if c["family"] != "planted":
            continue
        _, region, which = c["id"].split(":", 2)
        ev = c["evidence"]
        r = by.setdefault(region, {"region": region, "bound": _fmt(ev.get("bound"), "g")})
        runs = ev.get("runs", [])
        r["picked"] = ", ".join(_fmt(p.get("picked"), ".3g") for p in runs) or "-"
        r["start"] = ", ".join(_fmt(p.get("first"), ".3g") for p in runs) or "-"
        r["recovered" if which == "recovered" else "untrained"] = c["status"].upper()
    return [dict({"recovered": "-", "untrained": "-"}, **r) for r in by.values()]


# ── the host ───────────────────────────────────────────────────────────────────────────────────


def _args(p):
    p.add_argument("hist", nargs="*", help="learning histories (JSON lines)")
    p.add_argument("--arms", default=None, help="an arms table (grl_snam.arms/v1 or the harness's)")
    p.add_argument("--follow", action="store_true", help="re-read the histories as they grow")
    p.add_argument("--every", type=float, default=2.0, help="--follow: seconds between claims")
    p.add_argument("--report", default=None, help="write learn-report's JSON here")
    p.add_argument("--strict", action="store_true", help="exit 1 when any claim fails")
    p.add_argument("--trained", default="trained", help="the trained arm's name (trained)")
    p.add_argument("--against", default=None, help="arms to beat, comma-separated (all others)")
    p.add_argument("--better", action="append", default=[], metavar="METRIC=up|down")
    p.add_argument("--maximize", action="append", default=[], metavar="TERM")
    p.add_argument("--planted-bound", type=float, default=0.2)
    p.add_argument("--boot", type=int, default=5000, help="bootstrap resamples (5000)")


def set_rows(host, key: str, rows: list[dict], columns: list[str]) -> None:
    """``rows`` (dicts) at ``key`` for a ``repeat:`` table: ``key.<i>.<column>``, ``key`` = count."""
    host.set_table(key, [[r.get(c, "") for c in columns] for r in rows], columns)


class LearnPort:
    """What the document shows, kept in step with the histories and the windows' choices."""

    def __init__(self, host, runs, arms, args):
        self.host, self.runs, self.arms, self.args = host, runs, arms, args
        self.charts = hasattr(host, "set_plot")
        self.rep: dict | None = None
        self.version = 0
        self._choice = None
        self._claims_at = -math.inf

    # ---- the claims
    def make_report(self) -> dict:
        a = self.args
        better = {}
        for t in a.better:
            k, _, v = t.partition("=")
            better[k] = v
        self.rep = make_report(
            self.runs,
            self.arms,
            trained=a.trained,
            against=[x for x in (a.against or "").split(",") if x] or None,
            better=better,
            maximize=a.maximize,
            planted_bound=a.planted_bound,
            n_boot=a.boot,
            sources={"histories": list(a.hist), "arms": a.arms},
        )
        self._claims_at = time.perf_counter()
        if a.report:
            Path(a.report).write_text(json.dumps(self.rep, indent=2) + "\n")
        return self.rep

    # ---- the data changed
    def publish_all(self) -> None:
        host, runs = self.host, self.runs
        self.version += 1
        terms, heads, labels = heldout_names(runs), head_names(runs), run_labels(runs)
        host.set("learn.terms", "\n".join(terms))
        host.set("learn.heads", "\n".join(heads))
        host.set("learn.runs", "\n".join(labels))
        for key, options in (("learn.term", terms), ("learn.head", heads), ("learn.run", labels)):
            if options and host.get(key, "") not in options:
                host.set(key, options[0])
        host.set("learn.charts", "1" if self.charts else "0")
        host.set("learn.n_runs", len(runs))
        sch = sorted({h.schema or "?" for h in runs})
        host.set("learn.caption", f"{len(runs)} run(s): {', '.join(sch) or 'none yet'}")
        rep = self.rep
        host.set("learn.summary", verdict_summary(rep))
        set_rows(host, "learn.claims", verdict_rows(rep), ["status", "family", "claim", "detail"])
        arms = arms_rows(self.arms, n_boot=self.args.boot)
        set_rows(host, "learn.arms", arms, ["case", "metric", "arm", "value", "diff"])
        planted = planted_rows(rep)
        cols = ["region", "bound", "picked", "start", "recovered", "untrained"]
        set_rows(host, "learn.planted", planted, cols)
        self._choice = None
        self.publish_choice()

    # ---- a window's choice changed
    def publish_choice(self) -> None:
        host, runs = self.host, self.runs
        term, head = host.get("learn.term", ""), host.get("learn.head", "")
        seeds, run = host.get("learn.seeds", "band"), host.get("learn.run", "")
        choice = (self.version, term, head, seeds, run)
        if choice == self._choice:
            return
        self._choice = choice
        labels = run_labels(runs)
        h = runs[labels.index(run)] if run in labels else (runs[0] if len(runs) else None)
        self._chart("learn.loss", loss_chart(runs, term, seeds) if term else ([], [], False, ""))
        self._chart("learn.head_chart", heads_chart(runs, head) if head else ([], [], False, ""))
        self._chart("learn.grad", grad_chart(runs))
        self._chart("learn.dtheta", dtheta_chart(h))
        rows = checkpoint_rows(h, term)
        set_rows(host, "learn.checkpoints", rows, ["x", "kind", "value", "mark", "blob"])
        what = term or "fitness"
        host.set("learn.ck_caption", f"{run or '-'}: {len(rows)} checkpoints; value = {what}")

    def _chart(self, key, built) -> None:
        series, markers, log_y, text = built
        self.host.set(f"{key}_text", text)
        self.host.set(f"{key}_log", "1" if log_y else "0")
        if self.charts:
            self.host.set_plot(key, series, markers=markers)

    # ---- each frame
    def frame(self, dt) -> None:
        if self.args.follow and self.runs.refresh():
            if time.perf_counter() - self._claims_at >= self.args.every:
                self.make_report()
            self.publish_all()
        self.publish_choice()


def main(argv=None) -> int:
    from grl_snam.ariadne import _host

    try:
        args = _host.parse(argv, _args)
    except RuntimeError as e:
        print(f"grl-snam ariadne learn: {e}", file=sys.stderr)
        return 2
    if not args.hist:
        print("grl-snam ariadne learn: name at least one learning history", file=sys.stderr)
        return 2
    try:
        runs = load(args.hist, follow=args.follow)
    except (FileNotFoundError, ValueError) as e:
        print(f"grl-snam ariadne learn: {e}", file=sys.stderr)
        return 2
    arms = json.loads(Path(args.arms).read_text()) if args.arms else None
    host = _host.open_host("learn.ari", "learn", args)
    port = LearnPort(host, runs, arms, args)
    try:
        port.make_report()
    except ValueError as e:
        host.close()
        print(f"grl-snam ariadne learn: {e}", file=sys.stderr)
        return 2
    port.publish_all()
    once = host.mode == "headless" and not args.follow and not args.frames

    @host.on_frame
    def _frame(dt):
        port.frame(dt)
        if once:
            host.request_close()

    code = _host.finish(host, args)
    if args.strict and port.rep and port.rep["summary"]["fail"]:
        return 1
    return code
