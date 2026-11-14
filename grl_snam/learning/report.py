"""Claims that a learner learned, each marked pass or fail: ``grl-snam learn-report``.

    grl-snam learn-report run*/history.jsonl --arms arms.json --json report.json
    grl-snam learn-report seed*/history.jsonl --arms seed0.json --arms seed1.json ...

The claims, by family:

- **loss**, against the arms: for an arms table's loss case (``"family": "loss"``, e.g.
  ``grl-snam arms-eval --cases objective-sdf``: the trainer's own objective on held-out units),
  per term, the trained arm's held-out term is lower than each other arm's (the untrained step 0,
  the constant), i.e. the 95% CI of ``trained - other`` over paired units lies below 0. With
  several tables (one per training run, ``--arms`` repeated) the runs pool: the CI resamples the
  runs and the units (:func:`grl_snam.learning.arms.runs_paired_diff`).
- **loss**, per held-out LOSS term in the histories (one with a training twin, the losses that
  carry gradient, or the run's ``fitness``): the term at the LAST evaluation is better than at the
  first, the untrained policy (step 0, or the ``init`` record). The last and not the best: the
  best is picked on the same held-out data, so best against first is biased in the best's favour
  (the evidence lists the picked value too). With several runs (seeds) the per-run gains are
  bootstrapped and the CI must lie above 0. With one run of at least 5 evaluations the gain must
  exceed ``max(delta * IQR, z * noise)`` of the run's own curve; the evidence says that basis,
  which is weaker than a CI. One shorter run cannot pass.
  Held-out metrics with no twin (reach, a clearance) get their marks but no claim, and are listed
  under ``untested``.
- **selection**: each run picked its checkpoint on held-out data (a held-out best, or an ``end``
  record's pick at a held-out evaluation), and the pick is not the first evaluation. A pick at
  step 0 means no step improved on the untrained policy.
- **task**, from ``--arms`` (``grl_snam.arms/v1``, or the A/B harness's JSON with ``--better``):
  on each primary metric of a task case, the trained arm beats each other arm on paired seeds,
  i.e. the 95% CI of ``trained - other`` lies on the better side of 0 (pooled over the tables as
  the loss case is, when there are several).
- **planted**, for a planted-latent exercise that logs ``...planted.<region>.rel_err``: at each
  run's pick the recovered latent's relative error is within ``--planted-bound``, and at the first
  evaluation (the untrained net) it is not.

The report also carries, per run, the marks of every held-out curve (best, elbow, plateau,
divergence; :mod:`grl_snam.learning.curves`), each head's plateau, and the checkpoint table.
A claim the data cannot support fails and says why; nothing is passed by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .arms import arm_values, case_result, runs_paired_diff, runs_summary
from .curves import DEFAULT, MarkParams, marks
from .history import History, TermPair, classify, load
from .stats import N_BOOT, as_rng, negate, paired_diff, summarize

REPORT_SCHEMA = "grl_snam.learn_report/v1"
#: A single run's loss claim needs this many evaluations (the noise floor needs 3 second
#: differences); with fewer, only several runs can show a gain.
MIN_POINTS_ONE_RUN = 5
FAMILIES = ("loss", "selection", "task", "planted")


@dataclass
class Claim:
    id: str
    family: str
    statement: str
    passed: bool
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "family": self.family,
            "statement": self.statement,
            "status": "pass" if self.passed else "fail",
            "evidence": self.evidence,
        }


def _num(v):
    return None if v is None or not math.isfinite(v) else float(v)


def _ordered_names(runs: Sequence[History], pick) -> list[str]:
    seen: list[str] = []
    for h in runs:
        for n in pick(h):
            if n not in seen:
                seen.append(n)
    return seen


# --------------------------------------------------------------------------- loss


def _loss_pairs(h: History) -> dict[str, TermPair]:
    return {p.heldout.name: p for p in h.pairs() if p.is_loss}


def _mode(h: History, name: str, maximize: Sequence[str]) -> str:
    term = (classify(name) or ("", name))[1]
    if term in maximize or name in maximize:
        return "max"
    return h.mode_of(name)


def loss_claims(
    runs: Sequence[History],
    *,
    params: MarkParams = DEFAULT,
    maximize: Sequence[str] = (),
    n_boot: int = N_BOOT,
    rng=None,
) -> list[Claim]:
    """The loss family: one claim per held-out LOSS term (one with a training twin, or the run's
    ``fitness``); other held-out metrics are marked but not claimed (:func:`unclaimed`)."""
    rng = as_rng(rng)
    out = []
    names = _ordered_names(runs, lambda h: list(_loss_pairs(h)))
    for name in names:
        term = classify(name)[1]
        owner = next(h for h in runs if name in _loss_pairs(h))
        mode = _mode(owner, name, maximize)
        per = [(h, h[name].finite()) for h in runs if name in h]
        per = [(h, s) for h, s in per if len(s) >= 2]
        better = "lower" if mode == "min" else "higher"
        statement = f"held-out {term} is {better} at the last evaluation than at the first"
        if not per:
            out.append(
                Claim(f"loss:{name}", "loss", statement, False, {"reason": "fewer than 2 points"})
            )
            continue
        first = [float(s.y[0]) for _, s in per]
        last = [float(s.y[-1]) for _, s in per]
        gain = [(a - b) if mode == "min" else (b - a) for a, b in zip(first, last, strict=True)]
        ev: dict = {
            "series": name,
            "mode": mode,
            "runs": [h.label for h, _ in per],
            "first": first,
            "last": last,
            "picked": [_picked(h, s) for h, s in per],
        }
        twin = _loss_pairs(owner)[name].train
        if twin is not None:
            ev["train"] = twin.name
        if len(per) >= 2:
            g = summarize(gain, n_boot, rng)
            passed = g["ci95"][0] > 0.0
            ev.update(gain=g, basis=f"{len(per)} runs: bootstrap CI of the per-run gain > 0")
        elif len(per[0][1]) < MIN_POINTS_ONE_RUN:
            passed = False
            ev.update(
                gain={"mean": gain[0], "n": 1},
                reason=f"1 run of {len(per[0][1])} points: too short to tell a gain from noise "
                f"(needs {MIN_POINTS_ONE_RUN}, or more runs)",
            )
        else:
            m = marks(per[0][1].x, per[0][1].y, mode, params)
            margin = max(params.delta * m.iqr, params.z * m.noise)
            margin = margin if math.isfinite(margin) else math.inf
            passed = gain[0] > margin
            ev.update(
                gain={"mean": gain[0], "n": 1},
                margin=margin,
                basis="1 run: gain > max(delta * IQR, z * noise) of its curve (no CI)",
            )
        out.append(Claim(f"loss:{name}", "loss", statement, passed, ev))
    return out


def unclaimed(runs: Sequence[History]) -> list[dict]:
    """Held-out series no loss claim is made on: metrics of the held-out evaluation with no
    training twin (reach, a clearance, a path length). Their marks are in the report."""
    out = []
    for name in _ordered_names(
        runs, lambda h: [s.name for s in h.heldout_series() if s.name not in _loss_pairs(h)]
    ):
        out.append({"series": name, "reason": "no training twin: a metric, not a loss term"})
    return out


def _picked(h: History, s) -> float | None:
    sel = h.selected()
    return None if sel is None else _num(s.at(sel.x))


# --------------------------------------------------------------------------- selection


def selection_claims(runs: Sequence[History]) -> list[Claim]:
    if not runs:
        return []
    per = []
    for h in runs:
        rows = h.checkpoints()
        sel = next((r for r in rows if r.selected), None)
        on_heldout = sel is not None and any(math.isfinite(v) for v in sel.heldout.values())
        first_x = rows[0].x if rows else None
        per.append(
            {
                "run": h.label,
                "picked": None if sel is None else sel.x,
                "axis": None if sel is None else sel.axis,
                "blob": None if sel is None else sel.blob,
                "on_heldout": on_heldout,
                "first": first_x,
                "after_first": sel is not None and first_x is not None and sel.x > first_x,
            }
        )
    return [
        Claim(
            "selection:heldout",
            "selection",
            "each run picked its checkpoint on held-out data",
            all(p["on_heldout"] for p in per),
            {"runs": per},
        ),
        Claim(
            "selection:not-untrained",
            "selection",
            "each run's pick comes after its first evaluation (the untrained policy)",
            all(p["after_first"] for p in per),
            {"runs": per},
        ),
    ]


# --------------------------------------------------------------------------- task (arms)


def _arm_entry(res: Mapping, metric: str, arm: str) -> dict | None:
    return res.get("metrics", {}).get(metric, {}).get(arm)


def trained_minus(
    table: Mapping, res: Mapping, metric: str, trained: str, other: str, n_boot: int, rng
) -> tuple[dict | None, str]:
    """The paired difference ``trained - other`` and how it was had: from the per-seed values
    when the table has them, else from the table's own paired difference to its reference."""
    a = arm_values(table, res["case"], metric, trained)
    b = arm_values(table, res["case"], metric, other)
    if a is not None and b is not None:
        return paired_diff(b, a, n_boot, rng), "per-seed values"
    names = [x["name"] for x in table.get("arms", [])]
    ref = table.get("reference") or (names[0] if names else None)
    if other == ref or trained == ref:
        e = _arm_entry(res, metric, trained if other == ref else other)
        d = None if e is None else e.get("paired_diff")
        if d is None:
            return None, "no per-seed values, and the table has no paired difference for it"
        if other == ref:
            return d, "the table's paired difference"
        return negate(d), "the table's paired difference, negated"
    return None, "no per-seed values, and neither arm is the table's reference"


def _tables(arms) -> list[Mapping]:
    """One table (a mapping) or several, as a list."""
    if arms is None:
        return []
    return [arms] if isinstance(arms, Mapping) else list(arms)


def _label(table: Mapping, k: int) -> str:
    return str(table.get("meta", {}).get("label") or f"table {k}")


def arms_claims(
    arms,
    *,
    trained: str = "trained",
    against: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    better: Mapping[str, str] | None = None,
    n_boot: int = N_BOOT,
    rng=None,
) -> tuple[list[Claim], list[dict]]:
    """The claims on an arms table (or several, one per training run), and the metrics no claim
    could be made on (with why). A loss case's claims are family ``"loss"``, the others
    ``"task"``. One table: each difference from its per-unit values, else from its own paired
    difference. Several: the cases, metrics and arms of the first table, pooled over every table
    that has them (:func:`grl_snam.learning.arms.runs_paired_diff`)."""
    rng = as_rng(rng)
    tables = _tables(arms)
    if not tables:
        return [], []
    names = [a["name"] for a in tables[0].get("arms", [])]
    for k, t in enumerate(tables):
        got = [a["name"] for a in t.get("arms", [])]
        if trained not in got:
            which = "the arms table" if len(tables) == 1 else f"the arms table {_label(t, k)!r}"
            raise ValueError(f"{which} has no {trained!r} arm (arms: {', '.join(got)})")
    others = list(against) if against else [a for a in names if a != trained]
    missing = [a for a in others if a not in names]
    if missing:
        raise ValueError(f"the arms table has no {missing} arm (arms: {', '.join(names)})")
    direction = {}
    for t in reversed(tables):
        direction.update(t.get("direction", {}))
    direction.update(better or {})
    claims, skipped = [], []
    for res in tables[0].get("results", []):
        case = res.get("case", "default")
        family = "loss" if res.get("family") == "loss" else "task"
        chosen = list(metrics or res.get("primary") or res.get("metrics", {}))
        for metric in chosen:
            if metric not in res.get("metrics", {}):
                continue
            d = direction.get(metric)
            if d not in ("up", "down"):
                skipped.append({"case": case, "metric": metric, "reason": "direction unknown"})
                continue
            for other in others:
                if len(tables) == 1:
                    diff, how = trained_minus(tables[0], res, metric, trained, other, n_boot, rng)
                    sides = {a: _summary(_arm_entry(res, metric, a)) for a in (trained, other)}
                else:
                    diff, how, sides = _pooled(tables, case, metric, trained, other, n_boot, rng)
                if diff is None:
                    skipped.append(
                        {"case": case, "metric": metric, "against": other, "reason": how}
                    )
                    continue
                lo, hi = diff["ci95"]
                word = "higher" if d == "up" else "lower"
                ev = {"case": case, "metric": metric, "direction": d, **sides}
                ev.update(diff=diff, **{"from": how})
                passed = lo > 0.0 if d == "up" else hi < 0.0
                if family == "loss":
                    unit = res.get("unit", "unit")
                    ev["unit"] = unit
                    statement = (
                        f"held-out {metric} ({case}) is {word} for {trained} than for {other} "
                        f"(paired by held-out {unit}, 95% CI)"
                    )
                    cid = f"loss:{case}:{metric}:{trained}-vs-{other}"
                else:
                    statement = (
                        f"{case} {metric} is {word} for {trained} than for {other} "
                        "(paired seeds, 95% CI)"
                    )
                    cid = f"task:{case}:{metric}:{trained}-vs-{other}"
                claims.append(Claim(cid, family, statement, passed, ev))
    return claims, skipped


def task_claims(table: Mapping, **kw) -> tuple[list[Claim], list[dict]]:
    """:func:`arms_claims` on one table (the name the task family had first)."""
    return arms_claims(table, **kw)


def _pooled(tables, case, metric, trained, other, n_boot, rng):
    """``(diff, how, sides)`` over several tables: the two-level bootstrap of the per-unit
    differences, each arm's own pooled level, and the runs that took part."""
    a_runs, b_runs, labels = [], [], []
    for k, t in enumerate(tables):
        if case_result(t, case) is None:
            continue
        a = arm_values(t, case, metric, trained)
        b = arm_values(t, case, metric, other)
        if a is None or b is None:
            return None, f"{_label(t, k)} has no per-unit values for both arms", {}
        a_runs.append(a)
        b_runs.append(b)
        labels.append(_label(t, k))
    diff = runs_paired_diff(b_runs, a_runs, n_boot, rng)
    if diff is None:
        return None, "no unit pairs in any table", {}
    diff["labels"] = labels
    sides = {
        trained: _pooled_level(a_runs, n_boot, rng),
        other: _pooled_level(b_runs, n_boot, rng),
    }
    return diff, f"per-unit values of {len(labels)} runs (runs x units bootstrap)", sides


def _pooled_level(per_run, n_boot, rng) -> dict | None:
    got = runs_summary(per_run, n_boot, rng)
    return None if got is None else {k: got[k] for k in ("mean", "ci95", "n", "runs")}


def _summary(e: Mapping | None) -> dict | None:
    if e is None:
        return None
    return {k: e[k] for k in ("mean", "ci95", "n") if k in e}


# --------------------------------------------------------------------------- planted


def planted_claims(runs: Sequence[History], *, bound: float = 0.2) -> list[Claim]:
    def pick(h):
        return [n for n in h.series if ".planted." in n and n.endswith(".rel_err")]

    out = []
    for name in _ordered_names(runs, pick):
        parts = name.split(".")
        region = ".".join(parts[parts.index("planted") + 1 : -1]) or name
        per = []
        for h in runs:
            if name not in h:
                continue
            s = h[name].finite()
            if not len(s):
                continue
            sel = h.selected()
            at = s.at(sel.x) if sel is not None else math.nan
            per.append(
                {
                    "run": h.label,
                    "picked": _num(at) if math.isfinite(at) else float(s.y[-1]),
                    "at": sel.x if sel is not None and math.isfinite(at) else float(s.x[-1]),
                    "first": float(s.y[0]),
                }
            )
        ev = {"series": name, "bound": bound, "runs": per}
        out.append(
            Claim(
                f"planted:{region}:recovered",
                "planted",
                f"the planted {region} latent is recovered within {bound:g} (relative) at the pick",
                bool(per) and all(p["picked"] <= bound for p in per),
                ev,
            )
        )
        out.append(
            Claim(
                f"planted:{region}:untrained-fails",
                "planted",
                f"the untrained policy does not recover the planted {region} latent "
                f"within {bound:g}",
                bool(per) and all(p["first"] > bound for p in per),
                ev,
            )
        )
    return out


# --------------------------------------------------------------------------- the report


def curve_marks(runs: Sequence[History], params: MarkParams = DEFAULT, maximize=()) -> list[dict]:
    """Per run: every held-out curve's marks, each head's plateau, and the checkpoint table."""
    out = []
    for h in runs:
        curves = {}
        losses = _loss_pairs(h)
        for s in h.heldout_series():
            d = marks(s.x, s.y, _mode(h, s.name, maximize), params).to_dict()
            d["loss_term"] = s.name in losses
            curves[s.name] = d
        heads = {
            g: {st: _plateau(stats[st], params) for st in ("mean", "spread") if st in stats}
            for g, stats in h.heads().items()
            if "mean" in stats or "spread" in stats
        }
        sel = h.selected()
        out.append(
            {
                "run": h.label,
                "path": h.path,
                "schema": h.schema,
                "curves": curves,
                "heads": heads,
                "selected": None if sel is None else sel.to_dict(),
                "checkpoints": [c.to_dict() for c in h.checkpoints()],
            }
        )
    return out


def _plateau(s, params):
    m = marks(s.x, s.y, "min", params)
    return None if m.plateau is None else m.plateau.to_dict()


def report(
    runs: Sequence[History],
    arms: Mapping | None = None,
    *,
    trained: str = "trained",
    against: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    better: Mapping[str, str] | None = None,
    maximize: Sequence[str] = (),
    planted_bound: float = 0.2,
    params: MarkParams = DEFAULT,
    n_boot: int = N_BOOT,
    seed: int = 1,
    sources: Mapping | None = None,
) -> dict:
    """The ``grl_snam.learn_report/v1`` dict (see the module docstring). ``arms`` is one arms
    table or a list of them (one per training run)."""
    rng = as_rng(seed)
    claims = loss_claims(runs, params=params, maximize=maximize, n_boot=n_boot, rng=rng)
    skipped: list[dict] = unclaimed(runs)
    on_arms: list[Claim] = []
    if arms is not None:
        on_arms, arms_skipped = arms_claims(
            arms,
            trained=trained,
            against=against,
            metrics=metrics,
            better=better,
            n_boot=n_boot,
            rng=rng,
        )
        skipped += arms_skipped
    claims += [c for c in on_arms if c.family == "loss"]
    claims += selection_claims(runs)
    claims += [c for c in on_arms if c.family != "loss"]
    claims += planted_claims(runs, bound=planted_bound)
    families = {}
    for f in FAMILIES:
        got = [c for c in claims if c.family == f]
        if got:
            families[f] = {
                "pass": sum(c.passed for c in got),
                "fail": sum(not c.passed for c in got),
            }
    return {
        "schema": REPORT_SCHEMA,
        "sources": dict(sources or {}),
        "params": params.to_dict(),
        "boot": n_boot,
        "seed": seed,
        "claims": [c.to_dict() for c in claims],
        "untested": skipped,
        "summary": {
            "pass": sum(c.passed for c in claims),
            "fail": sum(not c.passed for c in claims),
            "families": families,
        },
        "runs": curve_marks(runs, params, maximize),
    }


def format_claims(rep: Mapping) -> str:
    lines = []
    for c in rep["claims"]:
        ev = c["evidence"]
        detail = ""
        g = ev.get("diff") or ev.get("gain")
        if isinstance(g, Mapping) and "ci95" in g:
            lo, hi = g["ci95"]
            detail = f"  {g['mean']:+.4g} [{lo:+.4g}, {hi:+.4g}] n={g['n']}"
            if "runs" in g:
                detail += f" over {g['runs']} runs"
        elif "reason" in ev:
            detail = f"  ({ev['reason']})"
        elif isinstance(g, Mapping):
            detail = f"  {g['mean']:+.4g} (1 run; margin {ev.get('margin', math.nan):.4g})"
        lines.append(f"{c['status'].upper():4}  {c['id']}: {c['statement']}{detail}")
    s = rep["summary"]
    lines.append(f"\n{s['pass']} pass, {s['fail']} fail")
    for u in rep.get("untested", []):
        if "series" in u:
            what = u["series"]
        else:
            what = f"{u['case']}/{u['metric']}" + (f" vs {u['against']}" if "against" in u else "")
        lines.append(f"untested: {what}: {u['reason']}")
    return "\n".join(lines)


def _kv(items: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for t in items:
        k, sep, v = t.partition("=")
        if not sep or v not in ("up", "down"):
            raise SystemExit(f"{what}: NAME=up|down, not {t!r}")
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="grl-snam learn-report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("hist", nargs="+", metavar="HIST", help="learning histories (JSON lines)")
    ap.add_argument(
        "--arms",
        action="append",
        default=[],
        help="an arms table (grl_snam.arms/v1, or the A/B harness's JSON); repeat it for one "
        "table per training run",
    )
    ap.add_argument("--json", dest="out", metavar="R.json", help="write the report here")
    ap.add_argument("--trained", default="trained", help="the trained arm's name (trained)")
    ap.add_argument("--against", help="comma-separated arms to beat (default: every other arm)")
    ap.add_argument(
        "--metric", action="append", default=[], help="claim on this metric (default: primary)"
    )
    ap.add_argument(
        "--better",
        action="append",
        default=[],
        metavar="METRIC=up|down",
        help="a metric's better direction, for a table that does not state it",
    )
    ap.add_argument(
        "--maximize",
        action="append",
        default=[],
        metavar="TERM",
        help="a held-out term where higher is better (fitness already is)",
    )
    ap.add_argument("--planted-bound", type=float, default=0.2, help="relative error bound (0.2)")
    ap.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap resamples (5000)")
    ap.add_argument("--seed", type=int, default=1, help="the bootstrap's seed (1)")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any claim fails")
    args = ap.parse_args(argv)

    runs = load(args.hist)
    tables = [json.loads(Path(p).read_text()) for p in args.arms]
    table = tables[0] if len(tables) == 1 else (tables or None)
    try:
        rep = report(
            runs,
            table,
            trained=args.trained,
            against=[a for a in (args.against or "").split(",") if a] or None,
            metrics=args.metric or None,
            better=_kv(args.better, "--better"),
            maximize=args.maximize,
            planted_bound=args.planted_bound,
            n_boot=args.boot,
            seed=args.seed,
            sources={
                "histories": list(args.hist),
                "arms": args.arms[0] if args.arms else None,
                "arms_tables": list(args.arms),
            },
        )
    except ValueError as e:
        print(f"learn-report: {e}", file=sys.stderr)
        return 2
    print(format_claims(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 1 if args.strict and rep["summary"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
