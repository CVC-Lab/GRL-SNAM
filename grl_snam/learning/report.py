"""Claims that a learner learned, each marked pass or fail: ``grl-snam learn-report``.

    grl-snam learn-report run*/history.jsonl --arms arms.json --json report.json

The claims, by family:

- **loss**, per held-out term in the histories: the term at the LAST evaluation is better than at
  the first, the untrained policy at step 0. The last and not the best: the best is picked on the
  same held-out data, so best against first is biased in the best's favour (the evidence lists
  the picked value too). With several runs (seeds) the per-run gains are bootstrapped and the CI
  must lie above 0. With one run the gain must exceed ``delta * IQR`` of the run's own smoothed
  curve; the evidence says that basis, which is weaker than a CI.
- **selection**: each run picked its checkpoint on held-out data (a held-out best, or an ``end``
  record's pick at a held-out evaluation), and the pick is not the first evaluation. A pick at
  step 0 means no step improved on the untrained policy.
- **task**, from ``--arms`` (``grl_snam.arms/v1``, or the A/B harness's JSON with ``--better``):
  on each primary metric, the trained arm beats each other arm on paired seeds, i.e. the 95% CI of
  ``trained - other`` lies on the better side of 0.
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

from .arms import arm_values
from .curves import DEFAULT, MarkParams, marks
from .history import History, classify, load, mode_of
from .stats import N_BOOT, as_rng, negate, paired_diff, summarize

REPORT_SCHEMA = "grl_snam.learn_report/v1"
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


def loss_claims(
    runs: Sequence[History],
    *,
    params: MarkParams = DEFAULT,
    maximize: Sequence[str] = (),
    n_boot: int = N_BOOT,
    rng=None,
) -> list[Claim]:
    rng = as_rng(rng)
    out = []
    names = _ordered_names(runs, lambda h: [s.name for s in h.heldout_series()])
    for name in names:
        term = classify(name)[1]
        mode = "max" if term in maximize or name in maximize else mode_of(term)
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
        if len(per) >= 2:
            g = summarize(gain, n_boot, rng)
            passed = g["ci95"][0] > 0.0
            ev.update(gain=g, basis=f"{len(per)} runs: bootstrap CI of the per-run gain > 0")
        else:
            m = marks(per[0][1].x, per[0][1].y, mode, params)
            margin = params.delta * m.iqr if math.isfinite(m.iqr) else math.inf
            passed = gain[0] > margin
            ev.update(
                gain={"mean": gain[0], "n": 1},
                margin=margin,
                basis="1 run: gain > delta * IQR of its smoothed curve (no CI)",
            )
        out.append(Claim(f"loss:{name}", "loss", statement, passed, ev))
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


def task_claims(
    table: Mapping,
    *,
    trained: str = "trained",
    against: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    better: Mapping[str, str] | None = None,
    n_boot: int = N_BOOT,
    rng=None,
) -> tuple[list[Claim], list[dict]]:
    """The task claims, and the metrics no claim could be made on (with why)."""
    rng = as_rng(rng)
    names = [a["name"] for a in table.get("arms", [])]
    if trained not in names:
        raise ValueError(f"the arms table has no {trained!r} arm (arms: {', '.join(names)})")
    others = list(against) if against else [a for a in names if a != trained]
    missing = [a for a in others if a not in names]
    if missing:
        raise ValueError(f"the arms table has no {missing} arm (arms: {', '.join(names)})")
    direction = {**table.get("direction", {}), **(better or {})}
    claims, skipped = [], []
    for res in table.get("results", []):
        case = res.get("case", "default")
        chosen = list(metrics or res.get("primary") or res.get("metrics", {}))
        for metric in chosen:
            if metric not in res.get("metrics", {}):
                continue
            d = direction.get(metric)
            if d not in ("up", "down"):
                skipped.append({"case": case, "metric": metric, "reason": "direction unknown"})
                continue
            for other in others:
                diff, how = trained_minus(table, res, metric, trained, other, n_boot, rng)
                if diff is None:
                    skipped.append(
                        {"case": case, "metric": metric, "against": other, "reason": how}
                    )
                    continue
                lo, hi = diff["ci95"]
                word = "higher" if d == "up" else "lower"
                claims.append(
                    Claim(
                        f"task:{case}:{metric}:{trained}-vs-{other}",
                        "task",
                        f"{case} {metric} is {word} for {trained} than for {other} "
                        "(paired seeds, 95% CI)",
                        lo > 0.0 if d == "up" else hi < 0.0,
                        {
                            "case": case,
                            "metric": metric,
                            "direction": d,
                            trained: _summary(_arm_entry(res, metric, trained)),
                            other: _summary(_arm_entry(res, metric, other)),
                            "diff": diff,
                            "from": how,
                        },
                    )
                )
    return claims, skipped


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
        for s in h.heldout_series():
            term = classify(s.name)[1]
            mode = "max" if term in maximize or s.name in maximize else mode_of(term)
            curves[s.name] = marks(s.x, s.y, mode, params).to_dict()
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
    """The ``grl_snam.learn_report/v1`` dict (see the module docstring)."""
    rng = as_rng(seed)
    claims = loss_claims(runs, params=params, maximize=maximize, n_boot=n_boot, rng=rng)
    claims += selection_claims(runs)
    skipped: list[dict] = []
    if arms is not None:
        task, skipped = task_claims(
            arms,
            trained=trained,
            against=against,
            metrics=metrics,
            better=better,
            n_boot=n_boot,
            rng=rng,
        )
        claims += task
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
        elif isinstance(g, Mapping):
            detail = f"  {g['mean']:+.4g} (1 run; margin {ev.get('margin', math.nan):.4g})"
        elif "reason" in ev:
            detail = f"  ({ev['reason']})"
        lines.append(f"{c['status'].upper():4}  {c['id']}: {c['statement']}{detail}")
    s = rep["summary"]
    lines.append(f"\n{s['pass']} pass, {s['fail']} fail")
    for u in rep.get("untested", []):
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
    ap.add_argument("--arms", help="an arms table (grl_snam.arms/v1, or the A/B harness's JSON)")
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
    table = json.loads(Path(args.arms).read_text()) if args.arms else None
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
            sources={"histories": list(args.hist), "arms": args.arms},
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
