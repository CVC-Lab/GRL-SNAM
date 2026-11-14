"""Read learning histories: the JSON-lines files the trainers write, one record per line.

Five writers are read today, two in GRL-SNAM (``grl_snam.coef_train_history/v1`` from
``coef-train``, ``grl_snam.coef_energy_history/v1`` from ``train_coef_energy``), one in libcvc
(``cvc.nav.coef_train_history/v1``) and two downstream. There is no per-schema adapter: every
writer follows the same conventions, and the reader reads the conventions.

- One JSON object per line. ``kind`` names the record (``header``, ``step``, ``epoch``,
  ``checkpoint``, ``heldout``, ``end``, ...).
- A ``header`` record starts a run, and its fields are the run's :attr:`History.meta`. A file a
  trainer appended to twice holds two runs.
- A record's x is its ``step``, or its ``epoch`` when it has no step. A record with neither (an
  ``end``) is kept whole in :attr:`History.extras`, by kind. The exception is an ``init``
  record, the evaluation of the untrained net before any training: it joins the series of the
  run's first evaluated kind, one before that kind's first x (``epoch`` 0's init is epoch -1).
- Every numeric leaf is a point of the series ``<kind>.<dotted.path>``: ``{"kind": "step",
  "step": 4, "grad_norm": {"mean": 0.3}}`` adds ``(4, 0.3)`` to ``step.grad_norm.mean``. A
  ``null`` or ``NaN`` is a gap. Strings, booleans and lists are not series.
- A dict named ``heldout`` or ``heldout_terms``, or a record whose kind starts with ``heldout``,
  holds held-out values. A dict named ``train`` or ``terms`` holds training values. A bare
  ``loss`` is the total, and ``fitness`` is a held-out score where higher is better. A held-out
  term and the training term of the same name are twins (:meth:`History.pairs`); a held-out
  ``<prefix>_<term>`` (``obj_goal``) with no exact twin pairs with ``<term>``. A twin is what
  makes a held-out value a LOSS TERM (lower is better, and it carries gradient); a held-out value
  with no twin is a metric of the held-out evaluation (reach, a clearance), higher-is-better when
  its name is in :data:`HIGHER_IS_BETTER` or the header's ``maximize`` list. A ``planted`` dict
  (a planted latent's truth and recovery) is not a loss term.
- ``heads.<g>.{mean,spread,dtheta_rel,...}`` describe head ``g`` (:meth:`History.heads`).
  ``grad_norm`` is a number or a dict of numbers (:meth:`History.grad_norms`).
- A true leaf whose name starts with ``best`` (``best``, ``best_heldout``, ``heldout_pick.best``)
  flags a new held-out best. An ``end`` record's ``best_step`` or ``selected_step`` names the
  final pick (:meth:`History.selected`).
- ``blob`` or ``checkpoint`` is the path of the policy saved at that record.

``load(paths)`` returns every run of every file as a :class:`Runs` list. ``load(paths,
follow=True)`` is for a file a trainer is still writing: :meth:`Runs.refresh` re-reads a file
when its modification time or size changes, a file that does not exist yet reads as empty, and a
half-written last line is skipped rather than refused.
"""

from __future__ import annotations

import functools
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Dict names whose leaves are held-out values, and those whose leaves are training values.
HELDOUT_KEYS = frozenset({"heldout", "heldout_terms"})
TRAIN_KEYS = frozenset({"train", "terms"})
#: A record's x, in order of preference.
X_KEYS = ("step", "epoch")
#: The path of the policy a record saved.
BLOB_KEYS = ("blob", "checkpoint")
#: An ``end`` record's final pick.
PICK_KEYS = ("best_step", "selected_step")
#: Held-out scores where higher is better, twin or not.
MAXIMIZE = frozenset({"fitness"})
#: Held-out metrics with no training twin where higher is better (every other one is lower).
HIGHER_IS_BETTER = frozenset(
    {"fitness", "reach", "reach_rate", "arrival_rate", "success_rate", "min_clear_m", "clear"}
)
#: The record kind of the untrained net's evaluation before training.
INIT_KIND = "init"


def record_x(rec: dict) -> tuple[float | None, str | None]:
    """``(x, axis)``: the record's ``step``, else its ``epoch``; ``(None, None)`` with neither."""
    for k in X_KEYS:
        v = rec.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v), k
    return None, None


def flatten(rec: dict, prefix: str = "") -> dict[str, float]:
    """The numeric leaves of ``rec`` by dotted path. ``None`` is NaN (a gap); booleans, strings
    and lists are left out."""
    out: dict[str, float] = {}
    for k, v in rec.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif v is None:
            out[key] = math.nan
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


@functools.cache
def classify(name: str) -> tuple[str, str] | None:
    """``("heldout" | "train", term)`` for a loss-term series name, ``None`` for any other
    (heads, gradient norms, the learning rate, ...). Cached: a run repeats its names on every
    record."""
    kind, _, path = name.partition(".")
    parts = path.split(".") if path else []
    if not parts or "heads" in parts or "grad_norm" in parts or "planted" in parts:
        return None
    held_kind = kind.startswith("heldout")
    for i, p in enumerate(parts):
        if p in HELDOUT_KEYS or p in TRAIN_KEYS:
            term = ".".join(parts[i + 1 :])
            if not term:
                return None
            return ("heldout" if held_kind or p in HELDOUT_KEYS else "train"), term
    if parts == ["loss"]:
        return ("heldout" if held_kind else "train"), "loss"
    if parts == ["fitness"]:
        return "heldout", "fitness"
    return None


def mode_of(term: str, twin: bool = True, maximize=()) -> str:
    """``"max"`` where higher is better, else ``"min"``: a loss term (one with a training twin)
    is lower-is-better unless it is a ``fitness``; a held-out metric with no twin is
    higher-is-better when its name is in :data:`HIGHER_IS_BETTER` or ``maximize``."""
    leaf = term.rsplit(".", 1)[-1]
    if term in maximize or leaf in maximize or leaf in MAXIMIZE:
        return "max"
    return "max" if not twin and leaf in HIGHER_IS_BETTER else "min"


def _flags_best(rec: dict) -> bool:
    for k, v in rec.items():
        if isinstance(v, dict):
            if _flags_best(v):
                return True
        elif v is True and k.startswith("best"):
            return True
    return False


def _series_leaves(rec: dict, axis: str | None) -> dict[str, float]:
    skip = {"kind", axis, *BLOB_KEYS}
    return flatten({k: v for k, v in rec.items() if k not in skip})


@dataclass
class Series:
    """One named curve: ``y`` against ``x`` (``axis`` says whether x is a step or an epoch)."""

    name: str
    x: np.ndarray
    y: np.ndarray
    axis: str = "step"

    @property
    def kind(self) -> str:
        return self.name.partition(".")[0]

    @property
    def path(self) -> str:
        return self.name.partition(".")[2]

    def __len__(self) -> int:
        return len(self.x)

    def finite(self) -> Series:
        """The points with a finite y."""
        m = np.isfinite(self.y)
        return Series(self.name, self.x[m], self.y[m], self.axis)

    def at(self, x: float) -> float:
        """y at ``x`` (the last such point), NaN when the series has none there."""
        idx = np.nonzero(self.x == float(x))[0]
        return float(self.y[idx[-1]]) if idx.size else math.nan

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "axis": self.axis,
            "x": [float(v) for v in self.x],
            "y": [float(v) if np.isfinite(v) else None for v in self.y],
        }


@dataclass
class Checkpoint:
    """One evaluated point of a run: a ``checkpoint`` record, a record that saved a policy, or
    one that carries held-out values."""

    x: float
    axis: str
    kind: str
    #: The saved policy's path, when the record names one.
    blob: str | None
    #: The writer flagged it a new held-out best.
    best: bool
    #: The held-out ``fitness`` (higher is better), when the record has one.
    fitness: float | None
    #: Held-out terms by name (``fitness`` included when present).
    heldout: dict[str, float]
    #: Per head, its numbers (``mean``, ``spread``, ``dtheta_rel``, ...).
    heads: dict[str, dict[str, float]]
    record: dict = field(repr=False)
    #: The run's final pick (:meth:`History.selected`).
    selected: bool = False

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "axis": self.axis,
            "kind": self.kind,
            "blob": self.blob,
            "best": self.best,
            "selected": self.selected,
            "fitness": self.fitness,
            "heldout": _jsonable(self.heldout),
            "heads": {g: _jsonable(d) for g, d in self.heads.items()},
        }


def _jsonable(d: dict[str, float]) -> dict:
    return {k: (v if math.isfinite(v) else None) for k, v in d.items()}


@dataclass
class TermPair:
    """A loss term's training curve and its held-out twin (either may be missing)."""

    term: str
    train: Series | None
    heldout: Series | None
    #: The header's ``maximize`` list (names where higher is better).
    maximize: tuple[str, ...] = ()

    @property
    def mode(self) -> str:
        return mode_of(self.term, self.train is not None, self.maximize)

    @property
    def is_loss(self) -> bool:
        """A held-out loss term: it has a training twin, or it is the run's ``fitness``."""
        return self.heldout is not None and (
            self.train is not None or self.term.rsplit(".", 1)[-1] in MAXIMIZE
        )


@dataclass
class History:
    """One run: its header (``meta``), its records, and the series flattened from them."""

    path: str | None = None
    run: int = 0
    meta: dict = field(default_factory=dict)
    records: list[dict] = field(default_factory=list, repr=False)
    series: dict[str, Series] = field(default_factory=dict, repr=False)
    #: Records with no step or epoch, by kind (the last of each kind): the ``end`` record.
    extras: dict[str, dict] = field(default_factory=dict, repr=False)

    @classmethod
    def from_records(cls, records: Iterable[dict], *, path=None, run: int = 0) -> History:
        h = cls(path=None if path is None else str(path), run=run)
        h.records = list(records)
        pts: dict[str, tuple[list[float], list[float], str]] = {}
        for rec in h.records:
            kind = str(rec.get("kind", "record"))
            if kind == "header":
                h.meta.update({k: v for k, v in rec.items() if k != "kind"})
                continue
            kind, x, axis = h._place(rec)
            if x is None:
                h.extras[kind] = rec
                continue
            for p, v in _series_leaves(rec, axis).items():
                xs, ys, _ = pts.setdefault(f"{kind}.{p}", ([], [], axis))
                xs.append(x)
                ys.append(v)
        for name, (xs, ys, axis) in pts.items():
            y = np.asarray(ys, dtype=float)
            if np.isfinite(y).any():
                h.series[name] = Series(name, np.asarray(xs, dtype=float), y, axis)
        return h

    def _place(self, rec: dict) -> tuple[str, float | None, str | None]:
        """``(series kind, x, axis)`` of a record; an ``init`` record without an x goes one
        before the first x of the run's first evaluated kind (see the module docstring)."""
        kind = str(rec.get("kind", "record"))
        x, axis = record_x(rec)
        if kind != INIT_KIND:
            return kind, x, axis
        first = self._first_evaluated()
        if first is None:
            return kind, (0.0 if x is None else x), (axis or "step")
        fkind, fx, faxis = first
        return fkind, (fx - 1.0 if x is None else x), (faxis if x is None else axis)

    def _first_evaluated(self):
        if not hasattr(self, "_first_cache"):
            self._first_cache = None
            for rec in self.records:
                kind = str(rec.get("kind", "record"))
                if kind in ("header", INIT_KIND):
                    continue
                x, axis = record_x(rec)
                if x is not None:
                    self._first_cache = (kind, x, axis)
                    break
        return self._first_cache

    # ---- naming
    @property
    def schema(self) -> str | None:
        return self.meta.get("schema")

    @property
    def maximize(self) -> tuple[str, ...]:
        """The header's ``maximize`` list: held-out names where higher is better."""
        m = self.meta.get("maximize") or ()
        return tuple(str(v) for v in m) if isinstance(m, (list, tuple)) else ()

    @property
    def label(self) -> str:
        base = Path(self.path).name if self.path else "history"
        return f"{base}#{self.run}" if self.run else base

    def names(self, prefix: str = "") -> list[str]:
        return [n for n in self.series if n.startswith(prefix)]

    def __contains__(self, name: str) -> bool:
        return name in self.series

    def __getitem__(self, name: str) -> Series:
        return self.series[name]

    def get(self, name: str) -> Series | None:
        return self.series.get(name)

    # ---- the conventions
    def pairs(self) -> list[TermPair]:
        """Every loss term: each held-out curve with its training twin (the same term, from the
        same kind of record when there is one), then the training terms with no twin."""
        held: dict[str, list[Series]] = {}
        train: dict[str, list[Series]] = {}
        for name, s in self.series.items():
            c = classify(name)
            if c is not None:
                (held if c[0] == "heldout" else train).setdefault(c[1], []).append(s)
        out: list[TermPair] = []
        used: set[str] = set()
        mx = self.maximize
        for term, hs in held.items():
            twins = train.get(term) or train.get(term.partition("_")[2], [])
            for h in hs:
                twin = ([t for t in twins if t.kind == h.kind] or twins or [None])[0]
                if twin is not None:
                    used.add(twin.name)
                out.append(TermPair(term, twin, h, mx))
        for term, ts in train.items():
            out.extend(TermPair(term, t, None, mx) for t in ts if t.name not in used)
        return out

    def heldout_series(self) -> list[Series]:
        """Every held-out curve (the ones :mod:`grl_snam.learning.curves` marks)."""
        return [p.heldout for p in self.pairs() if p.heldout is not None]

    def mode_of(self, name: str) -> str:
        """``"min"`` or ``"max"`` for the held-out series ``name`` (see :func:`mode_of`)."""
        for p in self.pairs():
            if p.heldout is not None and p.heldout.name == name:
                return p.mode
        c = classify(name)
        return mode_of(c[1] if c else name, True, self.maximize)

    def heads(self) -> dict[str, dict[str, Series]]:
        """``{head: {stat: series}}`` from ``heads.<head>.<stat>`` (the first kind that has
        them)."""
        out: dict[str, dict[str, Series]] = {}
        for name, s in self.series.items():
            parts = s.path.split(".")
            if len(parts) >= 3 and parts[0] == "heads":
                out.setdefault(parts[1], {}).setdefault(".".join(parts[2:]), s)
        return out

    def grad_norms(self) -> dict[str, Series]:
        """The run's gradient norm: ``{"value": s}`` for a number, ``{"mean": s, "max": s, ...}``
        for a dict."""
        out: dict[str, Series] = {}
        for s in self.series.values():
            if s.path == "grad_norm":
                out.setdefault("value", s)
            elif s.path.startswith("grad_norm."):
                out.setdefault(s.path[len("grad_norm.") :], s)
        return out

    def checkpoints(self) -> list[Checkpoint]:
        """The checkpoint table, in order, with the final pick marked ``selected``. Built once
        per run: a report asks for it dozens of times, and a run's records do not change (a
        followed file that grew is read into a new History)."""
        cache = getattr(self, "_checkpoints_cache", None)
        if cache is None or cache[0] != len(self.records):
            cache = self._checkpoints_cache = (len(self.records), self._checkpoint_table())
        return list(cache[1])

    def _checkpoint_table(self) -> list[Checkpoint]:
        rows: list[Checkpoint] = []
        for rec in self.records:
            kind = str(rec.get("kind", "record"))
            if kind == "header":
                continue
            skind, x, axis = self._place(rec)
            if x is None:
                continue
            held = {}
            for p, v in _series_leaves(rec, axis).items():
                c = classify(f"{skind}.{p}")
                if c is not None and c[0] == "heldout":
                    held[c[1]] = v
            if not (kind in ("checkpoint", INIT_KIND) or held or any(k in rec for k in BLOB_KEYS)):
                continue
            blob = next((rec[k] for k in BLOB_KEYS if isinstance(rec.get(k), str)), None)
            fit = held.get("fitness")
            heads = {
                g: {s: float(v) for s, v in d.items() if _is_num(v)}
                for g, d in (rec.get("heads") or {}).items()
                if isinstance(d, dict)
            }
            rows.append(
                Checkpoint(
                    x=x,
                    axis=axis,
                    kind=kind,
                    blob=blob,
                    best=_flags_best(rec),
                    fitness=None if fit is None or not math.isfinite(fit) else fit,
                    heldout=held,
                    heads=heads,
                    record=rec,
                )
            )
        pick = self._pick_x()
        chosen = None
        if pick is not None:
            at = [r for r in rows if r.x == pick]
            chosen = ([r for r in at if r.best] or at or [None])[-1]
        if chosen is None:
            chosen = ([r for r in rows if r.best] or [None])[-1]
        if chosen is not None:
            chosen.selected = True
        return rows

    def selected(self) -> Checkpoint | None:
        """The run's final pick: the checkpoint the ``end`` record names, else the last one the
        writer flagged a new best; ``None`` when the run picked nothing."""
        return next((r for r in self.checkpoints() if r.selected), None)

    def _pick_x(self) -> float | None:
        for rec in self.extras.values():
            for k in PICK_KEYS:
                if _is_num(rec.get(k)):
                    return float(rec[k])
        return None


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- files


def parse_lines(lines: Iterable[str], *, path=None, tolerant: bool = False) -> list[History]:
    """The runs in a history's lines. ``tolerant`` skips an unparsable LAST line (a writer caught
    mid-line); any other bad line is an error."""
    lines = list(lines)
    last = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            if tolerant and i == last:
                break
            raise ValueError(f"{path or '<history>'}:{i + 1}: not a JSON record ({e})") from e
        if not isinstance(rec, dict):
            raise ValueError(f"{path or '<history>'}:{i + 1}: a record must be a JSON object")
        if rec.get("kind") == "header" and cur:
            groups.append(cur)
            cur = []
        cur.append(rec)
    if cur:
        groups.append(cur)
    return [History.from_records(g, path=path, run=k) for k, g in enumerate(groups)]


def read_runs(path, *, tolerant: bool = False) -> list[History]:
    """Every run in one history file."""
    with Path(path).open(encoding="utf-8") as f:
        return parse_lines(f, path=path, tolerant=tolerant)


def _stamp(path: str):
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return st.st_mtime_ns, st.st_size


_UNREAD = object()


class Runs(list):
    """Every run of every file :func:`load` read, in file order. With ``follow``, a missing file
    reads as empty and :meth:`refresh` picks up what a running trainer has written since."""

    def __init__(self, paths: Iterable, follow: bool = False):
        super().__init__()
        self.paths = [str(p) for p in paths]
        self.follow = bool(follow)
        self._stamps: dict[str, object] = {}
        self._parsed: dict[str, list[History]] = {}
        self.refresh()

    def refresh(self) -> bool:
        """Re-read each file whose modification time or size changed since the last read; True
        when any did."""
        changed = False
        for p in self.paths:
            st = _stamp(p)
            if self._stamps.get(p, _UNREAD) == st:
                continue
            if st is None and not self.follow:
                raise FileNotFoundError(p)
            self._parsed[p] = [] if st is None else read_runs(p, tolerant=self.follow)
            self._stamps[p] = st
            changed = True
        if changed:
            self[:] = [h for p in self.paths for h in self._parsed[p]]
        return changed


def load(paths, follow: bool = False) -> Runs:
    """Every run in ``paths`` (one path or several). See the module docstring for ``follow``."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return Runs(paths, follow=follow)


# --------------------------------------------------------------------------- across runs


@dataclass
class Band:
    """One series across runs (seeds), at each x the runs have: the mean, the median, the low and
    high edges (the min and max, or the quantiles ``band`` was asked for) and the count."""

    name: str
    x: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: np.ndarray

    def to_dict(self) -> dict:
        def f(a):
            return [float(v) for v in a]

        return {
            "name": self.name,
            "x": f(self.x),
            "mean": f(self.mean),
            "median": f(self.median),
            "lo": f(self.lo),
            "hi": f(self.hi),
            "n": [int(v) for v in self.n],
        }


def band(runs: Iterable[History], name: str, q: tuple[float, float] | None = None) -> Band | None:
    """``name`` across ``runs``: at each x any run has, the finite values there summarized.
    ``lo``/``hi`` are the min and max, or the ``q`` quantiles (e.g. ``(0.1, 0.9)``). ``None``
    when no run has the series."""
    by_x: dict[float, list[float]] = {}
    for h in runs:
        s = h.get(name)
        if s is None:
            continue
        for xv, yv in zip(s.x, s.y, strict=True):
            if math.isfinite(yv):
                by_x.setdefault(float(xv), []).append(float(yv))
    if not by_x:
        return None
    xs = sorted(by_x)
    vals = [np.asarray(by_x[x]) for x in xs]
    if q is None:
        lo = [v.min() for v in vals]
        hi = [v.max() for v in vals]
    else:
        lo = [np.quantile(v, q[0]) for v in vals]
        hi = [np.quantile(v, q[1]) for v in vals]
    return Band(
        name=name,
        x=np.asarray(xs),
        mean=np.asarray([v.mean() for v in vals]),
        median=np.asarray([np.median(v) for v in vals]),
        lo=np.asarray(lo, dtype=float),
        hi=np.asarray(hi, dtype=float),
        n=np.asarray([len(v) for v in vals]),
    )
