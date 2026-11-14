"""The run log: a recorded run of one or more arms, to replay and to analyse.

``grl-snam ariadne live austin-freedrive --arms ... --log run.npz`` writes one, and ``grl-snam
ariadne replay run.npz`` plays it back. The format is ``grl_snam.run_log/v1``, one ``.npz``:

- ``t`` (T,): the time at the end of each tick (s).
- Per field, (T, A) arrays with one column per arm, in the arms' order:
  - the cars template's log fields (``pycvc_gl.cars.CarsLog``, each arm a car): ``x``, ``y``
    (the rear axle, m), ``heading``, ``speed``, ``steer``, ``vx``, ``vy``, ``goal_x``,
    ``goal_y``, ``goal_dist``, ``reached``, and ``coef_<name>`` for the coefficients it keeps.
    So a reader of the template's logs reads this one;
  - ``ms`` (the arm's wall time for the tick), ``length``, ``width``, ``wheelbase`` (m);
  - ``tel_<field>``: every field of libcvc's drive telemetry;
  - ``book_<name>``: the tick's books against U_nav (dU, drive, projection, context, dKE,
    residual);
  - ``stats_<column>``: the arm's Stats row after the tick (its leg so far);
  - ``plant_<key>``: the plant ledger's items after the tick, NaN where one does not apply.
- ``leg_tick`` (L,), ``leg_goal`` (L, 2): the tick each leg starts at, and its goal.
- ``meta``, a JSON string:
  - ``schema`` and ``arms``: each arm's name, spec, kind, the ``.cvcnav`` it ran with its
    ``sha256``, its source's sha256 and its policy key;
  - ``seed``, ``start``, ``goal``, ``legs``, ``dt``, ``scale``, ``world``, ``plant``;
  - ``run`` and ``world_keys``: the state-tree keys that rebuild the world;
  - ``settings``;
  - ``provenance``: the GRL-SNAM version (and its checkout's commit when it runs from one),
    libcvc's version, and each training history's path and sha256 when the run names them.

:class:`RunLog` writes and reads it. :meth:`RunLog.from_run` takes anything with ``arrays()``
and ``meta()`` (``grl_snam.ariadne._arms.ArmsRun``), and :meth:`RunLog.at` gives an arm's values
at a tick, as the replay shows them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SCHEMA = "grl_snam.run_log/v1"
#: The coefficients the cars template's log keeps (``coef_<name>``), from the drive telemetry.
CARS_COEFS = ("alpha", "beta", "gamma", "lam_soft", "lam_hard", "potential", "clearance",
              "curvature", "v_risk", "mu")  # fmt: skip


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def libcvc_version() -> str | None:
    """libcvc's version as the library loaded into this process names it (``libcvc.so.3.7.13``
    -> ``"3.7.13"``), or None when none is loaded or the platform does not say."""
    if "pycvc" not in sys.modules:
        return None
    try:
        maps = Path("/proc/self/maps").read_text()
    except OSError:
        return None
    m = re.search(r"/libcvc\.so\.(\d+\.\d+\.\d+)", maps)
    return m.group(1) if m else None


def _commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=True)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def provenance(history: str | Sequence[str] | None = None) -> dict:
    """What made the run: the GRL-SNAM version (and its checkout's commit), libcvc's version, and
    each training ``history`` (a path, or several) with its sha256."""
    import grl_snam

    out = {"grl_snam": grl_snam.__version__, "grl_snam_commit": _commit(),
           "libcvc": libcvc_version()}  # fmt: skip
    paths = [history] if isinstance(history, (str, Path)) else list(history or [])
    if paths:
        out["history"] = [{"path": str(p), "sha256": _sha256(p)} for p in paths]
    return out


def cars_fields(arrays: Mapping) -> dict:
    """The cars template's log fields that ``arrays`` (a run's) lacks, derived from what it has."""
    out = {}
    x, y = arrays["x"], arrays["y"]
    heading, speed = arrays["heading"], arrays["speed"]
    if "vx" not in arrays:
        out["vx"], out["vy"] = speed * np.cos(heading), speed * np.sin(heading)
    if "steer" not in arrays and "tel_steer" in arrays:
        out["steer"] = arrays["tel_steer"]
    if "goal_dist" not in arrays:
        out["goal_dist"] = np.hypot(arrays["goal_x"] - x, arrays["goal_y"] - y)
    if "reached" not in arrays and "stats_reached" in arrays:
        out["reached"] = arrays["stats_reached"]
    for name in CARS_COEFS:
        if f"coef_{name}" not in arrays and f"tel_{name}" in arrays:
            out[f"coef_{name}"] = arrays[f"tel_{name}"]
    return out


class RunLog:
    """A recorded run: ``arrays`` (name -> array, as the module docstring lists them) and
    ``meta``. :meth:`save` writes it, :meth:`load` reads it back."""

    def __init__(self, arrays: Mapping, meta: Mapping):
        self.arrays = {k: np.asarray(v) for k, v in arrays.items() if k != "meta"}
        self.meta = dict(meta)
        self._cum: dict = {}

    @classmethod
    def from_run(cls, source, /, *, history=None, **meta) -> RunLog:
        """The log of ``source`` (a run: anything with ``arrays()`` and ``meta()``), with
        ``meta`` added, the cars template's fields filled in, and the run's provenance
        (``history``: the training histories it names)."""
        arrays = dict(source.arrays())
        arrays.update(cars_fields(arrays))
        m = {**source.meta(), **meta, "schema": SCHEMA, "provenance": provenance(history)}
        return cls(arrays, m)

    def save(self, path) -> str:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(self.meta, default=_jsonable), **self.arrays)
        return path

    @classmethod
    def load(cls, path) -> RunLog:
        """The log at ``path``. Raises ValueError when it is not a run log."""
        with np.load(path, allow_pickle=False) as z:
            if "meta" not in z.files:
                raise ValueError(f"{path}: not a run log (no meta)")
            meta = json.loads(str(z["meta"]))
            if meta.get("schema") != SCHEMA:
                raise ValueError(f"{path}: not a run log ({SCHEMA}): schema {meta.get('schema')!r}")
            return cls({k: z[k] for k in z.files}, meta)

    # ── reading ──

    @property
    def names(self) -> list[str]:
        return [a["name"] for a in self.meta.get("arms", [])]

    @property
    def t(self) -> np.ndarray:
        return self.arrays["t"]

    def __len__(self) -> int:
        return int(len(self.t))

    @property
    def dt(self) -> float:
        return float(self.meta.get("dt", 0.05))

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self) else 0.0

    def index(self, t: float) -> int:
        """The tick showing at time ``t`` (s): the last one ending at or before it (the first
        before that)."""
        k = int(np.searchsorted(self.t, float(t) + 1e-9, side="right")) - 1
        return min(max(k, 0), max(len(self) - 1, 0))

    def leg(self, k: int) -> int:
        """The leg (0, 1, ...) tick ``k`` is in."""
        ticks = self.arrays.get("leg_tick", np.zeros(1, int))
        return max(int(np.searchsorted(ticks, k, side="right")) - 1, 0)

    def fields(self, prefix: str) -> list[str]:
        """The names after ``prefix`` of the arrays named ``<prefix><name>``, sorted."""
        return sorted(k[len(prefix) :] for k in self.arrays if k.startswith(prefix))

    def cumulative(self, name: str) -> np.ndarray:
        """``name``'s running sum over the ticks, (T, A)."""
        if name not in self._cum:
            self._cum[name] = np.cumsum(np.asarray(self.arrays[name], float), axis=0)
        return self._cum[name]

    def at(self, k: int, j: int) -> dict:
        """Arm ``j``'s values at tick ``k``: ``pose`` (x, y, heading, speed, goal), ``size``
        (length, width, wheelbase), ``tel``, ``book`` (the tick's), ``book_sum`` (summed to it),
        ``stats`` and ``plant`` (None where an item does not apply)."""
        a = self.arrays

        def val(name):
            return float(a[name][k, j]) if name in a else math.nan

        def group(prefix):
            return {f: val(prefix + f) for f in self.fields(prefix)}

        plant = {f: (None if math.isnan(v) else v) for f, v in group("plant_").items()}
        return {
            "pose": {f: val(f) for f in ("x", "y", "heading", "speed", "goal_x", "goal_y")},
            "size": {f: val(f) for f in ("length", "width", "wheelbase")},
            "tel": group("tel_"),
            "book": group("book_"),
            "book_sum": {
                f: float(self.cumulative("book_" + f)[k, j]) for f in self.fields("book_")
            },
            "stats": group("stats_"),
            "plant": plant,
        }


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)
