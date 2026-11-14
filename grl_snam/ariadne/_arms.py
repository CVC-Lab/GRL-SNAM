"""``live austin-freedrive --arms``: several policies side by side, one libcvc sim_world each.

The Laboratory's comparison view (the engagement docs' Laboratory, section 6.5, Phase 5): the
same world, start, goal and seed for every arm, one ghost vehicle per arm in its own colour, a
Retarget that moves every arm's goal on the same tick, and two windows, Stats and Ledger
(``docs/stats_panels.ari``). This module is the part that needs no host: what an arm is, the
policy each world loads, the ledger's books, the Stats rows, the plant ledger, and
:class:`ArmsRun`, the one object that holds every arm's run, which ``--log`` saves as a run log
(:mod:`grl_snam.learning.runlog`). ``grl_snam/ariadne/live.py`` steps the worlds, draws them and
writes the panels from it; ``grl_snam/ariadne/replay.py`` writes the same panels from a run log,
through the same :func:`panel_tables` and :func:`chart_series`.

**Arms.** ``--arms trained,untrained,constant`` names them; each is a bare name or
``NAME=SPEC`` in ``grl-snam arms-eval``'s syntax (:mod:`grl_snam.learning.arms`): a ``.cvcnav``, a
``.pt`` or ``const:...``. The bare names are:

- ``trained``: the published policy (``grl-snam-weights``' ``coef_sdf.pt``), which the port
  drives without ``--arms``;
- ``untrained``: a ``CoefMLP`` as its constructor initializes it under ``torch.manual_seed(seed)``
  (``untrained=PATH`` takes a run's step-0 checkpoint instead);
- ``constant``: ``const:bias``, CoefMLP's output bias everywhere (alpha 1, beta 3, gamma 4).

As in ``arms-eval``, a constant arm takes its input layout from the first arm with a net. Each
arm's policy is written to a ``.cvcnav`` (``write_coef_mlp``) and loaded with
``nav_sim_world_set_policy``. Every arm runs in the frame of the first ``.pt`` arm that records
one (``meta.scale``), else the published policy's, 1/300.

**Stats** (one row per arm): the minimum and mean clearance (m), the penetration steps, the path
length (m), the elapsed time (s: to the goal once there), the wall time per tick (ms) and whether
it reached the goal. libcvc's ``nav_stats`` collector (``nav_sim_world_begin_nav_stats``) gives
all but the mean clearance, which is the mean of the drive telemetry's per-tick clearance, and
the wall time, which the port measures. A Retarget starts a new leg: every arm's collector is
armed again at the new goal on the same tick, so a row is the current leg's.

**The ledger** (per arm): the coefficients in force and the terms of U_nav from the drive
telemetry (``nav_sim_world_drive_telemetry``), and the tick's books against U_nav (the vehicle pH
white paper, ``docs/vehicle-ph-plant.tex``, "Context work and projection work"), per unit mass
in the drive's normalized units:

- ``dU``: U's change over the tick, from where the last tick ended to where this one ends;
- ``drive``: what the drive's motion did to U under the tick's own context, ``(potential_end -
  projection_work) - potential``;
- ``projection``: ``projection_work``, U's change across what moved the agent after the drive;
- ``context``: ``context_work``, what the sense, the field, the coefficients and the goal changed
  between the ticks (a Retarget books here);
- ``dKE``: the change of 1/2 v^2 (v the speed in normalized units per second);
- ``residual``: ``dU - (drive + projection + context)``, which is zero when nothing moved the agent
  between the ticks (the white paper's telescoping sum). It is what the books do not explain.

So ``dH = dKE + dU`` for ``H_nav = 1/2 v^2 + U_nav``. The cumulative books are their running sums.
The plant ledger (libcvc's ``plant_ledger.h``: the energy and the measurements a physics plant
feeds ``nav_stats``) is read from the collector's JSON, where an item that does not apply is left
out or null; it shows as n/a (on the kinematic plant, every item). pycvc does not bind
``plant_ledger_vehicle()``, so :data:`PLANT_LEDGER` lists its items in libcvc's order.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grl_snam.learning.runlog import SCHEMA, RunLog

#: The bare arm names, in their default order.
BARE = ("trained", "untrained", "constant")
#: The published policy's frame (normalized units per metre), when no ``.pt`` arm records one.
POLICY_SCALE = 1.0 / 300.0
#: Ariadne's chart: palette (light), slots c0..c7: arm ``j`` draws in slot ``j`` everywhere, its
#: ghost, its track and its chart series.
SLOTS = (0x2A78D6, 0xEB6834, 0x1BAF7A, 0xEDA100, 0xE87BA4, 0x008300, 0x6250D6, 0xE34948)
SLOT_NAMES = ("blue", "orange", "green", "amber", "pink", "dark green", "purple", "red")
#: The coefficients in force and the terms of U, as the drive telemetry names them.
COEFFICIENTS = ("alpha", "beta", "gamma", "lam_soft", "lam_hard", "mu")
U_TERMS = ("potential", "u_goal", "u_barrier", "u_soft", "u_hard", "u_terrain")
BOOKS = ("dU", "drive", "projection", "context", "dKE", "residual")
#: The Stats row's columns.
STATS = ("clear_min_m", "clear_mean_m", "pen_steps", "path_m", "elapsed_s", "ms_step", "reached")
#: libcvc's plant ledger, each vehicle's items in its order (src/cvc/nav/plant_ledger.cpp):
#: (key, unit, kind), the kind a ``measure``, a ``count`` or a ``flag`` (0 or 1).
PLANT_LEDGER = (
    ("energy_J", "J", "measure"),
    ("energy_per_km", "J/km", "measure"),
    ("building_contacts", "", "count"),
    ("plant_steps", "", "count"),
    ("rolled_over", "", "flag"),
    ("rolled_over_s", "s", "measure"),
    ("tilt_max_deg", "deg", "measure"),
    ("roll_max_deg", "deg", "measure"),
    ("pitch_max_deg", "deg", "measure"),
    ("tilt_s", "s", "measure"),
    ("ltr_max", "1", "measure"),
    ("lift_s", "s", "measure"),
    ("wheels_up_s", "s", "measure"),
    ("a_lat_max_g", "g", "measure"),
    ("steer_cmd_abs_mean", "rad", "measure"),
    ("steer_sat_share", "1", "measure"),
    ("accel_cmd_mean", "m/s^2", "measure"),
    ("brake_share", "1", "measure"),
    ("vehicle_hits", "", "count"),
    ("vehicle_hit_s", "s", "measure"),
)
#: The clearance libcvc reports before it measured one (a sentinel, not a distance).
_UNMEASURED = 1e20


def slot_rgb(j: int) -> tuple[float, float, float]:
    """Arm ``j``'s colour as (r, g, b) in [0, 1]."""
    c = SLOTS[j % len(SLOTS)]
    return ((c >> 16) / 255.0, ((c >> 8) & 0xFF) / 255.0, (c & 0xFF) / 255.0)


# ── arms ───────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Arm:
    """One arm: its ``name``, the ``spec`` it was given ("" for a bare name), its ``kind``
    (``trained``, ``untrained``, ``const``, ``cvcnav`` or ``pt``) and, once :func:`resolve` ran,
    the ``.cvcnav`` its world loads (``path``, ``sha256``), its source's sha256 and its colour
    slot. ``key`` is the policy key the world gave it."""

    name: str
    spec: str = ""
    kind: str = ""
    path: str = ""
    sha256: str = ""
    source_sha256: str = ""
    slot: int = 0
    key: str = ""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "spec": self.spec,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "source_sha256": self.source_sha256,
            "slot": self.slot,
            "key": self.key,
        }


def _kind(name: str, spec: str) -> str:
    if not spec:
        return {"trained": "trained", "untrained": "untrained", "constant": "const"}[name]
    if spec.startswith("const:"):
        return "const"
    return "cvcnav" if spec.endswith(".cvcnav") else "pt"


def _continues_const(token: str) -> bool:
    """Whether ``token`` (a piece of a comma list) belongs to the ``const:`` before it: a number,
    or ``HEAD=number``."""
    from grl_snam.learning.arms import HEADS

    head, sep, value = token.partition("=")
    try:
        float(value if sep else token)
    except ValueError:
        return False
    return not sep or head.strip() in HEADS


def parse_arms(arms: str | None, extra: Sequence[str] = ()) -> list[Arm]:
    """The arms ``--arms`` (a comma list) and each ``--arm NAME=SPEC`` name, in order.

    An item is a bare name (:data:`BARE`) or ``NAME=SPEC``. In the comma list a ``const:``'s own
    commas stay its own: ``const:1.2,3,4`` and ``c=const:alpha=1.2,gamma=5`` are one arm each.
    Raises ValueError on an unknown bare name, a repeated name, or more arms than colours."""
    from grl_snam.learning.arms import parse_arm

    items: list[str] = []
    for token in (t.strip() for t in (arms or "").split(",")):
        if not token:
            continue
        if items and "=const:" in items[-1] and _continues_const(token):
            items[-1] += "," + token
        else:
            items.append(token)
    items += [a.strip() for a in extra if a.strip()]
    out: list[Arm] = []
    for item in items:
        if "=" not in item:
            if item not in BARE:
                raise ValueError(
                    f"no arm {item!r}: a bare arm is one of {', '.join(BARE)}; "
                    "else NAME=PATH.cvcnav|PATH.pt|const:..."
                )
            name, spec = item, ""
        else:
            a = parse_arm(item)
            name, spec = a.name, a.spec
        if any(o.name == name for o in out):
            raise ValueError(f"the arm {name!r} is named twice")
        out.append(Arm(name, spec, _kind(name, spec), slot=len(out)))
    if not out:
        raise ValueError("--arms names no arm")
    if len(out) > len(SLOTS):
        raise ValueError(f"at most {len(SLOTS)} arms (one colour each), not {len(out)}")
    return out


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pt_scale(path) -> float | None:
    import torch

    ck = torch.load(path, map_location="cpu")
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    s = meta.get("scale") if isinstance(meta, dict) else None
    return float(s) if s else None


def resolve(arms: list[Arm], out_dir, *, seed: int = 0, trained_pt: str | None = None) -> float:
    """Write each arm's policy to ``out_dir/<name>.cvcnav`` (a ``.cvcnav`` arm is used as it is)
    and fill in its ``path`` and hashes. ``trained_pt`` is the bare ``trained`` arm's
    checkpoint (default: ``grl-snam-weights``' ``coef_sdf.pt``). Returns the frame every arm
    runs in (normalized units per metre). Raises RuntimeError when the bare ``trained`` arm has
    no published policy to load."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in arms:
        if a.kind == "trained":
            if trained_pt is None:
                from grl_snam.demos._common import default_nav_weights_pt

                trained_pt = default_nav_weights_pt()
            if not trained_pt or not os.path.exists(trained_pt):
                raise RuntimeError(
                    "the trained arm is the published policy: cvcpkg install grl-snam-weights "
                    "(or name it trained=PATH.pt|PATH.cvcnav)"
                )
            a.spec, a.kind = str(trained_pt), "pt"
    models = {}
    scale = None
    if any(a.kind != "cvcnav" for a in arms):
        import torch

        import grl_snam.sdf_nav as sdf_nav
        from grl_snam.learning.arms import ArmSpec, load_arm_model

        for a in arms:  # the nets first: a constant arm takes the first one's layout
            if a.kind == "untrained":
                torch.manual_seed(int(seed))
                models[a.name] = sdf_nav.CoefMLP().eval()
            elif a.kind in ("pt", "cvcnav"):
                models[a.name] = load_arm_model(ArmSpec(a.name, a.spec))
                if a.kind == "pt" and scale is None:
                    scale = _pt_scale(a.spec)
        like = next(iter(models.values()), None)
        for a in arms:
            if a.kind == "const":
                models[a.name] = load_arm_model(ArmSpec(a.name, a.spec or "const:bias"), like=like)
    from grl_snam.tools.coef_export import write_coef_mlp

    for a in arms:
        if a.kind in ("pt", "cvcnav") and os.path.isfile(a.spec):
            a.source_sha256 = _sha256(a.spec)
        if a.kind == "cvcnav":
            a.path = str(Path(a.spec).resolve())
        else:
            a.path = str(out_dir / f"{a.name}.cvcnav")
            write_coef_mlp(models[a.name], a.path)
        a.sha256 = _sha256(a.path)
    return float(scale or POLICY_SCALE)


# ── the books, the rows ────────────────────────────────────────────────────────────────────────


def _f(tel: Mapping, key: str, default: float = 0.0) -> float:
    v = tel.get(key) if tel else None
    if v is None:
        return default
    return float(np.asarray(v, float).reshape(-1)[0])


def books(tel: Mapping, prev_end: float | None, speed_n: float, prev_speed_n: float) -> dict:
    """One tick's books (:data:`BOOKS`) from its drive telemetry ``tel`` (one agent's values),
    last tick's ``potential_end`` (None on the first tick: then ``dU`` starts at this tick's
    ``potential``) and the speed before and after it (normalized units per second)."""
    u0, u1 = _f(tel, "potential"), _f(tel, "potential_end")
    proj, ctx = _f(tel, "projection_work"), _f(tel, "context_work")
    drive = (u1 - proj) - u0
    du = u1 - (u0 if prev_end is None else prev_end)
    return {
        "dU": du,
        "drive": drive,
        "projection": proj,
        "context": ctx,
        "dKE": 0.5 * (speed_n * speed_n - prev_speed_n * prev_speed_n),
        "residual": du - (drive + proj + ctx),
    }


def stats_row(per_vehicle: Mapping | None, clearance_m: Sequence[float], ms: Sequence[float],
              ticks: int, dt: float) -> dict:  # fmt: skip
    """An arm's Stats row (:data:`STATS`, numbers; NaN where there is nothing yet) from its
    collector's vehicle record (``nav_stats()``' ``per_vehicle[0]``), the leg's per-tick
    clearance (m) and wall times (ms), and its ticks and tick length."""
    v = dict(per_vehicle or {})
    clear = [c for c in clearance_m if math.isfinite(c) and abs(c) < _UNMEASURED]
    cmin = v.get("min_clearance_m")
    if cmin is None or not math.isfinite(float(cmin)) or abs(float(cmin)) >= _UNMEASURED:
        cmin = min(clear) if clear else math.nan
    arrived = bool(v.get("arrived", False))
    ttg = float(v.get("time_to_goal_s", -1.0) or -1.0)
    return {
        "clear_min_m": float(cmin),
        "clear_mean_m": float(np.mean(clear)) if clear else math.nan,
        "pen_steps": int(v.get("penetration_steps", 0) or 0),
        "path_m": float(v.get("total_path_m", 0.0) or 0.0),
        "elapsed_s": ttg if arrived and ttg >= 0.0 else ticks * dt,
        "ms_step": float(np.mean(ms)) if len(ms) else math.nan,
        "reached": arrived,
    }


def _num(v, digits=2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    x = float(v)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def _absent(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def plant_ledger_rows(values: Mapping | None) -> list[dict]:
    """The plant ledger for one vehicle: ``{key, value, unit}`` per :data:`PLANT_LEDGER` item from
    ``values`` (the collector's vehicle record, or a run log's values), the value ``"n/a"`` where
    it is left out, null or NaN (the item does not apply on this plant)."""
    v = dict(values or {})
    rows = []
    for key, unit, kind in PLANT_LEDGER:
        x = v.get(key)
        if _absent(x):
            rows.append({"key": key, "value": "n/a", "unit": ""})
            continue
        text = (
            _num(bool(x)) if kind == "flag" else (_num(int(x)) if kind == "count" else _num(x, 3))
        )
        rows.append({"key": key, "value": text, "unit": unit})
    return rows


def plant_values(per_vehicle: Mapping | None) -> dict:
    """The plant ledger's items as numbers (NaN where one does not apply), for a run's arrays."""
    v = dict(per_vehicle or {})
    return {k: (math.nan if _absent(v.get(k)) else float(v[k])) for k, _, _ in PLANT_LEDGER}


def stats_cells(stats: Mapping | None) -> dict:
    """A Stats row's text from its numbers (:data:`STATS`; "-" before the first tick)."""
    if not stats:
        return {k: "-" for k in ("clear_min", "clear_mean", "pen", "path", "elapsed", "ms",
                                 "reached")}  # fmt: skip
    return {"clear_min": _num(stats["clear_min_m"]), "clear_mean": _num(stats["clear_mean_m"]),
            "pen": _num(int(stats["pen_steps"])), "path": _num(stats["path_m"], 1),
            "elapsed": _num(stats["elapsed_s"], 1), "ms": _num(stats["ms_step"], 3),
            "reached": _num(bool(stats["reached"]))}  # fmt: skip


def panel_tables(arms: Sequence[Arm], stats, tel, book_sum, book_tick) -> dict:
    """The Stats and Ledger windows' tables (state key -> rows), from each arm's numbers: its Stats
    row, its drive telemetry this tick, and its books summed and this tick's (each a list over
    the arms of name -> number)."""
    nan = math.nan
    out = {"arms.stats": [], "ledger.coef": [], "ledger.terms": [], "ledger.books": [],
           "ledger.tick": []}  # fmt: skip
    for a, st, te, bs, bt in zip(arms, stats, tel, book_sum, book_tick, strict=True):
        out["arms.stats"].append({"arm": a.name, "colour": SLOT_NAMES[a.slot % len(SLOT_NAMES)],
                                  **stats_cells(st)})  # fmt: skip
        out["ledger.coef"].append(
            {"arm": a.name, **{k: _num(te.get(k, nan), 3) for k in COEFFICIENTS}}
        )
        out["ledger.terms"].append({"arm": a.name, **{k: _num(te.get(k, nan), 3) for k in U_TERMS}})
        out["ledger.books"].append({"arm": a.name, **{k: _num(bs.get(k, nan), 3) for k in BOOKS}})
        out["ledger.tick"].append({"arm": a.name, **{k: _num(bt.get(k, nan), 3) for k in BOOKS}})
    return out


def chart_series(arms: Sequence[Arm], t, clearance_m, book_sums: Mapping, legs, agent: int,
                 book: str) -> dict:  # fmt: skip
    """The charts' curves (state key -> (series, markers), as ``host.set_plot`` takes them): every
    arm's clearance (m, (T, A)), ``agent``'s cumulative books (``book_sums``: name -> (T, A)
    running sums), and ``book``'s for every arm; a marker where each leg after the first starts
    (``legs``: their start times, s)."""
    t = np.asarray(t, float)
    marks = [(float(x), f"leg {i}", "note") for i, x in enumerate(legs, start=1) if i > 1]
    book = book if book in BOOKS else "dU"
    clearance_m = np.asarray(clearance_m, float)
    return {
        "arms.clearance": ([{"x": t, "y": clearance_m[:, j], "label": a.name,
                             "style": f"c{a.slot}"} for j, a in enumerate(arms)], marks),
        "ledger.cum": ([{"x": t, "y": np.asarray(book_sums[b], float)[:, agent], "label": b}
                        for b in BOOKS], marks),
        "ledger.compare": ([{"x": t, "y": np.asarray(book_sums[book], float)[:, j],
                             "label": a.name, "style": f"c{a.slot}"} for j, a in enumerate(arms)],
                           marks),
    }  # fmt: skip


# ── every arm's run ────────────────────────────────────────────────────────────────────────────


@dataclass
class ArmSeries:
    """One arm's run, a value per tick: the pose (``x``, ``y``, ``heading``, ``speed``, the goal),
    the wall time, the drive telemetry (``tel``: every field), the books, the Stats row, the
    plant ledger's items and the vehicle's size. ``stats`` is the current leg's Stats row,
    ``nav_stats`` its collector's last record, ``legs`` the finished legs' records."""

    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    heading: list = field(default_factory=list)
    speed: list = field(default_factory=list)
    goal_x: list = field(default_factory=list)
    goal_y: list = field(default_factory=list)
    ms: list = field(default_factory=list)
    tel: dict = field(default_factory=dict)
    books: dict = field(default_factory=lambda: {k: [] for k in BOOKS})
    book_sum: dict = field(default_factory=lambda: dict.fromkeys(BOOKS, 0.0))
    stats_series: dict = field(default_factory=lambda: {k: [] for k in STATS})
    plant: dict = field(default_factory=lambda: {k: [] for k, _, _ in PLANT_LEDGER})
    size: dict = field(default_factory=lambda: {k: [] for k in ("length", "width", "wheelbase")})
    stats: dict = field(default_factory=dict)
    nav_stats: dict = field(default_factory=dict)
    legs: list = field(default_factory=list)
    reached_legs: int = 0


class ArmsRun:
    """Every arm's run on one world: the arms, the shared start, goal, seed and tick length, the
    legs (each Retarget's tick and goal), and per arm an :class:`ArmSeries`. The port calls
    :meth:`record` for each arm after each tick, then :meth:`end_tick`; :meth:`retarget` between
    ticks. :meth:`arrays` and :meth:`meta` are what a run log keeps (:meth:`save`)."""

    def __init__(self, arms: list[Arm], *, start, goal, seed=0, dt=0.05, scale=POLICY_SCALE,
                 world=None, plant="", demo="austin-freedrive"):  # fmt: skip
        self.arms = list(arms)
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        self.seed, self.dt, self.scale = int(seed), float(dt), float(scale)
        self.world, self.plant, self.demo = world, plant, demo
        self.tick = 0
        self.t: list[float] = []
        self.legs: list[dict] = [{"tick": 0, "goal": list(self.goal)}]
        self.series = [ArmSeries() for _ in self.arms]
        self._prev_end: list[float | None] = [None] * len(self.arms)
        self._prev_speed = [0.0] * len(self.arms)

    def __len__(self) -> int:
        return len(self.t)

    def index(self, name: str, default: int = 0) -> int:
        """The arm named ``name``'s index (``default`` when there is none)."""
        return next((j for j, a in enumerate(self.arms) if a.name == name), default)

    @property
    def leg_start(self) -> int:
        """The row (tick index) the current leg began at."""
        return int(self.legs[-1]["tick"])

    def record(self, j: int, *, x, y, heading, speed, goal, tel: Mapping | None, ms: float,
               nav_stats: Mapping | None = None, size=(math.nan,) * 3) -> None:  # fmt: skip
        """Arm ``j``'s tick: its pose (m, rad, m/s; the goal (x, y)), its drive telemetry (one
        agent's fields, arrays or numbers), its wall time (ms), its collector's record (the
        ``nav_stats()`` JSON, parsed) and its vehicle's (length, width, wheelbase) (m)."""
        s = self.series[j]
        s.x.append(float(x))
        s.y.append(float(y))
        s.heading.append(float(heading))
        s.speed.append(float(speed))
        s.goal_x.append(float(goal[0]))
        s.goal_y.append(float(goal[1]))
        s.ms.append(float(ms))
        tel = dict(tel or {})
        for k in set(s.tel) | set(tel):
            col = s.tel.setdefault(k, [math.nan] * (len(s.x) - 1))
            col.append(_f(tel, k, math.nan))
        v_n = float(speed) * self.scale
        b = books(tel, self._prev_end[j], v_n, self._prev_speed[j])
        if tel:
            self._prev_end[j] = _f(tel, "potential_end")
        self._prev_speed[j] = v_n
        for k in BOOKS:  # summed in order, as a run log's cumulative sums are
            s.books[k].append(b[k])
            s.book_sum[k] += b[k]
        if nav_stats is not None:
            s.nav_stats = dict(nav_stats)
        leg = slice(self.leg_start, None)
        clear = [c / self.scale for c in s.tel.get("clearance", [])[leg]]
        per = (s.nav_stats.get("per_vehicle") or [{}])[0]
        ticks = int(s.nav_stats.get("ticks", len(s.x) - self.leg_start))
        s.stats = stats_row(per, clear, s.ms[leg], ticks, self.dt)
        for k in STATS:
            s.stats_series[k].append(float(s.stats[k]))
        for k, v in plant_values(per).items():
            s.plant[k].append(v)
        for k, v in zip(("length", "width", "wheelbase"), size, strict=True):
            s.size[k].append(float(v))

    def end_tick(self) -> None:
        self.tick += 1
        self.t.append(self.tick * self.dt)

    def retarget(self, goal) -> None:
        """A new leg from the next tick, every arm to ``goal`` (x, y): the finished leg's records
        are kept, and each arm's books carry on (the goal's change is context work)."""
        for s in self.series:
            if s.stats.get("reached"):
                s.reached_legs += 1
            s.legs.append({"stats": dict(s.stats), "nav_stats": s.nav_stats})
            s.nav_stats = {}
        self.goal = (float(goal[0]), float(goal[1]))
        self.legs.append({"tick": len(self.t), "goal": list(self.goal)})

    @property
    def arrivals(self) -> int:
        """Goals reached, over every arm and leg."""
        return sum(s.reached_legs + int(bool(s.stats.get("reached"))) for s in self.series)

    def cumulative(self, j: int, book: str) -> np.ndarray:
        return np.cumsum(np.asarray(self.series[j].books[book], float))

    # ── what the panels show ──

    def _last(self, s: ArmSeries, k: str) -> float:
        col = s.tel.get(k)
        return col[-1] if col else math.nan

    def tables(self) -> dict:
        """The Stats and Ledger windows' tables now (:func:`panel_tables`)."""
        tel = [{k: self._last(s, k) for k in (*COEFFICIENTS, *U_TERMS)} for s in self.series]
        book_sum = [dict(s.book_sum) for s in self.series]
        book_tick = [{k: (s.books[k][-1] if s.books[k] else math.nan) for k in BOOKS}
                     for s in self.series]  # fmt: skip
        return panel_tables(self.arms, [s.stats for s in self.series], tel, book_sum, book_tick)

    def stats_rows(self) -> list[dict]:
        """One text row per arm (the Stats window)."""
        return self.tables()["arms.stats"]

    def plant_rows(self, j: int) -> list[dict]:
        return plant_ledger_rows(self.series[j].nav_stats.get("per_vehicle", [{}])[0])

    def column(self, get) -> np.ndarray:
        """(T, A): ``get(series)`` (a list a tick) for every arm."""
        return np.stack([np.asarray(get(s), float) for s in self.series], axis=1)

    def charts(self, agent: int, book: str) -> dict:
        """The charts' curves now (:func:`chart_series`)."""
        clearance = self.column(lambda s: s.tel.get("clearance", [math.nan] * len(s.x)))
        sums = {b: np.cumsum(self.column(lambda s, b=b: s.books[b]), axis=0) for b in BOOKS}
        legs = [g["tick"] * self.dt for g in self.legs]
        return chart_series(self.arms, self.t, clearance / self.scale, sums, legs, agent, book)

    # ── the log ──

    def arrays(self) -> dict:
        """The run as arrays: ``t`` (T,), and per field (T, A), one column per arm in order: the
        pose (``x``, ``y``, ``heading``, ``speed``, ``goal_x``, ``goal_y``), ``ms``, ``tel_<field>``
        (every drive telemetry field), ``book_<name>``, ``stats_<column>``, ``plant_<key>`` and
        the size (``length``, ``width``, ``wheelbase``); ``leg_tick`` (L,) and ``leg_goal``
        (L, 2). :class:`~grl_snam.learning.runlog.RunLog` documents the format."""
        out = {"t": np.asarray(self.t, float)}

        def cols(get):
            return np.stack([np.asarray(get(s), float) for s in self.series], axis=1)

        for k in ("x", "y", "heading", "speed", "goal_x", "goal_y", "ms"):
            out[k] = cols(lambda s, k=k: getattr(s, k))
        for k in sorted(set().union(*(s.tel for s in self.series))):
            out[f"tel_{k}"] = cols(lambda s, k=k: s.tel.get(k, [math.nan] * len(s.x)))
        for k in BOOKS:
            out[f"book_{k}"] = cols(lambda s, k=k: s.books[k])
        for k in STATS:
            out[f"stats_{k}"] = cols(lambda s, k=k: s.stats_series[k])
        for k, _, _ in PLANT_LEDGER:
            out[f"plant_{k}"] = cols(lambda s, k=k: s.plant[k])
        for k in ("length", "width", "wheelbase"):
            out[k] = cols(lambda s, k=k: s.size[k])
        out["leg_tick"] = np.asarray([g["tick"] for g in self.legs], int)
        out["leg_goal"] = np.asarray([g["goal"] for g in self.legs], float).reshape(-1, 2)
        return out

    def meta(self) -> dict:
        return {
            "schema": SCHEMA,
            "demo": self.demo,
            "arms": [a.describe() for a in self.arms],
            "seed": self.seed,
            "start": list(self.start),
            "goal": list(self.legs[0]["goal"]),
            "legs": self.legs,
            "dt": self.dt,
            "scale": self.scale,
            "world": self.world,
            "plant": self.plant,
            "nav_stats": [s.nav_stats for s in self.series],
            "finished_legs": [s.legs for s in self.series],
        }

    def save(self, path, /, *, history=None, **meta) -> str:
        """``path`` (.npz): the run as a :class:`~grl_snam.learning.runlog.RunLog`, with ``meta``
        added and ``history`` (the training histories it names) in its provenance."""
        return RunLog.from_run(self, history=history, **meta).save(path)


# ── where a Retarget goes ──────────────────────────────────────────────────────────────────────


def check_goal(world, x: float, y: float, radius: float) -> str:
    """Why ``(x, y)`` is not a goal on ``world`` for a vehicle of footprint ``radius`` (m), or ""
    when it is one: off the world, inside a building, or too tight (closer to one than the
    footprint), read from the world's signed distance."""
    mnx, mny, mxx, mxy = world.bounds
    if not (mnx <= x <= mxx and mny <= y <= mxy):
        return "off the world"
    r, c = world.cells(x, y)
    d = float(world.sdf[r, c])
    if d <= 0.0:
        return "inside a building"
    if d < radius:
        return f"too tight ({d:.1f} m clear, the footprint needs {radius:.1f} m)"
    return ""


def parse_point(text: str | None):
    """``"x, y"`` -> (x, y), or None for an empty or malformed one."""
    parts = [p for p in (text or "").replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def draw_goal(world, rng, positions, radius: float):
    """A fresh goal for every arm: a free point at least ``radius`` clear, and at least 30% of the
    world's span from each of ``positions`` ((A, 2), the arms' poses) when it can be."""
    pos = np.asarray(positions, float).reshape(-1, 2)
    best = None
    for _ in range(64):
        p = world.free_points(rng, 1, max(radius, 0.5), 0.0)[0]
        d = float(np.min(np.hypot(pos[:, 0] - p[0], pos[:, 1] - p[1]))) if len(pos) else math.inf
        if best is None or d > best[0]:
            best = (d, p)
        if d >= 0.3 * float(world.span):
            break
    return float(best[1][0]), float(best[1][1])
