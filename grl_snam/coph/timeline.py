"""What ``grl-snam ariadne coph`` draws, from an episode's messages (:mod:`grl_snam.coph.feed`).

A :class:`Timeline` takes one arm's messages in any order and answers what each agent looked like
at time ``t`` (:meth:`Timeline.state`): its pose and heading, its stages, its learner (version,
``pi``, damping), its energy ledger, its accepts and rejects, its bytes and the obstacles it knows.
:class:`Floor` draws an arm's floor as an image (corridors, the risk field as a heat map, the
stages and the sensor disc), with numpy only. :func:`log_arrays` lays the arms out as the cars
template's ``.npz`` log, plus the ledger columns. Nothing here needs pycvc_gl or a display.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .records import Config

CFG = Config()
#: The ledger's items, in the order the panels and the log show them.
LEDGER = ("work", "heat", "kinetic_change", "shaping", "residual")


@dataclass
class AgentState:
    """One agent at one time (:meth:`Timeline.state`)."""

    agent: int
    t: float
    active: bool
    q: tuple
    v: tuple
    heading: float
    stage: int
    version: int
    done: bool
    policy: tuple
    damping: float
    stages: list
    ledger: dict
    max_residual: float
    accepts: int
    rejects: int
    reasons: Counter
    last_reason: str
    bytes_sent: int
    bytes_received: int
    known: set
    path: float
    outcome: dict | None = None
    track: list = field(default_factory=list)


class _Agent:
    def __init__(self, initial, goal):
        self.initial, self.goal = [float(v) for v in initial], [float(v) for v in goal]
        self.poses = []  # (t, x4, stage, version, done)
        self.steps = []  # (t_end, u, work, heat, kinetic_change, residual, observation_jump)
        self.updates = []  # (t, tick, accepted, reason, shaping_jump, packets, snapshot, stats)
        self.geometry = []  # (t, kind, obstacle)
        self.outcome = None
        self._dirty = False
        self._index = None

    def index(self):
        """Sorted lists and cumulative sums, rebuilt after new messages."""
        if not self._dirty and self._index is not None:
            return self._index
        self.poses.sort(key=lambda p: p[0])
        self.steps.sort(key=lambda s: s[0])
        self.updates.sort(key=lambda u: (u[0], u[1]))
        self.geometry.sort(key=lambda g: g[0])
        q = np.array([p[1][:2] for p in self.poses], float).reshape(-1, 2)
        heading, h = [], math.atan2(self.goal[1] - self.initial[1], self.goal[0] - self.initial[0])
        for p in self.poses:
            vx, vy = p[1][2], p[1][3]
            if vx * vx + vy * vy > 1e-6:
                h = math.atan2(vy, vx)
            heading.append(h)
        steps = np.array([s[2:] for s in self.steps], float).reshape(-1, 5)
        jumps = np.array([u[4] if u[2] else 0.0 for u in self.updates], float)
        self._index = dict(
            pose_t=[p[0] for p in self.poses],
            path=(
                np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
                if len(q)
                else np.zeros(0)
            ),
            heading=heading,
            step_t=[s[0] for s in self.steps],
            step_sum=np.cumsum(steps, axis=0),
            step_max=np.maximum.accumulate(np.abs(steps[:, 3])) if len(steps) else np.zeros(0),
            update_t=[u[0] for u in self.updates],
            jump_sum=np.cumsum(jumps),
            accepts=np.cumsum([1 if u[2] else 0 for u in self.updates]),
            geometry_t=[g[0] for g in self.geometry],
        )
        self._dirty = False
        return self._index


class Timeline:
    """One arm of one episode, built from :mod:`~grl_snam.coph.feed` messages."""

    def __init__(self, demo, spec, arm, *, label=None):
        self.demo, self.spec, self.arm = demo, spec, arm
        self.label = label or arm
        self.agents = [_Agent(spec["initials"][i][:2], spec["goals"][i]) for i in range(2)]
        self.packets = {}  # identity -> (sender, recipient, sender_q, bytes, timestamp)
        self.deliveries = []  # (t, sender, receiver, identity, bytes, accepted, reason)
        self.messages = 0

    # ── input ──────────────────────────────────────────────────────────────────────────────

    def ingest(self, messages):
        for m in messages:
            kind = m[0]
            if kind == "pose":
                _, i, t, x, stage, version, done = m
                self.agents[i].poses.append((t, x, stage, version, done))
            elif kind == "step":
                _, i, t, u, work, heat, dke, residual, jump = m
                self.agents[i].steps.append((t, u, work, heat, dke, residual, jump))
            elif kind == "packet":
                _, identity, sender, recipient, q, nbytes, stamp = m
                self.packets[identity] = (sender, recipient, q, nbytes, stamp)
            elif kind == "update":
                _, i, t, tick, accepted, reason, jump, packets, snap, *stats = m
                stats = stats[0] if stats else {}
                self.agents[i].updates.append(
                    (t, tick, accepted, reason, jump, packets, snap, stats)
                )
                for identity in packets:
                    sender, _, _, nbytes, _ = self.packets[identity]
                    self.deliveries.append((t, sender, i, identity, nbytes, accepted, reason))
            elif kind == "geometry":
                _, i, t, k, obstacle = m
                self.agents[i].geometry.append((t, k, tuple(obstacle)))
            elif kind == "outcome":
                _, i, success, collision, start, finish, gap, cost, parts, nbytes = m
                self.agents[i].outcome = dict(
                    success=success,
                    collision=collision,
                    start_time=start,
                    finish_time=finish,
                    goal_gap=gap,
                    cost=cost,
                    components=parts,
                    bytes=nbytes,
                )
            else:
                raise ValueError(f"unknown message {kind!r}")
            if kind != "packet":
                self.agents[m[1]]._dirty = True
            self.messages += 1
        self.deliveries.sort(key=lambda d: (d[0], d[2], d[3]))
        return self

    @classmethod
    def from_episode(cls, demo, record, *, label=None):
        """A finished episode (a T1 episode file's record)."""
        from .feed import messages

        arm = record.get("arm", record.get("spec", {}).get("arm"))
        return cls(demo, record["spec"], arm, label=label).ingest(messages(record))

    # ── queries ────────────────────────────────────────────────────────────────────────────

    @property
    def finished(self):
        return all(a.outcome is not None for a in self.agents)

    @property
    def duration(self):
        """The latest control boundary of either agent (0 before any)."""
        return max((p[0] for a in self.agents for p in a.poses), default=0.0)

    def start(self, i):
        return self.spec.get("start_seconds", [0.0, 0.0])[i]

    def state(self, i, t, *, track=False):
        """Agent ``i`` at time ``t``: the last boundary at or before ``t`` (before its first, its
        start pose, inactive)."""
        a = self.agents[i]
        ix = a.index()
        k = bisect.bisect_right(ix["pose_t"], t + 1e-9) - 1
        if k < 0:
            x, stage, version, done = a.initial + [0.0, 0.0], 0, 0, False
            heading = ix["heading"][0] if ix["heading"] else 0.0
            path = 0.0
        else:
            _, x, stage, version, done = a.poses[k]
            heading, path = ix["heading"][k], float(ix["path"][k])
        s = bisect.bisect_right(ix["step_t"], t + 1e-9) - 1
        sums = ix["step_sum"][s] if s >= 0 else np.zeros(5)
        u = bisect.bisect_right(ix["update_t"], t + 1e-9) - 1
        if u >= 0:
            snap = a.updates[u][6]
            policy, damping, stages = tuple(snap["policy"]), snap["damping"], snap["stages"]
            accepts = int(ix["accepts"][u])
            rejects = u + 1 - accepts
            reasons = Counter(x[3] for x in a.updates[: u + 1])
            last_reason, jumps = a.updates[u][3], float(ix["jump_sum"][u])
        else:
            policy, damping, stages = tuple(CFG.policy_initial), CFG.damping_initial, []
            accepts = rejects = 0
            reasons, last_reason, jumps = Counter(), "", 0.0
        g = bisect.bisect_right(ix["geometry_t"], t + 1e-9)
        known = {o for _, _, o in a.geometry[:g]}
        sent = sum(d[4] for d in self.deliveries if d[0] <= t + 1e-9 and d[1] == i)
        received = sum(d[4] for d in self.deliveries if d[0] <= t + 1e-9 and d[2] == i)
        return AgentState(
            agent=i,
            t=t,
            active=k >= 0,
            q=(float(x[0]), float(x[1])),
            v=(float(x[2]), float(x[3])),
            heading=heading,
            stage=int(stage),
            version=int(version),
            done=bool(done),
            policy=policy,
            damping=float(damping),
            stages=stages,
            ledger=dict(
                work=float(sums[0]),
                heat=float(sums[1]),
                kinetic_change=float(sums[2]),
                shaping=float(sums[4]) + jumps,
                residual=float(sums[3]),
            ),
            max_residual=float(ix["step_max"][s]) if s >= 0 else 0.0,
            accepts=accepts,
            rejects=rejects,
            reasons=reasons,
            last_reason=last_reason,
            bytes_sent=int(sent),
            bytes_received=int(received),
            known=known,
            path=path,
            outcome=a.outcome if a.outcome is not None and k == len(a.poses) - 1 else None,
            track=[p[1][:2] for p in a.poses[: k + 1]] if track else [],
        )

    def deliveries_between(self, t0, t1):
        """Packets delivered in ``(t0, t1]``."""
        return [d for d in self.deliveries if t0 < d[0] <= t1 + 1e-9]

    def times(self):
        """Every control boundary of either agent, sorted."""
        return sorted({p[0] for a in self.agents for p in a.poses})

    def doors(self):
        """The episode's doors (crossing): ``{row: slab}``."""
        return {k: tuple(v) for k, v in self.spec.get("doors", {}).items()}


# ── the floor ──────────────────────────────────────────────────────────────────────────────────


def bounds(spec, margin=1.0):
    """``(x0, x1, y0, y1)``: the corridor network with ``margin`` metres around it."""
    nodes = np.asarray(spec["network_nodes"], float)
    return (
        float(nodes[:, 0].min() - margin),
        float(nodes[:, 0].max() + margin),
        float(nodes[:, 1].min() - margin),
        float(nodes[:, 1].max() + margin),
    )


#: The heat map's colours, low to high: the corridor's floor, amber, deep red.
_STOPS = np.array([[0.84, 0.83, 0.79], [0.97, 0.66, 0.24], [0.62, 0.08, 0.10]])
_OFF = np.array([0.17, 0.19, 0.21])  # off the corridor network


def heat(values, vmax):
    """``values`` (h, w) in 0..vmax as RGB in 0..1, through :data:`_STOPS`."""
    s = np.clip(np.nan_to_num(values, nan=0.0) / max(vmax, 1e-9), 0.0, 1.0) * 2.0
    lo = np.minimum(s.astype(int), 1)
    f = (s - lo)[..., None]
    return _STOPS[lo] * (1 - f) + _STOPS[lo + 1] * f


class Floor:
    """An arm's floor as an image: the corridors, a risk field as a heat map over them, and
    overlays. Row 0 is the south edge (``y0``), column 0 the west edge (``x0``)."""

    def __init__(self, spec, world=None, px_per_m=32):
        from .maze_geometry import distance_to_network

        self.spec, self.px = spec, px_per_m
        self.x0, self.x1, self.y0, self.y1 = bounds(spec)
        self.w = int(round((self.x1 - self.x0) * px_per_m))
        self.h = int(round((self.y1 - self.y0) * px_per_m))
        xs = self.x0 + (np.arange(self.w) + 0.5) / px_per_m
        ys = self.y0 + (np.arange(self.h) + 0.5) / px_per_m
        self.points = np.stack(np.meshgrid(xs, ys), axis=-1)  # (h, w, 2)
        nodes = [tuple(n) for n in spec["network_nodes"]]
        edges = [tuple(e) for e in spec["network_edges"]]
        self.corridor = (
            distance_to_network(self.points, nodes=nodes, edges=edges) <= spec["corridor_radius"]
        )
        self.truth = world.risk_truth(self.points) if world is not None else None
        #: A scale both the truth and an agent's field share.
        self.vmax = max(float(np.max(self.truth)) if self.truth is not None else 0.0, 0.25)

    def agent_risk(self, stages):
        """An agent's own risk field ``r`` over its kept stages (the newest stage wins where they
        overlap), NaN where it has none."""
        import torch

        from .model import risk, tensor

        r = np.full(self.points.shape[:2], np.nan)
        for s in sorted(stages, key=lambda s: s["id"]):
            x0, x1, y0, y1 = s["bounds"]
            mask = (
                (self.points[..., 0] >= x0)
                & (self.points[..., 0] <= x1)
                & (self.points[..., 1] >= y0)
                & (self.points[..., 1] <= y1)
            )
            if not mask.any():
                continue
            with torch.no_grad():
                values = risk(
                    tensor(self.points[mask]),
                    tensor(s["weights"]),
                    tensor(s["centers"]),
                    CFG.rbf_width,
                    s.get("local_residual"),
                )
            r[mask] = values.numpy()
        return r

    def image(self, field=None, overlays=()):
        """The floor as uint8 RGB (h, w, 3). ``field`` (h, w) is drawn as a heat map on the
        corridors; each overlay is ``(mask, rgb, alpha)``."""
        rgb = np.empty(self.points.shape[:2] + (3,))
        rgb[:] = _OFF
        floor = heat(field, self.vmax) if field is not None else _STOPS[0]
        known = self.corridor & ~np.isnan(field) if field is not None else self.corridor
        rgb[self.corridor] = _STOPS[0]
        rgb[known] = floor[known] if field is not None else _STOPS[0]
        for mask, colour, alpha in overlays:
            rgb[mask] = rgb[mask] * (1 - alpha) + np.asarray(colour) * alpha
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def ring(self, centre, radius, width_px=1.5):
        d = np.linalg.norm(self.points - np.asarray(centre, float), axis=-1)
        return np.abs(d - radius) <= width_px / self.px

    def outline(self, rect, width_px=1.5):
        x0, x1, y0, y1 = rect
        p, e = self.points, width_px / self.px
        inside = (p[..., 0] >= x0 - e) & (p[..., 0] <= x1 + e)
        inside &= (p[..., 1] >= y0 - e) & (p[..., 1] <= y1 + e)
        near = (np.abs(p[..., 0] - x0) <= e) | (np.abs(p[..., 0] - x1) <= e)
        near |= (np.abs(p[..., 1] - y0) <= e) | (np.abs(p[..., 1] - y1) <= e)
        return inside & near


# ── the log ────────────────────────────────────────────────────────────────────────────────────

#: The cars template's per-tick fields (pycvc_gl.cars.LOG_FIELDS), in its order.
# ── what the learners measured ─────────────────────────────────────────────────────────────────

#: :func:`learning`'s series: one value per update.
LEARNING = ("t", "version", "accepted", "field_before", "field_proposed", "field_heldout",
            "plant_heldout", "replay", "gradient_norm", "pi_risk", "pi_barrier", "damping",
            "error_seen", "error_seen_lo", "error_seen_hi", "error_unseen")  # fmt: skip


def _num(x):
    return math.nan if x is None else float(x)


def field_error(floor, stages, seen):
    """How far an agent's own field is from the truth over the corridor cells of ``floor`` that
    its kept stages cover (the newest stage wins where they overlap): the mean, 10th and 90th
    percentile of ``|r - r_true|`` where it has sensed (``seen``, a mask over ``floor.points``),
    and the mean elsewhere. NaN where there are no such cells."""
    r = floor.agent_risk(stages)
    covered = floor.corridor & ~np.isnan(r)
    if floor.truth is None:
        return (math.nan,) * 4
    e = np.abs(r - floor.truth)
    a, b = e[covered & seen], e[covered & ~seen]
    if not a.size:
        return (math.nan, math.nan, math.nan, float(b.mean()) if b.size else math.nan)
    return (float(a.mean()), float(np.percentile(a, 10)), float(np.percentile(a, 90)),
            float(b.mean()) if b.size else math.nan)  # fmt: skip


def learning(timeline, agent, floor=None, until=None):
    """What one agent's learner measured at each of its updates, up to time ``until`` (all of them
    when None), as arrays named by :data:`LEARNING`.

    - ``field_heldout`` and ``plant_heldout``: the held-out field and plant losses of the model in
      force after the update. They are measured on the learner's validation records, which its
      gate compares: the gate's ``after`` when it accepted, ``before`` when it did not.
      ``field_before`` is the model's it started from, and ``field_proposed`` the
      proposal's either way.
    - ``replay``: the replay cost after the update's line search, NaN where it could not replay.
    - ``pi_risk``, ``pi_barrier`` and ``damping``: the learner after the update.
    - ``error_seen`` (with its 10th and 90th percentiles) and ``error_unseen``, with ``floor`` (a
      coarse :class:`Floor` built with the world): :func:`field_error` after the update, where
      the agent has sensed being the cells within its sensor radius of the poses so far. A floor
      at 4 px/m costs about 2k risk evaluations an update; each update's value is kept.
    """
    a = timeline.agents[agent]
    a.index()  # sorts the poses and updates
    cache = a.__dict__.setdefault("_field_error", {})
    seen = np.zeros(floor.points.shape[:2], bool) if floor is not None else None
    radius2, poses, k = CFG.sensor_radius**2, a.poses, 0
    rows = []
    for u in a.updates:
        t, tick, accepted, snap = u[0], u[1], u[2], u[6]
        if until is not None and t > until + 1e-9:
            break
        stats = u[7] if len(u) > 7 else {}
        before, after = _num(stats.get("field_before")), _num(stats.get("field_after"))
        plant = _num(stats.get("plant_after" if accepted else "plant_before"))
        err = (math.nan,) * 4
        if floor is not None:
            while k < len(poses) and poses[k][0] <= t + 1e-9:
                d = floor.points - np.asarray(poses[k][1][:2])
                seen |= (d * d).sum(-1) <= radius2
                k += 1
            key = (t, tick, id(floor))
            if key not in cache:
                cache[key] = field_error(floor, snap["stages"], seen)
            err = cache[key]
        rows.append((
            t, snap["version"], float(accepted), before, after, after if accepted else before,
            plant,
            _num(stats.get("replay_after")), _num(stats.get("gradient_norm")), *snap["policy"],
            snap["damping"], *err,
        ))  # fmt: skip
    arr = np.array(rows, float).reshape(-1, len(LEARNING))
    return {k: arr[:, j] for j, k in enumerate(LEARNING)}


LOG_FIELDS = ("x", "y", "heading", "speed", "steer", "vx", "vy", "goal_x", "goal_y", "goal_dist",
              "reached", "collided", "cmd_steer", "cmd_accel")  # fmt: skip
#: The columns this log adds as ``coef_<name>``: the ledger and the learner.
COEF_FIELDS = (*LEDGER, "max_residual", "version", "pi_risk", "pi_barrier", "damping",
               "accepts", "rejects", "bytes_sent", "bytes_received", "ux", "uy", "active")  # fmt: skip


def log_arrays(timelines, meta=None):
    """The arms as the cars template's ``.npz`` log (``pycvc_gl.cars.CarsLog``): arrays (ticks,
    columns) named as :data:`LOG_FIELDS`, ``t``, ``coef_<name>`` for :data:`COEF_FIELDS`, and
    ``meta`` (JSON). A column is one agent of one arm (``meta["columns"]``); a tick is any control
    boundary of any arm, each agent at its last boundary then. CoPH drives a point with a planar
    force, so ``steer``, ``cmd_steer`` and ``cmd_accel`` are NaN and the force is ``coef_ux`` and
    ``coef_uy``."""
    import json

    ticks = sorted({t for tl in timelines for t in tl.times()})
    rows = {k: [] for k in LOG_FIELDS}
    coef = {k: [] for k in COEF_FIELDS}
    columns = [dict(arm=tl.arm, label=tl.label, agent=i) for tl in timelines for i in range(2)]
    for t in ticks:
        row = {k: [] for k in LOG_FIELDS}
        crow = {k: [] for k in COEF_FIELDS}
        for tl in timelines:
            for i in range(2):
                s = tl.state(i, t)
                a = tl.agents[i]
                goal = a.goal
                gap = math.hypot(s.q[0] - goal[0], s.q[1] - goal[1])
                k = bisect.bisect_right(a.index()["step_t"], t + 1e-9) - 1
                u = a.steps[k][1] if k >= 0 else [0.0, 0.0]
                out = a.outcome if s.done else None
                vals = dict(
                    x=s.q[0],
                    y=s.q[1],
                    heading=s.heading,
                    speed=math.hypot(*s.v),
                    steer=math.nan,
                    vx=s.v[0],
                    vy=s.v[1],
                    goal_x=goal[0],
                    goal_y=goal[1],
                    goal_dist=gap,
                    reached=float(bool(out and out["success"])),
                    collided=float(bool(out and out["collision"])),
                    cmd_steer=math.nan,
                    cmd_accel=math.nan,
                )
                for f in LOG_FIELDS:
                    row[f].append(vals[f])
                cvals = dict(
                    s.ledger,
                    max_residual=s.max_residual,
                    version=s.version,
                    pi_risk=s.policy[0],
                    pi_barrier=s.policy[1],
                    damping=s.damping,
                    accepts=s.accepts,
                    rejects=s.rejects,
                    bytes_sent=s.bytes_sent,
                    bytes_received=s.bytes_received,
                    ux=u[0],
                    uy=u[1],
                    active=float(s.active),
                )
                for f in COEF_FIELDS:
                    crow[f].append(cvals[f])
        for f in LOG_FIELDS:
            rows[f].append(row[f])
        for f in COEF_FIELDS:
            coef[f].append(crow[f])
    out = {k: np.array(v, float).reshape(len(ticks), len(columns)) for k, v in rows.items()}
    out["t"] = np.array(ticks, float)
    for k, v in coef.items():
        out[f"coef_{k}"] = np.array(v, float).reshape(len(ticks), len(columns))
    out["meta"] = np.array(json.dumps(dict(meta or {}, columns=columns), default=str))
    return out


# ── the comparison the demo headlines ──────────────────────────────────────────────────────────


def comparison(demo, base, other):
    """This run's own number for the headline, once both arms have finished, else None.

    - crossing: each agent's path in each arm and the route ``other`` saves against ``base``,
      as T1's results count it (path length over the samples; the shortest route cancels);
    - hazard: the carrier's cost in each arm, time plus risk, as T1's results count it."""
    if not (base.finished and other.finished):
        return None
    if demo == "crossing":
        paths = [[tl.state(i, tl.duration).path for i in range(2)] for tl in (base, other)]
        return dict(saving=[paths[0][i] - paths[1][i] for i in range(2)], paths=paths)
    cost = []
    for tl in (base, other):
        parts = tl.agents[1].outcome["components"]
        cost.append(parts["time"] + parts["risk"])
    return dict(carrier_cost=cost)
