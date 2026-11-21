"""G5–G8 of libcvc's CoPH golden: the learner's field fit, damping fit, replay and update.

``grl_snam.tools.coph_golden`` writes these sections into ``coph_golden.json`` (libcvc's
``NAV-COPH-ROADMAP.md`` §3.4, §5; C3 in §4.3). They pin ``CrossingLearner.update`` (the demos'
learner: ``MazeLearner`` on ``TerrainLearner`` on ``CooperativeLearner`` on ``Learner``) at three
update snapshots of a scripted crossing run.

The run is driven by a script, not by the learner's own controller, so its inputs are the same
bytes on every platform:
- The plant is the exact flow at the world's damping (0.85), evaluated in plain Python with
  ``exp`` and ``expm1`` as fixed Taylor sums, so no libm is involved.
- The force tracks a carrot that moves along fixed waypoints, with only ``+ - * /`` and ``sqrt``.
- The readings are a rational risk bump on the demos' 21-point sensor lattice.
- The learner observes, advances and updates every ``update_every`` ticks, in ``execute``'s order.
  After each update the script rounds what the update committed (the field weights, damping,
  policy and the terrain anchors built from the weights) to multiples of ``COMMIT_GRID``. Torch's
  last bits differ between platforms; the rounded values do not.

Every update's decisions, and every replay's in G7, are checked against two perturbed copies,
with the committed values nudged by 1e-12 relative each way. The run fails if any decision
changes, so no decision sits where a platform's last bits could flip it. As in G4, every tick's
geometry decisions that go through a BLAS dot product are kept more than ``coph_golden.MARGIN``
from their thresholds.
"""

from __future__ import annotations

import contextlib
import copy
import math

import numpy as np
import torch

from grl_snam.coph import replay_budget, stage_prediction
from grl_snam.coph.cooperative import CommunicationConfig
from grl_snam.coph.crossing_demo import EDGES, NODES, CrossingLearner, door
from grl_snam.coph.model import features, tensor
from grl_snam.coph.records import Config, Observation, Reading, Transition
from grl_snam.coph.shapes import is_slab, slab_distance

# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

AGENT_ID = "0"
GOAL = (12.0, 4.0)
WORLD_DAMPING = 0.85
# The carrot moves at CARROT_SPEED along the waypoints: east along the middle row until the
# middle door is in view, then back west.
WAYPOINTS = [(0.0, 4.0), (8.6, 4.0), (0.6, 4.0)]
CARROT_SPEED = 0.8  # m/s
GAIN_P, GAIN_D = 2.0, 2.5
FORCE_LIMIT = 3.0
OBSTACLES = [door("middle"), door("north")]
# The demos' sensor (world.sense): the 5 x 5 lattice on [-1, 1]^2 inside the unit disk, times the
# sensor radius. Reading j is validation when j % 5 == 0.
LATTICE = [(a, b) for a in (-1.0, -0.5, 0.0, 0.5, 1.0) for b in (-1.0, -0.5, 0.0, 0.5, 1.0)
           if a * a + b * b <= 1.0]  # fmt: skip
COMMIT_GRID = 2.0**-24
NUDGE = 1e-12


def risk_truth(x: float, y: float, tick: int, j: int) -> float:
    """A rational bump just north of the middle row's centre line, plus a small fixed pattern
    standing in for noise."""
    bump = 2.5 / (1.0 + ((x - 5.0) * (x - 5.0) + (y - 4.45) * (y - 4.45)) / 0.35)
    pattern = 0.003 * (((7 * tick + 13 * j) % 11) - 5) / 5.0
    return max(0.001, 0.04 + bump + pattern)


def _taylor_exp_neg(z: float) -> float:
    """exp(-z) for 0 <= z < 1, as a fixed sum: the same bits everywhere."""
    total, term = 1.0, 1.0
    for k in range(1, 30):
        term = term * (-z) / k
        total = total + term
    return total


def _taylor_neg_expm1_neg(z: float) -> float:
    """-expm1(-z) = 1 - exp(-z) for 0 <= z < 1, as a fixed sum."""
    total, term = 0.0, 1.0
    for k in range(1, 30):
        term = term * (-z) / k
        total = total - term
    return total


def script(cfg: Config, ticks: int) -> list[dict]:
    """The scripted run, in plain Python floats.

    Tick t holds the state at t, the force held from t to t + 1, the readings sensed at t and the
    visible obstacles. The state follows the exact flow at ``WORLD_DAMPING``.
    """
    d, dt = WORLD_DAMPING, cfg.dt
    a = _taylor_exp_neg(d * dt)
    integral = _taylor_neg_expm1_neg(d * dt) / d
    legs = []
    for p, q in zip(WAYPOINTS, WAYPOINTS[1:]):
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        legs.append((p, q, length))
    total = sum(length for *_, length in legs)

    def carrot(s: float):
        s = min(s, total)
        for p, q, length in legs:
            if s <= length:
                f = s / length
                ux, uy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
                return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f,
                        ux * CARROT_SPEED, uy * CARROT_SPEED)  # fmt: skip
            s = s - length
        p, q, _ = legs[-1]
        return (q[0], q[1], 0.0, 0.0)

    x = [WAYPOINTS[0][0], WAYPOINTS[0][1], 0.0, 0.0]
    out = []
    for tick in range(ticks):
        cx, cy, cvx, cvy = carrot(tick * dt * CARROT_SPEED)
        ux = GAIN_P * (cx - x[0]) + GAIN_D * (cvx - x[2])
        uy = GAIN_P * (cy - x[1]) + GAIN_D * (cvy - x[3])
        norm = math.sqrt(ux * ux + uy * uy)
        if norm > FORCE_LIMIT:
            ux, uy = ux * (FORCE_LIMIT / norm), uy * (FORCE_LIMIT / norm)
        readings = []
        for j, (ox, oy) in enumerate(LATTICE):
            px, py = x[0] + ox * cfg.sensor_radius, x[1] + oy * cfg.sensor_radius
            readings.append((px, py, risk_truth(px, py, tick, j)))
        visible = []
        for k, o in enumerate(OBSTACLES):
            q = np.array(x[:2])
            gap = float(slab_distance(q, o)) if is_slab(o) else math.hypot(*(q - o[:2])) - o[2]
            if abs(gap - cfg.sensor_radius) <= 1e-9:
                raise AssertionError(f"tick {tick}: obstacle {k} at the sensor's edge")
            if gap <= cfg.sensor_radius:
                visible.append(k)
        out.append(dict(state=tuple(x), u=(ux, uy), readings=readings, visible=visible))
        qx = x[0] + integral * x[2] + ((dt - integral) * ux) / d
        qy = x[1] + integral * x[3] + ((dt - integral) * uy) / d
        x = [qx, qy, a * x[2] + integral * ux, a * x[3] + integral * uy]
    return out


# ---------------------------------------------------------------------------
# Driving the learner
# ---------------------------------------------------------------------------


def _grid(v: float) -> float:
    """``v`` to the nearest multiple of ``COMMIT_GRID``, refusing a value within 1e-13 relative
    of a rounding boundary, where another platform's last bits could round it the other way."""
    v = float(v)
    k = v / COMMIT_GRID
    if abs((k - math.floor(k)) - 0.5) <= abs(k) * 1e-13 + 1e-300:
        raise AssertionError(f"{v!r} sits on a rounding boundary of the commit grid")
    return round(k) * COMMIT_GRID


def round_committed(agent) -> None:
    """Round what an update committed to the commit grid (see the module docstring)."""
    for s in agent.stages:
        s.weights = tensor([_grid(w) for w in s.weights.tolist()])
    agent.policy = tensor([_grid(p) for p in agent.policy.tolist()])
    agent.damping = _grid(agent.damping)
    agent.local_cells = {
        k: (t, (*a[:2], *(_grid(v) for v in a[2:]))) for k, (t, a) in agent.local_cells.items()
    }
    agent.local_model_anchors = [v[1] for v in agent.local_cells.values()]
    agent.refresh_terrain()


def make_learner(cfg: Config, first: dict, agent_id: str = AGENT_ID):
    comm = CommunicationConfig(byte_budget=4000000, ttl=300.0)
    agent = CrossingLearner(cfg, np.array(first["state"]), GOAL, agent_id=agent_id,
                            communication=comm)  # fmt: skip
    return agent


def observation(cfg, tick: int, t: dict) -> Observation:
    now = tick * cfg.dt
    readings = tuple(
        Reading(f"{tick}:{j}", now, (x, y), v) for j, (x, y, v) in enumerate(t["readings"])
    )
    visible = tuple(OBSTACLES[k] for k in t["visible"])
    return Observation(now, readings, visible, tuple(t["state"][:2]), cfg.sensor_radius)


def _nudged(agent, sign: float):
    out = copy.deepcopy(agent)
    f = 1.0 + sign * NUDGE
    for s in out.stages:
        s.weights = s.weights * f
    out.policy = out.policy * f
    out.damping = out.damping * f
    return out


def decisions(report: dict) -> dict:
    """An update report's decisions: what must match bit for bit."""
    trace = report.get("policy_trace") or {}
    plan = [
        dict(stage=r["stage"], horizon=r["horizon"], reason=r["reason"], eligible=r["eligible"],
             pilot_steps=r["pilot_steps"],
             controls=[(c.get("reason"), c.get("route_reason"), tuple(c.get("waypoint") or ()),
                        tuple(c.get("route_nodes") or ())) for c in r["pilot_controls"]],
             handoffs=[(h["step"], h["from_stage"], h["to_stage"]) for h in r["pilot_handoffs"]])
        for r in report.get("replay_plan", [])
    ]  # fmt: skip
    return dict(
        accepted=report["accepted"],
        reason=report["reason"],
        eligibility=[
            (e["stage"], e["eligible"], e["reason"]) for e in report["replay_eligibility"]
        ],
        plan=plan,
        attempts=[(a["power"], a["reason"], a["accepted"]) for a in trace.get("attempts", [])],
        exhausted=report.get("replay_budget_exhausted"),
        independent=report.get("policy_independent_replay"),
        steps=report["counterfactual_steps"],
        version=report["version_after"],
    )


@contextlib.contextmanager
def capture_prepare(holder: dict):
    """Keep a copy of the learner as ``Learner.update`` leaves ``replay_budget.prepare``."""
    original = replay_budget.prepare

    def wrapped(agent, weights, damping):
        original(agent, weights, damping)
        holder["agent"] = copy.deepcopy(agent)
        holder["weights"] = [w.detach().clone() for w in weights]
        holder["damping"] = damping

    replay_budget.prepare = wrapped
    try:
        yield
    finally:
        replay_budget.prepare = original


def traced_update(agent, x):
    agent.decision_tracing = True
    holder: dict = {}
    with capture_prepare(holder):
        report = agent.update(np.array(x))
    return report, holder


def _check_geometry(agent, cfg, centers, q) -> None:
    """Refuse a tick whose geometry decisions sit within ``coph_golden.MARGIN`` of a threshold.

    As in G4, only the decisions that go through a BLAS dot product are checked, since a
    platform's dot may differ in the last bit: the footprint against each stage ``observe`` saw,
    the geometry store's distance filter, and each obstacle against each kept stage.
    """
    from grl_snam.coph.shapes import reach
    from grl_snam.tools.coph_golden import _intersects_margin, _margin

    for c in centers:
        _margin("footprint_intersects", _intersects_margin(q, cfg.sensor_radius, c, cfg))
    v = np.asarray(agent.current_state[2:])
    stop = float(np.linalg.norm(v)) ** 2 / (2 * cfg.force_limit) + cfg.sensor_radius
    for o in OBSTACLES:
        gap = float(np.linalg.norm(np.asarray(q) - o[:2])) - (stop + reach(o) + 0.17)
        _margin("store_distance", gap)
        for s in agent.stages:
            _margin("obstacle_intersects", _intersects_margin(o[:2], reach(o), s.center, cfg))


def run(cfg: Config, ticks: int, snapshot_ticks=(), check=True, on_snapshot=None):
    """Drive the learner along the script; return the script (with each tick's version) and each
    update's decisions. At a snapshot tick, ``on_snapshot(tick, before, report, holder)`` gets a
    copy of the learner before the update, the update's report and ``capture_prepare``'s holder.
    """
    sc = script(cfg, ticks)
    agent = make_learner(cfg, sc[0])
    log = []
    for tick, t in enumerate(sc):
        obs = observation(cfg, tick, t)
        centers = [np.asarray(s.center) for s in agent.stages]
        if tick == 0:
            agent.observe(obs)
        else:
            prev = sc[tick - 1]
            tr = Transition(tick - 1, obs.time, tuple(prev["state"]), tuple(prev["u"]),
                            tuple(t["state"]), prev["version"])  # fmt: skip
            agent.observe(obs, tr)
            agent.advance(np.array(t["state"]))
        _check_geometry(agent, cfg, centers, t["state"][:2])
        if tick % cfg.update_every == 0:
            before = copy.deepcopy(agent)
            report, holder = traced_update(agent, t["state"])
            mine = decisions(report)
            if check:
                for sign in (1.0, -1.0):
                    other, _ = traced_update(_nudged(before, sign), t["state"])
                    if decisions(other) != mine:
                        raise AssertionError(f"tick {tick}: a {NUDGE} nudge changes the update")
            if tick in snapshot_ticks and on_snapshot is not None:
                on_snapshot(tick, before, report, holder)
            round_committed(agent)
            log.append(dict(tick=tick, **mine))
        t["version"] = agent.version
    return sc, log


# ---------------------------------------------------------------------------
# The snapshots and G5-G8
# ---------------------------------------------------------------------------

SNAPSHOTS = (
    ("near_the_bump", 64),
    ("door_in_view_rejected", 96),
    ("back_west_unsupported_entry", 112),
)
# G7's policies: the snapshot's committed policy first, then these.
REPLAY_POLICIES = [(0.0, 0.0), (10.0, 10.0), (0.25, 6.0), (6.0, 0.25), (3.0, 3.0)]
# FLOAT64 after a linear solve or an iterated step (NAV-COPH-ROADMAP §5): rtol 1e-9, with an atol
# for gradients that cancel to small values.
TIER = dict(rtol=1e-9, atol=1e-12)
SLACK_MARGIN = 1e-5  # G8: |validation change - slack| is at least this


def _index(uid: str) -> int:
    """A reading's index in the script: '<agent>:<tick>:<j>' -> len(LATTICE) * tick + j."""
    _, tick, j = uid.split(":")
    return len(LATTICE) * int(tick) + int(j)


def snapshot(name, tick, agent, cfg, sc) -> dict:
    """The learner's state before the update: everything ``update`` reads."""
    from grl_snam.tools.coph_golden import _hexes, _runs, hexbits

    footprint_tick = {
        (*sc[k]["state"][:2], cfg.sensor_radius, k * cfg.dt): k for k in range(tick + 1)
    }
    nav = agent.navigation
    assert set(nav) == {"maze_graph", "network", "reported_geometry", "terrain_anchors"}, set(nav)
    assert tuple(map(tuple, nav["network"][0])) == NODES
    assert tuple(map(tuple, nav["network"][1])) == EDGES
    assert not agent.received and not agent.remote_cells and not agent.geometry_overflow
    stages = []
    for s in agent.stages:
        assert not s.anchors and getattr(s, "local_residual", None) is None
        stages.append(dict(
            id=s.id, center=_hexes(s.center), entry=_hexes(s.entry), target=_hexes(s.target),
            version=s.version, weights=_hexes(s.weights.tolist()),
            readings=_runs([_index(r.uid) for r in s.readings]),
            transitions=_runs([t.uid for t in s.transitions]),
            footprints=_runs([footprint_tick[tuple(f)] for f in s.footprints]),
            obstacles=[OBSTACLES.index(o) for o in s.obstacles],
        ))  # fmt: skip
    return dict(
        name=name, tick=tick, now=hexbits(agent.now), x=_hexes(sc[tick]["state"]),
        version=agent.version, damping=hexbits(agent.damping), policy=_hexes(agent.policy.tolist()),
        geometry=[OBSTACLES.index(o) for o in agent.geometry_store],
        reported_geometry=[OBSTACLES.index(tuple(o)) for o in nav["reported_geometry"]],
        terrain_anchors=[_hexes(a) for a in nav["terrain_anchors"]],
        stages=stages,
    )  # fmt: skip


def field_fit_case(agent) -> dict:
    """G5: the vendored fit's normal equations, block by block, and its solution."""
    from grl_snam.tools.coph_golden import _hexes

    cfg = agent.cfg
    m = len(agent.stages[0].weights)
    total = len(agent.stages) * m
    rows, targets, owner = [], [], []
    for k, s in enumerate(agent.stages):
        for r in s.readings:
            if int(r.uid.split(":")[-1]) % 5 == 0:
                continue
            row = torch.zeros(total, dtype=torch.float64)
            row[k * m : (k + 1) * m] = features(tensor(r.q), s.centers, cfg.rbf_width)
            rows.append(row)
            targets.append(float(np.log(np.expm1(4 * max(r.value, 0.001)))))
            owner.append(k)
    A, y = torch.stack(rows), tensor(targets)
    normal = A.T @ A + cfg.fit_ridge * torch.eye(total, dtype=torch.float64)
    prior = torch.cat([s.weights for s in agent.stages])
    rhs = A.T @ y + cfg.fit_ridge * prior
    weights, cov = copy.deepcopy(agent)._field_fit()
    assert torch.equal(torch.cat(weights), torch.linalg.solve(normal, rhs)), "G5 is not _field_fit"
    out = []
    for k in range(len(agent.stages)):
        b = slice(k * m, (k + 1) * m)
        assert torch.count_nonzero(normal[b, : k * m]) == 0
        assert torch.count_nonzero(normal[b, (k + 1) * m :]) == 0
        out.append(dict(
            rows=owner.count(k),
            out=dict(targets=_hexes([t for t, o in zip(targets, owner) if o == k]),
                     normal=_hexes(normal[b, b].reshape(-1).tolist()), rhs=_hexes(rhs[b].tolist()),
                     weights=_hexes(weights[k].tolist()),
                     covariance=_hexes(cov[k].reshape(-1).tolist())),
        ))  # fmt: skip
    return dict(stages=out)


def damping_fit_case(agent, report) -> dict:
    """G6: the Newton iterates of ``Learner._fit_damping`` and the update's clipped proposal."""
    from grl_snam.tools.coph_golden import hexbits

    cfg = agent.cfg
    n = len(agent._transitions())
    iterates = []
    result = agent.damping
    if n >= 3:
        d = tensor(agent.damping).requires_grad_(True)
        for _ in range(cfg.solver_steps):
            loss = agent._plant_loss(d)
            g = torch.autograd.grad(loss, d, create_graph=True)[0]
            h = torch.autograd.grad(g, d)[0]
            step = (d - g / (h.abs() + 1e-5)).detach()
            nxt = step.clamp(0.05, 3.0)
            iterates.append(dict(damping=float(d.detach()), loss=float(loss.detach()),
                                 gradient=float(g.detach()), curvature=float(h), step=float(step),
                                 next=float(nxt)))  # fmt: skip
            d = nxt.requires_grad_(True)
        result = float(d.detach())
    assert result == copy.deepcopy(agent)._fit_damping(), "G6 is not _fit_damping"
    return dict(
        transitions=n,
        out=dict(iterates=[{k: hexbits(v) for k, v in it.items()} for it in iterates],
                 result=hexbits(result), proposal=hexbits(report["damping_proposal"])),
    )  # fmt: skip


def _replay(base, weights, damping, policy):
    """One ``retained_replay`` from a copy of the learner as ``prepare`` left it."""
    a = copy.deepcopy(base)
    start = a.counterfactual_steps
    p = tensor(policy).requires_grad_(True)
    cost, paths = stage_prediction.retained_replay(a, p, weights, damping)
    gradient = None
    if cost is not None and cost.requires_grad:
        gradient = torch.autograd.grad(cost, p, allow_unused=True)[0]
    return a, start, cost, paths, gradient


def _replay_decisions(a, start, cost, paths) -> dict:
    return dict(
        replayed=cost is not None,
        eligibility=[[e["stage"], e["eligible"], e["reason"]] for e in a.replay_eligibility],
        events=[[e["entry"], e["step"], e["from_stage"], e["to_stage"]]
                for e in a.replay_stage_events],
        controls=[[c["entry"], c["step"], c["reason"], c.get("route_reason"), c["guard_steps"],
                   c["projection_candidates"], [int(n) for n in c.get("route_nodes") or []]]
                  for c in a.replay_controls],
        lengths=[[sid, len(path)] for sid, path, _ in paths],
        steps=a.counterfactual_steps - start,
    )  # fmt: skip


def replay_case(holder, policy0, check=True) -> dict:
    """G7: ``retained_replay`` for 6 policies on the update's fitted weights and damping."""
    from grl_snam.tools.coph_golden import _hexes, hexbits

    base, weights, damping = holder["agent"], holder["weights"], holder["damping"]
    plan = [dict(stage=r["stage"], eligible=r["eligible"], horizon=r["horizon"], reason=r["reason"],
                 pilot_steps=r["pilot_steps"],
                 handoffs=[[h["step"], h["from_stage"], h["to_stage"]]
                           for h in r["pilot_handoffs"]])
            for r in base.replay_plan.values()]  # fmt: skip
    cases = []
    for policy in [tuple(policy0), *REPLAY_POLICIES]:
        a, start, cost, paths, gradient = _replay(base, weights, damping, policy)
        mine = _replay_decisions(a, start, cost, paths)
        if check:
            for sign in (1.0, -1.0):
                f = 1.0 + sign * NUDGE
                nudged = _replay(
                    base, [w * f for w in weights], damping * f, [p * f for p in policy]
                )
                if _replay_decisions(*nudged[:4]) != mine:
                    raise AssertionError(f"policy {policy}: a {NUDGE} nudge changes the replay")
        cases.append(dict(
            policy=_hexes(policy), **mine,
            out=dict(
                cost=None if cost is None else hexbits(float(cost.detach())),
                gradient=None if gradient is None else _hexes(gradient.tolist()),
                paths=[[_hexes(x) for x in path] for _, path, _ in paths],
                applied=[_hexes(c["applied"]) for c in a.replay_controls],
                waypoints=[_hexes(c.get("waypoint") or ()) for c in a.replay_controls],
                jumps=[hexbits(e["shaping_jump"]) for e in a.replay_stage_events],
            ),
        ))  # fmt: skip
    return dict(
        budget=dict(external_reserve=base.replay_external_reserve,
                    baseline_reserved=base.replay_baseline_reserved,
                    pilot_steps=base.replay_pilot_steps),
        plan=plan, policies=cases,
    )  # fmt: skip


def update_case(agent, report) -> dict:
    """G8: the update's decisions, and the numbers they compare."""
    from grl_snam.tools.coph_golden import _hexes, hexbits

    cfg = agent.cfg
    trace = report["policy_trace"]
    fv = (report["field_validation_before"], report["field_validation_after"])
    pv = (report["plant_validation_before"], report["plant_validation_after"])
    margins = ((fv[0] + cfg.validation_slack) - fv[1], (pv[0] + cfg.validation_slack) - pv[1])
    for m in margins:
        if abs(m) < SLACK_MARGIN:
            raise AssertionError(f"a validation change within {SLACK_MARGIN} of the slack ({m!r})")
    baseline = report["replay_before"]
    for a in trace["attempts"]:
        if a["cost"] is not None and abs((baseline + 1e-10) - a["cost"]) < 1e-13:
            raise AssertionError("a candidate's cost at the acceptance threshold")
    d = report["damping_proposal"]
    # The policy the feasibility check saw: the accepted candidate, else the committed one.
    chosen = [a["policy"] for a in trace["attempts"] if a["accepted"]]
    policy = np.asarray(chosen[0] if chosen else trace["initial"])

    def maybe(v):
        return None if v is None else (_hexes(v) if isinstance(v, (list, tuple)) else hexbits(v))

    return dict(
        field_valid=bool(margins[0] >= 0), plant_valid=bool(margins[1] >= 0),
        damping_feasible=bool(cfg.damping_bounds[0] <= d <= cfg.damping_bounds[1]),
        policy_in_box=bool((policy >= 0).all() and (policy <= cfg.policy_bounds[1]).all()),
        attempts=[[a["power"], a["reason"], a["accepted"]] for a in trace["attempts"]],
        exhausted=report["replay_budget_exhausted"],
        independent=report.get("policy_independent_replay"),
        accepted=report["accepted"], reason=report["reason"], version_after=report["version_after"],
        out=dict(
            field_validation=_hexes(fv), plant_validation=_hexes(pv), margins=_hexes(margins),
            replay_before=maybe(baseline), gradient=maybe(trace["gradient"]),
            unconstrained=maybe(trace["unconstrained_proposal"]),
            bounded=maybe(trace["bounded_proposal"]),
            candidates=[_hexes(a["policy"]) for a in trace["attempts"]],
            costs=[maybe(a["cost"]) for a in trace["attempts"]],
            policy_after=_hexes(report["policy"]), damping_after=hexbits(report["damping"]),
            shaping_jump=hexbits(report["shaping_jump"]),
        ),
    )  # fmt: skip


# Ticks whose learner the packet cases (coph_golden_packets) also keep: an export from the west
# end, before the agent reaches the bump.
EXTRA_TICKS = (24,)


def simulate(check=True):
    """The scripted run to the last snapshot: ``(cfg, script, found)``, where ``found[tick]`` is
    ``(learner before the update, report, holder)`` for each snapshot tick and ``EXTRA_TICKS``."""
    from grl_snam.tools.coph_golden import PROFILE

    cfg = Config(**{**PROFILE, "steps": 2500})
    ticks = tuple(t for _, t in SNAPSHOTS) + EXTRA_TICKS
    found = {}

    def keep(tick, before, report, holder):
        found[tick] = (before, report, holder)

    sc, _ = run(cfg, max(ticks) + 1, snapshot_ticks=ticks, check=check, on_snapshot=keep)
    return cfg, sc, found


def learner_cases(check=True, sim=None) -> dict:
    """G5-G8 at the snapshots, with the script and the world they need. ``sim`` is a
    ``simulate(check)`` result to reuse."""
    from grl_snam.tools.coph_golden import COMMUNICATION, _hexes, hexbits

    cfg, sc, found = sim if sim is not None else simulate(check)
    names = {t: n for n, t in SNAPSHOTS}
    snapshots, fits, dampings, replays, updates = [], [], [], [], []
    for tick, name in names.items():
        before, report, holder = found[tick]
        snapshots.append(snapshot(name, tick, before, cfg, sc))
        fits.append(field_fit_case(before))
        dampings.append(damping_fit_case(before, report))
        replays.append(replay_case(holder, before.policy.tolist(), check=check))
        updates.append(update_case(before, report))
    return dict(
        tier=TIER,
        steps=cfg.steps,
        world=dict(
            damping=hexbits(WORLD_DAMPING), goal=_hexes(GOAL), agent_id=AGENT_ID,
            nodes=[_hexes(n) for n in NODES], edges=[list(e) for e in EDGES],
            obstacles=[_hexes(o) for o in OBSTACLES], sensor_radius=hexbits(cfg.sensor_radius),
            lattice=[_hexes(o) for o in LATTICE],
            communication={k: hexbits(v) if isinstance(v, float) else v
                           for k, v in COMMUNICATION.items()},
        ),
        script=[dict(state=_hexes(t["state"]), u=_hexes(t["u"]), version=t["version"],
                     readings=[_hexes(r) for r in t["readings"]], visible=t["visible"])
                for t in sc],
        snapshots=snapshots,
        field_fit=fits,
        damping_fit=dampings,
        replay=replays,
        update=updates,
    )  # fmt: skip


def _bits(h):
    import struct

    if isinstance(h, list):
        return [_bits(x) for x in h]
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def _expand(runs) -> list[int]:
    return [i for start, count in runs for i in range(start, start + count)]


def rebuild(section: dict, snap: dict, agent_id: str = AGENT_ID):
    """A ``CrossingLearner`` in a snapshot's state, from the file's inputs and the rules alone.
    With another ``agent_id`` (G10's receiver), the readings' uids carry that id."""
    from grl_snam.coph.records import new_stage
    from grl_snam.tools.coph_golden import PROFILE

    cfg = Config(**{**PROFILE, "steps": section["steps"]})
    sc = section["script"]
    state = [tuple(_bits(t["state"])) for t in sc]
    agent = make_learner(cfg, dict(state=state[0]), agent_id)
    radius = _bits(section["world"]["sensor_radius"])
    network = (NODES, EDGES)
    stages = []
    for st in snap["stages"]:
        s = new_stage(st["id"], tuple(_bits(st["center"])), tuple(_bits(st["entry"])), GOAL, cfg)
        assert list(s.target) == _bits(st["target"])
        s.weights, s.version = tensor(_bits(st["weights"])), st["version"]
        n = len(LATTICE)
        s.readings = []
        for i in _expand(st["readings"]):
            tick, j = divmod(i, n)
            x, y, v = _bits(sc[tick]["readings"][j])
            s.readings.append(Reading(f"{agent_id}:{tick}:{j}", tick * cfg.dt, (x, y), v))
        s.transitions = [
            Transition(k, (k + 1) * cfg.dt, state[k], tuple(_bits(sc[k]["u"])), state[k + 1],
                       sc[k]["version"])
            for k in _expand(st["transitions"])
        ]  # fmt: skip
        s.footprints = [(*state[k][:2], radius, k * cfg.dt) for k in _expand(st["footprints"])]
        s.obstacles = tuple(OBSTACLES[k] for k in st["obstacles"])
        s.entry_navigation = dict(network=network, maze_graph=True)
        stages.append(s)
    agent.stages = stages
    agent.geometry_store = tuple(OBSTACLES[k] for k in snap["geometry"])
    agent.navigation = dict(
        network=network, maze_graph=True,
        reported_geometry=[OBSTACLES[k] for k in snap["reported_geometry"]],
        terrain_anchors=[tuple(_bits(a)) for a in snap["terrain_anchors"]],
    )  # fmt: skip
    # Nothing was received, so the terrain anchors are the learner's own (TerrainLearner).
    agent.local_model_anchors = list(agent.navigation["terrain_anchors"])
    agent.damping = _bits(snap["damping"])
    agent.policy = tensor(_bits(snap["policy"]))
    agent.version = snap["version"]
    agent.now = _bits(snap["now"])
    agent.current_state = tuple(_bits(snap["x"]))
    return agent


LEARNER_RULES = """\
G5-G8 pin CrossingLearner.update, the demos' learner (MazeLearner on TerrainLearner on
CooperativeLearner on Learner), under the profile (stages: the demos' rule, as in G4), at three
snapshots of a scripted crossing run. The C++ reads each snapshot's inputs and reproduces the four
outputs; grl_snam.tools.coph_golden_learner.rebuild is the reference for turning the inputs into a
learner.

The run (inputs; the same bytes on every OS):
- world: one agent, id '0', goal (12, 4), on the crossing's corridor network (nodes, edges) with
  its two doors (obstacles: slabs (x, y, r, hx, hy)); the world's damping 0.85.
- script[tick] = {state: (x, y, vx, vy) at the tick, u: the force held to the next tick, version:
  the learner's version after the tick, readings: 21 (x, y, value) from the demos' sensor lattice
  around state, visible: obstacle indices}. The state follows the exact flow at the world's
  damping; the force is a fixed tracker of a carrot moving east along the middle row to x = 8.6 and
  back. The readings are a rational risk bump at (5, 4.45).
- Each tick: the learner observes (tick 0 without a transition), advances, and updates when
  tick % update_every == 0. A reading's uid is '0:<tick>:<j>' (validation when j % 5 == 0) and its
  time tick * dt. Transition k is (uid k, time (k + 1) * dt, x = state[k], u = u[k], after =
  state[k + 1], version = version[k]). The footprint of tick k is (state[k].x, state[k].y,
  sensor_radius, k * dt).
- After each update the script rounds the committed field weights, damping and policy, and the
  value and gradient of each terrain anchor, to the nearest multiple of 2^-24.

A snapshot (snapshots[]) is the learner's state just before the update at its tick:
- version, damping, policy, now and x (the state passed to update);
- geometry: the geometry store, obstacle indices in the store's order; reported_geometry: the
  navigation context's reported shapes; terrain_anchors: its (x, y, value, gx, gy) anchors in order;
- stages[], oldest first: id, center, entry, target, version, the committed weights; readings and
  transitions and footprints as [start, count] runs of indices (a reading's index is 21 tick + j, a
  transition's its uid, a footprint's its tick), in the stage's own order; obstacles.
- Every stage's entry context, and the navigation context, hold the crossing's network
  (nodes, edges) and maze_graph = true. update adds terrain_anchors and reported_geometry to each
  entry context before it plans (MazeLearner.update, TerrainLearner.update). The demos never set
  Stage.anchors or a local residual, and nothing is received.

G5, field_fit[] (FLOAT64, tier): Learner._field_fit. For each kept stage k, the rows are its
readings with j % 5 != 0, in order: features phi_m(q) = exp(-0.5 |(q - c_m) / rbf_width|^2) over
the stage's 16 RBF centres (new_stage's), target log(expm1(4 max(value, 0.001))). The system is
block diagonal (no continuity rows in the profile): normal_k = A_k^T A_k + fit_ridge I, rhs_k =
A_k^T y_k + fit_ridge w_k (w_k the committed weights); weights_k = normal_k^-1 rhs_k (the Python
solves the stacked system by LU; any solver is within the tier), covariance_k = normal_k^-1.
out per stage: targets, normal (16 x 16, row-major), rhs, weights, covariance.

G6, damping_fit[] (FLOAT64): Learner._fit_damping, then update's reconciliation.
- The records are every kept stage's transitions with uid % 5 != 0 (oldest stage first); with
  fewer than 3, the result is the committed damping and there are no iterates.
- loss(d) = mean over records and the 4 components of (flow(x, u, d, dt) - after)^2
  + 0.05 mean over records of (delta - dt power)^2, with delta = (|v1|^2 - |v0|^2) / 2 and
  power = u . (v0 + v1) / 2 - d (|v0|^2 + |v1|^2) / 2, v0 = x's and v1 = after's velocity
  (flow as in G2).
- From d = the committed damping, solver_steps times: g = loss'(d), h = loss''(d),
  step = d - g / (|h| + 1e-5), next = clamp(step, 0.05, 3). result = the last next.
- proposal = clamp(result, max(damping_bounds[0], damping - damping_trust),
  min(damping_bounds[1], damping + damping_trust)).
out: iterates [{damping, loss, gradient, curvature, step, next}], result, proposal.

G7, replay[] (decisions BIT; values FLOAT64): stage_prediction.retained_replay on the fitted
weights (G5) and the proposal (G6), from the learner as replay_budget.prepare leaves it inside
update (with replay_external_reserve 12).
- plan: prepare's record per stage: eligible, horizon, reason, pilot_steps and the pilots'
  stage handoffs [step, from, to]. budget: the reserve, whether the baseline batch fits, and the
  pilots' steps.
- policies[]: the snapshot's policy, then (0, 0), (10, 10), (0.25, 6), (6, 0.25) and (3, 3).
  Each replays every kept stage from its entry, with a copy of its entry context, for the plan's
  horizon: per step the stage handoff (FrozenStages.advance in the 'spatial' stage mode), the
  routed control (navigation.guarded_control: maze_learning.maze_route on the network, then
  navigation.project with fast = true and the guard, which with maze_graph checks the corridor
  with maze_geometry.corridor_clear), the flow, the support check, then cost += dt (1 + 4 r(after)
  + 0.03 |u|^2) with r the active stage's risk (G3), 40 more on a swept collision, and a stop on
  a collision or within 0.25 of the goal. cost is the mean over the stages that replayed;
  gradient = d cost / d policy. A DomainExit ends the whole batch with no cost (replayed false).
- Decisions: replayed, eligibility [stage, eligible, reason], events [entry, step, from, to],
  controls [entry, step, reason, route_reason, guard_steps, projection_candidates, route_nodes],
  lengths [stage, states], steps (counterfactual steps, guard steps included).
  out: cost, gradient, paths (each replayed stage's states), applied (each control's force),
  waypoints, jumps (each event's shaping jump).
- The replay does not check the budget; update's Learner.replay does, before each batch.

G8, update[] (decisions BIT): Learner.update after G5-G7.
- field validation = mean over every kept stage's readings with j % 5 == 0 of
  (r(q) - value)^2, before with the committed weights and after with G5's; plant validation =
  G6's loss on the records with uid % 5 == 0, before at the committed damping and after at the
  proposal (0 with no records).
- The policy step: the baseline is the replay at the committed policy; the gradient there;
  unconstrained = policy - 0.2 gradient; bounded = clamp to [max(policy_bounds[0], policy -
  policy_trust), min(policy_bounds[1], policy + policy_trust)] per component. Attempt power
  0, 1, ... (up to solver_steps, while the budget holds a batch): candidate = policy + (bounded -
  policy) 0.5^power, accepted when it replays with cost <= baseline + 1e-10; the first accepted
  candidate becomes the policy.
- valid = field after <= field before + validation_slack and plant after <= plant before +
  validation_slack; feasible = proposal within damping_bounds and the policy within [0,
  policy_bounds[1]]; accepted = finite and feasible and valid, else reason
  'validation_or_feasibility'.
- On accept: version + 1, every kept stage takes G5's weights (and covariance) and the new version,
  the damping the proposal and the policy the step's. On reject nothing changes (rollback).
  shaping_jump = V(x) after - V(x) before on the newest stage (G3's progress potential, target the
  goal, the geometry store's obstacles).
- Decisions: field_valid, plant_valid, damping_feasible, policy_in_box, attempts [power, reason,
  accepted], exhausted, independent (the replay did not depend on the policy), accepted, reason,
  version_after. out: field_validation [before, after], plant_validation, margins (before + slack
  - after, each at least 1e-5 away from 0), replay_before, gradient, unconstrained, bounded,
  candidates, costs, policy_after, damping_after, shaping_jump.
"""
