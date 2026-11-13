"""The golden for libcvc's C++ twin of CoPH-Stagewise (``grl_snam.coph``).

libcvc's ``docs/roadmap/NAV-COPH-ROADMAP.md`` (§3.4, §5) puts the reference here: this module
writes ``coph_golden.json``, which libcvc commits in ``src/cvc/tests/data`` for its ``coph*``
tests to reproduce. GRL-SNAM keeps a copy in ``tests/fixtures/coph_golden``.
``tests/test_coph_golden.py`` regenerates it and compares bytes, so a change to the vendored code
fails there until the golden is regenerated and re-committed on both sides.

Every double is written as its raw bits: a 64-bit unsigned integer as 16 lowercase hex digits,
most significant first (``"3fbeb851eb851eb8"`` is 0.12). The file is the same bytes on every run.
G1, G4 and every input are the same bytes on every OS too. G2, G3 and G11's outputs are the
platform's float64 torch and libm results, so another OS may differ in their last bits; the test
compares those at their tier (FLOAT64: rtol 1e-12, atol 1e-12).

The cases are the C1 set (the records and the stage ledger, the spec's G1 and G4) and the C2 set
(the plant, the potential and command, and the allocation: G2, G3 and G11).

- ``config`` (G1): ``Config`` and ``CommunicationConfig`` built from overrides. An accepted one
  records every field after ``__post_init__`` (which fills ``replay_update_cap``) and the fields
  outside the demos' profile, which the C++ refuses (decision D7). A refused one records
  ``__post_init__``'s message; every message is covered, and the first failing check wins.
- ``stages`` (G4): the stage ledger of the demos' own learner (``crossing_demo.CrossingLearner``,
  a ``maze_learning.MazeLearner``) along a scripted path, under three configurations. Each tick
  records the advance and each kept stage's readings, transitions, footprints and obstacles.
  ``stage_rules`` below states the rules in full.
- ``plant`` (G2): ``model.flow`` and ``model.balance`` over 40 ``(x, u, d, dt)``, with ``d``
  from 1e-8 to 10 and the profile's damping bounds.
- ``potential`` (G3): ``model.potential``, its gradient and ``model.command`` at 25 points on
  five scenes (the crossing's doors, doors with disks, a hazard bump, disks alone and a fresh
  stage), each under legacy and progress shaping, on both sides of the force limit and of
  progress's alpha clip.
- ``allocation`` (G11): ``grl_snam.coph.allocation.allocate`` over 60 cases, at rest, in reverse
  and at the steer lock among them.

The demos' learner is not ``learner.Learner``. Its ``advance`` moves along either axis, pools the
readings of every kept stage and never sets ``Stage.anchors``, so ``_field_fit``'s overlap
continuity rows are always empty in the profile. G4 pins that rule, not ``Learner.advance``.

The script's geometry keeps every comparison more than 1e-9 from its threshold
(``MARGIN``), so a one-ulp difference between platforms' dot products cannot change a decision.

Run ``python -m grl_snam.tools.coph_golden --out DIR``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import struct
from pathlib import Path

import numpy as np

from grl_snam.coph import UPSTREAM
from grl_snam.coph.cooperative import CommunicationConfig
from grl_snam.coph.crossing_demo import CrossingLearner, door
from grl_snam.coph.records import Config, Observation, Reading, Transition
from grl_snam.coph.shapes import is_slab, reach, slab_distance

JSON_NAME = "coph_golden.json"
MARGIN = 1e-9

# The demos' profile (crossing_demo.configuration, hazard_demo's): its modes, and the
# communication both demos run.
PROFILE = dict(
    controller_mode="routed",
    shaping_mode="progress",
    replay_objective="mission",
    replay_domain="retained",
    replay_horizon_mode="supported_boundary",
    replay_update_cap=1024,
)
PROFILE_MODES = dict(
    stage_blend_mode="legacy",
    route_field_mode="anchors",
    controller_mode="routed",
    shaping_mode="progress",
    navigation_target="mission",
    replay_objective="mission",
    replay_domain="retained",
    replay_horizon_mode="supported_boundary",
)
COMMUNICATION = dict(byte_budget=4000000, ttl=300.0)
COMMUNICATION_MODES = dict(prediction_variant="legacy")


def hexbits(x) -> str:
    """A double's raw bits, as 16 hex digits."""
    return format(struct.unpack("<Q", struct.pack("<d", float(x)))[0], "016x")


def _hexes(values) -> list[str]:
    return [hexbits(v) for v in values]


def _encode(kind: str, value):
    """A field value by its declared type: doubles as bits, tuples of doubles as lists."""
    if value is None or kind in ("int", "str"):
        return value
    if kind == "float":
        return hexbits(value)
    if kind == "pair":
        return _hexes(value)
    raise ValueError(kind)


def _kinds(cls) -> dict[str, str]:
    """Each field's encoding, from its declared type."""
    out = {}
    for f in dataclasses.fields(cls):
        if f.type is str:
            out[f.name] = "str"
        elif f.type is float:
            out[f.name] = "float"
        elif f.type is int or str(f.type) == "int | None":
            out[f.name] = "int"
        elif "Tuple" in str(f.type):
            out[f.name] = "pair"
        else:
            raise TypeError(f"{cls.__name__}.{f.name}: {f.type}")
    return out


# ---------------------------------------------------------------------------
# G1: the configuration
# ---------------------------------------------------------------------------

P = PROFILE  # overrides on top of the profile

CONFIG_CASES = [
    # The profile, as each demo builds it, and the defaults (outside the profile).
    ("profile_crossing", {**P, "steps": 2500}),
    ("profile_hazard", {**P, "steps": 1800}),
    ("defaults", {}),
    # Partial overrides inside the profile.
    ("overlap_half_previous_3", {**P, "overlap": 0.5, "previous": 3}),
    ("previous_0_no_overlap", {**P, "previous": 0, "overlap": 0.0}),
    ("stage_2p5_by_3p5", {**P, "stage_width": 2.5, "stage_height": 3.5}),
    (
        "small_caps",
        {**P, "sample_cap": 30, "transition_cap": 15, "anchor_cap": 4, "footprint_cap": 6},
    ),
    ("rbf_side_2_wide", {**P, "rbf_side": 2, "rbf_width": 1.1}),
    ("dt_0p1_sensor_1p5", {**P, "dt": 0.1, "sensor_radius": 1.5, "force_limit": 2.5}),
    ("integer_scales", {**P, "dt": 1, "stage_width": 3, "sensor_radius": 2}),
    ("policy_at_the_box", {**P, "policy_initial": (0.0, 10.0)}),
    ("policy_box_0_5", {**P, "policy_bounds": (0.0, 5.0), "policy_trust": 0.1}),
    (
        "damping_and_fit",
        {**P, "damping_initial": 0.9, "damping_bounds": (0.2, 1.2), "damping_trust": 0.05,
         "fit_ridge": 0.1, "validation_slack": 0.01},
    ),  # fmt: skip
    ("update_cadence", {**P, "update_every": 4, "replay_steps": 6, "solver_steps": 3}),
    ("replay_cap_1", {**P, "replay_update_cap": 1}),
    ("boundary_edges", {**P, "boundary_max_steps": 8, "boundary_follow_steps": 1}),
    ("navigation_limits", {**P, "safety_margin": 0.0, "obstacle_cap": 1, "route_expansion_cap": 1}),
    # replay_update_cap filled by __post_init__.
    ("replay_cap_filled_by_controller", {**P, "replay_update_cap": None}),
    (
        "replay_cap_filled_by_horizon",
        {"replay_horizon_mode": "supported_boundary", "replay_domain": "retained",
         "replay_objective": "mission"},
    ),  # fmt: skip
    # Accepted by __post_init__ but outside the profile.
    ("protected_controller", {**P, "controller_mode": "protected"}),
    ("geometry_controller", {**P, "controller_mode": "geometry", "shaping_mode": "legacy"}),
    ("legacy_shaping", {**P, "shaping_mode": "legacy"}),
    ("smooth_blend", {**P, "stage_blend_mode": "smooth"}),
    ("supported_field_routing", {**P, "route_field_mode": "supported_field"}),
    ("stage_target", {**P, "navigation_target": "stage"}),
    ("fixed_horizon", {**P, "replay_horizon_mode": "fixed"}),
    # Every refusal in Config.__post_init__, in its order.
    ("bad_blend", {"stage_blend_mode": "hard"}),
    ("smooth_needs_retained_routed", {"stage_blend_mode": "smooth"}),
    ("smooth_needs_overlap", {**P, "stage_blend_mode": "smooth", "overlap": 0.0}),
    ("bad_route_field", {"route_field_mode": "dense"}),
    ("field_routing_needs_routed", {"route_field_mode": "supported_field"}),
    ("bad_shaping", {"shaping_mode": "quadratic"}),
    ("progress_needs_protected_or_routed", {"shaping_mode": "progress"}),
    ("progress_fixed_k", {**P, "shaping_k": 2.0}),
    ("progress_fixed_ell", {**P, "shaping_ell": 0.25}),
    ("progress_fixed_eta", {**P, "shaping_eta": 0.75}),
    ("progress_negative_box", {**P, "policy_bounds": (-1.0, 10.0)}),
    ("progress_infinite_box", {**P, "policy_bounds": (0.0, math.inf)}),
    ("progress_policy_outside_box", {**P, "policy_initial": (1.0, 10.5)}),
    ("bad_controller", {"controller_mode": "reactive"}),
    ("controller_needs_retained_mission", {"controller_mode": "routed"}),
    ("zero_obstacle_cap", {**P, "obstacle_cap": 0}),
    ("zero_footprint_cap", {**P, "footprint_cap": 0}),
    ("zero_route_expansion_cap", {**P, "route_expansion_cap": 0}),
    ("negative_safety_margin", {**P, "safety_margin": -0.01}),
    ("bad_horizon", {**P, "replay_horizon_mode": "adaptive"}),
    ("boundary_needs_retained", {"replay_horizon_mode": "supported_boundary"}),
    ("zero_replay_cap", {**P, "replay_update_cap": 0}),
    ("boundary_shorter_than_replay", {**P, "boundary_max_steps": 7}),
    ("boundary_follow_zero", {**P, "boundary_follow_steps": 0}),
    ("bad_replay_domain", {"replay_domain": "global"}),
    ("retained_needs_mission", {"replay_domain": "retained"}),
    ("bad_replay_objective", {"replay_objective": "best"}),
    ("bad_navigation_target", {"navigation_target": "goal"}),
    ("overlap_one", {**P, "overlap": 1.0}),
    ("overlap_negative", {**P, "overlap": -0.1}),
    ("previous_one", {**P, "previous": 1}),
    ("zero_dt", {**P, "dt": 0.0}),
    ("negative_stage_width", {**P, "stage_width": -1.0}),
    ("zero_stage_height", {**P, "stage_height": 0.0}),
    ("zero_rbf_width", {**P, "rbf_width": 0.0}),
    ("zero_sensor_radius", {**P, "sensor_radius": 0.0}),
    ("zero_steps", {**P, "steps": 0}),
    ("zero_sample_cap", {**P, "sample_cap": 0}),
    ("zero_transition_cap", {**P, "transition_cap": 0}),
    ("zero_anchor_cap", {**P, "anchor_cap": 0}),
    ("zero_update_every", {**P, "update_every": 0}),
    ("zero_replay_steps", {**P, "replay_steps": 0}),
    ("zero_solver_steps", {**P, "solver_steps": 0}),
    ("rbf_side_one", {**P, "rbf_side": 1}),
    # More than one check fails: the first in __post_init__'s order wins.
    ("blend_before_controller", {"stage_blend_mode": "hard", "controller_mode": "reactive"}),
    ("overlap_before_scales", {**P, "overlap": 1.0, "dt": 0.0}),
    ("cap_before_domain", {"replay_update_cap": 0, "replay_domain": "global"}),
    ("limits_before_horizon", {**P, "obstacle_cap": 0, "replay_horizon_mode": "adaptive"}),
]

COMMUNICATION_CASES = [
    ("profile", dict(COMMUNICATION)),
    ("defaults", {}),
    ("partial", {**COMMUNICATION, "ensemble_size": 4, "anchors": 3, "received_cap": 2,
                 "covariance_floor": 0.01, "delay_ticks": 2, "temporal_window": 1.5}),  # fmt: skip
    ("compact_prediction", {**COMMUNICATION, "prediction_variant": "compact"}),
    ("bad_prediction", {**COMMUNICATION, "prediction_variant": "bogus"}),
]


def _config_case(cls, modes, name, overrides):
    kinds = _kinds(cls)
    case = {
        "name": name,
        "overrides": {k: _encode(kinds[k], v) for k, v in overrides.items()},
    }
    try:
        cfg = cls(**overrides)
    except ValueError as e:
        return {**case, "accepted": False, "error": str(e)}
    values = {f.name: _encode(kinds[f.name], getattr(cfg, f.name)) for f in dataclasses.fields(cls)}
    outside = [k for k, v in modes.items() if getattr(cfg, k) != v]
    return {**case, "accepted": True, "values": values, "outside_profile": outside}


def config_cases() -> dict:
    return {
        "profile": {k: _encode(_kinds(Config)[k], v) for k, v in PROFILE.items()},
        "profile_modes": PROFILE_MODES,
        "communication_profile": {
            k: _encode(_kinds(CommunicationConfig)[k], v) for k, v in COMMUNICATION.items()
        },
        "communication_profile_modes": COMMUNICATION_MODES,
        "config": [_config_case(Config, PROFILE_MODES, n, o) for n, o in CONFIG_CASES],
        "communication": [
            _config_case(CommunicationConfig, COMMUNICATION_MODES, n, o)
            for n, o in COMMUNICATION_CASES
        ],
    }


# ---------------------------------------------------------------------------
# G4: the stage ledger
# ---------------------------------------------------------------------------

AGENT_ID = "0"
GOAL = (12.0, 4.0)
SPEED_M_PER_TICK = 0.12
# Out along the middle row to the door, a wait there, back, down the west column, along the
# south row, up the east column, then a diagonal leg so the advance axis comes from delta / size.
WAYPOINTS = [(0.0, 4.0), (8.6, 4.0), (0.0, 4.0), (0.0, 0.0), (12.0, 0.0), (12.0, 4.0),
             (10.2, 6.6)]  # fmt: skip
PAUSES = {1: 20}  # waypoint index -> ticks held there at rest (the transition caps fill)
OBSTACLES = [
    door("middle"),
    door("north"),
    (3.0, 4.6, 0.2),
    (0.65, 1.45, 0.25),
    (6.0, -0.5, 0.3),
    (11.3, 5.3, 0.15),
]
# A reading at the sensor's centre and at its radius along each axis.
READING_OFFSETS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
VERSION_EVERY = 24  # the script's version is tick // VERSION_EVERY

STAGE_VARIANTS = [
    ("profile", {}),
    ("small_caps_previous_3", {"sample_cap": 30, "transition_cap": 15, "footprint_cap": 6,
                               "previous": 3, "overlap": 0.5}),  # fmt: skip
    ("previous_0_no_overlap", {"previous": 0, "overlap": 0.0, "stage_width": 2.5,
                               "stage_height": 3.5}),  # fmt: skip
]


def stage_script(sensor_radius: float) -> dict:
    """The scripted path and what is sensed along it, in plain Python floats.

    Positions step ``SPEED_M_PER_TICK`` along each leg (``a + (b - a) * (k / n)``), and the
    velocity is the leg's unit direction. Lengths use ``math.hypot``, not a BLAS dot, so the
    script is the same bits on every platform.
    """
    states = [(*WAYPOINTS[0], 0.0, 0.0)]
    for i, (a, b) in enumerate(zip(WAYPOINTS, WAYPOINTS[1:])):
        states += [(*a, 0.0, 0.0)] * PAUSES.get(i, 0)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        n = math.ceil(length / SPEED_M_PER_TICK)
        for k in range(1, n + 1):
            f = k / n
            states.append((a[0] + dx * f, a[1] + dy * f, dx / length, dy / length))
    ticks = []
    for tick, s in enumerate(states):
        q = np.array(s[:2])
        readings = [
            (f"{tick}:{j}", s[0] + ox * sensor_radius, s[1] + oy * sensor_radius,
             0.05 + 0.01 * j + 0.001 * (tick % 7))
            for j, (ox, oy) in enumerate(READING_OFFSETS)
        ]  # fmt: skip
        visible = []
        for k, o in enumerate(OBSTACLES):
            gap = float(slab_distance(q, o)) if is_slab(o) else math.hypot(*(q - o[:2])) - o[2]
            if abs(gap - sensor_radius) <= MARGIN:
                raise AssertionError(f"tick {tick}: obstacle {k} at the sensor's edge")
            if gap <= sensor_radius:
                visible.append(k)
        ticks.append(dict(state=s, version=tick // VERSION_EVERY, readings=readings,
                          visible=visible))  # fmt: skip
    return dict(ticks=ticks, sensor_radius=sensor_radius)


def _delta(prev: list[int], cur: list[int]) -> list:
    """``cur`` as ``prev`` with its first ``dropped`` items removed and ``added`` appended:
    ``[dropped, runs(added)]``. A stage only ever drops from the front and appends."""
    for k in range(len(prev) + 1):
        keep = prev[k:]
        if cur[: len(keep)] == keep:
            return [k, _runs(cur[len(keep) :])]
    raise AssertionError("unreachable")


def _runs(indices) -> list[list[int]]:
    """``[start, count]`` runs of consecutive indices, in order."""
    runs: list[list[int]] = []
    for i in indices:
        if runs and runs[-1][0] + runs[-1][1] == i:
            runs[-1][1] += 1
        else:
            runs.append([i, 1])
    return runs


def _margin(what, value):
    """Refuse a script whose decision sits within ``MARGIN`` of its threshold.

    Only the decisions that go through a BLAS dot product (``d @ d``, ``numpy.linalg.norm``) are
    checked: a platform's dot may differ in the last bit. The rest of the ledger's arithmetic is
    single IEEE operations, the same everywhere.
    """
    if abs(float(value)) <= MARGIN:
        raise AssertionError(f"{what}: within {MARGIN} of its threshold ({float(value)!r})")


def _intersects_margin(shape_xy, shape_reach, center, cfg):
    d = np.maximum(
        np.abs(np.asarray(shape_xy) - center) - [cfg.stage_width / 2, cfg.stage_height / 2], 0
    )
    return float(d @ d) - (shape_reach + 0.12 + cfg.safety_margin) ** 2


def _check_observe(agent, cfg, centers, q, sensor_radius):
    """The footprint and the obstacles against the stages ``observe`` saw, and the store's
    distance filter (``navigation.observe_geometry``)."""
    for c in centers:
        _margin("footprint_intersects", _intersects_margin(q, sensor_radius, c, cfg))
    v = np.asarray(agent.current_state[2:])
    stop = float(np.linalg.norm(v)) ** 2 / (2 * cfg.force_limit) + cfg.sensor_radius
    for o in OBSTACLES:
        _margin(
            "store_distance",
            float(np.linalg.norm(np.asarray(q) - o[:2])) - (stop + reach(o) + 0.17),
        )


def _check_stages(agent, cfg):
    for s in agent.stages:
        for o in OBSTACLES:
            _margin(
                "obstacle_intersects",
                _intersects_margin(o[:2], reach(o), np.asarray(s.center), cfg),
            )


def stage_variant(script, name, overrides) -> dict:
    cfg = Config(**{**PROFILE, "steps": len(script["ticks"]), **overrides})
    comm = CommunicationConfig(**COMMUNICATION)
    first = script["ticks"][0]
    agent = CrossingLearner(cfg, np.array(first["state"]), GOAL, agent_id=AGENT_ID,
                            communication=comm)  # fmt: skip
    footprint_index = {}
    reading_index = {}
    n_readings = 0
    created, ticks = [], []
    held: dict[int, tuple] = {}  # stage id -> its (readings, transitions, footprints) last tick

    def record_stage(s, tick):
        created.append(dict(
            id=s.id, tick=tick, center=_hexes(s.center), entry=_hexes(s.entry),
            target=_hexes(s.target), centers=_hexes(s.centers.numpy().reshape(-1)),
            weights=_hexes(s.weights.numpy()), version=s.version,
        ))  # fmt: skip

    record_stage(agent.stages[0], 0)
    sizes = np.array([cfg.stage_width, cfg.stage_height])
    for tick, t in enumerate(script["ticks"]):
        now = tick * cfg.dt
        q = t["state"][:2]
        readings = tuple(Reading(uid, now, (x, y), v) for uid, x, y, v in t["readings"])
        for uid, *_ in t["readings"]:
            reading_index[f"{AGENT_ID}:{uid}"] = n_readings
            n_readings += 1
        obs = Observation(now, readings, tuple(OBSTACLES[k] for k in t["visible"]), tuple(q),
                          script["sensor_radius"])  # fmt: skip
        footprint_index[(*q, script["sensor_radius"], now)] = tick
        centers = [np.asarray(s.center) for s in agent.stages]
        advanced = None
        if tick == 0:
            agent.observe(obs)
        else:
            prev = script["ticks"][tick - 1]
            tr = Transition(tick - 1, now, tuple(prev["state"]), (0.0, 0.0), tuple(t["state"]),
                            prev["version"])  # fmt: skip
            agent.version = t["version"]
            agent.observe(obs, tr)
            s = agent.stages[-1]
            delta = np.abs(np.asarray(t["state"][:2]) - s.center)
            before = s.id
            if agent.advance(np.array(t["state"])):
                n = agent.stages[-1]
                axis = int(np.argmax(delta / sizes))
                advanced = dict(id=n.id, axis=axis, from_id=before)
                record_stage(n, tick)
        _check_observe(agent, cfg, centers, q, script["sensor_radius"])
        _check_stages(agent, cfg)
        store = [OBSTACLES.index(o) for o in agent.geometry_store]
        stages, now_held = [], {}
        for s in agent.stages:
            lists = (
                [reading_index[r.uid] for r in s.readings],
                [tr.uid for tr in s.transitions],
                [footprint_index[tuple(f[:4])] for f in s.footprints],
            )
            before_lists = held.get(s.id, ([], [], []))
            stages.append([s.id, *(_delta(b, c) for b, c in zip(before_lists, lists)),
                           [OBSTACLES.index(o) for o in s.obstacles]])  # fmt: skip
            now_held[s.id] = lists
        held = now_held
        ticks.append(dict(tick=tick, advanced=advanced, geometry=store, stages=stages))
    kinds = _kinds(Config)
    return dict(
        name=name,
        overrides={k: _encode(kinds[k], v) for k, v in overrides.items()},
        created=created,
        ticks=ticks,
    )


def stage_cases() -> dict:
    cfg = Config(**PROFILE)
    script = stage_script(cfg.sensor_radius)
    ticks = [
        dict(state=_hexes(t["state"]), version=t["version"], visible=t["visible"],
             readings=[[hexbits(x), hexbits(y)] for _, x, y, _ in t["readings"]])
        for t in script["ticks"]
    ]  # fmt: skip
    return dict(
        agent_id=AGENT_ID,
        goal=_hexes(GOAL),
        sensor_radius=hexbits(script["sensor_radius"]),
        obstacles=[_hexes(o) for o in OBSTACLES],
        script=ticks,
        variants=[stage_variant(script, n, o) for n, o in STAGE_VARIANTS],
    )


STAGE_RULES = """\
The script: tick 0 is observed without a transition. Each later tick observes, then advances.
- Its observation: time = tick * dt; the readings at their script uids, which the learner stores
  as '<agent_id>:<uid>'; the visible obstacles; a footprint (state.x, state.y, sensor_radius,
  time).
- Its transition: Transition(uid = tick - 1, time, x = state[tick - 1], u = (0, 0),
  after = state[tick], version = version[tick - 1]); the learner's version is version[tick]
  before it observes.
observe(observation, transition), on each kept stage s, in order:
1. footprints: the footprint is appended to every s that intersects (x, y, sensor_radius), then
   each list keeps its last footprint_cap.
2. readings: s appends each reading whose uid s does not hold and that s.contains, in order, then
   keeps its last sample_cap.
3. transitions: the newest stage appends the transition, then keeps its last transition_cap.
4. obstacles: s.obstacles = the geometry store's obstacles (ticks[].geometry, in order) that
   intersect s. The store comes from the navigation layer; G4 records it per tick.
advance(x), from the newest stage s:
- delta = |x[:2] - s.center|, size = (stage_width, stage_height). No advance unless
  delta[k] > (size[k] * (1 - overlap)) / 2 for some k.
- axis = argmax(delta / size) (x on a tie); the new centre is s.center with
  center[axis] += sign(x[axis] - center[axis]) * (size[axis] * (1 - overlap)).
- The new stage: id s.id + 1; entry x; target (min(goal.x, (center.x + stage_width / 2) - 0.1),
  goal.y); RBF centres [(cx, cy) for cx in xs for cy in ys], xs = numpy.linspace(center.x -
  stage_width / 2, center.x + stage_width / 2, rbf_side) (i * step + start, the last exactly
  stop), ys likewise with stage_height; weights all -1 / rbf_side^2; version = the learner's.
- Its readings: every kept stage's readings, oldest stage first, that it contains; a repeated uid
  keeps its first place; then the last sample_cap. Its footprints: every kept stage's
  footprints, oldest first, repeats dropped, then the last footprint_cap. No transitions.
- The ring keeps the last previous + 1 stages, and the obstacles are recomputed as in observe.
The tests:
- contains(q): |q.x - center.x| <= stage_width / 2 + 1e-10 and
  |q.y - center.y| <= stage_height / 2 + 1e-10.
- intersects(shape, center): d = max(|shape.xy - center| - (stage_width / 2, stage_height / 2),
  0); d.d <= (reach + 0.12 + safety_margin)^2 + 1e-12. reach is r for a disk (x, y, r) and
  r + hypot(hx, hy) for a slab (x, y, r, hx, hy).
The layout:
- script[tick] = {state: [x, y, vx, vy], version, visible: obstacle indices, readings:
  [[x, y], ...]}. Reading j of a tick has uid '<tick>:<j>' (validation when j % 5 == 0) and
  value (0.05 + 0.01 * j) + 0.001 * (tick % 7), which no ledger rule reads.
- variants[].ticks[tick] = {advanced: null or {id, axis, from_id}, geometry: the store, stages:
  one [id, readings, transitions, footprints, obstacles] per kept stage, oldest first}.
- Readings, transitions and footprints are given against the same stage's lists at the end of
  the previous tick (empty for a new stage) as [dropped, added]: drop that many from the front,
  then append added, written as [start, count] runs of indices. A reading's index counts the
  script's readings in order (5 * tick + j), a transition's is its uid (tick - 1) and a
  footprint's is its tick. Obstacles are indices into obstacles, in the stage's order.
- variants[].created lists each stage's fixed fields when it is made (the first at tick 0).
"""


# ---------------------------------------------------------------------------
# G2, G3, G11: the FLOAT64 cases
# ---------------------------------------------------------------------------

# FLOAT64 (NAV-COPH-ROADMAP §5): |mine - golden| <= rtol * |golden| + atol on each output. The
# inputs are exact decimals; the outputs are the platform's float64 torch and libm results, which
# can differ from another platform's in the last bits.
FLOAT64 = dict(rtol=1e-12, atol=1e-12)


def _float64_case(name, inputs: dict, out: dict) -> dict:
    """Inputs and outputs as bits: a double as 16 hex digits, a sequence as a list of them."""

    def enc(v):
        if isinstance(v, (int, str)) and not isinstance(v, bool):
            return v
        if isinstance(v, (list, tuple)):
            return [enc(x) for x in v]
        return hexbits(v)

    return dict(name=name, **{k: enc(v) for k, v in inputs.items()},
                out={k: enc(v) for k, v in out.items()})  # fmt: skip


# G2: the plant. 10 dampings, each under 4 (x, u, dt): d down to 1e-8, the profile's bounds
# (0.15, 1.5), its initial 0.6 and the demos' world 0.85; dt the profile's 0.12, the Humvee
# step 0.6 (§3.3.1), 0.05 and 1.
DAMPINGS = [1e-8, 1e-6, 1e-4, 0.01, 0.15, 0.6, 0.85, 1.5, 3.0, 10.0]
PLANT_STATES = [(0.0, 4.0, 0.0, 0.0), (2.5, 4.0, 0.8, 0.1), (9.3, 4.2, -0.6, 0.45),
                (11.0, 7.5, 1.0, 0.0), (0.4, 0.3, -0.2, -0.9)]  # fmt: skip
PLANT_FORCES = [(0.0, 0.0), (3.0, 0.0), (-1.2, 2.1), (2.12, -2.12), (0.5, 0.25)]
PLANT_DTS = [0.12, 0.6, 0.05, 1.0]


def plant_cases() -> dict:
    from grl_snam.coph.model import balance, flow, tensor

    cases = []
    for i, d in enumerate(DAMPINGS):
        for j, dt in enumerate(PLANT_DTS):
            x = PLANT_STATES[(i + j) % len(PLANT_STATES)]
            u = PLANT_FORCES[(2 * i + j) % len(PLANT_FORCES)]
            after = flow(tensor(x), tensor(u), d, dt).tolist()
            b = balance(tensor(x), tensor(u), d, dt)
            cases.append(_float64_case(
                f"d={d!r} dt={dt!r}", dict(x=x, u=u, d=d, dt=dt),
                dict(flow=after, work=b["work"], heat=b["heat"],
                     kinetic_change=b["kinetic_change"], residual=b["residual"]),
            ))  # fmt: skip
    return dict(tier=FLOAT64, cases=cases)


PLANT_RULES = """\
model.flow and model.balance, for a state x = (q, v) (q, v in R^2), a force u held over dt and a
damping d > 0: the exact flow of q' = v, v' = u - d v.
- flow: a = exp(-d dt); I = -expm1(-d dt) / d; q_after = (q + I v) + ((dt - I) u) / d;
  v_after = a v + I u.
- balance: steady = u / d; transient = v - steady; first = -expm1(-d dt) / d;
  second = -expm1(-2 d dt) / (2 d);
  heat = d ((steady . steady) dt + 2 (steady . transient) first + (transient . transient) second);
  work = u . (q_after - q); kinetic_change = (v_after . v_after - v . v) / 2;
  residual = (kinetic_change - work) + heat, zero but for round-off.
- Small d is ill-conditioned: dt - I cancels to about d dt^2 / 2, and heat sums terms of size
  |u|^2 / d. So q_after, heat and residual hold to the tier only with exp and expm1 correctly
  rounded (torch's are at every argument here; the Windows CRT's expm1 is not at most of them)
  and the operations in the order written.
The layout: plant.cases[] = {name, x, u, d, dt, out: {flow: [x, y, vx, vy], work, heat,
kinetic_change, residual}}.
"""


# G3: the potential, its gradient and the command. Each scene is a stage (the profile's 3 x 4 m
# box, 4 x 4 RBF centres of width 0.8), its field weights and the obstacles in the geometry store;
# every point is evaluated under legacy shaping (no cfg: Learner.act, and replay's command) and
# progress shaping (the profile's routed controller, navigation.guarded_control).

# The crossing's doors: vertical slabs (x, y, r, hx, hy) that span the corridor.
DOOR_MIDDLE, DOOR_NORTH = door("middle"), door("north")


def _cyclic_weights(n, mul, mod, shift, scale):
    return [((k * mul) % mod - shift) / scale for k in range(n)]


def _bump_weights(centers, at, top, floor):
    """A hazard bump: top - ((cx - at.x)^2 + (cy - at.y)^2) / 2, held above floor."""
    return [max(top - ((cx - at[0]) ** 2 + (cy - at[1]) ** 2) / 2, floor) for cx, cy in centers]


# name, stage centre, obstacles, weights (a function of the centres), policy_bounds, points.
# A point is (q, v, target, policy).
POTENTIAL_SCENES = [
    ("crossing", (8.4, 4.0), [DOOR_MIDDLE, DOOR_NORTH],
     lambda c: _cyclic_weights(len(c), 7, 11, 5, 8.0), (0.0, 10.0), [
         ((9.1, 4.0), (0.8, 0.0), (12.0, 4.0), (1.0, 2.0)),
         ((9.2, 4.9), (0.3, 0.6), (12.0, 4.0), (3.5, 0.5)),
         ((7.2, 3.1), (0.0, 0.0), (8.6, 4.0), (1.0, 2.0)),
         ((8.0, 4.4), (-0.5, 0.2), (0.0, 4.0), (10.0, 10.0)),
         ((9.3, 3.5), (0.1, -0.05), (9.0, 3.6), (0.25, 6.0)),
         ((6.95, 5.6), (1.0, 0.05), (12.0, 4.0), (0.0, 0.0)),
         ((9.25, 4.2), (0.05, 0.0), (9.15, 4.1), (1.0, 2.0)),
     ]),
    ("crossing_with_disks", (2.1, 4.0), [DOOR_MIDDLE, (3.0, 4.6, 0.2), (1.4, 3.2, 0.25), DOOR_NORTH],
     lambda c: _cyclic_weights(len(c), 5, 9, 4, 6.0), (0.0, 10.0), [
         ((2.6, 4.35), (0.6, 0.3), (12.0, 4.0), (1.0, 2.0)),
         ((3.3, 4.1), (0.2, 0.0), (3.6, 4.0), (2.0, 4.0)),
         ((1.0, 3.6), (-0.4, -0.4), (0.0, 0.0), (1.0, 2.0)),
         ((2.1, 5.8), (0.0, 0.9), (2.0, 6.2), (0.5, 1.0)),
         ((0.8, 2.3), (0.0, 0.0), (0.7, 2.2), (6.0, 0.25)),
     ]),
    ("hazard", (6.3, 4.0), [],
     lambda c: _bump_weights(c, (7.0, 4.0), 1.25, -1.5), (0.0, 10.0), [
         ((7.0, 4.0), (0.9, 0.0), (12.0, 4.0), (1.0, 2.0)),
         ((5.2, 4.3), (0.7, -0.1), (12.0, 4.0), (6.0, 2.0)),
         ((6.6, 2.4), (0.3, 0.3), (7.5, 2.6), (0.25, 2.0)),
         ((7.6, 5.5), (0.0, 0.0), (7.7, 5.4), (10.0, 0.0)),
         ((4.9, 3.0), (-0.2, 0.6), (5.1, 3.4), (1.0, 2.0)),
         ((6.0, 5.95), (0.5, 0.5), (12.0, 4.0), (3.0, 3.0)),
     ]),
    # Disks only: the car adapter's obstacles (§3.3.3), and model.potential's disk branch.
    ("disks", (4.2, 4.0), [(3.0, 4.6, 0.2), (4.8, 3.3, 0.3), (5.5, 4.9, 0.15), (3.6, 2.6, 0.25)],
     lambda c: _cyclic_weights(len(c), 3, 8, 3.5, 6.0), (0.0, 10.0), [
         ((4.5, 3.75), (0.5, -0.3), (12.0, 4.0), (1.0, 2.0)),
         ((5.25, 4.75), (0.4, 0.2), (6.0, 5.0), (1.0, 2.0)),
         ((3.0, 4.15), (0.0, 0.0), (3.2, 4.0), (0.5, 0.75)),
         ((4.0, 4.0), (0.1, 0.0), (4.3, 4.05), (1.0, 2.0)),
         ((2.95, 3.0), (-0.3, 0.1), (12.0, 4.0), (2.5, 8.0)),
     ]),
    # A fresh stage (weights -1/16), no obstacles and the box (0, 2): alpha is 1, unclipped.
    ("fresh_alpha_1", (0.0, 4.0), [], None, (0.0, 2.0), [
        ((0.3, 4.1), (0.2, 0.0), (0.5, 4.0), (1.0, 2.0)),
        ((1.2, 3.5), (0.4, 0.3), (1.4, 4.0), (2.0, 0.5)),
    ]),
]  # fmt: skip
FORCE_LIMIT = 3.0  # Config().force_limit; the command's limit


def _slab_margin(q, o):
    """Refuse a point at a slab's core segment ends (where the nearest point's t is clamped) or
    on its core: model.potential's gradient is not smooth there."""
    p = (o[0] - o[3], o[1] - o[4])
    d = (o[0] + o[3] - p[0], o[1] + o[4] - p[1])
    t = ((q[0] - p[0]) * d[0] + (q[1] - p[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])
    tc = min(max(t, 0.0), 1.0)
    gap = math.hypot(q[0] - (p[0] + tc * d[0]), q[1] - (p[1] + tc * d[1]))
    if min(abs(t), abs(t - 1)) < 0.01 or gap < 0.01:
        raise AssertionError(f"{q}: at the slab {o}'s ends or core")


def potential_cases() -> dict:
    import torch

    from grl_snam.coph.model import command, potential, risk, shaping_parameters, tensor
    from grl_snam.coph.records import new_stage

    scenes, cases = [], []
    for si, (name, center, obstacles, weights_of, box, points) in enumerate(POTENTIAL_SCENES):
        cfg = Config(**{**PROFILE, "policy_bounds": box, "policy_initial": (1.0, 2.0)})
        stage = new_stage(0, center, (*center, 0.0, 0.0), GOAL, cfg)
        centers = [tuple(c) for c in stage.centers.tolist()]
        weights = stage.weights.tolist() if weights_of is None else weights_of(centers)
        w, c = tensor(weights), stage.centers
        scenes.append(dict(
            name=name, center=_hexes(center), centers=_hexes(stage.centers.numpy().reshape(-1)),
            weights=_hexes(weights), obstacles=[_hexes(o) for o in obstacles],
            policy_bounds=_hexes(box),
        ))  # fmt: skip
        for pi, (q, v, target, policy) in enumerate(points):
            for o in obstacles:
                if len(o) == 5:
                    _slab_margin(q, o)
                elif math.hypot(q[0] - o[0], q[1] - o[1]) < 0.01:
                    raise AssertionError(f"{q}: at the disk {o}'s centre")
            for mode in ("legacy", "progress"):
                mode_cfg = cfg if mode == "progress" else None
                qt = tensor(q).requires_grad_(True)
                args = (tensor(target), w, c, cfg.rbf_width, tensor(policy), obstacles)
                value = potential(qt, *args, mode_cfg)
                (grad,) = torch.autograd.grad(value, qt)
                applied, proposed = command(tensor((*q, *v)), *args, FORCE_LIMIT, mode_cfg)
                out = dict(risk=float(risk(tensor(q), w, c, cfg.rbf_width)),
                           potential=float(value.detach()), grad=grad.tolist(),
                           proposed=proposed.tolist(), applied=applied.tolist())  # fmt: skip
                if mode == "progress":
                    params = shaping_parameters(w, cfg.rbf_width, obstacles, cfg)
                    out.update(alpha=params["alpha"], bound=params["bound"])
                cases.append(_float64_case(
                    f"{name} {pi} {mode}", dict(scene=si, mode=mode, q=q, v=v, target=target,
                                                policy=policy), out,
                ))  # fmt: skip
    shaping = dict(k=Config().shaping_k, ell=Config().shaping_ell, eta=Config().shaping_eta)
    return dict(
        tier=FLOAT64,
        rbf_width=hexbits(Config().rbf_width),
        limit=hexbits(FORCE_LIMIT),
        shaping={k: hexbits(v) for k, v in shaping.items()},
        scenes=scenes,
        cases=cases,
    )


POTENTIAL_RULES = """\
model.potential, model.risk, model.shaping_parameters and model.command (no local residual: the
demos never set Stage.local_residual). A case evaluates at q, with velocity v, target t, policy
(pi0, pi1), its scene's RBF centres c_k, weights w_k and obstacles, and rbf_width.
- softplus(z) = log1p(exp(z)), and z itself for z > 20 (torch's threshold); its derivative is
  sigmoid(z), and 1 for z > 20.
- risk(q) = softplus(sum_k w_k exp(-0.5 |(q - c_k) / rbf_width|^2)) / 4.
- An obstacle's clearance is its surface distance minus 0.12 (the agent's radius). A disk
  (x, y, r): |q - (x, y)| - r. A slab (x, y, r, hx, hy), with ends p = (x - hx, y - hy) and
  e = (x + hx, y + hy), d = e - p: |q - (p + clamp(((q - p) . d) / (d . d), 0, 1) d)| - r.
- barrier(q) = sum over obstacles of softplus(6 (0.35 - clearance))^2 / 36 (0 with none).
- shaping(q) = pi0 risk(q) + pi1 barrier(q); s = |q - t|^2.
- legacy (mode 'legacy'): V = 0.5 s + shaping.
- progress (mode 'progress'): V = 0.5 k s + ((alpha s) / (s + ell^2)) shaping, with k, ell, eta
  from shaping and alpha = min(1, 2 eta k ell / bound) (1 when bound = 0), bound =
  policy_bounds[1] (exp(-0.5) sum_k |w_k| / (4 rbf_width) + sum over obstacles of
  softplus(6 (r + 0.47)) / 3). alpha and bound do not depend on q or the policy.
- grad = dV/dq. command: proposed = -grad - 1.4 v; applied = proposed min(1, limit /
  max(|proposed|, 1e-12)).
The layout: potential.scenes[] = {name, center, centers (x0, y0, x1, y1, ...; x outer, as G4's),
weights, obstacles, policy_bounds}; potential.cases[] = {name, scene (an index), mode, q, v,
target, policy, out: {risk, potential, grad, proposed, applied; progress adds alpha, bound}}.
Every point is given under both modes. No point sits at a slab's segment ends or core, or a
disk's centre, where V is not smooth.
"""


# G11: the allocation (grl_snam.coph.allocation, §3.3.2). 10 speeds x 6 forces: at rest, below,
# at and above the speed floor, and in reverse; headings around the circle; a RoboRacer, a
# Humvee and a slow RoboRacer; the steer lock and the acceleration clip on both sides.
ALLOCATION_SPEEDS = [0.0, 0.0, 0.2, 0.5, 1.0, 2.5, -0.25, -1.0, 4.0, 1.5]
ALLOCATION_FORCES = [(3.0, 0.0), (0.0, 3.0), (-2.0, 1.5), (0.4, -0.2), (2.1, 2.1), (0.0, 0.0)]
ALLOCATION_HEADINGS = [0.0, 1.5707963267948966, -2.3, 3.0, 0.7]
# (wheelbase, steer_max, accel_max)
ALLOCATION_CARS = [(0.33, 0.4189, 3.0), (3.3, 0.6, 2.5), (0.33, 0.35, 1.0)]
ALLOCATION_FLOORS = [0.5, 0.5, 0.25, 1.0]


def allocation_cases() -> dict:
    from grl_snam.coph.allocation import allocate

    cases = []
    for i, v in enumerate(ALLOCATION_SPEEDS):
        for j, u in enumerate(ALLOCATION_FORCES):
            heading = ALLOCATION_HEADINGS[(i + 2 * j) % len(ALLOCATION_HEADINGS)]
            wheelbase, steer_max, accel_max = ALLOCATION_CARS[(i + j) % len(ALLOCATION_CARS)]
            floor = ALLOCATION_FLOORS[(i * j) % len(ALLOCATION_FLOORS)]
            a = allocate(u, heading, v, wheelbase, steer_max, accel_max, floor)
            cases.append(_float64_case(
                f"v={v!r} u={u!r}",
                dict(u=u, heading=heading, speed=v, wheelbase=wheelbase, steer_max=steer_max,
                     accel_max=accel_max, speed_floor=floor),
                dict(accel=a.accel, steer=a.steer, kappa=a.kappa, used=a.used,
                     unallocated=a.unallocated),
            ))  # fmt: skip
    return dict(tier=FLOAT64, cases=cases)


ALLOCATION_RULES = """\
grl_snam.coph.allocation.allocate (NAV-COPH-ROADMAP §3.3.2), for a force u, a heading psi (rad),
a signed speed v along it, a wheelbase L, a steer lock steer_max, an acceleration limit
accel_max and a speed floor:
- h = (cos psi, sin psi); n = (-sin psi, cos psi), the left normal.
- accel = clamp(u . h, -accel_max, accel_max), with u . h = u.x h.x + u.y h.y.
- kappa = (u . n) / max(v v, speed_floor speed_floor).
- steer = clamp(atan(L kappa), -steer_max, steer_max): the bicycle angle, + left.
- lateral = ((v v) tan(steer)) / L; used = accel h + lateral n; unallocated = u - used.
The layout: allocation.cases[] = {name, u, heading, speed, wheelbase, steer_max, accel_max,
speed_floor, out: {accel, steer, kappa, used, unallocated}}.
"""


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def generate(out_dir) -> Path:
    """Write ``coph_golden.json`` into ``out_dir``; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    golden = {
        "about": (
            "libcvc's C++ twin of CoPH-Stagewise (NAV-COPH-ROADMAP.md §3.4, §5), written by "
            "GRL-SNAM's grl_snam.tools.coph_golden. Doubles are raw bits: 16 hex digits of the "
            "uint64, most significant first. Cases: G1 (config), G2 (plant), G3 (potential), "
            "G4 (stages) and G11 (allocation). G1 and G4 are BIT; G2, G3 and G11 are FLOAT64 "
            "(each section's tier)."
        ),
        "upstream": UPSTREAM,
        "stage_rules": STAGE_RULES,
        "plant_rules": PLANT_RULES,
        "potential_rules": POTENTIAL_RULES,
        "allocation_rules": ALLOCATION_RULES,
        "config": config_cases(),
        "plant": plant_cases(),
        "potential": potential_cases(),
        "stages": stage_cases(),
        "allocation": allocation_cases(),
    }
    path = out_dir / JSON_NAME
    # Bytes, not text: a text-mode write would end the file with \r\n on Windows.
    path.write_bytes((json.dumps(golden, separators=(",", ":")) + "\n").encode("utf-8"))
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    print(generate(args.out))


if __name__ == "__main__":
    main()
