"""libcvc's CoPH golden, regenerated and compared with the committed copy.

``grl_snam.tools.coph_golden`` writes the golden libcvc commits in ``src/cvc/tests/data`` for its
C++ twin of ``grl_snam.coph`` (``NAV-COPH-ROADMAP.md`` §3.4). This test regenerates it and
compares it with the copy in ``tests/fixtures/coph_golden``: G1, G4 and every input exactly, and
G2, G3 and G11's float64 outputs at their tier, since another platform's torch and libm may differ
in the last bits. A change to the vendored code fails here until the golden is regenerated
(``--out tests/fixtures/coph_golden``) and the new file is committed in libcvc too.

The replays rebuild every G4 tick, and every G2, G3 and G11 output, from the file's own inputs
and its rules texts alone, in plain Python, so the rules the C++ follows are known to be complete.
"""

import ast
import copy
import inspect
import json
import math
import struct
import textwrap
from pathlib import Path

import mpmath
import numpy as np
import pytest
import torch

from grl_snam.coph.cooperative import CommunicationConfig
from grl_snam.coph.records import Config
from grl_snam.tools import coph_golden as golden
from grl_snam.tools import coph_golden_learner as learner

FIXTURE = Path(__file__).parent / "fixtures" / "coph_golden"


@pytest.fixture(scope="module")
def g():
    return json.loads((FIXTURE / golden.JSON_NAME).read_text())


def _f(h: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


FLOAT64_SECTIONS = ("plant", "potential", "allocation")


def _close(mine: float, theirs: float, tier: dict) -> bool:
    return abs(mine - theirs) <= tier["rtol"] * abs(theirs) + tier["atol"]


def _flat(value) -> list[str]:
    return value if isinstance(value, list) else [value]


def _pop_outputs(g) -> dict:
    """Remove each FLOAT64 case's outputs from ``g``; return them by section."""
    return {key: [c.pop("out") for c in g[key]["cases"]] for key in FLOAT64_SECTIONS}


def _pop_learner_outputs(node, found=None) -> list:
    """Remove every ``out`` under ``node`` (G5-G8's values); return them in document order."""
    found = [] if found is None else found
    if isinstance(node, dict):
        if "out" in node:
            found.append(node.pop("out"))
        for v in node.values():
            _pop_learner_outputs(v, found)
    elif isinstance(node, list):
        for v in node:
            _pop_learner_outputs(v, found)
    return found


def _same_values(a, b, tier, where=()):
    """Two ``out`` trees: the same shape, every double within ``tier``."""
    if isinstance(b, dict):
        assert isinstance(a, dict) and a.keys() == b.keys(), where
        for k in b:
            _same_values(a[k], b[k], tier, (*where, k))
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _same_values(x, y, tier, (*where, i))
    elif b is None:
        assert a is None, where
    else:
        assert _close(_f(a), _f(b), tier), (where, _f(a), _f(b))


def _same_golden(mine, theirs):
    """G1, G4, the rules, every input and every decision exactly; the FLOAT64 outputs at their
    tier."""
    mine, theirs = copy.deepcopy(mine), copy.deepcopy(theirs)
    mine_out, theirs_out = _pop_outputs(mine), _pop_outputs(theirs)
    mine_learner, theirs_learner = (_pop_learner_outputs(x["learner"]) for x in (mine, theirs))
    assert mine == theirs
    for key in FLOAT64_SECTIONS:
        tier = theirs[key]["tier"]
        for case, a, b in zip(theirs[key]["cases"], mine_out[key], theirs_out[key], strict=True):
            assert a.keys() == b.keys(), case["name"]
            for name in b:
                for x, y in zip(_flat(a[name]), _flat(b[name]), strict=True):
                    assert _close(_f(x), _f(y), tier), (key, case["name"], name)
    assert len(mine_learner) == len(theirs_learner)
    for i, (a, b) in enumerate(zip(mine_learner, theirs_learner)):
        _same_values(a, b, theirs["learner"]["tier"], ("learner", i))


def test_the_golden_regenerates(tmp_path, g):
    # The decision checks only guard regeneration; the file is the same without them.
    path = golden.generate(tmp_path, check=False)
    if path.read_bytes() != (FIXTURE / golden.JSON_NAME).read_bytes():  # another platform
        _same_golden(json.loads(path.read_text()), g)


def test_another_platforms_last_bits_pass_and_nothing_else_does(g):
    def nudged(section, field, ulps):
        out = copy.deepcopy(g)
        o = out[section]["cases"][3]["out"]
        values = _flat(o[field])  # o[field] itself when it is a list
        values[0] = format(int(values[0], 16) + ulps, "016x")
        if not isinstance(o[field], list):
            o[field] = values[0]
        return out

    for section, field in (("plant", "work"), ("potential", "grad"), ("allocation", "accel")):
        _same_golden(nudged(section, field, 2), g)
        with pytest.raises(AssertionError):
            _same_golden(nudged(section, field, 1 << 48), g)
    changed = copy.deepcopy(g)
    changed["potential"]["cases"][3]["q"][0] = golden.hexbits(9.30000001)
    with pytest.raises(AssertionError):
        _same_golden(changed, g)


def test_doubles_are_raw_bits():
    assert golden.hexbits(0.12) == "3fbeb851eb851eb8"
    assert golden.hexbits(-0.0625) == "bfb0000000000000"
    assert _f(golden.hexbits(1e-300)) == 1e-300


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------


def _messages(cls) -> set[str]:
    """Every ``raise ValueError('...')`` message in ``cls.__post_init__``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__post_init__)))
    return {
        node.exc.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
    }


def test_every_refusal_is_covered(g):
    for key, cls in (("config", Config), ("communication", CommunicationConfig)):
        errors = {c["error"] for c in g["config"][key] if not c["accepted"]}
        assert errors == _messages(cls), key


def test_the_profile_is_accepted_and_inside(g):
    cases = {c["name"]: c for c in g["config"]["config"]}
    for name in ("profile_crossing", "profile_hazard", "small_caps", "integer_scales"):
        assert cases[name]["accepted"] and cases[name]["outside_profile"] == [], name
    assert cases["defaults"]["outside_profile"] == [
        "controller_mode", "shaping_mode", "replay_objective", "replay_domain",
        "replay_horizon_mode",
    ]  # fmt: skip
    assert cases["protected_controller"]["outside_profile"] == ["controller_mode"]
    # __post_init__ fills replay_update_cap, from the controller or from the horizon.
    for name in ("replay_cap_filled_by_controller", "replay_cap_filled_by_horizon"):
        assert cases[name]["values"]["replay_update_cap"] == 1024, name
    assert cases["defaults"]["values"]["replay_update_cap"] is None
    # The first failing check wins.
    assert cases["blend_before_controller"]["error"] == "invalid stage blend mode"
    assert cases["overlap_before_scales"]["error"].startswith("overlap in [0,1)")
    comm = {c["name"]: c for c in g["config"]["communication"]}
    assert comm["profile"]["outside_profile"] == []
    assert comm["compact_prediction"]["outside_profile"] == ["prediction_variant"]


def test_values_are_bits_by_declared_type(g):
    values = {c["name"]: c for c in g["config"]["config"]}["profile_crossing"]["values"]
    assert values["dt"] == golden.hexbits(0.12) and values["steps"] == 2500
    assert values["policy_initial"] == [golden.hexbits(1.0), golden.hexbits(2.0)]
    assert values["controller_mode"] == "routed"
    # An integer given for a float field is the double.
    integers = {c["name"]: c for c in g["config"]["config"]}["integer_scales"]["values"]
    assert integers["dt"] == golden.hexbits(1.0)


# ---------------------------------------------------------------------------
# G4: an independent replay of stage_rules
# ---------------------------------------------------------------------------


class _Stage:
    def __init__(self, sid, center, entry, goal, cfg, version):
        w, h = cfg["stage_width"], cfg["stage_height"]
        self.id, self.center, self.entry, self.version = sid, tuple(center), tuple(entry), version
        self.target = (min(goal[0], (center[0] + w / 2) - 0.1), goal[1])
        n = cfg["rbf_side"]

        def linspace(start, stop):
            step = (stop - start) / (n - 1)
            return [i * step + start for i in range(n - 1)] + [stop]

        xs = linspace(center[0] - w / 2, center[0] + w / 2)
        ys = linspace(center[1] - h / 2, center[1] + h / 2)
        self.centers = [c for x in xs for y in ys for c in (x, y)]
        self.weights = [-1.0 / (n * n)] * (n * n)
        self.readings, self.transitions, self.footprints, self.obstacles = [], [], [], []

    def contains(self, q, cfg):
        return (abs(q[0] - self.center[0]) <= cfg["stage_width"] / 2 + 1e-10
                and abs(q[1] - self.center[1]) <= cfg["stage_height"] / 2 + 1e-10)  # fmt: skip

    def intersects(self, xy, reach, cfg):
        halves = (cfg["stage_width"] / 2, cfg["stage_height"] / 2)
        d = [max(abs(xy[k] - self.center[k]) - halves[k], 0.0) for k in range(2)]
        return d[0] * d[0] + d[1] * d[1] <= (reach + 0.12 + cfg["safety_margin"]) ** 2 + 1e-12


def _config(variant):
    cfg = Config(**{**golden.PROFILE, **dict(golden.STAGE_VARIANTS)[variant["name"]]})
    return {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}


def _expand(runs):
    return [s + i for s, n in runs for i in range(n)]


def _replay(g, variant):
    st = g["stages"]
    cfg = _config(variant)
    goal = [_f(h) for h in st["goal"]]
    radius = _f(st["sensor_radius"])
    obstacles = [[_f(h) for h in o] for o in st["obstacles"]]
    reach = [o[2] + math.hypot(o[3], o[4]) if len(o) == 5 else o[2] for o in obstacles]
    script = st["script"]
    w, h, overlap = cfg["stage_width"], cfg["stage_height"], cfg["overlap"]
    caps = cfg["sample_cap"], cfg["transition_cap"], cfg["footprint_cap"]
    first = [_f(x) for x in script[0]["state"]]
    stages = [_Stage(0, first[:2], first, goal, cfg, 0)]
    created, ticks, positions = [stages[0]], [], {}
    for tick, t in enumerate(script):
        state = [_f(x) for x in t["state"]]
        for j, (x, y) in enumerate(t["readings"]):
            positions[5 * tick + j] = (_f(x), _f(y))
        # observe: footprints, readings, the transition
        for s in stages:
            if s.intersects(state[:2], radius, cfg):
                s.footprints = (s.footprints + [tick])[-caps[2] :]
        for s in stages:
            held = set(s.readings)
            for r in range(5 * tick, 5 * tick + len(t["readings"])):
                if r not in held and s.contains(positions[r], cfg):
                    s.readings.append(r)
            s.readings = s.readings[-caps[0] :]
        advanced = None
        if tick > 0:
            stages[-1].transitions = (stages[-1].transitions + [tick - 1])[-caps[1] :]
            # advance
            s = stages[-1]
            delta = [abs(state[k] - s.center[k]) for k in range(2)]
            size = (w, h)
            if any(delta[k] > (size[k] * (1 - overlap)) / 2 for k in range(2)):
                axis = 0 if delta[0] / w >= delta[1] / h else 1
                center = list(s.center)
                sign = (state[axis] > center[axis]) - (state[axis] < center[axis])
                center[axis] += sign * (size[axis] * (1 - overlap))
                n = _Stage(s.id + 1, center, state, goal, cfg, t["version"])
                pooled = []
                for old in stages:
                    pooled += [
                        r for r in old.readings if r not in pooled and n.contains(positions[r], cfg)
                    ]
                n.readings = pooled[-caps[0] :]
                feet = []
                for old in stages:
                    feet += [f for f in old.footprints if f not in feet]
                n.footprints = feet[-caps[2] :]
                stages = (stages + [n])[-(cfg["previous"] + 1) :]
                created.append(n)
                advanced = dict(id=n.id, axis=axis, from_id=s.id)
        # obstacles: the tick's store, filtered per stage
        geometry = variant["ticks"][tick]["geometry"]
        for s in stages:
            s.obstacles = [k for k in geometry if s.intersects(obstacles[k][:2], reach[k], cfg)]
        ticks.append((advanced, [(s.id, list(s.readings), list(s.transitions), list(s.footprints),
                                  list(s.obstacles)) for s in stages]))  # fmt: skip
    return created, ticks


@pytest.mark.parametrize("index", range(len(golden.STAGE_VARIANTS)))
def test_the_stage_rules_rebuild_every_tick(g, index):
    variant = g["stages"]["variants"][index]
    created, ticks = _replay(g, variant)
    assert len(created) == len(variant["created"])
    for mine, theirs in zip(created, variant["created"]):
        assert mine.id == theirs["id"] and mine.version == theirs["version"]
        for key in ("center", "entry", "target", "centers", "weights"):
            assert [golden.hexbits(v) for v in getattr(mine, key)] == theirs[key], (mine.id, key)
    held = {}
    for tick, ((advanced, stages), recorded) in enumerate(zip(ticks, variant["ticks"])):
        assert advanced == recorded["advanced"], tick
        now_held = {}
        for mine, (sid, *lists, obstacles) in zip(stages, recorded["stages"], strict=True):
            before = held.get(sid, ([], [], []))
            full = [b[d:] + _expand(added) for b, (d, added) in zip(before, lists)]
            assert mine == (sid, *full, obstacles), (tick, sid)
            now_held[sid] = full
        held = now_held


def test_g4_exercises_the_ledger(g):
    """Both axes advance in both directions, retention drops stages, and every cap fills."""
    variants = {v["name"]: v for v in g["stages"]["variants"]}
    for name, v in variants.items():
        cfg = _config(v)
        _, ticks = _replay(g, v)
        center = {c["id"]: [_f(x) for x in c["center"]] for c in v["created"]}
        moves = set()
        for t in v["ticks"]:
            if a := t["advanced"]:
                step = center[a["id"]][a["axis"]] - center[a["from_id"]][a["axis"]]
                moves.add((a["axis"], step > 0))
        assert moves == {(0, True), (0, False), (1, True), (1, False)}, name
        assert max(len(s) for _, s in ticks) == cfg["previous"] + 1, name
        longest = [max(len(s[k]) for _, ss in ticks for s in ss) for k in (1, 2, 3)]
        caps = [cfg["sample_cap"], cfg["transition_cap"], cfg["footprint_cap"]]
        if name == "small_caps_previous_3":
            assert longest == caps, name
        assert any(s[4] for _, ss in ticks for s in ss), name  # some stage holds an obstacle
    profile = variants["profile"]
    assert max(len(s[1]) for _, ss in _replay(g, profile)[1] for s in ss) == 240


# ---------------------------------------------------------------------------
# G2, G3, G11: independent replays of the rules texts, at the FLOAT64 tier
# ---------------------------------------------------------------------------


def _check(section, case, out: dict):
    tier = section["tier"]
    assert out.keys() == case["out"].keys(), case["name"]
    for name, value in out.items():
        for x, y in zip(_flat(value), _flat(case["out"][name]), strict=True):
            assert _close(x, _f(y), tier), (case["name"], name, x, _f(y))


def _vec(hexes) -> list[float]:
    return [_f(h) for h in hexes]


# Correctly rounded exp and expm1 (plant_rules): mpmath at 200 bits, rounded once to a double.
def _cr(fn, z: float) -> float:
    with mpmath.workprec(200):
        return float(fn(mpmath.mpf(z)))


def _plant(x, u, d, dt):
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    q, v = x[:2], x[2:]
    a = _cr(mpmath.exp, -d * dt)
    integral = -_cr(mpmath.expm1, -d * dt) / d
    after = [q[k] + integral * v[k] + ((dt - integral) * u[k]) / d for k in range(2)]
    after += [a * v[k] + integral * u[k] for k in range(2)]
    steady = [u[k] / d for k in range(2)]
    transient = [v[k] - steady[k] for k in range(2)]
    first = -_cr(mpmath.expm1, -d * dt) / d
    second = -_cr(mpmath.expm1, -2 * d * dt) / (2 * d)
    heat = d * (dot(steady, steady) * dt + 2 * dot(steady, transient) * first
                + dot(transient, transient) * second)  # fmt: skip
    work = dot(u, [after[k] - q[k] for k in range(2)])
    kinetic = (dot(after[2:], after[2:]) - dot(v, v)) / 2
    return dict(flow=after, work=work, heat=heat, kinetic_change=kinetic,
                residual=kinetic - work + heat)  # fmt: skip


def test_the_plant_rules_rebuild_every_case(g):
    section = g["plant"]
    for case in section["cases"]:
        x, u, d, dt = _vec(case["x"]), _vec(case["u"]), _f(case["d"]), _f(case["dt"])
        _check(section, case, _plant(x, u, d, dt))


def test_torch_rounds_g2s_exp_and_expm1_correctly(g):
    """plant_rules' premise at small d, where q_after, heat and residual are ill-conditioned."""
    for case in g["plant"]["cases"]:
        d, dt = _f(case["d"]), _f(case["dt"])
        for z in (-d * dt, -2 * d * dt):
            got = float(torch.expm1(torch.tensor(z, dtype=torch.float64)))
            assert got == _cr(mpmath.expm1, z), (case["name"], z)


def _softplus(z):
    return (z, 1.0) if z > 20 else (math.log1p(math.exp(z)), 1 / (1 + math.exp(-z)))


def _clearance(q, o):
    """The surface distance minus 0.12 and its gradient (potential_rules)."""
    if len(o) == 5:
        p = (o[0] - o[3], o[1] - o[4])
        d = (o[0] + o[3] - p[0], o[1] + o[4] - p[1])
        t = min(max(((q[0] - p[0]) * d[0] + (q[1] - p[1]) * d[1]) / (d[0] ** 2 + d[1] ** 2), 0), 1)
        z = (q[0] - (p[0] + t * d[0]), q[1] - (p[1] + t * d[1]))
    else:
        z = (q[0] - o[0], q[1] - o[1])
    n = math.hypot(*z)
    return n - o[2] - 0.12, (z[0] / n, z[1] / n)


def _potential(scene, case, top, rbf_width, shaping, limit):
    q, v, t, (pi0, pi1) = (_vec(case[k]) for k in ("q", "v", "target", "policy"))
    c = _vec(scene["centers"])
    w = _vec(scene["weights"])
    raw, draw = 0.0, [0.0, 0.0]
    for k, wk in enumerate(w):
        dx, dy = (q[0] - c[2 * k]) / rbf_width, (q[1] - c[2 * k + 1]) / rbf_width
        phi = math.exp(-0.5 * (dx * dx + dy * dy))
        raw += phi * wk
        draw = [draw[0] - wk * phi * dx / rbf_width, draw[1] - wk * phi * dy / rbf_width]
    sp, sig = _softplus(raw)
    risk, drisk = sp / 4, [sig / 4 * x for x in draw]
    barrier, dbarrier = 0.0, [0.0, 0.0]
    obstacles = [_vec(o) for o in scene["obstacles"]]
    for o in obstacles:
        clearance, dc = _clearance(q, o)
        sp, sig = _softplus(6 * (0.35 - clearance))
        barrier += sp * sp / 36
        dbarrier = [dbarrier[j] + 2 * sp * sig * -6 * dc[j] / 36 for j in range(2)]
    shape = pi0 * risk + pi1 * barrier
    dshape = [pi0 * drisk[j] + pi1 * dbarrier[j] for j in range(2)]
    r = [q[0] - t[0], q[1] - t[1]]
    s = r[0] * r[0] + r[1] * r[1]
    out = dict(risk=risk)
    if case["mode"] == "legacy":
        out["potential"] = 0.5 * s + shape
        grad = [r[j] + dshape[j] for j in range(2)]
    else:
        k, ell, eta = shaping["k"], shaping["ell"], shaping["eta"]
        risk_bound = math.exp(-0.5) * sum(abs(x) for x in w) / (4 * rbf_width)
        barrier_bound = sum(_softplus(6 * (o[2] + 0.47))[0] / 3 for o in obstacles)
        bound = top * (risk_bound + barrier_bound)
        alpha = 1.0 if bound == 0 else min(1.0, 2 * eta * k * ell / bound)
        taper = s / (s + ell * ell)
        out["potential"] = 0.5 * k * s + alpha * taper * shape
        dtaper = [2 * ell * ell * r[j] / (s + ell * ell) ** 2 for j in range(2)]
        grad = [k * r[j] + alpha * (dtaper[j] * shape + taper * dshape[j]) for j in range(2)]
    proposed = [-grad[j] - 1.4 * v[j] for j in range(2)]
    scale = min(1.0, limit / max(math.hypot(*proposed), 1e-12))
    out.update(grad=grad, proposed=proposed, applied=[p * scale for p in proposed])
    if case["mode"] == "progress":
        out.update(alpha=alpha, bound=bound)
    return out


def test_the_potential_rules_rebuild_every_case(g):
    section = g["potential"]
    shaping = {k: _f(h) for k, h in section["shaping"].items()}
    for case in section["cases"]:
        scene = section["scenes"][case["scene"]]
        out = _potential(scene, case, _f(scene["policy_bounds"][1]), _f(section["rbf_width"]),
                         shaping, _f(section["limit"]))  # fmt: skip
        _check(section, case, out)


def _allocate(case):
    u = _vec(case["u"])
    psi, v, length, steer_max, accel_max, floor = (
        _f(case[k]) for k in ("heading", "speed", "wheelbase", "steer_max", "accel_max",
                              "speed_floor")
    )  # fmt: skip
    h = (math.cos(psi), math.sin(psi))
    n = (-h[1], h[0])
    accel = min(max(u[0] * h[0] + u[1] * h[1], -accel_max), accel_max)
    kappa = (u[0] * n[0] + u[1] * n[1]) / max(v * v, floor * floor)
    steer = min(max(math.atan(length * kappa), -steer_max), steer_max)
    lateral = v * v * math.tan(steer) / length
    used = [accel * h[j] + lateral * n[j] for j in range(2)]
    return dict(accel=accel, steer=steer, kappa=kappa, used=used,
                unallocated=[u[j] - used[j] for j in range(2)])  # fmt: skip


def test_the_allocation_rules_rebuild_every_case(g):
    section = g["allocation"]
    for case in section["cases"]:
        _check(section, case, _allocate(case))


def test_the_float64_cases_cover_the_roadmap(g):
    """G2: 40 cases, d down to 1e-8 and the profile's bounds. G3: 50, every point under both
    modes, both sides of the force limit in each, alpha clipped and not, slabs and disks, the
    barrier active. G11: 60, at rest, in reverse, at the lock and inside it, the clip both ways."""
    plant = g["plant"]["cases"]
    dampings = {_f(c["d"]) for c in plant}
    assert len(plant) == 40 and min(dampings) == 1e-8
    assert {0.15, 1.5} <= dampings and {0.15, 1.5} == set(Config().damping_bounds)

    section = g["potential"]
    cases = section["cases"]
    assert len(cases) == 50
    clipped = {(c["mode"], math.hypot(*_vec(c["out"]["proposed"])) > _f(section["limit"]))
               for c in cases}  # fmt: skip
    assert clipped == {(m, b) for m in ("legacy", "progress") for b in (True, False)}
    alphas = {_f(c["out"]["alpha"]) == 1.0 for c in cases if c["mode"] == "progress"}
    assert alphas == {True, False}
    points = {(c["scene"], tuple(c["q"])) for c in cases}
    assert all({c["mode"] for c in cases if (c["scene"], tuple(c["q"])) == p}
               == {"legacy", "progress"} for p in points)  # fmt: skip
    kinds = {len(o) for s in section["scenes"] for o in s["obstacles"]}
    assert kinds == {3, 5}
    near = [c for c in cases for o in section["scenes"][c["scene"]]["obstacles"]
            if _clearance(_vec(c["q"]), _vec(o))[0] < 0.35]  # fmt: skip
    assert len(near) >= 10

    alloc = g["allocation"]["cases"]
    assert len(alloc) == 60
    speeds = [_f(c["speed"]) for c in alloc]
    assert speeds.count(0.0) >= 6 and sum(v < 0 for v in speeds) >= 6
    locked = [abs(_f(c["out"]["steer"])) == _f(c["steer_max"]) for c in alloc]
    assert any(locked) and not all(locked)
    accels = [_f(c["out"]["accel"]) / _f(c["accel_max"]) for c in alloc]
    assert {1.0, -1.0} <= set(accels) and any(abs(a) < 1 for a in accels)


# ---------------------------------------------------------------------------
# G5-G8: the learner at three snapshots of a scripted crossing run
# ---------------------------------------------------------------------------


def _readings(section, runs):
    """A stage's readings from the script: (tick, j, x, y, value), in the stage's order."""
    n = len(section["world"]["lattice"])
    out = []
    for i in learner._expand(runs):
        tick, j = divmod(i, n)
        x, y, v = _vec(section["script"][tick]["readings"][j])
        out.append((tick, j, x, y, v))
    return out


def _records(section, stages, validation):
    """Every kept stage's transitions (oldest first) with uid % 5 == 0 iff ``validation``."""
    sc = section["script"]
    return [
        (_vec(sc[k]["state"]), _vec(sc[k]["u"]), _vec(sc[k + 1]["state"]))
        for st in stages
        for k in learner._expand(st["transitions"])
        if (k % 5 == 0) == validation
    ]


def _centers(center, cfg):
    half_w, half_h = cfg.stage_width / 2, cfg.stage_height / 2
    xs = np.linspace(center[0] - half_w, center[0] + half_w, cfg.rbf_side)
    ys = np.linspace(center[1] - half_h, center[1] + half_h, cfg.rbf_side)
    return [(float(x), float(y)) for x in xs for y in ys]


def _phi(q, centers, width):
    return [
        math.exp(-0.5 * (((q[0] - cx) / width) ** 2 + ((q[1] - cy) / width) ** 2))
        for cx, cy in centers
    ]


def _risk(q, weights, centers, width):
    return _softplus(sum(w * p for w, p in zip(weights, _phi(q, centers, width))))[0] / 4


def _plant_loss(records, d, dt):
    """G6's loss and its first two derivatives in d, in closed form (learner_rules)."""
    a, integral = math.exp(-d * dt), -math.expm1(-d * dt) / d
    da, dda = -dt * a, dt * dt * a
    di = (dt * a - integral) / d
    ddi = (dt * da - 2 * di) / d
    se = g1 = h1 = ee = g2 = h2 = 0.0
    for x, u, y in records:
        for k in range(2):
            t = (dt - integral) * u[k] / d
            dt_ = -di * u[k] / d - (dt - integral) * u[k] / d**2
            ddt = -ddi * u[k] / d + 2 * di * u[k] / d**2 + 2 * (dt - integral) * u[k] / d**3
            position = (x[k] + integral * x[2 + k] + t, di * x[2 + k] + dt_,
                        ddi * x[2 + k] + ddt, y[k])  # fmt: skip
            velocity = (a * x[2 + k] + integral * u[k], da * x[2 + k] + di * u[k],
                        dda * x[2 + k] + ddi * u[k], y[2 + k])  # fmt: skip
            for p, dp, ddp, target in (position, velocity):
                r = p - target
                se, g1, h1 = se + r * r, g1 + 2 * r * dp, h1 + 2 * (dp * dp + r * ddp)
        v0, v1 = x[2:], y[2:]
        kin = (v0[0] ** 2 + v0[1] ** 2 + v1[0] ** 2 + v1[1] ** 2) / 2
        power = u[0] * (v0[0] + v1[0]) / 2 + u[1] * (v0[1] + v1[1]) / 2
        delta = 0.5 * ((v1[0] ** 2 + v1[1] ** 2) - (v0[0] ** 2 + v0[1] ** 2))
        e = delta - dt * (power - d * kin)
        ee, g2, h2 = ee + e * e, g2 + 2 * e * dt * kin, h2 + 2 * (dt * kin) ** 2
    n = len(records)
    if not n:
        return 0.0, 0.0, 0.0
    return se / (4 * n) + 0.05 * ee / n, g1 / (4 * n) + 0.05 * g2 / n, h1 / (4 * n) + 0.05 * h2 / n


def _learner_cfg(section):
    return Config(**{**golden.PROFILE, "steps": section["steps"]})


def _cases(g):
    s = g["learner"]
    return zip(
        s["snapshots"], s["field_fit"], s["damping_fit"], s["replay"], s["update"], strict=True
    )


def test_the_field_fit_rules_rebuild_g5(g):
    section, tier = g["learner"], g["learner"]["tier"]
    cfg = _learner_cfg(section)
    for snap, fit, *_ in _cases(g):
        for st, case in zip(snap["stages"], fit["stages"], strict=True):
            centers = _centers(_vec(st["center"]), cfg)
            rows = [
                (_phi((x, y), centers, cfg.rbf_width), math.log(math.expm1(4 * max(v, 0.001))))
                for _, j, x, y, v in _readings(section, st["readings"])
                if j % 5
            ]
            assert len(rows) == case["rows"]
            m = len(centers)
            normal = np.array(
                [[sum(r[0][i] * r[0][k] for r in rows) for k in range(m)] for i in range(m)]
            ) + cfg.fit_ridge * np.eye(m)
            rhs = np.array([sum(r[0][i] * r[1] for r in rows) for i in range(m)])
            rhs = rhs + cfg.fit_ridge * np.array(_vec(st["weights"]))
            out = dict(
                targets=[r[1] for r in rows],
                normal=normal.reshape(-1).tolist(),
                rhs=rhs.tolist(),
                weights=np.linalg.solve(normal, rhs).tolist(),
            )
            for name, mine in out.items():
                for x, y in zip(mine, _vec(case["out"][name]), strict=True):
                    assert _close(x, y, tier), (snap["name"], st["id"], name, x, y)
            # The inverse's small entries carry the matrix's conditioning: compare against its
            # largest entry.
            cov = np.linalg.inv(normal)
            scale = np.abs(cov).max()
            for x, y in zip(cov.reshape(-1), _vec(case["out"]["covariance"]), strict=True):
                assert abs(x - y) <= tier["rtol"] * scale, (snap["name"], st["id"], "covariance")


def test_the_damping_rules_rebuild_g6(g):
    section, tier = g["learner"], g["learner"]["tier"]
    cfg = _learner_cfg(section)
    for snap, _, case, *_ in _cases(g):
        records = _records(section, snap["stages"], validation=False)
        assert len(records) == case["transitions"] >= 3
        d = _f(snap["damping"])
        for it in case["out"]["iterates"]:
            loss, grad, curv = _plant_loss(records, d, cfg.dt)
            step = d - grad / (abs(curv) + 1e-5)
            nxt = min(max(step, 0.05), 3.0)
            mine = dict(damping=d, loss=loss, gradient=grad, curvature=curv, step=step, next=nxt)
            for name, x in mine.items():
                assert _close(x, _f(it[name]), tier), (snap["name"], name, x, _f(it[name]))
            d = _f(it["next"])
        assert len(case["out"]["iterates"]) == cfg.solver_steps
        assert d == _f(case["out"]["result"])
        committed = _f(snap["damping"])
        low = max(cfg.damping_bounds[0], committed - cfg.damping_trust)
        high = min(cfg.damping_bounds[1], committed + cfg.damping_trust)
        assert _f(case["out"]["proposal"]) == min(max(d, low), high)


def test_g7s_cost_rebuilds_from_its_trace(g):
    """cost = the mean over replayed stages of sum dt (1 + 4 r(after) + 0.03 |u|^2), with r the
    active stage's fitted field; at the committed policy the cost and gradient are G8's."""
    section, tier = g["learner"], g["learner"]["tier"]
    cfg = _learner_cfg(section)
    for snap, fit, _, replay, update in _cases(g):
        fields = {
            st["id"]: (_vec(f["out"]["weights"]), _centers(_vec(st["center"]), cfg))
            for st, f in zip(snap["stages"], fit["stages"])
        }
        for case in replay["policies"]:
            assert case["replayed"]
            controls = iter(zip(case["controls"], case["out"]["applied"], strict=True))
            costs = []
            for (sid, length), path in zip(case["lengths"], case["out"]["paths"], strict=True):
                horizon = next(p["horizon"] for p in replay["plan"] if p["stage"] == sid)
                assert length == len(path) == horizon + 1  # no collision, no stop at the goal
                active, cost = sid, 0.0
                for step in range(horizon):
                    (entry, k, *_), u = next(controls)
                    assert (entry, k) == (sid, step)
                    for e in case["events"]:
                        if e[:2] == [sid, step]:
                            active = e[3]
                    u, q = _vec(u), _vec(path[step + 1])[:2]
                    w, c = fields[active]
                    effort = 0.03 * (u[0] ** 2 + u[1] ** 2)
                    cost += cfg.dt * (1 + 4 * _risk(q, w, c, cfg.rbf_width) + effort)
                costs.append(cost)
            assert _close(sum(costs) / len(costs), _f(case["out"]["cost"]), tier), snap["name"]
        first = replay["policies"][0]
        assert first["policy"] == snap["policy"]
        assert _close(_f(first["out"]["cost"]), _f(update["out"]["replay_before"]), tier)
        mine, theirs = _vec(first["out"]["gradient"]), _vec(update["out"]["gradient"])
        assert all(_close(x, y, tier) for x, y in zip(mine, theirs, strict=True))


def test_the_update_rules_rebuild_g8(g):
    section, tier = g["learner"], g["learner"]["tier"]
    cfg = _learner_cfg(section)
    for snap, fit, damping, _, case in _cases(g):
        out = case["out"]

        def field_validation(weights):
            errors = [
                (_risk((x, y), w, _centers(_vec(st["center"]), cfg), cfg.rbf_width) - v) ** 2
                for st, w in zip(snap["stages"], weights)
                for _, j, x, y, v in _readings(section, st["readings"])
                if j % 5 == 0
            ]
            return sum(errors) / len(errors)

        before = field_validation([_vec(st["weights"]) for st in snap["stages"]])
        after = field_validation([_vec(f["out"]["weights"]) for f in fit["stages"]])
        records = _records(section, snap["stages"], validation=True)
        d0, d1 = _f(snap["damping"]), _f(damping["out"]["proposal"])
        plant = (_plant_loss(records, d0, cfg.dt)[0], _plant_loss(records, d1, cfg.dt)[0])
        recorded = _vec(out["field_validation"]) + _vec(out["plant_validation"])
        for x, y in zip((before, after, *plant), recorded, strict=True):
            assert _close(x, y, tier), (snap["name"], x, y)
        fv, pv = _vec(out["field_validation"]), _vec(out["plant_validation"])
        margins = [(fv[0] + cfg.validation_slack) - fv[1], (pv[0] + cfg.validation_slack) - pv[1]]
        assert margins == _vec(out["margins"])
        assert min(abs(m) for m in margins) >= learner.SLACK_MARGIN
        assert [case["field_valid"], case["plant_valid"]] == [m >= 0 for m in margins]

        policy, grad = _vec(snap["policy"]), _vec(out["gradient"])
        unconstrained = [p - 0.2 * gr for p, gr in zip(policy, grad)]
        low = [max(cfg.policy_bounds[0], p - cfg.policy_trust) for p in policy]
        high = [min(cfg.policy_bounds[1], p + cfg.policy_trust) for p in policy]
        bounded = [max(lo, min(hi, x)) for lo, hi, x in zip(low, high, unconstrained)]
        recorded = _vec(out["unconstrained"]) + _vec(out["bounded"])
        assert all(_close(x, y, tier) for x, y in zip(unconstrained + bounded, recorded))
        chosen = policy
        baseline = _f(out["replay_before"])
        attempts = zip(case["attempts"], out["candidates"], out["costs"], strict=True)
        for (power, reason, accepted), candidate, cost in attempts:
            mine = [p + (b - p) * 0.5**power for p, b in zip(policy, _vec(out["bounded"]))]
            assert mine == _vec(candidate)
            assert accepted == (_f(cost) <= baseline + 1e-10)
            assert reason == ("accepted" if accepted else "replay_cost")
            if accepted:
                chosen = mine
                break
        feasible = cfg.damping_bounds[0] <= d1 <= cfg.damping_bounds[1]
        in_box = all(0 <= p <= cfg.policy_bounds[1] for p in chosen)
        assert (case["damping_feasible"], case["policy_in_box"]) == (feasible, in_box)
        accepted = feasible and in_box and all(m >= 0 for m in margins)
        assert case["accepted"] == accepted
        assert case["reason"] == ("accepted" if accepted else "validation_or_feasibility")
        assert case["version_after"] == snap["version"] + accepted
        assert _vec(out["policy_after"]) == (chosen if accepted else policy)
        assert _f(out["damping_after"]) == (d1 if accepted else d0)


def test_a_snapshot_holds_everything_the_update_reads(g):
    """The learner rebuilt from each snapshot's inputs alone reproduces G5-G8."""
    section = g["learner"]
    for snap, fit, damping, replay, update in _cases(g):
        before = learner.rebuild(section, snap)
        report, holder = learner.traced_update(copy.deepcopy(before), learner._bits(snap["x"]))
        mine = dict(
            field_fit=learner.field_fit_case(before),
            damping_fit=learner.damping_fit_case(before, report),
            replay=learner.replay_case(holder, before.policy.tolist(), check=False),
            update=learner.update_case(before, report),
        )
        theirs = dict(field_fit=fit, damping_fit=damping, replay=replay, update=update)
        mine, theirs = json.loads(json.dumps(mine)), copy.deepcopy(theirs)
        a, b = _pop_learner_outputs(mine), _pop_learner_outputs(theirs)
        assert mine == theirs, snap["name"]
        for x, y in zip(a, b, strict=True):
            _same_values(x, y, section["tier"], (snap["name"],))


def test_g5_to_g8_cover_the_roadmap(g):
    """Three snapshots of three kept stages; six replayed policies each; an accepted and a
    rejected update, both well away from the slack; a stage handoff, an unsupported entry and a
    supported_boundary horizon among the replays; the door in the geometry."""
    section = g["learner"]
    snaps = section["snapshots"]
    assert len(snaps) == 3 and all(len(s["stages"]) == 3 for s in snaps)
    assert all(len(r["policies"]) == 6 for r in section["replay"])
    assert {u["reason"] for u in section["update"]} == {"accepted", "validation_or_feasibility"}
    policies = [p for r in section["replay"] for p in r["policies"]]
    assert {"supported", "unsupported_entry"} <= {e[2] for p in policies for e in p["eligibility"]}
    assert "supported_boundary" in {p["reason"] for r in section["replay"] for p in r["plan"]}
    assert any(p["events"] for p in policies)
    assert any(s["geometry"] for s in snaps)
    assert any(st["obstacles"] for s in snaps for st in s["stages"])
    assert max(abs(x) for p in policies for x in _vec(p["out"]["gradient"])) > 1e-4


def test_another_platforms_last_bits_pass_in_g5_to_g8_and_nothing_else_does(g):
    def changed(edit):
        out = copy.deepcopy(g)
        edit(out["learner"])
        return out

    def nudge(ulps):
        def edit(s):
            w = s["field_fit"][1]["stages"][0]["out"]["weights"]
            w[3] = format(int(w[3], 16) + ulps, "016x")

        return edit

    _same_golden(changed(nudge(2)), g)
    with pytest.raises(AssertionError):
        _same_golden(changed(nudge(1 << 40)), g)
    with pytest.raises(AssertionError):
        _same_golden(changed(lambda s: s["update"][1].update(accepted=True)), g)
