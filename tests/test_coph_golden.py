"""libcvc's CoPH golden, regenerated and compared byte for byte.

``grl_snam.tools.coph_golden`` writes the golden libcvc commits in ``src/cvc/tests/data`` for its
C++ twin of ``grl_snam.coph`` (``NAV-COPH-ROADMAP.md`` §3.4). This test regenerates it and
compares bytes with the copy in ``tests/fixtures/coph_golden``. A change to the vendored records,
configuration or stage ledger fails here until the golden is regenerated (``--out
tests/fixtures/coph_golden``) and the new file is committed in libcvc too.

The G4 replay rebuilds every tick from the file's own inputs and ``stage_rules`` alone, in plain
Python, so the rules the C++ follows are known to be complete.
"""

import ast
import inspect
import json
import math
import struct
import textwrap
from pathlib import Path

import pytest

from grl_snam.coph.cooperative import CommunicationConfig
from grl_snam.coph.records import Config
from grl_snam.tools import coph_golden as golden

FIXTURE = Path(__file__).parent / "fixtures" / "coph_golden"


@pytest.fixture(scope="module")
def g():
    return json.loads((FIXTURE / golden.JSON_NAME).read_text())


def _f(h: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def test_the_golden_regenerates_byte_for_byte(tmp_path):
    path = golden.generate(tmp_path)
    assert path.read_bytes() == (FIXTURE / golden.JSON_NAME).read_bytes()


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
