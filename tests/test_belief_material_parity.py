"""Belief (observed) material against libcvc's golden (``tests/fixtures/belief_material_golden.json``).

The fixture is libcvc's ``NavBeliefMaterial.Golden`` (``src/cvc/tests/data``), written by the C++
kernels: four cases (the defaults, a 7 m material range, the range disc, pessimistic pins with
unseen cells open to the gate), three sense ticks, two belief planes. Each tick stores the
SHA-256 of every raster (the material mask, the planning occupancy, the belief inputs, the six
material planes, the gate surface and its clearance) and the witness gate's outputs. The numpy
path here must reproduce every value bit for bit: :meth:`BeliefGrid.track_material`,
:func:`seen_disc`, :func:`belief_material_runtime` and :func:`witness_gate_batch`.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from grl_snam import material as M
from grl_snam import nav_native
from grl_snam.belief import BeliefGrid
from grl_snam.belief_material import (
    BeliefMaterialParams,
    belief_gate_occ,
    belief_gate_surface,
    belief_material_inputs,
    belief_material_runtime,
    seen_disc,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "belief_material_golden.json").read_text()
)
R, C = GOLDEN["grid"]["rows"], GOLDEN["grid"]["cols"]
BOUNDS = tuple(GOLDEN["grid"]["bounds"])
TRUTH = np.array(GOLDEN["truth"], np.uint8).reshape(R, C).astype(bool)
RISK = np.array(GOLDEN["risk_raw"], np.float32).reshape(R, C)
HARD = np.array(GOLDEN["hard"]).reshape(R, C).astype(bool)
AMAP = [int(v) for v in GOLDEN["agent_map"]]
P = max(AMAP) + 1
SN, OC, MB, GP = GOLDEN["sense"], GOLDEN["occupancy"], GOLDEN["material_build"], GOLDEN["gate"]


def _h(a, dt):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a).astype(dt)).tobytes()).hexdigest()


def _same(golden, now):
    if isinstance(golden, str):
        return (golden == "nan" and math.isnan(now)) or (
            golden in ("inf", "-inf") and now == float(golden)
        )
    return golden == now


def _params(case):
    p = case["params"]
    return BeliefMaterialParams(
        unknown_risk=p["unknown_risk"],
        unknown_hard=bool(p["unknown_hard"]),
        unknown_gate_hard=bool(p["unknown_gate_hard"]),
        seen_rule=p["seen_rule"],
        material_range_m=p["material_range_m"],
    )


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["name"] for c in GOLDEN["cases"]])
def test_numpy_reproduces_the_golden(case):
    bp = _params(case)
    beliefs = [BeliefGrid((R, C), BOUNDS) for _ in range(P)]
    discs = [np.zeros((R, C), bool) for _ in range(P)]
    if bp.seen_rule == "raycast":
        for bg in beliefs:
            bg.track_material(bp.material_range_m)
    rng = float(SN["range_m"])
    cell_w = (BOUNDS[2] - BOUNDS[0]) / (C - 1)
    s = (C - 1) / (BOUNDS[2] - BOUNDS[0])
    for t, tk in enumerate(case["ticks"]):
        for i, (x, y, hd) in enumerate(GOLDEN["poses"][t]):
            beliefs[AMAP[i]].sense(
                TRUTH,
                (x, y),
                range_m=rng,
                n_rays=int(SN["n_rays"]),
                fov_rad=float(SN["fov_rad"]),
                heading_rad=hd,
            )
            if bp.seen_rule == "disc":
                rad = bp.material_range_m if bp.material_range_m > 0.0 else rng
                seen_disc(discs[AMAP[i]], (x, y), rad, BOUNDS)
        runtimes = []
        for m in range(P):
            seen = beliefs[m].material_seen if bp.seen_rule == "raycast" else discs[m]
            occ = beliefs[m].to_occupancy(
                unknown=OC["policy"], p_thresh=OC["p_thresh"], band=OC["band"]
            )
            risk_b, hard_b = belief_material_inputs(RISK, HARD, seen, bp)
            rt = belief_material_runtime(
                RISK, HARD, seen, occ, BOUNDS, (0.0, 0.0), MB["scale"], sigma=MB["sigma"], belief=bp
            )
            g = rt.grid
            # The explicit gate-surface twin agrees with the runtime's composition.
            gh, cl = belief_gate_surface(hard_b, occ, seen, cell_w, bp)
            assert np.array_equal(gh, rt._gate_hard)
            assert np.array_equal(cl, rt._gate_clear_m)
            got = {
                "material_seen": _h(seen, np.uint8),
                "occ": _h(occ, np.uint8),
                "risk_b": _h(risk_b, np.float32),
                "hard_b": _h(hard_b, np.uint8),
                "risk": _h(g.risk, np.float32),
                "phi_m": _h(g.phi_hard_m, np.float32),
                "grad_rx": _h(g.grad_rx, np.float32),
                "grad_ry": _h(g.grad_ry, np.float32),
                "grad_px": _h(g.grad_px, np.float32),
                "grad_py": _h(g.grad_py, np.float32),
                "gate_hard": _h(rt._gate_hard, np.uint8),
                "clear_m": _h(rt._gate_clear_m, np.float32),
            }
            want = tk["planes"][m]
            bad = [k for k, v in got.items() if want[k] != v]
            assert not bad, f"tick {t} plane {m}: {bad}"
            runtimes.append(rt)
        for i, (x, y, _hd) in enumerate(GOLDEN["poses"][t]):
            rt = runtimes[AMAP[i]]
            gx, gy = GOLDEN["goals"][i]
            act, nom, best, cnt = M.witness_gate_batch(
                rt.grid.risk,
                rt._gate_hard,
                rt._gate_clear_m,
                np.array([[(y - BOUNDS[1]) * s, (x - BOUNDS[0]) * s]]),
                np.array([[(gy - BOUNDS[1]) * s, (gx - BOUNDS[0]) * s]]),
                horizon_cells=GP["horizon_cells"],
                hard_margin_m=GP["hard_margin_m"],
                primitive_count=GP["primitive_count"],
                improvement_margin=GP["improvement_margin"],
                material_trigger=GP["material_trigger"],
                progress_slack_cells=GP["progress_slack_cells"],
            )
            gt = tk["gate"]
            assert int(act[0]) == gt["active"][i], (t, i)
            assert int(cnt[0]) == gt["count"][i], (t, i)
            assert _same(gt["nominal"][i], float(nom[0])), (t, i, gt["nominal"][i], nom[0])
            assert _same(gt["best"][i], float(best[0])), (t, i, gt["best"][i], best[0])


def test_the_full_arrays_match_their_hashes():
    full = GOLDEN["full"]
    planes = GOLDEN["cases"][0]["ticks"][full["tick"]]["planes"][full["plane"]]
    for key, dt in (
        ("material_seen", np.uint8),
        ("occ", np.uint8),
        ("risk_b", np.float32),
        ("hard_b", np.uint8),
        ("risk", np.float32),
        ("phi_m", np.float32),
        ("gate_hard", np.uint8),
        ("clear_m", np.float32),
    ):
        a = np.array(full[key], np.float64).astype(dt).reshape(R, C)
        assert _h(a, dt) == planes[key], key


def test_material_mask_at_the_sense_range_is_ever_seen():
    bg = BeliefGrid((R, C), BOUNDS)
    bg.track_material()
    for x, y, hd in GOLDEN["poses"][1]:
        bg.sense(TRUTH, (x, y), range_m=12.0, n_rays=90, heading_rad=hd)
        assert np.array_equal(bg.material_seen, bg.ever_seen)


def test_shorter_material_range_is_a_cast_at_that_range():
    for rm in (3.0, 5.5, 7.0):
        a = BeliefGrid((R, C), BOUNDS)
        a.track_material(rm)
        b = BeliefGrid((R, C), BOUNDS)
        for x, y, hd in GOLDEN["poses"][2]:
            a.sense(TRUTH, (x, y), range_m=12.0, n_rays=90, heading_rad=hd)
            b.sense(TRUTH, (x, y), range_m=rm, n_rays=90, heading_rad=hd)
        assert np.array_equal(a.material_seen, b.ever_seen), rm


def test_material_grew_flags_only_new_cells():
    bg = BeliefGrid((R, C), BOUNDS)
    bg.track_material()
    x, y, hd = GOLDEN["poses"][0][0]
    bg.sense(TRUTH, (x, y), range_m=12.0, n_rays=90, heading_rad=hd)
    assert bg.material_grew
    bg.sense(TRUTH, (x, y), range_m=12.0, n_rays=90, heading_rad=hd)
    assert not bg.material_grew
    bg.sense(TRUTH, (-50.0, -50.0), range_m=12.0)  # off the grid: senses nothing
    assert not bg.material_grew


def test_untracked_belief_is_unchanged():
    a, b = BeliefGrid((R, C), BOUNDS), BeliefGrid((R, C), BOUNDS)
    b.track_material(5.0)
    for x, y, hd in GOLDEN["poses"][0]:
        fa = a.sense(TRUTH, (x, y), range_m=12.0, n_rays=90, heading_rad=hd)
        fb = b.sense(TRUTH, (x, y), range_m=12.0, n_rays=90, heading_rad=hd)
        assert fa == fb
    assert a.material_seen is None
    for k in ("logodds", "last_visible", "ever_seen"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_disc_matches_a_full_scan_and_reports_growth():
    rnd = np.random.default_rng(3)
    bounds = (-3.5, 2.25, 41.0, 19.0)
    for _ in range(25):
        seen = np.zeros((23, 31), bool)
        px, py = rnd.uniform(-7.0, 45.0), rnd.uniform(-2.0, 23.0)
        rad = float(rnd.uniform(0.0, 15.0))
        grew = seen_disc(seen, (px, py), rad, bounds)
        want = np.zeros_like(seen)
        for r in range(23):
            for c in range(31):
                x = np.linspace(bounds[0], bounds[2], 31)[c]
                yv = np.linspace(bounds[1], bounds[3], 23)[r]
                want[r, c] = (x - px) ** 2 + (yv - py) ** 2 <= rad * rad
        assert np.array_equal(seen, want)
        assert grew == bool(want.any())
        assert not seen_disc(seen, (px, py), rad, bounds)


def test_inputs_and_gate_occ():
    seen = np.zeros((R, C), bool)
    seen[: R // 2] = True
    bp = BeliefMaterialParams(unknown_risk=0.25, unknown_hard=True)
    risk_b, hard_b = belief_material_inputs(RISK, HARD, seen, bp)
    assert risk_b.dtype == np.float32 and hard_b.dtype == np.bool_
    assert np.array_equal(risk_b[seen], RISK[seen]) and np.all(risk_b[~seen] == np.float32(0.25))
    assert np.array_equal(hard_b[seen], HARD[seen]) and np.all(hard_b[~seen])
    occ = np.zeros((R, C), bool)
    assert np.array_equal(belief_gate_occ(occ, seen, BeliefMaterialParams()), ~seen)
    assert not belief_gate_occ(occ, seen, BeliefMaterialParams(unknown_gate_hard=False)).any()


@pytest.mark.parametrize(
    "kw",
    [
        {"unknown_risk": 1.5},
        {"unknown_risk": float("nan")},
        {"seen_rule": "cone"},
        {"material_range_m": -1.0},
    ],
)
def test_params_are_validated(kw):
    with pytest.raises(ValueError):
        BeliefMaterialParams(**kw)


@pytest.mark.skipif(
    not nav_native.HAS_SENSE_BATCH_MATERIAL, reason="pycvc build lacks nav_sense_batch_material"
)
@pytest.mark.parametrize("rm", [0.0, 7.0])
def test_native_sense_mask_matches_numpy(rm):
    lo = np.zeros((P, R, C), np.float32)
    lv = np.zeros((P, R, C), bool)
    es = np.zeros((P, R, C), bool)
    ms = np.zeros((P, R, C), bool)
    ver = np.zeros(P, np.int32)
    beliefs = [BeliefGrid((R, C), BOUNDS) for _ in range(P)]
    for bg in beliefs:
        bg.track_material(rm)
    for t in range(len(GOLDEN["poses"])):
        pos = np.array([[q[0], q[1]] for q in GOLDEN["poses"][t]])
        hd = np.array([q[2] for q in GOLDEN["poses"][t]])
        flips, changed = nav_native.sense_batch(
            TRUTH,
            pos,
            hd,
            lo,
            lv,
            es,
            ver,
            np.array(AMAP, np.int32),
            range_m=12.0,
            n_rays=90,
            bounds=BOUNDS,
            material_seen=ms,
            material_range_m=rm,
            num_threads=1,
        )
        grew = [False] * P
        for i, (x, y, h) in enumerate(GOLDEN["poses"][t]):
            beliefs[AMAP[i]].sense(TRUTH, (x, y), range_m=12.0, n_rays=90, heading_rad=h)
            grew[AMAP[i]] |= beliefs[AMAP[i]].material_grew
        for m in range(P):
            assert np.array_equal(ms[m], beliefs[m].material_seen), (t, m)
            assert np.array_equal(es[m], beliefs[m].ever_seen), (t, m)
            assert bool(changed[m]) == grew[m], (t, m)
