"""The golden for libcvc's twin of the belief-material controller.

libcvc's ``docs/roadmap/NAV-BELIEF-MATERIAL-CONTROLLER-ROADMAP.md`` (§6) puts the reference here:
this module writes ``belief_material_controller_golden.json`` and
``belief_material_controller_golden.cvcnm``, which libcvc commits in ``src/cvc/tests/data`` and
its ``nav_belief_material_controller_test`` reproduces. GRL-SNAM keeps a copy in
``tests/fixtures/belief_material_controller_golden`` and its test regenerates both and compares
bytes, so a change to the controller fails here until the golden is regenerated and re-committed
on both sides.

The cases are the C1 set (the contract and the inputs; the spec's G1-G5):

- ``contract`` (G1): ``meta`` strings through :meth:`BeliefMaterialContract.from_meta`. An accepted
  one records the parsed values and ``to_meta()``; a refused one records only that it was refused.
- ``header_check`` (G2): the golden ``.cvcnm`` through :meth:`BeliefMaterialController.from_cvcnm`
  with belief and patch settings that each break one header field (one breaks two), plus two
  within the 1e-6 tolerance. Each records the fields the refusal names, in order.
- ``runtime_check`` (G3): ``material_config`` constants through
  :meth:`BeliefMaterialController.check_runtime`, recording the fields it names.
- ``material_config_for``: the gate horizon and hard margin ``MaterialRuntime`` derives from the
  contract's ``GateParams`` on a grid of each cell width.
- ``obstacles`` (G5): :func:`scan_visible_geometry` after each agent's sense tick on the belief
  golden's scene (and on a 9 x 9 scene of equal distances), with ``max_obstacles`` 36 and 5.
- ``tokens`` (G4): :func:`build_belief_material_controller_inputs` per agent and tick, on the
  belief planes of three of the belief golden's cases.

Every float32 is written as its raw bits (an unsigned 32-bit integer), so the C++ test compares
bits. The scene is libcvc's ``belief_material_golden.json`` (``--scene``; the copy in
``tests/fixtures`` by default). The net's weights come from numpy's PCG64 stream, not from
torch's initialisers, so the file does not depend on the torch version.

Run ``python -m grl_snam.tools.belief_material_controller_golden --out DIR``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from grl_snam.belief import BeliefGrid
from grl_snam.belief_material import BeliefMaterialParams, belief_material_runtime
from grl_snam.belief_material_controller import (
    BELIEF_MATERIAL_META_KEY,
    BeliefMaterialContract,
    BeliefMaterialController,
    build_belief_material_controller_inputs,
)
from grl_snam.material import GateParams, MaterialParams
from grl_snam.material_nav import CoefEnergyNetMaterial
from grl_snam.tools.austin_material_evaluation import scan_visible_geometry
from grl_snam.tools.matnet_export import BeliefHeader, write_matnet

JSON_NAME = "belief_material_controller_golden.json"
CVCNM_NAME = "belief_material_controller_golden.cvcnm"
DEFAULT_SCENE = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "belief_material_golden.json"
)

# The net: narrow, as golden 3 of libcvc's belief roadmap (spec §6).
NET = {"d_tok": 16, "nhead": 2, "num_layers": 1, "d_risk": 16, "patch_size": 9}
PATCH_EXTENT_M = 36.0
NET_SEED = 20261108

# The belief golden's cases whose planes the tokens are built on (all raycast).
TOKEN_CASES = ("defaults", "material_range_7", "pessimistic_ungated_unknown")
MAX_OBSTACLES = (36, 5)


def _bits(a) -> list[int]:
    return np.ascontiguousarray(np.asarray(a, np.float32)).reshape(-1).view(np.uint32).tolist()


def _u8(a) -> list[int]:
    return np.asarray(a).astype(np.uint8).reshape(-1).tolist()


def _fields(message: str, after: str) -> list[str]:
    """The ``name=`` fields a refusal names, in order."""
    return re.findall(r"(\w+)=", message.split(after, 1)[1])


# ---------------------------------------------------------------------------
# The net
# ---------------------------------------------------------------------------


def golden_net() -> CoefEnergyNetMaterial:
    """The golden's net, with weights from a fixed numpy stream.

    Each weight is uniform in ``[-1, 1) / sqrt(fan_in)``, LayerNorm scales are 1 and their biases
    0. The two lambda heads' last biases are 0, so both lambdas sit near the middle of their range
    and the gate's effect on ``lam_soft`` stays visible in C2's decisions.
    """
    model = CoefEnergyNetMaterial(risk_channels=3, **NET)
    rng = np.random.default_rng(NET_SEED)
    state = {}
    for name, tensor in model.state_dict().items():
        shape = tuple(tensor.shape)
        if ".norm" in name:
            value = np.ones(shape) if name.endswith("weight") else np.zeros(shape)
        else:
            fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
            value = (rng.random(shape) * 2.0 - 1.0) / math.sqrt(fan_in)
        state[name] = torch.from_numpy(np.asarray(value, np.float32))
    for head in ("lam_soft_head", "lam_hard_head"):
        state[f"{head}.2.bias"] = torch.zeros(1)
    model.load_state_dict(state)
    model.patch_size = NET["patch_size"]
    model.eval()
    return model


NET_HEADER = BeliefHeader(patch_extent_m=PATCH_EXTENT_M)


# ---------------------------------------------------------------------------
# G1: the contract
# ---------------------------------------------------------------------------


def _meta(record) -> str:
    return json.dumps({BELIEF_MATERIAL_META_KEY: record})


CONTRACT_METAS = [
    ("defaults", BeliefMaterialContract().to_meta()),
    ("empty_record", _meta({})),
    ("partial", _meta({"k_sharp": 4.5, "feature_scale_m": 12.5})),
    ("integers", _meta({"feature_scale_m": 20, "k_sharp": 5, "material_gate_trigger": 0})),
    (
        "short_reprs",
        _meta(
            {
                "feature_scale_m": 0.1,
                "k_sharp": 1e-05,
                "d_hat_sdf_m": 1e16,
                "material_gate_trigger": 1.0,
                "geometry_d_hat_m": 123456789.125,
            }
        ),
    ),
    (
        "repr_edges",
        _meta(
            {
                "feature_scale_m": 0.0001,
                "k_sharp": 9999999999999998.0,
                "d_hat_sdf_m": 1.7976931348623157e308,
                "material_gate_trigger": 5e-324,
                "geometry_d_hat_m": 0.30000000000000004,
            }
        ),
    ),
    ("exponent_text", '{"grl_snam.belief_material": {"k_sharp": 5E0, "d_hat_sdf_m": 3.0e+0}}'),
    (
        "other_keys",
        '{"run": {"seed": [1, 2.5, null, true, false], "name": "a\\u00e9\\"b\\n"},'
        ' "grl_snam.belief_material": {"feature_frame": "robot_relative"}, "x": NaN,'
        ' "y": -Infinity}',
    ),
    ("whitespace", ' \t\n{ "grl_snam.belief_material" : { } }\r\n '),
    ("duplicate_key", '{"grl_snam.belief_material": {"k_sharp": 1.0, "k_sharp": 2.0}}'),
    ("empty_meta", ""),
    ("not_json", "{grl_snam.belief_material: 1}"),
    ("trailing_text", BeliefMaterialContract().to_meta() + " x"),
    ("top_level_array", "[]"),
    ("no_key", '{"other": {}}'),
    ("key_not_an_object", _meta(5)),
    ("unknown_field", _meta({"feature_scale_m": 20.0, "extra": 1})),
    ("wrong_frame", _meta({"feature_frame": "absolute"})),
    ("frame_not_a_string", _meta({"feature_frame": 0})),
    ("zero_scale", _meta({"feature_scale_m": 0.0})),
    ("negative_scale", _meta({"feature_scale_m": -20.0})),
    ("infinite_scale", '{"grl_snam.belief_material": {"feature_scale_m": 1e999}}'),
    ("nan_k_sharp", '{"grl_snam.belief_material": {"k_sharp": NaN}}'),
    ("trigger_above_one", _meta({"material_gate_trigger": 1.5})),
    ("trigger_negative", _meta({"material_gate_trigger": -0.1})),
    ("string_number", _meta({"k_sharp": "5"})),
    ("boolean_number", _meta({"k_sharp": True})),
    ("null_number", _meta({"d_hat_sdf_m": None})),
]


def contract_cases() -> list[dict]:
    cases = []
    for name, meta in CONTRACT_METAS:
        case = {"name": name, "meta": meta}
        try:
            contract = BeliefMaterialContract.from_meta(meta)
        except ValueError:
            case["accepted"] = False
        else:
            case["accepted"] = True
            case["values"] = {
                "feature_frame": contract.feature_frame,
                "feature_scale_m": contract.feature_scale_m,
                "k_sharp": contract.k_sharp,
                "d_hat_sdf_m": contract.d_hat_sdf_m,
                "material_gate_trigger": contract.material_gate_trigger,
                "geometry_d_hat_m": contract.geometry_d_hat_m,
            }
            case["to_meta"] = contract.to_meta()
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# G2 / G3: the header and runtime checks; the derived gate constants
# ---------------------------------------------------------------------------

HEADER_CASES = [
    ("match", {}, {}),
    ("seen_rule", {"seen_rule": "disc"}, {}),
    ("unknown_risk", {"unknown_risk": 0.25}, {}),
    ("unknown_risk_in_tolerance", {"unknown_risk": 5e-7}, {}),
    ("unknown_risk_past_tolerance", {"unknown_risk": 2e-6}, {}),
    ("unknown_hard", {"unknown_hard": True}, {}),
    ("unknown_gate_hard", {"unknown_gate_hard": False}, {}),
    ("material_range_m", {"material_range_m": 7.0}, {}),
    ("patch_extent_m", {}, {"patch_extent_m": 40.0}),
    ("patch_extent_in_tolerance", {}, {"patch_extent_m": PATCH_EXTENT_M + 5e-7}),
    ("patch_size", {}, {"patch_size": 7}),
    ("two_fields", {"unknown_risk": 0.25, "material_range_m": 7.0}, {}),
]


def header_cases(cvcnm: Path) -> list[dict]:
    cases = []
    for name, belief, patch in HEADER_CASES:
        settings = {"patch_size": NET["patch_size"], "patch_extent_m": PATCH_EXTENT_M, **patch}
        bp = BeliefMaterialParams(**belief)
        case = {
            "name": name,
            "belief": {
                "unknown_risk": bp.unknown_risk,
                "unknown_hard": int(bp.unknown_hard),
                "unknown_gate_hard": int(bp.unknown_gate_hard),
                "seen_rule": bp.seen_rule,
                "material_range_m": bp.material_range_m,
            },
            **settings,
        }
        try:
            BeliefMaterialController.from_cvcnm(cvcnm, belief=bp, **settings)
        except ValueError as e:
            case["mismatches"] = _fields(str(e), "incompatible belief checkpoint:")
        else:
            case["mismatches"] = []
        cases.append(case)
    return cases


RUNTIME_CASES = [
    ("contract", 0.0, 0.0, 0.0),
    ("k_sharp", 1e-3, 0.0, 0.0),
    ("d_hat", 0.0, 1e-3, 0.0),
    ("trigger", 0.0, 0.0, 1e-3),
    ("in_tolerance", 4e-7, -4e-7, 5e-7),
    ("all_three", -1e-3, 1e-3, -1e-3),
]


def runtime_cases(controller: BeliefMaterialController) -> list[dict]:
    c = controller.contract
    cases = []
    for name, dk, dd, dt in RUNTIME_CASES:
        # libcvc's material_config holds k_sharp and d_hat_m as float32: check those values.
        k = float(np.float32(c.k_sharp + dk))
        d = float(np.float32(c.d_hat_sdf_m + dd))
        t = c.material_gate_trigger + dt
        runtime = SimpleNamespace(
            params=MaterialParams(
                lam_soft=1.5,
                lam_hard=1.0,
                k_sharp=k,
                d_hat_sdf_m=d,
                gate=GateParams(material_trigger=t),
            )
        )
        case = {"name": name, "k_sharp": k, "d_hat_m": d, "material_trigger": t}
        try:
            controller.check_runtime(runtime)
        except ValueError as e:
            case["mismatches"] = _fields(str(e), "contract:")
        else:
            case["mismatches"] = []
        cases.append(case)
    return cases


def material_config_cases() -> list[dict]:
    contract = BeliefMaterialContract()
    cases = []
    for cell_w in (0.3, 0.5, 1.0, 2.1, 10.0, 15.0, 50.0, 70.0):
        params = contract.material_params(lam_soft=1.5, lam_hard=1.0)
        ones = np.ones((6, 6), bool)  # the blur needs radius + 1 = 5 cells per axis
        runtime = belief_material_runtime(
            np.zeros((6, 6), np.float32),
            ~ones,
            ones,
            ~ones,
            (0.0, 0.0, 5.0 * cell_w, 5.0 * cell_w),
            (0.0, 0.0),
            1.0,
            sigma=1.0,
            params=params,
        )
        gate = runtime.grid.params.gate
        cases.append(
            {
                "cell_w": runtime.grid.cell_w,
                "horizon_cells": runtime._horizon_cells,
                "hard_margin_m": runtime._hard_margin_m,
                "k_sharp": params.k_sharp,
                "d_hat_sdf_m": params.d_hat_sdf_m,
                "material_trigger": gate.material_trigger,
                "primitive_count": gate.primitive_count,
                "improvement_margin": gate.improvement_margin,
                "progress_slack_cells": gate.progress_slack_cells,
                "sigma": runtime.grid.sigma,
            }
        )
    return cases


# ---------------------------------------------------------------------------
# G4 / G5: the obstacles and tokens on the belief golden's scene
# ---------------------------------------------------------------------------


def _obstacles(belief, truth, position, n_rays, range_m, max_obstacles) -> dict:
    centers, radii, weights, _ = scan_visible_geometry(
        belief, truth, position, n_rays, range_m, max_obstacles
    )
    return {
        "max_obstacles": max_obstacles,
        "centers": _bits(centers),
        "radii": _bits(radii),
        "weights": _bits(weights),
    }


def _belief_params(case) -> BeliefMaterialParams:
    p = case["params"]
    return BeliefMaterialParams(
        unknown_risk=p["unknown_risk"],
        unknown_hard=bool(p["unknown_hard"]),
        unknown_gate_hard=bool(p["unknown_gate_hard"]),
        seen_rule=p["seen_rule"],
        material_range_m=p["material_range_m"],
    )


def scene_cases(scene: dict) -> tuple[list[dict], list[dict], list[dict]]:
    R, C = scene["grid"]["rows"], scene["grid"]["cols"]
    bounds = tuple(scene["grid"]["bounds"])
    truth = np.array(scene["truth"], np.uint8).reshape(R, C).astype(bool)
    risk = np.array(scene["risk_raw"], np.float32).reshape(R, C)
    hard = np.array(scene["hard"]).reshape(R, C).astype(bool)
    amap = [int(v) for v in scene["agent_map"]]
    sense, occ_p, build = scene["sense"], scene["occupancy"], scene["material_build"]
    range_m, n_rays = float(sense["range_m"]), int(sense["n_rays"])
    by_name = {c["name"]: c for c in scene["cases"]}

    obstacles, planes, tokens = [], [], []
    for case_name in TOKEN_CASES:
        bp = _belief_params(by_name[case_name])
        beliefs = [BeliefGrid((R, C), bounds) for _ in range(max(amap) + 1)]
        for bg in beliefs:
            bg.track_material(bp.material_range_m)
        for t, poses in enumerate(scene["poses"]):
            visible = {}
            for i, (x, y, heading) in enumerate(poses):
                bg = beliefs[amap[i]]
                bg.sense(
                    truth,
                    (x, y),
                    range_m=range_m,
                    n_rays=n_rays,
                    fov_rad=float(sense["fov_rad"]),
                    heading_rad=heading,
                )
                position = np.array([x, y], np.float64)
                found = [
                    _obstacles(bg, truth, position, n_rays, range_m, mo) for mo in MAX_OBSTACLES
                ]
                visible[i] = found[0]
                entry = {
                    "tick": t,
                    "agent": i,
                    "position": [x, y],
                    "last_visible": _u8(bg.last_visible),
                    "cases": found,
                }
                if case_name == TOKEN_CASES[0]:
                    obstacles.append(entry)
                else:  # the material range and pins do not change what the rays touch
                    want = obstacles[t * len(poses) + i]
                    assert want["last_visible"] == entry["last_visible"]
                    assert want["cases"] == entry["cases"]

            runtimes = []
            for m, bg in enumerate(beliefs):
                seen = bg.material_seen
                occ = bg.to_occupancy(
                    unknown=occ_p["policy"], p_thresh=occ_p["p_thresh"], band=occ_p["band"]
                )
                rt = belief_material_runtime(
                    risk,
                    hard,
                    seen,
                    occ,
                    bounds,
                    (0.0, 0.0),
                    build["scale"],
                    sigma=build["sigma"],
                    belief=bp,
                )
                runtimes.append((rt, seen))
                planes.append(
                    {
                        "case": case_name,
                        "tick": t,
                        "plane": m,
                        "risk": _bits(rt.grid.risk),
                        "hard_b": _u8(rt.grid.hard),
                        "material_seen": _u8(seen),
                    }
                )

            for i, (x, y, _heading) in enumerate(poses):
                rt, seen = runtimes[amap[i]]
                goal = scene["goals"][i]
                variants = [("main", 20.0, visible[i])]
                if case_name == TOKEN_CASES[0] and t == 1:
                    variants.append(("scale_7_5", 7.5, visible[i]))
                if case_name == TOKEN_CASES[0] and t == 2 and i == 0:
                    variants.append(("no_obstacles", 20.0, None))
                for variant, scale, obs in variants:
                    if obs is None:
                        centers = np.empty((0, 2), np.float32)
                        radii = weights = np.empty((0,), np.float32)
                    else:
                        centers = np.array(obs["centers"], np.uint32).view(np.float32)
                        centers = centers.reshape(-1, 2)
                        radii = np.array(obs["radii"], np.uint32).view(np.float32)
                        weights = np.array(obs["weights"], np.uint32).view(np.float32)
                    obs_feats, obs_mask, goal_feats, patch = (
                        build_belief_material_controller_inputs(
                            np.array([x, y], np.float64),
                            np.array(goal, np.float64),
                            centers,
                            radii,
                            weights,
                            rt,
                            seen,
                            patch_extent_m=PATCH_EXTENT_M,
                            patch_size=NET["patch_size"],
                            feature_scale_m=scale,
                            belief=bp,
                        )
                    )
                    assert obs_mask.all() and len(obs_mask) == len(centers)
                    tokens.append(
                        {
                            "case": case_name,
                            "tick": t,
                            "agent": i,
                            "plane": amap[i],
                            "variant": variant,
                            "position": [x, y],
                            "goal": list(goal),
                            "feature_scale_m": scale,
                            "n_obstacles": int(len(centers)),
                            "obs_feats": _bits(obs_feats),
                            "goal_feats": _bits(goal_feats),
                            "risk_patch": _bits(patch),
                        }
                    )
    return obstacles, planes, tokens


def tie_cases() -> list[dict]:
    """A 9 x 9 unit grid, every cell visible and occupied, the agent on a cell centre: rings of
    equal distances, cut by max_obstacles (the stable row-major tie order, decision E5)."""
    shape, bounds = (9, 9), (0.0, 0.0, 8.0, 8.0)
    belief = BeliefGrid(shape, bounds)
    belief.last_visible[:] = True
    truth = np.ones(shape, bool)
    cases = []
    for position, max_obstacles in (((4.0, 4.0), 7), ((4.0, 4.0), 36), ((2.0, 6.0), 12)):
        p = np.array(position, np.float64)
        cases.append(
            {
                "rows": shape[0],
                "cols": shape[1],
                "bounds": list(bounds),
                "position": list(position),
                **_obstacles(belief, truth, p, 8, 10.0, max_obstacles),
            }
        )
    return cases


# ---------------------------------------------------------------------------


def generate(out_dir, scene_path=DEFAULT_SCENE) -> tuple[Path, Path]:
    """Write the golden's ``.cvcnm`` and JSON into ``out_dir``; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = json.loads(Path(scene_path).read_text())

    contract = BeliefMaterialContract()
    cvcnm = out_dir / CVCNM_NAME
    write_matnet(golden_net(), str(cvcnm), belief=NET_HEADER, meta=contract.to_meta())
    controller = BeliefMaterialController.from_cvcnm(
        cvcnm, patch_size=NET["patch_size"], patch_extent_m=PATCH_EXTENT_M
    )
    obstacles, planes, tokens = scene_cases(scene)

    golden = {
        "about": (
            "libcvc's belief-material controller twin (NAV-BELIEF-MATERIAL-CONTROLLER-ROADMAP.md "
            "§6), written by GRL-SNAM's grl_snam.tools.belief_material_controller_golden. "
            "Float32 values are raw bits (uint32). The scene is belief_material_golden.json."
        ),
        "net": {
            "file": CVCNM_NAME,
            **NET,
            "patch_extent_m": PATCH_EXTENT_M,
            "seed": NET_SEED,
            "meta": contract.to_meta(),
        },
        "scene": {
            "file": "belief_material_golden.json",
            "rows": scene["grid"]["rows"],
            "cols": scene["grid"]["cols"],
            "bounds": scene["grid"]["bounds"],
            "agent_map": scene["agent_map"],
            "cases": list(TOKEN_CASES),
        },
        "contract": contract_cases(),
        "header_check": header_cases(cvcnm),
        "runtime_check": runtime_cases(controller),
        "material_config_for": material_config_cases(),
        "obstacles": obstacles,
        "obstacle_ties": tie_cases(),
        "planes": planes,
        "tokens": tokens,
    }
    path = out_dir / JSON_NAME
    # Bytes, not text: a text-mode write would end the file with \r\n on Windows.
    path.write_bytes((json.dumps(golden, separators=(",", ":")) + "\n").encode("utf-8"))
    return path, cvcnm


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE), help="belief_material_golden.json")
    args = ap.parse_args(argv)
    for path in generate(args.out, args.scene):
        print(path)


if __name__ == "__main__":
    main()
