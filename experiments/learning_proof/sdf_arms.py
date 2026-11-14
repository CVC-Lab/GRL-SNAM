"""Score the SDF policy's arms on held-out data: is the retrained ``coef_sdf`` better than not
training it?

Arms:

- ``published``: ``grl-snam-weights``' ``coef_sdf.pt`` (200 steps, seed 0);
- ``trained``: per training seed, the held-out pick (``coef_sdf.best_heldout.pt``);
- ``untrained``: per training seed, its step-0 net (``coef_sdf.step000000.pt``);
- ``final``: per training seed, the last step (``coef_sdf.pt``), shown to say what the pick bought;
- ``constant``: the output layer zeroed, each coefficient the published policy's mean on the
  Austin test drive (``coef_export.constant_coef_mlp``);
- ``constant_trained``: per training seed, a constant net at that seed's ``trained`` means on the
  same drive. ``trained`` against it asks whether the net uses what it sees, or only moved its
  mean.
- ``constant_final``: the same at each seed's ``final`` means, the control for ``final`` (when
  the pick is step 0, ``constant_trained`` controls the untrained net, not what training made).

Scorers (every arm drives the same units, so comparisons are paired):

- ``austin``: the seek drive (``grl_snam.tools.sdf_eval.seek_drive``, the point plant the policy
  trained on) on fixed test pairs started inside the held-out blocks (the training run's split,
  read from its history header), drawn with ``--test-seed``: the pick's own pairs are not reused;
- ``city``: the vectorized Swarm on ``--city-scenes`` procedural city scenes (seeds 0..N-1, as
  ``grl_snam.tools.scorecard_eval``), one unit per agent: held-out geometry in another frame;
- ``native``: the same city scenes on libcvc's ``sim_world`` (``nav_native.sim_world_from_swarm``)
  driving each arm's exported ``.cvcnav``, when the pycvc has it;
- ``austin_native`` (``--austin-native``): the Austin test pairs on libcvc's ``sim_world``, every
  pair an agent of one world per arm, on nav's kinematic bicycle with the map known: the deployed
  navigator (its carrot FSM with the wall-follow escape, and the park at arrival), as
  ``grl-snam capture --plant kinematic`` drives it, for ``--native-ticks``.

    python experiments/learning_proof/sdf_arms.py --sdf nav_sdf.npz --runs runs/retrain \\
        --published coef_sdf.pt --out arms.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

import grl_snam.sdf_nav as sdf_nav
from grl_snam.learning_history import coef_heads, read_history
from grl_snam.tools import sdf_eval
from grl_snam.tools.coef_export import write_coef_mlp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import level, paired  # noqa: E402

METRICS = ("reached", "pen_ticks", "path_m", "min_clear_m", "time_to_goal_s", "gap_m")
#: Metrics defined only where both arms of a comparison arrived.
BOTH_ARRIVED = ("path_m", "time_to_goal_s")


def constant_coef_mlp(alpha, beta, gamma):
    """A ``CoefMLP`` that outputs ``(alpha, beta, gamma)`` whatever it sees: the output layer's
    weights zero and its biases ``log(expm1(c)) - log(expm1(bias))``, which the forward's
    ``softplus(net + log(expm1(bias)))`` turns back into ``c``. (CVC-Lab/GRL-SNAM#171 adds the
    same arm as ``coef_export --constant``.)"""
    target = torch.tensor([alpha, beta, gamma], dtype=torch.float32)
    m = sdf_nav.CoefMLP()
    with torch.no_grad():
        m.net[-1].weight.zero_()
        m.net[-1].bias.copy_(torch.log(torch.expm1(target)) - torch.log(torch.expm1(m.bias)))
    return m.eval()


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = sdf_nav.CoefMLP()
    m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    return m.eval()


def arms_from_runs(runs: str, published: str):
    seeds = sorted(glob.glob(os.path.join(runs, "seed*")))
    arms = {"published": [load(published)]}
    for kind, name in (
        ("trained", "coef_sdf.best_heldout.pt"),
        ("untrained", "coef_sdf.step000000.pt"),
        ("final", "coef_sdf.pt"),
    ):
        arms[kind] = [load(os.path.join(d, name)) for d in seeds]
    picks = []
    for d in seeds:
        end = [r for r in read_history(os.path.join(d, "history.jsonl")) if r["kind"] == "end"]
        picks.append(end[0]["best_step"])
    header = read_history(os.path.join(seeds[0], "history.jsonl"))[0]
    return arms, [os.path.basename(d) for d in seeds], picks, header


# --------------------------------------------------------------------------- Austin


def austin(arms, args, header):
    scene = sdf_eval.load_scene(args.sdf)
    ck = torch.load(args.published, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    kw = dict(rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"])
    pool = sdf_eval.free_pool(scene, meta["rr"])
    blocks = header["val_blocks"]
    held = header["heldout_blocks"]
    _, held_pool = sdf_eval.split_pool(pool, scene.region_n, blocks, held)
    starts, goals = sdf_eval.sample_pairs(
        scene, held_pool, args.test_pairs, args.test_seed, rr=meta["rr"]
    )
    block = sdf_eval.block_ids(starts.numpy(), scene.region_n, blocks)
    drive = dict(kw, nsub=meta["nsub"], ticks=args.ticks, reach_tol=args.reach_tol)
    out = {}
    for name, models in arms.items():
        out[name] = [
            _austin_unit(sdf_eval.seek_drive(m, scene, starts, goals, **drive), meta)
            for m in models
        ]
    return out, {"block": block, "meta": meta}


def _austin_unit(d, meta):
    t = d["ticks_to_goal"] * meta["dt"]
    return {
        "reached": d["reached"].astype(np.float64),
        "pen_ticks": d["pen_ticks"].astype(np.float64),
        "path_m": d["path_m"],
        "min_clear_m": d["min_clear_m"],
        "time_to_goal_s": np.where(d["reached"], t, np.nan),
        "gap_m": d["gap_m"],
        "alpha": d["alpha"],
        "beta": d["beta"],
        "gamma": d["gamma"],
    }


# --------------------------------------------------------------------------- city (torch + native)


def city(arms, args):
    from grl_snam.swarm import Swarm
    from grl_snam.tools.scorecard_eval import scene_specs

    out = {name: [_empty() for _ in models] for name, models in arms.items()}
    native = None
    try:
        from grl_snam import nav_native

        if nav_native.HAS_SIM_WORLD and getattr(nav_native, "HAS_SIM_WORLD_NAVSTATS", False):
            native = {name: [_empty() for _ in models] for name, models in arms.items()}
    except Exception:  # noqa: BLE001 - no pycvc: the native scorer is skipped
        native = None
    tmp = tempfile.mkdtemp(prefix="sdf-arms-")
    scene_of = []
    for seed in range(args.city_scenes):
        story, truth, specs = scene_specs(args.city_grid, seed, args.city_agents)
        scene_of += [seed] * len(specs)
        for name, models in arms.items():
            for k, m in enumerate(models):
                sw = Swarm(
                    story, specs, model=m, truth_occ=truth, prior_occ=truth, collect_stats=True
                )
                heads = _start_heads(sw, m)
                for _ in range(args.city_steps):
                    sw.step()
                    if sw.all_reached:
                        break
                _append(out[name][k], _swarm_units(sw), heads)
                if native is not None:
                    wp = os.path.join(tmp, f"{name}{k}.cvcnav")
                    write_coef_mlp(m, wp)
                    sw2 = Swarm(story, specs, model=m, truth_occ=truth, prior_occ=truth)
                    sw2._sense_shared = lambda: None
                    cw = nav_native.sim_world_from_swarm(sw2, wp, truth=truth, freeze_sense=True)
                    cw.begin_nav_stats(scene_id=f"city{seed}", seed=seed, checkpoint=name)
                    for _ in range(args.city_steps):
                        cw.step()
                        if bool(cw.snapshot()[4].all()):
                            break
                    _append(native[name][k], _native_units(cw, sw2), heads)
    for d in (out, native):
        if d is None:
            continue
        for runs in d.values():
            for u in runs:
                for k in list(u):
                    u[k] = np.asarray(u[k], np.float64)
    return out, native, np.asarray(scene_of)


def _empty():
    return {k: [] for k in (*METRICS, "alpha", "beta", "gamma")}


def _append(acc, units, heads):
    for k, v in units.items():
        acc[k] += list(v)
    for k, v in heads.items():
        acc[k] += list(v)


@torch.no_grad()
def _start_heads(sw, m):
    """The heads each agent's policy outputs at its start toward its first carrot."""
    dg = sw.goal - sw.o
    dist = dg.norm(dim=-1, keepdim=True)
    c = sw.o + dg / (dist + 1e-6) * dist.clamp(max=1.8)
    h = coef_heads(m, sdf_nav.coef_feats(sw.field, sw.o, c))
    return {k: h[k].numpy() for k in ("alpha", "beta", "gamma")}


def _swarm_units(sw):
    ns = sw._sc_stats
    tick = sw._sc_arrival_tick.tolist()
    t = [float(x) * sw.dt if int(x) >= 0 else np.nan for x in tick]
    return {
        "reached": [float(int(x) >= 0) for x in tick],
        "pen_ticks": [float(s.penetration_steps) for s in ns],
        "path_m": [s.total_path_m for s in ns],
        "min_clear_m": [s.min_clearance_m for s in ns],
        "time_to_goal_s": t,
        "gap_m": list(((sw.goal - sw.o).norm(dim=-1) / sw.S).numpy()),
    }


def _native_units(cw, sw):
    rec = cw.nav_stats()
    veh = rec["vehicles"] if "vehicles" in rec else rec["per_vehicle"]
    pos, *_ = cw.snapshot()
    goal_w = sw.n2w(sw.goal.clone()).cpu().numpy()
    out = {k: [] for k in METRICS}
    for i, v in enumerate(veh):
        arrived = bool(v.get("arrived", v.get("goals_reached", 0) >= 1))
        out["reached"].append(float(arrived))
        out["pen_ticks"].append(float(v.get("penetration_steps", 0)))
        out["path_m"].append(float(v.get("total_path_m", 0.0)))
        out["min_clear_m"].append(float(v.get("min_clearance_m", np.nan)))
        ttg = float(v.get("time_to_goal_s", -1.0))
        out["time_to_goal_s"].append(ttg if arrived and ttg >= 0 else np.nan)
        out["gap_m"].append(float(np.hypot(*(pos[i] - goal_w[i]))))
    return out


def austin_native(arms, args, header):
    """The Austin test pairs on libcvc's sim_world (see the module docstring)."""
    from grl_snam import nav_native
    from grl_snam.nav import SdfNavigator

    _pycvc = nav_native._pycvc
    scene = sdf_eval.load_scene(args.sdf)
    meta = torch.load(args.published, map_location="cpu", weights_only=False)["meta"]
    pool = sdf_eval.free_pool(scene, meta["rr"])
    _, held_pool = sdf_eval.split_pool(
        pool, scene.region_n, header["val_blocks"], header["heldout_blocks"]
    )
    starts, goals = sdf_eval.sample_pairs(
        scene, held_pool, args.test_pairs, args.test_seed, rr=meta["rr"]
    )
    occ = np.ascontiguousarray(scene.phi < 0.0, np.uint8)
    rows, cols = occ.shape
    mnx, mny, mxx, mxy = scene.bounds
    v = SdfNavigator.VEHICLE_DEFAULTS
    o = np.ascontiguousarray(starts.numpy(), np.float32)
    g = np.ascontiguousarray(goals.numpy(), np.float32)
    n = o.shape[0]
    tmp = tempfile.mkdtemp(prefix="sdf-arms-native-")
    out = {}
    for name, models in arms.items():
        out[name] = []
        for k, m in enumerate(models):
            wp = os.path.join(tmp, f"{name}{k}.cvcnav")
            write_coef_mlp(m, wp)
            h = _pycvc.nav_sim_world_create(
                occ, occ, wp, o, g, np.ones((n, 3), np.float32), rows, cols, mnx, mny, mxx, mxy,
                scene.center[0], scene.center[1], scene.scale, 60.0, 240, 2 * np.pi,
                meta["rr"], meta["d_hat"], meta["dt"], meta["vmax"], v["L"], v["delta_max"],
                v["a_max"], v["a_lat_max"], v["k_steer"], int(meta["nsub"]),
                int(bool(v["allow_reverse"])), args.reach_tol, 4, 1, 2.2, -1.4, 8.0, 1, 0.5, 0.15,
                4.0,
            )  # fmt: skip
            world = nav_native.NativeSimWorld(h, n)
            world.set_plant(None)
            world.begin_nav_stats(scene_id="austin-test", seed=args.test_seed, checkpoint=name)
            for _ in range(args.native_ticks):
                world.step()
                if bool(world.snapshot()[4].all()):
                    break
            pos, *_ = world.snapshot()
            veh = world.nav_stats()["per_vehicle"]
            gw = np.stack(
                [g[:, 0] / scene.scale + scene.center[0], g[:, 1] / scene.scale + scene.center[1]],
                1,
            )
            u = {key: [] for key in METRICS}
            for i, r in enumerate(veh):
                arrived = int(r.get("goals_reached", 0)) >= 1 or bool(r.get("arrived", False))
                u["reached"].append(float(arrived))
                u["pen_ticks"].append(float(r.get("penetration_steps", 0)))
                u["path_m"].append(float(r.get("total_path_m", 0.0)))
                u["min_clear_m"].append(float(r.get("min_clearance_m", np.nan)))
                ttg = float(r.get("time_to_goal_s", -1.0))
                u["time_to_goal_s"].append(ttg if arrived and ttg >= 0 else np.nan)
                u["gap_m"].append(float(np.hypot(*(pos[i] - gw[i]))))
            out[name].append({key: np.asarray(val, np.float64) for key, val in u.items()})
    return out


# --------------------------------------------------------------------------- reduction


def stack(runs, key):
    return np.stack([np.asarray(u[key], np.float64) for u in runs])


def comparisons(units, *, n_boot, seed):
    """Per metric, each arm's level and the paired differences the check asks for."""
    pairs = {
        "trained-untrained": ("trained", "untrained"),
        "trained-constant": ("trained", "constant"),
        "trained-published": ("trained", "published"),
        "trained-constant_trained": ("trained", "constant_trained"),
        "final-untrained": ("final", "untrained"),
        "published-untrained": ("published", "untrained"),
    }
    if "constant_final" in units:
        pairs["final-constant"] = ("final", "constant")
        pairs["final-constant_final"] = ("final", "constant_final")
    out = {"levels": {}, "diffs": {}}
    for metric in METRICS:
        out["levels"][metric] = {}
        for arm, runs in units.items():
            x = stack(runs, metric)
            out["levels"][metric][arm] = level(
                np.nan_to_num(x), ~np.isnan(x), n_boot=n_boot, seed=seed
            )
        out["diffs"][metric] = {}
        for label, (a, b) in pairs.items():
            xa, xb = stack(units[a], metric), stack(units[b], metric)
            valid = ~(np.isnan(xa) | np.isnan(xb))
            if metric in BOTH_ARRIVED:
                ra, rb = stack(units[a], "reached"), stack(units[b], "reached")
                valid = valid & (ra > 0) & (rb > 0)
            out["diffs"][metric][label] = paired(
                np.nan_to_num(xa), np.nan_to_num(xb), valid, n_boot=n_boot, seed=seed
            )
    return out


def heads_spread(units, scene_of):
    """Per arm and head: its mean, the spread (standard deviation) of the per-scene means across
    scenes, and the mean within-scene spread; one entry per seed of a multi-seed arm."""
    out = {}
    scenes = np.unique(scene_of)
    for arm, runs in units.items():
        out[arm] = []
        for u in runs:
            r = {}
            for h in ("alpha", "beta", "gamma"):
                v = np.asarray(u[h])
                means = np.array([v[scene_of == s].mean() for s in scenes])
                within = np.array([v[scene_of == s].std() for s in scenes])
                r[h] = {
                    "mean": float(v.mean()),
                    "spread": float(means.std()),
                    "within": float(within.mean()),
                }
            out[arm].append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--runs", required=True, help="a directory of seed*/ training runs")
    ap.add_argument("--published", required=True)
    ap.add_argument("--out", required=True, help="arms.json; per-unit arrays go beside it (.npz)")
    ap.add_argument("--test-pairs", type=int, default=1024)
    ap.add_argument("--test-seed", type=int, default=777)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--reach-tol", type=float, default=0.05)
    ap.add_argument("--city-scenes", type=int, default=8)
    ap.add_argument("--city-agents", type=int, default=64)
    ap.add_argument("--city-steps", type=int, default=400)
    ap.add_argument("--city-grid", type=int, default=96)
    ap.add_argument("--no-city", action="store_true")
    ap.add_argument("--austin-native", action="store_true", help="also score Austin on sim_world")
    ap.add_argument("--native-ticks", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument(
        "--constant-final", action="store_true", help="add constant_final (final's own means)"
    )
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    t0 = time.time()

    arms, seeds, picks, header = arms_from_runs(args.runs, args.published)
    # The constant arms: the published policy's mean coefficients on the Austin test drive, and
    # each trained seed's own.
    first, _ = austin(
        {"published": arms["published"], "trained": arms["trained"], "final": arms["final"]},
        args,
        header,
    )

    def means_of(u):
        return [float(np.mean(u[h])) for h in ("alpha", "beta", "gamma")]

    means = means_of(first["published"][0])
    arms["constant"] = [constant_coef_mlp(*means)]
    trained_means = [means_of(u) for u in first["trained"]]
    arms["constant_trained"] = [constant_coef_mlp(*m) for m in trained_means]
    final_means = [means_of(u) for u in first["final"]]
    if args.constant_final:
        arms["constant_final"] = [constant_coef_mlp(*m) for m in final_means]
    stem = os.path.splitext(args.out)[0]
    for name in ("constant", "constant_trained", "constant_final"):
        for k, m in enumerate(arms.get(name, [])):
            path = f"{stem}.{name}{k if name != 'constant' else ''}.pt"
            torch.save({"model_state_dict": m.state_dict(), "meta": {"kind": name}}, path)
    a_units, a_info = austin(arms, args, header)
    print(f"austin scored ({time.time() - t0:.0f}s)", flush=True)
    result = {
        "schema": "grl_snam.learnproof_arms/v1",
        "seeds": seeds,
        "picks": picks,
        "constant": means,
        "constant_trained": trained_means,
        "constant_final": final_means if args.constant_final else None,
        "args": vars(args),
        "scorers": {},
    }
    result["scorers"]["austin"] = comparisons(a_units, n_boot=args.n_boot, seed=args.boot_seed)
    result["scorers"]["austin"]["heads"] = heads_spread(a_units, a_info["block"])
    npz = {
        f"austin.{a}.{k}.{i}": np.asarray(u[k])
        for a, runs in a_units.items()
        for i, u in enumerate(runs)
        for k in u
    }
    if args.austin_native:
        an_units = austin_native(arms, args, header)
        print(f"austin native scored ({time.time() - t0:.0f}s)", flush=True)
        result["scorers"]["austin_native"] = comparisons(
            an_units, n_boot=args.n_boot, seed=args.boot_seed
        )
        npz.update(
            {
                f"austin_native.{a}.{k}.{i}": np.asarray(u[k])
                for a, runs in an_units.items()
                for i, u in enumerate(runs)
                for k in u
            }
        )
    if not args.no_city:
        c_units, n_units, scene_of = city(arms, args)
        print(f"city scored ({time.time() - t0:.0f}s)", flush=True)
        result["scorers"]["city"] = comparisons(c_units, n_boot=args.n_boot, seed=args.boot_seed)
        result["scorers"]["city"]["heads"] = heads_spread(c_units, scene_of)
        npz.update(
            {
                f"city.{a}.{k}.{i}": np.asarray(u[k])
                for a, runs in c_units.items()
                for i, u in enumerate(runs)
                for k in u
            }
        )
        if n_units is not None:
            result["scorers"]["native"] = comparisons(
                n_units, n_boot=args.n_boot, seed=args.boot_seed
            )
            npz.update(
                {
                    f"native.{a}.{k}.{i}": np.asarray(u[k])
                    for a, runs in n_units.items()
                    for i, u in enumerate(runs)
                    for k in u
                }
            )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.splitext(args.out)[0] + ".npz", **npz)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
