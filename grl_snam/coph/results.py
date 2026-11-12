"""Summaries of the vendored demos' episodes: ``results.json`` and ``REPORT.md``.

GRL-SNAM code, not vendored. Upstream writes these tables from its reporting modules
(``crossing_visuals.report`` and ``hazard_demo.render``), which also draw figures and GIFs and
stay out of ``grl_snam.coph``. ``doors_of``, ``door_name``, ``path_length``, ``door_events`` and
``metrics`` are copied from upstream's ``crossing_visuals.py`` so that the path, detour and saving
arithmetic is upstream's own; the tables follow upstream's wording.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import UPSTREAM
from .evaluation import dump, load
from .route_oracle import shortest

CROSSING_ARMS = ("disabled", "one_way_01", "one_way_10", "reciprocal_fixed")


# --- copied from upstream crossing_visuals.py ------------------------------------------------
def doors_of(spec):
    return [tuple(o) for o in spec["obstacles"] if len(o) == 5]


def door_name(spec, obstacle):
    """'door k' for a slab in the spec, None for anything else."""
    key = tuple(obstacle)
    doors = doors_of(spec)
    return f"door {doors.index(key) + 1}" if key in doors else None


def path_length(samples):
    p = np.array([s["x"][:2] for s in samples])
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def door_events(r):
    out = {}
    for i, events in enumerate(r["geometry_events"]):
        for e in events:
            name = door_name(r["spec"], e["obstacle"])
            if name is None:
                continue
            key = (name, i, "sensed" if e["kind"] == "locally_sensed" else "received")
            out[key] = min(out.get(key, np.inf), e["time"])
    return out


def metrics(r):
    spec = r["spec"]
    events = door_events(r)
    agents = []
    for i, a in enumerate(r["agents"]):
        length = path_length(r["samples"][i])
        best = shortest(
            spec["network_nodes"],
            spec["network_edges"],
            doors_of(spec),
            spec["initials"][i],
            spec["goals"][i],
        )
        agents.append(
            dict(
                agent=i,
                success=a["success"],
                collision=a["collision"],
                timeout=a["timeout"],
                finish_time=a["finish_time"],
                path=length,
                wasted=length - best,
                shortest=best,
                doors={
                    name: dict(
                        sensed=events.get((name, i, "sensed")),
                        received=events.get((name, i, "received")),
                    )
                    for name in {k[0] for k in events}
                },
                bytes=sum(a["bytes"].values()),
            )
        )
    return dict(
        seed=spec["seed"],
        arm=r["arm"],
        agents=agents,
        goals=sum(a["success"] for a in agents),
        team_wasted=sum(a["wasted"] for a in agents),
        team_finish=max(a["finish_time"] for a in agents),
        bytes=sum(a["bytes"] for a in agents),
        wall_seconds=r["wall_seconds"],
    )


# --- end of the copy -------------------------------------------------------------------------


def _episodes(directory: Path):
    for p in sorted((directory / "episodes").glob("*.json.gz")):
        r = load(p)
        if "agents" in r:
            yield r


def crossing(root) -> dict:
    """Summarize a crossing run: ``rows.json``, ``results.json`` and ``REPORT.md`` in *root*."""
    root = Path(root)
    manifest = load(root / "manifest.json")
    rows = [metrics(r) for r in _episodes(root)]
    rows.sort(key=lambda r: (r["seed"], CROSSING_ARMS.index(r["arm"])))
    dump(root / "rows.json", rows)
    pred = manifest["predictions"]
    oracle_team = {a: sum(v["wasted"] for v in pred[a].values()) for a in pred}
    oracle_saving = {a: oracle_team["disabled"] - oracle_team[a] for a in pred}

    per_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        mine = {r["arm"]: r for r in rows if r["seed"] == seed}
        entry = dict(
            arms=sorted(mine, key=CROSSING_ARMS.index),
            finish_times={a: [z["finish_time"] for z in r["agents"]] for a, r in mine.items()},
            paths={a: [z["path"] for z in r["agents"]] for a, r in mine.items()},
            bytes={a: r["bytes"] for a, r in mine.items()},
        )
        if "disabled" in mine:
            saving = {
                a: mine["disabled"]["team_wasted"] - r["team_wasted"]
                for a, r in mine.items()
                if a != "disabled"
            }
            entry["savings"] = saving
            if "one_way_01" in saving and "one_way_10" in saving:
                entry["one_way_saving"] = saving["one_way_01"] + saving["one_way_10"]
            if "reciprocal_fixed" in saving:
                entry["reciprocal_saving"] = saving["reciprocal_fixed"]
        per_seed[str(seed)] = entry

    missions = [z for r in rows for z in r["agents"]]
    result = dict(
        demo="crossing",
        upstream=UPSTREAM,
        steps=manifest["config"]["steps"],
        seeds=sorted({r["seed"] for r in rows}),
        episodes=len(rows),
        missions=len(missions),
        goals=sum(bool(z["success"]) for z in missions),
        collisions=sum(bool(z["collision"]) for z in missions),
        timeouts=sum(bool(z["timeout"]) for z in missions),
        per_seed=per_seed,
        oracle=dict(
            one_way_saving=oracle_saving["one_way_01"] + oracle_saving["one_way_10"],
            reciprocal_saving=oracle_saving["reciprocal_fixed"],
            savings={a: v for a, v in oracle_saving.items() if a != "disabled"},
        ),
        rows=rows,
    )
    dump(root / "results.json", result)
    (root / "REPORT.md").write_text(_crossing_report(manifest, rows, oracle_team) + "\n")
    return result


def _crossing_report(manifest, rows, oracle_team):
    spec = manifest["specs"][0]
    delay = manifest["communication"].get("delay_ticks", 0) * manifest["config"]["dt"]
    pred = manifest["predictions"]
    lines = [
        f'# {spec["name"]}: results',
        "",
        f"Development run. Message delay {delay:.2f} s. Doors are hidden from both agents. "
        "Oracle predictions were frozen in `manifest.json` before any episode ran.",
        "",
        "| Seed | Arm | Goals | Collisions | A path (m) | B path (m) | A detour, meas / oracle "
        "| B detour, meas / oracle | A finish (s) | B finish (s) | Bytes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        a, b = r["agents"]
        p = pred[r["arm"]]
        lines.append(
            f'| {r["seed"]} | {r["arm"]} | {r["goals"]}/2 '
            f'| {sum(z["collision"] for z in r["agents"])} | {a["path"]:.1f} | {b["path"]:.1f} '
            f'| {a["wasted"]:.1f} / {p["0"]["wasted"]:.1f} | {b["wasted"]:.1f} / {p["1"]["wasted"]:.1f} '
            f'| {a["finish_time"]:.1f} | {b["finish_time"]:.1f} | {r["bytes"]} |'
        )
    lines += [
        "",
        "## Saving from communication (team detour vs. no communication)",
        "",
        "| Seed | Arm | Measured | Oracle |",
        "|---|---|---|---|",
    ]
    for seed in sorted({r["seed"] for r in rows}):
        w = {r["arm"]: r["team_wasted"] for r in rows if r["seed"] == seed}
        for arm in CROSSING_ARMS[1:]:
            if arm in w and "disabled" in w:
                lines.append(
                    f'| {seed} | {arm} | {w["disabled"] - w[arm]:.1f} m '
                    f'| {oracle_team["disabled"] - oracle_team[arm]:.1f} m |'
                )
    lines += [
        "",
        "Detour is the measured path length minus the true shortest centerline route with every "
        "door known; corner cutting biases it low. The oracle assumes equal constant speed, instant "
        "turns and the declared delay, and ignores the planner's terrain costs. Agents are "
        "independent plants with no inter-agent contact dynamics.",
    ]
    return "\n".join(lines)


def hazard(root) -> dict:
    """Summarize a hazard run: ``results.json`` and ``REPORT.md`` in *root*.

    The headline is the carrier's excess cost: its time plus weighted risk cost, minus the best
    tested arm's in the same scene and seed, and its worst case over the scenes.
    """
    from .hazard_demo import ARMS, RAW_ANCHORS, SCENES, STAGGER, UNKNOWN_RISK, WEIGHTS

    root = Path(root)
    arms = list(ARMS)
    scenes = [s for s in SCENES if (root / s / "manifest.json").exists()]
    per_seed = {}
    missions = goals = collisions = 0
    steps = None
    for scene in scenes:
        d = root / scene
        m = load(d / "manifest.json")
        steps = m["config"]["steps"]
        oracle = {a: m["predictions"][a][1]["cost"] for a in arms}
        obest = min(oracle.values())
        for spec in m["specs"]:
            seed = spec["seed"]
            cost, ok = {}, {}
            for a in arms:
                r = load(d / "episodes" / f"{seed}_{a}.json.gz")
                z = r["agents"][1]
                cost[a] = z["components"]["time"] + z["components"]["risk"]
                ok[a] = sum(x["success"] for x in r["agents"])
                missions += len(r["agents"])
                goals += ok[a]
                collisions += sum(bool(x["collision"]) for x in r["agents"])
            best = min(cost.values())
            entry = per_seed.setdefault(str(seed), dict(scenes={}))
            entry["scenes"][scene] = {
                a: dict(
                    carrier_cost=cost[a],
                    excess=cost[a] - best,
                    oracle_excess=oracle[a] - obest,
                    goals=ok[a],
                )
                for a in arms
            }
    for entry in per_seed.values():
        entry["worst_case_excess"] = {
            a: max(s[a]["excess"] for s in entry["scenes"].values()) for a in arms
        }
    worst = {a: max(e["worst_case_excess"][a] for e in per_seed.values()) for a in arms}
    result = dict(
        demo="hazard",
        upstream=UPSTREAM,
        steps=steps,
        seeds=sorted(int(s) for s in per_seed),
        scenes=scenes,
        arms=arms,
        missions=missions,
        goals=goals,
        collisions=collisions,
        worst_case_excess=worst,
        per_seed=per_seed,
    )
    dump(root / "results.json", result)
    lines = [
        "# Graded hazards: share the environmental field, not the decision",
        "",
        f"Settings: {len(per_seed)} noise seed(s) on {len(scenes)} designed scenes. Scout risk "
        f"weight {WEIGHTS[0]}, carrier {WEIGHTS[1]}; carrier starts {STAGGER:.0f} s later. "
        f"Unknown-terrain prior {UNKNOWN_RISK}. Field packets carry 4 anchors, raw packets "
        f"{RAW_ANCHORS}. Oracle predictions were frozen in each scene manifest before the runs. "
        "Cost = time + weighted risk exposure (effort excluded).",
    ]
    for seed, entry in sorted(per_seed.items()):
        lines += ["", f"Seed {seed}:", "", "| Scene | " + " | ".join(arms) + " |"]
        lines.append("|---|" + "---|" * len(arms))
        for scene, cells in entry["scenes"].items():
            lines.append(
                f"| {scene} | "
                + " | ".join(
                    f'{c["excess"]:.1f} ({c["oracle_excess"]:.1f})'
                    + ("" if c["goals"] == 2 else " FAIL")
                    for c in (cells[a] for a in arms)
                )
                + " |"
            )
        lines.append(
            "| **worst-case excess cost** | "
            + " | ".join(f'**{entry["worst_case_excess"][a]:.1f}**' for a in arms)
            + " |"
        )
    lines += [
        "",
        "Cells: carrier excess cost relative to the best tested arm in that scene (not regret "
        "against a globally optimal policy), measured (oracle). Every episode reached both goals "
        "unless marked FAIL.",
        "Oracle agreement is qualitative: it idealizes constant speed and immediate full transfer, "
        "so magnitudes can differ (for example a timing cost the oracle omits).",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result
