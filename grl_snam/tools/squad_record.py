"""Record a squad run: one trace per agent, plus the shared manifest.

Deliberately reuses the single-agent trace layout rather than inventing a
multi-agent one. Each agent's rows/snapshots are exactly what
:class:`~grl_snam.fog_trace.Trace` already reads, so the renderer, the metrics
bridge and every existing test keep working, and a squad clip is N traces drawn
into one scene rather than a second parallel format to keep in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grl_snam.clock import WorldClock
from grl_snam.squad import AgentSpec, Squad
from grl_snam.tools.fog_record import _ROW_FIELDS, story_hash


def record_squad(
    story,
    agents: list[AgentSpec],
    out_dir: str | Path,
    *,
    model=None,
    seed: int = 0,
    max_steps: int | None = None,
    stall_ticks: int = 0,
    progress=None,
    truth_occ=None,
    prior_occ=None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    squad = Squad(story, agents, model, seed=seed, truth_occ=truth_occ, prior_occ=prior_occ)
    clock = WorldClock(fixed_dt=story.dt, mode="replay")

    rows = {a.key: {k: [] for k in _ROW_FIELDS} for a in agents}
    goal_x = {a.key: [] for a in agents}
    goal_y = {a.key: [] for a in agents}
    snaps = {a.key: {"tick": [], "occ": [], "dyn": []} for a in agents}
    fovs = {a.key: {"tick": [], "vis": [], "seen": []} for a in agents}
    routes = {a.key: [] for a in agents}

    def snapshot(key, sc, tick):
        s = snaps[key]
        s["tick"].append(tick)
        s["occ"].append(np.packbits(sc.belief.to_occupancy(unknown=story.unknown).ravel()))
        s["dyn"].append(np.packbits(sc.dyn.occupancy(sc._t()).ravel()))
        routes[key].append(np.asarray(sc.route or [], np.float32).reshape(-1, 2))

    for key, sc in squad.scenarios.items():
        snapshot(key, sc, 0)

    limit = max_steps or story.max_steps
    best = {a.key: float("inf") for a in agents}
    since = 0
    for _ in range(limit):
        recs = squad.step()
        improved = False
        for _k, _r in recs.items():
            if _r.goal_dist_m < best[_k] - 0.5:
                best[_k] = _r.goal_dist_m
                improved = True
        since = 0 if improved else since + 1
        clock.step_once()
        tick = clock.tick()
        import torch

        for key, rec in recs.items():
            sc = squad.scenarios[key]
            with torch.no_grad():
                phi, _ = sc.nav.field.sample(sc.nav.o)
            r = rows[key]
            r["tick"].append(tick)
            r["t"].append(clock.t())
            r["x"].append(rec.x)
            r["y"].append(rec.y)
            r["heading_rad"].append(rec.heading_rad)
            r["speed_mps"].append(rec.speed_mps)
            r["clearance_m"].append((float(phi.reshape(-1)[0]) - story.rr) / story.scale)
            r["goal_dist_m"].append(rec.goal_dist_m)
            r["goal_index"].append(rec.goal_index)
            r["belief_version"].append(rec.belief_version)
            r["rebuilt"].append(bool(rec.rebuilt))
            r["sensed"].append(sc.step_i % story.sense_every == 0)
            r["truth_penetration"].append(bool(rec.truth_penetration))
            gx, gy = sc.waypoints[sc.wp_i]
            goal_x[key].append(float(gx))
            goal_y[key].append(float(gy))

            if rec.rebuilt:
                snapshot(key, sc, tick)
            if r["sensed"][-1]:
                f = fovs[key]
                f["tick"].append(tick)
                f["vis"].append(np.packbits(sc.belief.last_visible.ravel()))
                f["seen"].append(np.packbits(sc.belief.ever_seen.ravel()))

        if progress and tick % 50 == 0:
            progress(tick, limit)
        if squad.done:
            break
        if stall_ticks and since >= stall_ticks:
            break

    for a in agents:
        k = a.key
        npz = {f"row_{f}": np.asarray(v) for f, v in rows[k].items()}
        npz["row_goal_x"] = np.asarray(goal_x[k])
        npz["row_goal_y"] = np.asarray(goal_y[k])
        npz["snap_tick"] = np.asarray(snaps[k]["tick"], np.int64)
        npz["snap_occ"] = np.stack(snaps[k]["occ"])
        npz["snap_dyn"] = np.stack(snaps[k]["dyn"])
        npz["fov_tick"] = np.asarray(fovs[k]["tick"], np.int64)
        blank = np.zeros((0, (story.n * story.n + 7) // 8), np.uint8)
        npz["fov_vis"] = np.stack(fovs[k]["vis"]) if fovs[k]["vis"] else blank
        npz["fov_seen"] = np.stack(fovs[k]["seen"]) if fovs[k]["seen"] else blank
        # Static truth for a squad run: peers are drawn from their own traces,
        # not baked into truth, so every agent shares this one grid.
        npz["truth_tick"] = np.asarray([0], np.int64)
        npz["truth_snap"] = np.stack(
            [np.packbits((story.truth_grid() if truth_occ is None else truth_occ).ravel())]
        )
        offs = np.cumsum([0] + [len(r) for r in routes[k]])
        npz["route_offsets"] = offs.astype(np.int64)
        npz["route_points"] = (
            np.concatenate(routes[k])
            if any(len(r) for r in routes[k])
            else np.empty((0, 2), np.float32)
        )
        agent_dir = out / k
        agent_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(agent_dir / "trace.npz", **npz)

        n = len(rows[k]["tick"])
        xs, ys = np.asarray(rows[k]["x"]), np.asarray(rows[k]["y"])
        manifest = {
            "story": story.key,
            "title": story.title,
            "subtitle": story.subtitle,
            "story_hash": story_hash(story),
            "fixed_dt": story.dt,
            "n_ticks": n,
            "shape": [story.n, story.n],
            "bounds": list(story.bounds),
            "scale": story.scale,
            "start": list(a.start),
            "waypoints": [list(a.goal)],
            "captions": [list(c) for c in story.captions],
            "sensor_range_m": float(story.sensor.get("range_m", 0.0)),
            "sense_every": int(story.sense_every),
            "use_planner": bool(story.use_planner),
            "nav": "route+sdf" if story.use_planner else "sdf-only",
            "agent": k,
            "color": list(a.color),
            "seed": seed,
            "summary": {
                "steps": n,
                "world_seconds": round(n * story.dt, 3),
                "map_updates": len(snaps[k]["tick"]) - 1,
                "penetration_steps": int(np.sum(rows[k]["truth_penetration"])),
                "reached_goal": bool(squad.scenarios[k].done),
                "path_length_m": round(float(np.hypot(np.diff(xs), np.diff(ys)).sum()), 1),
            },
        }
        (agent_dir / "trace.json").write_text(json.dumps(manifest, indent=2) + "\n")

    (out / "squad.json").write_text(
        json.dumps(
            {
                "story": story.key,
                "agents": [
                    {
                        "key": a.key,
                        "color": list(a.color),
                        "start": list(a.start),
                        "goal": list(a.goal),
                    }
                    for a in agents
                ],
                "ticks": len(rows[agents[0].key]["tick"]),
                "all_reached": all(sc.done for sc in squad.scenarios.values()),
            },
            indent=2,
        )
        + "\n"
    )
    return out
