"""Record a fog-of-war story to a measured trace bundle.

The simulation runs here and nowhere else. Everything downstream — the live
demo, the mp4, the end cards — replays what this wrote. See
:mod:`grl_snam.fog_trace` for the reader and the reasoning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from grl_snam.clock import WorldClock
from grl_snam.fog_stories import STORIES, Story, build_scenario

# Row fields captured per world tick. Everything the renderers need must be
# here: anything they have to re-derive is a chance to disagree with the sim.
_ROW_FIELDS = (
    "tick",
    "t",
    "x",
    "y",
    "heading_rad",
    "speed_mps",
    "clearance_m",
    "goal_dist_m",
    "goal_index",
    "belief_version",
    "rebuilt",
    "sensed",
    "truth_penetration",
)


def story_hash(story: Story) -> str:
    """Content hash of the story spec, so a stale trace can be detected."""
    payload = json.dumps(asdict(story), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def record(
    story_key: str,
    out_dir: str | Path | None = None,
    *,
    model=None,
    seed: int = 0,
    max_steps: int | None = None,
    story: Story | None = None,
    progress=None,
) -> Path:
    """Run ``story_key`` once and write ``trace.npz`` + ``trace.json``."""
    story = story or STORIES[story_key]
    out = Path(out_dir) if out_dir else Path("traces") / story.key
    out.mkdir(parents=True, exist_ok=True)

    sc = build_scenario(story, model, seed=seed)
    # replay mode: the recorder owns the tick, wall time is irrelevant here.
    clock = WorldClock(fixed_dt=story.dt, mode="replay")

    rows: dict[str, list] = {k: [] for k in _ROW_FIELDS}
    snap_ticks: list[int] = []
    snap_occ: list[np.ndarray] = []
    snap_dyn: list[np.ndarray] = []
    routes: list[np.ndarray] = []

    def snapshot(tick: int) -> None:
        occ = sc.belief.to_occupancy(unknown=story.unknown)
        dyn = sc.dyn.occupancy(sc._t())
        snap_ticks.append(tick)
        snap_occ.append(np.packbits(occ.ravel()))
        snap_dyn.append(np.packbits(dyn.ravel()))
        routes.append(np.asarray(sc.route or [], np.float32).reshape(-1, 2))

    snapshot(0)  # the prior map, before a single ray is cast

    limit = max_steps or story.max_steps
    for _ in range(limit):
        rec = sc.step()
        clock.step_once()

        # Clearance is measured from the navigator's own field, not
        # recomputed later from a grid the renderer happens to have.
        import torch

        with torch.no_grad():
            phi, _ = sc.nav.field.sample(sc.nav.o)
        clearance = (float(phi.reshape(-1)[0]) - story.rr) / story.scale

        rows["tick"].append(clock.tick())
        rows["t"].append(clock.t())
        rows["x"].append(rec.x)
        rows["y"].append(rec.y)
        rows["heading_rad"].append(rec.heading_rad)
        rows["speed_mps"].append(rec.speed_mps)
        rows["clearance_m"].append(clearance)
        rows["goal_dist_m"].append(rec.goal_dist_m)
        rows["goal_index"].append(rec.goal_index)
        rows["belief_version"].append(rec.belief_version)
        rows["rebuilt"].append(bool(rec.rebuilt))
        rows["sensed"].append(sc.step_i % story.sense_every == 0)
        rows["truth_penetration"].append(bool(rec.truth_penetration))

        if rec.rebuilt:
            snapshot(clock.tick())
        if progress and clock.tick() % 50 == 0:
            progress(clock.tick(), limit)
        if sc.done:
            break

    n = len(rows["tick"])
    xs = np.asarray(rows["x"])
    ys = np.asarray(rows["y"])
    # Straight-line reference from start to the final waypoint, so "detour" is
    # a measured quantity rather than an adjective.
    gx, gy = story.waypoints[-1]
    sx, sy = story.start
    seg = np.hypot(gx - sx, gy - sy)
    if seg > 1e-9:
        # perpendicular distance from the start->goal line
        detour = np.abs((gx - sx) * (sy - ys) - (sx - xs) * (gy - sy)) / seg
    else:
        detour = np.zeros_like(xs)

    summary = {
        "steps": n,
        "world_seconds": round(n * story.dt, 3),
        "map_updates": len(snap_ticks) - 1,
        "penetration_steps": int(np.sum(rows["truth_penetration"])),
        "waypoints_reached": int(sc.wp_i + (1 if sc.done else 0)),
        "reached_goal": bool(sc.done),
        "detour_peak_m": round(float(detour.max()), 1),
        "path_length_m": round(float(np.hypot(np.diff(xs), np.diff(ys)).sum()), 1),
        "straight_line_m": round(float(seg), 1),
        "end_xy": [round(float(xs[-1]), 1), round(float(ys[-1]), 1)],
    }

    npz = {f"row_{k}": np.asarray(v) for k, v in rows.items()}
    npz["snap_tick"] = np.asarray(snap_ticks, np.int64)
    npz["snap_occ"] = np.stack(snap_occ)
    npz["snap_dyn"] = np.stack(snap_dyn)
    offsets = np.cumsum([0] + [len(r) for r in routes])
    npz["route_offsets"] = offsets.astype(np.int64)
    npz["route_points"] = (
        np.concatenate(routes) if any(len(r) for r in routes) else np.empty((0, 2), np.float32)
    )
    np.savez_compressed(out / "trace.npz", **npz)

    try:
        import torch

        torch_version = torch.__version__
    except Exception:  # pragma: no cover -- torch is a hard dep of recording
        torch_version = "unknown"

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
        "start": list(story.start),
        "waypoints": [list(w) for w in story.waypoints],
        "truth_rects": [list(r) for r in story.truth_rects],
        "prior_rects": [list(r) for r in story.prior_rects],
        "captions": [list(c) for c in story.captions],
        "cam": story.cam,
        "seed": seed,
        "torch": torch_version,
        "summary": summary,
    }
    (out / "trace.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def is_stale(trace_dir: str | Path, story: Story) -> bool:
    """True if no trace exists or it was recorded from a different spec."""
    p = Path(trace_dir) / "trace.json"
    if not p.exists():
        return True
    try:
        return json.loads(p.read_text()).get("story_hash") != story_hash(story)
    except Exception:
        return True
