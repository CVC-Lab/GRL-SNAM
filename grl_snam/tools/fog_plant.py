"""Record a fog-of-war story on a vehicle plant: libcvc's sim_world drives the story's one agent
on nav's kinematic bicycle, the dynamic single-track vehicle or a Jolt vehicle, and the run is
written as the same trace bundle :func:`grl_snam.tools.fog_record.record` writes, so
``grl-snam fog capture`` and ``play`` replay it as they replay any trace.

The native world senses, believes and drives on its own; :func:`record_on_plant` keeps its world
honest to the story:

- **events**: ``add_rect`` and ``remove_rect`` change its truth (``sim_world::set_truth``), and
  on a physics plant ``add_rect`` raises a static box there too; ``unit_at`` stamps a decaying
  mark (``add_obstacle``);
- **movers**: each tick their footprints join the truth, and on a physics plant they are
  kinematic boxes moved to where they are;
- **a moving goal**: the agent is retargeted to where it is each tick.

What the native world does differently from the Python scenario: it steers by its own carrot
with no route spine (the trace's routes are empty), and a mover it sees is believed as a wall
until it sees those cells free again (the Python scenario demotes a seen mover to the decaying
layer). The clearance rows are each tick's distance to the truth, less the body radius.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grl_snam.clock import WorldClock


def record_on_plant(
    story,
    out: Path,
    plant,
    *,
    model=None,
    seed: int = 0,
    max_steps: int | None = None,
    progress=None,
    truth_occ=None,
    prior_occ=None,
    ground_mu: float = 0.9,
) -> Path:
    """Run ``story`` with its agent on ``plant`` (``"kinematic"``, ``"dynamic[:preset]"``,
    ``"jolt[:preset]"``; RoboRacers by default) on ground of grip ``ground_mu``, and write
    ``trace.npz`` + ``trace.json`` to ``out``."""
    import torch

    import grl_snam.sdf_nav as sdf_nav
    from grl_snam import nav_native
    from grl_snam.plant_nav import PlantNavigator
    from grl_snam.tools.fog_record import _ROW_FIELDS, _write_trace

    if not nav_native.HAS_FOG:
        raise RuntimeError(
            "a fog story on a plant needs pycvc's sim_world fog bindings "
            "(nav_sim_world_set_truth: libcvc's sim_world::set_truth)"
        )
    if story.moving_goal is None and len(story.waypoints) > 1:
        raise NotImplementedError("a plant drives one waypoint; the stories have one")
    if model is None:
        torch.manual_seed(seed)
        model = sdf_nav.CoefMLP()
        model.eval()
    truth = (story.truth_grid() if truth_occ is None else np.asarray(truth_occ)).astype(bool)
    prior = story.prior_grid() if prior_occ is None else prior_occ
    if prior is None:
        prior = np.zeros_like(truth)
    meta = story.meta()
    S = float(story.scale)
    phi, nxg, nyg = sdf_nav.build_sdf(truth, story.bounds, S)
    field = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], S)
    nav = PlantNavigator(
        field, model, meta, truth, plant, reach_tol=story.reach_tol, ground_mu=ground_mu,
        prior=prior, sense_every=story.sense_every, unknown=story.unknown,
        ttl_s=story.unit_ttl_s, sensor=dict(story.sensor),
    )  # fmt: skip
    goal0 = story.moving_goal.position_at(0.0) if story.moving_goal else story.waypoints[0]
    nav.start(story.start, goal0)
    world, pl = nav.world, nav.plant
    ny, nx = truth.shape
    mnx, mny, mxx, mxy = story.bounds
    cw = (mxx - mnx) / (nx - 1)

    def cell_of(x, y):
        c = int(round((x - mnx) / (mxx - mnx) * (nx - 1)))
        r = int(round((y - mny) / (mxy - mny) * (ny - 1)))
        return min(max(r, 0), ny - 1), min(max(c, 0), nx - 1)

    def rect_box(rect):
        r0, r1, c0, c1 = rect  # cells r0..r1-1, c0..c1-1: half a cell beyond the outer centres
        return (mnx + (c0 - 0.5) * cw, mny + (r0 - 0.5) * cw, mnx + (c1 - 0.5) * cw,
                mny + (r1 - 0.5) * cw)  # fmt: skip

    def stamp_movers(t):
        now = truth.copy()
        for mv in story.movers:
            r, c = cell_of(*mv.position_at(t))
            rad = max(1, int(round(mv.half_m / cw)))
            now[max(0, r - rad) : r + rad + 1, max(0, c - rad) : c + rad + 1] = True
        return now

    mover_boxes = {}
    if pl is not None:
        for mv in story.movers:
            x, y = mv.position_at(0.0)
            mover_boxes[mv.key] = pl.add_box(
                x - mv.half_m, y - mv.half_m, x + mv.half_m, y + mv.half_m, height_m=2.0,
                motion="kinematic",
            )  # fmt: skip

    clock = WorldClock(fixed_dt=story.dt, mode="replay")
    rows: dict[str, list] = {k: [] for k in _ROW_FIELDS}
    snap_ticks, snap_occ, snap_dyn, routes = [], [], [], []
    truth_ticks, truth_snaps, fov_ticks, fov_vis, fov_seen = [], [], [], [], []

    def snapshot(tick, t):
        st = world.dynamic_stamps(0)
        dyn = np.isfinite(st) & (t - st <= story.unit_ttl_s)
        snap_ticks.append(tick)
        snap_occ.append(np.packbits((world.belief_occ(0) & ~dyn).ravel()))
        snap_dyn.append(np.packbits(dyn.ravel()))
        routes.append(np.empty((0, 2), np.float32))

    events = sorted(story.events, key=lambda e: e.step)
    next_ev = 0
    truth_now = truth.copy()
    snapshot(0, 0.0)
    truth_ticks.append(0)
    truth_snaps.append(np.packbits(truth_now.ravel()))
    last_belief = world.belief_occ(0)
    edt = None
    limit = max_steps or story.max_steps
    reached = False
    for step_i in range(limit):
        t = step_i * story.dt
        touched = False
        while next_ev < len(events) and events[next_ev].step <= step_i:
            ev = events[next_ev]
            next_ev += 1
            if ev.kind in ("add_rect", "remove_rect"):
                r0, r1, c0, c1 = ev.args
                truth[r0:r1, c0:c1] = ev.kind == "add_rect"
                touched = True
                if pl is not None and ev.kind == "add_rect":
                    pl.add_box(*rect_box(ev.args))
                elif pl is not None and pl.world is not None:
                    raise NotImplementedError("a physics plant cannot take a wall down yet")
            elif ev.kind == "unit_at":
                r, c = ev.args
                world.add_obstacle(r - 1, r + 1, c - 1, c + 1)
            else:
                raise ValueError(f"unknown event kind {ev.kind!r}")
        if story.movers or touched:
            truth_now = stamp_movers(t)
            world.set_truth(truth_now)
            edt = None
            if pl is not None:
                for mv in story.movers:
                    (x, y), (x2, y2) = mv.position_at(t), mv.position_at(t + story.dt)
                    vx, vy = (x2 - x) / story.dt, (y2 - y) / story.dt
                    pl.move_box(mover_boxes.get(mv.key), x, y, vx, vy)
        if story.moving_goal is not None:
            nav.track_goal(story.moving_goal.position_at(t))

        m = nav.step()
        clock.step_once()
        tick = clock.tick()
        if edt is None:
            edt = np.sqrt(nav_native.edt2(truth_now)) * cw
        r, c = cell_of(m.x, m.y)
        belief = world.belief_occ(0)
        rebuilt = not np.array_equal(belief, last_belief)
        last_belief = belief
        sensed = step_i % story.sense_every == 0
        rows["tick"].append(tick)
        rows["t"].append(clock.t())
        rows["x"].append(m.x)
        rows["y"].append(m.y)
        rows["heading_rad"].append(m.heading_rad)
        rows["speed_mps"].append(m.speed_mps)
        rows["clearance_m"].append(float(edt[r, c]) - story.rr / S)
        rows["goal_dist_m"].append(m.goal_dist_m)
        rows["goal_index"].append(0)
        rows["goal_x"].append(float(m.goal_x))
        rows["goal_y"].append(float(m.goal_y))
        rows["belief_version"].append(len(snap_ticks) - 1 + int(rebuilt))
        rows["rebuilt"].append(bool(rebuilt))
        rows["sensed"].append(bool(sensed))
        rows["truth_penetration"].append(bool(truth_now[r, c]))
        if rebuilt:
            snapshot(tick, clock.t())
        if sensed:
            fov_ticks.append(tick)
            fov_vis.append(np.packbits(world.last_visible(0).ravel()))
            fov_seen.append(np.packbits(world.ever_seen(0).ravel()))
            truth_ticks.append(tick)
            truth_snaps.append(np.packbits(truth_now.ravel()))
        if progress and tick % 50 == 0:
            progress(tick, limit)
        reached = reached or m.reached
        if m.reached and story.moving_goal is None:
            break

    extra = {
        "nav": "sim_world carrot (no route spine)",
        "plant": nav.plant_spec,
        "ground_mu": float(ground_mu),
        "native_nav_stats": nav.nav_stats(),
    }
    return _write_trace(
        out, story, seed, rows, snap_ticks, snap_occ, snap_dyn, routes, truth_ticks, truth_snaps,
        fov_ticks, fov_vis, fov_seen, waypoints_reached=int(reached), reached=reached,
        extra=extra,
    )  # fmt: skip
