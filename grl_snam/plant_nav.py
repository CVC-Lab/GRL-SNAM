"""The trained policy on a vehicle plant: one agent on libcvc's ``sim_world``, driving nav's
kinematic bicycle, the dynamic single-track vehicle or a Jolt vehicle.

:class:`PlantNavigator` has :class:`~grl_snam.nav.SdfNavigator`'s drive interface (``start``,
``set_goal``, ``step``, ``drive_to_goal``) and fills the same
:class:`~grl_snam.metrics.NavMetrics`, plus ``plant``: the vehicle's sideslip, tire use,
lateral g and energy books (:meth:`grl_snam.nav_native.JoltPlant.telemetry`). The three plants
share one navigator (the C++ sim_world with the policy exported to ``.cvcnav``), so runs on
them differ only in the vehicle. The map is known: the belief starts as the truth and is not
sensed again.

A physics vehicle cannot be rewound, so unlike ``SdfNavigator`` a multi-goal drive continues
each leg from wherever the last one ended, and ``drive_to_goal`` returns no state to restore.

Needs a pycvc with libcvc's sim_world and plant bindings (:data:`grl_snam.nav_native.HAS_PLANT`;
the dynamic plant also :data:`~grl_snam.nav_native.HAS_DYN_PLANT`).
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import torch

from . import nav_native as _native
from .metrics import NavMetrics
from .nav import SdfNavigator

#: The plants: nav's ``kinematic`` bicycle, libcvc's ``dynamic`` single track, and ``jolt``
#: (each but the first optionally ``:<preset>``, e.g. ``jolt:humvee``).
PLANTS = ("kinematic", "dynamic", "jolt")


def plant_label(plant: str, vehicle: str | None = None) -> str:
    """The plant spec :func:`grl_snam.nav_native.make_plant` takes: ``plant`` with
    ``vehicle`` as its preset (``"jolt", "humvee"`` -> ``"jolt:humvee"``)."""
    if plant == "kinematic" or not vehicle or ":" in plant:
        return plant
    return f"{plant}:{vehicle}"


class PlantNavigator:
    """Drive the policy ``model`` (a ``CoefMLP``) across ``field`` on ``plant``.

    ``occ`` is the scene's (rows, cols) occupancy over the field's bounds, row 0 at min y, cell
    centres at the grid points (the field's own grid): the sim_world's truth and prior, and on a
    physics plant its static boxes (``obstacle_height_m`` tall). ``plant`` is ``"kinematic"``,
    ``"dynamic[:preset]"`` or ``"jolt[:preset]"``; ``ground_mu`` is the ground's grip.
    ``reach_tol`` is the arrival radius in normalized units, as SdfNavigator's, or ``reach_m``
    in metres (a trained metric's radius can be hundreds of metres, more than a vehicle needs).
    ``routes`` drives libcvc's vehicle routes (planned for the vehicle's turning circle) instead
    of the navigator's carrot. ``vehicle`` overrides SdfNavigator's ``VEHICLE_DEFAULTS``; on a
    plant the wheelbase is the vehicle's and the limits never exceed its own. ``known_map``
    False starts the belief empty (fog of war): the agent maps the buildings with its sensor
    (``sensor``: range_m, n_rays, fov_rad) as it drives, while the plant's boxes are there
    from the start. ``weights`` (a ``.cvcnav``, e.g. one trained through Jolt) is the policy
    the world drives instead of ``model``, which then only frames the run (``meta``)."""

    def __init__(
        self,
        field,
        model,
        meta,
        occ,
        plant: str = "jolt",
        *,
        reach_tol: float = 0.8,
        reach_m: float | None = None,
        ground_mu: float = 0.9,
        routes: bool = False,
        vehicle: dict | None = None,
        sensor: dict | None = None,
        obstacle_height_m: float = 6.0,
        known_map: bool = True,
        weights: str | None = None,
    ):
        if not _native.HAS_SIM_WORLD or not _native.HAS_PLANT:
            raise RuntimeError(
                "PlantNavigator needs a pycvc with libcvc's sim_world and plant bindings "
                "(nav_sim_world_create, nav_sim_world_set_plant)"
            )
        self.field = field
        self.model = model
        self.meta = meta
        self.S = float(meta["scale"])
        self.cx, self.cy = (float(c) for c in meta["center"])
        self.bounds = (field.mnx, field.mny, field.mxx, field.mxy)
        self.occ = np.ascontiguousarray(np.asarray(occ, bool), np.uint8)
        self.plant_spec = plant
        self.ground_mu = float(ground_mu)
        self.reach_tol = float(reach_tol) if reach_m is None else float(reach_m) * self.S
        self.routes = bool(routes)
        self.kw = dict(
            rr=float(meta["rr"]),
            d_hat=float(meta["d_hat"]),
            dt=float(meta["dt"]),
            vmax=float(meta["vmax"]),
        )
        self.dt = float(meta["dt"])
        self.nsub = int(meta.get("nsub", 1))
        self._veh = {**SdfNavigator.VEHICLE_DEFAULTS, **(vehicle or {})}
        self._sensor = {"range_m": 60.0, "n_rays": 240, "fov_rad": 2 * math.pi, **(sensor or {})}
        self._obstacle_h = float(obstacle_height_m)
        self.known_map = bool(known_map)
        self.weights = weights
        self.plant = None
        self.world = None
        self._routes = None
        self._weights = None
        self._telemetry = False
        self.goal_index = 0
        self.step_i = 0
        self._gw = (0.0, 0.0)

    # ── world <-> normalized ────────────────────────────────────────────────
    def w2n(self, p):
        return np.array([(p[0] - self.cx) * self.S, (p[1] - self.cy) * self.S], np.float32)

    def pos_world(self):
        pos = self.world.snapshot()[0]
        return np.array([pos[0, 0], pos[0, 1]], np.float32)

    def _phi(self, pw):
        on = torch.from_numpy(self.w2n(pw)).unsqueeze(0).float()
        return float(self.field.sample(on)[0][0])

    # ── setup ───────────────────────────────────────────────────────────────
    def start(self, start_world, goal_world):
        """Build the world with the agent at ``start_world`` (at rest) heading for
        ``goal_world``, and put it on the plant."""
        from .tools.coef_export import write_coef_mlp

        if self._phi(start_world) <= self.kw["rr"] and self.plant_spec != "kinematic":
            raise ValueError(
                f"start {tuple(np.round(start_world, 1))} is inside or against an obstacle; a "
                "physics vehicle cannot be added there"
            )
        if self.weights:
            path = str(self.weights)
        else:
            fd, path = tempfile.mkstemp(suffix=".cvcnav")
            os.close(fd)
            write_coef_mlp(self.model, path)
            self._weights = path  # ours to delete
        weights_path = path
        self.plant = _native.make_plant(
            self.plant_spec,
            bounds=self.bounds,
            obstacles=self.occ.astype(bool),
            ground_mu=self.ground_mu,
            obstacle_height_m=self._obstacle_h,
        )
        v, k = dict(self._veh), self.kw
        if self.plant is not None:
            lim = self.plant.nav_limits(self.S)
            v["a_max"] = min(float(v["a_max"]), lim["a_max"])
            v["a_lat_max"] = min(float(v["a_lat_max"]), lim["a_lat_max"])
            v["L"] = _native.wheelbase_m(self.plant.descs[0]) * self.S
        rows, cols = self.occ.shape
        mnx, mny, mxx, mxy = self.bounds
        o = self.w2n(start_world)[None].astype(np.float32)
        g = self.w2n(goal_world)[None].astype(np.float32)
        h = _native._pycvc.nav_sim_world_create(
            self.occ,
            self.occ if self.known_map else np.zeros_like(self.occ),
            weights_path,
            o,
            g,
            np.ones((1, 3), np.float32),
            rows,
            cols,
            mnx,
            mny,
            mxx,
            mxy,
            self.cx,
            self.cy,
            self.S,
            float(self._sensor["range_m"]),
            int(self._sensor["n_rays"]),
            float(self._sensor["fov_rad"]),
            k["rr"],
            k["d_hat"],
            k["dt"],
            k["vmax"],
            v["L"],
            v["delta_max"],
            v["a_max"],
            v["a_lat_max"],
            v["k_steer"],
            self.nsub,
            int(bool(v["allow_reverse"])),
            self.reach_tol,
            4,
            1 if self.known_map else 0,  # a known map is the prior, never sensed again
            2.2,
            -1.4,
            8.0,
            1,
            0.5,
            0.15,
            4.0,
        )
        self.world = _native.NativeSimWorld(h, 1)
        self.world.set_plant(self.plant)
        if _native.HAS_SIM_WORLD_NAVSTATS:
            # The world's own collector: it counts the plant's building and vehicle contacts.
            self.world.begin_nav_stats(scene_id="capture", contact_r=0.0)
        if hasattr(_native._pycvc, "nav_sim_world_set_capture_drive_telemetry"):
            _native._pycvc.nav_sim_world_set_capture_drive_telemetry(h, 1)
            self._telemetry = True
        self.step_i = 0
        self.set_goal(goal_world)
        return self

    def set_goal(self, goal_world, *, goal_index: int | None = None):
        """Retarget the agent; it carries on from where it is, as it is."""
        self._gw = (float(goal_world[0]), float(goal_world[1]))
        if goal_index is not None:
            self.goal_index = goal_index
        gn = self.w2n(self._gw)
        self.world.retarget(0, float(gn[0]), float(gn[1]))
        pos = self.pos_world()
        self._init = max(float(np.hypot(self._gw[0] - pos[0], self._gw[1] - pos[1])), 1e-6)
        self._best = self._init
        self._stall = 0
        self._reached = False
        if self.routes:
            if self.plant is None:
                raise ValueError("vehicle routes need a physics plant, not nav's bicycle")
            if self._routes is None:
                self._routes = _native.VehicleRoutes(self.world, self.bounds, [0])
            head = self.world.snapshot()[1]
            self._routes.start(pos[None], np.asarray(head), {0: self._gw}, t=self.step_i * self.dt)

    # ── stepping ────────────────────────────────────────────────────────────
    def step(self) -> NavMetrics:
        """Advance one navigation tick; return the metrics after it."""
        if self._routes is not None:
            pos0, head0 = self.world.snapshot()[:2]
            self._routes.apply(self.step_i * self.dt, pos0, head0)
        self.world.step()
        self.step_i += 1
        pos, head, speed, mode, reached = self.world.snapshot()
        x, y = float(pos[0, 0]), float(pos[0, 1])
        dg = float(np.hypot(self._gw[0] - x, self._gw[1] - y))
        if dg < self._best - 1e-3:
            self._best, self._stall = dg, 0
        else:
            self._stall += 1
        arrived = (
            bool(self._routes.done()[0])
            if self._routes is not None
            else bool(np.asarray(reached)[0])
        )
        self._reached = self._reached or arrived or dg * self.S < self.reach_tol
        phi = self._phi((x, y))
        al = be = ga = 0.0
        if self._telemetry:
            t = _native._pycvc.nav_sim_world_drive_telemetry(self.world._h)
            if t is not None:
                al, be, ga = (float(np.asarray(t[c])[0]) for c in ("alpha", "beta", "gamma"))
        m = NavMetrics(
            step=self.step_i,
            x=x,
            y=y,
            goal_x=self._gw[0],
            goal_y=self._gw[1],
            goal_dist_m=dg,
            clearance_m=(phi - self.kw["rr"]) / self.S,
            speed_mps=float(np.asarray(speed)[0]),
            alpha=al,
            beta=be,
            gamma=ga,
            mode="wall" if int(np.asarray(mode)[0]) == 1 else "seek",
            stall=self._stall,
            inside_building=phi < 0.0,
            progress=max(0.0, 1.0 - dg / self._init),
            goal_index=self.goal_index,
            reached=self._reached,
            heading_rad=float(np.asarray(head)[0]),
        )
        if self.plant is not None:
            m.plant = self.plant.telemetry(0)
            m.plant["name"] = self.plant_spec
            m.plant["mu"] = self.ground_mu
        else:
            m.plant = {"name": "kinematic"}
        return m

    def drive_to_goal(self, max_steps: int = 1300, *, stop_at_reach: bool = True, stall=340):
        """Step toward the current goal until it is reached, ``stall`` ticks pass without
        getting closer, or ``max_steps``. Returns ``(metrics, closest_idx, None, None)``:
        SdfNavigator's form, with no state to restore (a vehicle is not rewound)."""
        out = []
        best, best_i = 1e30, 0
        for _ in range(int(max_steps)):
            m = self.step()
            out.append(m)
            if m.goal_dist_m < best - 1e-3:
                best, best_i = m.goal_dist_m, len(out) - 1
            if stop_at_reach and m.reached:
                best_i = len(out) - 1
                break
            if m.stall > stall:
                break
        return out, best_i, None, None

    def nav_stats(self) -> dict | None:
        """The run's episode record from the sim_world's collector (its one vehicle's
        per_vehicle entry: goals reached, path, penetration, building and vehicle contacts,
        ...), or None when this pycvc lacks the collector. Ends the collection."""
        if not _native.HAS_SIM_WORLD_NAVSTATS or self.world is None:
            return None
        return self.world.nav_stats()["per_vehicle"][0]

    def __del__(self):
        p = getattr(self, "_weights", None)
        if p:
            try:
                os.unlink(p)
            except OSError:
                pass
