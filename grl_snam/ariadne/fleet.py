"""A live squad of Humvees, on the plant and in the world you pick, in an Ariadne document.

    python -m grl_snam.ariadne fleet                         # Austin; GRL-SNAM's torch squad
    python -m grl_snam.ariadne fleet --plant humvee          # Jolt Humvees, in libcvc's sim_world
    python -m grl_snam.ariadne fleet --plant dynamic:humvee  # libcvc's dynamic single-track Humvee
    python -m grl_snam.ariadne fleet --plant kinematic       # nav's bicycle, in libcvc's sim_world
    python -m grl_snam.ariadne fleet --world generated --set world.seed=4   # a generated world
    python -m grl_snam.ariadne fleet --uri DIR               # another city (a bundle, a package)
    python -m grl_snam.ariadne fleet --headless --frames 600 --log run.npz --min-arrivals 2

Unlike the recorded players this runs the squad live: ``grl_snam.squad.Squad``, each agent with
its own belief, starting blind, driving to a goal across the world. The document's settings are
the ones libcvc's multi-car template uses (pycvc_gl.cars; libcvc's docs/CARS.md), so they are
set the same four ways: in the document's ``init:``, on the sliders, with ``--set`` /
``--params``, or from Python (``host.set``).

- **Plant**, ``nav_plant.vehicle`` (the Plant combo; ``--plant``):
  - ``torch`` (the default): GRL-SNAM's Python squad, one FogScenario per agent, the path that
    trains;
  - any plant libcvc's ``plant_host`` names: ``kinematic`` (nav's bicycle), ``humvee`` and
    ``roboracer`` (Jolt), ``dynamic:humvee`` (the dynamic single-track model),
    ``dynamic4w:humvee`` (four wheels with suspension). The squad then runs in libcvc's
    sim_world (``Squad(plant=...)``), each agent's belief its own plane.

  Between libcvc's plants the switch is live: the agents keep their poses and start at rest. To
  or from ``torch`` the squad is rebuilt. On a physics plant the agents drive vehicle routes
  planned on their own beliefs (``Squad.use_vehicle_routes``). The old ``--plant`` names still
  work (``jolt:humvee`` is ``humvee``).
- **Navigator settings.** On libcvc's sim_world, every ``nav.*``, ``nav.vehicle.*`` and
  ``nav_plant.*`` key applies, as libcvc reads it. On the torch squad, the keys that mean the same
  thing on GRL-SNAM's navigator are mapped onto it (``grl_snam.ariadne._cars.TORCH_KEYS``; the
  README beside this file has the table): ``nav.vehicle.vmax``, ``rr``, ``d_hat``, ``a_max``,
  ``a_lat_max``, ``wheelbase``, ``delta_max``, ``k_steer``, ``allow_reverse``, and
  ``nav.reach_tol``, plus the scenario's ``nav.range_m`` and ``nav.sense_every``. The rest
  (``nav_plant.*``, the risk governor, ``nav.yield.*``, ...) are libcvc's alone.
- **World**, ``run.world`` (``--world``): the document's ``data:`` block (Austin from cvcpkg.org;
  ``--uri`` or the URI field for another package or a local bundle; ``--bundle DIR`` is
  ``--uri DIR``), libcvc's terrain generator (``world.*``), or a flat walled square. The squad,
  the physics and the scene all take the same world.
- **Panels**: a row per agent; the chosen agent's (``view.car``) coefficients (the policy's alpha,
  beta, gamma and its clearance); the pictures around it (the signed distance with its planned
  route, and its belief: fog of war, grey where it has not looked); the settings; the generator.
- **Headless**: ``--headless --log run.npz`` writes what libcvc's template writes (every tick's
  state and coefficients, the rasters every ``--raster-every`` ticks, every setting in ``meta``).
  The exit code is 0, 1 when the run failed, 3 below ``--min-arrivals`` (agents that arrived).

The policy is the published ``grl-snam-weights`` checkpoint when installed (else the untrained
seed). The navigator is sized for the story's bodies, not a 4.7 m Humvee (NATIVE_CVC_NAV.md), so
on Jolt Humvees it drives worse than on its own bicycle; libcvc's own
``examples/ariadne/python/nav.py`` follows routes planned for the vehicle's turning radius.
Needs pycvc-gl 3.7.12+ (libcvc's pycvc_gl.cars); with an older one it says what to install.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from grl_snam.ariadne import _cars, _host

#: The Plant combo's choices: GRL-SNAM's torch squad, or a plant libcvc's plant_host names.
PLANTS = ("torch", "kinematic", "humvee", "roboracer", "dynamic:humvee", "dynamic4w:humvee")
#: The first fleet's --plant names, as nav_plant.vehicle values.
OLD_PLANTS = {"jolt:humvee": "humvee", "jolt:roboracer": "roboracer", "jolt": "roboracer",
              "auto": "torch"}  # fmt: skip
#: An agent within this of its goal has arrived (m).
ARRIVED_M = 10.0
#: The grid's cell (m): 1200 m of Austin in 384 cells, as the first fleet had it.
CELL_M = 1200.0 / 383.0
#: The log keeps this many points of the chosen agent's planned route with each raster.
ROUTE_POINTS = 64
#: The scenario settings a torch squad takes besides its navigators': key -> FogScenario field.
SCENARIO_KEYS = {"nav.range_m": "range_m", "nav.sense_every": "sense_every"}


def _args(p):
    p.add_argument("--plant", default=None,
                   help="nav_plant.vehicle: torch (GRL-SNAM's squad), kinematic, humvee, "
                        "roboracer, dynamic:humvee, dynamic4w:humvee (old: jolt:humvee)")  # fmt: skip
    p.add_argument("--agents", type=int, default=None, help="how many agents (fleet.agents)")
    p.add_argument("--bundle", default=None, help="a local city bundle: the same as --uri DIR")
    p.add_argument("--seed", type=int, default=0, help="the starts and goals")
    if _cars.HAS_CARS:
        _cars.C.add_args(p)


def load_model():
    """The published CoefMLP (grl-snam-weights), or None (the untrained seed) without it."""
    import os

    from grl_snam.demos._common import default_nav_weights_pt

    path = default_nav_weights_pt()
    if not path or not os.path.exists(path):
        return None
    import torch

    import grl_snam.sdf_nav as sdf_nav

    ck = torch.load(path, map_location="cpu")
    model = sdf_nav.CoefMLP()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def plant_name(name) -> str:
    """A ``--plant`` (or nav_plant.vehicle) value as a nav_plant.vehicle value."""
    name = str(name or "torch").strip() or "torch"
    return OLD_PLANTS.get(name, name)


class Fleet:
    """A squad of ``n`` agents on ``world`` (a ``pycvc_gl.cars`` World) with random far-apart
    starts and goals, on ``vehicle``: ``torch`` (GRL-SNAM's Python squad) or a plant libcvc's
    plant_host names (the squad in libcvc's sim_world, the plant attached by plant_host)."""

    def __init__(self, world, story, model, n, vehicle, rng):
        from grl_snam.squad import AgentSpec, Squad
        from grl_snam.tools.finale_record import COLORS, _free_snapper

        occ, bounds = world.occ.astype(bool), world.bounds
        snap = _free_snapper(occ, bounds)
        span = bounds[2] - bounds[0]
        half, far = 0.85 * span / 2, min(300.0, 0.5 * span)
        agents = []
        for i in range(n):
            for _ in range(50):
                s = snap(*rng.uniform(-half, half, 2))
                g = snap(*rng.uniform(-half, half, 2))
                if np.hypot(g[0] - s[0], g[1] - s[1]) > far and all(
                    np.hypot(s[0] - a.start[0], s[1] - a.start[1]) > 15.0 for a in agents
                ):
                    break
            agents.append(AgentSpec(f"v{i}", s, g, COLORS[i % len(COLORS)]))
        self.world, self.agents, self.story = world, agents, story
        self.vehicle = vehicle
        self.native = vehicle != "torch"
        self.squad = Squad(
            story,
            agents,
            model,
            truth_occ=occ,
            prior_occ=None,
            plant="kinematic" if self.native else None,
            vehicle_routes=False if self.native else None,
        )
        self.plant = None
        self.routes = False
        if self.native:
            import pycvc

            h = self.squad.native._h
            self.plant = pycvc.plant_host(fixed_dt=0.01)
            if world.geometry is not None:
                self.plant.set_geometry(world.geometry)
            self.plant.use(h, vehicle)
            pycvc.nav_sim_world_set_capture_drive_telemetry(h, 1)
            self.routes = self.squad.use_vehicle_routes(True)
        self.recs = {}
        self.tracks = {a.key: [] for a in agents}
        self.ticks = 0

    @property
    def navs(self):
        """The torch squad's navigators, one per agent."""
        return [sc.nav for sc in self.squad.scenarios.values()]

    def driving(self) -> str:
        return self.plant.vehicle() if self.plant is not None else "torch"

    # ── settings ──
    def pull(self, host):
        """Read the settings (between ticks). On libcvc's sim_world: its nav.* keys, then the
        plant (a new nav_plant.vehicle switches it live, and the routes follow it)."""
        if not self.native:
            return
        import pycvc

        h = self.squad.native._h
        pycvc.nav_sim_world_pull(h, host.app, host.prefix)
        before = (self.plant.vehicle(), self.plant.kind())
        self.plant.pull(host.app, host.prefix, h)
        if (self.plant.vehicle(), self.plant.kind()) != before:
            self.vehicle = self.plant.vehicle()
            self.routes = self.squad.use_vehicle_routes(True)

    def publish(self, host):
        if self.native:
            import pycvc

            pycvc.nav_sim_world_publish(self.squad.native._h, host.app, host.prefix)

    # ── stepping ──
    def step(self, n):
        for _ in range(n):
            if self.squad.done:
                return
            self.recs = self.squad.step()
            self.ticks += 1
            if self.ticks % 4 == 0:
                for k, r in self.recs.items():
                    self.tracks[k] = (self.tracks[k] + [(r.x, r.y)])[-400:]

    # ── what the panels and the log read ──
    def coefficients(self) -> dict:
        """Each agent's coefficients this tick, name -> (n,) array: the policy's alpha, beta,
        gamma and the clearance (m) it measured (NaN where an agent has none this tick, as a
        route agent on a physics plant, which the follower drives)."""
        n = len(self.agents)
        if self.native:
            import pycvc

            tel = pycvc.nav_sim_world_drive_telemetry(self.squad.native._h) or {}
            return {k: np.asarray(tel[k], float).reshape(-1)[:n]
                    for k in ("alpha", "beta", "gamma", "clearance") if k in tel}  # fmt: skip
        out = {k: np.full(n, np.nan) for k in ("alpha", "beta", "gamma", "clearance")}
        for i, nav in enumerate(self.navs):
            m = nav.last_metrics
            if m is not None:
                out["alpha"][i], out["beta"][i], out["gamma"][i] = m.alpha, m.beta, m.gamma
                out["clearance"][i] = m.clearance_m
        return out

    def belief(self, i):
        """Agent ``i``'s belief over the world's grid as the probability each cell is blocked:
        the torch squad's log-odds, or on libcvc's sim_world its belief plane where the agent has
        looked (0 free, 1 blocked) and 0.5 where it has not."""
        if self.native:
            import pycvc

            h = self.squad.native._h
            plane = int(np.asarray(pycvc.nav_sim_world_agent_planes(h))[i])
            occ = np.asarray(pycvc.nav_sim_world_belief_occ(h, plane)) != 0
            seen = self.squad.native.ever_seen(plane)
            return np.where(seen, occ.astype(np.float32), np.float32(0.5))
        sc = list(self.squad.scenarios.values())[i]
        return (1.0 / (1.0 + np.exp(-sc.belief.logodds))).astype(np.float32)

    def route(self, i):
        """Agent ``i``'s planned route ahead ((m, 2) world points), or None: the torch
        scenario's route spine."""
        if self.native:
            return None
        sc = list(self.squad.scenarios.values())[i]
        return None if not sc.route else np.asarray(sc.route, float)[:, :2]

    def state(self, coeffs=None):
        """The squad now as a ``pycvc_gl.cars`` CarsState (Humvee-sized cars): what the log, the
        pictures and the panels read."""
        C = _cars.C
        xs, ys, hs, vs, goals = [], [], [], [], []
        for a in self.agents:
            r = self.recs.get(a.key)
            if r is None:
                x, y = self.squad.position_of(a.key)
                h, v = 0.0, 0.0
            else:
                x, y, h, v = r.x, r.y, r.heading_rad, r.speed_mps
            xs.append(x)
            ys.append(y)
            hs.append(h)
            vs.append(v)
            goals.append(self.squad.goal_of(a.key))
        length, width, wheelbase = _cars.humvee_sizes()
        return C.state_from(
            self.world, xs, ys, hs, vs, goals, t=self.ticks * self.story.dt, tick=self.ticks,
            dt=self.story.dt, reach=ARRIVED_M, driving=self.driving(), length=length,
            width=width, wheelbase=wheelbase, nav=coeffs,
        )  # fmt: skip


class ScenarioSettings:
    """``nav.range_m`` and ``nav.sense_every`` on a torch squad's scenarios (the sensor's reach
    and the ticks between looks), seeded from the first one and applied when they change."""

    def __init__(self, host, scenarios):
        self.host, self.scs, self._last = host, list(scenarios), {}

    def seed(self):
        sc = self.scs[0]
        for key, field in SCENARIO_KEYS.items():
            if self.host.get(key) in (None, ""):
                v = sc.sensor.get("range_m", 60.0) if field == "range_m" else sc.sense_every
                self.host.set(key, f"{float(v):.6g}")
        self.apply()

    def apply(self):
        for key, field in SCENARIO_KEYS.items():
            v = self.host.get(key)
            if v in (None, "") or self._last.get(key) == v:
                continue
            try:
                x = float(v)
            except ValueError:
                continue
            for sc in self.scs:
                if field == "range_m":
                    sc.sensor = {**sc.sensor, "range_m": x}
                else:
                    sc.sense_every = max(1, int(round(x)))
            self._last[key] = v


def main(argv=None) -> int:
    if not _cars.HAS_CARS:
        print(f"grl-snam ariadne fleet: needs {_cars.INSTALL}", file=sys.stderr)
        return 2
    args = _host.parse(argv, _args)

    from grl_snam.tools.austin import austin_story

    C = _cars.C
    bundle = args.bundle or os.environ.get("GRL_SNAM_SCENE_BUNDLE") or None
    host = _host.open_host("fleet.ari", "fleet", args, load=False)
    _host.load_document(host, "fleet.ari", worlds=True)
    flags = {"fleet.agents": args.agents}
    if bundle and not args.uri:  # a local city, loaded (from there, no download) when it is used
        flags.update({"run.uri": str(bundle), "run.data_path": ""})
    if args.plant is not None:
        flags["nav_plant.vehicle"] = plant_name(args.plant)
    C.apply_settings(host, args, flags)
    model = load_model()
    drawn = host.mode != "headless"
    log = C.CarsLog() if args.log else None
    st = {"fleet": None, "nodes": [], "world": None, "torch": None, "scen": None, "coef": {},
          "speed": [], "error": None, "camera": None, "model": None}  # fmt: skip
    doc = str(_host.DOCS / "fleet.ari")

    def build(seed=None):
        kind = host.get("run.world", "data") or "data"
        try:
            world = C.make_world(host, kind, cell=CELL_M)
        except Exception as e:  # noqa: BLE001 - say why, and fall back to a world that loads
            host.set("stats.note", f"{kind} world: {e}")
            print(f"fleet: the {kind} world failed ({e}); flat ground instead", file=sys.stderr)
            world = C.flat_world(_cars.humvee_sizes()[0])
        n = max(1, host.get_int("fleet.agents", 6))
        vehicle = plant_name(host.get("nav_plant.vehicle", "torch"))
        story = austin_story(world.occ.astype(bool), world.bounds, key="fleet", fog=True)[0]
        rng = np.random.default_rng(args.seed if seed is None else seed)
        fleet = Fleet(world, story, model, n, vehicle, rng)
        st.update(fleet=fleet, world=world, coef={}, speed=[])
        if fleet.native:
            fleet.pull(host)  # the tree's settings apply (unset ones are seeded)
            fleet.publish(host)
            st["torch"] = st["scen"] = None
        else:
            st["torch"] = _cars.TorchSettings(host, fleet.navs)
            st["torch"].seed()
            st["scen"] = ScenarioSettings(host, fleet.squad.scenarios.values())
            st["scen"].seed()
            if host.get("nav_plant.vehicle") in (None, ""):
                host.set("nav_plant.vehicle", "torch")
        C.show_world(host, world.kind, doc)
        if log is not None:
            log.meta.update(world=world.info, agents=n, seed=args.seed if seed is None else seed,
                            plant=vehicle, policy="grl-snam-weights" if model else "untrained")  # fmt: skip
        if drawn:
            draw(fleet)

    def draw(fleet):
        from grl_snam.ariadne.squad import _pyramid

        for name in st["nodes"]:
            if host.scene.hasGraphics(name):
                host.scene.removeGraphics(name)
        st["nodes"] = []
        C.draw_world(host, st)
        model_mesh = C.humvee_model(host, _host.HUMVEE_LEN_M) or _host.humvee_model(
            _host.bundle_or_none(args.bundle), host=host
        )
        st["model"] = model_mesh
        z = fleet.world.z
        for a in fleet.agents:
            k = a.key
            _host.add_vehicle(host, f"car_{k}", model_mesh)
            host.lab.add_mesh(f"pole_{k}", *_host.humvee_box(0.35, 0.35, 7.0), color=a.color)
            host.lab.add_mesh(f"goal_{k}", *_pyramid(6.0), color=a.color)
            gx, gy = a.goal
            host.scene.getGraphics(f"goal_{k}").setTransform(
                [1, 0, 0, gx, 0, 1, 0, gy, 0, 0, 1, z(gx, gy) + 8.0, 0, 0, 0, 1]
            )
            host.lab.add_path(f"track_{k}", [(0, 0, 0), (0.1, 0, 0)], a.color)
            st["nodes"] += [f"car_{k}", f"pole_{k}", f"goal_{k}", f"track_{k}"]

    def regenerate():
        if (host.get("run.world") or "data") != "generated":
            host.set("run.world", "generated")  # the watch rebuilds
        else:
            build()
        if drawn:
            host.set("world.request", "regenerate")  # the scene's generated_world node redraws it

    def load_world():
        C.load_data_world(host, host.get("run.uri", C.AUSTIN), host.get("run.data_path", ""))
        if host.get("run.world") != "data":
            host.set("run.world", "data")
        else:
            build()

    host.on("restart", build)
    host.on("shuffle", lambda: build(int(np.random.default_rng().integers(1 << 30))))
    host.on("regenerate_world", regenerate)
    host.on("load_world", load_world)

    @host.watch("fleet.agents")
    @host.watch("run.world")
    def _rebuild(_v):
        if st["fleet"] is not None:
            build()

    @host.watch("nav_plant.vehicle")
    def _plant(v):
        fleet = st["fleet"]
        if fleet is not None and (plant_name(v) == "torch") == fleet.native:
            build()  # to or from the torch squad; between libcvc's plants pull() switches live

    if drawn:
        _host.track_camera(host)
    try:
        build()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"fleet: {e}", file=sys.stderr)
        host.close()
        return 1
    camera = _host.CameraSwitch(
        host,
        lambda: host.frame_bounds((*st["world"].bounds[:2], 0.0), (*st["world"].bounds[2:], 60.0)),
    )

    @host.on_frame
    def tick(dt):
        fleet = st["fleet"]
        try:
            if fleet.native:
                fleet.pull(host)
            else:
                st["torch"].apply()
                st["scen"].apply()
            if not host.get_bool("view.paused"):
                for _ in range(max(1, host.get_int("fleet.ticks", 2))):
                    if fleet.squad.done:
                        break
                    fleet.step(1)
                    if log is not None:
                        coeffs = fleet.coefficients()
                        s = fleet.state(coeffs)
                        log.record(s)
                        log.coeffs(coeffs, s.n)
                        every = max(1, args.raster_every)
                        if fleet.ticks % every == 1 or every == 1:
                            log.raster(s.t, **rasters(fleet, s, for_log=True))
            fleet.publish(host)
            panels(fleet)
            if drawn:
                scene(fleet, dt)
        except Exception as e:  # noqa: BLE001 - a failure ends the run with its reason
            import traceback

            traceback.print_exc()
            st["error"] = str(e)
            host.request_close()

    def rasters(fleet, s, for_log=False):
        k = C.chosen(host, s.n)
        size = C.raster_size(host, s.length[k])
        r = C.raster_arrays(fleet.world, s, k, size, belief=fleet.belief(k))
        if for_log:  # the chosen agent's planned route: its next ROUTE_POINTS points, NaN after
            route = fleet.route(k)
            out = np.full((ROUTE_POINTS, 2), np.nan, np.float32)
            if route is not None:
                out[: min(len(route), ROUTE_POINTS)] = route[:ROUTE_POINTS]
            r["route"] = out
        return r

    def panels(fleet):
        coeffs = fleet.coefficients()
        s = fleet.state(coeffs)
        k = C.chosen(host, s.n)
        rows = []
        for i, a in enumerate(fleet.agents):
            r = fleet.recs.get(a.key)
            if r is None:
                continue
            arrived = r.goal_dist_m < ARRIVED_M
            rows.append({"key": a.key, "speed": f"{abs(r.speed_mps):.1f}",
                         "goal": f"{r.goal_dist_m:.0f}",
                         "mode": "arrived" if arrived else r.mode})  # fmt: skip
        host.set_table("agents", rows)
        host.set_many(
            {
                "plant": fleet.driving() + (" (routes)" if fleet.routes else ""),
                "world": fleet.world.kind,
                "time": f"{fleet.ticks * fleet.story.dt:.1f}",
                "arrived": f"{sum(r['mode'] == 'arrived' for r in rows)} / {len(fleet.agents)}",
            },
            prefix="stats",
        )
        if coeffs:
            C.plot_coefficients(host, st["coef"], coeffs, k)
        st["speed"] = (st["speed"] + [float(abs(s.speed[k]))])[-240:]
        host.set_series("plot.speed", st["speed"])
        if host.mode != "headless" and host.frame_count % 4 == 0:
            r = rasters(fleet, s)
            size = C.raster_size(host, s.length[k])
            for name, img in C.raster_pictures(r, s, k, size, path=fleet.route(k)).items():
                host.publish_image(name, img)

    def scene(fleet, dt):
        from pycvc_gl.vehicle import seat_transform

        from grl_snam.fog_scene import _style

        z = fleet.world.z
        sampler = None if fleet.world.height is None else z
        for a in fleet.agents:
            r = fleet.recs.get(a.key)
            if r is None:
                continue
            host.scene.getGraphics(f"car_{a.key}").setTransform(
                seat_transform(r.x, r.y, r.heading_rad, sampler)
            )
            host.scene.getGraphics(f"pole_{a.key}").setTransform(
                [1, 0, 0, r.x, 0, 1, 0, r.y, 0, 0, 1, z(r.x, r.y) + 2.0, 0, 0, 0, 1]
            )
            pts = fleet.tracks[a.key]
            if host.frame_count % 6 == 0 and len(pts) >= 2:
                host.lab.add_path(
                    f"track_{a.key}", [(x, y, z(x, y) + 0.6) for x, y in pts], a.color
                )
                _style(host.scene, f"track_{a.key}", color=a.color, mode="lines", line_width=3)
        if st["model"] is None and host.frame_count % 30 == 0 and C.humvee_model_path(host):
            draw(fleet)  # the Humvee's model has come since: draw the agents with it
        k = C.chosen(host, len(fleet.agents))
        rk = fleet.recs.get(fleet.agents[k].key)
        if rk is not None:
            camera.update((rk.x, rk.y, z(rk.x, rk.y)))

    host.run(fps=30, frames=args.frames or None)
    if args.png and drawn:
        host.snapshot(args.png)
    fleet = st["fleet"]
    arrived = sum(
        1 for a in fleet.agents
        if (r := fleet.recs.get(a.key)) is not None and r.goal_dist_m < ARRIVED_M
    )  # fmt: skip
    if log is not None:
        path = C.save_log(
            log, host, args.log, fleet={k: host.get(f"fleet.{k}") for k in ("agents", "ticks")},
            run={k: host.get(f"run.{k}") for k in ("world", "uri", "data_path")},
            arrivals=arrived, error=st["error"],
        )  # fmt: skip
        print(f"fleet: {len(log)} ticks to {path}")
    print(f"fleet: {fleet.driving()} on {fleet.world.kind}, {len(fleet.agents)} agents, "
          f"{fleet.ticks * fleet.story.dt:.1f} s: {arrived} arrived")  # fmt: skip
    code = host.exit_code()
    host.close()
    return C.exit_code(code, error=st["error"], arrivals=arrived, min_arrivals=args.min_arrivals,
                       who="fleet")  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
