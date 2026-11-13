"""GRL-SNAM's live demos (:mod:`grl_snam.demos`) as Python hosts of an Ariadne document.

    python -m grl_snam.ariadne live austin-freedrive     # the learned SDF policy, no route
    python -m grl_snam.ariadne live austin-learned       # A* spine + learned local control
    python -m grl_snam.ariadne live austin-patrol        # grounded A* patrol (baseline)
    python -m grl_snam.ariadne live austin-planner       # the surrogate planner
    python -m grl_snam.ariadne live austin-freedrive --plant humvee        # on a Jolt Humvee
    python -m grl_snam.ariadne live austin-freedrive --world generated     # in a generated world
    python -m grl_snam.ariadne live austin-freedrive --headless --frames 900 --log run.npz
    grl-snam ariadne live austin-freedrive --plant kinematic --arms trained,untrained,constant

Each demo runs unchanged: its ``setup()`` builds the scene and its ``step(dt)`` drives it, through
:class:`~grl_snam.ariadne._host.DemoHost` instead of VolRover3 or a bare window. The document
(``docs/live.ari``) adds the navigator's HUD, pause / sim-speed / camera controls and a soft
afternoon light, and the settings, worlds, panels and log of libcvc's multi-car template
(pycvc_gl.cars; libcvc's docs/CARS.md). On Austin the vehicle is a Humvee at its real size.

What each demo takes:

- **Settings** (libcvc's keys, set in ``init:``, on the sliders, with ``--set`` / ``--params`` or
  from Python). On GRL-SNAM's torch navigator the keys that mean the same thing are mapped onto
  it (``grl_snam.ariadne._cars``): austin-freedrive (a point navigator) takes
  ``nav.vehicle.vmax``, ``rr``, ``d_hat`` and ``nav.reach_tol``; austin-learned's corridor
  follower ``nav.vehicle.vmax``, ``rr`` and ``d_hat``. austin-patrol (a route played back) and
  austin-planner (its own coefficients) take none. On a plant every ``nav.*``, ``nav.vehicle.*``
  and ``nav_plant.*`` key applies, as libcvc reads it.
- **Plant**, ``nav_plant.vehicle`` (``--plant``): ``torch`` (the default) is the demo's own
  stepping. austin-freedrive also runs on libcvc's sim_world on any plant ``plant_host`` names
  (``kinematic``, ``humvee``, ``roboracer``, ``dynamic:humvee``, ...): libcvc's navigator (the
  same CoefMLP policy, GRL-SNAM's ``grl-snam-weights`` exported to ``.cvcnav``, in its trained
  frame) drives the vehicle from the same start to the same goal, switching plants live. The
  others stay on their own stepping (:data:`REASONS` says why).
- **World**, ``run.world`` (``--world``, ``--uri``): Austin from the document's ``data:`` block by
  default (``--bundle DIR`` is ``--uri DIR``). austin-freedrive also drives another city
  (``--uri``), libcvc's terrain generator or the flat square, from a start to a goal picked far
  apart in it (``GRL_SNAM_START`` / ``GRL_SNAM_GOAL`` still set them). The others are Austin's.
- **Panels**: the coefficients (the policy's alpha, beta, gamma and the clearance) and the
  pictures around the vehicle: the signed distance with its predicted path, and where the wall
  barrier reaches (``rr`` + ``d_hat``; on a plant, the navigator's belief).
- **Headless**: ``--headless --log run.npz`` writes what libcvc's template writes, one row a frame
  (a plant run: one a tick); the exit code is 0, 1 when the run failed, 3 below
  ``--min-arrivals`` (1 for reaching the goal).
- **Arms** (``--arms trained,untrained,constant``, austin-freedrive on a plant): one libcvc
  sim_world per arm, with the same world, start, goal and seed, each a ghost in its own colour; a
  Retarget (the button, or ``--retarget-at``) that moves every arm's goal on the same tick; and
  the Stats and Ledger windows (``docs/stats_panels.ari``). An arm is a bare name or
  ``grl-snam arms-eval``'s ``NAME=PATH.cvcnav|PATH.pt|const:...`` (:mod:`grl_snam.ariadne._arms`
  says what each bare name is, and what the windows show). ``--log`` then writes every arm's
  run (``grl_snam.live_arms/v1``). The torch path refuses arms (:data:`TORCH_ARMS`).

The learned demos need ``grl-snam-weights``, as ``grl-snam demo`` does. ``lab`` stays a VolRover3
demo. Needs pycvc-gl 3.7.12+ (libcvc's pycvc_gl.cars); with an older one it says what to install.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys

import numpy as np

from grl_snam.ariadne import _arms, _cars, _host

DEMOS = {
    "austin-freedrive": ("grl_snam.demos.austin_freedrive", "the learned SDF navigator, no route"),
    "austin-learned": ("grl_snam.demos.austin_learned", "an A* spine + learned SDF control"),
    "austin-patrol": ("grl_snam.demos.austin_patrol", "a grounded A* street patrol"),
    "austin-planner": ("grl_snam.demos.austin_planner", "the surrogate planner, sparse obstacles"),
}
HUMVEE_DEMOS = ("austin-freedrive", "austin-learned", "austin-patrol")
#: The demos that take any world and any plant; the others are Austin's, on their own stepping.
ANYWHERE = ("austin-freedrive",)
#: Why the others stay on Austin and on their own stepping.
REASONS = {
    "austin-learned": "its A* spine and its corner start and goal are placed for Austin, and its "
                      "corridor follower is its own Python on the torch rollout",
    "austin-patrol": "it plays back an A* loop through six waypoints 430 m out on Austin, with no "
                     "navigator and no physics",
    "austin-planner": "it runs the surrogate planner on an analytic terrain of its own",
}  # fmt: skip
#: austin-freedrive's start and goal on Austin (austin_south).
AUSTIN_START, AUSTIN_GOAL = (-361.0, 114.0), (185.0, 50.0)
#: Where the HUD's readouts go (the navigator's NavMetrics), apart from libcvc's nav.* settings.
HUD = "hud"
#: The predicted path's length (navigator ticks) and how often it is worked out (frames).
PATH_STEPS, PATH_EVERY = 40, 15
#: Why the torch path runs no arms.
TORCH_ARMS = (
    "the torch path runs one policy: its demo's state is module-level "
    "(grl_snam.demos._common._STANDALONE holds the one demo a process runs), so a second arm "
    "would drive the first's; run the arms on a plant, each in its own libcvc sim_world "
    "(--plant kinematic, humvee, roboracer, dynamic:humvee)"
)
#: How often the arms' charts are rewritten (frames).
CHART_EVERY = 6


def _args(p):
    p.add_argument("demo", choices=sorted(DEMOS), help="which live demo")
    p.add_argument("--bundle", default=None, help="a local Austin bundle: the same as --uri DIR")
    p.add_argument("--plant", default=None,
                   help="nav_plant.vehicle: torch (the demo's own stepping) or a plant libcvc's "
                        "plant_host names (austin-freedrive)")  # fmt: skip
    p.add_argument("--arms", default=None, metavar="A,B,...",
                   help="austin-freedrive on a plant: run these policies side by side, one "
                        "sim_world each: trained, untrained, constant, or NAME=PATH.cvcnav|"
                        "PATH.pt|const:... as grl-snam arms-eval takes them")  # fmt: skip
    p.add_argument("--arm", action="append", default=[], metavar="NAME=SPEC",
                   help="one more arm, as arms-eval's --arm (repeatable; after --arms)")  # fmt: skip
    p.add_argument("--seed", type=int, default=0,
                   help="with --arms: the untrained arm's initialization and the Retarget goals")  # fmt: skip
    p.add_argument("--retarget-at", default=None, metavar="S,S,...",
                   help="with --arms: a Retarget at each of these sim times (s), as the button "
                        "does (to arms.goal, e.g. --set arms.goal=120,-40, else a drawn goal)")  # fmt: skip
    if _cars.HAS_CARS:
        _cars.C.add_args(p)


def _refuse(msg) -> int:
    print(f"grl-snam ariadne live: {msg}", file=sys.stderr)
    return 2


def start_and_goal(world, kind, uri):
    """austin-freedrive's start and goal: GRL_SNAM_START / GRL_SNAM_GOAL when set, Austin's on
    Austin, else two free points far apart in ``world``."""

    def env(name):
        v = os.environ.get(name)
        return tuple(float(c) for c in v.split(",")) if v else None

    if env("GRL_SNAM_START") and env("GRL_SNAM_GOAL"):
        return env("GRL_SNAM_START"), env("GRL_SNAM_GOAL")
    if kind == "data" and str(uri or _cars.C.AUSTIN).split("@")[0] == _cars.C.AUSTIN:
        return AUSTIN_START, AUSTIN_GOAL
    span = world.span
    pts = world.free_points(np.random.default_rng(0), 2, 4.0, 0.6 * span)
    return tuple(map(float, pts[0])), tuple(map(float, pts[1]))


def barrier_field(host, world):
    """``cost(X, Y)``: where the wall barrier acts, 1 at the footprint's edge falling to 0 at
    ``d_hat`` beyond it (``nav.vehicle.rr``, ``nav.vehicle.d_hat``, metres), on ``world``."""

    def cost(X, Y):
        rr = host.get_float("nav.vehicle.rr", 1.2)
        d_hat = max(host.get_float("nav.vehicle.d_hat", 3.0), 1e-6)
        r, c = world.cells(X, Y)
        return np.clip(1.0 - (world.sdf[r, c] - rr) / d_hat, 0.0, 1.0)

    return cost


def main(argv=None) -> int:
    if not _cars.HAS_CARS:
        return _refuse(f"needs {_cars.INSTALL}")
    args = _host.parse(argv, _args)
    C = _cars.C
    demo = args.demo
    bundle = args.bundle or os.environ.get("GRL_SNAM_SCENE_BUNDLE") or None
    planner = demo == "austin-planner"
    if planner and args.log:
        return _refuse("austin-planner has no navigator state to log (" + REASONS[demo] + ")")
    arms = None
    if args.arms or args.arm:
        if demo not in ANYWHERE:
            return _refuse(f"--arms is austin-freedrive's, on a plant: {demo}: {REASONS[demo]}")
        try:
            arms = _arms.parse_arms(args.arms, args.arm)
            retarget_at = sorted(float(t) for t in (args.retarget_at or "").split(",") if t.strip())
        except ValueError as e:
            return _refuse(f"--arms: {e}")
    host = _host.open_host("live.ari", "live", args, load=False)
    _host.load_document(host, "live.ari", city=not planner, worlds=not planner)
    flags = {"nav_plant.vehicle": args.plant}
    if bundle and not args.uri:  # a local city, loaded (from there, no download) when it is used
        flags.update({"run.uri": str(bundle), "run.data_path": ""})
    C.apply_settings(host, args, flags)
    kind = host.get("run.world", "data") or "data"
    plant = host.get("nav_plant.vehicle", "torch") or "torch"
    if demo not in ANYWHERE and (kind != "data" or args.uri):
        host.close()
        return _refuse(f"{demo} runs on Austin only: {REASONS[demo]}")
    if demo not in ANYWHERE and plant != "torch":
        host.close()
        return _refuse(f"{demo} runs on its own stepping only (--plant torch): {REASONS[demo]}")
    if arms is not None and plant == "torch":
        host.close()
        return _refuse(f"--arms: {TORCH_ARMS}")
    host.set("demo.title", f"{demo}: {DEMOS[demo][1]}")
    world = None
    if not planner:
        world = C.make_world(host, kind)
        C.show_world(host, kind, str(_host.DOCS / "live.ari"))
    if arms is not None:
        return run_arms(host, args, world, plant, kind, arms, retarget_at)
    if plant != "torch":
        return run_on_plant(host, args, world, plant, kind)
    return run_demo(host, args, demo, world, kind)


# ── a demo on its own stepping ─────────────────────────────────────────────────────────────────


def run_demo(host, args, demo, world, kind) -> int:
    """The demo's ``setup()`` and ``step(dt)``, with the settings mapped onto its navigator and the
    panels and log around it."""
    from grl_snam.demos import _common

    C = _cars.C
    module = importlib.import_module(DEMOS[demo][0])
    drawn = host.mode != "headless"
    endpoints = {}  # the demo reads its start and goal from these while it sets up
    if demo in ANYWHERE and world is not None:
        start, goal = start_and_goal(world, kind, host.get("run.uri"))
        endpoints = {"GRL_SNAM_START": f"{start[0]},{start[1]}",
                     "GRL_SNAM_GOAL": f"{goal[0]},{goal[1]}"}  # fmt: skip
    city = None if kind == "data" or world is None else _cars.city_of(world)
    demo_host = _host.DemoHost(host, city=city, metrics_prefix=HUD)
    log = C.CarsLog() if args.log else None
    st = {"settings": None, "coef": {}, "series": {"clearance": [], "speed": []}, "path": None,
          "prev": None, "heading": 0.0, "t": 0.0, "frame": 0, "error": None, "reached": False}  # fmt: skip

    @host.watch("view.camera")
    def _camera(mode):
        if mode == "orbit" and drawn:
            import pycvc_gl

            host.camera.setMode(pycvc_gl.CameraController.Mode_Orbit)

    @host.on_frame
    def _settings(dt):  # before the demo's step: the tree's settings on its navigator
        if st["settings"] is not None:
            st["settings"].apply()

    _common._STANDALONE = demo_host  # current_host() hands the demo this host
    try:
        saved = {k: os.environ.get(k) for k in endpoints}
        os.environ.update(endpoints)
        try:
            module.setup()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        nav = getattr(module, "navigator", lambda: None)()
        if nav is not None:
            st["settings"] = _cars.TorchSettings(host, [nav], getattr(module, "SETTINGS", None))
            st["settings"].seed()
        if drawn and demo in HUMVEE_DEMOS:  # the demo's box -> a Humvee, posed by its VehiclePose
            model = C.humvee_model(host, _host.HUMVEE_LEN_M) or _host.humvee_model(
                _host.bundle_or_none(args.bundle), host=host
            )
            _host.add_vehicle(host, "agent0", model)
        # The published policy's metric (scale 1/300) lets it drive at up to 270 m/s on Austin,
        # so the chase sits well back and catches up quickly.
        host.set_many(
            {"back": 40.0, "height": 16.0, "look_ahead": 8.0, "look_up": 2.0, "cam_tau": 0.04,
             "pos_tau": 0.03},
            prefix="viewers.main.camera.track",
        )  # fmt: skip
        demo_host.run(module.step)  # the demo's step, every frame

        @host.on_frame
        def _panels(dt):  # after it: what it published, as plots, pictures and the log
            try:
                panels(host, args, module, demo_host, world, st, log, dt)
            except Exception as e:  # noqa: BLE001 - a failure ends the run with its reason
                import traceback

                traceback.print_exc()
                st["error"] = str(e)
                host.request_close()

        host.run(fps=30, frames=args.frames or None)
        if args.png and drawn:
            host.snapshot(args.png)
        arrivals = int(st["reached"])
        if log is not None:
            path = C.save_log(log, host, args.log, demo=demo, world=world.info if world else None,
                              plant="torch", arrivals=arrivals, error=st["error"])  # fmt: skip
            print(f"live: {len(log)} frames to {path}")
        code = host.exit_code()
        return C.exit_code(code, error=st["error"], arrivals=arrivals,
                           min_arrivals=args.min_arrivals, who="live")  # fmt: skip
    finally:
        _common._STANDALONE = None
        host.close()


def panels(host, args, module, demo_host, world, st, log, dt):
    C = _cars.C
    m = demo_host.last_metrics
    for key, field in (("clearance", f"{HUD}.clearance_m"), ("speed", f"{HUD}.speed_mps")):
        v = host.get_float(field)
        if v == v:  # not NaN
            st["series"][key] = (st["series"][key] + [v])[-240:]
            host.set_series(f"series.{key}", st["series"][key])
    if m is None or world is None:
        return
    st["frame"] += 1
    if not host.get_bool("view.paused"):
        st["t"] += dt * max(host.get_float("view.speed", 1.0), 0.0)
    if st["prev"] is not None and math.hypot(m.x - st["prev"][0], m.y - st["prev"][1]) > 1e-3:
        st["heading"] = math.atan2(m.y - st["prev"][1], m.x - st["prev"][0])
    st["prev"] = (m.x, m.y)
    heading = m.heading_rad if m.heading_rad else st["heading"]
    coeffs = {}
    if st["settings"] is not None:  # a navigator's demo: its policy's coefficients
        coeffs = {"alpha": [m.alpha], "beta": [m.beta], "gamma": [m.gamma],
                  "clearance": [m.clearance_m]}  # fmt: skip
        C.plot_coefficients(host, st["coef"], coeffs, 0)
    length, width, wheelbase = _cars.humvee_sizes()
    s = C.state_from(world, [m.x], [m.y], [heading], [m.speed_mps], [[m.goal_x, m.goal_y]],
                     t=st["t"], tick=st["frame"], dt=dt, reach=host.get_float("nav.reach_tol", 0.0) or 7.26,
                     driving="torch", length=length, width=width, wheelbase=wheelbase,
                     nav=coeffs)  # fmt: skip
    st["reached"] = st["reached"] or bool(m.reached)
    want_path = host.get_bool("show.plans", True) and hasattr(module, "predicted_path")
    if want_path and (st["path"] is None or st["frame"] % PATH_EVERY == 1):
        st["path"] = module.predicted_path(PATH_STEPS)
    size = C.raster_size(host, length)
    cost = barrier_field(host, world) if st["settings"] is not None else None
    if log is not None:
        log.record(s)
        log.coeffs(coeffs, 1)
        every = max(1, args.raster_every)
        if st["frame"] % every == 1 or every == 1:
            r = C.raster_arrays(world, s, 0, size, cost=cost)
            out = np.full((PATH_STEPS, 2), np.nan, np.float32)
            if st["path"]:
                p = np.asarray(st["path"], float)[:PATH_STEPS]
                out[: len(p)] = p
            log.raster(s.t, plan=out, **r)
    if host.mode != "headless" and host.frame_count % 4 == 0:
        r = C.raster_arrays(world, s, 0, size, cost=cost)
        for name, img in C.raster_pictures(r, s, 0, size, path=st["path"]).items():
            host.publish_image(name, img)


# ── austin-freedrive on a plant ────────────────────────────────────────────────────────────────


def run_on_plant(host, args, world, plant, kind) -> int:
    """austin-freedrive on libcvc's sim_world: libcvc's navigator, with GRL-SNAM's published
    policy, drives one vehicle on ``plant`` from the demo's start to its goal (pycvc_gl.cars'
    Cars, its scene and panels)."""
    from grl_snam.ariadne.cars import grl_policy
    from grl_snam.demos._common import SimPacer

    C = _cars.C
    path, scale = grl_policy()
    start, goal = start_and_goal(world, kind, host.get("run.uri"))
    cars = C.Cars(world, 1, plant, navigator=True, nav_weights=path, scale=scale, app=host.app,
                  prefix=host.prefix, starts=[start], goals=[goal])  # fmt: skip
    drawn = host.mode != "headless"
    log = C.CarsLog() if args.log else None
    st = {"cars": cars, "world": world, "nodes": [], "tracks": [[]], "paths": None,
          "camera": None, "coef": {}, "series": {"clearance": [], "speed": []}, "error": None,
          "pacer": SimPacer(), "d0": None, "reached": False}  # fmt: skip
    host.set("demo.title", f"austin-freedrive on {plant}: libcvc's navigator, GRL-SNAM's policy")
    if log is not None:
        log.meta.update(world=world.info, demo="austin-freedrive", plant=plant, nav_weights=path,
                        nav_scale=scale, start=start, goal=goal)  # fmt: skip
    if drawn:
        C.draw_world(host, st)
        host.lab.add_markers("start", [(start[0], start[1], world.z(*start) + 1.0)],
                             color=(0.15, 0.85, 0.25))  # fmt: skip
        host.lab.add_markers("goal", [(goal[0], goal[1], world.z(*goal) + 1.0)],
                             color=(0.95, 0.80, 0.10))  # fmt: skip

    @host.on_frame
    def tick(dt):
        try:
            cars.pull()
            s = cars.state()
            if not host.get_bool("view.paused"):
                speed = max(host.get_float("view.speed", 1.0), 0.0)
                for _ in range(st["pacer"].ticks(dt, C.DT, speed)):
                    s = cars.step_navigator()
                    if log is not None:
                        log.record(s, cars.last_command)
                        log.coeffs(s.nav, s.n)
                        every = max(1, args.raster_every)
                        if cars.tick % every == 1 or every == 1:
                            log.raster(s.t, **pictures(cars, s, for_log=True))
            cars.publish()
            st["reached"] = st["reached"] or bool(s.reached[0])
            st["d0"] = st["d0"] or float(s.goal_dist[0]) or 1.0
            m = plant_metrics(cars, s, st["d0"])
            _host.publish_metrics(host, m, prefix=HUD)
            for key, v in (("clearance", m.clearance_m), ("speed", m.speed_mps)):
                st["series"][key] = (st["series"][key] + [v])[-240:]
                host.set_series(f"series.{key}", st["series"][key])
            coeffs = {k: s.nav[k] for k in ("alpha", "beta", "gamma", "clearance") if k in s.nav}
            if coeffs:
                C.plot_coefficients(host, st["coef"], coeffs, 0)
            if drawn:
                if host.frame_count % 4 == 0:
                    size = C.raster_size(host, s.length[0])
                    r = pictures(cars, s)
                    for name, img in C.raster_pictures(r, s, 0, size).items():
                        host.publish_image(name, img)
                C.update_scene(host, cars, st)
        except Exception as e:  # noqa: BLE001 - a failure ends the run with its reason
            import traceback

            traceback.print_exc()
            st["error"] = str(e)
            host.request_close()

    def pictures(cars, s, for_log=False):
        return belief_rasters(host, world, cars, s)

    host.run(fps=30, frames=args.frames or None)
    if args.png and drawn:
        host.snapshot(args.png)
    arrivals = int(st["reached"])
    if log is not None:
        out = C.save_log(log, host, args.log, arrivals=arrivals, error=st["error"])
        print(f"live: {len(log)} ticks to {out}")
    s = cars.state()
    print(f"live: austin-freedrive on {s.driving} ({s.world}), {s.t:.1f} s, "
          f"{float(s.goal_dist[0]):.0f} m to go")  # fmt: skip
    code = host.exit_code()
    host.close()
    return C.exit_code(code, error=st["error"], arrivals=arrivals, min_arrivals=args.min_arrivals,
                       who="live")  # fmt: skip


def plant_metrics(cars, s, d0):
    """The HUD's NavMetrics for the one vehicle of ``cars`` (state ``s``), ``d0`` m from its goal
    at the start."""
    from grl_snam.metrics import NavMetrics

    nav = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in s.nav.items()}
    gx, gy = (float(c) for c in s.goal[0])
    return NavMetrics(step=cars.tick, x=float(s.x[0]), y=float(s.y[0]), goal_x=gx, goal_y=gy,
                      goal_dist_m=float(s.goal_dist[0]),
                      clearance_m=nav.get("clearance", 0.0) / cars.scale,
                      speed_mps=float(s.speed[0]), alpha=nav.get("alpha", 0.0),
                      beta=nav.get("beta", 0.0), gamma=nav.get("gamma", 0.0),
                      mode="arrived" if s.reached[0] else "seek",
                      progress=max(0.0, 1.0 - float(s.goal_dist[0]) / d0),
                      reached=bool(s.reached[0]), heading_rad=float(s.heading[0]),
                      plant={"name": s.driving} if cars.host.kind() == "kinematic" else None)  # fmt: skip


def belief_rasters(host, world, cars, s):
    """The rasters around the vehicle of ``cars``: its signed distance, and its navigator's
    belief as the second picture."""
    import pycvc

    size = _cars.C.raster_size(host, s.length[0])
    return _cars.C.raster_arrays(world, s, 0, size,
                                 belief=np.asarray(pycvc.nav_sim_world_belief_occ(cars.sim, 0)))  # fmt: skip


# ── austin-freedrive's arms, side by side ──────────────────────────────────────────────────────


def run_arms(host, args, world, plant, kind, arms, retarget_at=()) -> int:
    """Each arm in its own libcvc sim_world on ``plant``, from the demo's start to its goal, all
    stepped tick for tick together (:mod:`grl_snam.ariadne._arms`): their ghosts, the Retarget,
    the Stats and Ledger windows, the HUD and the rasters around the arm the camera follows."""
    import tempfile
    import time

    import pycvc

    from grl_snam.demos._common import SimPacer

    C = _cars.C
    start, goal = start_and_goal(world, kind, host.get("run.uri"))
    try:
        scale = _arms.resolve(arms, tempfile.mkdtemp(prefix="grl-snam-arms-"), seed=args.seed)
    except (RuntimeError, ValueError, OSError) as e:
        host.close()
        return _refuse(f"--arms: {e}")
    library = pycvc.nav_policy_library_create()
    fleet = []
    for a in arms:
        cars = C.Cars(world, 1, plant, navigator=True, scale=scale, seed=args.seed, app=host.app,
                      prefix=host.prefix, starts=[start], goals=[goal])  # fmt: skip
        a.key = pycvc.nav_sim_world_set_policy(cars.sim, 0, library, a.path, a.sha256)
        fleet.append(cars)
    run = _arms.ArmsRun(arms, start=start, goal=goal, seed=args.seed, dt=C.DT, scale=scale,
                        world=world.info, plant=plant)  # fmt: skip

    def arm_stats():  # every arm's collector, from where it stands, to the current goal
        for a, cars in zip(arms, fleet, strict=True):
            pycvc.nav_sim_world_begin_nav_stats(cars.sim, 0.10, 0.20, 2.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, world.kind, int(args.seed), a.name)  # fmt: skip

    arm_stats()
    rng = np.random.default_rng(args.seed + 1)
    radius = 0.5 * float(fleet[0].width[0])
    st = {"pending": None, "due": list(retarget_at), "nodes": [], "tracks": [[] for _ in arms],
          "world": world, "camera": None, "follow": None, "coef": {},
          "series": {"clearance": [], "speed": []}, "error": None, "pacer": SimPacer(),
          "d0": None, "modeled": None}  # fmt: skip
    drawn = host.mode != "headless"
    charts = hasattr(host, "set_plot")
    host.set_many({"on": "1", "names": "\n".join(a.name for a in arms), "status": "",
                   "plant": plant, "scale": f"{scale:.6g}", "seed": str(args.seed)}, prefix="arms")  # fmt: skip
    for key in ("view.arm", "ledger.agent"):
        if host.get(key, "") not in [a.name for a in arms]:
            host.set(key, arms[0].name)
    host.set("demo.title", f"austin-freedrive on {plant}: {len(arms)} arms, "
                           f"{', '.join(a.name for a in arms)}")  # fmt: skip

    def retarget(to=None):
        """Every arm's goal moves to ``to`` (else a drawn one) before the next tick."""
        pos = [(c.state().x[0], c.state().y[0]) for c in fleet]
        if to is None:
            to = _arms.draw_goal(world, rng, pos, 2.0 * radius)
        else:
            why = _arms.check_goal(world, to[0], to[1], radius)
            if why:
                host.set("arms.status", f"not there: {why}")
                return
        for cars in fleet:
            cars.goals[0] = to
            pycvc.nav_sim_world_retarget(cars.sim, 0, (to[0] - cars.cx) * cars.scale,
                                         (to[1] - cars.cy) * cars.scale)  # fmt: skip
        arm_stats()
        run.retarget(to)
        host.set("arms.status", f"leg {len(run.legs)}: every arm to ({to[0]:.0f}, {to[1]:.0f}), "
                                f"from tick {run.tick}")  # fmt: skip

    def press():  # the Retarget button: to arms.goal when it names a point, else a drawn one
        st["pending"] = _arms.parse_point(host.get("arms.goal")) or "draw"

    host.on("arms_retarget", press)
    if drawn:
        C.draw_world(host, st)
        host.lab.add_markers("start", [(start[0], start[1], world.z(*start) + 1.0)],
                             color=(0.15, 0.85, 0.25))  # fmt: skip

    @host.on_frame
    def tick(dt):
        try:
            if st["pending"] is not None:  # between ticks: every arm's goal moves at once
                to, st["pending"] = st["pending"], None
                retarget(None if to == "draw" else to)
            for cars in fleet:
                cars.pull()
            if not host.get_bool("view.paused"):
                speed = max(host.get_float("view.speed", 1.0), 0.0)
                for _ in range(st["pacer"].ticks(dt, C.DT, speed)):
                    if st["due"] and run.tick * C.DT >= st["due"][0] - 1e-9:
                        st["due"].pop(0)
                        retarget(_arms.parse_point(host.get("arms.goal")))
                    for j in np.roll(np.arange(len(fleet)), -run.tick):  # each first in turn
                        cars = fleet[j]
                        t0 = time.perf_counter()
                        s = cars.step_navigator()
                        ms = 1e3 * (time.perf_counter() - t0)
                        run.record(j, x=s.x[0], y=s.y[0], heading=s.heading[0], speed=s.speed[0],
                                   goal=s.goal[0], ms=ms,
                                   tel=pycvc.nav_sim_world_drive_telemetry(cars.sim),
                                   nav_stats=json.loads(pycvc.nav_sim_world_nav_stats(cars.sim)))  # fmt: skip
                    run.end_tick()
            fleet[0].publish()
            publish_arms(host, run, fleet, st, charts)
            if drawn:
                update_arms_scene(host, run, fleet, st)
        except Exception as e:  # noqa: BLE001 - a failure ends the run with its reason
            import traceback

            traceback.print_exc()
            st["error"] = str(e)
            host.request_close()

    host.run(fps=30, frames=args.frames or None)
    if args.png and drawn:
        host.snapshot(args.png)
    if args.log:
        settings = {d["path"]: d["value"] for d in C.settings(host.app, host.prefix)}
        out = run.save(args.log, settings=settings, arrivals=run.arrivals, error=st["error"])
        print(f"live: {len(run)} ticks of {len(arms)} arms to {out}")
    for row in run.stats_rows():
        print(f"live: {row['arm']:>12}: clearance {row['clear_min']} / {row['clear_mean']} m, "
              f"{row['pen']} penetrating, {row['path']} m in {row['elapsed']} s, "
              f"{row['ms']} ms/tick, reached {row['reached']}")  # fmt: skip
    code = host.exit_code()
    host.close()
    return C.exit_code(code, error=st["error"], arrivals=run.arrivals,
                       min_arrivals=args.min_arrivals, who="live")  # fmt: skip


def publish_arms(host, run, fleet, st, charts):
    """The Stats and Ledger windows' rows and charts, and the HUD, plots and rasters of the arm the
    camera follows."""
    C = _cars.C
    k = run.index(host.get("view.arm", ""))
    if st["follow"] != k:  # another arm's plots start afresh
        st["follow"], st["coef"], st["d0"] = k, {}, None
        st["series"] = {"clearance": [], "speed": []}
    cars = fleet[k]
    s = cars.state()
    st["d0"] = st["d0"] or float(s.goal_dist[0]) or 1.0
    m = plant_metrics(cars, s, st["d0"])
    _host.publish_metrics(host, m, prefix=HUD)
    for key, v in (("clearance", m.clearance_m), ("speed", m.speed_mps)):
        st["series"][key] = (st["series"][key] + [v])[-240:]
        host.set_series(f"series.{key}", st["series"][key])
    coeffs = {c: s.nav[c] for c in ("alpha", "beta", "gamma", "clearance") if c in s.nav}
    if coeffs:
        C.plot_coefficients(host, st["coef"], coeffs, 0)
    host.set_table("arms.stats", run.stats_rows())
    host.set_table("ledger.coef", run.coefficient_rows())
    host.set_table("ledger.terms", run.term_rows())
    host.set_table("ledger.books", run.book_rows(cumulative=True))
    host.set_table("ledger.tick", run.book_rows(cumulative=False))
    j = run.index(host.get("ledger.agent", ""))
    rows = run.plant_rows(j)
    host.set_table("ledger.plant", rows)
    applies = sum(r["value"] != "n/a" for r in rows)
    host.set("ledger.plant_note", f"{run.arms[j].name} on {s.driving}: "
             + (f"{applies} of {len(rows)} items apply" if applies else
                "no item applies (a physics plant feeds them)"))  # fmt: skip
    host.set("arms.time", f"t = {run.tick * run.dt:.1f} s, tick {run.tick}, leg {len(run.legs)}")
    if host.mode != "headless" and host.frame_count % 4 == 0:
        size = C.raster_size(host, s.length[0])
        r = belief_rasters(host, fleet[k].world, cars, s)
        for name, img in C.raster_pictures(r, s, 0, size).items():
            host.publish_image(name, img)
    if charts and len(run) and (host.frame_count % CHART_EVERY == 0 or len(run) <= 2):
        chart_arms(host, run, j)


def chart_arms(host, run, j):
    """The charts (pycvc_gl's chart:, libcvc 3.7.14): every arm's clearance, agent ``j``'s
    cumulative books, and one book's cumulative sum for every arm (``ledger.book``)."""
    t = np.asarray(run.t, float)
    marks = [(t[g["tick"]] if g["tick"] < len(t) else t[-1], f"leg {i + 1}", "note")
             for i, g in enumerate(run.legs[1:], start=1)]  # fmt: skip
    style = [f"c{a.slot}" for a in run.arms]
    host.set_plot("arms.clearance", [
        {"x": t, "y": np.asarray(s.tel.get("clearance", []), float) / run.scale, "label": a.name,
         "style": style[i]} for i, (a, s) in enumerate(zip(run.arms, run.series, strict=True))
    ], markers=marks)  # fmt: skip
    host.set_plot("ledger.cum", [{"x": t, "y": run.cumulative(j, b), "label": b}
                                 for b in _arms.BOOKS], markers=marks)  # fmt: skip
    book = host.get("ledger.book", "dU")
    book = book if book in _arms.BOOKS else "dU"
    host.set_plot("ledger.compare", [{"x": t, "y": run.cumulative(i, book), "label": a.name,
                                      "style": style[i]} for i, a in enumerate(run.arms)],
                  markers=marks)  # fmt: skip


def draw_arms(host, fleet, st, arms):
    """Each arm's ghost: the Humvee's model (or a box its size) in the arm's colour, see-through;
    a tag above it; and the goal's post."""
    from pycvc_gl.vehicle import seat_transform  # noqa: F401 - fail here, not mid-frame

    C = _cars.C
    for name in st["nodes"]:
        if host.scene.hasGraphics(name):
            host.scene.removeGraphics(name)
    st["nodes"], modeled = [], []
    for j, (a, cars) in enumerate(zip(arms, fleet, strict=True)):
        col = _arms.slot_rgb(a.slot)
        length, width = float(cars.length[0]), float(cars.width[0])
        model = C.humvee_model(host, length) if cars.preset[0] in (None, "humvee") else None
        name = f"arm{j}_car"
        if model is not None:
            model.add(host.scene, name, col)
        else:
            host.lab.add_mesh(name, *_host.humvee_box(length, width, 0.4 * length), color=col)
        gn = host.scene.geometry_node(name)
        gn.setUseSingleColor(True)
        gn.setColor(*col)
        gn.setOpacity(0.55)
        host.lab.add_mesh(f"arm{j}_tag", *_host.humvee_box(0.7, 0.7, 0.5), color=col)
        st["nodes"] += [name, f"arm{j}_tag"]
        modeled.append(model is not None)
    u = float(np.max(fleet[0].length))
    host.lab.add_mesh("arms_goal", *_host.humvee_box(0.16 * u, 0.16 * u, 2 * u),
                      color=(0.95, 0.80, 0.10))  # fmt: skip
    st["nodes"].append("arms_goal")
    st["modeled"] = modeled


def update_arms_scene(host, run, fleet, st):
    """The ghosts at their poses, their tracks, the goal, and the camera on the followed arm."""
    import pycvc_gl
    from pycvc_gl.vehicle import seat_transform

    C = _cars.C
    world = fleet[0].world
    sampler = None if world.height is None else world.z
    sized = [float(c.length[0]) for c in fleet]
    if (
        st["modeled"] is None
        or st.get("sized") != sized
        or (not all(st["modeled"]) and host.frame_count % 30 == 0 and C.humvee_model_path(host))
    ):  # a new plant's vehicles, or the model has come since
        draw_arms(host, fleet, st, run.arms)
        st["sized"] = sized
    for j, cars in enumerate(fleet):
        s = cars.state()
        c = s.center[0]
        host.scene.getGraphics(f"arm{j}_car").setTransform(
            seat_transform(c[0], c[1], s.heading[0], sampler)
        )
        host.scene.getGraphics(f"arm{j}_tag").setTransform(
            seat_transform(c[0], c[1], s.heading[0], sampler, lift=2.3 + 0.6 * j)
        )
        if host.frame_count % 3 == 0:
            st["tracks"][j] = (st["tracks"][j] + [(float(s.x[0]), float(s.y[0]))])[-600:]
    gx, gy = run.goal
    host.scene.getGraphics("arms_goal").setTransform(
        [1, 0, 0, gx, 0, 1, 0, gy, 0, 0, 1, world.z(gx, gy), 0, 0, 0, 1]
    )
    if host.frame_count % 6 == 0:
        for j, pts in enumerate(st["tracks"]):
            name = f"arm{j}_track"
            if host.scene.hasGraphics(name):
                host.scene.removeGraphics(name)
            if len(pts) >= 2:
                host.lab.add_path(name, [(x, y, world.z(x, y) + 0.3) for x, y in pts],
                                  _arms.slot_rgb(run.arms[j].slot))  # fmt: skip
                g = host.scene.geometry_node(name)
                g.setRenderMode(pycvc_gl.GeometryRenderMode_LINES)
                g.setLineWidth(2.5)
    k = run.index(host.get("view.arm", ""))
    s = fleet[k].state()
    c = s.center[0]
    mode = host.get("view.camera", "follow")
    if mode == "follow":
        if st["camera"] != "follow":
            u = float(np.max(fleet[0].length))
            host.set_many({"back": 4.0 * u, "height": 2.0 * u, "look_ahead": 1.5 * u,
                           "look_up": 0.4 * u, "cam_tau": 0.15, "pos_tau": 0.1},
                          prefix="viewers.main.camera.track")  # fmt: skip
        host.camera.setMode(pycvc_gl.CameraController.Mode_Track)
        host.camera.feedTrackTarget(float(c[0]), float(c[1]), world.z(c[0], c[1]))
    elif st["camera"] != mode:  # once: the whole world, then the orbit is the viewer's
        mnx, mny, mxx, mxy = world.bounds
        host.camera.setMode(pycvc_gl.CameraController.Mode_Orbit)
        host.frame_bounds((mnx, mny, 0.0), (mxx, mxy, 40.0))
    st["camera"] = mode


if __name__ == "__main__":
    sys.exit(main())
