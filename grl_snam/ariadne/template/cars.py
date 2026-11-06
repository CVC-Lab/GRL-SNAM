"""cars.py: N Ackermann cars driven by YOUR controller, shown in an Ariadne document (cars.ari).

    python cars.py                                   # a window: 6 cars, the controller below
    python cars.py --cars 10 --plant jolt:humvee --world blocks
    python cars.py --plant kinematic                 # nav's bicycle: no physics
    python cars.py --controller navigator            # libcvc's navigator, as a baseline
    python cars.py --headless --frames 600 --log run.npz      # a batch run, no GL (or .csv)

Make it yours: ``grl-snam ariadne cars --copy my_cars`` writes this file and cars.ari into
my_cars/, and ``python my_cars/cars.py`` runs your copy. Edit ``controller`` below. Every tick
(0.05 s) it gets every car's state as numpy arrays and returns one (steer, accel) per car. The
plant is the host's choice, not the document's: ``--plant`` or the Plant combo picks nav's
kinematic bicycle, Jolt vehicles (``roboracer``, ``humvee``, or a comma list over the cars), or
the dynamic single track. grl_snam/ariadne/README.md, "Write your own multi-car controller", says
what each plant means for a controller.

The document (cars.ari) is the panels: the run's controls, YOUR PARAMETERS (each slider bound
to ``ctl.<name>`` reaches ``controller`` as ``p["<name>"]``; add one to both PARAMS and
cars.ari), a row per car, two plots, and New goals / Reset buttons that call back into Python.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from grl_snam.cars import Cars, CarsLog, default_plant

# ══ YOUR CONTROLLER ═════════════════════════════════════════════════════════════════════════════

#: Your parameters and their starting values. cars.ari's Controller window has a slider for each
#: (bound to ctl.<name>); the values reach controller() as p[name] every tick.
PARAMS = {
    "speed": 0.5,  # cruise, as a fraction of each car's top speed
    "spacing": 3.0,  # keep this many car lengths from the other cars
    "gain": 1.5,  # steering per radian of heading error
    "look": 4.0,  # react to walls this many car lengths away
}


def controller(s, p):
    """Every car's command for the next tick, from its state ``s`` and your parameters ``p``.

    ``s`` holds (n,) arrays (SI units): ``x``, ``y`` (the rear axle), ``heading`` (rad, 0 along
    +x, counter-clockwise), ``speed`` (m/s along the heading), ``steer`` (the front wheels'
    angle now, rad, + left), ``vx``, ``vy``, ``goal`` (n, 2), ``goal_dist``, ``reached``,
    ``collided``; ``ranges`` (n, R), the distance to a building or wall along each ray at
    ``ray_angles`` (R,) off the heading (``range_max`` when nothing is that close); the limits
    ``max_steer``, ``max_accel``, ``max_decel``, ``max_speed``; the sizes ``length``, ``width``,
    ``wheelbase``; and ``t``, ``dt``, ``n``. ``s.pos`` is (n, 2). The other cars are the rest of
    the arrays: car i's neighbours are every j != i.

    Return ``(steer, accel)``: (n,) arrays of the front wheels' angle (rad, + left; held to
    each car's lock) and the acceleration along the heading (m/s^2; held to
    [-max_decel, max_accel]). This one heads for the goal, keeps its spacing from the other
    cars, turns away from walls and slows for what is ahead. Replace it with yours.
    """
    L = s.length
    d = s.goal - s.pos
    dist = np.maximum(s.goal_dist, 1e-6)
    pull = d / dist[:, None]

    # Spacing: a push away from each car closer than `spacing` car lengths, stronger nearer.
    rel = s.pos[None, :, :] - s.pos[:, None, :]  # rel[i, j] = car j's position - car i's
    gap = np.hypot(rel[..., 0], rel[..., 1]) + np.eye(s.n) * 1e9
    near = np.clip(1.0 - gap / (p["spacing"] * L[:, None]), 0.0, None)
    push = -(rel / gap[..., None] * near[..., None]).sum(axis=1)

    # Walls: a push away along each ray that sees a wall within `look` car lengths.
    ang = s.heading[:, None] + s.ray_angles[None, :]
    close = np.clip(1.0 - s.ranges / (p["look"] * L[:, None]), 0.0, None)
    wall = -np.stack([(close * np.cos(ang)).sum(1), (close * np.sin(ang)).sum(1)], axis=1)

    want = pull + 2.0 * push + 1.5 * wall
    err = np.arctan2(want[:, 1], want[:, 0]) - s.heading
    err = (err + np.pi) % (2 * np.pi) - np.pi
    steer = p["gain"] * err

    ahead = s.ranges[:, np.abs(s.ray_angles) < 0.6].min(axis=1)
    target = p["speed"] * s.max_speed
    target = target * np.clip(dist / (3.0 * L), 0.0, 1.0)  # slow into the goal
    target = target * np.clip(np.cos(err), 0.2, 1.0)  # and through a sharp turn
    target = target * np.clip((ahead - L) / (p["look"] * L), 0.1, 1.0)  # and for a wall ahead
    target = np.where(s.reached, 0.0, target)

    # Wedged against a wall: back out, steering the other way.
    stuck = (ahead < 0.8 * L) & ~s.reached
    target = np.where(stuck, -0.15 * s.max_speed, target)
    steer = np.where(stuck, -steer, steer)

    accel = 2.0 * (target - s.speed)
    return steer, accel


# ══ the host: the cars, the document and the panels ═════════════════════════════════════════════

HERE = Path(__file__).resolve().parent
COLORS = [(0.90, 0.33, 0.20), (0.18, 0.55, 0.90), (0.95, 0.75, 0.15), (0.35, 0.75, 0.35),
          (0.70, 0.40, 0.85), (0.20, 0.80, 0.80), (0.95, 0.55, 0.70), (0.55, 0.55, 0.55)]  # fmt: skip


def document() -> str:
    """cars.ari, beside this file (in the package, or in your copy)."""
    return str(HERE / "cars.ari")


def copy_template(dest, force=False) -> int:
    """Write this file and cars.ari into ``dest`` to edit."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("cars.py", "cars.ari"):
        out = dest / name
        if out.exists() and not force:
            print(f"cars: {out} exists (--force replaces it)", file=sys.stderr)
            return 1
        shutil.copyfile(HERE / name, out)
    print(f"cars: wrote {dest / 'cars.py'} and {dest / 'cars.ari'}\n"
          f"  edit controller() in cars.py, then:  python {dest / 'cars.py'}")  # fmt: skip
    return 0


def box_mesh(x0, y0, x1, y1, z0, z1):
    v = [
        x0,
        y0,
        z0,
        x1,
        y0,
        z0,
        x1,
        y1,
        z0,
        x0,
        y1,
        z0,
        x0,
        y0,
        z1,
        x1,
        y0,
        z1,
        x1,
        y1,
        z1,
        x0,
        y1,
        z1,
    ]
    t = [4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    return [float(a) for a in v], t


def merge(meshes):
    verts, tris = [], []
    for v, t in meshes:
        b = len(verts) // 3
        verts += v
        tris += [b + k for k in t]
    return verts, tris


def car_mesh(length, width, height):
    """A car body and a cabin set back from its nose, +X forward, base at z = 0."""
    hl, hw = 0.5 * length, 0.5 * width
    body = box_mesh(-hl, -hw, hl, hw, 0.12 * height, 0.55 * height)
    cabin = box_mesh(-0.6 * hl, -0.85 * hw, 0.25 * hl, 0.85 * hw, 0.55 * height, height)
    return merge([body, cabin])


class Scene:
    """What the window draws: the world, a car, a goal pole and a track per car."""

    def __init__(self, host):
        self.host, self.names, self.world = host, [], None

    def build(self, cars):
        host = self.host
        for name in self.names:
            if host.scene.hasGraphics(name):
                host.scene.removeGraphics(name)
        self.names = []
        w = cars.world
        if self.world != (w.name, w.bounds):
            for name in ("ground", "buildings", "terrain"):
                if host.scene.hasGraphics(name):
                    host.scene.removeGraphics(name)
            if w.bundle:
                from pycvc_gl.scenes import load_geometry_bundle

                load_geometry_bundle(host.lab, w.bundle, terrain_color=(0.42, 0.46, 0.36),
                                     building_color=(0.80, 0.78, 0.74))  # fmt: skip
            else:
                mnx, mny, mxx, mxy = w.bounds
                host.lab.add_mesh("ground", [mnx, mny, 0, mxx, mny, 0, mxx, mxy, 0, mnx, mxy, 0],
                                  [0, 1, 2, 0, 2, 3], color=(0.45, 0.50, 0.40))  # fmt: skip
                if w.boxes:
                    host.lab.add_mesh("buildings",
                                      *merge([box_mesh(*b[:4], 0.0, b[4]) for b in w.boxes]),
                                      color=(0.80, 0.78, 0.74))  # fmt: skip
            self.world = (w.name, w.bounds)
        u = cars.unit
        for i in range(cars.n):
            col = COLORS[i % len(COLORS)]
            v = cars.vehicles[i]
            host.lab.add_mesh(f"car_{i}", *car_mesh(v.length, v.width, max(v.height, 0.3 * v.length)),
                              color=col)  # fmt: skip
            host.lab.add_mesh(f"goal_{i}", *box_mesh(-0.08 * u, -0.08 * u, 0.08 * u, 0.08 * u, 0, 2 * u),
                              color=col)  # fmt: skip
            host.lab.add_path(f"track_{i}", [(0, 0, 0), (0.01, 0, 0)], col)
            self.names += [f"car_{i}", f"goal_{i}", f"track_{i}"]
        self.tracks = [[] for _ in range(cars.n)]

    def update(self, cars, s, frame):
        import pycvc_gl
        from pycvc_gl.vehicle import seat_transform

        host, w = self.host, cars.world
        sampler = None if w.height is None else w.z
        centre = s.center
        for i in range(s.n):
            host.scene.getGraphics(f"car_{i}").setTransform(
                seat_transform(centre[i, 0], centre[i, 1], s.heading[i], sampler)
            )
            gx, gy = s.goal[i]
            host.scene.getGraphics(f"goal_{i}").setTransform(
                [1, 0, 0, gx, 0, 1, 0, gy, 0, 0, 1, w.z(gx, gy), 0, 0, 0, 1]
            )
            if frame % 3 == 0:
                self.tracks[i] = (self.tracks[i] + [(s.x[i], s.y[i])])[-300:]
        if frame % 9 == 0:
            for i in range(s.n):
                pts = self.tracks[i]
                if len(pts) >= 2:
                    host.scene.removeGraphics(f"track_{i}")
                    host.lab.add_path(f"track_{i}", [(x, y, w.z(x, y) + 0.05 * cars.unit) for x, y in pts],
                                      COLORS[i % len(COLORS)])  # fmt: skip
                    g = host.scene.geometry_node(f"track_{i}")
                    g.setRenderMode(pycvc_gl.GeometryRenderMode_LINES)
                    g.setLineWidth(2.0)


def parse(argv):
    from pycvc_gl.ariadne import main_args

    host_ns = main_args(argv)
    p = argparse.ArgumentParser(prog="cars.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--cars", type=int, default=6, help="how many cars (default 6)")
    p.add_argument("--plant", default=None,
                   help="kinematic | jolt | dynamic, each with :<presets> (roboracer, humvee, or a "
                        "comma list), or a preset name (default: jolt:roboracer, kinematic without Jolt)")  # fmt: skip
    p.add_argument("--world", choices=("flat", "blocks", "austin"), default="flat")
    p.add_argument("--bundle", default=None, help="the Austin scene bundle (--world austin)")
    p.add_argument("--controller", choices=("yours", "navigator"), default="yours")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=1, help="ticks per frame (default 1)")
    p.add_argument("--regoal", type=int, default=1, help="a car at its goal gets a new one (1/0)")
    p.add_argument(
        "--log", default=None, help="write every tick's states and commands (.npz or .csv)"
    )
    p.add_argument("--copy", metavar="DIR", default=None, help="copy cars.py and cars.ari to DIR")
    p.add_argument("--force", action="store_true", help="with --copy, replace existing files")
    ns = p.parse_args(host_ns.rest)
    for k, v in vars(host_ns).items():
        if k != "rest":
            setattr(ns, k, v)
    return ns


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--copy" in argv:  # no pycvc_gl needed to copy
        k = argv.index("--copy")
        if k + 1 >= len(argv):
            print("cars: --copy DIR", file=sys.stderr)
            return 2
        return copy_template(argv[k + 1], force="--force" in argv)
    args = parse(argv)
    from pycvc_gl.ariadne import Host

    host = Host(mode=args.mode, name="cars", width=args.width, height=args.height)
    host.load(document())
    host.set_many({"cars": args.cars, "plant": args.plant or default_plant(), "world": args.world,
                   "controller": args.controller, "ticks": args.ticks, "regoal": args.regoal},
                  prefix="run")  # fmt: skip
    host.set_many(PARAMS, prefix="ctl")
    drawn = host.mode != "headless"
    scene = Scene(host) if drawn else None
    log = CarsLog() if args.log else None
    st = {"cars": None, "speed": [], "gap": [], "camera": None}

    def build(seed=None):
        plant = host.get("run.plant", args.plant or default_plant())
        nav = host.get("run.controller", "yours") == "navigator"
        try:
            cars = Cars(max(1, host.get_int("run.cars", args.cars)), plant,
                        host.get("run.world", args.world), seed=args.seed if seed is None else seed,
                        navigator=nav, bundle=args.bundle)  # fmt: skip
            note = ""
        except (RuntimeError, ValueError) as e:  # a plant or world this pycvc cannot run
            note = str(e)
            cars = Cars(max(1, host.get_int("run.cars", args.cars)), "kinematic", "flat",
                        seed=args.seed, navigator=nav)  # fmt: skip
        st.update(cars=cars, speed=[], gap=[])
        host.set("stats.note", note)
        if note:
            print(f"cars: {note}; running kinematic on flat ground", file=sys.stderr)
        if scene is not None:
            scene.build(cars)
            st["camera"] = None

    host.on("new_goals", lambda: st["cars"].new_goals())
    host.on("reset", lambda: build(int(np.random.default_rng().integers(1 << 30))))

    @host.watch("run.plant")
    @host.watch("run.world")
    @host.watch("run.cars")
    @host.watch("run.controller")
    def _changed(_v):
        if st["cars"] is not None:
            build()

    build()

    @host.on_frame
    def tick(dt):
        cars = st["cars"]
        if not host.get_bool("run.paused"):
            p = {k: host.get_float(f"ctl.{k}", v) for k, v in PARAMS.items()}
            for _ in range(max(1, host.get_int("run.ticks", 1))):
                s = cars.state()
                if cars.navigator:
                    s = cars.step_navigator()
                else:
                    s = cars.step(*controller(s, p))
                if log is not None:
                    log.record(s, cars.last_command)
                if host.get_bool("run.regoal", True) and s.reached.any():
                    cars.new_goals(s.reached)
        publish(host, cars, st)
        if scene is not None:
            scene.update(cars, cars.state(), host.frame_count)
            camera(host, cars, st)

    host.run(fps=30, frames=args.frames or None)
    if args.png and drawn:
        host.snapshot(args.png)
    if log is not None:
        print(f"cars: {len(log)} ticks to {log.save(args.log)}")
    s = st["cars"].state()
    print(f"cars: {s.plant} on {s.world}, {s.n} cars, {s.t:.1f} s: {st['cars'].arrivals} arrivals, "
          f"{int(s.reached.sum())} at their goals, {int(st['cars'].collisions.sum())} car-ticks "
          f"touching")  # fmt: skip
    code = host.exit_code()
    host.close()
    return code


def publish(host, cars, st):
    """The readouts: the run's numbers, a row per car, and the two plots."""
    s = cars.state()
    k = min(max(0, host.get_int("view.car", 0)), s.n - 1)
    st["speed"] = (st["speed"] + [float(s.speed[k])])[-240:]
    if s.n > 1:
        d = np.hypot(s.x[:, None] - s.x[None, :], s.y[:, None] - s.y[None, :]) + np.eye(s.n) * 1e9
        st["gap"] = (st["gap"] + [float(d.min())])[-240:]
    host.set_series("plot.speed", st["speed"])
    host.set_series("plot.gap", st["gap"])
    host.set_many({"time": f"{s.t:.1f}", "plant": s.plant, "world": s.world,
                   "backend": cars.backend_name + (" (navigator)" if cars.navigator else ""),
                   "arrived": f"{int(s.reached.sum())} / {s.n}", "arrivals": cars.arrivals,
                   "touching": int(cars.collisions.sum())}, prefix="stats")  # fmt: skip
    cmd_s, cmd_a = cars.last_command
    host.set_table("carrows", [
        {"car": str(i), "speed": f"{s.speed[i]:.2f}", "steer": f"{math.degrees(s.steer[i]):.0f}",
         "accel": "" if np.isnan(cmd_a[i]) else f"{cmd_a[i]:.2f}", "goal": f"{s.goal_dist[i]:.1f}",
         "status": "touching" if s.collided[i] else ("arrived" if s.reached[i] else "")}
        for i in range(s.n)])  # fmt: skip


def camera(host, cars, st):
    import pycvc_gl

    mode = host.get("view.camera", "overview")
    s = cars.state()
    if mode == "follow":
        k = min(max(0, host.get_int("view.car", 0)), s.n - 1)
        if st["camera"] != "follow":
            u = cars.unit
            host.set_many({"back": 2.5 * u, "height": 1.2 * u, "look_ahead": 1.2 * u,
                           "look_up": 0.4 * u, "cam_tau": 0.15, "pos_tau": 0.1},
                          prefix="viewers.main.camera.track")  # fmt: skip
        host.camera.setMode(pycvc_gl.CameraController.Mode_Track)
        c = s.center[k]
        host.camera.feedTrackTarget(float(c[0]), float(c[1]), cars.world.z(c[0], c[1]))
    elif st["camera"] != "overview":
        mnx, mny, mxx, mxy = cars.world.bounds
        host.camera.setMode(pycvc_gl.CameraController.Mode_Orbit)
        host.frame_bounds((mnx, mny, 0.0), (mxx, mxy, 2.0 * cars.unit))
    st["camera"] = mode


if __name__ == "__main__":
    sys.exit(main())
