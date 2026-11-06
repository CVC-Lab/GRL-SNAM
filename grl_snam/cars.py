"""N Ackermann cars that a controller of your own drives, on the plant you choose.

    from grl_snam.cars import Cars

    cars = Cars(6, plant="jolt:roboracer", world="flat", seed=0)
    s = cars.state()
    while s.t < 30.0:
        steer, accel = my_controller(s)       # (n,) arrays: rad (+ left), m/s^2
        s = cars.step(steer, accel)

Every tick (:data:`DT`, 0.05 s) each car's plant takes your ``(steer, accel)``, and
:meth:`Cars.step` returns the new :class:`CarsState`: every car's pose, speed, wheel angle and
goal, its range readings, and whether it touches a building or another car. Units are SI
throughout. A car's ``(x, y)`` is its rear axle's midpoint, the point the bicycle model turns about.

The plant (``plant=``) decides what your command does:

- ``kinematic``: nav's kinematic bicycle. The wheels point where you steer at once, the speed
  integrates your acceleration, nothing slips, and nothing collides (a car drives through a
  building; :attr:`CarsState.collided` says so).
- ``jolt`` (or a preset name): a Jolt vehicle per car (libcvc's ``cvc::physics``): an engine and
  gearbox that a driver model works to give your acceleration, brakes, tires that slip,
  suspension, and the steering's rate limit. Cars collide with the buildings and each other.
- ``dynamic``: libcvc's dynamic single-track vehicle (Pacejka tires, load transfer, actuator
  lag), a box per chassis in a Jolt world. Needs a pycvc with ``nav_sim_world_override``.

``:<presets>`` names the vehicles: ``roboracer`` (the lab's 1/10 car, the default), ``humvee``,
or a comma list cycled over the cars (``jolt:roboracer,humvee``). On ``kinematic`` the preset
gives each car its size and limits.

The world (``world=``) is ``flat`` (an open walled square), ``blocks`` (a block city), both
scaled to the largest car, or ``austin`` (the Austin scene bundle at its real size; cvcpkg's
``scene-austin-south`` or ``scene-austin-south-small``).

How a command reaches the plant: with a pycvc that has ``nav_sim_world_override`` (libcvc after
3.7.11), every plant runs in libcvc's ``sim_world`` and takes your command in its navigator's
place. With an older pycvc, ``kinematic`` runs the same bicycle in numpy and ``jolt`` drives each
vehicle through ``physics_world.drive``, which cannot back up (a negative acceleration only
brakes). ``Cars(..., navigator=True)`` runs libcvc's navigator (GRL-SNAM's, in C++) instead of
your controller, as a baseline: :meth:`Cars.step_navigator`.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field

import numpy as np

try:
    import pycvc as _pycvc
except Exception:  # pragma: no cover - pycvc is what drives the cars
    _pycvc = None

#: The control tick (s): your controller runs once per tick.
DT = 0.05
#: The physics step (s) under a tick, on Jolt and the dynamic plant.
PHYSICS_DT = 0.01
#: The kinematic bicycle's substeps per tick.
NSUB = 5
#: The vehicle presets libcvc ships.
PRESETS = ("roboracer", "humvee")
#: The plant kinds.
KINDS = ("kinematic", "jolt", "dynamic")
#: The worlds.
WORLDS = ("flat", "blocks", "austin")

HAS_PYCVC = _pycvc is not None and hasattr(_pycvc, "nav_sim_world_create")
#: The pycvc can take a caller's command in sim_world's navigator's place (libcvc after 3.7.11).
HAS_OVERRIDE = HAS_PYCVC and hasattr(_pycvc, "nav_sim_world_override")
HAS_SET_POSE = HAS_PYCVC and hasattr(_pycvc, "nav_sim_world_set_pose")
HAS_PHYSICS = _pycvc is not None and hasattr(_pycvc, "physics_world")
HAS_DYN = HAS_PYCVC and hasattr(_pycvc, "dyn_bike_plant")


def has_jolt() -> bool:
    """The pycvc can run Jolt vehicles."""
    if not HAS_PHYSICS:
        return False
    try:
        return "jolt" in _pycvc.physics_backends() and bool(
            _pycvc.physics_backend_info("jolt")["available"]
        )
    except Exception:  # noqa: BLE001 - an older pycvc without the backend queries
        return False


# ── the vehicles ───────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Vehicle:
    """A preset's size and limits (SI). ``rear_x`` is the rear axle's offset from the body's
    centre (negative), ``body_z`` the chassis box's half height."""

    name: str
    length: float
    width: float
    height: float
    wheelbase: float
    rear_x: float
    max_steer: float
    max_steer_rate: float
    max_accel: float
    max_decel: float
    max_speed: float


# libcvc's presets (pycvc.vehicle_desc.<name>()), for a pycvc without the physics bindings.
_FALLBACK = {
    "roboracer": Vehicle("roboracer", 0.535, 0.281, 0.16, 0.324, -0.162, 0.425, 3.0, 6.0, 6.0, 10.0),
    "humvee": Vehicle("humvee", 4.84, 2.18, 1.9, 3.30, -1.65, 0.457, 0.35, 2.5, 7.5, 24.6),
}  # fmt: skip


def vehicle(name: str) -> Vehicle:
    """A preset's :class:`Vehicle`, read from libcvc's ``vehicle_desc`` when pycvc has it."""
    if name not in PRESETS:
        raise ValueError(f"no vehicle preset {name!r}: one of {', '.join(PRESETS)}")
    if not HAS_PHYSICS:
        return _FALLBACK[name]
    d = getattr(_pycvc.vehicle_desc, name)()
    hx, hy, _hz = d.half_extents()
    axles = [d.axle(i) for i in range(d.axle_count())]
    rear = min(a["x"] for a in axles if not a["steered"])
    front = max(a["x"] for a in axles if a["steered"])
    fb = _FALLBACK[name]
    return Vehicle(name, 2 * hx, 2 * hy, fb.height, front - rear, rear, float(d.max_steer),
                   float(d.max_steer_rate), float(d.max_accel), float(d.max_decel),
                   float(d.max_speed))  # fmt: skip


def parse_plant(plant: str) -> tuple[str, list[str]]:
    """``"jolt:roboracer,humvee"`` -> ``("jolt", ["roboracer", "humvee"])``. A bare preset name is a
    Jolt plant; no presets is a RoboRacer."""
    kind, _, names = str(plant).partition(":")
    if kind in PRESETS and not names:
        kind, names = "jolt", kind
    if kind not in KINDS:
        raise ValueError(
            f"plant {plant!r}: kinematic, jolt or dynamic, each optionally ':<presets>' "
            f"({', '.join(PRESETS)}, comma separated), or a preset name"
        )
    presets = [p.strip() for p in names.split(",") if p.strip()] or ["roboracer"]
    for p in presets:
        vehicle(p)  # raises on an unknown one
    return kind, presets


def default_plant() -> str:
    """``jolt:roboracer`` when this pycvc runs Jolt, else ``kinematic:roboracer``."""
    return "jolt:roboracer" if has_jolt() else "kinematic:roboracer"


# ── the world ──────────────────────────────────────────────────────────────────────────────────


@dataclass
class World:
    """The ground the cars drive on: an occupancy grid (row 0 at ``min_y``, cell centres on the
    grid points) over ``bounds`` (min_x, min_y, max_x, max_y, m), the buildings as boxes
    ``(x0, y0, x1, y1, height)`` for drawing, and a terrain height sampler (None: flat at 0)."""

    name: str
    occ: np.ndarray
    bounds: tuple
    boxes: list = field(default_factory=list)
    height: object = None
    bundle: str | None = None
    building_height: float = 6.0

    @property
    def cell(self) -> float:
        return (self.bounds[2] - self.bounds[0]) / (self.occ.shape[1] - 1)

    @property
    def center(self) -> tuple:
        return 0.5 * (self.bounds[0] + self.bounds[2]), 0.5 * (self.bounds[1] + self.bounds[3])

    def cells(self, x, y):
        """Grid (row, col) of world points (arrays), clamped to the grid."""
        mnx, mny = self.bounds[0], self.bounds[1]
        rows, cols = self.occ.shape
        c = np.clip(np.rint((np.asarray(x) - mnx) / self.cell), 0, cols - 1).astype(int)
        r = np.clip(np.rint((np.asarray(y) - mny) / self.cell), 0, rows - 1).astype(int)
        return r, c

    def occupied(self, x, y):
        """Whether world points (arrays) are on a building or outside the grid."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        mnx, mny, mxx, mxy = self.bounds
        out = (x < mnx) | (x > mxx) | (y < mny) | (y > mxy)
        r, c = self.cells(x, y)
        return out | (self.occ[r, c] > 0)

    def z(self, x, y) -> float:
        if self.height is None:
            return 0.0
        h = self.height(float(x), float(y))
        return 0.0 if math.isnan(h) else float(h)

    def free_points(self, rng, n, clearance, min_sep, avoid=None, avoid_dist=0.0):
        """``n`` random points at least ``clearance`` m from a building, ``min_sep`` m apart, and
        ``avoid_dist`` m from the matching point of ``avoid`` ((n, 2) or None)."""
        free = ~_dilate(self.occ > 0, max(1, int(math.ceil(clearance / self.cell))))
        rr, cc = np.nonzero(free)
        if len(rr) == 0:
            raise RuntimeError(f"the {self.name} world has no free ground for the cars")
        mnx, mny = self.bounds[0], self.bounds[1]
        pts = []
        for i in range(n):
            best = None
            for _ in range(400):
                k = int(rng.integers(len(rr)))
                p = (mnx + cc[k] * self.cell, mny + rr[k] * self.cell)
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) < min_sep for q in pts):
                    continue
                if (
                    avoid is not None
                    and math.hypot(p[0] - avoid[i][0], p[1] - avoid[i][1]) < avoid_dist
                ):
                    best = best or p
                    continue
                best = p
                break
            pts.append(best or (mnx + cc[0] * self.cell, mny + rr[0] * self.cell))
        return np.array(pts, float)


def _dilate(mask, k):
    out = mask.copy()
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if dr * dr + dc * dc <= k * k:
                out |= np.roll(np.roll(mask, dr, 0), dc, 1)
    return out


def _border(occ):
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1


def _grid(half, cell):
    n = int(round(2 * half / cell)) + 1
    return np.zeros((n, n), np.uint8), (-half, -half, half, half)


def flat_world(unit: float) -> World:
    """An open square 40 car lengths (``unit``) across, walled at its edge."""
    occ, bounds = _grid(20.0 * unit, 0.5 * unit)
    _border(occ)
    h = 1.2 * unit
    m = 20.0 * unit
    boxes = [(-m, -m, m, -m + 0.5 * unit, h), (-m, m - 0.5 * unit, m, m, h),
             (-m, -m, -m + 0.5 * unit, m, h), (m - 0.5 * unit, -m, m, m, h)]  # fmt: skip
    return World("flat", occ, bounds, boxes, building_height=h)


def blocks_world(unit: float) -> World:
    """A block city scaled to the cars: blocks 10 car lengths (``unit``) square, streets 5 wide,
    five lots a side with every seventh an open square, buildings 2.5 to 8 car lengths tall."""
    half, cell = 40.0 * unit, 0.5 * unit
    occ, bounds = _grid(half, cell)
    block, street = 10.0 * unit, 5.0 * unit
    boxes, k = [], 0
    x0 = -half + street
    while x0 + block < half - street:
        y0 = -half + street
        while y0 + block < half - street:
            k += 1
            if k % 7 != 3:
                h = (2.5 + 5.5 * ((k * 37) % 11) / 10.0) * unit
                boxes.append((x0, y0, x0 + block, y0 + block, h))
                c0, c1 = int(round((x0 + half) / cell)), int(round((x0 + block + half) / cell))
                r0, r1 = int(round((y0 + half) / cell)), int(round((y0 + block + half) / cell))
                occ[r0 : r1 + 1, c0 : c1 + 1] = 1
            y0 += block + street
        x0 += block + street
    _border(occ)
    return World("blocks", occ, bounds, boxes, building_height=8.0 * unit)


def austin_world(bundle=None, half=400.0, cell=2.0, city=None) -> World:
    """The Austin scene bundle (terrain.json + buildings), cropped to ``half`` m around its
    centre. ``city`` is the bundle as already loaded, ``(world_geometry, height_field)`` (an
    Ariadne host's data store: ``pycvc.data_world`` / ``data_height_field``), so nothing is read
    again; otherwise the bundle is read from ``bundle``, else ``GRL_SNAM_SCENE_BUNDLE``, else the
    installed one."""
    if city is not None:
        geometry, field = city
        return _austin_from(geometry, field, field.bounds2d(), half, cell, None)
    if bundle is None:
        bundle = os.environ.get("GRL_SNAM_SCENE_BUNDLE")
    if bundle is None:
        try:
            from pycvc_gl.scenes import find_scene

            bundle = find_scene()
        except Exception:  # noqa: BLE001 - no pycvc_gl, or no scene
            bundle = None
    if not bundle or not os.path.isfile(os.path.join(str(bundle), "terrain.json")):
        raise RuntimeError(
            "the austin world needs the Austin scene bundle: cvcpkg install scene-austin-south-small "
            "(or scene-austin-south), or --bundle DIR / GRL_SNAM_SCENE_BUNDLE"
        )
    loaded = _pycvc.load_city_bundle(str(bundle))
    return _austin_from(
        loaded.world(), loaded.terrain(), loaded.bounds2d(), half, cell, str(bundle)
    )


def _austin_from(geometry, field, extent, half, cell, bundle) -> World:
    mnx, mny, mxx, mxy = extent
    half = min(half, 0.5 * (mxx - mnx), 0.5 * (mxy - mny))
    cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    bounds = (cx - half, cy - half, cx + half, cy + half)
    n = int(round(2 * half / cell)) + 1
    occ = np.ascontiguousarray(
        _pycvc.nav_world_rasterize(geometry, n, n, *bounds)["occupancy"], np.uint8
    )
    _border(occ)
    return World("austin", occ, bounds, [], field.height, bundle, building_height=12.0)


def make_world(name: str, unit: float, bundle=None, city=None) -> World:
    if name == "flat":
        return flat_world(unit)
    if name == "blocks":
        return blocks_world(unit)
    if name == "austin":
        return austin_world(bundle, city=city)
    raise ValueError(f"world {name!r}: one of {', '.join(WORLDS)}")


def obstacle_boxes(occ, bounds):
    """The occupied cells as few rectangles (x0, y0, x1, y1), each its cells' extent: a row's runs,
    grown down while the next rows have the same run."""
    occ = np.asarray(occ, bool)
    rows, cols = occ.shape
    mnx, mny, mxx, mxy = bounds
    cw, ch = (mxx - mnx) / (cols - 1), (mxy - mny) / (rows - 1)
    done = np.zeros_like(occ)
    out = []
    for r in range(rows):
        c = 0
        while c < cols:
            if not occ[r, c] or done[r, c]:
                c += 1
                continue
            c1 = c
            while c1 < cols and occ[r, c1] and not done[r, c1]:
                c1 += 1
            r1 = r + 1
            while r1 < rows and occ[r1, c:c1].all() and not done[r1, c:c1].any():
                r1 += 1
            done[r:r1, c:c1] = True
            out.append((mnx + (c - 0.5) * cw, mny + (r - 0.5) * ch, mnx + (c1 - 0.5) * cw,
                        mny + (r1 - 0.5) * ch))  # fmt: skip
            c = c1
    return out


# ── what the controller sees ───────────────────────────────────────────────────────────────────


@dataclass
class CarsState:
    """Every car at one tick, as (n,) arrays unless noted (SI: m, rad, m/s, m/s^2).

    ``x``, ``y`` (the rear axle's midpoint), ``heading`` (0 along +x, counter-clockwise),
    ``speed`` (along the heading; negative backing up), ``steer`` (the front wheels' angle now,
    + left; on Jolt it lags the command at the steering's rate), ``vx``, ``vy`` (the velocity in
    the world), ``goal`` (n, 2), ``goal_dist``, ``reached`` (within ``reach`` m of the goal),
    ``ranges`` (n, R): the distance to a building or wall along each ray, at ``ray_angles`` (R,)
    off the heading, up to ``range_max``; ``collided``: touching a building or another car.
    The limits ``length``, ``width``, ``wheelbase``, ``max_steer``, ``max_accel``, ``max_decel``
    and ``max_speed`` are per car. ``t`` is the time (s), ``tick`` the ticks stepped, ``dt``
    the tick."""

    t: float
    tick: int
    dt: float
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    steer: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    goal: np.ndarray
    goal_dist: np.ndarray
    reached: np.ndarray
    ranges: np.ndarray
    ray_angles: np.ndarray
    range_max: float
    collided: np.ndarray
    length: np.ndarray
    width: np.ndarray
    wheelbase: np.ndarray
    max_steer: np.ndarray
    max_accel: np.ndarray
    max_decel: np.ndarray
    max_speed: np.ndarray
    reach: float
    plant: str
    world: str

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def pos(self) -> np.ndarray:
        """(n, 2) positions."""
        return np.stack([self.x, self.y], axis=1)

    @property
    def center(self) -> np.ndarray:
        """(n, 2) the bodies' centres (half a wheelbase ahead of the rear axle)."""
        h = 0.5 * self.wheelbase
        return np.stack([self.x + h * np.cos(self.heading), self.y + h * np.sin(self.heading)], 1)


# ── the backends ───────────────────────────────────────────────────────────────────────────────


class _Kinematic:
    """nav's kinematic bicycle (libcvc's plant_types.h kinematic_step, after sim_world's speed hold)
    in numpy: what a pycvc without nav_sim_world_override drives on ``kinematic``."""

    def __init__(self, cars, starts, headings):
        self.cars = cars
        self.x, self.y = starts[:, 0].copy(), starts[:, 1].copy()
        self.th = headings.copy()
        self.sp = np.zeros(cars.n)
        self.steer = np.zeros(cars.n)

    def command(self, steer, accel):
        c = self.cars
        vmax, sp_min = c.max_speed, -0.25 * c.max_speed
        acc = np.where(self.sp >= 0.0, np.minimum(accel, (vmax - self.sp) / DT),
                       np.maximum(accel, (sp_min - self.sp) / DT))  # fmt: skip
        h, t = DT / NSUB, np.tan(steer)
        for _ in range(NSUB):
            self.sp = np.clip(self.sp + h * acc, sp_min, vmax)
            self.th = self.th + h * (self.sp / c.wheelbase) * t
            self.x = self.x + h * self.sp * np.cos(self.th)
            self.y = self.y + h * self.sp * np.sin(self.th)
        self.steer = np.asarray(steer, float).copy()

    def read(self):
        return (self.x, self.y, self.th, self.sp, self.steer, self.sp * np.cos(self.th),
                self.sp * np.sin(self.th))  # fmt: skip


def _yaw(q):
    w, qx, qy, qz = q
    return math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _physics_world(world: World, mu=0.9, rolling_resistance=0.015):
    """A Jolt world: a flat ground box under ``world`` and its buildings as static boxes."""
    w = _pycvc.physics_world(backend="jolt", fixed_dt=PHYSICS_DT)
    ground_m = w.add_material("ground", friction=mu, rolling_resistance=rolling_resistance)
    wall_m = w.add_material("building", friction=0.6)
    mnx, mny, mxx, mxy = world.bounds
    cx, cy = world.center
    ground = w.add_box((0.5 * (mxx - mnx) + 10.0, 0.5 * (mxy - mny) + 10.0, 0.5),
                       position=(cx, cy, -0.5), motion="static", material=ground_m)  # fmt: skip
    h = world.building_height
    for x0, y0, x1, y1 in obstacle_boxes(world.occ, world.bounds):
        w.add_box((0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * h),
                  position=(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * h), motion="static",
                  material=wall_m)  # fmt: skip
    return w, ground


class _Jolt:
    """A Jolt vehicle per car, each driven through ``physics_world.drive`` (the driver model takes
    the acceleration along the vehicle's heading; it brakes on a negative one and never backs up):
    what a pycvc without nav_sim_world_override drives on ``jolt``."""

    def __init__(self, cars, starts, headings):
        self.cars = cars
        self.world, _ = _physics_world(cars.world)
        self.ids, self.rear = [], []
        for i in range(cars.n):
            d = getattr(_pycvc.vehicle_desc, cars.vehicles[i].name)()
            axles = [d.axle(k) for k in range(d.axle_count())]
            rear_x = min(a["x"] for a in axles if not a["steered"])
            rest = max(-a["z"] + a["suspension_max"] + a["wheel_radius"] for a in axles) + 0.03
            c, s = math.cos(headings[i]), math.sin(headings[i])
            x, y = starts[i]
            origin = (x - c * rear_x, y - s * rear_x, cars.world.z(x, y) * 0.0 + rest)
            q = (math.cos(0.5 * headings[i]), 0.0, 0.0, math.sin(0.5 * headings[i]))
            self.ids.append(self.world.add_vehicle(d, position=origin, rotation=q))
            self.rear.append(rear_x)
        self.steps = int(round(DT / PHYSICS_DT))

    def command(self, steer, accel):
        for i, vid in enumerate(self.ids):
            self.world.drive(vid, steer=float(steer[i]), accel=float(accel[i]))
        self.world.step(self.steps)

    def read(self):
        n = self.cars.n
        out = [np.zeros(n) for _ in range(7)]
        for i, vid in enumerate(self.ids):
            vs = self.world.vehicle_state(vid)
            th = _yaw(vs["rotation"])
            px, py, _ = vs["position"]
            vx, vy, _ = vs["linear_velocity"]
            fwd = vx * math.cos(th) + vy * math.sin(th)
            vals = (px + self.rear[i] * math.cos(th), py + self.rear[i] * math.sin(th), th, fwd,
                    vs["steer"], vx, vy)  # fmt: skip
            for k, v in enumerate(vals):
                out[k][i] = v
        return tuple(out)


class _Sim:
    """libcvc's sim_world over the world's grid with the plant attached: your command goes in its
    navigator's place (nav_sim_world_override), or with ``navigator`` its navigator drives."""

    def __init__(self, cars, starts, headings, navigator):
        from grl_snam import nav_native

        c, w = cars, cars.world
        self.cars, self.navigator = c, navigator
        unit = c.unit
        self.scale = 0.05 * 4.84 / unit  # the navigator's metric, as nav's Humvee demos at 0.05
        self.cx, self.cy = w.center
        S = self.scale
        rows, cols = w.occ.shape
        occ = np.ascontiguousarray(w.occ, np.uint8)
        on = np.array([((p[0] - self.cx) * S, (p[1] - self.cy) * S) for p in starts], np.float32)
        goal = np.array([((p[0] - self.cx) * S, (p[1] - self.cy) * S) for p in c.goals],
                        np.float32)  # fmt: skip
        # The top speed holds a command; the navigator cruises at about 0.4 of it.
        vmax = float(np.max(c.max_speed)) * (0.4 if navigator else 1.0)
        L = float(np.mean(c.wheelbase))
        lock = float(np.min(c.max_steer))
        a_max = 0.8 * float(np.min(c.max_accel))
        self.h = _pycvc.nav_sim_world_create(
            occ, occ, "", on, goal, np.ones((c.n, 3), np.float32), rows, cols, *w.bounds,
            self.cx, self.cy, S, 12.0 * unit, 120, 2 * math.pi, 0.25 * unit * S, 0.62 * unit * S,
            DT, vmax * S, L * S, lock, a_max * S, 0.5 * 9.80665 * S, 0.8, NSUB, 1, c.reach * S,
            4, 1, 2.2, -1.4, 8.0, 1, 0.5, 0.15, 4.0)  # fmt: skip
        if HAS_SET_POSE:
            for i in range(c.n):
                _pycvc.nav_sim_world_set_pose(self.h, i, float(on[i, 0]), float(on[i, 1]),
                                              float(headings[i]))  # fmt: skip
        self.kind = c.kind
        self.plant = None
        self.steer = np.zeros(c.n)
        if c.kind != "kinematic":
            names = [v.name for v in c.vehicles]
            self.plant = nav_native.make_plant(
                f"{c.kind}:{','.join(names)}", bounds=w.bounds, obstacles=w.occ > 0,
                obstacle_height_m=w.building_height, fixed_dt=PHYSICS_DT,
            ) if c.kind == "jolt" else nav_native.make_plant(
                f"{c.kind}:{','.join(names)}", bounds=w.bounds, obstacles=w.occ > 0,
                obstacle_height_m=w.building_height, substep=PHYSICS_DT,
            )  # fmt: skip
            _pycvc.nav_sim_world_set_plant(self.h, self.plant.plant)

    def retarget(self, goals):
        S = self.scale
        for i, (gx, gy) in enumerate(goals):
            _pycvc.nav_sim_world_retarget(self.h, i, (gx - self.cx) * S, (gy - self.cy) * S)

    def command(self, steer, accel):
        _pycvc.nav_sim_world_override(self.h, np.asarray(steer, float),
                                      np.asarray(accel, float) * self.scale)  # fmt: skip
        _pycvc.nav_sim_world_step(self.h)
        self.steer = np.asarray(steer, float).copy()

    def step_navigator(self):
        _pycvc.nav_sim_world_step(self.h)

    def read(self):
        pos, head, speed = _pycvc.nav_sim_world_snapshot(self.h)[:3]
        x, y = pos[:, 0].astype(float), pos[:, 1].astype(float)
        th, sp = head.astype(float), speed.astype(float)
        steer = self.steer.copy()
        vx, vy = sp * np.cos(th), sp * np.sin(th)
        if self.plant is not None:
            for i in range(self.cars.n):
                st = self.plant.state(i)
                if "steer" in st:
                    steer[i] = float(st["steer"])
                if self.kind == "jolt":
                    v = self.plant.vehicle_state(i)["linear_velocity"]
                    vx[i], vy[i] = v[0], v[1]
                else:
                    b = self.plant.body(i)
                    c, s = math.cos(b["psi"]), math.sin(b["psi"])
                    vx[i], vy[i] = b["vx"] * c - b["vy"] * s, b["vx"] * s + b["vy"] * c
                    steer[i] = float(b.get("delta", steer[i]))
        return x, y, th, sp, steer, vx, vy


# ── the cars ───────────────────────────────────────────────────────────────────────────────────


class Cars:
    """``n`` cars on ``plant`` in ``world``, each with a random start and a goal, driven by
    :meth:`step` (your ``(steer, accel)``) or, with ``navigator``, :meth:`step_navigator`.

    ``seed`` makes the starts and goals repeatable. ``rays`` and ``range_max`` (car lengths) shape
    the range readings. ``bundle`` is the Austin bundle, or ``city`` the bundle as already
    loaded (``world="austin"``; :func:`austin_world`). ``backend`` forces ``"sim"``
    (libcvc's sim_world) or ``"direct"`` (numpy bicycle / ``physics_world.drive``); by default
    it is ``"sim"`` when the pycvc has ``nav_sim_world_override`` (or ``navigator``), else
    ``"direct"``."""

    def __init__(self, n=6, plant=None, world="flat", *, seed=0, navigator=False, rays=16,
                 range_max=12.0, bundle=None, city=None, backend=None, headings="goal"):  # fmt: skip
        if not HAS_PYCVC:
            raise RuntimeError(
                "grl_snam.cars needs pycvc with cvc::nav (cvcpkg install pycvc-gl-cp312)"
            )
        self.n = int(max(1, n))
        self.plant = plant or default_plant()
        self.kind, presets = parse_plant(self.plant)
        self.vehicles = [vehicle(presets[i % len(presets)]) for i in range(self.n)]
        if self.kind == "jolt" and not has_jolt():
            raise RuntimeError("this pycvc has no Jolt physics: use plant='kinematic'")
        if backend is None:
            backend = "sim" if (navigator or HAS_OVERRIDE) else "direct"
        if backend == "sim" and not navigator and not HAS_OVERRIDE:
            raise RuntimeError(
                "your own commands on libcvc's sim_world need a pycvc with nav_sim_world_override "
                "(libcvc after 3.7.11)"
            )
        if self.kind == "dynamic" and (backend != "sim" or not HAS_DYN):
            raise RuntimeError(
                "the dynamic plant needs a pycvc with dyn_bike_plant and, for your own commands, "
                "nav_sim_world_override (libcvc after 3.7.11)"
            )
        self.backend_name, self.navigator = backend, bool(navigator)
        self.world_name = world
        self.unit = max(v.length for v in self.vehicles)
        self.world = make_world(world, self.unit, bundle, city)
        self.reach = 1.5 * self.unit
        self.rng = np.random.default_rng(seed)
        self.ray_angles = np.linspace(-math.pi, math.pi, int(rays), endpoint=False)
        self.range_max = float(range_max) * self.unit
        for k in ("length", "width", "wheelbase", "max_steer", "max_accel", "max_decel",
                  "max_speed"):  # fmt: skip
            setattr(self, k, np.array([getattr(v, k) for v in self.vehicles], float))
        self._headings = headings
        self.reset()

    # ── the episode ──
    def reset(self, seed=None):
        """New starts and goals (with ``seed``, a new random stream), everything at rest."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        clear = 1.2 * self.unit
        self.starts = self.world.free_points(self.rng, self.n, clear, 3.0 * self.unit)
        self.goals = self._new_goals(self.starts)
        if self._headings == "goal":
            d = self.goals - self.starts
            headings = np.arctan2(d[:, 1], d[:, 0])
        else:
            headings = np.full(self.n, float(self._headings))
        if self.backend_name == "sim":
            self._b = _Sim(self, self.starts, headings, self.navigator)
        elif self.kind == "jolt":
            self._b = _Jolt(self, self.starts, headings)
        else:
            self._b = _Kinematic(self, self.starts, headings)
        self.t, self.tick = 0.0, 0
        self.last_command = (np.zeros(self.n), np.zeros(self.n))
        self.collisions = np.zeros(self.n, int)
        self.arrivals = 0
        self._state = self._read()
        return self._state

    def _new_goals(self, near):
        span = self.world.bounds[2] - self.world.bounds[0]
        return self.world.free_points(self.rng, self.n, 1.2 * self.unit, 4.0 * self.unit,
                                      avoid=near, avoid_dist=0.35 * span)  # fmt: skip

    def new_goals(self, which=None):
        """Give every car (or the cars in ``which``, a bool mask or indices) a new goal. A car at
        its goal that gets a new one counts in :attr:`arrivals`."""
        fresh = self._new_goals(self._state.pos)
        mask = np.ones(self.n, bool) if which is None else np.zeros(self.n, bool)
        if which is not None:
            mask[which] = True
        self.arrivals += int((mask & self._state.reached).sum())
        self.goals = np.where(mask[:, None], fresh, self.goals)
        if isinstance(self._b, _Sim):
            self._b.retarget(self.goals)
        self._state = self._read()
        return self._state

    # ── stepping ──
    def step(self, steer, accel) -> CarsState:
        """One tick: each car's plant takes ``steer`` (rad, + left; held to its lock) and ``accel``
        (m/s^2 along the heading; held to [-max_decel, max_accel]). Returns the new state."""
        if self.navigator:
            raise RuntimeError("these cars run libcvc's navigator: step_navigator(), or "
                               "Cars(..., navigator=False) for your own commands")  # fmt: skip
        steer = np.clip(np.nan_to_num(np.asarray(steer, float).reshape(self.n)), -self.max_steer,
                        self.max_steer)  # fmt: skip
        accel = np.clip(np.nan_to_num(np.asarray(accel, float).reshape(self.n)), -self.max_decel,
                        self.max_accel)  # fmt: skip
        self._b.command(steer, accel)
        self.last_command = (steer, accel)
        return self._advance()

    def step_navigator(self) -> CarsState:
        """One tick of libcvc's navigator driving every car to its goal (a baseline)."""
        if not self.navigator:
            raise RuntimeError("Cars(..., navigator=True) runs the navigator")
        self._b.step_navigator()
        self.last_command = (np.full(self.n, np.nan), np.full(self.n, np.nan))
        return self._advance()

    def _advance(self):
        self.t += DT
        self.tick += 1
        self._state = self._read()
        self.collisions += self._state.collided
        return self._state

    def state(self) -> CarsState:
        return self._state

    # ── reading ──
    def _read(self) -> CarsState:
        x, y, th, sp, steer, vx, vy = (np.asarray(a, float) for a in self._b.read())
        d = self.goals - np.stack([x, y], 1)
        dist = np.hypot(d[:, 0], d[:, 1])
        return CarsState(
            t=self.t, tick=self.tick, dt=DT, x=x, y=y, heading=th, speed=sp, steer=steer,
            vx=vx, vy=vy, goal=self.goals.copy(), goal_dist=dist, reached=dist < self.reach,
            ranges=self._ranges(x, y, th), ray_angles=self.ray_angles.copy(),
            range_max=self.range_max, collided=self._collided(x, y, th),
            length=self.length, width=self.width, wheelbase=self.wheelbase,
            max_steer=self.max_steer, max_accel=self.max_accel, max_decel=self.max_decel,
            max_speed=self.max_speed, reach=self.reach, plant=self.plant, world=self.world_name,
        )  # fmt: skip

    def _ranges(self, x, y, th):
        """(n, R) the distance along each ray from the body's centre to a building, ray-marched over
        the grid at half a cell, up to range_max."""
        h = 0.5 * self.wheelbase
        cx, cy = x + h * np.cos(th), y + h * np.sin(th)
        step = 0.5 * self.world.cell
        d = np.arange(step, self.range_max + 1e-9, step)
        ang = th[:, None] + self.ray_angles[None, :]
        px = cx[:, None, None] + np.cos(ang)[:, :, None] * d[None, None, :]
        py = cy[:, None, None] + np.sin(ang)[:, :, None] * d[None, None, :]
        hit = self.world.occupied(px, py)
        first = np.where(hit.any(axis=2), hit.argmax(axis=2), len(d))
        return np.where(first < len(d), d[np.minimum(first, len(d) - 1)], self.range_max)

    def _collided(self, x, y, th):
        """Touching a building (a point of the body's outline on one) or another car (their boxes
        overlap, by the separating axis test)."""
        n = self.n
        c, s = np.cos(th), np.sin(th)
        h = 0.5 * self.wheelbase
        cx, cy = x + h * c, y + h * s
        hl, hw = 0.5 * self.length, 0.5 * self.width
        u = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        pts_x = np.concatenate([cx[:, None] + (u * hl[:, None]) * c[:, None] + hw[:, None] * s[:, None] * sgn
                                for sgn in (-1.0, 1.0)], 1)  # fmt: skip
        pts_y = np.concatenate([cy[:, None] + (u * hl[:, None]) * s[:, None] - hw[:, None] * c[:, None] * sgn
                                for sgn in (-1.0, 1.0)], 1)  # fmt: skip
        out = self.world.occupied(pts_x, pts_y).any(axis=1)
        for i in range(n):
            for j in range(i + 1, n):
                if math.hypot(cx[i] - cx[j], cy[i] - cy[j]) > hl[i] + hl[j] + hw[i] + hw[j]:
                    continue
                if _boxes_overlap((cx[i], cy[i], c[i], s[i], hl[i], hw[i]),
                                  (cx[j], cy[j], c[j], s[j], hl[j], hw[j])):  # fmt: skip
                    out[i] = out[j] = True
        return out


def _boxes_overlap(a, b):
    """Two oriented rectangles (cx, cy, cos, sin, half length, half width) overlap."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    for ax, ay in ((a[2], a[3]), (-a[3], a[2]), (b[2], b[3]), (-b[3], b[2])):
        ra = a[4] * abs(a[2] * ax + a[3] * ay) + a[5] * abs(-a[3] * ax + a[2] * ay)
        rb = b[4] * abs(b[2] * ax + b[3] * ay) + b[5] * abs(-b[3] * ax + b[2] * ay)
        if abs(dx * ax + dy * ay) > ra + rb:
            return False
    return True


# ── logging ────────────────────────────────────────────────────────────────────────────────────

#: What :class:`CarsLog` records each tick, per car.
LOG_FIELDS = ("x", "y", "heading", "speed", "steer", "vx", "vy", "goal_x", "goal_y", "goal_dist",
              "reached", "collided", "cmd_steer", "cmd_accel")  # fmt: skip


class CarsLog:
    """Every tick's state and command, per car: :meth:`record` after each step, then :meth:`save`
    (``.npz``: arrays (ticks, n) named as :data:`LOG_FIELDS`, plus ``t``; ``.csv``: one row per car
    per tick)."""

    def __init__(self):
        self.t, self.rows = [], {k: [] for k in LOG_FIELDS}
        self.meta = {}

    def record(self, s: CarsState, command=None):
        steer, accel = command if command is not None else (np.full(s.n, np.nan),) * 2
        vals = {"x": s.x, "y": s.y, "heading": s.heading, "speed": s.speed, "steer": s.steer,
                "vx": s.vx, "vy": s.vy, "goal_x": s.goal[:, 0], "goal_y": s.goal[:, 1],
                "goal_dist": s.goal_dist, "reached": s.reached, "collided": s.collided,
                "cmd_steer": steer, "cmd_accel": accel}  # fmt: skip
        self.t.append(s.t)
        for k in LOG_FIELDS:
            self.rows[k].append(np.asarray(vals[k], float).copy())
        if not self.meta:
            self.meta = {"plant": s.plant, "world": s.world, "dt": s.dt, "n": s.n}

    def __len__(self):
        return len(self.t)

    def arrays(self) -> dict:
        out = {k: np.array(v) for k, v in self.rows.items()}
        out["t"] = np.array(self.t)
        return out

    def save(self, path) -> str:
        path = str(path)
        if path.lower().endswith(".csv"):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tick", "t", "car", *LOG_FIELDS])
                for k, t in enumerate(self.t):
                    for i in range(len(self.rows["x"][k])):
                        w.writerow(
                            [k, f"{t:.4f}", i, *(f"{self.rows[f][k][i]:.6g}" for f in LOG_FIELDS)]
                        )
        else:
            np.savez_compressed(
                path, **self.arrays(), **{f"meta_{k}": v for k, v in self.meta.items()}
            )
        return path
