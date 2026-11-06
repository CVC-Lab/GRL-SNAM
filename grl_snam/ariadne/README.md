# GRL-SNAM on Ariadne

GRL-SNAM's demos as Python hosts of Ariadne (`.ari`) documents. The document (`docs/*.ari`) holds
the panels, the controls and the look: a pale sky, a low warm three-point sun with a cool fill,
and soft-edged shadows that follow the camera. The port (`grl_snam/ariadne/*.py`) runs the demo,
or replays its recording, draws it into the document's scene, and writes what the panels show.
Vehicles on Austin are Humvees at their real size (4.7 m; the `vehicle-humvee` package's model,
else a box that size).

```bash
grl-snam ariadne --list               # or: python -m grl_snam.ariadne --list
grl-snam ariadne <port> --help
```

Every port takes `--offscreen` (GL, no window), `--headless` (no GL: the readouts only),
`--frames N` (stop after N frames) and `--png PATH` (save the last frame).

**Testing a multi-car controller of your own?** Start from the `cars` template:
[Write your own multi-car controller](#write-your-own-multi-car-controller).

## Requirements

- pycvc_gl with its Python host, `pycvc_gl.ariadne` (libcvc 3.7.9+, transfix/libcvc#985). With an
  older pycvc_gl the ports refuse to start and say so; their tests skip.
- torch (every port but `live austin-planner` and `cars`).
- Austin, for `live`, `squad` and `fleet`:
  - `live`'s document names its city itself: the `scene-austin-south-small` package on
    cvcpkg.org, which the host downloads once (libcvc's Ariadne data, transfix/libcvc#998).
  - `squad` and `fleet` find a bundle as `grl-snam demo` does: `GRL_SNAM_SCENE_BUNDLE`, else
    `<prefix>/share/cvc-scenes/austin_south` (`scene-austin-south`, or
    `scene-austin-south-small`).
  - The Humvee comes from the city's package when it carries one (the small one does), else
    `vehicle-humvee` (`share/cvc-scenes/shared/Humvee.glb`) or the bundle's own copy.
- The published policy (`grl-snam-weights`) for `live austin-freedrive` / `austin-learned`; `fleet`
  uses it when installed.

```bash
cvcpkg install grl-snam-cp312 scene-austin-south vehicle-humvee grl-snam-weights --prefix <prefix>
```

## The ports

### `live`: the live Austin demos

```bash
python -m grl_snam.ariadne live austin-freedrive   # the learned SDF navigator, no route
python -m grl_snam.ariadne live austin-learned     # an A* spine + learned SDF control
python -m grl_snam.ariadne live austin-patrol      # a grounded A* patrol
python -m grl_snam.ariadne live austin-planner     # the surrogate planner (needs vtkmodules)
```

The demo modules in `grl_snam.demos` run unchanged: `setup()` builds the scene and `step(dt)`
drives it, through an Ariadne host instead of VolRover3 or a bare window. The document adds the
navigator's HUD as panels (its NavMetrics, clearance and speed plots, progress), and pause,
sim speed and camera controls. On Austin the box becomes a Humvee. Windowed or `--offscreen`
only. The published policy's metric (scale 1/300) lets it reach 270 m/s on Austin; the Sim speed
slider goes down to 0.1x.

The city is the document's (`docs/live.ari`'s `data:` block):
- **Where it comes from:** the host downloads the package once, and later runs read the local
  cache. It unpacks the package into the state tree and decodes it.
- **What draws it:** the scene's `city` node draws the terrain, with the package's aerial image
  draped over it, and the buildings.
- **What the demo reads:** the same objects, through `DemoHost.city`. Its terrain sampler,
  occupancy grids and SDF come from that one load, so no bundle file is read twice.
- **A local bundle instead:** `--bundle DIR` (or `GRL_SNAM_SCENE_BUNDLE`) points the document's
  city at a bundle on disk, with nothing downloaded.
- **When the package cannot load** (offline, no cache), the installed bundle stands in.
- **`austin-planner`** drives a world of its own, so the city never loads for it.

### `fog`: the fog-of-war stories

```bash
python -m grl_snam.ariadne fog ghost                 # ghost, blocker, city, traffic, pursuit, unit
python -m grl_snam.ariadne fog city --record         # record again first
```

Records the story into `traces/<story>` when that trace is missing or stale (as
`grl-snam fog record`), then replays it with `grl_snam.fog_scene`, the scene the capture and the
live player draw, seen from above. The document is the transport (play, speed, loop, scrubbing),
the recording's captions and the vehicle's HUD. Overview or follow camera.

### `squad`: squad recordings on Austin

```bash
python -m grl_snam.ariadne squad finale-rendezvous   # finale act one
python -m grl_snam.ariadne squad finale-pursuit      # finale act two
python -m grl_snam.ariadne squad convoy
python -m grl_snam.ariadne squad pursuit
python -m grl_snam.ariadne squad convoy --plant jolt:humvee   # recorded on a vehicle plant
```

Records the run into `traces/<run>` when it is missing (minutes; as `finale_record`,
`convoy_record`, `pursuit_record` do), then replays the recorded poses in 3-D Austin: a Humvee per
agent seated at its recorded heading, a pole in its colour, its goal (a pyramid) and its track.
`--plant` records on libcvc's sim_world and that plant (`kinematic`, `jolt:humvee`,
`dynamic:humvee`; [docs/PLANTS.md](../../docs/PLANTS.md)) into `traces/<run>-<plant>`.
The panels are the transport, the captions, a row per agent and the first agent's HUD.

### `fleet`: a live squad of Humvees on Austin

```bash
python -m grl_snam.ariadne fleet                      # GRL-SNAM's torch squad
python -m grl_snam.ariadne fleet --plant jolt:humvee     # libcvc's sim_world, Jolt Humvees
python -m grl_snam.ariadne fleet --plant dynamic:humvee  # the dynamic single-track Humvee
python -m grl_snam.ariadne fleet --plant kinematic       # libcvc's sim_world, nav's bicycle
```

`grl_snam.squad.Squad` run live, each agent starting blind; the Plant combo switches the runtime
(`dynamic` needs a pycvc with `dyn_bike_plant`) and New goals deals new ones. Measured with 4 agents over 18 s: torch moves all four (~10 m/s),
kinematic all four (14-18 m/s), Jolt Humvees leave two stuck in wall-escape mode (the navigator
is sized for the story's bodies, not a 4.7 m Humvee), so torch is the default.

### `material`: material-aware navigation

```bash
python -m grl_snam.ariadne material
```

`grl-snam material-demo`'s two stories (the fog scenario and the reactive swarm), each with
material off and on, run as async functions on the app's background pool while the window
draws; each shows a picture (drawn with numpy, no matplotlib) and the off/on numbers.
The runs have no time cap, since each story ends on its own step budget. With
`--headless` or `--offscreen`, a run that does not finish ends the port with exit code 1 and
the reason.

### `cars`: N Ackermann cars driven by your own controller

```bash
python -m grl_snam.ariadne cars                          # 6 cars on flat ground
python -m grl_snam.ariadne cars --plant kinematic --world blocks --cars 12
python -m grl_snam.ariadne cars --copy my_cars           # the template, to edit
```

A template rather than a demo: see [Write your own multi-car controller](#write-your-own-multi-car-controller)
below. It needs no torch and no Austin (Austin is `--world austin`).

## Write your own multi-car controller

The `cars` port is a template to copy: one Python file (`cars.py`) and one Ariadne document
(`cars.ari`). It drives N Ackermann cars with a controller of yours, on nav's kinematic
bicycle, on Jolt vehicles, or on libcvc's dynamic single track. It runs with nothing beyond
`pycvc_gl` (no torch, no scene download).

```bash
cvcpkg install grl-snam-cp312 --prefix ./cars-env     # or cp311 / cp313 for your Python
. ./cars-env/bin/activate
grl-snam ariadne cars --copy my_cars                  # writes my_cars/cars.py and my_cars/cars.ari
python my_cars/cars.py                                # a window: 6 cars, the default controller
```

### Your controller

Edit `controller(s, p)` at the top of `cars.py`. It runs every tick (0.05 s), once for all the
cars, and returns `(steer, accel)`: one front-wheel angle (rad, + left) and one acceleration
along the heading (m/s²) per car, as numpy arrays. The cars' plants take them directly; no
navigator is involved.

`s` holds every car's state as (n,) arrays, in SI units:

| field | meaning |
|---|---|
| `x`, `y`, `heading` | the rear axle's midpoint (the point the bicycle turns about), and the heading (0 along +x, counter-clockwise) |
| `speed`, `vx`, `vy` | the speed along the heading (negative backing up), and the velocity in the world |
| `steer` | the front wheels' angle now; on Jolt it lags your command at the steering's rate |
| `goal` (n, 2), `goal_dist`, `reached` | each car's goal, the distance to it, and whether it is within `s.reach` |
| `ranges` (n, R), `ray_angles` (R,) | the distance to a building or wall along each ray, at angles off the heading, up to `range_max` |
| `collided` | touching a building or another car this tick |
| `max_steer`, `max_accel`, `max_decel`, `max_speed` | each car's limits (your command is held to them) |
| `length`, `width`, `wheelbase` | each car's size |
| `t`, `dt`, `n`, `pos` (n, 2) | the time, the tick, the number of cars, the positions |

The other cars are the rest of the arrays: car i's neighbours are every j != i. The shipped
controller heads for the goal, keeps a spacing from the other cars, turns away from walls,
slows for what is ahead, and backs out when wedged. It is a starting point, not a planner: in
the block city it can stall in a dead end.

`p` is your parameters. `PARAMS` in `cars.py` names them and their starting values, and
`cars.ari`'s Controller window has a slider for each, bound to `ctl.<name>`; moving a slider
changes `p["<name>"]` from the next tick. To add a parameter, add it to `PARAMS` and add a
`slider_float` bound to `ctl.<name>` in `cars.ari`.

### The panels and the buttons

- **Run**: the plant, the world, the number of cars, the controller (yours, or libcvc's navigator
  as a baseline), ticks per frame, a new goal on arrival, pause, the camera.
- **New goals** and **Reset** are document buttons that call back into Python (`host.on`).
  Reset draws new starts and goals.
- **Cars**: the time, the arrivals, how many car-ticks touched something, a row per car (speed,
  wheel angle, your last acceleration, distance to goal), and two plots: the chosen car's speed
  and the closest gap between two cars.

### The plant is the host's choice

Physics comes from the plant `cars.py` builds (`Cars(n, plant, world)` in `grl_snam.cars`), not
from the `.ari` document. The document has no physics in it: its Plant combo only writes
`run.plant`, which `cars.py` reads when it builds the cars. `--plant` sets the same thing on the
command line:

- `kinematic` (or `kinematic:<preset>`, for that car's size and limits): nav's kinematic bicycle.
- `jolt:roboracer` (the default when the pycvc has Jolt), `jolt:humvee`, or a comma list cycled
  over the cars (`jolt:roboracer,humvee`). A bare preset name is a Jolt plant.
- `dynamic:roboracer` / `dynamic:humvee`: libcvc's dynamic single track.

The RoboRacer is the default because it is the lab's own 1/10 Ackermann car: libcvc's preset
comes from the lab's configuration (0.324 m wheelbase, 0.425 rad lock, 6 m/s² either way,
10 m/s), so a controller tuned here starts from the hardware's geometry and limits. Sixteen fit an arena
and run in real time. The Humvee is a 4.8 m, 2.7 t truck with a slow steering (0.35 rad/s), for
Austin at its real size. The flat and block worlds scale to the largest car.

### What each plant means for your controller

- **Kinematic (nav's bicycle).** The wheels point where you steer at once, the speed integrates
  your acceleration exactly (between -0.25 and 1 times the top speed), and the yaw rate is
  `speed / wheelbase * tan(steer)`. Nothing slips and nothing collides: a car drives through a
  building or another car, and `collided` says so. Use it to get the logic right.
- **Jolt.** Each car is a rigid body on four wheels with suspension, an engine and gearbox, and
  brakes. Three things differ from the bicycle:
  - *The steering has a rate.* `s.steer` follows your command at the vehicle's steering rate:
    a RoboRacer reaches 90% of full lock in 0.15 s, a Humvee in 1.2 s. High steering gains that are
    stable on the bicycle can oscillate here.
  - *A driver model makes your acceleration.* It works throttle, brake and gear to give the
    acceleration you ask for; from rest it tracks it closely (2 m/s² gives 1 m/s at 0.50 s on a
    RoboRacer and 0.60 s on a Humvee, against the bicycle's 0.50 s), but the engine's torque
    and the tire's grip cap what it can deliver. There is no transport delay beyond the tick.
  - *The tires slip.* Below about 0.2 g of cornering a car turns as the bicycle predicts. Near
    the grip limit it turns less: measured at about 0.8 g on full lock, a RoboRacer at 3 m/s
    yaws at 0.64 of the bicycle's rate and a Humvee at 8 m/s at 0.85. Past the limit it slides,
    and braking distances grow with speed squared over the grip. Leave room in your spacing.

  Cars collide with the buildings and with each other.
- **Dynamic.** A single-track model with Pacejka tires, load transfer and actuator lag, a box per
  chassis in a Jolt world for collisions. It turns about as the Jolt car does near the limit.

**Backing up.** A negative acceleration from rest backs a car up on every plant. One exception:
a Jolt car on a pycvc without `nav_sim_world_override` (libcvc 3.7.11 and earlier). There,
`grl_snam.cars` drives Jolt through `physics_world.drive`, which only brakes. The Cars panel's
"commands via" line says which path is in use:
- `sim`: libcvc's sim_world, your command in its navigator's place (`nav_sim_world_override`);
- `direct`: the numpy bicycle, or `physics_world.drive`.

The dynamic plant needs the newer pycvc.

### Batch runs

```bash
python my_cars/cars.py --headless --frames 1200 --log run.npz      # 60 s, no GL
python my_cars/cars.py --headless --frames 1200 --plant jolt:humvee --cars 8 --log run.csv
```

`--headless` runs the cars and the panels' state with no window and no GL, so a server runs it
too. `--log` records every tick, per car: the state (`x`, `y`, `heading`, `speed`, `steer`, `vx`,
`vy`, `goal_x`, `goal_y`, `goal_dist`, `reached`, `collided`) and your command (`cmd_steer`,
`cmd_accel`, after the limits). In `.npz` each is an array (ticks, cars), with `t` and
`meta_plant` / `meta_world` / `meta_dt`. In `.csv` it is one row per car per tick. `--seed`
makes the starts and goals repeatable. `--regoal 0` keeps each car's first goal.

Or skip the document and drive `grl_snam.cars` from a script of your own:

```python
from grl_snam.cars import Cars, CarsLog

cars = Cars(8, plant="jolt:roboracer", world="blocks", seed=3)
log, s = CarsLog(), cars.state()
while s.t < 60.0:
    s = cars.step(*my_controller(s))
    log.record(s, cars.last_command)
log.save("run.npz")
```

`Cars(..., navigator=True)` with `cars.step_navigator()` runs libcvc's navigator (GRL-SNAM's, in
C++) as a baseline. On vehicles it does worse than a controller made for them. In 30 s on the
block city with Jolt RoboRacers, it reached 1 goal against the shipped controller's 8;
[docs/PLANTS.md](../../docs/PLANTS.md) says why.

## Every demo and its port

| GRL-SNAM demo | How it runs today | Ariadne port |
|---|---|---|
| `grl-snam demo austin-freedrive` | VolRover3 or a pycvc_gl window | `live austin-freedrive` |
| `grl-snam demo austin-learned` | VolRover3 or a pycvc_gl window | `live austin-learned` |
| `grl-snam demo austin-patrol` | VolRover3 or a pycvc_gl window | `live austin-patrol` |
| `grl-snam demo austin-planner` | VolRover3 or a pycvc_gl window | `live austin-planner` (needs vtkmodules) |
| `grl-snam demo lab` | VolRover3 only | not ported: it drives VolRover3's own scene |
| `grl-snam fog record / capture / play / all` | trace, mp4, window | `fog <story>` (records, then plays) |
| `finale` (act one, rendezvous) | mp4 | `squad finale-rendezvous` |
| `finale` (act two, pursuit) | mp4 | `squad finale-pursuit` |
| `convoy_record` | traces only | `squad convoy` |
| `pursuit_record` | traces only | `squad pursuit` |
| `squad_record` / `squad_capture` | traces, mp4 | `squad` (Austin runs); `fleet` (live) |
| `Squad(plant=...)`, `--plant` on the recorders | library, `capture --plant` | `fleet --plant jolt:humvee / dynamic:humvee / kinematic`, `squad --plant` |
| `grl-snam material-demo` | two PNGs | `material` |
| `grl-snam capture drive / multigoal` | mp4 | covered by `live austin-freedrive` / `austin-learned` (the same navigator, live) |
| `grl-snam pipeline`, `build-sdf`, `train` | batch | not ported: no picture of their own |
| `wall_vis` | a post-pass for the finale | not ported |
| `*_bench` | console timings | not ported |
| `selftest`, `eval`, `experiments/e*` | console / paper figures | not ported |
| (new) your own multi-car controller | | `cars` (a template: `--copy DIR`) |
