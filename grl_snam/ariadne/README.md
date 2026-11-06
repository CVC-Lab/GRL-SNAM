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

**Testing a multi-car controller of your own?** Start from the `cars` template, libcvc's
`pycvc_gl.cars`: [Write your own multi-car controller](#write-your-own-multi-car-controller).

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
grl-snam ariadne cars                                       # Austin, 6 Humvees
grl-snam ariadne cars --vehicle kinematic --world generated --cars 12
grl-snam ariadne cars --copy my_cars                        # the template, to edit
grl-snam ariadne cars --controller navigator --grl-policy   # GRL-SNAM's policy as the baseline
```

A template rather than a demo, and libcvc's, not GRL-SNAM's: `pycvc_gl.cars` (pycvc-gl 3.7.12+),
so any multi-car experiment can start from it. It needs no torch. See
[Write your own multi-car controller](#write-your-own-multi-car-controller) below.

## Write your own multi-car controller

`grl-snam ariadne cars --copy my_cars` (or `python -m pycvc_gl.cars --copy my_cars`) writes
libcvc's template: `cars.py`, with a `controller(s, p)` to edit, and `cars.ari`, its panels.
- **Each tick** (0.05 s), `controller(s, p)` gets every car's state as numpy arrays (SI units):
  pose, speed, wheel angle, velocity, goal, range rays, collisions, limits and sizes.
- **It returns** `(steer, accel)` per car, which each car's plant takes in the navigator's place.
- **The plant is a state-tree setting**, `nav_plant.vehicle`: `kinematic`, `humvee`,
  `roboracer`, a comma list, or `dynamic[:presets]`. The document's Vehicle combo, its `init:`,
  `--vehicle` or `--set` switch it live. Physics is the host's plant, not something in the `.ari`
  file.
- **So are the navigator's and the vehicles' settings** (`nav.*`, `nav.vehicle.*`, `nav_plant.*`)
  and your own parameters (`ctl.*`).
- **The world:** the document's `data:` block (Austin from cvcpkg.org by default, or another
  package or local bundle), libcvc's terrain generator (its parameters in a panel, recorded in the
  log), or a flat square.
- **Headless:** `--headless --frames N --log run.npz` records every tick's states and commands,
  the coefficients, the rasters and every setting, and sets the exit code.

libcvc's [docs/CARS.md](https://github.com/transfix/libcvc/blob/master/docs/CARS.md) is the full guide. It covers the state table, the worlds, what
kinematic vs Jolt vs dynamic means for a controller (the steering rate, the driver model, tire slip,
measured), every setting with its unit and range, the panels, and batch runs.

What GRL-SNAM adds is `--grl-policy`:
- **With it,** libcvc's navigator (GRL-SNAM's navigator, in C++) drives with the published policy
  (`grl-snam-weights`, which needs torch to export it to `.cvcnav`), in the frame it was trained in
  (1 unit = 300 m: `--nav-scale`).
- **Without it,** `--controller navigator` uses libcvc's built-in policy.

The published policy was trained on the point surrogate. On vehicles it does worse than a
controller made for them ([docs/PLANTS.md](../../docs/PLANTS.md#what-to-expect)).

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
| (new) your own multi-car controller | | `cars` (libcvc's template: `--copy DIR`) |
