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
- `live` and `fleet` also need libcvc's multi-car template's pieces, `pycvc_gl.cars` (pycvc-gl
  3.7.12+): its settings, worlds, panels and log ([Settings, worlds and plants on `live` and
  `fleet`](#settings-worlds-and-plants-on-live-and-fleet)). With an older pycvc_gl they say what
  to install.
- torch (every port but `live austin-planner` and `cars`).
- Austin, for `live`, `squad` and `fleet`:
  - `live`'s and `fleet`'s documents name their city themselves: the `scene-austin-south-small`
    package on cvcpkg.org, which the host downloads once (libcvc's Ariadne data,
    transfix/libcvc#998). `--uri` or `--bundle DIR` (or `GRL_SNAM_SCENE_BUNDLE`) points them at
    another package or a bundle on disk.
  - `squad` finds a bundle as `grl-snam demo` does: `GRL_SNAM_SCENE_BUNDLE`, else
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
python -m grl_snam.ariadne live austin-freedrive --plant humvee      # on a Jolt Humvee
python -m grl_snam.ariadne live austin-freedrive --world generated   # in a generated world
python -m grl_snam.ariadne live austin-freedrive --headless --frames 900 --log run.npz
```

The demo modules in `grl_snam.demos` run unchanged: `setup()` builds the scene and `step(dt)`
drives it, through an Ariadne host instead of VolRover3 or a bare window. The document adds:
- the navigator's HUD as panels: its NavMetrics (at `hud.*`), clearance and speed plots,
  progress, and the policy's coefficients as plots;
- the pictures around the vehicle: the signed distance with its predicted path, and where the wall
  barrier reaches;
- pause, sim speed and camera controls, and libcvc's settings panel.

On Austin the box becomes a Humvee. The published policy's metric (scale 1/300) lets it reach
270 m/s on Austin; the Sim speed slider goes down to 0.1x, and `nav.vehicle.vmax` caps it.

`austin-freedrive` also runs:
- **in another world:** another city (`--uri`), libcvc's terrain generator (`--world generated`)
  or the flat square (`--world flat`). Its start and goal are picked far apart there.
- **on a plant:** `--plant kinematic`, `humvee`, `roboracer`, `dynamic:humvee`. libcvc's
  navigator then drives a vehicle in its sim_world from the same start to the same goal, with
  the same policy (`grl-snam-weights`, exported to `.cvcnav`, in its trained frame). Between
  plants the Plant combo switches live.

The other demos stay on Austin and on their own stepping:
- `austin-learned`: its A* spine and corner goals are placed for Austin, and its corridor
  follower is its own Python on the torch rollout.
- `austin-patrol`: it plays back an A* loop with no navigator and no physics.
- `austin-planner`: it has an analytic world of its own.

Every demo but `austin-planner` runs `--headless` too (a lab that draws nothing).
`--log run.npz` writes what the `cars` template writes, one row a frame (on a plant, one a tick):
states, coefficients, the rasters every `--raster-every` frames with the predicted path, and
every setting in `meta`. The exit code is 0, 1 when the run failed, and 3 below `--min-arrivals`
(1 for reaching the goal).

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

### `fleet`: a live squad of Humvees, on the plant and in the world you pick

```bash
python -m grl_snam.ariadne fleet                         # Austin; GRL-SNAM's torch squad
python -m grl_snam.ariadne fleet --plant humvee          # libcvc's sim_world, Jolt Humvees
python -m grl_snam.ariadne fleet --plant dynamic:humvee  # the dynamic single-track Humvee
python -m grl_snam.ariadne fleet --plant kinematic       # libcvc's sim_world, nav's bicycle
python -m grl_snam.ariadne fleet --world generated --set world.seed=4   # a generated world
python -m grl_snam.ariadne fleet --headless --frames 600 --log run.npz --min-arrivals 2
```

`grl_snam.squad.Squad` run live, each agent starting blind, with New goals and Restart.
- **The plant** is `nav_plant.vehicle` (the Plant combo, `--plant`, `--set`):
  - `torch`, the default: GRL-SNAM's Python squad, the path that trains;
  - any plant libcvc's `plant_host` names: `kinematic`, `humvee`, `roboracer`,
    `dynamic:humvee`, `dynamic4w:humvee`. The squad then runs in libcvc's sim_world, each agent's
    belief its own plane. On a physics plant the agents drive vehicle routes planned on their own
    beliefs.

  Between libcvc's plants the switch is live: the agents keep their poses. To or from `torch`
  the squad is rebuilt. The old names still work (`jolt:humvee` is `humvee`).
- **The world:** Austin from the document's `data:` block, another city (`--uri`), libcvc's
  terrain generator (`--world generated`), or the flat square (`--world flat`). The squad, the
  physics and the scene take the same one.
- **The panels:**
  - a row per agent;
  - the chosen agent's (`view.car`) coefficients (alpha, beta, gamma, and its clearance);
  - its pictures: the signed distance with its planned route, and its belief (fog of war: grey
    where it has not looked);
  - libcvc's settings and the generator.
- **Headless:** `--log run.npz` writes what the `cars` template writes, a row a tick. The exit code
  is 0, 1 when the run failed, and 3 below `--min-arrivals` (agents that arrived).

Measured on the first fleet with 4 agents over 18 s: torch moves all four (~10 m/s), kinematic all
four (14-18 m/s). On Jolt Humvees two stayed stuck in wall-escape mode before vehicle routes (the
navigator is sized for the story's bodies, not a 4.7 m Humvee). So torch is the default.

### Settings, worlds and plants on `live` and `fleet`

`live` and `fleet` take libcvc's multi-car template's settings, so they are set as in `cars`:
- in the document's `init:`;
- on the sliders (libcvc's Settings panel, `cars_panels.ari`);
- with `--set KEY=VALUE` / `--params FILE.json`;
- from Python (`host.set`).

On libcvc's sim_world (fleet on a plant, `live austin-freedrive` on a plant) every `nav.*`,
`nav.vehicle.*` and `nav_plant.*` key applies, as libcvc reads it. libcvc's
[docs/CARS.md](https://github.com/transfix/libcvc/blob/master/docs/CARS.md#settings-the-state-tree)
lists them with their units and ranges.

On GRL-SNAM's torch navigator (`SdfNavigator`, normalized units) the keys that mean the same
thing are mapped onto it (`grl_snam/ariadne/_cars.py`). A length or speed is metres in the tree
and times the navigator's scale on it:

| key | torch `fleet` (bicycle) | `live austin-freedrive` (point) | `live austin-learned` | what it sets on the torch navigator |
|---|---|---|---|---|
| `nav.vehicle.vmax` | yes | yes | yes | `kw["vmax"]`, the top speed |
| `nav.vehicle.rr` | yes | yes | yes | `kw["rr"]`, the footprint disc |
| `nav.vehicle.d_hat` | yes | yes | yes | `kw["d_hat"]`, the barrier's range |
| `nav.reach_tol` | yes | yes | | `reach_tol`, the arrival radius |
| `nav.vehicle.a_max`, `a_lat_max` | yes | | | the bicycle's acceleration and cornering |
| `nav.vehicle.wheelbase`, `delta_max`, `k_steer` | yes | | | the bicycle's `L`, steering lock, wall steering |
| `nav.vehicle.allow_reverse` | yes | | | whether it may back up |
| `nav.range_m`, `nav.sense_every` | yes | | | the scenario's sensor range and ticks between looks |

The rest are libcvc's alone and do nothing on the torch navigator: `nav_plant.*`, the risk
governor, `nav.yield.*` and the multi-disc footprint. `austin-patrol` (a route played back) and
`austin-planner` (its own coefficients) take none.

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
| `Squad(plant=...)`, `--plant` on the recorders | library, `capture --plant` | `fleet --plant humvee / dynamic:humvee / kinematic` (any plant libcvc's `plant_host` names), `squad --plant` |
| `grl-snam material-demo` | two PNGs | `material` |
| `grl-snam capture drive / multigoal` | mp4 | covered by `live austin-freedrive` / `austin-learned` (the same navigator, live) |
| `grl-snam pipeline`, `build-sdf`, `train` | batch | not ported: no picture of their own |
| `wall_vis` | a post-pass for the finale | not ported |
| `*_bench` | console timings | not ported |
| `selftest`, `eval`, `experiments/e*` | console / paper figures | not ported |
| (new) your own multi-car controller | | `cars` (libcvc's template: `--copy DIR`) |
