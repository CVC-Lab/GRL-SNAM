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
- `scrub` deploys on libcvc's sim_world, so it needs `pycvc_gl.cars`, and torch for `.pt`
  checkpoints. `scrub --view` needs neither. The learning check's checkpoints (every one its
  retrains saved) are the `grl-snam-learning-checkpoints` package.

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

- **with several policies side by side** (`--arms`, on a plant). This is the Laboratory's
  comparison view; see [Arms](#arms-policies-side-by-side) below.

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

#### Arms: policies side by side

```bash
python -m grl_snam.ariadne live austin-freedrive --plant kinematic --arms trained,untrained,constant
python -m grl_snam.ariadne live austin-freedrive --plant humvee --arms trained,step0=run/coef.step000000.cvcnav \
    --arm flat=const:hist=run/history.jsonl
python -m grl_snam.ariadne live austin-freedrive --plant kinematic --world flat --headless --frames 900 \
    --arms trained,untrained,constant --set arms.goal=-20,-5 --retarget-at 12 --log arms.npz
```

Each arm drives its own libcvc sim_world, with the same world, start, goal and seed as the
others. All the worlds step tick for tick together, and each arm is a see-through ghost in its own
colour (its chart colour: blue, orange, green, ...).
- **An arm** is a bare name or what `grl-snam arms-eval --arm` takes, `NAME=PATH.cvcnav`,
  `NAME=PATH.pt` or `NAME=const:...` (`--arm` adds one more). The bare names are:
  - `trained`: the published policy, which `--plant` without `--arms` drives (needs
    `grl-snam-weights`);
  - `untrained`: a CoefMLP as `torch.manual_seed(--seed)` initializes it (`untrained=PATH` takes a
    run's step-0 checkpoint instead);
  - `constant`: `const:bias`, CoefMLP's output bias everywhere (alpha 1, beta 3, gamma 4).

  As in `arms-eval`, a constant arm takes its input layout from the first arm with a net. Each
  policy is exported to `.cvcnav` and loaded with `nav_sim_world_set_policy`. Every arm runs in
  the frame of the first `.pt` arm that records one, else the published policy's (1/300).
- **Retarget** (the Stats window's button, the Demo menu, or `--retarget-at S,...`) moves every
  arm's goal on the same tick. It goes to the Goal field's `x, y` (`arms.goal`) once that point
  is clear of buildings by the vehicle's half-width, read from the world's signed distance;
  otherwise it says why not. With the field empty, it goes to a free point far from every arm.
  Each Retarget starts a new leg: every arm's `nav_stats` collector restarts there.
- **Stats**: one row per arm for the current leg (Laboratory §6.5):
  - the minimum and mean clearance (m) and the ticks penetrating;
  - the path length (m) and the elapsed time (s, to the goal once there);
  - the wall time per tick (ms; the arms take turns stepping first) and whether it reached the
    goal.

  The rows come from libcvc's `nav_stats` (`nav_sim_world_begin_nav_stats`). The mean clearance
  is the drive telemetry's, and the wall time the port's own.
- **Ledger**, per arm:
  - the coefficients in force and the terms of U_nav, from `nav_sim_world_drive_telemetry`
    (every field; `pycvc_gl.cars` keeps only the coefficients);
  - the books against U_nav, summed over the run and for this tick: `dU`, `drive`, `projection`
    (`projection_work`), `context` (`context_work`: a Retarget books here), `dKE` and the
    `residual`, with `dU = drive + projection + context + residual`. The residual is zero while
    nothing moves the agent between ticks (`docs/vehicle-ph-plant.tex`, "Context work and
    projection work");
  - the plant ledger (libcvc's `plant_ledger.h`) for the arm the Agent combo picks: energy, tilt,
    load transfer, commands and contacts as a physics plant measures them. An item shows as n/a
    where it does not apply, which on the kinematic plant is every item.
- **Charts** (pycvc_gl's `chart:`, libcvc 3.7.14): every arm's clearance, the picked arm's
  cumulative books, and one book (the Book combo) summed for every arm. A marker shows where each
  leg starts. On an older pycvc_gl the charts are hidden and the rows show the same numbers.
- **Follow** picks the arm that the camera, the Navigator window and the rasters follow.
- **`--log run.npz`** writes a run log (`grl_snam.run_log/v1`, `grl_snam.learning.runlog`), which
  [`replay`](#replay-a-recorded-run-played-back) plays back. Per tick and arm it holds:
  - the pose, in the `cars` template's log fields (each arm a car);
  - the wall time and the vehicle's size;
  - every drive telemetry field, the books, the Stats row and the plant ledger's items.

  It also keeps the legs. Its `meta` has:
  - each arm's spec, its `.cvcnav`'s sha256 and its policy key;
  - the seed, start, goal and frame;
  - the keys that rebuild the world, and the settings;
  - the provenance: the GRL-SNAM version and commit, libcvc's version, and each `--history
    H.jsonl` (the training histories the arms come from) with its sha256.

  The exit code counts each arm's arrival at each leg's goal against `--min-arrivals`.

The windows are `docs/stats_panels.ari`'s units, which `docs/live.ari` imports; without `--arms`
they stay hidden. The torch path (`--plant torch`, the default) refuses `--arms` and says why: its
demo's state is module-level (`grl_snam.demos._common._STANDALONE` holds the one demo a process
runs), so a second arm would drive the first one's demo.

On open ground the policies tell apart only near walls and near the goal. Elsewhere every arm
accelerates at the plant's limit straight at its goal. The published policy sits close to
CoefMLP's bias (`docs/learning-validation.md`), so on Austin the rows part after tens of
seconds: `--set view.speed=4` gets there in 300 frames.

#### Traces: a session recorded and replayed exactly

```bash
python -m grl_snam.ariadne live austin-freedrive --plant kinematic --arms trained,untrained,constant \
    --record-trace session.trace
python -m grl_snam.ariadne live austin-freedrive --replay-trace session.trace
python -m grl_snam.ariadne live austin-freedrive --replay-trace session.trace --arms trained=other.cvcnav,untrained,constant
```

This is the Laboratory's live/replay harness (R49). It uses libcvc's input trace,
`cvc.input_trace/v1` (libcvc's `docs/INPUT_TRACE.md`, transfix/libcvc#1168). That needs a pycvc
after 3.7.14; with an older one, the flags say so.
- **What is recorded:** the session's inputs.
  - The starting state: the world, the start and goal, the settings, and the policies' sha256.
    The arms, the seed and the frame go in the header's `meta`.
  - Every write to the `nav`, `nav_plant`, `run` and `world` keys (a slider, `--set`), stamped
    with the world clock's tick.
  - Each Retarget, as nav's own `nav.retarget` event.
  - The state's hash after each frame.

  A recording steps on the app's world clock (a 0.01 s quantum, the navigator every 5 quanta),
  so every input carries the tick that consumed it. Without `--arms` it records one arm, the
  trained policy.
- **What a replay does:** it rebuilds the worlds from the trace and puts the clock in replay
  mode. It writes the recorded settings at their ticks, applies the events, and compares every
  hash.
  - Headless, it runs to the end and prints whether the replay matches and, if not, the tick
    where it departs. The exit code is 1 when the replay departs with the recorded policies.
  - In a window, it plays at the sim speed. The Trace window shows the hashes compared, and has
    a Position slider and Restart. `sim_world` has no save and restore, so a seek back rebuilds
    the worlds and runs again from tick 0.
- **`--arms` on a replay** runs other policies on the same inputs. The run is the recording's
  until the first tick where the policies' coefficients differ. The replay says where it
  departs, and exits 0.
- **A one-arm trace** is libcvc's own: `cvc trace verify session.trace` replays it headlessly.
  With several arms the hash is over every arm's world, which only this port replays.
- **v1 covers the kinematic plant.** libcvc refuses to describe a world on a vehicle plant: Jolt
  is deterministic only within one binary.

`tests/test_lab_acceptance.py` holds the Laboratory's acceptance tests 5 and 6 (§9):
- one Retarget moves every arm's goal on the same tick, and the Stats rows differ;
- the replay gives byte-identical poses and the same final hash;
- with another build of a policy, the run matches until the first tick where the coefficients
  differ.

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

### `coph`: CoPH-Stagewise's worlds in 3-D (demo 2a)

```bash
python -m grl_snam.ariadne coph crossing                       # live, seed 76000
python -m grl_snam.ariadne coph hazard --scene severe          # live, seed 79000
grl-snam coph crossing --out runs/crossing                     # every arm, headless (T1)
python -m grl_snam.ariadne coph crossing --from runs/crossing  # replay two of them
python -m grl_snam.ariadne coph crossing --headless --log run.npz --min-goals 4
```

The vendored crossing and hazard demos (`grl_snam.coph`, `grl-snam coph`), with communication off
on the left and CoPH on the right, on the same seed. Crossing shows `disabled` beside
`reciprocal_fixed`; hazard shows `none` beside `field` (`--arms A,B` picks others). The headline
gives the reference's established result with its seed counts, this run's own number once both
arms finish, and what the reference has not established. Replay, receiver field fitting and
adaptive-rate sharing run all the same.
- **Live, or a replay.**
  - Live, each arm runs in a worker process. The runner is called exactly as `grl-snam coph`
    calls it, so the outcomes are the same (`tests/test_coph_feed.py`). The play head follows the
    run while Live is on.
  - A full crossing arm takes about 20 minutes of computing for about 130 s of episode, and a
    hazard arm about 6 minutes. A live run therefore plays slower than real time.
  - `--out DIR` keeps the episodes in `grl-snam coph`'s layout, and `--from DIR` replays them.
- **The floor:**
  - the surveyed corridors;
  - the risk field as a heat map: the truth, or one agent's own `r` over its kept stages (the
    Heat map combo, `--heat`);
  - each agent's three stages, the newest brightest;
  - its sensor disc.
- **The rest of the scene:**
  - Crossing's doors are full-width slabs, drawn once an agent of that arm has sensed or been
    told about them.
  - The agents are RoboRacer-sized (0.58 m), heading along their velocity. pycvc_gl packages no
    RoboRacer model, so they are the Humvee model scaled down when it is installed, else boxes.
  - Each packet is an arc from sender to receiver at delivery: green when the receiver's gate
    accepted the update that took it in, red when not, and as wide as its bytes.
- **The panels:** one row per agent per arm:
  - status, version, `pi` (risk, barrier), damping;
  - accepts and rejects, with the last update's reason;
  - packet payload bytes sent and received;
  - the energy ledger: work, heat, kinetic change, shaping jumps and the balance residual.
- **The learning panels** (the Learning window). They draw on pycvc_gl's `chart:` (libcvc
  3.7.14, `host.set_plot`). On an older pycvc_gl the document hides the window, and the port runs
  as before. For one agent (the Agent combo), each arm's learner at each of its updates:
  - **Held-out field loss**, before and after the update. The gate compares the two on the
    learner's validation readings, and the chart draws the gate's threshold, before + 0.003.
    That slack is about 100 times the losses here, so the gate accepts every update.
  - **One statistic below** (the Show combo):
    - the own field against the truth, `|r - r_true|` over the corridor its stages cover,
      where it has sensed (with a 10th..90th percentile band) and elsewhere;
    - the damping estimate against the world's;
    - `pi`;
    - the held-out plant loss;
    - the replay cost.
  - **Markers** where the agent first knew of a door and where it reached its goal. A cursor
    follows the play head, and a click on either chart seeks there.
- **Headless:** the run goes to its end. `--log run.npz` writes the `cars` template's log shape,
  with one column per agent per arm. CoPH drives a point, so `steer` and the commands are NaN, and
  the force is `coef_ux` and `coef_uy`. The ledger and the learner are `coef_*` columns too.
  - The exit code is 0, 1 when a run failed, 2 for bad arguments, and 3 below `--min-goals`.
  - `--at SECONDS` starts the play head mid-run, for a screenshot (`--offscreen --png`).

It needs torch, as `grl-snam coph` does, and no city.

### `learn`: did it learn?

```bash
grl-snam ariadne learn run/history.jsonl                          # one training run
grl-snam ariadne learn seed*/history.jsonl --arms arms.json       # seeds, and the arms
grl-snam ariadne learn run/history.jsonl --follow                 # a run still training
grl-snam ariadne learn seed*/history.jsonl --arms arms.json --headless --report report.json
```

The learning instruments as windows. The port reads learning histories with `grl_snam.learning`:
any trainer that follows its conventions (`coef-train`, `grl-snam train`, `train_coef_energy`,
libcvc's `coef_trainer` and downstream trainers alike). It also reads an optional arms table
(`grl-snam arms-eval`'s `grl_snam.arms/v1`, or the A/B harness's JSON), and makes
`grl-snam learn-report`'s claims on them.

- **Did it learn?**: every claim, pass or fail, with its evidence, and the summary.
- **Loss**: the held-out curve Term picks against its training twin, marked where it was best,
  bent (Kneedle), went flat and turned back up. With several runs, Runs shows their band (the
  marks are the mean curve's) or each run's own curves and pick.
- **Heads**: a head's mean over the probe or held-out set, with its spread as a band, per run,
  marked where it settled.
- **Training**: each run's gradient norm, and each head's dtheta / theta per checkpoint for the
  run Run picks.
- **Checkpoints**: that run's checkpoints with Term's value, the pick marked `*`.
- **Arms** (with `--arms`): each arm's mean [95% CI] per metric, primary metrics first, and its
  paired difference to the untrained arm. `*` marks a CI that excludes 0, with better or worse.
- **Planted** (when a history logs `...planted.<region>.rel_err`): each region's recovered error
  at the pick and at the start, against `--planted-bound`.

The curves are pycvc_gl's `chart:` (libcvc 3.7.14). On an older pycvc_gl each window shows its
marks as text, and the tables are the same. `--follow` re-reads a history when it changes and
re-makes the claims at most every `--every` seconds. Headless, the port writes `--report` and
exits; `--strict` makes it exit 1 when a claim fails. It needs no torch and no city.

It also runs in a browser, on libcvc's `ariglx-web` (CPython, pycvc, pycvc_gl and numpy in
WebAssembly): there the port's Host adopts the page's, `host.run()` returns at once and the page
runs the frames, and the documents are read from files even when a package is in a zip
(`_host.BROWSER`, `_host.on_disk`). [`pages/`](../../pages/README.md) builds that page for
GitHub Pages.

### `replay`: a recorded run, played back

```bash
python -m grl_snam.ariadne live austin-freedrive --plant kinematic --arms trained,untrained,constant \
    --retarget-at 20 --log run.npz --headless --frames 900
python -m grl_snam.ariadne replay run.npz
python -m grl_snam.ariadne replay run.npz --at 26 --paused --camera orbit --offscreen --frames 90 --png at26.png
```

`replay` plays back a run log that `live --arms --log` wrote.
- **What it draws:** it rebuilds the recorded world from the log (the city, a generated world or
  the flat square). Each arm's ghost and track appear where the arm was at the play head.
- **The windows:** the Stats and Ledger windows (`docs/stats_panels.ari`) show what the live run
  showed at that tick, cell for cell. The replay writes them with the live run's own code, and
  `tests/test_ariadne_live_arms.py` checks this at the run's end and mid-run.
- **The transport:** `docs/replay.ari` has libcvc's `sim_transport` component:
  - Paused and Speed;
  - Step, which moves the play head one tick on while paused;
  - Restart.

  Beside it, a Position slider scrubs, and Loop and the camera are there too.
- **The charts** draw the whole run with a cursor at the play head (`play.t`), and a click on one
  seeks there.
- **The flags:** `--at SECONDS` starts the play head there, `--paused` holds it, `--loop` wraps it,
  and `--camera orbit` frames the world.
- **Headless,** nothing is drawn and no world is built: the panels follow the play head for
  `--frames` frames.
- **The exit code** is 0, and 2 for a file that is not a run log.

### `scrub`: watch it learn

```bash
python -m grl_snam.ariadne scrub run/history.jsonl                           # Austin, Humvee
python -m grl_snam.ariadne scrub run/history.jsonl --plant humvee --plant kinematic  # both
python -m grl_snam.ariadne scrub run/history.jsonl --world flat --ticks 600
python -m grl_snam.ariadne scrub run/history.jsonl --headless --report scrub.json
python -m grl_snam.ariadne scrub --view scrub.json [more.json ...]          # deployed before
```

`scrub` deploys every checkpoint of a training run. The learning history names the policy each
checkpoint saved: its `blob`, a `.cvcnav` or a `.pt`. A relative path is the history's
directory's, and blobs that are not there are listed and skipped.
- **The scenario:** each policy drives libcvc's sim_world on `--plant` from the scenario's start
  to its goal for `--ticks` ticks, as `live --arms` drives an arm, tick for tick. So checkpoint 0
  deployed here is the untrained arm deployed there (on the same plant), which the tests check.
- **The plant:** `humvee` by default, the Jolt Humvee, which collides with the world's buildings.
  A car that drives into one stops against it. `kinematic`, nav's bicycle, has no collision
  response, so its cars drive through buildings: the scrubber's default until 2026-11-19.
  - `--plant` given again deploys every checkpoint on each plant, and the Scrub window's
    Plant combo shows the same run and step on another.
  - One `--report` holds one plant's run.
- **The cache:** each run is a run log, cached at `<history>.scrub/<blob sha256>-<scenario
  hash>.npz`. A second look costs nothing. The scenario hash covers the world (its `run.*`
  keys, and a generated world's `world.*`), the start and goal, the seed, the ticks, the plant,
  the policies' frame (`--scale`, else the first `.pt` checkpoint's), the settings set on the
  command line, and libcvc's version.
- **What to look for:** [`docs/scrubber-guide.md`](../../docs/scrubber-guide.md) describes each
  window, and what a policy that is learning something useful looks like there.
- **Checkpoint k** is picked by a combo, a slider, Previous and Next, or a click on a chart. It
  opens on the history's pick (`--at K` for another). Its ghost (orange) runs beside step 0's
  (blue) over the episode, with the Stats and Ledger windows for the two and libcvc's sim
  transport.
- **The deployed metric** (the Metric combo: reached, time, path, ticks penetrating, clearance,
  what is left to the goal) is charted against the checkpoint's step, with step 0's as a dashed
  line, and shown as a table, one row per checkpoint.
- **The training curve** is a held-out curve from the history, with the `learn` port's marks.
  Both charts carry a cursor at checkpoint k.
- **Headless,** it deploys and reports: `--report R.json` writes every checkpoint's metrics
  and run log (`grl_snam.scrub/v1`), the scenario, and what deployed it (the GRL-SNAM commit,
  libcvc's version). `--blobs DIR` finds the saved policies there first (by file name), and
  `--cache DIR` caches the run logs there instead of `<history>.scrub/`. `--data-set NAME` puts
  the run's data set in the report (the learning check's `retrain`, `fixed/retrain`,
  `world-units/retrain`), and `--lean` caches lean run logs (`grl_snam.run_log/v2`: single
  precision, the cars template's derived fields left out, the arrays packed losslessly; about
  70 KB a checkpoint of 1200 ticks, read back the same), as the Pages site does. A lean log
  cached before they were packed is read, and packed in place.
- **View only:** `--view R.json ...` shows runs deployed before and deploys nothing, so it needs
  neither torch nor the policies, and no sim_world steps. It reads the reports, the run log each
  checkpoint lists (read when first shown) and the history each names, found as given, relative
  to the report, or by file name beside it, so a report travels with its files. A report may also
  be a run's bundle: one zip with the report (`scrub.json`), the history and the logs, which the
  view reads from it (the Pages site ships those; `pages/assemble.py` writes them). It does not
  hash the scenario again. With several reports a Run combo picks one (their worlds must be the same),
  and a Set combo their data set when the reports name several. Picking another set shows its run
  of the same name at the same training step, so the learning check's retrains compare side by
  side.
  The world is rebuilt from the report: a generated world from its generator settings (the
  view says when this libcvc's differs from the deploy's). With `--no-world`, a city that is not
  on the machine in a browser, or a pycvc_gl without `pycvc_gl.cars`, it shows the windows
  alone. The Pages site's scrub page is this mode (`pages/README.md`).
- **`--windows ID,...`** opens only those windows (`scrub`, `training`, `episode`,
  `sim_transport`, `arms_stats`, `arms_ledger`); the Windows menu has the rest. `--camera
  orbit` looks at the world from above instead of following checkpoint k.
- **The exit code** is 0, and 2 when the history names no saved policy that is there, the plant
  is torch, or a report or a file it names is not there.

On the learning check's retrain (seed 0, 41 checkpoints, Austin, 60 s each), measured on the
kinematic plant, the default then, so through the buildings, no checkpoint reaches the goal, and
there are two outcomes:
- 25 checkpoints, step 0 among them, stop 516 to 558 m short, after 255 to 266 ticks
  penetrating.
- 16 checkpoints take another turn and end 865 to 883 m away, after 323 to 334 ticks. They are
  scattered from step 400 to step 4000, and include the history's pick (step 2100) and the last
  checkpoint.

Nothing orders the outcomes by training step. That is the null of `docs/learning-validation.md`,
seen checkpoint by checkpoint.

The Pages page's runs, deployed again on both plants (2026-11-19): the three retrains, five seeds
each, 41 checkpoints a seed, 205 a set, on libcvc's generated city for 1200 ticks (60 s), libcvc
4.0.0. No checkpoint reaches the goal on either plant.

| set | plant | ticks penetrating | least clearance | path | left to the goal |
|---|---|---|---|---|---|
| `retrain` | kinematic | 375–636, every checkpoint | −28.9 m | 480–631 m | 272–353 m |
| `retrain` | humvee | 0 | 1.33 m | 414–415 m | 579–580 m |
| `fixed/retrain` | kinematic | 372–634, every checkpoint | −28.9 m | 479–636 m | 231–350 m |
| `fixed/retrain` | humvee | 0 at 199, 71–186 at 6 | −25.7 to 49.0 m | 303–472 m | 503–667 m |
| `world-units/retrain` | kinematic | 355–459, every checkpoint | −29.8 m | 471–485 m | 329–339 m |
| `world-units/retrain` | humvee | 0 | 0.70–0.82 m | 212–218 m | 588 m |

- "Ticks penetrating" counts the ticks a car's centre is on a cell the navigator marks blocked:
  a building, or deep water. On this world most of them are not buildings. libcvc 4.0.0's
  generated world marks dry ground between its −4 m water level and −0.5 m as deep water
  (transfix/libcvc#1374 fixes it).
- On kinematic, every checkpoint spends 355 to 636 ticks on blocked cells, so its progress toward
  the goal was never real. 79–93% of those ticks are on that mislabelled water; the rest are in
  buildings.
- On the Humvee, `retrain` and `world-units/retrain` each come to one outcome whatever the seed or
  the step: the car meets a building and stays against it.
- `fixed/retrain`'s Humvees range further (303–472 m). Six of its checkpoints (seeds 0, 1, 3 and
  4) count 71 to 186 ticks penetrating with no building contact. They are not in a building: they
  drive into a dry basin near (−290, 175), 0.6–3.5 m above the water level, that the navigator
  reads as deep water and the physics has nothing to collide with.
- The page's runs are deployed on libcvc 4.0.0, so these counts stay until a libcvc with #1374 is
  released and the runs are deployed again.

To see it yourself, with the retrain's checkpoints from cvcpkg.org:

```bash
cvcpkg install grl-snam-learning-checkpoints --prefix <prefix>
P=<prefix>/share/grl-snam-learning-checkpoints
python -m grl_snam.ariadne scrub docs/learning-validation/retrain/seed0.history.jsonl.gz --blobs $P/retrain/seed0
python -m grl_snam.ariadne scrub docs/learning-validation/fixed/retrain/seed0.history.jsonl.gz \
  --blobs $P/fixed/retrain/seed0                      # the retrain after the fix (§8)
```

**In a browser.** The Pages site's scrub page is `scrub --view` on the learning check's three
retrains (`retrain`, `fixed/retrain`, `world-units/retrain`; the Set combo picks one), five seeds
each, deployed when the site is built on libcvc's generated world on two plants, the Humvee and
kinematic (the Plant combo picks one; `pages/README.md`). It needs no download beyond the page.
On kinematic, through the buildings, none of the 615 checkpoints reaches the goal (each ends 231 to
353 m short), and in no set are the outcomes ordered by training step. On the Humvee none does
either: world-units/retrain seed 0's 41 checkpoints all stop against a building about 150 m in,
588 m from the goal, with no tick penetrating.

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
  (`grl-snam-weights`, which needs torch to export it to `.cvcnav`), in the frame its checkpoint
  records (`meta["scale"]`, passed as `--nav-scale`: 1 unit = 300 m for the published policy; a
  `grl-snam train` checkpoint since docs/learning-validation.md §9 is 20 m per unit).
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
| `grl-snam coph crossing / hazard` | episodes, `results.json`, `REPORT.md` | `coph crossing` / `coph hazard` (live, or `--from` a run) |
| `grl-snam learn-report` | text, `report.json` | `learn` (the claims, and the curves, heads, checkpoints and arms behind them) |
| (new) policies side by side, recorded | | `live austin-freedrive --arms ... --log`, then `replay run.npz` |
| (new) a training run's checkpoints, deployed | | `scrub run/history.jsonl` |
| `grl-snam capture drive / multigoal` | mp4 | covered by `live austin-freedrive` / `austin-learned` (the same navigator, live) |
| `grl-snam pipeline`, `build-sdf`, `train` | batch | not ported: no picture of their own |
| `wall_vis` | a post-pass for the finale | not ported |
| `*_bench` | console timings | not ported |
| `selftest`, `eval`, `experiments/e*` | console / paper figures | not ported |
| (new) your own multi-car controller | | `cars` (libcvc's template: `--copy DIR`) |
