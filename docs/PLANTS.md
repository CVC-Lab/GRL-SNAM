# Running GRL-SNAM on a vehicle plant: kinematic, dynamic, Jolt

The policy trains on a point-mass surrogate. GRL-SNAM can also drive it on a vehicle: libcvc's
pure-C++ `sim_world` runs the navigator, with the policy exported to `.cvcnav`, and a *plant*
moves the vehicle. The three plants share that navigator, so runs on them differ only in the
vehicle:

| Plant | Name | What moves the vehicle | Collisions | Energy |
|---|---|---|---|---|
| nav's kinematic bicycle | `kinematic` | `cvc::nav`'s bicycle, no tires | none (nav scores penetration) | none |
| the dynamic single track | `dynamic`, `dynamic:<preset>` | libcvc's `dyn_bike_plant`: Pacejka tires, load transfer, actuator lag | each chassis is a box in a Jolt world with the buildings | its port-Hamiltonian books (drive, brake, dissipated, contact, residual) |
| a Jolt vehicle | `jolt`, `jolt:<preset>`, or a preset name | libcvc's `nav_plant`: Jolt's vehicle, drivetrain and tires | the buildings are static boxes | Jolt's ledger (source, tire, brake, aero, rolling) |

The presets are `roboracer` (the UT-AV lab's 1/10 car, the default for the stories and the
squads; it fits GRL-SNAM's 0.7 m bodies) and `humvee` (the default for `capture`, sized for the
Austin scene). `capture` also keeps `point`, the Python surrogate the policy trains on.

On a plant the navigator plans with the vehicle's limits, never more than the policy's: 0.8 of
its `max_accel`, and corners at 0.5 g. The corner cap does not know the ground's grip, which is
why low grip slides (see [What to expect](#what-to-expect)).

To drive the plants with a controller of your own instead of GRL-SNAM's navigator, start from the
`cars` template: [Write your own multi-car controller](../grl_snam/ariadne/README.md#write-your-own-multi-car-controller).

Contents:
- [What you need](#what-you-need)
- [`capture`: one vehicle, rendered with its read-out](#capture-one-vehicle-rendered-with-its-read-out)
- [`pipeline`](#pipeline)
- [The finale, the squad, the convoy and the pursuit](#the-finale-the-squad-the-convoy-and-the-pursuit)
- [The fog-of-war stories](#the-fog-of-war-stories)
- [The material demo](#the-material-demo)
- [The benches](#the-benches)
- [Not on a plant yet](#not-on-a-plant-yet)
- [Training through Jolt](#training-through-jolt)
- [What to expect](#what-to-expect)

## What you need

- **A pycvc with libcvc's plants.** `grl_snam.nav_native` probes for each piece:

  ```bash
  python -c "from grl_snam import nav_native as n; print(n.HAS_PLANT, n.HAS_DYN_PLANT, n.HAS_TRAIN_JOLT)"
  ```

  `HAS_PLANT` is the Jolt plant (`pycvc.nav_plant`, libcvc P6), `HAS_DYN_PLANT` the dynamic plant
  (`pycvc.dyn_bike_plant`, transfix/libcvc#979), and `HAS_TRAIN_JOLT` the Jolt rollout's trainer
  (`pycvc.nav_train_coef_mlp_jolt`). Until a pycvc release carries `dyn_bike_plant`, build pycvc
  from libcvc master with `-DCVC_BUILD_PYCVC=ON -DCVC_WITH_JOLT=ON` ([NATIVE_CVC_NAV.md,
  "Getting a pycvc"](NATIVE_CVC_NAV.md#getting-a-pycvc-that-carries-cvcnav)). On Windows,
  `os.add_dll_directory` the build's and the deps' `bin` before `import pycvc`.
- **For `capture`:** VTK's Python modules (`vtk-python-cp311` on cvcpkg), `ffmpeg` on `PATH`, a
  scene bundle (`scene-austin-south` on cvcpkg: `share/cvc-scenes/austin_south`) and a checkpoint
  (`grl-snam-weights`: `share/grl-snam-weights/coef_sdf.pt`). `build-sdf` rasterizes the
  buildings with `pycvc_gl.scenes`.
- **The checkpoint's frame.** The shipped `coef_sdf.pt` was trained at 1 normalized unit = 300 m
  (`--region 1500`). Build the SDF in the same frame, or the field and the policy disagree:

  ```bash
  grl-snam build-sdf path/to/austin_south --region 1500
  ```

## `capture`: one vehicle, rendered with its read-out

```bash
# The point surrogate, as before (paced by arc length; legs end at closest approach)
grl-snam capture multigoal path/to/austin_south coef_sdf.pt --sdf path/to/austin_south/nav_sdf.npz

# A Jolt Humvee up a street on grip 0.6, drone camera
grl-snam capture drive path/to/austin_south coef_sdf.pt --sdf path/to/austin_south/nav_sdf.npz \
  --start 499.2 -699.7 --goal 496.7 -403.7 --plant jolt --vehicle humvee --mu 0.6 --cam drone \
  -o drive_jolt.mp4

# The dynamic Humvee on two of the multi-goal drive's goals
grl-snam capture multigoal path/to/austin_south coef_sdf.pt --sdf path/to/austin_south/nav_sdf.npz \
  --plant dynamic --goals 2 -o multigoal_dynamic.mp4
```

The plant options (both subcommands):

| Option | Default | Meaning |
|---|---|---|
| `--plant` | `point` | `point`, `kinematic`, `dynamic` or `jolt` |
| `--vehicle` | `humvee` | the dynamic or Jolt plant's preset |
| `--mu` | 0.9 | the ground's grip |
| `--routes` | off | follow libcvc's vehicle routes (planned for the turning circle) instead of the carrot |
| `--fog` | off | the map starts unknown: the vehicle maps the buildings with its sensor as it drives; they are in the physics from the start |
| `--weights` | | drive this `.cvcnav` (say, one [trained through Jolt](#training-through-jolt)) instead of the checkpoint's policy |
| `--reach-m` | 15 | arrival radius, m |
| `--max-steps` | 3000 / 1300 | ticks per leg |
| `--speed` | 1 | world seconds per played second; raised only as far as fitting `--minutes` needs |
| `--caption` | | a line drawn along the bottom |

On a plant:
- **The whole run is kept**, failures included, and played by the clock: a stall or a spin takes
  the time it took. A yellow trail shows the path driven so far, and the box is the vehicle's
  own chassis, at its own heading.
- **A leg ends** when the vehicle is within `--reach-m`, after 340 ticks without getting closer
  (stalled), or at `--max-steps`. A multi-goal leg starts where the last one ended: a vehicle is
  not rewound.
- **The HUD adds three lines.** The plant and its grip. The sideslip (the chassis' velocity off
  its heading), the tire use (libcvc's: the most used tire's combined slip over its peak; past 1
  it slides; Jolt counts wheel spin and lock too) and speed x yaw rate in g, with `[PAST PEAK]`,
  `[SPIN]` (sideslip past 20 degrees) and `[CONTACT]` flags. Then the energy, in kJ: on Jolt
  what the source drew and where it went, on the dynamic plant its books.
- **`<out>.json`** records each leg: how it ended (`reached`, `stalled`, `out of ticks`), ticks,
  closest approach, penetration, minimum clearance, maximum sideslip and tire use, ticks past
  the peak, contacts and the final books, and the sim_world's own episode record (`nav_stats`).
  Its `building_contacts` counts the contacts the chassis starts with a building or another
  static body. The ground is not one where the pycvc's `nav_plant` takes `ground_bodies`
  (libcvc's `nav_plant_options::ground_bodies`); with an older pycvc a chassis scraping the
  ground counts too.

## `pipeline`

```bash
grl-snam pipeline path/to/austin_south --plant jolt --vehicle humvee --drive single
```

It builds the SDF and trains on the surrogate as before, then drives the video on the plant
(`grl_snam_single_jolt.mp4`). `--mu` sets the grip.

## The finale, the squad, the convoy and the pursuit

They run a `Squad`. With a plant the squad runs in libcvc's `sim_world`: one private belief plane
per agent, peers meet as bodies in the physics world, and on a physics plant the agents with a
fixed goal drive vehicle routes. The native world keeps its belief and its plan to itself, so
these traces carry no map updates, field of view or route spine to draw.

```bash
grl-snam finale path/to/austin_south --plant jolt:humvee
grl-snam finale path/to/austin_south --plant dynamic:humvee --max-steps 600   # a short run
```

The finale records both acts' traces, then renders them through `pycvc_gl` (its `Lab` and scene
renderer), as without a plant.

```python
from grl_snam.squad import Squad
from grl_snam.tools.convoy_record import record_convoy
from grl_snam.tools.pursuit_record import record_pursuit_v2

sq = Squad(story, agents, truth_occ=truth, prior_occ=truth, plant="dynamic")  # or "jolt", ...
res = sq.run(max_steps=1500)
res.reached, res.energy_J      # on "dynamic": what each one's axles delivered (J)

record_convoy("path/to/austin_south", "convoy", plant="jolt:humvee")
# The native world has one top speed for the squad, so the quicker quarry must go:
record_pursuit_v2("path/to/austin_south", "pursuit", plant="jolt:humvee", runner_vmax_scale=1.0)
```

`squad_record.record_squad(..., plant=...)` is what all three use. `Squad(plant=...)` refuses
per-agent `vmax` or sensor overrides, and a start inside a building. A material grid
(`material=`) attaches to the native world: its forces and witness gate, without the Python
path's route surcharge.

## The fog-of-war stories

```bash
grl-snam fog record blocker --plant jolt -o traces/blocker_jolt
grl-snam fog record traffic --plant dynamic --mu 0.6 -o traces/traffic_dynamic
grl-snam fog all --plant jolt -o fog_jolt          # every story, then the reel
```

`--plant` runs the story's one agent on libcvc's sim_world (RoboRacers unless the name says
`:humvee`) and writes the same trace the Python scenario does, so `fog capture` and `fog play`
replay it. All six stories run: `ghost`, `blocker`, `unit`, `city`, `traffic` and `pursuit`.
The recorder keeps the native world honest to the story:

- **A wall raised mid-run** (`blocker`) goes into the world's truth (`sim_world::set_truth`,
  transfix/libcvc#986), and on a physics plant it is a box in the physics world from then on.
  The agent learns it by sensing, as in the Python scenario.
- **A unit's marks** (`unit`) are the native world's decaying marks (`add_obstacle`), believed
  until `unit_ttl_s` passes.
- **Moving vehicles** (`traffic`) join the truth each tick and are kinematic boxes in the
  physics world.
- **A moving target** (`pursuit`) is retargeted each tick.

The native world steers by its own carrot with no route spine, so the trace's routes are empty.
A mover it sees is believed as a wall until it sees those cells free again; the Python scenario
demotes it to the decaying layer instead. The trace's clearance is each tick's distance to the
truth, less the body radius, and `trace.json` adds the plant and the world's own episode record
(`native_nav_stats`). Needs a pycvc with transfix/libcvc#986 (`nav_native.HAS_FOG`).

## The material demo

```bash
grl-snam material-demo --plant jolt --out-dir material_jolt
```

Both stories run on libcvc's sim_world, their agents driving the plant with the material's forces
and witness gate attached (`NativeSimWorld.set_material_grid`). The native world has no route
spine, so the scenario story's planner surcharge, which routes the Python agent through the dry
gap, does not apply: the agent meets the water strip head on. The demo prints the same table and
writes the same two PNGs, and exits 1 when material leaves a vehicle in hazard cells, as without
a plant.

The material's forces steer a vehicle off risky ground, but they slow it only by what they take
from its goal pull's surplus over its top speed. A physics plant's top speed is under the
policy's cruise, so the surplus is all of it: the dynamic RoboRacer meets the water at its top
speed of 10 m/s, and the water's push (half the pull's surplus at most) leaves its command,
and so its run, exactly as without material. Head on, the push has no sideways part to steer it
either. What slows a vehicle on risky ground is libcvc's risk governor
(`sim_world::set_risk_governor`, `NativeSimWorld.set_risk_governor`, `PlantNavigator(...,
risk_governor=True)`): the risk caps its speed, down to a stop at risk 1, looking 10 m ahead. The
demo turns it on for the dynamic and Jolt plants where the running pycvc has it
(`nav_native.HAS_RISK_GOVERNOR`); `--no-governor` shows the forces alone, `--governor` puts it on
the kinematic bicycle too.

## The benches

```bash
python -m grl_snam.tools.squad_bench --plant jolt --n 8          # the squad's tick on Jolt
python -m grl_snam.tools.drive_bench --plant dynamic --n 64      # sim_world's tick, sense frozen
```

`swarm_bench` and `belief_bench` measure the torch `Swarm`'s sensing and belief planes. A plant
replaces only the drive, which `drive_bench --plant` times, so they take none.

## Not on a plant yet

- **A pursuit with a quicker quarry**, as above: the native world has one top speed.
- **A fog story with a wall taken down** (`remove_rect`) on a physics plant: none of the stories
  has one, and the recorder refuses it rather than leave the box standing.

## Training through Jolt

libcvc's coefficient trainer can train through Jolt (`rollout_kind::jolt`): Jolt drives the
vehicles in a copy of the scene per agent, and the gradient is the four-wheel port-Hamiltonian
model's along Jolt's trajectory, after the model is calibrated to Jolt's energy ledger. libcvc's
[NAV_TRAINING.md, "Training on Jolt"](https://github.com/transfix/libcvc/blob/master/docs/NAV_TRAINING.md#training-on-jolt-rollout_kindjolt)
describes it and what it measured. It runs on the CPU and needs no torch.

**From the CLI**, on the city story:

```bash
grl-snam coef-train --backend native --rollout jolt --vehicle humvee --steps 400 --out coef_jolt.cvcnav
```

`--w-coll` (default 3) and `--w-energy` (a price on what Jolt's source draws; default 0) apply.
It prints the calibration.

**From Python, on your own scene.** Give the trainer the scene's occupancy in the frame you
will drive it in. Each agent gets its own copy of the scene in one Jolt world, so keep the grid
coarse and the batch small:

```python
import numpy as np, torch
from grl_snam import nav_native
from grl_snam.tools.sdf import terrain_bounds

meta = torch.load("coef_sdf.pt", map_location="cpu")["meta"]
occ = np.load("path/to/austin_south/nav_sdf.npz")["phi"][::4, ::4] < 0
cal = nav_native.train_coef_mlp_jolt(
    occ, "austin_jolt.cvcnav", bounds=terrain_bounds("path/to/austin_south"),
    scale=meta["scale"], rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"],
    vehicle="humvee", steps=400, n=8, calib_agents=4, calib_steps=60,
)
```

`veh_L` (the wheelbase), `veh_a_max` and `veh_a_lat_max` default to the vehicle's at `scale`,
with the plants' limits. `nav_native.train_coef_mlp(..., rollout="jolt")` is the same with the
defaults, and `pycvc.nav_train_coef_mlp_jolt` the binding underneath (all positional; libcvc
documents it).

**Drive the result** on the plant it trained for:

```bash
grl-snam capture drive path/to/austin_south coef_sdf.pt --sdf path/to/austin_south/nav_sdf.npz \
  --start 499.2 -699.7 --goal 496.7 -403.7 --plant jolt --weights austin_jolt.cvcnav
```

## What to expect

With the shipped checkpoint on Austin, 24 random routes of 120-400 m between open streets
(Humvees; arrival within 15 m; 1500 ticks):

| Runtime | Arrived |
|---|---|
| the point surrogate (Python) | 3 / 24 |
| nav's kinematic bicycle | 3 / 24 |
| the dynamic Humvee, grip 0.9 | 0 / 24 |
| the Jolt Humvee, grip 0.9 | 0 / 24 |
| the Jolt Humvee on vehicle routes, grip 0.9 | 0 / 24 |

The checkpoint was trained on the surrogate in a frame where 1 unit is 300 m and arrival is
within 0.8 units, 240 m. It does not transfer to a vehicle that has to arrive within 15 m:

- **The bicycle dead-ends.** The policy's wall-follow escape needs the point mass's sideways
  step; nav's kinematic bicycle stops where the surrogate slides past (the C++ one and
  `SdfNavigator(dynamics="bicycle")` both do).
- **The physics vehicles orbit.** The wall-follow carrot sits inside a Humvee's turning circle,
  so it circles at 7 m/s. Vehicle routes fix this on the city story's RoboRacers
  (`tests/test_squad_plant.py`), not on Austin's Humvees with this policy.
- **Low grip slides.** The navigator corners at 0.5 g whatever the grip. On grip 0.3 the Humvee
  slides past the tire's peak and spins out (sideslip near 90 degrees); with the map unknown it
  meets walls it cannot stop for.

Each of these failures is on video in the
[plant reel](https://github.com/CVC-Lab/GRL-SNAM/releases/tag/plant-reel-2026-11), next to three
successes. Each clip comes with the `capture` command that made it and its measured outcome.

The city story's squads arrive on Jolt RoboRacers ([NATIVE_CVC_NAV.md](NATIVE_CVC_NAV.md#a-physics-plant--squadplant)).

**The fog-of-war stories** (RoboRacers, an untrained policy, `fog record --plant`, at most 1500
ticks): all six arrive on the kinematic bicycle, the dynamic plant and Jolt. Only `traffic` puts
the vehicle inside the truth: a moving vehicle drives onto it for 3 ticks on Jolt and 8 on the
dynamic plant. The physics plants take two to three times the bicycle's ticks (blocker: 357 on
the bicycle, 826 dynamic, 893 Jolt).

**The material demo** (RoboRacers, `material-demo --plant`, material on; without material every
run crosses the water):

| | swarm | scenario |
|---|---|---|
| Jolt, governor | no hazard or mud cells; 3 of 6 arrive | no hazard cells, mud exposure 42.7; never reaches the goal |
| Jolt, forces only | no hazard or mud cells; 2 of 6 arrive | no hazard cells, mud exposure 119.7; never reaches the goal |
| dynamic, governor | no hazard cells; 3 of 6 arrive | no hazard cells; never reaches the goal |
| dynamic, forces only | no hazard cells; 4 of 6 arrive | 14 hazard steps: the run without material, to the millimetre |
| kinematic, forces only | no hazard cells; 6 of 6 arrive | 8 hazard steps, as without material; arrives |
| kinematic, governor | no hazard cells; 6 of 6 arrive | 62 hazard steps, creeping through the water; arrives |

Off the goal line the forces and the gate keep every vehicle out of the hazards. Square across
it, with no planner to find the dry gap, the forces alone do not keep the dynamic vehicle out;
the governor stops both physics vehicles short of the water, a dead end the policy cannot leave.
The kinematic bicycle creeps on through the water under the governor instead, so it stays off
there by default.
