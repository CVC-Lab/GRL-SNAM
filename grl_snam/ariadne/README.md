# GRL-SNAM on Ariadne

GRL-SNAM's demos as Python hosts of Ariadne (`.ari`) documents. The document (`docs/*.ari`) holds
the panels, the controls and the look: a pale sky, a low warm three-point sun with a cool fill,
and soft-edged shadows that follow the camera. The port (`grl_snam/ariadne/*.py`) runs the demo,
or replays its recording, draws it into the document's scene, and writes what the panels show.
Vehicles on Austin are Humvees at their real size (4.7 m; the `vehicle-humvee` package's model,
else a box that size).

```bash
python -m grl_snam.ariadne --list
python -m grl_snam.ariadne <port> --help
```

Every port takes `--offscreen` (GL, no window), `--headless` (no GL: the readouts only),
`--frames N` (stop after N frames) and `--png PATH` (save the last frame).

## Requirements

- pycvc_gl with its Python host, `pycvc_gl.ariadne` (libcvc 3.7.9+, transfix/libcvc#985). With an
  older pycvc_gl the ports refuse to start and say so; their tests skip.
- torch (every port but `live austin-planner`).
- Austin (`scene-austin-south`, or `scene-austin-south-small`) for `live`, `squad` and `fleet`,
  found as `grl-snam demo` finds it: `GRL_SNAM_SCENE_BUNDLE`, else
  `<prefix>/share/cvc-scenes/austin_south`. The Humvee comes from `vehicle-humvee`
  (`share/cvc-scenes/shared/Humvee.glb`) or the bundle's own copy.
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
```

Records the run into `traces/<run>` when it is missing (minutes; as `finale_record`,
`convoy_record`, `pursuit_record` do), then replays the recorded poses in 3-D Austin: a Humvee per
agent seated at its recorded heading, a pole in its colour, its goal (a pyramid) and its track.
The panels are the transport, the captions, a row per agent and the first agent's HUD.

### `fleet`: a live squad of Humvees on Austin

```bash
python -m grl_snam.ariadne fleet                      # GRL-SNAM's torch squad
python -m grl_snam.ariadne fleet --plant humvee       # libcvc's sim_world, Jolt Humvees
python -m grl_snam.ariadne fleet --plant kinematic    # libcvc's sim_world, nav's bicycle
```

`grl_snam.squad.Squad` run live, each agent starting blind; the Plant combo switches the runtime
and New goals deals new ones. Measured with 4 agents over 18 s: torch moves all four (~10 m/s),
kinematic all four (14-18 m/s), Jolt Humvees leave two stuck in wall-escape mode (the navigator
is sized for the story's bodies, not a 4.7 m Humvee), so torch is the default.

### `material`: material-aware navigation

```bash
python -m grl_snam.ariadne material
```

`grl-snam material-demo`'s two stories (the fog scenario and the reactive swarm), each with
material off and on, run as async functions on the app's background pool while the window
draws; each shows a picture (drawn with numpy, no matplotlib) and the off/on numbers.

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
| `Squad(plant=...)` | library, tests | `fleet --plant humvee / kinematic` |
| `grl-snam material-demo` | two PNGs | `material` |
| `grl-snam capture drive / multigoal` | mp4 | covered by `live austin-freedrive` / `austin-learned` (the same navigator, live) |
| `grl-snam pipeline`, `build-sdf`, `train` | batch | not ported: no picture of their own |
| `wall_vis` | a post-pass for the finale | not ported |
| `*_bench` | console timings | not ported |
| `selftest`, `eval`, `experiments/e*` | console / paper figures | not ported |
