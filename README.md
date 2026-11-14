# GRL-SNAM: Geometric Reinforcement Learning for Navigation and Mapping

This repository contains the implementation of **GRL-SNAM**, a geometric reinforcement learning framework for simultaneous navigation and mapping in unknown environments using Hamiltonian mechanics and differential policy optimization.

📖 **Documentation:** [Developer Guide](docs/developer-guide.md) · [Native `cvc::nav` acceleration](docs/NATIVE_CVC_NAV.md) · [Running on a vehicle plant (kinematic, dynamic, Jolt)](docs/PLANTS.md) · [Material-aware navigation](docs/MATERIAL_NAV.md) · [Vehicle refinements](docs/VEHICLE_REFINEMENTS.md) · [Navigation stats & scorecard](docs/NAV_STATS.md) · [Does the policy learn? The learning check](docs/learning-validation.md) · [PyPI Publishing Roadmap](docs/pypi-publishing-roadmap.md) · [Headless server quickstart](docs/headless-quickstart.md) · [The demos on Ariadne, and a template for your own multi-car controller](grl_snam/ariadne/README.md) · [The vehicle plant in port-Hamiltonian form (PDF)](docs/vehicle-ph-plant.pdf)

## Overview

GRL-SNAM addresses navigation in unknown environments by:
- **Minimal Mapping**: Achieves high-quality navigation using only ~10% environment coverage
- **Hamiltonian Structure**: Formulates navigation as energy minimization with symplectic dynamics
- **Multi-Policy Architecture**: Decomposes control into sensor, frame, and shape policies operating at different timescales
- **Deformable Robots**: Demonstrates hyperelastic ring navigation through narrow gaps and cluttered spaces

## Key Features

- ✅ Feedforward control without Bellman bootstrapping
- ✅ Physics-informed energy-based policy learning
- ✅ Online adaptation through contextual Hamiltonian corrections
- ✅ Radius-aware collision avoidance with IPC barriers
- ✅ Multi-start robustness training for near-obstacle scenarios

## Installation

GRL-SNAM is a Poetry-managed, pip-installable Python package. The simplest install:

```bash
pip install grl-snam        # once published to PyPI
```

For local development:

```bash
git clone <repository-url>
cd GRL-SNAM
poetry install              # installs runtime + dev dependencies
poetry run pytest           # run the smoke tests
```

To use GRL-SNAM as a dependency in a downstream project, add it to your own
``pyproject.toml``:

```toml
[tool.poetry.dependencies]
grl-snam = "^0.1"
```

### Requirements
- Python >= 3.10, < 3.14
- PyTorch >= 2.1
- NumPy, Matplotlib, imageio (installed automatically)

> **Headless server / want the graphics (`pycvc_gl`)?** `pip install` does not provide the
> graphics bindings. Install from cvcpkg instead — see the
> [headless quickstart](docs/headless-quickstart.md).

### Manual setup (without Poetry)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Command-line interface (`grl-snam`)

Installing GRL-SNAM puts one console script on your `PATH` — **`grl-snam`** — the single
entry point for every workflow, from a raw world model to a running demo. (`grl-snam`,
`grl-snam-selftest`, and `grl-snam-lab-demo` are the installed scripts; the last two are
aliases for `grl-snam selftest` / `grl-snam lab-demo`.)

```bash
grl-snam --help
```

| Command | What it does |
|---|---|
| `grl-snam selftest` | numerical correctness check (surrogate dynamics; no data) |
| `grl-snam obstacles BUNDLE` | world model → circular-obstacle set (`.npz`) |
| `grl-snam build-sdf BUNDLE` | world model → navigation SDF (`.npz`) |
| `grl-snam train SDF.npz` | self-supervised SDF-coefficient training |
| `grl-snam arms-eval --arm NAME=PATH …` | trained, untrained and constant policies on paired held-out seeds |
| `grl-snam learn-report HIST…` | did the learner learn? held-out claims, each pass or fail |
| `grl-snam capture drive\|multigoal …` | drive the policy → mp4 offscreen, with a live HUD |
| `grl-snam pipeline BUNDLE` | the whole chain: world model → SDF → train → video |
| `grl-snam demo NAME` | run a demo live inside VolRover3 |
| `grl-snam lab-demo [PNG]` | standalone lab visualization |
| `grl-snam ariadne PORT` | a demo as an Ariadne document; `cars` is libcvc's template for your own multi-car controller |
| `grl-snam eval` / `train-coef` | the CoefEnergyNet evaluator / trainer (legacy track) |

A `BUNDLE` is a **world model** — a directory with `terrain.json` (heightfield) +
`buildings.glb` (city mesh), e.g. an OpenStreetMap + SRTM export. The heavy work lives
in `grl_snam.tools`; the demos in `grl_snam.demos`; the graphics deps (`pycvc_gl`, VTK,
`vrhost`) load lazily, so `grl-snam --help` runs anywhere.

### Ariadne: the demos in a document, and a template for your own multi-car controller

`grl-snam ariadne <port>` runs GRL-SNAM's demos as Python hosts of Ariadne (`.ari`) documents:
the live Austin demos, the fog-of-war stories, the squad recordings, a live fleet on a vehicle
plant, and the material demo. `grl-snam ariadne learn HIST... [--arms A.json] [--follow]` shows
the learning instruments ([Did it learn?](#did-it-learn-held-out-losses-arms-and-claims)) as
windows: the held-out curves and their marks, the heads, the checkpoints, the arms and
`learn-report`'s claims. `grl-snam ariadne live austin-freedrive --plant kinematic --arms
trained,untrained,constant` drives policies side by side, one sim_world each, with Stats and
Ledger windows. `--log` records the run, and `grl-snam ariadne replay` plays it back. `live` and
`fleet` take the `cars` template's settings (libcvc's
`nav.*`, `nav.vehicle.*` and `nav_plant.*` keys, mapped onto GRL-SNAM's torch navigator where
they mean the same thing), its worlds (Austin, another city, a generated world, a flat square),
its plants, its panels and its headless log.

`grl-snam ariadne cars` runs libcvc's multi-car template (`pycvc_gl.cars`, pycvc-gl 3.7.12+). A
controller of your own drives N Ackermann cars:
- on nav's kinematic bicycle, Jolt vehicles or the dynamic single track;
- in Austin (the default, from cvcpkg.org), a generated world or a flat square;
- with panels, rasters, sliders bound to the navigator's settings, and headless batch runs.

`--grl-policy` runs GRL-SNAM's published policy as the navigator baseline.

```bash
cvcpkg install grl-snam-cp312 --prefix ./env && . ./env/bin/activate   # pycvc_gl comes along
grl-snam ariadne --list
grl-snam ariadne cars --copy my_cars && python my_cars/cars.py
```

[grl_snam/ariadne/README.md](grl_snam/ariadne/README.md) describes each port. libcvc's
[docs/CARS.md](https://github.com/transfix/libcvc/blob/master/docs/CARS.md) is the template's guide: the state your controller gets, the worlds, the
plants and what each means for a controller, every setting with its unit and range, and batch runs.

### The learned-nav pipeline, end to end

From nothing but a world model to a rendered demo, in one command:

```bash
grl-snam pipeline path/to/austin_south           # build-sdf → train → multigoal video
```

…or step by step (each artifact is reusable):

```bash
grl-snam build-sdf path/to/austin_south                              # → nav_sdf.npz
grl-snam train     path/to/austin_south/nav_sdf.npz -o coef_sdf.pt --steps 1500
grl-snam capture multigoal path/to/austin_south coef_sdf.pt --minutes 3 -o drive.mp4
```

`build-sdf`/`train` run on CPU in a few minutes (GPU auto-used if present).
`capture` renders the real 3-D scene **offscreen** through VTK (the VolRover3 engine —
not a screen grab) and draws a **live metrics HUD** on every frame: the network's
predicted coefficients (α, β, γ), wall clearance, speed, escape mode, and building
penetration. `--source cvc` on `build-sdf` uses CVC's mesh-exact 3-D SDF instead of the
footprint EDT. See [docs/training-navigation-on-geometry.md](docs/training-navigation-on-geometry.md).

`--plant kinematic|dynamic|jolt` drives the same policy on a vehicle instead of the point
surrogate: nav's kinematic bicycle, libcvc's dynamic single-track vehicle, or a Jolt vehicle
(`--vehicle humvee`, `--mu` the ground's grip). The HUD then adds the vehicle's sideslip, tire
use and energy books. [docs/PLANTS.md](docs/PLANTS.md) covers every demo and bench on a plant,
and training through Jolt.

**On video:** the [plant reel](https://github.com/CVC-Lab/GRL-SNAM/releases/tag/plant-reel-2026-11)
has eight clips of the shipped checkpoint on Austin, each with its measured outcome and the command
that made it:
- **three successes:** the surrogate's multi-goal drive, and the kinematic bicycle and a Jolt Humvee
  on grip 0.3 arriving;
- **five failures:** a dead end, an orbit, sliding past the tire's peak, a spin-out on low grip, and
  a collision with the map unknown.

### Did it learn? Held-out losses, arms and claims

A falling training loss does not show that a policy learned anything. `grl_snam.learning` checks
three things on held-out data, each against two arms: the untrained policy the run started from,
and a constant one (the same coefficients everywhere, so no state dependence).

1. The per-term held-out losses: the trainer's own objective, scored for every arm on the same
   held-out units, and the trained run's own curve from its learning history.
2. The task metric on paired seeds, with a bootstrap CI.
3. In a planted exercise, the error of the recovered latent.

```bash
grl-snam coef-train --rollout bicycle --out run/coef.cvcnav \
    --history run/history.jsonl --checkpoint-every 20 --keep-best
grl-snam arms-eval --arm trained=run/coef.best.cvcnav \
    --arm untrained=run/coef.step000000.cvcnav \
    --arm constant=const:hist=run/history.jsonl --seeds 20 --out run/arms.json
grl-snam arms-eval --cases objective-coef --rollout bicycle --arm trained=run/coef.best.cvcnav \
    --arm untrained=run/coef.step000000.cvcnav \
    --arm constant=const:hist=run/history.jsonl --out run/objective.json
grl-snam learn-report run/history.jsonl --arms run/arms.json --arms run/objective.json \
    --json run/report.json
```

For `grl-snam train` the loss case is `objective-sdf`, on the run's own held-out blocks:

```bash
grl-snam arms-eval --cases objective-sdf --sdf nav_sdf.npz --split-history seed0/history.jsonl \
    --arm trained=seed0/coef_sdf.best_heldout.pt --arm untrained=seed0/coef_sdf.step000000.pt \
    --arm constant=const:hist=seed0/history.jsonl --label seed0 --out seed0.objective.json
grl-snam learn-report seed*/history.jsonl --arms seed0.objective.json --arms seed1.objective.json ...
```

- **The reader** (`grl_snam.learning.history`) reads any trainer's JSON-lines history by its
  conventions, with no per-schema adapter. It knows `kind`, `step`/`epoch`, `train`/`terms` beside
  `heldout`, `heads.<g>.{mean,spread,dtheta_rel}`, `grad_norm`, `best*` and `blob`/`checkpoint`.
  So it reads `coef-train`'s, `grl-snam train`'s and `train_coef_energy`'s histories, libcvc's
  `cvc.nav.coef_train_history/v1` and downstream trainers' alike. A held-out value with a
  training twin (the same name, or `<prefix>_<name>`) is a loss term. An `init` record (the
  untrained net) is the first point. `load(paths)` gives named series
  (`<kind>.<path>`), a checkpoint table with the pick, train/held-out term pairs and `band()`
  across seeds. `follow=True` re-reads a file a trainer is still writing.
- **The marks** (`grl_snam.learning.curves`) are read on each held-out curve after a 5-point
  running median: best, the Kneedle elbow, the plateau (a windowed Theil–Sen slope that stays
  small next to the curve's IQR) and divergence after best. Each one that compares a change to a
  threshold also clears a noise floor, so a flat, noisy curve gets no elbow and no divergence.
  `tests/fixtures/learning_marks_golden.json` holds synthetic curves and their marks, for other
  implementations to check against.
- **The arms** (`arms-eval`): an arm is a `.cvcnav` (a run's step-0 blob is the untrained arm),
  a `.pt`, or `const:…` (`const:1.2,3,4`, `const:beta=2.5`, `const:bias`, or `const:hist=H.jsonl`
  for the trained run's head means at its pick). Every arm runs on the same seeds through
  `coef_eval` and `scorecard_eval`. The output is `grl_snam.arms/v1`: per metric and arm, the
  per-seed values, mean, 95% CI and the paired difference to the reference arm (a seeded
  percentile bootstrap over seeds, 5000 resamples, pairs kept together).
  `python -m grl_snam.tools.coef_export --constant 1.2,3,4 c.cvcnav` writes a constant arm for
  libcvc's navigator, and `read_coef_mlp` loads any `.cvcnav` back into torch.
- **The held-out objectives** (`grl_snam.learning.objective`, `arms-eval --cases objective-sdf`
  or `objective-coef`) score each arm with the trainer's own loss terms, one unit per held-out
  unit, with no gradient and a forked torch RNG:
  - `objective-sdf`: `grl-snam train`'s loss, goal, col and reg per held-out pair (`--pairs`
    drawn with `--pair-seed` inside the run's held-out blocks, `--split-history`). Their mean is
    the `obj_<term>` the run's history logs.
  - `objective-coef`: coef-train's loss, goal and coll per held-out start (`--heldout-seed`,
    `--heldout-agents`, `--horizon`, `--grid`, `--rollout surrogate|bicycle`). Their mean is
    what the trainer's step record logs at a fixed net.

  A run with several training seeds is one table per seed (`--label`). `learn-report` takes
  them all (`--arms` repeated) and pools them: the CI resamples the runs and the units.
- **The report** (`learn-report`) marks each claim pass or fail:
  - `loss`, against the arms: on a loss case, each held-out term is lower for trained than for
    each other arm (the CI of the paired difference lies below 0);
  - `loss`, along the run: each held-out loss term at the last evaluation beats the first, with a
    CI over seeds (held-out metrics with no training twin, such as reach, get marks but no claim);
  - `selection`: the pick was made on held-out data, and is not step 0;
  - `task`: on each primary metric, trained beats each other arm (the CI of the paired
    difference excludes 0 on the better side);
  - `planted`: the recovered latent is within its bound, and the untrained net's is not.

  A claim the data cannot support fails and says why. `--strict` exits 1 on any failure.

  On the five-seed `grl-snam train` retrain in [docs/learning-validation.md](docs/learning-validation.md),
  the loss family fails against both arms. The picks' held-out objective is worse than untrained
  (+0.75 [+0.26, +1.35]), and it ties a constant at each pick's own head means (+0.03
  [−0.01, +0.07]). The one term that beats untrained, goal (−0.06), ties that constant
  (+0.0004).

### Live inside VolRover3

```bash
grl-snam demo --list                             # the registered demos
grl-snam demo austin-freedrive --bundle path/to/austin_south --checkpoint coef_sdf.pt
```

`grl-snam demo NAME` shells out to `volrover3 --run-job` with the demo module (set
`VOLROVER3_BIN` if `volrover3` isn't on `PATH`); you can also load the module from the
app's **Jobs tab → Load Script → Run as Job**. Each demo publishes the same live metrics
into the state tree under `grl_snam.metrics.*` — the hook for an in-app HUD panel — and
prints them to the console. Demos:

- **`austin-freedrive`** — end-to-end learned SDF free-drive, **no route**, live goal retarget.
- **`austin-learned`** — stagewise: an A* route spine + learned SDF local control (robust).
- **`austin-planner`** — the surrogate in its native sparse-obstacle regime.
- **`austin-patrol`** — a grounded A* street patrol (non-learned baseline).
- **`lab`** — the analytic terrain + an agent walking a draped loop.

Env for the Austin demos: `GRL_SNAM_SCENE_BUNDLE`, `GRL_SNAM_CHECKPOINT`, `GRL_SNAM_SDF`
(prebuilt `nav_sdf.npz`; else built at load).

### selftest & lab-demo

```bash
grl-snam selftest            # four PASS checks: goal-seeking, IPC-finite, deterministic
grl-snam lab-demo out.png    # offscreen snapshot of the analytic lab (omit PNG for a window)
```

### Dataset-specific extensions
Applied, dataset-specific work lives in **separate downstream projects** that depend on
GRL-SNAM (not the other way round) and keep their dataset-specific code and models out
of this general library. Such an extension reuses exactly the pieces shown above — the
differentiable surrogate, the coefficient network, and the `pycvc`/`pycvc_gl` scene
bindings — and versions its own additions in its own repository. This library stays
general; the extensions stay separate.

## Repository Structure

```
.
├── train_coef_energy.py       # Main training script for coefficient network
├── eval_coef_energy.py         # Evaluation and visualization
├── surrogate_robust.py         # Radius-aware surrogate dynamics
├── scripts/
│   ├── ring_dataset_maxmin.py  # Dataset generation utilities
│   ├── spline_stagewise6.py    # Stagewise planner implementation
│   ├── stagewise_dataset.py    # Stagewise episode generator
│   └── ...
├── experiments/                # Experimental configurations
│   ├── e1.py, e3.py, ...      # Various experimental setups
├── src/utils/
│   ├── dijkstra.py            # A* baseline implementation
│   ├── online_stage_manager.py # Stagewise navigation manager
│   └── ...
└── README.md
```

## Quick Start

### 1. Generate Training Data

First, generate stagewise navigation episodes:

```bash
python -m scripts.stagewise_dataset \
  --root ./nav_stagewise_hyperring \
  --num-episodes 500 \
  --case case1-tight
```

This creates a dataset of short navigation trajectories with local obstacle observations.

### 2. Train Coefficient Network

Train the Hamiltonian coefficient predictor (α, β, γ):

```bash
python -m train_coef_energy \
  --root ./nav_stagewise_hyperring \
  --epochs 50 \
  --bs 128 \
  --lr 1e-4 \
  --workers 4 \
  --outdir checkpoints/coef_energy \
  --w_friction 0.1 \
  --w_multi 0.5
```

**Training parameters:**
- `--w_friction`: Weight for friction matching loss (default: 0.1)
- `--w_multi`: Weight for multi-start robustness penalty (default: 0.5)
- `--w_prox`: Weight for the penetration penalty on the main rollout (default: 0.1; 0 = the pre-2026-10 objective)
- `--gamma_rel`: Use relative gamma scaling (flag)

### 3. Evaluate Trained Model

Visualize navigation performance:

```bash
python -m eval_coef_energy \
  --ckpt checkpoints/coef_energy/epoch_049.pt \
  --case case1-tight \
  --steps 800 \
  --fps 20 \
  --alpha_mode weight
```

**Evaluation modes:**
- `weight`: Map α to obstacle weights (W_j ← W_j × α_j)
- `radius`: Map α to effective radii (R_j ← R_j + k × α_j)
- `both`: Apply both mappings
- `none`: Ablation mode (ignore α)

**Optional flags:**
- `--correction`: Enable online history-based secant controller
- `--online_finetune`: Enable test-time fine-tuning

### 4. View Results

Generated outputs:
- `snaps_coef/<timestamp>/`: PNG frames for each timestep
- `rollout.gif`: Animated visualization
- `rollout.mp4`: Video encoding (if FFmpeg available)

## Core Components

### Coefficient Network (`CoefEnergyNet`)

Predicts Hamiltonian coefficients from local context:
- **α_j ≥ 0**: Per-obstacle barrier weights (via softplus)
- **β ≥ 0**: Goal attraction strength (via softplus)
- **γ ≥ 0**: Damping/friction coefficient (via softplus)

**Architecture:**
- Obstacle encoder (MLP)
- Goal encoder (MLP)
- Transformer fusion (2 layers, 4 heads)
- Per-coefficient prediction heads

### Surrogate Integrator (`integrate_surrogate_v2`)

Differentiable forward dynamics for training:
- Radius-aware clearance: `d = ||o - C_j|| - (R_j + margin × r_robot)`
- IPC-style barrier forces: `F_barrier = -Σ α_j × (db/dd) × n̂_j`
- Goal attraction: `F_goal = -β × (o - goal)`
- Friction damping: `F_friction = -γ × v`

### Multi-Start Robustness (`multi_start_penalty`)

Samples auxiliary starts near obstacles and penalizes penetrations:
```python
L_multi = multi_start_penalty(
    o0, v0, goal, C, R, mask,
    alphas, beta, gamma, d_hat, dt, H,
    ms_count=20,    # Number of aux starts
    ms_h=3,         # Short rollout horizon
    ms_dt_mult=4.0, # Enlarged timestep
    frac_range=(0.8, 0.98),  # each start moves U(frac_range) of the clearance toward the obstacle
)
```

Ensures robustness in near-contact scenarios. Each start draws its own fraction
(pass `generator=` for a reproducible draw); `frac_range=(0.9, 0.9)` reproduces the
pre-2026-10 penalty, whose `ms_count` rollouts were all the same start.

### Online Adaptation

**History-based Secant Controller** (`HistSecantController`):
- Rank-1 Jacobian approximation from consecutive frames
- Adjusts β, γ, and top-K α without extra simulations
- EMA smoothing for stability

**Test-Time Fine-Tuning** (`OnlineFinetuner`):
- Proximal regularization to checkpoint weights
- Constraint-based losses (speed limits, clearance)
- Updates only final layers for stability

## Experimental Configurations

The `experiments/` directory contains various scenario setups:

- `e1.py`: Basic tight corridor navigation
- `e3.py`: Multi-gap cluttered environments
- `e5.py`: Out-of-distribution generalization tests
- `e7.py`: Dense obstacle fields
- `e10.py`: Comparative baseline evaluations

Run experiments via:
```bash
python -m experiments.e1  # Example
```

## Baselines

Implemented comparison methods (in evaluation scripts):
- **Rigid A\***: Grid-based planning with inflated obstacles
- **Deformable A\***: Clearance-aware penalty costs
- **Potential Fields (PF)**: Reactive gradient descent
- **Control Barrier Functions (CBF)**: QP-based safety filters
- **Dynamic Window Approach (DWA)**: Local velocity sampling

All baselines use identical stagewise information constraints.

## Key Results

From our experiments (see paper for full details):

| Method      | SPL ↑ | Detour ↓ | Min. Clearance ↑ | Mapping % ↓ |
|-------------|-------|----------|------------------|-------------|
| PF          | 0.77  | 1.42     | 0.18             | 10.3        |
| CBF         | 0.96  | 1.04     | 0.32             | 11.2        |
| **GRL-SNAM**| 0.95  | 1.09     | 0.26             | **10.7**    |

✅ GRL-SNAM achieves near-CBF navigation quality with minimal mapping budget.

## Reproducing Paper Results

### Main Comparison (Table 1, Figure 4)
```bash
# Generate test environments
python -m scripts.stagewise_dataset --root ./test_envs --num-episodes 50

# Train model
python -m train_coef_energy --root ./test_envs --epochs 50

# Evaluate all baselines
python -m experiments.e10  # Runs full comparison suite
```

### Ablation Studies (Table 4)
```bash
# No friction term
python -m train_coef_energy --w_friction 0.0 --w_multi 0.0

# Multi-start only
python -m train_coef_energy --w_friction 0.0 --w_multi 0.5

# Friction only
python -m train_coef_energy --w_friction 0.1 --w_multi 0.0

# Both (default)
python -m train_coef_energy --w_friction 0.1 --w_multi 0.5
```

### Robustness Analysis (Table 5)
```bash
# Add noise parameters to evaluation
python -m eval_coef_energy \
  --ckpt checkpoints/best.pt \
  --noise_level 0.05  # Mild noise
```

## Hyperparameters

**Training (TrainCfg):**
- `epochs`: 50
- `batch_size`: 128
- `learning_rate`: 1e-4
- `w_traj`: 1.0 (trajectory loss weight)
- `w_vel`: 1.0 (velocity loss weight)
- `w_friction`: 0.1 (damping matching weight)
- `w_multi`: 0.5 (multi-start robustness weight)
- `w_prox`: 0.1 (penetration penalty on the main rollout, softness `prox_tau` 0.05 m)
- `margin_factor`: 0.5 (robot radius margin)

**Surrogate Integration:**
- `mass`: 1.0
- `d_hat`: 0.5 (IPC barrier activation distance)
- `dt`: 0.03 (base timestep)

**Online Adaptation:**
- Secant controller: `lr_beta=0.25, lr_gamma=0.05, lr_alpha=0.4`
- Test-time finetuning: `lr=1e-4, max_steps=1, prox_lambda=1e-3`

## Troubleshooting

**Issue: Training loss oscillates**
- Reduce learning rate: `--lr 5e-5`
- Increase friction weight: `--w_friction 0.2`

**Issue: Evaluation penetrates obstacles**
- Increase margin factor in config
- Check `d_hat` barrier threshold
- Verify obstacle radii are properly inflated

**Issue: Slow convergence**
- Increase batch size: `--bs 256`
- Reduce multi-start count during training
- Ensure sufficient training episodes (>500)

## Contact

For questions or issues, please open a GitHub issue.
