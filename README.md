# GRL-SNAM: Geometric Reinforcement Learning for Navigation and Mapping

This repository contains the implementation of **GRL-SNAM**, a geometric reinforcement learning framework for simultaneous navigation and mapping in unknown environments using Hamiltonian mechanics and differential policy optimization.

📖 **Documentation:** [Developer Guide](docs/developer-guide.md) · [PyPI Publishing Roadmap](docs/pypi-publishing-roadmap.md)

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
| `grl-snam capture drive\|multigoal …` | drive the policy → mp4 offscreen, with a live HUD |
| `grl-snam pipeline BUNDLE` | the whole chain: world model → SDF → train → video |
| `grl-snam demo NAME` | run a demo live inside VolRover3 |
| `grl-snam lab-demo [PNG]` | standalone lab visualization |
| `grl-snam eval` / `train-coef` | the CoefEnergyNet evaluator / trainer (legacy track) |

A `BUNDLE` is a **world model** — a directory with `terrain.json` (heightfield) +
`buildings.glb` (city mesh), e.g. an OpenStreetMap + SRTM export. The heavy work lives
in `grl_snam.tools`; the demos in `grl_snam.demos`; the graphics deps (`pycvc_gl`, VTK,
`vrhost`) load lazily, so `grl-snam --help` runs anywhere.

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
general; the extensions stay separate and private.

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
    ms_dt_mult=4.0  # Enlarged timestep
)
```

Ensures robustness in near-contact scenarios.

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
