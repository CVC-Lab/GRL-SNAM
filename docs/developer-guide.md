# GRL-SNAM Developer Guide

**Geometric Reinforcement Learning for Simultaneous Navigation and Mapping**

*Reference paper: Ellendula, Wang, Nguyen, Bajaj — arXiv:2601.00116v1 (Dec 2025)*

---

## 1. What Is GRL-SNAM?

GRL-SNAM is a framework for navigating a robot from a start position to a goal in an **unknown 2D environment** without a pre-built global map. Instead of constructing a full map and planning on it, the robot only senses a small local window around itself at each moment and makes navigation decisions based on that partial information. The key innovation is formulating this problem using **Hamiltonian mechanics**: the robot learns an energy landscape where valleys correspond to safe, efficient paths, and navigation reduces to following the energy gradient downhill.

The framework is evaluated on two tasks:
1. **Hyperelastic ring navigation** — a deformable ring robot that can squeeze through gaps narrower than its resting diameter.
2. **Dungeon point-agent navigation** — a point mass navigating indoor maze layouts.

---

## 2. Core Concepts

### 2.1 Stages and the Stage Manager

The most important architectural concept to understand is **stagewise decomposition**. Rather than planning a single long path from start to goal, GRL-SNAM decomposes the journey into a chain of short hops through overlapping rectangular regions called **stages**.

**A stage** is an axis-aligned bounding box (AABB) — a rectangular region of the 2D workspace — with:
- A **center** — the geometric midpoint of the box
- A **size** — default `(W=2.6, H=2.0)` meters (width along the path direction, height orthogonal)
- An **entry point** — where the robot enters the stage (the previous stage's exit)
- An **exit point** — where the robot must reach to graduate to the next stage
- **bounds** — `(xmin, xmax, ymin, ymax)` derived from center and size

The full navigation path looks like:

```
START → [Stage 0] → exit₀ → [Stage 1] → exit₁ → ... → [Stage N-1] → GOAL
```

Adjacent stages **overlap** by a configurable fraction (default 30% of width = 0.78m). The exit point of stage *i* lies within this overlap region, ensuring a smooth handoff. This overlap zone is sometimes referred to informally as the **stageline** — the frontier or boundary between consecutive stages where the robot transitions from one local planning context to the next.

**Offline StageManager** (used in training data generation and evaluation — `spline_stagewise6.py`):
1. Computes the direction vector from start to goal
2. Places stage centers uniformly along this line, spaced by `W × (1 - overlap_ratio)`
3. For each consecutive stage pair, finds a collision-free exit point in the overlap region using Dijkstra on a KNN visibility graph

**Online StageManager** (`src/utils/online_stage_manager.py`):
- Builds stages reactively, one at a time, as the robot moves
- When the robot reaches the boundary of its current stage, it proposes a new box ahead in the direction of travel + goal
- Includes stuck detection and recovery: if progress stalls, it tries random relocations of the next-box center

Stage advancement triggers when either:
- The robot center is within a threshold of the current exit point (0.45m offline, 0.15m online)
- The robot has overshot the stage center in the goal direction

### 2.2 What Is a "Stageline"?

The term **stageline** is used informally to refer to the **boundary between adjacent stages** — specifically the edge of the overlap region where the robot transitions from one stage to the next. In the codebase, this manifests as the `exit_point` of each `Stage` object (which equals the `entry_point` of the next stage). There is no explicit `Stageline` class; the concept is encoded in:

- `stage_exit` — the per-frame subgoal stored in training data checkpoints
- `StageManager.advance_if_needed()` — the logic that detects when the robot crosses the stageline
- The overlap rectangle computed between adjacent stages in `_rect_overlap()`

Conceptually, crossing a stageline means: the robot has completed its current local planning task and is ready to receive a new local goal (the next stage's exit point) and a new local obstacle context.

### 2.3 Hamiltonian Mechanics Formulation

The paper's central insight: instead of learning a policy $\pi(a|s)$ that maps states to actions (standard RL), GRL-SNAM learns **Hamiltonian coefficients** $(\alpha_j, \beta, \gamma)$ that parameterize a physics-based energy landscape:

$$H(q, p) = \underbrace{\frac{1}{2m}\|p\|^2}_{\text{kinetic}} + \underbrace{\beta \cdot V_{\text{goal}}(q)}_{\text{goal attraction}} + \underbrace{\sum_j \alpha_j \cdot b(d_j)}_{\text{obstacle barriers}} + \underbrace{\gamma \cdot D(p)}_{\text{damping}}$$

Where:
- $\beta \geq 0$ — goal attraction strength (how strongly the robot is pulled toward the current stage exit)
- $\alpha_j \geq 0$ — per-obstacle barrier weight (how strongly obstacle $j$ repels the robot)
- $\gamma \geq 0$ — damping coefficient (energy dissipation for stability)
- $b(d_j)$ — IPC (Incremental Potential Contact) barrier function, a log-barrier that activates when distance $d_j < \hat{d}$

The resulting force field is:

$$F = -\beta(o - g) - \sum_j \alpha_j \frac{\partial b}{\partial d}(d_j) \hat{n}_j - \gamma v$$

Navigation then becomes forward integration of these dynamics — no Bellman equation, no value function, no reward engineering.

---

## 3. Spatial Geometry

### 3.1 Workspace

The workspace is a 2D rectangular region. For the hyperelastic ring task:
- Default bounds: $x \in [-0.5, 9.0]$, $y \in [-1.8, 1.8]$
- Start: e.g. `[0.0, 2.5]`, Goal: e.g. `[9.0, -1.2]`

### 3.2 Obstacles

All obstacles are **circles** defined by:
- **Center** $c_j \in \mathbb{R}^2$ — position in the 2D plane
- **Radius** $r_j$ — geometric radius
- **Weight** $w_j$ — scaling factor for the barrier potential (controls repulsion strength)

For planning (exit-point finding), obstacles are **inflated** by a tube radius (default 0.05m) to ensure clearance. For the surrogate dynamics, obstacles are further inflated by `margin_factor × robot_radius` (default 0.5 × robot radius).

Obstacles are generated procedurally via:
- `sample_obstacles_case1_tight()` — creates a corridor with narrow gates, intruder obstacles, and Poisson-distributed clutter
- `sample_obstacles_case2_harder()` — sinusoidal corridor, more gates, clustered clutter, wall belts

### 3.3 Robot Geometry (Hyperelastic Ring)

The robot is a **periodic cubic B-spline ring** with:
- `nctrl = 20` control points — defining the ring shape
- `K = 240` sample points — for force evaluation and collision checking
- A **uniform scale** parameter $s(t) > 0$ — the single deformation DOF (squeeze or expand)
- **Translation** $o(t) \in \mathbb{R}^2$ and **rotation** $\theta(t)$

Reference shape: control points form a unit circle of radius `rbase`, transformed to world coordinates by $P_i(t) = o(t) + s(t) \cdot P_{0,i}$. The bulk energy penalizes area deviation from a target that depends on local clearance — in tight passages, the target area shrinks (encouraging squeezing).

### 3.4 Local Sensing Window

At each timestep, the robot observes obstacles within a local window $W(q_t)$ of size $2\hat{d} \times 2\hat{d}$ centered at its position. The **mapping ratio** quantifies sensing effort:

$$\rho_{\text{map}} = \frac{\text{area}(\bigcup_t W(q_t))}{L^2}$$

GRL-SNAM achieves ~10-11% mapping ratio while maintaining near-planner path quality.

---

## 4. Code Architecture

### 4.1 Repository Structure

```
GRL-SNAM/
├── train_coef_energy.py       # Main training script
├── eval_coef_energy.py        # Evaluation and visualization
├── surrogate_robust.py        # Radius-aware surrogate dynamics
├── scripts/
│   ├── spline_stagewise6.py   # Core planner: robot model, stage manager, force field
│   ├── stagewise_dataset.py   # Dataset generation (environments + episodes)
│   ├── ring_dataset_maxmin.py # Dataset utilities
│   └── ...
├── experiments/
│   ├── e1.py ... e10.py       # Experimental configurations
├── src/utils/
│   ├── dijkstra.py            # Grid-based exit solvers (Dijkstra, BFS, exhaustive)
│   ├── online_stage_manager.py # Online reactive stage manager
│   └── tensorboard_logger_mixin.py
```

### 4.2 Data Pipeline

```
                    ┌─────────────────────────────────┐
                    │   stagewise_dataset.py           │
                    │   - Generate random environments │
                    │   - Run episodes with planner    │
                    │   - Save .pt + .jsonl per episode│
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │   train_coef_energy.py            │
                    │   ShortRollouts dataset           │
                    │   - Random short windows from     │
                    │     stagewise checkpoints         │
                    │   - Horizon H ∈ [2,6] steps       │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │   CoefEnergyNet                   │
                    │   - Obstacle encoder (MLP)        │
                    │   - Goal encoder (MLP)            │
                    │   - Transformer fusion (2L, 4H)   │
                    │   - α, β, γ prediction heads      │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │   surrogate_robust.py             │
                    │   integrate_surrogate_v2          │
                    │   + multi_start_penalty           │
                    │   → predicted (oT, vT, clearance) │
                    └───────────────┬──────────────────┘
                                    │
                         Loss = w_traj·‖oT-o_tgt‖²
                               + w_vel·‖vT-v_tgt‖²
                               + w_friction·‖γ-γ₀‖²
                               + w_multi·L_multi
```

### 4.3 Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `CoefEnergyNet` | `train_coef_energy.py` | Neural network predicting (α,β,γ) from obstacle/goal context |
| `ShortRollouts` | `train_coef_energy.py` | Dataset: random short trajectory windows from stagewise episodes |
| `Trainer` | `train_coef_energy.py` | Training loop: forward → integrate → loss → backprop |
| `HyperelasticRingSystem` | `scripts/spline_stagewise6.py` | The deformable robot: B-spline ring with scale, rotation, translation |
| `StageManager` | `scripts/spline_stagewise6.py` | Offline stage chain builder (pre-plans all stages at init) |
| `StageForceFieldTorch` | `scripts/spline_stagewise6.py` | Per-stage force field: goal + radial + tangent + flow + boundary |
| `IPCBarrier` | `scripts/spline_stagewise6.py` | IPC log-barrier energy and force computation |
| `StagewiseHyperelasticPlanner` | `scripts/spline_stagewise6.py` | Top-level planner wrapping StageManager + RingSystem + ForceField |
| `StageManagerOnline` | `src/utils/online_stage_manager.py` | Online reactive stage builder with stuck detection |
| `HistSecantController` | `eval_coef_energy.py` | Online correction via rank-1 secant Jacobian estimation |
| `OnlineFinetuner` | `eval_coef_energy.py` | Test-time fine-tuning of prediction heads |

---

## 5. How the Code Works End-to-End

### 5.1 Dataset Generation (`scripts/stagewise_dataset.py`)

1. **Generate obstacles**: procedural placement of corridor walls, gates, intruders, and Poisson clutter
2. **Build planner**: create `StagewiseHyperelasticPlanner` with the offline `StageManager`
3. **Run episode**: step the planner for up to 2000 timesteps at dt=0.03
   - Each step: compute forces (IPC barriers + stage field + bulk energy), integrate dynamics, check for stage advancement
   - Log each frame to `stagewise_checkpoints.jsonl` with: robot state `(o, v, θ, s)`, current `stage_exit`, `obstacles_effective` (centers, radii, weights in the current stage window), `d_hat`
4. **Save**: episode `.pt` file + manifest

### 5.2 Training (`train_coef_energy.py`)

1. **Load dataset**: `ShortRollouts` reads `.jsonl` checkpoint files, builds an index of extractable short windows
2. **Sample batch**: for each item, randomly pick an episode and time offset, extract a window of $H \in [2,6]$ steps
   - Input features: per-obstacle `[cx, cy, r, w, dx_goal, dy_goal]` (6D) + goal `[dg_x, dg_y, ‖dg‖, 1.0]` (4D)
   - Target: `(o_target, v_target)` at the end of the window
3. **Forward**: `CoefEnergyNet` predicts `(α[B,N], β[B], γ[B])`
4. **Integrate**: `integrate_surrogate_v2` runs the surrogate dynamics forward $H$ steps from `(o0, v0)` using the predicted coefficients
5. **Multi-start penalty**: `multi_start_penalty` samples 20 perturbed starts near the nearest obstacle, runs short rollouts, penalizes any penetrations
6. **Loss**: weighted sum of position MSE, velocity MSE, friction matching, and multi-start penalty
7. **Backprop**: gradient through the differentiable integrator, clip at 5.0, Adam update

### 5.3 Evaluation (`eval_coef_energy.py`)

1. **Load checkpoint**: restore trained `CoefEnergyNet`
2. **Create environment**: sample obstacles, build planner with `StageManager`
3. **Rollout loop** (e.g. 800 steps):
   - `planner.stage_slice()` crops obstacles to current stage window
   - Build features, predict `(α, β, γ)` with the trained network
   - **Map α to world**: scale obstacle weights (`W_j ← W_j × α_j`) and/or inflate radii (`R_j ← R_j + k·α_j`)
   - Inject `β` → stage field goal weight, `γ` → robot damping
   - Rebuild world obstacles, step planner forward
   - Optionally apply `HistSecantController` for online correction (after 5 steps in a stage)
   - Optionally apply `OnlineFinetuner` for test-time head tuning
   - Detect stage transitions → reset correction controller
   - Capture PNG frames
4. **Output**: GIF/MP4 animation of the rollout

### 5.4 Online Adaptation (Section 3.3 of the paper)

When deployed in a new environment, the frozen network may produce suboptimal coefficients. Two lightweight online mechanisms correct this without retraining:

**HistSecantController**: Observes a 3D signal $y = [-\text{clearance}, \text{distance}, -\text{speed}]$ and a target $y^*$. Estimates a rank-1 Jacobian $J \approx \partial y / \partial \zeta$ from consecutive frames using exponential moving average. Solves a Tikhonov-regularized least-squares update $\Delta\zeta = (J^T J + \lambda I)^{-1} J^T (y^* - y)$ to adjust $(\beta, \gamma, \alpha_{\text{top-K}})$.

**OnlineFinetuner**: Freezes all network parameters except the prediction heads. Runs 1-2 Adam steps on a rollout + constraint loss (speed limits, clearance) with proximal regularization to the checkpoint weights.

---

## 6. The Three-Policy Architecture

The paper decomposes navigation into three policies operating at different timescales (Figure 9 of the paper):

| Policy | Controls | Timescale | What It Does |
|--------|----------|-----------|-------------|
| **Sensor** $\pi_y$ | $y \in Q_y$ | Slow (once per stage) | Queries environment, establishes obstacle constraints $C_t$ |
| **Frame/Path** $\pi_f$ | $c \in Q_f$ | Medium (within each stage) | Computes waypoints toward stage exit, controls translation |
| **Shape** $\pi_o$ | $\psi \in Q_o$ | Fast (every integration step) | Adapts robot morphology (scale $s$) for squeezing |

This creates a temporal hierarchy: shape equilibrates within each frame update, frame settles before the sensor policy evolves. In the codebase:
- The sensor policy maps to stage transitions and obstacle context updates
- The frame policy maps to the `StageForceFieldTorch` driving translation
- The shape policy maps to the bulk energy term in `HyperelasticRingSystem` that drives uniform scaling

---

## 7. Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage_size` | `(2.6, 2.0)` | Width × Height of each stage box (meters) |
| `overlap` | `0.30` | Fraction of width shared by adjacent stages |
| `inflate` | `0.05` | Obstacle inflation for collision-free path planning |
| `d_hat` | `0.5–1.0` | IPC barrier activation distance |
| `dt` | `0.03` | Integration timestep |
| `radius` | `0.30–0.50` | Nominal robot ring radius |
| `nctrl` | `20` | B-spline control points on the ring |
| `K` | `240` | Sample points on the ring boundary |
| `epochs` | `50` | Training epochs |
| `bs` | `128` | Batch size |
| `lr` | `1e-4` | Learning rate |
| `w_traj` | `1.0` | Position matching loss weight |
| `w_vel` | `1.0` | Velocity matching loss weight |
| `w_friction` | `0.1` | Damping matching loss weight |
| `w_multi` | `0.5` | Multi-start robustness penalty weight |
| `margin_factor` | `0.5` | Obstacle radius inflation = margin_factor × robot_radius |
| `ms_count` | `20` | Number of perturbed starts for multi-start penalty |
| `mass` | `1.0` | Robot translational mass |
| `k_bulk` | `1.5` | Bulk modulus (area stiffness) for hyperelastic ring |

---

## 8. Quick Start

### Generate training data
```bash
python -m scripts.stagewise_dataset \
  --root ./nav_stagewise_hyperring \
  --num-episodes 500 \
  --case case1-tight
```

### Train the coefficient network
```bash
python -m train_coef_energy \
  --root ./nav_stagewise_hyperring \
  --epochs 50 --bs 128 --lr 1e-4 \
  --w_friction 0.1 --w_multi 0.5 \
  --outdir checkpoints/coef_energy
```

### Evaluate
```bash
python -m eval_coef_energy \
  --ckpt checkpoints/coef_energy/best.pt \
  --case case1-tight \
  --steps 800 --fps 20 \
  --alpha_mode weight
```

Alpha modes for evaluation:
- `weight` — map α to obstacle weights: $W_j \leftarrow W_j \times \alpha_j$
- `radius` — map α to effective radii: $R_j \leftarrow R_j + k \cdot \alpha_j$
- `both` — apply both mappings
- `none` — ablation mode (ignore α)

---

## 9. Baselines Implemented

| Method | Type | Map Access | Description |
|--------|------|-----------|-------------|
| Rigid A* | Global planner | Full | A* on inflated grid (rigid disc) |
| Deformable A* | Global planner | Full | Clearance-aware A* with deformation cost |
| Potential Fields (PF) | Local reactive | Stagewise | Attractive + repulsive potentials |
| Control Barrier Functions (CBF) | Local reactive | Stagewise | QP-based safety filter |
| Dynamic Window Approach (DWA) | Local reactive | Stagewise | Velocity sampling + scoring |
| PPO / SAC / TRPO | Deep RL | Full or stagewise | Standard RL with Kin/Dyn/Coef parameterizations |

All stagewise baselines share the same stage manager and local sensing window as GRL-SNAM.

---

## 10. Key Results Summary

| Method | SPL ↑ | Detour ↓ | Min. Clearance ↑ | Mapping % ↓ |
|--------|-------|----------|------------------|-------------|
| PF | 0.77 | 1.42 | 0.18 | 10.3 |
| CBF | 0.96 | 1.04 | 0.32 | 11.2 |
| **GRL-SNAM** | **0.95** | **1.09** | **0.26** | **10.7** |
| Best RL (TRPO-Coef) | 0.57 | 1.44 | 0.004 | 15.3 |

GRL-SNAM matches CBF-level path efficiency with PF-level mapping budget — the best of both worlds.

---

## 11. Important Implementation Details

### IPC Barrier Function
The barrier uses piecewise log form (Equation 26 in the paper):

$$b_{\text{IPC}}(d; \hat{d}) = \begin{cases}
-(d - \hat{d})^2 \ln(d / \hat{d}) & 0 < d < \hat{d} \\
0 & d \geq \hat{d} \\
V_{\text{penalty}} & d \leq 0
\end{cases}$$

This produces smooth repulsion that increases as the robot approaches an obstacle surface, with a hard penalty for actual penetration.

### Surrogate Integrator (v2)
The training dynamics use **semi-implicit Euler** (velocity updates first, position uses new velocity):
```python
a = (F_barrier + F_goal - γ * v) / mass
v ← v + dt * a        # velocity update first
o ← o + dt * v        # position uses new velocity
```

This provides better energy conservation than explicit Euler and makes the full pipeline differentiable end-to-end through PyTorch autograd.

### Feature Encoding
Per-obstacle features (6D): `[cx, cy, r, w, dx_goal, dy_goal]` where `(dx_goal, dy_goal)` is the vector from the obstacle center to the goal.

Goal features (4D): `[dg_x, dg_y, ‖dg‖, 1.0]` where `(dg_x, dg_y)` is the vector from robot to goal.

The constant `1.0` is a bias term.

---

## 12. Experiment Configurations

The `experiments/` directory contains scenario configs:

| File | Description |
|------|-------------|
| `e1.py` | Basic tight corridor navigation |
| `e3.py` | Multi-gap cluttered environments |
| `e5.py` | Out-of-distribution generalization |
| `e7.py` | Dense obstacle fields |
| `e10.py` | Full comparison suite (Table 1 results) |

Run via: `python -m experiments.e1`

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **Stage** | Rectangular region (AABB) the robot navigates through; part of a chain from start to goal |
| **Stage exit / stageline** | The handoff point between adjacent stages in the overlap region |
| **IPC** | Incremental Potential Contact — log-barrier function for smooth collision avoidance |
| **CoefEnergyNet** | Neural network that predicts Hamiltonian coefficients (α, β, γ) from local context |
| **Surrogate dynamics** | Simplified rigid-point dynamics used for differentiable training |
| **DfPO** | Differential Policy Optimization — the paper's term for gradient-based Hamiltonian learning |
| **Mapping ratio** | Fraction of workspace area ever observed through local sensing windows |
| **SPL** | Success-weighted Path Length — path efficiency metric relative to a reference planner |
| **Multi-start penalty** | Robustness term: samples perturbed starts near obstacles and penalizes penetrations |
| **Secant controller** | Online adaptation mechanism using rank-1 Jacobian estimation from consecutive observations |

---

*This guide was generated from the codebase and the reference paper (arXiv:2601.00116v1). For theoretical details, see Sections 3–4 of the paper. For experimental methodology, see Section 5.*
