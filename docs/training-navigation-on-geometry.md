# Training Navigation on Real Geometry

How to teach the GRL-SNAM navigation policy to drive a **specific real-world
scene** — a terrain heightfield plus a city mesh — and then run the trained
policy as a live demo (e.g. a vehicle finding its own way through a city in
volrover3).

This is the end-to-end path from *"here is a new map"* to *"the agent navigates
it"*. It needs **no expert demonstrations and no pre-generated dataset** — the
navigation surrogate is differentiable, so the policy trains itself directly on
the scene's obstacles.

---

## 1. What "training on geometry" means

GRL-SNAM navigation is a **differentiable, physics-informed potential field**.
The surrogate rollout (`surrogate_robust.integrate_surrogate_v2`) advances a point
agent under three forces:

- a **goal spring** (strength `beta`) pulling toward the target,
- **IPC-barrier repulsion** from each nearby obstacle (per-obstacle strength
  `alphas`), and
- **damping** (`gamma`).

A small transformer, **`CoefEnergyNet`** (`train_coef_energy.py`, re-exported as
`grl_snam.network`), predicts `(alphas, beta, gamma)` from the local situation:
the nearby obstacle circles and the goal direction. Good coefficients produce a
smooth, collision-free path; bad (or untrained) coefficients stall or clip walls.

**Training on a scene = fitting `CoefEnergyNet` so its predicted coefficients
navigate *that scene's* obstacles well.** The scene enters training as a set of
**circular obstacles** (building footprints) plus a pool of **free (drivable)
points**.

---

## 2. Do we have everything to train right now?

**Yes.** There are two ways to train the coefficient network; we use the second,
which requires nothing pre-generated:

| Mode | Signal | Needs | Use when |
|------|--------|-------|----------|
| **Imitation** | MSE against expert rollouts | an expert dataset from the ring/stagewise generators (`scripts.ring_dataset_maxmin`) + a base checkpoint | you already have curated expert trajectories |
| **Self-supervised** (this doc) | reach-goal + no-penetration + speed loss, **backprop through the differentiable rollout** | just `torch` + `numpy` + the repo | you have a **new scene** and no labels |

Because `integrate_surrogate_v2` is fully differentiable, the self-supervised mode
backpropagates a task loss straight through the H-step rollout — no labels, no
reward bootstrapping. It trains directly on **any** geometry.

**Requirements, all currently satisfied:**

- **Training** (`scripts/train_on_geometry.py`): `torch`, `numpy`, and the
  GRL-SNAM repo on `PYTHONPATH`. Runs on a plain CPU box or a GPU cluster — **no
  graphics**.
- **Geometry ingestion** (`scripts/extract_obstacles.py`): the `pycvc_gl` scene
  helpers (they rasterize the city mesh to an occupancy grid with VTK). Run this
  in an environment where volrover3's Python / cvcGL bindings are importable
  (i.e. the volrover deps prefix). Ingestion is a **one-time** step per scene and
  its output is cached.
- **The scene**: a `geometry_bundle` — `terrain.json` (heightfield) +
  `buildings.glb` (glTF city mesh).

---

## 3. The pipeline

```
 geometry bundle              obstacles.npz              coef_energy.pt
 (terrain.json +   ──[1 ingest]──▶  (obstacle circles  ──[2 train]──▶ (CoefEnergyNet   ──[3 run]──▶ live demo
  buildings.glb)                     + free points)                     weights + scale)
```

### Step 1 — Ingest the scene into an obstacle set

```bash
python scripts/extract_obstacles.py <bundle_dir> -o obstacles.npz
```

Rasterizes the city footprint to a **solid** top-down occupancy grid, coarsens it,
and emits one circular obstacle per occupied cell plus a pool of free world points.
The occupancy raster is cached next to the `.glb`, so re-runs are instant. The
`obstacles.npz` is self-contained (world-unit obstacle centers + radius + free
pool + bounds) and has no graphics dependency — it can be shipped and reused.

### Step 2 — Train the policy (self-supervised)

```bash
python scripts/train_on_geometry.py obstacles.npz -o coef_energy.pt \
    --steps 5000 --region 430
```

Each optimizer step samples a batch of **local** navigation problems (start at a
random free point, aim at a nearby *reachable* goal 1.0–2.0 units away), rolls the
surrogate forward `H` steps with the net's predicted coefficients, and minimizes:

```
L = L_goal  +  3.0 * L_collision  +  0.3 * L_coef_reg
      │              │                    │
   reach the      penalize ACTUAL      anchor coefficients to the
   local goal     penetration only     known-good navigating regime
                  (not proximity)      (beta~3, gamma~4, alpha~3)
```

all backpropagated through the rollout. Two hard-won details make or break this:

1. **Do not cap speed, and penalize collision — not proximity.** Streets are
   narrow, so navigating them *requires* low positive clearance. A speed cap or a
   proximity penalty makes staying put (max clearance, min speed) beat moving, and
   the net collapses to a **crawl** (damping `gamma` ≫ goal-spring `beta`, agent
   immobile). Penalize only *actual* penetration (clearance below a thin margin).
2. **Regularize toward the known-good regime.** The self-supervised rollout
   objective has degenerate optima (crawl, or erratic overshoot). Anchoring the
   predicted coefficients to the working hand-set values keeps the optimizer in the
   stable, navigating basin while the task terms adapt them per situation.

**Local** goals are deliberate: a policy trained to reach *nearby* goals learns
obstacle-avoiding local progress that *chains* into a route at inference — the
paper's **stagewise** decomposition (§Scope).

#### The scale-normalization step (important)

The surrogate's coefficients are tuned for a **~10-unit world**. Real scenes are
much bigger — the Austin bundle spans **3 km** (bounds ±1500 m). The trainer
normalizes a **working region** into the tuned regime:

- `--region R` picks a half-extent (metres) around the scene center;
- that `2R`-wide region maps to ~10 units for the rollout: `scale = 10 / (2R)`;
- obstacle radius and robot radius scale the same way;
- the checkpoint records `scale` + `center` so the demo maps world ↔ normalized
  consistently.

For Austin, `--region 430` gives `scale ≈ 0.0116` (a ~860 m working area at the
tuned zoom). Two geometry knobs matter in a real city: the obstacle circles come
from `extract_obstacles.py --block 2` (~7 m circles that leave 20 m streets
navigable — `radn ≈ 0.08`; coarser 14 m circles crowd them shut), and the IPC
barrier reach is kept **local** (`--d-hat-world 25`, ~one street width) so the
agent isn't stalled by a "sea of repulsion" from many overlapping barriers. Widen
`--region` to cover more map at coarser zoom, or run several regions for a large
city.

### Step 3 — Run the trained policy

At inference the loop, per step:

1. gathers the nearby obstacles + goal direction and builds local features
   (`eval_coef_energy.build_local_feats`);
2. `CoefEnergyNet` predicts `(alphas, beta, gamma)` (~7 ms/step — real-time);
3. **online adaptation** refines those coefficients on the fly:
   - **`HistSecantController`** (`grl_snam.adaptation`) — a rank-1 secant update
     that nudges the coefficients from the recent motion history, and
   - **`OnlineFinetuner`** — optional test-time training back through the
     differentiable rollout;
4. `integrate_surrogate_v2` advances one step; the agent is draped onto the
   terrain and rendered.

Two demo scaffolds load from volrover3's Jobs tab:

- [`examples/volrover_grl_snam_planner.py`](../examples/volrover_grl_snam_planner.py)
  — the learned policy's **native regime**: a handful of sparse circular obstacles,
  where the surrogate navigates end-to-end. Load a `coef_energy.pt` checkpoint to
  drive it with the learned network + online adaptation.
- [`examples/volrover_grl_snam_austin_learned.py`](../examples/volrover_grl_snam_austin_learned.py)
  — the **stagewise** demo on real Austin: an A* occupancy route is the
  collision-free spine, and the trained `CoefEnergyNet` + `HistSecantController` do
  the local reactive control between route sub-goals (see §Scope for why).

### Obstacle model — circles vs. SDF (why the city needs an SDF)

The **base** surrogate (`integrate_surrogate_v2`) is a **circular-obstacle** field.
It navigates cleanly when obstacles are sparse and round (rings, dungeons, a few
pillars — the `volrover_grl_snam_planner.py` regime). A **dense rectilinear city**
defeats it: thousands of building footprints become thousands of overlapping circular
barriers that conflict, and a point-agent is pushed *through* corners no matter the
coefficients (measured on Austin: even hand-set good coefficients reach 0–1/4 goals
with heavy penetration). That is a property of the obstacle model, not the training.

The fix is a **signed distance field (SDF)** obstacle model (`sdf_nav.py`): the
barrier repels along the *true wall normal*, so the learned surrogate navigates
streets and corners cleanly. On Austin, stagewise + SDF reaches **3/4 with ~0
penetration** and the learned surrogate **drives every step** (vs the circle model,
which drove none). Same self-supervised recipe — the field is differentiable
(`grid_sample`), so `sdf_nav.CoefMLP` trains through the rollout exactly like
`CoefEnergyNet`, biased toward the known-good regime so it starts near-optimal.

Build the SDF, then train:

```bash
python scripts/build_sdf.py <bundle> --source edt      # grl-snam's exact footprint EDT (default)
#                                     --source cvc      # OR CVC's mesh-exact 3-D SDF (pycvc.sdf, SDF_V2)
python scripts/train_sdf.py <bundle>/nav_sdf.npz -o checkpoints/coef_sdf.pt --steps 1500
```

`--source cvc` uses the CVC compute layer and returns a full **3-D** volume (sliced
to the ground plane for 2-D nav) — the natural substrate for extending GRL-SNAM to
3-D navigation later. The `SDFField` / surrogate / net consume either source
identically.

**Global topology still needs a planner.** A pure potential field — SDF or not — has
local minima (dead-ends, U-shaped clusters), so a long cross-city A→B can still trap.
So the demo keeps the paper's **stagewise** structure: an occupancy-grid A\* route is
the collision-free spine (global path), and the *learned SDF surrogate* does the
local navigation between route sub-goals (now genuinely, not as a fallback). The
route + a footprint check guarantee the drive never enters a building.

---

## 4. How long does training take?

Measured on this workstation (6 CPU threads, batch 64, horizon 28):

| Metric | Value |
|--------|-------|
| Per step (`--block 2`, ~28k obstacles) | ~270–400 ms |
| 5000 steps | **~22–33 min** |
| Convergence | the goal term drops steadily; per-stage reaching becomes reliable within the first few thousand steps |
| Inference | ~7 ms/step (real-time driving) |

A **GPU** cuts training roughly 10–20× (single-digit minutes for 5000 steps); the
rollout and transformer are both small and batch-parallel. Ingestion (step 1) is
seconds, and cached thereafter.

So a full "new scene → trained policy" cycle is well under an hour on CPU and a few
minutes on GPU.

---

## 5. Pre-generated datasets (future)

Nothing above depends on pre-generated data, but two artifacts are worth reusing:

- **`obstacles.npz`** — ship the ingested obstacle set so a training box never
  needs the graphics stack. A "pre-generated dataset" for self-supervised training
  is simply this file (obstacle circles + free pool).
- **`coef_energy.pt`** — ship a trained checkpoint so the demo runs with zero
  training. Re-train only when the geometry changes or you want a different
  working region / behavior.

When curated **expert** trajectories become available, the imitation mode
(`train_coef_energy.py`'s `Trainer`) can pre-train or fine-tune `CoefEnergyNet`
against them; self-supervised training on the scene geometry then adapts it to the
specific map. The two are complementary.

---

## 6. Reference

| Piece | Where |
|-------|-------|
| Differentiable surrogate rollout | `surrogate_robust.integrate_surrogate_v2` |
| Coefficient network | `CoefEnergyNet` (`train_coef_energy.py` → `grl_snam.network`) |
| Local feature builder | `eval_coef_energy.build_local_feats` |
| Online adaptation | `HistSecantController`, `OnlineFinetuner` (`grl_snam.adaptation`) |
| Geometry ingestion (circles) | [`scripts/extract_obstacles.py`](../scripts/extract_obstacles.py) |
| Self-supervised trainer (circles) | [`scripts/train_on_geometry.py`](../scripts/train_on_geometry.py) |
| **SDF obstacle model** (city-grade) | [`sdf_nav.py`](../sdf_nav.py) — EDT + cvc::sdf builders, `SDFField`, `sdf_rollout`, `CoefMLP` |
| SDF field builder (`edt`/`cvc`) | [`scripts/build_sdf.py`](../scripts/build_sdf.py) |
| SDF trainer | [`scripts/train_sdf.py`](../scripts/train_sdf.py) |
| Live demo (native sparse-obstacle regime) | [`examples/volrover_grl_snam_planner.py`](../examples/volrover_grl_snam_planner.py) |
| Live demo (stagewise, real Austin) | [`examples/volrover_grl_snam_austin_learned.py`](../examples/volrover_grl_snam_austin_learned.py) |
| Scene helpers (terrain/glTF/occupancy/route) | `pycvc_gl.scenes` |

See the [Developer Guide](developer-guide.md) for the underlying stagewise
navigation model.
