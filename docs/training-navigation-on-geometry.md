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
random free point, aim at a nearby goal 1.5–3.0 units away), rolls the surrogate
forward `H` steps with the net's predicted coefficients, and minimizes:

```
L = L_goal  +  8.0 * L_penetration  +  1.0 * L_speed
      │              │                      │
   reach the      keep clearance ≥        don't rush
   local goal     robot radius            (cap terminal speed)
```

all backpropagated through the rollout. **Local** goals are deliberate: a policy
trained to reach *nearby* goals learns obstacle-avoiding local progress that
*chains* into a full cross-map route at inference. Training on far goals instead
rewards rushing in a straight line — which clips buildings.

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
tuned zoom, `radn ≈ 0.16`, robot radius `≈ 0.035`). Widen `--region` to cover more
map at coarser zoom, or run several regions to cover a large city.

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

The demo scaffold is
[`examples/volrover_grl_snam_planner.py`](../examples/volrover_grl_snam_planner.py)
(load it from volrover3's Jobs tab). It currently ships with hand-set coefficients
in a known-good regime; loading a `coef_energy.pt` checkpoint swaps the hand-set
coefficients for the learned network + online adaptation — "the entire feature set
in play."

---

## 4. How long does training take?

Measured on this workstation (6 CPU threads, batch 64, horizon 24):

| Metric | Value |
|--------|-------|
| Per step | ~270–310 ms |
| 5000 steps | **~22–26 min** |
| Convergence | `L_goal` falls 100 → ~6 within the first ~1500 steps |
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
| Geometry ingestion | [`scripts/extract_obstacles.py`](../scripts/extract_obstacles.py) |
| Self-supervised trainer | [`scripts/train_on_geometry.py`](../scripts/train_on_geometry.py) |
| Live demo scaffold | [`examples/volrover_grl_snam_planner.py`](../examples/volrover_grl_snam_planner.py) |
| Scene helpers (terrain/glTF/occupancy) | `pycvc_gl.scenes` |

See the [Developer Guide](developer-guide.md) for the underlying stagewise
navigation model.
