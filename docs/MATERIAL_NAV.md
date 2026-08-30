# Material-aware navigation

> **Campaign roadmap:** the full end-to-end torch-free C++/CUDA port (learned
> model + training loop for TACC campaigns) is planned in
> [CVCNAV_MATERIAL_PORT_ROADMAP.md](CVCNAV_MATERIAL_PORT_ROADMAP.md). This
> document is the merged base feature.


Port of the material-aware GRL-SNAM extension (published by the core
researcher at `github.com/SetasAditya/material-aware-grl-snam`) into the main
repo, plus its `cvc::nav` C++ twin in libcvc. This document is the design
record: what the method is, what was ported at which fidelity, where the sim
integration deliberately deviates from the source, and the parity contract.

## The method

The source augments the geometry-only port-Hamiltonian navigation field with
an explicit material-risk force, an always-on hard-hazard force, and a local
feasibility-witness gate controlling when the material force is exposed:

    F = F_geom + g(context) * lam_soft * f_material + lam_hard * f_hazard

    f_material = -grad r~            r~   = smoothed material risk in [0, 1]
    f_hazard   = -(db/dphi) grad phi phi  = unsigned metres distance to the
    db/dphi    = -sigmoid(k (d_hat - phi))       nearest hard-hazard cell

The gate `g` is a frame-wise witness: K rays uniform over 360 degrees; a ray
counts only if its endpoint makes progress toward the goal and every sample
clears hard terrain; the gate activates iff a feasible ray exists, the
straight-to-goal ray's mean risk reaches a trigger, and the best ray improves
on it by a margin. The gate multiplies `lam_soft` ONLY — `lam_hard` is never
gated. The witness never chooses the executed action.

## What was ported (three layers)

### 1. Research core — `material_nav.py` (root, torch, faithful)

`CoefEnergyNetMaterial` (state-dict compatible with the researcher's
checkpoints), `integrate_surrogate_material` (6-channel rollout patch,
semi-implicit Euler), `sdf_barrier_grad`, `bilinear_sample_patch`, and the
frame-wise `primitive_feasibility_gate`, verbatim in math and semantics
(fork units: pixels/metres, float32). Proven bit-identical to the source
implementation across every surface; `tests/test_material_fork_xcheck.py`
re-proves it against a live checkout (`GRL_SNAM_MATERIAL_FORK=<path>`).

Deliberately NOT ported: the frozen "v7" repair controller (failed its
preregistered efficacy stage gate — retained upstream for reproducibility
only), the `route_aware_stage2` grid heuristic (the paper's dyn-table
controller, a discrete route-following baseline distinct from the executed
field; this port carries the executed-field method and does not claim to
reproduce the paper's dynamic table numbers), and the highway lateral/TTC
channels (`mu_lat` is kept in the model class for checkpoint compatibility,
unused off-highway).

### 2. Sim integration — `grl_snam/material.py` (default-on when attached)

`MaterialGrid(risk, hard, bounds, center, scale)` attaches to `FogScenario`,
`Squad`, or `Swarm` and turns the whole feature on in pure Python: the
planner pays a per-cell risk surcharge (soft bias + a large-but-finite hard
penalty; composed additively with a user `route_cost_fn`, never clobbering
it), the drive feels both forces, and the witness gate modulates the soft
channel every tick. No MaterialGrid — the default everywhere — leaves every
existing trajectory bit-for-bit unchanged (asserted by the test suite; an
all-zero grid is also value-identical).

Documented deviations from the source (each with its reason):

* **Units.** phi stays in world METRES end-to-end (no barrier rescale trap);
  risk gradients are per NORMALIZED unit so `F_soft` composes with the goal
  spring's scale; `np.gradient` replaces the source's `sobel/(2 res)` (which
  silently carries a ~4x magnitude factor).
* **Force constants are sim-frame retunes**, validated behaviorally, not
  ported numbers: the source's `lam_soft = 1.5` launches a vehicle off-world
  in this frame, and its 3 m barrier reach is under ONE grid cell here (six
  cells on its own 0.5 m/cell BEV) — reach is kept cell-relative
  (`d_hat = 12 m`, `k*d_hat` preserved). Defaults: `lam_soft 0.5`,
  `lam_hard 1.0`, `k 1.25/m`, `d_hat 12 m`.
* **The material force STEERS the bicycle.** The source integrates a point
  mass, so its force bends the trajectory directly; a bicycle discards the
  lateral force component, and an F-sum-only port degenerates to speed
  modulation. The material force therefore also joins the steering bias
  (`tanh((F_rep + F_mat) . left)`), alongside the existing barrier bias.
* **Gate feasibility includes occupancy** (`gate_hard = material.hard | occ`):
  a ray through a building is not evidence of a feasible detour. FogScenario
  gates against its belief-composited planning surface; the planner-less
  Swarm (and C++ `sim_world`) gate against truth — the source's oracle-maps
  setting.
* **Gate geometry is metre-denominated** (`horizon_m = 25` ~ the source's 12
  cells at the default story grid; `hard_margin_m = None` -> 2 cells) so the
  parameters survive grid-resolution changes.
* **Scope.** The executed field is a LOCAL layer; in the source it always ran
  under a planner's waypoint scaffold. Same here: with a planner the cost
  surcharge routes around hazards and forces polish locally; a planner-less
  reactive agent gets the no-entry guarantee but can dead-end against a
  hazard squarely blocking its goal line (a potential-field minimum, as in
  the source's own field). Material-as-belief (the source's "Setting 3"
  perception track) is out of scope — material is oracle world truth.

### 3. C++ twin — libcvc `cvc::nav` (`inc/cvc/nav/material.h`)

`material_build` (derived planes), `witness_gate(+batch)`,
`material_sample`, `bicycle_rollout_material` / `drive_step_material`
(new entry points — existing signatures untouched, so version skew degrades
cleanly), and `sim_world::set_material` for pure-C++ material-aware swarms.
pycvc bindings: `nav_material_build`, `nav_witness_gate(_batch)`,
`nav_material_sample`, `nav_bicycle_rollout_material`.

## Using it

### Attach a grid (this is the whole opt-in)

```python
import numpy as np
from grl_snam.material import MaterialGrid, MaterialParams, GateParams
from grl_snam.scenario import FogScenario

risk = np.zeros((96, 96), np.float32)   # smoothed by the grid (sigma=1 cell)
hard = np.zeros((96, 96), bool)         # lethal-but-not-geometry cells
risk[38:58, 34:52] = 0.95               # a mud field
hard[16:58, 60:64] = True               # a water strip

grid = MaterialGrid(risk, hard, bounds, center, scale)
sc = FogScenario(truth, bounds, scale, model, meta,
                 waypoints=[goal], material=grid).start(start)
sc.run()          # planner cost + forces + witness gate, all active
```

`Squad(..., material=grid)` and `Swarm(..., material=grid)` take the same
object (one shared oracle plane for the whole squad/swarm). Not passing
`material` — or passing an all-zero grid — leaves every trajectory exactly
as before.

### Tuning

```python
params = MaterialParams(
    lam_soft=0.5,        # soft risk force (gated); 1.5 launches vehicles here
    lam_hard=1.0,        # hazard barrier (never gated)
    k_sharp=1.25,        # 1/m — keep k_sharp * d_hat_sdf_m ~= 15
    d_hat_sdf_m=12.0,    # barrier reach; must span several grid cells
    risk_weight=10.0,    # A* surcharge per unit risk
    hard_penalty=25.0,   # A* surcharge on hard cells (finite: bias, not forbid)
    gate=GateParams(horizon_m=25.0, material_trigger=0.45, improvement_margin=0.05),
    gate_enabled=True,
)
grid = MaterialGrid(risk, hard, bounds, center, scale, params=params)
```

### Runtime events

`grid.stamp_risk(r0, r1, c0, c1, value)` / `grid.stamp_hard(...)` mutate the
raw rasters, re-derive the planes, and bump `grid.version` — an attached
scenario forces a replan on the next tick (bypassing the route hysteresis),
which is how a mud-onset event lands mid-run.

### Reading it back

`NavMetrics.material_risk` / `.material_gate` carry the per-step risk at the
agent and the gate state (0/False without material); a scenario's
`sc.material.last_gate` is the full `GateResult`; a swarm exposes
`swarm.material_gate` per agent.

### Demo

`grl-snam material-demo` runs both stories (planner-backed FogScenario and
the reactive Swarm) with and without material and writes side-by-side
trajectory renders + stats to `material_demo/`. The pure-C++ twin of the
swarm story is libcvc's `examples/nav_material_demo.cpp`
(`-DCVC_BUILD_NAV_EXAMPLE=ON`).

## Backends and defaults

| tier | env var | default |
|---|---|---|
| material feature (forces/gate/cost) | — (attach a MaterialGrid) | pure Python, ON when attached |
| material kernels in C++ | `GRL_SNAM_MATERIAL_BACKEND` | `python` (opt-in `native`) |
| fused native drive with material | `GRL_SNAM_NAV_DRIVE=native` | requires a pycvc with `nav_drive_step_material`; Swarm raises loudly rather than silently dropping material |

Unlike the geometry kernels (bit-identical, default native when present),
the material native path is opt-in by decision: pure Python is the feature's
default; C++ is the accelerator. The BIT-tier kernels are flip-candidates
after soak.

## Parity contract

| surface | tier | where tested |
|---|---|---|
| gaussian blur / EDT / gradients (`material_build`) | BIT (tobytes) | `test_material_parity.py`, gtest goldens, scipy-equivalence pinned by a corner-impulse discriminator |
| witness gate, serial + batch | BIT (every field; batch == serial bytes; thread-count stable) | `test_material_gate_sim.py`, `test_material_parity.py`, `NavMaterialGate.*` |
| `material_sample` | FLOAT (rtol 1e-5) + non-contractual bit tripwire | `test_material_parity.py` |
| rollout with material vs torch | FLOAT (rtol 1e-4 / atol 1e-5, fixed lambda columns) | `test_material_parity.py` |
| no-material invariance | BIT | null-material delegation byte-identity (gtest), zero-grid value-identity, full suite green |
| behavior | acceptance | mud detour, hazard no-entry, arrival, replan-on-stamp, determinism (Python + gtest sim_world) |

The Layer-B gate in `grl_snam/material.py` is the NORMATIVE bit spec for the
C++ twin: float64 end-to-end, a shared exact 16-direction table (libm
sin/cos and CPython's hypot are excluded from the contract by construction),
sequential accumulation, round-half-even cells, risk recorded before the
feasibility break. The research-core gate keeps the source's float32 math
verbatim; the two are the same algorithm and may disagree only in the last
ulp on adversarial ties.

## Grip — `FrictionField` (separate from risk, and deliberately so)

Risk and grip are **independent** surface properties, so grip is a separate
one-plane sampler (`grl_snam.material.FrictionField`) rather than a seventh
`MaterialField` channel. Ice is innocuous to look at and lethal to drive on;
rubble is the reverse. Folding grip into `risk` would make the one case worth
simulating unrepresentable, and it would cost the bit-identical
`material_sample` twin, every stored bundle, and every golden trace. The
6-plane stack is untouched.

`mu = 1` is the reference dry surface the vehicle constants are already quoted
against, so an absent or uniform-1 field is bit-for-bit the pre-grip rollout.
Rough table: ice 0.15, mud 0.35, grass 0.45, gravel 0.6, wet 0.75, dry 1.0.

```python
from grl_snam.material import FrictionField
nav.friction = FrictionField.uniform(occ.shape, bounds, center, scale, mu=1.0)
nav.friction.stamp(r0, r1, c0, c1, 0.15)   # an ice patch
```

**Both** actuator limits scale with mu, because both are grip-limited:
`a_max` (brake/traction) and `a_lat_max` (cornering). That coupling is the
whole point — wiring only one would leave the vehicle an escape via the other.

Three properties of this model are worth stating plainly, because each is easy
to mistake for a bug:

1. **It understeers; it does not skid.** A kinematic bicycle has no lateral
   velocity state, so it cannot fishtail. On ice `d_cap = atan(mu a_lat L/v^2)`
   collapses and the vehicle simply ploughs on toward the outside of the turn.
2. **Ice therefore makes a free turn FASTER, not slower.** The dry vehicle
   brakes for a corner it is actually taking; the ice vehicle fails to take it
   and carries its speed. "Ice is slower" is true only when the corner is
   forced (a wall, a convoy slot), so do not assert it as an invariant —
   `tests/test_vehicle_refinements.py` pins the understeer instead.
3. **mu is sampled at the CURRENT position, so the vehicle cannot anticipate.**
   Enter ice at speed and the stopping governor's guarantee is already broken —
   it budgeted for grip it no longer has. That is the intended failure mode.

### Anticipation: mu as a coefficient feature

`coef_feats(..., friction=f)` appends the sampled mu as a **sixth** feature, so
the learned coefficients can slow *before* a transition instead of discovering
ice by standing on it. `None` still returns the 5-feature vector bit-for-bit.

The obvious objection to a 5 -> 6 change is that it invalidates every trained
`.cvcnav` weight file and demands a retrain — and training this net from a fresh
init is known to *collapse* reach against the shipped seed. So do not retrain
from scratch:

```python
wide = sdf_nav.widen_coef_mlp(trained_5_feature_net)   # in_dim 5 -> 6
```

The new mu column is **zero**, so the widened net computes exactly the same
function of the original five features — identical outputs, bit-for-bit, for any
mu. It is a strictly-equivalent starting point, so fine-tuning begins inside the
basin that already works and the only thing it can do is discover a use for mu.
`SdfNavigator` reads the width off the model, so attaching a friction field to a
5-input net still feeds it exactly five features.

**Still open:** nobody has actually fine-tuned a widened net. Until that runs,
mu-in-features is plumbing with a proof of equivalence, not a demonstrated
anticipation win.

### C++ / CUDA

Ported. `veh_params` carries `body_offsets`/`body_rr`, `track_width` and a
`friction_field *grip`; each is inert at its default, so `bicycle_rollout`,
`bicycle_rollout_material` and `drive_step` pick them up with no signature
change. Validated against the torch reference by a standalone harness that links
`drive.cpp` directly (the SWIG binding does not expose the new fields yet, so the
usual Python gate cannot reach them):

| case | worst residual |
|---|---|
| legacy (regression gate) | **0.0 — bit-identical** |
| footprint | **0.0 — bit-identical** |
| steering lock | 1.8e-07 |
| grip | 1.5e-08 |
| all three | 1.8e-07 |

against a ~1e-5 float-equivalence contract, and re-checked on a real GPU
(RTX 3050 Ti, sm_86) via `bicycle_rollout_cuda` — the unfused device entry point
added so the vehicle math can be compared without a trained net in the way.
Worst residual there is 1.19e-07.

`veh_params` also carries **`body_gain`**, which scales the summed multi-disc
barrier. Set it to `1/n_body`: the sum is a K-times gain on the learned `alpha`,
and uncorrected it cost the city story its entire reach (45% → 0% at matched
radius) while *improving* standoff and collision rate. Gain-corrected it
recovers to 35% and keeps both safety gains.

Supported everywhere now — `bicycle_rollout`, `bicycle_rollout_material`,
`drive_step`, `bicycle_rollout_cuda`, the device-resident `sim_world_cuda`, and
the SWIG binding. One path still refuses rather than diverging quietly, because
a native path honouring fewer constraints than the torch reference is the "fast
digital twin that moves differently" failure and no parity gate would catch it —
the gates hand both paths the same params:

* `drive_step_cuda` throws on a 6-feature net, because `drive_kernel` assembles
  the 5-feature vector inline in registers. Use the CPU `drive_step` for a
  grip-widened policy.
