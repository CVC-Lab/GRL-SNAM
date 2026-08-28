# Material-aware navigation

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
