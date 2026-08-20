# The Native (libcvc `cvc::nav`) Acceleration Layer

GRL-SNAM's numerics are **canonical in Python (numpy + PyTorch)**. Separately,
the navigation hot path — and now the policy *trainer* — has a torch-free C++
implementation in **libcvc's `cvc::nav`** module, reached from Python through
**`grl_snam/nav_native.py`** (a thin adapter over the `pycvc` bindings). Every C++
path is **opt-in behind a feature flag and falls back to the Python/torch
implementation** when the binding is absent, so importing `grl_snam` is always safe
and the reference behavior never changes.

This document is the GRL-SNAM-side reference for that layer: what is accelerated,
the flags that switch it on, the fidelity contract, and how to get a `pycvc` that
carries it. The C++ *design* (the phased port plan) lives in
[`docs/CVCNAV_CPP_PORT_ROADMAP.md`](CVCNAV_CPP_PORT_ROADMAP.md); the libcvc-side
API reference for the trainer lives in libcvc `docs/NAV_TRAINING.md`.

## TL;DR — what runs where

| Layer | Python (canonical) | Native C++ (`cvc::nav`, opt-in) | Feature flag | Fidelity |
|---|---|---|---|---|
| **Kernels** — EDT, `build_sdf`, inflate, line-of-sight, `nearest_free`, A*, string-pull, `sense_batch`, neighbours | `grl_snam/planner.py`, `sdf_nav.py`, `belief.py` | `nav_native.*` → `pycvc.nav_*` | `GRL_SNAM_NAV_BACKEND` (default `native`; `python` forces off) | **bit-identical** |
| **Drive** — sample → CoefMLP → bicycle rollout, carrot FSM, `sim_world`, `sim_thread` | `sdf_nav.py`, `swarm.py` | `nav_native.drive_step` / `sim_world_*` | `GRL_SNAM_NAV_DRIVE=native` (Swarm; default `torch`) | float-equivalent (~1e-4) |
| **Trainer** — self-supervised CoefMLP training | `grl_snam/tools/coef_train.py` (torch) | `nav_native.train_coef_mlp` → `cvc::nav::coef_train` | `GRL_SNAM_TRAIN_BACKEND=native` (or `--backend native`) | torch-independent (gradcheck) |

**Nothing above is on by default that changes results.** The kernels are a
bit-identical drop-in (so they *can* be the default when present); the drive and
trainer are float-equivalent / torch-independent and stay explicit opt-ins — torch
remains the golden reference and generator.

## The adapter: `grl_snam/nav_native.py`

`nav_native` imports `pycvc` and probes for each capability with a `HAS_*` flag, so
partial builds degrade gracefully:

```python
from grl_snam import nav_native

nav_native.AVAILABLE        # pycvc importable AND carries nav_astar
nav_native.enabled()        # AVAILABLE and GRL_SNAM_NAV_BACKEND != "python"

nav_native.HAS_SENSE_BATCH  # nav_sense_batch
nav_native.HAS_SDF_SAMPLE   # nav_sdf_sample
nav_native.HAS_COEF_MLP     # nav_coef_mlp_forward
nav_native.HAS_DRIVE        # nav_bicycle_rollout / nav_drive_step
nav_native.HAS_SIM_WORLD    # nav_sim_world_*
nav_native.HAS_SIM_THREAD   # nav_sim_thread_*
nav_native.HAS_CUDA_DRIVE   # nav_drive_step_cuda (pycvc built with CUDA)
nav_native.HAS_TRAIN        # nav_train_coef_mlp  (the trainer)
```

Every adapter returns the **same Python type** as the function it replaces, so
callers dispatch with a one-line early return and nothing downstream can tell which
path ran. The libcvc parity tests (`tests/test_nav_cpp_parity.py`,
`tests/test_*_parity.py`) assert this byte-for-byte / float-equivalently on every
release.

## Kernels — `GRL_SNAM_NAV_BACKEND`

`planner.py` / `sdf_nav.py` / `belief.py` call `nav_native.enabled()` and dispatch
to the C++ kernel when present; `GRL_SNAM_NAV_BACKEND=python` forces the pure-Python
reference (the parity tests use this to obtain the golden). These are a
**bit-identical** port (float64 EDT, exact heap pop-order A*, `std::rint` half-even
log-odds), threaded across cores for the batch variants (`astar_batch`,
`build_sdf_batch`, `inflate_batch`, `sense_batch`) with the GIL released.

## Drive — `GRL_SNAM_NAV_DRIVE`

The vectorized `Swarm` (`swarm.py`) can drive one tick through the torch-free C++
path instead of the torch coef-net + bicycle rollout:

```bash
GRL_SNAM_NAV_DRIVE=native   # Swarm.step drives via nav_native.drive_step
```

Off by default (torch is the reference twin). It engages only when the flag is set
**and** `nav_native.HAS_DRIVE`; otherwise the Swarm silently stays on torch. The C++
drive is float-equivalent to torch (~1.5e-5 over 120 ticks, identical reach-set).
The whole shared-belief swarm can also run entirely in C++ via
`nav_native.sim_world_from_swarm(...)` (CPU `sim_world`) or the device-resident
`sim_world_cuda` — the pure-C++ deployment path a renderer / game engine embeds.

## Trainer — `GRL_SNAM_TRAIN_BACKEND`

**Canonical:** `grl_snam.tools.coef_train.train()` (PyTorch). **Opt-in native:** the
torch-free `cvc::nav::coef_train`, which trains the CoefMLP by self-supervised
differentiable rollout (no dataset, no labels) and writes the same versioned
`.cvcnav`.

```bash
# CLI: torch stays the default; --backend native uses the libcvc trainer
python -m grl_snam.tools.coef_train --backend native --out coef_mlp.cvcnav
python -m grl_snam.tools.coef_train --backend native --rollout bicycle --cuda
GRL_SNAM_TRAIN_BACKEND=native python -m grl_snam.tools.coef_train   # env form
```

```python
# Programmatic, fully torch-free (needs only pycvc + numpy):
from grl_snam import nav_native
nav_native.train_coef_mlp(
    occ, "coef_mlp.cvcnav",              # (H,W) uint8 free(0)/obstacle grid
    bounds=(-100, -100, 100, 100), scale=0.05,
    rr=0.15, d_hat=0.35, dt=0.06, vmax=0.9,
    steps=400, rollout="surrogate",      # or "bicycle"
    use_cuda=False,                       # True -> device-resident GPU trainer
)
```

Note `coef_train.py` itself imports torch (via `sdf_nav`), so its `--backend native`
CLI still needs torch installed; the *genuinely* torch-free entry is
`nav_native.train_coef_mlp` (used above) or the libcvc `nav_train_demo` C++ CLI.

### Two rollout integrators

`rollout="surrogate"` (default) differentiates the smooth point-mass `sdf_rollout` —
the same proxy the torch trainer uses; it refines the hand-tuned `(1,3,4)` basin and
**improves** reach (~62% → ~65% on the city scene). `rollout="bicycle"`
differentiates the *full deployment* kinematic-bicycle integrator (no
surrogate→deployment dynamics gap), but its governor branches make the loss
landscape far more sensitive — it needs a **much** lower learning rate (`~1e-5` vs
the surrogate's `2e-4`; `train_coef_mlp` auto-picks this) — and because the carrot
FSM that feeds sub-goals at deployment is non-differentiable, the bicycle rollout
chases the goal directly, so on the city scene it *holds* the basin rather than
beating it. **Prefer the surrogate** unless you specifically want the deployment
integrator in the loop.

### CUDA

`use_cuda=True` runs the **fully device-resident** GPU trainer (field, params, Adam
moments and scratch stay on the GPU for the whole run; in-place device Adam; only
the final weights come back), when the `pycvc` was built with CUDA and a device is
present. It is a device transcription of the same hand-written adjoints and
reproduces the CPU loss+gradient to ~1e-7.

## Fidelity contract

- **Kernels: bit-identical.** A C++ kernel result equals the numpy reference
  byte-for-byte; that's why they may be the transparent default.
- **Drive: float-equivalent (~1 ULP / ~1e-4).** Never wired transparently into the
  torch path — opt-in forever; torch stays the golden generator.
- **Trainer: torch-independent.** Correctness is a **finite-difference gradcheck**
  (analytic gradient == numeric), not a torch-parity claim — so the native trainer
  needs no reference to torch's autograd. It is *not* expected to reproduce the
  torch trainer's exact trajectory (different init, different float order); it is a
  gradient-correct refinement of the same basin.

## Getting a `pycvc` that carries `cvc::nav`

The published `pycvc` may lag; build one from libcvc `feat/cvc-nav-kernels` (or
newer, ≥ 3.3.0 for the trainer) with `CVC_BUILD_PYCVC=ON` and point `PYTHONPATH` at
`build*/bindings/pycvc`. Two gotchas:

- The stale `libcvc.so` in a sibling `deps/` prefix has **no nav symbols** — put the
  fresh `build*/lib` **first** on `LD_LIBRARY_PATH` or `import pycvc` dies with an
  undefined-symbol error.
- Adding a binding regenerates the SWIG wrapper — if a `HAS_*` probe is `False` after
  a build, force a `touch` on `pycvc_nav.i` and rebuild; verify with
  `hasattr(pycvc, "nav_train_coef_mlp")`.

Everything degrades cleanly without it: `HAS_* == False`, `enabled() == False`, and
the Python/torch path runs.

## See also

- [`docs/CVCNAV_CPP_PORT_ROADMAP.md`](CVCNAV_CPP_PORT_ROADMAP.md) — the C++ port
  design (phases P0–P8, fidelity boundary, `sim_world`/`sim_thread`, the trainer).
- [`docs/CVCNAV_CUDA_ASSESSMENT.md`](CVCNAV_CUDA_ASSESSMENT.md) — when the GPU wins.
- [`docs/PERFORMANCE.md`](PERFORMANCE.md) — kernel speedups and the scaling wall.
- libcvc `docs/NAV_TRAINING.md` — the trainer's libcvc-side API reference.
