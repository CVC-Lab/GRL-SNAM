# cvc::nav material-aware port roadmap — end-to-end torch-free C++/CUDA

Goal: run the **entire** material-aware GRL-SNAM stack — inference *and*
training — with zero libtorch, like the base `cvc::nav` port
([`docs/CVCNAV_CPP_PORT_ROADMAP.md`](CVCNAV_CPP_PORT_ROADMAP.md)), with CUDA
paths where they pay off, so large training campaigns run on the TACC cluster
as fast as possible. This is the organizing document for that campaign; each
phase is a shippable PR pair (libcvc + GRL-SNAM) with a parity/gradcheck gate.

The base material feature is already merged (GRL-SNAM #31, libcvc #230): the
force field, witness gate, and material-coupled drive run torch-free in C++,
consuming **fixed** `lam_soft`/`lam_hard`. What remains is (a) the last native
plumbing, (b) CUDA, (c) the **learned** coefficient model `CoefEnergyNetMaterial`
that *produces* those λ from context, and (d) its **training loop** — the
performance-critical TACC target.

## Status

| Phase | Scope | State |
|---|---|---|
| **P1** | Native fused material inference drive (`nav_drive_step_material`) | **DONE** — libcvc #232, GRL-SNAM #34 |
| **P2** | Deferred native plumbing: sim_world material binding, grouped M-planes, batched Squad drive | planned |
| **P3** | CUDA material inference (`drive.cu` material twin, `sim_world_cuda` material) | planned |
| **P4** | Torch-free `CoefEnergyNetMaterial` forward (transformer + CNN) | **DONE** — libcvc #235, GRL-SNAM #35 |
| **P5** | Torch-free material **training** loop — the TACC target | **DONE (CPU + GPU rollout)** — see below |
| **P6** | Scope completions: route_aware baseline, μ_lat/TTC, Setting-3 belief, v7 decision | planned |

**P5 is complete on CPU and drivable from Python.** The whole `train_material.py`
stage-2 learned-policy training loop runs torch-free in C++, every hand-written
adjoint validated by a finite-difference gradcheck:

| P5 piece | PR |
|---|---|
| Rollout backward (`integrate_surrogate_material_vjp`) | libcvc #240 |
| Model backward (masked MHA / LayerNorm / Conv2d / heads) | libcvc #241 |
| Full loss + full-chain gradcheck (the release gate) | libcvc #243 |
| Adam + global-norm clip + cosine + `.cvcnm` write | libcvc #245 |
| `L_multi` (geometry `integrate_surrogate_v2` + multi-start) | libcvc #246 |
| Python driver (`MaterialTrainer` pycvc binding + wrapper) | libcvc #247, GRL-SNAM #37 |
| CUDA material rollout fwd+VJP (CPU-parity on GTX 1650) | libcvc #248 |

**Deferred (not in the stage-2 learned trainer):** `mu_lat` (in the forward,
unused by the loss), `selectivity_loss` (default `w_selectivity=0`), Stage-1
freeze (warm-start via `.cvcnm` load instead), Stage-3 belief patches
(unimplemented upstream → P6), `route_aware`/`v7` (P6 / dropped). The dataset and
epoch loop stay in Python by design — C++ provides the step. The one CUDA piece
left is **the model (transformer+CNN) forward+backward on device** — the dominant
cost, so it's what a full-GPU trainer needs; the rollout landing first proves the
device path. It's a large batched-GEMM effort held pending a throughput decision.
Deployment note: the pycvc-gl recipe must be rebuilt/bumped so the GRL-SNAM CI
closure picks up the `nav_material_trainer_*` binding (the native test skips until
then, as the material parity tests did after #234).

Fidelity tiers (as the base port): **BIT** = `array_equal`/`tobytes`;
**FLOAT** = float-equivalent (rtol 1e-4 / atol 1e-5), the realistic tier for
anything through a GEMM/attention/bilinear; **GRADCHECK** = finite-difference
gradient check is the ground-truth oracle for training (torch is a secondary
cross-check, never the gate). Every new TU: no `-ffast-math`, explicit
`-ffp-contract=off` on BIT surfaces; new `.cu` gets `-fmad=false --prec-div/sqrt
--ftz=false` per the nav convention.

---

## P1 — Native fused material inference drive ✅

`drive_step_material` (sample → coef_feats → coef_mlp → bicycle_material)
existed in C++ but was unbound. Added the `nav_drive_step_material` pycvc
binding and wired `Swarm._native_drive_step` to use it under
`GRL_SNAM_NAV_DRIVE=native` + material, with the #33 torch fallback for older
pycvc. Parity: null-material == plain drive_step (BIT gtest); drive_step_material
vs torch material drive (FLOAT); native-material Swarm tracks torch. This closes
the "drive material torch-free from Python" hole for the shared-belief swarm.

---

## P2 — Deferred native plumbing

Three items; **grouped M-plane support is the root blocker** for the other two,
so it lands first.

### P2a — Grouped material planes (M>1) — do first
Today the kernels are half M-aware and half shared-only:
- M-aware already: `material_sample` (plane = `map_id[i]` when `M>1`),
  `drive_step_material`/`bicycle_rollout_material` (per-agent plane from the
  geometry `map_id`), and the `nav_bicycle_rollout_material` binding (`(Mm,6,H,W)`).
- Shared-only, must extend: `witness_gate`/`witness_gate_batch` take one raster
  and no `map_id`; `sim_world::set_material` builds exactly one plane and one
  gate surface (`mat_stack_` is `[1,6,H,W]`, `material_view().M = 1`).

Tasks: add optional `map_id` + stacked `[M,H,W]` rasters to
`witness_gate_batch` (nullptr → plane 0, back-compat); a `material_build_batch`
(or an M-loop in `set_material`) producing `[M,6,H,W]`; store sim_world material
as M planes with per-plane gate surfaces. Parity: `bicycle_rollout_material`
M=2 + map_id == two serial single-plane runs (BIT); map_id-aware
`witness_gate_batch` == per-plane serial `witness_gate` (BIT).

### P2b — sim_world material from Python (pure binding)
`sim_world::set_material` is C++-only. Add `nav_sim_world_set_material`
(+ `_clear_material`, `_material_gate_active`) bindings and extend the
`nav_native.NativeSimWorld` wrapper + `sim_world_from_swarm` to attach a
`MaterialRuntime`. Parity: material sim_world vs torch-material Swarm (the
Python mirror of the existing `NavMaterialSimWorld` gtest); thread-count
determinism.

### P2c — Batched Squad material drive
`Squad._can_batch_drive` punts to serial when material is attached, and the
sense-tick `astar_batch` punts when a material cost raster is present. With P2a
done, batch it: squad-level vectorized `witness_gate_batch` → λ columns (the
`Swarm._material_kw` pattern), assemble the shared/`map_id` material stack, swap
`bicycle_rollout` → `bicycle_rollout_material` in `_batched_drive`, and wire the
risk cost into `astar_batch`. Parity: batched-material squad vs serial-material
squad (FLOAT, the batched-drive tier).

---

## P3 — CUDA material inference

Mirror `drive.cu`'s device path for the material drive: a `drive_step_material`
CUDA twin (one thread/agent, reusing the device sampler + adding the 6-channel
material sample + the two force terms + the steer coupling), and material planes
in `sim_world_cuda` (device-resident, static-map deploy path). The material
sample and witness gate can stay CPU where they're not hot; the fused per-agent
drive is the kernel that matters for thousands of agents. Build precise
(`-fmad=false`, no `--use_fast_math`). Parity: CUDA material drive vs CPU
material drive (FLOAT, `rel<5e-3`), auto-skip without a device.

---

## P4 — Torch-free `CoefEnergyNetMaterial` forward (CPU + CUDA)

The learned model that produces `(alpha,beta,gamma,lam_soft,lam_hard)` from
obstacle features + goal + a risk patch. This replaces the fixed-λ option with
learned, context-dependent λ in the torch-free sim — and is the inference half
that P5's training produces. **First conv2d / multi-head-attention / LayerNorm
/ softmax in libcvc.**

Exact forward (source: `material_nav.py` / `train_material.py`): `d_tok=64`,
`nhead=4`, `dim_feedforward=128`, `num_layers=2`, patch 2×32×32.
- `obs_enc` Linear(6→128)→ReLU→Linear(128→64) per obstacle token; `goal_enc`
  Linear(4→64)→ReLU→Linear(64→64) → the goal token.
- `fuser` = 2× `TransformerEncoderLayer(d=64, nhead=4, ff=128)` over `1+N`
  tokens with a key-padding mask. **POST-norm, ReLU FFN, no final encoder
  norm** (torch defaults, not passed — hardcode them):
  `x = norm1(x + MHA(x)); x = norm2(x + linear2(relu(linear1(x))))`.
  MHA: packed `in_proj` → Q,K,V (4×16 heads), `scores = QKᵀ·0.25`, mask padded
  keys to −∞, row-max-stable softmax, `·V`, merge heads, `out_proj`.
- `alpha = softplus(head(z_all[:,1:]))` masked; `beta,gamma = softplus(head(ctx))`.
- `RiskPatchEncoder`: Conv2d(2→16,k3,s1,p1)→ReLU→Conv2d(16→32,k3,s2,p1)→ReLU→
  Conv2d(32→64,k3,s2,p1)→ReLU→AdaptiveAvgPool2d(4)→Flatten→Linear(1024→64)→ReLU.
- `lam_soft = 5·σ(head([risk_ctx, ctx]))`, `lam_hard = 10·σ(...)`; `mu_lat`
  (unused off-highway) may be omitted.

Weight format: a **new** sibling to `.cvcnav` (magic e.g. `CVNM`) — `.cvcnav`
is a Linear chain by construction and can't express the conv/attention/DAG
topology. Reuse every convention (little-endian, f32 row-major, FNV-1a
`arch_hash`, meta tail); a Python exporter mirrors `coef_export.py`. ~207k
params ≈ 0.83 MB.

**#1 parity hazard: torch's fused-attention fast path.** In eval/no-grad,
`TransformerEncoder` may dispatch to BetterTransformer / flash / mem-efficient
SDPA / NestedTensor (which *skips* padded tokens) — all round differently from
naive math attention. Goldens MUST be generated with the math path forced
(`enable_nested_tensor=False`, disable flash/mem-efficient SDPA, or on CPU).
Never validate C++ against fast-path goldens. Tier: FLOAT (rtol 1e-4), not BIT.
CUDA: one block per agent, `(1+N)×64` tokens in shared memory; the CNN is the
heaviest component and should be cached when agents share a risk tile — at
`d=64` everything is launch/memory-bound, so fuse aggressively.

---

## P5 — Torch-free material **training** loop (CPU + CUDA) — the TACC target ✅ (CPU + GPU rollout)

> Implemented — see the P5 PR table under **Status** above. The design notes
> below are the plan that was built; the one remaining item is the model
> (transformer+CNN) forward+backward on device (the rollout landed first).

The performance-critical piece. Differentiate the loss end-to-end without
libtorch, validated by finite-difference gradcheck (the base `coef_train`
discipline: delicate VJPs as `CVC_HD` in a shared `detail/` header so one
gradcheck covers host + device).

Loss assembly (stage 2, `train_material.py:step_batch`):
`L = 0.3·w_traj·L_traj + 0.3·w_vel·L_vel + w_friction·L_fric + w_clear·L_clear
+ w_multi·L_multi + L_nav + w_lreg·L_lreg` where `L_nav = cvar_loss(J,
α=0.95)`, `J = w_goal·‖oT−goal‖² + w_len·arc_len + w_risk·cum_risk +
w_hard·hard_count`. `cvar_loss` uses a **detached** quantile → per-agent tail
mask `1[J_i>η]/((1−α)B)`. Stage 1 drops L_nav/L_lreg and zeros material.
Adam(lr=1e-4) + global-norm clip 5.0 + cosine LR. `mu_lat` unused → omit.

Two new adjoint families beyond base `coef_train`:
1. **Rollout forces** (small): the 6-channel patch bilinear VJP (`d/do`),
   `_sdf_barrier_grad` adjoint (`∂/∂φ = k·σ(1−σ)`), the soft/hard force VJPs
   (inject `d/d(lam_soft,lam_hard)`), the explicit-obstacle IPC sum with the
   `[-200,200]`/`vp` branches, and the semi-implicit step (no vmax clamp, mass).
2. **Backprop through `CoefEnergyNetMaterial`** (the real new surface) — ranked
   by risk (all fail *silently*, so per-op FD gradcheck before the end-to-end
   check): **MHA backward** (softmax-VJP `dS = P⊙(g−(g·P)1)`, four projection
   matmuls, the `1/√d` scale, head reshape, the key-padding mask → exactly-zero
   grad on padded keys, post-norm ordering) > **Conv2d backward** ×3 (stride-2
   weight-grad indexing, AdaptiveAvgPool bins) > **LayerNorm backward** ×4
   (normalized-Jacobian — a dropped mean term biases every gradient silently) >
   Linear/ReLU/softplus/`max·σ`.

Phased with a gradcheck per piece (`dir_rel<2e-2`, `worst_rel<5e-2` on the
smooth surrogate, evaluated at interior points away from the CVaR kink and the
`sdf<1` / `risk∈{0,1}` thresholds): P0 patch sampler + primitives; P1 rollout
fwd+bwd (coefficients fixed); P2 model fwd + per-op adjoints; P3 full loss
assembly end-to-end gradcheck (the release gate); P4 host Adam + `.cvcnm` IO;
P5 CUDA transcription (CUDA-vs-CPU grad parity `rel<5e-3, cos>0.9999`); P6
convergence + deployment-reach test.

**CUDA / TACC.** Device-resident loop (field/patches/params/Adam-moments
resident, in-place device Adam, one-float grad-norm D2H per window, no
`--use_fast_math`). Unlike base `coef_train` (heavy shared field, tiny per-agent
MLP), here the **per-item transformer+CNN dominates** — so **batch the model
across the B scenes** (batched GEMM/conv + batched softmax), don't thread-per-item
tiny GEMMs (latency-bound at `d=64`). Cache the model activations once per item
(reused across the H rollout steps); checkpoint/recompute the rollout samples.
**Multi-GPU is task-parallel, not data-parallel:** the model is ~10²k params, so
one config saturates one GPU — run one full config per GPU/rank (MPS to fill
SMs), many configs concurrently across nodes via the batch scheduler, zero
inter-rank communication. Near-linear campaign scaling.

Stage 3 (belief-patch perception) is documented but **unimplemented** in the
source (CLI restricts `--stage` to {1,2}); it's P6, not P5.

---

## Remaining work & resumption guide (P5 is CPU-complete)

Everything below is what is **left**; the CPU trainer + Python driver + CUDA
rollout are all merged to `master`/`main` (see the P5 PR table under **Status**).
This section is written so the work can resume cleanly on another machine.

### Where it all lives (on libcvc `master`)

- **CPU forward/backward/loss/optimizer:**
  `inc/cvc/nav/{coef_energy_net.h, material.h, material_train.h, geom_rollout.h}`,
  `inc/cvc/nav/detail/{material_rollout.h (CVC_HD primitives + VJPs), nn_ops.h (the
  matched fwd+bwd op library: linear/relu/layernorm/mha/conv2d/adaptive_avg_pool)}`,
  `src/cvc/nav/{coef_energy_net.cpp (forward), coef_energy_net_backward.cpp
  (backward_one), coef_energy_net_io.cpp (.cvcnm serialize/save), material_rollout.cpp
  (rollout fwd+vjp), material_train.cpp (loss + Adam + cosine), geom_rollout.cpp
  (L_multi)}`.
- **CUDA (done):** `src/cvc/nav/material_rollout.cu` (rollout fwd+vjp on device).
- **Gradcheck / parity tests:** `src/cvc/tests/nav_{material_rollout_grad,
  coef_energy_grad, material_train, material_optim, geom_rollout_grad,
  material_rollout_cuda}_test.cpp` + the shared model builder
  `src/cvc/tests/coef_energy_test_model.h`.
- **Python:** `bindings/pycvc/pycvc_nav.i` (the `MaterialTrainer` handle:
  `nav_material_trainer_{create,step,save}`), GRL-SNAM
  `grl_snam/material_train_native.py` + `tests/test_material_train_native.py`.

### P5-P5b — the model (transformer+CNN) on device (the one big item left)

Only the rollout is on GPU. The **transformer+CNN dominates cost**, so a full-GPU
trainer needs `coef_energy_net` forward **and** backward on device. Unlike the
rollout, this is a genuine re-implementation, not a CVC_HD reuse: `detail/nn_ops.h`
is host-only (the ops allocate `std::vector` internally, and `mha_bwd` caches the
softmax matrix), so the kernels must be written fresh. Plan:

1. **Batch across B, not thread-per-agent.** At `d=64` the per-agent GEMMs are
   latency-bound; stack the `B` agents' tokens (`B·T × d`) and use batched GEMM.
   Two routes: **(a) cuBLAS** (`cublasSgemmStridedBatched` for the Linear/attention
   projections; custom kernels for masked softmax, LayerNorm, ReLU, Conv2d,
   AdaptiveAvgPool, and the head activations) — link `CUDA::cublas`; or **(b)
   hand-rolled** shared-memory batched kernels (no cuBLAS dep, more code). (a) is
   recommended for the GEMMs; the conv/pool/softmax/LayerNorm are small custom
   kernels either way.
2. **Mirror the `material_rollout.cu` scaffold** — `CVC_ENABLE_CUDA` gate, a
   `coef_energy_net_cuda_available()`, host wrappers that malloc + H2D the weights
   once and the batch per step, and a **CPU-parity test** `nav_coef_energy_cuda_test`
   (forward float-equivalent; backward `rel < 5e-3 / cos > 0.9999` vs the CPU
   `backward_one`, seeded with the same output grads). Precise-math flags
   (`-fmad=false --prec-div=true --prec-sqrt=true --ftz=false`) as the other nav
   `.cu`. Note fused attention / accumulation order will NOT be bit-exact — the
   parity tier is FLOAT/`rel`, and the CPU FD gradcheck remains the ground truth.
3. **Device-resident training step** (`material_train` twin): keep weights +
   Adam moments device-resident, run model→rollout→loss→backward→Adam on device
   with **one grad-norm D2H per window**; the model forward+backward runs **once
   per training step** (it produces the coefficients; the rollout then runs its H
   steps — the rollout kernel from #248 is already there). Reuse `coef_train.cu`'s
   `grad_sqnorm_kernel` / `adam_kernel` shapes for the device Adam.
4. **Sizing first (recommended):** before investing, benchmark the CPU trainer's
   steps/sec on a realistic (B, model, batch) — the model is ~10²k params and the
   CPU path is OpenMP-parallel across agents, so measure whether GPU is needed for
   the intended TACC campaign scale. Multi-GPU is **task-parallel** (one config per
   GPU/rank, MPS to fill SMs, zero inter-rank comms) — see the CUDA/TACC note above.

### Building & testing on a fresh machine

- **Worktree per session:** `git worktree add -b feat/<x> wt-libcvc-<x> origin/master`
  (never share a checkout; see the `worktree-per-session` convention).
- **Deps:** reuse a sibling deps prefix via `-DCMAKE_PREFIX_PATH=<...>/deps` (the
  libcvc-deps action pins these; a local full build reuses a sibling `deps/`).
- **CPU config (fast iteration):** `cmake -S . -B build -G Ninja
  -DCVC_ENABLE_CUDA=OFF -DCVC_BUILD_PYCVC=OFF -DCVC_BUILD_CLI=OFF
  -DCVC_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=<deps>`; build a single test target;
  run with `LD_LIBRARY_PATH=build/lib:<deps>/lib`.
- **CUDA config:** add `-DCVC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<CC>`
  (`75` on a GTX 1650; set to the box's GPU compute capability, `nvidia-smi`).
  Register a new `.cu` in the `CVC_ENABLE_CUDA` `SOURCE_FILES` list **and** the
  precise-math `set_source_files_properties(...)` line in `src/cvc/CMakeLists.txt`.
- **pycvc build:** `-DCVC_BUILD_PYCVC=ON -DCVC_BUILD_PYCVC_GL=OFF`
  (GL needs the `vtk-python` package) `-DPython3_EXECUTABLE=<py-with-numpy>`;
  target `pycvc`; drive with `PYTHONPATH=build/bindings/pycvc`.
- **Discipline:** the FD gradcheck is the oracle (directional `dir_rel < 2e-2`,
  per-tensor spot `worst < 5e-2` restricted to large-`|g|` params where the central
  FD is well-conditioned; per-test **local** RNG seed; freeze the detached CVaR
  quantile in the FD). `clang-format` (v18) `.cpp`/`.h` — **never** `.i` (it mangles
  the SWIG `%{` directives). Every test exe **must** be in `TEST_TARGETS`
  (`src/cvc/tests/CMakeLists.txt` has a FATAL-error drift check).

### Deployment follow-up

The `pycvc-gl` cvcpkg recipe must be **rebuilt / `cvc_revision`-bumped** against the
libcvc that carries #247, so the GRL-SNAM CI closure ships a pycvc with the
`nav_material_trainer_*` binding. Until then `test_material_train_native` **skips**
(the same pattern the material parity tests followed after #234). A recipe fix
without a revision bump silently no-ops on the published catalog and the dev-server
cache.

### Not part of the stage-2 learned trainer (P6 / dropped)

`mu_lat` (computed in the forward for checkpoint compat, unused by the loss),
`selectivity_loss` (default `w_selectivity = 0`), Stage-1 geometry-freeze mode
(warm-start via a `.cvcnm` load instead), Stage-3 belief patches (unimplemented
upstream), `route_aware_stage2`, and the `v7` controller (failed its own efficacy
gate — recommend not porting). The **dataset (DFC2018) and the epoch/validation
loop stay in Python by design** — C++ exposes the per-step `MaterialTrainer.step`.

---

## P6 — Scope completions

- **route_aware_stage2** grid heuristic (the paper's dyn-table baseline) as a
  C++ baseline for campaign comparison — grid A*/beam, no learned field.
- **Highway μ_lat / TTC channels** — the μ_lat head already exists (kept for
  checkpoint compat); wire the lateral + time-to-collision force channels for
  highway-env-style campaigns if wanted.
- **Setting-3 belief-patch training** — the source's stage-3 (train on inferred
  belief patches + perception loss) is unimplemented upstream; a genuine
  research extension, sequenced after P5.
- **v7 controller** — the source's frozen "v7" repair controller failed its own
  preregistered efficacy gate; **recommend NOT porting** as a deployment target.
  Confirm with the user before spending effort (reproduce-only at most).

---

## Sequencing rationale

P1 (done) unblocks torch-free material inference from Python. P2a (grouped
planes) is the root blocker for P2b/P2c and for per-cluster material campaigns,
so it precedes them. P3 (CUDA inference) and P4 (learned model) are independent
and can run in parallel; P4 is the prerequisite for P5. **P5 is the TACC target**
and the largest, riskiest phase — its correctness rides entirely on the FD
gradcheck harness, so that harness is built first (P5-P0). P6 is opportunistic.
