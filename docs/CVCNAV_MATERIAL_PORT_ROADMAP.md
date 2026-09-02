# cvc::nav material-aware port roadmap — end-to-end torch-free C++/CUDA

Goal: run the **entire** material-aware GRL-SNAM stack — inference *and*
training — with zero libtorch, like the base `cvc::nav` port
([`docs/CVCNAV_CPP_PORT_ROADMAP.md`](CVCNAV_CPP_PORT_ROADMAP.md)), with CUDA
paths where they pay off, so large training campaigns run on the TACC cluster
as fast as possible. This is the organizing document for that campaign; each
phase is a shippable PR pair (libcvc + GRL-SNAM) with a parity/gradcheck gate.

The base material feature is merged (GRL-SNAM #31, libcvc #230): the force
field, witness gate, and material-coupled drive run torch-free in C++, consuming
**fixed** `lam_soft`/`lam_hard`. On top of that sit the learned coefficient model
`CoefEnergyNetMaterial` that *produces* those λ from context (P4), and its
**training loop** — the performance-critical TACC target (P5).

**Where the normative source lives.** The doc cites `train_material.py`,
`material_nav.py` and the exp-* trees as the authority for every forward, loss
and hyperparameter. None of them are in this repo. They resolve as
`$GRL_SNAM_MATERIAL_FORK/full_code/train_material.py` — a checkout of
`github.com/SetasAditya/material-aware-grl-snam`, which
`tests/test_material_fork_xcheck.py:22-24` skips against when the env var is
unset. A clone exists at `/home/joe/src/cvc/material-aware-grl-snam`. Set
`GRL_SNAM_MATERIAL_FORK` to it before touching any parity claim; the bit-identity
cross-check is otherwise silently green.

**Companion C++-side doc:** libcvc `docs/NAV_MATERIAL.md` describes the same
subsystem from the library side (API surface, pycvc entry points, deferred
items). It is independently stale — its `## Deferred` section still lists the
`nav_sim_world_set_material` binding and grouped material planes, both of which
shipped in #271 — and is being corrected separately. Cross-check both before
trusting either.

## Status

| Phase | Scope | State |
|---|---|---|
| **P1** | Native fused material inference drive (`nav_drive_step_material`) | **DONE** — libcvc #232, GRL-SNAM #34 |
| **P2** | Deferred native plumbing: grouped M-planes, sim_world material binding, batched Squad drive | **partial** — P2a done (libcvc #271); P2b C++/SWIG done, GRL-SNAM wrapper not written; P2c not started |
| **P3** | CUDA material inference (`drive.cu` material twin, `sim_world_cuda` material) | not started — zero lines |
| **P4** | Torch-free `CoefEnergyNetMaterial` forward (transformer + CNN) | **DONE** — libcvc #238, GRL-SNAM #35 + #36 |
| **P5** | Torch-free material **training** loop — the TACC target | **C++ step complete** (CPU gradcheck-gated; CUDA for model fwd/bwd, rollout fwd/VJP, device Adam), reachable from Python via libcvc #301; **no campaign driver** |
| **P6** | Scope completions: route_aware baseline, highway μ_lat/TTC, Setting-3 belief, v7 decision | deliberately deferred — see the re-scoping below |

Two PR numbers that appear in older revisions of this document, #234 and #235,
match no commit on any ref in either repo. P4 is libcvc **#238** (f6cd932),
which carries both the P4 forward and P4b `integrate_surrogate_material`.

### Blocking everything downstream: nothing material-aware is published

**No material binding has ever reached cvcpkg.org.** The newest libcvc tag is
`v3.2.4` (2026-05-23) and the last `publish-cvcpkg.yml` run was 2026-08-21 at
2d850d1 — which predates #230, #232, #238, #247, #271 and #301. The live catalog
tops out at `pycvc-cp312 3.2.4+cvc.7`. Consequently, in the GRL-SNAM CI closure
**every native material capability flag is False**: `HAS_MATERIAL`
(`grl_snam/nav_native.py:667`), `HAS_MATERIAL_ROLLOUT` (`:672`),
`HAS_MATERIAL_DRIVE` (`:673`), `HAS_MATNET` (`:856`),
`HAS_MATERIAL_ROLLOUT_INTEGRATOR` (`:879`) and `HAS_MATERIAL_TRAINER`
(`grl_snam/material_train_native.py`). So `test_material_parity` (module-level
skip), `test_material_rollout_parity`, `test_matnet_parity`,
`test_material_scenario`'s drive cases and `test_material_train_native` all skip
in every CI leg, and `grl_snam/swarm.py:216` silently falls back off the native
drive. P1/P2/P4/P5 are entirely unexercised downstream — the green CI is green
because it is testing nothing.

A recipe edit does not fix this. The recipes have already been bumped twice
(#271, #272) with no effect, because #272 synced all 15 recipes to
`upstream_version 3.3.0` and **reset every `cvc_revision` to 1** — the usual
"bump the revision" reflex is now the wrong lever. The unblocking action is a
**`v3.3.0` tag push** (which fires `publish-cvcpkg.yml` and `release.yml` with
both CUDA lanes auto-enabled), or a `workflow_dispatch` of `publish-cvcpkg.yml`
on master with `dry_run` explicitly `'false'` (it defaults to `'true'`) and
`publish_cuda` `'true'` (defaults `'false'` on dispatch). Before tagging, decide
what to do about the two lanes that failed in the last run (run 32520794989:
`macos-15-intel` and `ubuntu-22.04-arm`, both marked experimental so the run
still reported success) — a 3.3.0 publish on the same matrix leaves macOS-x86_64
and linux-arm64 without a bundle. Confirm prettyhatemachine is online at publish
time or the CUDA lanes skip with a warning and ship CPU-only.

### P5 pieces, as merged

| P5 piece | PR |
|---|---|
| Rollout backward (`integrate_surrogate_material_vjp`) | libcvc #240 |
| Model backward (masked MHA / LayerNorm / Conv2d / heads) | libcvc #241 |
| Full loss + full-chain gradcheck (the release gate) | libcvc #243 |
| Adam + global-norm clip + cosine + `.cvcnm` write | libcvc #245 |
| `L_multi` (geometry `integrate_surrogate_v2` + multi-start) | libcvc #246 |
| Python driver (`MaterialTrainer` pycvc binding + wrapper) | libcvc #247, GRL-SNAM #37 |
| CUDA material rollout fwd+VJP | libcvc #248 |
| CUDA model fwd+bwd (`coef_energy_net.cu`), device Adam (`material_train.cu`), `use_cuda` routing in `material_loss_and_grad` | libcvc #271 |
| `use_cuda` through the pycvc binding, `nav_material_trainer_loss`, horizon-cap fallback, two binding crash fixes, CPU-only probe stubs | libcvc #301 |

Note #271 (068a220) was **mixed-scope**: besides the material CUDA work it links
the ImageMagick codec delegates PRIVATE in `src/cvc/CMakeLists.txt` (to stop
`PkgConfig::_im_dep_*` leaking into the exported `cvcTargets.cmake`) and adds a
Windows `%pythonbegin add_dll_directory(<prefix>/bin)` to `pycvc.i` and
`pycvc_gl.i`. Anyone bisecting the material work lands on that commit.

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

## P2 — Deferred native plumbing (partial)

Three items. Grouped M-plane support was the root blocker for the other two; it
landed first, in #271, and now has no consumer.

### P2a — Grouped material planes (M>1) ✅

Shipped in libcvc #271. `witness_gate_batch` takes an optional
`const int *map_id = nullptr` alongside `[M,rows,cols]` stacked rasters
(`inc/cvc/nav/material.h:136-146`; the per-agent offset `map_id[i] * rows*cols`
is at `src/cvc/nav/material.cpp:300-320`), `nullptr` selecting plane 0 for
back-compat. `sim_world::set_material(..., int planes = 1)` builds per-plane
surfaces: `mat_stack_` is `[mat_planes_,6,H,W]` with per-plane `mat_risk_`,
`mat_gate_hard_`, `mat_clear_m_` (`inc/cvc/nav/sim_world.h:242-249`;
`src/cvc/nav/sim_world.cpp:530-578`), and `material_view()` reports
`M = mat_planes_` (`sim_world.cpp:514-520`). The already-M-aware pieces —
`material_sample`, `drive_step_material`/`bicycle_rollout_material`, and the
`nav_bicycle_rollout_material` binding's `(Mm,6,H,W)` — are unchanged.

Of the two parity gates this phase specified, one exists and one does not.
`NavMaterialGate.GroupedPlanesBatchMatchesPerPlaneSerial`
(`src/cvc/tests/nav_material_test.cpp:240`) meets "map_id-aware
`witness_gate_batch` == per-plane serial `witness_gate` (BIT)". The other,
"`bicycle_rollout_material` M=2 + map_id == two serial single-plane runs (BIT)",
was never written: every rollout test in that file (lines 358, 417, 428, 462)
passes `map_id = nullptr`. And the one M>1 end-to-end test that does exist,
`NavMaterialSimWorld.GroupedIdenticalPlanesMatchShared` (`:545`), uses two
byte-**identical** planes, so reading plane 0 where plane 1 was intended is
indistinguishable from correct behaviour — a wrong stride or a dropped `map_id`
would still pass. Both holes are worth closing before P2c builds on this.

### P2b — sim_world material from Python (half done)

The C++/SWIG half shipped in #271: `nav_sim_world_set_material`,
`nav_sim_world_clear_material` and `nav_sim_world_material_gate_active`
(`bindings/pycvc/pycvc_nav.i:1042, 1091, 1097`), validated against the
`rows()`/`cols()`/`has_material()`/`material_gate_active()` getters
(`inc/cvc/nav/sim_world.h:134-135, :207, :210`). #301 hardens both — the
gate-active getter now refuses a world with no material attached instead of
reading an empty vector's `data()`, and `set_material`'s `map_id >= planes`
precondition is checked **before** the `Py_BEGIN_ALLOW_THREADS` region rather
than throwing across it.

The Python half does not exist. `grl_snam/nav_native.py` `NativeSimWorld`
(`:446`) and `sim_world_from_swarm` (`:468`) never attach a `MaterialRuntime` —
`git grep set_material -- '*.py'` returns nothing — and the phase's stated
parity gate (material sim_world vs torch-material Swarm from Python, plus
thread-count determinism, the Python mirror of the `NavMaterialSimWorld` gtest)
has not been written. That wrapper plus that test is all that remains of P2b.

The grouped-plane Python surface is also asymmetric: `nav_witness_gate_batch`
(`pycvc_nav.i:1644-1704`) still takes rank-2 `(H,W)` rasters via `take(..., 2)`
and passes no `map_id`; `nav_material_sample` (`:1747`) passes `nullptr`;
`nav_material_build` (`:1548`) is single-plane. Only
`nav_bicycle_rollout_material` (`:1766-1820`) accepts `(M,6,H,W)` + `map_id`.
A Python caller therefore cannot run a grouped gate or grouped sample at all.

### P2c — Batched Squad material drive (not started)

`Squad._can_batch_drive` still punts to serial when material is attached
(`grl_snam/squad.py:334`: `if any(nav.material is not None for nav in navs)`),
and the sense-tick `astar_batch` still punts on a material cost raster
(`:409-412`). With P2a done the kernels are all there; what remains is GRL-SNAM
wiring — squad-level vectorized `witness_gate_batch` → λ columns (the
`Swarm._material_kw` pattern), assemble the shared/`map_id` material stack, swap
`bicycle_rollout` → `bicycle_rollout_material` in `_batched_drive` (`:346+`),
and wire the risk cost into `astar_batch`. `tests/test_material_scenario.py:188`
currently asserts the fallback and changes with it. Per-group (rather than one
shared `MaterialGrid`, `squad.py:150-155`) material also needs the grouped gate
and sample bindings above. Parity: batched-material squad vs serial-material
squad (FLOAT, the batched-drive tier).

---

## P3 — CUDA material inference (not started)

Zero lines. `git grep -i material` over `src/cvc/nav/drive.cu` and
`inc/cvc/nav/sim_world_cuda.h` returns no hits; there is no
`drive_step_material_cuda` or `bicycle_rollout_material_cuda` symbol anywhere.
drive.cu's device path (`dev_field`, `drive_kernel:321`, `bicycle_kernel:448`,
`sim_world_cuda::impl:781`) has no material stack, no 6-channel sample, no
`lam_soft`/`lam_hard`, no `k_sharp`/`d_hat_m` force terms.

**Do not mistake the recent CUDA drive work for progress here.** 6b5933c
(`bicycle_rollout_cuda`), 8575f56 (device-resident `sim_world_cuda`) and f2f7c02
(fused grip) are the geometry/vehicle drive plus the μ friction raster, which
`inc/cvc/nav/drive.h:66-70` explicitly calls "Deliberately separate from the
material planes."

The plan is unchanged: mirror `drive.cu`'s device path for the material drive —
a `drive_step_material` CUDA twin (one thread/agent, reusing the device sampler
+ adding the 6-channel material sample + the two force terms + the steer
coupling), and material planes in `sim_world_cuda` (device-resident, static-map
deploy path). The material sample and witness gate can stay CPU where they're
not hot; the fused per-agent drive is the kernel that matters for thousands of
agents. Build precise (`-fmad=false`, no `--use_fast_math`). Parity: CUDA
material drive vs CPU material drive (FLOAT, `rel<5e-3`), auto-skip without a
device.

---

## P4 — Torch-free `CoefEnergyNetMaterial` forward (CPU + CUDA) ✅

The learned model that produces `(alpha,beta,gamma,lam_soft,lam_hard)` from
obstacle features + goal + a risk patch. This replaces the fixed-λ option with
learned, context-dependent λ in the torch-free sim — and is the inference half
that P5's training produces. **First conv2d / multi-head-attention / LayerNorm
/ softmax in libcvc.** Shipped in libcvc #238 (CPU forward), with a CUDA twin
in #271; FD-gradchecked by `nav_coef_energy_grad_test.cpp`.

Exact forward (source: `material_nav.py` / `train_material.py`): `d_tok=64`,
`nhead=4`, `dim_feedforward=128`, `num_layers=2`, patch 2×32×32.
- `obs_enc` Linear(6→128)→ReLU→Linear(128→64) per obstacle token; `goal_enc`
  Linear(4→64)→ReLU→Linear(64→64) → the goal token.
- `fuser` = 2× `TransformerEncoderLayer(d=64, nhead=4, ff=128)` over `1+N`
  tokens with a key-padding mask. **POST-norm, ReLU FFN, no final encoder
  norm** (torch defaults, not passed — hardcoded):
  `x = norm1(x + MHA(x)); x = norm2(x + linear2(relu(linear1(x))))`.
  MHA: packed `in_proj` → Q,K,V (4×16 heads), `scores = QKᵀ·0.25`, mask padded
  keys to −∞, row-max-stable softmax, `·V`, merge heads, `out_proj`.
- `alpha = softplus(head(z_all[:,1:]))` masked; `beta,gamma = softplus(head(ctx))`.
- `RiskPatchEncoder`: Conv2d(2→16,k3,s1,p1)→ReLU→Conv2d(16→32,k3,s2,p1)→ReLU→
  Conv2d(32→64,k3,s2,p1)→ReLU→AdaptiveAvgPool2d(4)→Flatten→Linear(1024→64)→ReLU.
- `lam_soft = 5·σ(head([risk_ctx, ctx]))`, `lam_hard = 10·σ(...)`; `mu_lat` is
  computed for checkpoint compatibility.

Weight format: `.cvcnm`, a sibling to `.cvcnav` (magic `CVNM`,
`src/cvc/nav/coef_energy_net_io.cpp:13`) — `.cvcnav` is a Linear chain by
construction and cannot express the conv/attention/DAG topology. Same
conventions (little-endian, f32 row-major, FNV-1a `arch_hash`, meta tail); the
exporter is `grl_snam/tools/matnet_export.py:write_matnet`, mirroring
`coef_export.py`. ~207k params ≈ 0.83 MB.

**#1 parity hazard: torch's fused-attention fast path.** In eval/no-grad,
`TransformerEncoder` may dispatch to BetterTransformer / flash / mem-efficient
SDPA / NestedTensor (which *skips* padded tokens) — all round differently from
naive math attention. Goldens MUST be generated with the math path forced
(`enable_nested_tensor=False`, disable flash/mem-efficient SDPA, or on CPU).
Never validate C++ against fast-path goldens. Tier: FLOAT (rtol 1e-4), not BIT.

---

## P5 — Torch-free material training loop (CPU + CUDA) — the TACC target

The C++ **step** is complete and gradcheck-gated. The **campaign** — dataset,
epoch loop, validation, checkpoint selection, resume — does not exist. Both
halves matter to the phase's stated purpose, so P5 is not done.

### What exists

The whole `train_material.py` stage-2 learned-policy loss runs torch-free in
C++, every hand-written adjoint validated by a finite-difference gradcheck.
Loss assembly (stage 2, `train_material.py:step_batch`):
`L = 0.3·w_traj·L_traj + 0.3·w_vel·L_vel + w_friction·L_fric + w_clear·L_clear
+ w_multi·L_multi + L_nav + w_lreg·L_lreg` where `L_nav = cvar_loss(J,
α=0.95)`, `J = w_goal·‖oT−goal‖² + w_len·arc_len + w_risk·cum_risk +
w_hard·hard_count`. `cvar_loss` uses a **detached** quantile → per-agent tail
mask `1[J_i>η]/((1−α)B)`. Stage 1 drops L_nav/L_lreg and zeros material.
Adam(lr=1e-4) + global-norm clip 5.0 + cosine LR.

Two adjoint families beyond base `coef_train`, both landed:
1. **Rollout forces**: the 6-channel patch bilinear VJP (`d/do`),
   `_sdf_barrier_grad` adjoint (`∂/∂φ = k·σ(1−σ)`), the soft/hard force VJPs
   (injecting `d/d(lam_soft,lam_hard)`), the explicit-obstacle IPC sum with the
   `[-200,200]`/`vp` branches, and the semi-implicit step (no vmax clamp, mass).
2. **Backprop through `CoefEnergyNetMaterial`**: MHA backward (softmax-VJP
   `dS = P⊙(g−(g·P)1)`, four projection matmuls, the `1/√d` scale, head
   reshape, key-padding mask → exactly-zero grad on padded keys, post-norm
   ordering); Conv2d backward ×3 (stride-2 weight-grad indexing,
   AdaptiveAvgPool bins); LayerNorm backward ×4 (normalized Jacobian — a
   dropped mean term biases every gradient silently);
   Linear/ReLU/softplus/`max·σ`.

The release gate is `nav_material_train_test.FullChainGradcheck`, backed by
`nav_material_rollout_grad_test`, `nav_coef_energy_grad_test`,
`nav_geom_rollout_grad_test` and `nav_material_optim_test`. Note that despite the
comment at the top of `nav_material_train_test.cpp`, **`L_multi` is inside the
finite-differenced loss**: the test uses a default `material_loss_config`
(`w_multi = 0.5`) with `N = 3`, and `compute_loss` adds L_multi whenever
`cfg.w_multi != 0.0f && N > 0`.

**CUDA coverage** (#271): the model forward and backward
(`src/cvc/nav/coef_energy_net.cu` `forward_batch_cuda` / `backward_batch_cuda`),
the rollout forward and VJP (`src/cvc/nav/material_rollout.cu`), and a
device-resident Adam (`src/cvc/nav/material_train.cu` `material_adam_cuda`,
holding weights and the `(m,u)` moments on device across steps with `sync_to()`
to write back). `material_loss_and_grad(..., bool use_cuda = false)` routes the
four heavy ops. Probes: `coef_energy_cuda_available()`
(`inc/cvc/nav/coef_energy_net.h:176`), `material_rollout_cuda_available()` and
`material_rollout_cuda_max_horizon()` (`inc/cvc/nav/material.h:252, :257`),
`material_train_cuda_available()` (`inc/cvc/nav/material_train.h:169`). #301
adds `#ifndef CVC_USING_CUDA` stubs for all three availability probes so a
CPU-only libcvc links — they are declared unconditionally in the public headers
and were defined only in the `.cu` files, so any caller that followed the
headers' own "guard with `X_cuda_available()`" instruction hit an unresolved
symbol. (Guard on `CVC_USING_CUDA`, the PUBLIC compile define; `CVC_ENABLE_CUDA`
is the CMake variable and guarding on it double-defines the symbol — the same
trap `drive.cpp` documents.)

**Reachable from Python** as of libcvc #301 (merged) + GRL-SNAM #57.
`nav_material_trainer_create`/`_step` take `use_cuda`;
`nav_material_trainer_cuda_active(handle)` reports what the handle is actually
using (the request is not a guarantee); `nav_material_cuda_available()` and
`nav_material_cuda_max_horizon()` are module-level probes. The forward-only
`material_loss` — the validation path, which previously had no binding at all —
is bound as `nav_material_trainer_loss`, and `material_loss` itself gained a
`use_cuda` parameter (it hardcoded `false` before, so even a C++ caller could
not score a held-out batch on device). On the Python side
`grl_snam/material_train_native.py` exposes `use_cuda=`,
`MaterialTrainer.cuda_active`, `MaterialTrainer.loss()` and module-level
`cuda_available()` / `cuda_max_horizon()`.

`material_loss_and_grad(use_cuda=true)` used to throw when `max(H)` exceeded the
device VJP's 64-step cap (`material_rollout.cu:32` `MAX_H_CUDA`, because the
device backward stores each agent's whole trajectory in per-thread local memory
at `:147`), contradicting the header's promise that "it is always safe to pass
true". #301 computes `max_horizon(b)` and falls back to the host adjoint above
the cap; the device *forward* has no such cap, so a long-horizon batch still
gets the GPU forward and only that one op degrades. This mattered because the
existing CUDA tests could never have caught it — `material_batch_fixture.h:97`
sets `H` to 2 or 3, so every device test ran 2–3 steps against a 64-step cap.

**Measured** (GTX 1650, CUDA 12.0, Linux, this session): GPU training driven
from Python is **4.2× faster per step** than CPU (3.49 s → 0.83 s for 30 steps)
and agrees with the CPU path to 3e-6 on the loss after 30 optimizer updates.
The full material suite is green: 31 tests across 7 binaries
(`nav_material_test` 19, `nav_material_train_test` 1, `nav_material_optim_test`
2, `nav_material_train_cuda_test` 4, `nav_material_rollout_cuda_test` 2,
`nav_coef_energy_cuda_test` 2, `nav_material_adam_cuda_test` 1). This
independently reproduces #271's RTX 3050 Ti / Windows validation on a second
platform.

### The shipped CUDA design (not the one this document originally planned)

Recorded here because the code took the route the earlier plan rejected, and the
plan is what a reader would otherwise implement.

`coef_energy_net.cu` is **hand-rolled, one CUDA block per agent**:
`blockDim = DT = d_tok = 64` with every per-dim loop strided by 64
(`ce_fwd_k<<<n, DT, smem>>>` at `:494`, `ce_bwd_k<<<n, DT>>>` at `:1336`), the
tokens and transformer activations living in dynamic shared memory (`:182`), and
the ragged batch handled by an `obs_offsets` array. There is **no cuBLAS
anywhere** — `git grep cublas` over `src/cvc` and `inc/cvc` returns zero hits,
and nothing links `CUDA::cublas`. `material_rollout.cu` is coarser still: one
*thread* per agent (`mat_fwd_k<<<G, T>>>` at `:359`, `T = 128`).

The kernels had to be written fresh rather than reusing `CVC_HD` primitives the
way the rollout did: `inc/cvc/nav/detail/nn_ops.h` is host-only (the ops
allocate `std::vector` internally and `mha_bwd` caches the softmax matrix), and
`coef_energy_net.cu` does not include it. Parity is FLOAT, not BIT — fused
attention and accumulation order differ — with the CPU FD gradcheck remaining
the ground truth; `nav_coef_energy_cuda_test.cpp` gates forward
float-equivalence and backward `rel < 5e-3 / cos > 0.9999` against the CPU
`backward_one`, seeded with the same output grads.

**Possible future optimization, not a plan of record: batch across B.** At
`d = 64` the per-agent GEMMs are latency-bound in principle, so stacking the `B`
agents' tokens (`B·T × d`) and using `cublasSgemmStridedBatched` for the
Linear/attention projections (with custom kernels for masked softmax, LayerNorm,
ReLU, Conv2d, AdaptiveAvgPool and the head activations) is the textbook
alternative. **Benchmark before rewriting anything.** The 4.2× datapoint above
is the only throughput measurement that exists, it was taken on a small GPU, and
nobody has profiled where the time actually goes. A rewrite justified only by
theory is how the shipped design would get replaced by something slower.

### What does not exist

- **No campaign driver.** `grl_snam/material_train_native.py` is a per-step
  handle and nothing more. Grepping for `material_train_native`, `MaterialTrainer`
  or `BATCH_KEYS` returns exactly three files: this roadmap, the module, and its
  own test — there is no production caller. There
  is no dataset adapter (`git grep -i dfc2018` returns nothing), no `collate_fn`
  doing obstacle padding + bool mask, no epoch loop, no running-metric logging,
  no per-epoch/best checkpoint writes, and no `train-material` subcommand in
  `grl_snam/cli.py` (which registers only `material-demo`, `train`, `train-coef`,
  `train-geo`). The only thing in the repo that builds the 17 `BATCH_KEYS`
  arrays is `_synthetic_batch()` in `tests/test_material_train_native.py:105`.
  The dataset and epoch loop staying in Python is the design — but the Python
  has not been written.
- **`L_multi` has no CUDA twin.** There is no `src/cvc/nav/geom_rollout.cu`;
  `inc/cvc/nav/geom_rollout.h` declares no `*_cuda` entry points. Both call
  sites (`src/cvc/nav/material_train.cpp:140` forward-only inside
  `compute_loss`, `:249` again for the grads) sit outside any `#ifdef
  CVC_USING_CUDA`, so with the default `cfg` (`w_multi = 0.5`, `N = 3`) every
  `use_cuda=true` step serializes on a full B-agent geometry rollout on the
  host — **twice**, since the first call's returned loss is discarded. The
  double call is pure waste on the CPU path too.
- **Weights and grads round-trip host↔device every step.** `material_adam_cuda`
  genuinely keeps weights and moments on device, but there is no way to feed
  those device weights into the model kernels: `forward_batch_cuda` and
  `backward_batch_cuda` both `cudaMalloc` + H2D the full weight set on **every**
  call (`coef_energy_net.cu:407-409`, `:1233-1235`) and `cudaFree` at exit, and
  `backward_batch_cuda` D2Hs every grad and accumulates it on the host. One
  composed step therefore costs ~55 weight tensors H2D twice, all grads D2H
  once, all grads H2D once, all weights D2H once. `material_adam_cuda` also has
  zero production callers, and there is no `train_material_cuda` orchestrator
  analogous to `train_coef_mlp_cuda` (`inc/cvc/nav/coef_train.h:198`). The file
  comment in `material_train.cu` claiming a training loop needs no per-step
  weight round-trip is true of the optimizer in isolation and false as composed.
- **No torch-free `.cvcnm` initializer.** `coef_energy_net` has no fresh-init
  constructor — `inc/cvc/nav/coef_energy_net.h:60-62` exposes only
  `load()`/`load_from_memory`, and `MaterialTrainer` is built from a path. The
  only shipped writer, `grl_snam/tools/matnet_export.py:write_matnet`, requires
  a torch module's `state_dict()`. The sole torch-free writer is a test-local
  helper (`tests/test_material_train_native.py:78 _write_random_cvcnm`) whose
  distribution (uniform ±0.15) is not torch's init and which omits the source's
  deliberate `self.mu_lat_head[-1].bias.fill_(-5.0)`
  (`train_material.py:192-193`). So starting a campaign still requires torch —
  which also makes the "Stage-1 freeze: warm-start via a `.cvcnm` load instead"
  mitigation circular. (There is no freeze API either: `material_adam::step`
  unconditionally updates every tensor, there is no `requires_grad` concept, and
  `--train_risk_only` / `load_geometry_weights` have no analogue.)
- **No optimizer-state resume.** `material_adam`'s `(m_, u_, t_)` and
  `material_adam_cuda`'s device `(d_m_, d_u_)` are private with no serializer
  (`inc/cvc/nav/material_train.h:118-124, :151-158`) and no accessor beyond
  `steps()`. The port's only persistence is `coef_energy_net::save()` —
  weights-only. The source checkpoints
  `{'epoch','model_state_dict','opt_state_dict','train_metrics','val_loss','cfg'}`
  every epoch (`train_material.py:1141-1149`). A preempted multi-day TACC job
  restarts with cold Adam moments and a reset bias correction, which is not a
  resume; it is a different training run.
- **No CUDA test runs in CI.** `.github/workflows/ci.yml:162` runs
  `package-linux` on `ubuntu-latest` (no NVIDIA GPU) and merely installs the
  CUDA toolkit before `-DCVC_ENABLE_CUDA=ON` (`:353`); Windows is the same
  shape; macOS builds with `CVC_ENABLE_CUDA=OFF` (`:620`). Every CUDA test body
  opens with `if (!..._cuda_available()) GTEST_SKIP()`. CI proves the `.cu`
  files compile and nothing more.

**Deferred by design (not in the stage-2 learned trainer):** `mu_lat` (computed
in the forward for checkpoint compat, unused by the loss — but note its backward
*is* fully implemented on CPU `coef_energy_net_backward.cpp:191` and CUDA
`coef_energy_net.cu:1040`; only the training seed is zero,
`material_train.cpp:266`, so a port-trained `.cvcnm` ships an untrained
`mu_lat` head that a downstream highway inference path would read as
meaningful); `selectivity_loss` (default `w_selectivity = 0`); Stage-1 freeze;
Stage-3 belief patches (unimplemented upstream — the CLI restricts `--stage` to
`{1,2}` at `train_material.py:1218`) → P6; `route_aware`/`v7` → P6 / dropped.

**CUDA / TACC target shape.** Device-resident loop (field/patches/params/Adam
moments resident, in-place device Adam, one-float grad-norm D2H per window, no
`--use_fast_math`). **Multi-GPU is task-parallel, not data-parallel:** the model
is ~10²k params, so one config saturates one GPU — run one full config per
GPU/rank (MPS to fill SMs), many configs concurrently across nodes via the batch
scheduler, zero inter-rank communication. Near-linear campaign scaling.

---

## Remaining work & resumption guide

Written so the work can resume cleanly on another machine. The ranking below is
by what unblocks what, not by size.

### Where it all lives (on libcvc `master`)

- **CPU forward/backward/loss/optimizer:**
  `inc/cvc/nav/{coef_energy_net.h, material.h, material_train.h, geom_rollout.h}`,
  `inc/cvc/nav/detail/{material_rollout.h (CVC_HD primitives + VJPs), nn_ops.h (the
  matched fwd+bwd op library: linear/relu/layernorm/mha/conv2d/adaptive_avg_pool —
  host-only, no CVC_HD)}`,
  `src/cvc/nav/{coef_energy_net.cpp (forward), coef_energy_net_backward.cpp
  (backward_one), coef_energy_net_io.cpp (.cvcnm serialize/save), material_rollout.cpp
  (rollout fwd+vjp), material_train.cpp (loss + Adam + cosine), geom_rollout.cpp
  (L_multi)}`.
- **CUDA:** `src/cvc/nav/material_rollout.cu` (rollout fwd+VJP, `MAX_H_CUDA = 64`),
  `src/cvc/nav/coef_energy_net.cu` (model fwd+bwd, block-per-agent, `DT = 64`),
  `src/cvc/nav/material_train.cu` (`material_adam_cuda` +
  `material_train_cuda_available`). All three are registered in the
  `CVC_ENABLE_CUDA` source list and given the precise-math flags
  (`-fmad=false --prec-div=true --prec-sqrt=true --ftz=false`). No
  `geom_rollout.cu` exists; no material twin in `drive.cu`.
- **Gradcheck / parity tests:** `src/cvc/tests/nav_{material_rollout_grad,
  coef_energy_grad, material_train, material_optim, geom_rollout_grad,
  material_rollout_cuda, coef_energy_cuda, material_adam_cuda,
  material_train_cuda}_test.cpp` + `src/cvc/tests/nav_material_test.cpp` (gate,
  sample, sim_world) + the shared builders `src/cvc/tests/coef_energy_test_model.h`
  and `src/cvc/tests/material_batch_fixture.h`.
- **Python (libcvc side):** `bindings/pycvc/pycvc_nav.i` — the `MaterialTrainer`
  handle (`nav_material_trainer_{create,step,loss,save,cuda_active}`), the CUDA
  probes (`nav_material_cuda_{available,max_horizon}`), the P2b sim_world trio
  (`nav_sim_world_{set_material,clear_material,material_gate_active}` at
  `:1042, 1091, 1097`), `nav_drive_step_material`, `nav_matnet_forward`,
  `nav_bicycle_rollout_material`, `nav_witness_gate_batch`, `nav_material_sample`,
  `nav_material_build`. `pycvc_nav.i` is `%include`d by **`pycvc.i`** (line 835) —
  these symbols ship in the CORE `pycvc-cp3XX` package, not `pycvc-gl`.
- **Python (GRL-SNAM side):** `grl_snam/material_train_native.py` +
  `tests/test_material_train_native.py`; the capability flags in
  `grl_snam/nav_native.py:667-879`; the material drive wiring in
  `grl_snam/swarm.py:216`; the squad fallback at `grl_snam/squad.py:334`.

### Ranked remaining work

**1. Publish the 3.3.0 family (blocker, small, libcvc).** See "Blocking
everything downstream" above. Until this lands, nothing below can be validated
downstream and every material test in GRL-SNAM CI is a no-op.

**2. ~~Land libcvc #301 and its GRL-SNAM companion~~ — done.** libcvc #301 and
#302 are merged; GRL-SNAM #57 carries the Python half (`use_cuda`,
`MaterialTrainer.cuda_active`, `MaterialTrainer.loss()`, the module-level
probes). This was #271's headline deliverable being unreachable from the only
driver that exists; it now switches on.

**3. Write the P2b Python half (important, medium, GRL-SNAM).** Extend
`NativeSimWorld` (`grl_snam/nav_native.py:446`) and `sim_world_from_swarm`
(`:468`) to call the sim_world material trio, and add the parity test this phase
specified: material sim_world vs torch-material Swarm from Python, plus
thread-count determinism. #271 landed the C++ and SWIG halves and touched zero
GRL-SNAM files, so a material-aware pure-C++ sim_world is unreachable from
Python and the phase's gate does not exist.

**4. Build the actual training driver (important, medium, GRL-SNAM).** A
`DFC2018ShortRollouts`-equivalent dataset (`train_material.py:561-677` —
including `waypoint_mode='geom'` dijkstra-derived local goals with the
`geom_waypoint_cache_v1_*.pt` cache, the random `(H, dt_mult, t0)` sampling at
`:607-621`, and the 6-channel `rollout_patch` fallback synthesis at `:660-677`);
a `collate_fn` doing obstacle padding + bool mask (`:713-748`); an epoch loop
with running metrics (`:1116-1138`); a validation pass (needs
`nav_material_trainer_loss` from item 2); per-epoch `epoch_%03d.pt` / `best.pt`
writes (`:1141-1153`); and a `train-material` subcommand in `grl_snam/cli.py`.
Model it on the geometry trainer's complete chain: `nav_native.train_coef_mlp` →
`tools/coef_train.py train_native` → argparse `--backend native --cuda`. This is
what makes P5's stated purpose runnable.

**5. Benchmark CPU vs GPU steps/sec at campaign scale (important, small,
libcvc).** The 4.2× GTX 1650 datapoint is a smoke test, not a sizing study. Add
a CPU-vs-GPU benchmark for `material_loss_and_grad` at a realistic
`(B, N, H, patch_size)` and profile where the time goes. This is what decides
whether items 6 and 7 are worth doing at all, and whether the block-per-agent
design should ever be revisited.

**6. Give `L_multi` a CUDA twin, and stop evaluating `multi_start_penalty`
twice (important, medium, libcvc).** Port `integrate_surrogate_v2` + its VJP +
`multi_start_penalty` to a `geom_rollout.cu` twin (the
`detail/material_rollout.h` primitives are already `CVC_HD`, so the same reuse
trick applies) and add `nav_geom_rollout_cuda_test`. The double evaluation
(`material_train.cpp:140` and `:249`) is worth fixing independently of CUDA.
Sequence after item 5 — the benchmark should say whether this term is on the
critical path.

**7. Compose a device-resident training loop (important, large, libcvc).** Give
`coef_energy_net` a device-weight handle (or let `material_adam_cuda` expose its
`d_p_` and have `forward_batch_cuda`/`backward_batch_cuda` accept a device
weight pointer), add persistent device buffers for the batch and scratch instead
of `cudaMalloc`/`cudaFree` per call, and add a `train_material_cuda(...)`
orchestrator analogous to `train_coef_mlp_cuda`. Do item 5 first.

**8. Get the CUDA nav tests onto a GPU runner (important, medium, libcvc).**
Route `nav_coef_energy_cuda_test`, `nav_material_rollout_cuda_test`,
`nav_material_adam_cuda_test` and `nav_material_train_cuda_test` to a
self-hosted GPU runner (prettyhatemachine is the fleet's only CUDA box), or at
minimum a nightly GPU job. Every parity number quoted so far comes from a
developer box and nothing re-verifies it on change.

**9. Strengthen the grouped-plane (M>1) tests (important, small, libcvc).** Add
a `NavMaterialSimWorld` case with two **different** planes where each agent's
trace matches its own single-plane run, and write the missing
`bicycle_rollout_material` M=2 + `map_id` BIT gate. P2a is the foundation P2c
builds on and plane indexing fails silently.

**10. P2c: batched Squad material drive (important, large, GRL-SNAM).** See the
P2c section. Now unblocked; P2a currently has no consumer.

**11. Add a torch-free `.cvcnm` initializer (important, small, GRL-SNAM).**
Promote a `write_random_matnet(path, *, patch_size, seed)` into
`grl_snam/tools/matnet_export.py` with correct per-layer fan-in
(Kaiming-uniform) init and the source's `mu_lat_head[-1].bias.fill_(-5.0)`.
Until this exists the "zero libtorch" claim breaks at step zero.

**12. Add optimizer-state checkpoint and resume (important, medium, libcvc).**
Serialize `material_adam`'s moments, step count and LR-schedule position, and
extend `nav_material_trainer_save` to write it alongside the `.cvcnm` weights.

**13. Broaden CUDA test coverage beyond the fixture's 2–3 step horizons
(nice-to-have, small, libcvc).** `material_batch_fixture.h:97` sets
`b.H[i] = (i % 2 == 0) ? 2 : 3`, so no device test approaches the 64-step cap —
which is why the `max(H) > 64` throw went unnoticed. Add cases at and above the
cap, with `w_multi = 0` (isolating the material path), with `N = 0`, and with
ragged obstacle counts at the training-step level; add a thread-determinism case
for `material_loss_and_grad` across `cfg.num_threads`
(`nav_material_rollout_grad_test` has one, the full loss does not). Sequence
after item 8 so these actually run somewhere.

**14. Close the asymmetric grouped-plane Python surface (nice-to-have, small,
libcvc).** Extend `nav_witness_gate_batch`, `nav_material_sample` and
`nav_material_build` to the grouped `(M,...)` + `map_id` form. This bites item 10
the moment a squad wants per-group material.

**15. P3: CUDA material inference drive (nice-to-have, large, libcvc).** See the
P3 section. Ranked low because nothing currently blocks on it.

**16. Note the untrained `mu_lat` head in the `.cvcnm` metadata (nice-to-have,
small, libcvc).** Cheap to record now, expensive to discover later — see the
deferred-by-design note under P5.

**17. Stop `material_train_native.py` charging ~25 uncovered statements against
the 80% coverage gate (nice-to-have, small, GRL-SNAM).** `pyproject.toml`
`[tool.coverage.run]` omits demos, tools and `cli.py` but not this module, whose
function bodies are all dead while `HAS_MATERIAL_TRAINER` is False. Either omit
it until the closure republishes, or add a non-skipped unit test for
`cosine_lr` and the `BATCH_KEYS` dtype coercion (both pure Python). Resolves
itself when item 1 lands.

**Documentation debt** (distinct from the engineering gaps above): libcvc
`docs/NAV_MATERIAL.md` needs the same correction this document just received —
its `## Deferred` section still lists the `nav_sim_world_set_material` binding
and grouped material planes (only the `drive.cu` material twin is genuinely
deferred), its pycvc surface list (`:155-158`) omits `nav_drive_step_material`,
the sim_world trio, `nav_matnet_forward` and `nav_material_trainer_*`, and its
`set_material` example (`:98-101`) never mentions the `planes` argument or the
`[planes,H,W]` stacked-raster convention. Two source comments also mislead:
`nav_material_train_test.cpp`'s header claims L_multi is out of scope (it is
inside the finite-differenced loss), and `material_train.cu`'s header claims a
training loop needs no per-step weight round-trip (true of the optimizer alone,
false as composed).

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
  FD is well-conditioned, evaluated at interior points away from the CVaR kink and
  the `sdf<1` / `risk∈{0,1}` thresholds; per-test **local** RNG seed; freeze the
  detached CVaR quantile in the FD). `clang-format` (v18) `.cpp`/`.h` — **never**
  `.i` (it mangles the SWIG `%{` directives). Every test exe **must** be in
  `TEST_TARGETS` (`src/cvc/tests/CMakeLists.txt` has a FATAL-error drift check).
- **Cross-check against the source:** export `GRL_SNAM_MATERIAL_FORK` before
  running the GRL-SNAM suite, or `tests/test_material_fork_xcheck.py` skips and
  the bit-identity claims go unverified.

---

## P6 — Scope completions

All four remain correctly deferred. They are deferred for four different
reasons and are four different sizes; the earlier one-bullet-each framing
concealed that.

- **`route_aware_stage2`** — an **evaluation baseline**, not a training feature.
  It lives at `full_code/exp-rellis/eval_rellis_dyn.py:594` (`_route_aware_step`)
  and is the paper's dynamic-table comparator. Porting it as a C++ baseline
  (grid A*/beam over `cvc::nav`'s existing A* + `clearance_cost`, no learned
  field) is a contained job, and libcvc `docs/MATERIAL_NAV.md` already disclaims
  reproducing the paper's dynamic-table numbers, so nothing downstream depends
  on it.
- **Highway μ_lat / TTC** — **not a one-line wiring job, and not in
  `train_material.py` at all.** It is a separate experiment tree,
  `full_code/exp-highway-env/`: the lateral probe `n_lat_world`
  (`surrogate_integrator.py:189-213`), `_ttc_longitudinal_force` (`:220-314`),
  plus `bicycle_surrogate.py`, `episode_costs.py`, `onpolicy_buffer.py`,
  `collect_onpolicy.py`, a 72 KB `train_stage2.py`, and a vendored HighwayEnv
  fork — roughly 100 KB of trainer code. Wiring the existing `mu_lat` head is a
  day (its forward *and* backward are already implemented on CPU and CUDA;
  only the training seed is zeroed). Porting the campaign is its own phase, and
  should be commissioned as one if a highway campaign is ever wanted.
- **Setting-3 belief-patch training** — a **research task, not a port.** The
  source's stage 3 (train on inferred belief patches + a perception loss) has no
  upstream implementation: `train_material.py:1218` restricts `--stage` to
  `choices=[1,2]`. There is nothing to port; someone has to design it. Needs an
  owner and a scope before it is a phase.
- **v7 controller** — **settled: do not port.** The source's frozen "v7" repair
  controller failed its own preregistered efficacy gate; `FROZEN_V7_STATE.md`
  records that the 54-item matched subset "did not pass the efficacy stage gate"
  and that "no held-out or acceptance claim is authorized." Reproduce-only at
  most, and confirm with the user before spending any effort.

---

## Sequencing rationale

P1 (done) unblocks torch-free material inference from Python. P2a (grouped
planes) was the root blocker for P2b/P2c and for per-cluster material campaigns;
it landed in #271 and now waits on its Python consumers. P3 (CUDA inference) and
P4 (learned model) are independent; P4 was the prerequisite for P5 and is done.
**P5 is the TACC target** and the largest, riskiest phase — its correctness
rides entirely on the FD gradcheck harness, which is why that harness was built
first and remains the release gate.

What changed about the sequencing is where the risk now sits. The C++ is ahead
of everything around it: the kernels, adjoints and device paths are written and
gradcheck-gated, while the *distribution* (nothing published) and the *driver*
(no dataset, no epoch loop, no validation, no resume) are not. So the next three
moves are publish, land #301, and write the Python — not more C++. P6 is
opportunistic and, per the re-scoping above, is really four unrelated items
rather than one phase.
