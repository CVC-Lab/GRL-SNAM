# cvc::nav — CUDA acceleration assessment

_Decision-ready analysis (multi-agent, code-verified). Companion to CVCNAV_CPP_PORT_ROADMAP.md — read that first._

I have verified every load-bearing claim against the actual code. Synthesis follows.

---

# cvc::nav CUDA — decision-ready assessment

**Bottom line:** NOT YET. The one measured wall (single-threaded shared-belief `sense_batch`, 542 ms @ N=1024) is a **CPU decomposition artifact**, fixable bit-identically without a GPU. Do the CPU raycast-parallelization + the torch-free C++ drive first. A CUDA path is a **conditional, device-resident, shared-belief, float-equivalent third twin** that earns its keep only for a *named* deployment above ~10⁴ agents (or ~2–3k on a CPU-weak/GPU-strong box), measured on the deploy GPU — never on the local 4 GB GTX 1650. The four expert analyses converge on this; they differ only in emphasis (A1/A3 bullish on the device-resident build, A4 skeptical), and the honest lead call is A4's gating with A1/A3's design as the blueprint *if triggered*.

## Verified ground truth (I read the code, not just the analyses)

- **CMake CUDA wiring** (`wt-libcvc-nav/CMakeLists.txt`): `CVC_ENABLE_CUDA` default ON, auto-disables without nvcc (L5, L25, L80-81); arch list `50 60 70 75 80` set **before** `enable_language(CUDA)` (L45, L75, L78 — the documented sm_52 fix); `+90` when nvcc≥11.8, `+100 120` when ≥12.8 (L63-67) — **local nvcc 12.0 compiles 50/60/70/75/80/90**; `CUDA_RESOLVE_DEVICE_SYMBOLS ON` so device-link emits SASS and **PTX is stripped — the arch list is the entire compat surface** (L54-57, SetupCUDA.cmake L29). `CVC_USING_CUDA` is a PUBLIC define (`src/cvc/CMakeLists.txt` L815); `voxels_kernels.cu` is appended to `SOURCE_FILES` only under `if(CVC_ENABLE_CUDA)` (L611-612) — the exact slot a nav `.cu` drops into.
- **The `--use_fast_math` bug is REAL** (`CMake/SetupCUDA.cmake` L88-96): it is attached **target-wide** to `cvc` for `$<CONFIG:Release>` × `$<COMPILE_LANGUAGE:CUDA>`, not per-source. Any future nav parity `.cu` compiled into `cvc` in Release inherits `ftz=1`, `prec-div=0`, `prec-sqrt=0`, and fast-intrinsic transcendentals — **silently breaking the float-equivalence contract**. This is a hard prerequisite to fix, confirmed by inspection (A3/A4 flagged it; it checks out).
- **`sense_batch` structure** (`src/cvc/nav/grid_nav.cpp`): `parallel_for(planes.K, …)` (L593) — threads **across planes**, so shared M=1 ⇒ K=1 ⇒ one worker. The per-cell fold is accumulate-then-clamp reading the clamped prior value (L561-569, "Model C: running delta is f32"), i.e. **order-dependent**; the ray-march reads only `truth` (L497-551), i.e. **order-free**. The split the analyses propose is valid against the actual code.
- **PERFORMANCE.md**: batched EDT 0.440 ms/agent vs 9.285 ms single-core = GPU ≈ 21 CPU cores (not a win vs 32); A* GPU 18.4 ms vs CPU 2.84 ms = **6.5× worse**; D2H 0.301 + compute 0.440 = 0.741 ms **loses to 16-core CPU 0.580 ms** ("a GPU EDT with a host consumer never wins at any agent count"); the "~60 agents" crossover is EDT-only/device-resident/private-ish, **not** the shared deployment crossover.
- **Recipe** (`cvcpkg/recipes/libcvc-cuda/recipe.yaml`): `provides:[libcvc]`, `requires_capabilities:[cuda]`, `CVC_ENABLE_CUDA=ON`, linux+windows only (no macOS — no CUDA on Apple), self-hosted CUDA runner, `cvc_revision: 2`. Peers exist: `cvcgl-cuda`, `pycvc-cp311-cuda`, `pycvc-gl-cp311-cuda`. **No new recipe needed.**
- **Roadmap fidelity boundary** (`wt-grl-snam-nav/docs/CVCNAV_CPP_PORT_ROADMAP.md`): drawn **exactly at the bilinear sample** — belief→to_occupancy→EDT→field is **BIT**; sample→MLP→rollout is **FLOAT-EQUIV**. Rule 1: only BIT tiers may be a transparent Python default; torch stays the reference/golden forever. §10.1 ships all three belief modes (shared M=1 / clustered / private M=N); §10.2 deploy `nsub=1`, goldens pinned nsub=1. Flags: `GRL_SNAM_NAV_DRIVE` (drive, default `torch`) separate from `GRL_SNAM_NAV_BACKEND` (bit-kernels, default `native`).

## (1) Per-component GPU-fit table

Speedups are device-resident on sm_80-class; "vs current" = vs today's single-threaded shared-mode CPU, "vs fixed CPU" = vs the agent-parallel CPU sense from step 1.

| Component | GPU verdict | Best fidelity tier on GPU | Expected speedup | Note |
|---|---|---|---|---|
| **drive: sdf_sample** (bilinear + normal renorm) | OK, only inside a fused sim | **Bit-identical** to CPU f32 sampler achievable (pure mul/add + one sqrt, fmad off) | trivial / fill-bound | **HW texture filter DISQUALIFIED** — `cudaFilterModeLinear` uses 9-bit fixed-point weights (~1e-3 err ≈ 10⁴× the ≤1-ULP contract). Manual bilinear only; read field as `float*` via `__ldg`. |
| **drive: CoefMLP** 5→64→64→3 | OK, thread-per-agent, weights in `__constant__` | Float-equiv `rtol=1e-4` (libdevice `expf`≠glibc) | trivial | Third numeric peer; **`-fmad=false`** or MACs fuse and diverge. |
| **drive: bicycle_rollout** | OK | Float-equiv `rtol=1e-5` | trivial | Highest transcendental count but still ~1.6k FLOP/agent-step. |
| **drive: carrot FSM** | OK (predicated selects, shallow divergence) | discrete-exact on identical inputs | trivial | `pos_hist` ring in registers/local. |
| **DRIVE (whole)** | Wins **only** device-resident + CUDA-Graph-replayed, at large N | float-equiv/behavioral third twin | ~44 µs @10⁴ on 1650, ~7 µs on A100; **stops being the frame bottleneck** | Loses below a few-k agents or any per-tick host round-trip (PCIe field re-upload ~147 µs ≫ ~4 µs compute). |
| **sense_batch — raycast** | **STRONG** (N×n_rays≈246k independent DDAs) | counts **bit-exact** via integer atomics; cell-selection trig is float-equiv → **overall behavioral** | ~20–50× vs current; **~par-to-modest vs fixed CPU** | The headline GPU shape — but the CPU fix already captures most of it at N~1k. |
| **sense_batch — fold** | GPU-hostile (non-assoc clamp-after-each) | behavioral (single end-of-tick clamp) or bit (per-cell ordered agent walk, complex) | cheap | Never fold with float atomics (reorders + drops per-agent clamp). |
| **EDT / build_sdf** (Felzenszwalb) | **MARGINAL** — device-resident + batched (private/clustered M>1) only | bit-reproducible *on paper* (f64/f32 correctly-rounded, no transcendentals) | ~whole-CPU (0.44 vs 9.3 ms/agent); **loses on D2H to a host consumer** | Only H/W=384 lines/pass ⇒ under-occupies for a single field. **fp64 tie-break trap:** consumer GPUs run f64 at 1/32–1/64 ⇒ a bit-exact EDT is fast only on A100/H100. Shared M=1 = 1 rebuild/tick, not worth a launch. |
| **inflate** | Trivial stencil, bit-exact — but off the swarm hot path (route only) | bit | trivial | Worth it only inside a device route pipeline; else CPU. |
| **astar** | **POOR — CPU forever** | can't reproduce heapq `(f,g,node,parent)` order | **6.5× worse** on GPU | Threaded `astar_batch` on CPU. |
| **neighbors (CGAL kd-tree)** | **CPU** | n/a | n/a | Not on shared/clustered hot path (swarm passes `peer_boxes=None`). A GPU fixed-radius grid is a separate, later concern. |

## (2) Recommended scope and ORDER

**Step 0 — FIX `SetupCUDA.cmake` (blocking).** Move `--use_fast_math` per-source onto `voxels_kernels.cu` (`set_source_files_properties`); give nav parity `.cu` files `-fmad=false --prec-div=true --prec-sqrt=true --ftz=false` (the CUDA analog of the TU's no-fast-math/no-fp-contract discipline). Verify with a trivial `expf`/`tanh` kernel diffed vs libm **before** any parity kernel. Nothing is trustworthy until this lands.

**Step 1 — CPU raycast-parallelization (DO NOW, no GPU).** Split `sense_batch` into (a) `parallel_for` over **agents** producing per-agent integer free/occ counts (order-free ⇒ bit-exact under any parallelization, even at K=1) and (b) a serial per-plane fold in ascending index (the existing `sense_agent` inner loop, byte-identical). 542 ms → ~40 ms on 32 cores, **bit-for-bit, one reference, zero new fidelity contract.** This dissolves the wall *in the shared mode that is the thousands-of-vehicles target*. Highest value/effort ratio in the whole assessment.

**Step 2 — Land the torch-free C++ drive (roadmap P1–P6).** This is the real deployment payoff: it removes torch from VolRover / a C++ game-engine host entirely and is the correctness reference. The 32-core CPU already meets "thousands @ 60 Hz drive-only."

**Step 3 — CUDA, CONDITIONAL.** Build the device-resident fused CUDA sim **only if** a named deployment's measured N exceeds the CPU ceiling on **its own hardware** (measured on the deploy GPU / prettyhatemachine, never the 1650). Scope: **shared belief, reactive (no per-tick A*), declared float-equiv/behavioral third twin.** Internal order:
1. Device `coef_mlp` (`__constant__`) + fused `drive_step` kernel, validated vs the roadmap golden.
2. CUDA sense raycast (agent×ray, int-atomic counts + single clamp) + `to_occupancy` + per-line EDT.
3. `sim_world.cu`: one stream, **two CUDA Graphs** (drive-only vs sense+drive, selected by tick counter — kills launch overhead that sank the drive on the 1650), COW device field ping-pong (atomic pointer swap, ref-counted buffers), pinned double-buffered pose snapshot.
4. CUDA-GL interop pose VBO in `cvcgl-cuda` / `pycvc-gl-cp311-cuda` for the zero-copy C++/GL renderer.

**Deployment scenarios → verdict:**
- **VolRover on a workstation, or any many-core box at <~few-thousand agents** → CPU only; CUDA never triggers (and the GPU is busy rendering).
- **Headless farm** → CUDA only if the farm has GPUs *and* sim (not rendering) is proven the offline bottleneck — usually it is not.
- **10⁴–10⁵ shared-belief agents on a datacenter/game GPU** → CUDA sim_world wins (~2–4× whole-sim at N~10k on A100-class; drive+sense stop being the bottleneck).
- **Asymmetric CPU-weak/GPU-strong (8-core game/console + big GPU)** → crossover drops to ~1.5–3k; **the one genuinely compelling case.**
- **Private belief (M=N)** → VRAM-bound: ~700–900 agents on 4 GB, ~18–20k on 80 GB. Keep the private/fog-of-war twin on CPU/torch; GPU-private only on the big-VRAM box.

## (3) How it slots into libcvc's CUDA build + recipe + parity harness

- **Build:** new TUs (`nav/drive.cu`, `nav/nav_kernels.cu`, `nav/sim_world.cu`) appended to `SOURCE_FILES` under `if(CVC_ENABLE_CUDA)`, mirroring `voxels_kernels.cu` (L612). Host dispatch CPU-vs-`cuda::` under `#ifdef CVC_USING_CUDA` (already PUBLIC, L815). Each `nav_*_cuda_test` needs **all four/five** CMake entries (`add_executable`, `TEST_TARGETS`, `target_link_libraries`, `cxx_std_17`, `gtest_discover_tests`) gated on `CVC_ENABLE_CUDA` or it is silently omitted.
- **Arch/recipe:** arch list already covers the fleet (sm_75 local, sm_80 deploy; sm_86/89 via Ampere/Ada same-major forward-compat from the sm_80 SASS). Because PTX is stripped, **verify the actual deploy GPU's compute capability is in the compiled set before shipping.** Rides the **existing** `libcvc-cuda` recipe (capability-probe auto-selects it on CUDA hosts, filters to CPU `libcvc` elsewhere); GL interop rides `cvcgl-cuda`/`pycvc-gl-cp311-cuda`. No new recipe. (Compile-time `-cuda` *column* auto-select for downstream consumers is still unimplemented — memory: Phase 10 — so consumers pin columns manually.)
- **Weights/parity sharing:** the **same `.cvcnav` blob** feeds torch, the CPU `coef_mlp`, and a `coef_mlp_cuda` that `cudaMemcpyToSymbol`s the identical folded f32 weights into `__constant__`; `arch_hash` pinning unchanged. Reuse the roadmap's **golden traces** (torch, single-thread, TF32-off, f32 — box-independent). The CUDA path is validated **against the golden at tolerance and against the CPU twin behaviorally**, never bit-vs-CPU past the first FSM flip, and **never becomes the golden generator**. Add a dedicated CUDA CI lane on prettyhatemachine; the CPU twin stays the fleet-wide gate.

## (4) What stays CPU — and the hybrid host/device cost

**Permanently CPU:** `astar` (6.5× worse, can't reproduce heapq tie-break), CGAL kd-tree `neighbors`, `simplify`, `nearest_free`, `line_of_sight`, `inflate` (route-spine path). **The bit-exact fidelity twin (private/fog-of-war) also stays CPU/torch.**

**Hybrid cost implication:** the device-resident CUDA sim is viable **only for pure reactive shared-belief** with the route-spine (A*) orchestrated on a CPU layer *above* `sim_world` — which the roadmap already defers there. If a deployment needs **per-tick A* replanning or neighbor queries**, every such tick round-trips device→host→device, and PERFORMANCE.md shows even a single D2H sinks the device-resident advantage. So per-tick A* = CUDA is off for that deployment. This is the boundary that keeps the GPU case honest: it accelerates the reactive steady state, not the planning layer.

## (5) Top risks

1. **Fidelity of a second/third reference.** A GPU sense fold (atomics + single clamp) is **behavioral-tier, not bit**, and a fp32 GPU field is a third trajectory matching neither torch nor the CPU twin — this is the honest cost of unlocking the sense win. *Mitigation:* declare the whole CUDA sim a float-equiv/behavioral twin, gate behaviorally (reach-set, min-clearance no-regression, <0.5% mode-flip budget, every flip threshold-adjacent) vs the **CPU twin** (not torch directly), keep the CPU bit-kernels canonical.
2. **`--use_fast_math` target-wide (VERIFIED) + nvcc default `fmad=true`** silently break float-equiv. *Mitigation:* Step 0 above; verify before any parity kernel.
3. **fp64 EDT trap.** Bit-identity needs f64 parabola arithmetic; consumer GPUs run f64 at 1/32–1/64 ⇒ a bit-correct CUDA EDT is fast only on A100/H100 — the opposite of the game-GPU target. *Mitigation:* keep EDT on CPU, or accept the fp32 float-equiv field fork; never attempt a bit-exact consumer-GPU EDT.
4. **sm-arch / stripped PTX.** The arch list is the whole compat surface. *Mitigation:* sm_80 covers 86/89; sm_90 needs nvcc≥11.8 (present); Blackwell needs 12.8 (local is 12.0) — verify the deploy GPU's cc is compiled in and add it to `CMAKE_CUDA_ARCHITECTURES` if not.
5. **Transfer.** Per-tick field H2D/D2H sinks it (the EDT-with-host-consumer trap). *Mitigation:* device-resident only; snapshot = pose-only D2H (~3 µs) or CUDA-GL interop; field uploaded only on gated sense ticks.
6. **4 GB-local vs deploy-GPU gap + bus-factor-1 CI.** The GTX 1650 can't bench perf, can't run torch on its GPU, and OOMs private mode at ~700–900 agents; only prettyhatemachine has CUDA in the fleet. *Mitigation:* the 1650 is a **correctness/parity/interop oracle only** — all perf and the go/no-go crossover are measured on prettyhatemachine (sm_80) or the real deploy GPU; price bus-factor-1 CUDA CI into the decision (another reason to build only for a deployment that demonstrably needs it).

**The single decision variable:** after Step 1, does any *named* deployment's real N-at-frame-rate on *its own* hardware exceed the agent-parallel CPU ceiling? The GPU crushes today's *broken* single-threaded sense but only ties-to-modestly-beats a *fixed* CPU sense at N~1k; it pulls decisively ahead only as N outgrows core-count saturation. Measure that number before writing a single `.cu`.

Relevant files: `/home/joe/src/cvc/wt-libcvc-nav/CMake/SetupCUDA.cmake` (L88-96, the fast-math fix), `/home/joe/src/cvc/wt-libcvc-nav/CMakeLists.txt` (L42-78, arch list), `/home/joe/src/cvc/wt-libcvc-nav/src/cvc/CMakeLists.txt` (L611-612, L815, wiring slot), `/home/joe/src/cvc/wt-libcvc-nav/src/cvc/nav/grid_nav.cpp` (L497-601, sense split target), `/home/joe/src/cvc/wt-libcvc-nav/cvcpkg/recipes/libcvc-cuda/recipe.yaml`, `/home/joe/src/cvc/wt-grl-snam-nav/docs/CVCNAV_CPP_PORT_ROADMAP.md` (§1 fidelity boundary, §10 belief modes/nsub), `/home/joe/src/cvc/wt-grl-snam-nav/docs/PERFORMANCE.md` (L59-79, CUDA verdict).