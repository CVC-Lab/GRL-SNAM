# cvc::nav — torch-free C++ port roadmap

> **STATUS: COMPLETE (2026-08-19).** The port shipped end-to-end — all phases P0–P8,
> the CUDA twins (drive, device-resident `sim_world_cuda`, the trainer), and the
> torch-free self-supervised **trainer** (which was *beyond* the original P0–P8
> plan). Every item once listed under "Deferred" is done. This document is kept as
> the **design record** — the fidelity contract (§1), the resolved decisions (§0,
> §10), and the `.cvcnav` format (§4) still define the shipped behavior — not as an
> active plan. The one honest gap is that the *installed default* weights are the
> untrained biased seed (~57% reach), not yet a trained checkpoint. For USING the
> C++ layer from GRL-SNAM see `docs/NATIVE_CVC_NAV.md`; for the trainer API see
> libcvc `docs/NAV_TRAINING.md`.

_Design synthesis (multi-agent). Plan for bringing the grl-snam nav simulator into cvc::nav for pure-C++ hosts while the Python API stays the reference twin and CI stays green._

I have read all the load-bearing files (both worktrees, the FSM, the sampler, the binding conventions, the CMake gtest wiring, the parity-test patterns, and the `parallel_for`). Here is the unified roadmap.

---

# ROADMAP — Porting all of grl-snam nav into torch-free `cvc::nav`

## 0. The load-bearing decisions (contradictions resolved)

The five designs agree on the shape and disagree on six points. My rulings, each with the reason:

| # | Contradiction | Ruling |
|---|---|---|
| A | **TU layout** — D4 folds the drive into `grid_nav.cpp`; D3 splits into 7 TUs | Keep `grid_nav.h/.cpp` **untouched** (its header *documents* the bit discipline; the drive is a different fidelity contract and must not share the TU). Extract the existing `parallel_for` into a shared `detail/parallel.h`. Add **five** new TUs (below). D4's "put it in grid_nav" is rejected; D3's 7-TU split is trimmed. |
| B | **`sdf_sample` fidelity** — D5 wants "BIT if the fuzz proves it, then transparent"; D1/D3/D4 say float-equiv | **Contract = FLOAT-EQUIVALENT (≤1 ULP target).** Run D5's exact-equality fuzz as a *diagnostic* to know the residual, but **never wire the sampler transparently into the Python torch path even if bit-exact** — it only ever feeds the float-equiv drive, so transparency buys nothing and risks silently redefining the reference. |
| C | **Does native drive become the Python default?** — D4 plans a "P5 flip"; D3/D5 say never | **Never flip the Python default.** The drive is opt-in in Python *forever*; torch stays the reference/training twin. The C++ drive earns its keep in a C++ renderer/game-engine host (no Python at all) and in *new* parity tests. D4's default-flip is rejected — it reroutes existing 5e-3 tolerance tests through an independent float32 implementation for zero deployment benefit. |
| D | **Drive backend flag** | A **new, independent** env flag `GRL_SNAM_NAV_DRIVE` (default `torch`), **separate** from `GRL_SNAM_NAV_BACKEND` (the bit-kernel flag, default `native`). D4 is right that one flag cannot govern both tiers. |
| E | **Weight file format** — 4 competing layouts | Adopt **D2's generic dense-stack `.cvcnav`** (arch_hash answers "hidden-size change bumps the version" automatically) + D2/D5's optional length-prefixed **provenance trailer**. Store **raw** bias (fold `log(expm1)` once at load). |
| F | **sim_thread publish** | Ship **both** (D3): `triple_buffer` (atomic-shared_ptr, exposes field/occ by ref — 1:1 with Python `Snapshot`) *and* `pose_seqlock<NMAX>` (genuinely lock-free, pose-only). triple_buffer first. |

Two hard rules fall out of B/C and govern the whole plan:

> **Rule 1 (transparency).** Only **BIT** tiers may be a transparent Python default (like today's kernels). **FLOAT-equivalent** components (sampler, MLP, rollout, whole drive) are **explicit opt-in**, torch is the Python default and the golden generator.
>
> **Rule 2 (Python-green by construction).** No phase edits `SdfNavigator`, `Squad`, or the default `Swarm`/torch path. New C++ is additive; every grl-snam test that touches it is new and `pytest.importorskip("pycvc")`-guarded, so an old pycvc build skips → green.

---

## 1. Fidelity contract per component

Boundary is drawn **exactly at the bilinear sample**. Everything upstream stays byte-identical to numpy (or the occupancy raster silently diverges and the whole field with it); everything from the sample onward is float-equivalent.

| Component | Tier | Gate vs torch/numpy | Gate vs itself |
|---|---|---|---|
| `sense_batch`, EDT, `build_sdf`, `inflate`, `astar`, LoS, `nearest_free`, `simplify` | **BIT — DONE** | `array_equal` + `.tobytes()` | 1-vs-8-thread exact |
| `to_occupancy` / `composite` / `dynamic_layer` / `world_to_cell` | **BIT — new surface** | `array_equal` on bool mask; `world_to_cell` exact (`std::rint`, half-even) | serial |
| `sdf_sample` (bilinear + normal renorm) | **FLOAT-EQUIV ≤1 ULP** | fuzz: `array_equal` on raw phi if it holds, else `allclose(rtol=1e-6,atol=1e-7)`; unit normal `rtol=1e-5` | exact (pure fn) |
| `coef_mlp.forward` | **FLOAT-EQUIV** | `allclose(rtol=1e-4, atol=1e-5)` over 1e5 feats incl. a `raw>20` softplus row | exact across threads |
| `bicycle_step` | **FLOAT-EQUIV** | single substep `allclose(rtol=1e-5,atol=1e-6)`; 200-step trajectory bounded | exact across threads/reruns |
| carrot FSM | **discrete-exact on identical inputs** | FSM-logic test: identical `mode/stall/turn/hist_count` given bit-identical `phi/nrm`; whole-drive mode-flip budget | exact |
| **whole drive** (`sim_world.step` vs `Swarm.step`) | **BEHAVIORAL** | reach-set ± tiny budget; `min_clearance` no-regression; short-horizon (pre-first-flip) pos < 1e-3 normalized; mode-flip-rate < 0.5% and every flip threshold-adjacent | 1-vs-8-thread exact |

**The critical new bit surface is `to_occupancy`.** It feeds the bit-exact EDT/`build_sdf`. It must compute `p = 1/(1+expf(-logodds))` in **float32** (numpy `BeliefGrid.p()` is float32 logodds → float32 sigmoid) and compare against float32 thresholds `max(p_thresh, 0.5+band)` / `min(1-p_thresh, 0.5-band)`. A float64 sigmoid here flips threshold cells and silently changes every downstream field. Dedicated raster-parity test, per plane.

---

## 2. Target `cvc::nav` inventory (signatures)

Namespace `cvc::nav`, raw-pointer SoA cores, `int num_threads` on every batch, all new TUs compiled under the existing discipline (**no `-ffast-math`, no `-ffp-contract=fast`**, float32 interior to track torch f32). **Name collision to avoid:** `cvc::nav::sdf_field` already exists in `grid_nav.h` (the *built* field). The sampler stack is named **`field_stack`**.

Files (all under `/home/joe/src/cvc/wt-libcvc-nav/`):

```
inc/cvc/nav/detail/parallel.h   NEW  (extract parallel_for from grid_nav.cpp)
inc/cvc/nav/coef_mlp.h          NEW  |  src/cvc/nav/coef_mlp.cpp        NEW
inc/cvc/nav/drive.h             NEW  |  src/cvc/nav/drive.cpp           NEW
inc/cvc/nav/belief_occupancy.h  NEW  |  src/cvc/nav/belief_occupancy.cpp NEW
inc/cvc/nav/sim_world.h         NEW  |  src/cvc/nav/sim_world.cpp       NEW
inc/cvc/nav/sim_thread.h        NEW  |  src/cvc/nav/sim_thread.cpp      NEW
```

### 2.1 `detail/parallel.h` (refactor)
Move the `template<class F> void parallel_for(int n, int num_threads, F&&)` verbatim out of the `grid_nav.cpp` anonymous namespace into `cvc::nav::detail`; `grid_nav.cpp` includes it (no behavior change — keeps the byte-identical batch guarantee). `drive.cpp`/`sim_world.cpp` reuse it.

### 2.2 `coef_mlp.h`
```cpp
class coef_mlp {  // 5→64→64→3 SiLU; softplus(net + log(expm1(bias)))
 public:
  static constexpr std::uint32_t kSupportedFormat = 1;
  static constexpr std::uint32_t kFlagSoftplusLogExpm1 = 1u << 0;
  static coef_mlp load(const std::string& path);
  static coef_mlp load_from_memory(const void* data, std::size_t nbytes);
  void  forward (const float* feats, int n, float* out) const;      // feats[n*in]→out[n*out] (a,b,g)
  std::array<float,3> forward1(const float feat5[5]) const;
  int in_features() const;  int out_features() const;
  std::uint32_t format_version() const;  std::uint64_t arch_hash() const;
 private:
  struct Layer { int rows, cols; std::uint32_t act; std::vector<float> w, b; };
  std::vector<Layer> layers_;  std::vector<float> out_bias_off_;  // = log(expm1(bias)), f32, folded once
  int in_=0, out_=0;  std::uint32_t fmt_=0, flags_=0;  std::uint64_t arch_hash_=0;
  void build_from_bytes(const std::uint8_t*, std::size_t);         // throws on any mismatch
};
```

### 2.3 `drive.h` — the three numerics + FSM + one-tick drive
```cpp
struct field_stack {                     // immutable sampler; NOT the built-field struct
  const float* data;   // borrowed [M*3*H*W] row-major: (plane*3+ch)*H*W + r*W + c; ch 0=phi 1=nx 2=ny
  int M, H, W;
  double mnx, mny, mxx, mxy, cx, cy, S;   // world<->grid transform (== SDFField)
};

// (1) grid_sample(bilinear, align_corners=True, padding=border) + unit-normal renorm.
void sdf_sample1(const field_stack&, int plane, float ox, float oy,
                 float& phi, float& nx, float& ny);
void sdf_sample (const field_stack&, const std::int32_t* plane /*null=>0*/,
                 const float* o /*[n*2]*/, int n, float* phi, float* nrm /*[n*2]*/, int num_threads=0);

// (2) features toward the CARROT: [phi, |carrot-o|, gdir_x, gdir_y, gdir·nrm]  (== Swarm._coef_feats)
void coef_feats(const float* phi, const float* nrm, const float* o, const float* carrot,
                int n, float* feat /*[n*5]*/);

struct veh_params { float L=.035f, delta_max=.6f, a_max=1.5f, a_lat_max=1.f, k_steer=.8f;
                    bool allow_reverse=true; };            // SdfNavigator.VEHICLE_DEFAULTS
struct integ_params { float rr, d_hat, dt, vmax; int nsub=1; };

// SoA over caller memory (numpy buffers OR sim_world's own vectors)
struct agent_soa {
  float *o /*[n*2]*/, *th, *sp, *goal /*[n*2]*/;
  std::int32_t *mode, *stall, *hist_count;                 // 0=SEEK 1=WALL
  float *turn, *dhit, *best, *init, *wall_entry /*[n*2]*/, *pos_hist /*[n*40*2] ring*/;
  std::uint8_t *we_valid, *tracking, *parked, *reached, *active;
  const std::int32_t* map_id;  int n;
};

// (3) kinematic bicycle, nsub substeps, steps==1; samples the field each substep; updates o/th/sp in place.
void bicycle_step(const field_stack&, const agent_soa&, const float* carrot,
                  const float* al, const float* be, const float* ga,
                  const veh_params&, const integ_params&, float* min_clr /*[n] nullable*/);

// carrot FSM (Swarm._plan_carrot, 6 masked branches) as a per-agent scalar loop; may brake parked sp.
void carrot_step(agent_soa&, const float* phi, const float* nrm,
                 float reach_tol, float a_max, float dt, float* carrot /*[n*2]*/);

// One drive tick over caller-owned SoA + field (sample→carrot→feats→mlp→bicycle→reach/park).
// GIL-free, threaded across agents (per-agent independent ⇒ deterministic in thread count).
void drive_step(const field_stack&, agent_soa&, const coef_mlp&,
                const veh_params&, const integ_params&, float reach_tol, int num_threads=0);
```

### 2.4 `belief_occupancy.h` (BIT tier)
```cpp
enum class unknown_policy { optimistic, pessimistic };
void to_occupancy(const float* logodds, int H, int W, unknown_policy,
                  float p_thresh, float band, std::uint8_t* occ);          // f32 sigmoid
struct dynamic_layer { int H, W; double ttl_s; std::vector<double> stamp;  // init -inf
  void mark(int r, int c, double t, int radius_cells);
  void occupancy(double t, std::uint8_t* occ) const; };
void composite_occupancy(const float* logodds, const dynamic_layer*, double t, int H, int W,
                         unknown_policy, float p_thresh, float band, std::uint8_t* occ);
std::pair<int,int> world_to_cell(double x, double y, double mnx, double mny,
                                 double mxx, double mxy, int H, int W);     // int(rint()) half-even
```

### 2.5 `sim_world.h` — owning aggregate (the pure-C++ runtime)
```cpp
enum class belief_mode { shared, clustered, private_ };
struct agent_spec { float start_x, start_y, goal_x, goal_y, color[3]; };
struct sim_config {
  int H, W; double mnx, mny, mxx, mxy, scale, center_x, center_y;
  std::vector<std::uint8_t> truth;             // [H*W]
  std::vector<float> prior_logodds;            // [H*W] or empty
  double sensor_range_m; int sensor_n_rays; double sensor_fov;
  int sense_every; float reach_tol = 0.8f;     // NORMALIZED units (NOT 0.15 — see §9)
  veh_params veh; integ_params integ;
  belief_mode mode = belief_mode::shared;
  std::vector<std::int32_t> map_id;  int M = 1; // caller-resolved grouping
  unknown_policy unknown = unknown_policy::optimistic;
  double unit_ttl_s = 4.0, l_occ = 2.2, l_free = -1.4, l_clamp = 8.0;
};
class sim_world {
 public:
  sim_world(sim_config, std::vector<agent_spec>);
  void load_policy(const std::string& path);              // coef_mlp::load
  void load_policy_bytes(const void*, std::size_t);
  void step();                                            // sense(gated)→rebuild→drive→metrics
  std::shared_ptr<const snapshot> take_snapshot() const;  // copy-out; field/occ by shared_ptr
  void retarget(int i, double gx, double gy);             // track_goal semantics
  void add_obstacle(double x0, double y0, double x1, double y1);
  bool all_reached() const;  int n() const;  int field_version() const;
};
```

### 2.6 `sim_thread.h`
```cpp
struct snapshot { std::uint64_t gen, tick; double world_t; int n;
  std::vector<float> pos, heading, speed, color, goal;   // WORLD metres; speed = sp/S
  std::vector<std::int8_t> mode; std::vector<std::uint8_t> active, reached;
  std::vector<std::int32_t> map_id; int field_ver;
  std::shared_ptr<const field_stack_owned> field;        // by ref, COW, never mutated
  std::shared_ptr<const std::vector<std::uint8_t>> occ; };
class triple_buffer { void publish(std::shared_ptr<const snapshot> prev,
                                   std::shared_ptr<const snapshot> curr);
  std::shared_ptr<const snapshot> read() const;
  std::pair<std::shared_ptr<const snapshot>,std::shared_ptr<const snapshot>> read_pair() const; };
template<int NMAX> class pose_seqlock { /* even→odd→memcpy→even; reader retries on odd/mismatch */ };
using command = std::variant<retarget_goal, move_obstacle, set_rate, pause, nudge, stop>;
class sim_thread { sim_thread(sim_world&, double hz=60);
  void start(); void stop(); void send(command);         // MPSC queue, drained top-of-tick
  triple_buffer& buffer(); double step_ms() const; std::uint64_t ticks() const, behind() const; };
```

---

## 3. Torch-free numeric replacements (the three, exactly)

All interior math **float32**. The one subtlety that decides fidelity is **torch's python-scalar promotion**: a float32 tensor op with a python `float` casts the scalar to f32 *then* operates; a `(max-min)` python subtraction is done in f64 *once* and cast at the division.

### 3.1 `sdf_sample` — transcribe `SDFField.sample` / `BatchedSDFField.sample`
```cpp
float Sf=(float)S, cxf=(float)cx, cyf=(float)cy, mnxf=(float)mnx, mnyf=(float)mny;
float denx=(float)(mxx-mnx), deny=(float)(mxy-mny);        // (max-min) in double, cast ONCE
float wx = ox/Sf + cxf,  wy = oy/Sf + cyf;                 // == on/S + c
float gx = 2.0f*(wx-mnxf)/denx - 1.0f,  gy = 2.0f*(wy-mnyf)/deny - 1.0f;
// grid_sample align_corners=True, border: unnormalize THEN clamp the continuous coord
float ix = (gx+1.0f)*0.5f*(float)(W-1);  ix = fminf(fmaxf(ix,0.0f),(float)(W-1));
float iy = (gy+1.0f)*0.5f*(float)(H-1);  iy = fminf(fmaxf(iy,0.0f),(float)(H-1));
int x0=(int)floorf(ix), y0=(int)floorf(iy), x1=x0+1, y1=y0+1;
float wx1=ix-x0, wy1=iy-y0, wx0=1-wx1, wy0=1-wy1;
float NW=wx0*wy0, NE=wx1*wy0, SW=wx0*wy1, SE=wx1*wy1;
// per-corner index clamp to [0,W-1]/[0,H-1] (= border for the +1 corner at an edge)
// accumulate NW,NE,SW,SE order: v = ((NW*p_nw + NE*p_ne) + SW*p_sw) + SE*p_se
// channel 0=phi (returned raw), 1=nx, 2=ny  from base=(plane*3+ch)*H*W
float g = sqrtf(nx*nx+ny*ny) + 1e-6f;  nx/=g;  ny/=g;      // == nrm/(nrm.norm()+1e-6) — the float point
```
`plane = map_id[i]` collapses shared (M=1→plane 0) / private (plane=i) / clustered (dense label). **No `groups` machinery** — the torch three-branch structure existed only to keep `grid_sample` one batched call. **Contract:** float-equiv; `array_equal` on raw phi where the fuzz shows it holds, `allclose` on the normalized normal.

### 3.2 `coef_mlp.forward` — transcribe `CoefMLP.forward`
```cpp
inline float siluf(float x){ return x/(1.0f+expf(-x)); }
inline float softplusf(float x){ return x>20.0f ? x : log1pf(expf(x)); }   // torch threshold=20 REQUIRED
// per layer: acc=b[o]; for i: acc += w[o*cols+i]*a[i];  (sequential f32, no FMA reliance)  a[o]=act(acc)
// output: y[j] = softplusf( net[j] + out_bias_off_[j] );   out_bias_off_ = log(expm1(bias)) folded at load
```
Sequential f32 accumulation ("Model-C"). Reduction lengths are 5 and 64 → a few ULP vs torch's blocked sgemm. **Contract:** float-equiv `rtol=1e-4`. If it ever exceeds tolerance, the lever is to *match torch's summation order*, **not** switch to f64 (f64 deviates from torch's f32 result, it does not converge to it).

### 3.3 `bicycle_step` — transcribe `sdf_nav.bicycle_rollout` lines 417–574, elementwise
Per agent independent ⇒ `parallel_for` over agents; `min_clr` init 9.9f; `hdt=dt/nsub`; `tan_dmax=tanf(delta_max)`. Each substep is a **line-for-line** port of the barrier `_ipc_dbdd`, pure-pursuit steering with the behind/turn-around branch, adaptive lookahead `L_d_eff`, repulsive-only steering bias `k_steer*tanh(F_rep·left)`, corner cap `v_corner`, **directional** stopping governor `v_stop_dir`, maneuvering creep floor, `behind`/`stuck_turning`/`allow_reverse` `a_long` overrides, semi-implicit `sp→th→o`, and the `d_cap` lateral clamp. Preserve op order and the `torch.where` branch arithmetic exactly (compute both sides, select) so libm-vs-ATen transcendental drift stays sub-ULP and does not reorder. **Contract:** float-equiv single substep; bounded 200-step trajectory.

The carrot FSM (`carrot_step`) is the **ring-buffer** vectorization from `swarm.py` (NOT the list-pop form in `nav.py` — `swarm.py` is the vectorized reference the tests pin). Its integer state (`stall/mode/turn/hist_count`) is **exact**; transcribe branches 1–5 verbatim, including that `pos_hist[slot]` is written for all agents but `hist_count` advances only for `tracked = tracking & active`, and the final `stall/best` select uses `tracking` (not `tracked`).

---

## 4. Weight file format `.cvcnav` + loader + exporter

Little-endian, f32 payload, header ints assembled byte-by-byte (host-endian-agnostic), CRC32 trailer, exact-size assertion.

```
off  size        field
0    8           magic "CVCNMLP\0"
8    4  u32      format_version = 1   (loader rejects > kSupportedFormat)
12   4  u32      flags  (bit0 = softplus(x + log(expm1(bias))) output transform)
16   4  u32      in_features = 5
20   4  u32      out_features = 3
24   4  u32      num_layers L
28   4  u32      out_bias_len (== out_features)
32   8  u64      arch_hash  (FNV-1a/64 over the arch descriptor below)
--- 40-byte header ---
L × { u32 rows; u32 cols; u32 act }        act: 0=identity 1=SiLU 2=ReLU 3=GELU-tanh
per layer i:  f32 W[rows*cols] (torch Linear [out,in] row-major) ; f32 b[rows]
              f32 out_bias[out_bias_len]   (RAW (1,3,4); loader folds log(expm1))
[optional]    u32 meta_len ; meta_len bytes  (git sha / checkpoint id / train date — optional)
              u32 crc32   (IEEE, over every preceding byte)
arch descriptor (hashed): pack("<IIII", in, out, L, flags) ++ per-layer pack("<III", rows, cols, act)
```

- **`arch_hash` is the "hidden-size change bumps the version" token.** Any change to in/out/layers/flags/per-layer shape-or-act flips it; the loader self-checks it and grl-snam pins it (`assert torch_arch_hash == native.arch_hash`), so `hidden=128` weights can never be paired with a 64-wide net silently. `format_version` is reserved for byte-layout/semantic changes (loader refuses newer).
- **Loader** (`coef_mlp::build_from_bytes`): validate magic → CRC → format ≤ supported → flags known → endpoints match in/out → layers chain → recompute and match arch_hash → read weights → fold `out_bias_off_ = log(expm1(raw))` in f32 → assert `r.off == body_end`. Throws (never a partial model). Add a static LE-only guard.
- **Exporter** `/home/joe/src/cvc/wt-grl-snam-nav/grl_snam/coef_export.py`: `serialize_coef_mlp(model)->bytes` walks the **live** `nn.Module` (activations discovered from `model.net`, `bias` buffer present), plus `serialize_checkpoint(pt_path)` that rebuilds a `CoefMLP` from a saved state_dict. `write_coef_mlp(model, path)`; CLI `python -m grl_snam.coef_export coef_sdf.pt coef_mlp.cvcnav`. Wire `write_coef_mlp` into `grl_snam/tools/train.py` and `grl_snam/tools/pipeline.py` so every checkpoint ships its deployable twin.
- **In-memory path is the parity substrate:** the same `bytes` feed torch (its source) and `pycvc.NavCoefMLP(blob)` — no file round-trip in the tolerance test.

---

## 5. Python-green seam + migration order

**pycvc surface** (append to `/home/joe/src/cvc/wt-libcvc-nav/bindings/pycvc/pycvc_nav.i`, following the existing writable-borrow discipline — validate exact dtype/shape/contig, never coerce-copy an in-place buffer, `Py_BEGIN_ALLOW_THREADS` across compute):

- Fine-grained (parity + optional acceleration): `nav_sdf_sample(field[M,3,H,W]f32, o[N,2]f32, plane[N]i32|None, mnx..S) -> (phi[N], nrm[N,2])`; `NavCoefMLP(bytes)` / `nav_coef_mlp_load(path)` with `.forward(feat[N,5])->(N,3)`; `nav_drive_step(field, plane|None, <SoA borrowed in place>, mlp, veh scalars, integ, reach_tol, mnx..S, num_threads) -> None`.
- Object surface (pure-C++ twin + full delegation): opaque `nav_sim_world_create(...)`, `_step`, `_snapshot(->dict of fresh numpy)`, `_retarget`, `_add_obstacle`, `_free`.

**`grl_snam/nav_native.py`** gains `HAS_DRIVE`, `HAS_SIM_WORLD`, `NativeCoefMLP`, thin `sdf_sample/drive_step/SimWorld` adapters, and a **separate** gate:
```python
def drive_enabled():
    return AVAILABLE and hasattr(_pycvc,'nav_drive_step') \
        and os.environ.get('GRL_SNAM_NAV_DRIVE','torch').lower() == 'native'
```

**Migration order** (each step green on its own):
1. C++ additive → no Python impact.
2. pycvc additive → grl-snam untouched, tests skip where absent.
3. grl-snam **tests only** (new parity tests) → this is where the C++ drive earns trust; tolerances tuned *here*, never in existing tests.
4. **Optional, late:** in-Swarm native dispatch behind `GRL_SNAM_NAV_DRIVE=native`. Realized by allocating the FSM SoA as **persistent numpy** with `self.o/th/sp` as `torch.from_numpy` **views** that alias it (never reassigned in native mode), keeping `Swarm.step`'s Python structure so `_sense_shared` monkeypatch (which `test_swarm.py` relies on) still gates sensing, and replacing only the sample→carrot→mlp→bicycle→metrics block with one `nav_drive_step`. `SdfNavigator`/`Squad` are **never touched** ⇒ `test_squad`/`test_nav` green by construction. This phase carries the most Python-green risk for the least deployment value, so it is last and independently revertible.

---

## 6. Validation plan (parity gates)

**Two-location oracle** (mirrors the existing SHA-pinned-action discipline):
- **grl-snam CI (torch present):** C++-via-pycvc vs *live* torch — exact for bit tiers, tolerance for float.
- **libcvc CI (no torch):** C++ vs a small **checked-in golden trace** generated by torch and regenerated deliberately, so libcvc's own gtests catch a drive regression torch-free.

**Golden capture environment (pin it in the test):** `torch.set_num_threads(1)`, TF32 disabled, float32 — so the "reference" is deterministic across boxes and `rtol=1e-4/1e-5` absorbs cross-BLAS variation.

**New grl-snam tests** (all `importorskip("pycvc")`):
- `tests/test_coef_mlp_parity.py` — `NavCoefMLP(blob)` vs torch `CoefMLP` over 1e5 feats incl. a `raw>20` row; round-trip (`serialize→load→forward` byte-stable); corruption (flip one payload byte → CRC raises; hand-edit a shape → arch_hash raises).
- `tests/test_sdf_sample_parity.py` — `nav_sdf_sample` vs torch `grid_sample`; the exact-equality fuzz over position×field, all three plane geometries.
- `tests/test_bicycle_parity.py` — single-substep and 200-step trajectory at fixed coeffs/carrot (decoupled from MLP/FSM).
- `tests/test_sim_world_parity.py` — torch `Swarm` vs C++ `sim_world` from identical truth/specs/weights/mode; **behavioral** gate (reach-set, min-clearance no-regression) + short-horizon tight pos/th + mode-flip budget; all belief modes; sensing frozen (Variant 1) then live (Variant 2, after belief port).
- `to_occupancy`/`composite` raster bit-parity per plane.

**New libcvc gtests** (`nav_drive_test.cpp`, `sim_world_test.cpp`): hand-golden bilinear value; MLP vs a torch-exported vector; bicycle single substep vs golden; each carrot branch; `DriveDeterministicAcrossThreadCounts` (byte-identical 1 vs 8, like `NavSense`); `triple_buffer`/`pose_seqlock` 1-writer/N-reader no-tear checksum; `sim_world` smoke (reach→park, no NaN, `field_version` bumps).

---

## 7. Phased plan (P0…P8) — each shippable, never reds Python CI

- **P0 — `parallel.h` extraction.** *Deliverable:* `detail/parallel.h`; `grid_nav.cpp` includes it. *Gate:* full existing `nav_test` + `test_nav_cpp_parity`/`test_nav_sense_parity` still byte-green. *Risk:* trivial; confined refactor. *Python:* untouched.
- **P1 — `sdf_sample`.** *Deliverable:* `drive.{h,cpp}` sample only + `nav_sdf_sample` + `nav_native.sdf_sample` + gtest + `test_sdf_sample_parity`. *Gate:* float-equiv fuzz (record the exact-equality residual as a diagnostic). *Risk:* float32 coordinate promotion offset — the fuzz catches a systematic shift immediately. *Python:* opt-in only; torch path unchanged. Independently useful to any C++ SDF consumer.
- **P2 — `coef_mlp` + `.cvcnav`.** *Deliverable:* `coef_mlp.{h,cpp}`, format+loader, `coef_export.py`, `NavCoefMLP`, `test_coef_mlp_parity`. *Gate:* `rtol=1e-4`; round-trip + corruption + arch_hash. *Risk:* softplus threshold=20 (exercised by a `raw>20` row); sgemm reduction order (bounded by tiny net). *Python:* torch `CoefMLP` stays default/training/reference.
- **P3 — `bicycle_step` + `carrot_step` + `coef_feats`.** *Deliverable:* the rest of `drive.cpp` + `test_bicycle_parity` + FSM-logic gtest. *Gate:* single-substep + 200-step at fixed coeffs; identical FSM transitions on bit-identical inputs. *Risk:* **highest arithmetic risk** — many transcendentals over `nsub` substeps; mitigated by exact op order, no-fast-math, golden traces. *Python:* torch rollout stays reference/gradient path.
- **P4 — `drive_step` (assembles P1–P3).** *Deliverable:* `nav_drive_step` binding + `DriveDeterministicAcrossThreadCounts`. *Gate:* threaded == serial byte-identical. *Risk:* none new (per-agent independent). *Python:* opt-in dispatch flag defined but default `torch`.
- **P5 — `belief_occupancy` (BIT).** *Deliverable:* `to_occupancy/composite/dynamic_layer/world_to_cell` + raster bit-parity gtest/pytest. *Gate:* `array_equal` bool mask; `world_to_cell` exact. *Risk:* the f32-sigmoid new bit surface — dedicated per-plane test. *Python:* untouched.
- **P6 — `sim_world` (single-goal reactive, shared belief).** *Deliverable:* `sim_world.{h,cpp}` (COW `field_stack` swapped by `shared_ptr`) + `nav_sim_world_*` + `sim_world_test` + `test_sim_world_parity`. *Gate:* behavioral whole-drive vs torch `Swarm`, both belief-frozen then live. **This ships the headline pure-C++ thousands-of-vehicles-on-known-Austin capability, zero torch.** *Risk:* emergent mode-flip divergence — behavioral gate + flip budget + "every flip threshold-adjacent" assertion. *Python:* torch `Swarm` stays default.
- **P7 — `sim_thread` + snapshot + a C++ renderer/game-engine host wiring.** *Deliverable:* `triple_buffer` (then `pose_seqlock`), `sim_thread`, pycvc_gl exposure. *Gate:* threaded==synchronous determinism; port `test_swarm`'s no-tear reader test to C++; live retarget/pause take effect. *Risk:* C++17 `atomic<shared_ptr>` may spinlock internally → offer `pose_seqlock` for the hot pose-only path. *Python:* untouched.
- **P8 (optional) — in-Swarm native dispatch.** *Deliverable:* `Swarm` native branch behind `GRL_SNAM_NAV_DRIVE`, SoA-view surgery, a dedicated `native` CI lane running `test_swarm` at the 5e-3 tier. *Gate:* existing suite green in `torch` mode (unchanged) AND in the `native` lane. *Risk:* view-aliasing desync — assert `sw.o` shares its base buffer after a step; **default never flips.** *Python:* green in both modes by construction.

**Deferred (explicitly out of scope, designed-for):** the multi-waypoint route-spine driver (`SdfNavigator.drive_to_goal` / `Squad`: A* belief spine + lookahead subgoal + closest-approach bookkeeping) is a thin C++ orchestration layer *above* `sim_world` that reuses the already-ported `astar`/`simplify` kernels and `sim_world.retarget`; `sim_world`'s API is shaped to accept it later. Point-mode `sdf_rollout` (training/`dynamics='point'`) is not the deployment vehicle — not ported.

---

## 8. Top risks

**To bit/float fidelity:**
1. **Chaotic divergence from a threshold-adjacent FSM flip** (`stall>70`, `moved<0.15`, `dg<best-1e-3`): one ~1-ULP sample difference at a wall-follow/turn-around bifurcation sends an agent a different way and late-tick positions separate without bound. *Mitigation:* guarantee bit-identity **up to and including the built SDF field**; gate the drive on **behavior** (reach-set, min-clearance no-regression), reserve tight pos/th tolerance for the pre-first-flip horizon only, and require every flip to be provably threshold-adjacent (a flip that isn't localizes a real port bug).
2. **`to_occupancy` — the new bit surface feeding the EDT.** A float64-vs-float32 sigmoid or numpy scalar-promotion mismatch flips a threshold cell and silently changes the whole field. *Mitigation:* f32 `expf` sigmoid + f32 threshold compares; per-plane raster bit-parity test in P5 *before* `sim_world` depends on it.
3. **ATen `grid_sample`/`sgemm` are not bit-stable across builds/threads**, so "the reference" itself varies. *Mitigation:* capture goldens single-threaded, TF32-off, f32; assert at `rtol` that absorbs cross-BLAS variation; document the capture env.
4. **libm vs ATen transcendentals over `nsub` substeps** (`atan2/tan/tanh/sqrt`). *Mitigation:* no-fast-math/no-fp-contract TU; libm is what numpy dispatches to anyway; golden trajectory bound + behavioral end-state.

**To keeping Python green:**
5. **A silent float-equivalent dispatch** would leave existing 5e-3 tests passing while redefining the reference. *Mitigation:* Rule 1 — only bit tiers default-on; drive is explicit opt-in, torch is the Python default and golden generator. **The default never flips.**
6. **P8 SoA view-aliasing desync** (a stray torch op reallocating a column). *Mitigation:* native mode never reassigns `o/th/sp`; assert `sw.o` shares its base after a step; P8 is optional, last, revertible by a one-line flag.
7. **Stale weights** (retrain without re-export). *Mitigation:* `arch_hash` + provenance trailer; `write_coef_mlp` wired into `train.py`/`pipeline.py`; `load_policy` logs the hash.

---

## 9. Corrections to the input designs + open questions for sign-off

**Errors caught against the actual code:**
- **Design 3's `sim_config` `reach_tol = .15f` is wrong.** Both `Swarm` and `SdfNavigator` default `reach_tol=0.8`, in **normalized** units. Using 0.15 would change the reach/park set on every run. Corrected to `0.8f` in §2.5.
- **Name collision:** `cvc::nav::sdf_field` already exists in `grid_nav.h` (the *built* field). Designs 3/5 reuse `sdf_field` for the *sampler*. Renamed the sampler to **`field_stack`** (§2.3).
- **FSM source of truth:** port `swarm.py._plan_carrot` (ring-buffer form), **not** `nav.py._plan_carrot` (list-pop form). They compute `moved` against different history references; `swarm.py` is the vectorized reference the tests pin, and its integer state is exact.
- **`allow_reverse`:** `bicycle_rollout`'s own default is `False`, but the deployment path (`SdfNavigator.VEHICLE_DEFAULTS`, inherited by `Swarm.veh`) is `True`. C++ `veh_params` default must be `true` (§2.3), and the golden must exercise the reverse branch.
- **Weight bias:** keep it **raw** in the file (D2), not pre-folded (D5) — `log(expm1)` is a load-time constant folded once in f32, and a raw mirror of the state_dict is auditable. D1's fixed-4742-float blob is rejected in favor of D2's generic+arch_hash layout.
- **CMake gtest wiring:** each new `*_test` needs **all five** `nav_test` entries in `/home/joe/src/cvc/wt-libcvc-nav/src/cvc/tests/CMakeLists.txt` — `add_executable` (L100), the `TEST_TARGETS`/list membership (L168), `target_link_libraries` (L320), `target_compile_features(... cxx_std_17)` (L850), `gtest_discover_tests` (L1014) — or the target is silently omitted (the "four-entry" trap in MEMORY).

**Open questions needing a decision (do not block P0–P5):**
- **Deployment belief mode:** does a C++ renderer/game-engine host need clustered/private belief, or is **shared (M=1)** the whole deployment path? Shared is the thousands-of-agents target; if only shared, `map_id` is all-zeros and the M>1 COW/rebuild cost is untested-in-anger. Confirm before P6 sizing.
- **`nsub` at deploy** (`meta.get("nsub",1)`): if always 1, the golden and gtests can pin `nsub=1`; if >1, the multi-substep transcendental accumulation must be in the golden.
- **Whole-drive tolerance/horizon sign-off:** the proposed short-horizon `1e-3` normalized, `<0.5%` flip budget are from the existing `Squad(batched_drive)` 5e-3 tier and need empirical confirmation across all six stories and multiple seeds in P3/P6 — this is the single number that gates P6, and it is the top fidelity risk.
- **Canonical `.cvcnav` home + provenance policy:** where the blessed weights live (libcvc test-data vs pycvc-published vs per-deployment bundle) (kept out of public repos where required); and whether the provenance trailer is required for an audit trail.

Files to create are listed in §2; the two files to edit are `/home/joe/src/cvc/wt-libcvc-nav/src/cvc/CMakeLists.txt` (add headers ~L88, sources ~L171 next to `nav/grid_nav.cpp`) and `/home/joe/src/cvc/wt-libcvc-nav/bindings/pycvc/pycvc_nav.i` (append the marshalling), plus the new grl-snam `coef_export.py`, `nav_native.py` additions, and the parity tests under `/home/joe/src/cvc/wt-grl-snam-nav/tests/`.
---

## 10. Decisions on the §9 open questions

### 10.1 Belief modes — ship all three, benchmark each

`belief_mode` is a first-class knob with three values, each benchmarked (`bench_modes_nsub.py`):

- **all-shared** (`M=1`) — one belief plane, the O(1)-map / thousands-of-agents path.
- **clustered-shared** (`clusters=K`, `1<K<N`) — K groups each sharing a belief; isolation is structural.
- **all-private** (`M=N`) — one belief per agent, the fog-of-war fidelity twin.

The C++ `sim_world` carries the same `map_id[N]` seam, so all three are one data parameter in the port too;
P6 sizes for whichever the deployment host actually needs (see the bench for the memory/parallelism trade).

### 10.2 `nsub` — configurable, deployment default 1

`nsub` (bicycle substeps per world dt) is now a `Swarm` constructor arg. `None` inherits the story's meta
value (so the serial-navigator parity test, which reads the same meta, still holds); an explicit value
overrides. **The deployment / C++-port default is `nsub=1`**; the goldens and gtests pin `nsub=1`, and the
`bicycle_step` parity harness additionally exercises `nsub∈{2,4}` (the multi-substep transcendental
accumulation) so a host that raises it stays covered. `nsub` scales only the drive (not the sense), so at
the sense-bound steady state its effect is second-order; the drive-only sweep isolates it.

### 10.3 Whole-drive validation — options beyond the behavioral gate

The problem is chaos, not per-op error: the drive is float-equivalent (~1 ULP) to the torch reference, and
the carrot FSM makes threshold decisions (`stall>70`, `moved<0.15`, `dg<best-1e-3`, `|p-wall_entry|>2.0`),
so a sub-ULP sample difference can flip a decision on a *different* tick and separate the two trajectories
without bound — while both remain perfectly valid drives. A single tight position tolerance over a long
horizon is therefore the wrong test. Seven alternatives, and the recommended layering:

| # | Option | What it validates | Tolerance reach | Cost |
|---|---|---|---|---|
| 1 | **Behavioral gate** (roadmap default) | outcome: reach-set, min-clearance no-regression, mode-flip rate < budget, every flip threshold-adjacent | loose, whole horizon | needs empirical budget calibration; a bug that doesn't change the outcome can slip |
| 2 | **Reference-injection / lockstep** | feed C++ and torch the SAME upstream each tick (same carrot + same α,β,γ → test rollout alone; same phi/nrm → test MLP alone; same field+pos → test sampler alone) — re-sync every tick kills chaotic amplification | **tight** per-op (rtol 1e-4…1e-6), every tick | a lockstep harness; doesn't test the assembled long-horizon dynamics |
| 3 | **FSM-decision replay** | record torch's discrete FSM transitions and force them into C++ → decisions can't diverge, so the CONTINUOUS math is comparable exactly | **tight**, full horizon | plumbing to inject FSM state; the FSM-on-drifted-phi decision still tested separately |
| 4 | **Fixed-coefficient goldens** | bypass the MLP (constant α,β,γ) → rollout + FSM determinism without MLP variance | tight, per trajectory | doesn't test the MLP or the coupled MLP↔rollout loop |
| 5 | **Threshold quantization** (opt-in, a *different* sim) | snap `phi/dg/moved` to a coarse grid before the threshold compares → both drives make identical decisions by construction | **bit-ish**, full horizon | changes behavior slightly (more robust/reproducible); goldens regenerate under it; could be the deploy default |
| 6 | **Deterministic reference capture** (prereq) | pin torch nondeterminism (single-thread, TF32 off, deterministic algos, fixed BLAS) so "the reference" is itself reproducible | shrinks every budget | doesn't solve chaos; must document the capture env |
| 7 | **Error-bound certification** | prove accumulated per-op error can't cross the nearest threshold within N ticks | a guarantee | heavy, brittle vs integer stall counters; overkill |
| — | **Chase bit-identity** (match ATen reduction order) | — | — | infeasible for the MLP sgemm (BLAS/tile/thread-dependent); the 4-tap sampler *could* be bit-exact but buys nothing alone |

**Recommendation — a layered gate, not one number:**
- **L0** deterministic reference capture (6) — prerequisite.
- **L1** per-op lockstep injection (2) — tight rtol on sampler / MLP / rollout individually.
- **L2** FSM-decision replay (3) — tight, full-horizon check that the assembled *continuous* math tracks torch.
- **L3** discrete-FSM test — given bit-identical inputs (which hold up to the sample), the FSM makes identical
  decisions; a discrete-exact assertion.
- **L4** behavioral gate (1) — the only *loose* layer, reserved for emergent whole-system behavior.
- **Optional deploy lever:** threshold quantization (5) turns L4 tight and makes the drive
  reproducible-by-construction, at the cost of being a slightly-different (more robust) simulator — offer it
  as `fidelity="reproducible"`, never the silent default.

This gives a tight, non-chaotic test for every numeric component *and* for the assembled continuous dynamics,
and only leans on a loose gate for genuinely emergent behavior — which is where looseness is honest.

### 10.4 Canonical `.cvcnav` weights location in the install prefix

**`$PREFIX/share/cvc/nav/coef_mlp.cvcnav`** — following the cvcpkg payload convention (`$PREFIX/share/<pkg>/`,
and libcvc's CMake package is `cvc`) and CMake's `CMAKE_INSTALL_DATADIR`. Install rule (P2):

```cmake
install(FILES ${CVC_NAV_WEIGHTS_FILE}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/cvc/nav COMPONENT libcvc)
```

`coef_mlp::default_weights_path()` resolves in this order (first hit wins):
1. an explicit path passed to `coef_mlp::load(path)` — tests, custom deployments;
2. the `CVC_NAV_WEIGHTS` environment variable — redeploy without recompiling;
3. `CVC_NAV_DATADIR "/coef_mlp.cvcnav"` — a constant baked into `inc/cvc/core/config.h.cmake` at configure
   time as `${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/cvc/nav` (libcvc already generates `config.h`);
4. a relocatable fallback resolved from the loaded libcvc `.so` via `dladdr` —
   `<so_dir>/../share/cvc/nav/coef_mlp.cvcnav` — for prefix-relocated / cvcpkg installs where the baked
   prefix moved.

The exporter (`coef_export.py`, P2) writes the blessed weights into the build tree / test-data dir; the
`libcvc` recipe packages `share/cvc/nav/`. This matches the cvcpkg prefix-payload layout (installed artifacts
live in the prefix under `share/<pkg>/`, never the root).

---

## 11. Benchmarks — belief modes × nsub (measured)

384² city, 32 cores, C++ `sense_batch` live, steady state (sense + rebuild every 4th tick, drive on the
other 3), `nsub=1`. Reproduce with `python -m grl_snam.tools.belief_bench`.

Steady-state mean tick, **after** the raycast-parallelization (libcvc `abfb87d`); the "was" column is the
pre-optimization number (plane-parallel sense, shared single-threaded at K=1):

| N | all-shared (M=1) was → now | clustered/8 | all-private (M=N) |
|---|---|---|---|
| 256 | 46 → **~20 ms** | ~25 ms | 203 ms · 5 fps |
| 512 | 74 → **25 ms · 40 fps** | **31 ms · 32 fps** | 374 ms · 3 fps |
| 1024 | 144 → **44 ms · 23 fps** | **42 ms · 24 fps** | 768 ms · 1 fps |
| 2048 | 291 → **88 ms · 11 fps** | **68 ms · 15 fps** | 1593 ms · 0.6 fps |
| 4096 | 586 → **~200 ms** | **196 ms · 5 fps** | (memory) |

**The drive is cheap; the sense raycast was the steady-state wall — now parallelized across agents,
bit-identically.** The raycast reads only `truth` + the agent's boxes, so it is order-free and fans out over
all N agents; only the cheap log-odds fold stays serial per plane in ascending index (the order N serial
`BeliefGrid.sense` calls fold in). Result: **shared-mode ~3.3× faster** (144 → 44 ms @1024), the plane-count
no longer caps sense parallelism. So:
- **shared (M=1)** — smallest footprint (one ~4 MB plane, one EDT); sense now scales across cores. Best when
  the map is largely known (drive-bound: thousands @ 60 Hz, `bench_swarm.py`).
- **clustered/K** — still the balanced choice (K-EDT cost, K ≈ cores); now the raycast is N-way regardless.
- **private (M=N)** — **N EDT rebuilds** dominate and memory is O(N); the fidelity twin, at a handful of
  agents. Unchanged by this optimization (EDT-bound, not raycast-bound).

Scaling plateaus at ~4× (≈200 ms raycast @1024, num_threads 8→32 barely moves). **Measured cause: this box's
memory subsystem, not the code.** The dev box is a dual-socket Xeon E5-2650 v2 — **16 physical cores + HT (32
logical), 2 NUMA nodes, DDR3** — so a memory-touching parallel loop caps near ~4× (HT gives ~nothing on
memory-bound work; cross-socket NUMA access on the per-agent stores throttles bandwidth). A **window-local
scratch** (index the gen-stamped counts by FoV offset instead of the full grid) was implemented and measured:
**no change** — the full-grid scratch was already cache-resident (a 40×40 window touches only ~40 strided
cache lines), so it was reverted rather than ship complexity for nothing. The remaining lever on this hardware
is **NUMA-aware allocation** (a flat per-socket arena for the Phase-A stores, first-touch on the owning
socket); on a modern single-socket many-core box the current code should scale much further as-is. Independent
of scaling: `pipeline_edt` (rebuild off the critical path) and staggered/subsampled sensing for shared belief.

`nsub` is **second-order at steady state** (sense dominates): shared N=4096 is 586/591/570 ms at nsub=1/2/4.
In the **drive-only / known-map** regime `nsub` multiplies the drive roughly linearly (a direct
throughput-vs-substep-accuracy trade), so a host raises it only where thin-wall tunnelling matters.

---

## 12. CUDA — see the separate assessment

A decision-ready, code-verified CUDA assessment lives in [CVCNAV_CUDA_ASSESSMENT.md](CVCNAV_CUDA_ASSESSMENT.md). Bottom line: **not yet** — the one measured wall (single-threaded shared `sense_batch`) is a CPU decomposition artifact fixable bit-identically without a GPU (the raycast-parallelization); do that + the torch-free CPU drive first. A CUDA path is a conditional, device-resident, shared-belief, float-equivalent third twin that earns its keep only for a *named* deployment above the CPU ceiling measured on its **own** GPU. It found a real prerequisite bug: `--use_fast_math` is applied target-wide in `CMake/SetupCUDA.cmake`, which would silently break the float-equivalence contract for any future nav `.cu`.

---

## 13. Implementation status — the port is complete

Every phase landed (each build → test → commit, Python green throughout). A pure-C++
host runs the whole shared-belief swarm with **zero libtorch**, float-equivalent to torch.

| phase | delivered | fidelity vs torch |
|---|---|---|
| P0 `detail/parallel.h` | ✓ | refactor, byte-green |
| P1 `sdf_sample` | ✓ | **bit-exact** |
| P2 `coef_mlp` + `.cvcnav` + exporter | ✓ | 9.5e-7 |
| P3 `coef_feats` + `bicycle_rollout` | ✓ | ~1e-7 |
| P4 fused `drive_step` | ✓ | float-exact end-to-end |
| P5 `belief_occupancy` (BIT surface) | ✓ | **byte-identical** field |
| P6 `sim_world` + carrot FSM | ✓ | behavioral: identical reach-set, sub-5cm/80 ticks |
| P7 `sim_thread` | ✓ | concurrent, lock-free, no-GIL |
| P8 in-`Swarm` native dispatch (`GRL_SNAM_NAV_DRIVE=native`) | ✓ | float-equiv: 1.5e-5/120 ticks, identical reach-set |
| CUDA `drive.cu` (GTX 1650) | ✓ | float-equiv ~5e-7 |
| device-resident `sim_world_cuda` (GTX 1650) | ✓ | bit-exact/step, p50 bit-tight/250 ticks, reach-set match |
| self-supervised trainer `coef_train` (CPU) | ✓ | finite-diff gradcheck: dir 2.2e-4, per-param 2.8e-3 |
| CUDA trainer `coef_train.cu` (GTX 1650) | ✓ | loss+grad vs CPU 4.2e-7, cos 1.000000 |
| pure-C++ ergonomics | ✓ | `default_biased`, `from_occupancy`, `default_weights_path`, examples |

**52 pytest + 39 gtest green.**

### Training is torch-free too (`coef_train`)

The last torch dependency is gone: `coef_train.py` is ported into `cvc::nav` as pure
C++ — no dataset, no labels, the gradient comes from a differentiable rollout over a
scene's SDF straight into the coefficient net. It differentiates the *simple point-mass
`sdf_rollout` surrogate* (not the branch-heavy bicycle), so the reverse pass is small
hand-written adjoints (bilinear-sample position VJP, MLP backward, IPC-barrier
derivative, rollout chain) with truncated BPTT. Correctness is **torch-independent**: a
finite-difference gradcheck is the ground truth. The CUDA trainer (`coef_train.cu`) is a
device transcription of the same adjoints (atomicAdd param grads, checkpoint/recompute)
and reproduces the CPU loss+gradient to 4.2e-7. Scene source: `city_scene()` ports the
Python `STORIES["city"]`, `occupancy_scene()` takes any rasterized map (train on the
deployment terrain directly). Trained weights export to the same `.cvcnav`.

Result — end to end, pure C++: a policy trained on the city scene (CPU or GPU) **drives
the bicycle `sim_world` and improves reach ~62% → ~65%**, above the hand-tuned (1,3,4)
basin. It is a *refinement* of that basin, not from-scratch learning (the surrogate has
no turning limits, so the default lr is 2e-4; `coef_train.py`'s never-run 1e-3 over-fits
the surrogate and collapses navigation). `nav_train_demo` retrains on the box and writes
the `.cvcnav` with zero torch. Note: this validated the objective for the first time —
the shipped 57% policy was the *untrained* seeded basin (torch training had segfaulted).

The device-resident GPU twin (`sim_world_cuda`) keeps the field, `.cvcnav` weights
and **every** SoA agent column (pose + full carrot-FSM state) on the GPU across
ticks: `step()` launches sample → carrot FSM → fused drive → reached/park with no
host round-trip, and `snapshot()` copies only the pose-sized columns a renderer
needs. It is a per-agent transcription of `carrot_step` sharing `drive.cu`'s device
math, gated against the CPU `sim_world` (`NavSimWorldCuda.TracesCpuSimWorld`): after
one tick GPU==CPU **to the bit**, the median agent stays bit-tight over a 250-tick
roll, and the reach count matches — the tail past that horizon is the documented FSM
mode-flip chaos, not drift. Belief is now **M static planes via a per-agent
`map_id`** (the same shared/grouped/private grouping the CPU twin has, mirrored on
the device as an `[M,3,H,W]` field block): shared (M=1, the thousands-of-agents
path) or grouped/private, where different groups can carry genuinely *different*
known maps — the GPU analog of the CPU's grouped belief under `freeze_sense`, but
capable of per-group intel. `GroupedIdenticalPlanesMatchShared` gates the
plane-offset math bit-for-bit; `GroupedDifferentPlanesRouteApart` shows two agents
with one start+goal but different `map_id` diverge onto their own maps. Still
static-map (no on-device sensing — live fog-of-war stays on the CPU `sim_world`).
Bench on a bigger GPU box next.

### The pure-C++ path (dropping agents into a cvcGL scene / lsystem_forest)

```cpp
sim_world::config cfg = /* rows/cols, bounds, scale, vehicle params */;
// zero setup: a default policy; or coef_mlp::load(coef_mlp::default_weights_path())
sim_world world = sim_world::from_occupancy(cfg, scene_occupancy, coef_mlp::default_biased(), N);
// per frame: step, then feed world poses into per-agent GeometryNodes
world.step();
world.snapshot(pos_world, heading, speed, mode, reached);
// or run it off the render thread, lock-free:
sim_thread sim(world, 60.0); sim.start();  auto frame = sim.read();  sim.retarget(i, gx, gy);
```

See `examples/nav_swarm_demo.cpp` (`-DCVC_BUILD_NAV_EXAMPLE=ON`), a compilable template.

### Was deferred — now DONE

- **Device-resident `sim_world_cuda`** ✓ — field + `.cvcnav` weights + full SoA agent
  columns (pose *and* carrot-FSM state) stay GPU-resident across ticks; fused
  sample→carrot→drive→park; pose-only D2H. Validated bit-exact/step, p50 bit-tight
  over 250 ticks vs the CPU `sim_world` on a GTX 1650 (`NavSimWorldCuda.TracesCpuSimWorld`).
  (CUDA-Graph replay / CUDA-GL interop remain a throughput follow-up for a bigger box.)
- **P8 in-Swarm native dispatch** (`GRL_SNAM_NAV_DRIVE=native`) ✓ — the torch `Swarm`
  drives via the C++ path when opted in; float-equiv 1.5e-5/120 ticks, default stays torch.
- **The torch-free TRAINER** ✓ (beyond the original plan) — `cvc::nav::coef_train`
  (CPU + device-resident CUDA), surrogate *and* bicycle rollouts, gradcheck-validated,
  exposed to Python behind `GRL_SNAM_TRAIN_BACKEND=native`. See libcvc `docs/NAV_TRAINING.md`.

### Remaining (non-blocking)

- **Ship an actually-trained default** — the installed `share/cvc/nav/coef_mlp.cvcnav` is
  the untrained biased seed (~57% reach; zero-config driving works). Run `coef_train` (native
  or torch) to the ~65% target and install that as the default.
- **The live-demo path** (next round) — wire `sim_world`/`sim_thread` snapshot poses into
  per-agent cvcGL `GeometryNode`s in the `lsystem_forest` scene, rasterize the procedural
  island to an occupancy grid, and add a `swarm_live` demo entry point. See the readiness
  audit in the session notes.
