# Navigation statistics & the base-policy scorecard

grl-snam is the **base (RF-free)** owner of the two-layer nav-stats design: per-vehicle + per-episode
navigation telemetry and a corpus scorecard that ranks base-policy checkpoints. It is field-mirrored
by the C++ `cvc::nav::nav_stats` (libcvc `docs/NAV_STATS.md`) — the same schema and the same
hand-computed numbers — and the RF/comms layer (grl_snam_dbg) extends it, joined by vehicle index.

## The schema (`grl_snam.metrics`)

- **`NavMetrics`** — one per-step snapshot (position, heading, speed, `clearance_m`, `inside_building`,
  `reached`, …).
- **`NavStats`** — the rolling per-vehicle accumulator: `steps`, `penetration_steps`, `min_clearance_m`,
  `total_path_m`, `turn_total_rad`, `fuel_used`, `goals_reached`. Fed one `NavMetrics` per tick via
  `NavStats.update(m)`.
- **`NavStats.seed_start(x0, y0)`** — prime the accumulator with the true start pose so `total_path_m`
  counts the start→first-step leg, matching the C++ collector (which seeds `prev_pos`). Turn/fuel still
  skip the first sample. Call once before the first `update()`.

## The three collection paths

All three collect the **same** base row, so a checkpoint scores identically whichever runs:

1. **Pure-Python `SdfNavigator`** (`grl_snam/nav.py`) — `_metrics()` builds a `NavMetrics` each step;
   the caller accumulates via `NavStats.update` (the reference collector).
2. **Vectorized `Swarm`** (`grl_snam/swarm.py`) — opt-in `Swarm(collect_stats=True, contact_radius_m=…)`
   folds each tick (re-sampling `phi` post-rollout, converting normalized→metres), and
   `sw.episode_stats()` reduces to an `EpisodeStats`. Default-off keeps the thousands-of-agents hot
   path byte-identical.
3. **Native `sim_world`** (`grl_snam/nav_native.py`) — `NativeSimWorld.begin_nav_stats(...)` arms the
   C++ `sim_world`'s own internal collector (torch-free, on the sim thread) and `episode_stats()`
   reduces via the same scorecard reducer. Requires a pycvc with the `nav_sim_world_*` bindings
   (gated behind `HAS_SIM_WORLD_NAVSTATS`).

## The scorecard (`grl_snam.scorecard`)

`EpisodeStats.from_nav_stats(vehicles, straights_m=, arrival_times_s=, veh_contacts=, min_sep_m=)`
reduces a list of per-vehicle `NavStats` (plus the fields `NavStats` doesn't carry) into one episode
record; `aggregate_nav(episodes, checkpoint)` reduces a corpus into a single `NavScorecard` — the
RF-free fitness row (arrival, economy, safety, material). Mirrors C++ `cvc::nav::nav_scorecard`.

## Material & terrain-risk buckets

The scorecard's material dimension is **discrete**, mirroring C++ `veh_nav_stats`: each vehicle's
`NavStats` carries `time_over_material_s[13]` and `dist_over_material_m[13]`, indexed by the shared
13-class palette (`grl_snam.material_palette`, positional with `cvc::nav::kNumMaterials` and the
`cvc::dbg` `MATERIAL_TABLE`). `aggregate_nav` reduces the time buckets into
`NavScorecard.material_time_share[13]`, and `terrain_risk_share(...)` sums the traversable
terrain-risk classes (foliage/soil/water/rock) into the single **"time in terrain-risk areas"**
number the eval CLI prints as `risk-time%`.

The id source is a `grl_snam.material.MaterialIdRaster` — a nearest-cell per-agent sampler, the twin
of C++ `nav_samplers.material_id(x,y)`. It is **stats-only**: attaching it (`Swarm(...,
material_ids=…)`, `SdfNavigator.material_ids`, or `scorecard_eval` by default) classifies what each
vehicle drives over and populates the buckets **without changing navigation** — the arrival/economy/
safety numbers are bit-identical with or without it. `city_material_ids(truth, …)` builds a corpus
raster (buildings→concrete, free→open_air, deterministic terrain-risk disks). With no raster attached
every bucket stays zero, so a non-material drive is byte-identical.

## Consuming it — eval and training

- **Eval CLI:** `python -m grl_snam.tools.scorecard_eval [--checkpoint run/coef.pt] [--scenes N] [--json card.json]`
  runs a CoefMLP over a city scene corpus with the `Swarm` collector and prints/writes the
  `NavScorecard`.
- **Training tie-in:** `python -m grl_snam.tools.coef_train --score [--score-scenes N] [--score-json …]`
  scores the freshly-trained checkpoint with the base scorecard (arrival/economy/safety) instead of
  raw reach — the single fitness signal a campaign ranks on.

## Training against terrain risk (opt-in `--w-risk`)

`coef_train --rollout bicycle --w-risk W` trains the policy to spend less time in terrain-risk
areas. Three pieces combine (all default off → the objective is byte-identical to the geometry-only
trainer):

- **Risk-lookahead feature** — `coef_feats(..., material=…)` appends the WORST terrain risk between
  the vehicle and its carrot (the terrain twin of the grip μ-probe; `add_risk_feature(model)` widens a
  trained net to see it, output-identical at init). The coefficients can now *see* risk ahead.
- **Reroute force** — the material force `F_soft = -lam_soft·∇risk` steers the trajectory around risk.
  This is the lever that actually reroutes. `--lam-soft` sets a **fixed** strength; `--learned-lam`
  instead gives the net a 4th output (`add_lam_head`) so it **learns the per-position reroute strength**
  — the deployable lever. `add_lam_head` is identity-preserving (α/β/γ bit-unchanged, `lam` starts at
  `--lam-soft`), and the output stack stays one uniform `softplus(net + log(expm1(out_bias)))` so the
  `.cvcnav` just carries a 4th output (a risk `.cvcnav` still needs the C++ `coef_mlp`/drive update
  before it runs on the pure-C++ host). Smoke (grid 64, 40 steps, 2 seeds): learned `lam` trims exposure
  a further ~3% over the fixed dial (0.082→0.079) at equal reach.
- **Risk-exposure loss** — `w_risk · Σ risk(pose)` over the rollout, differentiable through the
  trajectory, penalizes dwelling in risk. Reported as the third `last_loss_terms` summand.

Only the **bicycle** rollout can learn this (the holonomic surrogate never sees material); the trainer
raises if a risk net is used without `material=`. A widened net is a torch research artifact — the
pure-C++ host's `coef_feats` does not build the risk column yet, so a risk `.cvcnav` is not deployable
until the coordinated `cvc::nav` update lands (and `--score` skips widened nets, since the Swarm drive
is base-5-feature).

Smoke measurement (grid 64, 40 steps, `lam_soft=0.4`, 2 seeds — indicative, not the full study):

| arm | risk exposure | reach |
|---|---|---|
| `w_risk=0` | 0.108 | 0.166 |
| `w_risk=8` | **0.082** (−24%) | 0.172 |

Reproduce with `coef_train.eval_risk_exposure(model, grid=64)` after training each arm. The exposure
drop is consistent across both seeds; the reach delta (+0.006) is within run-to-run noise, so the fair
read is "exposure down, reach not measurably changed" rather than a proven Pareto win — a proper
3-seed, longer-horizon study on a stable/GPU box (the BPTT graph is CPU-fragile) is the validation of
record, in the style of the `w_coll` operating-point table in `coef_train.train_bicycle`.

## Scorecard-driven curriculum (opt-in `--curriculum`)

Uniform start sampling spends most of the gradient on easy open corridors. `--curriculum`
(`grl_snam.curriculum.RegionCurriculum`) instead biases *where drives start* toward the regions the
current policy handles worst. It partitions the free space into a coarse bin grid and tracks a per-bin
difficulty EMA fed from **each batch's own outcomes** — goal shortfall + terrain-risk exposure, the
scorecard axes a campaign ranks on — with no extra eval pass. A uniform floor (`--curriculum-eps`) keeps
any bin from starving, a cap keeps one bin from dominating, and a uniform cold start prevents early noise
from locking it. Default off → uniform sampling, byte-identical.

Smoke (grid 64, 60 steps, `w_risk=6`, `lam_soft=0.4`, 6×6 bins): the learned start weights correlate
**+0.97** with per-bin terrain risk — riskier-than-median bins draw 0.030 of the mass vs 0.025 for the
rest (uniform 0.028) — so the curriculum concentrates training exactly on the hard, risky ground while
the floor keeps coverage everywhere. Magnitude scales with `--curriculum-eps`/bin count.

## Parity

`tests/test_scorecard.py` pins the reducer to the SAME hand-computed corpus as libcvc's
`nav_stats_test.cpp` and cvcdbg's `tests/nav_stats_test.cpp` (the shared-schema contract);
`tests/test_swarm.py` checks the `Swarm` collector against N serial `SdfNavigator`s to float32;
`tests/test_material_buckets.py` hand-computes the material buckets (the time/dist-per-id contract
mirrored from `nav_stats_test.cpp`), the id-raster sampler, and the byte-identical no-material default;
`tests/test_risk_lever.py` pins the risk feature + `w_risk` loss wiring; `tests/test_curriculum.py`
pins the difficulty reweighting (floor, cap, cold start, hard-bin bias).
