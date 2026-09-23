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

## Consuming it — eval and training

- **Eval CLI:** `python -m grl_snam.tools.scorecard_eval [--checkpoint run/coef.pt] [--scenes N] [--json card.json]`
  runs a CoefMLP over a city scene corpus with the `Swarm` collector and prints/writes the
  `NavScorecard`.
- **Training tie-in:** `python -m grl_snam.tools.coef_train --score [--score-scenes N] [--score-json …]`
  scores the freshly-trained checkpoint with the base scorecard (arrival/economy/safety) instead of
  raw reach — the single fitness signal a campaign ranks on.

## Parity

`tests/test_scorecard.py` pins the reducer to the SAME hand-computed corpus as libcvc's
`nav_stats_test.cpp` and cvcdbg's `tests/nav_stats_test.cpp` (the shared-schema contract);
`tests/test_swarm.py` checks the `Swarm` collector against N serial `SdfNavigator`s to float32.
