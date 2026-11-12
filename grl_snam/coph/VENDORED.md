# Vendored: CoPH-Stagewise

`grl_snam.coph` is a copy of the CoPH-Stagewise package from CoPH-Sense, pinned at one commit.
It holds the method's core and the two demos the CoPH roadmap builds on: crossing traverses and
graded hazards. The copy is kept in sync with upstream by re-vendoring and by the goldens
(libcvc `docs/roadmap/NAV-COPH-ROADMAP.md`, §3.1 and §3.4).

| | |
|---|---|
| Upstream | [SetasAditya/CoPH-Sense](https://github.com/SetasAditya/CoPH-Sense), `CoPH-Stagewise/coph_stagewise/` |
| Commit | `c1f1ff34d0db98c1ac61240778266d6e49b9c853` (2026-11-07) |
| License | MIT, like GRL-SNAM. The authors confirmed this on 2026-11-09. Upstream has no LICENSE file and no per-file copyright lines at this commit, so there are none to carry over. |
| Dependencies | torch, numpy and matplotlib. Upstream's scipy and Pillow are used only by modules that stay out. |

The copied files keep upstream's formatting and names, so that re-syncing is a mechanical diff.
black and ruff skip them (`pyproject.toml` lists them). `__init__.py` and `results.py` are
GRL-SNAM's own code and are linted.

## What was copied

**Verbatim** (byte-identical to upstream): `hazard_oracle`, `learner`, `maze_geometry`,
`maze_learning`, `model`, `navigation`, `records`, `replay_budget`, `route_oracle`, `scenarios`,
`shapes`, `sharing`, `stage_prediction`, `terrain_warning`, `world`.

**Changed** (whole module, with the local changes below): `cooperative`, `route_field`,
`crossing_demo`, `hazard_demo`.

**Trimmed** (a subset of upstream's top-level definitions, each one verbatim, under upstream's
module name and docstring, with a `# grl_snam.coph: trimmed ...` note and only the imports those
definitions need):

| Module | Kept | Why it is here |
|---|---|---|
| `diagnostics` | `snapshot`; `InstrumentedLearner`'s hooks as the `Instrumented` mixin | the runner records `snapshot` at every update |
| `cooperative_evaluation` | `SCENES`, `specification`, `CooperativeWorld` | the demos' worlds derive from `CooperativeWorld`; the tests use the scenes |
| `staggered_demo` | `sense`, `packet_effect`, `execute` | `execute` is the two-agent runner both demos call |
| `maze_family` | `NetworkWorld`, `NetworkLearner`, `learner_for` | the hazard demo's world and learner base |
| `evaluation` | `load`, `dump` | episode I/O |
| `cli` | `rollout`, `source_hashes`, `archive_sources` | the single-agent reference loop; manifest provenance |

**New** (GRL-SNAM code): `__init__.py` (the `UPSTREAM` pin) and `results.py`, which writes
`results.json`, `rows.json` and `REPORT.md` from a run's episodes. Its `doors_of`, `door_name`,
`path_length`, `door_events` and `metrics` are copied from upstream's `crossing_visuals.py`, so the
path, detour and saving arithmetic is upstream's.

This is 25 of upstream's 120 modules. It is more than the roadmap's estimate of about 15 because
the demos call into `staggered_demo`, `cooperative_evaluation`, `maze_family`, `terrain_warning`,
`sharing` and `diagnostics` at run time. Of those, only the definitions the demos reach were kept.

## What was left out

Everything else in `coph_stagewise/`, in four groups:
- reporting and figures: `crossing_visuals`, `maze_visuals`, `visualize`, `terrain_warning_visuals`,
  `campaign_landscape_visuals` and every `*_report*` module;
- campaigns: `campaign*`, `physical_*`, `completion_*`, `routed_validation`;
- diagnostics and receiver experiments: `decision_*`, `attribution*`, `receiver_*` (including
  `receiver_fitting` and `receiver_summary`), `reconstruction_*`, `localized_receiver*`,
  `packet_feedback`, `model_prediction`, `continuations`, `scheduling`, `solver_diagnosis` and
  `decision_sensitive` (the only scipy users), among others;
- the other demos and scene generators: `maze_demo`, the `maze_family` generator,
  `staggered_demo`'s scenes, `terrain_warning_demo`, `complex_environments`, plus
  `additive_routing` and `route_decision`.

Some verbatim modules still name left-out modules inside branches that the demos' profile and
the default `Config` never reach. If one of those experiments is switched on, the branch raises
`ModuleNotFoundError`:
- `maze_learning` imports `additive_routing` when `route_integral_mode == 'additive_v1'`, and
  `route_decision` when `route_candidate_schema == 1`;
- `route_field.validate_context` imports `additive_routing` and `localized_receiver` (nothing in
  the package calls it);
- `cooperative.CooperativeLearner._field_fit` imports `receiver_fitting` when a
  `receiver_diagnostic_variant` attribute is set.

## Local changes

Every change carries a `grl_snam.coph:` comment in the code.

1. **`cooperative`: `CooperativeLearner` no longer subclasses `diagnostics.InstrumentedLearner`.**
   Its base is `learner.Learner`, so the core no longer loads the diagnostics suite. The hooks
   became the `diagnostics.Instrumented` mixin. An evaluation that wants them composes it first,
   for example `class X(Instrumented, CooperativeLearner)`; `tests/coph/test_navigation.py` does.
   The hooks only record replay calls, proposals and the raw damping proposal (with
   `current_budget=None`), and nothing in the demos reads them, so results are unchanged.
   - Because the mixin now sits above the learner, `Instrumented.update` calls
     `super().update(x)` when `force_invalid` is false: the maze and terrain learners'
     `update()` takes `x` only.
2. **`cooperative`: receiver modes other than legacy are refused.** `receiver_fit_mode` and
   `receiver_validation_mode` other than `'legacy'` raise `NotImplementedError`, because
   `receiver_fitting` and `localized_receiver` are not vendored. Every non-default solver,
   search and localization mode already requires `precision_bounded` fitting upstream, so this
   one check covers all of them.
3. **`cooperative`: `_receiver_acceptance` short-circuits.** It returns `(True, None, None)` when
   there is no receiver fit diagnostic, which is exactly what `receiver_fitting.acceptance` returns
   in that case. Legacy receivers always take this path, and `Learner.update` calls it on every
   update.
4. **`cooperative`: `evaluate_packet` removed.** It wrapped `packet_feedback.evaluate_receiver`, an
   evaluation module that is not vendored.
5. **`route_field`: `terrain_warning` is imported lazily.** `values()` imports `terrain_values`
   when called instead of at module load, so loading the core does not load the cooperative stack.
6. **`crossing_demo.main` ends with `results.crossing(out)`.** This writes `results.json`,
   `rows.json` and `REPORT.md`, and returns the summary. Upstream calls
   `crossing_visuals.report`, which writes the same tables plus figures and a GIF.
   `--render-only` rewrites those files from existing episodes.
7. **`hazard_demo`: `render()` removed, `main()` extended.**
   - `render()` is gone. It drew `crossing_visuals` GIFs and wrote `REPORT.md`; `results.hazard`
     writes `results.json` and the same excess-cost table.
   - `main()` takes `--steps` (default 1800, upstream's) and `--seeds` (default `[79000]`,
     upstream's). Each scene manifest lists one spec per seed, and the jobs run over seeds × arms.
   - `--render` calls `results.hazard`, and a run ends by calling it.
8. **`cli`: `source_hashes` and `archive_sources` cover this package.** They hash and archive
   `grl_snam/coph/*.py`. Upstream covers the whole CoPH-Stagewise tree. Manifest
   `source_hashes` keys are therefore bare file names.

The entry point is `grl-snam coph crossing|hazard --out DIR [--seeds N] [--steps N] [--workers N]
[--summary-only]` in `grl_snam/cli.py`. `--seeds N` runs the first N seeds counted from the
demo's first seed: 76000 for crossing (default 2) and 79000 for hazard (default 1). The command
sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 unless they are already
set, as upstream's reproduction recipe does.

## Tests

`tests/coph/` holds the upstream tests that cover the copied modules. Each has `coph_stagewise`
renamed to `grl_snam.coph`.

| File | Change |
|---|---|
| `test_core`, `test_route_oracle`, `test_sharing` | none |
| `test_crossing` | `door_name` comes from `results` (copied from `crossing_visuals`) |
| `test_cooperative` | dropped `test_audits_do_not_change_learner_or_work_counters` and `test_cost_components_and_budget` (the cooperative evaluation suite); `test_rollback_and_audit_isolation` became `test_rollback`, without its `audit_receiver` half |
| `test_navigation` | dropped `test_summary_context_and_braking` (`receiver_summary`); the update-budget test composes `Instrumented` for `replay_calls` |
| `test_progress` | dropped `test_summary_context_and_braking`; imports `.test_navigation` |
| `test_stage_prediction` | dropped the four tests on `receiver_summary`, `model_prediction` and `stagewise_diagnostics`; `test_no_evaluator_imports` checks `stage_prediction` and `replay_budget` |
| `test_stage_blending` | dropped `test_logged_landscape_reproduces_blended_and_localized_force` (`maze_visuals`) |
| `test_maze` | `maze_demo`'s `specification`, `MazeWorld` and `configuration` copied in as fixtures; the landscape test became `test_exact_controller_gradients`, without its `maze_visuals` checks |
| `test_staggered_demo` | `complex_environments.scene` and `staggered_demo.specification` copied in as fixtures |
| `test_terrain_warning` | `terrain_warning_demo.specification` copied in as a fixture |

The other 31 upstream test files test modules that are not vendored. That includes
`test_maze_family`, which tests the maze generator.

`tests/test_coph.py` is GRL-SNAM's own test. It covers the CLI, short-horizon smokes of both
demos pinned to numbers that upstream's code produces at the same horizon, and the two
untanglings.

## The numbers gate

At this commit the vendored demos must reproduce upstream's per-arm results: exactly on the same
OS and torch version; otherwise with the same goal and collision counts, a reciprocal saving
within 0.5 m and a worst-case excess within 0.5. The full demos take 20 to 30 minutes on a
many-core box, so PR CI runs the ported tests and the short smokes. The full gate runs by hand,
and the roadmap proposes it as a nightly.

```bash
grl-snam coph crossing --out gate/crossing --workers 8
grl-snam coph hazard --out gate/hazard --workers 12
```

**Measured 2026-11-12** at this commit, on Windows 11 with cvcpkg's Python 3.12.10, torch 2.8.0
(CPU) and numpy 2.4.6. That is the stack upstream's own reproduction ran on (2026-11-09,
`CoPH-Stagewise/results/{crossing,hazard}_reproduction/`).
- **Crossing** (23.6 min, 8 workers). 16/16 goals and 0 collisions. On both seeds the one-way
  saving is 13.4 m and the reciprocal saving is 27.1 m (oracle 14.0 and 28.0). Finish times are
  132.7 / 134.4 s with communication off and 92.0 / 92.5–93.0 s reciprocal.
- **Hazard** (17.8 min, 12 workers). 64/64 goals and 0 collisions. The carrier's worst-case
  excess cost:

  | none | field | occupancy_0p1 | occupancy_0p25 | occupancy_1 | decision | raw | no_learning |
  |---|---|---|---|---|---|---|---|
  | 68.1 | 4.6 | 12.3 | 14.9 | 25.3 | 25.3 | 4.6 | 4.6 |

- **Exact.** All 40 episode files (8 crossing, 32 hazard) equal upstream's reproduction field
  for field, apart from wall-clock timings. The crossing `rows.json` and its report tables match
  upstream's line for line.

The box was also running other builds, so the times are not benchmarks.

**Packet byte counts depend on the platform.** A packet's size is its compact JSON (decision
D9). The first export bootstraps from a single time block, so its 8 ensemble members are
identical and its covariance is pure round-off. Whether `np.cov` returns exact zeros (`0.0`, 3
characters) or ~1e-35 residues (about 22 characters) depends on numpy's summation order. On this
box the same 8 rows give 112 or 132 exact zeros, 380 bytes apart, depending only on that order.

Linux CI (cp311–cp313) measured the 50-step crossing smoke's `one_way_01` at 80,817–80,834
bytes, against 79,488 here, with the paths equal to 1e-6. So `tests/test_coph.py` pins the
packets' structure, not their bytes. For the goldens (roadmap G9, where `bytes()` is BIT), either
inject the covariance or avoid single-block exports.

## Re-syncing

1. Check out CoPH-Sense at the new commit. Diff the verbatim modules against it and copy them.
2. Re-extract the trimmed modules' kept definitions.
3. Re-apply the local changes above, then update the commit at the top of this file and in
   `__init__.py`.
4. Re-port any changed tests. Run `pytest tests/coph tests/test_coph.py` and the full gate, and
   compare the gate with upstream's own run of both demos at the new commit.
