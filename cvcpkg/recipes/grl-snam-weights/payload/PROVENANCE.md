# grl-snam-weights — base GRL-SNAM nav coefficient provenance

Base GRL-SNAM navigation coefficients (no RF-comm forces) for the differentiable
SDF navigator's `CoefMLP` — predicts `(alpha, beta, gamma)`.

- `coef_sdf.cvcnav` — native `CVNV` blob for the C++ `cvc::nav` forward.
- `coef_sdf.pt` — torch checkpoint; load with `grl_snam` / `sdf_nav.CoefMLP`.

## How it was trained (v1.0.0)

- **Pipeline:** `grl-snam train <nav_sdf.npz>` (self-supervised SDF-coefficient
  training — geometry only, NO comm-force term) → `grl_snam.tools.coef_export`
  (checkpoint → `CVNV`).
- **Data:** the `austin_south` navigation SDF (`nav_sdf.npz`, 1024×1024 φ field
  over the public Austin geometry).
- **Schedule:** 200 self-supervised steps, seed 0 (goal-seeking + collision
  loss on ~400k drivable sample points).
- **Character:** an initial, deliberately short proof-of-pipeline run — enough
  that the coefficients have moved and the weights are usable, NOT a full
  training campaign. Regenerate from a longer run for production accuracy.

Bump `cvc_revision` in `recipe.yaml` whenever these bytes change.
