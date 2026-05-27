# Binding reproducible Stage-1 case-suite data into training/evaluation

This repo supports two data schemas:

1. **NavMax local-patch data** (`nav_stagewise_hyperring`, `manifest.json`, `episodes/*.pt`) used by the original dual-energy trainer.
2. **Stage-1 Ackermann rollout data** (`train.pt`, `val.pt`, `test.pt`, `scenes_*.json`) generated from MPC-teacher rollouts.

`train_dual_weight_energy.py` now accepts both.  Stage-1 data is converted through
`scripts/stage1_energy_adapter.py`, which reconstructs a NavMax-style local energy
patch from each saved pose and exact scene record.

## 1. Generate a reproducible case suite

```bash
python -m generate_stage1_case_suite \
  --root data/stage1_case_suite \
  --train-episodes 300 \
  --val-episodes 60 \
  --test-episodes 100 \
  --seed 2026
```

Each case directory contains:

```text
train.pt / val.pt / test.pt
scenes_train.json / scenes_val.json / scenes_test.json
episodes_train.json / episodes_val.json / episodes_test.json
summary.json
```

`train.pt` is for offline fitting, `val.pt` for model selection, and
`scenes_test.json` is the exact held-out closed-loop scenario bank.

## 2. Train on one generated case

```bash
python -m train_dual_weight_energy \
  --data data/stage1_case_suite/parallel_parking \
  --data-format stage1 \
  --stage1-split train \
  --stage1-direction-from cmd \
  --H 64 --W 64 --hat_d 3.5 \
  --training-mode auglag \
  --signed-speed \
  --w-reverse 1.0 \
  --epochs 80 \
  --bs 16 \
  --teacher-candidates 32 \
  --clear-rollout-steps 2 \
  --mu-max 50.0 \
  --failure-replay-capacity 512 \
  --failure-replay-prob 0.35 \
  --failure-replay-topk 4 \
  --parking-augment-batches 2 \
  --parking-loss-scale 0.25 \
  --outdir checkpoints/dual_energy_stage1_parallel_parking
```

## 3. Train on multiple cases from the suite

```bash
python -m train_dual_weight_energy \
  --data data/stage1_case_suite \
  --data-format stage1 \
  --stage1-split train \
  --stage1-cases narrow_gate,s_turn,front_trap,parallel_parking,terminal_bay \
  --stage1-direction-from cmd \
  --H 64 --W 64 --hat_d 3.5 \
  --training-mode auglag \
  --signed-speed \
  --w-reverse 1.0 \
  --epochs 100 \
  --bs 16 \
  --teacher-candidates 32 \
  --clear-rollout-steps 2 \
  --failure-replay-capacity 1024 \
  --failure-replay-prob 0.35 \
  --parking-augment-batches 2 \
  --outdir checkpoints/dual_energy_stage1_mixed
```

The adapter preserves ordered episode/snapshot traversal, and it assigns unique
episode offsets across cases so recurrent dual state does not leak between cases.

## 4. Quick debug run

```bash
python -m train_dual_weight_energy \
  --data data/stage1_case_suite/parallel_parking \
  --data-format stage1 \
  --stage1-split train \
  --stage1-max-samples-per-case 256 \
  --H 32 --W 32 --hat_d 3.5 \
  --training-mode auglag \
  --signed-speed \
  --epochs 2 \
  --bs 8 \
  --teacher-candidates 4 \
  --outdir checkpoints/debug_stage1_binding
```

## 5. Closed-loop evaluation on exact saved scenes

Evaluate a trained checkpoint on the identical held-out scene bank:

```bash
python -m eval_dual_weight_scenario_mpc \
  --ckpt checkpoints/dual_energy_stage1_mixed/best.pt \
  --scene-bank data/stage1_case_suite/parallel_parking/scenes_test.json \
  --episodes 100 \
  --max-steps 220 \
  --H 64 --W 64 --hat-d 3.5 \
  --modes fixed,forward,sensitivity,mpc \
  --signed-speed \
  --mpc-p-reverse 0.55 \
  --outdir eval_stage1_parallel_parking_exact
```

For a multi-case scene bank, use `--case-filter parallel_parking` or another
comma-separated subset.

## 6. Important interpretation

The Stage-1 adapter uses exact scene geometry from `scenes_*.json` whenever it is
available.  It does not rely on the lidar token approximation for obstacle maps
unless the scene bank is missing.  Therefore, the recommended workflow is:

```text
generate train/val/test + scenes_*.json
train from train.pt through Stage1EnergyPatchDataset
select checkpoint using val.pt or short val scene-bank rollouts
report closed-loop performance on scenes_test.json
```

This makes offline fitting and closed-loop evaluation reproducible under the
same named scenarios and random seeds.

## Stage-1 parking stability fix: signed clearance instead of raw barrier AL

For Stage-1 parking data, the raw rasterized obstacle barrier can be large even
for valid states because a parking slot is intentionally bounded by nearby cars
and curbs.  The trainer therefore now defaults to

```bash
--clearance-mode auto
```

which uses `signed` clearance for `--data-format stage1` and keeps the older raw
barrier exposure for NavMax local patches.  In signed mode, the augmented
Lagrangian clearance constraint is based on the predicted rollout's minimum
local signed distance to obstacles:

```text
C_clear = clip( relu((d_safe - d_min) / tau)^2, max=clear_cost_clip )
```

The raw barrier value is still logged as `C_clear_raw` and the old current-origin
barrier diagnostic remains `C_clear_current`.  The signed physical distance is
logged as `min_clear_signed`.

Recommended stable Stage-1 parking command:

```bash
python -m train_dual_weight_energy \
  --data data/stage1_case_suite/parallel_parking \
  --data-format stage1 \
  --stage1-split train \
  --stage1-direction-from cmd \
  --H 64 --W 64 --hat_d 3.5 \
  --training-mode auglag \
  --signed-speed \
  --clearance-mode auto \
  --clear-safe-margin 0.05 \
  --clear-tau 0.10 \
  --clear-cost-clip 5.0 \
  --eps-clear 0.05 \
  --eps-act 0.5 \
  --rho-clear 0.05 \
  --rho-act 0.05 \
  --mu-max 5.0 \
  --mu-step-clip 0.05 \
  --failure-replay-start-epoch 5 \
  --failure-replay-prob 0.0 \
  --parking-augment-start-epoch 5 \
  --parking-augment-batches 0 \
  --lambda-max 3.0 \
  --integrator-damping 0.40 \
  --momentum-clip 0.75 \
  --outdir checkpoints/debug_parallel_parking_signed_clearance
```

Once the base signed-speed policy is stable, gradually enable replay:

```bash
--failure-replay-capacity 512 \
--failure-replay-start-epoch 5 \
--failure-replay-prob 0.05 \
--failure-replay-loss-scale 0.2
```

and then optional synthetic parking augmentation:

```bash
--parking-augment-start-epoch 10 \
--parking-augment-batches 1 \
--parking-loss-scale 0.05
```

Synthetic parking augmentation is supervised-only by default; pass
`--parking-use-auglag` only after the signed-clearance and actuation constraints
are calibrated.

## Contact-aware parking supervision patch

For reverse/parallel-parking cases, the Stage-1 adapter now preserves MPC teacher
sequences in the training batch:

- `cmd`: first teacher `[v, omega]` command.
- `cmd_seq`: teacher command horizon.
- `pose_seq_xy`: future teacher positions expressed in the current local frame.
- `waypoint_xy`: a short-horizon teacher waypoint.

The trainer can use these fields to avoid learning a purely repulsive barrier
field that bounces out of the slot.  Recommended parking-specific losses:

```bash
--w-cmd 0.5 \
--w-waypoint 0.3 \
--w-rollout 0.2 \
--rollout-supervision-steps 4 \
--w-contact 0.2 \
--contact-band 0.35 \
--contact-normal-cap 0.30
```

`--w-contact` activates a near-contact sliding loss: when the vehicle is safely
near obstacles, the model is encouraged to align the tangential component of its
force with the MPC teacher direction while limiting excessive outward normal
force.  This is intended for parking slots, terminal bays, and curb-following
where nearby obstacles define a safe contact corridor rather than an absolute
repulsive region.
