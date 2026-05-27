# Reproducible Stage-1 Ackermann Case Suite

This patch turns `generate_stage1_ackermann_rollouts.py` into a reproducible offline-training data generator and benchmark seed bank.

The generator still saves MPC-teacher supervised data:

```text
train.pt
val.pt
test.pt
```

but now it also saves exact scenario geometry:

```text
scenes_train.json
scenes_val.json
scenes_test.json
episodes_train.json
episodes_val.json
episodes_test.json
summary.json
```

The scene files are the key addition.  They let a learned policy be evaluated later on exactly the same held-out scenarios used to define the test split.

## Generate one named case

```bash
python -m generate_stage1_ackermann_rollouts \
  --root data/stage1_case_suite/parallel_parking \
  --case parallel_parking \
  --train-episodes 300 \
  --val-episodes 60 \
  --test-episodes 100 \
  --max-steps 220 \
  --v-reverse-max 0.35 \
  --mpc-p-reverse 0.55 \
  --mpc-samples 1536 \
  --seed 2026
```

Available cases:

```text
random
easy_open
dense_clutter
narrow_gate
s_turn
front_trap
parallel_parking
terminal_bay
noisy_scan
long_horizon_mixed
```

## Generate the full suite

```bash
python -m generate_stage1_case_suite \
  --root data/stage1_case_suite \
  --train-episodes 300 \
  --val-episodes 60 \
  --test-episodes 100 \
  --seed 2026
```

This creates one subdirectory per case and then writes:

```text
data/stage1_case_suite/manifest.json
data/stage1_case_suite/teacher_report.csv
data/stage1_case_suite/teacher_report.md
```

## Reuse exact scenes

To regenerate a split from a saved scene bank:

```bash
python -m generate_stage1_ackermann_rollouts \
  --root data/replay_parallel_parking \
  --case parallel_parking \
  --scene-bank-test data/stage1_case_suite/parallel_parking/scenes_test.json \
  --train-episodes 0 \
  --val-episodes 0 \
  --test-episodes 1
```

When a scene bank is provided, the number of episodes is inferred from the bank, not from `--test-episodes`.

## What is stored per sample

Each training sample contains the original MPC target fields plus reproducibility metadata:

```text
obs_feats
goal_feats
cmd
cmd_seq
pose
next_pose
pose_seq
beta_seq
mode
ranges
recovery_sample
episode_id
step
case_id
scene_seed
true_clearance
mpc_cost
```

## What is stored per episode

Each episode summary contains:

```text
success
collision
stopped_by_safety
steps
min_clearance
final_dist
final_heading_err
reverse_steps
reverse_episode
stall_steps
mode_switches
case
scene_seed
samples
```

For `parallel_parking`, the generator also fixes `start_heading` and `goal_heading`, so `final_heading_err` is meaningful.

## Suggested offline-training protocol

Use:

```text
train.pt  -> model fitting
val.pt    -> hyperparameter/model selection
test.pt   -> final offline imitation metrics
scenes_test.json -> closed-loop learned-policy rollout on exact test scenes
```

For reporting learned policies, compare at least:

```text
MPC teacher
fixed-energy policy
learned dual-energy policy without replay
learned dual-energy policy with failure replay
learned dual-energy policy with signed-speed/reverse support
```

Recommended metrics:

```text
success_rate ↑
collision_rate ↓
safety_stop_rate ↓
mean_final_dist ↓
p05_min_clearance ↑
reverse_episode_rate
mean_reverse_steps
mean_stall_steps ↓
mode_switches
final_heading_err ↓ for parking
```
