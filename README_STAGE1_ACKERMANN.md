# Stage-1 Ackermann + Synthetic LaserScan Pipeline

This patch adds a first sim-to-real bridge before Isaac Sim or real ROS 2 deployment.
It keeps the core GRL-SNAM idea as a local-frame policy, but replaces the old idealized velocity-frame rollout with:

```text
procedural navigation scene
    -> synthetic LaserScan
    -> scan adapter / pseudo-obstacle tokens
    -> learned policy command (v, omega)
    -> Ackermann projection with calibrated limits
    -> bicycle-model rollout
```

The intent is to train/test on the same abstraction that a later ROS 2 node will build from `/scan`, `/tf`, `/odom`, and a local goal.

## New files

```text
scripts/stage1_ackermann_scan.py   # simulator, scan renderer, adapter, Ackermann dynamics, rollout metrics
train_stage1_ackermann.py          # training entry point
eval_stage1_ackermann.py           # closed-loop evaluation and visualization
```

## What is modeled

### Observation

The policy does not receive clean simulator obstacle circles. It receives pseudo-obstacle tokens derived from synthetic LaserScan:

```text
obs_feats[j] = [cx, cy, r, ||c||, atan2(cy,cx), proximity]
goal_feats   = [gx, gy, ||g||, atan2(gy,gx), current_speed]
```

All quantities are in the robot/body frame.

### Command

The policy predicts raw `(v, omega)`. Before dynamics, the command is projected into an Ackermann-feasible envelope:

```text
|v| <= v_max
|delta| <= max_steering_angle
|omega / v| <= 1 / min_turning_radius
```

The same projection should be used later in the ROS 2 policy node before publishing `/cmd_vel` or `AckermannDriveStamped`.

### Dynamics

The rollout uses the bicycle/Ackermann kinematics:

```text
x_dot     = v cos(theta)
y_dot     = v sin(theta)
theta_dot = omega = v / L * tan(delta)
```

Closed-loop evaluation also applies simple acceleration and steering-rate limits.

## Quick smoke test

From the repo root:

```bash
python -m train_stage1_ackermann \
  --epochs 1 \
  --samples 256 \
  --val-samples 64 \
  --bs 32 \
  --workers 0 \
  --threads 1 \
  --outdir checkpoints/stage1_ackermann_smoke

python -m eval_stage1_ackermann \
  --ckpt checkpoints/stage1_ackermann_smoke/best.pt \
  --episodes 5 \
  --max-steps 40 \
  --plot-count 2 \
  --baseline-expert \
  --threads 1 \
  --outdir eval_stage1_smoke
```

## Longer training run

```bash
python -m train_stage1_ackermann \
  --epochs 30 \
  --samples 50000 \
  --val-samples 4096 \
  --bs 256 \
  --lr 3e-4 \
  --threads 1 \
  --outdir checkpoints/stage1_ackermann
```

Evaluate:

```bash
python -m eval_stage1_ackermann \
  --ckpt checkpoints/stage1_ackermann/best.pt \
  --episodes 100 \
  --max-steps 180 \
  --plot-count 8 \
  --baseline-expert \
  --threads 1 \
  --outdir eval_stage1_ackermann
```

## Calibration knobs to replace with real car values

These command-line options should be set from the UT car calibration/navigation values:

```bash
--wheelbase 0.324
--v-max 0.8
--max-steering-angle 0.396
--min-turning-radius 0.8
--dt 0.1
--robot-radius 0.23
```

The most important real-car value is `--min-turning-radius`, because it defines the curvature bound:

```text
kappa_max = 1 / min_turning_radius
```

## How this connects to ROS 2 later

The ROS 2 adapter should reproduce exactly this interface:

```text
/scan + /tf + /odom + local goal
    -> scan_to_obstacle_features(...)
    -> local_goal_features(...)
    -> Stage1PolicyNet(...)
    -> project_twist_np(...)
    -> publish /cmd_vel or AckermannDriveStamped
```

This lets us debug the learning/control interface before adding Isaac Sim or the real car.

## Visualizing policy vs. MPC

After training a Stage-1 checkpoint, compare the learned scan-token policy against a sampling-based Ackermann MPC baseline:

```bash
python -m compare_stage1_policy_mpc \
  --ckpt checkpoints/stage1_ackermann/best.pt \
  --episodes 20 \
  --plot-count 6 \
  --outdir eval_policy_vs_mpc \
  --mpc-horizon 12 \
  --mpc-samples 512
```

This writes:

- `eval_policy_vs_mpc/compare_ep_*.png`: trajectory overlay, distance-to-goal, clearance, and command traces.
- `eval_policy_vs_mpc/summary_policy_vs_mpc.json`: aggregate success/collision/final-distance/path-length metrics.
- `eval_policy_vs_mpc/episodes_policy.json` and `episodes_mpc.json`: per-episode scalar metrics.

The MPC baseline in `scripts/stage1_mpc.py` is geometry-aware: it uses the procedural obstacle circles and should be interpreted as a model-aware upper-bound baseline. The learned policy only consumes synthetic-LaserScan-derived local obstacle tokens.

## Recovery-aware Stage-1 revision

This version extends Stage-1 training for the quasi-static failure mode where the
policy slows to `v≈0` near a contact region and loses Ackermann turning authority.
The update adds:

```text
signed speed:       v ∈ [-v_reverse_max, v_max]
recovery samples:   near-front-obstacle states with stall hints
recovery context:   min/front clearance and normalized stall count
reverse expert:     reverse + steer-away labels under near-contact stall
MPC reverse:        reverse candidates in the sampling MPC baseline
metrics:            stall_steps and reverse_steps
```

The policy context is now:

```text
obs_feats[j] = [cx, cy, r, ||c||, atan2(cy,cx), proximity]
goal_feats   = [gx, gy, ||g||, atan2(gy,gx), current_speed,
                min_clearance, front_clearance, normalized_stall_count]
```

Train with recovery cases emphasized:

```bash
python -m train_stage1_ackermann \
  --epochs 30 \
  --samples 50000 \
  --val-samples 4096 \
  --bs 256 \
  --v-max 0.8 \
  --v-reverse-max 0.25 \
  --p-recovery 0.30 \
  --w-reverse 0.75 \
  --w-move 0.05 \
  --outdir checkpoints/stage1_ackermann_recovery
```

Compare with reverse-capable MPC:

```bash
python -m compare_stage1_policy_mpc \
  --ckpt checkpoints/stage1_ackermann_recovery/best.pt \
  --episodes 50 \
  --plot-count 8 \
  --mpc-samples 768 \
  --mpc-p-reverse 0.30 \
  --outdir eval_policy_vs_mpc_recovery
```

The online port-injection hook is implemented but disabled by default. Use it as
a fallback after the policy has been trained with recovery examples; otherwise it
can hide the fact that the base policy has not learned reverse/reaccelerate
maneuvers.

## Hybrid horizon policy upgrade

This patch upgrades the Stage-1 learner from a one-step reactive command head to a hybrid short-horizon policy. The model now predicts:

- an `H x 2` raw command sequence proposal `(v, omega)`,
- a maneuver-mode distribution over `STOP_SAFE`, `REVERSE_ALIGN`, `CREEP_FORWARD`, `FORWARD_ARC_LEFT`, `FORWARD_ARC_RIGHT`, and `GO_FORWARD`,
- a soft barrier schedule `beta[0:H]`,
- a scalar value/cost diagnostic.

The executed action is still the first command of the predicted sequence. Training, however, supervises the full sequence using a teacher rollout and an Ackermann rollout loss. This is intended to reduce locally myopic behavior where the one-step controller slows down near contact and cannot commit to a longer reverse/arc maneuver.

Recommended training command:

```bash
python -m train_stage1_ackermann \
  --epochs 30 \
  --samples 50000 \
  --val-samples 4096 \
  --bs 256 \
  --horizon 12 \
  --p-recovery 0.35 \
  --w-seq 1.0 \
  --w-mode 0.25 \
  --w-reverse 0.75 \
  --outdir checkpoints/stage1_ackermann_horizon
```

Evaluation:

```bash
python -m compare_stage1_policy_mpc \
  --ckpt checkpoints/stage1_ackermann_horizon/best.pt \
  --episodes 50 \
  --plot-count 8 \
  --mpc-horizon 12 \
  --mpc-samples 768 \
  --outdir eval_policy_vs_mpc_horizon
```

For local smoke testing, use much smaller values such as `--samples 512 --val-samples 128 --epochs 2 --horizon 8`. Such smoke tests only check the execution path; they are not meaningful performance evidence.

## Stage-1 pH/Hamiltonian coefficient training path

The preferred research path is now `train_stage1_ph_energy.py`.  It restores the original GRL-SNAM logic:

```text
synthetic LaserScan / local obstacle tokens / local goal context
    -> StagePHCoefNet
    -> Hamiltonian coefficients for goal, ICP barrier, damping, stage blend, memory bump, tangential port
    -> differentiable pH surrogate rollout
    -> Ackermann command adapter for closed-loop evaluation
```

This is different from the earlier direct horizon-action policy. The network predicts coefficients of the shaped Hamiltonian/potential rather than a conventional action sequence.

Train a small pH coefficient model:

```bash
python -m train_stage1_ph_energy \
  --epochs 30 \
  --samples 50000 \
  --val-samples 4096 \
  --bs 256 \
  --horizon 12 \
  --p-recovery 0.30 \
  --outdir checkpoints/stage1_ph_energy
```

Evaluate it:

```bash
python -m eval_stage1_ph_energy \
  --ckpt checkpoints/stage1_ph_energy/best.pt \
  --episodes 50 \
  --plot-count 8 \
  --outdir eval_stage1_ph_energy
```

Compare against the geometry-aware MPC upper-bound baseline:

```bash
python -m compare_stage1_ph_mpc \
  --ckpt checkpoints/stage1_ph_energy/best.pt \
  --episodes 50 \
  --plot-count 8 \
  --mpc-horizon 12 \
  --mpc-samples 768 \
  --outdir eval_ph_vs_mpc
```

The reported smoke tests in the assistant response only verify that the code path executes. They are not meaningful performance evidence; use the larger run above for performance claims.

### Multi-stage pH fine-tuning schedule

`train_stage1_ph_energy.py` now supports model initialization from a checkpoint:

```bash
python -m train_stage1_ph_energy \
  --resume checkpoints/stage1_ph_energy_bootstrap/best.pt \
  --epochs 20 \
  --horizon 16 \
  --w-cmd 1.0 \
  --w-roll 3.0 \
  --w-final 1.5 \
  --w-clear 3.0 \
  --w-pass 0.05 \
  --w-beta 0.001 \
  --outdir checkpoints/stage1_ph_energy_rollout
```

By default, `--resume` restores only the model parameters and starts a fresh optimizer. This is the recommended behavior for phase-to-phase fine-tuning because each phase changes the loss weights. Add `--resume-optimizer` only when continuing an interrupted run with the same training objective.

For the full recommended staged schedule, use the wrapper:

```bash
python -m train_stage1_ph_multistage \
  --samples 50000 \
  --val-samples 4096 \
  --bs 256 \
  --outdir checkpoints/stage1_ph_energy_multistage
```

The wrapper creates:

```text
checkpoints/stage1_ph_energy_multistage/bootstrap/best.pt
checkpoints/stage1_ph_energy_multistage/rollout/best.pt
checkpoints/stage1_ph_energy_multistage/safety/best.pt
```

The default phases are:

```text
bootstrap: w_cmd=5.0, w_roll=1.0, w_final=0.5, w_clear=1.0, w_pass=0.005, w_beta=0.0005, horizon=12
rollout:   w_cmd=1.0, w_roll=3.0, w_final=1.5, w_clear=3.0, w_pass=0.05,  w_beta=0.001,  horizon=16
safety:    w_cmd=0.5, w_roll=3.0, w_final=2.0, w_clear=8.0, w_pass=0.10,  w_beta=0.002,  horizon=16
```

Use the final safety checkpoint for evaluation:

```bash
python -m compare_stage1_ph_mpc \
  --ckpt checkpoints/stage1_ph_energy_multistage/safety/best.pt \
  --episodes 50 \
  --plot-count 8 \
  --mpc-horizon 12 \
  --mpc-samples 768 \
  --outdir eval_ph_vs_mpc_multistage
```

For a fast code-path test, reduce epochs and samples:

```bash
python -m train_stage1_ph_multistage \
  --bootstrap-epochs 1 \
  --rollout-epochs 1 \
  --safety-epochs 1 \
  --samples 64 \
  --val-samples 32 \
  --bs 16 \
  --bootstrap-horizon 4 \
  --rollout-horizon 4 \
  --safety-horizon 4 \
  --outdir /tmp/stage1_ph_multistage_smoke
```
