# General Short-Horizon Planning + Dual-Energy Tracking

This patch adds a scenario-independent planning layer above the dual-weight energy controller.

The training pipeline now separates two questions:

1. **What short-horizon motion is feasible?**
   A learned planner head predicts a local K-step motion segment from scan/obstacle/goal context.
   During offline training the head is supervised by saved teacher `pose_seq`; at deployment no MPC is called.

2. **How should the robot execute that motion safely?**
   The pH/dual-energy controller uses the predicted intermediate waypoint as the local energy target and allocates dual weights for goal tracking, barriers, tangent/flow, damping, and memory.

This is intended to replace scenario-specific terms such as parking-only contact losses.  The same supervision works for parking, front-trap recovery, narrow gates, terminal bays, and S-turns, because the common target is a short feasible motion segment.

## Main flags

```bash
--use-plan-goal              # use learned planner waypoint as local energy target
--w-plan 1.0                 # supervise predicted plan with teacher pose_seq
--w-plan-track 0.3           # train energy rollout to track the predicted plan
--plan-horizon 6             # number of local waypoints predicted
--plan-goal-index 3          # waypoint used as the energy target
--plan-supervision-steps 6   # pose_seq steps used for planner supervision
--plan-track-steps 4         # rollout steps used for controller tracking
```

`--w-rollout` remains available for direct energy-rollout-to-teacher supervision.  It is optional once `--w-plan` and `--w-plan-track` are enabled.

## Recommended training command

```bash
python -m train_dual_weight_energy \
  --data data/stage1_case_suite \
  --data-format stage1 \
  --stage1-split train \
  --stage1-cases narrow_gate,s_turn,front_trap,parallel_parking,terminal_bay,long_horizon_mixed \
  --stage1-direction-from cmd \
  --H 64 --W 64 --hat_d 3.5 \
  --training-mode auglag \
  --signed-speed \
  --use-plan-goal \
  --plan-horizon 6 \
  --plan-goal-index 3 \
  --w-plan 1.0 \
  --w-plan-track 0.3 \
  --w-rollout 0.1 \
  --rollout-supervision-steps 4 \
  --w-task 0.5 \
  --w-teacher 0.15 \
  --w-cmd 0.2 \
  --w-reverse 0.3 \
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
  --lambda-max 3.0 \
  --integrator-damping 0.40 \
  --momentum-clip 0.75 \
  --failure-replay-start-epoch 10 \
  --failure-replay-prob 0.05 \
  --failure-replay-loss-scale 0.2 \
  --outdir checkpoints/dual_energy_flow_planner_mixed
```

For debugging a single case, first disable replay and synthetic augmentation:

```bash
--failure-replay-prob 0.0 --parking-augment-batches 0
```

## Online evaluation

The checkpoint stores `use_plan_goal=True`, so online evaluation automatically uses the predicted planner waypoint.  MPC is optional only as a baseline method in the evaluator; the learned controller does not query MPC.

```bash
python -m eval_dual_weight_scenario_mpc \
  --ckpt checkpoints/dual_energy_flow_planner_mixed/best.pt \
  --scene-bank data/stage1_case_suite/parallel_parking/scenes_test.json \
  --episodes 20 \
  --modes fixed,forward,sensitivity \
  --signed-speed \
  --outdir eval_flow_planner_parallel_parking
```

## Logs to check

- `L_plan`: predicted short-horizon plan matches teacher pose sequence.
- `L_plan_track`: energy rollout follows the predicted plan.
- `plan_goal_norm`: predicted intermediate target is not collapsing to zero.
- `C_clear`: signed-clearance safety cost, not raw barrier exposure.
- `C_clear_raw`: raw rasterized barrier diagnostic; can be high in tight slots.
- `force_norm`: should not grow without progress.

