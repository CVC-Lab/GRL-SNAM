# Alternating short-horizon planner and local energy-controller training

This patch upgrades `train_dual_weight_energy.py` from a single deterministic local-goal head into a staged/alternating training pipeline:

1. **Planner phase** learns short-horizon sub-goals from offline teacher `pose_seq`.
2. **Energy/controller phase** learns a local energy/force field that tracks those sub-goals through differentiable time integration.
3. Optional **multi-candidate plans** let the planner represent multiple feasible local maneuvers without hand-coded scenario losses.

No MPC is used online. MPC/teacher rollouts are used only to create offline supervision targets.

## New model behavior

`DualWeightEnergyNet` now supports:

```text
plan_candidates = M
plan_horizon    = K
```

The planner emits:

```text
plan_seq_candidates: [B, M, K, 2]
plan_scores:         [B, M]
plan_best_idx:       [B]
plan_seq:            [B, K, 2]
plan_goal:           [B, 2]
```

The selected candidate is chosen by a generic feasibility/progress score:

```text
score = goal-distance + signed-clearance + curvature + path-length
```

This is deliberately not parking-specific.

## Stage A: planner-only pretraining

```bash
python -m train_dual_weight_energy \
  --data data/stage1_case_suite \
  --data-format stage1 \
  --stage1-split train \
  --stage1-cases narrow_gate,s_turn,front_trap,parallel_parking,terminal_bay,long_horizon_mixed \
  --stage1-direction-from cmd \
  --H 64 --W 64 --hat_d 3.5 \
  --use-plan-goal \
  --planner-only \
  --plan-candidates 8 \
  --plan-horizon 6 \
  --plan-goal-index 3 \
  --plan-target-mode arclength \
  --plan-arc-step 0.25 \
  --w-plan 1.0 \
  --w-plan-clear 0.05 \
  --w-plan-dyn 0.05 \
  --w-plan-diversity 0.02 \
  --epochs 50 \
  --bs 16 \
  --outdir checkpoints/planner_only_mixed
```

## Stage B: controller training against teacher sub-goals

```bash
python -m train_dual_weight_energy \
  --resume checkpoints/planner_only_mixed/best.pt \
  --data data/stage1_case_suite \
  --data-format stage1 \
  --stage1-split train \
  --stage1-cases narrow_gate,s_turn,front_trap,parallel_parking,terminal_bay,long_horizon_mixed \
  --stage1-direction-from cmd \
  --H 64 --W 64 --hat_d 3.5 \
  --training-mode auglag \
  --signed-speed \
  --use-plan-goal \
  --plan-candidates 8 \
  --plan-horizon 6 \
  --plan-goal-index 3 \
  --energy-plan-source teacher \
  --w-plan 0.2 \
  --w-plan-track 0.5 \
  --w-rollout 0.3 \
  --w-cmd 0.2 \
  --clearance-mode auto \
  --epochs 50 \
  --bs 16 \
  --outdir checkpoints/controller_teacher_plan
```

## Stage C: alternating planner / energy training

```bash
python -m train_dual_weight_energy \
  --resume checkpoints/controller_teacher_plan/best.pt \
  --data data/stage1_case_suite \
  --data-format stage1 \
  --stage1-split train \
  --stage1-cases narrow_gate,s_turn,front_trap,parallel_parking,terminal_bay,long_horizon_mixed \
  --stage1-direction-from cmd \
  --H 64 --W 64 --hat_d 3.5 \
  --training-mode auglag \
  --signed-speed \
  --use-plan-goal \
  --plan-candidates 8 \
  --alternate-plan-energy \
  --alt-plan-steps 1 \
  --alt-energy-steps 1 \
  --energy-plan-source mixed \
  --teacher-plan-prob 0.3 \
  --w-plan 0.5 \
  --w-plan-clear 0.05 \
  --w-plan-dyn 0.05 \
  --w-plan-diversity 0.02 \
  --w-plan-track 0.3 \
  --w-rollout 0.2 \
  --w-cmd 0.2 \
  --clearance-mode auto \
  --failure-replay-start-epoch 10 \
  --failure-replay-prob 0.05 \
  --epochs 80 \
  --bs 16 \
  --outdir checkpoints/coupled_plan_energy
```

## Important diagnostics

Watch these logs:

```text
plan_phase/L_plan
plan_phase/L_plan_clear
plan_phase/L_plan_dyn
plan_phase/L_plan_diversity
energy_phase/L_plan_track
energy_phase/L_rollout
energy_phase/C_clear
energy_phase/plan_score
```

A good run should show planner imitation decreasing first, then local integration tracking errors decreasing. If the planner looks good but `L_plan_track` remains high, the issue is the energy/controller rather than sub-goal placement.
