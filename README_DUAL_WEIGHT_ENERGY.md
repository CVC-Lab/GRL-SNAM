# Dual-weight energy reshaping patch

This patch adds a stateful dual-weight architecture for multi-stage pH/energy navigation.  The learned state is no longer a one-shot potential coefficient vector.  It is a slow dual vector

```text
lambda = [goal, barrier, tangent, flow, boundary, memory, stage, damping]
```

that is updated online from local obstacle tokens, barrier maps, local goal features, stage changes, progress, and stall indicators.

## Files

```text
scripts/dual_weight_energy_nav.py      # model, dual update law, field composition, projection helpers
train_dual_weight_energy.py            # trainer for NavMaximinLocalPatches
scripts/failure_replay.py              # medium-level failure replay buffer
scripts/parking_aug.py                 # synthetic parallel-parking/reverse curriculum patches
scripts/energy_data_navmax.py          # copied dataset loader
scripts/stagewise_dataset.py           # copied dataset generator
scripts/spline_stagewise6.py           # copied stagewise/ICP reference implementation
```

## Why this helps quasi-static failures

The previous static coefficient model could increase the obstacle barrier and then freeze.  The new model updates barrier, tangent, flow, memory, and stage weights together.  Near obstacles or during stalls, the analytic scaffold increases the tangential and flow weights in addition to the barrier weight, while the neural update learns the context-dependent correction.

## Training

The dual state is sequential, so do **not** shuffle the dataset during training.  The trainer uses `shuffle=False` by default.

```bash
python -m train_dual_weight_energy \
  --data nav_stagewise_hyperring \
  --epochs 50 \
  --bs 16 \
  --H 64 \
  --W 64 \
  --outdir checkpoints/dual_weight_energy
```

Important losses:

```text
--w-align          align induced field with recorded desired direction
--w-dual-target    weakly regularize dual weights toward proximity/stall/stage triggers
--w-dual-tv        discourage violent dual jumps
--w-anti-static    prevent zero vector field at the vehicle origin
--force-floor      minimum local field norm used by anti-static loss
```

A useful quasi-static-focused run is:

```bash
python -m train_dual_weight_energy \
  --data nav_stagewise_hyperring \
  --epochs 80 \
  --bs 16 \
  --w-align 1.0 \
  --w-dual-target 0.10 \
  --w-dual-tv 0.005 \
  --w-anti-static 0.25 \
  --force-floor 0.08 \
  --lambda-step 0.16 \
  --lambda-max 8.0 \
  --outdir checkpoints/dual_weight_energy_quasistatic
```

## Online use

Use `DualWeightOnlineController` and call `reset()` at the start of every episode.  The controller preserves `lambda_state` across time, raises tangential/flow/memory weights under proximity/stall, and applies a hard Ackermann box/rate projection if you request commands.

```python
from scripts.dual_weight_energy_nav import DualWeightEnergyNet, DualWeightOnlineController

model = DualWeightEnergyNet(H=64, W=64)
model.load_state_dict(torch.load("checkpoints/dual_weight_energy/best.pt")["model_state_dict"])
model.eval()
controller = DualWeightOnlineController(model)
controller.reset()

out = controller.step(batch)
F0 = out["F0"]          # local reshaped pH/energy field at vehicle origin
u  = out["u"]           # hard-projected Ackermann-like (v, steering) command
lam = out["lambda"]     # online dual weights
```

## Constraint handling

The model contains a differentiable soft input barrier helper, but deployment should still use the hard `project_ackermann_box_rate` shield.  Barrier potentials shape behavior; projection enforces speed, steering, acceleration, and steering-rate feasibility.

## Sensitivity / adjoint-style online dual update

This revision adds a second online adaptation mode for the dual weights.  The original controller mode was a forward recurrent update:

```text
lambda_{t+1} = Proj(lambda_t + learned_delta + analytic_trigger)
```

The new sensitivity mode first applies that forward update, then freezes the network parameters and corrects only the dual state by differentiating a short local rollout objective:

```text
lambda_{t+1} = Proj(lambda_forward - eta * dJ/dlambda)
```

The short-horizon objective contains terminal goal progress, path progress, fixed barrier-map exposure, force/goal alignment, actuator projection defect, and anti-static force floor terms.  This means feedback can directly alter the next energy field through the dual weights without changing the trained neural parameters online.

### Python API

```python
from scripts.dual_weight_energy_nav import (
    DualWeightOnlineController,
    SensitivityUpdateConfig,
)

controller = DualWeightOnlineController(
    model,
    sensitivity_cfg=SensitivityUpdateConfig(
        eta=0.08,
        horizon=8,
        w_goal=1.0,
        w_barrier=0.25,
        w_align=0.25,
        w_act=0.05,
        w_stall=0.25,
    ),
)

# no dual adaptation: fixed lambda prior
out_fixed = controller.step(batch, adaptation="fixed")

# learned + analytic recurrent dual update
out_forward = controller.step(batch, adaptation="forward")

# learned + analytic update, then projected sensitivity correction
out_sens = controller.step(batch, adaptation="sensitivity")
```

Returned diagnostics in sensitivity mode include:

```text
sens_loss
sens_terminal
sens_barrier
sens_act_defect
sens_grad
sens_grad_norm
lambda_before_sens
lambda_after_sens
q_seq
u_raw_seq
u_proj_seq
F_seq
```

### Evaluation and visualization

Compare fixed, forward-dual, and sensitivity-dual adaptation on an ordered dataset:

```bash
python -m eval_dual_weight_sensitivity \
  --data nav_stagewise_hyperring \
  --ckpt checkpoints/dual_weight_energy/best.pt \
  --steps 120 \
  --H 64 \
  --W 64 \
  --outdir eval_dual_weight_sensitivity
```

The evaluator writes:

```text
eval_dual_weight_sensitivity/summary.json
eval_dual_weight_sensitivity/metrics.json
eval_dual_weight_sensitivity/lambda_trace_fixed.npy
eval_dual_weight_sensitivity/lambda_trace_forward.npy
eval_dual_weight_sensitivity/lambda_trace_sensitivity.npy
eval_dual_weight_sensitivity/lambda_traces_fixed.png
eval_dual_weight_sensitivity/lambda_traces_forward.png
eval_dual_weight_sensitivity/lambda_traces_sensitivity.png
eval_dual_weight_sensitivity/method_metrics.png
eval_dual_weight_sensitivity/energy_fields_step_*.png
```

The `energy_fields_step_*.png` files visualize how the potential `U` and the force field `F` differ under the three adaptation modes at the same dataset snapshot.

Useful sensitivity knobs:

```bash
--sens-eta 0.08
--sens-horizon 8
--sens-w-goal 1.0
--sens-w-path 0.15
--sens-w-barrier 0.25
--sens-w-align 0.25
--sens-w-act 0.05
--sens-w-stall 0.25
--force-floor 0.06
```

If the sensitivity update oscillates, reduce `--sens-eta` or `--sens-grad-clip`.  If it still freezes in quasi-static cases, increase `--sens-w-stall` and `--sens-w-align`.  If it cuts too close to obstacles, increase `--sens-w-barrier`.

## Closed-loop scenario evaluation vs MPC

`eval_dual_weight_scenario_mpc.py` samples procedural navigation scenarios and compares:

```text
fixed        fixed/default dual weights
forward      learned + analytic online dual update
sensitivity  forward update plus short-horizon sensitivity correction
mpc          geometry-aware sampling MPC upper-bound baseline
```

Run after training a dual-weight checkpoint:

```bash
python -m eval_dual_weight_scenario_mpc \
  --ckpt checkpoints/dual_weight_energy/best.pt \
  --scenario front_trap \
  --episodes 5 \
  --max-steps 120 \
  --H 64 \
  --W 64 \
  --hat-d 3.5 \
  --mpc-horizon 12 \
  --mpc-samples 512 \
  --outdir eval_dual_weight_scenario_mpc
```

Available scenario presets:

```text
front_trap
narrow_gate
s_turn
parallel_parking
random
```

The script writes:

```text
summary.json                     aggregate success/collision/final-distance metrics
records.json                     per-episode scalar metrics
scenario_*.png                   trajectory, clearance, command comparison
lambda_*.png                     online dual-weight traces by method
energy_*_step_*.png              local U and |F|/arrow snapshots for fixed/forward/sensitivity
```

This scenario evaluator is different from `eval_dual_weight_sensitivity.py`: the older script evaluates dataset snapshots and energy fields, while this one performs a closed-loop bicycle/Ackermann rollout in a procedural scene.

## Energy-controller training with adaptive constraints

The previous trainer used a fixed weighted sum of local losses.  For quasi-static failures this is brittle, because a decreasing scalar loss can coexist with poor closed-loop progress.  The trainer now supports an energy-controller mode:

```bash
python -m train_dual_weight_energy \
  --data nav_stagewise_hyperring \
  --training-mode auglag \
  --epochs 80 \
  --bs 16 \
  --w-task 1.0 \
  --w-teacher 0.35 \
  --teacher-candidates 32 \
  --eps-clear 0.05 \
  --eps-act 0.01 \
  --eps-stall 0.001 \
  --eps-tv 0.08 \
  --rho-clear 2.0 \
  --rho-act 1.0 \
  --rho-stall 4.0 \
  --rho-tv 0.5 \
  --clear-rollout-steps 1 \
  --mu-max 50.0 \
  --energy-integrator semi_implicit \
  --integrator-damping 0.20 \
  --momentum-clip 1.50 \
  --lambda-step 0.12 \
  --lambda-max 6.0 \
  --outdir checkpoints/dual_weight_energy_controller
```

This mode uses:

```text
primary objective:
    force/progress alignment

adaptive augmented-Lagrangian constraints:
    obstacle/barrier exposure
    actuator projection defect
    quasi-static force floor
    dual-weight total variation
    energy smoothness

lambda-space teacher:
    random shooting over candidate dual-weight updates,
    scored by local progress, clearance, action feasibility, and anti-static force.
```

The learned object remains an energy controller:

```text
context -> dual weights lambda -> reshaped energy U_lambda -> force F_lambda -> projected Ackermann command
```

The lambda-space teacher does not supervise raw actions.  It supervises which dual weights should be raised or lowered so the induced energy field is more useful.

Implementation note: `C_clear` is now the barrier exposure along a short predicted local rollout under the learned force field, so the augmented-Lagrangian multiplier acts on a model-dependent constraint.  The fixed exposure at the current local origin is still logged as `C_clear_current` for diagnosing dataset difficulty.


## Medium-level failure replay and reverse-motion curriculum

The medium scheme adds two mechanisms on top of the stabilized augmented-Lagrangian trainer.

1. **Failure replay.** During normal ordered training, each mini-batch is scored for clearance exposure, actuation projection defect, stall, dual-weight jump, and reverse-command mismatch. Hard samples are copied into a replay buffer. Additional replay updates then train on these mined failure states without disrupting the ordered recurrent pass through the dataset.

2. **Parallel-parking augmentation.** Optional synthetic local patches expose the controller to multi-phase parking-like energy switches: reverse entry, reverse counter-steer, forward straighten, and final backward correction. These patches use local goals with negative x in the vehicle frame, so they require signed-speed training.

A recommended medium run is:

```bash
python -m train_dual_weight_energy \
  --data nav_stagewise_hyperring \
  --training-mode auglag \
  --signed-speed \
  --epochs 80 \
  --bs 16 \
  --w-task 1.0 \
  --w-teacher 0.35 \
  --w-reverse 1.0 \
  --teacher-candidates 32 \
  --eps-clear 0.05 \
  --eps-act 0.01 \
  --eps-stall 0.001 \
  --eps-tv 0.08 \
  --rho-clear 2.0 \
  --rho-act 1.0 \
  --rho-stall 4.0 \
  --rho-tv 0.5 \
  --clear-rollout-steps 2 \
  --mu-max 50.0 \
  --energy-integrator semi_implicit \
  --integrator-damping 0.20 \
  --momentum-clip 1.50 \
  --failure-replay-capacity 256 \
  --failure-replay-prob 0.35 \
  --failure-replay-topk 4 \
  --failure-replay-loss-scale 0.5 \
  --parking-augment-batches 2 \
  --parking-augment-bs 16 \
  --parking-loss-scale 0.25 \
  --outdir checkpoints/dual_weight_energy_medium_replay
```

Important new logs:

```text
replay_buffer_size
replay_steps
parking_steps
failure/replay_score
failure/reverse_needed
failure/reverse_mismatch
replay_tag_clear
replay_tag_act
replay_tag_stall
replay_tag_reverse
parking/C_clear
parking/C_reverse
parking/reverse_needed
```

Evaluate reverse-capable behavior on the procedural parking preset:

```bash
python -m eval_dual_weight_scenario_mpc \
  --ckpt checkpoints/dual_weight_energy_medium_replay/best.pt \
  --scenario parallel_parking \
  --episodes 5 \
  --max-steps 160 \
  --H 64 \
  --W 64 \
  --hat-d 3.5 \
  --modes forward,sensitivity,mpc \
  --signed-speed \
  --mpc-p-reverse 0.45 \
  --outdir eval_parallel_parking_medium
```

The fixed/default controller generally cannot solve this preset unless `--signed-speed` is enabled and the model has seen reverse local goals during training.

## Global-scale scenario evaluation

The scenario evaluator still supports the previous local energy snapshots and now also saves global-frame energy/force snapshots by transforming each local energy grid into world coordinates at the vehicle pose.

```bash
python -m eval_dual_weight_scenario_mpc \
  --ckpt checkpoints/dual_weight_energy_controller/best.pt \
  --scenario random \
  --episodes 5 \
  --max-steps 120 \
  --H 64 \
  --W 64 \
  --hat-d 3.5 \
  --modes fixed,forward,sensitivity,mpc \
  --field-steps 0,15,40,80 \
  --outdir eval_energy_controller_global
```

New files include:

```text
global_energy_000_step_0000.png
global_energy_000_step_0015.png
...
```

Each global energy figure overlays:

```text
scenario obstacles
executed trajectory
vehicle box
local U field transformed into world coordinates
world-frame force arrows
```

Use `--no-global-energy` to disable these additional plots when running large evaluation sweeps.

## Pluggable numerical integration schemes

This patch adds `scripts/integrators.py` and routes the sensitivity rollout,
lambda-space teacher, and scenario Ackermann rollout through reusable integrator
templates.

### Local pH / energy-field integrator

The local energy controller now integrates a virtual state `(q, p)` instead of
only using the explicit first-order update `q <- q + dt F(q)`.  Supported
schemes are:

```text
explicit
semi_implicit
midpoint
velocity_verlet
```

The default is `semi_implicit`, corresponding to:

```text
p_{k+1} = (p_k + dt F(q_k)) / (1 + dt*damping)
q_{k+1} = q_k + dt p_{k+1}
```

Use it in sensitivity evaluation:

```bash
python -m eval_dual_weight_sensitivity \
  --data nav_stagewise_hyperring \
  --ckpt checkpoints/dual_weight_energy_controller/best.pt \
  --energy-integrator semi_implicit \
  --integrator-damping 0.20 \
  --momentum-clip 1.50 \
  --outdir eval_dual_weight_sensitivity_semiimplicit
```

Try a second-order option:

```bash
python -m eval_dual_weight_sensitivity \
  --data nav_stagewise_hyperring \
  --ckpt checkpoints/dual_weight_energy_controller/best.pt \
  --energy-integrator velocity_verlet \
  --sens-horizon 8 \
  --outdir eval_dual_weight_sensitivity_verlet
```

The lambda-space teacher in training also accepts the same integrator:

```bash
python -m train_dual_weight_energy \
  --data nav_stagewise_hyperring \
  --training-mode auglag \
  --energy-integrator semi_implicit \
  --integrator-damping 0.20 \
  --momentum-clip 1.50 \
  --epochs 80 \
  --outdir checkpoints/dual_weight_energy_controller
```

### Ackermann pose integrator

Closed-loop scenario evaluation and MPC shooting now share a configurable
Ackermann pose step.  Supported pose schemes are:

```text
explicit
semi_implicit
midpoint
```

The default is `semi_implicit`, meaning the projected actuator state `(v,delta)`
is treated as the new feasible actuator state before updating pose.  For smoother
heading integration, use midpoint:

```bash
python -m eval_dual_weight_scenario_mpc \
  --ckpt checkpoints/dual_weight_energy_controller/best.pt \
  --scenario random \
  --episodes 5 \
  --energy-integrator semi_implicit \
  --ackermann-integrator midpoint \
  --outdir eval_energy_controller_midpoint
```

### Files touched

```text
scripts/integrators.py              new reusable integration templates
scripts/dual_weight_energy_nav.py   sensitivity rollout uses local pH integrator
train_dual_weight_energy.py         lambda-space teacher uses local pH integrator
eval_dual_weight_sensitivity.py     exposes --energy-integrator flags
eval_dual_weight_scenario_mpc.py    exposes --energy-integrator and --ackermann-integrator
```

This keeps the current default behavior close to the previous implementation but
makes the rollout numerics explicit and easy to replace with higher-order schemes.

---

# Reproducible Stage-1 Case-Suite Data Generation

This repo now includes a reproducible MPC-teacher data pipeline for offline policy training and later learned-policy evaluation on the same scenarios.

Generate a full case suite:

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
train.pt
val.pt
test.pt
scenes_train.json
scenes_val.json
scenes_test.json
episodes_train.json
episodes_val.json
episodes_test.json
summary.json
```

The `.pt` files are used for offline training and validation.  The `scenes_*.json` files save exact geometry, headings, case labels, and seeds, so the learned policy can later be evaluated on the identical held-out scenarios.

See `README_STAGE1_CASE_SUITE.md` for details.

## Stage-1 reproducible case-suite binding

See `README_BIND_STAGE1_PIPELINE.md` for the full workflow.  In brief,
`train_dual_weight_energy.py` now accepts `--data-format stage1` and converts
MPC-teacher rollout samples plus `scenes_*.json` into the local energy-patch
schema through `scripts/stage1_energy_adapter.py`.  Closed-loop evaluation can
consume exact held-out scene banks using `eval_dual_weight_scenario_mpc.py
--scene-bank path/to/scenes_test.json`.

## General short-horizon planning mode

See `README_FLOW_PLANNING_PIPELINE.md` for the new scenario-independent flow/planning-style training path.  It adds a learned K-step local planner head supervised by offline `pose_seq` and uses the predicted intermediate waypoint as the energy target at deployment, so online testing does not require MPC.
