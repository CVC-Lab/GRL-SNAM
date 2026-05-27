#!/bin/bash
#------------------------------------------------------------
# Lonestar6 A100 single-node workflow for three-stage training:
#   A. planner-only pretraining
#   B. energy controller with teacher sub-goals
#   C. alternating coupled planner-energy training
#
# Submit:
#   sbatch ls6_three_stage_a100.slurm
#
# Notes:
#   - Lonestar6 A100 nodes have 3x A100 GPUs.
#   - The safe default below runs one Python process per stage.
#   - Set USE_TORCHRUN=1 only if train_dual_weight_energy.py is DDP-aware.
#     Otherwise torchrun will start three duplicate trainings that overwrite
#     the same checkpoint directory.
#------------------------------------------------------------

#SBATCH -J dual_energy_3stage
#SBATCH -o logs/dual_energy_3stage.%j.out
#SBATCH -e logs/dual_energy_3stage.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A YOUR_TACC_ALLOCATION
#SBATCH --mail-type=all
#SBATCH --mail-user=example@email.com

# If your local Slurm configuration requires explicit GPU GRES, uncomment:
##SBATCH --gres=gpu:3

set -euo pipefail

mkdir -p logs

echo "===== Job info ====="
date
hostname
pwd
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---------------- User-editable paths ----------------
# Put repo, venv, data, and outputs on $SCRATCH for speed.
PROJECT_DIR="${PROJECT_DIR:-$HOME/GRL-SNAM/dual-weight-energy-alternating-plan-energy}"
DATA_ROOT="${DATA_ROOT:-$HOME/GRL-SNAM/data/stage1_case_suite}"
VENV_DIR="${VENV_DIR:-$HOME/GRL-SNAM/venv}"
RUN_ROOT="${RUN_ROOT:-$HOME/GRL-SNAM/dual_energy_runs/three_stage_${SLURM_JOB_ID:-manual}}"

# Case suite used in all stages.
CASES="${CASES:-narrow_gate,s_turn,front_trap,parallel_parking,terminal_bay,long_horizon_mixed}"

# Training scale knobs.
BS="${BS:-256}"
WORKERS="${WORKERS:-4}"
H="${H:-64}"
W="${W:-64}"
HAT_D="${HAT_D:-0.35}"
PLAN_CANDIDATES="${PLAN_CANDIDATES:-8}"
PLAN_HORIZON="${PLAN_HORIZON:-6}"

# Epochs.
EPOCHS_A="${EPOCHS_A:-50}"
EPOCHS_B="${EPOCHS_B:-50}"
EPOCHS_C="${EPOCHS_C:-80}"

# Launcher mode:
#   USE_TORCHRUN=0: robust default, one Python process, generally one GPU.
#   USE_TORCHRUN=1: true 3-GPU launch, only if your trainer supports DDP.
USE_TORCHRUN="${USE_TORCHRUN:-0}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29531}"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "VENV_DIR=$VENV_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "USE_TORCHRUN=$USE_TORCHRUN"

# ---------------- Environment ----------------
module load cuda/12.8

source "${VENV_DIR}/bin/activate"

cd "$PROJECT_DIR"

python - <<'PYCHK'
import os, torch
print("Python OK")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PYCHK
nvidia-smi || true

mkdir -p "$RUN_ROOT"

run_train () {
    if [[ "$USE_TORCHRUN" == "1" ]]; then
        echo "[launcher] torchrun on ${NGPUS} local processes"
        torchrun \
          --standalone \
          --nnodes=1 \
          --nproc_per_node="${NGPUS}" \
          --master_port="${MASTER_PORT}" \
          -m train_dual_weight_energy "$@"
    else
        echo "[launcher] python single process"
        # Keep all GPUs visible in case your code wraps DataParallel internally;
        # otherwise PyTorch will use cuda:0 by default.
        python -u -m train_dual_weight_energy "$@"
    fi
}

# Shared arguments.
COMMON_ARGS=(
  --data "$DATA_ROOT"
  --data-format stage1
  --stage1-split train
  --stage1-cases "$CASES"
  --stage1-direction-from cmd
  --H "$H" --W "$W" --hat_d "$HAT_D"
  --bs "$BS"
  --workers "$WORKERS"
  --use-plan-goal
  --plan-candidates "$PLAN_CANDIDATES"
  --plan-horizon "$PLAN_HORIZON"
  --plan-goal-index 3
  --clearance-mode auto
  --clear-safe-margin 0.05
  --clear-tau 0.10
  --clear-cost-clip 5.0
  --lambda-max 3.0
  --integrator-damping 0.40
  --momentum-clip 0.75
)

# ---------------- Stage A: planner-only pretraining ----------------
STAGE_A="$RUN_ROOT/stageA_planner_only"
echo "===== Stage A: planner-only pretraining ====="
date
run_train \
  "${COMMON_ARGS[@]}" \
  --planner-only \
  --plan-target-mode arclength \
  --plan-arc-step 0.25 \
  --w-plan 1.0 \
  --w-plan-clear 0.05 \
  --w-plan-dyn 0.05 \
  --w-plan-diversity 0.02 \
  --training-mode weighted \
  --epochs "$EPOCHS_A" \
  --outdir "$STAGE_A"

test -f "$STAGE_A/best.pt" || { echo "Missing Stage A checkpoint: $STAGE_A/best.pt"; exit 10; }

# ---------------- Stage B: controller with teacher sub-goals ----------------
STAGE_B="$RUN_ROOT/stageB_controller_teacher_plan"
echo "===== Stage B: local energy/controller using teacher sub-goals ====="
date
run_train \
  "${COMMON_ARGS[@]}" \
  --resume "$STAGE_A/best.pt" \
  --training-mode auglag \
  --signed-speed \
  --energy-plan-source teacher \
  --w-plan 0.2 \
  --w-plan-track 0.5 \
  --w-rollout 0.3 \
  --w-cmd 0.2 \
  --eps-clear 0.05 \
  --eps-act 0.5 \
  --rho-clear 0.05 \
  --rho-act 0.05 \
  --mu-max 5.0 \
  --mu-step-clip 0.05 \
  --epochs "$EPOCHS_B" \
  --outdir "$STAGE_B"

test -f "$STAGE_B/best.pt" || { echo "Missing Stage B checkpoint: $STAGE_B/best.pt"; exit 20; }

# ---------------- Stage C: alternating coupled planner-energy training ----------------
STAGE_C="$RUN_ROOT/stageC_coupled_alternating"
echo "===== Stage C: alternating coupled training ====="
date
run_train \
  "${COMMON_ARGS[@]}" \
  --resume "$STAGE_B/best.pt" \
  --training-mode auglag \
  --signed-speed \
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
  --eps-clear 0.05 \
  --eps-act 0.5 \
  --rho-clear 0.05 \
  --rho-act 0.05 \
  --mu-max 5.0 \
  --mu-step-clip 0.05 \
  --failure-replay-start-epoch 10 \
  --failure-replay-prob 0.05 \
  --failure-replay-loss-scale 0.2 \
  --epochs "$EPOCHS_C" \
  --outdir "$STAGE_C"

test -f "$STAGE_C/best.pt" || { echo "Missing Stage C checkpoint: $STAGE_C/best.pt"; exit 30; }

echo "===== Done ====="
date
echo "Stage A: $STAGE_A/best.pt"
echo "Stage B: $STAGE_B/best.pt"
echo "Stage C: $STAGE_C/best.pt"
