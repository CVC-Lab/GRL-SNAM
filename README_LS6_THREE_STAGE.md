# Lonestar6 A100 three-stage training scripts

This package contains:

- `ls6_three_stage_a100.slurm`: one Slurm job that runs Stage A, Stage B, and Stage C sequentially on one `gpu-a100` node.
- `setup_env_ls6.sh`: one-time Python virtual environment setup.
- `submit_example.sh`: example `sbatch` submission wrapper.

## Recommended use

1. Put the repo and data under `$SCRATCH`, for example:

```bash
cds
unzip dual-weight-energy-alternating-plan-energy.zip -d $SCRATCH/
# or clone/copy your repo to:
# $SCRATCH/dual-weight-energy-alternating-plan-energy

# data should look like:
# $SCRATCH/data/stage1_case_suite/{narrow_gate,s_turn,...}/train.pt
```

2. Create the virtual environment. Prefer doing this in an interactive A100 dev job:

```bash
idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00
bash setup_env_ls6.sh
```

3. Edit `ls6_three_stage_a100.slurm`:

```bash
#SBATCH -A YOUR_TACC_ALLOCATION
```

4. Submit:

```bash
sbatch ls6_three_stage_a100.slurm
```

Or use:

```bash
bash submit_example.sh
```

## Important note on multi-GPU

The Slurm script requests one A100 node. Each Lonestar6 A100 node has three A100 GPUs. The script has:

```bash
USE_TORCHRUN=0
```

as the safe default. Set `USE_TORCHRUN=1` only if `train_dual_weight_energy.py` is DDP-aware. If the trainer is not DDP-aware, `torchrun --nproc_per_node=3` will launch three duplicate trainings that all write to the same checkpoint directory.

If DDP is not implemented yet, the job still benefits from the A100 node but each stage will typically use one GPU. To truly use all three GPUs for one stage, patch the trainer with `DistributedDataParallel` or another safe multi-GPU path.

## Useful commands

```bash
squeue -u $USER
tail -f logs/dual_energy_3stage.<JOBID>.out
tail -f logs/dual_energy_3stage.<JOBID>.err
```
