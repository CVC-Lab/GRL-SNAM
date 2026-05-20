#!/usr/bin/env python3
"""Run the recommended staged pH/Hamiltonian coefficient fine-tuning schedule.

This wrapper calls ``python -m train_stage1_ph_energy`` three times:

  1. bootstrap: learn nonzero Ackermann steering from pH vector-field commands
  2. rollout:   shift authority toward Hamiltonian/pH rollout consistency
  3. safety:    harden clearance and passivity regularization

Each phase writes to a subdirectory and the next phase resumes from the previous
phase's ``best.pt`` checkpoint using the model-only ``--resume`` path.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class PhaseSpec:
    name: str
    epochs: int
    horizon: int
    p_recovery: float
    w_cmd: float
    w_roll: float
    w_final: float
    w_clear: float
    w_pass: float
    w_beta: float


def phase_specs(args: argparse.Namespace) -> List[PhaseSpec]:
    return [
        PhaseSpec(
            name="bootstrap",
            epochs=args.bootstrap_epochs,
            horizon=args.bootstrap_horizon,
            p_recovery=args.bootstrap_p_recovery,
            w_cmd=args.bootstrap_w_cmd,
            w_roll=args.bootstrap_w_roll,
            w_final=args.bootstrap_w_final,
            w_clear=args.bootstrap_w_clear,
            w_pass=args.bootstrap_w_pass,
            w_beta=args.bootstrap_w_beta,
        ),
        PhaseSpec(
            name="rollout",
            epochs=args.rollout_epochs,
            horizon=args.rollout_horizon,
            p_recovery=args.rollout_p_recovery,
            w_cmd=args.rollout_w_cmd,
            w_roll=args.rollout_w_roll,
            w_final=args.rollout_w_final,
            w_clear=args.rollout_w_clear,
            w_pass=args.rollout_w_pass,
            w_beta=args.rollout_w_beta,
        ),
        PhaseSpec(
            name="safety",
            epochs=args.safety_epochs,
            horizon=args.safety_horizon,
            p_recovery=args.safety_p_recovery,
            w_cmd=args.safety_w_cmd,
            w_roll=args.safety_w_roll,
            w_final=args.safety_w_final,
            w_clear=args.safety_w_clear,
            w_pass=args.safety_w_pass,
            w_beta=args.safety_w_beta,
        ),
    ]


def add_common_args(cmd: List[str], args: argparse.Namespace) -> None:
    pairs = [
        ("--samples", args.samples),
        ("--val-samples", args.val_samples),
        ("--bs", args.bs),
        ("--lr", args.lr),
        ("--workers", args.workers),
        ("--threads", args.threads),
        ("--seed", args.seed),
        ("--wheelbase", args.wheelbase),
        ("--v-max", args.v_max),
        ("--v-reverse-max", args.v_reverse_max),
        ("--max-steering-angle", args.max_steering_angle),
        ("--min-turning-radius", args.min_turning_radius),
        ("--dt", args.dt),
        ("--robot-radius", args.robot_radius),
        ("--n-beams", args.n_beams),
        ("--fov-deg", args.fov_deg),
        ("--range-max", args.range_max),
        ("--scan-noise", args.scan_noise),
        ("--scan-dropout", args.scan_dropout),
        ("--max-tokens", args.max_tokens),
        ("--n-obs-min", args.n_obs_min),
        ("--n-obs-max", args.n_obs_max),
        ("--corridor-half-width", args.corridor_half_width),
        ("--move-speed-floor", args.move_speed_floor),
        ("--stall-steps-trigger", args.stall_steps_trigger),
        ("--w-coef", args.w_coef),
    ]
    for k, v in pairs:
        cmd.extend([k, str(v)])


def run_phase(phase: PhaseSpec, args: argparse.Namespace, resume: str | None) -> str:
    outdir = os.path.join(args.outdir, phase.name)
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "train_stage1_ph_energy",
        "--outdir",
        outdir,
        "--epochs",
        str(phase.epochs),
        "--horizon",
        str(phase.horizon),
        "--p-recovery",
        str(phase.p_recovery),
        "--w-cmd",
        str(phase.w_cmd),
        "--w-roll",
        str(phase.w_roll),
        "--w-final",
        str(phase.w_final),
        "--w-clear",
        str(phase.w_clear),
        "--w-pass",
        str(phase.w_pass),
        "--w-beta",
        str(phase.w_beta),
    ]
    add_common_args(cmd, args)
    if resume is not None:
        cmd.extend(["--resume", resume])
        if args.resume_optimizer:
            cmd.append("--resume-optimizer")
        if args.continue_epoch_numbering:
            cmd.append("--continue-epoch-numbering")
    print("\n=== Running phase:", phase.name, "===")
    print(" ".join(shlex.quote(x) for x in cmd))
    if not args.dry_run:
        subprocess.check_call(cmd)
    best_path = os.path.join(outdir, "best.pt")
    if not args.dry_run and not os.path.exists(best_path):
        raise FileNotFoundError(f"Expected phase checkpoint not found: {best_path}")
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser("Run staged Stage-1 pH/Hamiltonian training")
    ap.add_argument("--outdir", type=str, default="checkpoints/stage1_ph_energy_multistage")
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--val-samples", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", type=str, default=None,
                    help="Optional initial checkpoint for the bootstrap phase.")
    ap.add_argument("--resume-optimizer", action="store_true",
                    help="Restore optimizer state when resuming between phases. Usually leave this off for fine-tuning.")
    ap.add_argument("--continue-epoch-numbering", action="store_true")

    # Vehicle/sensor/scene args passed through to train_stage1_ph_energy.
    ap.add_argument("--wheelbase", type=float, default=0.324)
    ap.add_argument("--v-max", type=float, default=0.80)
    ap.add_argument("--v-reverse-max", type=float, default=0.25)
    ap.add_argument("--max-steering-angle", type=float, default=0.396)
    ap.add_argument("--min-turning-radius", type=float, default=0.80)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--robot-radius", type=float, default=0.23)
    ap.add_argument("--n-beams", type=int, default=181)
    ap.add_argument("--fov-deg", type=float, default=270.0)
    ap.add_argument("--range-max", type=float, default=6.0)
    ap.add_argument("--scan-noise", type=float, default=0.015)
    ap.add_argument("--scan-dropout", type=float, default=0.01)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--n-obs-min", type=int, default=8)
    ap.add_argument("--n-obs-max", type=int, default=20)
    ap.add_argument("--corridor-half-width", type=float, default=0.70)
    ap.add_argument("--move-speed-floor", type=float, default=0.10)
    ap.add_argument("--stall-steps-trigger", type=int, default=8)
    ap.add_argument("--w-coef", type=float, default=1e-4)

    # Phase A defaults: steering/bootstrap.
    ap.add_argument("--bootstrap-epochs", type=int, default=10)
    ap.add_argument("--bootstrap-horizon", type=int, default=12)
    ap.add_argument("--bootstrap-p-recovery", type=float, default=0.30)
    ap.add_argument("--bootstrap-w-cmd", type=float, default=5.0)
    ap.add_argument("--bootstrap-w-roll", type=float, default=1.0)
    ap.add_argument("--bootstrap-w-final", type=float, default=0.5)
    ap.add_argument("--bootstrap-w-clear", type=float, default=1.0)
    ap.add_argument("--bootstrap-w-pass", type=float, default=0.005)
    ap.add_argument("--bootstrap-w-beta", type=float, default=0.0005)

    # Phase B defaults: pH rollout.
    ap.add_argument("--rollout-epochs", type=int, default=20)
    ap.add_argument("--rollout-horizon", type=int, default=16)
    ap.add_argument("--rollout-p-recovery", type=float, default=0.35)
    ap.add_argument("--rollout-w-cmd", type=float, default=1.0)
    ap.add_argument("--rollout-w-roll", type=float, default=3.0)
    ap.add_argument("--rollout-w-final", type=float, default=1.5)
    ap.add_argument("--rollout-w-clear", type=float, default=3.0)
    ap.add_argument("--rollout-w-pass", type=float, default=0.05)
    ap.add_argument("--rollout-w-beta", type=float, default=0.001)

    # Phase C defaults: safety hardening.
    ap.add_argument("--safety-epochs", type=int, default=20)
    ap.add_argument("--safety-horizon", type=int, default=16)
    ap.add_argument("--safety-p-recovery", type=float, default=0.40)
    ap.add_argument("--safety-w-cmd", type=float, default=0.5)
    ap.add_argument("--safety-w-roll", type=float, default=3.0)
    ap.add_argument("--safety-w-final", type=float, default=2.0)
    ap.add_argument("--safety-w-clear", type=float, default=8.0)
    ap.add_argument("--safety-w-pass", type=float, default=0.10)
    ap.add_argument("--safety-w-beta", type=float, default=0.002)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    phases = phase_specs(args)
    with open(os.path.join(args.outdir, "multistage_schedule.json"), "w") as f:
        json.dump({"phases": [asdict(p) for p in phases], "args": vars(args)}, f, indent=2)

    resume = args.resume
    best_by_phase: Dict[str, str] = {}
    for phase in phases:
        if phase.epochs <= 0:
            print(f"Skipping phase {phase.name} because epochs <= 0")
            continue
        best = run_phase(phase, args, resume)
        best_by_phase[phase.name] = best
        resume = best

    with open(os.path.join(args.outdir, "multistage_outputs.json"), "w") as f:
        json.dump(best_by_phase, f, indent=2)
    print("\nFinal checkpoints:")
    print(json.dumps(best_by_phase, indent=2))


if __name__ == "__main__":
    main()
