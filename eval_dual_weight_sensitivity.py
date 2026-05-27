#!/usr/bin/env python3
"""Evaluate fixed/forward/sensitivity dual-weight adaptation and visualize U/F changes."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from scripts.energy_data_navmax import NavMaximinLocalPatches, collate_navmax
    from scripts.dual_weight_energy_nav import (
        AckermannLimitConfig,
        DualWeightConfig,
        DualWeightEnergyNet,
        DualWeightOnlineController,
        SensitivityUpdateConfig,
        LAMBDA_NAMES,
        input_barrier_loss,
    )
except Exception:  # pragma: no cover
    from energy_data_navmax import NavMaximinLocalPatches, collate_navmax
    from dual_weight_energy_nav import (
        AckermannLimitConfig,
        DualWeightConfig,
        DualWeightEnergyNet,
        DualWeightOnlineController,
        SensitivityUpdateConfig,
        LAMBDA_NAMES,
        input_barrier_loss,
    )


def load_model(ckpt_path: str, H: int, W: int, device: str) -> DualWeightEnergyNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_raw = ckpt.get("model_cfg", {})
    cfg_raw = {k: v for k, v in cfg_raw.items() if k in DualWeightConfig.__dataclass_fields__}
    cfg = DualWeightConfig(**cfg_raw)
    model = DualWeightEnergyNet(H=H, W=W, cfg=cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print({"load_missing_keys": missing, "load_unexpected_keys": unexpected})
    model.eval()
    return model


def tensor_to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def step_metrics(out: Dict[str, torch.Tensor], batch: Dict[str, Any], limit_cfg: AckermannLimitConfig) -> Dict[str, float]:
    F0 = out["F0"]
    d = batch["dir_xy"].to(F0.device)
    align = 1.0 - (
        (F0 / torch.linalg.norm(F0, dim=-1, keepdim=True).clamp_min(1e-8))
        * (d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-8))
    ).sum(dim=-1).clamp(-1, 1)
    force_norm = torch.linalg.norm(F0, dim=-1)
    proj_defect = ((out["u_raw"] - out["u"]) ** 2).sum(dim=-1)
    ibar = input_barrier_loss(out["u_raw"], limit_cfg)
    return {
        "align": tensor_to_float(align.mean()),
        "force_norm": tensor_to_float(force_norm.mean()),
        "proj_defect": tensor_to_float(proj_defect.mean()),
        "input_barrier": tensor_to_float(ibar),
        "speed": tensor_to_float(out["u"][:, 0].mean()),
        "steering": tensor_to_float(out["u"][:, 1].mean()),
    }


def summarize(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = sorted(records[0].keys())
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in records], dtype=np.float32)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


def save_lambda_plot(path: str, traces: Dict[str, List[np.ndarray]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    for mode, arrs in traces.items():
        if not arrs:
            continue
        A = np.stack(arrs, axis=0)
        fig = plt.figure(figsize=(10.5, 5.2), dpi=150)
        ax = fig.add_subplot(111)
        for i, name in enumerate(LAMBDA_NAMES):
            ax.plot(np.arange(A.shape[0]), A[:, i], label=name)
        ax.set_title(f"dual weights over time: {mode}")
        ax.set_xlabel("snapshot step")
        ax.set_ylabel("lambda")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=4)
        fig.tight_layout()
        fig.savefig(path.replace(".png", f"_{mode}.png"), bbox_inches="tight")
        plt.close(fig)


def save_metric_plot(path: str, metrics: Dict[str, List[Dict[str, float]]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["align", "force_norm", "proj_defect", "speed", "steering"]
    fig = plt.figure(figsize=(12.5, 8.5), dpi=150)
    for j, key in enumerate(keys, start=1):
        ax = fig.add_subplot(3, 2, j)
        for mode, recs in metrics.items():
            vals = [r[key] for r in recs]
            ax.plot(np.arange(len(vals)), vals, label=mode)
        ax.set_title(key)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("snapshot step")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _field_norm(F: np.ndarray) -> np.ndarray:
    return np.sqrt(F[0] ** 2 + F[1] ** 2)


def save_energy_field_plot(path: str, snapshots: Dict[int, Dict[str, Dict[str, np.ndarray]]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    for step, by_mode in snapshots.items():
        modes = list(by_mode.keys())
        fig = plt.figure(figsize=(4.3 * len(modes), 8.0), dpi=150)
        for i, mode in enumerate(modes):
            U = by_mode[mode]["U"]
            F = by_mode[mode]["F"]
            ax = fig.add_subplot(2, len(modes), i + 1)
            im = ax.imshow(U, origin="lower")
            ax.set_title(f"{mode}: U")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax2 = fig.add_subplot(2, len(modes), len(modes) + i + 1)
            im2 = ax2.imshow(_field_norm(F), origin="lower")
            # sparse quiver for direction inspection
            H, W = U.shape
            stride = max(4, H // 12)
            yy, xx = np.mgrid[0:H:stride, 0:W:stride]
            ax2.quiver(xx, yy, F[0, ::stride, ::stride], F[1, ::stride, ::stride], angles="xy", scale_units="xy", scale=1.0)
            ax2.set_title(f"{mode}: |F| + arrows")
            ax2.set_xticks([]); ax2.set_yticks([])
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        fig.suptitle(f"energy/field comparison at snapshot step {step}")
        fig.tight_layout()
        fig.savefig(path.replace(".png", f"_step_{step:04d}.png"), bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Evaluate dual-weight sensitivity updates")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="eval_dual_weight_sensitivity")
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--hat-d", type=float, default=None)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--modes", type=str, default="fixed,forward,sensitivity")
    ap.add_argument("--plot-steps", type=str, default="0,20,60,100")
    # actuation limits
    ap.add_argument("--v-min", type=float, default=-0.25)
    ap.add_argument("--v-max", type=float, default=0.80)
    ap.add_argument("--steering-max", type=float, default=0.396)
    ap.add_argument("--accel-max", type=float, default=1.5)
    ap.add_argument("--steering-rate-max", type=float, default=1.5)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--energy-integrator", type=str, default="semi_implicit",
                    choices=["explicit", "semi_implicit", "midpoint", "velocity_verlet"],
                    help="Local pH/energy integrator used by sensitivity rollouts.")
    ap.add_argument("--integrator-damping", type=float, default=0.20)
    ap.add_argument("--momentum-clip", type=float, default=1.50)
    # sensitivity knobs
    ap.add_argument("--sens-eta", type=float, default=0.08)
    ap.add_argument("--sens-horizon", type=int, default=8)
    ap.add_argument("--sens-grad-clip", type=float, default=1.0)
    ap.add_argument("--sens-w-goal", type=float, default=1.0)
    ap.add_argument("--sens-w-path", type=float, default=0.15)
    ap.add_argument("--sens-w-barrier", type=float, default=0.25)
    ap.add_argument("--sens-w-align", type=float, default=0.25)
    ap.add_argument("--sens-w-act", type=float, default=0.05)
    ap.add_argument("--sens-w-stall", type=float, default=0.25)
    ap.add_argument("--force-floor", type=float, default=0.06)
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    os.makedirs(args.outdir, exist_ok=True)
    model = load_model(args.ckpt, args.H, args.W, args.device)
    ds = NavMaximinLocalPatches(args.data, hat_d=args.hat_d, H=args.H, W=args.W)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_navmax)
    limit_cfg = AckermannLimitConfig(
        v_min=args.v_min, v_max=args.v_max, steering_max=args.steering_max,
        accel_max=args.accel_max, steering_rate_max=args.steering_rate_max, dt=args.dt,
    )
    sens_cfg = SensitivityUpdateConfig(
        eta=args.sens_eta, horizon=args.sens_horizon, dt=args.dt, grad_clip=args.sens_grad_clip,
        w_goal=args.sens_w_goal, w_path=args.sens_w_path, w_barrier=args.sens_w_barrier,
        w_align=args.sens_w_align, w_act=args.sens_w_act, w_stall=args.sens_w_stall,
        force_floor=args.force_floor,
        integrator=args.energy_integrator, damping=args.integrator_damping, momentum_clip=args.momentum_clip,
    )
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    controllers = {m: DualWeightOnlineController(model, limit_cfg=limit_cfg, sensitivity_cfg=sens_cfg) for m in modes}
    plot_steps = {int(x) for x in args.plot_steps.split(",") if x.strip()}
    metrics: Dict[str, List[Dict[str, float]]] = {m: [] for m in modes}
    lambdas: Dict[str, List[np.ndarray]] = {m: [] for m in modes}
    snapshots: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    prev_episode: Optional[int] = None
    prev_stage: Optional[int] = None
    for step, batch in enumerate(dl):
        if step >= args.steps:
            break
        ep = int(batch["meta"]["episode"][0].item())
        st = int(batch["meta"]["stage"][0].item())
        new_ep = prev_episode is None or ep != prev_episode
        if new_ep:
            for c in controllers.values():
                c.reset()
            stage_changed = torch.zeros(1, device=args.device)
        else:
            stage_changed = torch.tensor([1.0 if st != prev_stage else 0.0], device=args.device)
        prev_episode, prev_stage = ep, st
        for mode, controller in controllers.items():
            out = controller.step(batch, stage_changed=stage_changed, adaptation=mode)
            rec = step_metrics(out, batch, limit_cfg)
            if "sens_loss" in out:
                rec["sens_loss"] = tensor_to_float(out["sens_loss"])
                rec["sens_grad_norm"] = tensor_to_float(out["sens_grad_norm"].mean())
            metrics[mode].append(rec)
            lambdas[mode].append(out["lambda"][0].detach().cpu().numpy())
            if step in plot_steps:
                snapshots.setdefault(step, {})[mode] = {
                    "U": out["U"][0, 0].detach().cpu().numpy(),
                    "F": out["F"][0].detach().cpu().numpy(),
                }
    summary = {mode: summarize(recs) for mode, recs in metrics.items()}
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.outdir, "lambda_names.json"), "w") as f:
        json.dump(list(LAMBDA_NAMES), f, indent=2)
    for mode, arrs in lambdas.items():
        if arrs:
            np.save(os.path.join(args.outdir, f"lambda_trace_{mode}.npy"), np.stack(arrs, axis=0))
    save_lambda_plot(os.path.join(args.outdir, "lambda_traces.png"), lambdas)
    save_metric_plot(os.path.join(args.outdir, "method_metrics.png"), metrics)
    save_energy_field_plot(os.path.join(args.outdir, "energy_fields.png"), snapshots)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
