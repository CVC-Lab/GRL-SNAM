"""Synthetic local patches for reverse/parallel-parking style maneuvers.

These batches intentionally match the ``NavMaximinLocalPatches`` collated schema
used by the dual-weight energy trainer.  They are not a replacement for a real
parking dataset; they provide a medium-level curriculum that forces the energy
controller to represent backward motion and multi-phase energy switching.
"""
from __future__ import annotations

from typing import Dict, Optional
import math
import torch


def make_coord_grid(H: int, W: int, half: float, *, device="cpu", dtype=torch.float32) -> torch.Tensor:
    ys = torch.linspace(-half, half, H, device=device, dtype=dtype)
    xs = torch.linspace(-half, half, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([X, Y], dim=-1)


def barrier_map_from_sd(sd: torch.Tensor, d_hat: float, max_b: float = 8.0) -> torch.Tensor:
    # Same smooth finite barrier style as the online scenario evaluator.
    eps = 1e-6
    dh = sd.new_tensor(float(d_hat))
    safe = torch.clamp(sd, min=eps)
    b_in = -(sd - dh) ** 2 * torch.log(safe / dh)
    b = torch.where(sd < dh, b_in, torch.zeros_like(sd))
    b = torch.where(sd <= eps, dh * dh * (-torch.log(sd.new_tensor(eps) / dh)) + (eps - sd) ** 2, b)
    return torch.clamp(b, 0.0, max_b)


def _phase_goal(phase: torch.Tensor, device, dtype):
    B = int(phase.shape[0])
    g = torch.zeros(B, 2, device=device, dtype=dtype)
    # Phase 0: reverse into the slot from a forward-staged start.
    m = phase == 0
    g[m, 0] = -1.25; g[m, 1] = -0.70
    # Phase 1: continue reverse but counter-steer toward the slot center.
    m = phase == 1
    g[m, 0] = -0.55; g[m, 1] = -0.25
    # Phase 2: small forward correction / straighten.
    m = phase == 2
    g[m, 0] = 0.45; g[m, 1] = 0.03
    # Phase 3: final tiny backward alignment correction.
    m = phase == 3
    g[m, 0] = -0.18; g[m, 1] = 0.00
    return g


def sample_parallel_parking_batch(
    batch_size: int,
    H: int,
    W: int,
    hat_d: float,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Return a synthetic B-sample parking batch with fixed N obstacle disks.

    The local frame is the vehicle frame: +x is current forward direction.  The
    reverse phases have local goals with negative x, which is what trains the
    signed-speed energy switch.
    """
    B = int(batch_size)
    device = torch.device(device)
    grid_xy = make_coord_grid(H, W, float(hat_d), device=device, dtype=dtype)
    coord_map_single = torch.stack([grid_xy[..., 0] / float(hat_d), grid_xy[..., 1] / float(hat_d)], dim=0)

    phase = torch.randint(0, 4, (B,), device=device, generator=generator)
    goal_l = _phase_goal(phase, device, dtype)
    goal_l = goal_l + 0.08 * torch.randn(B, 2, device=device, dtype=dtype, generator=generator)

    # Fixed parking geometry in local coordinates with slight per-sample jitter:
    # front/rear parked cars, curb/wall disks, and a near front-corner obstacle.
    base_C = torch.tensor([
        [-2.05, -0.58], [-1.55, -0.58],   # rear parked car footprint
        [ 1.55, -0.58], [ 2.05, -0.58],   # front parked car footprint
        [-1.70, -1.18], [-0.55, -1.20], [0.55, -1.20], [1.70, -1.18],  # curb/wall samples
        [ 0.90,  0.18],                  # traffic-side front corner / cone
    ], device=device, dtype=dtype)
    base_R = torch.tensor([0.33, 0.33, 0.33, 0.33, 0.18, 0.18, 0.18, 0.18, 0.20], device=device, dtype=dtype)
    N = int(base_R.numel())
    jitter = 0.04 * torch.randn(B, N, 2, device=device, dtype=dtype, generator=generator)
    C = base_C.view(1, N, 2).expand(B, -1, -1) + jitter
    R = base_R.view(1, N).expand(B, -1)

    barrier = []
    obs_feats = []
    for b in range(B):
        maps = []
        feats = []
        for j in range(N):
            c = C[b, j]
            r = R[b, j]
            sd = torch.linalg.norm(grid_xy - c.view(1, 1, 2), dim=-1) - r
            maps.append(barrier_map_from_sd(sd, float(hat_d)))
            norm = torch.linalg.norm(c)
            ang = torch.atan2(c[1], c[0])
            prox = (float(hat_d) - torch.clamp(norm - r, min=0.0)) / float(hat_d)
            feats.append(torch.stack([c[0], c[1], r, norm, ang, prox.clamp(0.0, 1.0)]))
        barrier.append(torch.stack(maps, dim=0))
        obs_feats.append(torch.stack(feats, dim=0))
    barrier_stack = torch.stack(barrier, dim=0)
    obs_feats_t = torch.stack(obs_feats, dim=0)
    obs_mask = torch.ones(B, N, device=device, dtype=torch.bool)

    goal_map = 0.5 * ((grid_xy.view(1, H, W, 2) - goal_l.view(B, 1, 1, 2)) ** 2).sum(dim=-1).unsqueeze(1)
    coord_map = coord_map_single.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    gl_norm = torch.linalg.norm(goal_l, dim=-1)
    gl_ang = torch.atan2(goal_l[:, 1], goal_l[:, 0])
    goal_feats = torch.stack([goal_l[:, 0], goal_l[:, 1], gl_norm, gl_ang], dim=-1)
    dir_xy = goal_l / gl_norm.view(B, 1).clamp_min(1e-8)

    return {
        "grid_xy": grid_xy.unsqueeze(0).expand(B, -1, -1, -1).contiguous(),
        "goal_map": goal_map,
        "coord_map": coord_map,
        "barrier_stack": barrier_stack,
        "obs_mask": obs_mask,
        "obs_feats": obs_feats_t,
        "obs_weights": torch.ones(B, N, device=device, dtype=dtype),
        "goal_feats": goal_feats,
        "pos_xy": torch.zeros(B, 2, device=device, dtype=dtype),
        "dir_xy": dir_xy,
        "meta": {
            "hat_d": torch.full((B,), float(hat_d), device=device, dtype=dtype),
            "episode": torch.arange(B, device=device, dtype=torch.long),
            "stage": phase.to(torch.long),
            "snap": torch.zeros(B, device=device, dtype=torch.long),
            "success": torch.ones(B, device=device, dtype=torch.long),
            "parking_phase": phase.to(torch.long),
        },
    }
