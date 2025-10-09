# plot_dataset_trajectories.py
import os
import random
from typing import List, Optional, Sequence
import numpy as np

import torch
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def _as_tensor(x, dtype=torch.float32):
    return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=dtype)

def _get_xy(t: torch.Tensor) -> torch.Tensor:
    """
    Take first two dims as (x,y). Supports [T,2+] or [2+].
    Returns shape [T,2].
    """
    t = _as_tensor(t).detach().cpu().float()
    if t.ndim == 1:
        if t.shape[0] < 2:
            raise ValueError("Need at least 2D to plot (x,y).")
        return t[:2].unsqueeze(0)
    if t.ndim >= 2:
        if t.shape[-1] < 2:
            raise ValueError(f"Last dim < 2: {tuple(t.shape)}")
        return t[..., :2]
    raise ValueError(f"Unexpected tensor shape: {tuple(t.shape)}")

def _draw_obstacles(ax, centers: torch.Tensor, radii: torch.Tensor,
                    facecolor=(0.95,0.4,0.4), edgecolor=(0.35,0.05,0.05), alpha=0.25):
    centers = _as_tensor(centers).cpu().float()
    radii   = _as_tensor(radii).cpu().float()
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError(f"obstacle_centers should be (C,2), got {tuple(centers.shape)}")
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("obstacle_radii should be (C,) matching centers.")
    for (cx, cy), r in zip(centers.numpy(), radii.numpy()):
        circ = Circle((cx, cy), r, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, alpha=alpha)
        ax.add_patch(circ)

def _compute_limits(items: Sequence[torch.Tensor], pad_ratio: float = 0.08):
    xs, ys = [], []
    for it in items:
        a = _as_tensor(it).cpu().float()
        if a.ndim == 1 and a.numel() >= 2:
            xs.append(a[0].item()); ys.append(a[1].item())
        elif a.ndim == 2 and a.shape[-1] >= 2:
            xs.extend(a[:, 0].tolist()); ys.extend(a[:, 1].tolist())
    if not xs:
        return (-1, 1), (-1, 1)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    px, py = dx * pad_ratio, dy * pad_ratio
    return (xmin - px, xmax + px), (ymin - py, ymax + py)

def plot_single_trajectory(
    traj: dict,
    ax=None,
    show_velocity_every: int = 5,
    velocity_scale: float = 1.0,
    path_kwargs: Optional[dict] = None,
    quiver_kwargs: Optional[dict] = None,
    show_goal: bool = True,
    show_start: bool = True,
    draw_obstacles: bool = True,
):
    """
    Plot one trajectory dict from the dataset on an axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # ---- meta
    goal = _get_xy(traj["goal_position"])
    centers = traj.get("obstacle_centers", None)
    radii   = traj.get("obstacle_radii", None)
    if draw_obstacles and centers is not None and radii is not None:
        _draw_obstacles(ax, centers, radii)

    # ---- time series: q_frame & p_frame
    states: List[dict] = traj["trajectory_states"]
    q_seq = torch.stack([_get_xy(s["q_frame"]).squeeze(0) for s in states], dim=0)  # [T,2]
    p_seq = torch.stack([_get_xy(s["p_frame"]).squeeze(0) for s in states], dim=0)  # [T,2]

    # ---- path
    pk = dict(color="tab:blue", linewidth=2.0, alpha=0.95)
    if path_kwargs: pk.update(path_kwargs)
    ax.plot(q_seq[:, 0].numpy(), q_seq[:, 1].numpy(), **pk)

    # ---- start & goal
    if show_start:
        ax.scatter(q_seq[0, 0].item(), q_seq[0, 1].item(), marker="o", s=40, color="tab:blue", zorder=3, label="start")
    if show_goal:
        ax.scatter(goal[0, 0].item(), goal[0, 1].item(), marker="*", s=120, color="tab:green", edgecolor="k", zorder=4, label="goal")

    # ---- velocity arrows
    if show_velocity_every is not None and show_velocity_every > 0:
        idx = torch.arange(0, q_seq.shape[0], show_velocity_every)
        qv = q_seq[idx]
        pv = p_seq[idx] * float(velocity_scale)
        qk = dict(angles="xy", scale_units="xy", scale=1.0, width=0.004, alpha=0.9)
        if quiver_kwargs: qk.update(quiver_kwargs)
        ax.quiver(qv[:, 0].numpy(), qv[:, 1].numpy(), pv[:, 0].numpy(), pv[:, 1].numpy(), **qk)

    # ---- cosmetics
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    return ax, q_seq, p_seq

def plot_sampled_trajectories(
    pt_path: str,
    num_trajs: int = 4,
    seed: Optional[int] = 0,
    velocity_every: int = 5,
    velocity_scale: float = 1.0,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """
    Draw num_trajs random trajectories with scene layout (obstacles + goal).
    """
    data = torch.load(pt_path, map_location="cpu")
    assert isinstance(data, list) and len(data) > 0, "Top-level should be a non-empty list of trajectories."

    rng = random.Random(seed) if seed is not None else random
    idxs = list(range(len(data)))
    rng.shuffle(idxs)
    idxs = idxs[:min(num_trajs, len(data))]

    # Prepare grid
    n = len(idxs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))

    # Flatten axes to a simple list
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]
    axes = [a for row in (axes if isinstance(axes, list) else [axes]) for a in (row if isinstance(row, (list, tuple)) else [row])]
    axes = axes[:n]

    # First pass: plot and track limits
    all_items_for_limits = []
    for ax, i in zip(axes, idxs):
        traj = data[i]
        ax.set_title(f"Trajectory {i} | success={traj.get('success', 'NA')}")
        ax, q_seq, _ = plot_single_trajectory(
            traj, ax=ax,
            show_velocity_every=velocity_every,
            velocity_scale=velocity_scale,
        )
        all_items_for_limits.append(q_seq)
        if "obstacle_centers" in traj:
            all_items_for_limits.append(_get_xy(traj["obstacle_centers"]))
        if "goal_position" in traj:
            all_items_for_limits.append(_get_xy(traj["goal_position"]).squeeze(0))

    # Harmonize axis limits across subplots
    xlim, ylim = _compute_limits(all_items_for_limits, pad_ratio=0.12)
    for ax in axes:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    return fig

# ---- Optional: log to TensorBoard ----
def log_sampled_trajectories_tb(
    pt_path: str,
    log_dir: str,
    tag: str = "Eval/Trajectories",
    global_step: int = 0,
    **kwargs,
):
    """Create the plot and push it to TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter
    fig = plot_sampled_trajectories(pt_path, **kwargs)
    w = SummaryWriter(log_dir=log_dir)
    w.add_figure(tag, fig, global_step=global_step, close=True)
    w.flush(); w.close()


# Basic: draw 4 random trajectories and save
fig = plot_sampled_trajectories(
    "./complete_dpo_navigation_dataset/complete_trajectories_torch.pt",
    num_trajs=30,
    velocity_every=6,     # put arrows every 6 steps
    velocity_scale=0.8,   # shorten/lenghten arrows
    save_path="sampled_trajs.png",
    suptitle="Dataset samples",
)
