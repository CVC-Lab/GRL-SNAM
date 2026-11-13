"""Held-out scoring for the SDF coefficient policy (``grl-snam train``'s ``CoefMLP``).

``grl-snam train`` fits the policy to a reach-goal + no-collision objective on random start/goal
pairs drawn over one SDF. This module holds the pieces that judge it on data it did not train on:

- **The held-out split** (:func:`block_split`): the working region cut into ``blocks x blocks``
  square blocks and a fraction of whole blocks held out, so a held-out start is never a pixel
  next to a training one.
- **Held-out pairs** (:func:`sample_pairs`): starts from a pool, goals ``d_min..d_max`` away in
  free space (the training draw, minus goals inside geometry).
- **The objective's terms** (:func:`objective_terms`): the training loss on fixed pairs, without
  a gradient.
- **The seek drive** (:func:`seek_drive`): the deployed navigator's seek mode
  (:class:`grl_snam.nav.SdfNavigator`: a carrot ``min(1.8, d)`` toward the goal, the policy's
  coefficients at the carrot, one ``sdf_rollout`` step of ``nsub`` substeps) for a whole batch of
  pairs at once. It has no wall-follow escape, so it scores the policy rather than the escape
  machinery. Per pair: reached, time to goal, path length, minimum clearance and penetration
  ticks (clearance below ``-0.5 * rr``, :func:`grl_snam.tools.coef_eval.pen_threshold`).

Everything is in the field's normalized units except where a name says ``_m`` (world metres).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import grl_snam.sdf_nav as sdf_nav


@dataclass
class SdfScene:
    """One ``nav_sdf.npz``: its field and the frame it is in."""

    field: sdf_nav.SDFField
    phi: np.ndarray
    bounds: list[float]
    center: list[float]
    scale: float
    region: float

    @property
    def region_n(self) -> float:
        """The working region's half-extent in normalized units."""
        return self.region * self.scale


def load_scene(sdf_npz: str, device="cpu") -> SdfScene:
    d = np.load(sdf_npz)
    bounds = [float(v) for v in d["bounds"]]
    center = [float(v) for v in d["center"]]
    scale = float(d["scale"])
    field = sdf_nav.SDFField(d["phi"], d["normal_x"], d["normal_y"], bounds, center, scale, device)
    return SdfScene(field, d["phi"], bounds, center, scale, float(d["region"]))


def free_pool(scene: SdfScene, rr: float, margin: float = 0.02) -> np.ndarray:
    """``[N, 2]`` normalized points of the field's grid that are free by ``rr + margin`` and inside
    the working region: ``grl-snam train``'s start pool."""
    mnx, mny, mxx, mxy = scene.bounds
    ny, nx = scene.phi.shape
    gx, gy = np.meshgrid(np.linspace(mnx, mxx, nx), np.linspace(mny, mxy, ny))
    cx, cy = scene.center
    reg = (np.abs(gx - cx) < scene.region) & (np.abs(gy - cy) < scene.region)
    sel = (scene.phi > rr + margin) & reg
    return np.stack([(gx[sel] - cx) * scene.scale, (gy[sel] - cy) * scene.scale], 1).astype(
        np.float32
    )


def block_ids(points: np.ndarray, region_n: float, blocks: int) -> np.ndarray:
    """Each normalized point's block in a ``blocks x blocks`` grid over ``[-region_n, region_n]^2``."""
    cell = 2.0 * region_n / blocks
    ij = np.clip(((points + region_n) / cell).astype(np.int64), 0, blocks - 1)
    return ij[:, 1] * blocks + ij[:, 0]


def block_split(blocks: int, frac: float, seed: int) -> list[int]:
    """The held-out blocks: ``round(frac * blocks^2)`` of them (at least one, never all), drawn
    with ``seed``."""
    n = blocks * blocks
    if not 0.0 < frac < 1.0:
        raise ValueError("a held-out split needs 0 < frac < 1")
    k = min(n - 1, max(1, round(frac * n)))
    return sorted(np.random.default_rng(seed).choice(n, size=k, replace=False).tolist())


def split_pool(pool: np.ndarray, region_n: float, blocks: int, held: list[int]):
    """``(train, held_out)``: the pool's points outside and inside the held-out blocks."""
    inside = np.isin(block_ids(pool, region_n, blocks), held)
    return pool[~inside], pool[inside]


@torch.no_grad()
def sample_pairs(
    scene: SdfScene,
    pool: np.ndarray,
    n: int,
    seed: int,
    *,
    rr: float,
    d_min: float = 1.0,
    d_max: float = 2.0,
    margin: float = 0.02,
):
    """``n`` fixed ``(starts, goals)`` ``[n, 2]`` tensors: a start from ``pool``, a goal at a
    uniform heading and distance ``d_min..d_max`` that is free by ``rr + margin`` and inside the
    working region (redrawn until it is)."""
    rng = np.random.default_rng(seed)
    starts, goals = [], []
    lim = scene.region_n
    while len(starts) < n:
        m = 4 * (n - len(starts))
        s = pool[rng.integers(0, len(pool), m)]
        ang = rng.uniform(0.0, 2.0 * np.pi, m)
        dist = rng.uniform(d_min, d_max, m)
        g = (s + np.stack([np.cos(ang), np.sin(ang)], 1) * dist[:, None]).astype(np.float32)
        phi, _ = scene.field.sample(torch.from_numpy(g).to(scene.field.dev))
        ok = (phi.cpu().numpy() > rr + margin) & (np.abs(g) < lim).all(1)
        starts += list(s[ok])
        goals += list(g[ok])
    dev = scene.field.dev
    return (
        torch.from_numpy(np.stack(starts[:n])).to(dev),
        torch.from_numpy(np.stack(goals[:n])).to(dev),
    )


@torch.no_grad()
def objective_terms(model, scene: SdfScene, starts, goals, *, rr, d_hat, dt, vmax, horizon):
    """``grl-snam train``'s loss terms on fixed pairs: ``goal`` (mean squared end distance after
    ``horizon`` steps), ``col`` (the clearance softplus), ``reg`` (the pull toward the bias
    regime) and their weighted ``loss``."""
    al, be, ga = model(sdf_nav.coef_feats(scene.field, starts, goals))
    oT, _, clr = sdf_nav.sdf_rollout(
        scene.field,
        starts,
        torch.zeros_like(starts),
        goals,
        al,
        be,
        ga,
        horizon,
        nsub=1,
        rr=rr,
        d_hat=d_hat,
        dt=dt,
        vmax=vmax,
    )
    goal = float(((oT - goals) ** 2).sum(-1).mean())
    col = float(F.softplus((0.02 - clr) / 0.02).mean())
    reg = float(((be - 3.0) ** 2).mean() + ((ga - 4.0) ** 2).mean() + ((al - 1.0) ** 2).mean())
    return {"loss": goal + 3.0 * col + 0.1 * reg, "goal": goal, "col": col, "reg": reg}


@torch.no_grad()
def seek_drive(
    model,
    scene: SdfScene,
    starts,
    goals,
    *,
    rr,
    d_hat,
    dt,
    vmax,
    nsub=3,
    ticks=200,
    reach_tol=0.05,
    carrot=1.8,
):
    """Drive every pair with the policy (see the module docstring) and return per-pair numpy
    arrays: ``reached``, ``ticks_to_goal`` (``ticks`` when not reached), ``path_m``,
    ``min_clear_m``, ``pen_ticks``, ``gap_m`` (the distance left at the end), and the
    coefficients' per-pair means ``alpha``/``beta``/``gamma`` over the ticks driven. An agent stops at arrival; its numbers are those up to it.
    """
    from grl_snam.tools.coef_eval import pen_threshold

    field = scene.field
    o = starts.clone()
    v = torch.zeros_like(o)
    n = o.shape[0]
    dev = o.device
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    t_goal = torch.full((n,), ticks, dtype=torch.int64, device=dev)
    path = torch.zeros(n, device=dev)
    worst = torch.full((n,), 1e9, device=dev)
    pen = torch.zeros(n, dtype=torch.int64, device=dev)
    sums = torch.zeros(3, n, device=dev)
    driven = torch.zeros(n, device=dev)
    pen_at = pen_threshold(rr)
    for t in range(ticks):
        live = ~done
        if not bool(live.any()):
            break
        dg = goals - o
        dist = dg.norm(dim=-1, keepdim=True)
        c = o + dg / (dist + 1e-6) * dist.clamp(max=carrot)
        al, be, ga = model(sdf_nav.coef_feats(field, o, c))
        o2, v2, _ = sdf_nav.sdf_rollout(
            field, o, v, c, al, be, ga, 1, nsub=nsub, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax
        )
        m = live.unsqueeze(-1)
        path += torch.where(live, (o2 - o).norm(dim=-1), torch.zeros_like(path))
        o = torch.where(m, o2, o)
        v = torch.where(m, v2, v)
        lf = live.float()
        sums += torch.stack([al, be, ga]) * lf
        driven += lf
        phi, _ = field.sample(o)
        clr = phi - rr
        worst = torch.where(live, torch.minimum(worst, clr), worst)
        pen += (live & (clr < pen_at)).long()
        arrived = live & ((goals - o).norm(dim=-1) < reach_tol)
        t_goal = torch.where(arrived, torch.full_like(t_goal, t + 1), t_goal)
        done |= arrived
    means = (sums / driven.clamp_min(1.0)).cpu().numpy()
    S = scene.scale
    return {
        "reached": done.cpu().numpy(),
        "ticks_to_goal": t_goal.cpu().numpy(),
        "path_m": (path / S).cpu().numpy(),
        "min_clear_m": (worst / S).cpu().numpy(),
        "pen_ticks": pen.cpu().numpy(),
        "gap_m": ((goals - o).norm(dim=-1) / S).cpu().numpy(),
        "alpha": means[0],
        "beta": means[1],
        "gamma": means[2],
    }


def drive_summary(d: dict, dt: float) -> dict:
    """A :func:`seek_drive` result reduced to the held-out terms a history records."""
    reached = d["reached"]
    out = {
        "reach": float(reached.mean()),
        "pen_ticks": float(d["pen_ticks"].mean()),
        "path_m": float(d["path_m"].mean()),
        "min_clear_m": float(d["min_clear_m"].mean()),
        "gap_m": float(d["gap_m"].mean()),
    }
    out["time_to_goal_s"] = (
        float((d["ticks_to_goal"][reached] * dt).mean()) if reached.any() else None
    )
    return out
