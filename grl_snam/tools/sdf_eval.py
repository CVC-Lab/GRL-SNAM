"""Held-out scoring for the SDF coefficient policy (``grl-snam train``'s ``CoefMLP``).

``grl-snam train`` fits the policy to a reach-goal + no-collision objective on random start/goal
pairs drawn over one SDF. This module holds the pieces that judge it on data it did not train on:

- **The held-out split** (:func:`block_split`): the working region cut into ``blocks x blocks``
  square blocks and a fraction of whole blocks held out, so a held-out start is never a pixel
  next to a training one.
- **Held-out pairs** (:func:`sample_pairs`): starts from a pool, goals ``d_min..d_max`` away in
  free space (the training draw, minus goals inside geometry).
- **The objective's terms** (:func:`objective_terms`): the training loss on fixed pairs, without
  a gradient. Two objectives (:data:`OBJECTIVES`, :func:`loss_terms`): ``task`` (the default since
  docs/learning-validation.md §9) and ``end-distance`` (the objective before it).
- **The sim frame** (:func:`in_frame`, :data:`SIM_METRES_PER_UNIT`): the physics constants are
  numbers in sim units, tuned where one unit is 20 m. ``grl-snam sdf`` frames a map at about 10
  units across its region (Austin's ``nav_sdf.npz``: 300 m per unit), so there the same numbers
  are a 270 m/s robot with a 105 m barrier band. A scene read in the 20 m frame is the same
  geometry, in metres, with the robot at its real size. Pairs are drawn in the npz's own frame and
  converted (:func:`heldout_pairs`), so a held-out pair is the same place in every frame.
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

#: Metres per sim unit in the frame ``grl-snam train``'s physics constants were tuned in: rr 3 m
#: (0.15), ``d_hat`` 0.35 = 7 m, ``vmax`` 0.9 = 18 m/s, the seek carrot 1.8 = 36 m, and libcvc's
#: carrot machine (``sim_world``: stall 0.15 = 3 m, wall-follow 1.6 = 32 m). The procedural city
#: and :mod:`grl_snam.tools.austin`'s story run in it.
SIM_METRES_PER_UNIT = 20.0

#: ``grl-snam train``'s objectives (:func:`loss_terms`).
OBJECTIVES = ("task", "end-distance")


@dataclass
class SdfScene:
    """One ``nav_sdf.npz``: its field and the frame it is in. ``task`` is the scene in the npz's
    own frame when this one was read in another (:func:`in_frame`), else ``None``."""

    field: sdf_nav.SDFField
    phi: np.ndarray
    bounds: list[float]
    center: list[float]
    scale: float
    region: float
    task: SdfScene | None = None

    @property
    def region_n(self) -> float:
        """The working region's half-extent in normalized units."""
        return self.region * self.scale

    @property
    def task_k(self) -> float:
        """Sim units per unit of the frame pairs are drawn in (1 in the npz's own frame)."""
        return 1.0 if self.task is None else self.scale / self.task.scale


def sim_scale(npz_scale: float, metres_per_unit: float) -> float:
    """The sim frame's units per metre: ``1 / metres_per_unit``, or the npz's own scale when
    ``metres_per_unit`` is 0 (the frame ``grl-snam train`` used before §9)."""
    if metres_per_unit <= 0:
        return float(npz_scale)
    s = 1.0 / float(metres_per_unit)
    # An npz already in that frame (its scale a float32 of it) is read as it is.
    return float(npz_scale) if abs(s - float(npz_scale)) <= 1e-6 * s else s


def load_scene(sdf_npz: str, device="cpu", scale: float | None = None) -> SdfScene:
    """``sdf_npz``'s scene, in the npz's own frame or, with ``scale`` (sim units per metre, as a
    checkpoint's ``meta["scale"]``), in that frame (:func:`in_frame`)."""
    d = np.load(sdf_npz)
    bounds = [float(v) for v in d["bounds"]]
    center = [float(v) for v in d["center"]]
    s0 = float(d["scale"])
    field = sdf_nav.SDFField(d["phi"], d["normal_x"], d["normal_y"], bounds, center, s0, device)
    scene = SdfScene(field, d["phi"], bounds, center, s0, float(d["region"]))
    return scene if scale is None else in_frame(scene, float(scale))


def in_frame(scene: SdfScene, scale: float) -> SdfScene:
    """``scene`` read in the sim frame of ``scale`` units per metre. The geometry is the same in
    metres: ``phi`` is multiplied by ``scale / scene.scale``; normals, bounds and region are
    unchanged. The result keeps ``scene`` as its ``task``. The same frame (to float32
    precision) returns ``scene``."""
    if abs(float(scale) - scene.scale) <= 1e-6 * scene.scale:
        return scene
    k = float(scale) / scene.scale
    planes = scene.field.field[0].cpu().numpy()
    phi = (scene.phi.astype(np.float64) * k).astype(np.float32)
    field = sdf_nav.SDFField(
        phi, planes[1], planes[2], scene.bounds, scene.center, float(scale), scene.field.dev
    )
    return SdfScene(field, phi, scene.bounds, scene.center, float(scale), scene.region, scene)


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


def heldout_pairs(scene: SdfScene, blocks: int, held: list[int], n: int, seed: int, *, rr: float):
    """``n`` fixed ``(starts, goals)`` started inside the held-out blocks, in ``scene``'s frame
    (``rr`` too). They are drawn in the npz's own frame (``scene.task``), as ``grl-snam train``
    draws them, and converted, so a pair is the same place in metres in every sim frame."""
    t, k = scene.task or scene, scene.task_k
    rt = rr / k
    pool = free_pool(t, rt)
    _, held_pool = split_pool(pool, t.region_n, blocks, held)
    if not len(held_pool):
        raise ValueError("no free start point lies inside the held-out blocks")
    s, g = sample_pairs(t, held_pool, n, seed, rr=rt)
    dev = scene.field.dev
    return (s * k).to(dev), (g * k).to(dev)


def carrot_of(o, goals, carrot=1.8):
    """The seek carrot: ``min(carrot, d)`` from ``o`` toward ``goals`` (``grl_snam.nav``'s and
    libcvc's ``carrot_step`` seek branch)."""
    dg = goals - o
    dist = dg.norm(dim=-1, keepdim=True)
    return o + dg / (dist + 1e-6) * dist.clamp(max=carrot)


def loss_terms(
    model,
    field,
    starts,
    goals,
    *,
    rr,
    d_hat,
    dt,
    vmax,
    horizon,
    window=0,
    col_grad=False,
    kind="end-distance",
    carrot_m=36.0,
    col_margin_m=0.4,
    col_temp_m=0.4,
    reach_tol_m=15.0,
    per_pair=False,
    v0=None,
    state=None,
    reg_centre=(1.0, 3.0, 4.0),
):
    """``grl-snam train``'s objective on ``(starts, goals)``, as tensors: ``goal``, ``col``,
    ``reg`` (the pull toward a centre, ``(alpha-a)^2 + (beta-b)^2 + (gamma-c)^2``; the bias regime
    ``(1, 3, 4)`` unless ``task``'s ``reg_centre`` says otherwise) and
    ``loss = goal + 3 col + 0.1 reg``. Two objectives (``kind``):

    - ``task`` (docs/learning-validation.md §9): the deployed seek drive's rule each step (the
      carrot ``min(carrot_m, d)`` toward the goal, the coefficients at it), and terms in metres
      (``field.S`` converts each ``_m``): with a constant policy its value is the same in any sim
      frame.
      ``goal`` is the share of the possible progress toward arrival not made: ``1 - (r0 - rT) /
      min(r0, vmax * dt * horizon)``, with ``r`` the distance left to within ``reach_tol_m`` of
      the goal. 0 is arrival, or every step at full speed toward it. Each window's progress
      carries that window's gradient. ``col`` is ``softplus((m - c) / tau)`` of the rollout's
      minimum clearance ``c``, with ``m = col_margin_m`` and ``tau = col_temp_m``. ``reg`` is
      averaged over the steps, around ``reg_centre`` ``(alpha, beta, gamma)``.
    - ``end-distance`` (before §9): the coefficients once, at the start, toward the goal itself;
      ``goal`` the squared distance left after ``horizon`` steps, ``col`` the softplus with
      margin and temperature 0.02 units (6 m on Austin's npz, 0.4 m in a 20 m frame).

    Two settings change the gradient and no value:

    - ``window`` (0: the whole horizon) cuts the rollout into windows of that many steps and
      detaches the state between them, so the backward runs through at most one window
      (truncated BPTT).
    - ``col_grad``: the collision term carries gradient. Without it the clearance is detached
      and ``col`` is a constant to the optimizer.

    ``per_pair`` returns ``[n]`` tensors (``loss`` per pair) in place of their means. ``task``
    only: ``v0`` starts the rollout at that velocity (else at rest), and a dict ``state``
    receives its end (``o``, ``v``, detached), so a leg can be driven across calls. Returns
    ``(terms, (alpha, beta, gamma))``, the coefficients at the start.
    """
    if kind == "task":
        return _task_terms(
            model, field, starts, goals, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax, horizon=horizon,
            window=window, col_grad=col_grad, carrot_m=carrot_m, col_margin_m=col_margin_m,
            col_temp_m=col_temp_m, reach_tol_m=reach_tol_m, per_pair=per_pair, v0=v0,
            state=state, reg_centre=reg_centre,
        )  # fmt: skip
    if kind != "end-distance":
        raise ValueError(f"unknown objective {kind!r} (objectives: {', '.join(OBJECTIVES)})")
    if v0 is not None or state is not None:
        raise ValueError("v0= and state= are the task objective's")
    al, be, ga = model(sdf_nav.coef_feats(field, starts, goals))
    o, v = starts, torch.zeros_like(starts)
    w = window if window > 0 else horizon
    clr = None
    for t0 in range(0, horizon, w):
        if t0:
            o, v = o.detach(), v.detach()
        o, v, c = sdf_nav.sdf_rollout(
            field, o, v, goals, al, be, ga, min(w, horizon - t0), nsub=1, rr=rr, d_hat=d_hat,
            dt=dt, vmax=vmax, clr_grad=col_grad,
        )  # fmt: skip
        clr = c if clr is None else torch.minimum(clr, c)
    if per_pair:
        goal = ((o - goals) ** 2).sum(-1)
        col = F.softplus((0.02 - clr) / 0.02)
        reg = (be - 3.0) ** 2 + (ga - 4.0) ** 2 + (al - 1.0) ** 2
        return {"loss": goal + 3.0 * col + 0.1 * reg, "goal": goal, "col": col, "reg": reg}, (
            al,
            be,
            ga,
        )
    goal = ((o - goals) ** 2).sum(-1).mean()
    col = F.softplus((0.02 - clr) / 0.02).mean()
    reg = ((be - 3.0) ** 2).mean() + ((ga - 4.0) ** 2).mean() + ((al - 1.0) ** 2).mean()
    return {"loss": goal + 3.0 * col + 0.1 * reg, "goal": goal, "col": col, "reg": reg}, (
        al,
        be,
        ga,
    )


def _task_terms(
    model, field, starts, goals, *, rr, d_hat, dt, vmax, horizon, window, col_grad, carrot_m,
    col_margin_m, col_temp_m, reach_tol_m, per_pair, v0=None, state=None,
    reg_centre=(1.0, 3.0, 4.0),
):  # fmt: skip
    """:func:`loss_terms`'s ``task`` objective."""
    ca, cb, cg = (float(x) for x in reg_centre)
    S = field.S
    tol, m, tau, carrot = reach_tol_m * S, col_margin_m * S, col_temp_m * S, carrot_m * S

    def left(p):
        return ((goals - p).norm(dim=-1) - tol).clamp_min(0.0)

    r0 = left(starts)
    possible = torch.minimum(r0, torch.full_like(r0, vmax * dt * horizon))
    o, v = starts, (torch.zeros_like(starts) if v0 is None else v0)
    w = window if window > 0 else horizon
    clr, first = None, None
    prog = torch.zeros_like(r0)
    reg = torch.zeros_like(r0)
    for t0 in range(0, horizon, w):
        if t0:
            o, v = o.detach(), v.detach()
        r_in = left(o)
        for _ in range(min(w, horizon - t0)):
            c = carrot_of(o, goals, carrot)
            al, be, ga = model(sdf_nav.coef_feats(field, o, c))
            if first is None:
                first = (al, be, ga)
            o, v, cl = sdf_nav.sdf_rollout(
                field, o, v, c, al, be, ga, 1, nsub=1, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax,
                clr_grad=col_grad,
            )  # fmt: skip
            clr = cl if clr is None else torch.minimum(clr, cl)
            reg = reg + (be - cb) ** 2 + (ga - cg) ** 2 + (al - ca) ** 2
        prog = prog + (r_in - left(o))
    if state is not None:
        state.update(o=o.detach(), v=v.detach())
    # A pair that starts within reach of its goal has nothing left to make: 0.
    goal = torch.where(possible > 0, 1.0 - prog / possible.clamp_min(1e-9), torch.zeros_like(r0))
    col = F.softplus((m - clr) / tau)
    reg = reg / horizon
    t = {"goal": goal, "col": col, "reg": reg}
    if not per_pair:
        t = {k: x.mean() for k, x in t.items()}
    t["loss"] = t["goal"] + 3.0 * t["col"] + 0.1 * t["reg"]
    return {k: t[k] for k in ("loss", "goal", "col", "reg")}, first


@torch.no_grad()
def objective_terms(
    model, scene: SdfScene, starts, goals, *, rr, d_hat, dt, vmax, horizon, **objective
):
    """:func:`loss_terms` on fixed pairs, as floats, without a gradient (``objective``: its
    ``window`` and ``col_grad``, which change no value)."""
    t, _ = loss_terms(
        model, scene.field, starts, goals, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax,
        horizon=horizon, **objective,
    )  # fmt: skip
    goal, col, reg = float(t["goal"]), float(t["col"]), float(t["reg"])
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
        c = carrot_of(o, goals, carrot)
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


def clean_reach(d: dict) -> np.ndarray:
    """Per pair of a :func:`seek_drive` result: reached the goal with no penetration tick."""
    return d["reached"] & (d["pen_ticks"] == 0)


def drive_summary(d: dict, dt: float) -> dict:
    """A :func:`seek_drive` result reduced to the held-out terms a history records:
    ``clean_reach`` is :func:`clean_reach`'s share, ``pen_frac`` the share of pairs with a
    penetration tick."""
    reached = d["reached"]
    out = {
        "reach": float(reached.mean()),
        "clean_reach": float(clean_reach(d).mean()),
        "pen_frac": float((d["pen_ticks"] > 0).mean()),
        "pen_ticks": float(d["pen_ticks"].mean()),
        "path_m": float(d["path_m"].mean()),
        "min_clear_m": float(d["min_clear_m"].mean()),
        "gap_m": float(d["gap_m"].mean()),
    }
    out["time_to_goal_s"] = (
        float((d["ticks_to_goal"][reached] * dt).mean()) if reached.any() else None
    )
    return out
