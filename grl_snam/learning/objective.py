"""Held-out objectives: a trainer's own loss terms, per held-out unit, for any arm.

A learning history holds the held-out loss terms of the run it records, and nothing else: its
step 0 is the only other policy it can be compared with. To test the loss family against the
untrained AND the constant arm, as the task metric is, every arm has to be scored on the same
held-out units with the trainer's own objective. This module does that for the two trainers whose
arms :mod:`grl_snam.learning.arms` builds:

- **``grl-snam train``** (:func:`sdf_terms`): the SDF policy's objective on fixed held-out pairs
  started inside the run's held-out blocks (:func:`sdf_heldout_pairs`). Per pair: ``goal``
  (squared end distance after ``horizon`` steps of the training rollout), ``col`` (the clearance
  softplus), ``reg`` (the pull toward the bias regime) and ``loss = goal + 3 col + 0.1 reg``. The
  mean over pairs is :func:`grl_snam.tools.sdf_eval.objective_terms`, the ``obj_<term>`` a
  ``--val-frac`` history logs.
- **coef-train** (:func:`coef_terms`): its loss on held-out starts and goals in the procedural
  city, drawn with ``--heldout-seed`` (``--heldout-agents`` of them), rolled out for
  ``--horizon`` steps on ``--grid``. Per start: the training loop's windows (the graph is cut
  every ``window`` steps), each window's terms as the trainer applies them, averaged over the
  windows as the history's step record averages them. ``rollout="surrogate"`` is
  :func:`grl_snam.tools.coef_train.train`'s (``goal`` the end distance, ``coll`` the penetration
  depth, ``w_coll`` 6); ``"bicycle"`` is :func:`grl_snam.tools.coef_train.train_bicycle`'s, with
  no material (``goal`` over the region's half-extent, ``coll`` the margin shortfall over
  ``d_hat``, ``w_coll`` 3). ``loss = goal + coll``.

Scoring takes no gradient, puts the net in eval mode for the call (its mode is restored), and
forks the torch RNG, so scoring a policy between training steps does not change the run.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SDF_TERMS: tuple[str, ...] = ("loss", "goal", "col", "reg")
COEF_TERMS: tuple[str, ...] = ("loss", "goal", "coll")
ROLLOUTS: tuple[str, ...] = ("surrogate", "bicycle")
#: Per rollout: coef-train's collision weight and BPTT window (``train`` / ``train_bicycle``).
COEF_DEFAULTS = {"surrogate": {"w_coll": 6.0, "window": 7}, "bicycle": {"w_coll": 3.0, "window": 6}}
#: The bicycle trainer's vehicle (``train_bicycle``'s ``kw``).
BICYCLE = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8, allow_reverse=True)
#: ``grl-snam train``'s physics defaults (``train_sdf``), for when no arm carries its ``meta``.
SDF_DEFAULTS = dict(robot_radius_world=3.0, d_hat=0.35, dt=0.06, vmax=0.9, horizon=28)


@contextmanager
def scoring(model):
    """No gradient, a forked torch RNG, and ``model`` in eval mode (restored on exit)."""
    import torch

    was = model.training
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model.eval()
        try:
            yield model
        finally:
            model.train(was)


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- grl-snam train


def read_header(path: str | Path) -> dict:
    """The first record of a JSON-lines history (``.jsonl`` or ``.jsonl.gz``)."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"{path}: empty history")


def sdf_split(
    history: str | Path | None = None,
    *,
    val_frac: float | None = None,
    val_seed: int = 0,
    val_blocks: int = 8,
) -> tuple[int, list[int]]:
    """``(blocks, held_out_blocks)``: from a ``grl-snam train --val-frac`` history's header, else
    drawn as that run drew them (:func:`grl_snam.tools.sdf_eval.block_split`)."""
    if history is not None:
        hdr = read_header(history)
        if "heldout_blocks" not in hdr:
            raise ValueError(f"{history}: the header names no held-out blocks (no --val-frac)")
        return int(hdr["val_blocks"]), [int(b) for b in hdr["heldout_blocks"]]
    if val_frac is None:
        raise ValueError("a held-out split needs a history or val_frac")
    from ..tools.sdf_eval import block_split

    return int(val_blocks), block_split(val_blocks, val_frac, val_seed)


def sdf_meta(scene, ckpt_meta: Mapping | None = None) -> dict:
    """The rollout settings: a checkpoint's ``meta`` when there is one, else ``train_sdf``'s
    defaults on ``scene``'s scale."""
    if ckpt_meta:
        return {k: ckpt_meta[k] for k in ("rr", "d_hat", "dt", "vmax", "horizon")}
    d = SDF_DEFAULTS
    return {
        "rr": d["robot_radius_world"] * scene.scale,
        "d_hat": d["d_hat"],
        "dt": d["dt"],
        "vmax": d["vmax"],
        "horizon": d["horizon"],
    }


def sdf_heldout_pairs(scene, blocks: int, held: list[int], n: int, seed: int, *, rr: float):
    """``n`` fixed ``(starts, goals)`` started inside the held-out blocks (the training draw)."""
    from ..tools import sdf_eval

    pool = sdf_eval.free_pool(scene, rr)
    _, held_pool = sdf_eval.split_pool(pool, scene.region_n, blocks, held)
    if not len(held_pool):
        raise ValueError("no free start point lies inside the held-out blocks")
    return sdf_eval.sample_pairs(scene, held_pool, n, seed, rr=rr)


def sdf_terms(
    model, scene, starts, goals, *, rr, d_hat, dt, vmax, horizon
) -> dict[str, np.ndarray]:
    """``grl-snam train``'s loss terms per pair (``[n]`` arrays, see the module docstring)."""
    import torch.nn.functional as F

    import grl_snam.sdf_nav as sdf_nav

    with scoring(model):
        al, be, ga = model(sdf_nav.coef_feats(scene.field, starts, goals))
        oT, _, clr = sdf_nav.sdf_rollout(
            scene.field, starts, starts.new_zeros(starts.shape), goals, al, be, ga, int(horizon),
            nsub=1, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax,
        )  # fmt: skip
        goal = ((oT - goals) ** 2).sum(-1)
        col = F.softplus((0.02 - clr) / 0.02)
        reg = (be - 3.0) ** 2 + (ga - 4.0) ** 2 + (al - 1.0) ** 2
        loss = goal + 3.0 * col + 0.1 * reg
    return {"loss": _np(loss), "goal": _np(goal), "col": _np(col), "reg": _np(reg)}


# --------------------------------------------------------------------------- coef-train


def coef_scene(grid: int = 96):
    """coef-train's world on ``grid``: ``(field, meta, rand_on)``."""
    from ..tools.coef_train import _scene

    return _scene(grid, 0)


def coef_terms(
    model,
    *,
    grid: int = 96,
    horizon: int = 28,
    n: int = 200,
    seed: int = 123,
    rollout: str = "surrogate",
    w_coll: float | None = None,
    window: int | None = None,
    scene=None,
) -> dict[str, np.ndarray]:
    """coef-train's loss terms per held-out start (``[n]`` arrays, see the module docstring).
    ``scene`` is a :func:`coef_scene` triple to reuse across arms."""
    import torch

    import grl_snam.sdf_nav as sdf_nav

    if rollout not in ROLLOUTS:
        raise ValueError(f"rollout {rollout!r}: one of {', '.join(ROLLOUTS)}")
    if getattr(model, "use_mu", False) or getattr(model, "use_risk", False):
        raise ValueError(
            "the held-out coef-train objective is the geometry-only one: a grip or risk net "
            "needs its friction field or material grid"
        )
    w_coll = COEF_DEFAULTS[rollout]["w_coll"] if w_coll is None else float(w_coll)
    window = COEF_DEFAULTS[rollout]["window"] if window is None else int(window)
    if horizon < 1 or window < 1:
        raise ValueError("horizon and window must be >= 1")
    field, meta, rand_on = coef_scene(grid) if scene is None else scene
    rr, d_hat, dt, vmax = meta["rr"], meta["d_hat"], meta["dt"], meta["vmax"]
    rng = np.random.default_rng(seed)
    o = torch.from_numpy(rand_on(n, rng))
    goal = torch.from_numpy(rand_on(n, rng))
    bicycle = rollout == "bicycle"
    if bicycle:
        th = torch.from_numpy(rng.uniform(-np.pi, np.pi, n).astype(np.float32))
        sp = torch.from_numpy(rng.uniform(0.0, float(vmax), n).astype(np.float32))
        region_n = float(meta["region"]) * float(meta["scale"])
        d_safe = float(d_hat)
    else:
        v = torch.zeros(n, 2)
    goal_sum = torch.zeros(n, dtype=torch.float64)
    coll_sum = torch.zeros(n, dtype=torch.float64)
    windows = 0
    with scoring(model):
        coll = torch.zeros(n)
        for t in range(horizon):
            al, be, ga = model(sdf_nav.coef_feats(field, o, goal))
            if bicycle:
                o, th, sp, _ = sdf_nav.bicycle_rollout(
                    field, o, th, sp, goal, al, be, ga, 1, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax,
                    **BICYCLE,
                )  # fmt: skip
                phi, _ = field.sample(o)
                coll = coll + torch.relu(d_safe - (phi - rr)) / d_safe
            else:
                o, v, _ = sdf_nav.sdf_rollout(
                    field, o, v, goal, al, be, ga, 1, rr=rr, d_hat=d_hat, dt=dt, vmax=vmax
                )
                phi, _ = field.sample(o)
                coll = coll + torch.relu(rr - phi)
            if (t + 1) % window == 0 or t == horizon - 1:
                g = (goal - o).norm(dim=1)
                goal_sum += (g / region_n if bicycle else g).double()
                coll_sum += (w_coll * coll / window).double()
                windows += 1
                coll = torch.zeros(n)
    goal_t = (goal_sum / windows).numpy()
    coll_t = (coll_sum / windows).numpy()
    return {"loss": goal_t + coll_t, "goal": goal_t, "coll": coll_t}


# --------------------------------------------------------------------------- per arm


def per_unit(terms: Mapping[str, np.ndarray]) -> dict[str, dict[int, float]]:
    """``{term: {unit: value}}`` from per-unit arrays, the unit being the array index."""
    return {t: {i: float(x) for i, x in enumerate(v)} for t, v in terms.items()}
