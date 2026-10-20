"""Belief (observed) material: material planes built from what a belief plane has seen.

The numpy twin of libcvc's ``cvc::nav`` ``belief_material.h`` (its spec is libcvc's
``docs/roadmap/NAV-BELIEF-MATERIAL-ROADMAP.md``). A vehicle that only senses within a fixed
range should navigate on the material it has observed, not on the whole map, so the truth
rasters (``risk_raw``, ``hard``) are kept where the plane's sensor has seen and replaced by
pinned values everywhere else. :class:`~grl_snam.material.MaterialGrid` and
:class:`~grl_snam.material.MaterialRuntime` then run unchanged on the result:

* :func:`belief_material_inputs` — the truth where seen, ``unknown_risk`` / ``unknown_hard``
  elsewhere (defaults 0 / False: an unseen cell exerts no force of its own);
* :func:`belief_gate_occ` — what the witness gate's surface needs besides the material:
  ``occ | ~seen`` while ``unknown_gate_hard`` (the default), so a ray through unseen ground
  never certifies a detour. ``MaterialRuntime.update_occ`` ORs in ``hard_b``;
* :func:`belief_gate_surface` — that surface and its metres clearance explicitly, the
  ``belief_gate_surface`` kernel's twin;
* :func:`seen_disc` — the optional range-disc seen rule (no occlusion);
* :func:`belief_material_grid` / :func:`belief_material_runtime` — the composition.

The default seen mask is the sense ray-cast's field of view:
:meth:`grl_snam.belief.BeliefGrid.track_material` keeps it (at the default range it is
``ever_seen``). Everything here is BIT-identical to libcvc; libcvc's
``belief_material_golden.json`` is the fixture (``tests/test_belief_material_parity.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import grl_snam.sdf_nav as sdf_nav

from .material import MaterialGrid, MaterialParams, MaterialRuntime

__all__ = [
    "BeliefMaterialParams",
    "belief_material_inputs",
    "belief_gate_occ",
    "belief_gate_surface",
    "seen_disc",
    "belief_material_grid",
    "belief_material_runtime",
]


@dataclass(frozen=True)
class BeliefMaterialParams:
    """The pinned unseen values and the seen rule (libcvc ``belief_material_params``).

    ``seen_rule`` is ``"raycast"`` (the sense ray-cast's field of view, occluded; the
    default) or ``"disc"`` (every corner sample within the material range of the agent, no
    occlusion: it sees through walls). ``material_range_m`` 0 means the sense range."""

    unknown_risk: float = 0.0
    unknown_hard: bool = False
    unknown_gate_hard: bool = True
    seen_rule: str = "raycast"
    material_range_m: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.unknown_risk) or not 0.0 <= self.unknown_risk <= 1.0:
            raise ValueError("unknown_risk must be in [0, 1]")
        if self.seen_rule not in ("raycast", "disc"):
            raise ValueError("seen_rule must be 'raycast' or 'disc'")
        if not math.isfinite(self.material_range_m) or self.material_range_m < 0.0:
            raise ValueError("material_range_m must be finite and >= 0")


def belief_material_inputs(
    risk_raw: np.ndarray, hard: np.ndarray, seen: np.ndarray, params: BeliefMaterialParams
) -> tuple[np.ndarray, np.ndarray]:
    """``(risk_b float32, hard_b bool)``: the truth where ``seen``, the pins elsewhere."""
    seen = np.asarray(seen, bool)
    risk_b = np.where(seen, np.asarray(risk_raw, np.float32), np.float32(params.unknown_risk))
    hard_b = np.where(seen, np.asarray(hard).astype(bool), bool(params.unknown_hard))
    return risk_b.astype(np.float32), hard_b


def belief_gate_occ(occ: np.ndarray, seen: np.ndarray, params: BeliefMaterialParams) -> np.ndarray:
    """The occupancy to hand ``MaterialRuntime.update_occ`` on belief planes:
    ``occ | ~seen`` while ``unknown_gate_hard``, else ``occ``."""
    occ = np.asarray(occ, bool)
    return occ | ~np.asarray(seen, bool) if params.unknown_gate_hard else occ.copy()


def belief_gate_surface(
    hard_b: np.ndarray,
    occ: np.ndarray | None,
    seen: np.ndarray,
    cell_w: float,
    params: BeliefMaterialParams,
) -> tuple[np.ndarray, np.ndarray]:
    """``(gate_hard bool, clear_m float32)``: ``hard_b | occ | (unknown_gate_hard & ~seen)``
    and its metres clearance, ``MaterialRuntime``'s chain (f64, one f32 store)."""
    gate = np.asarray(hard_b, bool).copy()
    if occ is not None:
        gate |= np.asarray(occ, bool)
    if params.unknown_gate_hard:
        gate |= ~np.asarray(seen, bool)
    clear = (np.sqrt(sdf_nav._edt2(gate)) * float(cell_w)).astype(np.float32)
    return gate, clear


def seen_disc(seen: np.ndarray, pos_world, radius_m: float, bounds) -> bool:
    """``seen_rule="disc"`` for one agent: ORs into ``seen`` (bool, in place) every corner
    sample within ``radius_m`` of the world point ``pos_world`` (not snapped to a cell), on
    ``np.linspace`` sample coordinates. Returns True iff some cell changed."""
    ny, nx = seen.shape
    if not (math.isfinite(radius_m) and radius_m >= 0.0):
        return False
    px, py = float(pos_world[0]), float(pos_world[1])
    if not (math.isfinite(px) and math.isfinite(py)):
        return False
    xs = np.linspace(float(bounds[0]), float(bounds[2]), nx, dtype=np.float64)
    ys = np.linspace(float(bounds[1]), float(bounds[3]), ny, dtype=np.float64)
    disc = (xs[None, :] - px) ** 2 + (ys[:, None] - py) ** 2 <= radius_m * radius_m
    grew = bool((disc & ~seen).any())
    seen |= disc
    return grew


def belief_material_grid(
    risk_raw: np.ndarray,
    hard: np.ndarray,
    seen: np.ndarray,
    bounds,
    center,
    scale: float,
    *,
    sigma: float = 1.0,
    params: MaterialParams | None = None,
    belief: BeliefMaterialParams = BeliefMaterialParams(),
) -> MaterialGrid:
    """The :class:`MaterialGrid` of the belief inputs (its derived planes are the belief
    planes: the blur and the hard-distance field see only what was seen)."""
    risk_b, hard_b = belief_material_inputs(risk_raw, hard, seen, belief)
    return MaterialGrid(risk_b, hard_b, bounds, center, scale, sigma=sigma, params=params)


def belief_material_runtime(
    risk_raw: np.ndarray,
    hard: np.ndarray,
    seen: np.ndarray,
    occ: np.ndarray,
    bounds,
    center,
    scale: float,
    *,
    sigma: float = 1.0,
    params: MaterialParams | None = None,
    belief: BeliefMaterialParams = BeliefMaterialParams(),
) -> MaterialRuntime:
    """A :class:`MaterialRuntime` on belief planes: the belief grid, with the gate's surface
    built from ``occ`` (the plane's own planning raster) and the unseen cells."""
    grid = belief_material_grid(
        risk_raw, hard, seen, bounds, center, scale, sigma=sigma, params=params, belief=belief
    )
    rt = MaterialRuntime(grid)
    rt.update_occ(belief_gate_occ(occ, seen, belief))
    return rt
