"""Canonical model inputs for a range-limited belief-material controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .belief import BeliefGrid
from .belief_material import (
    BeliefMaterialParams,
    belief_gate_occ,
    belief_material_runtime,
    material_patch_belief,
)
from .material import GateResult, MaterialParams, MaterialRuntime
from .material_nav import CoefEnergyNetMaterial, integrate_surrogate_material

__all__ = [
    "BeliefMaterialCoefficients",
    "BeliefMaterialControlCommand",
    "BeliefMaterialController",
    "BeliefMaterialRollout",
    "BeliefMaterialSenseUpdate",
    "BeliefMaterialSensor",
    "belief_material_ray_count",
    "build_belief_material_controller_inputs",
]


def belief_material_ray_count(fov_rad: float, range_m: float, cell_size_m: float) -> int:
    """Minimum shared ray count whose far-edge arc spacing is at most one cell."""
    if not math.isfinite(fov_rad) or fov_rad <= 0.0:
        raise ValueError("fov_rad must be finite and > 0")
    if not math.isfinite(range_m) or range_m < 0.0:
        raise ValueError("range_m must be finite and >= 0")
    if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and > 0")
    return max(1, math.ceil(fov_rad * range_m / cell_size_m))


@dataclass(frozen=True)
class BeliefMaterialSenseUpdate:
    """Observable effects of one belief-material sensing tick."""

    belief_flips: int
    material_grew: bool
    runtime_rebuilt: bool
    gate_updated: bool
    n_rays: int


class BeliefMaterialSensor:
    """Ray-cast sensing and belief-material runtime rebuild coordination."""

    def __init__(
        self,
        belief_grid: BeliefGrid,
        truth_occupancy: np.ndarray,
        risk_raw: np.ndarray,
        hard: np.ndarray,
        center,
        scale: float,
        *,
        sigma: float = 1.0,
        material_params: MaterialParams | None = None,
        belief_params: BeliefMaterialParams = BeliefMaterialParams(),
        unknown_occupancy: str = "optimistic",
        occupancy_p_threshold: float = 0.5,
        occupancy_band: float = 0.15,
    ):
        if belief_params.seen_rule != "raycast":
            raise ValueError("BeliefMaterialSensor requires seen_rule='raycast'")
        shape = belief_grid.logodds.shape
        self.truth_occupancy = np.asarray(truth_occupancy, dtype=bool)
        self.risk_raw = np.asarray(risk_raw, dtype=np.float32)
        self.hard = np.asarray(hard, dtype=bool)
        for name, plane in (
            ("truth_occupancy", self.truth_occupancy),
            ("risk_raw", self.risk_raw),
            ("hard", self.hard),
        ):
            if plane.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {plane.shape}")

        self.belief_grid = belief_grid
        self.center = (float(center[0]), float(center[1]))
        self.scale = float(scale)
        self.sigma = float(sigma)
        self.material_params = material_params
        self.belief_params = belief_params
        self.unknown_occupancy = unknown_occupancy
        self.occupancy_p_threshold = float(occupancy_p_threshold)
        self.occupancy_band = float(occupancy_band)

        if self.belief_grid.material_seen is None:
            self.belief_grid.track_material(belief_params.material_range_m)
        elif self.belief_grid.material_range_m != belief_params.material_range_m:
            raise ValueError(
                "belief_grid material_range_m does not match belief_params.material_range_m"
            )

        self._last_occupancy = self._planning_occupancy()
        self.runtime = self._build_runtime(self._last_occupancy)

    @property
    def cell_size_m(self) -> float:
        """Conservative cell size for ray coverage on an anisotropic raster."""
        bg = self.belief_grid
        cell_w = (bg.mxx - bg.mnx) / (bg.nx - 1)
        cell_h = (bg.mxy - bg.mny) / (bg.ny - 1)
        return min(cell_w, cell_h)

    def _material_seen(self) -> np.ndarray:
        seen = self.belief_grid.material_seen
        if seen is None:  # guarded in __init__; keeps the type invariant explicit
            raise RuntimeError("belief material tracking is not enabled")
        return seen

    def _planning_occupancy(self, dynamic_occupancy=None) -> np.ndarray:
        occ = self.belief_grid.to_occupancy(
            unknown=self.unknown_occupancy,
            p_thresh=self.occupancy_p_threshold,
            band=self.occupancy_band,
        )
        if dynamic_occupancy is None:
            return occ
        dynamic = np.asarray(dynamic_occupancy, dtype=bool)
        if dynamic.shape != occ.shape:
            raise ValueError(f"dynamic_occupancy must have shape {occ.shape}, got {dynamic.shape}")
        return occ | dynamic

    def _build_runtime(self, occupancy: np.ndarray) -> MaterialRuntime:
        bg = self.belief_grid
        return belief_material_runtime(
            self.risk_raw,
            self.hard,
            self._material_seen(),
            occupancy,
            (bg.mnx, bg.mny, bg.mxx, bg.mxy),
            self.center,
            self.scale,
            sigma=self.sigma,
            params=self.material_params,
            belief=self.belief_params,
        )

    def sense(
        self,
        position_world,
        *,
        range_m: float,
        fov_rad: float = 2.0 * math.pi,
        heading_rad: float = 0.0,
        dynamic_occupancy: np.ndarray | None = None,
    ) -> BeliefMaterialSenseUpdate:
        """Run one ray-cast tick and update only the derived surfaces that changed."""
        n_rays = belief_material_ray_count(fov_rad, range_m, self.cell_size_m)
        flips = self.belief_grid.sense(
            self.truth_occupancy,
            position_world,
            range_m=range_m,
            n_rays=n_rays,
            fov_rad=fov_rad,
            heading_rad=heading_rad,
        )
        occupancy = self._planning_occupancy(dynamic_occupancy)
        occupancy_changed = not np.array_equal(occupancy, self._last_occupancy)
        material_grew = bool(self.belief_grid.material_grew)

        if material_grew:
            self.runtime = self._build_runtime(occupancy)
        elif occupancy_changed:
            self.runtime.update_occ(
                belief_gate_occ(occupancy, self._material_seen(), self.belief_params)
            )
        self._last_occupancy = occupancy.copy()

        return BeliefMaterialSenseUpdate(
            belief_flips=flips,
            material_grew=material_grew,
            runtime_rebuilt=material_grew,
            gate_updated=material_grew or occupancy_changed,
            n_rays=n_rays,
        )


@dataclass(frozen=True)
class BeliefMaterialCoefficients:
    """The six outputs of :class:`CoefEnergyNetMaterial` for one agent."""

    alphas: np.ndarray
    beta: float
    gamma: float
    lam_soft: float
    lam_hard: float
    mu_lat: float


@dataclass(frozen=True)
class BeliefMaterialControlCommand:
    """Predicted and witness-gated coefficients for one controller tick."""

    predicted: BeliefMaterialCoefficients
    effective: BeliefMaterialCoefficients
    gate: GateResult | None


@dataclass(frozen=True)
class BeliefMaterialRollout:
    """One-agent output from the shared material surrogate integrator."""

    position: np.ndarray
    velocity: np.ndarray
    min_clearance: float
    cumulative_risk: float
    hard_count: float
    arc_length: float


class BeliefMaterialController:
    """Inference-only adapter from canonical numpy inputs to the material model."""

    def __init__(
        self,
        model: CoefEnergyNetMaterial,
        *,
        patch_size: int = 25,
        patch_extent_m: float = 100.0,
        feature_scale_m: float = 20.0,
        belief: BeliefMaterialParams = BeliefMaterialParams(),
    ):
        if model.risk_channels != 3:
            raise ValueError("BeliefMaterialController requires a 3-channel CoefEnergyNetMaterial")
        if patch_size < 2:
            raise ValueError("patch_size must be >= 2")
        model_patch_size = int(getattr(model, "patch_size", patch_size))
        if model_patch_size != patch_size:
            raise ValueError(
                f"model patch_size is {model_patch_size}, controller patch_size is {patch_size}"
            )
        if not math.isfinite(patch_extent_m) or patch_extent_m <= 0.0:
            raise ValueError("patch_extent_m must be finite and > 0")
        if not math.isfinite(feature_scale_m) or feature_scale_m <= 0.0:
            raise ValueError("feature_scale_m must be finite and > 0")
        if belief.seen_rule != "raycast":
            raise ValueError("BeliefMaterialController requires belief.seen_rule='raycast'")
        self.model = model
        self.patch_size = int(patch_size)
        self.patch_extent_m = float(patch_extent_m)
        self.feature_scale_m = float(feature_scale_m)
        self.belief = belief
        self.model.eval()

    @classmethod
    def from_cvcnm(
        cls,
        path,
        *,
        device=None,
        dtype: torch.dtype = torch.float32,
        patch_size: int = 25,
        patch_extent_m: float = 100.0,
        feature_scale_m: float = 20.0,
        belief: BeliefMaterialParams = BeliefMaterialParams(),
    ) -> BeliefMaterialController:
        """Load and validate a belief-material ``.cvcnm`` deployment checkpoint.

        Loading goes through GRL-SNAM's shared ``.cvcnm`` reader. The version-2 belief block is
        then checked against the controller's sensing and unknown-cell semantics before any
        inference can run.
        """
        from .material_train_torch import load_cvcnm

        model = load_cvcnm(path, device=device, dtype=dtype)
        header = getattr(model, "belief_header", None)
        if model.risk_channels != 3 or header is None:
            raise ValueError(f"{path}: belief controller requires a 3-channel .cvcnm version 2")

        expected_seen_rule = {"raycast": 0, "disc": 1}[belief.seen_rule]
        mismatches = []

        def check(name, actual, expected) -> None:
            if actual != expected:
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")

        def check_float(name, actual, expected) -> None:
            if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6):
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")

        check("risk_channels", header.risk_channels, 3)
        check("seen_rule", header.seen_rule, expected_seen_rule)
        check_float("unknown_risk", header.unknown_risk, belief.unknown_risk)
        check("unknown_hard", bool(header.unknown_hard), belief.unknown_hard)
        check("unknown_gate_hard", bool(header.unknown_gate_hard), belief.unknown_gate_hard)
        check_float("material_range_m", header.material_range_m, belief.material_range_m)
        check_float("patch_extent_m", header.patch_extent_m, patch_extent_m)
        check("patch_size", int(model.patch_size), int(patch_size))
        if mismatches:
            raise ValueError(f"{path}: incompatible belief checkpoint: {', '.join(mismatches)}")

        return cls(
            model,
            patch_size=patch_size,
            patch_extent_m=patch_extent_m,
            feature_scale_m=feature_scale_m,
            belief=belief,
        )

    def predict(
        self,
        obs_feats: np.ndarray,
        obs_mask: np.ndarray,
        goal_feats: np.ndarray,
        risk_patch: np.ndarray,
    ) -> BeliefMaterialCoefficients:
        """Run one inference step on unbatched inputs from the canonical builder."""
        obs = np.asarray(obs_feats)
        mask = np.asarray(obs_mask, dtype=bool)
        goal = np.asarray(goal_feats)
        patch = np.asarray(risk_patch)

        if obs.ndim != 2 or obs.shape[1:] != (6,):
            raise ValueError(f"obs_feats must have shape (N, 6), got {obs.shape}")
        if mask.shape != (obs.shape[0],):
            raise ValueError(f"obs_mask must have shape ({obs.shape[0]},), got {mask.shape}")
        if goal.shape != (4,):
            raise ValueError(f"goal_feats must have shape (4,), got {goal.shape}")
        expected_patch_shape = (3, self.patch_size, self.patch_size)
        if patch.shape != expected_patch_shape:
            raise ValueError(
                f"risk_patch must have shape {expected_patch_shape}, got {patch.shape}"
            )

        parameter = next(self.model.parameters())
        device, dtype = parameter.device, parameter.dtype
        with torch.inference_mode():
            alphas, beta, gamma, lam_soft, lam_hard, mu_lat = self.model(
                torch.as_tensor(obs, device=device, dtype=dtype).unsqueeze(0),
                torch.as_tensor(mask, device=device, dtype=torch.bool).unsqueeze(0),
                torch.as_tensor(goal, device=device, dtype=dtype).unsqueeze(0),
                torch.as_tensor(patch, device=device, dtype=dtype).unsqueeze(0),
            )

        return BeliefMaterialCoefficients(
            alphas=alphas[0].detach().cpu().numpy().copy(),
            beta=float(beta[0].item()),
            gamma=float(gamma[0].item()),
            lam_soft=float(lam_soft[0].item()),
            lam_hard=float(lam_hard[0].item()),
            mu_lat=float(mu_lat[0].item()),
        )

    def decide(
        self,
        position_world,
        goal_world,
        obstacle_centers_world,
        obstacle_radii,
        obstacle_weights,
        runtime: MaterialRuntime,
        seen: np.ndarray,
    ) -> BeliefMaterialControlCommand:
        """Predict coefficients with the validated belief config, then gate ``lam_soft``."""
        inputs = build_belief_material_controller_inputs(
            position_world,
            goal_world,
            obstacle_centers_world,
            obstacle_radii,
            obstacle_weights,
            runtime,
            seen,
            patch_extent_m=self.patch_extent_m,
            patch_size=self.patch_size,
            feature_scale_m=self.feature_scale_m,
            belief=self.belief,
        )
        predicted = self.predict(*inputs)

        if runtime.params.gate_enabled:
            gate = runtime.gate(position_world, goal_world)
            lam_soft = predicted.lam_soft if gate.active else 0.0
        else:
            runtime.last_gate = None
            gate = None
            lam_soft = predicted.lam_soft

        effective = BeliefMaterialCoefficients(
            alphas=predicted.alphas,
            beta=predicted.beta,
            gamma=predicted.gamma,
            lam_soft=lam_soft,
            lam_hard=predicted.lam_hard,
            mu_lat=predicted.mu_lat,
        )
        return BeliefMaterialControlCommand(predicted=predicted, effective=effective, gate=gate)

    def rollout(
        self,
        command: BeliefMaterialControlCommand,
        position,
        velocity,
        goal,
        obstacle_centers,
        obstacle_radii,
        rollout_patch,
        runtime: MaterialRuntime,
        *,
        d_hat: float,
        dt: float,
        steps: int,
        robot_radius: float = 0.0,
        margin_factor: float = 0.5,
        mass: float = 1.0,
        ext=None,
    ) -> BeliefMaterialRollout:
        """Execute ``command`` with GRL-SNAM's existing material surrogate.

        ``position``, ``velocity``, ``goal``, obstacles, radii, and ``rollout_patch`` must use the
        same coordinate frame. The patch is the existing unmasked six-plane belief rollout patch
        ``[risk, phi, grad_r_x, grad_r_y, grad_phi_x, grad_phi_y]``; it is intentionally distinct
        from the masked three-channel model patch. ``mu_lat`` is retained in the command but is not
        used by this off-highway point-mass integrator.
        """
        position = np.asarray(position)
        velocity = np.asarray(velocity)
        goal = np.asarray(goal)
        centers = np.asarray(obstacle_centers)
        radii = np.asarray(obstacle_radii)
        patch = np.asarray(rollout_patch)
        if position.shape != (2,):
            raise ValueError(f"position must have shape (2,), got {position.shape}")
        if velocity.shape != (2,):
            raise ValueError(f"velocity must have shape (2,), got {velocity.shape}")
        if goal.shape != (2,):
            raise ValueError(f"goal must have shape (2,), got {goal.shape}")
        if centers.size == 0:
            centers = np.empty((0, 2), dtype=np.float32)
        elif centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(f"obstacle_centers must have shape (N, 2), got {centers.shape}")
        n_obstacles = centers.shape[0]
        if radii.shape != (n_obstacles,):
            raise ValueError(f"obstacle_radii must have shape ({n_obstacles},), got {radii.shape}")
        if command.effective.alphas.shape != (n_obstacles,):
            raise ValueError(
                "command alphas must match the number of obstacles: "
                f"{command.effective.alphas.shape} vs ({n_obstacles},)"
            )
        if patch.ndim != 3 or patch.shape[0] != 6 or min(patch.shape[1:]) < 2:
            raise ValueError(f"rollout_patch must have shape (6, H>=2, W>=2), got {patch.shape}")
        if not math.isfinite(d_hat) or d_hat <= 0.0:
            raise ValueError("d_hat must be finite and > 0")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0")
        if steps < 0:
            raise ValueError("steps must be >= 0")

        parameter = next(self.model.parameters())
        device, dtype = parameter.device, parameter.dtype

        def tensor(value):
            return torch.as_tensor(value, device=device, dtype=dtype)

        coefficients = command.effective
        outputs = integrate_surrogate_material(
            tensor(position).unsqueeze(0),
            tensor(velocity).unsqueeze(0),
            tensor(goal).unsqueeze(0),
            tensor(centers).unsqueeze(0),
            tensor(radii).unsqueeze(0),
            torch.ones((1, n_obstacles), device=device, dtype=torch.bool),
            tensor(coefficients.alphas).unsqueeze(0),
            tensor([coefficients.beta]),
            tensor([coefficients.gamma]),
            tensor([coefficients.lam_soft]),
            tensor([coefficients.lam_hard]),
            tensor(patch).unsqueeze(0),
            tensor([d_hat]),
            tensor([dt]),
            torch.tensor([steps], device=device, dtype=torch.int64),
            robot_radius=robot_radius,
            margin_factor=margin_factor,
            mass=mass,
            d_hat_sdf=float(runtime.params.d_hat_sdf_m),
            k_sharp=float(runtime.params.k_sharp),
            ext=ext,
        )
        position_out, velocity_out, min_clear, risk, hard, arc = outputs
        return BeliefMaterialRollout(
            position=position_out[0].detach().cpu().numpy().copy(),
            velocity=velocity_out[0].detach().cpu().numpy().copy(),
            min_clearance=float(min_clear[0].item()),
            cumulative_risk=float(risk[0].item()),
            hard_count=float(hard[0].item()),
            arc_length=float(arc[0].item()),
        )

    def live_material_kwargs(
        self,
        command: BeliefMaterialControlCommand,
        runtime: MaterialRuntime,
    ) -> dict[str, object]:
        """Build the material arguments consumed by GRL-SNAM's live rollouts.

        The returned mapping can be expanded directly into :func:`sdf_rollout` or
        :func:`bicycle_rollout`. It deliberately contains only the material field and
        material coefficients: per-obstacle ``alphas`` remain the responsibility of the
        base geometry adapter instead of being collapsed into one SDF coefficient here.
        """
        parameter = next(self.model.parameters())
        material = runtime.grid.field(device=str(parameter.device))
        dtype = material.field.dtype
        coefficients = command.effective
        return {
            "material": material,
            "lam_soft": torch.tensor([coefficients.lam_soft], device=parameter.device, dtype=dtype),
            "lam_hard": torch.tensor([coefficients.lam_hard], device=parameter.device, dtype=dtype),
            "mat_k_sharp": float(runtime.params.k_sharp),
            "mat_d_hat_m": float(runtime.params.d_hat_sdf_m),
        }


def build_belief_material_controller_inputs(
    position_world,
    goal_world,
    obstacle_centers_world,
    obstacle_radii,
    obstacle_weights,
    runtime: MaterialRuntime,
    seen: np.ndarray,
    *,
    patch_extent_m: float = 100.0,
    patch_size: int = 25,
    feature_scale_m: float = 20.0,
    belief: BeliefMaterialParams = BeliefMaterialParams(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one agent's inputs for :class:`CoefEnergyNetMaterial`.

    The caller supplies the already range-limited obstacle set and current local goal, all in
    world coordinates. The obstacle layout deliberately matches the existing geometry model:
    ``[cx, cy, radius, weight, goal_x - cx, goal_y - cy]``. The material patch is cropped from
    the runtime's belief planes, never from truth, and has channels ``[risk, hard, seen]``. As
    in the established CoefEnergyNet adapters, geometry features use a robot-relative frame and
    are divided by ``feature_scale_m``; material and rollout coordinates remain in world units.

    Returns unbatched ``(obs_feats, obs_mask, goal_feats, risk_patch)`` with shapes ``(N, 6)``,
    ``(N,)``, ``(4,)``, and ``(3, patch_size, patch_size)``. Callers add a leading batch axis
    before invoking the Torch model.
    """
    position = np.asarray(position_world, dtype=np.float32)
    goal = np.asarray(goal_world, dtype=np.float32)
    if position.shape != (2,):
        raise ValueError(f"position_world must have shape (2,), got {position.shape}")
    if goal.shape != (2,):
        raise ValueError(f"goal_world must have shape (2,), got {goal.shape}")
    if not math.isfinite(feature_scale_m) or feature_scale_m <= 0.0:
        raise ValueError("feature_scale_m must be finite and > 0")

    centers = np.asarray(obstacle_centers_world, dtype=np.float32)
    if centers.size == 0:
        centers = np.empty((0, 2), dtype=np.float32)
    elif centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"obstacle_centers_world must have shape (N, 2), got {centers.shape}")

    radii = np.asarray(obstacle_radii, dtype=np.float32)
    weights = np.asarray(obstacle_weights, dtype=np.float32)
    n_obstacles = centers.shape[0]
    if radii.shape != (n_obstacles,):
        raise ValueError(f"obstacle_radii must have shape ({n_obstacles},), got {radii.shape}")
    if weights.shape != (n_obstacles,):
        raise ValueError(f"obstacle_weights must have shape ({n_obstacles},), got {weights.shape}")

    material_seen = np.asarray(seen, dtype=bool)
    if material_seen.shape != runtime.grid.risk.shape:
        raise ValueError(
            f"seen must have shape {runtime.grid.risk.shape}, got {material_seen.shape}"
        )

    goal_delta = (goal - position) / np.float32(feature_scale_m)
    goal_feats = np.array(
        [goal_delta[0], goal_delta[1], np.linalg.norm(goal_delta), 1.0], dtype=np.float32
    )

    if n_obstacles:
        local_centers = (centers - position[None, :]) / np.float32(feature_scale_m)
        local_radii = radii / np.float32(feature_scale_m)
        goal_from_obstacles = goal_delta[None, :] - local_centers
        obs_feats = np.concatenate(
            [local_centers, local_radii[:, None], weights[:, None], goal_from_obstacles], axis=1
        ).astype(np.float32, copy=False)
    else:
        obs_feats = np.empty((0, 6), dtype=np.float32)
    obs_mask = np.ones(n_obstacles, dtype=bool)

    risk_patch = material_patch_belief(
        runtime.grid.risk,
        runtime.grid.hard,
        material_seen,
        runtime.grid.bounds,
        position,
        patch_extent_m,
        patch_size,
        belief,
    )
    return obs_feats, obs_mask, goal_feats, risk_patch
