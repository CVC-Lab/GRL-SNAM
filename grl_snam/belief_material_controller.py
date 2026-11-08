"""Canonical model inputs for a range-limited belief-material controller."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .belief import BeliefGrid
from .belief_material import (
    BeliefMaterialParams,
    belief_gate_occ,
    belief_material_runtime,
    material_patch_belief,
)
from .material import GateParams, GateResult, MaterialParams, MaterialRuntime
from .material_nav import CoefEnergyNetMaterial, integrate_surrogate_material

__all__ = [
    "BELIEF_MATERIAL_META_KEY",
    "BeliefMaterialCoefficients",
    "BeliefMaterialContract",
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


BELIEF_MATERIAL_META_KEY = "grl_snam.belief_material"
"""Key of the belief-material contract in a ``.cvcnm`` file's JSON ``meta`` string."""

ROBOT_RELATIVE_FRAME = "robot_relative"


@dataclass(frozen=True)
class BeliefMaterialContract:
    """What a belief-material ``.cvcnm`` was trained with, recorded in the file's ``meta``.

    The ``.cvcnm`` header pins the network's shapes and belief semantics, but not the frame of its
    geometry inputs or the material force constants. This controller feeds obstacle and goal
    tokens in a **robot-relative** frame divided by ``feature_scale_m``. That differs from
    ``train_coef_energy.collate_short`` and ``eval_coef_energy.build_local_feats``, which feed
    absolute centres in metres; the token layout (libcvc's ``coef_energy_net.h``) is the same in
    both. A checkpoint therefore records which frame it expects, and
    :meth:`BeliefMaterialController.from_cvcnm` refuses one that does not.

    ``k_sharp`` / ``d_hat_sdf_m`` are the hazard barrier's steepness and reach,
    ``material_gate_trigger`` is the witness gate's soft-risk trigger, and ``geometry_d_hat_m`` is
    the obstacle barrier's reach in the training rollouts; inference must use the values the heads
    were trained under.
    """

    feature_frame: str = ROBOT_RELATIVE_FRAME
    feature_scale_m: float = 20.0
    k_sharp: float = 5.0
    d_hat_sdf_m: float = 3.0
    material_gate_trigger: float = 0.30
    geometry_d_hat_m: float = 3.0

    def __post_init__(self) -> None:
        if self.feature_frame != ROBOT_RELATIVE_FRAME:
            raise ValueError(
                f"feature_frame must be {ROBOT_RELATIVE_FRAME!r}, got {self.feature_frame!r}"
            )
        for name in ("feature_scale_m", "k_sharp", "d_hat_sdf_m", "geometry_d_hat_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        trigger = float(self.material_gate_trigger)
        if not math.isfinite(trigger) or not 0.0 <= trigger <= 1.0:
            raise ValueError("material_gate_trigger must be finite and in [0, 1]")

    def material_params(self, *, lam_soft: float, lam_hard: float) -> MaterialParams:
        """``MaterialParams`` carrying this contract's barrier and gate constants."""
        return MaterialParams(
            lam_soft=lam_soft,
            lam_hard=lam_hard,
            k_sharp=self.k_sharp,
            d_hat_sdf_m=self.d_hat_sdf_m,
            gate=GateParams(material_trigger=self.material_gate_trigger),
        )

    def to_meta(self) -> str:
        """The ``.cvcnm`` ``meta`` string recording this contract."""
        return json.dumps({BELIEF_MATERIAL_META_KEY: asdict(self)}, sort_keys=True)

    @classmethod
    def from_meta(cls, meta: str, *, source="checkpoint") -> BeliefMaterialContract:
        """Parse a ``.cvcnm`` ``meta`` string; raise if it does not record a contract."""
        try:
            record = json.loads(meta) if meta else {}
        except json.JSONDecodeError:
            record = {}
        contract = record.get(BELIEF_MATERIAL_META_KEY) if isinstance(record, dict) else None
        if not isinstance(contract, dict):
            raise ValueError(
                f"{source}: no {BELIEF_MATERIAL_META_KEY!r} record in the .cvcnm meta, so its "
                "geometry feature frame and material constants are unknown"
            )
        unknown = sorted(set(contract) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"{source}: unknown belief-material contract fields {unknown}")
        return cls(**contract)


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
        contract: BeliefMaterialContract = BeliefMaterialContract(),
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
        if belief.seen_rule != "raycast":
            raise ValueError("BeliefMaterialController requires belief.seen_rule='raycast'")
        self.model = model
        self.patch_size = int(patch_size)
        self.patch_extent_m = float(patch_extent_m)
        self.contract = contract
        self.feature_scale_m = float(contract.feature_scale_m)
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
        belief: BeliefMaterialParams = BeliefMaterialParams(),
    ) -> BeliefMaterialController:
        """Load and validate a belief-material ``.cvcnm`` deployment checkpoint.

        Loading goes through GRL-SNAM's shared ``.cvcnm`` reader. The version-2 belief block is
        checked against the controller's sensing and unknown-cell semantics, and the file must
        record a :class:`BeliefMaterialContract` (its feature frame and material constants) in its
        ``meta``, before any inference can run.
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
        contract = BeliefMaterialContract.from_meta(getattr(model, "meta", ""), source=path)

        return cls(
            model,
            patch_size=patch_size,
            patch_extent_m=patch_extent_m,
            contract=contract,
            belief=belief,
        )

    def check_runtime(self, runtime: MaterialRuntime) -> None:
        """Refuse a runtime whose barrier or gate constants differ from this controller's contract.

        Build the sensor with ``material_params=controller.contract.material_params(...)``; a
        runtime on the ``MaterialParams`` defaults (k 1.25, d_hat 12 m) would otherwise silently
        run heads trained under other constants.
        """
        params = runtime.params
        gate = params.gate or GateParams()
        expected = self.contract
        mismatches = [
            f"{name}={actual!r} (contract {wanted!r})"
            for name, actual, wanted in (
                ("k_sharp", float(params.k_sharp), expected.k_sharp),
                ("d_hat_sdf_m", float(params.d_hat_sdf_m), expected.d_hat_sdf_m),
                ("material_trigger", float(gate.material_trigger), expected.material_gate_trigger),
            )
            if not math.isclose(actual, float(wanted), rel_tol=0.0, abs_tol=1e-6)
        ]
        if mismatches:
            raise ValueError(
                f"material runtime does not match the checkpoint contract: {', '.join(mismatches)}"
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
        self.check_runtime(runtime)
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
        self.check_runtime(runtime)

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
        self.check_runtime(runtime)
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
    the runtime's belief planes, never from truth, and has channels ``[risk, hard, seen]``.
    Geometry features use a robot-relative frame divided by ``feature_scale_m`` — unlike
    ``train_coef_energy``'s absolute-metre features, which is why a belief-material checkpoint
    records its :class:`BeliefMaterialContract`. Material and rollout coordinates remain in world
    units.

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
