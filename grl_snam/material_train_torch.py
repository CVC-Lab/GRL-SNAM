"""The torch material trainer, held to libcvc's native one.

:class:`MaterialTrainerTorch` is the torch twin of
:class:`grl_snam.material_train_native.MaterialTrainer` (libcvc's ``material_loss_and_grad`` +
``material_adam``, ``src/cvc/nav/material_train.cpp``): the same constructor arguments, the same
``step`` / ``loss`` / ``save``, the same batch dict (the 17
:data:`~grl_snam.material_train_native.BATCH_KEYS` arrays), the same external force ``ext=``. The
model is :class:`~grl_snam.material_nav.CoefEnergyNetMaterial`; the loss is taken through the torch
rollout twins (:func:`~grl_snam.material_nav.integrate_surrogate_material`,
:func:`~grl_snam.material_bike.integrate_bicycle_material`) and differentiated by autograd. It runs
on the CPU or a CUDA device, in float32 (the native's precision) or float64.

The stage-2 loss, term by term as ``compute_loss`` adds it (``material_train.h``)::

    J_i   = w_goal |oT_i - goal_i|^2 + w_len arc_i + w_risk cum_risk_i + w_hard hard_count_i
    eta   = quantile(J, cvar_alpha)                       (linear interpolation, DETACHED)
    traj  = sum |oT - o_tgt|^2 / 2B          vel  = sum |vT - v_tgt|^2 / 2B
    fric  = mean (gamma - clamp(gamma_o, 0, 20))^2
    clear = mean softplus(-min_clear / 0.05)
    nav   = eta + sum max(J_i - eta, 0) / (B (1 - cvar_alpha))
    lreg  = -0.25 mean [log ls + log(1 - ls) + log lh + log(1 - lh)],  ls = lam_soft / lam_soft_max
    multi = mean softplus(-min_clear_ms / tau) over the multi-start geometry rollouts
    L     = 0.3 w_traj traj + 0.3 w_vel vel + w_fric fric + w_clear clear + nav + w_lreg lreg
            + w_multi multi

``hard_count`` is a step count (no gradient); ``multi`` backprops only into alphas, beta and gamma
(its rollouts carry no material force and no external force); ``mu_lat`` is unused. The optimizer
is ``material_adam``: global-norm clipping at ``grad_clip``, then bias-corrected Adam (0.9, 0.999,
1e-8) at the step's learning rate.

Where it follows the native rather than a literal torch reading (each is how libcvc computes it):

- The model runs without dropout (``eval()``) and through the transformer's math path (no fused
  attention fast path), as ``cvc::nav::coef_energy_net`` does; ``loss()`` and ``step()`` therefore
  see the same forward. On CUDA, TF32 is off while it runs (cuDNN convolutions default to it).
- Every configuration scalar is rounded to float32, as the native's ``material_loss_config`` holds
  them, and the reductions run in float64 from the per-agent values, as ``compute_loss`` does.
- ``lreg``'s clamp passes its gradient through: the native's seed is
  ``(1/ls - 1/(1-ls)) / lam_soft_max`` at the clamped value, with no zero on the clamp.
- ``multi`` is libcvc's ``multi_start_penalty`` (``geom_rollout.cpp``): explicit Euler by default
  (``semi_implicit=True`` gives GRL-SNAM's :func:`~grl_snam.surrogate_robust.multi_start_penalty`),
  the trainer's ``mass``, and the starts drawn exactly as ``multi_start_fracs`` draws them (the
  stream of ``torch.rand`` on a CPU generator seeded with ``seed``). The ``k``-th :meth:`step`
  draws with ``seed + k``; :meth:`loss` draws with ``seed``.
- ``loss(frozen_eta=...)`` rounds the quantile to float32, as ``nav_material_trainer_loss`` does.

``num_threads`` is accepted for API parity and ignored (torch has its own thread pool). The dynamic
integrator (``material_integrator::dynamic``) has no torch twin yet: asking for it raises
:class:`NotImplementedError`.

Parity with the native trainer is tested in ``tests/test_material_train_torch.py``: against the
libcvc golden (``GRL_SNAM_MATERIAL_TRAIN_GOLDEN``), and against a live pycvc trainer when one with
the binding is installed.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from grl_snam.material_bike import BicycleParams, integrate_bicycle_material
from grl_snam.material_ext import MaterialExt
from grl_snam.material_nav import CoefEnergyNetMaterial, integrate_surrogate_material
from grl_snam.material_train_native import BATCH_KEYS, cosine_lr
from grl_snam.tools.matnet_export import read_matnet, write_matnet
from grl_snam.train_coef_energy import ipc_piecewise

__all__ = [
    "BATCH_KEYS",
    "LOSS_PARTS",
    "MaterialTrainerTorch",
    "cosine_lr",
    "load_cvcnm",
    "save_cvcnm",
]

# The loss's parts, named as libcvc's material_loss_parts names them.
LOSS_PARTS = ("L", "traj", "vel", "fric", "clear", "nav", "lreg", "multi", "eta")
_INTEGRATORS = ("surrogate", "bicycle", "dynamic")


def _f32(x) -> float:
    """``x`` rounded to float32, as a Python float."""
    return float(np.float32(x))


# ---------------------------------------------------------------------------
# .cvcnm <-> CoefEnergyNetMaterial
# ---------------------------------------------------------------------------


def load_cvcnm(path, *, device=None, dtype: torch.dtype = torch.float32) -> CoefEnergyNetMaterial:
    """A :class:`CoefEnergyNetMaterial` with the weights of the ``.cvcnm`` at ``path``.

    The file's hyper-parameters (``lam_soft_max``, ``lam_hard_max``, ``mu_lat_max``, the LayerNorm
    eps, ``patch_size``) carry over; ``patch_size`` is kept on the model as an attribute so
    :func:`save_cvcnm` writes it back. The model is returned in ``eval()`` mode (no dropout: the
    native net has none).
    """
    mn = read_matnet(str(path))
    m = CoefEnergyNetMaterial(
        d_tok=mn.d_tok,
        patch_size=mn.patch_size,
        d_risk=mn.d_risk,
        nhead=mn.nhead,
        num_layers=mn.num_layers,
        lam_soft_max=float(mn.lam_soft_max),
        lam_hard_max=float(mn.lam_hard_max),
        mu_lat_max=float(mn.mu_lat_max),
    )
    own = m.state_dict()
    missing = sorted(set(own) - set(mn.t))
    extra = sorted(set(mn.t) - set(own))
    if missing or extra:
        raise ValueError(
            f"{path}: tensors do not match the model (missing {missing}, extra {extra})"
        )
    for k, v in mn.t.items():
        if tuple(v.shape) != tuple(own[k].shape):
            raise ValueError(f"{path}: {k} is {v.shape}, the model's is {tuple(own[k].shape)}")
    m.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in mn.t.items()})
    for mod in m.modules():
        if isinstance(mod, nn.LayerNorm):
            mod.eps = float(mn.eps)
    m.patch_size = int(mn.patch_size)
    _no_nested(m)
    m.eval()
    return m.to(device=device, dtype=dtype)


def _no_nested(model: nn.Module) -> None:
    """Keep the transformer off its nested-tensor conversion (a fast-path-only shortcut)."""
    model.fuser.enable_nested_tensor = False
    if hasattr(model.fuser, "use_nested_tensor"):
        model.fuser.use_nested_tensor = False


def save_cvcnm(model: nn.Module, path) -> str:
    """Write ``model``'s weights to ``path`` as a ``.cvcnm`` (float32, the native's format)."""
    return write_matnet(model, str(path))


@contextlib.contextmanager
def _math_path():
    """The transformer's math path: no fused fast path, math scaled-dot-product attention.

    The native ``coef_energy_net`` computes the explicit post-norm layers with a masked softmax;
    torch's fast path (taken in eval mode without grad) and its fused attention kernels round
    differently, and would make ``loss()`` disagree with ``step()``.
    """
    mha = getattr(torch.backends, "mha", None)
    if mha is not None and not hasattr(mha, "set_fastpath_enabled"):  # torch < 2.2
        mha = None
    prev = mha.get_fastpath_enabled() if mha is not None else None
    if mha is not None:
        mha.set_fastpath_enabled(False)
    try:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            ctx = sdpa_kernel([SDPBackend.MATH])
        except ImportError:  # torch < 2.3
            ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
        with ctx:
            yield
    finally:
        if mha is not None:
            mha.set_fastpath_enabled(prev)


@contextlib.contextmanager
def _full_float32():
    """No TF32 anywhere (cuDNN convolutions default to it on CUDA): the native computes in full
    float32, and TF32's 10-bit mantissa would put ~1e-4 into the risk encoder's gradients."""
    cudnn, mm = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    prec = torch.get_float32_matmul_precision()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cuda.matmul.allow_tf32 = mm
        torch.set_float32_matmul_precision(prec)


# ---------------------------------------------------------------------------
# The pieces of the loss that have no torch twin elsewhere
# ---------------------------------------------------------------------------


def _quantile_linear(x: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile's linear interpolation, in the native's op order (detached)."""
    xs = torch.sort(x.detach().reshape(-1)).values
    n = xs.numel()
    if n == 0:
        return x.new_zeros(())
    pos = q * float(n - 1)
    lo = int(math.floor(pos))
    if lo + 1 >= n:
        return xs[-1]
    frac = pos - float(lo)
    return xs[lo] + frac * (xs[lo + 1] - xs[lo])


def _clamp_pass(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """clamp(x, lo, hi) in value, the identity in gradient (the native lreg seed)."""
    return x + (x.clamp(lo, hi) - x).detach()


def multi_start_fracs(ms_count: int, B: int, frac_lo: float, frac_hi: float, seed: int):
    """libcvc's ``multi_start_fracs``: the (K, B) float32 start fractions, start-major.

    A degenerate range draws nothing and gives one start (K = 1) of ``frac_lo``. Otherwise
    ``frac_lo + (frac_hi - frac_lo) * u``, ``u`` the CPU stream of
    ``torch.rand(K*B, generator=torch.Generator().manual_seed(seed))`` (only the low 32 bits of
    ``seed`` count), with both scalars rounded to float32 as the Python reference rounds them.
    """
    lo, hi = float(frac_lo), float(frac_hi)
    if not 0.0 <= lo <= hi < 1.0:
        raise ValueError(f"frac_range must satisfy 0 <= lo <= hi < 1, got {(frac_lo, frac_hi)!r}")
    if ms_count <= 0 or B <= 0:
        return torch.zeros(0, max(B, 0), dtype=torch.float32)
    if not hi > lo:
        return torch.full((1, B), _f32(lo), dtype=torch.float32)
    g = torch.Generator().manual_seed(int(seed) & 0xFFFFFFFF)
    u = torch.rand(ms_count * B, generator=g, dtype=torch.float32)
    span = torch.tensor(_f32(hi - lo), dtype=torch.float32)
    return (torch.tensor(_f32(lo), dtype=torch.float32) + span * u).view(ms_count, B)


@torch.no_grad()
def multi_start_origins(
    o0: torch.Tensor,
    C: torch.Tensor,
    R_eff: torch.Tensor,
    mask: torch.Tensor,
    fracs: torch.Tensor,
) -> torch.Tensor:
    """libcvc's ``multi_start_origins``: the (K, B, 2) starts, data (no gradient).

    Each start moves every agent ``frac`` of its clearance toward its nearest valid obstacle, or
    half a step AWAY if that would penetrate one; an agent with no valid obstacle stays at ``o0``.
    """
    B = o0.shape[0]
    rows = torch.arange(B, device=o0.device)
    diff = o0.unsqueeze(1) - C
    r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
    d = torch.where(mask, r - R_eff, torch.full_like(r, float("inf")))
    dmin0, j = d.min(dim=1)
    n0 = diff[rows, j] / r[rows, j].unsqueeze(-1)
    fr = fracs.to(device=o0.device, dtype=o0.dtype)
    finite = torch.isfinite(dmin0)
    fd = fr * torch.where(finite, dmin0, torch.zeros_like(dmin0))  # (K, B)
    step = fd.unsqueeze(-1) * n0  # (K, B, 2)
    o_ms = o0 - step
    r_ms = torch.linalg.norm(o_ms.unsqueeze(2) - C, dim=-1).clamp_min(1e-9)
    d_ms = torch.where(mask, r_ms - R_eff, torch.full_like(r_ms, float("inf"))).min(dim=-1).values
    o_ms = torch.where((d_ms < 0.0).unsqueeze(-1), o0 + 0.5 * step, o_ms)
    return torch.where(finite.unsqueeze(-1), o_ms, o0.expand_as(o_ms))


def geom_rollout_min_clear(
    o: torch.Tensor,
    v: torch.Tensor,
    goal: torch.Tensor,
    C: torch.Tensor,
    R_eff: torch.Tensor,
    mask: torch.Tensor,
    alphas: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    d_hat: torch.Tensor,
    dt: torch.Tensor,
    steps: int,
    mass: float = 1.0,
    semi_implicit: bool = False,
) -> torch.Tensor:
    """libcvc's ``integrate_surrogate_v2`` (``geom_rollout.cpp``): the minimum clearance.

    The geometry-only rollout L_multi scores: the per-obstacle IPC barriers, the goal spring and
    damping, ``a = F / mass``; explicit Euler (the position steps with the OLD velocity) or, with
    ``semi_implicit``, GRL-SNAM's order. ``min_clear`` is taken before each step, ungated.
    """
    B = o.shape[0]
    min_clear = torch.full((B,), float("inf"), dtype=o.dtype, device=o.device)
    dtc = dt.unsqueeze(-1)
    for _ in range(int(steps)):
        f_goal = -beta.unsqueeze(-1) * (o - goal)
        diff = o.unsqueeze(1) - C
        r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
        n_hat = diff / r.unsqueeze(-1)
        d = r - R_eff
        _, dbdd = ipc_piecewise(torch.where(mask, d, torch.full_like(d, 1e6)), d_hat.view(-1, 1))
        f_bar = (-(alphas * dbdd).unsqueeze(-1) * n_hat).sum(dim=1)
        dmin = torch.where(mask, d, torch.full_like(d, float("inf"))).min(dim=1).values
        min_clear = torch.minimum(min_clear, dmin)
        a = (f_bar + f_goal - gamma.unsqueeze(-1) * v) / mass
        if semi_implicit:
            v = v + dtc * a
            o = o + dtc * v
        else:
            o_next = o + dtc * v
            v = v + dtc * a
            o = o_next
    return min_clear


def _bicycle_start(o0: torch.Tensor, v0: torch.Tensor, goal: torch.Tensor):
    """The native ``bicycle_start``: speed |v0|, heading along v0 (toward the goal at rest)."""
    vx, vy = v0[:, 0], v0[:, 1]
    speed = torch.sqrt(vx * vx + vy * vy)
    toward = torch.atan2(goal[:, 1] - o0[:, 1], goal[:, 0] - o0[:, 0])
    th = torch.where(speed > 1e-6, torch.atan2(vy, vx), toward)
    return th, speed


def _validate_bicycle(v) -> None:
    """libcvc's ``validate(bicycle_material_params)``."""
    ok = (
        all(math.isfinite(float(x)) and float(x) > 0.0 for x in (v.L, v.a_max, v.a_lat_max, v.vmax))
        and 0.0 < float(v.delta_max) < math.pi / 2
        and math.isfinite(float(v.k_steer))
        and float(v.k_steer) >= 0.0
    )
    if not ok:
        raise ValueError(
            "bicycle: L, a_max, a_lat_max and vmax must be finite and > 0, delta_max in "
            f"(0, pi/2) and k_steer finite and >= 0; got {v!r}"
        )


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------


class MaterialTrainerTorch:
    """The torch material trainer: :class:`~grl_snam.material_train_native.MaterialTrainer`'s API.

    Construct from a ``.cvcnm`` checkpoint (or a :class:`CoefEnergyNetMaterial`, which is copied);
    drive with :meth:`step` over batches; :meth:`save` the trained weights as a ``.cvcnm`` the
    native trainer, the native forward and the CUDA paths all read. Every keyword means what it
    means for ``MaterialTrainer`` (see its docstring), with these additions:

    - ``integrator``: ``"surrogate"`` (the point mass, the default), ``"bicycle"`` (implied by
      ``bicycle=``; the defaults are :class:`~grl_snam.material_bike.BicycleParams`'s, which are
      the native ``set_bicycle``'s) or ``"dynamic"``, which raises :class:`NotImplementedError`.
    - ``device``: a torch device. Without one, ``use_cuda=True`` asks for ``"cuda"`` and falls back
      to the CPU when no device is present (a request, as the native's is: read
      :attr:`cuda_active`).
    - ``dtype``: ``torch.float32`` (the native's precision) or ``torch.float64``.

    Every configuration (sampled starts, ``semi_implicit``, the bicycle, ``ext`` from any source,
    a callable included) runs here; the native's binding-version gates do not apply.
    """

    def __init__(
        self,
        cvcnm_path,
        *,
        grad_clip: float = 5.0,
        w_traj: float = 1.0,
        w_vel: float = 0.5,
        w_fric: float = 0.1,
        w_clear: float = 5e-3,
        w_lreg: float = 0.01,
        w_goal: float = 2.0,
        w_len: float = 0.01,
        w_risk: float = 1.0,
        w_hard: float = 5.0,
        cvar_alpha: float = 0.95,
        w_multi: float = 0.5,
        lam_soft_max: float = 5.0,
        lam_hard_max: float = 10.0,
        margin_factor: float = 0.5,
        mass: float = 1.0,
        d_hat_sdf: float = 3.0,
        k_sharp: float = 5.0,
        tau: float = 0.05,
        ms_h: int = 3,
        ms_dt_mult: float = 4.0,
        use_cuda: bool = False,
        ms_count: int = 1,
        frac_range: tuple[float, float] = (0.9, 0.9),
        seed: int = 0,
        semi_implicit: bool = False,
        bicycle=None,
        integrator: str | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        if integrator is None:
            integrator = "bicycle" if bicycle is not None else "surrogate"
        if integrator not in _INTEGRATORS:
            raise ValueError(f"integrator must be one of {_INTEGRATORS}, got {integrator!r}")
        if integrator == "dynamic":
            raise NotImplementedError(
                "the dynamic integrator (libcvc material_integrator::dynamic, "
                "integrate_dynamic_material) has no torch twin yet; train on 'surrogate' or "
                "'bicycle' here, or use the native MaterialTrainer"
            )
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
        frac_lo, frac_hi = (float(f) for f in frac_range)
        if not 0.0 <= frac_lo <= frac_hi < 1.0:
            raise ValueError(f"frac_range must satisfy 0 <= lo <= hi < 1, got {frac_range!r}")
        if integrator == "bicycle":
            bicycle = BicycleParams() if bicycle is None else bicycle
            _validate_bicycle(bicycle)
            # Rounded to float32, as the native's bicycle_material_params holds them.
            bicycle = BicycleParams(
                L=_f32(bicycle.L),
                delta_max=_f32(bicycle.delta_max),
                a_max=_f32(bicycle.a_max),
                a_lat_max=_f32(bicycle.a_lat_max),
                k_steer=_f32(bicycle.k_steer),
                vmax=_f32(bicycle.vmax),
                allow_reverse=bool(bicycle.allow_reverse),
            )
        else:
            bicycle = None

        if device is None:
            device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        if isinstance(cvcnm_path, nn.Module):
            model = copy.deepcopy(cvcnm_path)
            _no_nested(model)
            model.eval()
            self.model = model.to(device=self.device, dtype=dtype)
        else:
            self.model = load_cvcnm(cvcnm_path, device=self.device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._integrator = integrator
        self.bicycle = bicycle
        self.grad_clip = _f32(grad_clip)
        # The native's material_loss_config holds every scalar as a float.
        self.cfg = {
            k: _f32(v)
            for k, v in dict(
                w_traj=w_traj,
                w_vel=w_vel,
                w_fric=w_fric,
                w_clear=w_clear,
                w_lreg=w_lreg,
                w_goal=w_goal,
                w_len=w_len,
                w_risk=w_risk,
                w_hard=w_hard,
                cvar_alpha=cvar_alpha,
                w_multi=w_multi,
                lam_soft_max=lam_soft_max,
                lam_hard_max=lam_hard_max,
                margin_factor=margin_factor,
                mass=mass,
                d_hat_sdf=d_hat_sdf,
                k_sharp=k_sharp,
                tau=tau,
                ms_dt_mult=ms_dt_mult,
            ).items()
        }
        self.ms_h = int(ms_h)
        self.ms_count = int(ms_count)
        self.frac_range = (frac_lo, frac_hi)
        self.seed = int(seed) & 0xFFFFFFFF
        self.semi_implicit = bool(semi_implicit)

        # material_adam: beta1, beta2, eps as floats; the moments keyed like the parameters.
        self._b1, self._b2, self._eps = _f32(0.9), _f32(0.999), _f32(1e-8)
        self._t = 0
        self._m = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self._u = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.last_eta = 0.0

    # -- properties ------------------------------------------------------------------------------

    @property
    def integrator(self) -> str:
        """``"surrogate"`` or ``"bicycle"``: the rollout the loss is taken through."""
        return self._integrator

    @property
    def cuda_active(self) -> bool:
        """Whether this trainer runs on a CUDA device (``use_cuda`` is a request)."""
        return self.device.type == "cuda"

    @property
    def steps(self) -> int:
        """Optimizer steps taken (the native ``material_adam::steps``)."""
        return self._t

    # -- batches ---------------------------------------------------------------------------------

    def _batch(self, batch: Mapping) -> dict:
        """The batch as tensors on this trainer's device: float32 values (as the binding
        coerces them) in this trainer's dtype, ``obs_mask`` bool and ``H`` int64."""
        out = {}
        for k in BATCH_KEYS:
            x = batch[k]
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            if k == "obs_mask":
                t = torch.from_numpy(np.ascontiguousarray(x, np.uint8)).bool()
            elif k == "H":
                t = torch.from_numpy(np.ascontiguousarray(x, np.int32)).long()
            else:
                t = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(self.dtype)
            out[k] = t.to(self.device)
        B = out["o0"].shape[0] if out["o0"].dim() == 2 else -1
        N = out["C"].shape[1] if out["C"].dim() == 3 else -1
        rp, rl = out["risk_patch"], out["rollout_patch"]
        want = {
            "obs_feats": (B, N, 6),
            "obs_mask": (B, N),
            "goal_feats": (B, 4),
            "o0": (B, 2),
            "v0": (B, 2),
            "goal": (B, 2),
            "C": (B, N, 2),
            "R": (B, N),
            "rr": (B,),
            "d_hat": (B,),
            "dt": (B,),
            "H": (B,),
            "o_tgt": (B, 2),
            "v_tgt": (B, 2),
            "gamma_o": (B,),
        }
        bad = [k for k, s in want.items() if tuple(out[k].shape) != s]
        if rp.dim() != 4 or rp.shape[0] != B or rp.shape[1] != 2 or rp.shape[2] != rp.shape[3]:
            bad.append("risk_patch")
        if rl.dim() != 4 or rl.shape[0] != B or rl.shape[1] != 6:
            bad.append("rollout_patch")
        if B < 0 or N < 0 or bad:
            raise ValueError(f"batch shape mismatch ({', '.join(bad) or 'o0/C rank'})")
        P = getattr(self.model, "patch_size", None)
        if P is not None and int(rp.shape[2]) != int(P):
            raise ValueError(f"risk_patch P {int(rp.shape[2])} != model patch_size {int(P)}")
        return out

    def _ext(self, ext):
        """``ext`` with its tensors on this trainer's device (values rounded to float32, as the
        native takes them); a callable source is passed through."""
        if ext is None:
            return None

        def T(x, integer=False):
            if x is None:
                return None
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            if integer:
                return torch.from_numpy(np.ascontiguousarray(x, np.int64)).to(self.device)
            a = np.ascontiguousarray(x, np.float32)
            return torch.from_numpy(a).to(device=self.device, dtype=self.dtype)

        return MaterialExt(
            patch=T(getattr(ext, "patch", None)),
            patch_of_step=T(getattr(ext, "patch_of_step", None), integer=True),
            F=T(getattr(ext, "F", None)),
            J=T(getattr(ext, "J", None)),
            sample=getattr(ext, "sample", None),
            hold=int(getattr(ext, "hold", 1)),
            steer=bool(getattr(ext, "steer", True)),
        )

    # -- the loss --------------------------------------------------------------------------------

    def _rollout(self, b, coefs, ext):
        """(oT, vT, min_clear, cum_risk, hard_count, arc, th0, sp0) through the integrator."""
        alphas, beta, gamma, lam_soft, lam_hard = coefs
        c = self.cfg
        mask = b["obs_mask"]
        if self._integrator == "bicycle":
            th0, sp0 = _bicycle_start(b["o0"], b["v0"], b["goal"])
            oT, thT, spT, min_clear, cum, hard, arc = integrate_bicycle_material(
                b["o0"],
                th0,
                sp0,
                b["goal"],
                b["C"],
                b["R"],
                mask,
                alphas,
                beta,
                gamma,
                lam_soft,
                lam_hard,
                b["rollout_patch"],
                b["rr"],
                b["d_hat"],
                b["dt"],
                b["H"],
                vehicle=self.bicycle,
                margin_factor=c["margin_factor"],
                mass=c["mass"],
                d_hat_sdf=c["d_hat_sdf"],
                k_sharp=c["k_sharp"],
                ext=ext,
            )
            vT = torch.stack([spT * torch.cos(thT), spT * torch.sin(thT)], dim=-1)
            return oT, vT, min_clear, cum, hard, arc, th0, sp0
        oT, vT, min_clear, cum, hard, arc = integrate_surrogate_material(
            b["o0"],
            b["v0"],
            b["goal"],
            b["C"],
            b["R"],
            mask,
            alphas,
            beta,
            gamma,
            lam_soft,
            lam_hard,
            b["rollout_patch"],
            b["d_hat"],
            b["dt"],
            b["H"],
            robot_radius=b["rr"],
            margin_factor=c["margin_factor"],
            mass=c["mass"],
            d_hat_sdf=c["d_hat_sdf"],
            k_sharp=c["k_sharp"],
            ext=ext,
        )
        return oT, vT, min_clear, cum, hard, arc, None, None

    def _multi(self, b, alphas, beta, gamma, th0, sp0, seed) -> torch.Tensor:
        """L_multi (float64 scalar): libcvc's multi_start_penalty / _bicycle."""
        c = self.cfg
        o0, C, mask = b["o0"], b["C"], b["obs_mask"]
        B, N = C.shape[:2]
        fracs = multi_start_fracs(self.ms_count, B, *self.frac_range, seed)
        K = fracs.shape[0]
        if K == 0 or N == 0 or B == 0:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        rr = b["rr"]
        R_eff = b["R"] + c["margin_factor"] * rr.unsqueeze(-1)
        o_ms = multi_start_origins(o0, C, R_eff, mask, fracs).reshape(K * B, 2)

        def tile(x):
            return x.repeat(K, *([1] * (x.dim() - 1)))

        dt_ms = c["ms_dt_mult"] * b["dt"]
        if self._integrator == "bicycle":
            KB = K * B
            zeros = torch.zeros(KB, dtype=self.dtype, device=self.device)
            patch = torch.zeros(KB, 6, 2, 2, dtype=self.dtype, device=self.device)
            mc = integrate_bicycle_material(
                o_ms,
                tile(th0),
                tile(sp0),
                tile(b["goal"]),
                tile(C),
                tile(b["R"]),
                tile(mask),
                tile(alphas),
                tile(beta),
                tile(gamma),
                zeros,
                zeros,
                patch,
                tile(rr),
                tile(b["d_hat"]),
                tile(dt_ms),
                torch.full((KB,), self.ms_h, dtype=torch.long, device=self.device),
                vehicle=self.bicycle,
                margin_factor=c["margin_factor"],
                mass=c["mass"],
                d_hat_sdf=c["d_hat_sdf"],
                k_sharp=c["k_sharp"],
            )[3]
        else:
            mc = geom_rollout_min_clear(
                o_ms,
                tile(b["v0"]),
                tile(b["goal"]),
                tile(C),
                tile(R_eff),
                tile(mask),
                tile(alphas),
                tile(beta),
                tile(gamma),
                tile(b["d_hat"]),
                tile(dt_ms),
                self.ms_h,
                mass=c["mass"],
                semi_implicit=self.semi_implicit,
            )
        return F.softplus(-mc / c["tau"]).double().mean()

    def _compute(self, b, ext, eta_frozen, seed) -> dict:
        """The forward: every part of the loss (``LOSS_PARTS``), as float64 tensors."""
        c = self.cfg
        with _math_path():
            alphas, beta, gamma, lam_soft, lam_hard, _ = self.model(
                b["obs_feats"], b["obs_mask"], b["goal_feats"], b["risk_patch"]
            )
        oT, vT, min_clear, cum, hard, arc, th0, sp0 = self._rollout(
            b, (alphas, beta, gamma, lam_soft, lam_hard), ext
        )
        B = oT.shape[0]
        inv_b = 1.0 / float(B)

        # Episode cost J and the detached CVaR quantile.
        g = (oT - b["goal"]).double()
        J = (
            c["w_goal"] * (g * g).sum(-1)
            + c["w_len"] * arc.double()
            + c["w_risk"] * cum.double()
            + c["w_hard"] * hard.double()
        )
        eta = (
            _quantile_linear(J, c["cvar_alpha"])
            if eta_frozen is None
            else torch.tensor(float(eta_frozen), dtype=torch.float64, device=J.device)
        )

        t = (oT - b["o_tgt"]).double()
        v = (vT - b["v_tgt"]).double()
        traj = (t * t).sum() * (inv_b * 0.5)
        vel = (v * v).sum() * (inv_b * 0.5)
        go = b["gamma_o"].clamp(0.0, 20.0).double()
        dg = gamma.double() - go
        fric = (dg * dg).sum() * inv_b
        clear = F.softplus(-min_clear / 0.05).double().sum() * inv_b
        nav = eta + torch.relu(J - eta).sum() * inv_b / (1.0 - c["cvar_alpha"])
        lo = _f32(1e-6)
        hi = float(np.float32(1.0) - np.float32(1e-6))
        ls = _clamp_pass(lam_soft / c["lam_soft_max"], lo, hi).double()
        lh = _clamp_pass(lam_hard / c["lam_hard_max"], lo, hi).double()
        lreg = (torch.log(ls) + torch.log(1.0 - ls) + torch.log(lh) + torch.log(1.0 - lh)).sum() * (
            -0.25 * inv_b
        )
        L = (
            0.3 * c["w_traj"] * traj
            + 0.3 * c["w_vel"] * vel
            + c["w_fric"] * fric
            + c["w_clear"] * clear
            + nav
            + c["w_lreg"] * lreg
        )
        multi = torch.zeros((), dtype=torch.float64, device=L.device)
        if c["w_multi"] != 0.0 and b["C"].shape[1] > 0:
            multi = self._multi(b, alphas, beta, gamma, th0, sp0, seed)
            L = L + c["w_multi"] * multi
        return dict(
            L=L,
            traj=traj,
            vel=vel,
            fric=fric,
            clear=clear,
            nav=nav,
            lreg=lreg,
            multi=multi,
            eta=eta,
        )

    @staticmethod
    def _floats(parts: dict) -> dict:
        return {k: float(parts[k].detach()) for k in LOSS_PARTS}

    def loss_parts(self, batch: Mapping, frozen_eta: float | None = None, ext=None, seed=None):
        """Every part of the loss (libcvc's ``material_loss_parts``: ``L``, the unweighted
        ``traj``/``vel``/``fric``/``clear``/``nav``/``lreg``/``multi`` and ``eta``) as floats.

        No backward and no optimizer step. ``frozen_eta`` pins the quantile EXACTLY (no float32
        rounding, unlike :meth:`loss`); ``seed`` is the multi-start draw's (the base seed by
        default).
        """
        b, e = self._batch(batch), self._ext(ext)
        with torch.no_grad(), _full_float32():
            parts = self._compute(b, e, frozen_eta, self.seed if seed is None else seed)
        return self._floats(parts)

    def loss_and_grad(self, batch: Mapping, ext=None, frozen_eta: float | None = None, seed=None):
        """(parts, grads): :meth:`loss_parts` and the weight gradients, without a step.

        ``grads`` maps every parameter name (the ``.cvcnm`` / state-dict names) to its gradient,
        zeros for the parameters the loss does not reach (``mu_lat_head``). The native
        ``material_loss_and_grad``: the quantile is computed from J unless ``frozen_eta`` pins it.
        """
        b, e = self._batch(batch), self._ext(ext)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad(), _full_float32():
            parts = self._compute(b, e, frozen_eta, self.seed if seed is None else seed)
            parts["L"].backward()
        grads = {
            n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in self.model.named_parameters()
        }
        self.model.zero_grad(set_to_none=True)
        return self._floats(parts), grads

    def loss(
        self,
        batch: Mapping,
        frozen_eta: float = float("nan"),
        num_threads: int = 0,
        ext=None,
    ) -> float:
        """Forward-only loss: the native ``MaterialTrainer.loss``.

        Runs no backward and does not touch the optimizer. ``frozen_eta`` pins the detached CVaR
        quantile, rounded to float32 as ``nav_material_trainer_loss`` rounds it; the default NaN
        computes it from this batch's own costs. The multi-start draw uses the base ``seed``.
        ``num_threads`` is ignored.
        """
        del num_threads
        eta = None if math.isnan(float(frozen_eta)) else _f32(frozen_eta)
        return self.loss_parts(batch, eta, ext)["L"]

    def step(self, batch: Mapping, lr: float, num_threads: int = 0, ext=None) -> float:
        """One training step at learning rate ``lr``; returns the pre-update loss.

        The native ``MaterialTrainer.step``: the loss and its gradient (the quantile from this
        batch's J, its float32 value kept as :attr:`last_eta`), then ``material_adam``. The k-th
        step draws the multi-start starts with ``seed + k``. ``num_threads`` is ignored.
        """
        del num_threads
        parts, grads = self.loss_and_grad(batch, ext=ext, seed=(self.seed + self._t) & 0xFFFFFFFF)
        self._adam(grads, lr)
        self.last_eta = _f32(parts["eta"])
        return parts["L"]

    @torch.no_grad()
    def _adam(self, grads: dict, lr: float) -> None:
        """``material_adam::step``: global-norm clip, then bias-corrected Adam, in place."""
        self._t += 1
        sq = sum(float((g.double() * g.double()).sum()) for g in grads.values())
        norm = np.float32(math.sqrt(sq))
        clip = np.float32(self.grad_clip)
        gscale = float(clip / norm) if (clip > 0 and norm > clip) else 1.0
        b1, b2 = np.float32(self._b1), np.float32(self._b2)
        bc1 = float(np.float32(1.0) - b1 ** np.float32(self._t))
        bc2 = float(np.float32(1.0) - b2 ** np.float32(self._t))
        one_b1 = float(np.float32(1.0) - b1)
        one_b2 = float(np.float32(1.0) - b2)
        lr = _f32(lr)
        for name, w in self.model.named_parameters():
            g = grads[name] * gscale
            m = self._m[name]
            u = self._u[name]
            m.copy_(self._b1 * m + one_b1 * g)
            u.copy_(self._b2 * u + one_b2 * g * g)
            mhat = m / bc1
            uhat = u / bc2
            w.sub_(lr * mhat / (torch.sqrt(uhat) + self._eps))

    # -- weights ---------------------------------------------------------------------------------

    def save(self, path) -> None:
        """Write the trained weights to ``path`` as a ``.cvcnm``."""
        save_cvcnm(self.model, os.fspath(path))
