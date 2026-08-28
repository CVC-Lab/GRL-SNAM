"""Torch-free C++ material trainer (``cvc::nav::material_train``) driven from Python.

The dataset stays in Python; a batch dict of numpy arrays is handed to the C++
trainer (``coef_energy_net`` + Adam), which runs the whole material loss +
backward + Adam step GIL-free and returns the scalar loss. Trained weights
checkpoint to the ``.cvcnm`` container the forward / CUDA paths (and torch, via
``matnet_export``) all read.

This is the deployment path for TACC training campaigns: build batches from the
dataset in Python, run the fast torch-free step in C++. :data:`HAS_MATERIAL_TRAINER`
is False (and constructing a :class:`MaterialTrainer` raises) until a pycvc with
the ``nav_material_trainer_*`` binding is installed (libcvc PR #247 + a pycvc-gl
rebuild).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from grl_snam import nav_native as _nn

_pycvc = _nn._pycvc
HAS_MATERIAL_TRAINER = _nn.AVAILABLE and hasattr(_pycvc, "nav_material_trainer_create")

# The arrays nav_material_trainer_step expects, in order. Shapes (B agents,
# N padded obstacles, P model patch, Hp*Wp rollout patch):
#   obs_feats (B,N,6) f32   obs_mask (B,N) u8       goal_feats (B,4) f32
#   risk_patch (B,2,P,P) f32   o0/v0/goal (B,2) f32  C (B,N,2) f32   R (B,N) f32
#   rollout_patch (B,6,Hp,Wp) f32   rr/d_hat/dt (B,) f32   H (B,) i32
#   o_tgt/v_tgt (B,2) f32   gamma_o (B,) f32
BATCH_KEYS = (
    "obs_feats",
    "obs_mask",
    "goal_feats",
    "risk_patch",
    "o0",
    "v0",
    "goal",
    "C",
    "R",
    "rollout_patch",
    "rr",
    "d_hat",
    "dt",
    "H",
    "o_tgt",
    "v_tgt",
    "gamma_o",
)
_U8_KEYS = frozenset({"obs_mask"})
_I32_KEYS = frozenset({"H"})


def cosine_lr(lr0: float, t: int, t_max: int, eta_min_frac: float = 0.1) -> float:
    """torch ``CosineAnnealingLR``: anneal ``lr0`` -> ``lr0*eta_min_frac`` over ``t_max`` steps."""
    import math

    eta_min = lr0 * eta_min_frac
    tt = min(max(t, 0), t_max)
    return eta_min + (lr0 - eta_min) * 0.5 * (1.0 + math.cos(math.pi * tt / max(t_max, 1)))


class MaterialTrainer:
    """A stateful C++ material trainer (``coef_energy_net`` + Adam), reached by handle.

    Construct from a ``.cvcnm`` checkpoint (initial / Stage-1 warm-start weights);
    drive with :meth:`step` over batches; :meth:`save` the trained weights.
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
    ):
        if not HAS_MATERIAL_TRAINER:
            raise RuntimeError(
                "pycvc lacks nav_material_trainer_create — install a pycvc with the "
                "material-trainer binding (libcvc #247 + a pycvc-gl rebuild)."
            )
        self._h = _pycvc.nav_material_trainer_create(
            str(cvcnm_path),
            float(grad_clip),
            float(w_traj),
            float(w_vel),
            float(w_fric),
            float(w_clear),
            float(w_lreg),
            float(w_goal),
            float(w_len),
            float(w_risk),
            float(w_hard),
            float(cvar_alpha),
            float(w_multi),
            float(lam_soft_max),
            float(lam_hard_max),
            float(margin_factor),
            float(mass),
            float(d_hat_sdf),
            float(k_sharp),
            float(tau),
            int(ms_h),
            float(ms_dt_mult),
        )

    def step(self, batch: Mapping[str, np.ndarray], lr: float, num_threads: int = 0) -> float:
        """One training step over ``batch`` (keys :data:`BATCH_KEYS`) at learning rate ``lr``.

        Returns the scalar loss (computed on the pre-update weights, as in torch).
        """
        args = []
        for k in BATCH_KEYS:
            dt = np.uint8 if k in _U8_KEYS else (np.int32 if k in _I32_KEYS else np.float32)
            args.append(np.ascontiguousarray(batch[k], dtype=dt))
        return float(_pycvc.nav_material_trainer_step(self._h, *args, float(lr), int(num_threads)))

    def save(self, path) -> None:
        """Write the trained weights to ``path`` as a ``.cvcnm``."""
        _pycvc.nav_material_trainer_save(self._h, str(path))
