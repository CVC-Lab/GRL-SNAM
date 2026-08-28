"""The torch-free C++ material trainer, driven from Python (material_train_native).

Skips until a pycvc with the nav_material_trainer_* binding is installed
(libcvc #247 + a pycvc-gl rebuild). The check is an end-to-end LOSS-DECREASE run
on a synthetic batch — an independent confirmation (complementing libcvc's C++
finite-difference gradchecks) that driving the C++ trainer from Python trains:
build a random .cvcnm, run Adam for a while on a fixed batch, assert the loss
drops substantially, then save + round-trip load. No torch required.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

pytest.importorskip("pycvc")
from grl_snam import material_train_native as mtn  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mtn.HAS_MATERIAL_TRAINER,
    reason="pycvc lacks nav_material_trainer_create (rebuild pycvc-gl after libcvc #247)",
)

# The CoefEnergyNetMaterial parameter table (name, shape) — matnet_export's layout.
_TENSORS = [
    ("goal_enc.0.weight", (64, 4)),
    ("goal_enc.0.bias", (64,)),
    ("goal_enc.2.weight", (64, 64)),
    ("goal_enc.2.bias", (64,)),
    ("obs_enc.0.weight", (128, 6)),
    ("obs_enc.0.bias", (128,)),
    ("obs_enc.2.weight", (64, 128)),
    ("obs_enc.2.bias", (64,)),
]
for _L in (0, 1):
    _p = f"fuser.layers.{_L}."
    _TENSORS += [
        (_p + "self_attn.in_proj_weight", (192, 64)),
        (_p + "self_attn.in_proj_bias", (192,)),
        (_p + "self_attn.out_proj.weight", (64, 64)),
        (_p + "self_attn.out_proj.bias", (64,)),
        (_p + "linear1.weight", (128, 64)),
        (_p + "linear1.bias", (128,)),
        (_p + "linear2.weight", (64, 128)),
        (_p + "linear2.bias", (64,)),
        (_p + "norm1.weight", (64,)),
        (_p + "norm1.bias", (64,)),
        (_p + "norm2.weight", (64,)),
        (_p + "norm2.bias", (64,)),
    ]
for _h in ("alpha_head", "beta_head", "gamma_head"):
    _TENSORS += [
        (_h + ".0.weight", (64, 64)),
        (_h + ".0.bias", (64,)),
        (_h + ".2.weight", (1, 64)),
        (_h + ".2.bias", (1,)),
    ]
_TENSORS += [
    ("risk_enc.net.0.weight", (16, 2, 3, 3)),
    ("risk_enc.net.0.bias", (16,)),
    ("risk_enc.net.2.weight", (32, 16, 3, 3)),
    ("risk_enc.net.2.bias", (32,)),
    ("risk_enc.net.4.weight", (64, 32, 3, 3)),
    ("risk_enc.net.4.bias", (64,)),
    ("risk_enc.net.8.weight", (64, 1024)),
    ("risk_enc.net.8.bias", (64,)),
]
for _h in ("lam_soft_head", "lam_hard_head", "mu_lat_head"):
    _TENSORS += [
        (_h + ".0.weight", (64, 128)),
        (_h + ".0.bias", (64,)),
        (_h + ".2.weight", (1, 64)),
        (_h + ".2.bias", (1,)),
    ]


def _write_random_cvcnm(path, patch_size, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "wb") as f:
        f.write(b"CVNM")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", 0xABCD))
        f.write(struct.pack("<IIIII", 64, 4, 2, 64, patch_size))
        f.write(struct.pack("<ffff", 5.0, 10.0, 5.0, 1e-5))
        f.write(struct.pack("<I", len(_TENSORS)))
        for name, shape in _TENSORS:
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            n = int(np.prod(shape))
            if "norm" in name and "weight" in name:
                w = rng.uniform(0.8, 1.2, n)
            else:
                w = rng.uniform(-0.15, 0.15, n)
            f.write(w.astype("<f4").tobytes())
        f.write(struct.pack("<I", 0))  # meta_len


def _synthetic_batch(B=16, N=3, P=16, Hp=13, Wp=13, seed=7):
    rng = np.random.default_rng(seed)
    o0 = rng.uniform(-0.5, 0.5, (B, 2)).astype(np.float32)
    goal = rng.uniform(1.5, 2.5, (B, 2)).astype(np.float32)
    ang = rng.uniform(0, 6.28, (B, N))
    dist = rng.uniform(1.0, 1.4, (B, N))
    C = np.stack([o0[:, 0:1] + dist * np.cos(ang), o0[:, 1:2] + dist * np.sin(ang)], -1).astype(
        np.float32
    )
    R = rng.uniform(0.4, 0.6, (B, N)).astype(np.float32)
    mask = np.ones((B, N), np.uint8)
    mask[:, N - 1] = 0
    obs_feats = np.concatenate(
        [C, R[..., None], rng.uniform(0.5, 1.5, (B, N, 1)), goal[:, None, :] - C], -1
    ).astype(np.float32)
    goal_feats = np.concatenate(
        [goal - o0, np.linalg.norm(goal - o0, axis=1, keepdims=True), np.ones((B, 1))], 1
    ).astype(np.float32)
    risk_patch = np.stack(
        [rng.uniform(0.1, 0.9, (B, P, P)), (rng.uniform(0, 1, (B, P, P)) > 0.7).astype(np.float32)],
        1,
    ).astype(np.float32)
    rollout_patch = rng.uniform(-0.3, 0.3, (B, 6, Hp, Wp)).astype(np.float32)
    rollout_patch[:, 0] = rng.uniform(0.1, 0.9, (B, Hp, Wp))
    rollout_patch[:, 1] = rng.uniform(1.0, 4.0, (B, Hp, Wp))
    return {
        "obs_feats": obs_feats,
        "obs_mask": mask,
        "goal_feats": goal_feats,
        "risk_patch": risk_patch,
        "o0": o0,
        "v0": rng.uniform(-0.1, 0.1, (B, 2)).astype(np.float32),
        "goal": goal,
        "C": C,
        "R": R,
        "rollout_patch": rollout_patch,
        "rr": np.full(B, 0.5, np.float32),
        "d_hat": np.full(B, 3.0, np.float32),
        "dt": np.full(B, 0.1, np.float32),
        "H": np.where(np.arange(B) % 2 == 0, 2, 3).astype(np.int32),
        "o_tgt": (goal + rng.uniform(-0.3, 0.3, (B, 2))).astype(np.float32),
        "v_tgt": rng.uniform(-0.2, 0.2, (B, 2)).astype(np.float32),
        "gamma_o": rng.uniform(3.0, 5.0, B).astype(np.float32),
    }


def test_native_trainer_reduces_loss(tmp_path):
    ckpt = tmp_path / "init.cvcnm"
    _write_random_cvcnm(ckpt, patch_size=16, seed=1)
    trainer = mtn.MaterialTrainer(ckpt)
    batch = _synthetic_batch()

    steps = 70
    L0 = L = None
    for s in range(steps):
        L = trainer.step(batch, lr=mtn.cosine_lr(1e-3, s, steps))
        if s == 0:
            L0 = L
    assert np.isfinite(L)
    assert L < 0.8 * L0, f"loss did not decrease (first={L0:.3f} last={L:.3f})"

    # checkpoint round-trips (write -> re-create a trainer from it)
    out = tmp_path / "trained.cvcnm"
    trainer.save(out)
    assert out.stat().st_size > 0
    mtn.MaterialTrainer(out)  # loads without error
