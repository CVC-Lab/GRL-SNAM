"""The torch-free C++ material trainer, driven from Python (material_train_native).

Skips until a pycvc with the nav_material_trainer_* binding is installed
(libcvc #247 + a pycvc-gl rebuild). The check is an end-to-end LOSS-DECREASE run
on a synthetic batch — an independent confirmation (complementing libcvc's C++
finite-difference gradchecks) that driving the C++ trainer from Python trains:
build a random .cvcnm, run Adam for a while on a fixed batch, assert the loss
drops substantially, then save + round-trip load. No torch required.
"""

from __future__ import annotations

import inspect
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


@pytest.mark.skipif(
    not mtn.HAS_MATERIAL_TRAINER_LOSS, reason="pycvc lacks nav_material_trainer_loss"
)
def test_forward_only_loss_does_not_train(tmp_path):
    """`loss()` scores a batch without moving the weights — the validation path."""
    ckpt = tmp_path / "init.cvcnm"
    _write_random_cvcnm(ckpt, patch_size=16, seed=3)
    trainer = mtn.MaterialTrainer(ckpt)
    batch = _synthetic_batch(seed=11)

    first = trainer.loss(batch)
    again = trainer.loss(batch)
    assert np.isfinite(first)
    # No optimizer, no backward: scoring twice must be identical.
    assert first == again, f"loss() moved the weights ({first} -> {again})"

    # ...and it agrees with the loss step() reports for the same pre-update weights.
    stepped = trainer.step(batch, lr=0.0)
    assert abs(stepped - first) / (abs(first) + 1e-6) < 1e-6


@pytest.mark.skipif(
    not mtn.HAS_MATERIAL_TRAINER_CUDA, reason="pycvc lacks the material CUDA surface"
)
def test_cuda_request_is_honest_about_what_it_uses():
    """`use_cuda` is a request; `cuda_active` reports the truth on this machine."""
    assert mtn.cuda_max_horizon() >= 0
    if not mtn.cuda_available():
        # No device: asking for CUDA must still work, and must admit it is on CPU.
        pytest.skip("no CUDA device on this machine")
    assert mtn.cuda_max_horizon() > 0


@pytest.mark.skipif(
    not mtn.HAS_MATERIAL_TRAINER_CUDA, reason="pycvc lacks the material CUDA surface"
)
def test_cuda_and_cpu_training_agree(tmp_path):
    """The GPU path must train, and track the CPU path it replaces."""
    if not mtn.cuda_available():
        pytest.skip("no CUDA device on this machine")
    batch = _synthetic_batch(seed=5)
    steps = 30
    out = {}
    for use_cuda in (False, True):
        ckpt = tmp_path / f"init_{int(use_cuda)}.cvcnm"
        _write_random_cvcnm(ckpt, patch_size=16, seed=2)  # same weights both runs
        tr = mtn.MaterialTrainer(ckpt, use_cuda=use_cuda)
        assert tr.cuda_active is use_cuda
        losses = [tr.step(batch, lr=mtn.cosine_lr(1e-3, s, steps)) for s in range(steps)]
        out[use_cuda] = (losses[0], losses[-1], tr.loss(batch))

    (_, c1, cval), (g0, g1, gval) = out[False], out[True]
    assert g1 < 0.9 * g0, f"GPU path did not train ({g0:.3f} -> {g1:.3f})"
    # Float-equivalent ops compounded over `steps` optimizer updates.
    assert abs(c1 - g1) / abs(c1) < 1e-3, f"CPU {c1:.6f} vs GPU {g1:.6f}"
    assert abs(cval - gval) / abs(cval) < 1e-3


# --------------------------------------------------------------------------- L_multi sampling

_needs_multistart = pytest.mark.skipif(
    not (mtn.HAS_MATERIAL_TRAINER_MULTISTART and mtn.HAS_MATERIAL_TRAINER_LOSS),
    reason="pycvc's nav_material_trainer_create lacks the L_multi sampling args (libcvc #596)",
)
_SAMPLED = dict(ms_count=4, frac_range=(0.8, 0.98))


def _close(x, y):
    return abs(x - y) / (abs(y) + 1e-6) < 1e-6


def test_multistart_probe_agrees_with_the_binding_signature():
    """The capability probe must not be fooled by its own argument types: SWIG answers a type
    mismatch with the same TypeError an old arity gives. The proxy signature cross-checks it."""
    try:
        params = inspect.signature(mtn._pycvc.nav_material_trainer_create).parameters
    except (TypeError, ValueError):
        pytest.skip("this pycvc's create has no Python-level signature to compare against")
    assert mtn.HAS_MATERIAL_TRAINER_MULTISTART == ("ms_semi_implicit" in params)


@_needs_multistart
def test_sampled_multi_start_args(tmp_path):
    """libcvc's test_sampled_multi_start_args, through the wrapper: a degenerate range is the
    legacy single start, a sampled range and the integrator switch reach the loss, validation
    draws with the base seed, and step k draws with seed + k."""
    ckpt = tmp_path / "init.cvcnm"
    _write_random_cvcnm(ckpt, patch_size=16, seed=5)
    batch = _synthetic_batch(seed=13)

    def loss(**kw):
        return mtn.MaterialTrainer(ckpt, **kw).loss(batch)

    legacy = loss()
    # The wrapper routes this through the legacy signature (no draw, one rollout)...
    assert loss(ms_count=10, frac_range=(0.9, 0.9), seed=123) == legacy
    # ...so check native's own degenerate handling on a configuration that needs the new args.
    semi_one = loss(semi_implicit=True)
    assert loss(ms_count=7, frac_range=(0.9, 0.9), seed=3, semi_implicit=True) == semi_one
    assert semi_one != legacy, "semi_implicit did not reach the rollout"

    sampled = mtn.MaterialTrainer(ckpt, seed=5, **_SAMPLED)
    base = sampled.loss(batch)
    assert base != legacy, "sampled starts did not reach the loss"
    assert sampled.loss(batch) == base, "validation loss is not deterministic"
    assert loss(seed=5, semi_implicit=True, **_SAMPLED) != base, "semi_implicit ignored"
    # Only the low 32 bits of the seed count, as with torch.Generator.manual_seed.
    assert loss(seed=5 + 2**32, **_SAMPLED) == base

    # At lr=0 the weights never move, so the k-th step scores what a trainer seeded seed + k
    # scores at its base seed. (step vs loss: the same value, compared to a tolerance.)
    tr = mtn.MaterialTrainer(ckpt, seed=5, **_SAMPLED)
    s0 = tr.step(batch, lr=0.0)
    s1 = tr.step(batch, lr=0.0)
    assert _close(s0, base), "step 0 did not draw with the base seed"
    assert not _close(s1, s0), "the draw did not advance between steps"
    assert _close(s1, loss(seed=6, **_SAMPLED)), "step 1 did not draw with seed + 1"
    assert tr.loss(batch) == base, "steps moved the validation draw"

    with pytest.raises(ValueError, match="frac_range"):
        mtn.MaterialTrainer(ckpt, ms_count=4, frac_range=(0.98, 0.8))


@_needs_multistart
@pytest.mark.parametrize("case", [0, 1, 2])
def test_semi_implicit_l_multi_matches_the_torch_reference(tmp_path, case):
    """Native L_multi with semi_implicit=True is GRL-SNAM's multi_start_penalty, fed the native
    model's coefficients and a manual_seed generator; the explicit default is not.

    L_multi is isolated as loss(w_multi=1) - loss(w_multi=0): every other term is the same
    deterministic function of the same weights and batch.
    """
    torch = pytest.importorskip("torch")
    from grl_snam.surrogate_robust import multi_start_penalty

    rng = np.random.default_rng(100 + case)
    seed = int(rng.integers(0, 2**32)) + (2**32 if case == 2 else 0)  # high bits must not matter
    ms_count = int(rng.integers(2, 7))
    lo = float(rng.uniform(0.5, 0.85))
    frac_range = (lo, float(rng.uniform(lo + 0.05, 0.99)))
    ckpt = tmp_path / "init.cvcnm"
    _write_random_cvcnm(ckpt, patch_size=16, seed=200 + case)
    b = _synthetic_batch(seed=300 + case)
    ms = dict(ms_count=ms_count, frac_range=frac_range, seed=seed)

    def native(semi):
        on, off = (
            mtn.MaterialTrainer(ckpt, w_multi=w, semi_implicit=semi, **ms).loss(b) for w in (1, 0)
        )
        return on - off

    B, N = b["C"].shape[:2]
    alphas, beta, gamma = mtn._pycvc.nav_matnet_forward(
        str(ckpt),
        b["obs_feats"].reshape(B * N, 6),
        b["obs_mask"].reshape(B * N),
        (np.arange(B + 1) * N).astype(np.int32),
        b["goal_feats"],
        b["risk_patch"],
    )[:3]
    t = torch.from_numpy
    ref = multi_start_penalty(
        t(b["o0"]), t(b["v0"]), t(b["goal"]), t(b["C"]), t(b["R"]), t(b["obs_mask"].astype(bool)),
        t(alphas.reshape(B, N)), t(beta), t(gamma), t(b["d_hat"]), t(b["dt"]), t(b["H"]),
        robot_radius=t(b["rr"]), margin_factor=0.5, ms_h=3, ms_dt_mult=4.0, tau=0.05,
        generator=torch.Generator().manual_seed(seed), ms_count=ms_count, frac_range=frac_range,
    ).item()  # fmt: skip

    semi, explicit = native(True), native(False)
    assert ref > 1e-3, "fixture must actually score near-obstacle starts"
    assert abs(semi - ref) <= 1e-5 * ref, f"semi-implicit native {semi!r} vs torch {ref!r}"
    assert abs(explicit - ref) > 1e-3 * ref, "the explicit default should NOT match"
