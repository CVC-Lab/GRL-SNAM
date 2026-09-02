"""The parts of ``material_train_native`` that need no pycvc at all.

The rest of that module's tests (test_material_train_native.py) skip wherever the
installed pycvc lacks the trainer binding — which is every CI runner today, since
nothing material-aware has been published to cvcpkg.org yet. These cover the
pure-Python contract that still has to hold there: the batch ordering and dtype
coercion the binding rejects on mismatch, the LR schedule, and the capability
gates that must degrade to a clear error rather than a segfault or a silent CPU
run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grl_snam import material_train_native as mtn


def test_batch_keys_are_the_binding_contract():
    """17 arrays, in the order the C binding takes them positionally."""
    assert len(mtn.BATCH_KEYS) == 17
    assert len(set(mtn.BATCH_KEYS)) == 17, "duplicate key would silently shift the argument list"
    # The two non-float columns; everything else is f32.
    assert mtn._U8_KEYS == {"obs_mask"}
    assert mtn._I32_KEYS == {"H"}
    assert mtn._U8_KEYS | mtn._I32_KEYS <= set(mtn.BATCH_KEYS)


def _batch(B=3, N=2, P=4, Hp=5, Wp=5):
    """A shape-correct batch in deliberately WRONG dtypes, to prove coercion."""
    return {
        "obs_feats": np.zeros((B, N, 6), np.float64),
        "obs_mask": np.ones((B, N), np.int64),
        "goal_feats": np.zeros((B, 4), np.float64),
        "risk_patch": np.zeros((B, 2, P, P), np.float64),
        "o0": np.zeros((B, 2), np.float64),
        "v0": np.zeros((B, 2), np.float64),
        "goal": np.zeros((B, 2), np.float64),
        "C": np.zeros((B, N, 2), np.float64),
        "R": np.zeros((B, N), np.float64),
        "rollout_patch": np.zeros((B, 6, Hp, Wp), np.float64),
        "rr": np.zeros(B, np.float64),
        "d_hat": np.zeros(B, np.float64),
        "dt": np.zeros(B, np.float64),
        "H": np.full(B, 3, np.int64),
        "o_tgt": np.zeros((B, 2), np.float64),
        "v_tgt": np.zeros((B, 2), np.float64),
        "gamma_o": np.zeros(B, np.float64),
    }


def test_pack_batch_orders_and_coerces_dtypes():
    packed = mtn.pack_batch(_batch())
    assert len(packed) == len(mtn.BATCH_KEYS)
    for arr, key in zip(packed, mtn.BATCH_KEYS):
        want = (
            np.uint8 if key in mtn._U8_KEYS else (np.int32 if key in mtn._I32_KEYS else np.float32)
        )
        assert arr.dtype == want, f"{key}: {arr.dtype} != {want}"
        assert arr.flags["C_CONTIGUOUS"], f"{key} not C-contiguous"


def test_pack_batch_makes_non_contiguous_input_contiguous():
    """A transposed / sliced array must be copied, not passed through."""
    b = _batch()
    b["risk_patch"] = np.asfortranarray(b["risk_patch"])
    b["o0"] = np.zeros((3, 4), np.float32)[:, ::2]  # strided view
    for arr in mtn.pack_batch(b):
        assert arr.flags["C_CONTIGUOUS"]


def test_pack_batch_reports_a_missing_key():
    b = _batch()
    del b["gamma_o"]
    with pytest.raises(KeyError):
        mtn.pack_batch(b)


def test_cosine_lr_matches_the_torch_schedule():
    lr0, t_max, frac = 1e-3, 50, 0.1
    eta_min = lr0 * frac
    assert mtn.cosine_lr(lr0, 0, t_max) == pytest.approx(lr0)
    assert mtn.cosine_lr(lr0, t_max, t_max) == pytest.approx(eta_min)
    # Half way through a cosine anneal sits at the midpoint of [eta_min, lr0].
    assert mtn.cosine_lr(lr0, t_max // 2, t_max) == pytest.approx((lr0 + eta_min) / 2, rel=1e-6)
    # Monotone non-increasing.
    seq = [mtn.cosine_lr(lr0, t, t_max) for t in range(t_max + 1)]
    assert all(b <= a + 1e-15 for a, b in zip(seq, seq[1:]))
    # Clamped outside [0, t_max] rather than diverging.
    assert mtn.cosine_lr(lr0, -5, t_max) == pytest.approx(lr0)
    assert mtn.cosine_lr(lr0, t_max + 99, t_max) == pytest.approx(eta_min)
    assert math.isfinite(mtn.cosine_lr(lr0, 0, 0))  # t_max=0 must not divide by zero


def test_capability_flags_are_consistent():
    """The newer surfaces can never claim more than the base binding."""
    assert isinstance(mtn.HAS_MATERIAL_TRAINER, bool)
    if not mtn.HAS_MATERIAL_TRAINER:
        assert not mtn.HAS_MATERIAL_TRAINER_LOSS
        assert not mtn.HAS_MATERIAL_TRAINER_CUDA


def test_cuda_probes_degrade_without_the_binding():
    """They must answer safely, not raise, on a pycvc that lacks the CUDA surface."""
    assert isinstance(mtn.cuda_available(), bool)
    assert isinstance(mtn.cuda_max_horizon(), int)
    assert mtn.cuda_max_horizon() >= 0
    if not mtn.HAS_MATERIAL_TRAINER_CUDA:
        assert mtn.cuda_available() is False
        assert mtn.cuda_max_horizon() == 0


@pytest.mark.skipif(mtn.HAS_MATERIAL_TRAINER, reason="pycvc HAS the trainer binding")
def test_construction_without_the_binding_explains_itself():
    with pytest.raises(RuntimeError, match="nav_material_trainer_create"):
        mtn.MaterialTrainer("does-not-matter.cvcnm")


@pytest.mark.skipif(
    not mtn.HAS_MATERIAL_TRAINER or mtn.HAS_MATERIAL_TRAINER_CUDA,
    reason="needs a trainer binding built WITHOUT the CUDA surface",
)
def test_cuda_request_without_cuda_support_raises_rather_than_silently_using_cpu():
    with pytest.raises(RuntimeError, match="use_cuda"):
        mtn.MaterialTrainer("does-not-matter.cvcnm", use_cuda=True)
