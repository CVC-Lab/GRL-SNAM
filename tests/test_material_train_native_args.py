"""MaterialTrainer's create-argument contract, against a fake binding so it runs without pycvc.

The L_multi sampling args (``ms_count``, ``frac_range``, ``seed``, ``semi_implicit``) are new
trailing positionals on ``nav_material_trainer_create`` (libcvc #596), and an older binding rejects
the extra arguments with TypeError. So on an older binding the wrapper must keep calling the old
signature for the legacy single start (explicit Euler, ``semi_implicit=False``). It may refuse,
with a clear error, only a configuration that needs the new args, and the default
``semi_implicit=True`` is one. A binding that takes them always gets them, the legacy start
included: the binding's own ``ms_semi_implicit`` default changed with libcvc #981 (0, then 1), so
leaving them out would leave the integrator to the binding's version.
tests/test_material_train_native.py checks the same args against a real pycvc.
"""

from __future__ import annotations

import pytest

from grl_snam import material_train_native as mtn

_LEGACY_ARITY = 23  # path, grad_clip..tau (19 doubles), ms_h, ms_dt_mult, use_cuda
_NEW_ARITY = _LEGACY_ARITY + 5  # + ms_count, frac_lo, frac_hi, seed, ms_semi_implicit
_SAMPLED = dict(ms_count=4, frac_range=(0.8, 0.98), seed=5, semi_implicit=True)


class _FakeBinding:
    """Records create() calls and rejects an unknown arity, as the SWIG proxy does."""

    def __init__(self, takes_multi_start: bool):
        self.max_args = _NEW_ARITY if takes_multi_start else _LEGACY_ARITY
        self.calls: list[tuple] = []

    def nav_material_trainer_create(self, *args):
        if len(args) > self.max_args:
            raise TypeError(
                f"nav_material_trainer_create() takes from 1 to {self.max_args} positional "
                f"arguments but {len(args)} were given"
            )
        if len(args) == _NEW_ARITY and not args[24] <= args[25]:
            raise RuntimeError("cvc::nav::multi_start: the frac range must satisfy lo <= hi")
        self.calls.append(args)
        return object()


@pytest.fixture
def binding(monkeypatch):
    """Install a fake binding; the capability flag comes from the real probe run against it."""

    def install(takes_multi_start: bool) -> _FakeBinding:
        fake = _FakeBinding(takes_multi_start)
        monkeypatch.setattr(mtn, "_pycvc", fake)
        monkeypatch.setattr(mtn, "HAS_MATERIAL_TRAINER", True)
        monkeypatch.setattr(mtn, "HAS_MATERIAL_TRAINER_MULTISTART", mtn._takes_multi_start_args())
        return fake

    return install


@pytest.mark.parametrize("takes", [False, True])
def test_probe_reads_the_binding_without_building_a_trainer(binding, takes):
    fake = binding(takes)
    assert mtn.HAS_MATERIAL_TRAINER_MULTISTART is takes
    assert fake.calls == []


_EXPLICIT = dict(semi_implicit=False)
_LEGACY_STARTS = pytest.mark.parametrize(
    ("kw", "tail"),
    [
        (dict(_EXPLICIT), (1, 0.9, 0.9, 0, 0)),
        (dict(_EXPLICIT, ms_count=10), (10, 0.9, 0.9, 0, 0)),
        (dict(_EXPLICIT, ms_count=10, frac_range=(0.9, 0.9), seed=123), (10, 0.9, 0.9, 123, 0)),
    ],
    ids=["explicit", "ms_count", "degenerate-seeded"],
)


@_LEGACY_STARTS
def test_the_legacy_start_keeps_the_old_signature_on_an_old_binding(binding, kw, tail):
    """A degenerate 0.9 range with explicit Euler draws nothing and runs one rollout, whatever
    ms_count (>= 1) and seed are: it is the legacy start, so it must not need the new binding."""
    fake = binding(False)
    mtn.MaterialTrainer("m.cvcnm", **kw)
    (args,) = fake.calls
    assert len(args) == _LEGACY_ARITY
    assert args[-1] == 0  # use_cuda


@_LEGACY_STARTS
def test_the_legacy_start_says_explicit_euler_to_a_new_binding(binding, kw, tail):
    """libcvc #981 made the binding's ms_semi_implicit default 1. The legacy start is explicit
    Euler, so a binding that takes the args gets them all, with ms_semi_implicit 0."""
    fake = binding(True)
    mtn.MaterialTrainer("m.cvcnm", **kw)
    (args,) = fake.calls
    assert len(args) == _NEW_ARITY
    assert args[_LEGACY_ARITY - 1] == 0  # use_cuda
    assert args[_LEGACY_ARITY:] == tail
    assert [type(a) for a in args[_LEGACY_ARITY:]] == [int, float, float, int, int]


def test_the_default_is_semi_implicit(binding):
    """As libcvc's own default since #981: one start from the 0.9 range, ms_semi_implicit 1. An
    older binding cannot run it, and says what to pass instead."""
    fake = binding(True)
    mtn.MaterialTrainer("m.cvcnm")
    (args,) = fake.calls
    assert len(args) == _NEW_ARITY
    assert args[_LEGACY_ARITY:] == (1, 0.9, 0.9, 0, 1)

    fake = binding(False)
    with pytest.raises(RuntimeError, match="semi_implicit=False"):
        mtn.MaterialTrainer("m.cvcnm")
    assert fake.calls == []


def test_sampled_args_are_appended_after_the_legacy_ones(binding):
    fake = binding(True)
    mtn.MaterialTrainer("m.cvcnm", **_EXPLICIT)
    mtn.MaterialTrainer("m.cvcnm", **_SAMPLED)
    legacy, sampled = fake.calls
    assert sampled[:_LEGACY_ARITY] == legacy[:_LEGACY_ARITY]
    tail = sampled[_LEGACY_ARITY:]
    assert tail == (4, 0.8, 0.98, 5, 1)
    assert [type(a) for a in tail] == [int, float, float, int, int]


@pytest.mark.parametrize(
    ("seed", "want"), [(5 + 2**32, 5), (2**32 - 1, 2**32 - 1), (-1, 2**32 - 1)]
)
def test_only_the_seeds_low_32_bits_are_passed(binding, seed, want):
    """As torch.Generator.manual_seed: its mt19937 keeps the low 32 bits (a negative seed wraps)."""
    fake = binding(True)
    mtn.MaterialTrainer("m.cvcnm", **{**_SAMPLED, "seed": seed})
    assert fake.calls[0][26] == want


@pytest.mark.parametrize(
    "kw",
    [
        dict(_SAMPLED),
        {},
        dict(semi_implicit=True),
        dict(_EXPLICIT, frac_range=(0.8, 0.8)),
        dict(_EXPLICIT, ms_count=0),
    ],
    ids=["sampled", "defaults", "semi-implicit", "other-degenerate-frac", "ms_count=0"],
)
def test_configs_beyond_the_legacy_start_need_the_new_binding(binding, kw):
    fake = binding(False)
    with pytest.raises(RuntimeError, match="libcvc #596"):
        mtn.MaterialTrainer("m.cvcnm", **kw)
    assert fake.calls == []

    fake = binding(True)
    mtn.MaterialTrainer("m.cvcnm", **kw)
    (args,) = fake.calls
    assert len(args) == _NEW_ARITY


@pytest.mark.parametrize("takes", [False, True])
@pytest.mark.parametrize("bad", [(-0.1, 0.5), (0.5, 1.0), (0.9, 0.8), (float("nan"),) * 2])
def test_a_bad_frac_range_is_a_value_error_on_any_binding(binding, takes, bad):
    """Checked in Python, with multi_start_penalty's rule, before the capability check."""
    fake = binding(takes)
    with pytest.raises(ValueError, match="frac_range"):
        mtn.MaterialTrainer("m.cvcnm", ms_count=4, frac_range=bad)
    assert fake.calls == []
