"""grl_snam.learning.stats: the seeded percentile bootstrap and paired differences."""

import random

import pytest

from grl_snam.learning import stats


def _reference_bootstrap(xs, n_boot, rng, alpha=0.05):
    """The harness's interval, written out: the mean of k draws, n_boot times, sorted, read at the
    percentile indices."""
    k = len(xs)
    means = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def test_bootstrap_matches_the_harness_draw_for_draw():
    xs = [0.1, 0.4, 0.35, 0.8, 0.2, 0.55, 0.9, 0.05]
    got = stats.bootstrap_ci(xs, 5000, random.Random(7))
    assert got == _reference_bootstrap(xs, 5000, random.Random(7))


def test_bootstrap_is_seeded_and_reproducible():
    xs = [float(i % 7) for i in range(30)]
    a = stats.bootstrap_ci(xs, 2000, 3)
    b = stats.bootstrap_ci(xs, 2000, random.Random(3))
    c = stats.bootstrap_ci(xs, 2000, 4)
    assert a == b
    assert a != c
    # The default generator is seed 1.
    assert stats.bootstrap_ci(xs, 500) == stats.bootstrap_ci(xs, 500, 1)
    lo, hi = a
    assert lo <= stats.mean(xs) <= hi


def test_bootstrap_degenerate_cases():
    assert stats.bootstrap_ci([2.5]) == (2.5, 2.5)
    lo, hi = stats.bootstrap_ci([])
    assert lo != lo and hi != hi  # NaN, as the harness's mean of nothing
    assert stats.bootstrap_ci([1.0, 1.0, 1.0], 100) == (1.0, 1.0)


def test_one_generator_is_consumed_in_order():
    rng = random.Random(11)
    first = stats.bootstrap_ci([1.0, 2.0, 3.0], 100, rng)
    second = stats.bootstrap_ci([1.0, 2.0, 3.0], 100, rng)
    fresh = random.Random(11)
    assert stats.bootstrap_ci([1.0, 2.0, 3.0], 100, fresh) == first
    assert stats.bootstrap_ci([1.0, 2.0, 3.0], 100, fresh) == second


def test_pairs_by_position_and_by_unit():
    assert stats.pairs([1, None, 3, float("nan")], [2, 5, None, 1]) == [(1.0, 2.0)]
    ref = {"s1": 1.0, "s2": 2.0, "s3": 3.0}
    arm = {"s3": 4.0, "s1": 1.5, "s9": 0.0}
    assert stats.pairs(ref, arm) == [(1.0, 1.5), (3.0, 4.0)]
    assert stats.differences(ref, arm) == [0.5, 1.0]
    with pytest.raises(TypeError):
        stats.pairs(ref, [1.0])


def test_paired_difference_keeps_pairs_together():
    # Noisy seeds, a constant shift: the paired interval has zero width, the unpaired ones do not.
    ref = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8]
    arm = [v + 0.25 for v in ref]
    d = stats.paired_diff(ref, arm, 1000, 1)
    assert d["n"] == 7
    assert d["mean"] == pytest.approx(0.25)
    assert d["ci95"] == pytest.approx([0.25, 0.25])
    assert stats.excludes_zero(d["ci95"])
    s = stats.summarize(arm, 1000, 1)
    assert s["ci95"][1] - s["ci95"][0] > 0.2
    assert stats.paired_diff([None], [1.0]) is None


def test_summarize_and_negate():
    s = stats.summarize([1.0, None, 3.0], 200, 5)
    assert s["n"] == 2 and s["mean"] == 2.0
    n = stats.negate({"mean": 0.5, "ci95": [0.1, 0.9], "n": 4})
    assert n == {"mean": -0.5, "ci95": [-0.9, -0.1], "n": 4}
    assert not stats.excludes_zero([-0.1, 0.2])
    assert stats.excludes_zero([-0.3, -0.1])
