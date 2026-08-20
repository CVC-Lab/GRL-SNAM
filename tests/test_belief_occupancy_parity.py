"""The torch-free C++ belief->occupancy (cvc::nav to_occupancy / composite) vs
the numpy reference (grl_snam.belief).

This is the one NEW bit surface between the raw belief and the (already
bit-identical) EDT/build_sdf, so it must be byte-exact: a float32-vs-float64
sigmoid or a threshold-compare mismatch flips a cell near the boundary and the
whole downstream field changes (docs/CVCNAV_CPP_PORT_ROADMAP.md §1, P5). Checked
on a realistically evolved belief for both unknown-space policies, with and
without the decaying dynamic layer.
"""

import numpy as np
import pytest

pytest.importorskip("pycvc")

from grl_snam import nav_native  # noqa: E402
from grl_snam.belief import BeliefGrid, DynamicLayer, composite_occupancy  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402

pytestmark = pytest.mark.skipif(
    not nav_native.HAS_OCCUPANCY, reason="pycvc build lacks nav_composite_occupancy"
)


def _evolved_belief(n=192, senses=40):
    story = shrunk(STORIES["city"], n=n, max_steps=100)
    truth = story.truth_grid()
    H, W = truth.shape
    b = BeliefGrid((H, W), story.bounds)
    dyn = DynamicLayer((H, W), ttl_s=4.0)
    rng = np.random.default_rng(0)
    mnx, mny, mxx, mxy = story.bounds
    for _ in range(senses):
        b.sense(
            truth,
            (rng.uniform(mnx, mxx), rng.uniform(mny, mxy)),
            range_m=60.0,
            n_rays=240,
            heading_rad=rng.uniform(-np.pi, np.pi),
        )
    for _ in range(30):
        dyn.mark(rng.integers(0, H), rng.integers(0, W), 2.0, radius_cells=2)
    return b, dyn


@pytest.mark.parametrize("unknown", ["optimistic", "pessimistic"])
def test_to_occupancy_bit_identical(unknown):
    b, _ = _evolved_belief()
    ref = b.to_occupancy(unknown=unknown)
    got = nav_native.to_occupancy(b.logodds, unknown=unknown)
    assert np.array_equal(ref, got), f"{int((ref != got).sum())} cells flipped"


@pytest.mark.parametrize("unknown", ["optimistic", "pessimistic"])
def test_composite_occupancy_bit_identical(unknown):
    b, dyn = _evolved_belief()
    t_now = 3.5
    ref = composite_occupancy(b, dyn, t_now, unknown=unknown)
    got = nav_native.composite_occupancy(b.logodds, dyn._stamp, t_now, dyn.ttl_s, unknown=unknown)
    assert np.array_equal(ref, got), f"{int((ref != got).sum())} cells flipped"


def test_composite_feeds_bit_identical_field():
    """The whole point: the C++ occupancy must produce the SAME SDF field as the
    numpy occupancy (the fidelity boundary is the built field)."""
    import sdf_nav

    b, dyn = _evolved_belief()
    story = shrunk(STORIES["city"], n=192, max_steps=100)
    ref_occ = composite_occupancy(b, dyn, 3.5)
    got_occ = nav_native.composite_occupancy(b.logodds, dyn._stamp, 3.5, dyn.ttl_s)
    rp = sdf_nav.build_sdf(ref_occ, story.bounds, story.meta()["scale"])
    gp = sdf_nav.build_sdf(got_occ, story.bounds, story.meta()["scale"])
    for a, c in zip(rp, gp):
        assert np.array_equal(a, c)
