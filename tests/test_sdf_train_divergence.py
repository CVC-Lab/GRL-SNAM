"""``grl-snam train`` on dense geometry: its gradient is bounded, and the collision term has one.

On Austin (1 unit = 300 m) the retrain clipped every optimizer step, with pre-clip gradient
norms up to 7e13, and did not descend its own loss (docs/learning-validation.md §3.2, §8). Two
causes, both fixed in ``train_sdf``'s defaults:

- the backward ran through the whole 28-step rollout, and on dense geometry the rollout is
  chaotic near walls, so the gradient grew exponentially with the horizon (``bptt_window``);
- the collision term's clearance was detached, so ``col`` had no gradient at all (``col_grad``).

The fixture is Austin's frame in small: 1 unit = 300 m, rr = 3 m = 0.01 units, a 3 m SDF cell,
and buildings over ~40% of the ground.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grl_snam.sdf_nav as sdf_nav
from grl_snam import learning_history as lh
from grl_snam.tools.train import train_sdf

SCALE = 1.0 / 300.0
RUN = dict(steps=12, batch=32, horizon=28, threads=1, device="cpu", log_every=1000, seed=0)


@pytest.fixture(scope="module")
def dense_npz(tmp_path_factory):
    n, half = 96, 144.0
    rng = np.random.default_rng(0)
    occ = np.zeros((n, n), bool)
    for _ in range(n * n // 110):
        w, h = rng.integers(4, 12, 2)
        x, y = rng.integers(0, n - 12, 2)
        occ[y : y + h, x : x + w] = True
    bounds = (-half, -half, half, half)
    phi, nx, ny = sdf_nav.build_sdf(occ, bounds, SCALE)
    path = tmp_path_factory.mktemp("dense") / "nav_sdf.npz"
    np.savez(
        path, phi=phi, normal_x=nx, normal_y=ny, bounds=np.array(bounds, np.float32),
        center=np.zeros(2, np.float32), scale=np.float32(SCALE), region=np.float32(0.95 * half),
    )  # fmt: skip
    return str(path)


def _steps(path):
    return [r for r in lh.read_history(path) if r["kind"] == "step"]


def test_the_default_trainer_s_gradient_is_bounded_on_dense_geometry(dense_npz, tmp_path):
    train_sdf(dense_npz, str(tmp_path / "a.pt"), history=str(tmp_path / "a.jsonl"), **RUN)
    steps = _steps(tmp_path / "a.jsonl")
    norms = [r["grad_norm"]["mean"] for r in steps]
    clipped = np.mean([r["grad_norm"]["clipped_frac"] for r in steps])
    # Full BPTT here: median ~1e3-1e4 and every step clipped. Truncated at 7: median ~2-4.
    assert np.median(norms) < 50.0, norms
    assert clipped <= 0.5, clipped
    loss = [r["terms"]["loss"] for r in steps]
    assert all(np.isfinite(loss))


def test_the_trainer_before_the_fix_still_explodes_there(dense_npz, tmp_path):
    """``bptt_window=0, col_grad=False`` is the old trainer, kept to reproduce the published
    weights; on the same draws its gradient is the exploding one."""
    train_sdf(
        dense_npz, str(tmp_path / "b.pt"), history=str(tmp_path / "b.jsonl"), bptt_window=0,
        col_grad=False, grad_terms_every=1, **RUN,
    )  # fmt: skip
    steps = _steps(tmp_path / "b.jsonl")
    assert np.median([r["grad_norm"]["mean"] for r in steps]) > 200.0
    assert np.mean([r["grad_norm"]["clipped_frac"] for r in steps]) > 0.9
    assert all(r["grad_norm"]["terms"]["col"] == 0.0 for r in steps)  # col had no gradient


def test_the_collision_term_carries_gradient(dense_npz, tmp_path):
    train_sdf(
        dense_npz, str(tmp_path / "c.pt"), history=str(tmp_path / "c.jsonl"),
        grad_terms_every=3, **RUN,
    )  # fmt: skip
    steps = _steps(tmp_path / "c.jsonl")
    measured = [r["grad_norm"]["terms"] for r in steps if "terms" in r["grad_norm"]]
    assert [r["step"] for r in steps if "terms" in r["grad_norm"]] == [1, 4, 7, 10]
    assert all(t["col"] > 0.0 and t["goal"] > 0.0 for t in measured)
    assert set(measured[0]) == {"goal", "col", "reg"}


def test_window_and_col_grad_change_no_value_of_the_objective(dense_npz):
    from grl_snam.tools import sdf_eval

    scene = sdf_eval.load_scene(dense_npz)
    rr = 3.0 * SCALE
    pool = sdf_eval.free_pool(scene, rr)
    starts, goals = sdf_eval.sample_pairs(scene, pool, 64, 1, rr=rr)
    kw = dict(rr=rr, d_hat=0.35, dt=0.06, vmax=0.9, horizon=28)
    model = (torch.manual_seed(0), sdf_nav.CoefMLP())[1]
    ref = sdf_eval.objective_terms(model, scene, starts, goals, window=0, col_grad=False, **kw)
    for window, col_grad in ((7, True), (3, False), (1, True)):
        got = sdf_eval.objective_terms(
            model, scene, starts, goals, window=window, col_grad=col_grad, **kw
        )
        assert got == pytest.approx(ref, rel=1e-6, abs=1e-7), (window, col_grad)
    # The truncated, collision-aware gradient is finite and reaches all three heads.
    t, (al, be, ga) = sdf_eval.loss_terms(
        model, scene.field, starts, goals, window=7, col_grad=True, **kw
    )
    g = torch.autograd.grad(t["col"], [al, be, ga])
    assert all(torch.isfinite(x).all() for x in g)
    assert float(g[0].abs().sum()) > 0.0  # the barrier gain alpha moves the clearance
