"""The torch-free libcvc trainer, dispatched from Python behind a feature flag.

The canonical trainer is torch (grl_snam.tools.coef_train.train). This exercises
the OPT-IN native path — cvc::nav::coef_train via pycvc — which trains the CoefMLP
with NO torch and writes the same versioned .cvcnav the torch exporter does. Both
rollout integrators (surrogate / bicycle) are covered, plus the coef_train.py
--backend native dispatch. Skips cleanly when pycvc lacks the trainer.
"""

import numpy as np
import pytest

pytest.importorskip("pycvc")

from grl_snam import nav_native  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (nav_native.HAS_TRAIN and nav_native.HAS_COEF_MLP),
    reason="pycvc built without cvc::nav::coef_train",
)


def _city(grid=48):
    story = shrunk(STORIES["city"], n=grid, max_steps=100)
    return story, story.meta()


@pytest.mark.parametrize("rollout", ["surrogate", "bicycle"])
def test_native_trainer_writes_loadable_cvcnav(tmp_path, rollout):
    story, meta = _city()
    out = str(tmp_path / f"native_{rollout}.cvcnav")
    lr = 1e-5 if rollout == "bicycle" else 2e-4
    ret = nav_native.train_coef_mlp(
        story.truth_grid().astype(np.uint8),
        out,
        bounds=story.bounds,
        scale=meta["scale"],
        rr=meta["rr"],
        d_hat=meta["d_hat"],
        dt=meta["dt"],
        vmax=meta["vmax"],
        steps=25,
        n=32,
        hidden=16,
        lr=lr,
        rollout=rollout,
    )
    assert ret == out

    # The trained .cvcnav loads and produces valid (softplus > 0) coefficients.
    feat = np.array([[0.3, 3.0, 0.7, 0.71, 0.0]], np.float32)
    coef = np.asarray(nav_native.coef_mlp_forward(out, feat))
    assert coef.shape == (1, 3)
    assert np.all(np.isfinite(coef))
    assert np.all(coef > 0.0)


def test_coef_train_backend_flag(tmp_path):
    # The coef_train.py CLI dispatches to the native trainer on --backend native
    # (torch stays the default). Importing coef_train needs torch (via sdf_nav);
    # the native training itself does not.
    pytest.importorskip("torch")
    from grl_snam.tools import coef_train

    out = str(tmp_path / "cli_native.cvcnav")
    coef_train.main(["--backend", "native", "--out", out, "--steps", "20"])
    coef = np.asarray(
        nav_native.coef_mlp_forward(out, np.array([[0.2, 2.0, 1.0, 0.0, 0.0]], np.float32))
    )
    assert coef.shape == (1, 3) and np.all(coef > 0.0)
