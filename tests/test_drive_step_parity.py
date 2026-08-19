"""The fused torch-free drive tick (cvc::nav::drive_step) vs the torch Swarm drive.

drive_step = sample -> coef_feats -> coef_mlp -> bicycle_rollout in one C++ call.
This validates the whole per-agent drive end-to-end against torch
(sdf_nav.coef_feats + CoefMLP + bicycle_rollout) — the assembly of the P1-P3
pieces (docs/CVCNAV_CPP_PORT_ROADMAP.md P4). The carrot is given here; the carrot
FSM that produces it rides with the SoA sim_world (P6).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import sdf_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402
from grl_snam.tools import coef_export  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (nav_native.HAS_DRIVE and hasattr(nav_native._pycvc, "nav_drive_step")),
    reason="pycvc build lacks nav_drive_step",
)

VEH = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8, allow_reverse=True)


def _scene():
    story = shrunk(STORIES["city"], n=96, max_steps=100)
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(story.truth_grid(), story.bounds, meta["scale"])
    sf = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    return sf, story.bounds, meta


@pytest.mark.parametrize("nsub", [1, 2])
def test_drive_step_matches_torch(tmp_path, nsub):
    sf, bounds, meta = _scene()
    field = sf.field.numpy()
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP()
    model.eval()
    path = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(model, str(path))

    rng = np.random.default_rng(1)
    n = 3000
    mnx, mny, mxx, mxy = bounds
    S = meta["scale"]
    o = np.stack([rng.uniform(mnx, mxx, n) * S, rng.uniform(mny, mxy, n) * S], 1).astype(np.float32)
    carrot = (o + rng.uniform(-3, 3, (n, 2))).astype(np.float32)
    th = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    sp = rng.uniform(0, meta["vmax"], n).astype(np.float32)
    P = dict(rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"], nsub=nsub, **VEH)

    # torch reference: coef_feats -> CoefMLP -> bicycle_rollout.
    with torch.no_grad():
        feat = sdf_nav.coef_feats(sf, torch.from_numpy(o), torch.from_numpy(carrot))
        al, be, ga = model(feat)
        ro, rt, rs, _ = sdf_nav.bicycle_rollout(
            sf,
            torch.from_numpy(o.copy()),
            torch.from_numpy(th.copy()),
            torch.from_numpy(sp.copy()),
            torch.from_numpy(carrot),
            al,
            be,
            ga,
            1,
            nsub=nsub,
            rr=P["rr"],
            d_hat=P["d_hat"],
            dt=P["dt"],
            vmax=P["vmax"],
            **VEH,
        )

    co, ct, cs, _ = nav_native.drive_step(
        field,
        o,
        th,
        sp,
        carrot,
        str(path),
        bounds=bounds,
        center=meta["center"],
        scale=meta["scale"],
        params=P,
    )
    assert np.allclose(co, ro.numpy(), rtol=1e-4, atol=1e-5), np.abs(co - ro.numpy()).max()
    assert np.allclose(ct, rt.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(cs, rs.numpy(), rtol=1e-4, atol=1e-5)


def test_drive_step_deterministic_across_threads(tmp_path):
    sf, bounds, meta = _scene()
    field = sf.field.numpy()
    torch.manual_seed(0)
    model = sdf_nav.CoefMLP().eval()
    path = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(model, str(path))
    rng = np.random.default_rng(2)
    n = 2000
    S = meta["scale"]
    o = np.stack([rng.uniform(-100, 100, n) * S, rng.uniform(-100, 100, n) * S], 1).astype(
        np.float32
    )
    carrot = (o + rng.uniform(-3, 3, (n, 2))).astype(np.float32)
    th = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    sp = rng.uniform(0, meta["vmax"], n).astype(np.float32)
    P = dict(rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"], nsub=2, **VEH)
    a = nav_native.drive_step(
        field,
        o,
        th,
        sp,
        carrot,
        str(path),
        bounds=bounds,
        center=meta["center"],
        scale=meta["scale"],
        params=P,
        num_threads=1,
    )
    b = nav_native.drive_step(
        field,
        o,
        th,
        sp,
        carrot,
        str(path),
        bounds=bounds,
        center=meta["center"],
        scale=meta["scale"],
        params=P,
        num_threads=8,
    )
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
