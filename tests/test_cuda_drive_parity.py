"""The GPU drive (cvc::nav drive_step_cuda, nav/drive.cu) vs the CPU drive_step
and the torch reference.

Skipped unless pycvc was built with CUDA AND a CUDA device is present. Proves the
nav CUDA build works end-to-end and that the no-fast-math discipline
(--use_fast_math kept OFF for the nav .cu: -fmad=false, IEEE div/sqrt) keeps the
GPU drive float-equivalent to the CPU/torch reference. This is the validation
target on a workstation GPU; throughput is benchmarked on a bigger GPU box.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import sdf_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402
from grl_snam.tools import coef_export  # noqa: E402

VEH = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8, allow_reverse=True)


def _cuda_ready(tmp_path):
    if not nav_native.HAS_CUDA_DRIVE:
        return False
    # A tiny probe: raises if built without CUDA or no device.
    try:
        m = sdf_nav.CoefMLP().eval()
        wp = tmp_path / "probe.cvcnav"
        coef_export.write_coef_mlp(m, str(wp))
        field = np.zeros((1, 3, 8, 8), np.float32)
        o = np.zeros((1, 2), np.float32)
        nav_native.drive_step_cuda(
            field,
            o,
            np.zeros(1, np.float32),
            np.zeros(1, np.float32),
            o,
            wp,
            bounds=(-1, -1, 1, 1),
            center=(0, 0),
            scale=1.0,
            params=dict(rr=0.1, d_hat=0.2, dt=0.06, vmax=0.9, nsub=1, **VEH),
        )
        return True
    except Exception:
        return False


@pytest.mark.parametrize("nsub", [1, 2])
def test_cuda_drive_matches_torch(tmp_path, nsub):
    if not _cuda_ready(tmp_path):
        pytest.skip("no CUDA build / device")
    story = shrunk(STORIES["city"], n=96, max_steps=100)
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(story.truth_grid(), story.bounds, meta["scale"])
    sf = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    field = sf.field.numpy()
    torch.manual_seed(0)
    m = sdf_nav.CoefMLP().eval()
    wp = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(wp))

    rng = np.random.default_rng(1)
    n = 4000
    mnx, mny, mxx, mxy = story.bounds
    S = meta["scale"]
    o = np.stack([rng.uniform(mnx, mxx, n) * S, rng.uniform(mny, mxy, n) * S], 1).astype(np.float32)
    carrot = (o + rng.uniform(-3, 3, (n, 2))).astype(np.float32)
    th = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    sp = rng.uniform(0, meta["vmax"], n).astype(np.float32)
    P = dict(rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"], nsub=nsub, **VEH)

    with torch.no_grad():
        feat = sdf_nav.coef_feats(sf, torch.from_numpy(o), torch.from_numpy(carrot))
        al, be, ga = m(feat)
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
    go, gt, gs, _ = nav_native.drive_step_cuda(
        field,
        o,
        th,
        sp,
        carrot,
        wp,
        bounds=story.bounds,
        center=meta["center"],
        scale=meta["scale"],
        params=P,
    )
    assert np.allclose(go, ro.numpy(), rtol=1e-4, atol=1e-5), np.abs(go - ro.numpy()).max()
    assert np.allclose(gt, rt.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(gs, rs.numpy(), rtol=1e-4, atol=1e-5)


def test_cuda_matches_cpu(tmp_path):
    if not _cuda_ready(tmp_path):
        pytest.skip("no CUDA build / device")
    story = shrunk(STORIES["city"], n=96, max_steps=100)
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(story.truth_grid(), story.bounds, meta["scale"])
    sf = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    field = sf.field.numpy()
    m = sdf_nav.CoefMLP().eval()
    wp = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(m, str(wp))
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
    kw = dict(bounds=story.bounds, center=meta["center"], scale=meta["scale"], params=P)
    cpu = nav_native.drive_step(field, o, th, sp, carrot, wp, **kw)
    gpu = nav_native.drive_step_cuda(field, o, th, sp, carrot, wp, **kw)
    for a, b in zip(gpu, cpu):
        assert np.allclose(a, b, rtol=1e-4, atol=1e-5), np.abs(a - b).max()
