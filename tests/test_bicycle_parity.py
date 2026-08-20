"""The torch-free C++ drive numerics (cvc::nav coef_feats + bicycle_rollout) vs
the torch reference (sdf_nav.coef_feats / bicycle_rollout).

These are the two float-equivalent pieces of the pure-C++ drive
(docs/CVCNAV_CPP_PORT_ROADMAP.md P3). The carrot FSM rides with the SoA sim_world
(P6). Contract: per-substep float-equivalence (~1e-4); a multi-step trajectory
stays bounded. On this platform the residual is float32 epsilon (~1e-7).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc")

import sdf_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.fog_stories import STORIES, shrunk  # noqa: E402

pytestmark = pytest.mark.skipif(
    not nav_native.HAS_DRIVE, reason="pycvc build lacks nav_bicycle_rollout"
)

VEH = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8, allow_reverse=True)


def _field():
    story = shrunk(STORIES["city"], n=96, max_steps=100)
    meta = story.meta()
    phi, nxg, nyg = sdf_nav.build_sdf(story.truth_grid(), story.bounds, meta["scale"])
    sf = sdf_nav.SDFField(phi, nxg, nyg, story.bounds, meta["center"], meta["scale"])
    return sf, story.bounds, meta


def _pose(bounds, scale, n, seed):
    rng = np.random.default_rng(seed)
    mnx, mny, mxx, mxy = bounds
    o = np.stack([rng.uniform(mnx, mxx, n) * scale, rng.uniform(mny, mxy, n) * scale], 1).astype(
        np.float32
    )
    carrot = (o + rng.uniform(-3, 3, (n, 2))).astype(np.float32)
    th = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    sp = rng.uniform(0, 0.9, n).astype(np.float32)
    al = rng.uniform(0.3, 4, n).astype(np.float32)
    be = rng.uniform(0.3, 6, n).astype(np.float32)
    ga = rng.uniform(0.3, 6, n).astype(np.float32)
    return o, carrot, th, sp, al, be, ga


def test_coef_feats_matches_torch():
    sf, bounds, meta = _field()
    field = sf.field.numpy()
    o, carrot, *_ = _pose(bounds, meta["scale"], 3000, 0)
    ref = sdf_nav.coef_feats(sf, torch.from_numpy(o), torch.from_numpy(carrot)).numpy()
    got = nav_native.coef_feats(
        field, o, carrot, bounds=bounds, center=meta["center"], scale=meta["scale"]
    )
    assert got.shape == (3000, 5)
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5), np.abs(got - ref).max()


@pytest.mark.parametrize("nsub", [1, 2, 4])
def test_bicycle_rollout_matches_torch(nsub):
    sf, bounds, meta = _field()
    field = sf.field.numpy()
    o, carrot, th, sp, al, be, ga = _pose(bounds, meta["scale"], 3000, 1)
    P = dict(rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"], nsub=nsub, **VEH)
    ro, rt, rs, rm = sdf_nav.bicycle_rollout(
        sf,
        torch.from_numpy(o.copy()),
        torch.from_numpy(th.copy()),
        torch.from_numpy(sp.copy()),
        torch.from_numpy(carrot),
        torch.from_numpy(al),
        torch.from_numpy(be),
        torch.from_numpy(ga),
        1,
        nsub=nsub,
        rr=P["rr"],
        d_hat=P["d_hat"],
        dt=P["dt"],
        vmax=P["vmax"],
        **VEH,
    )
    co, ct, cs, cm = nav_native.bicycle_rollout(
        field,
        o,
        th,
        sp,
        carrot,
        al,
        be,
        ga,
        bounds=bounds,
        center=meta["center"],
        scale=meta["scale"],
        params=P,
    )
    assert np.allclose(co, ro.numpy(), rtol=1e-4, atol=1e-5), np.abs(co - ro.numpy()).max()
    assert np.allclose(ct, rt.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(cs, rs.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(cm, rm.numpy(), rtol=1e-4, atol=1e-5)


def test_multistep_trajectory_stays_bounded():
    """Loop the drive: the C++ and torch trajectories track over many ticks (a
    fixed carrot, so no FSM branching — pure integrator drift)."""
    sf, bounds, meta = _field()
    field = sf.field.numpy()
    o, carrot, th, sp, al, be, ga = _pose(bounds, meta["scale"], 500, 2)
    P = dict(rr=meta["rr"], d_hat=meta["d_hat"], dt=meta["dt"], vmax=meta["vmax"], nsub=2, **VEH)
    co, cth, csp = o.copy(), th.copy(), sp.copy()
    ro = torch.from_numpy(o.copy())
    rth, rsp = torch.from_numpy(th.copy()), torch.from_numpy(sp.copy())
    gt = torch.from_numpy(carrot)
    tal, tbe, tga = torch.from_numpy(al), torch.from_numpy(be), torch.from_numpy(ga)
    for _ in range(30):
        co, cth, csp, _ = nav_native.bicycle_rollout(
            field,
            co,
            cth,
            csp,
            carrot,
            al,
            be,
            ga,
            bounds=bounds,
            center=meta["center"],
            scale=meta["scale"],
            params=P,
        )
        ro, rth, rsp, _ = sdf_nav.bicycle_rollout(
            sf,
            ro,
            rth,
            rsp,
            gt,
            tal,
            tbe,
            tga,
            1,
            nsub=2,
            rr=P["rr"],
            d_hat=P["d_hat"],
            dt=P["dt"],
            vmax=P["vmax"],
            **VEH,
        )
    assert np.abs(co - ro.numpy()).max() < 1e-3, np.abs(co - ro.numpy()).max()


def test_clustered_map_id_gather():
    sf, bounds, meta = _field()
    M = 4
    # M distinct fields (jitter the base so planes differ).
    rng = np.random.default_rng(3)
    base = sf.field.numpy()[0]
    planes = [
        sdf_nav.SDFField(
            *(
                base[c] + rng.standard_normal(base[c].shape).astype(np.float32) * 0.3
                for c in range(3)
            ),
            bounds,
            meta["center"],
            meta["scale"],
        )
        for _ in range(M)
    ]
    field = np.stack([p.field.numpy()[0] for p in planes]).astype(np.float32)
    o, carrot, th, sp, al, be, ga = _pose(bounds, meta["scale"], 2000, 4)
    map_id = rng.integers(0, M, 2000).astype(np.int32)
    bf = sdf_nav.BatchedSDFField.stack([planes[m] for m in map_id])
    ref = sdf_nav.coef_feats(bf, torch.from_numpy(o), torch.from_numpy(carrot)).numpy()
    got = nav_native.coef_feats(
        field, o, carrot, bounds=bounds, center=meta["center"], scale=meta["scale"], map_id=map_id
    )
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5), np.abs(got - ref).max()
