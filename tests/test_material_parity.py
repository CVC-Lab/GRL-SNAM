"""Python-vs-C++ parity for the material-aware kernels (cvc::nav material).

Tier map (mirroring the geometry kernels' contract):
  nav_material_build, nav_witness_gate(+batch)  — BIT (array_equal / tobytes)
  nav_material_sample                           — FLOAT contract + a
      non-contractual bit-exactness tripwire (the sdf_sample precedent)
  nav_bicycle_rollout_material vs torch         — FLOAT (rtol 1e-4 / atol 1e-5,
      the drive_step tier), with FIXED lambda columns (never live gate output —
      discrete divergence stays out of the FLOAT surface)

The native path is OPT-IN (GRL_SNAM_MATERIAL_BACKEND=native); the dispatch
test proves both the opt-in and the default-off.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pycvc = pytest.importorskip("pycvc")

from grl_snam import nav_native  # noqa: E402
from grl_snam.material import (  # noqa: E402
    MaterialGrid,
    MaterialRuntime,
    witness_gate,
    witness_gate_batch,
)

if not nav_native.HAS_MATERIAL:
    pytest.skip("this pycvc build has no material kernels", allow_module_level=True)

import sdf_nav  # noqa: E402

BOUNDS = (-100.0, -100.0, 100.0, 100.0)
CENTER = (0.0, 0.0)
SCALE = 0.05

GATE_KW = dict(
    horizon_cells=12,
    hard_margin_m=1.0,
    primitive_count=16,
    improvement_margin=0.05,
    material_trigger=0.45,
    progress_slack_cells=0.5,
)


def _random_grid(rng, h, w):
    risk = rng.random((h, w)).astype(np.float32)
    hard = (rng.random((h, w)) < 0.06).astype(np.uint8)
    clear = (rng.random((h, w)) * 4).astype(np.float32)
    return risk, hard, clear


# ---------------------------------------------------------------------------
# material_build — BIT
# ---------------------------------------------------------------------------


def test_material_build_bit_identical():
    rng = np.random.default_rng(11)
    for h, w in ((8, 8), (24, 31), (64, 64), (17, 40)):
        risk_raw = rng.random((h, w)).astype(np.float32)
        hard = rng.random((h, w)) < 0.08
        g = MaterialGrid(risk_raw, hard, BOUNDS, CENTER, SCALE)
        planes = nav_native.material_build(risk_raw, hard.astype(np.uint8), g.cell_w, SCALE, 1.0)
        names = ("risk", "phi_hard_m", "grad_rx", "grad_ry", "grad_px", "grad_py")
        for name, native in zip(names, planes):
            ref = getattr(g, name)
            assert ref.tobytes() == np.asarray(native).tobytes(), name


def test_material_build_sigma_zero_and_error_paths():
    rng = np.random.default_rng(12)
    risk_raw = rng.random((16, 16)).astype(np.float32)
    hard = np.zeros((16, 16), np.uint8)
    g = MaterialGrid(risk_raw, hard, BOUNDS, CENTER, SCALE, sigma=0.0)
    planes = nav_native.material_build(risk_raw, hard, g.cell_w, SCALE, 0.0)
    assert g.risk.tobytes() == np.asarray(planes[0]).tobytes()
    with pytest.raises(Exception):
        pycvc.nav_material_build(
            np.zeros((3, 30), np.float32), np.zeros((3, 30), np.uint8), 1.0, 1.0, 1.0
        )


# ---------------------------------------------------------------------------
# witness gate — BIT
# ---------------------------------------------------------------------------


def test_witness_gate_bit_identical_randomized():
    rng = np.random.default_rng(21)
    for _ in range(60):
        h, w = int(rng.integers(20, 70)), int(rng.integers(20, 70))
        risk, hard, clear = _random_grid(rng, h, w)
        pos = (float(rng.uniform(0, h - 1)), float(rng.uniform(0, w - 1)))
        goal = (float(rng.uniform(0, h - 1)), float(rng.uniform(0, w - 1)))
        ref = witness_gate(risk, hard.astype(bool), clear, pos, goal, **GATE_KW)
        nat = nav_native.witness_gate(risk, hard, clear, pos, goal, **GATE_KW)
        assert nat.active == ref.active
        assert nat.nominal_risk == ref.nominal_risk
        assert nat.best_risk == ref.best_risk
        assert nat.feasible_count == ref.feasible_count
        assert nat.direction_rc == ref.direction_rc
        assert nat.endpoint_rc == ref.endpoint_rc
        same_clear = nat.min_clearance_m == ref.min_clearance_m or (
            np.isnan(nat.min_clearance_m) and np.isnan(ref.min_clearance_m)
        )
        assert same_clear


def test_witness_gate_half_even_adversarials():
    # positions on exact .5 boundaries with the exactly-representable axis
    # direction (table index 0 = (0.0, 1.0)); half-even rounding must agree.
    rng = np.random.default_rng(22)
    h = w = 30
    risk = rng.random((h, w)).astype(np.float32)
    hard = np.zeros((h, w), np.uint8)
    clear = np.full((h, w), 9.0, np.float32)
    for _ in range(50):
        pos = (float(rng.integers(2, 27)) + 0.5, float(rng.integers(2, 27)) + 0.5)
        goal = (pos[0], min(float(w - 2), pos[1] + 15.0))
        ref = witness_gate(risk, hard.astype(bool), clear, pos, goal, **GATE_KW)
        nat = nav_native.witness_gate(risk, hard, clear, pos, goal, **GATE_KW)
        assert (nat.active, nat.nominal_risk, nat.best_risk, nat.feasible_count) == (
            ref.active,
            ref.nominal_risk,
            ref.best_risk,
            ref.feasible_count,
        )


def test_witness_gate_batch_bit_identical_and_thread_stable():
    rng = np.random.default_rng(23)
    h, w = 48, 52
    risk, hard, clear = _random_grid(rng, h, w)
    n = 33
    pos = np.stack([rng.uniform(0, h - 1, n), rng.uniform(0, w - 1, n)], axis=1)
    goal = np.stack([rng.uniform(0, h - 1, n), rng.uniform(0, w - 1, n)], axis=1)
    ref = witness_gate_batch(risk, hard.astype(bool), clear, pos, goal, **GATE_KW)
    for threads in (1, 8):
        nat = nav_native.witness_gate_batch(
            risk, hard, clear, pos, goal, num_threads=threads, **GATE_KW
        )
        assert np.array_equal(nat[0], ref[0])
        assert nat[1].tobytes() == ref[1].tobytes()
        assert nat[2].tobytes() == ref[2].tobytes()
        assert np.array_equal(nat[3], ref[3])


# ---------------------------------------------------------------------------
# material_sample — FLOAT + tripwire
# ---------------------------------------------------------------------------


def _grid_with_features(rng, n=64):
    risk = rng.random((n, n)).astype(np.float32)
    hard = rng.random((n, n)) < 0.05
    return MaterialGrid(risk, hard, BOUNDS, CENTER, SCALE)


def test_material_sample_float_contract():
    rng = np.random.default_rng(31)
    g = _grid_with_features(rng)
    f = g.field()
    on = (rng.random((257, 2)).astype(np.float32) * 12.0) - 6.0  # incl. OOB (border clamp)
    r_ref, p_ref, gr_ref, gp_ref = f.sample(torch.from_numpy(on))
    r_nat, p_nat, gr_nat, gp_nat = nav_native.material_sample(f, on)
    assert np.allclose(r_nat, r_ref.numpy(), rtol=1e-5, atol=1e-6)
    assert np.allclose(p_nat, p_ref.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(gr_nat, gr_ref.numpy(), rtol=1e-4, atol=1e-6)
    assert np.allclose(gp_nat, gp_ref.numpy(), rtol=1e-4, atol=1e-6)


def test_material_sample_bit_exact_tripwire():
    """NON-CONTRACTUAL: on this platform the C++ sampler has matched torch
    bit-for-bit (the sdf_sample precedent). If this trips without a torch/
    compiler change, investigate before relaxing."""
    rng = np.random.default_rng(32)
    g = _grid_with_features(rng)
    f = g.field()
    on = (rng.random((64, 2)).astype(np.float32) * 8.0) - 4.0
    r_ref, p_ref, gr_ref, gp_ref = f.sample(torch.from_numpy(on))
    r_nat, p_nat, gr_nat, gp_nat = nav_native.material_sample(f, on)
    assert np.array_equal(r_nat, r_ref.numpy())
    assert np.array_equal(p_nat, p_ref.numpy())
    assert np.array_equal(gr_nat, gr_ref.numpy())
    assert np.array_equal(gp_nat, gp_ref.numpy())


# ---------------------------------------------------------------------------
# rollout with material — FLOAT (fixed lambda columns)
# ---------------------------------------------------------------------------


def test_bicycle_rollout_material_float_parity_vs_torch():
    rng = np.random.default_rng(41)
    n_grid = 64
    occ = np.zeros((n_grid, n_grid), bool)
    occ[20:30, 30:34] = True
    phi, nx_g, ny_g = sdf_nav.build_sdf(occ, BOUNDS, SCALE)
    field = sdf_nav.SDFField(phi, nx_g, ny_g, BOUNDS, CENTER, SCALE)
    g = _grid_with_features(rng, n_grid)
    mf = g.field()

    N = 96
    o = ((rng.random((N, 2)) * 8.0) - 4.0).astype(np.float32)
    th = (rng.random(N).astype(np.float32) * 6.0) - 3.0
    sp = rng.random(N).astype(np.float32) * 0.8
    goal = ((rng.random((N, 2)) * 8.0) - 4.0).astype(np.float32)
    al = rng.random(N).astype(np.float32) * 2
    be = rng.random(N).astype(np.float32) * 4
    ga = rng.random(N).astype(np.float32) * 4
    lam_s = rng.random(N).astype(np.float32) * 0.8  # FIXED columns, no live gate
    lam_h = rng.random(N).astype(np.float32) * 1.5
    kw = dict(rr=0.15, d_hat=0.35, dt=0.06, vmax=0.9)

    for nsub in (1, 2):
        to, tth, tsp, _ = sdf_nav.bicycle_rollout(
            field,
            torch.from_numpy(o.copy()),
            torch.from_numpy(th.copy()),
            torch.from_numpy(sp.copy()),
            torch.from_numpy(goal),
            torch.from_numpy(al),
            torch.from_numpy(be),
            torch.from_numpy(ga),
            1,
            nsub=nsub,
            allow_reverse=True,
            material=mf,
            lam_soft=torch.from_numpy(lam_s),
            lam_hard=torch.from_numpy(lam_h),
            mat_k_sharp=1.25,
            mat_d_hat_m=12.0,
            **kw,
        )
        no, nth, nsp, _ = pycvc.nav_bicycle_rollout_material(
            field.field.numpy(),
            o.copy(),
            th.copy(),
            sp.copy(),
            goal,
            al,
            be,
            ga,
            mf.field.numpy(),
            lam_s,
            lam_h,
            1.25,
            12.0,
            None,
            *BOUNDS,
            *CENTER,
            SCALE,
            kw["rr"],
            kw["d_hat"],
            kw["dt"],
            kw["vmax"],
            0.035,
            0.6,
            1.5,
            1.0,
            0.8,
            nsub,
            1,
            0,
        )
        assert np.allclose(no, to.numpy(), rtol=1e-4, atol=1e-5), f"o nsub={nsub}"
        assert np.allclose(nth, tth.numpy(), rtol=1e-4, atol=1e-5), f"th nsub={nsub}"
        assert np.allclose(nsp, tsp.numpy(), rtol=1e-4, atol=1e-5), f"sp nsub={nsub}"


# ---------------------------------------------------------------------------
# dispatch / opt-in
# ---------------------------------------------------------------------------


def test_material_backend_is_opt_in(monkeypatch):
    monkeypatch.delenv("GRL_SNAM_MATERIAL_BACKEND", raising=False)
    assert not nav_native.material_enabled()  # default: pure Python
    monkeypatch.setenv("GRL_SNAM_MATERIAL_BACKEND", "native")
    assert nav_native.material_enabled()
    monkeypatch.setenv("GRL_SNAM_MATERIAL_BACKEND", "python")
    assert not nav_native.material_enabled()


def test_runtime_gate_dispatch_transparent(monkeypatch):
    """MaterialRuntime.gate must return identical decisions on both backends."""
    rng = np.random.default_rng(51)
    risk = rng.random((64, 64)).astype(np.float32)
    hard = rng.random((64, 64)) < 0.04

    def run():
        g = MaterialGrid(risk, hard, BOUNDS, CENTER, SCALE)
        rt = MaterialRuntime(g)
        rt.update_occ(np.zeros((64, 64), bool))
        out = []
        for _ in range(40):
            pos = (float(rng2.uniform(-90, 90)), float(rng2.uniform(-90, 90)))
            goal = (float(rng2.uniform(-90, 90)), float(rng2.uniform(-90, 90)))
            r = rt.gate(pos, goal)
            out.append((r.active, r.nominal_risk, r.best_risk, r.feasible_count))
        return out

    rng2 = np.random.default_rng(52)
    monkeypatch.setenv("GRL_SNAM_MATERIAL_BACKEND", "python")
    a = run()
    rng2 = np.random.default_rng(52)
    monkeypatch.setenv("GRL_SNAM_MATERIAL_BACKEND", "native")
    b = run()
    assert a == b


# ---------------------------------------------------------------------------
# fused native material drive (nav_drive_step_material) — FLOAT
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not nav_native.HAS_MATERIAL_DRIVE, reason="pycvc lacks nav_drive_step_material")
def test_drive_step_material_float_parity_vs_torch(tmp_path):
    """The fused torch-free material drive (sample -> coef_feats -> coef_mlp ->
    bicycle_material) matches the torch material drive at the drive tier."""
    from grl_snam.tools import coef_export

    rng = np.random.default_rng(71)
    n_grid = 64
    occ = np.zeros((n_grid, n_grid), bool)
    occ[18:30, 30:34] = True
    phi, nx_g, ny_g = sdf_nav.build_sdf(occ, BOUNDS, SCALE)
    field = sdf_nav.SDFField(phi, nx_g, ny_g, BOUNDS, CENTER, SCALE)
    g = _grid_with_features(rng, n_grid)
    mf = g.field()
    model = sdf_nav.CoefMLP()
    model.eval()
    wp = tmp_path / "coef.cvcnav"
    coef_export.write_coef_mlp(model, str(wp))

    N = 80
    o = ((rng.random((N, 2)) * 8.0) - 4.0).astype(np.float32)
    th = (rng.random(N).astype(np.float32) * 6.0) - 3.0
    sp = rng.random(N).astype(np.float32) * 0.8
    carrot = ((rng.random((N, 2)) * 8.0) - 4.0).astype(np.float32)
    lam_s = rng.random(N).astype(np.float32) * 0.6
    lam_h = rng.random(N).astype(np.float32) * 1.2
    kw = dict(rr=0.15, d_hat=0.35, dt=0.06, vmax=0.9)
    veh = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8)
    params = {**kw, **veh, "nsub": 1, "allow_reverse": True}

    # torch reference: coef_feats -> CoefMLP -> bicycle_rollout(material=...)
    with torch.no_grad():
        feat = sdf_nav.coef_feats(field, torch.from_numpy(o), torch.from_numpy(carrot))
        al, be, ga = model(feat)
        to, tth, tsp, _ = sdf_nav.bicycle_rollout(
            field,
            torch.from_numpy(o.copy()),
            torch.from_numpy(th.copy()),
            torch.from_numpy(sp.copy()),
            torch.from_numpy(carrot),
            al,
            be,
            ga,
            1,
            nsub=1,
            allow_reverse=True,
            material=mf,
            lam_soft=torch.from_numpy(lam_s),
            lam_hard=torch.from_numpy(lam_h),
            mat_k_sharp=1.25,
            mat_d_hat_m=12.0,
            **kw,
        )
    no, nth, nsp, _ = nav_native.drive_step_material(
        field.field.numpy(),
        o,
        th,
        sp,
        carrot,
        str(wp),
        mf.field.numpy(),
        lam_s,
        lam_h,
        mat_k_sharp=1.25,
        mat_d_hat_m=12.0,
        bounds=BOUNDS,
        center=CENTER,
        scale=SCALE,
        params=params,
    )
    assert np.allclose(no, to.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(nth, tth.numpy(), rtol=1e-4, atol=1e-5)
    assert np.allclose(nsp, tsp.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not nav_native.HAS_MATERIAL_DRIVE, reason="pycvc lacks nav_drive_step_material")
def test_swarm_native_material_drive_tracks_torch(monkeypatch):
    """GRL_SNAM_NAV_DRIVE=native + material now uses the fused C++ material
    drive (no fallback warning) and tracks the torch material Swarm to the
    drive tier over a short run."""
    import warnings

    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.material import MaterialGrid
    from grl_snam.squad import AgentSpec
    from grl_snam.swarm import Swarm

    story = shrunk(STORIES["city"], n=64, max_steps=10_000_000)
    specs = [
        AgentSpec(f"a{i}", (-45.0, -20.0 + 10.0 * i), (45.0, -20.0 + 10.0 * i)) for i in range(4)
    ]
    rr, cc = np.mgrid[0:64, 0:64]
    risk = np.where((rr - 30) ** 2 + (cc - 32) ** 2 <= 49, 0.9, 0.0).astype(np.float32)

    def make_model():
        torch.manual_seed(0)
        m = sdf_nav.CoefMLP()
        m.eval()
        return m

    def run(nav_drive):
        monkeypatch.setenv("GRL_SNAM_NAV_DRIVE", nav_drive)
        grid = MaterialGrid(risk, np.zeros((64, 64), bool), story.bounds, (0, 0), story.scale)
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no fallback warning may fire
            s = Swarm(
                story,
                specs,
                make_model(),
                seed=0,
                truth_occ=np.zeros((64, 64), bool),
                material=grid,
            )
        pos = []
        for _ in range(60):
            s.step()
            pos.append(s.n2w(s.o).cpu().numpy().copy())
        return np.stack(pos), s

    torch_pos, _ = run("torch")
    native_pos, s_native = run("native")
    assert s_native._native_drive and s_native._native_drive_material
    # drive-tier tracking (same float-equivalence budget as test_drive_step_parity)
    assert np.abs(native_pos - torch_pos).max() < 0.05
