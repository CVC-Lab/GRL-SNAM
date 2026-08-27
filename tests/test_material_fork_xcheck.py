"""Bit-identity cross-check against a live checkout of the source repo.

Skipped unless GRL_SNAM_MATERIAL_FORK points at a checkout of
github.com/SetasAditya/material-aware-grl-snam.  When it does, this re-proves
that material_nav.py is a bit-for-bit port of the researcher's originals:
the barrier, the patch sampler, the material integrator (six outputs), the
model (state-dict compatible + bitwise forward), and the witness gate.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

FORK = os.environ.get("GRL_SNAM_MATERIAL_FORK", "")
pytestmark = pytest.mark.skipif(
    not (FORK and Path(FORK, "full_code", "train_material.py").is_file()),
    reason="GRL_SNAM_MATERIAL_FORK not set to a material-aware-grl-snam checkout",
)


@pytest.fixture(scope="module")
def fork_modules():
    # The fork's train_material imports its DFC dataset builder at module
    # scope; stub it (unused by the functions under test) so import succeeds
    # without scipy/matplotlib.
    stub_pkg = types.ModuleType("scripts")
    stub_mod = types.ModuleType("scripts.build_dfc2018_stagewise")
    for name in (
        "extract_local_geom_obstacles",
        "extract_risk_patch",
        "extract_rollout_patch",
        "dijkstra_geom",
        "_pick_local_goal_index",
    ):
        setattr(stub_mod, name, lambda *a, **k: None)
    stub_pkg.build_dfc2018_stagewise = stub_mod
    saved = {k: sys.modules.get(k) for k in ("scripts", "scripts.build_dfc2018_stagewise")}
    sys.modules["scripts"] = stub_pkg
    sys.modules["scripts.build_dfc2018_stagewise"] = stub_mod
    sys.path.insert(0, str(Path(FORK, "full_code")))
    try:
        import train_material as fork_tm  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location(
            "fork_exp1", Path(FORK, "rebuttal_experiments", "exp1_gate_ablation.py")
        )
        fork_exp1 = importlib.util.module_from_spec(spec)
        sys.modules["fork_exp1"] = fork_exp1
        spec.loader.exec_module(fork_exp1)
        yield fork_tm, fork_exp1
    finally:
        sys.path.remove(str(Path(FORK, "full_code")))
        sys.modules.pop("train_material", None)
        sys.modules.pop("fork_exp1", None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_barrier_sampler_integrator_bitwise(fork_modules):
    import material_nav as mnav

    fork_tm, _ = fork_modules
    tg = torch.Generator().manual_seed(7)

    x = torch.linspace(-5, 60, 999)
    assert all(
        torch.equal(a, b) for a, b in zip(fork_tm._sdf_barrier_grad(x), mnav.sdf_barrier_grad(x))
    )

    patch = torch.randn(5, 6, 33, 31, generator=tg)
    o0 = torch.randn(5, 2, generator=tg) * 50
    o = o0 + torch.randn(5, 2, generator=tg) * 40
    assert torch.equal(
        fork_tm.bilinear_sample_patch(patch, o, o0), mnav.bilinear_sample_patch(patch, o, o0)
    )

    B, N = 4, 3
    kw = dict(
        o0=(torch.rand(B, 2, generator=tg) * 60) + 10,
        v0=torch.randn(B, 2, generator=tg) * 0.5,
        goal=(torch.rand(B, 2, generator=tg) * 60) + 10,
        C=torch.rand(B, N, 2, generator=tg) * 80,
        R=torch.rand(B, N, generator=tg) * 3 + 0.5,
        mask=torch.rand(B, N, generator=tg) > 0.3,
        alphas=torch.rand(B, N, generator=tg) * 2,
        beta=torch.rand(B, generator=tg) * 2,
        gamma=torch.rand(B, generator=tg),
        lam_soft=torch.rand(B, generator=tg) * 5,
        lam_hard=torch.rand(B, generator=tg) * 10,
        rollout_patch=torch.randn(B, 6, 32, 32, generator=tg),
        d_hat=torch.rand(B, generator=tg) * 2 + 1,
        dt=torch.full((B,), 0.01),
        H=torch.randint(1, 9, (B,), generator=tg),
        robot_radius=torch.rand(B, generator=tg) * 2,
    )
    a = fork_tm.integrate_surrogate_material(
        **{k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    )
    b = mnav.integrate_surrogate_material(
        **{k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    )
    assert all(torch.equal(x0, x1) for x0, x1 in zip(a, b))


def test_model_state_dict_and_forward_bitwise(fork_modules):
    import material_nav as mnav

    fork_tm, _ = fork_modules
    m_fork = fork_tm.CoefEnergyNetMaterial()
    m_port = mnav.CoefEnergyNetMaterial()
    assert set(m_fork.state_dict()) == set(m_port.state_dict())
    m_port.load_state_dict(m_fork.state_dict())
    m_fork.eval()
    m_port.eval()
    tg = torch.Generator().manual_seed(11)
    with torch.no_grad():
        args = (
            torch.randn(3, 4, 6, generator=tg),
            torch.tensor([[1, 1, 0, 1]] * 3, dtype=torch.bool),
            torch.randn(3, 4, generator=tg),
            torch.rand(3, 2, 32, 32, generator=tg),
        )
        assert all(torch.equal(a, b) for a, b in zip(m_fork(*args), m_port(*args)))


def test_gate_bitwise_random_grids(fork_modules):
    import material_nav as mnav

    _, fork_exp1 = fork_modules
    rng = np.random.default_rng(3)
    for _ in range(200):
        h, w = int(rng.integers(20, 60)), int(rng.integers(20, 60))
        maps = {
            "risk_map": rng.random((h, w)).astype(np.float32),
            "hard_mask": (rng.random((h, w)) < 0.06).astype(np.uint8),
            "sdf_hard": (rng.random((h, w)) * 5).astype(np.float32),
        }
        pos = np.array([rng.uniform(0, w - 1), rng.uniform(0, h - 1)], dtype=np.float32)
        goal = np.array([rng.uniform(0, w - 1), rng.uniform(0, h - 1)], dtype=np.float32)
        kw = dict(
            primitive_count=16,
            horizon_cells=12,
            hard_margin_m=1.0,
            improvement_margin=0.05,
            material_trigger=0.45,
        )
        g0 = fork_exp1.primitive_feasibility_gate(maps, pos, goal, **kw)
        g1 = mnav.primitive_feasibility_gate(maps, pos, goal, **kw)
        assert (g0.active, g0.nominal_risk, g0.best_risk, g0.feasible_count) == (
            g1.active,
            g1.nominal_risk,
            g1.best_risk,
            g1.feasible_count,
        )
        assert g0.selected_direction_rc == g1.selected_direction_rc
        assert g0.selected_endpoint_rc == g1.selected_endpoint_rc
        same_clear = g0.selected_min_clearance_m == g1.selected_min_clearance_m or (
            np.isnan(g0.selected_min_clearance_m) and np.isnan(g1.selected_min_clearance_m)
        )
        assert same_clear
