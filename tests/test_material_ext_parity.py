"""Parity for the external force in the material rollouts (libcvc material_ext, #734).

libcvc's ``integrate_surrogate_material``, ``integrate_bicycle_material`` and
``integrate_dynamic_material`` with an external force, and their hand-written reverses, against the torch twins with ``ext=``
(:mod:`grl_snam.material_ext`), differentiated by autograd. The reference is libcvc's
``src/cvc/tests/data/material_ext_golden.json`` (copied to ``tests/fixtures``; written by
``NavMaterialExt.Golden`` with ``CVC_MATERIAL_EXT_GOLDEN_WRITE=1``): a small batch per integrator
with two force patches per agent, switched for half the agents at step 2, and each case's
outputs, coefficient gradients and dL/dF_ext for fixed output seeds.

Tier: FLOAT contract, as the other material parity tests: the golden is libcvc's float rollout,
the twin runs in float64, forward rtol 1e-4 and gradients rtol 1e-3. With a pycvc that takes the
channel, its bindings are held to the same golden (surrogate and bicycle; pycvc has no dynamic
binding), and the trainer to the null force.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam import nav_native  # noqa: E402
from grl_snam.dyn_bike import DynBikeParams  # noqa: E402
from grl_snam.material_bike import (  # noqa: E402
    BicycleParams,
    integrate_bicycle_material,
    integrate_dynamic_material,
)
from grl_snam.material_ext import ExtForce, MaterialExt  # noqa: E402
from grl_snam.material_nav import integrate_surrogate_material  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "material_ext_golden.json"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
COEFS = ("alphas", "beta", "gamma", "lam_soft", "lam_hard")


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-material-ext-golden-1"
    return data


def _cases():
    return _golden()["cases"]


def _rel(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))


def _run(c, dev, ext, coefs=None):
    """The torch twin on a golden case: (outputs dict, coefficient tensors)."""
    B, N, Hp, Wp = c["B"], c["N"], c["Hp"], c["Wp"]

    def T(k, *shape):
        return torch.tensor(np.asarray(c[k]), dtype=torch.float64, device=dev).view(*shape)

    co = coefs or {
        "alphas": T("alphas", B, N),
        **{k: T(k, B) for k in COEFS[1:]},
    }
    common = dict(
        goal=T("goal", B, 2),
        C=T("C", B, N, 2),
        R=T("R", B, N),
        mask=torch.tensor(c["mask"], device=dev).view(B, N).bool(),
        rollout_patch=T("rollout_patch", B, 6, Hp, Wp),
        d_hat=T("d_hat", B),
        dt=T("dt", B),
        H=torch.tensor(c["H"], device=dev),
        **co,
    )
    p = c["params"]
    if c["integrator"] == "surrogate":
        o, v, minc, cum, hard, arc = integrate_surrogate_material(
            T("o0", B, 2),
            T("v0", B, 2),
            **common,
            robot_radius=T("rr", B),
            margin_factor=p["margin_factor"],
            mass=p["mass"],
            d_hat_sdf=p["d_hat_sdf"],
            ext=ext,
        )
        return {
            "o": o,
            "th": v,
            "min_clear": minc,
            "cum_risk": cum,
            "hard_count": hard,
            "arc_length": arc,
        }, co
    vh = c["vehicle"]
    veh = BicycleParams(
        L=vh["L"],
        delta_max=vh["delta_max"],
        a_max=vh["a_max"],
        a_lat_max=vh["a_lat_max"],
        k_steer=vh["k_steer"],
        vmax=vh["vmax"],
        allow_reverse=bool(vh["allow_reverse"]),
    )
    kw = dict(
        rr=T("rr", B),
        vehicle=veh,
        margin_factor=p["margin_factor"],
        mass=p["mass"],
        d_hat_sdf=p["d_hat_sdf"],
        k_sharp=p["k_sharp"],
        ext=ext,
    )
    if c["integrator"] == "dynamic":
        o, th, sp, x, minc, cum, hard, arc = integrate_dynamic_material(
            T("o0", B, 2),
            T("th0", B),
            T("sp0", B),
            T("x0", B, 4),
            **common,
            dyn=DynBikeParams(**c["dynamic"]),
            **kw,
        )
        return {
            "o": o,
            "th": th,
            "sp": sp,
            "x": x,
            "min_clear": minc,
            "cum_risk": cum,
            "hard_count": hard,
            "arc_length": arc,
        }, co
    o, th, sp, minc, cum, hard, arc = integrate_bicycle_material(
        T("o0", B, 2), T("th0", B), T("sp0", B), **common, **kw
    )
    return {
        "o": o,
        "th": th,
        "sp": sp,
        "min_clear": minc,
        "cum_risk": cum,
        "hard_count": hard,
        "arc_length": arc,
    }, co


def _patch_ext(c, dev, zero=False):
    B, K, Hp, Wp, steps = c["B"], c["K"], c["Hp"], c["Wp"], c["steps"]
    patch = torch.tensor(np.asarray(c["ext_patch"]), dtype=torch.float64, device=dev)
    patch = patch.view(B, K, 2, Hp, Wp)
    pos = torch.tensor(c["patch_of_step"], dtype=torch.long, device=dev).view(B, steps)
    return MaterialExt(
        patch=torch.zeros_like(patch) if zero else patch, patch_of_step=pos, steer=bool(c["steer"])
    )


def _scalar(c, outs, dev):
    q = c["seeds"]

    def S(k):
        return torch.tensor(np.asarray(q[k]), dtype=torch.float64, device=dev)

    L = (outs["o"].reshape(-1) * S("o")).sum() + (outs["th"].reshape(-1) * S("th")).sum()
    if "sp" in outs:
        L = L + (outs["sp"] * S("sp")).sum()
    if "x" in outs:
        L = L + (outs["x"].reshape(-1) * S("x")).sum()
    for k in ("min_clear", "cum_risk", "arc_length"):
        L = L + (outs[k] * S(k)).sum()
    return L


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("case", range(len(_cases())))
def test_forward_and_gradients_match_libcvc(case, dev, monkeypatch):
    c = _cases()[case]
    B, N, steps = c["B"], c["N"], c["steps"]
    # dL/dF_ext per step: keep each step's force as a node of its own, so its gradient is
    # recorded even where the force is sampled at the fixed start (step 0).
    forces = []
    at = ExtForce.at

    def recording_at(self, s, o):
        F = at(self, s, o)
        node = F if F.requires_grad else F.detach().requires_grad_(True)
        if node.requires_grad and not node.is_leaf:
            node.retain_grad()
        forces.append((s, node))
        return node

    monkeypatch.setattr(ExtForce, "at", recording_at)
    coefs = {
        "alphas": torch.tensor(np.asarray(c["alphas"]), dtype=torch.float64, device=dev)
        .view(B, N)
        .requires_grad_(True),
        **{
            k: torch.tensor(np.asarray(c[k]), dtype=torch.float64, device=dev).requires_grad_(True)
            for k in COEFS[1:]
        },
    }
    outs, co = _run(c, dev, _patch_ext(c, dev), coefs)
    for k, v in outs.items():
        r = _rel(v.detach().cpu().numpy(), c["out"][k])
        assert r < 1e-4, f"{c['integrator']} {k}: {r:.3e}"
    _scalar(c, outs, dev).backward()
    for k in COEFS:
        r = _rel(co[k].grad.detach().cpu().numpy(), c["grad"][k])
        assert r < 1e-3, f"{c['integrator']} d/d{k}: {r:.3e}"
    gF = np.zeros((B, steps, 2))
    for s, F in forces:
        if F.grad is not None:
            gF[:, s] += F.grad.detach().cpu().numpy()
    r = _rel(gF, c["grad"]["fext"])
    assert r < 1e-3, f"{c['integrator']} dL/dF_ext: {r:.3e}"


@pytest.mark.parametrize("case", range(len(_cases())))
def test_a_zero_force_is_the_rollout_without_one(case):
    c = _cases()[case]
    none, _ = _run(c, "cpu", None)
    zero, _ = _run(c, "cpu", _patch_ext(c, "cpu", zero=True))
    for k in none:
        assert torch.equal(none[k], zero[k]), k


def test_the_force_steers_the_bicycle_only_with_steer():
    c = next(c for c in _cases() if c["integrator"] == "bicycle")
    on = _patch_ext(c, "cpu")
    off = MaterialExt(patch=on.patch, patch_of_step=on.patch_of_step, steer=False)
    a, _ = _run(c, "cpu", on)
    b, _ = _run(c, "cpu", off)
    assert not torch.equal(a["th"], b["th"])


def test_values_carry_their_J_into_the_gradient():
    # Per-step values with a J: the forward uses F as given, the gradient takes J (libcvc chains
    # J^T dL/dF into the position adjoint the same way).
    o = torch.tensor([[0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    F = torch.tensor([[[0.5, -0.1]]], dtype=torch.float64)
    J = torch.tensor([[[1.0, 2.0, -3.0, 0.5]]], dtype=torch.float64)
    f = ExtForce(MaterialExt(F=F, J=J), o.detach()).at(0, o)
    assert torch.equal(f.detach(), F[:, 0])
    w = torch.tensor([[0.7, -1.3]], dtype=torch.float64)
    (f * w).sum().backward()
    want = torch.tensor([[0.7 * 1.0 - 1.3 * -3.0, 0.7 * 2.0 - 1.3 * 0.5]], dtype=torch.float64)
    assert torch.allclose(o.grad, want)


def test_a_held_callable_keeps_its_sample():
    calls = []

    def sample(s, o):
        calls.append(s)
        return o * 0.0 + float(s)

    o = torch.zeros(2, 2, dtype=torch.float64)
    ef = ExtForce(MaterialExt(sample=sample, hold=3), o)
    got = [float(ef.at(s, o)[0, 0]) for s in range(7)]
    assert calls == [0, 3, 6]
    assert got == [0, 0, 0, 3, 3, 3, 6]


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"patch": torch.zeros(1, 1, 2, 3, 3), "F": torch.zeros(1, 2, 2)},
        {"F": None, "J": torch.zeros(1, 2, 4), "sample": None},
        {"patch_of_step": torch.zeros(1, 2), "F": torch.zeros(1, 2, 2)},
        {"sample": lambda s, o: o, "hold": 0},
        {"patch": torch.zeros(2, 3, 3)},
    ],
)
def test_malformed_channels_raise(kw):
    with pytest.raises(ValueError):
        MaterialExt(**kw)


@pytest.mark.skipif(not nav_native.HAS_MATERIAL_EXT, reason="pycvc lacks the material ext channel")
def test_pycvc_bindings_match_the_golden():
    from grl_snam.material_bike import BicycleParams as BP

    for c in _cases():
        if c["integrator"] == "dynamic":
            continue
        B, Hp, Wp, K, steps = c["B"], c["Hp"], c["Wp"], c["K"], c["steps"]
        a = {k: np.asarray(c[k], np.float32) for k in c if isinstance(c[k], list)}
        ext = MaterialExt(
            patch=torch.tensor(a["ext_patch"]).view(B, K, 2, Hp, Wp),
            patch_of_step=torch.tensor(c["patch_of_step"]).view(B, steps),
        )
        p = c["params"]
        common = (
            a["goal"],
            a["C"],
            a["R"],
            np.asarray(c["mask"], np.uint8),
            a["alphas"],
            a["beta"],
            a["gamma"],
            a["lam_soft"],
            a["lam_hard"],
            a["rollout_patch"].reshape(B, 6, Hp, Wp),
            a["rr"],
            a["d_hat"],
            a["dt"],
            np.asarray(c["H"], np.int32),
        )
        kw = dict(
            margin_factor=p["margin_factor"],
            mass=p["mass"],
            d_hat_sdf=p["d_hat_sdf"],
            k_sharp=p["k_sharp"],
            ext=ext,
        )
        if c["integrator"] == "surrogate":
            out = nav_native.integrate_surrogate_material(
                a["o0"].reshape(B, 2), a["v0"].reshape(B, 2), *common, **kw
            )
            names = ("o", "th", "min_clear", "cum_risk", "hard_count", "arc_length")
        else:
            v = c["vehicle"]
            veh = BP(
                L=v["L"],
                delta_max=v["delta_max"],
                a_max=v["a_max"],
                a_lat_max=v["a_lat_max"],
                k_steer=v["k_steer"],
                vmax=v["vmax"],
                allow_reverse=bool(v["allow_reverse"]),
            )
            out = nav_native.integrate_bicycle_material(
                a["o0"].reshape(B, 2), a["th0"], a["sp0"], *common, veh, **kw
            )
            names = ("o", "th", "sp", "min_clear", "cum_risk", "hard_count", "arc_length")
        for k, got in zip(names, out):
            assert _rel(got, c["out"][k]) < 1e-5, k


@pytest.mark.skipif(not nav_native.HAS_MATERIAL_EXT, reason="pycvc lacks the material ext channel")
def test_the_trainer_takes_the_force(tmp_path):
    from grl_snam import material_train_native as mtn

    if not mtn.HAS_MATERIAL_TRAINER_LOSS:
        pytest.skip("pycvc lacks the forward-only trainer loss")
    from tests.test_material_train_native import _synthetic_batch, _write_random_cvcnm  # noqa: E402

    init = tmp_path / "init.cvcnm"
    _write_random_cvcnm(init, patch_size=16, seed=9)
    tr = mtn.MaterialTrainer(init)
    batch = _synthetic_batch(seed=17)
    B, _, Hp, Wp = batch["rollout_patch"].shape
    base = tr.loss(batch)
    zero = MaterialExt(patch=torch.zeros(B, 1, 2, Hp, Wp))
    assert tr.loss(batch, ext=zero) == base
    yy, xx = np.meshgrid(np.linspace(0, 1, Hp), np.linspace(0, 1, Wp), indexing="ij")
    one = np.stack([0.3 * np.sin(5 * xx + 2 * yy), 0.3 * np.cos(3 * xx - 4 * yy)])
    force = MaterialExt(patch=torch.tensor(np.broadcast_to(one, (B, 1, 2, Hp, Wp)).copy()))
    assert tr.loss(batch, ext=force) != base
