"""The torch material trainer (material_train_torch), held to libcvc's native one.

Three layers:

1. **The native golden** (the gate). libcvc's ``NavMaterialTrain.TorchTrainerGolden`` (libcvc #734,
   commit 6001a840, with ``CVC_MATERIAL_TRAIN_GOLDEN_DIR`` set) runs the native trainer for 3 Adam
   steps on its fixture batch (B 16, N 3, P 16, 13x13 rollout patch) with a scheduled two-patch
   external force, and writes ``material_train_golden.json`` (config, batch, force, the parameter
   index, each step's loss, CVaR eta and unweighted terms), ``_w0.cvcnm`` / ``_w3.cvcnm`` (the
   weights before and after) and ``_grad0.f32`` (step 0's weight gradients). The blobs are ~3 MB
   and are not committed here: point ``GRL_SNAM_MATERIAL_TRAIN_GOLDEN`` at that directory, or
   these tests skip.
2. **A live native trainer**, when the installed pycvc has the ``nav_material_trainer_*`` binding:
   the same weights and batch through both trainers, for the configurations the golden does not
   cover (the bicycle, with and without the external force, and sampled and semi-implicit
   multi-start). Against libcvc #734's pycvc: loss <= 1e-7 relative over 3 steps, L_multi alone
   <= 1e-8, weights <= 1.9e-7 at the 99.99th percentile.
3. **Self-consistency**, always: a float64 finite-difference gradcheck of the full loss -> weights
   chain, the pass-through's semantics (``loss`` does not train, ``step`` returns the pre-update
   loss, step k draws with ``seed + k``, ``ext=None`` is a zero force), determinism, the
   ``.cvcnm`` round trip, and CUDA == CPU.

Tolerances. The native runs its model and rollouts in float32 and accumulates the loss in double.
So the golden holds at the float tier, whatever the torch dtype: each loss term to 1e-5 relative
(measured <= 5e-7, CPU and CUDA, float32 and float64), each weight tensor's gradient to 1e-4 of its
largest entry (measured <= 1e-6), and the weights after 3 Adam steps (lr 1e-3, so up to 3e-3 of
travel) to 1e-6 absolute on 99.99% of the 207k entries (measured <= 2e-7) and 2.5e-5 on every
entry (measured 3.5e-6 against the golden, 1.2e-5 against a live trainer). The few entries above
1e-6 are those whose gradient is ~1e-7 of their tensor's: Adam divides by the gradient's own
size, so a float32 rounding of it there moves the step by a percent of lr. One exception, by
construction: the key
projection's bias (``in_proj_bias[64:128]``) has an identically zero gradient (softmax is
invariant to a shift shared by every key), so both trainers' gradients there are roundoff
(~1e-8) and Adam, which normalizes the gradient, turns that roundoff into steps of up to ``lr``.
Those 128 entries are checked for that, not for agreement.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam import material_train_native as mtn  # noqa: E402
from grl_snam import material_train_torch as mtt  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.material_bike import BicycleParams  # noqa: E402
from grl_snam.material_ext import MaterialExt  # noqa: E402
from grl_snam.material_nav import CoefEnergyNetMaterial  # noqa: E402
from grl_snam.tools.matnet_export import read_matnet  # noqa: E402

GOLDEN_ENV = "GRL_SNAM_MATERIAL_TRAIN_GOLDEN"
GOLDEN_FILES = (
    "material_train_golden.json",
    "material_train_golden_w0.cvcnm",
    "material_train_golden_w3.cvcnm",
    "material_train_golden_grad0.f32",
)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.float64]
PARTS = ("traj", "vel", "fric", "clear", "nav", "lreg", "multi")
WEIGHT_MAX = 2.5e-5  # weights after 3 steps, every entry (see the module docstring)
VEHICLE = BicycleParams(L=0.3, a_max=2.0, a_lat_max=1.5, vmax=1.5)  # sized for the batch below


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def _key_bias(name: str, n: int) -> np.ndarray:
    """True on the key projection's bias: the entries whose gradient is identically zero."""
    m = np.zeros(n, bool)
    if name.endswith("self_attn.in_proj_bias"):
        d = n // 3
        m[d : 2 * d] = True
    return m


def _weights(tr) -> dict:
    return {k: v.detach().cpu().double().numpy() for k, v in tr.model.state_dict().items()}


def _weight_gap(got: dict, want: dict):
    """(max |got - want| off the key bias, its 99.99th percentile, the key-bias pairs)."""
    off, keyb = [], []
    for k, w in want.items():
        g = np.asarray(got[k], np.float64).ravel()
        w = np.asarray(w, np.float64).ravel()
        kb = _key_bias(k, w.size)
        off.append(np.abs(g - w)[~kb])
        if kb.any():
            keyb.append((k, g[kb], w[kb]))
    off = np.concatenate(off)
    return float(off.max()), float(np.quantile(off, 0.9999)), keyb


# --------------------------------------------------------------------------- fixtures


def _write_model(path, seed=0, patch_size=16):
    torch.manual_seed(seed)
    m = CoefEnergyNetMaterial(patch_size=patch_size)
    m.patch_size = patch_size
    mtt.save_cvcnm(m, path)
    return path


def _batch(B=8, N=3, P=16, Hp=13, Wp=13, seed=7):
    """A padded batch on the native tests' recipe: discs around the start, the goal beyond them."""
    rng = np.random.default_rng(seed)
    o0 = rng.uniform(-0.5, 0.5, (B, 2)).astype(np.float32)
    goal = rng.uniform(1.5, 2.5, (B, 2)).astype(np.float32)
    ang = rng.uniform(0, 6.28, (B, N))
    dist = rng.uniform(1.0, 1.4, (B, N))
    C = np.stack([o0[:, 0:1] + dist * np.cos(ang), o0[:, 1:2] + dist * np.sin(ang)], -1)
    C = C.astype(np.float32)
    R = rng.uniform(0.4, 0.6, (B, N)).astype(np.float32)
    mask = np.ones((B, N), np.uint8)
    mask[:, N - 1] = 0
    obs_feats = np.concatenate(
        [C, R[..., None], rng.uniform(0.5, 1.5, (B, N, 1)), goal[:, None, :] - C], -1
    ).astype(np.float32)
    goal_feats = np.concatenate(
        [goal - o0, np.linalg.norm(goal - o0, axis=1, keepdims=True), np.ones((B, 1))], 1
    ).astype(np.float32)
    risk_patch = np.stack(
        [rng.uniform(0.1, 0.9, (B, P, P)), (rng.uniform(0, 1, (B, P, P)) > 0.7).astype(np.float32)],
        1,
    ).astype(np.float32)
    rollout_patch = rng.uniform(-0.3, 0.3, (B, 6, Hp, Wp)).astype(np.float32)
    rollout_patch[:, 0] = rng.uniform(0.1, 0.9, (B, Hp, Wp))
    rollout_patch[:, 1] = rng.uniform(1.0, 4.0, (B, Hp, Wp))
    return {
        "obs_feats": obs_feats,
        "obs_mask": mask,
        "goal_feats": goal_feats,
        "risk_patch": risk_patch,
        "o0": o0,
        "v0": rng.uniform(-0.1, 0.1, (B, 2)).astype(np.float32),
        "goal": goal,
        "C": C,
        "R": R,
        "rollout_patch": rollout_patch,
        "rr": np.full(B, 0.5, np.float32),
        "d_hat": np.full(B, 3.0, np.float32),
        "dt": np.full(B, 0.1, np.float32),
        "H": np.where(np.arange(B) % 2 == 0, 2, 3).astype(np.int32),
        "o_tgt": (goal + rng.uniform(-0.3, 0.3, (B, 2))).astype(np.float32),
        "v_tgt": rng.uniform(-0.2, 0.2, (B, 2)).astype(np.float32),
        "gamma_o": rng.uniform(3.0, 5.0, B).astype(np.float32),
    }


def _force(batch, K=2):
    """libcvc's golden force: two smooth patches per agent, the odd agents switching at step 1."""
    B, _, Hp, Wp = batch["rollout_patch"].shape
    steps = int(batch["H"].max())
    x = np.arange(Wp, dtype=np.float32) / (Wp - 1)
    y = np.arange(Hp, dtype=np.float32) / (Hp - 1)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    patch = np.zeros((B, K, 2, Hp, Wp), np.float32)
    for i in range(B):
        for k in range(K):
            ph = 0.7 * k + 0.3 * i
            patch[i, k, 0] = 0.3 * np.sin(5 * xx + 2 * yy + ph)
            patch[i, k, 1] = 0.3 * np.cos(3 * xx - 4 * yy + ph)
    sched = np.zeros((B, steps), np.int64)
    sched[1::2, 1:] = 1
    return MaterialExt(patch=torch.from_numpy(patch), patch_of_step=torch.from_numpy(sched))


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return _write_model(tmp_path_factory.mktemp("mtt") / "init.cvcnm", seed=3)


# --------------------------------------------------------------------------- 1. the golden


def _golden_dir() -> Path:
    env = os.environ.get(GOLDEN_ENV)
    if not env:
        pytest.skip(
            f"{GOLDEN_ENV} is unset: point it at the output directory of libcvc's "
            "NavMaterialTrain.TorchTrainerGolden (libcvc #734; run it with "
            "CVC_MATERIAL_TRAIN_GOLDEN_DIR set). The ~3 MB golden is not committed."
        )
    d = Path(env)
    missing = [f for f in GOLDEN_FILES if not (d / f).is_file()]
    if missing:
        pytest.skip(f"{GOLDEN_ENV}={env} lacks {', '.join(missing)}")
    return d


def _golden():
    d = _golden_dir()
    g = json.loads((d / "material_train_golden.json").read_text(encoding="utf-8"))
    assert g["format"] == "cvc-material-train-golden-1"
    b = g["batch"]
    B, N, P, Hp, Wp = b["B"], b["N"], b["P"], b["Hp"], b["Wp"]
    shapes = {
        "obs_feats": (B, N, 6),
        "obs_mask": (B, N),
        "goal_feats": (B, 4),
        "risk_patch": (B, 2, P, P),
        "o0": (B, 2),
        "v0": (B, 2),
        "goal": (B, 2),
        "C": (B, N, 2),
        "R": (B, N),
        "rollout_patch": (B, 6, Hp, Wp),
        "rr": (B,),
        "d_hat": (B,),
        "dt": (B,),
        "H": (B,),
        "o_tgt": (B, 2),
        "v_tgt": (B, 2),
        "gamma_o": (B,),
    }
    batch = {k: np.asarray(b[k]).reshape(s) for k, s in shapes.items()}
    e = g["ext"]
    ext = MaterialExt(
        patch=torch.tensor(np.asarray(e["patch"], np.float32).reshape(B, e["K"], 2, Hp, Wp)),
        patch_of_step=torch.tensor(np.asarray(e["patch_of_step"]).reshape(B, e["steps"])),
        steer=bool(e["steer"]),
    )
    c = g["config"]
    assert c["integrator"] == "surrogate"
    kw = {
        k: c[k]
        for k in (
            "w_traj w_vel w_fric w_clear w_lreg w_goal w_len w_risk w_hard cvar_alpha w_multi "
            "lam_soft_max lam_hard_max margin_factor mass d_hat_sdf k_sharp tau ms_h ms_dt_mult "
            "ms_count grad_clip"
        ).split()
    }
    kw["frac_range"] = (c["frac_lo"], c["frac_hi"])
    kw["semi_implicit"] = bool(c["semi_implicit"])
    grad0 = np.fromfile(d / "material_train_golden_grad0.f32", dtype="<f4")
    grads = {
        p["name"]: grad0[p["offset"] : p["offset"] + p["count"]].astype(np.float64)
        for p in g["params"]
    }
    return dict(
        dir=d,
        kw=kw,
        lr=c["lr"],
        batch=batch,
        ext=ext,
        grads=grads,
        steps=g["steps"],
        w0=read_matnet(str(d / "material_train_golden_w0.cvcnm")).t,
        w3=read_matnet(str(d / "material_train_golden_w3.cvcnm")).t,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dev", DEVICES)
def test_golden_step0_loss_terms_and_gradients(dev, dtype):
    G = _golden()
    tr = mtt.MaterialTrainerTorch(
        G["dir"] / "material_train_golden_w0.cvcnm", device=dev, dtype=dtype, **G["kw"]
    )
    s0 = G["steps"][0]
    # With the CVaR quantile frozen at the native's: the loss and every term.
    parts, grads = tr.loss_and_grad(G["batch"], ext=G["ext"], frozen_eta=s0["eta"])
    for k in ("L", *PARTS):
        assert _rel(parts[k], s0[k]) < 1e-5, f"{k}: torch {parts[k]!r} native {s0[k]!r}"
    # The quantile the torch trainer computes itself.
    fresh = tr.loss_parts(G["batch"], ext=G["ext"])
    assert _rel(fresh["eta"], s0["eta"]) < 1e-6
    assert _rel(fresh["L"], s0["L"]) < 1e-5
    # Each weight tensor's gradient, against the native's step-0 gradient.
    assert set(grads) == set(G["grads"])
    for name, want in G["grads"].items():
        got = grads[name].detach().cpu().double().numpy().ravel()
        scale = float(np.abs(want).max())
        if scale == 0.0:  # mu_lat: unused by the loss, no gradient on either side
            assert not got.any(), name
            continue
        gap = float(np.abs(got - want).max()) / scale
        assert gap < 1e-4, f"d/d{name}: {gap:.3e} of its largest entry"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dev", DEVICES)
def test_golden_three_adam_steps(dev, dtype):
    G = _golden()
    tr = mtt.MaterialTrainerTorch(
        G["dir"] / "material_train_golden_w0.cvcnm", device=dev, dtype=dtype, **G["kw"]
    )
    for t in range(3):
        L = tr.step(G["batch"], lr=G["lr"], ext=G["ext"])
        want = G["steps"][t]
        assert _rel(L, want["L"]) < 1e-5, f"step {t}: torch {L!r} native {want['L']!r}"
        assert _rel(tr.last_eta, want["eta"]) < 1e-6, f"step {t} eta"
    assert tr.steps == 3
    after = tr.loss_parts(G["batch"], ext=G["ext"])
    for k in ("L", "eta", *PARTS):
        assert _rel(after[k], G["steps"][3][k]) < 1e-5, f"after 3 steps, {k}"
    worst, p9999, keyb = _weight_gap(_weights(tr), G["w3"])
    assert worst < WEIGHT_MAX, f"weights after 3 steps: max |torch - native| = {worst:.3e}"
    assert p9999 < 1e-6, f"weights after 3 steps: 99.99th percentile {p9999:.3e}"
    # The key bias: a zero gradient in exact arithmetic, roundoff in both trainers, so each
    # moved at most 3 lr on noise.
    lr = G["lr"]
    for name, got, want in keyb:
        kb = _key_bias(name, G["grads"][name].size)
        w0 = np.asarray(G["w0"][name], np.float64).ravel()[kb]
        assert np.abs(G["grads"][name][kb]).max() < 1e-6, name
        assert np.abs(want - w0).max() <= 3 * lr * 1.001, name
        assert np.abs(got - w0).max() <= 3 * lr * 1.001, name


def test_golden_terms_add_up_to_the_loss():
    """The double-precision assembly: both sides' terms sum to their L (the native's
    TheLossPartsAddUpToTheLoss), to 1e-12."""
    G = _golden()
    kw = G["kw"]
    f = {k: float(np.float32(kw[k])) for k in ("w_traj", "w_vel", "w_fric", "w_clear", "w_lreg")}
    wm = float(np.float32(kw["w_multi"]))

    def total(q):
        return (
            0.3 * f["w_traj"] * q["traj"]
            + 0.3 * f["w_vel"] * q["vel"]
            + f["w_fric"] * q["fric"]
            + f["w_clear"] * q["clear"]
            + q["nav"]
            + f["w_lreg"] * q["lreg"]
            + wm * q["multi"]
        )

    for s in G["steps"]:
        assert abs(total(s) - s["L"]) <= 1e-12 * abs(s["L"])
    tr = mtt.MaterialTrainerTorch(
        G["dir"] / "material_train_golden_w0.cvcnm", dtype=torch.float64, **G["kw"]
    )
    q = tr.loss_parts(G["batch"], ext=G["ext"])
    assert abs(total(q) - q["L"]) <= 1e-12 * abs(q["L"])


# --------------------------------------------------------------------------- 2. a live native


_needs_native = pytest.mark.skipif(
    not mtn.HAS_MATERIAL_TRAINER_LOSS,
    reason="pycvc lacks the nav_material_trainer_* binding (with its forward-only loss)",
)
_LIVE = {
    "legacy": ({}, None),
    "sampled": ({"ms_count": 4, "frac_range": (0.8, 0.98), "seed": 5}, "multistart"),
    "sampled-semi": (
        {"ms_count": 4, "frac_range": (0.8, 0.98), "seed": 5, "semi_implicit": True},
        "multistart",
    ),
    "bicycle": ({"bicycle": VEHICLE}, "bicycle"),
    "legacy+ext": ({}, "ext"),
    "bicycle+ext": ({"bicycle": VEHICLE}, "bicycle+ext"),
}


@_needs_native
@pytest.mark.parametrize("case", list(_LIVE))
def test_live_native_trainer_agrees(tmp_path, case):
    kw, need = _LIVE[case]
    have = {
        None: True,
        "multistart": mtn.HAS_MATERIAL_TRAINER_MULTISTART,
        "bicycle": mtn.HAS_MATERIAL_TRAINER_BICYCLE,
        "ext": nav_native.HAS_MATERIAL_EXT,
        "bicycle+ext": mtn.HAS_MATERIAL_TRAINER_BICYCLE and nav_native.HAS_MATERIAL_EXT,
    }[need]
    if not have:
        pytest.skip(f"this pycvc's trainer cannot run the {case!r} configuration")
    init = _write_model(tmp_path / "init.cvcnm", seed=11)
    batch = _batch(B=16, seed=21)
    ext = _force(batch) if "ext" in case else None
    native = mtn.MaterialTrainer(init, **kw)
    torch_tr = mtt.MaterialTrainerTorch(init, **kw)
    assert torch_tr.integrator == native.integrator
    assert _rel(torch_tr.loss(batch, ext=ext), native.loss(batch, ext=ext)) < 1e-5
    # L_multi on its own: the native's as loss(w_multi=1) - loss(w_multi=0).
    multi = [mtn.MaterialTrainer(init, **kw, w_multi=w).loss(batch, ext=ext) for w in (1.0, 0.0)]
    want = multi[0] - multi[1]
    assert want > 1e-3, "the fixture must score near-obstacle starts"
    assert _rel(torch_tr.loss_parts(batch, ext=ext)["multi"], want) < 1e-4, case
    for t in range(3):
        a = torch_tr.step(batch, lr=1e-3, ext=ext)
        b = native.step(batch, lr=1e-3, ext=ext)
        assert _rel(a, b) < 1e-5, f"step {t}: torch {a!r} native {b!r}"
    native.save(tmp_path / "native.cvcnm")
    worst, p9999, _ = _weight_gap(_weights(torch_tr), read_matnet(str(tmp_path / "native.cvcnm")).t)
    assert worst < WEIGHT_MAX and p9999 < 1e-6, f"{case}: weights {worst:.3e} ({p9999:.3e})"


# --------------------------------------------------------------------------- 3. self-consistency


_GRADCHECK = {
    "surrogate+ext": ({}, True, True),
    "bicycle+ext": ({"bicycle": VEHICLE}, True, False),
    "sampled": ({"ms_count": 3, "frac_range": (0.8, 0.98), "seed": 5}, False, False),
    "sampled-semi": (
        {"ms_count": 3, "frac_range": (0.8, 0.98), "seed": 5, "semi_implicit": True},
        False,
        False,
    ),
}


@pytest.mark.parametrize("case", list(_GRADCHECK))
def test_float64_gradcheck_of_the_full_chain(ckpt, case):
    """libcvc's FullChainGradcheck in float64: the analytic loss -> weights gradient against
    central differences of the loss with the CVaR quantile frozen at the base point's. A
    directional difference along the whole gradient, then (for the main case) the largest entry
    of every weight tensor."""
    kw, with_ext, per_tensor = _GRADCHECK[case]
    tr = mtt.MaterialTrainerTorch(ckpt, dtype=torch.float64, **kw)
    batch = _batch(B=8, seed=4)
    ext = _force(batch) if with_ext else None
    parts, grads = tr.loss_and_grad(batch, ext=ext)
    eta = parts["eta"]
    params = dict(tr.model.named_parameters())
    gnorm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    assert gnorm > 1e-3

    def L():
        return tr.loss_parts(batch, frozen_eta=eta, ext=ext)["L"]

    def nudge(scale):
        with torch.no_grad():
            for n, p in params.items():
                p.add_(grads[n], alpha=scale)

    eps = 1e-5
    nudge(eps / gnorm)
    lp = L()
    nudge(-2 * eps / gnorm)
    lm = L()
    nudge(eps / gnorm)
    fd = (lp - lm) / (2 * eps)
    assert abs(fd - gnorm) / gnorm < 1e-5, f"{case}: directional FD {fd!r} vs |g| {gnorm!r}"

    if not per_tensor:
        return
    checked = 0
    for n, p in params.items():
        g = grads[n].reshape(-1)
        i = int(torch.argmax(g.abs()))
        if float(g[i].abs()) < 1e-3:
            continue
        flat = p.data.view(-1)
        orig = float(flat[i])
        flat[i] = orig + 1e-6
        lp = L()
        flat[i] = orig - 1e-6
        lm = L()
        flat[i] = orig
        fd = (lp - lm) / 2e-6
        assert abs(fd - float(g[i])) <= 1e-4 * abs(float(g[i])) + 1e-8, f"d/d{n}[{i}]"
        checked += 1
    assert checked >= 40


def test_no_force_is_a_zero_force(ckpt):
    """ext=None and an all-zero force patch are the same loss and the same gradient."""
    tr = mtt.MaterialTrainerTorch(ckpt)
    batch = _batch()
    zero = _force(batch)
    zero = MaterialExt(patch=torch.zeros_like(zero.patch), patch_of_step=zero.patch_of_step)
    p0, g0 = tr.loss_and_grad(batch)
    p1, g1 = tr.loss_and_grad(batch, ext=zero)
    assert p0 == p1
    for n in g0:
        assert torch.equal(g0[n], g1[n]), n
    assert tr.loss(batch, ext=_force(batch)) != tr.loss(batch)


def test_every_force_source_reaches_the_loss(ckpt):
    """Per-step values, a callable and a patch: a constant force is the same loss whichever
    source carries it, and it moves the loss."""
    tr = mtt.MaterialTrainerTorch(ckpt, dtype=torch.float64)
    batch = _batch()
    B, _, Hp, Wp = batch["rollout_patch"].shape
    steps = int(batch["H"].max())
    f = np.array([0.2, -0.1], np.float32)
    by_values = MaterialExt(F=torch.from_numpy(np.broadcast_to(f, (B, steps, 2)).copy()))
    by_values_j = MaterialExt(F=by_values.F, J=torch.zeros(B, steps, 4))
    by_call = MaterialExt(sample=lambda s, o: torch.as_tensor(f, dtype=o.dtype).expand_as(o))
    by_patch = MaterialExt(
        patch=torch.from_numpy(np.broadcast_to(f[:, None, None], (B, 1, 2, Hp, Wp)).copy())
    )
    vals = [tr.loss(batch, ext=e) for e in (by_values, by_values_j, by_call, by_patch)]
    assert max(vals) - min(vals) <= 1e-12 * abs(vals[0])
    assert abs(vals[0] - tr.loss(batch)) > 1e-6


def test_loss_does_not_train_and_step_returns_the_pre_update_loss(ckpt):
    tr = mtt.MaterialTrainerTorch(ckpt)
    batch = _batch(seed=11)
    w = _weights(tr)
    first = tr.loss(batch)
    assert tr.loss(batch) == first
    assert all(np.array_equal(w[k], v) for k, v in _weights(tr).items())
    assert tr.step(batch, lr=0.0) == first  # lr 0: the moments move, the weights do not
    assert all(np.array_equal(w[k], v) for k, v in _weights(tr).items())
    assert tr.step(batch, lr=1e-3) == first
    assert tr.loss(batch) < first
    # frozen_eta pins the quantile, rounded to float32 as the binding rounds it.
    tr2 = mtt.MaterialTrainerTorch(ckpt)
    eta = tr2.loss_parts(batch)["eta"]
    assert tr2.loss(batch, frozen_eta=eta) == tr2.loss_parts(batch, frozen_eta=mtt._f32(eta))["L"]
    assert tr2.loss(batch, frozen_eta=eta + 1.0) != first


def test_training_is_deterministic(ckpt):
    batch = _batch(seed=12)
    runs = []
    for _ in range(2):
        tr = mtt.MaterialTrainerTorch(ckpt, ms_count=3, frac_range=(0.8, 0.98), seed=9)
        runs.append(([tr.step(batch, lr=1e-3) for _ in range(3)], _weights(tr)))
    assert runs[0][0] == runs[1][0]
    assert all(np.array_equal(runs[0][1][k], runs[1][1][k]) for k in runs[0][1])


@pytest.mark.parametrize("bicycle", [None, VEHICLE])
def test_it_trains(ckpt, bicycle):
    """The loss descends, on the point mass and on the bicycle (libcvc P8b)."""
    tr = mtt.MaterialTrainerTorch(ckpt, bicycle=bicycle)
    assert tr.integrator == ("surrogate" if bicycle is None else "bicycle")
    batch = _batch(seed=13)
    losses = [tr.step(batch, lr=mtt.cosine_lr(1e-3, s, 12)) for s in range(12)]
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < (0.8 if bicycle is None else 1.0) * losses[0], losses


def test_sampled_multi_start_semantics(ckpt):
    """The native's test_sampled_multi_start_args on the torch trainer."""
    batch = _batch(seed=13)

    def loss(**kw):
        return mtt.MaterialTrainerTorch(ckpt, **kw).loss(batch)

    legacy = loss()
    assert loss(ms_count=10, frac_range=(0.9, 0.9), seed=123) == legacy  # degenerate: no draw
    semi = loss(semi_implicit=True)
    assert semi != legacy, "semi_implicit did not reach the rollout"
    sampled = dict(ms_count=4, frac_range=(0.8, 0.98))
    tr = mtt.MaterialTrainerTorch(ckpt, seed=5, **sampled)
    base = tr.loss(batch)
    assert base != legacy and tr.loss(batch) == base
    assert loss(seed=5 + 2**32, **sampled) == base  # only the low 32 bits count
    s0 = tr.step(batch, lr=0.0)
    s1 = tr.step(batch, lr=0.0)
    assert s0 == base, "step 0 did not draw with the base seed"
    assert s1 != s0 and s1 == loss(seed=6, **sampled), "step 1 did not draw with seed + 1"
    assert tr.loss(batch) == base, "steps moved the validation draw"
    with pytest.raises(ValueError, match="frac_range"):
        mtt.MaterialTrainerTorch(ckpt, ms_count=4, frac_range=(0.98, 0.8))


def test_semi_implicit_multi_is_grl_snams_multi_start_penalty(ckpt):
    """With semi_implicit, L_multi is surrogate_robust.multi_start_penalty fed the model's
    coefficients and a manual_seed generator; the explicit default (libcvc's) is not."""
    from grl_snam.surrogate_robust import multi_start_penalty

    batch = _batch(B=16, seed=31)
    ms = dict(ms_count=5, frac_range=(0.7, 0.95))
    semi = mtt.MaterialTrainerTorch(ckpt, semi_implicit=True, seed=77, **ms)
    expl = mtt.MaterialTrainerTorch(ckpt, seed=77, **ms)
    b = semi._batch(batch)
    with torch.no_grad(), mtt._math_path():
        alphas, beta, gamma = semi.model(
            b["obs_feats"], b["obs_mask"], b["goal_feats"], b["risk_patch"]
        )[:3]
        ref = multi_start_penalty(
            b["o0"], b["v0"], b["goal"], b["C"], b["R"], b["obs_mask"], alphas, beta, gamma,
            b["d_hat"], b["dt"], b["H"], robot_radius=b["rr"], margin_factor=0.5, ms_h=3,
            ms_dt_mult=4.0, tau=0.05, generator=torch.Generator().manual_seed(77), **ms,
        ).item()  # fmt: skip
    got = semi.loss_parts(batch)["multi"]
    assert ref > 1e-3, "the fixture must score near-obstacle starts"
    assert _rel(got, ref) < 1e-5, f"semi-implicit {got!r} vs multi_start_penalty {ref!r}"
    assert _rel(expl.loss_parts(batch)["multi"], ref) > 1e-3


def test_the_draw_is_libcvcs():
    """multi_start_fracs: the CPU torch.rand stream, lo + span * u in float32, start-major."""
    f = mtt.multi_start_fracs(3, 4, 0.8, 0.98, 5 + 2**32)
    u = torch.rand(12, generator=torch.Generator().manual_seed(5), dtype=torch.float32)
    want = (np.float32(0.8) + np.float32(0.98 - 0.8) * u.numpy()).reshape(3, 4)
    assert np.array_equal(f.numpy(), want)
    assert mtt.multi_start_fracs(7, 4, 0.9, 0.9, 1).shape == (1, 4)  # degenerate: one start
    assert mtt.multi_start_fracs(0, 4, 0.8, 0.9, 1).numel() == 0


def test_cvcnm_round_trip(ckpt, tmp_path):
    """.cvcnm -> torch -> .cvcnm is lossless, and the header keeps patch_size."""
    tr = mtt.MaterialTrainerTorch(ckpt, dtype=torch.float64)
    out = tmp_path / "again.cvcnm"
    tr.save(out)
    a, b = read_matnet(str(ckpt)), read_matnet(str(out))
    assert b.patch_size == a.patch_size == 16
    assert set(a.t) == set(b.t) and all(np.array_equal(a.t[k], b.t[k]) for k in a.t)
    # From a module: the trainer copies it, and saves what it trained.
    m = mtt.load_cvcnm(ckpt)
    tr2 = mtt.MaterialTrainerTorch(m)
    tr2.step(_batch(), lr=1e-3)
    assert not torch.equal(m.alpha_head[0].weight, tr2.model.alpha_head[0].weight)
    tr2.save(out)
    back = mtt.MaterialTrainerTorch(out)
    assert all(
        torch.equal(back.model.state_dict()[k], tr2.model.state_dict()[k])
        for k in back.model.state_dict()
    )


def test_api_contract(ckpt):
    with pytest.raises(NotImplementedError, match="dynamic"):
        mtt.MaterialTrainerTorch(ckpt, integrator="dynamic")
    with pytest.raises(ValueError, match="integrator"):
        mtt.MaterialTrainerTorch(ckpt, integrator="hovercraft")
    with pytest.raises(ValueError, match="bicycle"):
        mtt.MaterialTrainerTorch(ckpt, bicycle=BicycleParams(L=0.0))
    with pytest.raises(ValueError, match="dtype"):
        mtt.MaterialTrainerTorch(ckpt, dtype=torch.float16)
    tr = mtt.MaterialTrainerTorch(ckpt, integrator="bicycle")  # the default vehicle
    assert tr.bicycle.L == mtt._f32(BicycleParams().L)  # held as float32, as the native holds it
    tr = mtt.MaterialTrainerTorch(ckpt, device="cpu", use_cuda=True)
    assert not tr.cuda_active
    assert mtt.MaterialTrainerTorch(ckpt, use_cuda=True).cuda_active == torch.cuda.is_available()
    with pytest.raises(ValueError, match="patch_size"):
        tr.loss(_batch(P=8))
    bad = _batch()
    bad["R"] = bad["R"][:, :2]
    with pytest.raises(ValueError, match="shape"):
        tr.loss(bad)
    assert mtt.cosine_lr is mtn.cosine_lr and mtt.BATCH_KEYS == mtn.BATCH_KEYS
    assert tr.step(_batch(), lr=1e-3, num_threads=4) > 0.0  # num_threads: accepted, ignored


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_matches_cpu(ckpt, dtype):
    batch = _batch(seed=17)
    ext = _force(batch)
    tol = 1e-9 if dtype == torch.float64 else 1e-5
    runs = {}
    for dev in ("cpu", "cuda"):
        tr = mtt.MaterialTrainerTorch(
            ckpt, device=dev, dtype=dtype, ms_count=3, frac_range=(0.8, 0.98)
        )
        losses = [tr.step(batch, lr=1e-3, ext=ext) for _ in range(3)]
        runs[dev] = (losses, tr.loss_parts(batch, ext=ext), _weights(tr))
    (lc, pc, wc), (lg, pg, wg) = runs["cpu"], runs["cuda"]
    assert all(_rel(a, b) < tol for a, b in zip(lg, lc)), (lg, lc)
    assert all(_rel(pg[k], pc[k]) < tol for k in ("L", *PARTS) if pc[k] != 0.0)
    worst, _, _ = _weight_gap(wg, wc)
    assert worst < (1e-9 if dtype == torch.float64 else 1e-5), worst
