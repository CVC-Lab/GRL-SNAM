"""grl_snam.learning.arms, coef_export's --constant and read_coef_mlp, and tools.arms_eval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import grl_snam.sdf_nav as sdf_nav  # noqa: E402
from grl_snam import nav_native  # noqa: E402
from grl_snam.learning import arms as A  # noqa: E402
from grl_snam.tools import arms_eval, coef_export  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "learning_history"
SMALL = [
    "--grid", "48", "--agents", "16", "--ticks", "15",
    "--scene-agents", "4", "--scene-steps", "20", "--scene-grid", "48", "--boot", "200",
]  # fmt: skip


def _feats(n=64, d=5, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).uniform(-2, 2, (n, d)).astype(np.float32))


def test_parse_arm_and_kind():
    a = A.parse_arm("trained = run/x.cvcnav")
    assert (a.name, a.spec, a.kind) == ("trained", "run/x.cvcnav", "cvcnav")
    assert A.parse_arm("u=ck.pt").kind == "pt"
    assert A.parse_arm("c=const:1,3,4").kind == "const"
    for bad in ("noequals", "=x.cvcnav", "a="):
        with pytest.raises(ValueError):
            A.parse_arm(bad)


def test_parse_constant_forms():
    assert A.parse_constant("const:") == {"alpha": 1.0, "beta": 3.0, "gamma": 4.0}
    assert A.parse_constant("bias") == A.parse_constant("const:")
    assert A.parse_constant("const:1.2,3.5,4") == {"alpha": 1.2, "beta": 3.5, "gamma": 4.0}
    assert A.parse_constant("1,2,3,0.5")["lam_soft"] == 0.5
    named = A.parse_constant("const:gamma=5,lam_soft=0.3")
    assert named == {"alpha": 1.0, "beta": 3.0, "gamma": 5.0, "lam_soft": 0.3}
    with pytest.raises(ValueError):
        A.parse_constant("const:delta=1")
    with pytest.raises(ValueError):
        A.parse_constant("1,2,3,4,5,6")


def test_constants_from_a_history_take_the_picks_head_means():
    got = A.parse_constant(f"const:hist={FIX / 'coef_train_history.jsonl'}")
    # coef_train_history's pick is step 4: alpha 1 + 0.2 * 4/6, beta 3 - 0.3 * 4/6, gamma 4 + ...
    assert got["alpha"] == pytest.approx(1.0 + 0.2 * 4 / 6, abs=1e-6)
    assert got["beta"] == pytest.approx(3.0 - 0.3 * 4 / 6, abs=1e-6)
    assert set(got) == {"alpha", "beta", "gamma"}
    with pytest.raises(ValueError, match="lacks alpha/beta/gamma"):
        A.constants_from_history(str(FIX / "ext_native_history.jsonl"))


@pytest.mark.parametrize(
    "values",
    [
        {"alpha": 1.3, "beta": 2.7, "gamma": 4.4},
        {"alpha": 0.05, "beta": 8.0, "gamma": 0.5, "lam_soft": 0.7},
        {"alpha": 1.0, "beta": 3.0, "gamma": 4.0, "lam_soft": 1.5, "lam_hard": 4.0},
    ],
)
def test_constant_net_outputs_its_constants_everywhere(values):
    m = A.constant_coef_mlp(values)
    f = _feats()
    out = m.coeffs_and_lam(f) if "lam_soft" in values else m(f)
    for col, (h, c) in zip(out, values.items(), strict=True):
        assert torch.allclose(col, torch.full_like(col, c), rtol=1e-5), h
    lin = [x for x in m.net if isinstance(x, torch.nn.Linear)]
    assert all(float(x.weight.abs().max()) == 0.0 for x in lin)


def test_constant_net_takes_its_layout_from_like():
    like = sdf_nav.add_risk_feature(sdf_nav.CoefMLP(hidden=32))
    m = A.constant_coef_mlp([1.1, 2.2, 3.3], like=like)
    assert m.in_dim == like.in_dim and m.use_risk and m.net[0].out_features == 32
    al, be, ga = m(_feats(d=m.in_dim))
    assert float(al[0]) == pytest.approx(1.1, rel=1e-5) and float(ga[-1]) == pytest.approx(3.3)


def test_constant_net_refuses_what_it_cannot_represent():
    with pytest.raises(ValueError, match="positive"):
        A.constant_coef_mlp([0.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="lam_soft"):
        A.constant_coef_mlp({"lam_hard": 1.0})
    with pytest.raises(ValueError, match="sigmoid"):
        A.constant_coef_mlp({"lam_soft": 9.0, "lam_hard": 1.0})
    with pytest.raises(ValueError, match="lam_hard"):
        A.constant_coef_mlp({"lam_soft": 1.0, "lam_hard": 20.0})
    with pytest.raises(ValueError, match="unknown heads"):
        A.constant_coef_mlp({"delta": 1.0})


@pytest.mark.parametrize(
    "build",
    [
        lambda: sdf_nav.CoefMLP(),
        lambda: sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP()),
        lambda: sdf_nav.add_risk_feature(sdf_nav.CoefMLP(hidden=16)),
        lambda: sdf_nav.add_lam_head(sdf_nav.add_risk_feature(sdf_nav.CoefMLP())),
        lambda: sdf_nav.CoefMLP(use_lam=True, use_lam_hard=True, lam_init=0.6, lam_hard_init=2.0),
    ],
)
def test_read_coef_mlp_round_trips(tmp_path, build):
    torch.manual_seed(3)
    m = build().eval()
    with torch.no_grad():  # move every weight, so the round trip is not of an init
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    a, b = tmp_path / "a.cvcnav", tmp_path / "b.cvcnav"
    coef_export.write_coef_mlp(m, str(a))
    r = coef_export.read_coef_mlp(str(a))
    f = _feats(d=m.in_dim)
    lam = m.use_lam
    want = m.coeffs_and_lam(f) if lam else m(f)
    got = r.coeffs_and_lam(f) if lam else r(f)
    for x, y in zip(want, got, strict=True):
        assert torch.equal(x, y)
    coef_export.write_coef_mlp(r, str(b))
    assert a.read_bytes() == b.read_bytes()


def test_read_coef_mlp_refuses_a_bad_blob(tmp_path):
    p = tmp_path / "x.cvcnav"
    p.write_bytes(b"NOPE" + bytes(40))
    with pytest.raises(ValueError, match="not a .cvcnav"):
        coef_export.read_coef_mlp(str(p))
    coef_export.write_coef_mlp(sdf_nav.CoefMLP(), str(p))
    raw = bytearray(p.read_bytes())
    raw[28] ^= 0xFF  # the arch_hash
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="arch_hash"):
        coef_export.read_coef_mlp(str(p))


def test_coef_export_constant_cli(tmp_path, capsys):
    out = tmp_path / "c.cvcnav"
    assert coef_export.main(["--constant", "1.09,3.05,3.99", str(out)]) == 0
    m = coef_export.read_coef_mlp(str(out))
    al, be, ga = m(_feats())
    assert float(al.mean()) == pytest.approx(1.09, rel=1e-5)
    assert torch.unique(be).numel() == 1 and float(ga[0]) == pytest.approx(3.99, rel=1e-5)
    assert b"constant policy" in out.read_bytes()
    pt = tmp_path / "c.pt"
    assert coef_export.main(["--constant", "alpha=2", str(pt)]) == 0
    from grl_snam.tools.scorecard_eval import load_model

    al, be, _ = load_model(str(pt))(_feats())
    assert float(al[0]) == pytest.approx(2.0, rel=1e-5) and float(be[0]) == pytest.approx(3.0)
    like = tmp_path / "like.cvcnav"
    coef_export.write_coef_mlp(sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP()), str(like))
    assert coef_export.main(["--constant", "bias", "--like", str(like), str(out)]) == 0
    assert coef_export.read_coef_mlp(str(out)).in_dim == 6
    with pytest.raises(SystemExit):
        coef_export.main(["--constant", "bias", str(out), str(pt)])
    capsys.readouterr()


@pytest.mark.skipif(not nav_native.HAS_COEF_MLP, reason="pycvc lacks nav_coef_mlp_forward")
def test_constant_blob_runs_on_libcvcs_coef_mlp(tmp_path):
    out = tmp_path / "c.cvcnav"
    coef_export.write_coef_mlp(A.constant_coef_mlp([1.25, 2.5, 5.0]), str(out))
    coef = nav_native.coef_mlp_forward(str(out), _feats().numpy())
    assert np.allclose(coef, [[1.25, 2.5, 5.0]] * 64, rtol=1e-5)


def test_arms_table_shape_and_pairing():
    values = {
        "c1": {
            "m": {
                "trained": {1: 0.5, 2: 0.7, 3: 0.6},
                "untrained": {1: 0.3, 2: 0.5, 3: None},
            }
        }
    }
    t = A.arms_table(values, ["trained", "untrained"], direction={"m": "up"}, n_boot=100)
    assert t["schema"] == A.ARMS_SCHEMA and t["reference"] == "trained"
    res = t["results"][0]
    assert res["case"] == "c1" and res["episodes"] == 3 and res["seeds"] == [1, 2, 3]
    tr, un = res["metrics"]["m"]["trained"], res["metrics"]["m"]["untrained"]
    assert set(tr) == {"mean", "ci95", "n", "seeds", "values"}  # the reference: no difference
    assert un["values"] == [0.3, 0.5, None] and un["n"] == 2
    # untrained - trained over the two seeds both have: -0.2 each.
    assert un["paired_diff"]["mean"] == pytest.approx(-0.2) and un["paired_diff"]["n"] == 2
    assert A.arm_values(t, "c1", "m", "untrained") == {"1": 0.3, "2": 0.5, "3": None}
    assert A.arm_values(t, "nope", "m", "untrained") is None
    with pytest.raises(ValueError):
        A.arms_table(values, ["trained"], reference="x")
    # The same draws, the same table.
    assert A.arms_table(values, ["trained", "untrained"], n_boot=100) == A.arms_table(
        values, ["trained", "untrained"], n_boot=100
    )


def test_arms_eval_end_to_end(tmp_path, capsys):
    torch.manual_seed(0)
    u = tmp_path / "u.cvcnav"
    coef_export.write_coef_mlp(sdf_nav.CoefMLP(), str(u))
    out = tmp_path / "arms.json"
    argv = ["--arm", f"untrained={u}", "--arm", "constant=const:bias", "--arm", "low=const:0.3,1,1"]
    rc = arms_eval.main(argv + ["--seeds", "3", "--out", str(out)] + SMALL)
    assert rc == 0
    t = json.loads(out.read_text())
    assert t["schema"] == "grl_snam.arms/v1" and [a["name"] for a in t["arms"]] == [
        "untrained",
        "constant",
        "low",
    ]
    assert len(t["arms"][0]["sha256"]) == 64 and t["arms"][1]["kind"] == "const"
    cases = {r["case"]: r for r in t["results"]}
    assert set(cases) == {"coef", "scorecard"}
    assert cases["coef"]["seeds"] == [1000, 1001, 1002]
    assert cases["coef"]["primary"] == ["reach", "pen"]
    assert set(cases["coef"]["metrics"]) == {"reach", "pen", "clear", "gap"}
    assert "veh_contacts_per_run" not in cases["scorecard"]["metrics"]  # --contact-r 0
    assert t["direction"]["pen"] == "down"
    low = cases["coef"]["metrics"]["gap"]["low"]
    assert len(low["values"]) == 3 and low["paired_diff"]["n"] == 3
    assert "| reach |" in capsys.readouterr().out


def test_arms_eval_refuses_bad_arguments(capsys):
    with pytest.raises(SystemExit):
        arms_eval.main(["--arm", "a=const:", "--arm", "a=const:1,2,3"])
    with pytest.raises(SystemExit):
        arms_eval.main(["--arm", "a=const:", "--cases", "coef,nope"])
    with pytest.raises(ValueError):
        arms_eval.evaluate_arms([A.parse_arm("a=const:")], [0], cases=["nope"])
    capsys.readouterr()
