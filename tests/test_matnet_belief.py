"""``.cvcnm`` version 2 and the belief-plane trainer inputs (libcvc's NAV-BELIEF-MATERIAL-ROADMAP
§5-§6, the torch side T2): a 3-channel ``CoefEnergyNetMaterial``, its belief block, the loader's
shape check, and the witness gate in training."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grl_snam import material_train_torch as mtt  # noqa: E402
from grl_snam.material_nav import CoefEnergyNetMaterial  # noqa: E402
from grl_snam.tools.matnet_export import BeliefHeader, read_matnet, write_matnet  # noqa: E402


def _net(channels, seed=0, **kw):
    torch.manual_seed(seed)
    m = CoefEnergyNetMaterial(patch_size=16, risk_channels=channels, **kw)
    m.patch_size = 16
    return m


def test_a_two_channel_net_is_version_1_byte_for_byte(tmp_path):
    """Version 1 is unchanged: a 2-channel net writes no belief block, and reading it back gives
    2 channels, no belief header and the same bytes on a re-write."""
    a, b = tmp_path / "a.cvcnm", tmp_path / "b.cvcnm"
    write_matnet(_net(2), str(a), meta="m")
    mn = read_matnet(str(a))
    assert (mn.version, mn.risk_channels, mn.belief, mn.meta) == (1, 2, None, "m")
    assert a.read_bytes()[4:8] == (1).to_bytes(4, "little")
    write_matnet(mtt.load_cvcnm(a), str(b), meta="m")
    assert a.read_bytes() == b.read_bytes()


def test_a_three_channel_net_round_trips_as_version_2(tmp_path):
    hdr = BeliefHeader(
        risk_channels=3,
        seen_rule=1,
        unknown_risk=0.25,
        unknown_hard=1,
        unknown_gate_hard=0,
        material_range_m=7.0,
        patch_extent_m=100.0,
    )
    p = tmp_path / "b.cvcnm"
    write_matnet(_net(3), str(p), belief=hdr)
    mn = read_matnet(str(p))
    assert (mn.version, mn.risk_channels) == (2, 3)
    assert mn.belief == hdr
    assert mn.t["risk_enc.net.0.weight"].shape == (16, 3, 3, 3)
    # The loader keeps the belief block on the model; a save writes it back unchanged.
    m = mtt.load_cvcnm(p)
    assert m.risk_channels == 3 and m.belief_header == hdr
    q = tmp_path / "c.cvcnm"
    mtt.save_cvcnm(m, q)
    assert q.read_bytes() == p.read_bytes()


def test_a_three_channel_net_defaults_its_belief_block(tmp_path):
    p = tmp_path / "d.cvcnm"
    write_matnet(_net(3), str(p))
    assert read_matnet(str(p)).belief == BeliefHeader()


def test_the_reader_refuses_a_channel_count_its_weights_contradict(tmp_path):
    """A version-1 file whose encoder takes 3 channels (what a pre-belief writer would have made
    of a 3-channel net) is refused, not mis-strided."""
    p = tmp_path / "bad.cvcnm"
    write_matnet(_net(3), str(p))
    buf = bytearray(p.read_bytes())
    # Rewrite it as version 1 by dropping the 28-byte belief block after the 4 floats.
    head = 4 + 4 + 8 + 5 * 4 + 4 * 4
    buf[4:8] = (1).to_bytes(4, "little")
    del buf[head : head + 28]
    p.write_bytes(bytes(buf))
    with pytest.raises(ValueError, match="2-channel file needs"):
        read_matnet(str(p))


def test_an_unknown_version_is_refused(tmp_path):
    p = tmp_path / "v9.cvcnm"
    write_matnet(_net(2), str(p))
    buf = bytearray(p.read_bytes())
    buf[4:8] = (9).to_bytes(4, "little")
    p.write_bytes(bytes(buf))
    with pytest.raises(ValueError, match="unsupported .cvcnm version 9"):
        read_matnet(str(p))


def _batch(B=4, N=3, P=16, C=3, seed=1):
    rng = np.random.default_rng(seed)
    f = np.float32
    return {
        "obs_feats": rng.normal(size=(B, N, 6)).astype(f),
        "obs_mask": np.ones((B, N), np.uint8),
        "goal_feats": rng.normal(size=(B, 4)).astype(f),
        "risk_patch": rng.random((B, C, P, P)).astype(f),
        "o0": rng.normal(size=(B, 2)).astype(f),
        "v0": rng.normal(size=(B, 2)).astype(f) * 0.1,
        "goal": rng.normal(size=(B, 2)).astype(f) + 4.0,
        "C": rng.normal(size=(B, N, 2)).astype(f) * 2.0,
        "R": rng.random((B, N)).astype(f) * 0.5 + 0.2,
        "rollout_patch": rng.random((B, 6, 9, 9)).astype(f),
        "rr": np.full(B, 0.3, f),
        "d_hat": np.full(B, 1.0, f),
        "dt": np.full(B, 0.1, f),
        "H": np.array([2, 3] * (B // 2), np.int32),
        "o_tgt": rng.normal(size=(B, 2)).astype(f),
        "v_tgt": rng.normal(size=(B, 2)).astype(f),
        "gamma_o": rng.random(B).astype(f),
    }


def _trainer(tmp_path, C=3):
    p = tmp_path / "t.cvcnm"
    write_matnet(_net(C, seed=5), str(p))
    return mtt.MaterialTrainerTorch(p, dtype=torch.float64, w_multi=0.0)


def test_the_trainer_takes_a_three_channel_batch_and_checks_it(tmp_path):
    tr = _trainer(tmp_path)
    assert np.isfinite(tr.loss_parts(_batch())["L"])
    with pytest.raises(ValueError, match="takes 3 channels"):
        tr.loss_parts(_batch(C=2))


def test_an_all_on_gate_is_the_ungated_trainer(tmp_path):
    tr = _trainer(tmp_path)
    b = _batch()
    want = tr.loss_parts(b)
    got = tr.loss_parts({**b, "gate_active": np.ones(4, np.uint8)})
    for k in want:
        assert got[k] == want[k], k


def test_a_closed_gate_takes_lam_soft_out_of_the_rollout(tmp_path):
    """With every gate closed, the rollout no longer depends on lam_soft: its head's gradient is
    only lreg's, the same as with the soft force's weight zeroed in the loss."""
    tr = _trainer(tmp_path)
    b = _batch()
    gated = tr.loss_parts({**b, "gate_active": np.zeros(4, np.uint8)})
    open_ = tr.loss_parts(b)
    assert gated["lreg"] == open_["lreg"], "lreg reads the head's own lam_soft"
    assert gated["L"] != open_["L"], "the soft force acted in the open rollout"
    with pytest.raises(ValueError, match="gate_active has 3 entries"):
        tr.loss_parts({**b, "gate_active": np.ones(3, np.uint8)})
