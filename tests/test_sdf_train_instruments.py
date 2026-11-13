"""``grl-snam train``'s learning instruments and the held-out scorer behind them
(grl_snam.tools.sdf_eval).

With the history and checkpoints on, the trained weights are the same as without. ``val_frac``
holds out whole blocks of the region: training starts never fall in them, every checkpoint scores
fixed held-out pairs (the objective's terms and the seek drive), and the best held-out reach is
kept as ``<stem>.best_heldout.pt``, step 0 (the untrained net) included.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grl_snam.sdf_nav as sdf_nav
from grl_snam import learning_history as lh
from grl_snam.tools import sdf_eval
from grl_snam.tools.train import train_sdf

RR_WORLD = 3.0


@pytest.fixture(scope="module")
def sdf_npz(tmp_path_factory):
    """A 96 x 96 field over +-300 m (1 unit = 100 m) with two buildings."""
    n, half = 96, 300.0
    occ = np.zeros((n, n), bool)
    occ[30:50, 20:40] = True
    occ[55:75, 60:80] = True
    scale = 1.0 / 100.0
    bounds = (-half, -half, half, half)
    phi, nx, ny = sdf_nav.build_sdf(occ, bounds, scale)
    path = tmp_path_factory.mktemp("sdf") / "nav_sdf.npz"
    np.savez(
        path, phi=phi, normal_x=nx, normal_y=ny, bounds=np.array(bounds, np.float32),
        center=np.zeros(2, np.float32), scale=np.float32(scale), region=np.float32(280.0),
    )  # fmt: skip
    return str(path)


def _state(path):
    return torch.load(path, weights_only=False)["model_state_dict"]


SMALL = dict(steps=4, batch=16, horizon=8, threads=1, device="cpu", log_every=100)


def test_the_instruments_do_not_change_training(sdf_npz, tmp_path):
    plain = train_sdf(sdf_npz, str(tmp_path / "a.pt"), seed=3, **SMALL)
    inst = train_sdf(
        sdf_npz, str(tmp_path / "b.pt"), seed=3, history=str(tmp_path / "h.jsonl"),
        checkpoint_every=2, **SMALL,
    )  # fmt: skip
    a, b = _state(plain), _state(inst)
    assert all(torch.equal(a[k], b[k]) for k in a)
    recs = lh.read_history(tmp_path / "h.jsonl")
    assert recs[0]["kind"] == "header" and recs[0]["trainer"] == "grl-snam train"
    steps = [r for r in recs if r["kind"] == "step"]
    assert [r["step"] for r in steps] == [1, 2, 3, 4]
    assert set(steps[0]["terms"]) == {"loss", "goal", "col", "reg"}
    cks = [r for r in recs if r["kind"] == "checkpoint"]
    assert [c["step"] for c in cks] == [0, 2, 4]
    assert all(c["fitness"] is None for c in cks)
    step0 = _state(tmp_path / "b.step000000.pt")
    fresh = (torch.manual_seed(3), sdf_nav.CoefMLP())[1].state_dict()
    assert all(torch.equal(step0[k], fresh[k]) for k in fresh)  # step 0 is the untrained net


def test_the_held_out_split_scores_every_checkpoint_and_keeps_the_pick(sdf_npz, tmp_path):
    out = tmp_path / "c.pt"
    train_sdf(
        sdf_npz, str(out), seed=0, history=str(tmp_path / "h.jsonl"), checkpoint_every=2,
        val_frac=0.25, val_blocks=4, val_pairs=32, val_ticks=40, **SMALL,
    )  # fmt: skip
    recs = lh.read_history(tmp_path / "h.jsonl")
    hdr = recs[0]
    assert hdr["heldout_blocks"] == sdf_eval.block_split(4, 0.25, 0)
    cks = [r for r in recs if r["kind"] == "checkpoint"]
    for c in cks:
        assert c["fitness"] == c["heldout"]["reach"]
        assert {
            "obj_loss",
            "obj_goal",
            "obj_col",
            "obj_reg",
            "pen_ticks",
            "path_m",
            "gap_m",
        } <= set(c["heldout"])
    best = max(cks, key=lambda c: c["fitness"])  # the first maximum wins ties
    end = recs[-1]
    assert end["best_step"] == best["step"]
    picked = _state(tmp_path / "c.best_heldout.pt")
    blob = _state(tmp_path / f"c.step{best['step']:06d}.pt")
    assert all(torch.equal(picked[k], blob[k]) for k in blob)


def test_block_split_and_pool(sdf_npz):
    scene = sdf_eval.load_scene(sdf_npz)
    rr = RR_WORLD * scene.scale
    pool = sdf_eval.free_pool(scene, rr)
    phi, _ = scene.field.sample(torch.from_numpy(pool))
    assert float(phi.min()) > rr
    held = sdf_eval.block_split(4, 0.25, 7)
    assert len(held) == 4 and held == sdf_eval.block_split(4, 0.25, 7)
    train, test = sdf_eval.split_pool(pool, scene.region_n, 4, held)
    assert len(train) + len(test) == len(pool)
    assert set(sdf_eval.block_ids(test, scene.region_n, 4)) <= set(held)
    assert not set(sdf_eval.block_ids(train, scene.region_n, 4)) & set(held)
    with pytest.raises(ValueError):
        sdf_eval.block_split(4, 1.0, 0)


def test_pairs_are_fixed_free_and_in_range(sdf_npz):
    scene = sdf_eval.load_scene(sdf_npz)
    rr = RR_WORLD * scene.scale
    pool = sdf_eval.free_pool(scene, rr)
    s, g = sdf_eval.sample_pairs(scene, pool, 64, 5, rr=rr, d_min=0.5, d_max=1.0)
    s2, g2 = sdf_eval.sample_pairs(scene, pool, 64, 5, rr=rr, d_min=0.5, d_max=1.0)
    assert torch.equal(s, s2) and torch.equal(g, g2)
    d = (g - s).norm(dim=-1)
    assert float(d.min()) >= 0.5 - 1e-5 and float(d.max()) <= 1.0 + 1e-5
    phi, _ = scene.field.sample(g)
    assert float(phi.min()) > rr
    assert float(g.abs().max()) < scene.region_n


def test_the_seek_drive_arrives_in_the_open_and_stops_there(sdf_npz):
    scene = sdf_eval.load_scene(sdf_npz)
    rr = RR_WORLD * scene.scale
    starts = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    goals = torch.tensor([[2.4, -2.0], [-0.7, -0.4]])  # open ground; behind a building
    torch.manual_seed(0)
    d = sdf_eval.seek_drive(
        sdf_nav.CoefMLP(), scene, starts, goals, rr=rr, d_hat=0.35, dt=0.06, vmax=0.9,
        nsub=3, ticks=150, reach_tol=0.05,
    )  # fmt: skip
    assert bool(d["reached"][0])
    assert 0 < d["ticks_to_goal"][0] < 150
    assert d["gap_m"][0] < 5.0  # stopped at arrival, 0.05 units = 5 m
    assert 30.0 < d["path_m"][0] < 60.0
    assert d["alpha"][0] == pytest.approx(1.0, abs=0.2)
    s = sdf_eval.drive_summary(d, 0.06)
    assert set(s) == {"reach", "pen_ticks", "path_m", "min_clear_m", "gap_m", "time_to_goal_s"}
    assert s["time_to_goal_s"] == pytest.approx(
        float(d["ticks_to_goal"][d["reached"]].mean()) * 0.06
    )
    terms = sdf_eval.objective_terms(
        sdf_nav.CoefMLP(), scene, starts, goals, rr=rr, d_hat=0.35, dt=0.06, vmax=0.9, horizon=8
    )
    assert terms["loss"] == pytest.approx(terms["goal"] + 3 * terms["col"] + 0.1 * terms["reg"])
