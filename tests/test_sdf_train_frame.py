"""``grl-snam train`` simulates the robot at its real size, whatever frame the map was built in.

The physics constants are numbers in sim units, tuned where one unit is 20 m: rr 3 m (0.15),
``d_hat`` 0.35 (7 m), ``vmax`` 0.9 (18 m/s). ``grl-snam sdf`` frames a map at ~10 units across
its region, so Austin's ``nav_sdf.npz`` is 300 m per unit, and the trainer used the npz's frame:
there the same numbers were a 270 m/s robot with a 105 m barrier band, and the collision
margin of 0.02 units was 6 m (docs/learning-validation.md §8.8, §9). The trainer now reads the
field in a 20 m sim frame (``metres_per_unit``), draws its pairs in the npz's frame as before,
and states its objective's lengths in metres.

The fixture is one piece of geometry, in metres, written twice: at 300 m per unit (Austin's
frame) and at 20 m per unit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grl_snam.sdf_nav as sdf_nav
from grl_snam import learning_history as lh
from grl_snam.tools import sdf_eval
from grl_snam.tools.train import train_sdf

RUN = dict(steps=2, batch=8, horizon=8, threads=1, device="cpu", log_every=1000, seed=0)


def _geometry():
    n, half = 96, 144.0  # 3 m cells
    rng = np.random.default_rng(0)
    occ = np.zeros((n, n), bool)
    for _ in range(n * n // 110):
        w, h = rng.integers(4, 12, 2)
        x, y = rng.integers(0, n - 12, 2)
        occ[y : y + h, x : x + w] = True
    return occ, (-half, -half, half, half), 0.95 * half


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    occ, bounds, region = _geometry()
    out = {}
    for mpu in (300.0, 20.0):
        phi, nx, ny = sdf_nav.build_sdf(occ, bounds, 1.0 / mpu)
        path = tmp_path_factory.mktemp(f"f{int(mpu)}") / "nav_sdf.npz"
        np.savez(
            path, phi=phi, normal_x=nx, normal_y=ny, bounds=np.array(bounds, np.float32),
            center=np.zeros(2, np.float32), scale=np.float32(1.0 / mpu), region=np.float32(region),
        )  # fmt: skip
        out[mpu] = str(path)
    return out


def _meta(path):
    return torch.load(path, weights_only=False)["meta"]


def test_the_robot_is_its_real_size_whatever_the_npz_frame(frames, tmp_path):
    for mpu, npz in frames.items():
        out = tmp_path / f"m{int(mpu)}.pt"
        train_sdf(npz, str(out), **RUN)
        meta = _meta(out)
        S = meta["scale"]
        assert meta["vmax"] / S == pytest.approx(18.0)  # metres per second (not 270)
        assert meta["d_hat"] / S == pytest.approx(7.0)  # metres (not 105)
        assert meta["rr"] / S == pytest.approx(3.0)  # metres
        assert meta["metres_per_unit"] == pytest.approx(20.0)
        assert meta["sdf_scale"] == pytest.approx(1.0 / mpu)
        assert meta["objective"]["kind"] == "task"


def test_metres_per_unit_0_is_the_npz_s_own_frame(frames, tmp_path):
    """The trainer before: on Austin's frame its robot drives at 270 m/s."""
    out = tmp_path / "old.pt"
    train_sdf(frames[300.0], str(out), metres_per_unit=0, objective="end-distance", **RUN)
    meta = _meta(out)
    assert meta["scale"] == pytest.approx(1.0 / 300.0)
    assert meta["vmax"] / meta["scale"] == pytest.approx(270.0)


def test_a_scene_read_in_another_frame_is_the_same_geometry_in_metres(frames):
    a = sdf_eval.load_scene(frames[300.0], scale=1.0 / 20.0)
    b = sdf_eval.load_scene(frames[20.0])
    assert a.scale == pytest.approx(b.scale) and a.task is not None and b.task is None
    assert sdf_eval.load_scene(frames[20.0], scale=1.0 / 20.0).task is None  # already there
    pts_m = torch.from_numpy(
        np.random.default_rng(1).uniform(-120, 120, (256, 2)).astype(np.float32)
    )
    pa, na = a.field.sample(pts_m * a.scale)
    pb, nb = b.field.sample(pts_m * b.scale)
    assert torch.allclose(pa / a.scale, pb / b.scale, atol=1e-3)  # clearance in metres
    assert torch.allclose(na, nb, atol=1e-4)
    # Held-out pairs are drawn in the npz's frame and converted: the same places in any frame.
    sa, ga = sdf_eval.heldout_pairs(a, 4, [1, 6, 11], 64, 7, rr=3.0 * a.scale)
    st, gt = sdf_eval.heldout_pairs(a.task, 4, [1, 6, 11], 64, 7, rr=3.0 * a.task.scale)
    k = a.scale / a.task.scale
    assert torch.equal(sa, st * k) and torch.equal(ga, gt * k)


def _constant(alpha=1.0, beta=3.0, gamma=4.0):
    target = torch.tensor([alpha, beta, gamma])
    m = sdf_nav.CoefMLP()
    with torch.no_grad():
        m.net[-1].weight.zero_()
        m.net[-1].bias.copy_(torch.log(torch.expm1(target)) - torch.log(torch.expm1(m.bias)))
    return m.eval()


def _terms(scene, kind, model):
    S = scene.scale  # the physics in metres, converted to this frame
    starts, goals = sdf_eval.heldout_pairs(scene, 4, [1, 6, 11], 128, 3, rr=3.0 * S)
    return sdf_eval.objective_terms(
        model, scene, starts, goals, rr=3.0 * S, d_hat=7.0 * S, dt=0.06, vmax=18.0 * S,
        horizon=28, window=7, col_grad=True, kind=kind,
    )  # fmt: skip


def test_the_task_objective_is_in_metres_and_the_old_one_is_not(frames):
    """With a constant policy the dynamics are the same in metres in any frame, so an objective
    in metres has one value; the old one's terms are in the frame's units."""
    m = _constant()
    own = sdf_eval.load_scene(frames[300.0])  # Austin's frame
    sim = sdf_eval.load_scene(frames[300.0], scale=1.0 / 20.0)
    a, b = _terms(own, "task", m), _terms(sim, "task", m)
    for t in ("goal", "col", "reg", "loss"):
        assert a[t] == pytest.approx(b[t], rel=1e-3, abs=1e-4), (t, a, b)
    old_a, old_b = _terms(own, "end-distance", m), _terms(sim, "end-distance", m)
    assert old_b["goal"] / old_a["goal"] == pytest.approx(15.0**2, rel=1e-2)  # units squared
    assert old_a["col"] != pytest.approx(old_b["col"], rel=0.1)  # a 0.02-unit margin: 6 m vs 0.4 m


def test_the_task_objective_rewards_arrival(tmp_path):
    """Its goal term is the share of the possible progress toward arrival not made: 0 for an
    agent that arrives, higher for one held back, and 0 for a pair that starts arrived."""
    occ = np.zeros((64, 64), bool)
    occ[:4, :4] = True  # one building, far from the leg
    bounds = (-96.0, -96.0, 96.0, 96.0)
    phi, nx, ny = sdf_nav.build_sdf(occ, bounds, 1.0 / 20.0)
    f = sdf_nav.SDFField(phi, nx, ny, bounds, (0.0, 0.0), 1.0 / 20.0)
    scene = sdf_eval.SdfScene(f, phi, list(bounds), [0.0, 0.0], 1.0 / 20.0, 90.0)
    S = scene.scale
    start = torch.tensor([[0.0, 0.0]])
    kw = dict(rr=3.0 * S, d_hat=7.0 * S, dt=0.06, vmax=18.0 * S, horizon=28, kind="task")
    goal = torch.tensor([[25.0, 0.0]]) * S  # 10 m to arrival (15 m), 1.7 s at up to 18 m/s
    fast = sdf_eval.objective_terms(_constant(), scene, start, goal, **kw)
    slow = sdf_eval.objective_terms(_constant(gamma=40.0), scene, start, goal, **kw)
    assert fast["goal"] == pytest.approx(0.0, abs=1e-6)
    assert slow["goal"] > 0.1
    there = torch.tensor([[10.0, 0.0]]) * S
    assert sdf_eval.objective_terms(_constant(), scene, start, there, **kw)["goal"] == 0.0


def test_the_pick_is_held_out_reach_without_penetration(frames, tmp_path):
    hist = tmp_path / "h.jsonl"
    train_sdf(
        frames[20.0], str(tmp_path / "p.pt"), history=str(hist), checkpoint_every=1,
        val_frac=0.25, val_blocks=4, val_pairs=16, val_ticks=30, **RUN,
    )  # fmt: skip
    recs = lh.read_history(hist)
    assert recs[0]["pick"] == "clean-reach"
    assert recs[0]["fitness"] == "heldout seek-drive reach without penetration"
    for c in (r for r in recs if r["kind"] == "checkpoint"):
        assert c["fitness"] == c["heldout"]["clean_reach"] <= c["heldout"]["reach"]


def test_a_leg_continues_across_steps_until_it_arrives_or_times_out(frames, tmp_path, monkeypatch):
    """``leg_ticks``: each batch row starts a step where it stopped the step before, at its
    speed then, and is redrawn (from rest) after ``leg_ticks`` ticks."""
    calls = []
    real = sdf_eval.loss_terms

    def spy(model, field, starts, goals, **kw):
        out = real(model, field, starts, goals, **kw)
        calls.append((starts.clone(), goals.clone(), kw.get("v0"), dict(kw.get("state") or {})))
        return out

    monkeypatch.setattr(sdf_eval, "loss_terms", spy)
    run = dict(RUN, steps=4, horizon=8)
    train_sdf(frames[300.0], str(tmp_path / "l.pt"), leg_ticks=16, **run)
    assert _meta(tmp_path / "l.pt")["leg_ticks"] == 16
    (s0, g0, v0, e0), (s1, g1, v1, e1), (s2, g2, v2, _), _ = calls
    assert torch.count_nonzero(v0) == 0  # a leg starts at rest
    go = (g0 - e0["o"]).norm(dim=-1) >= 15.0 * 0.05  # not arrived after one step
    assert torch.equal(s1[go], e0["o"][go]) and torch.equal(v1[go], e0["v"][go])
    assert torch.equal(g1[go], g0[go])
    assert not torch.equal(g2, g1)  # 16 ticks = 2 steps of 8: every row redrawn
    assert torch.count_nonzero(v2) == 0
    with pytest.raises(ValueError, match="task objective"):
        train_sdf(
            frames[300.0], str(tmp_path / "x.pt"), objective="end-distance", leg_ticks=8, **RUN
        )
