"""train_coef_energy's opt-in held-out split, per-epoch history and best_heldout.pt.

``best.pt`` is picked on the TRAINING traj loss; ``--val-frac`` holds out whole episodes, scores a
fixed held-out set every epoch and keeps ``best_heldout.pt`` on its traj loss. ``--history`` writes
a record per epoch: the loss terms, the grad norm before the clip, and per head its held-out mean,
spread across episodes and ||dtheta||/||theta||. With neither, training is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest
import torch

import grl_snam.train_coef_energy as tce
from grl_snam import learning_history as lh


def _episode(root: Path, k: int, T: int = 14) -> str:
    """One synthetic stagewise episode: an agent driving past two discs toward a stage exit."""
    rng = random.Random(k)
    gx, gy = 4.0 + rng.uniform(-1, 1), 2.0 + rng.uniform(-1, 1)
    cks = []
    for t in range(T):
        f = t / (T - 1)
        cks.append(
            {
                "center": [f * gx + 0.1 * rng.uniform(-1, 1), f * gy],
                "stage_exit": [gx, gy],
                "obstacles_effective": {
                    "C": [[1.5 + 0.2 * k, 0.4], [2.5, 1.8 - 0.1 * k]],
                    "R_eff": [0.35, 0.3],
                    "W": [1.0, 1.0],
                },
                "barrier": {"barrier_d_hat": 0.5},
                "dt": 0.03,
            }
        )
    jl = root / f"ep{k}.jsonl"
    jl.write_text("\n".join(json.dumps(c) for c in cks) + "\n")
    pt = root / f"ep{k}.pt"
    torch.save(
        {"logs": {"checkpoints_jsonl": str(jl)}, "meta": {"dt": 0.03}, "params": {"gamma_o": 4.0}},
        pt,
    )
    return str(pt)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("stagewise")
    recs = [{"path": _episode(root, k)} for k in range(6)]
    (root / "manifest.json").write_text(json.dumps(recs))
    return root


def _argv(dataset: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--root", str(dataset), "--epochs", "3", "--bs", "8", "--workers", "0",
        "--outdir", str(out), *extra,
    ]  # fmt: skip


def _run(argv):
    random.seed(0)
    torch.manual_seed(0)
    tce.main(argv)


def _hashes(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.pt"))}


def test_history_alone_does_not_change_training(dataset, tmp_path):
    """--history (no held-out split) writes the same checkpoints as a plain run."""
    out = tmp_path / "ck"
    _run(_argv(dataset, out))
    plain = {k: torch.load(out / k, weights_only=False)["model_state_dict"] for k in _hashes(out)}
    hist = tmp_path / "h.jsonl"
    _run(_argv(dataset, tmp_path / "ck2", "--history", str(hist)))
    for k, sd in plain.items():
        other = torch.load(tmp_path / "ck2" / k, weights_only=False)["model_state_dict"]
        for name in sd:
            assert torch.equal(sd[name], other[name]), (k, name)
    recs = lh.read_history(hist)
    assert recs[0]["kind"] == "header" and recs[0]["heldout_samples"] == 0
    epochs = [r for r in recs if r["kind"] == "epoch"]
    assert [r["epoch"] for r in epochs] == [0, 1, 2]
    assert set(epochs[0]["grad_norm"]) == {"mean", "max", "clipped_frac"}
    assert set(epochs[0]["heads"]) == {"alpha", "beta", "gamma", "trunk"}
    assert epochs[-1]["heads"]["trunk"]["dtheta_rel_init"] > 0.0


def test_heldout_split_scores_whole_episodes_and_keeps_its_own_best(dataset, tmp_path):
    out, hist = tmp_path / "ck", tmp_path / "h.jsonl"
    _run(_argv(dataset, out, "--val-frac", "0.34", "--val-samples", "32", "--history", str(hist)))
    recs = lh.read_history(hist)
    header = recs[0]
    train_eps, held_eps = tce.split_episodes(6, 0.34, 0)
    assert header["heldout_episodes"] == held_eps and len(held_eps) == 2
    assert not set(train_eps) & set(held_eps)
    epochs = [r for r in recs if r["kind"] == "epoch"]
    for r in epochs:
        assert {"traj", "vel", "loss"} <= set(r["heldout"])
        for h in ("alpha", "beta", "gamma"):
            assert r["heads"][h]["n_scenes"] == 2
            assert {"mean", "spread", "within", "dtheta_rel"} <= set(r["heads"][h])
    best = min(epochs, key=lambda r: r["heldout"]["traj"])
    assert best["best_heldout"]
    ck = torch.load(out / "best_heldout.pt", weights_only=False)
    assert ck["epoch"] == best["epoch"]
    assert ck["heldout_logs"]["traj"] == pytest.approx(best["heldout"]["traj"])
    assert ck["heldout_episodes"] == held_eps
    assert (out / "best.pt").exists()  # the training-loss pick is still written


def test_the_heldout_set_is_fixed_and_leaves_the_training_rng_alone(dataset):
    _, held = tce.split_episodes(6, 0.34, 0)
    a, ea = tce.heldout_samples(str(dataset), held, 16, seed=3)
    b, eb = tce.heldout_samples(str(dataset), held, 16, seed=3)
    assert ea == eb and set(ea) <= set(held)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x["o0"], y["o0"]) and torch.equal(x["H"], y["H"])
    random.seed(5)
    before = random.getstate()
    tce.heldout_samples(str(dataset), held, 4, seed=3)
    assert random.getstate() == before


def test_the_default_dataset_draws_as_before(dataset):
    """No episodes / rng: the global random module, every manifest episode."""
    ds = tce.ShortRollouts(str(dataset))
    assert ds.rng is None and ds.item_episode == list(range(6))
    random.seed(11)
    first = ds[0]
    random.seed(11)
    item, _ep = ds.draw(random)
    assert torch.equal(first["o0"], item["o0"]) and torch.equal(first["H"], item["H"])


def test_split_needs_two_episodes_and_a_fraction():
    with pytest.raises(ValueError):
        tce.split_episodes(1, 0.5, 0)
    with pytest.raises(ValueError):
        tce.split_episodes(6, 0.0, 0)
    tr, he = tce.split_episodes(5, 0.99, 0)
    assert len(tr) == 1 and len(he) == 4


def _planted_episode(root: Path, k: int, T: int = 16) -> str:
    """An episode crossing x = 2 with the damping the generator logs (gamma_o_true): 2.5 before
    it, 6.0 after."""
    pt = _episode(root, k, T)
    jl = root / f"ep{k}.jsonl"
    rows = [json.loads(line) for line in jl.read_text().splitlines()]
    for r in rows:
        r["gamma_o_true"] = 2.5 if r["center"][0] < 2.0 else 6.0
    jl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return pt


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("planted")
    recs = [{"path": _planted_episode(root, k)} for k in range(6)]
    (root / "manifest.json").write_text(json.dumps(recs))
    spec = {
        "M_o": 1.5,
        "regions": [
            {"name": "lo", "x_lo": -1e9, "x_hi": 2.0, "gamma": 2.5},
            {"name": "hi", "x_lo": 2.0, "x_hi": 1e9, "gamma": 6.0},
        ],
    }
    (root / "planted.json").write_text(json.dumps(spec))
    return root, root / "planted.json"


def test_planted_logs_per_region_truth_mean_and_error_from_the_untrained_net_on(planted, tmp_path):
    root, spec = planted
    out, hist = tmp_path / "ck", tmp_path / "h.jsonl"
    argv = ["--val-frac", "0.34", "--val-samples", "48", "--history", str(hist)]
    _run(_argv(root, out, *argv, "--planted", str(spec), "--w_friction", "0"))
    recs = lh.read_history(hist)
    assert recs[0]["planted"]["M_o"] == 1.5
    assert [r["kind"] for r in recs[1:]] == ["init", "epoch", "epoch", "epoch"]
    for r in recs[1:]:
        p = r["heldout"]["planted"]
        assert p["lo"]["truth"] == pytest.approx(2.5 / 1.5)
        assert p["hi"]["truth"] == pytest.approx(4.0)
        for name in ("lo", "hi"):
            if p[name]["n"]:
                assert p[name]["rel_err"] == pytest.approx(
                    abs(p[name]["mean"] - p[name]["truth"]) / p[name]["truth"]
                )
                assert p[name]["truth_logged"] == pytest.approx(p[name]["truth"])
        assert p["ratio"]["of"] == ["hi", "lo"] and p["ratio"]["truth"] == pytest.approx(2.4)
    init = torch.load(out / "init.pt", weights_only=False)
    assert init["epoch"] == -1 and init["heldout_logs"]["kind"] == "init"


def test_planted_needs_a_held_out_split(planted, tmp_path):
    root, spec = planted
    with pytest.raises(SystemExit):
        tce.main(_argv(root, tmp_path / "ck", "--planted", str(spec)))


def test_planted_summary_counts_windows_inside_a_region_at_both_ends():
    spec = {
        "M_o": 2.0,
        "regions": [
            {"name": "a", "x_hi": 1.0, "gamma": 2.0},
            {"name": "b", "x_lo": 1.0, "gamma": 8.0},
        ],
    }

    def item(x0, x1, g):
        return {
            "o0": torch.tensor([x0, 0.0]),
            "o_tgt": torch.tensor([x1, 0.0]),
            "gamma_true": (g, g),
        }

    items = [item(0.0, 0.5, 2.0), item(0.5, 1.5, 2.0), item(2.0, 3.0, 8.0), item(1.2, 1.4, 8.0)]
    got = tce.planted_summary(spec, items, torch.tensor([1.5, 9.0, 3.0, 5.0]))
    assert got["a"] == {"truth": 1.0, "mean": 1.5, "n": 1, "rel_err": 0.5, "truth_logged": 1.0}
    assert (
        got["b"]["n"] == 2 and got["b"]["mean"] == pytest.approx(4.0) and got["b"]["truth"] == 4.0
    )
    assert got["ratio"]["value"] == pytest.approx(4.0 / 1.5)
    assert got["ratio"]["rel_err"] == pytest.approx(abs(4.0 / 1.5 - 4.0) / 4.0)
