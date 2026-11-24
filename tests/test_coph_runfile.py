"""Run files (grl_snam.coph.runfile): a run packed so that it plays back without torch, as a web
page must. The pieces that stand in for torch (the numpy risk field, the floor's patches, the
lazy config) are checked against what they replace. These tests need no pycvc_gl."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from grl_snam.ariadne import coph
from grl_snam.coph import feed, runfile
from grl_snam.coph.evaluation import load
from grl_snam.coph.timeline import (
    CFG,
    Floor,
    Patch,
    Timeline,
    config_values,
    learning,
    risk,
)

STEPS = 50  # 6 s: agent 1 senses the middle door at 4.0 s; agent 0 hears of it at 4.8 s


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    """Both crossing arms at 50 steps, as episode files in grl-snam coph's layout, and the run
    file baked from them."""
    root = tmp_path_factory.mktemp("coph-runfile")
    where = str(feed.run_dir("crossing", root / "run", "severe"))
    for arm in feed.DEFAULT_ARMS["crossing"]:
        feed.run("crossing", arm, 76000, STEPS, out=where)
    path = coph.bake(root / "crossing.coph.npz", "crossing", feed.DEFAULT_ARMS["crossing"], 76000,
                     STEPS, "severe", root / "run")  # fmt: skip
    episodes = [load(coph.Source.episode("crossing", a, 76000, "severe", root / "run"))
                for a in feed.DEFAULT_ARMS["crossing"]]  # fmt: skip
    return dict(root=root, path=path, episodes=episodes)


def test_the_numpy_risk_is_models(baked):
    import torch

    from grl_snam.coph.model import risk as torch_risk
    from grl_snam.coph.model import tensor

    rng = np.random.default_rng(3)
    q = rng.uniform(-2.0, 6.0, (500, 2))
    centers, weights = rng.uniform(0.0, 4.0, (25, 2)), rng.normal(0.0, 2.0, 25)
    residual = dict(centers=rng.uniform(0.0, 4.0, (6, 2)).tolist(), radius=1.5,
                    coefficients=rng.normal(0.0, 0.5, 6).tolist())  # fmt: skip
    for res in (None, residual, dict(residual, coefficients=[])):
        with torch.no_grad():
            want = torch_risk(tensor(q), tensor(weights), tensor(centers), CFG.rbf_width, res)
        got = risk(q, weights, centers, CFG.rbf_width, res)
        assert np.allclose(got, want.numpy(), rtol=0, atol=1e-12)
    # And on every kept stage of the run, through the floor.
    tl = Timeline.from_episode("crossing", baked["episodes"][1])
    floor = Floor(tl.spec, feed.world("crossing", tl.spec), px_per_m=8)
    for s in tl.state(0, tl.duration).stages + tl.state(1, tl.duration).stages:
        with torch.no_grad():
            want = torch_risk(tensor(floor.points.reshape(-1, 2)), tensor(s["weights"]),
                              tensor(s["centers"]), CFG.rbf_width, s.get("local_residual"))  # fmt: skip
        got = risk(floor.points.reshape(-1, 2), s["weights"], s["centers"], CFG.rbf_width,
                   s.get("local_residual"))  # fmt: skip
        assert np.allclose(got, want.numpy(), rtol=0, atol=1e-12)


def _full(floor, patch):
    mask = np.zeros((floor.h, floor.w), bool)
    h, w = patch.mask.shape
    mask[patch.y0 : patch.y0 + h, patch.x0 : patch.x0 + w] = patch.mask
    return mask


def test_patches_are_the_whole_floor_masks_they_replace(baked):
    spec = baked["episodes"][0]["spec"]
    floor = Floor(spec, px_per_m=32)
    p, e = floor.points, 1.5 / floor.px
    for centre, radius in (((3.0, 4.0), 1.25), ((0.1, 0.2), 1.25), ((13.5, 9.0), 2.0)):
        d = np.linalg.norm(p - np.asarray(centre), axis=-1)
        want = np.abs(d - radius) <= e
        patch = floor.ring(centre, radius)
        assert isinstance(patch, Patch) and np.array_equal(_full(floor, patch), want)
    for x0, x1, y0, y1 in ((2.0, 6.0, 1.0, 5.0), (-1.0, 3.0, 7.0, 12.0)):
        inside = (p[..., 0] >= x0 - e) & (p[..., 0] <= x1 + e)
        inside &= (p[..., 1] >= y0 - e) & (p[..., 1] <= y1 + e)
        near = (np.abs(p[..., 0] - x0) <= e) | (np.abs(p[..., 0] - x1) <= e)
        near |= (np.abs(p[..., 1] - y0) <= e) | (np.abs(p[..., 1] - y1) <= e)
        patch = floor.outline((x0, x1, y0, y1))
        assert np.array_equal(_full(floor, patch), inside & near)
    # A patch draws what its whole-floor mask draws.
    overlays = [(floor.outline((2.0, 6.0, 1.0, 5.0)), (0.2, 0.4, 1.0), 0.55),
                (floor.ring((3.0, 4.0), 1.25), (1.0, 0.5, 0.0), 0.85)]  # fmt: skip
    masks = [(_full(floor, o), c, a) for o, c, a in overlays]
    assert np.array_equal(floor.image(None, overlays), floor.image(None, masks))


def test_the_run_file_holds_the_run(baked):
    run = runfile.read(baked["path"])
    assert (run.demo, run.seed, run.steps, run.scene) == ("crossing", 76000, STEPS, None)
    assert [a.arm for a in run.arms] == list(feed.DEFAULT_ARMS["crossing"])
    assert run.config == json.loads(json.dumps(config_values()))  # tuples stored as lists
    assert run.meta["schema"] == runfile.SCHEMA
    for arm, record, tl, floor in zip(run.arms, baked["episodes"], run.timelines(), run.floors()):
        want = Timeline.from_episode("crossing", record)
        for t in (0.0, 2.5, 4.0, 4.8, want.duration):
            for i in range(2):
                a, b = tl.state(i, t), want.state(i, t)
                assert (a.q, a.version, a.accepts, a.rejects, a.bytes_received) == (
                    b.q, b.version, b.accepts, b.rejects, b.bytes_received)  # fmt: skip
                assert a.stages == b.stages and a.ledger == b.ledger and a.known == b.known
        assert tl.deliveries == want.deliveries and tl.packets == want.packets
        world = Floor(record["spec"], feed.world("crossing", record["spec"]))
        assert np.array_equal(floor.corridor, world.corridor)
        assert np.allclose(floor.truth, world.truth, atol=1e-6)
        coarse = Floor(record["spec"], feed.world("crossing", record["spec"]), px_per_m=4)
        for i in range(2):
            fresh = learning(want, i, coarse)
            assert set(arm.learning[i]) == set(fresh)
            for k, v in fresh.items():
                assert np.array_equal(arm.learning[i][k], v, equal_nan=True), k
        if arm.plate is not None:  # drawn with Pillow when it is installed
            assert arm.plate.dtype == np.uint8 and arm.plate.ndim == 3


def test_a_run_file_refuses_another_run(baked):
    path = baked["path"]
    with pytest.raises(ValueError, match="is a crossing run"):
        coph.Source.from_runfile(path, "hazard")
    with pytest.raises(ValueError, match="seed 76000"):
        coph.Source.from_runfile(path, "crossing", seed=1)
    with pytest.raises(ValueError, match="holds the arms"):
        coph.Source.from_runfile(path, "crossing", arms=("disabled", "local_only"))
    np.savez(baked["root"] / "other.npz", x=np.zeros(2))
    with pytest.raises(ValueError, match="not a run file"):
        runfile.read(baked["root"] / "other.npz")


def test_a_run_file_plays_back_without_torch_or_pillow(baked):
    """What a page does with a run file, in a Python where torch and Pillow cannot import."""
    script = textwrap.dedent(f"""
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("torch", "PIL"):
                    raise ImportError(name + " is blocked")

        sys.meta_path.insert(0, Block())
        from grl_snam.ariadne import coph

        src = coph.Source.from_runfile({str(baked["path"])!r}, "crossing")
        tl = src.timelines[1]
        s = tl.state(0, tl.duration)
        field = src.floors[1].agent_risk(s.stages)
        assert (field[~(field != field)] > 0).all()
        image = src.floors[1].image(field, [(src.floors[1].ring(s.q, 1.25), (1, 0, 0), 1.0)])
        print(coph.result_line("crossing", src.timelines, 76000, "severe"))
        print(coph.gate_line(src.timelines))
        print(len(src.learning[1][0]["t"]), image.shape, coph.slack())
        print("torch" in sys.modules, "PIL" in sys.modules)
        """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         check=False)  # fmt: skip
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0].startswith("This run (seed 76000): CoPH both ways saves agent 0 0.2 m")
    assert "slack (0.003)" in lines[1]
    assert lines[2].endswith("0.003") and lines[3] == "False False"
