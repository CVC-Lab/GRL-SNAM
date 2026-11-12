"""grl-snam ariadne coph: the port runs both arms live, keeps and replays their episodes, and
writes the log. Needs pycvc_gl's Python host; the timeline and the feed are tested without it
(tests/test_coph_feed.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from grl_snam.ariadne import _host

pytestmark = pytest.mark.skipif(
    not _host.HAS_HOST, reason="needs pycvc_gl's Python host (pycvc_gl.ariadne)"
)


def _main(args):
    from grl_snam.ariadne import coph

    return coph.main(args)


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    root = tmp_path_factory.mktemp("ariadne-coph")
    args = ["crossing", "--headless", "--steps", "50", "--out", str(root / "run")]
    code = _main([*args, "--log", str(root / "live.npz"), "--min-goals", "0"])
    return dict(root=root, code=code)


def test_live_run_keeps_its_episodes_and_logs(live):
    root = live["root"]
    assert live["code"] == 0
    assert {p.name for p in (root / "run" / "episodes").iterdir()} == {
        "76000_disabled.json.gz",
        "76000_reciprocal_fixed.json.gz",
    }
    z = np.load(root / "live.npz")
    assert z["x"].shape == (51, 4) and z["t"][-1] == pytest.approx(6.0)
    meta = json.loads(str(z["meta"]))
    assert meta["arms"] == ["disabled", "reciprocal_fixed"] and meta["origin"] == "live, 50 steps"
    # Communication off sends nothing; CoPH both ways sends both ways.
    assert (z["coef_bytes_sent"][-1, :2] == 0).all() and (z["coef_bytes_sent"][-1, 2:] > 0).all()


def test_replay_draws_what_the_live_run_drew(live, capsys):
    root = live["root"]
    code = _main(["crossing", "--headless", "--from", str(root / "run"),
                  "--log", str(root / "replay.npz")])  # fmt: skip
    assert code == 0
    a, b = np.load(root / "live.npz"), np.load(root / "replay.npz")
    for k in ("x", "y", "heading", "coef_work", "coef_version", "coef_bytes_received"):
        assert np.array_equal(a[k], b[k]), k
    out = capsys.readouterr().out
    assert "saves agent 0 0.2 m" in out  # 1.8597 - 1.6794 m at 50 steps (tests/test_coph.py)


def test_refusals(tmp_path, capsys):
    assert _main(["crossing", "--headless", "--arms", "disabled"]) == 2
    assert _main(["crossing", "--headless", "--arms", "disabled,field"]) == 2
    assert _main(["hazard", "--headless", "--from", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "two different arms" in err and "no arm 'field'" in err and "is missing" in err


def test_min_goals_sets_exit_code_3(live):
    root = live["root"]
    assert _main(["crossing", "--headless", "--from", str(root / "run"), "--min-goals", "1"]) == 3
