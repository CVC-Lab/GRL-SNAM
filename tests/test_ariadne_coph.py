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


def test_a_run_file_draws_what_its_episodes_drew(live, capsys, monkeypatch):
    """--bake packs the run (tests/test_coph_runfile.py); --from it replays it alike, and a page's
    ?demo= (GRL_SNAM_COPH_DEMO) fills in {demo}."""
    from grl_snam.ariadne import coph

    root = live["root"]
    path = root / "crossing.coph.npz"
    assert (
        _main(["crossing", "--from", str(root / "run"), "--steps", "50", "--bake", str(path)]) == 0
    )
    assert "wrote" in capsys.readouterr().out and path.is_file()
    assert _main(["crossing", "--headless", "--from", str(path),
                  "--log", str(root / "runfile.npz")]) == 0  # fmt: skip
    a, b = np.load(root / "live.npz"), np.load(root / "runfile.npz")
    for k in ("x", "y", "heading", "coef_work", "coef_version", "coef_bytes_received"):
        assert np.array_equal(a[k], b[k]), k
    meta = json.loads(str(b["meta"]))
    assert meta["origin"] == "run file" and meta["episodes"] == str(path)
    assert "saves agent 0 0.2 m" in capsys.readouterr().out
    monkeypatch.setattr(coph._host, "BROWSER", True)
    monkeypatch.setenv(coph.PAGE_DEMO, "hazard")
    template = str(root / "{demo}.coph.npz")
    assert _main(["crossing", "--headless", "--from", template]) == 2
    assert "hazard.coph.npz" in capsys.readouterr().err  # the page's demo, not the argument's


def test_refusals(tmp_path, capsys):
    assert _main(["crossing", "--headless", "--arms", "disabled"]) == 2
    assert _main(["crossing", "--headless", "--arms", "disabled,field"]) == 2
    assert _main(["hazard", "--headless", "--from", str(tmp_path)]) == 2
    assert _main(["crossing", "--offscreen", "--tour", "--arms", "disabled,one_way_10"]) == 2
    assert _main(["crossing", "--headless", "--tour", "--from", str(tmp_path)]) == 2
    assert _main(["crossing", "--from", str(tmp_path / "x.coph.npz"), "--bake", "y.npz"]) == 2
    err = capsys.readouterr().err
    assert "two different arms" in err and "no arm 'field'" in err and "is missing" in err
    assert "--tour follows the default arms" in err and "--tour draws" in err
    assert "--from a directory, not a run file" in err


def test_walkthrough_runs_crossing_then_hazard(tmp_path, monkeypatch):
    from grl_snam.ariadne import coph

    calls, relayed, made = [], [], []

    class Popen:
        def __init__(self, cmd, env=None):
            calls.append([str(c) for c in cmd])
            self.key = (env or {}).get(coph.RECORD_KEY)

    class Recorder:
        def __init__(self, path, ffmpeg=None):
            made.append((str(path), ffmpeg))

        def close(self):
            return 0

    monkeypatch.setattr(coph.subprocess, "Popen", Popen)
    monkeypatch.setattr(coph, "Recorder", Recorder)
    monkeypatch.setattr(coph, "relay", lambda listener, rec, proc: relayed.append(proc.key) or 0)
    out = tmp_path / "walk.mp4"
    args = ["walkthrough", "--from", str(tmp_path / "runs"), "--offscreen", "--width", "1920",
            "--height", "1080", "--record", str(out), "--ffmpeg", "ff"]  # fmt: skip
    assert _main(args) == 0
    crossing, hazard = calls
    assert crossing[3:7] == ["coph", "crossing", "--tour", "--from"]
    assert crossing[7] == str(tmp_path / "runs" / "crossing")
    assert hazard[4] == "hazard" and hazard[7] == str(tmp_path / "runs" / "hazard")
    assert ["--scene", "severe"] == hazard[hazard.index("--scene") :][:2]
    for cmd in (crossing, hazard):  # one video, recorded here: each tour pipes its frames
        assert {"--offscreen", "1920", "1080"} <= set(cmd) and "--record" not in cmd
        assert cmd.count("--record-to") == 1
    assert made == [(str(out), "ff")]
    assert len(relayed) == 2 and relayed[0] and relayed[0] == relayed[1]


def test_min_goals_sets_exit_code_3(live):
    root = live["root"]
    assert _main(["crossing", "--headless", "--from", str(root / "run"), "--min-goals", "1"]) == 3


@pytest.mark.skipif(
    not (_host.HAS_HOST and hasattr(_host.Host, "set_plot")),
    reason="needs pycvc_gl's chart: (libcvc 3.7.14, host.set_plot)",
)
def test_the_learning_charts(live, monkeypatch):
    hosts = []
    real = _host.open_host

    def open_host(*args, **kwargs):
        hosts.append(real(*args, **kwargs))
        return hosts[-1]

    monkeypatch.setattr(_host, "open_host", open_host)
    assert _main(["crossing", "--headless", "--from", str(live["root"] / "run")]) == 0
    host = hosts[0]
    heldout, stat = host.get_tree("learn.heldout"), host.get_tree("learn.stat")
    assert [s["label"] for s in heldout["s"]] == [
        "off before", "off after", "CoPH before", "CoPH after", "gate (CoPH)",
    ]  # fmt: skip
    assert [m["label"] for m in heldout["m"]] == ["door (CoPH)"]
    assert [s["label"] for s in stat["s"]] == [
        "off sensed", "off not sensed", "CoPH sensed", "CoPH not sensed",
    ]  # fmt: skip
    assert "lo" in stat["s"][0] and host.get("learn.show") == "field"
