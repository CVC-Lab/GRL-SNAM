"""Smoke tests for the Ariadne ports (grl_snam.ariadne).

They need pycvc_gl's Python host (pycvc_gl.ariadne, libcvc 3.7.9+) and skip without it. The
headless ones run in CI once a pycvc release carries it; a port that needs the Austin bundle skips
without one. GRL_SNAM_ARIADNE_GL=1 adds an offscreen run (needs a GL context).
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycvc_gl.ariadne", reason="pycvc_gl has no Python host (libcvc 3.7.9+)")

from grl_snam.ariadne import PORTS, _host  # noqa: E402
from grl_snam.ariadne.__main__ import main as ports_main  # noqa: E402

GL = os.environ.get("GRL_SNAM_ARIADNE_GL", "0") not in ("", "0")


def test_every_port_is_registered_with_its_document():
    import importlib

    for name, (module, _what, _original) in PORTS.items():
        assert hasattr(importlib.import_module(module), "main"), name
    for doc in ("live.ari", "player.ari", "fleet.ari", "material.ari"):
        assert (_host.DOCS / doc).is_file(), doc


def test_the_list(capsys):
    assert ports_main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in PORTS)
    assert ports_main(["nope"]) == 2


def test_the_cli_command():
    from click.testing import CliRunner

    from grl_snam.cli import main

    r = CliRunner().invoke(main, ["ariadne", "--list"])
    assert r.exit_code == 0 and "fleet" in r.output


def test_fog_records_then_replays(tmp_path):
    rc = ports_main(
        [
            "fog",
            "unit",
            "--headless",
            "--frames",
            "10",
            "--max-steps",
            "60",
            "--trace",
            str(tmp_path / "unit"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "unit" / "trace.json").exists()


def test_squad_replays_a_recording(tmp_path):
    """A small squad recording (no bundle needed headless), replayed by the squad player."""
    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.squad import AgentSpec
    from grl_snam.tools.squad_record import record_squad

    story = shrunk(STORIES["city"], n=48, max_steps=40)
    agents = [
        AgentSpec("v0", (-60.0, -40.0), (60.0, 40.0), (1.0, 0.5, 0.2)),
        AgentSpec("v1", (-60.0, 40.0), (60.0, -40.0), (0.2, 0.6, 1.0)),
    ]
    out = record_squad(story, agents, tmp_path / "run", max_steps=40)
    spec = json.loads((out / "squad.json").read_text())
    assert [a["key"] for a in spec["agents"]] == ["v0", "v1"]
    assert ports_main(["squad", "convoy", "--headless", "--frames", "10", "--out", str(out)]) == 0


def test_material_runs_both_stories():
    from grl_snam.ariadne.material import picture

    img = picture(
        np.zeros((8, 8), np.float32),
        np.zeros((8, 8), bool),
        [(np.array([[0.0, 0.0]]), (255, 0, 0))],
        (-1.0, -1.0, 1.0, 1.0),
        scale=2,
    )
    assert img.shape == (16, 16, 3) and img.dtype == np.uint8
    assert ports_main(["material", "--headless", "--frames", "2", "--swarm-steps", "20"]) == 0


def test_material_returns_when_a_run_is_stopped(monkeypatch, capsys):
    import time

    from grl_snam.ariadne import material
    from grl_snam.tools import material_demo as md

    def slow(*_a, **_k):
        time.sleep(1.0)
        raise AssertionError("the cap should have stopped this run's program first")

    monkeypatch.setattr(md, "scenario_runs", slow)
    monkeypatch.setattr(md, "swarm_runs", slow)
    monkeypatch.setattr(material, "RUN_LIMITS", {"action_max_time": 0.2, "action_max_steps": 0})
    t0 = time.monotonic()
    assert ports_main(["material", "--headless", "--frames", "2"]) == 1
    assert time.monotonic() - t0 < 30
    err = capsys.readouterr().err
    assert "the scenario run did not finish" in err and "the swarm run did not finish" in err


def test_fleet_runs_on_austin():
    if _host.bundle_or_none() is None:
        pytest.skip("no Austin scene bundle (scene-austin-south)")
    assert ports_main(["fleet", "--headless", "--frames", "10", "--agents", "2"]) == 0


def test_live_needs_a_scene():
    with pytest.raises(RuntimeError, match="draw into the scene"):
        ports_main(["live", "austin-patrol", "--headless", "--frames", "1"])


@pytest.mark.skipif(not GL, reason="GRL_SNAM_ARIADNE_GL=1 runs the offscreen checks")
def test_fog_offscreen(tmp_path):
    png = tmp_path / "fog.png"
    rc = ports_main(
        [
            "fog",
            "unit",
            "--offscreen",
            "--frames",
            "20",
            "--max-steps",
            "60",
            "--trace",
            str(tmp_path / "unit"),
            "--png",
            str(png),
        ]
    )
    assert rc == 0 and png.stat().st_size > 0
