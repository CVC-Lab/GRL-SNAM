"""grl-snam ariadne cars: libcvc's multi-car template (pycvc_gl.cars) run through GRL-SNAM.

The template is libcvc's (pycvc-gl 3.7.12+); with an older pycvc_gl the entry says what to install
and the runs here skip. --grl-policy needs torch and the published policy (grl-snam-weights).
"""

import json
import os
import sys

import numpy as np
import pytest

from grl_snam.ariadne import PORTS
from grl_snam.ariadne import cars as entry


def _template():
    try:
        from pycvc_gl import cars
    except ImportError:
        return None
    return cars if cars.has_cars() else None


needs_template = pytest.mark.skipif(
    _template() is None, reason="pycvc_gl has no cars template (pycvc-gl 3.7.12+)"
)


def test_the_port_is_the_entry():
    assert PORTS["cars"][0] == "grl_snam.ariadne.cars"


def test_without_the_template_it_says_what_to_install(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pycvc_gl.cars", None)
    assert entry.main(["--headless", "--frames", "1"]) == 2
    err = capsys.readouterr().err
    assert "pycvc-gl 3.7.12" in err and "CARS.md" in err


@needs_template
def test_a_headless_run_logs_every_tick(tmp_path):
    log = tmp_path / "run.npz"
    args = ["--headless", "--frames", "10", "--world", "flat", "--vehicle", "kinematic"]
    assert entry.main([*args, "--log", str(log)]) == 0
    z = np.load(log)
    meta = json.loads(str(z["meta"]))
    assert z["x"].shape == (10, 6) and meta["world"]["source"] == "flat"
    assert "nav.vehicle.vmax" in meta["settings"]


@needs_template
def test_copy_writes_the_template(tmp_path):
    out = tmp_path / "mine"
    assert entry.main(["--copy", str(out)]) == 0
    assert (out / "cars.py").is_file() and (out / "cars.ari").is_file()


def _policy_available():
    try:
        import torch  # noqa: F401

        from grl_snam.demos._common import default_nav_weights_pt
    except ImportError:
        return False
    p = default_nav_weights_pt()
    return bool(p) and os.path.exists(p)


@needs_template
@pytest.mark.skipif(not _policy_available(), reason="needs torch and grl-snam-weights")
def test_the_navigator_drives_with_grl_snams_policy(tmp_path):
    path, scale = entry.grl_policy()
    assert os.path.getsize(path) > 0 and 0.0 < scale < 1.0
    log = tmp_path / "nav.npz"
    args = ["--headless", "--frames", "10", "--world", "flat", "--vehicle", "kinematic",
            "--controller", "navigator", "--grl-policy", "--log", str(log)]  # fmt: skip
    assert entry.main(args) == 0
    meta = json.loads(str(np.load(log)["meta"]))
    assert meta["controller"] == "navigator" and meta["nav_weights"].endswith(".cvcnav")
    assert meta["nav_scale"] == pytest.approx(scale)
