"""grl_snam.coph: the CLI, short-horizon smokes of both demos, and the two untanglings.

The full demos (the numbers gate in grl_snam/coph/VENDORED.md) take 20-30 minutes, so PR CI runs
them at a short horizon. The crossing smoke is pinned to what upstream's own code (CoPH-Sense
c1f1ff3) produces at the same horizon: at 50 steps the first door warnings already shorten the
receivers' paths, so the pins cover sensing, packets, assimilation and rerouting.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from grl_snam.cli import main
from grl_snam.coph.evaluation import load

# Upstream crossing_demo, seed 76000, --steps 50 (Windows, torch 2.8.0 CPU), per arm:
# - the two agents' path lengths (m);
# - every packet in send order as (sender, support points), each with 4 anchors and one source
#   per support point;
# - each agent's geometry events.
#
# Byte counts are not pinned. A packet's size is its compact JSON, and the first export
# bootstraps from a single time block, so its 8 ensemble members are identical and its
# covariance is round-off. Whether np.cov returns exact zeros ("0.0") or ~1e-35 residues
# (22 characters each) depends on numpy's summation order. The same 8 rows give 112 or 132 exact
# zeros, 380 bytes apart, and Linux CI measured one_way_01 at 80817-80834 bytes against Windows'
# 79488, with the paths equal to 1e-6.
CROSSING_50 = {
    "disabled": ([1.859716928807241, 1.6610842771994443], [], ([], ["locally_sensed"])),
    "one_way_01": (
        [1.859716928807241, 1.6474403222232064],
        [(0, 10), (0, 90), (0, 163), (0, 181), (0, 96), (0, 170), (0, 196)],
        ([], ["locally_sensed"]),
    ),
    "one_way_10": (
        [1.6794203471596572, 1.6610842771994443],
        [(1, 10), (1, 90), (1, 170), (1, 196), (1, 110), (1, 166), (1, 194)],
        (["received_geometry"], ["locally_sensed"]),
    ),
    "reciprocal_fixed": (
        [1.6794203471439542, 1.6474403222232064],
        [(0, 10), (1, 10), (0, 90), (1, 90), (0, 163), (1, 170), (0, 181)]
        + [(1, 196), (0, 91), (1, 110), (0, 165), (1, 166), (0, 196), (1, 194)],
        (["received_geometry"], ["locally_sensed"]),
    ),
}


def _run(args):
    result = CliRunner().invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def test_cli_registers_coph():
    assert {"crossing", "hazard"} <= set(main.commands["coph"].commands)
    assert "coph crossing|hazard" in _run(["--help"])


def test_crossing_smoke_matches_upstream(tmp_path):
    out = tmp_path / "crossing"
    args = ["--seeds", "1", "--steps", "50", "--workers", "4"]
    text = _run(["coph", "crossing", "--out", str(out), *args])
    assert "goals 0/8, collisions 0" in text
    r = json.loads((out / "results.json").read_text())
    assert (r["episodes"], r["missions"], r["collisions"], r["seeds"]) == (4, 8, 0, [76000])
    assert r["upstream"]["commit"].startswith("c1f1ff3")
    rows = {row["arm"]: row for row in r["rows"]}
    assert set(rows) == set(CROSSING_50)
    for arm, (paths, packets, geometry) in CROSSING_50.items():
        np.testing.assert_allclose([a["path"] for a in rows[arm]["agents"]], paths, rtol=1e-6)
        assert [a["finish_time"] for a in rows[arm]["agents"]] == pytest.approx([6.0, 6.0])
        e = load(out / "episodes" / f"76000_{arm}.json.gz")
        sent = [(int(p["sender"]), len(p["support_points"])) for p in e["packets"]]
        assert sent == packets
        assert all(len(p["anchors"]) == 4 for p in e["packets"])
        assert all(len(p["sources"]) == len(p["support_points"]) for p in e["packets"])
        delivered = [x for x in e["events"] if x["kind"] == "delivered"]
        assert len(delivered) == len(packets)
        assert all(x["reason"] == "eligible" and x["accepted"] for x in delivered)
        assert tuple([g["kind"] for g in es] for es in e["geometry_events"]) == geometry
        assert (rows[arm]["bytes"] > 0) == bool(packets)
    seed = r["per_seed"]["76000"]
    assert seed["savings"]["one_way_01"] == pytest.approx(0.0136440, abs=1e-5)
    assert seed["savings"]["one_way_10"] == pytest.approx(0.1802966, abs=1e-5)
    assert seed["reciprocal_saving"] == pytest.approx(0.1939405, abs=1e-5)
    # The manifest freezes the oracle's predictions before any episode runs.
    assert r["oracle"]["reciprocal_saving"] == pytest.approx(28.0, abs=0.05)
    assert (out / "REPORT.md").read_text().startswith("# crossing_traverses: results")

    # --summary-only rewrites the summaries from the episodes alone.
    (out / "results.json").unlink()
    _run(["coph", "crossing", "--out", str(out), "--summary-only"])
    assert json.loads((out / "results.json").read_text())["rows"] == r["rows"]


def test_hazard_smoke(tmp_path):
    out = tmp_path / "hazard"
    text = _run(["coph", "hazard", "--out", str(out), "--steps", "10", "--workers", "2"])
    assert "goals 0/64, collisions 0" in text
    r = json.loads((out / "results.json").read_text())
    assert r["scenes"] == ["mild", "moderate", "severe", "double_mild"]
    assert len(r["arms"]) == 8 and r["seeds"] == [79000] and r["steps"] == 10
    # Ten 0.12 s steps short of every hazard: time plus the carrier's (w=6) background risk 0.02.
    for cells in r["per_seed"]["79000"]["scenes"].values():
        for cell in cells.values():
            assert cell["carrier_cost"] == pytest.approx(10 * 0.12 * (1 + 4 * 6 * 0.02))
    assert list(r["worst_case_excess"].values()) == pytest.approx([0.0] * 8)
    assert "worst-case excess cost" in (out / "REPORT.md").read_text()


def test_route_field_does_not_load_the_cooperative_stack():
    # A fresh interpreter, so modules other tests imported do not mask the check.
    probe = (
        "import sys; from grl_snam.coph import route_field; "
        "loaded = sorted(m for m in sys.modules if m.startswith('grl_snam.coph')); "
        "print(loaded); print(route_field.values([[0.0, 0.0]], {}).tolist())"
    )
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, "-c", probe], cwd=root, capture_output=True, text=True, check=True
    )
    loaded, values = done.stdout.splitlines()
    assert "terrain_warning" not in loaded and "cooperative" not in loaded
    assert values == "[1.0]"


def test_cooperative_learner_carries_no_diagnostics():
    from grl_snam.coph.cooperative import CooperativeLearner
    from grl_snam.coph.diagnostics import Instrumented, InstrumentedLearner
    from grl_snam.coph.learner import Learner
    from grl_snam.coph.records import Config

    assert CooperativeLearner.__mro__[1] is Learner
    assert InstrumentedLearner not in CooperativeLearner.__mro__

    class Probe(Instrumented, CooperativeLearner):
        pass

    a = Probe(Config(), (0.0, 0.0, 0.0, 0.0), (4.0, 0.0), agent_id="0")
    assert a.replay_calls == [] and a.current_budget is None
    for mode in ("receiver_fit_mode", "receiver_validation_mode"):
        value = "precision_bounded" if mode == "receiver_fit_mode" else "relative_local"
        with pytest.raises(NotImplementedError):
            CooperativeLearner(
                Config(), (0.0, 0.0, 0.0, 0.0), (4.0, 0.0), agent_id="0", **{mode: value}
            )
