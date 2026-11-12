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

# Upstream crossing_demo, seed 76000, --steps 50: per arm, the two agents' path lengths (m) and
# the team's payload + status + ack bytes. Run on Windows, torch 2.8.0 CPU.
CROSSING_50 = {
    "disabled": ([1.859716928807241, 1.6610842771994443], 0),
    "one_way_01": ([1.859716928807241, 1.6474403222232064], 79488),
    "one_way_10": ([1.6794203471596572, 1.6610842771994443], 80999),
    "reciprocal_fixed": ([1.6794203471439542, 1.6474403222232064], 159977),
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
    for arm, (paths, team_bytes) in CROSSING_50.items():
        np.testing.assert_allclose([a["path"] for a in rows[arm]["agents"]], paths, rtol=1e-6)
        assert [a["finish_time"] for a in rows[arm]["agents"]] == pytest.approx([6.0, 6.0])
        # Packet sizes are compact JSON, so last-bit float differences across platforms can move
        # a few bytes; a lost or extra packet moves several percent.
        assert rows[arm]["bytes"] == pytest.approx(team_bytes, rel=0.01)
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
