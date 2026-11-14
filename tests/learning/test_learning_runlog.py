"""grl_snam.learning.runlog: the run log ``live --arms --log`` writes and ``replay`` reads, without a
host (a stand-in run with the arrays and meta grl_snam.ariadne._arms.ArmsRun gives)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from grl_snam.learning import runlog
from grl_snam.learning.runlog import SCHEMA, RunLog


class _Run:
    """Two arms, four ticks, a second leg from tick 2."""

    def arrays(self):
        T, A = 4, 2
        col = np.arange(T, dtype=float)[:, None] + np.array([0.0, 10.0])
        return {
            "t": 0.05 * np.arange(1, T + 1),
            "x": col, "y": 2 * col, "heading": np.zeros((T, A)), "speed": np.full((T, A), 2.0),
            "goal_x": np.full((T, A), 3.0), "goal_y": np.full((T, A), 4.0), "ms": np.ones((T, A)),
            "length": np.full((T, A), 4.84), "width": np.full((T, A), 2.18),
            "wheelbase": np.full((T, A), 3.3),
            "tel_alpha": np.full((T, A), 1.5), "tel_steer": np.full((T, A), 0.1),
            "book_dU": np.full((T, A), 0.5), "book_drive": np.full((T, A), -0.25),
            "stats_path_m": col, "stats_reached": np.array([[0, 0], [0, 1], [0, 1], [1, 1.0]]),
            "plant_energy_J": np.array([[np.nan, 1.0]] * T),
            "leg_tick": np.array([0, 2]), "leg_goal": np.array([[3.0, 4.0], [5.0, 6.0]]),
        }  # fmt: skip

    def meta(self):
        return {"schema": "replaced", "arms": [{"name": "a"}, {"name": "b"}], "dt": 0.05,
                "seed": 7}  # fmt: skip


def test_a_run_log_round_trips_and_reads_as_the_cars_templates(tmp_path):
    hist = tmp_path / "history.jsonl"
    hist.write_text('{"kind": "header"}\n')
    log = RunLog.from_run(_Run(), history=str(hist), note="x")
    assert log.meta["schema"] == SCHEMA and log.meta["note"] == "x" and log.meta["seed"] == 7
    prov = log.meta["provenance"]
    assert prov["grl_snam"] and prov["history"][0]["path"] == str(hist)
    assert len(prov["history"][0]["sha256"]) == 64
    a = log.arrays  # the template's fields, derived
    assert np.allclose(a["vx"], 2.0) and np.allclose(a["vy"], 0.0)
    assert np.allclose(a["steer"], 0.1) and np.allclose(a["coef_alpha"], 1.5)
    assert np.allclose(a["goal_dist"][0], [5.0, np.hypot(3 - 10, 4 - 20)])
    assert a["reached"][3].tolist() == [1, 1]
    back = RunLog.load(log.save(tmp_path / "run.npz"))
    assert back.meta == json.loads(json.dumps(log.meta)) and back.names == ["a", "b"]
    assert sorted(back.arrays) == sorted(log.arrays)
    assert all(np.array_equal(back.arrays[k], log.arrays[k], equal_nan=True) for k in log.arrays)


def test_reading_a_tick():
    log = RunLog.from_run(_Run())
    assert len(log) == 4 and log.dt == 0.05 and log.duration == pytest.approx(0.2)
    # The tick showing at t is the last that ended by then; the first before it.
    assert [log.index(t) for t in (0.0, 0.05, 0.07, 0.1, 0.2, 9.0)] == [0, 0, 0, 1, 3, 3]
    assert [log.leg(k) for k in range(4)] == [0, 0, 1, 1]
    assert log.fields("book_") == ["dU", "drive"]
    at = log.at(2, 1)
    assert at["pose"]["x"] == 12.0 and at["size"]["wheelbase"] == 3.3
    assert at["tel"] == {"alpha": 1.5, "steer": 0.1}
    assert at["book"] == {"dU": 0.5, "drive": -0.25}
    assert at["book_sum"] == {"dU": 1.5, "drive": -0.75}  # summed to the tick
    assert at["stats"]["reached"] == 1.0 and at["plant"] == {"energy_J": 1.0}
    assert log.at(2, 0)["plant"] == {"energy_J": None}  # NaN: it does not apply


def test_load_refuses_what_is_not_a_run_log(tmp_path):
    np.savez(tmp_path / "bare.npz", x=np.zeros(3))
    with pytest.raises(ValueError, match="no meta"):
        RunLog.load(tmp_path / "bare.npz")
    np.savez(tmp_path / "other.npz", meta=json.dumps({"schema": "cars"}))
    with pytest.raises(ValueError, match="schema 'cars'"):
        RunLog.load(tmp_path / "other.npz")


def test_provenance_without_libcvc(monkeypatch):
    monkeypatch.delitem(sys.modules, "pycvc", raising=False)
    assert runlog.libcvc_version() is None
    prov = runlog.provenance()
    assert prov["libcvc"] is None and "history" not in prov
