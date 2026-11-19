"""grl_snam.learning.runlog: the run log ``live --arms --log`` writes and ``replay`` reads, without a
host (a stand-in run with the arrays and meta grl_snam.ariadne._arms.ArmsRun gives)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from grl_snam.learning import runlog
from grl_snam.learning.runlog import PACKED_SCHEMA, SCHEMA, RunLog


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


def test_a_lean_run_log_is_smaller_and_reads_the_same(tmp_path):
    """A lean log leaves out the cars template's derived fields and stores single precision, its
    arrays packed into one (v2); read back, it has every field again, equal to single precision."""
    log = RunLog.from_run(_Run())
    path = log.save(tmp_path / "lean.npz", lean=True)
    with np.load(path) as z:
        assert sorted(z.files) == ["meta", "packed"] and z["packed"].dtype == np.uint8
        meta = json.loads(str(z["meta"]))
    assert meta["schema"] == PACKED_SCHEMA and meta["packed"]["encoding"] == "delta-shuffle"
    stored = {name for name, _, _ in meta["packed"]["arrays"]}
    assert not stored & set(runlog.LEAN_DROPS) and {"x", "tel_steer", "t"} <= stored
    back = RunLog.load(path)
    assert back.meta["lean"] is True and sorted(back.arrays) == sorted(log.arrays)
    assert back.meta["schema"] == PACKED_SCHEMA and "packed" not in back.meta
    assert back.arrays["x"].dtype == np.float32 and back.arrays["t"].dtype == np.float64
    for k, v in log.arrays.items():
        assert np.allclose(back.arrays[k], v, rtol=1e-6, atol=1e-6, equal_nan=True), k
    assert back.at(3, 1)["pose"] == pytest.approx(log.at(3, 1)["pose"])


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint8).tobytes()


def test_packing_is_exact_bit_for_bit():
    """Every value comes back with its bit pattern: NaNs with payloads, infinities, -0.0,
    subnormals, sign changes and wrapping differences; several arms, other dtypes, empty arrays."""
    rng = np.random.default_rng(0)
    f4 = rng.standard_normal((50, 3)).astype(np.float32) * 300
    f4[3, 1], f4[7, 0], f4[9, 2], f4[11, 1] = np.nan, np.inf, -np.inf, -0.0
    f4[13, 0] = np.float32(1e-45)  # subnormal
    f4[15, 2] = np.uint32(0x7FC0BEEF).view(np.float32)  # a NaN with a payload
    f4[16, 2] = np.uint32(0xFFFFFFFF).view(np.float32)  # the largest bit pattern after one smaller
    arrays = {
        "t": 0.05 * np.arange(1, 51),
        "x": f4,
        "y": rng.standard_normal(50).astype(np.float32),
        "leg_tick": np.array([0, 17, 40]),
        "leg_goal": np.array([[3.0, 4.0], [-5.0, 6.0], [7.5, -8.25]], np.float32),
        "flag": np.array([True, False, True]),
        "small": np.array([65535, 0, 1], np.uint16),
        "none": np.zeros((0, 3), np.float32),
        "one": np.float64(2.5),
    }
    packed, layout = runlog.pack(arrays)
    assert packed.dtype == np.uint8 and packed.ndim == 1
    assert packed.size == sum(np.asarray(v).nbytes for v in arrays.values())
    back = runlog.unpack(packed, json.loads(json.dumps(layout)))  # as meta stores it
    assert sorted(back) == sorted(arrays)
    for k, v in arrays.items():
        v = np.asarray(v)
        assert back[k].dtype == v.dtype and back[k].shape == v.shape, k
        assert _bits(back[k]) == _bits(v), k
        assert back[k].flags.c_contiguous, k


def test_a_lean_log_reads_back_bit_for_bit(tmp_path):
    """What a lean log keeps (its float32 casts, ``t``, the legs) is stored exactly."""
    log = RunLog.from_run(_Run())
    log.arrays["x"][1, 0] = np.nan
    back = RunLog.load(log.save(tmp_path / "lean.npz", lean=True))
    for k, v in log.arrays.items():
        kept = v.astype(np.float32) if v.dtype == np.float64 and k != "t" else v
        if k not in runlog.LEAN_DROPS:
            assert back.arrays[k].dtype == kept.dtype and _bits(back.arrays[k]) == _bits(kept), k


def _lean_v1(log, path):
    """A lean log as ``save(lean=True)`` wrote one before they were packed: grl_snam.run_log/v1,
    an array a member, single precision, the derived fields left out."""
    arrays = {k: (v.astype(np.float32) if v.dtype == np.float64 and k != "t" else v)
              for k, v in log.arrays.items() if k not in runlog.LEAN_DROPS}  # fmt: skip
    np.savez_compressed(path, meta=json.dumps({**log.meta, "lean": True}), **arrays)
    return path


def test_a_lean_log_from_before_packing_still_reads(tmp_path):
    log = RunLog.from_run(_Run())
    old = RunLog.load(_lean_v1(log, tmp_path / "v1.npz"))
    new = RunLog.load(log.save(tmp_path / "v2.npz", lean=True))
    assert old.meta["schema"] == SCHEMA and old.meta["lean"] is True
    assert sorted(old.arrays) == sorted(new.arrays) == sorted(log.arrays)
    for k in log.arrays:
        assert old.arrays[k].dtype == new.arrays[k].dtype, k
        assert _bits(old.arrays[k]) == _bits(new.arrays[k]), k
    # Saved lean again it is packed, and the same.
    again = RunLog.load(old.save(tmp_path / "again.npz", lean=True))
    assert again.meta["schema"] == PACKED_SCHEMA
    assert all(_bits(again.arrays[k]) == _bits(new.arrays[k]) for k in log.arrays)
    # A full save of a lean log read back is a v1 log, with no packing left in its meta.
    full = RunLog.load(new.save(tmp_path / "full.npz"))
    assert full.meta["schema"] == SCHEMA and "packed" not in full.meta


def test_load_refuses_a_packed_log_it_cannot_unpack(tmp_path):
    log = RunLog.from_run(_Run())
    with np.load(log.save(tmp_path / "lean.npz", lean=True)) as z:
        meta, packed = json.loads(str(z["meta"])), z["packed"]
    np.savez(tmp_path / "other.npz", meta=json.dumps({**meta, "packed": {"encoding": "zstd"}}),
             packed=packed)  # fmt: skip
    with pytest.raises(ValueError, match="not packed as 'delta-shuffle'"):
        RunLog.load(tmp_path / "other.npz")
    np.savez(tmp_path / "short.npz", meta=json.dumps(meta), packed=packed[:-3])
    with pytest.raises(ValueError, match="too few"):
        RunLog.load(tmp_path / "short.npz")
    np.savez(tmp_path / "long.npz", meta=json.dumps(meta), packed=np.append(packed, [1, 2]))
    with pytest.raises(ValueError, match="2 more than the layout"):
        RunLog.load(tmp_path / "long.npz")


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
    np.savez(tmp_path / "later.npz", meta=json.dumps({"schema": "grl_snam.run_log/v3"}))
    with pytest.raises(ValueError, match="run_log/v3"):
        RunLog.load(tmp_path / "later.npz")


def test_provenance_without_libcvc(monkeypatch):
    monkeypatch.delitem(sys.modules, "pycvc", raising=False)
    assert runlog.libcvc_version() is None
    prov = runlog.provenance()
    assert prov["libcvc"] is None and "history" not in prov
