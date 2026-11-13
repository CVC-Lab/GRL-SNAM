"""grl_snam.learning.curves: the held-out curve marks, against the golden other repos reuse."""

import json
from pathlib import Path

import numpy as np
import pytest

from grl_snam.learning import curves
from grl_snam.learning.history import Series
from grl_snam.tools import learning_marks_golden as golden

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "learning_marks_golden.json"


@pytest.fixture(scope="module")
def stored():
    return json.loads(FIXTURE.read_text())


def test_the_stored_golden_matches_a_fresh_one(stored):
    fresh = golden.golden()
    assert golden.compare(fresh, stored) == []
    # The curves themselves are stored exactly, so another implementation needs nothing from here.
    assert [c["y"] for c in fresh["cases"]] == [c["y"] for c in stored["cases"]]
    assert stored["params"] == curves.DEFAULT.to_dict()


def test_golden_cli_writes_and_checks(tmp_path, capsys):
    out = tmp_path / "g.json"
    assert golden.main(["--out", str(out)]) == 0
    assert golden.main(["--check", str(out)]) == 0
    bad = json.loads(out.read_text())
    bad["cases"][0]["expect"]["best"]["index"] += 5
    bad["cases"][1]["expect"]["elbow"] = None
    out.write_text(json.dumps(bad))
    assert golden.main(["--check", str(out)]) == 1
    assert "differs" in capsys.readouterr().out


def _case(stored, name):
    return next(c for c in stored["cases"] if c["name"] == name)


def test_the_marks_say_what_each_curve_was_built_to_show(stored):
    e = {c["name"]: c["expect"] for c in stored["cases"]}
    t = {c["name"]: c["truth"] for c in stored["cases"]}
    for name, case in e.items():
        assert (case["divergence"] is not None) == t[name]["diverges"], name
    # An overfitting curve: best near the noise-free minimum, divergence after it, no plateau.
    o = e["overfit_u"]
    assert abs(o["best"]["x"] - t["overfit_u"]["best_x"]) <= 30
    assert o["divergence"]["x"] > o["best"]["x"]
    assert o["plateau"] is None
    assert o["elbow"]["x"] < o["best"]["x"]
    # A decay that settles: elbow, then a plateau, no divergence.
    d = e["decay_to_floor"]
    assert d["elbow"]["x"] < d["plateau"]["x"] < d["best"]["x"]
    # A learner that learned nothing: no elbow; a straight descent: no elbow, no plateau.
    assert e["flat_noise"]["elbow"] is None
    assert e["still_falling"]["elbow"] is None and e["still_falling"]["plateau"] is None
    assert e["still_falling"]["best"]["x"] == t["still_falling"]["best_x"]
    # One bad evaluation does not move anything: same marks as the clean twin, within a step.
    assert e["spike_on_floor"]["divergence"] is None
    # The step's elbow is at the drop; a fitness curve's best is late and it plateaus.
    lo, hi = t["step_drop"]["drop_x"]
    assert lo <= e["step_drop"]["elbow"]["x"] <= hi
    f = e["fitness_saturating"]
    assert f["best"]["x"] > f["elbow"]["x"] and f["plateau"] is not None
    assert e["three_points"]["best"]["index"] == 0
    assert all(e["three_points"][k] is None for k in ("elbow", "plateau", "divergence"))


def test_noise_is_the_white_noise_sd_and_ignores_a_straight_trend():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.05, 4000)
    assert curves.noise(y) == pytest.approx(0.05, rel=0.08)
    assert curves.noise(y + np.linspace(0, 50, 4000)) == pytest.approx(curves.noise(y), rel=1e-6)
    assert curves.noise([1.0, 2.0]) == 0.0


def test_median_smooth_and_theil_sen():
    y = np.array([0.0, 0.0, 9.0, 0.0, 0.0, np.nan, 1.0])
    s = curves.median_smooth(y, 5)
    assert s[2] == 0.0  # the spike is gone
    assert np.isfinite(s).all()  # a NaN in a window is skipped
    assert curves.median_smooth([np.nan, np.nan], 3).tolist() != [0.0, 0.0]
    assert curves.theil_sen([0, 1, 2, 3, 4], [0, 2, 4, 6, 100]) == 2.0  # robust to the outlier
    assert curves.theil_sen([1, 1], [0, 5]) == 0.0


def test_marks_drop_nonfinite_points_and_handle_degenerate_curves():
    m = curves.marks([0, 1, 2, 3], [np.nan, np.nan, np.nan, np.nan])
    assert m.best is None and m.to_dict()["n"] == 0
    c = curves.marks(np.arange(30.0), np.full(30, 2.0))
    # A constant, noise-free curve: flat from the start, nothing else.
    assert c.best.index == 0 and c.elbow is None and c.divergence is None
    assert c.plateau.index == 0
    with pytest.raises(ValueError):
        curves.marks([0, 1], [0, 1], mode="up")


def test_max_mode_is_min_mode_of_the_negated_curve():
    x = np.arange(60.0)
    y = 0.2 + 0.8 * np.exp(-x / 8) + np.random.default_rng(3).normal(0, 0.01, 60)
    a = curves.marks(x, y, "min").to_dict()
    b = curves.marks(x, -y, "max").to_dict()
    for k in ("best", "elbow", "plateau", "divergence"):
        assert (a[k] is None) == (b[k] is None)
        if a[k] is not None:
            assert a[k]["index"] == b[k]["index"]


def test_series_marks_and_explicit_window():
    s = Series("epoch.heldout.traj", np.arange(40.0), 1.0 / (1.0 + np.arange(40.0)), "epoch")
    m = curves.series_marks(s)
    assert m.best.index == 39 and m.elbow is not None
    p = curves.MarkParams(window=8)
    assert p.window_for(1000) == 8 and curves.DEFAULT.window_for(12) == 5
    assert curves.DEFAULT.window_for(600) == 100
