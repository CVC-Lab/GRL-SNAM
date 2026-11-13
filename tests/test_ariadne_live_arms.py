"""``grl-snam ariadne live austin-freedrive --arms``: several policies side by side.

The first half needs no host: the arm specs, the policies the worlds load, the ledger's books,
the Stats rows, the plant ledger and the run object (grl_snam.ariadne._arms). The second runs the
port headless on the flat square (no download), on nav's kinematic plant, with three arms that
need no grl-snam-weights, and skips without pycvc_gl's Python host and its multi-car template
(pycvc-gl 3.7.12+), as tests/test_ariadne_coph.py does. It checks that the Stats rows tell the
arms apart, that one Retarget moves every arm's goal on the same tick, and that the torch path
refuses arms. With grl-snam-weights installed, the trained arm is checked against the port's
single-policy plant run.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from grl_snam.ariadne import _arms

# ── no host ────────────────────────────────────────────────────────────────────────────────────


def test_bare_arms_and_arms_eval_specs():
    arms = _arms.parse_arms("trained,untrained,constant")
    assert [(a.name, a.kind, a.slot) for a in arms] == [
        ("trained", "trained", 0), ("untrained", "untrained", 1), ("constant", "const", 2),
    ]  # fmt: skip
    arms = _arms.parse_arms(
        "a=run/best.cvcnav,c=const:1.2,3,4,b=run/ck.pt,d=const:alpha=1.2,gamma=5,untrained",
        ["e=const:bias"],
    )
    assert [(a.name, a.spec, a.kind) for a in arms] == [
        ("a", "run/best.cvcnav", "cvcnav"),
        ("c", "const:1.2,3,4", "const"),
        ("b", "run/ck.pt", "pt"),
        ("d", "const:alpha=1.2,gamma=5", "const"),
        ("untrained", "", "untrained"),
        ("e", "const:bias", "const"),
    ]


@pytest.mark.parametrize(
    "text, why",
    [
        ("trained,trianed", "no arm 'trianed'"),
        ("untrained,untrained", "named twice"),
        (",".join(f"a{i}=const:bias" for i in range(9)), "at most 8 arms"),
        ("", "names no arm"),
        ("x=", "NAME=PATH"),
    ],
)
def test_bad_arms_say_why(text, why):
    with pytest.raises(ValueError, match=why):
        _arms.parse_arms(text)


def test_the_books_close_and_book_a_goal_change_as_context():
    """dU = drive + projection + context + residual, the residual zero while nothing moves the
    agent between ticks; a goal moved between two ticks is context work."""
    first = {"potential": 5.0, "potential_end": 4.0, "projection_work": 0.0, "context_work": 0.0}
    b = _arms.books(first, None, 0.2, 0.0)
    assert b["dU"] == pytest.approx(-1.0) and b["drive"] == pytest.approx(-1.0)
    assert b["residual"] == 0.0 and b["dKE"] == pytest.approx(0.02)
    # The goal moved: U at the same pose rose from 4 to 6 before the drive took it to 5.5.
    moved = {"potential": 6.0, "potential_end": 5.5, "projection_work": 0.25, "context_work": 2.0}
    b = _arms.books(moved, 4.0, 0.2, 0.2)
    assert b["context"] == 2.0 and b["projection"] == 0.25 and b["dKE"] == 0.0
    assert b["drive"] == pytest.approx(-0.75) and b["dU"] == pytest.approx(1.5)
    assert b["residual"] == pytest.approx(0.0)
    # Something moved the agent between the ticks (U 0.5 higher than where the last ended) and
    # nothing booked it: the residual says so.
    b = _arms.books({**moved, "potential": 6.5, "potential_end": 6.0}, 4.0, 0.2, 0.2)
    assert b["residual"] == pytest.approx(0.5)


def test_a_stats_row():
    per = {"min_clearance_m": 1e29, "arrived": True, "time_to_goal_s": 7.5,
           "penetration_steps": 3, "total_path_m": 81.25}  # fmt: skip
    row = _arms.stats_row(per, [2.0, 1.0, 3.0, 1e29], [0.5, 1.5], ticks=200, dt=0.05)
    assert row == {"clear_min_m": 1.0, "clear_mean_m": 2.0, "pen_steps": 3, "path_m": 81.25,
                   "elapsed_s": 7.5, "ms_step": 1.0, "reached": True}  # fmt: skip
    row = _arms.stats_row({"min_clearance_m": -0.5}, [], [], ticks=40, dt=0.05)
    assert row["clear_min_m"] == -0.5 and math.isnan(row["clear_mean_m"])
    assert row["elapsed_s"] == pytest.approx(2.0) and not row["reached"]


def test_the_plant_ledger_shows_n_a_where_an_item_does_not_apply():
    rows = _arms.plant_ledger_rows({"total_path_m": 4.0})  # the kinematic plant feeds none
    assert [r["key"] for r in rows] == [k for k, _ in _arms.PLANT_LEDGER]
    assert {r["value"] for r in rows} == {"n/a"}
    rows = {r["key"]: r for r in _arms.plant_ledger_rows(
        {"energy_J": 1250.5, "rolled_over": False, "ltr_max": None, "vehicle_hits": 2})}  # fmt: skip
    assert rows["energy_J"]["value"] == "1250.500" and rows["energy_J"]["unit"] == "J"
    assert rows["rolled_over"]["value"] == "no" and rows["vehicle_hits"]["value"] == "2"
    assert rows["ltr_max"]["value"] == "n/a" and rows["tilt_s"]["value"] == "n/a"


def _tel(u0, u1, ctx=0.0, **kw):
    return {"potential": np.float32(u0), "potential_end": np.float32(u1), "context_work": ctx,
            "projection_work": 0.0, "clearance": 0.03, "alpha": 1.0, **kw}  # fmt: skip


def test_the_run_keeps_every_arm_and_its_legs(tmp_path):
    arms = _arms.parse_arms("one=const:bias,two=const:1,6,2")
    run = _arms.ArmsRun(arms, start=(0, 0), goal=(10, 0), seed=4, dt=0.05, scale=0.01)
    ends = [0.0, 0.0]
    for tick in range(5):
        if tick == 3:
            run.retarget((0, 10))
        for j in range(2):
            u0 = ends[j] + (1.0 if tick == 3 else 0.0)  # the new goal raises U: context work
            u1 = u0 - 0.1 * (j + 1)
            per = {"total_path_m": float(tick + j), "arrived": tick == 2 and j == 1}
            run.record(j, x=tick, y=j, heading=0.0, speed=2.0, goal=run.goal,
                       tel=_tel(u0, u1, ctx=u0 - ends[j] if tick else 0.0), ms=1.0,
                       nav_stats={"ticks": 1, "per_vehicle": [per]})  # fmt: skip
            ends[j] = u1
        run.end_tick()
    a = run.arrays()
    assert len(run) == 5 and a["x"].shape == (5, 2) and a["tel_alpha"].shape == (5, 2)
    assert a["goal_y"][:, 0].tolist() == [0, 0, 0, 10, 10] == a["goal_y"][:, 1].tolist()
    assert a["leg_tick"].tolist() == [0, 3] and a["leg_goal"].tolist() == [[10, 0], [0, 10]]
    assert np.abs(a["book_residual"]).max() < 1e-6
    assert a["book_context"][3].tolist() == pytest.approx([1.0, 1.0])
    assert run.cumulative(1, "drive")[-1] == pytest.approx(-1.0)
    assert run.arrivals == 1  # arm two reached leg 1's goal
    rows = run.stats_rows()
    assert [r["arm"] for r in rows] == ["one", "two"] and rows[1]["colour"] == "orange"
    assert rows[0]["clear_mean"] == "3.00"  # 0.03 normalized at 0.01 per metre
    assert [r["alpha"] for r in run.coefficient_rows()] == ["1.000", "1.000"]
    assert run.book_rows()[1]["drive"] == "-1.000" and run.book_rows(False)[1]["drive"] == "-0.200"
    assert {r["value"] for r in run.plant_rows(0)} == {"n/a"}
    path = run.save(tmp_path / "run.npz", note="x")
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    assert meta["schema"] == _arms.SCHEMA and meta["note"] == "x" and meta["seed"] == 4
    assert [m["name"] for m in meta["arms"]] == ["one", "two"] and len(meta["legs"]) == 2
    assert len(meta["finished_legs"][1]) == 1 and z["stats_path_m"].shape == (5, 2)


class _World:
    """A 21 x 21 m square with a 4 x 4 m block in the middle, as pycvc_gl.cars' World has it."""

    kind, cell, span = "test", 1.0, 20.0
    bounds = (0.0, 0.0, 20.0, 20.0)

    def __init__(self):
        Y, X = np.mgrid[0:21, 0:21].astype(float)
        dx = np.maximum(np.maximum(8 - X, X - 12), 0.0)
        dy = np.maximum(np.maximum(8 - Y, Y - 12), 0.0)
        self.sdf = np.where((dx == 0) & (dy == 0), -1.0, np.hypot(dx, dy))

    def cells(self, x, y):
        return int(round(y)), int(round(x))

    def free_points(self, rng, n, clearance, min_sep):
        rr, cc = np.nonzero(self.sdf >= clearance)
        k = rng.integers(len(rr), size=n)
        return np.stack([cc[k], rr[k]], 1).astype(float)


def test_where_a_retarget_may_go():
    w = _World()
    assert _arms.check_goal(w, 10, 10, 1.0) == "inside a building"
    assert _arms.check_goal(w, 13, 10, 2.0).startswith("too tight")
    assert _arms.check_goal(w, 30, 10, 1.0) == "off the world"
    assert _arms.check_goal(w, 2, 2, 1.0) == ""
    assert _arms.parse_point(" 3.5, -2") == (3.5, -2.0) and _arms.parse_point("3") is None
    assert _arms.parse_point("a,b") is None and _arms.parse_point("") is None
    goal = _arms.draw_goal(w, np.random.default_rng(0), [(1.0, 1.0), (2.0, 1.0)], 1.5)
    assert w.sdf[int(goal[1]), int(goal[0])] >= 1.5
    assert math.hypot(goal[0] - 1.0, goal[1] - 1.0) >= 6.0  # 30% of the span away


def test_resolve_writes_each_arms_policy(tmp_path):
    torch = pytest.importorskip("torch")
    import grl_snam.sdf_nav as sdf_nav
    from grl_snam.tools.coef_export import read_coef_mlp, write_coef_mlp

    torch.manual_seed(11)
    net = sdf_nav.CoefMLP()
    write_coef_mlp(net, str(tmp_path / "net.cvcnav"))
    torch.save({"model_state_dict": net.state_dict(), "meta": {"scale": 0.01}}, tmp_path / "n.pt")
    spec = f"untrained,constant,blob={tmp_path / 'net.cvcnav'},ck={tmp_path / 'n.pt'}"
    arms = _arms.parse_arms(spec)
    scale = _arms.resolve(arms, tmp_path / "out", seed=5)
    assert scale == 0.01  # the .pt arm's frame
    assert all(os.path.isfile(a.path) and len(a.sha256) == 64 for a in arms)
    assert arms[2].path == str((tmp_path / "net.cvcnav").resolve())
    assert arms[2].sha256 == arms[2].source_sha256  # a .cvcnav arm runs as it is
    assert arms[3].sha256 == arms[2].sha256 != arms[3].source_sha256  # the same net, from a .pt
    assert arms[0].sha256 != arms[2].sha256
    again = _arms.parse_arms("untrained")
    _arms.resolve(again, tmp_path / "again", seed=5)
    assert again[0].sha256 == arms[0].sha256  # the seed is the untrained arm's init
    alpha, beta, gamma = read_coef_mlp(arms[1].path)(torch.randn(4, net.in_dim))[:3]
    for head, c in ((alpha, 1.0), (beta, 3.0), (gamma, 4.0)):  # const:bias, for any input
        assert torch.allclose(head, torch.full_like(head, c), atol=1e-5)
    with pytest.raises(RuntimeError, match="grl-snam-weights"):
        _arms.resolve(_arms.parse_arms("trained"), tmp_path / "t", trained_pt=str(tmp_path / "no"))


# ── the port, headless ─────────────────────────────────────────────────────────────────────────

ARMS = "untrained,constant,brisk=const:1,6,2"
#: The frame the test presses Retarget on, how long it runs, and where the Retarget sends every
#: arm. On open ground a policy shows near its goal and near walls: far from both, every arm
#: accelerates at the plant's limit straight at its goal.
PRESS, FRAMES, GOAL = 120, 450, "-20, -5"


def _host_ready() -> bool:
    try:
        from grl_snam.ariadne import _cars, _host
    except Exception:  # noqa: BLE001
        return False
    return _host.HAS_HOST and _cars.HAS_CARS


needs_host = pytest.mark.skipif(
    not _host_ready(), reason="needs pycvc_gl's Python host and pycvc_gl.cars (pycvc-gl 3.7.12+)"
)


def _main(args):
    from grl_snam.ariadne.__main__ import main

    return main(args)


@pytest.fixture(scope="module")
def arms_run(tmp_path_factory):
    """Three arms on the flat square for FRAMES frames, the Retarget pressed once, at PRESS."""
    pytest.importorskip("torch")
    from grl_snam.ariadne import _host

    root = tmp_path_factory.mktemp("live-arms")
    hosts = []
    real = _host.open_host

    def open_host(*args, **kwargs):
        host = real(*args, **kwargs)
        hosts.append(host)
        frames = [0]

        @host.on_frame
        def press(dt):
            frames[0] += 1
            if frames[0] == PRESS:
                host.emit("arms_retarget")

        return host

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_host, "open_host", open_host)
        code = _main(["live", "austin-freedrive", "--headless", "--frames", str(FRAMES),
                      "--world", "flat", "--plant", "kinematic", "--arms", ARMS, "--seed", "3",
                      "--set", f"arms.goal={GOAL}", "--log", str(root / "arms.npz")])  # fmt: skip
    z = np.load(root / "arms.npz")
    return dict(code=code, host=hosts[0], z=z, meta=json.loads(str(z["meta"])))


@needs_host
def test_the_stats_rows_tell_the_arms_apart(arms_run):
    assert arms_run["code"] == 0
    rows = arms_run["host"].get_tree("arms.stats")
    assert [r["arm"] for r in rows] == ["untrained", "constant", "brisk"]
    shown = [tuple(r[c] for c in ("clear_min", "clear_mean", "pen", "path", "elapsed"))
             for r in rows]  # fmt: skip
    assert len(set(shown)) == 3, shown
    z = arms_run["z"]
    last = np.stack([z[f"stats_{k}"][-1] for k in ("clear_mean_m", "path_m")], 1)
    assert len({tuple(r) for r in last.round(6)}) == 3
    # The ledger's rows: the constant arm's coefficients are its constants.
    coef = {r["arm"]: r for r in arms_run["host"].get_tree("ledger.coef")}
    assert (coef["constant"]["alpha"], coef["constant"]["beta"]) == ("1.000", "3.000")
    assert (coef["brisk"]["beta"], coef["brisk"]["gamma"]) == ("6.000", "2.000")
    assert arms_run["host"].get("ledger.plant_note").endswith("no item applies (a physics plant "
                                                             "feeds them)")  # fmt: skip


@needs_host
def test_one_retarget_moves_every_arms_goal_on_the_same_tick(arms_run):
    z, meta = arms_run["z"], arms_run["meta"]
    assert len(meta["legs"]) == 2
    tick = int(z["leg_tick"][1])
    assert 0 < tick < len(z["t"])
    gx, gy = z["goal_x"], z["goal_y"]
    for j in range(3):
        changed = np.flatnonzero((gx[:, j] != gx[0, j]) | (gy[:, j] != gy[0, j]))
        assert changed[0] == tick, (j, changed[:3])
    assert (gx == gx[:, :1]).all() and (gy == gy[:, :1]).all()  # every arm, every tick
    assert tuple(gx[tick]) == (-20.0,) * 3 and tuple(gy[tick]) == (-5.0,) * 3
    assert meta["legs"][1]["goal"] == [-20.0, -5.0]
    ctx = np.abs(z["book_context"])
    assert (ctx[tick] > 10 * np.median(ctx, axis=0)).all()  # the goal's move is context work
    # Each arm's collector began a new leg there.
    assert all(s["ticks"] == len(z["t"]) - tick for s in meta["nav_stats"])
    assert arms_run["host"].get("arms.status").startswith("leg 2: every arm to")


@needs_host
def test_the_books_close_on_the_kinematic_plant(arms_run):
    z = arms_run["z"]
    scale = np.abs(z["tel_potential"]).max()
    assert np.abs(z["book_residual"]).max() <= 1e-5 * scale
    total = z["book_drive"] + z["book_projection"] + z["book_context"] + z["book_residual"]
    assert np.allclose(total, z["book_dU"])


@needs_host
def test_the_log_names_every_arms_policy(arms_run):
    meta = arms_run["meta"]
    assert meta["schema"] == _arms.SCHEMA and meta["plant"] == "kinematic" and meta["seed"] == 3
    assert [a["name"] for a in meta["arms"]] == ["untrained", "constant", "brisk"]
    assert all(a["key"] == "sha256:" + a["sha256"] for a in meta["arms"])
    assert meta["settings"]["nav_plant.vehicle"] == "kinematic"


@needs_host
def test_the_charts(arms_run):
    """On a host with chart: (libcvc 3.7.14, host.set_plot): every arm's clearance, the followed
    agent's cumulative books, and one book for every arm, each arm in its colour slot."""
    host = arms_run["host"]
    if not hasattr(host, "set_plot"):
        pytest.skip("needs pycvc_gl's chart: (libcvc 3.7.14, host.set_plot)")
    clearance, cum, compare = (host.get_tree(k) for k in ("arms.clearance", "ledger.cum",
                                                          "ledger.compare"))  # fmt: skip
    assert [(s["label"], s["style"]) for s in clearance["s"]] == [
        ("untrained", "c0"), ("constant", "c1"), ("brisk", "c2"),
    ]  # fmt: skip
    assert [s["label"] for s in cum["s"]] == list(_arms.BOOKS)
    assert [s["label"] for s in compare["s"]] == ["untrained", "constant", "brisk"]
    assert [m["label"] for m in compare["m"]] == ["leg 2"] and host.get("ledger.book") == "dU"


@needs_host
def test_the_torch_path_refuses_arms(capsys):
    args = ["live", "austin-freedrive", "--headless", "--world", "flat", "--arms", ARMS]
    assert _main(args) == 2  # the default plant is the demo's own stepping, torch
    assert _main([*args, "--plant", "torch"]) == 2
    err = capsys.readouterr().err
    assert "module-level" in err and "_STANDALONE" in err and "--plant kinematic" in err


@needs_host
def test_arms_refusals(capsys):
    assert _main(["live", "austin-patrol", "--headless", "--arms", "constant"]) == 2
    assert _main(["live", "austin-freedrive", "--headless", "--world", "flat", "--plant",
                  "kinematic", "--arms", "constant,nope"]) == 2  # fmt: skip
    err = capsys.readouterr().err
    assert "austin-freedrive's" in err and "no arm 'nope'" in err


@needs_host
@pytest.mark.skipif(
    os.environ.get("GRL_SNAM_ARIADNE_GL", "0") in ("", "0"),
    reason="GRL_SNAM_ARIADNE_GL=1 runs the offscreen checks (needs a GL context)",
)
def test_the_ghosts_offscreen(tmp_path):
    png = tmp_path / "arms.png"
    assert _main(["live", "austin-freedrive", "--offscreen", "--frames", "90", "--world", "flat",
                  "--plant", "kinematic", "--arms", ARMS, "--set", "view.camera=orbit",
                  "--retarget-at", "1", "--png", str(png)]) == 0  # fmt: skip
    assert png.stat().st_size > 10_000


def _weights():
    from grl_snam.demos._common import default_nav_weights_pt

    p = default_nav_weights_pt()
    return bool(p) and os.path.exists(p)


@needs_host
def test_the_trained_arm_is_the_single_policy_plant_run(tmp_path):
    """The bare trained arm drives as ``--plant kinematic`` without arms does, tick for tick, and
    the untrained one does not (it parts near the goal, 10 s in)."""
    if not _weights():
        pytest.skip("no grl-snam-weights")
    common = ["live", "austin-freedrive", "--headless", "--frames", "450", "--world", "flat",
              "--plant", "kinematic"]  # fmt: skip
    assert _main([*common, "--log", str(tmp_path / "one.npz")]) == 0
    assert _main([*common, "--arms", "trained,untrained", "--log", str(tmp_path / "two.npz")]) == 0
    one, two = np.load(tmp_path / "one.npz"), np.load(tmp_path / "two.npz")
    n = min(len(one["x"]), len(two["x"]))
    assert n > 100
    assert np.array_equal(one["x"][:n, 0], two["x"][:n, 0])
    assert np.array_equal(one["y"][:n, 0], two["y"][:n, 0])
    assert not np.array_equal(one["x"][:n, 0], two["x"][:n, 1])
