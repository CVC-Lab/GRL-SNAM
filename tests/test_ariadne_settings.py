"""libcvc's navigator keys on GRL-SNAM's torch navigator (grl_snam.ariadne._cars).

The fleet and live ports take libcvc's multi-car template's settings (``nav.vehicle.*``,
``nav.reach_tol``, metres); on GRL-SNAM's torch navigator (normalized units) TorchSettings maps
the keys that mean the same thing. These need neither pycvc nor a GL context.
"""

from types import SimpleNamespace

import pytest

from grl_snam.ariadne import _cars


class FakeHost:
    """The two calls TorchSettings makes on a pycvc_gl.ariadne.Host: get and set, as text."""

    def __init__(self, **values):
        self.values = {k: str(v) for k, v in values.items()}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = str(value)


def bicycle(scale=0.05):
    return SimpleNamespace(
        _dyn="bicycle", S=scale, rr=0.15, reach_tol=0.35,
        kw={"rr": 0.15, "d_hat": 0.35, "dt": 0.06, "vmax": 0.9},
        _veh={"L": 0.035, "delta_max": 0.6, "a_max": 1.5, "a_lat_max": 1.0, "k_steer": 0.8,
              "allow_reverse": True},
    )  # fmt: skip


def test_a_point_navigator_takes_its_speed_footprint_barrier_and_arrival():
    point = SimpleNamespace(_dyn="point")
    assert _cars.torch_keys(point) == _cars.POINT_KEYS
    assert set(_cars.torch_keys(bicycle())) == set(_cars.TORCH_KEYS)


def test_seed_writes_the_navigators_values_in_metres():
    host = FakeHost()
    _cars.TorchSettings(host, [bicycle()]).seed()
    assert float(host.values["nav.vehicle.vmax"]) == pytest.approx(18.0)  # 0.9 / 0.05
    assert float(host.values["nav.vehicle.rr"]) == pytest.approx(3.0)
    assert float(host.values["nav.vehicle.d_hat"]) == pytest.approx(7.0)
    assert float(host.values["nav.reach_tol"]) == pytest.approx(7.0)
    assert float(host.values["nav.vehicle.wheelbase"]) == pytest.approx(0.7)
    assert float(host.values["nav.vehicle.delta_max"]) == pytest.approx(0.6)  # an angle as is
    assert host.values["nav.vehicle.allow_reverse"] == "1"


def test_the_trees_values_reach_every_navigator_in_its_units():
    navs = [bicycle(), bicycle()]
    host = FakeHost(**{"nav.vehicle.vmax": 4, "nav.vehicle.rr": 2.0, "nav.reach_tol": 10,
                       "nav.vehicle.allow_reverse": 0, "nav.vehicle.k_steer": 0.3})  # fmt: skip
    ts = _cars.TorchSettings(host, navs)
    ts.seed()  # set keys stay; the unset ones are seeded
    for nav in navs:
        assert nav.kw["vmax"] == pytest.approx(0.2)
        assert nav.kw["rr"] == pytest.approx(0.1) and nav.rr == pytest.approx(0.1)
        assert nav.reach_tol == pytest.approx(0.5)
        assert nav._veh["allow_reverse"] is False
        assert nav._veh["k_steer"] == pytest.approx(0.3)
        assert nav.kw["d_hat"] == pytest.approx(0.35)  # unset: seeded from the navigator, unchanged
    host.set("nav.vehicle.vmax", 10)  # a slider moved
    ts.apply()
    assert all(nav.kw["vmax"] == pytest.approx(0.5) for nav in navs)


def test_a_value_that_is_not_a_number_is_left_alone():
    nav = bicycle()
    host = FakeHost(**{"nav.vehicle.vmax": "fast"})
    _cars.TorchSettings(host, [nav], keys=("nav.vehicle.vmax",)).apply()
    assert nav.kw["vmax"] == pytest.approx(0.9)


def test_without_the_template_the_ports_say_what_to_install(monkeypatch, capsys):
    pytest.importorskip("pycvc_gl.ariadne", reason="the ports need pycvc_gl's Python host")
    from grl_snam.ariadne import fleet, live

    monkeypatch.setattr(_cars, "HAS_CARS", False)
    assert fleet.main(["--headless", "--frames", "1"]) == 2
    assert live.main(["austin-freedrive", "--headless", "--frames", "1"]) == 2
    err = capsys.readouterr().err
    assert err.count("pycvc-gl 3.7.12") == 2
