"""grl-snam ariadne coph's Learning window, against a stand-in for pycvc_gl's host.

It covers what the port publishes to the charts (``host.set_plot``, libcvc 3.7.14), the seek, the
redraw throttle, and that it stays quiet without ``set_plot``. No pycvc_gl is needed: the port's
:class:`~grl_snam.ariadne.coph.Learning` only calls ``set``, ``get*``, ``on`` and ``set_plot``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grl_snam.ariadne import coph
from grl_snam.coph import feed
from grl_snam.coph.timeline import Timeline

STEPS = 50  # agent 0 hears of the middle door from agent 1 within it (tests/test_coph_feed.py)


class FakeHost:
    """The host calls the Learning window makes, with the state as plain strings."""

    def __init__(self, plots=True):
        self.state, self.plots, self.handlers, self.draws = {}, {}, {}, 0
        if plots:
            self.set_plot = self._set_plot

    def set(self, key, value):
        self.state[key] = value

    def get(self, key, default=None):
        value = self.state.get(key)
        return default if value is None else str(value)

    def get_int(self, key, default=0):
        return int(self.state.get(key, default))

    def get_float(self, key, default=math.nan):
        return float(self.state.get(key, default))

    def get_bool(self, key, default=False):
        return str(self.state.get(key, default)).lower() in ("1", "true")

    def on(self, event, fn):
        self.handlers[event] = fn

    def _set_plot(self, key, series, *, markers=(), **_):
        self.plots[key] = (list(series), list(markers))
        self.draws += 1


@pytest.fixture(scope="module")
def timelines(tmp_path_factory):
    out = []
    for arm in ("disabled", "reciprocal_fixed"):
        messages = []
        feed.run("crossing", arm, 76000, STEPS, sink=messages.extend)
        spec = feed.job("crossing", arm, 76000, STEPS)[0]
        out.append(Timeline("crossing", spec, arm, label=coph.ARM_LABELS[arm]).ingest(messages))
    return out


def _learning(timelines, host=None):
    host = host or FakeHost()
    transport = coph.Transport(host, "title")
    return host, transport, coph.Learning(host, "crossing", timelines, transport)


def test_the_held_out_chart(timelines):
    host, _, panel = _learning(timelines)
    panel.publish(timelines, 3.0, finished=True)
    series, markers = host.plots["learn.heldout"]
    assert [s["label"] for s in series] == [
        "off before", "off after", "CoPH before", "CoPH after", "gate (CoPH)",
    ]  # fmt: skip
    for s in series:
        assert len(s["x"]) == len(s["y"]) == len(timelines[0 if "off" in s["label"] else 1]
                                                   .agents[0].updates)  # fmt: skip
    assert series[0]["style"] == "c0 points" and series[3]["style"] == "c1"
    np.testing.assert_allclose(series[4]["y"], series[2]["y"] + coph.SLACK)
    # Agent 0 hears of the middle door from agent 1 in the CoPH arm, and senses nothing with
    # communication off; nobody arrives within 50 steps.
    assert [m[1] for m in markers] == ["door (CoPH)"]
    assert host.state["learn.t"] == 3.0 and host.state["learn.log"] == "0"
    assert host.state["learn.caption"] == ("Agent 0 west. off: comms off; CoPH: CoPH both ways")


@pytest.mark.parametrize("show", list(coph.STATS))
def test_each_statistic(timelines, show):
    host, _, panel = _learning(timelines)
    host.state["learn.show"] = show
    panel.publish(timelines, 0.0, finished=True)
    series, _ = host.plots["learn.stat"]
    caption, log_y, keys = coph.STATS[show]
    assert host.state["learn.note"] == caption and host.state["learn.log"] == (
        "1" if log_y else "0"
    )
    arms = len(keys) * 2
    assert [s["label"] for s in series[:arms]] == [
        f"{arm} {suffix}" for arm in ("off", "CoPH") for _, suffix, _, _ in keys
    ]
    if show == "field":
        assert "lo" in series[0] and "hi" in series[0] and "lo" not in series[1]
    if show == "damping":
        assert series[-1]["label"] == "the world's" and series[-1]["y"] == [0.85, 0.85]
    else:
        assert len(series) == arms


def test_it_redraws_when_the_choice_or_the_data_changes(timelines):
    host, _, panel = _learning(timelines)
    panel.publish(timelines, 1.0, finished=True)
    assert host.draws == 2  # both charts
    panel.publish(timelines, 2.0, finished=True)
    assert host.draws == 2 and host.state["learn.t"] == 2.0  # the cursor moves, nothing redraws
    host.state["learn.agent"] = "1"
    panel.publish(timelines, 2.0)
    assert host.draws == 4 and host.state["learn.caption"].startswith("Agent 1 east.")
    host.state["learn.show"] = "nonsense"  # an unknown choice draws the default
    panel.publish(timelines, 2.0)
    assert host.draws == 4


def test_a_click_seeks(timelines):
    host, transport, _ = _learning(timelines)
    host.state["play.live"] = True
    host.state["learn.pick"] = 2.5
    host.handlers["learn_seek"]()
    assert transport.t == 2.5 and host.state["play.live"] is False


def test_without_set_plot_it_does_nothing(timelines):
    host, _, panel = _learning(timelines, FakeHost(plots=False))
    assert not panel.enabled and "learn_seek" not in host.handlers
    panel.publish(timelines, 1.0, finished=True)
    assert not any(k.startswith("learn.") for k in host.state)
