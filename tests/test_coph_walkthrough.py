"""grl-snam ariadne coph's walkthrough and claim labels, without pycvc_gl: the gate's count, the
camera's fit to the room the docked panels leave, the tour's steps from an episode, and the tour
playing over the transport (a stand-in host and view)."""

from __future__ import annotations

import math
import os
import shutil

import numpy as np
import pytest

from grl_snam.ariadne import coph
from grl_snam.coph import feed
from grl_snam.coph.timeline import Timeline

STEPS = 50  # 6 s: agent 1 senses the middle door at 4.0 s; agent 0 hears of it at 4.8 s


@pytest.fixture(scope="module")
def timelines():
    out = []
    for arm in feed.DEFAULT_ARMS["crossing"]:
        messages = []
        feed.run("crossing", arm, 76000, STEPS, sink=messages.extend)
        spec = feed.job("crossing", arm, 76000, STEPS)[0]
        out.append(Timeline("crossing", spec, arm, label=coph.label(arm)).ingest(messages))
    return out


def test_the_gate_kept_every_update(timelines):
    kept, rejected, worse, rise = coph.gate_counts(timelines)
    total = sum(len(a.updates) for tl in timelines for a in tl.agents)
    assert kept == total > 0 and rejected == 0
    assert 0 <= worse <= kept and 0.0 <= rise < coph.SLACK
    assert coph.gate_counts(timelines, t=0.0)[0] < kept  # counted up to the play head
    line = coph.gate_line(timelines)
    assert "can't reject" in line and f"{kept} kept, 0 rejected" in line
    assert coph.GATE_REFERENCE in line


def test_a_rejection_is_counted(timelines):
    off = timelines[0]
    agent = off.agents[0]
    u = agent.updates[-1]
    agent.updates.append((u[0], u[1] + 8, False, "validation_or_feasibility", 0.0, [], u[6], {}))
    try:
        assert coph.gate_counts(timelines)[1] == 1
    finally:
        agent.updates.pop()


def test_the_labels_name_the_reference_and_its_seed_counts():
    assert "2 seeds x 4 arms" in coph.REFERENCE["crossing"]
    assert "16/16 goals" in coph.REFERENCE["crossing"] and "27.1 m" in coph.REFERENCE["crossing"]
    assert "4 scenes x 8 arms" in coph.REFERENCE["hazard"]
    assert "4.6" in coph.REFERENCE["hazard"] and "68.1" in coph.REFERENCE["hazard"]
    for text in coph.UNESTABLISHED.values():
        assert "replay" in text and "field fitting" in text and "adaptive-rate" in text


def _project(pose, width, height, point):
    """Pixel (u, v) of ground ``point`` under the orbit ``pose`` (pinhole, v down), with numpy:
    a check on coph.project."""
    az, el = math.radians(pose["azimuth"]), math.radians(pose["elevation"])
    c = np.array([pose["center.x"], pose["center.y"], pose["center.z"]])
    eye = c + pose["distance"] * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    fwd = (c - eye) / np.linalg.norm(c - eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    p = np.array([point[0], point[1], 0.0]) - eye
    f = 0.5 * height / math.tan(math.radians(coph.VFOV) / 2)
    z = p @ fwd
    return 0.5 * width + f * (p @ right) / z, 0.5 * height - f * (p @ up) / z


@pytest.mark.parametrize("size", [(1600, 900), (1920, 1080), (1024, 768)])
def test_the_overview_fits_the_room_the_panels_leave(size):
    w, h = size
    work = h - coph.MENU_PX
    region = (8.0, (1 - coph.RIGHT) * w - 8.0, coph.MENU_PX + coph.TOP * work + 8.0,
              h - coph.BOTTOM * work - 8.0)  # fmt: skip
    box = (-1.0, 30.0, -1.0, 10.2)  # both floors and their nameplates
    pose = coph.fit(w, h, box, region)
    corners = [_project(pose, w, h, (x, y)) for x in box[:2] for y in box[2:]]
    for u, v in corners:
        assert region[0] - 1 <= u <= region[1] + 1 and region[2] - 1 <= v <= region[3] + 1
    u, v = _project(pose, w, h, (0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])))
    assert abs(u - 0.5 * (region[0] + region[1])) < 0.05 * w
    assert abs(v - 0.5 * (region[2] + region[3])) < 0.05 * h


def test_tour_steps_read_the_episode(timelines):
    steps = coph.tour_steps("crossing", timelines, 76000, "severe")
    assert len(steps) == 8
    assert "4.0 s" in steps[1].caption and "4.8 s" in steps[1].caption
    ends = [s.until for s in steps if s.until is not None]
    assert ends == sorted(ends)
    assert steps[-1].learn == (0, "field")


class _Host:
    """The Host calls Transport and Tour make, over a dict."""

    def __init__(self):
        self.state, self.handlers, self.closed = {}, {}, False

    def set(self, key, value):
        self.state[key] = ("1" if value else "0") if isinstance(value, bool) else str(value)

    def get(self, key, default=None):
        return self.state.get(key, default)

    def get_bool(self, key, default=False):
        return self.state.get(key, "0") in ("1", "true")

    def get_float(self, key, default=math.nan):
        try:
            return float(self.state[key])
        except (KeyError, ValueError):
            return default

    def on(self, name, fn):
        self.handlers[name] = fn

    def request_close(self):
        self.closed = True


class _View:
    pose = None

    def pose_for(self, view):
        return {"center.x": float(len(view)), "distance": 10.0, "azimuth": -90.0}

    def set_pose(self, pose):
        self.pose = dict(pose)


def test_the_tour_plays_every_step_and_hands_the_run_back(timelines):
    host, view = _Host(), _View()
    transport = coph.Transport(host, "title")
    steps = coph.tour_steps("crossing", timelines, 76000, "severe")
    tour = coph.Tour(host, view, transport, steps, demo="crossing", shots="unused",
                     close_at_end=True)  # fmt: skip
    head = max(tl.duration for tl in timelines)
    seen, shots = set(), 0
    for _ in range(20000):
        tour.frame(1.0 / coph.FPS, head)
        transport.tick(1.0 / coph.FPS, head, False)
        seen.add(host.get("tour.step"))
        if tour.want_shot:
            shots, tour.want_shot = shots + 1, None
        if host.closed:
            break
    assert host.closed and host.get("tour.step") == "End of the walkthrough"
    assert {f"Step {i} of 8" for i in range(1, 9)} <= seen and shots == 8
    assert transport.t == pytest.approx(head)  # the run is shorter than the steps' ends
    assert host.get("view.learning") == "1" and host.get("learn.agent") == "0"
    # Restart, then the presenter presses Playing: the tour hands the run back.
    host.handlers["tour_restart"]()
    assert transport.t == 0.0 and host.get("tour.step") == "Step 1 of 8"
    host.set("play.playing", True)
    assert tour.frame(1.0 / coph.FPS, head) is False and host.get("view.tour") == "0"
    host.handlers["tour_next"]()  # Next resumes it at step 2
    assert host.get("view.tour") == "1" and host.get("tour.step") == "Step 2 of 8"


def test_the_recorder_needs_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.delenv("FFMPEG", raising=False)
    monkeypatch.setattr(coph.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="needs ffmpeg"):
        coph.Recorder(tmp_path / "r.mp4")


@pytest.mark.skipif(not (os.environ.get("FFMPEG") or shutil.which("ffmpeg")), reason="needs ffmpeg")
def test_the_recorder_writes_an_mp4_of_odd_sized_frames(tmp_path):
    rec = coph.Recorder(tmp_path / "r.mp4")
    for i in range(10):
        rec.write(np.full((49, 65, 3), 20 * i, np.uint8))
    assert rec.close() == 10 and (tmp_path / "r.mp4").stat().st_size > 0


def test_two_tours_relay_their_frames_into_one_recorder(monkeypatch):
    """walkthrough's pipe: each tour's PipeRecorder (here a thread) sends its frames, odd-sized,
    and the relay hands them on to the one recorder, trimmed to even sizes, in order."""
    import threading
    from multiprocessing.connection import Listener

    key = os.urandom(32)
    monkeypatch.setenv(coph.RECORD_KEY, key.hex())

    class Got:
        def __init__(self):
            self.frames = []

        def write(self, rgb):
            self.frames.append((rgb.shape, int(rgb[0, 0, 0])))

    class Proc:  # the tour's process: running until its thread is done
        def __init__(self, thread):
            self.thread = thread

        def poll(self):
            return None if self.thread.is_alive() else 0

        def wait(self):
            self.thread.join()
            return 0

    def tour(address, start, n):
        rec = coph.PipeRecorder(address)
        for i in range(n):
            rec.write(np.full((49, 65, 3), start + i, np.uint8))
        assert rec.close() == n

    got = Got()
    with Listener(authkey=key) as listener:
        for start, n in ((0, 4), (100, 3)):
            t = threading.Thread(target=tour, args=(listener.address, start, n))
            t.start()
            assert coph.relay(listener, got, Proc(t)) == 0
    assert got.frames == [((48, 64, 3), v) for v in (0, 1, 2, 3, 100, 101, 102)]


def test_a_tour_that_dies_before_connecting_ends_the_relay():
    from multiprocessing.connection import Listener

    class Dead:
        def poll(self):
            return 3

        def wait(self):
            return 3

    with Listener(authkey=b"k") as listener:
        assert coph.relay(listener, None, Dead()) == 3


def test_the_pose_tween_zooms_evenly_and_turns_the_short_way():
    a = {"azimuth": 170.0, "distance": 10.0, "center.x": 0.0}
    b = {"azimuth": -170.0, "distance": 40.0, "center.x": 4.0}
    mid = coph._lerp_pose(a, b, 0.5)
    assert mid["azimuth"] == pytest.approx(180.0) and mid["distance"] == pytest.approx(20.0)
    assert mid["center.x"] == pytest.approx(2.0)
