"""The play head of ``docs/player.ari``: recorded time, driven by the document's transport.

Every frame :meth:`Player.tick` advances the play head by the frame's dt times ``play.speed``
(while ``play.playing``), wraps it with ``play.loop`` or stops at the end, follows the position
slider when the user drags it, and calls ``draw(t)`` with the recorded time to show.
"""

from __future__ import annotations


class Player:
    def __init__(self, host, duration_s, draw, *, title=""):
        self.host = host
        self.duration = max(float(duration_s), 1e-6)
        self.draw = draw
        self.t = 0.0
        self._pos = 0.0  # the position this player last wrote
        host.set("play.title", title)
        host.on("restart", self.restart)

    def restart(self):
        self.t = 0.0
        self.host.set("play.playing", True)

    def tick(self, dt):
        host = self.host
        pos = host.get_float("play.pos", 0.0)
        if abs(pos - self._pos) > 1e-6:  # dragged: the slider wins
            self.t = min(max(pos, 0.0), 1.0) * self.duration
        elif host.get_bool("play.playing"):
            self.t += dt * max(host.get_float("play.speed", 1.0), 0.0)
            if self.t > self.duration:
                if host.get_bool("play.loop"):
                    self.t = 0.0
                else:
                    self.t = self.duration
                    host.set("play.playing", False)
        self._pos = self.t / self.duration
        host.set("play.pos", round(self._pos, 6))
        self._pos = host.get_float("play.pos", self._pos)  # as the state text reads back
        host.set("play.time", f"{self.t:6.1f} / {self.duration:.1f} s")
        self.draw(self.t)
