"""A position-driven third-person chase camera.

The live GRL-SNAM planner feeds a STREAM of agent positions — the camera must not
assume the path's shape (a circle, a spline, anything). ``ChaseCamera`` estimates
the travel direction from a *smoothed velocity* of the incoming positions and
critically damps the camera pose toward a trailing target, so the view eases
smoothly even when the raw direction is noisy or snaps around.

It is pure math — no rendering, no volrover3, no assumption about the path. Feed
it ``update(pos, dt)`` and read back an ``(eye, target, up)`` to drive any camera
(in volrover3, write those into the ``volrover3.camera`` state tree; see
``examples/volrover_lab_animated.py``).

Robustness for real input:
  * two-stage smoothing — a light EMA on position (kills jitter) feeds a heavier
    EMA on the velocity estimate (a stable heading);
  * frame-rate independent — every smoothing factor is ``1 - exp(-dt/tau)``, so
    irregular tick spacing doesn't change the feel;
  * stop-safe — below ``min_speed`` the last good heading is held, so the camera
    doesn't spin when the agent pauses or pivots in place.
"""

from __future__ import annotations

import math


def _ema(prev, new, dt, tau):
    """Exponential moving average toward ``new`` with time-constant ``tau`` (s)."""
    if prev is None:
        return list(new)
    a = 1.0 - math.exp(-dt / tau) if tau > 0.0 else 1.0
    return [prev[i] + (new[i] - prev[i]) * a for i in range(len(new))]


class ChaseCamera:
    """Third-person follow camera driven by a stream of world positions.

    Parameters (world units / seconds):
      back       trailing distance behind the agent
      height     camera height above the agent
      look_ahead aim this far ahead of the agent along its heading (0 = at it)
      look_up    aim this far above the agent's base (so it isn't at the feet)
      pos_tau    position-smoothing time constant (small; de-jitter)
      vel_tau    velocity/heading-smoothing time constant (larger; stable heading)
      cam_tau    camera-pose damping time constant (the "ease" of the follow)
      min_speed  below this horizontal speed the heading is frozen (stop-safe)
      up         world up axis (Z-up scenes: (0, 0, 1))
    """

    def __init__(
        self,
        back: float = 55.0,
        height: float = 40.0,
        look_ahead: float = 0.0,
        look_up: float = 3.0,
        pos_tau: float = 0.15,
        vel_tau: float = 0.40,
        cam_tau: float = 0.55,
        min_speed: float = 0.05,
        up=(0.0, 0.0, 1.0),
    ):
        self.back = float(back)
        self.height = float(height)
        self.look_ahead = float(look_ahead)
        self.look_up = float(look_up)
        self.pos_tau = float(pos_tau)
        self.vel_tau = float(vel_tau)
        self.cam_tau = float(cam_tau)
        self.min_speed = float(min_speed)
        self.up = tuple(float(c) for c in up)
        self.reset()

    def reset(self):
        """Forget all history (snaps to the next update's target pose)."""
        self._p = None  # smoothed position
        self._pprev = None
        self._v = None  # smoothed velocity (heading source)
        self._head = None  # last good horizontal unit heading (held while stopped)
        self._eye = None
        self._tgt = None

    def update(self, pos, dt: float):
        """Feed the agent's current world position ``(x, y, z)`` and the seconds
        since the last update; returns ``(eye, target, up)`` for the camera."""
        pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        dt = max(float(dt), 1e-4)

        # 1) light position smoothing (de-jitter without noticeable lag)
        self._p = _ema(self._p, pos, dt, self.pos_tau)

        # 2) velocity from the smoothed-position delta, then smooth the velocity
        if self._pprev is not None:
            raw_v = [(self._p[i] - self._pprev[i]) / dt for i in range(3)]
            self._v = _ema(self._v, raw_v, dt, self.vel_tau)
        self._pprev = list(self._p)

        # 3) heading = horizontal unit velocity; hold the last one if too slow
        if self._v is not None:
            speed = math.hypot(self._v[0], self._v[1])
            if speed >= self.min_speed:
                self._head = (self._v[0] / speed, self._v[1] / speed)
        hx, hy = self._head if self._head is not None else (1.0, 0.0)

        # 4) target pose: trail behind + above; look at (a touch ahead of) the agent
        p = self._p
        target_eye = [p[0] - hx * self.back, p[1] - hy * self.back, p[2] + self.height]
        target_look = [
            p[0] + hx * self.look_ahead,
            p[1] + hy * self.look_ahead,
            p[2] + self.look_up,
        ]

        # 5) critically damp the camera toward the target pose (smooth follow)
        self._eye = _ema(self._eye, target_eye, dt, self.cam_tau)
        self._tgt = _ema(self._tgt, target_look, dt, self.cam_tau)
        return tuple(self._eye), tuple(self._tgt), self.up
