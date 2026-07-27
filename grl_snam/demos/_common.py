"""Shared scaffolding for the live VolRover3 demos.

The demo modules in this package are loaded two ways: imported as part of the
``grl_snam`` package (so the CLI can find them), and exec'd as standalone files by
``volrover3 --run-job``. They therefore use absolute imports and defer every
host-specific import (``pycvc``, ``vrhost``, ``pycvc_gl``) to run time, so importing
the package never needs the compiled bindings or the running host.

Common pieces: a host guard, the vehicle box mesh, a camera driver that writes the
VolRover3 camera through the state tree, and a metrics publisher that pushes the
navigator's live :class:`~grl_snam.metrics.NavMetrics` into the state tree (the hook
a HUD panel reads) and prints a compact line.
"""

from __future__ import annotations

import math

from grl_snam.metrics import NavMetrics, hud_lines


def require_host():
    """Return ``(pycvc, vrhost)`` or raise a clear error outside VolRover3."""
    try:
        import pycvc
        import vrhost
    except ImportError as exc:  # pragma: no cover - only meaningful inside volrover3
        raise RuntimeError(
            "this demo must run INSIDE volrover3's embedded Python "
            "(Python Console -> Jobs tab -> Load Script -> Run as Job, or "
            "`volrover3 --run-job <this file>`)."
        ) from exc
    return pycvc, vrhost


def vehicle_box_mesh(length=4.6, width=2.0, height=1.6):
    """Flat ``(verts, tris)`` for a simple vehicle box centred at the origin, base at z=0."""
    hx, hy = length / 2, width / 2
    v = [
        -hx,
        -hy,
        0,
        hx,
        -hy,
        0,
        hx,
        hy,
        0,
        -hx,
        hy,
        0,
        -hx,
        -hy,
        height,
        hx,
        -hy,
        height,
        hx,
        hy,
        height,
        -hx,
        hy,
        height,
    ]
    t = [
        0,
        1,
        2,
        0,
        2,
        3,
        4,
        6,
        5,
        4,
        7,
        6,
        1,
        2,
        6,
        1,
        6,
        5,
        0,
        7,
        4,
        0,
        3,
        7,
        3,
        2,
        6,
        3,
        6,
        7,
        0,
        5,
        1,
        0,
        4,
        5,
    ]
    return v, t


class CameraDriver:
    """Write the VolRover3 camera each frame through the state tree (`volrover3.camera.*`)."""

    def __init__(self, app, pycvc, path="volrover3.camera", fov=60.0):
        self._app = app
        self._pycvc = pycvc
        self._path = path
        self._fov = fov

    def _set(self, key, val):
        self._pycvc.state_set(self._app, f"{self._path}.{key}", "%.6f" % float(val))

    def look(self, eye, target, up=(0.0, 0.0, 1.0)):
        vx, vy, vz = (target[i] - eye[i] for i in range(3))
        mm = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
        self._set("position.x", eye[0])
        self._set("position.y", eye[1])
        self._set("position.z", eye[2])
        self._set("view_direction.x", vx / mm)
        self._set("view_direction.y", vy / mm)
        self._set("view_direction.z", vz / mm)
        self._set("up_vector.x", up[0])
        self._set("up_vector.y", up[1])
        self._set("up_vector.z", up[2])
        self._set("fov", self._fov)


class MetricsPublisher:
    """Publish live :class:`NavMetrics` into the state tree (for a HUD panel) + console.

    Writes each numeric field to ``grl_snam.metrics.<field>`` so a VolRover3 HUD/overlay
    can subscribe, and prints a compact multi-line read-out every ``print_every`` steps.
    """

    def __init__(self, app, pycvc, base="grl_snam.metrics", print_every=45):
        self._app = app
        self._pycvc = pycvc
        self._base = base
        self._print_every = print_every
        self._n = 0
        self._prev = None

    def publish(self, m: NavMetrics, dt: float | None = None) -> None:
        # report the on-screen ground speed (frame displacement / dt), not the sim-time
        # speed, so the HUD matches what the viewer sees.
        if dt and dt > 0 and self._prev is not None:
            dx, dy = m.x - self._prev[0], m.y - self._prev[1]
            m.speed_mps = (dx * dx + dy * dy) ** 0.5 / dt
        self._prev = (m.x, m.y)
        self._n += 1
        for k, v in m.as_dict().items():
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, int | float):
                self._pycvc.state_set(self._app, f"{self._base}.{k}", "%.4f" % float(v))
            else:
                self._pycvc.state_set(self._app, f"{self._base}.{k}", str(v))
        if self._print_every and self._n % self._print_every == 1:
            print("  |  ".join(hud_lines(m)), flush=True)
