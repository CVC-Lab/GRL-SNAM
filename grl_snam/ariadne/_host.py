"""What the Ariadne ports share: opening a ``pycvc_gl.ariadne.Host`` on a port's document, the
command line, the Humvee, the HUD readouts, and an Ariadne host for :mod:`grl_snam.demos`.

The ports are Python hosts of Ariadne (``.ari``) documents: the document declares the panels,
the look and the controls; the port steps or replays the simulation and writes what the panels
show. They need a pycvc_gl with the Python host (libcvc 3.7.9+); :data:`HAS_HOST` says whether
the installed one has it.
"""

from __future__ import annotations

import os
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"

try:  # pragma: no cover - depends on the installed bindings
    from pycvc_gl.ariadne import Host, main_args

    HAS_HOST = True
except Exception:  # noqa: BLE001 - an older pycvc_gl, or none
    Host = main_args = None
    HAS_HOST = False


def require_host():
    """``(Host, main_args)``, or a RuntimeError saying what to install."""
    if not HAS_HOST:
        raise RuntimeError(
            "the Ariadne ports need pycvc_gl with its Python host (pycvc_gl.ariadne, libcvc "
            "3.7.9+): cvcpkg install pycvc-gl-cp312 (or your Python's), or build libcvc's "
            "bindings."
        )
    return Host, main_args


def parse(argv, add_args=None, *, frames=0):
    """The ports' command line: the host's flags (``--offscreen | --headless``, ``--frames N``,
    ``--png PATH``, ``--width``, ``--height``) and the port's own, which ``add_args(parser)``
    declares. Returns one namespace."""
    import argparse

    _, main_args_ = require_host()
    host_ns = main_args_(argv, frames=frames)
    p = argparse.ArgumentParser(add_help=False)
    if add_args:
        add_args(p)
    ns = p.parse_args(host_ns.rest)
    for k, v in vars(host_ns).items():
        if k != "rest":
            setattr(ns, k, v)
    return ns


def open_host(doc, name, args, *, load=True, **kw):
    """A Host in ``args.mode`` with the port's document ``doc`` (a file name in ``docs/``)
    loaded; with ``load=False``, loaded later by ``host.load(str(DOCS / doc))``, once the
    functions its ``requires:`` names are registered."""
    Host_, _ = require_host()
    host = Host_(mode=args.mode, name=name, width=args.width, height=args.height, **kw)
    if load:
        host.load(str(DOCS / doc))
    return host


def finish(host, args) -> int:
    """Run the frames, write the PNG if asked, close; the document's exit code."""
    host.run(fps=30, frames=args.frames or None)
    if args.png and host.mode != "headless":
        host.snapshot(args.png)
    code = host.exit_code()
    host.close()
    return code


# ── the Humvee ─────────────────────────────────────────────────────────────────────────────────

HUMVEE_LEN_M = 4.72


def humvee_model(bundle=None):
    """The ``vehicle-humvee`` package's ``Humvee.glb`` (beside the scene bundle, or in a
    prefix's ``share/cvc-scenes/shared``) as a ``pycvc_gl.vehicle.VehicleModel`` at its real
    length, or None when there is none (then :func:`add_vehicle` draws a box of its size)."""
    try:
        from pycvc_gl.scenes import find_vehicle_model
        from pycvc_gl.vehicle import VehicleModel
    except ImportError:  # pragma: no cover - an older pycvc_gl
        return None
    path = find_vehicle_model(bundle)
    return VehicleModel(path, HUMVEE_LEN_M) if path else None


def humvee_box(length=4.84, width=2.18, height=1.9):
    """A box the Humvee's size (the physics preset's body), +X forward, base at z = 0."""
    hx, hy = length / 2, width / 2
    v = [-hx, -hy, 0, hx, -hy, 0, hx, hy, 0, -hx, hy, 0,
         -hx, -hy, height, hx, -hy, height, hx, hy, height, -hx, hy, height]  # fmt: skip
    t = [4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4,
         3, 4, 7]  # fmt: skip
    return [float(x) for x in v], t


def add_vehicle(host, name, model=None, color=(0.42, 0.45, 0.33)):
    """Add a Humvee scene node ``name``: the model when there is one, else a box of its size."""
    if host.scene.hasGraphics(name):
        host.scene.removeGraphics(name)
    if model is not None:
        return model.add(host.scene, name, color)
    v, t = humvee_box()
    host.lab.add_mesh(name, v, t, color=color)
    return host.scene.getGraphics(name)


# ── readouts ───────────────────────────────────────────────────────────────────────────────────


def publish_metrics(host, m, prefix="nav"):
    """A :class:`~grl_snam.metrics.NavMetrics` as the document's readouts: each field at
    ``<prefix>.<field>`` and the HUD's lines at ``<prefix>.hud`` (a list, one per line)."""
    from grl_snam.metrics import hud_lines

    values = {}
    for k, v in m.as_dict().items():
        if isinstance(v, bool):
            v = int(v)
        values[k] = f"{v:.3f}" if isinstance(v, float) else v
    host.set_many(values, prefix=prefix)
    host.set_tree(f"{prefix}.hud", hud_lines(m))


def bundle_or_none(bundle=None):
    """The scene bundle to use: ``bundle``, else the installed one (``default_scene_bundle``),
    when it has a terrain.json; else None."""
    from grl_snam.demos._common import default_scene_bundle

    b = bundle or default_scene_bundle()
    return b if b and os.path.isfile(os.path.join(str(b), "terrain.json")) else None


# ── an Ariadne host for grl_snam.demos ─────────────────────────────────────────────────────────


class _TrackCamera:
    """``look(eye, target, up)`` for a demo: the host's camera chases ``target`` (its Track
    mode, set up for a vehicle), unless the document's camera combo says otherwise."""

    def __init__(self, host):
        self.host = host

    def look(self, eye, target, up=(0.0, 0.0, 1.0)):
        if self.host.mode == "headless" or self.host.get("view.camera", "follow") != "follow":
            return
        import pycvc_gl

        self.host.camera.setMode(pycvc_gl.CameraController.Mode_Track)
        self.host.camera.feedTrackTarget(*(float(c) for c in target))


class _Metrics:
    def __init__(self, host):
        self.host, self._prev = host, None

    def publish(self, m, dt=None):
        if dt and dt > 0 and self._prev is not None:  # on-screen ground speed, as the demos do
            m.speed_mps = ((m.x - self._prev[0]) ** 2 + (m.y - self._prev[1]) ** 2) ** 0.5 / dt
        self._prev = (m.x, m.y)
        publish_metrics(self.host, m)


class DemoHost:
    """The demo-host interface :mod:`grl_snam.demos` expects (``make_lab``, ``camera``,
    ``metrics``, ``run``), over an Ariadne Host: the demo's scene is the document's, its
    metrics are the document's readouts, and its ``step(dt)`` runs every frame (scaled by the
    document's ``view.speed``, skipped while ``view.paused``)."""

    def __init__(self, host):
        self.host = host

    def make_lab(self):
        return self.host.lab

    def camera(self, fov=60.0):
        return _TrackCamera(self.host)

    def metrics(self):
        return _Metrics(self.host)

    def view(self):
        return self.host.view

    def run(self, step):
        host = self.host

        @host.on_frame
        def _frame(dt):
            if not host.get_bool("view.paused"):
                step(dt * max(host.get_float("view.speed", 1.0), 0.0))


class CameraSwitch:
    """The document's ``view.camera`` combo: ``follow`` chases a target (the camera's Track mode),
    anything else is the ``overview`` the port sets up (called once on each switch to it)."""

    def __init__(self, host, overview):
        self.host, self.overview, self.last = host, overview, None

    def update(self, target):
        host = self.host
        if host.mode == "headless":
            return
        import pycvc_gl

        mode = host.get("view.camera", "overview")
        if mode == "follow":
            host.camera.setMode(pycvc_gl.CameraController.Mode_Track)
            host.camera.feedTrackTarget(*(float(c) for c in target))
        elif self.last != mode:
            self.overview()
        self.last = mode


def track_camera(host, length=HUMVEE_LEN_M):
    """Set the host camera's Track (chase) mode up for a vehicle ``length`` metres long."""
    host.set_many(
        {
            "back": 2.5 * length,
            "height": 1.2 * length,
            "look_ahead": 1.2 * length,
            "look_up": 0.4 * length,
            "cam_tau": 0.15,
            "pos_tau": 0.1,
        },
        prefix="viewers.main.camera.track",
    )
