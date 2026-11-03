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


def humvee_model(bundle=None, host=None):
    """A ``Humvee.glb`` as a ``pycvc_gl.vehicle.VehicleModel`` at its real length.

    It comes from the document's city when ``host`` is given and the city's package carries one
    (the small Austin package does). Otherwise it is the ``vehicle-humvee`` package's (beside the
    scene bundle, or in a prefix's ``share/cvc-scenes/shared``). None when there is none, and then
    :func:`add_vehicle` draws a box of its size."""
    try:
        from pycvc_gl.scenes import find_vehicle_model
        from pycvc_gl.vehicle import VehicleModel
    except ImportError:  # pragma: no cover - an older pycvc_gl
        return None
    path = city_file(host, "Humvee.glb") if host is not None else None
    path = path or find_vehicle_model(bundle)
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


# ── the document's city ────────────────────────────────────────────────────────────────────────
# A port's document names its city in its data: block (docs/live.ari): the Austin scene package on
# cvcpkg.org. The host's data store downloads it once (later runs read its cache), unpacks it into
# the state tree and decodes it; the document's scene draws it and the demo reads the same objects
# (document_city). libcvc's docs/ARIADNE.md 5.17 has the data: block.

REGISTRY = "https://cvcpkg.org"


def registry_origin():
    """The origin of the cvcpkg registry the data store reaches: ``CVCPKG_URL``'s, else
    https://cvcpkg.org."""
    from urllib.parse import urlsplit

    parts = urlsplit(os.environ.get("CVCPKG_URL") or REGISTRY)
    return f"{parts.scheme}://{parts.netloc}"


def load_document(host, doc, *, bundle=None, city=True):
    """Load ``doc`` (a file in ``docs/``), whose data: block names the city.

    With a local ``bundle`` (a directory, or ``GRL_SNAM_SCENE_BUNDLE``), ``data.city`` points
    there instead and nothing is downloaded. Otherwise the data store may reach the registry for
    the package. With ``city`` False, neither happens: the document's city never loads, and its
    node stays empty (a demo with a world of its own)."""
    import pycvc

    local = (bundle or os.environ.get("GRL_SNAM_SCENE_BUNDLE")) if city else None
    if city and not local and hasattr(pycvc, "data_allow_network"):
        pycvc.data_allow_network(host.app, [registry_origin()])
    host.load(str(DOCS / doc))
    if local and hasattr(pycvc, "data_load"):
        pycvc.data_load(
            host.app, str(local), decoder="city_bundle", name="city", prefix=host.prefix
        )


def _data_error(host, path):
    """Why the dataset at ``path`` (a document's link) failed, as its store node says."""
    import pycvc

    try:
        target = pycvc.State.instance(host.app).child(path).linkTarget() or path
        return pycvc.state_get(host.app, target + ".error")
    except RuntimeError:
        return pycvc.data_status(host.app, path) or "not loaded"


def document_city(host, name="city", timeout=600.0):
    """The city the document's data: block loaded, which its scene draws.

    Returns a :class:`~grl_snam.demos._common.City` over the same objects (the terrain sampler,
    its bounds, occupancy from the world), so no file is read again. Returns None when the
    document names no city, or when this pycvc has no data store. When the city cannot load
    (offline, no cache), the installed bundle stands in. Waits for the load."""
    import pycvc

    from grl_snam.demos._common import City

    if not hasattr(pycvc, "data_world"):
        return None
    path = f"{host.prefix}.data.{name}"
    if not pycvc.data_status(host.app, path):
        return None
    if not pycvc.data_wait(host.app, path, timeout=timeout):
        why = _data_error(host, path)
        local = bundle_or_none()
        if local is None:
            raise RuntimeError(f"the document's {name} did not load: {why}")
        print(f"grl_snam: the document's {name} did not load ({why}); using {local}", flush=True)
        pycvc.data_load(host.app, str(local), decoder="city_bundle", name=name, prefix=host.prefix)
        if not pycvc.data_wait(host.app, path, timeout=timeout):
            raise RuntimeError(f"{local} did not load as a city: {_data_error(host, path)}")
    return City.of_world(pycvc.data_world(host.app, path), pycvc.data_height_field(host.app, path))


_CITY_FILES = None  # a temporary directory for files a path-based reader needs (city_file)


def city_file(host, file, name="city"):
    """A file of the document's city's package (``Humvee.glb``) as a local path a path-based
    reader opens, or None when the package has none. Written once, to a directory removed when
    the process ends."""
    import tempfile

    import pycvc

    global _CITY_FILES
    if not hasattr(pycvc, "data_file"):
        return None
    path = f"{host.prefix}.data.{name}"
    if pycvc.data_status(host.app, path) != "ready":
        return None
    data = pycvc.data_file(host.app, path, file)
    if data is None:
        return None
    if _CITY_FILES is None:
        _CITY_FILES = tempfile.TemporaryDirectory(prefix="grl-snam-city-")
    out = os.path.join(_CITY_FILES.name, os.path.basename(file))
    with open(out, "wb") as f:
        f.write(data)
    return out


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
    """The demo-host interface :mod:`grl_snam.demos` expects (``make_lab``, ``city``, ``camera``,
    ``metrics``, ``run``), over an Ariadne Host: the demo's scene is the document's, its city the
    one the document's data: block loaded, its metrics the document's readouts, and its
    ``step(dt)`` runs every frame (scaled by the document's ``view.speed``, skipped while
    ``view.paused``)."""

    def __init__(self, host):
        self.host = host

    def make_lab(self):
        return self.host.lab

    def city(self):
        """The document's city (:func:`document_city`), which the scene already draws, or None:
        the demo then loads the bundle itself (``grl_snam.demos._common.open_city``)."""
        return document_city(self.host)

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
