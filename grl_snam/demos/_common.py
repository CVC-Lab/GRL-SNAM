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
import os
import sys

from grl_snam.metrics import NavMetrics, hud_lines


# ── cvcpkg-installed data (weights + scene bundle) ───────────────────────────────
# The demos ship no data of their own: the pretrained nav coefficients come from the
# `grl-snam-weights` cvcpkg bundle (share/grl-snam-weights/coef_sdf.pt) and the Austin
# geometry from `scene-austin-south` (share/cvc-scenes/austin_south). Resolve those
# from the active prefix so a developer who has `cvcpkg install`ed them can just run
# the demo — no training run, no hand-placed files — while keeping env overrides and a
# local-dev fallback.
def _candidate_prefixes():
    """Prefixes to search for installed share/ data, most specific first."""
    roots = []
    for env in ("GRL_SNAM_PREFIX", "CVC_PREFIX", "CVCPKG_PREFIX", "CONDA_PREFIX", "VIRTUAL_ENV"):
        v = os.environ.get(env)
        if v:
            roots.append(v)
    roots.append(sys.prefix)  # the (embedded) interpreter's own prefix
    # …and relative to an installed host package: pycvc/grl_snam live under
    # <prefix>/lib/pythonX.Y/site-packages, so an ancestor with a share/ sibling is the
    # prefix. Covers relocated cvcpkg installs whose sys.prefix is the system python.
    for mod in ("pycvc", "grl_snam"):
        try:
            f = getattr(__import__(mod), "__file__", None)
        except Exception:
            f = None
        if not f:
            continue
        p = os.path.abspath(f)
        for _ in range(6):
            p = os.path.dirname(p)
            if os.path.isdir(os.path.join(p, "share")):
                roots.append(p)
                break
    seen, out = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def resolve_installed(relpath, *, env=None, fallback=None):
    """Resolve a cvcpkg-installed data path (e.g. ``share/grl-snam-weights/coef_sdf.pt``).

    Order: explicit ``env`` override → ``<prefix>/relpath`` for each candidate prefix →
    ``fallback`` (a local-dev path). Returns the first existing path, else ``fallback``
    so the caller's own load error names the missing file.
    """
    if env:
        v = os.environ.get(env)
        if v:
            return v
    for root in _candidate_prefixes():
        p = os.path.join(root, relpath)
        if os.path.exists(p):
            return p
    return fallback


def default_nav_weights_pt():
    """Pretrained torch checkpoint for ``sdf_nav.CoefMLP`` — the installed
    ``grl-snam-weights`` bundle (``share/grl-snam-weights/coef_sdf.pt``), overridable
    with ``GRL_SNAM_CHECKPOINT``; falls back to a locally-trained
    ``checkpoints/coef_sdf.pt``."""
    return resolve_installed(
        "share/grl-snam-weights/coef_sdf.pt",
        env="GRL_SNAM_CHECKPOINT",
        fallback="checkpoints/coef_sdf.pt",
    )


def default_scene_bundle():
    """Austin South geometry — the installed ``scene-austin-south`` bundle
    (``share/cvc-scenes/austin_south``), overridable with ``GRL_SNAM_SCENE_BUNDLE``;
    falls back to ``~/scenes/austin_south``."""
    return resolve_installed(
        "share/cvc-scenes/austin_south",
        env="GRL_SNAM_SCENE_BUNDLE",
        fallback=os.path.expanduser("~/scenes/austin_south"),
    )


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


# ── demo host: embedded in VolRover3, or a standalone pycvc_gl window ─────────
# The Austin demos are host-agnostic setup()/step(dt) pairs. By default they run
# EMBEDDED in VolRover3 (adopt vrhost's live app+scene; camera+metrics flow
# through the state tree — the original path, unchanged). `run_standalone()`
# swaps in a host that owns its OWN pycvc_gl window and drives a live animation
# loop, so the demos run straight off cvcpkg with no VolRover3 at all.
#
# The standalone loop mirrors the cvc::nav NATIVE demos (nav_city_drive.cpp):
# a non-blocking `pycvc_gl.SceneRenderer` (render()/processUIEvents()/
# windowClosed()/setCamera()) driven by a caller-owned `while not
# windowClosed()` loop — NOT a blocking `show()` (which owns the loop and so
# cannot be stepped by the sim). Same primitive the native/wasm demos use.

_STANDALONE = None  # a _StandaloneHost while a standalone run is active, else None


class _Vr3Host:
    """Embedded in VolRover3: adopt the running app+scene; camera+metrics through
    the state tree. This is the original behaviour, unchanged."""

    def __init__(self):
        self._pycvc, self._vrhost = require_host()
        self._app = self._vrhost.app()

    def make_lab(self):
        from pycvc_gl.lab import Lab

        return Lab(app=self._app, scene=self._vrhost.scene())

    def camera(self, fov=60.0):
        return CameraDriver(self._app, self._pycvc, fov=fov)

    def metrics(self):
        return MetricsPublisher(self._app, self._pycvc)

    def run(self, step) -> None:
        # VolRover3 owns the render loop and calls step(dt) itself — nothing to do.
        pass


class _DirectCamera:
    """Standalone camera driver: aim the live pycvc_gl window's camera each frame
    (no state tree). Mirrors :class:`CameraDriver`'s ``look()`` so the demos call
    it identically — the difference is a direct ``SceneRenderer.setCamera`` (the
    native demos' scripted-camera path) instead of a state-tree write."""

    def __init__(self, host, fov=60.0):
        self._host = host
        self._fov = float(fov)

    def look(self, eye, target, up=(0.0, 0.0, 1.0)):
        view = self._host.view()
        if view is None:
            return
        view.setCamera(
            float(eye[0]),
            float(eye[1]),
            float(eye[2]),
            float(target[0]),
            float(target[1]),
            float(target[2]),
            float(up[0]),
            float(up[1]),
            float(up[2]),
            self._fov,
        )


class _PrintMetrics:
    """Standalone metrics: print the live HUD line (no state tree). Same on-screen
    ground-speed derivation as :class:`MetricsPublisher`."""

    def __init__(self, print_every=30):
        self._n = 0
        self._prev = None
        self._print_every = print_every

    def publish(self, m: NavMetrics, dt: float | None = None) -> None:
        if dt and dt > 0 and self._prev is not None:
            dx, dy = m.x - self._prev[0], m.y - self._prev[1]
            m.speed_mps = (dx * dx + dy * dy) ** 0.5 / dt
        self._prev = (m.x, m.y)
        self._n += 1
        if self._print_every and self._n % self._print_every == 1:
            print("  |  ".join(hud_lines(m)), flush=True)


class _StandaloneHost:
    """No VolRover3: own a pycvc_gl ``Lab()`` and a live, non-blocking window.

    The ``SceneRenderer`` is created lazily in :meth:`run` — AFTER the demo's
    ``setup()`` has populated the scene — mirroring the native demos, which build
    the scene first and then attach one renderer to it."""

    def __init__(self, title="grl-snam demo", width=1280, height=800, fps=60.0):
        self._title = str(title)
        self._w = int(width)
        self._h = int(height)
        self._fps = max(1.0, float(fps))
        self._lab = None
        self._view = None

    def make_lab(self):
        import pycvc_gl
        from pycvc_gl.lab import Lab

        self._lab = Lab()  # standalone: builds its own app + SceneGraph
        # Attach ONE non-blocking on-screen renderer NOW, before the demo adds its
        # nodes — mirroring the embedded case, where the host's renderer is already
        # present as setup() populates the scene (so add_*/setTransform land on a
        # live renderer). offscreen=False opens a real window and needs a display.
        self._view = pycvc_gl.SceneRenderer(self._lab._scene, self._w, self._h, False, "main")
        return self._lab

    def camera(self, fov=60.0):
        return _DirectCamera(self, fov=fov)

    def metrics(self):
        return _PrintMetrics()

    def view(self):
        return self._view

    def run(self, step) -> None:
        import time

        view = self._view
        if view is None:  # setup() never built a Lab through this host
            raise RuntimeError("standalone demo did not create a Lab via the host")
        view.resetCamera()  # sane first frame until step() aims the chase camera
        view.render()

        min_dt, max_dt = 1.0 / 240.0, 0.25
        prev = None
        # Caller-owned loop (the native-demo shape): pump UI, step the sim, draw —
        # until the user closes the window. dt from a wall clock, clamped so the
        # first frame and any hitch can't spike the integrator.
        while not view.windowClosed():
            now = time.monotonic()
            dt = (now - prev) if prev is not None else (1.0 / 60.0)
            prev = now
            if dt < min_dt or dt > max_dt:
                dt = 1.0 / 60.0
            view.processUIEvents()
            step(dt)
            view.render()
        view.close()


def current_host():
    """The active demo host: the standalone window while a :func:`run_standalone`
    is in progress, else the VolRover3 embedded host (built lazily so importing a
    demo never needs the running host)."""
    if _STANDALONE is not None:
        return _STANDALONE
    return _Vr3Host()


def run_standalone(module, *, title=None, width=1280, height=800, fps=60.0) -> None:
    """Run a demo *module* (a ``setup()`` + ``step(dt)`` pair) in a standalone
    pycvc_gl window — no VolRover3. Backs ``grl-snam demo NAME --standalone``.

    The module's own ``setup()``/``step`` are used verbatim: switching the module
    global :data:`_STANDALONE` makes :func:`current_host` hand the demo a
    standalone host instead of the VolRover3 one, so the demo code is identical in
    both worlds."""
    global _STANDALONE
    name = getattr(module, "__name__", "demo").rsplit(".", 1)[-1].replace("_", "-")
    host = _StandaloneHost(title=title or f"grl-snam: {name}", width=width, height=height, fps=fps)
    _STANDALONE = host
    try:
        if hasattr(module, "setup"):
            module.setup()
        host.run(module.step)
    finally:
        _STANDALONE = None
