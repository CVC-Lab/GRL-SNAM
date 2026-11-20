"""The world a port's document runs: its world node, which libcvc's SceneWorld runs.

A port's document declares the worlds it can be, as libcvc's documents do (libcvc's
docs/ARIADNE.md, "Worlds"; ARIADNE_WORLDS.md sections 14 and 18). Each is a scene node with
``world: <name>``:

- ``data``: the city the document's ``data:`` block names (``source: { data: city }``);
- ``generated``: a ``generated_world`` node, its generator settings on the node;
- ``flat``: a walled square.

``scene: world: { source: ... }`` names the world on. libcvc's world runtime (SceneWorld) builds
that world once, and its bodies, what is drawn and the navigator's grid all come from it. A port
never builds a world of its own choosing:

- ``--world NAME`` is the document's ``scene.world.source`` (the World window's Source);
- ``--uri URI`` and ``--bundle DIR`` load another city into the document's ``data.city``, as the
  document's ``(data-load "city" ...)`` does;
- ``--set world.seed=3`` (any of the generator's ``world.*`` keys) asks the ``generated_world``
  node for a new world, as its Regenerate does.

:func:`world` is what a port drives on: the navigator's grid of the world on (SceneWorld's,
``pycvc.AriApp.grid``), cut to the document's ``nav_grid.half``, the terrain's height, and the
world's geometry for the plant's colliders. It is a ``pycvc_gl.cars.World`` because the plants
are still the Cars template's (``nav_sim_world`` with ``plant_host``, which collides with that
geometry) until libcvc's OW7 puts them in the document's world. Two things are still read beside
SceneWorld rather than from it, and :func:`check` holds them to what SceneWorld publishes
(``scene.world.grid.*``):

- **A drawn host's grid.** Under a window, offscreen or in a browser page, ``AriApp.grid`` finds no
  world: it reads ``AppRuntime::scene_world()``, which is empty when a renderer runs the world. So
  the grid is made as SceneWorld makes it (``cvc::nav::make_grid``), over the same world.
- **A generated world's geometry.** SceneWorld's colliders are not reachable from Python, so the
  plant's are generated again from the node's own ``world.*`` keys: the same generator, the same
  settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

#: The documents' worlds, by the names their nodes' ``world:`` give them.
NAMES = ("data", "generated", "flat")
#: The data world's dataset: the document's ``data.city``.
CITY = "city"
#: Where a document says how the navigator's grid is cut: ``nav_grid.<world>.cell``, else
#: ``nav_grid.cell`` (m; default the scene's ``world: { cell }``), and ``nav_grid.half`` (m either
#: side of the terrain's centre; 0,
#: the whole world). The crop cuts the grid only, and becomes the data node's region with
#: libcvc's world bundles (GB2, transfix/libcvc#1380), which cuts the bodies and the drawing too.
GRID = "nav_grid"
#: The Austin packages, whose bundle is a part of the package (as the documents' data: block has it).
AUSTIN = ("cvcpkg://cy-pca/scene-austin-south-small", "cvcpkg://cy-pca/scene-austin-south")
AUSTIN_PATH = "share/cvc-scenes/austin_south"


def _pycvc():
    import pycvc

    return pycvc


def _cars():
    from grl_snam.ariadne import _cars

    return _cars.require("the document's world")


# ── which world is on ──────────────────────────────────────────────────────────────────────────


def source(host) -> str:
    """The name of the world on: the document's ``scene.world.source``."""
    return host.get("scene.world.source") or ""


def _taken(host) -> bool:
    """Whether SceneWorld has taken the world asked for: its status names it. Until its next step
    (in a page, until the page runs a frame) it still publishes the world before."""
    return (host.get("scene.world.status") or "").split(" ", 1)[0] == source(host)


def kind(host) -> str:
    """The world on's kind (``data``, ``generated``, ``terrain`` or ``flat``), as SceneWorld says
    once it has taken it, else its name (the documents name their worlds by their kinds)."""
    return (host.get("scene.world.kind") if _taken(host) else "") or source(host)


def nodes(host) -> list[str]:
    """The scene nodes of the world on (``scene.world.node``), once SceneWorld has taken it."""
    return [n for n in (host.get("scene.world.node") or "").split(",") if n] if _taken(host) else []


def city_path(host) -> str:
    """The document's ``data.city``, by its path in the state tree."""
    return f"{host.prefix}.data.{CITY}"


def registry_origin() -> str:
    """The origin of the cvcpkg registry the data store reaches: ``CVCPKG_URL``'s, else
    https://cvcpkg.org."""
    from grl_snam.ariadne._host import registry_origin as origin

    return origin()


# ── choosing the world: the document's ──────────────────────────────────────────────────────────


def load_city(host, uri, path=None) -> None:
    """Point the document's ``data.city`` at ``uri`` (a ``cvcpkg://ORG/NAME`` package, an https
    archive or a bundle directory), decoded as a city bundle, as ``(data-load "city" URI
    "city_bundle" PATH)`` does. ``path`` is the part of a package to keep; an Austin package's is
    its bundle's directory, and a directory is the bundle itself."""
    pycvc = _pycvc()
    uri = str(uri)
    if os.path.isdir(uri):
        path = ""
    else:
        if not path and uri.split("@")[0] in AUSTIN:
            path = AUSTIN_PATH
        pycvc.data_allow_network(host.app, [registry_origin()])
    pycvc.data_load(host.app, uri, path=path or "", decoder="city_bundle", name=CITY,
                    prefix=host.prefix)  # fmt: skip
    host._grl_snam_city_path = path or ""


def apply_settings(host, args, flags=None, *, bundle=None) -> dict:
    """What a port's flags ask for, over the document's ``init:``, as ``pycvc_gl.cars``'s
    ``apply_settings`` sets them: the ``--params`` file's keys, each ``--set``, then ``flags``
    ({key: value}; None is skipped). Then the world, in the document: ``--world`` is its
    ``scene.world.source``, ``--uri`` (or ``bundle``, a local city) a city loaded into its
    ``data.city``, and a changed ``world.*`` key a new generated world. Returns the keys set."""
    overrides = {}
    if getattr(args, "params", None):
        overrides.update(json.loads(Path(args.params).read_text()))
    for kv in getattr(args, "set", None) or []:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = v.strip()
    for key, value in (flags or {}).items():
        if value is not None:
            overrides[key] = value
    if getattr(args, "world", None):
        overrides["scene.world.source"] = args.world
    uri = getattr(args, "uri", None) or bundle
    if uri:  # first, so the document's own city is not fetched for nothing
        load_city(host, uri, getattr(args, "data_path", None))
    use(host, overrides)
    return overrides


def use(host, keys: dict) -> None:
    """Set ``keys`` ({key: value}, relative to the document) and act on the world's: a
    ``world.*`` key whose value changes asks the generated_world node for a new world (its
    ``world.request``), and the data world, once it is on, may reach the registry. :func:`wait`
    then waits for the world asked for."""
    regenerate = False
    for k, v in keys.items():
        v = "" if v is None else str(v)
        if k.startswith("world.") and k != "world.request" and (host.get(k) or "") != v:
            regenerate = True
        if k == "scene.world.source" and source(host) != v:
            host._grl_snam_world_asked = True
        host.set(k, v)
    if regenerate:
        host.set("world.request", "regenerate")
        host._grl_snam_world_asked = True
    network(host)


def network(host) -> None:
    """Let the data store reach the registry while the data world is on, and load its city again
    (the document's ``(data-reload "city")``) if it failed for want of it. A city the document
    names is fetched as soon as its node is drawn, so while another world is on the network stays
    shut and nothing is downloaded."""
    if source(host) != "data":
        return
    pycvc = _pycvc()
    pycvc.data_allow_network(host.app, [registry_origin()])
    if pycvc.data_status(host.app, city_path(host)) == "error":
        host.run_program(f'(data-reload "{CITY}")')  # it runs in the next frame
        if _steps(host):
            host.step(1.0 / 60.0)


def from_record(host, run: dict | None, world_keys: dict | None) -> None:
    """The world a log or a report recorded (:func:`record`'s ``run``, and a generated world's
    ``world_keys``), chosen in the document: its source, its city, its generator settings. A
    record from before (``run.world``, ``run.uri``, ``run.data_path``) reads the same way."""
    run = {k: v for k, v in (run or {}).items() if v not in (None, "")}
    keys = {f"world.{k}": v for k, v in (world_keys or {}).items() if v not in (None, "")}
    if run.get("world"):
        keys["scene.world.source"] = run["world"]
    if run.get("crop") not in (None, ""):
        keys[f"{GRID}.half"] = run["crop"]
    uri = run.get("uri")
    if run.get("world") == "data" and uri and uri != city_uri(host):
        load_city(host, uri, run.get("data_path"))
    use(host, keys)


def city_key(host, key) -> str:
    """``key`` of the dataset the document's ``data.city`` links to (``uri``, ``canonical``,
    ``status``, ...): the store's node, read at the link's target."""
    pycvc = _pycvc()
    path = city_path(host)
    try:
        path = pycvc.State.instance(host.app).child(path).linkTarget() or path
    except RuntimeError:
        pass
    try:
        return pycvc.state_get(host.app, f"{path}.{key}") or ""
    except RuntimeError:
        return ""


def city_uri(host) -> str:
    """The city the document's ``data.city`` holds, as it was asked for (a package URI, an
    archive, a directory)."""
    return city_key(host, "uri")


def record(host) -> dict:
    """What a run log keeps to rebuild the world: the world on (``world``), the city its document
    loaded (``uri``, ``data_path``: the part of the package kept) and the grid's crop (``crop``)."""
    uri = city_uri(host)
    path = getattr(host, "_grl_snam_city_path", None)
    if path is None:
        path = AUSTIN_PATH if uri.split("@")[0] in AUSTIN else ""
    return {"world": source(host), "uri": uri, "data_path": path,
            "crop": host.get(f"{GRID}.half") or ""}  # fmt: skip


# ── the world on ───────────────────────────────────────────────────────────────────────────────


def _runtime(host):
    """A ``pycvc.AriApp`` over the host's document runtime: its own (headless), else one
    attached to a drawn host's. None when this pycvc has neither."""
    pycvc = _pycvc()
    rt = getattr(host, "runtime", None)
    if isinstance(rt, pycvc.AriApp):
        return rt
    cached = getattr(host, "_grl_snam_ari_app", None)
    if cached is not None:
        return cached
    capsule = getattr(rt, "app_runtime_capsule", None)
    if capsule is None or not hasattr(pycvc, "ari_app_attach"):
        return None
    app = pycvc.ari_app_attach(host.app, capsule())
    host._grl_snam_ari_app = app
    return app


def _headless(host) -> bool:
    return host.mode == "headless"


def _steps(host) -> bool:
    """Whether this program may run the host's frames itself: not in a scripting host's page
    (mode ``embedded``), which runs them once the program returns, nor from inside a frame (a
    handler or a watch)."""
    return not getattr(host, "embedded", False) and not getattr(host, "_in_frame", False)


def _settled(host) -> bool:
    """Whether the world on is the one asked for and has its world: SceneWorld's status names
    it, it is ready, and a generated world has no request or job left."""
    if not _taken(host) or host.get("scene.world.ready") != "1":
        return False
    if kind(host) == "generated":
        return not host.get("world.request") and host.get("world.busy") in (None, "", "0")
    return True


def wait(host, timeout=900.0) -> None:
    """Wait until the world on has its world: its dataset loaded (the data world), and SceneWorld
    on it (``scene.world.status`` and ``ready``), with any new world :func:`use` asked for made.
    In a page (mode ``embedded``) the frames are the page's, so only the dataset is waited for.
    Raises RuntimeError when it fails or does not come."""
    import time

    pycvc = _pycvc()
    name = source(host)
    if name not in NAMES:
        raise RuntimeError(f"the document has no world {name!r} (its worlds: "
                           f"{', '.join((host.get('scene.world.sources') or '').split())})")  # fmt: skip
    if name == "data":
        network(host)
        if not pycvc.data_wait(host.app, city_path(host), timeout=timeout):
            raise RuntimeError(f"the document's city did not load: {_data_error(host)}")
    if not _steps(host):
        if getattr(host, "_in_frame", False) and not _settled(host):
            raise RuntimeError(f"the {name} world is not ready: {host.get('scene.world.status')}")
        return
    deadline = time.monotonic() + timeout
    if getattr(host, "_grl_snam_world_asked", False):
        host.step(1.0 / 60.0)  # a new source or a request is taken at the next step
        host._grl_snam_world_asked = False
    while not _settled(host):
        if time.monotonic() > deadline:
            raise RuntimeError(f"the {name} world did not come: {host.get('scene.world.status')}")
        host.step(1.0 / 60.0)
        time.sleep(0.001)


def _data_error(host) -> str:
    return (
        city_key(host, "error") or _pycvc().data_status(host.app, city_path(host)) or "not loaded"
    )


def scene_cell(host) -> float:
    """The cell (m) of the grid SceneWorld publishes: the document's ``scene: world: { cell }``."""
    try:
        return float(host.get("scene.world.grid.cell") or 2.0)
    except ValueError:
        return 2.0


def grid_cut(host, cell=None, half=None) -> tuple[float, float]:
    """The navigator's grid's (cell, half), m: as given, else the document's: the world on's own
    cell (``nav_grid.<world>.cell``, the flat square's half a Humvee), else ``nav_grid.cell``,
    else the scene's; ``nav_grid.half``, else the whole world."""
    if cell is None:
        cell = (host.get_float(f"{GRID}.{source(host)}.cell", 0.0)
                or host.get_float(f"{GRID}.cell", 0.0) or scene_cell(host))  # fmt: skip
    if half is None:
        half = host.get_float(f"{GRID}.half", 0.0) or 0.0
    return float(cell), float(half)


def grid(host, cell=None, half=None, *, timeout=60.0):
    """The navigator's grid of the world on (a ``pycvc.world_grid``): SceneWorld's
    (``pycvc.AriApp.grid``), ``cell`` m samples, cut to ``half`` m either side of its centre
    (:func:`grid_cut`). Under a drawn host, where ``AriApp.grid`` finds no world, it is made as
    SceneWorld makes it, over the same world (:func:`made_grid`)."""
    import time

    cell, half = grid_cut(host, cell, half)
    rt = _runtime(host)
    deadline = time.monotonic() + timeout
    while rt is not None:
        for node in nodes(host):
            g = rt.grid(node, cell, half)
            if g is not None:
                return g
        if not _headless(host) or not _steps(host) or time.monotonic() > deadline:
            break  # a drawn host's runtime has no world to read (libcvc 4.0)
        host.step(1.0 / 60.0)  # SceneWorld takes a dataset's world at its next step
    if _headless(host):
        raise RuntimeError(f"SceneWorld has no grid of the {source(host)} world "
                           f"({host.get('scene.world.status')})")  # fmt: skip
    return made_grid(host, cell, half)


def made_grid(host, cell, half):
    """The grid SceneWorld makes of the world on (``SceneWorld::grid``: ``cvc::nav::make_grid``
    over its world, an open walled grid over a flat world's footprint), made here from the same
    world: the document's dataset, the generator on the node's keys, the flat world's extent."""
    pycvc = _pycvc()
    k = kind(host)
    if k == "data":
        return pycvc.nav_world_grid(pycvc.data_world(host.app, city_path(host)), cell, half)
    if k == "generated":
        return pycvc.nav_world_grid(generated(host).world(), cell, half)
    if k == "flat":
        return pycvc.nav_flat_grid(flat_half(host), cell)
    raise RuntimeError(f"a {k or 'missing'} world: the ports drive on data, generated or flat")


def flat_half(host) -> float:
    """Half the flat world's side (m): the document's ``nav_grid.flat.half``, else from the grid
    SceneWorld publishes of it."""
    half = host.get_float(f"{GRID}.flat.half", 0.0)
    if half > 0.0:
        return half
    cols = int(float(host.get("scene.world.grid.cols") or 0))
    if cols < 3:
        raise RuntimeError("the flat world has no grid yet (scene.world.grid)")
    return 0.5 * (cols - 1) * scene_cell(host)


_GENERATED = {}  # the generator's settings (as JSON) -> the world generated from them


def world_keys(host) -> dict:
    """The generator's settings: the generated_world node's ``world.*`` keys (those it has)."""
    keys = {}
    for k in _cars().WORLD_KEYS:
        v = host.get(f"world.{k}")
        if v not in (None, ""):
            keys[k] = v
    return keys


def generated(host):
    """The generated world on, generated here from the node's own keys (:func:`world_keys`), once
    per setting: what the plant's colliders and the terrain's height come from while SceneWorld's
    are not reachable from Python."""
    C = _cars()
    params = {}
    for k, v in world_keys(host).items():
        params[k] = v if k in ("terrain", "city", "terrain_file") else float(v)
    key = json.dumps(params, sort_keys=True)
    if key not in _GENERATED:
        _GENERATED.clear()
        _GENERATED[key] = C.generate(params, app=host.app)
    return _GENERATED[key]


def check(host, g=None) -> str:
    """Whether the grid made here (:func:`made_grid`) is SceneWorld's: its rows, columns and
    blocked samples against ``scene.world.grid.*``, at the scene's cell over the whole world.
    "" when they agree, or SceneWorld has published none of this world yet; else what differs."""
    rows = host.get("scene.world.grid.rows")
    if rows in (None, "") or not _taken(host):
        return ""
    want = (int(float(rows)), int(float(host.get("scene.world.grid.cols") or 0)),
            int(float(host.get("scene.world.grid.occupied") or 0)))  # fmt: skip
    g = g if g is not None else made_grid(host, scene_cell(host), 0.0)
    got = (g.rows(), g.cols(), int(g.occupied_count()))
    if got == want:
        return ""
    return (f"the {source(host)} world's grid here is {got[0]} x {got[1]} with {got[2]} blocked, "
            f"SceneWorld's {want[0]} x {want[1]} with {want[2]}")  # fmt: skip


def world(host, cell=None, half=None, *, timeout=900.0):
    """The world on, as a port drives on it (a ``pycvc_gl.cars.World``): the navigator's grid
    (:func:`grid`), the terrain's height, and the geometry the plant collides with (the data
    world's dataset; a generated world's, generated from the node's keys; none on flat ground).
    Waits for the world (:func:`wait`)."""
    C = _cars()
    pycvc = _pycvc()
    wait(host, timeout=timeout)
    g = grid(host, cell, half)
    occ = np.ascontiguousarray(g.occupancy(), np.uint8)
    bounds = tuple(float(b) for b in g.bounds())
    k = kind(host)
    if k == "data":
        path = city_path(host)
        geometry = pycvc.data_world(host.app, path)
        height = pycvc.data_height_field(host.app, path).height
        info = {"source": "data", "uri": city_uri(host), "canonical": city_key(host, "canonical"),
                "path": path}  # fmt: skip
    elif k == "generated":
        gen = generated(host)
        geometry, height = gen.world(), gen.height
        info = {"source": "generated", **dict(gen.info())}
    else:
        geometry = height = None
        info = {"source": k}
    w = C.World(k, occ, bounds, height, geometry, info)
    if not _headless(host):
        why = check(host)
        if why:
            print(f"grl_snam: {why}", flush=True)
    return w


def city(host, w):
    """The world ``w`` (:func:`world`) as a :class:`~grl_snam.demos._common.City`, the city a
    ``grl_snam.demos`` demo drives in: the terrain's height, its bounds, and occupancy at the demo's
    own grid, rasterized from the world's geometry by libcvc's nav rasterizer, or, on flat ground,
    from the navigator's grid. The data world's is its dataset's terrain and world, so nothing is
    read again."""
    from grl_snam.demos._common import City, world_occupancy

    if w.kind == "data":
        pycvc = _pycvc()
        path = city_path(host)
        return City.of_world(
            pycvc.data_world(host.app, path), pycvc.data_height_field(host.app, path)
        )
    if w.geometry is not None:
        geometry = w.geometry

        def occupancy(bounds, nx, ny, inflate_m=0.0):
            return world_occupancy(geometry, bounds, nx, ny, inflate_m=inflate_m)

    else:

        def occupancy(bounds, nx, ny, inflate_m=0.0):
            mnx, mny, mxx, mxy = bounds
            X, Y = np.meshgrid(np.linspace(mnx, mxx, nx), np.linspace(mny, mxy, ny))
            occ = w.occupied(X, Y)
            if inflate_m > 0.0 and occ.any():
                pycvc = _pycvc()
                cell = (mxx - mnx) / max(nx - 1, 1)
                d2 = np.asarray(pycvc.nav_edt2_squared(occ.astype(np.uint8)), float)
                occ = d2 * cell * cell <= inflate_m * inflate_m
            return occ

    return City(w.z, w.bounds, occupancy)
