"""The ports' world is their document's world node, which libcvc's SceneWorld runs.

The first part reads the documents alone: each one the scrub, live, fleet and replay ports load
declares its worlds as scene nodes (``world: data``, ``generated``, ``flat``) and picks the one on
with ``scene: world: { source }``, and none picks a world in its ``init:`` (no ``run.*``, no
``world.*``). The rest load them headless (pycvc's AriApp), so it skips without pycvc_gl's Python
host and its multi-car template (pycvc-gl 3.7.12+), as tests/test_ariadne_live_arms.py does:

- the navigator's grid a port drives on is SceneWorld's (``pycvc.AriApp.grid``), node for node;
- choosing a world is a change to the document (its ``scene.world.source``, a generator setting,
  another city in its ``data.city``), and a key a port once read (``run.world``) does nothing;
- what a drawn host makes in AriApp.grid's place (``_world.made_grid``) is SceneWorld's grid,
  sample for sample, and GRL_SNAM_ARIADNE_GL=1 checks an offscreen host against what SceneWorld
  publishes;
- while another world is on, the city the document names is not downloaded.

Austin's data world needs the installed bundle (``scene-austin-south-small``); without one those
tests skip.
"""

from __future__ import annotations

import os
import re
import types
from pathlib import Path

import numpy as np
import pytest

from grl_snam.ariadne import _host, _world

DOCS = ("scrub.ari", "live.ari", "fleet.ari", "replay.ari")


# ── the documents ──────────────────────────────────────────────────────────────────────────────


def _doc(name) -> str:
    return (Path(_host.DOCS) / name).read_text()


@pytest.mark.parametrize("name", DOCS)
def test_each_document_declares_its_worlds_as_nodes(name):
    text = _doc(name)
    scene = text[text.index("\nscene:") :]
    assert re.search(r"^  world: \{ source: data, cell: 2 \}", scene, re.M)
    worlds = re.findall(r"\bworld: (data|generated|flat)\b", scene)
    assert {"data", "generated", "flat"} <= set(worlds)
    assert "source: { data: city }" in scene and "type: generated_world" in scene
    # The generator's settings are the node's, and every world node is in the physics world.
    gen = scene[scene.index("- node: gen") : scene.index("- node: square")]
    for k in ("seed: 7", "half: 500", "water_level: -4", "buildings: 160"):
        assert k in gen
    assert gen.count("body: {}") == 1 and "visible:" not in scene


@pytest.mark.parametrize("name", DOCS)
def test_no_document_picks_a_world_in_its_init(name):
    text = _doc(name)
    init = text[text.index("init: |") : text.index("\n\n", text.index("init: |"))]
    assert not re.search(r'"run\.(world|uri|data_path|crop)"', init)
    assert not re.search(r'"world\.', init)
    assert '(state-set "nav_grid.half" "600")' in init


# ── headless ───────────────────────────────────────────────────────────────────────────────────


def _host_ready() -> bool:
    from grl_snam.ariadne import _cars

    return _host.HAS_HOST and _cars.HAS_CARS


needs_host = pytest.mark.skipif(
    not _host_ready(), reason="needs pycvc_gl's Python host and pycvc_gl.cars (pycvc-gl 3.7.12+)"
)


def _open(doc="live.ari", mode="headless", path=None, **flags):
    """A host on ``doc`` (or the document at ``path``), with ``flags`` as a port's command line
    gives them (``world``, ``uri``, ``set``)."""
    args = types.SimpleNamespace(mode=mode, width=320, height=240, params=None,
                                 set=flags.pop("set", []), world=flags.pop("world", None),
                                 uri=flags.pop("uri", None), data_path=None)  # fmt: skip
    host = _host.open_host(doc, "world", args, load=False)
    host.load(str(path or Path(_host.DOCS) / doc))
    _world.apply_settings(host, args, flags)
    return host


class _Spy:
    """An AriApp's grid(), recorded."""

    def __init__(self, rt):
        self.rt, self.calls = rt, []

    def grid(self, node, cell, half=0.0):
        g = self.rt.grid(node, cell, half)
        self.calls.append((node, cell, half, g is not None))
        return g


@needs_host
@pytest.mark.parametrize("name,node,shape,cell", [("generated", "gen", (501, 501), 2.0),
                                                  ("flat", "square", (81, 81), 2.42)])  # fmt: skip
def test_the_grid_a_port_drives_on_is_scene_worlds(name, node, shape, cell, monkeypatch):
    import pycvc

    host = _open(world=name)
    try:
        spy = _Spy(host.runtime)
        monkeypatch.setattr(_world, "_runtime", lambda h: spy)
        w = _world.world(host)
        assert isinstance(host.runtime, pycvc.AriApp)
        assert spy.calls[-1] == (node, cell, 600.0, True)  # the document's cell and nav_grid.half
        g = host.runtime.grid(node, cell, 600.0)
        assert w.kind == name and w.occ.shape == shape == (g.rows(), g.cols())
        assert np.array_equal(w.occ, np.asarray(g.occupancy()))
        assert w.bounds == tuple(g.bounds())
        # SceneWorld's own summary of the world on agrees (the scene's cell, the whole world).
        assert (
            int(host.get("scene.world.grid.occupied"))
            == host.runtime.grid(node, 2.0).occupied_count()
        )
        assert (w.geometry is not None) == (name == "generated")
    finally:
        host.close()


@needs_host
def test_choosing_a_world_is_a_change_to_the_document(tmp_path):
    # The document's own choice: the same live.ari with another world on, and no flag.
    text = _doc("live.ari").replace("world: { source: data, cell: 2 }",
                                    "world: { source: flat, cell: 2 }")  # fmt: skip
    doc = tmp_path / "live_flat.ari"
    doc.write_text(text)
    host = _open(path=doc)
    try:
        assert _world.source(host) == "flat" and _world.world(host).kind == "flat"
        # A key the ports once read picks nothing now.
        _world.use(host, {"run.world": "generated"})
        assert _world.world(host).kind == "flat"
        # The document's key does: the world on, then a generator setting (a new world).
        _world.use(host, {"scene.world.source": "generated"})
        w7 = _world.world(host)
        assert w7.kind == "generated" and w7.info["seed"] == 7
        rev = host.get("scene.world.rev")
        _world.use(host, {"world.seed": "8"})
        w8 = _world.world(host)
        assert host.get("scene.world.rev") != rev and w8.info["seed"] == 8
        assert not np.array_equal(w7.occ, w8.occ)
        assert np.array_equal(w8.occ, np.asarray(host.runtime.grid("gen", 2.0, 600.0).occupancy()))
    finally:
        host.close()


@needs_host
def test_the_document_s_flat_square_is_the_cars_templates():
    from pycvc_gl import cars as C

    host = _open(world="flat")
    try:
        w, old = _world.world(host), C.flat_world(4.84)
        assert np.array_equal(w.occ, old.occ)  # 81 x 81, walled at the edge
        assert np.allclose(w.bounds, old.bounds, atol=1e-4)  # the box's extent, in float32
    finally:
        host.close()


@needs_host
def test_the_flags_and_a_record_set_the_documents_world():
    host = _open(world="generated", set=["nav_grid.half=40"])
    try:
        assert host.get("scene.world.source") == "generated"
        w = _world.world(host)
        assert w.bounds == (-40.0, -40.0, 40.0, 40.0)
        assert _world.record(host)["world"] == "generated" and _world.record(host)["crop"] == "40"
        # A run log from before this (run.world, run.crop) chooses the same way.
        _world.from_record(host, {"world": "flat", "crop": "600"}, {"seed": "7"})
        assert _world.source(host) == "flat" and _world.world(host).kind == "flat"
    finally:
        host.close()


@needs_host
@pytest.mark.parametrize("name", ["generated", "flat"])
def test_what_a_drawn_host_makes_is_scene_worlds_grid(name):
    host = _open(world=name)
    try:
        _world.wait(host)
        cell, half = _world.grid_cut(host)
        for node in _world.nodes(host):
            g = host.runtime.grid(node, cell, half)
            if g is not None:
                break
        made = _world.made_grid(host, cell, half)
        assert (made.rows(), made.cols()) == (g.rows(), g.cols())
        assert np.allclose(made.bounds(), g.bounds(), atol=1e-4)  # a box node's extent is float32
        assert np.array_equal(np.asarray(made.occupancy()), np.asarray(g.occupancy()))
        assert _world.check(host) == ""
    finally:
        host.close()


@needs_host
def test_the_city_is_not_fetched_while_another_world_is_on():
    import pycvc

    host = _open(world="generated")
    try:
        _world.world(host)
        for _ in range(5):
            host.step(1.0 / 60.0)
        assert pycvc.data_status(host.app, _world.city_path(host)) in ("error", "idle", "")
    finally:
        host.close()


def _austin():
    b = _host.bundle_or_none()
    if b is None:
        pytest.skip("no Austin scene bundle (scene-austin-south)")
    return b


@needs_host
def test_austin_is_the_data_worlds_city():
    import pycvc

    host = _open(uri=str(_austin()))
    try:
        w = _world.world(host)
        g = host.runtime.grid("city", 2.0, 600.0)
        assert w.kind == "data" and np.array_equal(w.occ, np.asarray(g.occupancy()))
        assert w.geometry is not None and _world.record(host)["uri"] == str(_austin())
        # The geometry the plant collides with is the dataset SceneWorld took, and its grid is
        # made from it as SceneWorld makes it.
        made = pycvc.nav_world_grid(w.geometry, 2.0, 600.0)
        assert np.array_equal(np.asarray(made.occupancy()), w.occ)
    finally:
        host.close()


@pytest.mark.skipif(
    os.environ.get("GRL_SNAM_ARIADNE_GL", "0") in ("", "0"),
    reason="GRL_SNAM_ARIADNE_GL=1 runs the offscreen checks (needs a GL context)",
)
@needs_host
@pytest.mark.parametrize("name", ["generated", "flat"])
def test_an_offscreen_host_drives_on_scene_worlds_grid(name):
    host = _open(mode="offscreen", world=name)
    try:
        w = _world.world(host)
        assert w.kind == name and _world.check(host) == ""
        assert int(host.get("scene.world.grid.rows")) > 0
    finally:
        host.close()
