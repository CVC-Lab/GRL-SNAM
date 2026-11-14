"""The ports in a browser (libcvc's ariglx-web) and the Pages site that serves them.

grl_snam.ariadne._host's browser mode (the page's Host adopted, left open after run(), documents
unpacked from a zip), pages/main.py (the program the page runs) and pages/assemble.py (the site).
None of it needs a browser: the browser mode is switched on by hand. The site itself runs in
headless Chrome in pages.yml (pages/smoke.mjs).
"""

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from grl_snam.ariadne import _host

REPO = Path(__file__).resolve().parent.parent


def _pages_module(name):
    spec = importlib.util.spec_from_file_location(f"pages_{name}", REPO / "pages" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── _host: documents on disk ───────────────────────────────────────────────────────────────────


def test_on_disk_keeps_a_directory_and_a_path_in_no_zip(tmp_path):
    assert _host.on_disk(tmp_path) == tmp_path
    assert _host.on_disk(tmp_path / "nope") == tmp_path / "nope"
    assert _host.DOCS.is_dir() and (_host.DOCS / "learn.ari").is_file()


def test_on_disk_unpacks_a_directory_inside_a_zip_once(tmp_path):
    z = tmp_path / "python312.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("pycvc_gl/cars_panels.ari", "units: {}\n")
        f.writestr("pycvc_gl/cars.py", "x = 1\n")
        f.writestr("pycvc_gl/sub/more.ari", "root: []\n")
        f.writestr("other/elsewhere.ari", "root: []\n")
    d = _host.on_disk(z / "pycvc_gl")
    assert d.is_dir() and not str(d).startswith(str(z))
    assert (d / "cars_panels.ari").read_text() == "units: {}\n"
    assert (d / "sub" / "more.ari").is_file()
    assert not (d / "cars.py").exists()  # the loader reads documents, not the package's code
    assert not (d.parent / "other").exists()
    assert _host.on_disk(str(z / "pycvc_gl")) == d  # once


# ── _host: the browser mode ────────────────────────────────────────────────────────────────────


class _FakeHost:
    def __init__(self, embedded):
        self.embedded, self.mode, self.closed, self.runs = embedded, "window", False, []

    def run(self, **kw):
        self.runs.append(kw)
        return 0

    def snapshot(self, path):
        pass

    def exit_code(self):
        return 0

    def close(self):
        self.closed = True


def test_finish_leaves_the_pages_host_open(monkeypatch):
    args = SimpleNamespace(frames=0, png="")
    monkeypatch.setattr(_host, "BROWSER", True)
    h = _FakeHost(embedded=True)
    assert _host.finish(h, args) == 0
    assert not h.closed and h.runs == [{}]  # host.run(): the page runs the frames
    monkeypatch.setattr(_host, "BROWSER", False)
    h = _FakeHost(embedded=False)
    assert _host.finish(h, args) == 0
    assert h.closed and h.runs == [{"fps": 30, "frames": None}]  # as before


def test_parse_adopts_the_pages_host_in_a_browser(monkeypatch):
    pytest.importorskip("pycvc_gl.ariadne", reason="pycvc_gl has no Python host (libcvc 3.7.9+)")
    assert _host.parse(["x"], lambda p: p.add_argument("hist")).mode == "window"
    monkeypatch.setattr(_host, "BROWSER", True)
    ns = _host.parse(["x"], lambda p: p.add_argument("hist"))
    assert ns.mode == "embedded" and ns.hist == "x"
    assert _host.parse(["--headless", "x"], lambda p: p.add_argument("hist")).mode == "headless"


# ── the site ───────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """The site, assembled around a stand-in for ariglx-web."""
    assemble = _pages_module("assemble")
    root = tmp_path_factory.mktemp("pages")
    host = root / "ariglx-web"
    host.mkdir()
    for name in assemble.HOST_FILES:
        (host / name).write_text(f"stand-in {name}\n")
    out = root / "_site"
    assert assemble.main(["--host", str(host), "--out", str(out)]) == 0
    return out


def test_the_site_has_the_landing_page_and_learn(site):
    assert (site / "index.html").is_file() and (site / "style.css").is_file()
    assert (site / "img" / "curves.png").is_file() and (site / ".nojekyll").is_file()
    learn = site / "learn"
    for name in ("index.html", "ariglx.wasm", "python312.zip", "main.py", "grl_snam_web.zip"):
        assert (learn / name).is_file(), name


def test_learns_config_preloads_what_its_args_name(site):
    learn = site / "learn"
    config = json.loads((learn / "ariadne.config.json").read_text())
    assert config["src"] == "main.py" and config["src_param"] is False
    assert config["args"][0] == "learn"
    hists = [a for a in config["args"] if a.endswith(".history.jsonl")]
    assert len(hists) == 5
    assert set(config["preload"]) == {"grl_snam_web.zip", *hists}
    for f in config["preload"]:
        assert (learn / f).is_file(), f
    first = (learn / hists[0]).read_text().splitlines()[0]
    assert json.loads(first)  # unzipped: learn reads text


def test_the_package_zip_holds_grl_snam_and_its_documents(site):
    with zipfile.ZipFile(site / "learn" / "grl_snam_web.zip") as z:
        names = set(z.namelist())
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in z.infolist())
    assert {"grl_snam/__init__.py", "grl_snam/ariadne/learn.py",
            "grl_snam/ariadne/docs/learn.ari", "grl_snam/learning/history.py"} <= names  # fmt: skip
    assert not any("__pycache__" in n for n in names)


def test_assemble_refuses_an_arms_table_learn_cannot_read(tmp_path, site):
    assemble = _pages_module("assemble")
    with pytest.raises(SystemExit, match="not an arms table"):
        assemble.main(["--host", str(site / "learn"), "--out", str(tmp_path / "s"),
                       "--arms", str(REPO / "docs/learning-validation/arms.json")])  # fmt: skip


def test_main_unpacks_the_package_and_runs_a_port(site, tmp_path):
    page = tmp_path / "learn"
    page.mkdir()
    for name in ("main.py", "grl_snam_web.zip"):
        (page / name).write_bytes((site / "learn" / name).read_bytes())
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-s", str(page / "main.py"), "--list"], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=120)  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "learn" in r.stdout and "grl-snam: --list: started in" in r.stderr
    assert (page / "grl_snam" / "ariadne" / "docs" / "learn.ari").is_file()
