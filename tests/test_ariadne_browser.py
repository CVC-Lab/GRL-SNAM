"""The ports in a browser (libcvc's ariglx-web) and the Pages site that serves them.

grl_snam.ariadne._host's browser mode (the page's Host adopted, left open after run(), documents
unpacked from a zip), pages/main.py (the program the page runs) and pages/assemble.py (the site).
None of it needs a browser: the browser mode is switched on by hand. The site itself runs in
headless Chrome in pages.yml (pages/smoke.mjs).
"""

import gzip
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
from grl_snam.learning import history

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
    # No scrub page (no --scrub): no card for it, no picture.
    assert 'href="scrub/"' not in (site / "index.html").read_text()
    assert not (site / "scrub").exists() and not (site / "img" / "scrub.jpg").exists()
    assert (site / "img" / "curves.png").is_file() and (site / ".nojekyll").is_file()
    learn = site / "learn"
    for name in ("index.html", "ariglx.wasm", "python312.zip", "main.py", "grl_snam_web.zip"):
        assert (learn / name).is_file(), name


def test_learns_config_preloads_what_its_args_name(site):
    learn = site / "learn"
    config = json.loads((learn / "ariadne.config.json").read_text())
    assert config["src"] == "main.py" and config["src_param"] is False
    assert config["args"][0] == "learn"
    hists = [a for a in config["args"] if a.endswith(".history.jsonl.gz")]
    assert len(hists) == 5
    assert set(config["preload"]) == {"grl_snam_web.zip", *hists}
    for f in config["preload"]:
        assert (learn / f).is_file(), f
    # Shipped still gzipped, byte for byte, and learn's reader reads them as they are.
    retrain = REPO / "docs" / "learning-validation" / "retrain"
    assert (learn / hists[0]).read_bytes() == (retrain / Path(hists[0]).name).read_bytes()
    runs = history.load(learn / hists[0])
    assert runs[0].schema == "grl_snam.coef_train_history/v1"


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


def test_the_learn_page_alone_with_histories_of_your_choice(tmp_path, site):
    assemble = _pages_module("assemble")
    plain = tmp_path / "runs" / "a.history.jsonl"
    plain.parent.mkdir()
    seed0 = site / "learn" / "data" / "seed0.history.jsonl.gz"
    plain.write_bytes(gzip.decompress(seed0.read_bytes()))
    gz = REPO / "docs/learning-validation/retrain/seed1.history.jsonl.gz"
    out = tmp_path / "other" / "learn"
    host = site / "learn"
    assert assemble.main(["--host", str(host), "--out", str(out), "--learn-only",
                          "--histories", str(plain), str(gz)]) == 0  # fmt: skip
    config = json.loads((out / "ariadne.config.json").read_text())
    assert config["args"] == ["learn", "data/a.history.jsonl", "data/seed1.history.jsonl.gz"]
    assert all((out / f).is_file() for f in config["preload"])
    assert (out / "index.html").is_file() and not (out / "learn").exists()
    assert not (out.parent / "index.html").exists() and not (out / "style.css").exists()
    with pytest.raises(SystemExit, match="share a name"):
        assemble.main(["--host", str(host), "--out", str(tmp_path / "x"), "--learn-only",
                       "--histories", str(gz), str(gz)])  # fmt: skip
    zipped = tmp_path / "zipped" / "a.history.jsonl.gz"  # the same run label as plain's
    zipped.parent.mkdir()
    zipped.write_bytes(gzip.compress(plain.read_bytes()))
    with pytest.raises(SystemExit, match="share a name"):
        assemble.main(["--host", str(host), "--out", str(tmp_path / "y"), "--learn-only",
                       "--histories", str(plain), str(zipped)])  # fmt: skip


def _scrub_report(root: Path, name: str = "seed7", set_name: str | None = None) -> Path:
    """A scrub report (grl_snam.scrub/v1) whose history and two run logs are elsewhere, named by
    absolute paths, as `grl-snam ariadne scrub --report` writes it."""
    runs = root / "deployed"
    runs.mkdir(parents=True)
    hist = runs / f"{name}.history.jsonl.gz"
    hist.write_bytes(
        (REPO / "docs/learning-validation/retrain/seed0.history.jsonl.gz").read_bytes()
    )
    cks = []
    for step in (0, 100):
        log = runs / f"{step:016x}-abcd.npz"
        log.write_bytes(b"stand-in run log")
        cks.append({"step": step, "blob": f"/somewhere/coef_sdf.step{step:06d}.pt",
                    "cache": str(log), "metrics": {"path_m": 1.0 + step}})  # fmt: skip
    report = root / f"{name}.scrub.json"
    rep = {"schema": "grl_snam.scrub/v1", "history": str(hist), "run": -1,
           "scenario": {"run": {"world": "generated"}}, "checkpoints": cks}  # fmt: skip
    if set_name is not None:
        rep["set"] = set_name
    report.write_text(json.dumps(rep))
    return report


def _bundle(path: Path):
    """A run's bundle: its report and the names inside."""
    with zipfile.ZipFile(path) as z:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
        return json.loads(z.read("scrub.json")), sorted(z.namelist())


def test_the_scrub_page_carries_its_runs_beside_it(tmp_path, site):
    """--scrub-only --scrub-reports: a bundle per run, data/<run>.scrub.zip, with its report (its
    paths made the names inside), its history and its run logs; the page's config views them."""
    assemble = _pages_module("assemble")
    reports = [_scrub_report(tmp_path / "a", "seed7"), _scrub_report(tmp_path / "b", "seed8")]
    out = tmp_path / "other" / "scrub"
    assert assemble.main(["--host", str(site / "learn"), "--out", str(out), "--scrub-only",
                          "--scrub-reports", *map(str, reports)]) == 0  # fmt: skip
    config = json.loads((out / "ariadne.config.json").read_text())
    assert config["args"][:4] == ["scrub", "--view", "data/seed7.scrub.zip",
                                  "data/seed8.scrub.zip"]  # fmt: skip
    assert config["args"][4:] == assemble.SCRUB_VIEW
    # The Humvee's model ships when --python's prefix (here this Python's) has vehicle-humvee.
    model = Path(sys.executable).resolve().parent.parent / assemble.VEHICLE_MODEL
    assert config["preload"] == ["grl_snam_web.zip", "data/seed7.scrub.zip", "data/seed8.scrub.zip",
                                 *([assemble.VEHICLE_MODEL] if model.is_file() else [])]  # fmt: skip
    assert all((out / f).is_file() for f in config["preload"])
    rep, names = _bundle(out / "data/seed7.scrub.zip")
    assert names == sorted(["scrub.json", "seed7.history.jsonl.gz",
                            *(f"{s:016x}-abcd.npz" for s in (0, 100))])  # fmt: skip
    assert rep["history"] == "seed7.history.jsonl.gz"
    assert [c["cache"] for c in rep["checkpoints"]] == [f"{s:016x}-abcd.npz" for s in (0, 100)]
    assert rep["checkpoints"][1]["blob"] == "coef_sdf.step000100.pt"  # no machine's paths
    assert (out / "main.py").is_file() and not (out / "learn").exists()


def test_the_site_with_the_scrub_page(tmp_path, site):
    assemble = _pages_module("assemble")
    out = tmp_path / "_site"
    assert assemble.main(["--host", str(site / "learn"), "--out", str(out), "--scrub",
                          "--scrub-reports", str(_scrub_report(tmp_path))]) == 0  # fmt: skip
    assert (out / "learn" / "ariadne.config.json").is_file()
    assert (out / "scrub" / "data" / "seed7.scrub.zip").is_file()
    html = (out / "index.html").read_text()
    assert 'href="scrub/"' in html and (out / "img" / "scrub.jpg").is_file()


def test_assemble_refuses_what_the_scrub_page_cannot_show(tmp_path, site):
    assemble = _pages_module("assemble")
    host = str(site / "learn")
    with pytest.raises(SystemExit, match="not both"):
        assemble.main(
            ["--host", host, "--out", str(tmp_path / "x"), "--learn-only", "--scrub-only"]
        )
    with pytest.raises(SystemExit, match="not a scrub report"):
        assemble.main(["--host", host, "--out", str(tmp_path / "y"), "--scrub-only",
                       "--scrub-reports", str(REPO / "docs/learning-validation/arms.json")])  # fmt: skip
    report = _scrub_report(tmp_path / "z")
    next((tmp_path / "z" / "deployed").glob("*.npz")).unlink()
    with pytest.raises(SystemExit, match="run log is not there"):
        assemble.main(["--host", host, "--out", str(tmp_path / "w"), "--scrub-only",
                       "--scrub-reports", str(report)])  # fmt: skip
    with pytest.raises(SystemExit, match="no saved policies"):
        assemble.main(["--host", host, "--out", str(tmp_path / "v"), "--scrub-only",
                       "--checkpoints", str(tmp_path / "none")])  # fmt: skip


def test_the_scrub_page_finds_the_checkpoints_package_beside_its_python(tmp_path):
    """--checkpoints defaults to grl-snam-learning-checkpoints' retrain/ in --python's prefix."""
    assemble = _pages_module("assemble")
    python = tmp_path / "prefix" / "bin" / "python3.12"
    python.parent.mkdir(parents=True)
    python.write_text("")
    with pytest.raises(SystemExit, match="cvcpkg install grl-snam-learning-checkpoints"):
        assemble.default_checkpoints(str(python))
    root = tmp_path / "prefix" / "share" / "grl-snam-learning-checkpoints"
    root.mkdir(parents=True)
    assert assemble.default_checkpoints(str(python)) == root


def test_the_scrub_page_carries_several_data_sets(tmp_path, site):
    """Runs of the same name in two data sets: data/<set>/<run>.scrub.zip, viewed together."""
    assemble = _pages_module("assemble")
    reports = [_scrub_report(tmp_path / "a", "seed7", "retrain"),
               _scrub_report(tmp_path / "b", "seed7", "fixed/retrain")]  # fmt: skip
    out = tmp_path / "scrub"
    assert assemble.main(["--host", str(site / "learn"), "--out", str(out), "--scrub-only",
                          "--scrub-reports", *map(str, reports)]) == 0  # fmt: skip
    config = json.loads((out / "ariadne.config.json").read_text())
    assert config["args"][1:4] == ["--view", "data/retrain/seed7.scrub.zip",
                                   "data/fixed/retrain/seed7.scrub.zip"]  # fmt: skip
    assert all((out / f).is_file() for f in config["preload"])
    assert _bundle(out / "data/fixed/retrain/seed7.scrub.zip")[0]["set"] == "fixed/retrain"
    with pytest.raises(SystemExit, match="two scrub reports are of run retrain/seed7"):
        assemble.main(["--host", str(site / "learn"), "--out", str(tmp_path / "x"),
                       "--scrub-only", "--scrub-reports", str(reports[0]), str(reports[0])])  # fmt: skip


def test_the_scrub_page_carries_several_plants(tmp_path, site):
    """Runs deployed on two plants: data/<plant>/<set>/<run>.scrub.zip, the first report's first."""
    assemble = _pages_module("assemble")
    reports = []
    for plant in ("humvee", "kinematic"):
        r = _scrub_report(tmp_path / plant, "seed7", "retrain")
        rep = json.loads(r.read_text())
        rep["scenario"]["plant"] = plant
        r.write_text(json.dumps(rep))
        reports.append(r)
    out = tmp_path / "scrub"
    assert assemble.main(["--host", str(site / "learn"), "--out", str(out), "--scrub-only",
                          "--scrub-reports", *map(str, reports)]) == 0  # fmt: skip
    config = json.loads((out / "ariadne.config.json").read_text())
    assert config["args"][1:4] == ["--view", "data/humvee/retrain/seed7.scrub.zip",
                                   "data/kinematic/retrain/seed7.scrub.zip"]  # fmt: skip
    assert _bundle(out / "data/kinematic/retrain/seed7.scrub.zip")[0]["scenario"]["plant"] == (
        "kinematic"
    )
    assert assemble.SCRUB_PLANTS[0] == "humvee"  # the page opens on the plant that has walls


def test_the_learning_checks_data_sets(tmp_path):
    assemble = _pages_module("assemble")
    assert [h.name for h in assemble.set_histories("fixed/retrain")] == [
        f"seed{s}.history.jsonl.gz" for s in range(5)
    ]
    for bad in ("../retrain", "/tmp", "none"):
        with pytest.raises(SystemExit):
            assemble.set_histories(bad)


def test_the_scrub_page_refuses_runs_deployed_on_another_libcvc(tmp_path, site):
    """--expect-libcvc: the page draws libcvc's generated world with its own libcvc, so runs
    deployed on another (or on one the report does not name) are refused."""
    assemble = _pages_module("assemble")
    report = _scrub_report(tmp_path / "a")
    rep = json.loads(report.read_text())
    rep["provenance"] = {"libcvc": "4.0.0"}
    report.write_text(json.dumps(rep))
    args = ["--host", str(site / "learn"), "--scrub-only", "--scrub-reports", str(report)]
    assert assemble.main([*args, "--out", str(tmp_path / "ok"), "--expect-libcvc", "v4.0.0"]) == 0
    with pytest.raises(SystemExit, match=r"the page's libcvc is 3\.7\.13, but runs were deployed"):
        assemble.main([*args, "--out", str(tmp_path / "no"), "--expect-libcvc", "3.7.13"])
    other = _scrub_report(tmp_path / "b", "seed8")  # names no libcvc
    with pytest.raises(SystemExit, match="unknown"):
        assemble.main(["--host", str(site / "learn"), "--scrub-only", "--out", str(tmp_path / "x"),
                       "--scrub-reports", str(other), "--expect-libcvc", "4.0.0"])  # fmt: skip
