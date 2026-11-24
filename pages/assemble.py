#!/usr/bin/env python3
"""Assemble the GRL-SNAM Pages site: the landing page, and the ports that run in the browser.

    python pages/assemble.py --host HOST --out _site [--histories H...] [--arms TABLE]
    python pages/assemble.py --host HOST --out DIR --learn-only [--histories H...] [--arms TABLE]
    python pages/assemble.py --host HOST --out _site --scrub --python PY [--scrub-sets SET...] ...
    python pages/assemble.py --host HOST --out _site --scrub --scrub-austin --python PY ...
    python pages/assemble.py --host HOST --out DIR --scrub-only --scrub-reports R.json...

``--host`` is libcvc's ariglx-web, the ``share/ariglx-web`` directory pages/build-ariglx-web.sh
installs. The site:

    index.html, style.css, img/       the landing page (pages/site/), and its figures
    learn/                            grl-snam ariadne learn, on ariglx-web:
      index.html ariglx.js ariglx.wasm python312.zip arigl-web.js arigl-web.css
      LICENSE THIRD_PARTY_NOTICES.md    the host, as its recipe installs it
      main.py                         the program the page runs (pages/main.py)
      grl_snam_web.zip                grl_snam's files, its .ari documents with them
      data/<name>.history.jsonl[.gz]  the learning histories, as given (learn reads gzip)
      ariadne.config.json             src main.py, every file above in its preload, the args
    scrub/                            with --scrub: grl-snam ariadne scrub --view, on ariglx-web:
      (the host, main.py and grl_snam_web.zip, as learn/'s)
      data/<set>/<run>.scrub.zip      a run's bundle, stored: its scrub report (scrub.json,
                                      grl_snam.scrub/v1, its paths the names inside), its
                                      learning history and each checkpoint's run log (lean,
                                      packed: grl_snam.run_log/v2); the page's loader takes
                                      at most 512 files, and a run is 43
      ariadne.config.json             ["scrub", "--view", "data/<set>/<run>.scrub.zip", ...]
    scrub-austin/                     with --scrub-austin: the same page on the same runs,
                                      deployed in Austin (the data world), beside scrub/

``--learn-only`` writes the learn page alone, self-contained, at ``--out`` (for another site to
serve, at any path). Everything the page fetches is beside it, on the site's own origin.

``--histories`` names the learning histories the page shows (``.jsonl`` or ``.jsonl.gz``, copied
as they are: a gzipped one stays gzipped on the wire, and learn reads it); by default the learning
check's retrain, docs/learning-validation/retrain/seed*.history.jsonl.gz. A run is labelled by its
file's stem (``.gz`` dropped), so their names must differ. ``--arms`` adds an arms table (``grl_snam.arms/v1``, what ``grl-snam
arms-eval`` writes) as ``data/arms.json`` and learn's ``--arms``; any other schema is refused,
since learn would refuse it in the page.

``--scrub`` adds the checkpoint scrubber's page, ``--scrub-only`` writes it alone at ``--out``.
The page only views runs: deploying a checkpoint needs torch (a ``.pt``) and libcvc's sim_world,
so it is done here, natively, by ``--python`` (a Python with grl_snam's closure: torch and
pycvc-gl 3.7.12+, the same libcvc as ``--host``'s, since the page rebuilds the generated world
with its own generator; a cvcpkg prefix's python gets its ``lib/`` on the loader path).

``--scrub-sets`` (default ``retrain fixed/retrain world-units/retrain``) names the learning
check's data sets: each is a directory under docs/learning-validation/ whose
``*.history.jsonl.gz`` are its runs (or ``--scrub-histories``, one set named
``--scrub-set-name``). Each run is deployed with ``grl-snam ariadne scrub HISTORY --headless
--lean --data-set SET --report``, its saved policies found in ``--checkpoints``/<set>/<run>/ (else
``--checkpoints``/<run>/; <run> is the history's name without ``.history.jsonl[.gz]``). They are
not in the repository: the ``grl-snam-learning-checkpoints`` cvcpkg package has the learning
check's under the same names (retrain/, fixed/retrain/ and world-units/retrain/, each file checked
against the histories' blobs by sha256 when it was packed; cvcpkg/recipes/
grl-snam-learning-checkpoints), and by default ``--checkpoints`` is that package in ``--python``'s
prefix. The page's Set combo picks a set, its Run combo a run of it. Every run's logs are
preloaded (the page's loader has no fetch after start; libcvc's data roadmap D4), about 3 MB a run
lean, so each five-seed set adds about 15-18 MB a plant to the first visit. They are deployed on
``--scrub-world``, which the page builds again from the report:

- ``generated`` (the default): libcvc's generator from scrub.ari's world.* keys, with no data to
  fetch, provided the page's libcvc is the deploy's;
- ``data``: a city, scrub.ari's ``data.city`` (Austin: the ``cy-pca/scene-austin-south-small``
  package, the full bundle's buildings and terrain with a smaller satellite image, 10 MB) or
  ``--scrub-city`` (a ``cvcpkg://ORG/NAME[@VERSION]`` package or an https archive). The page's data
  store fetches the version the runs were deployed on from cvcpkg.org, which lets any origin read
  a public package, so the page reaches one origin beside its own;
- ``flat``: a walled square.

``--scrub-austin`` deploys the same runs a second time, on the data world (Austin, or
``--scrub-city``), into ``--scrub-cache``/austin/, and adds that page beside the first as
scrub-austin/, with its own card on the landing page: the default page stays on its world and
fetches nothing from another origin, and Austin's 10 MB load only for those who open its page.
``--scrub-only --scrub-world data`` writes that page alone.

``--scrub-cache`` keeps the run logs between builds
(a checkpoint already deployed on the same scenario is read back). ``--scrub-reports`` takes
reports deployed before instead, and deploys nothing.

Standard library only (the deploy is a subprocess); grl_snam's files are the ones git tracks
when the repo is a git checkout, else every file under grl_snam/ but caches.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics
import struct
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE_SRC = REPO / "pages" / "site"
RETRAIN = REPO / "docs" / "learning-validation" / "retrain"
HISTORIES = sorted(RETRAIN.glob("seed*.history.jsonl.gz"))
#: The cvcpkg package with the learning check's saved policies (<set>/<run>/coef_sdf.step*.pt).
CHECKPOINTS_PACKAGE = "grl-snam-learning-checkpoints"
LEARNING = REPO / "docs" / "learning-validation"
#: The learning check's data sets the scrub page shows: directories under LEARNING whose
#: *.history.jsonl.gz are their runs, and the package's directories of the same names.
SCRUB_SETS = ("retrain", "fixed/retrain", "world-units/retrain")
#: The plants the scrub page's runs are deployed on, the page's Plant combo: the Jolt Humvee, which
#: collides with the buildings, first; nav's kinematic bicycle, which drives through them, beside it.
SCRUB_PLANTS = ("humvee", "kinematic")
SCRUB_SCHEMA = "grl_snam.scrub/v1"
#: How the scrub page opens: the windows that leave the world in view (Training, Ledger and
#: Simulation are in its Windows menu; Episode has the pause and the speed), the camera following
#: checkpoint k's car (scrub.ari's own), on the data set :func:`open_set` picks, at the checkpoint
#: whose car gets farthest from step 0's, so the two cars a visit first sees part if any do.
SCRUB_VIEW = ["--windows", "scrub,episode,arms_stats", "--at", "far"]
#: The Humvee's model the scrub page draws its cars with (the cvcpkg package vehicle-humvee's),
#: where pycvc_gl.scenes looks for it in a prefix: main.py makes the page that prefix.
VEHICLE_MODEL = "share/cvc-scenes/shared/Humvee.glb"
FIGURES = {"img/curves.png": REPO / "docs" / "learning-validation" / "curves.png"}
#: The landing page's card for the scrub page, between these marks: dropped without one.
SCRUB_CARD = ("<!-- scrub:", "<!-- /scrub -->")
#: The scrub page in Austin (--scrub-austin), beside scrub/: its directory, its card's marks and
#: picture, and its runs' place in --scrub-cache. The pages' titles.
SCRUB_AUSTIN = "scrub-austin"
SCRUB_AUSTIN_CARD = ("<!-- scrub-austin:", "<!-- /scrub-austin -->")
SCRUB_AUSTIN_IMG = "img/scrub-austin.jpg"
SCRUB_AUSTIN_CACHE = "austin"
SCRUB_TITLE = "GRL-SNAM: watch it learn"
SCRUB_AUSTIN_TITLE = "GRL-SNAM: watch it learn in Austin"
#: The CoPH page (--coph): the demos it plays back, the first the one it opens on without a
#: ?demo=; where it goes, inside the CoPH poster's directory (a site that serves coph/ as it is
#: keeps the two together); its title; the variable its ?demo= sets (grl_snam.ariadne.coph's
#: PAGE_DEMO); and the poster's links to it, between these marks, dropped without the page.
COPH_DEMOS = ("crossing", "hazard")
COPH_PAGE = "coph/live"
COPH_TITLE = "GRL-SNAM: CoPH-Stagewise in 3-D"
COPH_ENV = "GRL_SNAM_COPH_DEMO"
COPH_LIVE = ("<!-- coph-live:", "<!-- /coph-live -->")
#: ariglx-web's files, as its recipe installs them in share/ariglx-web.
HOST_FILES = ("index.html", "ariglx.js", "ariglx.wasm", "python312.zip", "arigl-web.js",
              "arigl-web.css", "LICENSE", "THIRD_PARTY_NOTICES.md")  # fmt: skip
ARMS_SCHEMA = "grl_snam.arms/v1"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def package_files() -> list[Path]:
    """grl_snam's files, relative to the repo: the ones git tracks (no caches, nothing
    untracked), or, outside a git checkout (a release tarball), every file but caches."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "grl_snam"], cwd=REPO, check=True,
                             capture_output=True).stdout  # fmt: skip
        files = [Path(p) for p in out.decode().split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:
        files = [p.relative_to(REPO) for p in (REPO / "grl_snam").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]  # fmt: skip
    return sorted(files)


def write_package(dest: Path) -> int:
    """grl_snam as a zip, entries in order with a fixed time (the same files give the same
    bytes). Deflated: the page's Python has zlib, and a server may not compress a zip."""
    files = package_files()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in files:
            info = zipfile.ZipInfo(rel.as_posix(), FIXED_TIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (REPO / rel).read_bytes())
    return len(files)


def host_page(host: Path, page: Path) -> int:
    """A page at ``page`` (made here) with ariglx-web, the launcher and the package; the number of
    the package's files."""
    page.mkdir(parents=True)
    for name in HOST_FILES:
        shutil.copy2(host / name, page / name)
    shutil.copy2(REPO / "pages" / "main.py", page / "main.py")
    return write_package(page / "grl_snam_web.zip")


def run_name(history: Path) -> str:
    """A history's run, as learn labels it: its file's name without .gz, .jsonl, .history."""
    base = Path(history).name
    for ext in (".gz", ".jsonl", ".json", ".history"):
        base = base[: -len(ext)] if base.endswith(ext) else base
    return base


def _python_env(python: str) -> dict:
    """The environment the deploy runs in: this repo's grl_snam first on the path, no user
    site-packages, and a cvcpkg prefix's lib/ (``<prefix>/bin/python``'s) on the loader path,
    after what is there already."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH")) if p)
    env["PYTHONNOUSERSITE"] = "1"
    lib = Path(python).resolve().parent.parent / "lib"
    have = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    if any(lib.glob("libcvc.so*")) and str(lib) not in have:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*have, str(lib)])
    return env


def default_checkpoints(python: str) -> Path:
    """The learning check's saved policies in the cvcpkg prefix ``python`` is in
    (``<prefix>/bin/python``): the grl-snam-learning-checkpoints package. Exits saying how to
    install it when it is not there."""
    prefix = Path(python).resolve().parent.parent
    path = prefix / "share" / CHECKPOINTS_PACKAGE
    if not path.is_dir():
        sys.exit(f"assemble: no saved policies at {path}: cvcpkg install {CHECKPOINTS_PACKAGE} "
                 f"--prefix {prefix}, or name them with --checkpoints")  # fmt: skip
    return path


def set_histories(name: str) -> list[Path]:
    """The runs of the learning check's data set ``name``: its directory's histories."""
    if not name or Path(name).is_absolute() or ".." in Path(name).parts:
        sys.exit(f"assemble: {name!r} is not a data set under {LEARNING}")
    found = sorted((LEARNING / name).glob("*.history.jsonl.gz"))
    if not found:
        sys.exit(f"assemble: no histories in {LEARNING / name} (data set {name!r})")
    return found


def run_key(set_name: str, run: str) -> str:
    """A run's name across data sets: ``retrain/seed0``, or ``seed0`` in no set."""
    return f"{set_name}/{run}" if set_name else run


def deploy_scrub(sets, checkpoints: Path, python: str, cache: Path, world: str, ticks: int,
                 plants=("humvee",), city: str | None = None) -> list[Path]:  # fmt: skip
    """Each data set's runs deployed natively (``grl-snam ariadne scrub --headless --lean
    --report``) on each of ``plants``, on ``world`` (``city``, when given, is the data world's
    city: ``--uri``): ``sets`` is ``[(name, histories)]``; the reports in
    ``cache``/<plant>/<set>/<run>.scrub.json, the run logs in ``cache``/<plant>/<set>/<run>/. The
    first plant's runs come first, so the page opens on them."""
    keys, reports = [], []
    plants = list(dict.fromkeys(plants))
    if not plants:
        sys.exit("assemble: --scrub-plants names no plant")
    for plant, (set_name, histories) in ((p, s) for p in plants for s in sets):
        for h in map(Path, histories):
            key = run_key(set_name, run_name(h))
            if (plant, key) in keys:
                sys.exit(f"assemble: two histories are run {key}")
            keys.append((plant, key))
            blobs = next((d for d in (checkpoints / key, checkpoints / run_name(h)) if d.is_dir()),
                         None)  # fmt: skip
            if blobs is None:
                sys.exit(f"assemble: no saved policies for {key}: {checkpoints / key} is not a "
                         "directory")  # fmt: skip
            report = cache / plant / f"{key}.scrub.json"
            cmd = [python, "-s", "-m", "grl_snam.ariadne", "scrub", str(h.resolve()), "--blobs",
                   str(blobs.resolve()), "--cache", str((cache / plant / key).resolve()),
                   "--headless", "--lean", "--world", world, "--ticks", str(ticks), "--plant",
                   plant, "--report", str(report.resolve())]  # fmt: skip
            if set_name:
                cmd += ["--data-set", set_name]
            if city and world == "data":
                cmd += ["--uri", city]
            print(f"assemble: scrub: deploying {key}'s checkpoints on {world}, on {plant} ({blobs})",
                  flush=True)  # fmt: skip
            report.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(cmd, env=_python_env(python), cwd=REPO, check=False)
            if r.returncode != 0 or not report.is_file():
                sys.exit(f"assemble: scrub on {key} failed (exit {r.returncode})")
            reports.append(report)
    return reports


def deploy_coph(python: str, cache: Path) -> list[Path]:
    """Each CoPH demo's run file (grl_snam.coph.runfile), baked natively: ``grl-snam ariadne
    coph DEMO --from cache/DEMO --bake cache/DEMO.coph.npz``, which computes the demo's two
    episodes there first when they are missing (side by side; a crossing arm takes about 20
    minutes, a hazard arm about 6) and reads them back on later builds."""
    runs = []
    for demo in COPH_DEMOS:
        out = cache / f"{demo}.coph.npz"
        cmd = [python, "-s", "-m", "grl_snam.ariadne", "coph", demo, "--from",
               str((cache / demo).resolve()), "--bake", str(out.resolve())]  # fmt: skip
        print(f"assemble: coph: baking {demo}'s run file from {cache / demo}", flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(cmd, env=_python_env(python), cwd=REPO, check=False)
        if r.returncode != 0 or not out.is_file():
            sys.exit(f"assemble: coph {demo} --bake failed (exit {r.returncode})")
        runs.append(out)
    return runs


def coph_page(host: Path, page: Path, runs, vehicle: Path | None = None) -> dict:
    """The CoPH page at ``page`` (made here): ``grl-snam ariadne coph`` playing back a run file
    per demo (data/<demo>.coph.npz, from :func:`deploy_coph`), opening on its walkthrough. The
    page's ``?demo=crossing`` or ``?demo=hazard`` picks the demo (the config's url_env sets
    COPH_ENV, which coph reads in a page); without it, the first of :data:`COPH_DEMOS` it has.
    ``vehicle`` ships as the scrub page's does (the cars are boxes without it)."""
    by_demo = {}
    for r in map(Path, runs):
        demo = r.name[: -len(".coph.npz")] if r.name.endswith(".coph.npz") else None
        if demo not in COPH_DEMOS:
            sys.exit(f"assemble: {r} is not a CoPH run file named <demo>.coph.npz "
                     f"({', '.join(COPH_DEMOS)})")  # fmt: skip
        with zipfile.ZipFile(r) as z:  # an .npz is a zip of .npy arrays
            if "meta.npy" not in z.namelist():
                sys.exit(f"assemble: {r} is not a run file (no meta)")
        by_demo[demo] = r
    demos = [d for d in COPH_DEMOS if d in by_demo]
    if not demos:
        sys.exit("assemble: no CoPH run files to show")
    n = host_page(host, page)
    preload = ["grl_snam_web.zip"]
    for demo in demos:
        dest = page / "data" / f"{demo}.coph.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(by_demo[demo], dest)
        preload.append(f"data/{demo}.coph.npz")
    if vehicle is not None:
        (page / VEHICLE_MODEL).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vehicle, page / VEHICLE_MODEL)
        preload.append(VEHICLE_MODEL)
        with open(page / "THIRD_PARTY_NOTICES.md", "a", encoding="utf-8") as f:
            f.write("\n" + model_credit(vehicle))
    else:
        print("assemble: coph: no Humvee model (--vehicle-model): the page's cars are boxes")
    config = {
        "title": COPH_TITLE,
        "src": "main.py",
        "src_param": False,
        "preload": preload,
        "args": ["coph", demos[0], "--from", "data/{demo}.coph.npz", "--tour"],
        "url_env": {"demo": {"env": COPH_ENV, "match": f"^({'|'.join(demos)})$"}},
    }
    (page / "ariadne.config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"assemble: {page}: grl_snam_web.zip ({n} files), run files for {', '.join(demos)}")
    return config


def check_libcvc(reports, expect: str | None) -> None:
    """Say which libcvc deployed the runs, and exit when ``expect`` (the page's, ``--host``'s) is
    given and a run was deployed on another: the page builds libcvc's generated world again with
    its own generator, so the two must be the same release."""
    used = {}
    for r in map(Path, reports):
        prov = json.loads(r.read_text()).get("provenance") or {}
        used.setdefault(str(prov.get("libcvc") or "unknown"), []).append(r.name)
    print(f"assemble: scrub: the runs were deployed on libcvc {', '.join(sorted(used))}")
    if expect is None:
        return
    expect = expect.lstrip("v")
    other = {v: names for v, names in used.items() if v != expect}
    if other:
        what = "; ".join(f"{v}: {', '.join(n[:3])}{' ...' if len(n) > 3 else ''}"
                         for v, n in sorted(other.items()))  # fmt: skip
        sys.exit(f"assemble: the page's libcvc is {expect}, but runs were deployed on another "
                 f"({what}): deploy them with pycvc-gl {expect} (pages.yml pins it to LIBCVC_REF), "
                 "or build ariglx-web from the libcvc that deployed them")  # fmt: skip


def _found(report: Path, name) -> Path | None:
    """A file a scrub report names, as grl_snam.ariadne.scrub.find_file finds it."""
    if not name:
        return None
    p = Path(name)
    here = report.resolve().parent
    for c in [p] if p.is_absolute() else [here / p, Path.cwd() / p, here / p.name]:
        if c.is_file():
            return c
    return None


def _plant_of(rep: dict) -> str:
    return str((rep.get("scenario") or {}).get("plant") or "")


def model_credit(path: Path) -> str:
    """The credit a glTF binary names in its ``asset.extras`` (Sketchfab's: title, author, source,
    license), as a THIRD_PARTY_NOTICES.md section; a plain one when it names none."""
    extras = {}
    try:
        data = path.read_bytes()
        n = struct.unpack("<I", data[12:16])[0]
        extras = json.loads(data[20 : 20 + n]).get("asset", {}).get("extras") or {}
    except (OSError, ValueError, struct.error):
        pass
    head = f"## {path.name} ({VEHICLE_MODEL})\n\nThe scrub page's cars."
    if not extras.get("license"):
        return f"{head} From the cvcpkg package vehicle-humvee.\n"
    title = f'"{extras["title"]}"' if extras.get("title") else "The model"
    by = f" by {extras['author']}" if extras.get("author") else ""
    src = f", {extras['source']}" if extras.get("source") else ""
    return (
        f"{head} {title}{by}{src}, licensed {extras['license']}; from the cvcpkg package "
        "vehicle-humvee.\n"
    )


def _in_austin(rep: dict) -> bool:
    """Whether a report's runs were deployed in Austin (one of the scene-austin-south packages)."""
    world = (rep.get("scenario") or {}).get("world") or {}
    return "/scene-austin-south" in str(world.get("canonical") or world.get("uri") or "")


def open_set(reps, plant: str) -> str:
    """The data set the scrub page opens on: of the runs deployed on ``plant``, the set whose
    checkpoints get farthest from step 0 (the median, over all its checkpoints after step 0, of
    the farthest each one's car gets from step 0's: its metric ``apart_m``), so the two cars a
    visit first sees drive apart if any do. The first set when no report has the distance."""
    far: dict[str, list[float]] = {}
    for rep in reps:
        if _plant_of(rep) != plant:
            continue
        got = far.setdefault(str(rep.get("set") or ""), [])
        for c in (rep.get("checkpoints") or [])[1:]:
            v = (c.get("metrics") or {}).get("apart_m")
            if isinstance(v, (int, float)) and math.isfinite(v):
                got.append(float(v))
    scored = {k: statistics.median(v) for k, v in far.items() if v}
    if not scored:
        return next(iter(far), "")
    best = max(far, key=lambda k: scored.get(k, -1.0))
    each = ", ".join(f"{k or '(none)'} {v:.1f} m" for k, v in scored.items())
    print(f"assemble: scrub: opens on {best or '(no set)'} (the median checkpoint's farthest "
          f"from step 0, on {plant or '?'}: {each})")  # fmt: skip
    return best


def scrub_page(host: Path, page: Path, reports, vehicle: Path | None = None) -> dict:
    """The scrub page at ``page`` (made here): the host, the launcher, the package, a bundle per
    run, data/<set>/<run>.scrub.zip (data/<run>.scrub.zip in no set; data/<plant>/<set>/<run>.scrub.zip
    when the runs were deployed on more than one plant): one zip with its report (scrub.json, its
    paths made the names inside), its history and its run logs, stored. The page's loader fetches
    at most 512 files, and a run is 43; scrub --view reads the bundles. ``vehicle``, the Humvee's
    glTF, ships at :data:`VEHICLE_MODEL` with its credit in THIRD_PARTY_NOTICES.md (without it
    the cars are boxes). The page config views them, the first plant's runs first, opening on the
    data set :func:`open_set` picks, titled for Austin when they were deployed there."""
    reports = [Path(r) for r in reports]
    if not reports:
        sys.exit("assemble: no scrub reports to show")
    reps = [json.loads(r.read_text()) for r in reports]
    plants = list(dict.fromkeys(_plant_of(rep) for rep in reps))
    n = host_page(host, page)
    args, preload, names, logs = ["scrub", "--view"], ["grl_snam_web.zip"], [], 0
    for r, rep in zip(reports, reps, strict=True):
        if rep.get("schema") != SCRUB_SCHEMA:
            sys.exit(f"assemble: {r} is {rep.get('schema')!r}, not a scrub report ({SCRUB_SCHEMA})")
        hist = _found(r, rep.get("history"))
        if hist is None:
            sys.exit(f"assemble: {r}: its history {rep.get('history')!r} is not there")
        name = run_key(str(rep.get("set") or ""), run_name(hist))
        if len(plants) > 1:
            name = f"{_plant_of(rep) or 'unknown'}/{name}"
        if name in names:
            sys.exit(f"assemble: two scrub reports are of run {name}")
        names.append(name)
        files = {hist.name: hist}
        rep["history"] = hist.name
        for c in rep.get("checkpoints") or []:
            log = _found(r, c.get("cache"))
            if log is None:
                sys.exit(f"assemble: {r}: step {c.get('step')}'s run log is not there")
            files[log.name] = log
            c["cache"], c["blob"] = log.name, Path(str(c.get("blob", ""))).name
            logs += 1
        bundle = page / "data" / f"{name}.scrub.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as z:
            entries = [("scrub.json", (json.dumps(rep, indent=1) + "\n").encode())]
            entries += [(k, f.read_bytes()) for k, f in sorted(files.items())]
            for entry, data in entries:
                info = zipfile.ZipInfo(entry, FIXED_TIME)
                info.external_attr = 0o644 << 16
                z.writestr(info, data)
        args.append(f"data/{name}.scrub.zip")
        preload.append(f"data/{name}.scrub.zip")
    if vehicle is not None:
        (page / VEHICLE_MODEL).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vehicle, page / VEHICLE_MODEL)
        preload.append(VEHICLE_MODEL)
        with open(page / "THIRD_PARTY_NOTICES.md", "a", encoding="utf-8") as f:
            f.write("\n" + model_credit(vehicle))
    else:
        print("assemble: scrub: no Humvee model (--vehicle-model): the page's cars are boxes")
    view = list(SCRUB_VIEW)
    if len({str(rep.get("set") or "") for rep in reps}) > 1:
        first = open_set(reps, plants[0])
        view += ["--data-set", first] if first else []
    config = {
        "title": SCRUB_AUSTIN_TITLE if all(map(_in_austin, reps)) else SCRUB_TITLE,
        "src": "main.py",
        "src_param": False,
        "preload": preload,
        "args": args + view,
    }
    (page / "ariadne.config.json").write_text(json.dumps(config, indent=2) + "\n")
    sets = len({str(rep.get("set") or "") for rep in reps})
    on = ", ".join(p or "?" for p in plants)
    print(f"assemble: {page}: grl_snam_web.zip ({n} files), {len(names)} runs in {sets} data "
          f"sets on {len(plants)} plants ({on}), {logs} run logs")  # fmt: skip
    return config


def learn_page(host: Path, page: Path, histories, arms: Path | None) -> dict:
    """The learn page at ``page`` (made here): the host, the launcher, the package, the data and
    the page config."""
    histories = [Path(h) for h in histories]
    if not histories:
        sys.exit("assemble: no learning histories to show")
    names = [h.name[: -len(".gz")] if h.name.endswith(".gz") else h.name for h in histories]
    if len(set(names)) != len(names):
        sys.exit(f"assemble: two histories share a name ({', '.join(names)}); learn labels a run "
                 "by its file's name")  # fmt: skip
    n = host_page(host, page)
    data = page / "data"
    data.mkdir()
    for h in histories:
        shutil.copy2(h, data / h.name)
    args = ["learn", *(f"data/{h.name}" for h in histories)]
    if arms is not None:
        schema = json.loads(arms.read_text()).get("schema")
        if schema != ARMS_SCHEMA:
            sys.exit(f"assemble: {arms} is {schema!r}, not an arms table ({ARMS_SCHEMA}) that "
                     "learn --arms reads")  # fmt: skip
        shutil.copy2(arms, data / "arms.json")
        args += ["--arms", "data/arms.json"]
    preload = ["grl_snam_web.zip", *(a for a in args if a.startswith("data/"))]
    config = {
        "title": "GRL-SNAM: did it learn?",
        "src": "main.py",
        "src_param": False,
        "preload": preload,
        "args": args,
    }
    (page / "ariadne.config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"assemble: {page}: grl_snam_web.zip ({n} files), {len(names)} histories"
          + (", arms.json" if arms else ""))  # fmt: skip
    return config


def drop_card(site: Path, card=SCRUB_CARD, picture: str | None = "img/scrub.jpg") -> None:
    """The landing page (``site``/index.html) without a page's card (and the card's picture),
    for a site without the page: the scrub page's by default."""
    index = site / "index.html"
    html = index.read_text()
    start, end = html.find(card[0]), html.find(card[1])
    if start >= 0 and end > start:
        cut = html[: html.rfind("\n", 0, start) + 1] + html[end + len(card[1]) + 1 :]
        index.write_text(cut)
    if picture:
        (site / picture).unlink(missing_ok=True)


def sizes(site: Path) -> None:
    """What a first visit to each page downloads, raw and gzip -9 (Pages compresses on the wire,
    and the browser caches)."""
    print("assemble: sizes, bytes: raw / gzip -9")
    total_raw = total_gz = 0
    for f in sorted(p for p in site.rglob("*") if p.is_file()):
        raw = f.read_bytes()
        gz = len(gzip.compress(raw, 9))
        total_raw, total_gz = total_raw + len(raw), total_gz + gz
        print(f"  {f.relative_to(site).as_posix():40} {len(raw):>11} {gz:>11}")
    print(f"  {'(all)':40} {total_raw:>11} {total_gz:>11}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", required=True, type=Path, help="ariglx-web's share/ariglx-web")
    ap.add_argument("--out", required=True, type=Path, help="the site directory (replaced)")
    ap.add_argument("--learn-only", action="store_true",
                    help="write the learn page alone at --out, not the whole site")  # fmt: skip
    ap.add_argument("--histories", nargs="+", type=Path, default=HISTORIES, metavar="H",
                    help="learning histories (.jsonl or .jsonl.gz; default: the retrain's)")  # fmt: skip
    ap.add_argument("--arms", type=Path, help=f"an arms table ({ARMS_SCHEMA}) for learn")
    ap.add_argument("--scrub", action="store_true",
                    help="add the checkpoint scrubber's page, scrub/ (deployed here)")  # fmt: skip
    ap.add_argument("--scrub-only", action="store_true",
                    help="write the scrub page alone at --out, not the whole site")  # fmt: skip
    ap.add_argument("--scrub-sets", nargs="+", default=list(SCRUB_SETS), metavar="SET",
                    help="the learning check's data sets it deploys, directories under "
                         f"docs/learning-validation/ (default: {' '.join(SCRUB_SETS)})")  # fmt: skip
    ap.add_argument("--scrub-histories", nargs="+", type=Path, default=None, metavar="H",
                    help="deploy these histories instead, as one data set (--scrub-set-name)")  # fmt: skip
    ap.add_argument(
        "--scrub-set-name",
        default="",
        metavar="NAME",
        help="the data set --scrub-histories are (default: none)",
    )
    ap.add_argument("--checkpoints", type=Path, default=None, metavar="DIR",
                    help="the saved policies: DIR/<set>/<run>/ (or DIR/<run>/) for each history "
                         f"(default: the {CHECKPOINTS_PACKAGE} package in --python's prefix)")  # fmt: skip
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="the Python that deploys them: grl_snam's closure (torch, pycvc-gl)",
    )
    ap.add_argument("--scrub-world", default="generated", choices=("generated", "flat", "data"),
                    help="the world they are deployed on (default generated)")  # fmt: skip
    ap.add_argument("--scrub-city", default=None, metavar="URI",
                    help="with --scrub-world data or --scrub-austin, the city: a "
                         "cvcpkg://ORG/NAME[@VERSION] package or an https archive the page can "
                         "fetch (default scrub.ari's, cvcpkg://cy-pca/scene-austin-south-small)")  # fmt: skip
    ap.add_argument("--scrub-austin", action="store_true",
                    help=f"with --scrub, deploy the runs on Austin too and add that page beside "
                         f"scrub/, as {SCRUB_AUSTIN}/ (its runs in --scrub-cache/"
                         f"{SCRUB_AUSTIN_CACHE}/)")  # fmt: skip
    ap.add_argument("--scrub-ticks", type=int, default=1200, help="each checkpoint's ticks")
    ap.add_argument("--scrub-plants", nargs="+", default=list(SCRUB_PLANTS), metavar="PLANT",
                    help="the vehicles they are deployed on, each a choice in the page's Plant "
                         f"combo, the first shown first (default: {' '.join(SCRUB_PLANTS)})")  # fmt: skip
    ap.add_argument("--scrub-cache", type=Path, default=None, metavar="DIR",
                    help="keep the deployed run logs here between builds (default: a temporary "
                         "directory)")  # fmt: skip
    ap.add_argument("--scrub-reports", nargs="+", type=Path, default=None, metavar="R",
                    help="scrub reports deployed before (grl_snam.scrub/v1): deploy nothing")  # fmt: skip
    ap.add_argument("--vehicle-model", type=Path, default=None, metavar="GLB",
                    help="the Humvee's model the scrub page draws its cars with (default: "
                         f"{VEHICLE_MODEL} in --python's prefix, the cvcpkg package "
                         "vehicle-humvee's; without one the cars are boxes)")  # fmt: skip
    ap.add_argument("--coph", action="store_true",
                    help=f"add the CoPH page ({COPH_PAGE}/: grl-snam ariadne coph playing back "
                         "each demo's run file, baked here by --python, which needs torch)")  # fmt: skip
    ap.add_argument(
        "--coph-only",
        action="store_true",
        help="write the CoPH page alone at --out (for another site to serve)",
    )
    ap.add_argument(
        "--coph-runs",
        nargs="+",
        type=Path,
        default=None,
        metavar="RUN",
        help="the CoPH page's run files, baked before (<demo>.coph.npz): bake nothing",
    )
    ap.add_argument("--coph-cache", type=Path, default=None, metavar="DIR",
                    help="keep the CoPH demos' episodes and run files here between builds "
                         "(default: a temporary directory)")  # fmt: skip
    ap.add_argument("--expect-libcvc", default=None, metavar="VERSION",
                    help="the libcvc --host is built from (e.g. 4.0.0): refuse runs deployed on "
                         "another, whose generated world the page would not draw the same")  # fmt: skip
    a = ap.parse_args(argv)
    if sum((a.learn_only, a.scrub_only, a.coph_only)) > 1:
        sys.exit("assemble: one of --learn-only, --scrub-only and --coph-only")
    if a.scrub_city and ((a.scrub_world != "data" and not a.scrub_austin)
                         or "://" not in a.scrub_city):  # fmt: skip
        sys.exit("assemble: --scrub-city is a data world's city, a URI the page can fetch "
                 "(cvcpkg://ORG/NAME or https://...): with --scrub-world data or --scrub-austin")  # fmt: skip
    if a.scrub_austin and (not a.scrub or a.scrub_only or a.scrub_reports is not None):
        sys.exit("assemble: --scrub-austin adds a page to the whole site (--scrub) and deploys its "
                 "runs: not with --scrub-only or --scrub-reports (--scrub-only --scrub-world data "
                 "writes that page alone)")  # fmt: skip
    missing = [n for n in HOST_FILES if not (a.host / n).is_file()]
    if missing:
        sys.exit(f"assemble: {a.host} is not ariglx-web (no {', '.join(missing)})")
    reports, austin, tmp = None, None, None
    if a.scrub or a.scrub_only:
        reports = a.scrub_reports
        if reports is None:
            cache = a.scrub_cache or Path(tmp := tempfile.mkdtemp(prefix="grl-snam-scrub-"))
            cache.mkdir(parents=True, exist_ok=True)
            checkpoints = a.checkpoints or default_checkpoints(a.python)
            sets = ([(a.scrub_set_name, a.scrub_histories)] if a.scrub_histories else
                    [(name, set_histories(name)) for name in a.scrub_sets])  # fmt: skip
            reports = deploy_scrub(sets, checkpoints, a.python, cache, a.scrub_world,
                                   a.scrub_ticks, a.scrub_plants, a.scrub_city)  # fmt: skip
            if a.scrub_austin:
                austin = deploy_scrub(sets, checkpoints, a.python, cache / SCRUB_AUSTIN_CACHE,
                                      "data", a.scrub_ticks, a.scrub_plants, a.scrub_city)  # fmt: skip
        check_libcvc(reports + (austin or []), a.expect_libcvc)
    coph_runs, coph_tmp = None, None
    if a.coph or a.coph_only:
        coph_runs = a.coph_runs
        if coph_runs is None:
            cache = a.coph_cache or Path(coph_tmp := tempfile.mkdtemp(prefix="grl-snam-coph-"))
            coph_runs = deploy_coph(a.python, cache)
    vehicle = a.vehicle_model or Path(a.python).resolve().parent.parent / VEHICLE_MODEL
    if not vehicle.is_file():
        if a.vehicle_model:
            sys.exit(f"assemble: no Humvee model at {vehicle}")
        vehicle = None
    if a.out.exists():
        shutil.rmtree(a.out)
    try:
        if a.learn_only:
            learn_page(a.host, a.out, a.histories, a.arms)
        elif a.scrub_only:
            scrub_page(a.host, a.out, reports, vehicle)
        elif a.coph_only:
            coph_page(a.host, a.out, coph_runs, vehicle)
        else:
            shutil.copytree(SITE_SRC, a.out)
            for dest, src in FIGURES.items():
                (a.out / dest).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, a.out / dest)
            (a.out / ".nojekyll").touch()  # served as is: no Jekyll
            learn_page(a.host, a.out / "learn", a.histories, a.arms)
            if reports is not None:
                scrub_page(a.host, a.out / "scrub", reports, vehicle)
            else:
                drop_card(a.out)
            if austin is not None:
                scrub_page(a.host, a.out / SCRUB_AUSTIN, austin, vehicle)
            else:
                drop_card(a.out, SCRUB_AUSTIN_CARD, SCRUB_AUSTIN_IMG)
            if coph_runs is not None:
                coph_page(a.host, a.out / COPH_PAGE, coph_runs, vehicle)
            else:  # the poster without its links to a page the site does not have
                drop_card(a.out / "coph", COPH_LIVE, picture=None)
    finally:
        for t in (tmp, coph_tmp):
            if t:
                shutil.rmtree(t, ignore_errors=True)
    sizes(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
