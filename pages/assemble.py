#!/usr/bin/env python3
"""Assemble the GRL-SNAM Pages site: the landing page, and the ports that run in the browser.

    python pages/assemble.py --host HOST --out _site [--histories H...] [--arms TABLE]
    python pages/assemble.py --host HOST --out DIR --learn-only [--histories H...] [--arms TABLE]
    python pages/assemble.py --host HOST --out _site --scrub --python PY [--scrub-sets SET...] ...
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
      data/<run>/scrub.json           a run's scrub report (grl_snam.scrub/v1), its paths made
                                      the file names beside it
      data/<run>/<history>            the run's learning history
      data/<run>/<sha>-<hash>.npz     each checkpoint's run log (grl_snam.run_log/v1)
      ariadne.config.json             ["scrub", "--view", "data/<run>/scrub.json", ...]

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
with its own generator; a cvcpkg prefix's python gets its ``lib/`` on the loader path). Each of
``--scrub-sets`` (default ``retrain fixed/retrain``) names the learning check's data sets: each is
a directory under docs/learning-validation/ whose ``*.history.jsonl.gz`` are its runs (or
``--scrub-histories``, one set named ``--scrub-set-name``). Each run is deployed with
``grl-snam ariadne scrub HISTORY --headless --lean --data-set SET --report``, its saved policies
found in ``--checkpoints``/<set>/<run>/ (else ``--checkpoints``/<run>/; <run> is the history's
name without ``.history.jsonl[.gz]``). They are not in the repository: the
``grl-snam-learning-checkpoints`` cvcpkg package has the learning check's under the same names
(its retrain/ and fixed/retrain/, each file checked against the histories' blobs by sha256 when it
was packed; cvcpkg/recipes/grl-snam-learning-checkpoints), and by default ``--checkpoints`` is
that package in ``--python``'s prefix. The page's Set combo picks a set, its Run combo a run of it.
Every run's logs are preloaded (the page's loader has no fetch after start; libcvc's data roadmap
D4), about 5 MB a run lean, so each five-seed set adds about 25 MB to the first visit. They are
deployed on ``--scrub-world`` (default generated:
libcvc's generator from scrub.ari's world.* keys, which the page builds again from the report
with no data to fetch; Austin's bundle is 228 MB and cvcpkg.org sends no CORS for github.io, so a
data world's page shows the windows alone). ``--scrub-cache`` keeps the run logs between builds
(a checkpoint already deployed on the same scenario is read back). ``--scrub-reports`` takes
reports deployed before instead, and deploys nothing.

Standard library only (the deploy is a subprocess); grl_snam's files are the ones git tracks
when the repo is a git checkout, else every file under grl_snam/ but caches.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
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
SCRUB_SETS = ("retrain", "fixed/retrain")
SCRUB_SCHEMA = "grl_snam.scrub/v1"
#: How the scrub page opens: the world seen from above, and the windows that leave it in view
#: (Training, Ledger and Simulation are in its Windows menu).
SCRUB_VIEW = ["--windows", "scrub,episode,arms_stats", "--camera", "orbit"]
FIGURES = {"img/curves.png": REPO / "docs" / "learning-validation" / "curves.png"}
#: The landing page's card for the scrub page, between these marks: dropped without one.
SCRUB_CARD = ("<!-- scrub:", "<!-- /scrub -->")
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


def deploy_scrub(sets, checkpoints: Path, python: str, cache: Path, world: str,
                 ticks: int) -> list[Path]:  # fmt: skip
    """Each data set's runs deployed natively (``grl-snam ariadne scrub --headless --lean
    --report``): ``sets`` is ``[(name, histories)]``; the reports in ``cache``/<set>/<run>.scrub.json,
    the run logs in ``cache``/<set>/<run>/."""
    keys, reports = [], []
    for set_name, histories in sets:
        for h in map(Path, histories):
            key = run_key(set_name, run_name(h))
            if key in keys:
                sys.exit(f"assemble: two histories are run {key}")
            keys.append(key)
            blobs = next((d for d in (checkpoints / key, checkpoints / run_name(h)) if d.is_dir()),
                         None)  # fmt: skip
            if blobs is None:
                sys.exit(f"assemble: no saved policies for {key}: {checkpoints / key} is not a "
                         "directory")  # fmt: skip
            report = cache / f"{key}.scrub.json"
            cmd = [python, "-s", "-m", "grl_snam.ariadne", "scrub", str(h.resolve()), "--blobs",
                   str(blobs.resolve()), "--cache", str((cache / key).resolve()), "--headless",
                   "--lean", "--world", world, "--ticks", str(ticks), "--report",
                   str(report.resolve())]  # fmt: skip
            if set_name:
                cmd += ["--data-set", set_name]
            print(
                f"assemble: scrub: deploying {key}'s checkpoints on {world} ({blobs})", flush=True
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(cmd, env=_python_env(python), cwd=REPO, check=False)
            if r.returncode != 0 or not report.is_file():
                sys.exit(f"assemble: scrub on {key} failed (exit {r.returncode})")
            reports.append(report)
    return reports


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


def scrub_page(host: Path, page: Path, reports) -> dict:
    """The scrub page at ``page`` (made here): the host, the launcher, the package, each report
    with its history and run logs in data/<set>/<run>/ (data/<run>/ in no set; the report's paths
    made their file names), and the page config, which views them."""
    reports = [Path(r) for r in reports]
    if not reports:
        sys.exit("assemble: no scrub reports to show")
    n = host_page(host, page)
    args, preload, names, logs = ["scrub", "--view"], ["grl_snam_web.zip"], [], 0
    for r in reports:
        rep = json.loads(r.read_text())
        if rep.get("schema") != SCRUB_SCHEMA:
            sys.exit(f"assemble: {r} is {rep.get('schema')!r}, not a scrub report ({SCRUB_SCHEMA})")
        hist = _found(r, rep.get("history"))
        if hist is None:
            sys.exit(f"assemble: {r}: its history {rep.get('history')!r} is not there")
        name = run_key(str(rep.get("set") or ""), run_name(hist))
        if name in names:
            sys.exit(f"assemble: two scrub reports are of run {name}")
        names.append(name)
        out = page / "data" / name
        out.mkdir(parents=True)
        shutil.copy2(hist, out / hist.name)
        rep["history"] = hist.name
        for c in rep.get("checkpoints") or []:
            log = _found(r, c.get("cache"))
            if log is None:
                sys.exit(f"assemble: {r}: step {c.get('step')}'s run log is not there")
            shutil.copy2(log, out / log.name)
            c["cache"], c["blob"] = log.name, Path(str(c.get("blob", ""))).name
            preload.append(f"data/{name}/{log.name}")
            logs += 1
        (out / "scrub.json").write_text(json.dumps(rep, indent=1) + "\n")
        args.append(f"data/{name}/scrub.json")
        preload += [f"data/{name}/scrub.json", f"data/{name}/{hist.name}"]
    config = {
        "title": "GRL-SNAM: watch it learn",
        "src": "main.py",
        "src_param": False,
        "preload": preload,
        "args": args + SCRUB_VIEW,
    }
    (page / "ariadne.config.json").write_text(json.dumps(config, indent=2) + "\n")
    sets = len({x.rpartition("/")[0] for x in names})
    print(f"assemble: {page}: grl_snam_web.zip ({n} files), {len(names)} runs in {sets} data "
          f"sets, {logs} run logs")  # fmt: skip
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


def drop_card(site: Path) -> None:
    """The landing page without its scrub card (and the card's picture), for a site with no
    scrub page."""
    index = site / "index.html"
    html = index.read_text()
    start, end = html.find(SCRUB_CARD[0]), html.find(SCRUB_CARD[1])
    if start >= 0 and end > start:
        cut = html[: html.rfind("\n", 0, start) + 1] + html[end + len(SCRUB_CARD[1]) + 1 :]
        index.write_text(cut)
    (site / "img" / "scrub.jpg").unlink(missing_ok=True)


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
    ap.add_argument("--scrub-ticks", type=int, default=1200, help="each checkpoint's ticks")
    ap.add_argument("--scrub-cache", type=Path, default=None, metavar="DIR",
                    help="keep the deployed run logs here between builds (default: a temporary "
                         "directory)")  # fmt: skip
    ap.add_argument("--scrub-reports", nargs="+", type=Path, default=None, metavar="R",
                    help="scrub reports deployed before (grl_snam.scrub/v1): deploy nothing")  # fmt: skip
    a = ap.parse_args(argv)
    if a.learn_only and a.scrub_only:
        sys.exit("assemble: --learn-only or --scrub-only, not both")
    missing = [n for n in HOST_FILES if not (a.host / n).is_file()]
    if missing:
        sys.exit(f"assemble: {a.host} is not ariglx-web (no {', '.join(missing)})")
    reports, tmp = None, None
    if a.scrub or a.scrub_only:
        reports = a.scrub_reports
        if reports is None:
            cache = a.scrub_cache or Path(tmp := tempfile.mkdtemp(prefix="grl-snam-scrub-"))
            cache.mkdir(parents=True, exist_ok=True)
            checkpoints = a.checkpoints or default_checkpoints(a.python)
            sets = ([(a.scrub_set_name, a.scrub_histories)] if a.scrub_histories else
                    [(name, set_histories(name)) for name in a.scrub_sets])  # fmt: skip
            reports = deploy_scrub(sets, checkpoints, a.python, cache,
                                   a.scrub_world, a.scrub_ticks)  # fmt: skip
    if a.out.exists():
        shutil.rmtree(a.out)
    try:
        if a.learn_only:
            learn_page(a.host, a.out, a.histories, a.arms)
        elif a.scrub_only:
            scrub_page(a.host, a.out, reports)
        else:
            shutil.copytree(SITE_SRC, a.out)
            for dest, src in FIGURES.items():
                (a.out / dest).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, a.out / dest)
            (a.out / ".nojekyll").touch()  # served as is: no Jekyll
            learn_page(a.host, a.out / "learn", a.histories, a.arms)
            if reports is not None:
                scrub_page(a.host, a.out / "scrub", reports)
            else:
                drop_card(a.out)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    sizes(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
