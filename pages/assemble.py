#!/usr/bin/env python3
"""Assemble the GRL-SNAM Pages site: the landing page, and the ports that run in the browser.

    python pages/assemble.py --host HOST --out _site [--histories H...] [--arms TABLE]
    python pages/assemble.py --host HOST --out DIR --learn-only [--histories H...] [--arms TABLE]

``--host`` is libcvc's ariglx-web, the ``share/ariglx-web`` directory pages/build-ariglx-web.sh
installs. The site:

    index.html, style.css, img/       the landing page (pages/site/), and its figures
    learn/                            grl-snam ariadne learn, on ariglx-web:
      index.html ariglx.js ariglx.wasm python312.zip arigl-web.js arigl-web.css
      LICENSE THIRD_PARTY_NOTICES.md    the host, as its recipe installs it
      main.py                         the program the page runs (pages/main.py)
      grl_snam_web.zip                grl_snam's files, its .ari documents with them
      data/<name>.history.jsonl       the learning histories, unzipped
      ariadne.config.json             src main.py, every file above in its preload, the args

``--learn-only`` writes the learn page alone, self-contained, at ``--out`` (for another site to
serve, at any path). Everything the page fetches is beside it, on the site's own origin.

``--histories`` names the learning histories the page shows (``.jsonl``, or ``.jsonl.gz``, which
is unzipped here because learn reads text); by default the learning check's retrain,
docs/learning-validation/retrain/seed*.history.jsonl.gz. A run is labelled by its file's stem,
so their names must differ. ``--arms`` adds an arms table (``grl_snam.arms/v1``, what ``grl-snam
arms-eval`` writes) as ``data/arms.json`` and learn's ``--arms``; any other schema is refused,
since learn would refuse it in the page. Standard library only; grl_snam's files are the ones git
tracks when the repo is a git checkout, else every file under grl_snam/ but caches.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE_SRC = REPO / "pages" / "site"
RETRAIN = REPO / "docs" / "learning-validation" / "retrain"
HISTORIES = sorted(RETRAIN.glob("seed*.history.jsonl.gz"))
FIGURES = {"img/curves.png": REPO / "docs" / "learning-validation" / "curves.png"}
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
    page.mkdir(parents=True)
    for name in HOST_FILES:
        shutil.copy2(host / name, page / name)
    shutil.copy2(REPO / "pages" / "main.py", page / "main.py")
    n = write_package(page / "grl_snam_web.zip")
    data = page / "data"
    data.mkdir()
    for h, name in zip(histories, names):
        if h.name.endswith(".gz"):
            with gzip.open(h, "rb") as f:
                (data / name).write_bytes(f.read())
        else:
            shutil.copy2(h, data / name)
    args = ["learn", *(f"data/{name}" for name in names)]
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
    a = ap.parse_args(argv)
    missing = [n for n in HOST_FILES if not (a.host / n).is_file()]
    if missing:
        sys.exit(f"assemble: {a.host} is not ariglx-web (no {', '.join(missing)})")
    if a.out.exists():
        shutil.rmtree(a.out)
    if a.learn_only:
        learn_page(a.host, a.out, a.histories, a.arms)
    else:
        shutil.copytree(SITE_SRC, a.out)
        for dest, src in FIGURES.items():
            (a.out / dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, a.out / dest)
        (a.out / ".nojekyll").touch()  # served as is: no Jekyll
        learn_page(a.host, a.out / "learn", a.histories, a.arms)
    sizes(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
