#!/usr/bin/env python3
"""Assemble the GRL-SNAM Pages site: the landing page, and the ports that run in the browser.

    python pages/assemble.py --host OUT/share/ariglx-web --out _site [--arms TABLE]

``--host`` is libcvc's ariglx-web (pages/build-ariglx-web.sh). The site:

    index.html, style.css, img/       the landing page (pages/site/), and its figures
    learn/                            grl-snam ariadne learn, on ariglx-web:
      index.html ariglx.js ariglx.wasm python312.zip arigl-web.js arigl-web.css
      LICENSE THIRD_PARTY_NOTICES.md    the host, as its recipe installs it
      main.py                         the program the page runs (pages/main.py)
      grl_snam_web.zip                grl_snam's files, its .ari documents with them
      data/seed{0..4}.history.jsonl   the learning check's retrain histories, unzipped
      ariadne.config.json             src main.py, every file above in its preload, the args

Everything the page fetches is on the site's own origin. The histories are unzipped here because
learn reads them as text. ``--arms`` adds an arms table (``grl_snam.arms/v1``, what ``grl-snam
arms-eval`` writes) as ``data/arms.json`` and learn's ``--arms``; any other schema is refused,
since learn would refuse it in the page. Standard library only.
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
HISTORIES = REPO / "docs" / "learning-validation" / "retrain"
FIGURES = {"img/curves.png": REPO / "docs" / "learning-validation" / "curves.png"}
#: ariglx-web's files, as its recipe installs them in share/ariglx-web.
HOST_FILES = ("index.html", "ariglx.js", "ariglx.wasm", "python312.zip", "arigl-web.js",
              "arigl-web.css", "LICENSE", "THIRD_PARTY_NOTICES.md")  # fmt: skip
ARMS_SCHEMA = "grl_snam.arms/v1"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def package_files() -> list[Path]:
    """grl_snam's tracked files (no caches, nothing untracked), relative to the repo."""
    out = subprocess.run(["git", "ls-files", "-z", "grl_snam"], cwd=REPO, check=True,
                         capture_output=True).stdout  # fmt: skip
    return sorted(Path(p) for p in out.decode().split("\0") if p)


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


def learn_page(host: Path, page: Path, arms: Path | None) -> dict:
    page.mkdir(parents=True)
    for name in HOST_FILES:
        shutil.copy2(host / name, page / name)
    shutil.copy2(REPO / "pages" / "main.py", page / "main.py")
    n = write_package(page / "grl_snam_web.zip")
    data = page / "data"
    data.mkdir()
    hists = []
    for gz in sorted(HISTORIES.glob("seed*.history.jsonl.gz")):
        name = gz.name[: -len(".gz")]
        with gzip.open(gz, "rb") as f:
            (data / name).write_bytes(f.read())
        hists.append(f"data/{name}")
    if not hists:
        sys.exit(f"assemble: no histories in {HISTORIES}")
    args = ["learn", *hists]
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
    print(f"assemble: learn/: grl_snam_web.zip ({n} files), {len(hists)} histories"
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
    ap.add_argument("--arms", type=Path, help=f"an arms table ({ARMS_SCHEMA}) for learn")
    a = ap.parse_args(argv)
    missing = [n for n in HOST_FILES if not (a.host / n).is_file()]
    if missing:
        sys.exit(f"assemble: {a.host} is not ariglx-web (no {', '.join(missing)})")
    if a.out.exists():
        shutil.rmtree(a.out)
    shutil.copytree(SITE_SRC, a.out)
    for dest, src in FIGURES.items():
        (a.out / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, a.out / dest)
    (a.out / ".nojekyll").touch()  # served as is: no Jekyll
    learn_page(a.host, a.out / "learn", a.arms)
    sizes(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
