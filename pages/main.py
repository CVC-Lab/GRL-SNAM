"""The Pages site's program: one of GRL-SNAM's Ariadne ports, in the browser on libcvc's ariglx-web.

A page's ``ariadne.config.json`` runs this file in place of a document (``"src": "main.py"``),
with the port and its arguments as the config's ``args`` (``["learn", "data/seed0.history.jsonl",
...]``), and preloads ``grl_snam_web.zip`` and the port's data beside it. The page writes those
files into its file system at their URL paths, so they sit in this file's directory.

grl_snam ships as files, not as a wheel: its package metadata requires torch, which has no
browser build. The zip holds the package with its ``.ari`` documents, and is unpacked here because
the document loader reads files, not zip members. Relative arguments are relative to this file's
directory.

The port's Host adopts the page's (grl_snam.ariadne._host's browser mode) and returns once its
document is loaded; the page then runs the frames. A port that fails to start exits non-zero,
which the page shows as its error screen. On stderr (the browser's console) it says how long the
port took to start and, once the page runs its frames, that it is running.
"""

import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv) -> int:
    t0 = time.perf_counter()
    with zipfile.ZipFile(os.path.join(HERE, "grl_snam_web.zip")) as z:
        z.extractall(HERE)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    os.chdir(HERE)
    from grl_snam.ariadne.__main__ import main as port

    code = port(argv)
    name = argv[0] if argv else "?"
    print(f"grl-snam: {name}: started in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    try:
        from pycvc_gl.ariadne import current_host
    except ImportError:  # no pycvc_gl: a port that needs none (--list)
        return code
    host = current_host() if code == 0 else None
    if host is not None:
        frames = []

        def running(dt):
            frames.append(dt)
            if len(frames) == 3:
                print(f"grl-snam: {name}: running", file=sys.stderr)

        host.on_frame(running)
    return code


if __name__ == "__main__":
    code = main(sys.argv[1:])
    if code:
        raise SystemExit(code)
