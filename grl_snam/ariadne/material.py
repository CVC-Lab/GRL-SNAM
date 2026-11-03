"""The material-aware navigation demo (``grl-snam material-demo``) in an Ariadne document.

    python -m grl_snam.ariadne material
    python -m grl_snam.ariadne material --offscreen --frames 600 --png material.png
    python -m grl_snam.ariadne material --headless --swarm-steps 120

The same two stories as ``grl_snam.tools.material_demo``, each run with material off and on:
the fog scenario (belief + planner + drive) around a mud field and through a dry gap in a water
strip, and the reactive swarm (no planner) detouring around mud and a lake. Here they run as
async functions on the app's background pool when the document starts (and again on its Run
buttons), so the window draws a spinner while torch works; each result is a picture (terrain,
mud, water and both runs' tracks, drawn with numpy instead of matplotlib) and a table of the
off/on numbers: arrivals, mud exposure, and steps on hazard cells.
"""

from __future__ import annotations

import sys

import numpy as np

from grl_snam.ariadne import _host

OFF, ON = (192, 57, 43), (30, 132, 73)  # the original's colours


def _args(p):
    p.add_argument("--swarm-steps", type=int, default=700, help="the swarm story's step budget")


def picture(risk, hard, runs, bounds, scale=4):
    """Terrain + trajectories as an RGB array (row 0 north). runs = [(positions (N, 2), rgb)]."""
    n = risk.shape[0]
    img = np.ones((n, n, 3)) * np.array([0.93, 0.93, 0.89])
    mud = np.clip(risk, 0, 1)[..., None]
    img = img * (1 - mud) + mud * np.array([0.45, 0.30, 0.16])
    img[hard.astype(bool)] = (0.28, 0.43, 0.78)
    img = np.kron((img * 255).astype(np.uint8), np.ones((scale, scale, 1), np.uint8))
    size = n * scale
    for pts, color in runs:
        c = ((pts[:, 0] - bounds[0]) / (bounds[2] - bounds[0]) * (size - 1)).round().astype(int)
        r = ((pts[:, 1] - bounds[1]) / (bounds[3] - bounds[1]) * (size - 1)).round().astype(int)
        ok = (c >= 0) & (c < size) & (r >= 0) & (r < size)
        img[r[ok], c[ok]] = color
    return img[::-1]  # row 0 at max y, the top of the picture


def _rows(label, plain, mat, of=""):
    return [
        {"what": "reached", "off": f"{plain[1]}{of}", "on": f"{mat[1]}{of}"},
        {"what": "mud exposure", "off": f"{plain[2]:.1f}", "on": f"{mat[2]:.1f}"},
        {"what": "hazard steps", "off": str(plain[3]), "on": str(mat[3])},
    ]


def main(argv=None) -> int:
    args = _host.parse(argv, _args)
    from grl_snam.tools import material_demo as md

    host = _host.open_host("material.ari", "material", args, load=False)
    ready = {}  # a story's finished results, for the next frame to publish

    # The runs take seconds of torch, so they are async functions: each runs on the app's
    # background pool while its program waits and the window keeps drawing. A picture is
    # published on the frame's thread (below), never from the pool.
    @host.async_function("run-scenario")
    def run_scenario():
        risk, hard, plain, mat = md.scenario_runs()
        ready["scenario"] = (
            picture(risk, hard, [(plain[0], OFF), (mat[0], ON)], md.BOUNDS),
            _rows("scenario", plain, mat),
        )
        return "done"

    @host.async_function("run-swarm")
    def run_swarm():
        from grl_snam.fog_stories import STORIES

        risk, hard, plain, mat = md.swarm_runs(steps=args.swarm_steps)
        ready["swarm"] = (
            picture(risk, hard, [(plain[0], OFF), (mat[0], ON)], STORIES["city"].bounds, scale=6),
            _rows("swarm", plain, mat, of=" / 6"),
        )
        return "done"

    @host.on_frame
    def publish(dt):
        for story in list(ready):
            img, rows = ready.pop(story)
            host.publish_image(story, img)
            host.set_table(story, rows)

    host.load(str(_host.DOCS / "material.ari"))  # now that what it requires is registered
    for story in ("scenario", "swarm"):  # as the Run buttons do
        host.run_program(
            f'(begin (state-set "{story}_status" "running") '
            f'(state-set "{story}_status" (run-{story})))'
        )
    if args.mode != "window":  # run until both are back, then the asked-for frames
        host.run(
            until=lambda: host.get("scenario_status") == "done"
            and host.get("swarm_status") == "done",
            realtime=False,
            frames=10**9,
        )
    return _host.finish(host, args)


if __name__ == "__main__":
    sys.exit(main())
