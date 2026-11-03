"""GRL-SNAM's live demos (:mod:`grl_snam.demos`) as Python hosts of an Ariadne document.

    python -m grl_snam.ariadne live austin-freedrive     # the learned SDF policy, no route
    python -m grl_snam.ariadne live austin-learned       # A* spine + learned local control
    python -m grl_snam.ariadne live austin-patrol        # grounded A* patrol (baseline)
    python -m grl_snam.ariadne live austin-planner       # the surrogate planner (needs vtkmodules)
    python -m grl_snam.ariadne live austin-freedrive --offscreen --frames 300 --png drive.png

Each demo runs unchanged: its ``setup()`` builds the scene and its ``step(dt)`` drives it, through
:class:`~grl_snam.ariadne._host.DemoHost` instead of VolRover3 or a bare window. The document
(``docs/live.ari``) adds what those hosts did not have: the navigator's HUD as panels (its
NavMetrics, a clearance and a speed plot), pause / sim-speed / camera controls, and a soft
afternoon light with shadows. On Austin the vehicle is a Humvee at its real size (the
``vehicle-humvee`` package's model, else a box its size) in place of the demos' box.

Data: the document names the city. ``docs/live.ari``'s data: block points at the Austin scene
package on cvcpkg.org, which the host downloads once (later runs read the cache), unpacks and
decodes. The scene draws it, and the demo takes the same objects (``DemoHost.city``) for its
terrain, occupancy and SDF, so no bundle is read twice. ``--bundle DIR`` (or
``GRL_SNAM_SCENE_BUNDLE``) points the document's city at a local bundle instead, with no
download, and an installed bundle stands in when the package cannot load. The learned demos also
need ``grl-snam-weights``, as ``grl-snam demo`` does. ``lab`` stays a VolRover3 demo (it drives
VolRover3's own scene).
"""

from __future__ import annotations

import importlib
import sys

from grl_snam.ariadne import _host

DEMOS = {
    "austin-freedrive": ("grl_snam.demos.austin_freedrive", "the learned SDF navigator, no route"),
    "austin-learned": ("grl_snam.demos.austin_learned", "an A* spine + learned SDF control"),
    "austin-patrol": ("grl_snam.demos.austin_patrol", "a grounded A* street patrol"),
    "austin-planner": ("grl_snam.demos.austin_planner", "the surrogate planner, sparse obstacles"),
}
HUMVEE_DEMOS = ("austin-freedrive", "austin-learned", "austin-patrol")


def _args(p):
    p.add_argument("demo", choices=sorted(DEMOS), help="which live demo")
    p.add_argument(
        "--bundle", default=None, help="a local Austin bundle for the city (default: the package)"
    )


def main(argv=None) -> int:
    args = _host.parse(argv, _args)
    from grl_snam.demos import _common

    module = importlib.import_module(DEMOS[args.demo][0])
    if args.mode == "headless":  # before the document, whose city would start loading
        raise RuntimeError(
            "the live demos draw into the scene: run them in a window or --offscreen"
        )
    host = _host.open_host("live.ari", "live", args, load=False)
    # The planner drives a world of its own, so the document's city stays unloaded for it.
    _host.load_document(host, "live.ari", bundle=args.bundle, city=args.demo in HUMVEE_DEMOS)
    host.set("demo.title", f"{args.demo}: {DEMOS[args.demo][1]}")
    demo_host = _host.DemoHost(host)
    series = {"clearance": [], "speed": []}

    @host.watch("view.camera")
    def _camera(mode):
        if mode == "orbit" and host.mode != "headless":
            import pycvc_gl

            host.camera.setMode(pycvc_gl.CameraController.Mode_Orbit)

    _common._STANDALONE = demo_host  # current_host() hands the demo this host
    try:
        module.setup()
        if args.demo in HUMVEE_DEMOS:  # the demo's box -> a Humvee, posed by the same VehiclePose
            model = _host.humvee_model(_host.bundle_or_none(args.bundle), host=host)
            _host.add_vehicle(host, "agent0", model)
        # The published policy's metric (scale 1/300) lets it drive at up to 270 m/s on Austin,
        # so the chase sits well back and catches up quickly.
        host.set_many(
            {
                "back": 40.0,
                "height": 16.0,
                "look_ahead": 8.0,
                "look_up": 2.0,
                "cam_tau": 0.04,
                "pos_tau": 0.03,
            },
            prefix="viewers.main.camera.track",
        )
        demo_host.run(module.step)  # the demo's step, every frame

        @host.on_frame
        def _plots(dt):  # after it: the readouts it published, as series
            for key, field in (("clearance", "nav.clearance_m"), ("speed", "nav.speed_mps")):
                v = host.get_float(field)
                if v == v:  # not NaN
                    series[key] = (series[key] + [v])[-240:]
                    host.set_series(f"series.{key}", series[key])

        return _host.finish(host, args)
    finally:
        _common._STANDALONE = None
        if not host.should_close():
            host.close()


if __name__ == "__main__":
    sys.exit(main())
