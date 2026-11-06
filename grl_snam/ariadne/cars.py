"""grl-snam ariadne cars: N Ackermann cars driven by your own controller, on libcvc's template.

    grl-snam ariadne cars                                    # Austin, 6 Humvees, the template's controller
    grl-snam ariadne cars --copy my_cars                     # my_cars/cars.py + cars.ari, to make yours
    grl-snam ariadne cars --controller navigator --grl-policy   # baseline: GRL-SNAM's trained policy
    grl-snam ariadne cars --headless --frames 1200 --log run.npz

The template is libcvc's, ``pycvc_gl.cars`` (pycvc-gl 3.7.12+), and not GRL-SNAM specific. A
``controller()`` of yours gets every car's state each tick as numpy arrays and returns (steer,
accel) per car. The cars run on nav's kinematic bicycle, Jolt vehicles or the dynamic single
track (the state-tree setting ``nav_plant.vehicle``, live). They drive in Austin (the default:
the document's ``data:`` block, ``scene-austin-south-small`` from cvcpkg.org), a world from
libcvc's terrain generator, or a flat square. The document has panels, rasters, sliders bound
to the navigator's settings, and headless batch runs. libcvc's docs/CARS.md describes it all;
every flag here is the template's.

GRL-SNAM adds ``--grl-policy``: libcvc's navigator (GRL-SNAM's navigator, in C++) then drives
with the published ``grl-snam-weights`` policy, exported to ``.cvcnav`` and run in the frame it
was trained in. Without it, ``--controller navigator`` uses libcvc's built-in policy.
"""

from __future__ import annotations

import os
import sys
import tempfile

CARS_DOC = "https://github.com/transfix/libcvc/blob/master/docs/CARS.md"


def grl_policy() -> tuple[str, float]:
    """``(path, scale)``: the published policy (grl-snam-weights' coef_sdf.pt) as a ``.cvcnav``
    file in a temporary directory, and the frame it was trained in (normalized units per metre)."""
    import torch

    import grl_snam.sdf_nav as sdf_nav
    from grl_snam.demos._common import default_nav_weights_pt
    from grl_snam.tools.coef_export import write_coef_mlp

    path = default_nav_weights_pt()
    if not path or not os.path.exists(path):
        raise RuntimeError(
            "--grl-policy needs the published policy: cvcpkg install grl-snam-weights"
        )
    ck = torch.load(path, map_location="cpu")
    model = sdf_nav.CoefMLP()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    out = os.path.join(tempfile.mkdtemp(prefix="grl-snam-policy-"), "coef_sdf.cvcnav")
    write_coef_mlp(model, out)
    return out, float(ck.get("meta", {}).get("scale", 1.0 / 300.0))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        from pycvc_gl import cars, cars_template
    except ImportError:
        print(
            "grl-snam ariadne cars runs libcvc's multi-car template, pycvc_gl.cars, which needs "
            "pycvc-gl 3.7.12+ (cvcpkg install pycvc-gl-cp312, or your Python's column). "
            f"See {CARS_DOC}",
            file=sys.stderr,
        )
        return 2
    if "--grl-policy" in argv:
        argv.remove("--grl-policy")
        path, scale = grl_policy()
        argv += ["--nav-weights", path, "--nav-scale", repr(scale)]
    return cars.run(
        cars_template.controller, cars_template.PARAMS, cost=cars_template.cost, argv=argv
    )


if __name__ == "__main__":
    sys.exit(main())
