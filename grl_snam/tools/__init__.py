"""grl_snam.tools — the importable core of the command-line pipeline.

Each module holds the real logic (previously buried in a ``scripts/*.py`` ``main()``)
as a plain function the Click CLI (:mod:`grl_snam.cli`) and other code can call:

* :func:`grl_snam.tools.obstacles.extract` — geometry bundle -> circular obstacles.
* :func:`grl_snam.tools.sdf.build` — geometry bundle -> navigation SDF (``nav_sdf.npz``).
* :func:`grl_snam.tools.train.train_sdf` — self-supervised SDF-coefficient training.
* :func:`grl_snam.tools.capture` — drive the learned policy and render an mp4 (HUD).
* :func:`grl_snam.tools.pipeline.run` — the whole world-model -> train -> video pipeline.

Heavy, environment-specific deps (``pycvc_gl``, VTK) are imported lazily inside the
functions so importing this package never requires the compiled bindings.
"""
