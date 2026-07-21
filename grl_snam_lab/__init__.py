"""grl_snam_lab — a general GRL-SNAM visualization laboratory.

A thin, **domain-general** layer over the CVC graphics stack (``pycvc`` +
``pycvc_gl``, the SWIG bindings for libcvc / cvcGL). It speaks the vocabulary
of GRL-SNAM — *terrain*, *obstacles*, *agent paths*, *agents*, and *scalar
fields* (risk / SDF / path-loss) — and turns each into a live 3D scene node.
It knows nothing about any specific mission or dataset.

Downstream projects adapt it by feeding their own data through this API — e.g.
a DBG loader reads a ``movement_bundle.v1`` and calls :meth:`Lab.add_path`
once per soldier track. Keep dataset-specific code in the downstream project;
keep this package general.

The graphics backend (``pycvc``/``pycvc_gl``) is imported lazily so the module
and its pure-Python geometry helpers import even where the compiled bindings
are not installed; the display/scene methods require them.

Example::

    from grl_snam_lab import Lab
    lab = Lab()
    lab.add_terrain(heights, bounds=(-100, -100, 100, 100))
    lab.add_mesh("bldg", verts, faces, color=(0.6, 0.6, 0.6))
    lab.add_path("agent0", [(x0, y0, z0), (x1, y1, z1), ...])
    lab.add_field("risk", risk_grid, bounds=(-100, -100, 0, 100, 100, 20))
    lab.show()                       # or lab.render_png("scene.png")
"""

from __future__ import annotations

from .lab import Lab, terrain_mesh

__all__ = ["Lab", "terrain_mesh"]
__version__ = "0.1.0"
