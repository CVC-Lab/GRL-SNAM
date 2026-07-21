# grl_snam_lab — a general GRL-SNAM visualization laboratory

A domain-general 3D lab for GRL-SNAM built on the CVC graphics stack
(`pycvc` + `pycvc_gl`, the Python bindings for libcvc / cvcGL). It speaks
general GRL-SNAM vocabulary — **terrain, obstacles, agent paths, agents,
scalar fields (risk / SDF / path-loss)** — and turns each into a live 3D
scene node. It has **no** mission- or dataset-specific knowledge.

```python
from grl_snam_lab import Lab
lab = Lab()
lab.add_terrain(heights, bounds=(-100, -100, 100, 100))
lab.add_mesh("bldg", verts, faces, color=(0.6, 0.6, 0.6))
lab.add_path("agent0", [(x0, y0, z0), (x1, y1, z1), ...])
lab.add_field("risk", grid, dims=(nx, ny, nz), bounds=(-100,-100,0, 100,100,20))
lab.show()                 # interactive window (needs a display)
# lab.render_png("s.png")  # offscreen snapshot
```

## Adapting for DBG (and other datasets)

Keep dataset-specific code **in the downstream project**. A DBG adapter reads
a `movement_bundle.v1` and drives the same general API — e.g. one `add_path`
per soldier track, `add_field` for a path-loss raster, `add_mesh` for the
scene's buildings. The lab stays general; the mapping lives with the data.

## Requirements

`pycvc` and `pycvc_gl` (from **libcvc** — install libcvc with its Python
bindings into your prefix, e.g. via cvcpkg, and put them on `PYTHONPATH`).
The pure-Python geometry helpers (`terrain_mesh`, …) work without them; the
scene/display methods require them. `show()`/`render_png()` need a GL display.

Run the demo: `python examples/lab_demo.py` (or `… out.png` for a snapshot).
