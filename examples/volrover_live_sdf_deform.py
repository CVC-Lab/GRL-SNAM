# examples/volrover_live_sdf_deform.py — a LIVE, fast SDF deformation demo.
#
# Generates a signed-distance field (SDF) volume from the built-in Stanford bunny
# (or any cvc::geometry), adds it to the running volrover3 scene as a volume, then
# each time slice mutates the voxel values IN PLACE from Python — via the zero-copy
# numpy view over the volume's buffer (volume.grid()) — and re-uploads, so the
# surface visibly deforms/decays in real time.
#
# HOW IT WORKS
#   * pycvc.sdf(app, geom, dim) -> cvc::volume (the SDF field).
#   * vol.grid() is a NUMPY VIEW of the volume's voxel buffer (no copy) — writing
#     to it writes the C++ voxels directly.
#   * node.setVolume(vol) re-copies the (now-modified) voxels into the VTK volume
#     mapper and re-renders. For a modest field this per-frame refresh is sub-ms —
#     "live and fast". (The zero-copy CUDA<->GL path that removes even this refresh
#     copy is the roadmap §C GPU-geometry / interop item.)
#
# RUN: volrover3 Jobs tab -> Load Script -> Run as Job (needs numpy).

import math

import numpy as np

try:
    import pycvc
    import vrhost
except ImportError as exc:  # pragma: no cover - only meaningful inside volrover3
    raise RuntimeError(
        "volrover_live_sdf_deform: run INSIDE volrover3 (Jobs tab -> Load Script)."
    ) from exc

_app = vrhost.app()
_sg = vrhost.scene()

# ── build the source geometry: the built-in bunny (a procedural .bunny mesh) ──
_bunny = pycvc.geometry(_app)
_bunny.load("bunny.bunny")  # the .bunny handler is procedural (CVC_GEOMETRY_ENABLE_BUNNY)
print("live-sdf: bunny has %d verts / %d tris — computing SDF..." %
      (_bunny.num_vertices(), _bunny.num_triangles()))

# ── compute the SDF volume from the mesh ─────────────────────────────────────
# pycvc flattens cvc::sdf's `dimension` arg to (nx, ny, nz); bbox/algorithm use
# their defaults (the mesh extents).
_N = 64
_vol = pycvc.sdf(_app, _bunny, _N, _N, _N)  # SDF over the mesh's extents
print("live-sdf: SDF volume %dx%dx%d built." % (_N, _N, _N))

# ── add it to the live scene as a volume, with a surface-highlighting TF ──────
_sg.addGraphics("sdf", _vol)
_node = _sg.getGraphics("sdf")
# A transfer function peaked around the zero level-set makes the SDF's *surface*
# (distance ~ 0) opaque and the interior/exterior fade out — so deforming the
# field reads as the surface moving. (colorTable rows: scalar, r, g, b; opacity
# rows: scalar, alpha — normalized 0..1 over the value range; see VolumeNode.)
_node.setDefaultTransferFunction()

# ── the zero-copy voxel view we mutate every frame ───────────────────────────
_grid = _vol.grid()  # numpy view of the voxel buffer (no copy)
_grid_flat = _grid.reshape(-1)
_n = _grid_flat.size
# precompute a per-voxel spatial phase so we can add a travelling ripple cheaply
_zz, _yy, _xx = np.meshgrid(
    np.linspace(0, 1, _N), np.linspace(0, 1, _N), np.linspace(0, 1, _N), indexing="ij"
)
_phase = (2.0 * math.pi * (_xx + _yy + _zz)).reshape(-1).astype(_grid_flat.dtype)
_base = _grid_flat.copy()  # the pristine SDF to modulate around

print("live-sdf: driving live deformation — the bunny SDF surface ripples + erodes. Jobs tab -> Stop.")

_t = 0.0


def step(dt):
    global _t
    _t += dt
    # Deform: a travelling sinusoidal ripple on the distance field + a slow erosion
    # (adding a growing positive bias pushes the zero-crossing inward = decay).
    ripple = 2.0 * np.sin(_phase - 3.0 * _t)
    erode = 0.6 * _t
    _grid_flat[:] = _base + ripple + erode  # in-place write into the voxel buffer
    _node.setVolume(_vol)  # re-upload the modified voxels + re-render
    _sg.processEvents()
    if _t > 8.0:  # loop the decay so the demo runs indefinitely
        _t = 0.0
