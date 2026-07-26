# examples/volrover_live_texture.py — LIVE, zero-copy texture editing on a mesh.
#
# Maps a texture (a cvc::image) onto a UV'd mesh in the running volrover3 scene,
# then edits the texture's PIXELS in place from Python each frame — via the
# zero-copy numpy view over the image buffer — and the mesh's surface texture
# updates live, with NO per-frame copies.
#
# This is the acceptance test for PHASE 5 (pycvc image + texture bindings) built on
# PHASE 2/4 (geometry UVs -> SetTCoords; GeometryNode::setTexture). It needs these
# binding additions, which are NOT in the shipped pycvc/pycvc_gl yet:
#   * pycvc.image                 — the cvc::image value type (Phase 1) wrapped, with
#                                   img.numpy() returning a zero-copy (H,W,C) view.
#   * geometry.set_uvs(...) / uvs()— per-vertex UVs (Phase 2) on the pycvc geometry.
#   * node.set_texture(img)        — pycvc_gl binding of GeometryNode::setTexture,
#                                   ideally the ZERO-COPY variant that wraps the
#                                   image buffer (vtkUnsignedCharArray::SetArray) so
#                                   the vtkTexture aliases img's storage.
#   * node.texture_modified()      — signal the vtkTexture changed (no re-copy), so
#                                   an in-place pixel edit shows next render.
# Until then this script imports cleanly but raises a clear message at run time.
# (The fully zero-copy CUDA<->GL path is roadmap §C.)
#
# RUN: volrover3 Jobs tab -> Load Script -> Run as Job (needs numpy + Phase 5).

import numpy as np

try:
    import pycvc
    import pycvc_gl  # noqa: F401
    import vrhost
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "volrover_live_texture: run INSIDE volrover3 (Jobs tab -> Load Script)."
    ) from exc

_app = vrhost.app()
_sg = vrhost.scene()

if not hasattr(pycvc, "image"):
    raise RuntimeError(
        "volrover_live_texture needs the Phase-5 pycvc.image + node.set_texture "
        "bindings (not in this build yet). See the script header."
    )

# ── a UV'd quad to paint on ──────────────────────────────────────────────────
_g = pycvc.geometry(_app)
_g.add_vertices([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0])
_g.add_triangles([0, 1, 2, 0, 2, 3])
_g.set_uvs([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])  # Phase-2 UVs -> SetTCoords
_sg.addGraphics("quad", _g)
_node = _sg.getGraphics("quad")

# ── the texture: an RGBA8 cvc::image; set it once, then edit pixels in place ──
_W = _H = 256
_tex = pycvc.image(_W, _H, pycvc.image.RGBA, pycvc.image.u8)
_node.set_texture(_tex)  # Phase-4 setTexture, bound in Phase 5 (zero-copy variant)

# zero-copy view of the SAME buffer the vtkTexture aliases — writes are live.
_pix = _tex.numpy()  # (H, W, 4) uint8

# precompute coordinate grids for a cheap animated pattern
_yy, _xx = np.mgrid[0:_H, 0:_W].astype(np.float32)
_cx, _cy = _W / 2.0, _H / 2.0

print("live-texture: painting the quad's texture in place each frame (zero-copy). Jobs tab -> Stop.")

_t = 0.0


def step(dt):
    global _t
    _t += dt
    # A travelling radial ripple painted straight into the texture's pixels.
    r = np.hypot(_xx - _cx, _yy - _cy)
    wave = 0.5 + 0.5 * np.sin(0.15 * r - 4.0 * _t)
    ang = np.arctan2(_yy - _cy, _xx - _cx)
    _pix[..., 0] = (255 * wave).astype(np.uint8)                       # R
    _pix[..., 1] = (127 * (1.0 + np.sin(ang + _t))).astype(np.uint8)   # G
    _pix[..., 2] = (255 * (1.0 - wave)).astype(np.uint8)              # B
    _pix[..., 3] = 255                                                # A
    _node.texture_modified()  # mark the aliased vtkTexture dirty — no re-copy
    _sg.processEvents()
