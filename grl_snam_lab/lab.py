"""Scene Lab + pure-Python geometry helpers — moved to :mod:`pycvc_gl.lab`
(a generic scene utility that ships with the pycvc_gl bindings). Re-exported here
for backward compatibility."""
from pycvc_gl.lab import *  # noqa: F401,F403

# `import *` skips underscore-prefixed names, but tests/test_lab.py imports
# `_flatten_points` from here (its old home), so surface the helpers by name too.
from pycvc_gl.lab import Lab, terrain_mesh, polyline_indices, _flatten_points  # noqa: F401
