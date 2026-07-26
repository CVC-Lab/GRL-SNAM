"""Scene Lab + pure-Python geometry helpers — moved to :mod:`pycvc_gl.lab`
(a generic scene utility that ships with the pycvc_gl bindings). Re-exported here
for backward compatibility."""
from pycvc_gl.lab import *  # noqa: F401,F403
from pycvc_gl.lab import Lab, terrain_mesh, polyline_indices  # noqa: F401
