"""Which building SURFACE the squad actually saw, by ray-casting the real mesh.

The simulator's sensor is a 2-D ray cast over a 384x384 occupancy raster: it
answers "is this cell occupied and did a ray reach it", which is all the planner
needs and all it ever used. It says nothing about a wall's height, so a renderer
driven by it lights a 90 m tower to the roof because somebody drove past its
base.

This computes the missing thing honestly, as a POST-PASS over a finished trace:
replay the recorded vehicle positions, and from each one cast rays against the
actual 978k-face city mesh, marking the FIRST face each ray hits. First hit, so
a wall behind another wall stays dark -- occlusion is the entire point.

It changes nothing about the simulation. The planner never had this information
and does not get it now; the trace's dynamics are untouched and are not
re-recorded. What is produced is a rendering input derived from recorded
positions under a stated sensor model, and it should be described that way
rather than as something the agents "knew".

Output is one ``first_seen`` array: for each mesh face, the earliest tick at
which any agent saw it, or -1. That is compact (one int32 per face, ~4 MB) and
exactly replayable -- at render time a face is visible iff
``0 <= first_seen <= tick`` -- where storing a per-snapshot bitmask would be
tens of megabytes and still coarser in time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

#: Sensor geometry for the wall cast. The azimuth count is well under the
#: simulator's 360 rays because a wall face is metres across, not centimetres;
#: the elevation fan is what the 2-D sensor never had.
N_AZIMUTH = 180
N_ELEVATION = 7
ELEV_MIN_DEG = -6.0
ELEV_MAX_DEG = 46.0  # atan(100 m roof / 120 m range) ~ 40 deg, plus headroom
EYE_HEIGHT_M = 1.6


def _tree_and_faces(bundle_dir: str):
    import pycvc_gl
    from pycvc_gl.lab import Lab
    from pycvc_gl.scenes import load_geometry_bundle
    from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

    lab = Lab()
    sampler = load_geometry_bundle(lab, bundle_dir, buildings=True)
    act = pycvc_gl.prop(lab._scene, "buildings")
    if act is None:
        raise RuntimeError("bundle has no building mesh to cast against")
    pd = act.GetMapper().GetInput()
    # vtkStaticCellLocator, and specifically its FIRST-hit IntersectWithLine.
    # vtkOBBTree collects every intersection along the ray, which in a dense
    # city is most of a block: measured 401 us/cast against 21.8 here, and 4.2 s
    # of build against 0.2. Only the first hit is wanted anyway -- that is what
    # occlusion means.
    tree = vtkStaticCellLocator()
    tree.SetDataSet(pd)
    tree.BuildLocator()
    return lab, sampler, pd, tree


def compute(bundle_dir: str, trace_dir, *, stride: int = 4, progress=None) -> Path:
    """Ray-cast the city from every recorded position; write ``wall_seen.npz``."""
    from vtkmodules.vtkCommonCore import reference as vtk_ref
    from vtkmodules.vtkCommonDataModel import vtkGenericCell  # noqa: F401

    from grl_snam.fog_trace import Trace

    trace_dir = Path(trace_dir)
    spec = json.loads((trace_dir / "squad.json").read_text())
    keys = [a["key"] for a in spec["agents"]]
    traces = {k: Trace.load(trace_dir / k) for k in keys}

    lab, sampler, pd, tree = _tree_and_faces(bundle_dir)
    n_faces = int(pd.GetNumberOfPolys())
    first_seen = np.full(n_faces, -1, np.int32)

    rng_m = float(traces[keys[0]].manifest.get("sensor_range_m", 120.0))
    az = np.linspace(0.0, 2.0 * np.pi, N_AZIMUTH, endpoint=False)
    el = np.radians(np.linspace(ELEV_MIN_DEG, ELEV_MAX_DEG, N_ELEVATION))
    # Unit directions, azimuth-major.
    dirs = (
        np.stack(
            [
                (np.cos(el)[None, :] * np.cos(az)[:, None]).ravel(),
                (np.cos(el)[None, :] * np.sin(az)[:, None]).ravel(),
                np.tile(np.sin(el), N_AZIMUTH),
            ],
            1,
        )
        * rng_m
    )

    tpar, xhit, pcoords = vtk_ref(0.0), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    subid, cellid = vtk_ref(0), vtk_ref(0)
    n_ticks = min(len(traces[k].rows["x"]) for k in keys)
    t0 = time.time()
    for tick in range(0, n_ticks, stride):
        for k in keys:
            r = traces[k].rows
            ox, oy = float(r["x"][tick]), float(r["y"][tick])
            oz = float(sampler(ox, oy)) + EYE_HEIGHT_M
            origin = (ox, oy, oz)
            for d in dirs:
                if not tree.IntersectWithLine(
                    origin,
                    (ox + d[0], oy + d[1], oz + d[2]),
                    1e-6,
                    tpar,
                    xhit,
                    pcoords,
                    subid,
                    cellid,
                ):
                    continue
                # FIRST hit only: a face behind another face was not seen.
                fid = int(cellid)
                if 0 <= fid < n_faces and first_seen[fid] < 0:
                    first_seen[fid] = tick
        if progress and (tick // stride) % 25 == 0:
            progress(tick, n_ticks, int((first_seen >= 0).sum()), time.time() - t0)

    out = trace_dir / "wall_seen.npz"
    np.savez_compressed(
        out,
        first_seen=first_seen,
        n_faces=np.int64(n_faces),
        stride=np.int32(stride),
        n_azimuth=np.int32(N_AZIMUTH),
        n_elevation=np.int32(N_ELEVATION),
        elev_deg=np.array([ELEV_MIN_DEG, ELEV_MAX_DEG], np.float32),
        eye_height_m=np.float32(EYE_HEIGHT_M),
        range_m=np.float32(rng_m),
    )
    return out


def load(trace_dir) -> np.ndarray | None:
    """``first_seen`` per face, or None if the pass has not been run."""
    p = Path(trace_dir) / "wall_seen.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["first_seen"]
