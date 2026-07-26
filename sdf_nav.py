"""SDF-based navigation for GRL-SNAM — a drop-in obstacle model for real cities.

The base surrogate (``surrogate_robust.integrate_surrogate_v2``) repels from
**circular** obstacles. That works for sparse round obstacles but not a dense
rectilinear city: thousands of overlapping circle barriers conflict and a
point-agent is pushed through building corners regardless of coefficients
(measured on Austin: hand-tuned coefficients reach 0–1/4 goals with heavy
penetration).

This module replaces the circle field with a **signed distance field (SDF)** of
the building footprints. The barrier then repels along the true wall normal, so
the agent navigates streets and corners cleanly. Everything is differentiable
(``torch.nn.functional.grid_sample``), so a small coefficient net trains
self-supervised through the rollout exactly like the base ``CoefEnergyNet``.

Pieces:
  - ``build_sdf(occ, bounds)`` — footprint occupancy -> (phi, normal_x, normal_y)
    grids, via an exact Euclidean distance transform (no scipy).
  - ``SDFField`` — holds the field on a device; ``sample(pos)`` returns
    ``(phi, unit_normal)`` at normalized agent positions (differentiable).
  - ``sdf_rollout(...)`` — the differentiable SDF surrogate (semi-implicit Euler +
    IPC wall barrier + goal spring + damping), substepped + speed-clamped.
  - ``CoefMLP`` / ``coef_feats`` — predict ``(alpha, beta, gamma)`` from local SDF
    features; biased toward the known-good navigating regime for stability.

Scale: like the base surrogate, work in a ~10-unit normalized regime
(``pos_normalized = (world - center) * scale``); the SDF is stored normalized too.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── exact Euclidean distance transform (Felzenszwalb & Huttenlocher), no scipy ──
def _edt1d(f: np.ndarray) -> np.ndarray:
    n = len(f); d = np.empty(n); v = np.zeros(n, dtype=np.intp); z = np.empty(n + 1); INF = 1e20
    k = 0; v[0] = 0; z[0] = -INF; z[1] = INF
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1; v[k] = q; z[k] = s; z[k + 1] = INF
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]
    return d


def _edt2(mask: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance (grid units) from each cell to the nearest True."""
    f = np.where(mask, 0.0, 1e20)
    return np.apply_along_axis(_edt1d, 1, np.apply_along_axis(_edt1d, 0, f))


def build_sdf(occ: np.ndarray, bounds, scale: float):
    """Footprint occupancy -> normalized signed distance field + unit normals.

    ``occ[r][c]`` True = inside a building; ``bounds`` = ``(min_x,min_y,max_x,max_y)``
    (world); ``scale`` maps world -> the normalized regime. Returns ``(phi, nx, ny)``
    float32 grids (``phi`` positive OUTSIDE buildings, 0 at walls; ``(nx,ny)`` the
    unit OUTWARD normal, i.e. the direction of increasing clearance)."""
    ny, nx = occ.shape
    mnx, mny, mxx, mxy = bounds
    cell_w = (mxx - mnx) / (nx - 1)
    phi_w = (np.sqrt(_edt2(occ)) - np.sqrt(_edt2(~occ))) * cell_w   # signed world metres
    phi = (phi_w * scale).astype(np.float32)
    gy, gx = np.gradient(phi)                                        # dphi/dy(row), dphi/dx(col)
    gmag = np.sqrt(gx * gx + gy * gy) + 1e-9
    return phi, (gx / gmag).astype(np.float32), (gy / gmag).astype(np.float32)


def build_sdf_cvc(verts, tris, bounds, scale, *, dim=(256, 256, 48), z_frac=0.12,
                  algo=None, flip=False, return_volume=False):
    """MESH-EXACT signed distance field via **cvc::sdf** (the CVC compute layer),
    as an alternative to the footprint EDT (``build_sdf``). Builds a ``cvc::geometry``
    from the flat ``verts`` (``[x,y,z,...]``) + ``tris`` (``[i,j,k,...]``), runs
    ``pycvc.sdf(app, geom, nx,ny,nz, bbox, SDF_V2)`` to get a **3-D** SDF volume, and
    slices it at ``z_frac`` of the vertical extent for the 2-D ground-navigation field.

    Returns the same ``(phi, nx, ny)`` normalized 2-D grids as ``build_sdf`` (so the
    field/surrogate/net are source-agnostic). The full 3-D volume — the natural
    substrate for extending GRL-SNAM to 3-D navigation — is returned too when
    ``return_volume=True``. ``algo`` defaults to ``pycvc.SDF_V2`` (the faster method).
    Needs the ``pycvc`` bindings (and a mesh, e.g. extracted from a glTF)."""
    import pycvc

    app = pycvc.make_app()
    g = pycvc.geometry(app)
    g.add_vertices(list(verts))
    g.add_triangles(list(tris))
    mnx, mny, mxx, mxy = bounds
    # vertical slab around the footprints; SDF over the working box
    zs = [verts[i] for i in range(2, len(verts), 3)]
    zmin, zmax = (min(zs), max(zs)) if zs else (0.0, 1.0)
    algo = pycvc.SDF_V2 if algo is None else algo
    nx3, ny3, nz3 = dim
    vol = pycvc.sdf(app, g, nx3, ny3, nz3, float(mnx), float(mny), float(zmin),
                    float(mxx), float(mxy), float(zmax), algo, bool(flip))
    arr = np.asarray(vol.grid()).astype(np.float32)          # cvc grid() axis order is [Z, Y, X]
    kz = int(z_frac * (nz3 - 1))
    phi_w = arr[kz, :, :]                                     # [Y(row), X(col)] — matches the occupancy grid
    # NOTE: verify x/y orientation against your occupancy on first use (compare
    # sign(phi_w) to the footprint mask); transpose here if your scene is mirrored.
    phi = (phi_w * scale).astype(np.float32)
    gy, gx = np.gradient(phi)
    gmag = np.sqrt(gx * gx + gy * gy) + 1e-9
    out = (phi, (gx / gmag).astype(np.float32), (gy / gmag).astype(np.float32))
    return (out + (arr,)) if return_volume else out


class SDFField:
    """A normalized SDF on a device; differentiable sampling at agent positions."""

    def __init__(self, phi, nx_g, ny_g, bounds, center, scale, device="cpu"):
        self.dev = torch.device(device)
        self.field = torch.from_numpy(np.stack([phi, nx_g, ny_g], 0)[None]).float().to(self.dev)
        self.mnx, self.mny, self.mxx, self.mxy = [float(b) for b in bounds]
        self.cx, self.cy = float(center[0]), float(center[1])
        self.S = float(scale)

    def sample(self, on: torch.Tensor):
        """on: ``[B,2]`` normalized (centered) -> ``(phi[B], unit_normal[B,2])``."""
        wx = on[:, 0] / self.S + self.cx
        wy = on[:, 1] / self.S + self.cy
        gx = 2 * (wx - self.mnx) / (self.mxx - self.mnx) - 1
        gy = 2 * (wy - self.mny) / (self.mxy - self.mny) - 1
        grid = torch.stack([gx, gy], -1)[None, None]                # [1,1,B,2]
        out = F.grid_sample(self.field, grid, mode="bilinear", align_corners=True,
                            padding_mode="border")[0, :, 0, :].t()  # [B,3]
        nrm = out[:, 1:3]
        return out[:, 0], nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-6)


def _ipc_dbdd(d: torch.Tensor, d_hat: float) -> torch.Tensor:
    """IPC barrier derivative (matches surrogate_robust's piecewise form)."""
    d = d.clamp_min(1e-6)
    val = (d_hat - d) * (2 * torch.log(d / d_hat) - d_hat / d) + 1.0
    return torch.where(d < d_hat, val, torch.zeros_like(d))


def sdf_rollout(field: SDFField, o, v, goal, al, be, ga, steps, *, rr, d_hat, dt,
                nsub=1, vmax=0.9):
    """Differentiable SDF surrogate rollout. ``al,be,ga`` are ``[B]`` coefficients.
    Returns ``(oT, vT, min_clearance[B])``. Substep (``nsub``>1) + ``vmax`` clamp at
    inference so a fast step can't tunnel a thin wall; ``nsub=1`` is fine for the
    training gradient."""
    hdt = dt / nsub
    minclr = torch.full((o.shape[0],), 9.9, device=o.device)
    for _ in range(steps):
        for _s in range(nsub):
            phi, nrm = field.sample(o)
            d = phi - rr
            minclr = torch.minimum(minclr, d.detach())
            F_bar = -(al * _ipc_dbdd(d, d_hat)).unsqueeze(-1) * nrm   # push out along wall normal
            F_goal = -be.unsqueeze(-1) * (o - goal)
            a = F_bar + F_goal - ga.unsqueeze(-1) * v
            v = v + hdt * a
            sp = v.norm(dim=-1, keepdim=True)
            v = torch.where(sp > vmax, v * vmax / sp, v)
            o = o + hdt * v
    return o, v, minclr


class CoefMLP(nn.Module):
    """Predict ``(alpha, beta, gamma)`` from local SDF features, biased toward the
    known-good navigating regime (``bias``) so the self-supervised optimizer starts
    in — and stays near — the stable basin."""

    def __init__(self, hidden=64, bias=(1.0, 3.0, 4.0)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 3))
        self.register_buffer("bias", torch.tensor(bias))

    def forward(self, feat):
        raw = self.net(feat) + torch.log(torch.expm1(self.bias)).unsqueeze(0)
        c = F.softplus(raw)
        return c[:, 0], c[:, 1], c[:, 2]


def coef_feats(field: SDFField, o, goal):
    """Local features for ``CoefMLP``: ``[phi, goal_dist, goal_dir_x, goal_dir_y,
    goal·wall_normal]`` — the last says whether a wall stands between agent and goal."""
    phi, nrm = field.sample(o)
    dg = goal - o
    gd = dg.norm(dim=-1, keepdim=True)
    gdir = dg / (gd + 1e-6)
    align = (gdir * nrm).sum(-1, keepdim=True)
    return torch.cat([phi.unsqueeze(-1), gd, gdir, align], -1)
