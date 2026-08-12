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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── exact Euclidean distance transform (Felzenszwalb & Huttenlocher), no scipy ──
def _edt1d(f: np.ndarray) -> np.ndarray:
    n = len(f)
    d = np.empty(n)
    v = np.zeros(n, dtype=np.intp)
    z = np.empty(n + 1)
    INF = 1e20
    k = 0
    v[0] = 0
    z[0] = -INF
    z[1] = INF
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF
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
    phi_w = (np.sqrt(_edt2(occ)) - np.sqrt(_edt2(~occ))) * cell_w  # signed world metres
    phi = (phi_w * scale).astype(np.float32)
    gy, gx = np.gradient(phi)  # dphi/dy(row), dphi/dx(col)
    gmag = np.sqrt(gx * gx + gy * gy) + 1e-9
    return phi, (gx / gmag).astype(np.float32), (gy / gmag).astype(np.float32)


def build_sdf_cvc(
    verts,
    tris,
    bounds,
    scale,
    *,
    dim=(256, 256, 48),
    z_frac=0.12,
    algo=None,
    flip=False,
    return_volume=False,
):
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
    vol = pycvc.sdf(
        app,
        g,
        nx3,
        ny3,
        nz3,
        float(mnx),
        float(mny),
        float(zmin),
        float(mxx),
        float(mxy),
        float(zmax),
        algo,
        bool(flip),
    )
    arr = np.asarray(vol.grid()).astype(np.float32)  # cvc grid() axis order is [Z, Y, X]
    kz = int(z_frac * (nz3 - 1))
    phi_w = arr[kz, :, :]  # [Y(row), X(col)] — matches the occupancy grid
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
        grid = torch.stack([gx, gy], -1)[None, None]  # [1,1,B,2]
        out = F.grid_sample(
            self.field, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )[
            0, :, 0, :
        ].t()  # [B,3]
        nrm = out[:, 1:3]
        return out[:, 0], nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-6)


def _ipc_dbdd(d: torch.Tensor, d_hat: float) -> torch.Tensor:
    """IPC barrier derivative (matches surrogate_robust's piecewise form)."""
    d = d.clamp_min(1e-6)
    val = (d_hat - d) * (2 * torch.log(d / d_hat) - d_hat / d) + 1.0
    return torch.where(d < d_hat, val, torch.zeros_like(d))


def sdf_rollout(field: SDFField, o, v, goal, al, be, ga, steps, *, rr, d_hat, dt, nsub=1, vmax=0.9):
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
            F_bar = -(al * _ipc_dbdd(d, d_hat)).unsqueeze(-1) * nrm  # push out along wall normal
            F_goal = -be.unsqueeze(-1) * (o - goal)
            a = F_bar + F_goal - ga.unsqueeze(-1) * v
            v = v + hdt * a
            sp = v.norm(dim=-1, keepdim=True)
            v = torch.where(sp > vmax, v * vmax / sp, v)
            o = o + hdt * v
    return o, v, minclr


def bicycle_rollout(
    field: SDFField,
    o,
    th,
    sp,
    goal,
    al,
    be,
    ga,
    steps,
    *,
    rr,
    d_hat,
    dt,
    nsub=1,
    vmax=0.9,
    L=0.035,
    delta_max=0.6,
    a_max=1.5,
    a_lat_max=1.0,
    k_steer=0.8,
    allow_reverse=False,
):
    """Differentiable *kinematic bicycle* rollout over the same SDF barrier.

    Where :func:`sdf_rollout` integrates a holonomic point (it can translate
    sideways, so a turning radius is meaningless), this promotes heading to
    simulated state and moves like a vehicle::

        th' = (sp / L) * tan(delta)     x' = sp*cos(th)    y' = sp*sin(th)

    The learned coefficients keep their meaning — ``al`` scales the IPC wall
    barrier, ``be`` the goal spring, ``ga`` damping — but they act through the
    vehicle's actuators instead of directly on velocity:

    * longitudinal: ``a = clamp(F . heading - ga*sp, +-a_max)`` — the barrier
      decelerates an approach head-on, the goal spring accelerates, damping is
      drag on speed;
    * steering: pure pursuit toward ``goal`` (the carrot the navigator already
      feeds), ``delta = atan2(2 L sin(alpha), L_d)``, plus a bounded barrier
      bias ``k_steer * tanh(F_bar . left)`` so walls *steer* the vehicle away
      rather than shoving it sideways (a car cannot strafe);
    * the lateral-acceleration cap ``sp^2 * |tan(delta)| / L <= a_lat_max``
      makes it slow down for corners on its own — the single cheapest thing
      that reads as "vehicle" instead of "dot".

    ``delta_max`` fixes the minimum turning radius ``R_min = L / tan(delta_max)``.
    Speed is forward-only in ``[0, vmax]``: a carrot behind the vehicle produces
    an arcing turn-around at full steer, not a reverse. Everything is built from
    smooth/clamped torch ops, so gradients flow for self-supervised training
    exactly as they do through :func:`sdf_rollout`.

    Args are as :func:`sdf_rollout` except the state: ``th`` and ``sp`` are
    ``[B]`` heading (radians) and speed. Returns ``(o, th, sp, min_clearance)``.
    """
    hdt = dt / nsub
    tan_dmax = math.tan(delta_max)
    minclr = torch.full((o.shape[0],), 9.9, device=o.device)
    for _ in range(steps):
        for _s in range(nsub):
            phi, nrm = field.sample(o)
            d = phi - rr
            minclr = torch.minimum(minclr, d.detach())

            F_bar = -(al * _ipc_dbdd(d, d_hat)).unsqueeze(-1) * nrm
            F_goal = -be.unsqueeze(-1) * (o - goal)
            F = F_bar + F_goal

            head = torch.stack([torch.cos(th), torch.sin(th)], -1)  # [B,2]
            left = torch.stack([-torch.sin(th), torch.cos(th)], -1)  # [B,2]

            # longitudinal: project the virtual force onto the heading; damping
            # becomes drag on speed. Clamped to the actuator limit.
            a_long = ((F * head).sum(-1) - ga * sp).clamp(-a_max, a_max)

            # steering: pure pursuit toward the carrot + bounded barrier bias.
            to_goal = goal - o
            L_d = to_goal.norm(dim=-1).clamp_min(1e-6)
            ang = torch.atan2(to_goal[:, 1], to_goal[:, 0]) - th
            sin_a = torch.sin(ang)  # sin()/cos() absorb the angle wrap
            cos_a = torch.cos(ang)
            behind = cos_a < 0.0
            # Pure pursuit degenerates when the carrot is behind: sin(pi)=0
            # gives zero steer while the goal spring brakes, and the vehicle
            # parks facing away. Forward-only turn-around instead: full steer
            # toward sign(sin_a) (ties broken toward +) at a creep speed.
            turn_sign = torch.where(sin_a >= 0.0, torch.ones_like(sin_a), -torch.ones_like(sin_a))
            # Adaptive lookahead: with a carrot ~30 wheelbases away, textbook
            # pure pursuit commands ~3 degrees even at a 90-degree heading
            # error -- no low-speed steering authority at all. Shrink the
            # effective lookahead with speed (floor 4L) so a slow vehicle can
            # steer hard; at speed L_d_eff ~ L_d and tracking is unchanged.
            L_d_eff = torch.minimum(L_d, (1.2 * sp + 4.0 * L).clamp_min(4.0 * L))
            delta = torch.where(
                behind, turn_sign * delta_max, torch.atan2(2.0 * L * sin_a, L_d_eff)
            )
            # Steering bias uses only the REPULSIVE part of the barrier. The
            # inherited piecewise b' (surrogate_robust's +1-shifted form) is
            # positive over d in (0.39 d_hat, d_hat) -- an attraction band --
            # and tanh saturation made that bias out-vote pure pursuit ~15x,
            # dragging the vehicle onto the b'=0 shell instead of its lane.
            # The force term F_bar keeps the original form (point-mode parity);
            # only what feeds the steering wheel is clamped repulsive-only,
            # which also removes the 0.6 rad steering slew at the d_hat shell.
            F_rep = -(al * _ipc_dbdd(d, d_hat).clamp(max=0.0)).unsqueeze(-1) * nrm
            delta = delta + k_steer * torch.tanh((F_rep * left).sum(-1))
            delta = delta.clamp(-delta_max, delta_max)

            # corner speed limit from the lateral-acceleration cap.
            kappa = torch.tan(delta).abs() / L
            v_corner = torch.sqrt(a_lat_max / kappa.clamp_min(tan_dmax / (L * 400.0)))

            # stopping-distance governor: never drive faster than you can stop
            # inside the clearance ahead (v^2 <= 2 a_max (d - margin)). Forces
            # alone cannot guarantee this -- with weak/untrained coefficients
            # the goal spring's forward pull beats the barrier head-on, and
            # clamping the NET force to a_max means the barrier can never
            # out-brake the engine. Brakes are a constraint, not a force.
            #
            # DIRECTIONAL, or it deadlocks: only the velocity component INTO
            # the wall consumes stopping distance -- driving parallel to a
            # wall at small clearance is fine. An isotropic cap pins speed to
            # ~0 head-on, and a bicycle cannot turn at zero speed (th' ~ v),
            # so the vehicle would park nose-in forever.
            v_stop = torch.sqrt(2.0 * a_max * (d - 0.5 * rr).clamp_min(0.0))
            # Into-wall component of the MOTION direction (reversing flips
            # it): a vehicle backing away from a wall is not approaching it.
            motion_sign = torch.where(sp >= 0.0, torch.ones_like(sp), -torch.ones_like(sp))
            approach = (-(nrm * head).sum(-1) * motion_sign).clamp(0.0, 1.0)  # 1 = head-on
            # The constraint is on the INTO-wall component only:
            # sp * approach <= v_stop. When the heading has no into-wall
            # component the bound must vanish entirely -- v_stop/max(approach,
            # floor) gets this wrong at the margin, where v_stop == 0 makes
            # the limit 0 in EVERY direction and the vehicle deadlocks: too
            # close to move, unable to move away because it cannot move.
            v_stop_dir = torch.where(
                approach > 0.05,
                v_stop / approach.clamp_min(0.05),
                torch.full_like(v_stop, vmax),
            )
            v_lim = torch.minimum(
                torch.full_like(v_corner, vmax), torch.minimum(v_corner, v_stop_dir)
            )

            # Maneuvering creep: turning requires motion. When a hard steer is
            # commanded, keep a small speed floor so the vehicle can rotate
            # out of a nose-in stop; its own stopping distance (v^2 / 2a_max)
            # is far inside the governor margin, so this cannot cause
            # penetration -- only escape.
            hard_steer = delta.abs() >= 0.7 * delta_max
            # Creep is allowed slightly INSIDE the governor's stop margin
            # (0.5 rr): the governor parks the vehicle exactly at that margin,
            # and if the creep cutoff sat at the same distance the two would
            # deadlock nose-in at v=0, unable to rotate.
            can_move = d > 0.25 * rr
            v_floor = torch.where(
                hard_steer & can_move, torch.full_like(v_lim, 0.08), torch.zeros_like(v_lim)
            )
            v_lim = torch.maximum(v_lim, v_floor)

            # While turning around, hold a creep speed so th' = sp/L tan(delta)
            # stays nonzero -- braking to rest would freeze the arc. The same
            # applies at a nose-in stop against a wall: if the vehicle is
            # (near) stationary and a hard steer is commanded, drive the speed
            # toward the maneuvering floor.
            # Creep target capped at its DESIGN value (half the full-steer
            # corner speed): v_corner is computed from the biased/clamped
            # delta, so at partial steer it can be several times larger and
            # the turn-around would lunge. The barrier's decelerating
            # projection stays active in the behind branch (the full F.head
            # would re-introduce the parks-facing-away freeze: the goal
            # spring is negative when the goal is behind).
            v_creep = torch.minimum(
                0.5 * v_corner,
                torch.full_like(v_corner, 0.5 * math.sqrt(a_lat_max * L / tan_dmax)),
            )
            a_long = torch.where(
                behind,
                (v_creep - sp) / hdt + torch.clamp((F_bar * head).sum(-1), max=0.0),
                a_long,
            )
            stuck_turning = hard_steer & can_move & (sp.abs() < 0.06)
            a_long = torch.where(stuck_turning, torch.maximum(a_long, (0.08 - sp) / hdt), a_long)
            if allow_reverse:
                # A nose-in car with no reverse gear cannot escape: the forward
                # creep arc (radius R_min) dips toward the wall and burns its
                # clearance before the heading comes around. When the carrot is
                # behind AND the nose is against the wall, back out with
                # opposite steer -- reversing with delta rotates the heading
                # the other way, exactly as a real car backs out of a spot.
                head_on = (-(nrm * head).sum(-1)).clamp(0.0, 1.0) > 0.6
                nose_blocked = behind & head_on & (d < 0.5 * rr + 0.02)
                a_long = torch.where(nose_blocked, (-0.10 - sp) / hdt, a_long)
                delta = torch.where(nose_blocked, -delta, delta)
            a_long = a_long.clamp(-a_max, a_max)

            # semi-implicit: speed, then heading with the new speed, then
            # position. Speed approaches v_lim through the ACTUATOR, not a
            # clamp -- clamping straight to v_lim shed up to ~15x a_max of
            # speed in a single substep when the carrot flipped behind. The
            # brakes are still only a_max strong, so sp may exceed v_lim
            # transiently while shedding speed; the steering clamp below
            # keeps the lateral invariant exact during exactly that window
            # (the vehicle carves a wider arc while braking, as a car does).
            a_long = torch.minimum(a_long, (v_lim - sp) / hdt).clamp(-a_max, a_max)
            sp_min = -0.25 * vmax if allow_reverse else 0.0
            sp = (sp + hdt * a_long).clamp(torch.full_like(sp, sp_min), torch.full_like(sp, vmax))
            d_cap = torch.atan(a_lat_max * L / sp.square().clamp_min(1e-9))
            delta = torch.clamp(delta, -d_cap, d_cap)
            th = th + hdt * (sp / L) * torch.tan(delta)
            head = torch.stack([torch.cos(th), torch.sin(th)], -1)
            o = o + hdt * sp.unsqueeze(-1) * head
    return o, th, sp, minclr


class CoefMLP(nn.Module):
    """Predict ``(alpha, beta, gamma)`` from local SDF features, biased toward the
    known-good navigating regime (``bias``) so the self-supervised optimizer starts
    in — and stays near — the stable basin."""

    def __init__(self, hidden=64, bias=(1.0, 3.0, 4.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
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
