"""The shared learned-SDF navigator — one implementation of the drive loop used by
every demo and the video capture (previously copy-pasted across four scripts).

``SdfNavigator`` wraps a trained ``sdf_nav.CoefMLP`` + an ``SdfNavigator``-owned
``SDFField`` and drives a point agent toward a goal purely by reacting to the field:
a moving carrot toward the goal, deflected by the SDF wall barrier, with a
``bug``-style wall-follow escape when a potential-field dead-end stalls progress.
``step()`` advances one step and returns a :class:`~grl_snam.metrics.NavMetrics`
snapshot, so callers get the network's live coefficients and clearance for free.

Goals are live-retargetable (``set_goal``) — the basis of the dynamic multi-goal
free-drive. ``select_reachable_goals`` picks corner-ward goals in the SDF's *own*
free space so every leg is actually reachable.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import sdf_nav

from .metrics import NavMetrics


class SdfNavigator:
    """Drive a point agent across a trained SDF field toward a (retargetable) goal."""

    #: default kinematic-bicycle parameters (normalized units; L ~ 3 m at the
    #: Austin scale). Overridable per-instance via ``vehicle=dict(...)``.
    VEHICLE_DEFAULTS = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8)

    def __init__(
        self,
        field,
        model,
        meta,
        *,
        reach_tol: float = 0.8,
        dynamics: str = "point",
        vehicle: dict | None = None,
    ):
        if dynamics not in ("point", "bicycle"):
            raise ValueError(f"dynamics must be 'point' or 'bicycle', got {dynamics!r}")
        self._dyn = dynamics
        self._veh = {**self.VEHICLE_DEFAULTS, **(vehicle or {})}
        self.th = torch.zeros(1)  # heading (rad); meaningful in bicycle mode
        self.sp = torch.zeros(1)  # forward speed;  meaningful in bicycle mode
        self.field = field
        self.model = model
        self.meta = meta
        self.S = float(meta["scale"])
        self.cx, self.cy = (float(c) for c in meta["center"])
        self.kw = dict(
            rr=float(meta["rr"]),
            d_hat=float(meta["d_hat"]),
            dt=float(meta["dt"]),
            vmax=float(meta["vmax"]),
        )
        self.nsub = int(meta.get("nsub", 1))
        self.dt = float(meta["dt"])
        self.rr = float(meta["rr"])
        self.reach_tol = reach_tol
        self.o = torch.zeros(1, 2)
        self.v = torch.zeros(1, 2)
        self._gn = np.zeros(2, np.float32)
        self.step_i = 0
        self.goal_index = 0
        self._reset_goal_state(1e9)

    # ── world <-> normalized ────────────────────────────────────────────────
    def w2n(self, p):
        return np.array([(p[0] - self.cx) * self.S, (p[1] - self.cy) * self.S], np.float32)

    def n2w(self, on):
        return np.array([on[0] / self.S + self.cx, on[1] / self.S + self.cy], np.float32)

    def pos_world(self):
        return self.n2w(self.o[0].numpy())

    # ── setup ───────────────────────────────────────────────────────────────
    def start(self, start_world, goal_world):
        self.o = torch.from_numpy(self.w2n(start_world)).unsqueeze(0).float()
        self.v = torch.zeros(1, 2)
        gd = self.w2n(goal_world) - self.o[0].numpy()
        self.th = torch.tensor([math.atan2(float(gd[1]), float(gd[0]))])
        self.sp = torch.zeros(1)
        self.step_i = 0
        self.set_goal(goal_world)
        return self

    def set_goal(self, goal_world, *, goal_index: int | None = None):
        self._gn = self.w2n(goal_world)
        if goal_index is not None:
            self.goal_index = goal_index
        p = self.o[0].numpy()
        self._reset_goal_state(float(np.linalg.norm(self._gn - p)))

    def track_goal(self, goal_world, *, goal_index: int | None = None):
        """Retarget a *moving* goal without disturbing the escape machinery.

        ``set_goal`` resets the whole goal state — including ``_stall`` — which
        is right for a discrete new objective but wrong called per-frame on a
        moving target: zeroing the stall counter every frame permanently
        disables the wall-follow escape, and the agent livelocks in any concave
        corner. Conversely the counter itself measures *closing*, so a target
        that legitimately recedes would false-trigger the escape while tracking
        is going fine.

        ``track_goal`` therefore (a) updates the goal in place, leaving
        ``_mode``/``_stall``/``_turn`` alone, and (b) switches stall detection
        to *displacement* — "am I moving?" instead of "am I closing?" — until
        the next ``set_goal``/``start``.
        """
        self._gn = self.w2n(goal_world)
        if goal_index is not None:
            self.goal_index = goal_index
        dg = float(np.linalg.norm(self._gn - self.o[0].numpy()))
        # Re-base the progress accounting to the new goal; NOT the escape state.
        self._best = min(self._best, dg)
        self._init = max(self._init, dg, 1e-6)
        self._tracking = True
        self.reached = False

    def _reset_goal_state(self, dist0):
        self._best = dist0
        self._init = max(dist0, 1e-6)
        self._stall = 0
        self._mode = "seek"
        self._turn = 1.0
        self._dhit = 0.0
        self.reached = False
        self._tracking = False
        self._pos_hist = []  # recent positions (normalized) for displacement stall
        self._wall_entry = None  # position where wall-follow was entered

    @torch.no_grad()
    def _sdf_normal(self, on):
        _, nrm = self.field.sample(torch.from_numpy(on).unsqueeze(0).float())
        return nrm[0].numpy()

    # ── one drive step ──────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self) -> NavMetrics:
        """Advance one navigation step; return the metrics snapshot after it."""
        prev_world = self.pos_world()
        p = self.o[0].numpy()
        dg = float(np.linalg.norm(self._gn - p))
        gdir = (self._gn - p) / (dg + 1e-6)

        if self._tracking:
            # Displacement stall: "am I moving?", not "am I closing?". Against
            # a moving target the goal distance may legitimately never improve,
            # and (the reverse trap) resetting on every retarget would disable
            # the escape entirely. Window of 40 steps ~ 2.4 s at dt=0.06.
            self._pos_hist.append(p.copy())
            if len(self._pos_hist) > 40:
                self._pos_hist.pop(0)
                moved = float(np.linalg.norm(p - self._pos_hist[0]))
                if moved < 0.15:
                    self._stall += 1
                else:
                    self._stall = 0
            self._best = min(self._best, dg)  # progress metric only
        elif dg < self._best - 1e-3:
            self._best = dg
            self._stall = 0
        else:
            self._stall += 1

        # enter wall-follow when a potential-field dead-end stalls progress
        if self._mode == "seek" and self._stall > 70:
            nrm = self._sdf_normal(p)
            tang = np.array([-nrm[1], nrm[0]], np.float32)
            self._turn = 1.0 if np.dot(tang, gdir) >= 0 else -1.0
            self._dhit = dg
            self._wall_entry = p.copy()
            self._mode = "wall"
            self._stall = 0

        if self._mode == "wall":
            nrm = self._sdf_normal(p)
            tang = self._turn * np.array([-nrm[1], nrm[0]], np.float32)
            carrot = (p + (0.6 * tang + 0.4 * nrm) * 1.6).astype(np.float32)
            if self._tracking:
                # The dg-based exit assumes a fixed goal; under tracking, exit
                # once we have genuinely moved away from where we got stuck (or
                # give up). The moving carrot then re-acquires naturally.
                escaped = (
                    self._wall_entry is not None
                    and float(np.linalg.norm(p - self._wall_entry)) > 2.0
                )
                if escaped or dg < self._dhit - 1.2 or self._stall > 240:
                    self._mode = "seek"
                    self._best = dg
                    self._stall = 0
            elif dg < self._dhit - 1.2 or self._stall > 240:  # rounded it, or gave up
                self._mode = "seek"
                self._best = dg
                self._stall = 0
        else:
            carrot = (p + gdir * min(1.8, dg)).astype(np.float32)

        gt = torch.from_numpy(carrot).unsqueeze(0)
        al, be, ga = self.model(sdf_nav.coef_feats(self.field, self.o, gt))
        if self._dyn == "bicycle":
            self.o, self.th, self.sp, _ = sdf_nav.bicycle_rollout(
                self.field,
                self.o,
                self.th,
                self.sp,
                gt,
                al,
                be,
                ga,
                1,
                nsub=self.nsub,
                **self.kw,
                **self._veh,
            )
            head = torch.stack([torch.cos(self.th), torch.sin(self.th)], -1)
            self.v = self.sp.unsqueeze(-1) * head  # keep .v meaningful for callers
        else:
            self.o, self.v, _ = sdf_nav.sdf_rollout(
                self.field, self.o, self.v, gt, al, be, ga, 1, nsub=self.nsub, **self.kw
            )
        self.step_i += 1

        # metrics at the new position
        onew = self.o[0].numpy()
        phi, nrm = self.field.sample(self.o)
        phi = float(phi[0])
        dg_new = float(np.linalg.norm(self._gn - onew))
        world = self.n2w(onew)
        step_world = float(np.linalg.norm(world - prev_world))
        gwd = self._gn - onew
        gwn = gwd / (np.linalg.norm(gwd) + 1e-6)
        align = float(np.dot(gwn, nrm[0].numpy()))
        self.reached = dg_new < self.reach_tol
        return NavMetrics(
            step=self.step_i,
            x=float(world[0]),
            y=float(world[1]),
            goal_x=float(self._gn[0] / self.S + self.cx),
            goal_y=float(self._gn[1] / self.S + self.cy),
            goal_dist_m=dg_new / self.S,
            clearance_m=(phi - self.rr) / self.S,
            speed_mps=step_world / self.dt if self.dt > 0 else 0.0,
            alpha=float(al.mean()),
            beta=float(be.mean()),
            gamma=float(ga.mean()),
            mode=self._mode,
            stall=self._stall,
            inside_building=phi < 0.0,
            progress=max(0.0, 1.0 - dg_new / self._init),
            goal_wall_align=align,
            goal_index=self.goal_index,
            reached=self.reached,
            heading_rad=float(self.th[0]) if self._dyn == "bicycle" else 0.0,
        )

    def drive_to_goal(self, max_steps: int = 1300, *, stop_at_reach: bool = True):
        """Run steps toward the current goal until reached / stuck / ``max_steps``.
        Returns ``(metrics_list, closest_idx, o_at_closest, v_at_closest)`` — the
        closest-approach index truncates an orbiting tail for clean video, and its
        ``(o, v)`` lets a multi-goal driver continue the next leg from that point."""
        out = []
        best = 1e9
        best_i = 0
        since_best = 0
        best_o = self.o.clone()
        best_v = self.v.clone()
        for _ in range(max_steps):
            m = self.step()
            out.append(m)
            if m.goal_dist_m < best - 1e-3:
                best = m.goal_dist_m
                best_i = len(out) - 1
                best_o = self.o.clone()
                best_v = self.v.clone()
                since_best = 0
            else:
                since_best += 1
            if stop_at_reach and m.reached:
                best_i = len(out) - 1
                best_o = self.o.clone()
                best_v = self.v.clone()
                break
            if since_best > 340:  # stuck / orbiting — stop; caller re-targets
                break
        return out, best_i, best_o, best_v


def select_reachable_goals(
    field, model, meta, *, n_corners: int = 4, clear_world: float = 16.0, reach_world: float = 70.0
):
    """Pick ``n_corners`` corner-ward goals the navigator can actually REACH, in the
    SDF's own free space, chained from an open centre cell in a perimeter loop. Each
    candidate is *simulated* with the real model (not just checked geometrically), so
    selection and navigation agree — the fix for far-corner wandering. Returns
    ``(start_world, [goal_world, ...])``."""
    S = float(meta["scale"])
    cx, cy = (float(c) for c in meta["center"])
    region = float(meta["region"])
    mnx, mny, mxx, mxy = (float(b) for b in meta["bounds"])

    def clearance(pw):
        on = np.array([(pw[0] - cx) * S, (pw[1] - cy) * S], np.float32)
        d, _ = field.sample(torch.from_numpy(on).unsqueeze(0).float())
        return float(d[0]) / S

    def w2n(pw):
        return np.array([(pw[0] - cx) * S, (pw[1] - cy) * S], np.float32)

    start = np.array([cx, cy], np.float32)
    for rad in (0, 20, 40, 60, 80, 120):
        found = False
        for a in range(0, 360, 30):
            pw = np.array(
                [cx + math.cos(math.radians(a)) * rad, cy + math.sin(math.radians(a)) * rad],
                np.float32,
            )
            if clearance(pw) > clear_world:
                start = pw
                found = True
                break
        if found:
            break

    def candidates(sx, sy):
        diag = math.atan2(sy, sx)
        out = []
        for frac in (0.62, 0.56, 0.50, 0.44, 0.38, 0.32):
            for da in (0, -18, 18, -32, 32):
                ang = diag + math.radians(da)
                r = region * frac
                gw = np.array([cx + math.cos(ang) * r, cy + math.sin(ang) * r], np.float32)
                if mnx < gw[0] < mxx and mny < gw[1] < mxy and clearance(gw) > clear_world:
                    out.append(gw)
        return out

    nav = SdfNavigator(field, model, meta)
    o = torch.from_numpy(w2n(start)).unsqueeze(0).float()
    v = torch.zeros(1, 2)
    goals = []
    for sx, sy in [(1, 1), (1, -1), (-1, -1), (-1, 1)][:n_corners]:
        chosen = None
        chosen_ba = 1e9
        chosen_o, chosen_v = o, v
        for gw in candidates(sx, sy)[:8]:
            nav.o = o.clone()
            nav.v = v.clone()
            nav.set_goal(gw)
            ms, _bi, bo, bv = nav.drive_to_goal(max_steps=800)
            ba = float(np.linalg.norm(bo[0].numpy() - w2n(gw))) / S
            if ba < chosen_ba:
                chosen_ba, chosen, chosen_o, chosen_v = ba, gw, bo, bv
            if ba < reach_world:
                break
        if chosen is None:
            chosen = np.array([cx + sx * region * 0.4, cy + sy * region * 0.4], np.float32)
        o, v = chosen_o, chosen_v
        goals.append(chosen)
    return start, goals
