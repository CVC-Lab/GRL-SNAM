"""Live navigation metrics + a HUD overlay — a running read-out of what the learned
policy is doing while it drives.

``NavMetrics`` is one snapshot per navigation step (position, goal distance, wall
clearance, speed, the network's predicted coefficients alpha/beta/gamma, the escape
mode, whether the agent is penetrating a building, progress). ``SdfNavigator``
(``grl_snam.nav``) fills one in every step; the demos print/emit them live and the
video capture draws them as an on-frame HUD via ``hud_lines``.

The point is developer/operator insight: the coefficients are the policy's actual
output, and clearance + penetration + mode expose *why* it moves the way it does —
exactly the signals a HUD (still being designed) should surface in real time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .material_palette import NUM_MATERIALS


@dataclass
class NavMetrics:
    """One per-step snapshot of the navigator's state and the network's output."""

    step: int = 0
    x: float = 0.0  # agent world position
    y: float = 0.0
    goal_x: float = 0.0  # active goal world position
    goal_y: float = 0.0
    goal_dist_m: float = 0.0  # world metres to the active goal
    clearance_m: float = 0.0  # SDF clearance to nearest wall minus robot radius (world m)
    speed_mps: float = 0.0  # world metres/second
    alpha: float = 0.0  # network output: obstacle-barrier weight
    beta: float = 0.0  # network output: goal-spring stiffness
    gamma: float = 0.0  # network output: velocity damping
    mode: str = "seek"  # "seek" (head to goal) or "wall" (bug-style escape)
    stall: int = 0  # steps since last progress toward the goal
    inside_building: bool = False  # footprint penetration this step
    progress: float = 0.0  # 0..1, 1 - dist/initial_dist toward the active goal
    goal_wall_align: float = 0.0  # goal-direction . wall-normal (>0 = wall between agent and goal)
    goal_index: int = 0  # which goal (multi-goal drives)
    reached: bool = False  # active goal reached this step
    heading_rad: float = 0.0  # vehicle heading (bicycle dynamics; 0 in point mode)
    material_risk: float = 0.0  # smoothed material risk r~ at the agent (0 without material)
    material_gate: bool = False  # witness gate active this step (False without material)
    material_id: int = -1  # discrete material class 0..NUM_MATERIALS-1 at the agent (-1 unknown)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class NavStats:
    """Rolling aggregates over a drive — for an end-of-run summary and the HUD footer."""

    steps: int = 0
    penetration_steps: int = 0
    min_clearance_m: float = field(default=1e30)  # "unmeasured" sentinel; matches C++ veh_nav_stats
    total_path_m: float = 0.0
    # Economy signals for the base scorecard (grl_snam.scorecard), accumulated the same
    # way the C++ nav_stats collector does: total heading change and a Sigma|dSpeed| fuel
    # proxy. Zero until the drive feeds a second NavMetrics.
    turn_total_rad: float = 0.0
    fuel_used: float = 0.0
    # Per-material buckets, indexed by grl_snam.material_palette id. Mirrors the C++
    # veh_nav_stats time_over_material_s/dist_over_material_m (cvc/nav/nav_stats.h): at
    # each collected step the id at the CURRENT pose gets the step's dt (time) and the
    # step's travelled segment (dist). Stay all-zero until a material id is supplied
    # (update's default m.material_id == -1), so a non-material drive is byte-identical.
    time_over_material_s: list = field(default_factory=lambda: [0.0] * NUM_MATERIALS)
    dist_over_material_m: list = field(default_factory=lambda: [0.0] * NUM_MATERIALS)
    _prev: tuple | None = None
    _prev_head: float | None = None
    _prev_speed: float | None = None
    _reached: set = field(default_factory=set)

    def seed_start(self, x0: float, y0: float) -> None:
        """Prime the path accumulator with the true START pose, so the first ``update()``
        counts the start -> first-step segment. This matches the C++ ``cvc::nav`` nav_stats
        collector, which seeds ``prev_pos`` to the start (``total_path_m`` includes that
        first leg). Turn/fuel still skip the first sample — ``_prev_head``/``_prev_speed``
        stay ``None`` — exactly as the C++ collector skips turn/accel on the first step.
        Call once, before the first ``update()``; without it the first segment is dropped
        (the prior behavior, kept for callers that don't know the start, e.g. the HUD)."""
        self._prev = (float(x0), float(y0))

    def update(self, m: NavMetrics, dt: float | None = None) -> None:
        """Fold one step. ``dt`` (sim-seconds this step) is needed only for the
        time-over-material bucket; pass it whenever a material id is supplied so
        ``time_over_material_s`` matches the C++ collector (which holds the episode
        dt). ``dist_over_material_m`` needs no dt (it uses the travelled segment).
        With ``m.material_id == -1`` (the default) neither bucket moves, so a
        non-material drive is byte-identical to before."""
        self.steps += 1
        if m.inside_building:
            self.penetration_steps += 1
        if m.reached:
            self._reached.add(m.goal_index)  # count DISTINCT goals, not per-frame reached flags
        self.min_clearance_m = min(self.min_clearance_m, m.clearance_m)
        seg = 0.0
        if self._prev is not None:
            dx, dy = m.x - self._prev[0], m.y - self._prev[1]
            seg = (dx * dx + dy * dy) ** 0.5
            self.total_path_m += seg
        # Material buckets — parity with cvc/nav/nav_stats.cpp step(): id at the current
        # pose takes the step dt (time) and the step segment (dist). dist accrues even
        # without dt; time needs dt. First collected step: seg is start->step1 when
        # seed_start primed _prev, matching the C++ seeded prev_pos.
        mid = m.material_id
        if 0 <= mid < NUM_MATERIALS:
            self.dist_over_material_m[mid] += seg
            if dt is not None:
                self.time_over_material_s[mid] += float(dt)
        if self._prev_head is not None:
            dh = m.heading_rad - self._prev_head
            while dh > math.pi:
                dh -= 2.0 * math.pi
            while dh <= -math.pi:
                dh += 2.0 * math.pi
            self.turn_total_rad += abs(dh)
            self.fuel_used += abs(m.speed_mps - self._prev_speed)
        self._prev = (m.x, m.y)
        self._prev_head = m.heading_rad
        self._prev_speed = m.speed_mps

    @property
    def penetration_pct(self) -> float:
        return 100.0 * self.penetration_steps / max(1, self.steps)

    @property
    def goals_reached(self) -> int:
        """Distinct goals reached over the drive (by goal index)."""
        return len(self._reached)


def hud_lines(m: NavMetrics, stats: NavStats | None = None) -> list[str]:
    """Human-readable HUD lines for a metrics snapshot (used as on-frame overlay text
    and for live console output). Kept short so it fits a corner of the viewport."""
    clr = f"{m.clearance_m:5.1f} m" if m.clearance_m < 999 else "  open"
    lines = [
        "GRL-SNAM  learned SDF navigator",
        f"goal {m.goal_index}   dist {m.goal_dist_m:6.1f} m   {int(100 * m.progress):3d}%",
        f"coeffs  a {m.alpha:4.2f}  b {m.beta:4.2f}  g {m.gamma:4.2f}",
        f"clearance {clr}   speed {m.speed_mps:4.1f} m/s",
        f"mode {m.mode:4s}   stall {m.stall:3d}"
        + ("   [WALL CONTACT]" if m.inside_building else ""),
    ]
    if stats is not None:
        lines.append(
            f"reached {stats.goals_reached}   penetration {stats.penetration_pct:4.1f}%"
            f"   min clr {stats.min_clearance_m:4.1f} m"
        )
    return lines
