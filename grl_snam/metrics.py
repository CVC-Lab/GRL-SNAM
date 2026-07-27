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

from dataclasses import dataclass, field


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

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class NavStats:
    """Rolling aggregates over a drive — for an end-of-run summary and the HUD footer."""

    steps: int = 0
    penetration_steps: int = 0
    min_clearance_m: float = field(default=1e9)
    total_path_m: float = 0.0
    _prev: tuple | None = None
    _reached: set = field(default_factory=set)

    def update(self, m: NavMetrics) -> None:
        self.steps += 1
        if m.inside_building:
            self.penetration_steps += 1
        if m.reached:
            self._reached.add(m.goal_index)  # count DISTINCT goals, not per-frame reached flags
        self.min_clearance_m = min(self.min_clearance_m, m.clearance_m)
        if self._prev is not None:
            dx, dy = m.x - self._prev[0], m.y - self._prev[1]
            self.total_path_m += (dx * dx + dy * dy) ** 0.5
        self._prev = (m.x, m.y)

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
