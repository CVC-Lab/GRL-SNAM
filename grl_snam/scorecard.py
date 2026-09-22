"""Base navigation SCORECARD — aggregate a corpus of episodes into one fitness row.

This is the Python twin of the C++ ``cvc::dbg::nav_scorecard`` (CVC-DBG/cvcdbg
``inc/cvc/dbg/nav_stats.h``): the RF-free nav-fitness row grl-snam uses to rank its
OWN base-policy checkpoints over a scene corpus. The DBG campaign layers an
``rf_scorecard`` on top; the base half here is shared, so a grl-snam training run and
a cvcdbg ``dbg_arrival_check3 --episodes`` run report identical base numbers (the
aggregation logic + JSON keys are field-for-field the same, and ``tests/test_scorecard``
pins them to the same hand-computed values the C++ ``nav_stats_test`` uses).

Feed it ``EpisodeStats`` (one per eval episode). ``EpisodeStats.from_nav_stats`` builds
one from the per-vehicle ``NavStats`` rolling aggregates the drive already produces
(``grl_snam.metrics``), so a base eval loop is: run each corpus scene -> collect a
per-vehicle ``NavStats`` -> ``EpisodeStats.from_nav_stats(...)`` -> ``aggregate_nav(...)``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

# Dense material palette id space, matching cvc::dbg kNumMaterials (channel.cpp table).
NUM_MATERIALS = 13


@dataclass
class VehStats:
    """One vehicle's per-episode outcome — the fields the base scorecard reduces."""

    arrived: bool = False
    time_to_goal_s: float = -1.0  # < 0 = never
    total_path_m: float = 0.0
    straight_m: float = 0.0  # straight-line start->goal (for the path ratio)
    turn_total_rad: float = 0.0
    fuel_used: float = 0.0
    veh_contacts: int = 0
    penetration_steps: int = 0
    time_over_material_s: list[float] = field(default_factory=lambda: [0.0] * NUM_MATERIALS)


@dataclass
class EpisodeStats:
    """One eval episode — a reduction target for the scorecard. Mirrors the base half
    of the C++ ``episode_nav_stats``."""

    success: bool = False  # every vehicle arrived (and within budget)
    makespan_s: float = 0.0  # max arrival time over the fleet
    penetration_pct: float = 0.0  # fleet penetration steps / total steps * 100
    min_sep_m: float = 1e30  # smallest inter-vehicle distance over the run
    per_vehicle: list[VehStats] = field(default_factory=list)

    @staticmethod
    def from_nav_stats(
        vehicles: list,
        *,
        straights_m: list[float] | None = None,
        arrival_times_s: list[float] | None = None,
        min_sep_m: float = 1e30,
    ) -> EpisodeStats:
        """Build an EpisodeStats from a list of per-vehicle ``grl_snam.metrics.NavStats``.
        ``straights_m`` / ``arrival_times_s`` (per vehicle) supply the fields NavStats
        does not itself carry; a vehicle counts as arrived when it reached >= 1 goal.
        (NavStats.turn_total_rad / fuel_used are read when present — see metrics.py.)"""
        per = []
        for i, ns in enumerate(vehicles):
            arrived = getattr(ns, "goals_reached", 0) >= 1
            per.append(
                VehStats(
                    arrived=arrived,
                    time_to_goal_s=(arrival_times_s[i] if arrival_times_s and arrived else -1.0),
                    total_path_m=getattr(ns, "total_path_m", 0.0),
                    straight_m=(straights_m[i] if straights_m else 0.0),
                    turn_total_rad=getattr(ns, "turn_total_rad", 0.0),
                    fuel_used=getattr(ns, "fuel_used", 0.0),
                    penetration_steps=getattr(ns, "penetration_steps", 0),
                )
            )
        arrived_all = bool(per) and all(v.arrived for v in per)
        makespan = max((v.time_to_goal_s for v in per if v.arrived), default=0.0)
        tot_steps = sum(getattr(ns, "steps", 0) for ns in vehicles)
        tot_pen = sum(getattr(ns, "penetration_steps", 0) for ns in vehicles)
        pen_pct = 100.0 * tot_pen / tot_steps if tot_steps > 0 else 0.0
        return EpisodeStats(
            success=arrived_all,
            makespan_s=makespan,
            penetration_pct=pen_pct,
            min_sep_m=min_sep_m,
            per_vehicle=per,
        )


@dataclass
class NavScorecard:
    """One fitness row for a checkpoint's eval run over a corpus. Field names + JSON
    keys are identical to the C++ ``cvc::dbg::nav_scorecard``."""

    checkpoint: str = ""
    n_episodes: int = 0
    n_vehicle_runs: int = 0
    success_rate: float = 0.0
    arrival_rate: float = 0.0
    mean_time_to_goal_s: float = 0.0
    p95_time_to_goal_s: float = 0.0
    mean_makespan_s: float = 0.0
    mean_path_ratio: float = 0.0
    mean_turn_total_rad: float = 0.0
    mean_fuel: float = 0.0
    mean_penetration_pct: float = 0.0
    veh_contacts_per_run: float = 0.0
    mean_min_sep_m: float = 0.0
    material_time_share: list[float] = field(default_factory=lambda: [0.0] * NUM_MATERIALS)

    def to_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "n_episodes": self.n_episodes,
            "n_vehicle_runs": self.n_vehicle_runs,
            "success_rate": self.success_rate,
            "arrival_rate": self.arrival_rate,
            "mean_time_to_goal_s": self.mean_time_to_goal_s,
            "p95_time_to_goal_s": self.p95_time_to_goal_s,
            "mean_makespan_s": self.mean_makespan_s,
            "mean_path_ratio": self.mean_path_ratio,
            "mean_turn_total_rad": self.mean_turn_total_rad,
            "mean_fuel": self.mean_fuel,
            "mean_penetration_pct": self.mean_penetration_pct,
            "veh_contacts_per_run": self.veh_contacts_per_run,
            "mean_min_sep_m": self.mean_min_sep_m,
            "material_time_share": self.material_time_share,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def aggregate_nav(episodes: list[EpisodeStats], checkpoint: str = "") -> NavScorecard:
    """Reduce a corpus of episodes into one base scorecard row. Same reduction as the
    C++ ``aggregate_nav`` (success/arrival rates, mean+p95 ttg over arrivers, path
    ratio over straight>0 runs, fleet penetration, material share)."""
    s = NavScorecard(checkpoint=checkpoint, n_episodes=len(episodes))
    runs = arrived = contacts = ratio_n = sep_n = 0
    ttg_sum = makespan_sum = ratio_sum = turn_sum = fuel_sum = pen_sum = sep_sum = 0.0
    mat_sum = [0.0] * NUM_MATERIALS
    mat_total = 0.0
    ttgs: list[float] = []
    for e in episodes:
        makespan_sum += e.makespan_s
        pen_sum += e.penetration_pct
        if e.min_sep_m < 1e29:
            sep_sum += e.min_sep_m
            sep_n += 1
        for v in e.per_vehicle:
            runs += 1
            if v.arrived:
                arrived += 1
                ttg_sum += v.time_to_goal_s
                ttgs.append(v.time_to_goal_s)
            if v.straight_m > 1e-9:
                ratio_sum += v.total_path_m / v.straight_m
                ratio_n += 1
            turn_sum += v.turn_total_rad
            fuel_sum += v.fuel_used
            contacts += v.veh_contacts
            for m in range(NUM_MATERIALS):
                mat_sum[m] += v.time_over_material_s[m]
                mat_total += v.time_over_material_s[m]
    s.n_vehicle_runs = runs
    s.success_rate = sum(1 for e in episodes if e.success) / len(episodes) if episodes else 0.0
    s.arrival_rate = arrived / runs if runs else 0.0
    s.mean_time_to_goal_s = ttg_sum / arrived if arrived else 0.0
    if ttgs:
        ttgs.sort()
        k = int(math.ceil(0.95 * len(ttgs))) - 1
        s.p95_time_to_goal_s = ttgs[min(max(k, 0), len(ttgs) - 1)]
    s.mean_makespan_s = makespan_sum / len(episodes) if episodes else 0.0
    s.mean_path_ratio = ratio_sum / ratio_n if ratio_n else 0.0
    s.mean_turn_total_rad = turn_sum / runs if runs else 0.0
    s.mean_fuel = fuel_sum / runs if runs else 0.0
    s.mean_penetration_pct = pen_sum / len(episodes) if episodes else 0.0
    s.veh_contacts_per_run = contacts / runs if runs else 0.0
    s.mean_min_sep_m = sep_sum / sep_n if sep_n else 0.0
    if mat_total > 0:
        s.material_time_share = [mat_sum[m] / mat_total for m in range(NUM_MATERIALS)]
    return s
