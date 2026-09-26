"""Base navigation SCORECARD — aggregate a corpus of episodes into one fitness row.

This is the Python twin of the C++ ``cvc::nav::nav_scorecard`` (transfix/libcvc
``inc/cvc/nav/nav_stats.h``): the RF-free nav-fitness row grl-snam uses to rank its
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

from .material_palette import NUM_MATERIALS  # 13 == cvc::dbg kNumMaterials (channel.cpp table)


@dataclass
class NavCoverage:
    """Fleet belief-coverage fractions over an episode's belief planes — the Python twin of
    C++ ``cvc::nav::nav_coverage``. Each is a fraction in [0,1] of the (planes x cells) entries."""

    explored_frac: float = 0.0
    visible_frac: float = 0.0
    believed_free_frac: float = 0.0
    phantom_frac: float = 0.0


@dataclass
class VehStats:
    """One vehicle's per-episode outcome — the fields the base scorecard reduces. Field-for-field
    with the reduced subset of C++ ``cvc::nav::veh_nav_stats``."""

    arrived: bool = False
    time_to_goal_s: float = -1.0  # < 0 = never
    total_path_m: float = 0.0
    straight_m: float = 0.0  # straight-line start->goal (for the path ratio)
    turn_total_rad: float = 0.0
    fuel_used: float = 0.0
    veh_contacts: int = 0
    penetration_steps: int = 0
    time_over_material_s: list[float] = field(default_factory=lambda: [0.0] * NUM_MATERIALS)
    dist_over_material_m: list[float] = field(default_factory=lambda: [0.0] * NUM_MATERIALS)
    # identity / formation linkage (convoy_id + formation_parent key the per-episode form-mission;
    # -1 parent = the objective/anchor, or formation off). convoy_id must be >= 0 — the shared contract
    # the C++ side relies on (it keys the form-mission by a vector index; negative ids are undefined).
    convoy_id: int = 0
    formation_parent: int = -1
    # formation holding (0/off unless a formation_slot feed was used)
    slot_error_mean_m: float = 0.0
    formation_arrived: bool = False
    # progress toward goal (closest_approach seeded to straight_m by the collector; 1e30 = unmeasured)
    stall_steps: int = 0
    closest_approach_m: float = 1e30
    # epistemic churn
    sense_flips: int = 0
    # drive telemetry — per-vehicle reductions (0/off unless a drive feed was used). drive_steps is the
    # tick count the means average over.
    drive_steps: int = 0
    alpha_mean: float = 0.0
    beta_mean: float = 0.0
    gamma_mean: float = 0.0
    mu_mean: float = 0.0
    mrisk_mean: float = 0.0
    ext_force_mean: float = 0.0


@dataclass
class EpisodeStats:
    """One eval episode — a reduction target for the scorecard. Mirrors the base half
    of the C++ ``episode_nav_stats``."""

    success: bool = False  # every vehicle arrived (and within budget)
    makespan_s: float = 0.0  # max arrival time over the fleet
    penetration_pct: float = 0.0  # fleet penetration steps / total steps * 100
    min_sep_m: float = 1e30  # smallest inter-vehicle distance over the run
    coverage: NavCoverage = field(
        default_factory=NavCoverage
    )  # fleet belief coverage (0 unless fed)
    per_vehicle: list[VehStats] = field(default_factory=list)

    @staticmethod
    def from_nav_stats(
        vehicles: list,
        *,
        straights_m: list[float] | None = None,
        arrival_times_s: list[float] | None = None,
        veh_contacts: list[int] | None = None,
        min_sep_m: float = 1e30,
    ) -> EpisodeStats:
        """Build an EpisodeStats from a list of per-vehicle ``grl_snam.metrics.NavStats``.
        ``straights_m`` / ``arrival_times_s`` / ``veh_contacts`` (per vehicle) supply the
        fields NavStats does not itself carry — ``veh_contacts[i]`` is agent i's frame count
        within the contact radius, the pairwise quantity the episode owns (like ``min_sep_m``),
        so ``veh_contacts_per_run`` is non-zero on collisions. A vehicle counts as arrived when
        it reached >= 1 goal. (NavStats.turn_total_rad / fuel_used are read when present.)"""
        zeros = [0.0] * NUM_MATERIALS
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
                    veh_contacts=(veh_contacts[i] if veh_contacts else 0),
                    penetration_steps=getattr(ns, "penetration_steps", 0),
                    # Per-material buckets the accumulator carries (all-zero when no
                    # material id was ever supplied); copied so VehStats owns its lists.
                    time_over_material_s=list(getattr(ns, "time_over_material_s", zeros)),
                    dist_over_material_m=list(getattr(ns, "dist_over_material_m", zeros)),
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
    keys are identical to the C++ ``cvc::nav::nav_scorecard``."""

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
    # progress (mean_closest_approach over runs with a finite value; 0 when none — matches C++ /
    # mean_min_sep_m). mean_stall_steps over ALL vehicle-runs.
    mean_closest_approach_m: float = 0.0
    mean_stall_steps: float = 0.0
    material_time_share: list[float] = field(default_factory=lambda: [0.0] * NUM_MATERIALS)
    # formation holding (0 unless a formation feed was used)
    form_arrival_rate: float = 0.0
    form_mission_rate: float = 0.0
    mean_slot_error_m: float = 0.0
    # epistemic. mean_coverage divides by ALL n_episodes and mean_sense_flips by ALL vehicle-runs
    # (feature-off = 0 contributions still counted) — the divide-by-total convention the C++ side pins,
    # distinct from the finite-only mean_min_sep/closest rule.
    mean_coverage: NavCoverage = field(default_factory=NavCoverage)
    mean_sense_flips: float = 0.0
    # drive telemetry, averaged over the vehicle-runs that carried a drive feed (drive_steps > 0)
    mean_alpha: float = 0.0
    mean_beta: float = 0.0
    mean_gamma: float = 0.0
    mean_mu: float = 0.0
    mean_mrisk: float = 0.0
    mean_ext_force: float = 0.0

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
            "mean_closest_approach_m": self.mean_closest_approach_m,
            "mean_stall_steps": self.mean_stall_steps,
            "material_time_share": self.material_time_share,
            "form_arrival_rate": self.form_arrival_rate,
            "form_mission_rate": self.form_mission_rate,
            "mean_slot_error_m": self.mean_slot_error_m,
            "mean_coverage": {
                "explored_frac": self.mean_coverage.explored_frac,
                "visible_frac": self.mean_coverage.visible_frac,
                "believed_free_frac": self.mean_coverage.believed_free_frac,
                "phantom_frac": self.mean_coverage.phantom_frac,
            },
            "mean_sense_flips": self.mean_sense_flips,
            "mean_alpha": self.mean_alpha,
            "mean_beta": self.mean_beta,
            "mean_gamma": self.mean_gamma,
            "mean_mu": self.mean_mu,
            "mean_mrisk": self.mean_mrisk,
            "mean_ext_force": self.mean_ext_force,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> NavScorecard:
        """Parse a scorecard dict back into a ``NavScorecard`` — the inverse of :meth:`to_dict`, and
        the reader for the C++ ``cvc::nav::scorecard_json`` output (the keys are field-for-field the
        same, so a ``dbg_arrival_check3 --episodes`` / ``scorecard_json`` row loads directly). This is
        the plumbing a selection/loss stage reads a recorded corpus fitness from; nothing consumes it
        yet (Track 4 Phase 1+ wires it into training).

        Tolerant of a partial producer: any missing key keeps the dataclass default, so an older or
        newer writer (or one that omits zero fields) still loads. An ``rf`` sub-object, if present
        (the DBG ``scorecard_json`` embeds one), is ignored here — the base reader is RF-free; the DBG
        side reads the RF scorecard separately.
        """
        cov = d.get("mean_coverage") or {}
        return cls(
            checkpoint=str(d.get("checkpoint", "")),
            n_episodes=int(d.get("n_episodes", 0)),
            n_vehicle_runs=int(d.get("n_vehicle_runs", 0)),
            success_rate=float(d.get("success_rate", 0.0)),
            arrival_rate=float(d.get("arrival_rate", 0.0)),
            mean_time_to_goal_s=float(d.get("mean_time_to_goal_s", 0.0)),
            p95_time_to_goal_s=float(d.get("p95_time_to_goal_s", 0.0)),
            mean_makespan_s=float(d.get("mean_makespan_s", 0.0)),
            mean_path_ratio=float(d.get("mean_path_ratio", 0.0)),
            mean_turn_total_rad=float(d.get("mean_turn_total_rad", 0.0)),
            mean_fuel=float(d.get("mean_fuel", 0.0)),
            mean_penetration_pct=float(d.get("mean_penetration_pct", 0.0)),
            veh_contacts_per_run=float(d.get("veh_contacts_per_run", 0.0)),
            mean_min_sep_m=float(d.get("mean_min_sep_m", 0.0)),
            mean_closest_approach_m=float(d.get("mean_closest_approach_m", 0.0)),
            mean_stall_steps=float(d.get("mean_stall_steps", 0.0)),
            # missing/null -> the 13-zero default (NOT []), matching the dataclass default so a
            # partial producer that drops the all-zero share still loads a length-NUM_MATERIALS list
            # (a downstream `share[m]` never IndexErrors).
            material_time_share=[
                float(x) for x in (d.get("material_time_share") or [0.0] * NUM_MATERIALS)
            ],
            form_arrival_rate=float(d.get("form_arrival_rate", 0.0)),
            form_mission_rate=float(d.get("form_mission_rate", 0.0)),
            mean_slot_error_m=float(d.get("mean_slot_error_m", 0.0)),
            mean_coverage=NavCoverage(
                explored_frac=float(cov.get("explored_frac", 0.0)),
                visible_frac=float(cov.get("visible_frac", 0.0)),
                believed_free_frac=float(cov.get("believed_free_frac", 0.0)),
                phantom_frac=float(cov.get("phantom_frac", 0.0)),
            ),
            mean_sense_flips=float(d.get("mean_sense_flips", 0.0)),
            mean_alpha=float(d.get("mean_alpha", 0.0)),
            mean_beta=float(d.get("mean_beta", 0.0)),
            mean_gamma=float(d.get("mean_gamma", 0.0)),
            mean_mu=float(d.get("mean_mu", 0.0)),
            mean_mrisk=float(d.get("mean_mrisk", 0.0)),
            mean_ext_force=float(d.get("mean_ext_force", 0.0)),
        )

    @classmethod
    def from_json(cls, s: str) -> NavScorecard:
        """Parse a scorecard JSON string (``to_json`` / C++ ``scorecard_json``) into a
        ``NavScorecard``. See :meth:`from_dict`."""
        return cls.from_dict(json.loads(s))


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
    # formation holding (followers = formation_parent >= 0; anchors = -1)
    foll_runs = foll_arr = form_episodes = form_mission_ok = 0
    slot_sum = 0.0
    # progress (closest-approach over runs with a finite value only)
    approach_n = 0
    approach_sum = 0.0
    stall_sum = 0
    # epistemic (coverage over episodes; sense_flips over vehicle-runs)
    cov_sum = NavCoverage()
    flips_sum = 0
    # drive telemetry (over vehicle-runs that carried a drive feed, drive_steps > 0)
    drive_runs = 0
    al_sum = be_sum = ga_sum = dmu_sum = dmrisk_sum = dext_sum = 0.0
    for e in episodes:
        makespan_sum += e.makespan_s
        pen_sum += e.penetration_pct
        cov_sum.explored_frac += e.coverage.explored_frac
        cov_sum.visible_frac += e.coverage.visible_frac
        cov_sum.believed_free_frac += e.coverage.believed_free_frac
        cov_sum.phantom_frac += e.coverage.phantom_frac
        if e.min_sep_m < 1e29:
            sep_sum += e.min_sep_m
            sep_n += 1
        # per-episode formation mission: every convoy that has followers has its anchor arrived AND all
        # its followers in-slot. Keyed by convoy_id.
        conv_has_foll: dict[int, bool] = {}
        conv_ok: dict[int, bool] = {}
        for v in e.per_vehicle:
            conv_ok.setdefault(v.convoy_id, True)
            if v.formation_parent >= 0:  # follower
                conv_has_foll[v.convoy_id] = True
                if not v.formation_arrived:
                    conv_ok[v.convoy_id] = False
            elif not v.arrived:  # anchor/lead must reach the objective
                conv_ok[v.convoy_id] = False
        any_formation = any(conv_has_foll.values())
        if any_formation:
            form_episodes += 1
            if all(conv_ok[c] for c in conv_has_foll if conv_has_foll[c]):
                form_mission_ok += 1
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
            stall_sum += v.stall_steps
            flips_sum += v.sense_flips
            if v.drive_steps > 0:
                drive_runs += 1
                al_sum += v.alpha_mean
                be_sum += v.beta_mean
                ga_sum += v.gamma_mean
                dmu_sum += v.mu_mean
                dmrisk_sum += v.mrisk_mean
                dext_sum += v.ext_force_mean
            if v.closest_approach_m < 1e29:
                approach_sum += v.closest_approach_m
                approach_n += 1
            for m in range(NUM_MATERIALS):
                mat_sum[m] += v.time_over_material_s[m]
                mat_total += v.time_over_material_s[m]
            if v.formation_parent >= 0:  # followers only (the anchor has no slot)
                foll_runs += 1
                if v.formation_arrived:
                    foll_arr += 1
                slot_sum += v.slot_error_mean_m
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
    s.mean_closest_approach_m = approach_sum / approach_n if approach_n else 0.0
    s.mean_stall_steps = stall_sum / runs if runs else 0.0
    s.form_arrival_rate = foll_arr / foll_runs if foll_runs else 0.0
    s.form_mission_rate = form_mission_ok / form_episodes if form_episodes else 0.0
    s.mean_slot_error_m = slot_sum / foll_runs if foll_runs else 0.0
    if episodes:
        n = len(episodes)
        s.mean_coverage = NavCoverage(
            explored_frac=cov_sum.explored_frac / n,
            visible_frac=cov_sum.visible_frac / n,
            believed_free_frac=cov_sum.believed_free_frac / n,
            phantom_frac=cov_sum.phantom_frac / n,
        )
    s.mean_sense_flips = flips_sum / runs if runs else 0.0
    if drive_runs:
        s.mean_alpha = al_sum / drive_runs
        s.mean_beta = be_sum / drive_runs
        s.mean_gamma = ga_sum / drive_runs
        s.mean_mu = dmu_sum / drive_runs
        s.mean_mrisk = dmrisk_sum / drive_runs
        s.mean_ext_force = dext_sum / drive_runs
    return s


def compute_coverage(truth, belief, everseen, lastvis, planes: int, cells: int) -> NavCoverage:
    """Fleet belief coverage over one episode's rasters — the Python twin of C++
    ``cvc::nav::compute_coverage``. ``belief``/``everseen``/``lastvis`` are length ``planes*cells``
    flattened PLANE-MAJOR (plane m at ``m*cells``, C-order within a plane); ``truth`` is length
    ``cells`` — ONE shared plane, BROADCAST across every belief plane (indexed by the in-plane cell).
    ``belief`` is the BINARY ``to_occupancy`` output (0 = free, nonzero = occupied); to match C++ both
    sides must binarize with the SAME p_thresh 0.5 / band 0.15 / optimistic policy. Fractions are over
    ``planes*cells``: explored = everseen set, visible = lastvis set, believed_free = belief == 0,
    phantom = belief != 0 AND truth == 0. Any missing input or non-positive size yields all zero."""
    cov = NavCoverage()
    if truth is None or belief is None or everseen is None or lastvis is None:
        return cov
    if planes <= 0 or cells <= 0:
        return cov
    explored = visible = free = phantom = 0
    for m in range(planes):
        base = m * cells
        for c in range(cells):
            idx = base + c
            if everseen[idx]:
                explored += 1
            if lastvis[idx]:
                visible += 1
            if belief[idx] == 0:
                free += 1
            elif truth[c] == 0:  # believed occupied where truth is free -> a phantom obstacle
                phantom += 1
    total = float(planes * cells)
    cov.explored_frac = explored / total
    cov.visible_frac = visible / total
    cov.believed_free_frac = free / total
    cov.phantom_frac = phantom / total
    return cov
