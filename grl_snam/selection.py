"""Checkpoint SELECTION over nav scorecards — turn a recorded ``NavScorecard`` into a scalar
fitness and rank a set of candidates (Track 4 Phase 1, SELECTION only).

This is deliberately separate from :mod:`grl_snam.scorecard` (which stays a pure schema + reducer):
selection reads finished ``NavScorecard`` rows and never serializes back, so it can never perturb the
C++<->Python parity surface (the composite is a free function, NOT a scorecard field). It also stays
RF-free — the DBG campaign composes an RF fitness on top of ``composite_fitness`` in ``grl_snam_dbg``;
nothing here touches the loss or the rollout (that is Phase 2).

Typical use — rank recorded corpus rows (the C++/native collector fills the formation/coverage/grip
fields that the Python base eval leaves at zero):

    from grl_snam.scorecard import NavScorecard
    from grl_snam.selection import FitnessWeights, rank
    cards = [NavScorecard.from_json(open(p).read()) for p in card_paths]
    best = rank(cards, FitnessWeights(w_form_arrival=1.0))[0]
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .scorecard import NavScorecard


@dataclass
class FitnessWeights:
    """Weights for :func:`composite_fitness` (higher fitness = better checkpoint). Each term is added
    with the sign that makes "more is better": an ``_up`` field rewards a larger value, a ``_down``
    field (a cost) is subtracted so a smaller value scores higher.

    DEFAULTS reduce the composite to exactly ``arrival_rate`` — today's ranking signal (the raw
    ``reach_rate`` the base eval sorts on) — so enabling this changes NO ranking until a caller opts a
    term in. The Phase-1 fields (formation / belief / grip-margin) are structurally zero in the Python
    base corpus, so their weights move the ranking only on recorded (``from_dict``) rows the native
    collector filled.
    """

    # arrival / economy / safety (the base terms; only arrival is on by default = today's behavior)
    w_arrival: float = 1.0  # arrival_rate — up
    w_success: float = 0.0  # success_rate (every vehicle arrived) — up
    w_time: float = 0.0  # mean_time_to_goal_s — down (economy)
    w_path: float = 0.0  # mean_path_ratio — down (economy)
    w_penetration: float = 0.0  # mean_penetration_pct — down (safety)
    w_contacts: float = 0.0  # veh_contacts_per_run — down (safety)
    w_min_sep: float = 0.0  # mean_min_sep_m — up (safety)
    # formation holding
    w_form_arrival: float = 0.0  # form_arrival_rate — up
    w_form_mission: float = 0.0  # form_mission_rate — up
    w_slot_error: float = 0.0  # mean_slot_error_m — down
    # belief coverage
    w_explored: float = 0.0  # mean_coverage.explored_frac — up
    w_believed_free: float = 0.0  # mean_coverage.believed_free_frac — up
    w_phantom: float = 0.0  # mean_coverage.phantom_frac — down (false-belief)
    # grip-margin (drive telemetry): grip up, material risk down
    w_grip_mu: float = 0.0  # mean_mu — up (more underfoot grip)
    w_grip_mrisk: float = 0.0  # mean_mrisk — down (material risk)
    # progress — kept a SEPARATE term, default-0: mean_closest_approach_m == 0 is ambiguous
    # (0 = reached AND = unmeasured / no finite-distance run), so an all-zero corpus must not be
    # rewarded. Enable only when the corpus is known to carry finite closest-approach.
    w_closest: float = 0.0  # mean_closest_approach_m — down (closer to goal)
    w_stall: float = 0.0  # mean_stall_steps — down


def composite_fitness(card: NavScorecard, weights: FitnessWeights | None = None) -> float:
    """Scalar fitness of one scorecard, higher = better. A weighted linear combination (weights carry
    the scale); with default weights this is exactly ``card.arrival_rate``. Total over any card — a
    partial ``from_dict`` row (missing keys defaulted) never raises."""
    w = weights or FitnessWeights()
    cov = card.mean_coverage
    return (
        w.w_arrival * card.arrival_rate
        + w.w_success * card.success_rate
        - w.w_time * card.mean_time_to_goal_s
        - w.w_path * card.mean_path_ratio
        - w.w_penetration * card.mean_penetration_pct
        - w.w_contacts * card.veh_contacts_per_run
        + w.w_min_sep * card.mean_min_sep_m
        + w.w_form_arrival * card.form_arrival_rate
        + w.w_form_mission * card.form_mission_rate
        - w.w_slot_error * card.mean_slot_error_m
        + w.w_explored * cov.explored_frac
        + w.w_believed_free * cov.believed_free_frac
        - w.w_phantom * cov.phantom_frac
        + w.w_grip_mu * card.mean_mu
        - w.w_grip_mrisk * card.mean_mrisk
        - w.w_closest * card.mean_closest_approach_m
        - w.w_stall * card.mean_stall_steps
    )


def rank(cards: list[NavScorecard], weights: FitnessWeights | None = None) -> list[NavScorecard]:
    """Cards sorted best-first by :func:`composite_fitness` (stable; ties keep input order)."""
    w = weights or FitnessWeights()
    return sorted(cards, key=lambda c: composite_fitness(c, w), reverse=True)


def select_best(
    cards: list[NavScorecard], weights: FitnessWeights | None = None
) -> NavScorecard | None:
    """The single best card by :func:`composite_fitness`, or None for an empty list. On a tie the
    first in input order wins (max is stable)."""
    if not cards:
        return None
    w = weights or FitnessWeights()
    return max(cards, key=lambda c: composite_fitness(c, w))


def _main(argv: list[str] | None = None) -> int:
    """`python -m grl_snam.selection card1.json card2.json [--weights '{"w_form_arrival":1.0}']` —
    load recorded scorecard rows (cvc::nav ``scorecard_json`` / ``NavScorecard.to_json``) and print
    them best-first by composite fitness. This is the offline SELECTION entry point; the recorded
    (``from_json``) row is where the formation/coverage/grip fields are actually non-zero."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Rank nav scorecards by composite fitness (SELECTION)."
    )
    ap.add_argument("cards", nargs="+", help="scorecard JSON files")
    ap.add_argument(
        "--weights",
        default="",
        help="JSON object of FitnessWeights overrides, e.g. '{\"w_form_arrival\":1.0}'",
    )
    args = ap.parse_args(argv)
    w = FitnessWeights(**json.loads(args.weights)) if args.weights else FitnessWeights()
    rows = []
    for p in args.cards:
        with open(p) as f:
            sc = NavScorecard.from_json(f.read())
        rows.append((composite_fitness(sc, w), sc.checkpoint or p))
    rows.sort(key=lambda r: r[0], reverse=True)
    for score, label in rows:
        print(f"{score:.6f}\t{label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
