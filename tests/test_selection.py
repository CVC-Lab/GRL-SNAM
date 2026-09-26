"""Track 4 Phase 1 SELECTION — composite_fitness + rank/select_best over NavScorecard rows.

Pins: (1) default weights reduce the composite to today's ranking signal (arrival_rate) so enabling
selection changes no ranking until a term is opted in; (2) each weighted term moves fitness in the
correct direction; (3) rank/select_best order best-first; (4) the fitness is total over a partial
(from_dict) row.
"""

import math

from grl_snam.scorecard import NavCoverage, NavScorecard
from grl_snam.selection import FitnessWeights, composite_fitness, rank, select_best


def test_default_fitness_is_arrival_rate():
    # Back-compat: with default weights the composite IS arrival_rate (the raw reach_rate the base
    # eval sorts on today) — nothing else contributes, so no recorded ranking changes on adoption.
    # EVERY non-arrival field set non-zero, so a stray nonzero default on ANY weight would break the
    # == arrival_rate identity (pins the back-compat guarantee for the whole weight vector).
    sc = NavScorecard(
        arrival_rate=0.7,
        success_rate=1.0,
        mean_time_to_goal_s=99.0,
        mean_path_ratio=1.4,
        mean_penetration_pct=5.0,
        veh_contacts_per_run=2.0,
        mean_min_sep_m=3.0,
        form_arrival_rate=1.0,
        form_mission_rate=1.0,
        mean_slot_error_m=50.0,
        mean_sense_flips=9.0,
        mean_mu=0.9,
        mean_mrisk=0.4,
        mean_closest_approach_m=6.0,
        mean_stall_steps=8.0,
        mean_coverage=NavCoverage(
            explored_frac=1.0, visible_frac=1.0, believed_free_frac=1.0, phantom_frac=1.0
        ),
    )
    assert math.isclose(composite_fitness(sc), 0.7)


def _dir(field, weight_kw, up):
    """A field moves fitness the expected way: build two cards differing only in `field`, weight it,
    and assert the larger-field card scores higher (up=True) or lower (up=False)."""
    lo, hi = NavScorecard(), NavScorecard()
    setattr(lo, field, 1.0)
    setattr(hi, field, 2.0)
    w = FitnessWeights(**{weight_kw: 1.0})
    d = composite_fitness(hi, w) - composite_fitness(lo, w)
    assert (d > 0) if up else (d < 0), f"{field} via {weight_kw}: delta {d} wrong sign"


def test_per_field_directions():
    # up = larger value -> higher fitness; down (a cost) = larger value -> lower fitness.
    _dir("arrival_rate", "w_arrival", up=True)
    _dir("success_rate", "w_success", up=True)
    _dir("mean_time_to_goal_s", "w_time", up=False)
    _dir("mean_path_ratio", "w_path", up=False)
    _dir("mean_penetration_pct", "w_penetration", up=False)
    _dir("veh_contacts_per_run", "w_contacts", up=False)
    _dir("mean_min_sep_m", "w_min_sep", up=True)
    _dir("form_arrival_rate", "w_form_arrival", up=True)
    _dir("form_mission_rate", "w_form_mission", up=True)
    _dir("mean_slot_error_m", "w_slot_error", up=False)
    _dir("mean_mu", "w_grip_mu", up=True)  # grip up
    _dir("mean_mrisk", "w_grip_mrisk", up=False)  # material risk down
    _dir("mean_closest_approach_m", "w_closest", up=False)  # closer = smaller = better
    _dir("mean_stall_steps", "w_stall", up=False)


def test_coverage_directions():
    # nested mean_coverage fields, weighted.
    def card(explored=0.0, believed=0.0, phantom=0.0):
        return NavScorecard(
            mean_coverage=NavCoverage(
                explored_frac=explored, believed_free_frac=believed, phantom_frac=phantom
            )
        )

    we = FitnessWeights(w_explored=1.0)
    assert composite_fitness(card(explored=0.8), we) > composite_fitness(card(explored=0.2), we)
    wb = FitnessWeights(w_believed_free=1.0)
    assert composite_fitness(card(believed=0.8), wb) > composite_fitness(card(believed=0.2), wb)
    wp = FitnessWeights(w_phantom=1.0)  # phantom is a cost -> more phantom scores LOWER
    assert composite_fitness(card(phantom=0.8), wp) < composite_fitness(card(phantom=0.2), wp)


def test_rank_and_select_best():
    a = NavScorecard(checkpoint="a", arrival_rate=0.5)
    b = NavScorecard(checkpoint="b", arrival_rate=0.9)
    c = NavScorecard(checkpoint="c", arrival_rate=0.7)
    ordered = rank([a, b, c])  # default weights -> by arrival_rate
    assert [s.checkpoint for s in ordered] == ["b", "c", "a"]
    assert select_best([a, b, c]).checkpoint == "b"
    assert select_best([]) is None
    # a weighting can flip the order: reward formation instead of arrival
    a.form_arrival_rate, b.form_arrival_rate, c.form_arrival_rate = 1.0, 0.0, 0.0
    w = FitnessWeights(w_arrival=0.0, w_form_arrival=1.0)
    assert select_best([a, b, c], w).checkpoint == "a"


def test_fitness_total_over_partial_row():
    # a partial recorded row (from_dict defaults the missing keys) must not raise, even with every
    # Phase-1 term weighted.
    sc = NavScorecard.from_dict({"checkpoint": "p", "arrival_rate": 0.5})
    w = FitnessWeights(
        w_form_arrival=1.0, w_explored=1.0, w_grip_mu=1.0, w_slot_error=1.0, w_closest=1.0
    )
    assert isinstance(composite_fitness(sc, w), float)
    assert math.isclose(composite_fitness(sc), 0.5)  # only arrival present -> default == arrival
