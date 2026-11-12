"""CoPH-Stagewise: cooperative stagewise port-Hamiltonian navigation, vendored.

The method's core (plant, energy, risk field, stage ledger, learner, packets) and the crossing
and hazard demos, copied from SetasAditya/CoPH-Sense at a pinned commit. ``VENDORED.md`` lists
the commit, what was copied, what was left out and every local change. The code keeps
upstream's formatting so that re-syncing stays a mechanical diff.

Run the demos with ``grl-snam coph crossing|hazard --out DIR``; ``grl_snam.coph.results``
summarizes their episodes into ``results.json``.
"""

UPSTREAM = dict(
    repository="SetasAditya/CoPH-Sense",
    commit="c1f1ff34d0db98c1ac61240778266d6e49b9c853",
    path="CoPH-Stagewise/coph_stagewise",
    license="MIT",
)
