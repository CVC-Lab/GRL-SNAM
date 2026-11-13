"""Is a learner learning, or fitting noise? The reader, the curve marks, the statistics and the
report that answer it on held-out data.

A falling training loss proves nothing by itself: it can fall while the deployed behaviour stays
where it was. The proof here has three parts, each on held-out data and each against both an
untrained arm (the policy at step 0) and a constant arm (fixed coefficients, no state
dependence):

1. the per-term losses that carry gradient (:mod:`.history`, :mod:`.curves`);
2. the task metric on paired seeds, with a bootstrap CI (:mod:`.stats`, :mod:`.arms`);
3. in a planted exercise, the error of the recovered latent (:mod:`.report`).

Modules:

- :mod:`.history` reads JSON-lines learning histories (any trainer that follows the shared
  conventions) into named series, a checkpoint table and train/held-out term pairs.
- :mod:`.curves` marks a held-out curve: best, elbow (Kneedle), plateau and divergence.
- :mod:`.stats` is a seeded percentile bootstrap and paired differences.
- :mod:`.arms` builds the ``grl_snam.arms/v1`` table (per seed, mean, CI, paired difference) and
  the constant arm's policy.
- :mod:`.report` turns histories and an arms table into claims, each marked pass or fail
  (``grl-snam learn-report``).

Nothing here imports torch except :mod:`.arms`'s policy builders, which import it when called.
"""

from .curves import MarkParams, Marks, marks, median_smooth
from .history import Band, Checkpoint, History, Runs, Series, TermPair, band, load
from .stats import bootstrap_ci, paired_diff, summarize

__all__ = [
    "Band",
    "Checkpoint",
    "History",
    "MarkParams",
    "Marks",
    "Runs",
    "Series",
    "TermPair",
    "band",
    "bootstrap_ci",
    "load",
    "marks",
    "median_smooth",
    "paired_diff",
    "summarize",
]
