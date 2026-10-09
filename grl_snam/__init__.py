"""GRL-SNAM: Geometric Reinforcement Learning for Simultaneous Navigation and Mapping.

This module is a thin façade over the project's flat module layout
(`train_coef_energy.py`, `surrogate_robust.py`, `eval_coef_energy.py`,
`src/utils/`, `scripts/`, `experiments/`).  The flat layout is preserved
for backward compatibility with the original research code; this package
exposes a stable, importable API for downstream consumers (e.g. an
application-specific extension layer).

Typical use::

    from grl_snam import CoefEnergyNet
    from grl_snam.dynamics import integrate_surrogate_v2, multi_start_penalty
    from grl_snam.adaptation import HistSecantController

All re-exports are lazy where possible to keep import cost low.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("grl-snam")
except PackageNotFoundError:  # pragma: no cover - source checkouts
    __version__ = "0.2.1.dev0"

__all__ = [
    "__version__",
    "CoefEnergyNet",
    "integrate_surrogate",
]


def __getattr__(name: str):
    """Lazy attribute access so heavy imports (torch) happen on demand."""
    if name == "CoefEnergyNet":
        from grl_snam.train_coef_energy import CoefEnergyNet  # noqa: PLC0415

        return CoefEnergyNet
    if name == "integrate_surrogate":
        # The semi-implicit integrator the trainer, the runtime stepper and cvc::nav all use —
        # not the legacy explicit-Euler train_coef_energy.integrate_surrogate_explicit.
        from grl_snam.surrogate_robust import integrate_surrogate_v2  # noqa: PLC0415

        return integrate_surrogate_v2
    raise AttributeError(f"module 'grl_snam' has no attribute {name!r}")
