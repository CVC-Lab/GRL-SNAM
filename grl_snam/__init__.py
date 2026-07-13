"""GRL-SNAM: Geometric Reinforcement Learning for Simultaneous Navigation and Mapping.

This module is a thin façade over the project's flat module layout
(`train_coef_energy.py`, `surrogate_robust.py`, `eval_coef_energy.py`,
`src/utils/`, `scripts/`, `experiments/`).  The flat layout is preserved
for backward compatibility with the original research code; this package
exposes a stable, importable API for downstream consumers (e.g. the
`grl_snam_dbg` extension).

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
    __version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "CoefEnergyNet",
    "integrate_surrogate",
]


def __getattr__(name: str):
    """Lazy attribute access so heavy imports (torch) happen on demand."""
    if name in {"CoefEnergyNet", "integrate_surrogate"}:
        from train_coef_energy import CoefEnergyNet, integrate_surrogate  # noqa: PLC0415

        return {"CoefEnergyNet": CoefEnergyNet, "integrate_surrogate": integrate_surrogate}[name]
    raise AttributeError(f"module 'grl_snam' has no attribute {name!r}")
