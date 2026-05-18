"""Policy network re-exports.

`CoefEnergyNet` is the main learned coefficient network defined in
``train_coef_energy.py``.  This module provides a stable import path
for downstream packages.
"""

from __future__ import annotations

from train_coef_energy import CoefEnergyNet  # noqa: F401

__all__ = ["CoefEnergyNet"]
