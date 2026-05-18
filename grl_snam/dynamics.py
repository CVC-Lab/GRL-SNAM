"""Surrogate dynamics and multi-start robustness penalty.

Re-exports symbols from the top-level ``surrogate_robust`` module so
that downstream callers can ``from grl_snam.dynamics import ...``
without touching the flat layout.
"""

from __future__ import annotations

from surrogate_robust import (  # noqa: F401
    integrate_surrogate_v2,
    multi_start_penalty,
)

__all__ = ["integrate_surrogate_v2", "multi_start_penalty"]
