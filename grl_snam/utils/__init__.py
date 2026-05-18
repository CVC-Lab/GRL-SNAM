"""Utility re-exports (`dijkstra`, `online_stage_manager`).

The implementation modules live under ``src/utils/`` for historical
reasons; this façade exposes them under ``grl_snam.utils``.
"""

from __future__ import annotations

from src.utils import dijkstra, online_stage_manager  # noqa: F401

__all__ = ["dijkstra", "online_stage_manager"]
