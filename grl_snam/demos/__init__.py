"""grl_snam.demos — the live VolRover3 demos, in the package (previously ``examples/``).

Each demo is a module defining ``step(dt)`` (and usually ``setup()``); they run inside
VolRover3 via ``grl-snam demo <name>`` (which shells out to ``volrover3 --run-job``) or
the app's Jobs tab -> Load Script. Host-specific deps (``pycvc``, ``vrhost``,
``pycvc_gl``) import lazily, so importing this package never needs the running host.
"""

from __future__ import annotations

import importlib.util

_DEMOS = {
    "austin-freedrive": (
        "grl_snam.demos.austin_freedrive",
        "end-to-end learned SDF free-drive on real Austin (no route)",
    ),
    "austin-learned": (
        "grl_snam.demos.austin_learned",
        "stagewise: A* route spine + learned SDF local control",
    ),
    "austin-planner": (
        "grl_snam.demos.austin_planner",
        "surrogate planner in its native sparse-obstacle regime",
    ),
    "austin-patrol": (
        "grl_snam.demos.austin_patrol",
        "grounded A* street patrol (non-learned baseline)",
    ),
    "lab": ("grl_snam.demos.lab", "analytic terrain + an agent walking a draped loop"),
}


def registry() -> dict[str, str]:
    """``{demo_name: description}`` for every registered demo."""
    return {k: v[1] for k, v in _DEMOS.items()}


def demo_path(name: str) -> str | None:
    """Absolute file path of a demo module (for ``volrover3 --run-job``), or None if
    unknown. Resolves without importing the module (so no host deps are needed)."""
    entry = _DEMOS.get(name)
    if entry is None:
        return None
    spec = importlib.util.find_spec(entry[0])
    return spec.origin if spec and spec.origin else None
