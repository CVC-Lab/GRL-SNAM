"""The dense 13-class material palette — the id vocabulary the nav-stats material
buckets are indexed by.

This is the SINGLE source of the palette shared across the stack: it mirrors
``cvc::nav::kNumMaterials`` (libcvc ``inc/cvc/nav/nav_stats.h``) and is positional
with the ``cvc::dbg`` ``MATERIAL_TABLE`` (cvcdbg ``src/cvc/dbg/channel.cpp``), so a
material id means the same class in the C++ collector, the DBG RF layer, and here.
Keep the ORDER byte-for-byte with that table — the ids are the wire format.

The discrete buckets (``time_over_material_s`` / ``dist_over_material_m``) are the
parity field with the C++ scorecard. "Time in terrain-risk areas" is DERIVED from
them — :func:`terrain_risk_share` sums the traversable outdoor classes that carry
mobility risk — rather than stored as a separate schema field, so the scorecard
stays field-for-field with ``cvc::dbg::nav_scorecard``.
"""

from __future__ import annotations

# Positional — index == material id. MUST match cvc::dbg MATERIAL_TABLE order.
MATERIALS = (
    "reinforced_concrete",  # 0
    "brick",  # 1
    "glass",  # 2
    "wood",  # 3
    "foliage",  # 4
    "drywall",  # 5
    "metal",  # 6
    "open_air",  # 7
    "soil",  # 8
    "water",  # 9
    "glass_laminated",  # 10
    "composite_panel",  # 11
    "rock",  # 12
)
NUM_MATERIALS = len(MATERIALS)  # 13 == cvc::nav::kNumMaterials
MATERIAL_ID = {name: i for i, name in enumerate(MATERIALS)}

#: The traversable OUTDOOR terrain classes that carry mobility/terrain risk
#: (mud/rubble/soft ground/standing water), as opposed to ``open_air`` (free
#: driving) and the building materials (which are occupancy — not driven through).
#: This classification is the nav-domain reading of the shared palette; it is used
#: only to DERIVE a terrain-risk summary from the parity buckets, never stored.
RISK_MATERIAL_IDS = (
    MATERIAL_ID["foliage"],
    MATERIAL_ID["soil"],
    MATERIAL_ID["water"],
    MATERIAL_ID["rock"],
)

OPEN_AIR_ID = MATERIAL_ID["open_air"]


def terrain_risk_share(material_time_share) -> float:
    """Fraction of fleet time spent over terrain-risk materials — the "how often the
    rollout spends time in terrain-risk areas" number — summed from the parity
    ``material_time_share`` vector over :data:`RISK_MATERIAL_IDS`."""
    return float(sum(material_time_share[i] for i in RISK_MATERIAL_IDS))
