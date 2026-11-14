"""GRL-SNAM's demos as Python hosts of Ariadne (``.ari``) documents.

Each port runs a GRL-SNAM demo (or replays its recording) from Python and shows it in an Ariadne
document: the document (``docs/*.ari``) declares the panels, the controls and a soft afternoon
look; the port steps or replays the simulation, draws it into the document's scene, and writes
the readouts the panels show. Run one with::

    python -m grl_snam.ariadne <port> [args]        # --list names them; <port> --help explains

Every port takes ``--offscreen`` (GL with no window), ``--headless`` (no GL), ``--frames N`` and
``--png PATH``. They need pycvc_gl's Python host (``pycvc_gl.ariadne``, libcvc 3.7.9+); see
README.md beside this file for each port's data and how it maps to the original demo.
"""

from __future__ import annotations

#: port name -> (module, what it shows, the original demo(s) it ports)
PORTS = {
    "live": ("grl_snam.ariadne.live",
             "the live Austin demos (freedrive, learned, patrol, planner)",
             "grl-snam demo austin-*"),
    "fog": ("grl_snam.ariadne.fog", "a fog-of-war story, recorded then replayed",
            "grl-snam fog record / capture / play / all"),
    "squad": ("grl_snam.ariadne.squad",
              "a squad recording on Austin (finale acts, convoy, pursuit) with Humvees",
              "finale, convoy_record, pursuit_record, squad_capture"),
    "fleet": ("grl_snam.ariadne.fleet",
              "a live squad of Humvees, on the plant and in the world you pick",
              "Squad(plant=...)"),
    "material": ("grl_snam.ariadne.material", "material-aware navigation, off vs on",
                 "grl-snam material-demo"),
    "cars": ("grl_snam.ariadne.cars",
             "N Ackermann cars driven by your own controller: libcvc's template (--copy DIR)",
             "new: libcvc's pycvc_gl.cars, GRL-SNAM's policy as a baseline"),
    "coph": ("grl_snam.ariadne.coph",
             "CoPH-Stagewise's crossing and hazard episodes, communication off beside CoPH",
             "grl-snam coph crossing|hazard (live, or --from a run)"),
    "learn": ("grl_snam.ariadne.learn",
              "did it learn? held-out curves, heads, checkpoints, arms and the claims",
              "new: grl-snam learn-report's verdict, from learning histories"),
    "replay": ("grl_snam.ariadne.replay",
               "a recorded live --arms run: the ghosts, Stats and Ledger at the play head",
               "new: a run log, from live austin-freedrive --arms ... --log"),
}  # fmt: skip


def registry() -> dict[str, str]:
    """``{port: what it shows}``."""
    return {k: v[1] for k, v in PORTS.items()}
