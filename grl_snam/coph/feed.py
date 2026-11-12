"""One crossing or hazard episode as a stream of plain messages: live, or from a T1 episode file.

The viewer (``grl-snam ariadne coph``) reads an episode only through this stream, so a live run
and a replay draw the same thing. A live run calls the vendored runner exactly as
``crossing_demo.run`` and ``hazard_demo.run`` do (:func:`job`), adding only the runner's
read-only ``boundary_callback`` and ``delivery_callback``. A replay reads the episode those
functions dump. Both pass through :func:`emit`, which walks the episode's lists from a cursor,
so a live run's messages equal its own episode file's (``tests/test_coph_feed.py``).

Messages are tuples of plain JSON values (lists, floats, ints, strings, bools):

- ``("pose", agent, t, [qx, qy, vx, vy], stage, version, done)``: a control boundary.
- ``("step", agent, t_end, [ux, uy], work, heat, kinetic_change, residual, observation_jump)``: a
  held-input step and its energy balance.
- ``("packet", identity, sender, recipient, [qx, qy], bytes, timestamp)``: a delivered packet,
  sent just before the update that took it in. ``bytes`` is the length of its compact JSON.
- ``("update", agent, t, tick, accepted, reason, shaping_jump, [identities], snapshot)``: an
  update and the learner after it. ``snapshot`` holds ``version``, ``damping``, ``policy``
  ``[pi_risk, pi_barrier]`` and ``stages``, each ``{id, bounds, centers, weights}``.
- ``("geometry", agent, t, kind, obstacle)``: an obstacle the agent sensed or received.
- ``("outcome", agent, success, collision, start_time, finish_time, goal_gap, cost, components,
  bytes)``: the end of the episode, with the runner's cost components (time, risk, effort,
  failure) and byte totals (payload, status, ack).

A live run sends the messages as they happen, a replay agent by agent; each carries its agent
and time, and :class:`~grl_snam.coph.timeline.Timeline` does not depend on the order.
"""

from __future__ import annotations

import json
from dataclasses import asdict

#: The arms ``grl-snam ariadne coph`` shows by default: communication off, then the established
#: arm (crossing: shared geometry both ways; hazard: the shared graded field).
DEFAULT_ARMS = {"crossing": ("disabled", "reciprocal_fixed"), "hazard": ("none", "field")}
DEFAULT_SEEDS = {"crossing": 76000, "hazard": 79000}
#: The demos' full horizons (crossing_demo.main's and hazard_demo.main's --steps defaults).
DEFAULT_STEPS = {"crossing": 2500, "hazard": 1800}
HAZARD_SCENES = ("mild", "moderate", "severe", "double_mild")


def arms(demo):
    """Every arm the demo runs."""
    if demo == "crossing":
        from .crossing_demo import ARMS

        return tuple(ARMS)
    from .hazard_demo import ARMS

    return tuple(ARMS)


def job(demo, arm, seed, steps, scene="severe"):
    """``(spec, channel, communication, kwargs)``: the arguments ``crossing_demo.run`` or
    ``hazard_demo.run`` pass to ``staggered_demo.execute`` for this arm, seed and horizon."""
    if demo == "crossing":
        from .cooperative import CommunicationConfig
        from .crossing_demo import (
            CrossingLearner,
            CrossingWorld,
            configuration,
            specification,
        )

        if arm not in arms(demo):
            raise ValueError(f"crossing has no arm {arm!r}: {', '.join(arms(demo))}")
        spec = specification(seed, steps)
        kwargs = dict(
            learner_factory=CrossingLearner,
            world_factory=CrossingWorld,
            config=configuration(steps),
        )
        return spec, arm, CommunicationConfig(byte_budget=4000000, ttl=300.0), kwargs
    if demo == "hazard":
        from .crossing_demo import configuration
        from .hazard_demo import ARMS, communication, learner, specification
        from .maze_family import NetworkWorld

        if arm not in ARMS:
            raise ValueError(f"hazard has no arm {arm!r}: {', '.join(ARMS)}")
        if scene not in HAZARD_SCENES:
            raise ValueError(f"hazard has no scene {scene!r}: {', '.join(HAZARD_SCENES)}")
        spec = specification(scene, seed, steps)
        factories = learner(spec, arm)

        def factory(cfg, x, goal, agent_id, communication):
            return factories[int(agent_id)](
                cfg, x, goal, agent_id=agent_id, communication=communication
            )

        kwargs = dict(
            learner_factory=factory,
            world_factory=NetworkWorld,
            config=configuration(spec["steps"]),
        )
        return spec, ARMS[arm][2], communication(arm), kwargs
    raise ValueError(f"no demo {demo!r}: crossing or hazard")


def world(demo, spec, agent=0):
    """The episode's world for ``agent``, built as the runner builds it (its truth risk field
    and corridors)."""
    from .records import Config

    _, _, _, kwargs = job(demo, _first_arm(demo), spec["seed"], spec["steps"], spec.get("scene"))
    cfg = kwargs["config"] or Config()
    w = kwargs["world_factory"](
        dict(
            spec,
            initial=spec["initials"][agent],
            goal=spec["goals"][agent],
            risk_weight=spec.get("risk_weights", (1.0, 1.0))[agent],
        ),
        cfg,
    )
    w.seed = spec["seed"] * 2 + agent
    return w


def _first_arm(demo):
    return DEFAULT_ARMS[demo][0]


# ── the stream ─────────────────────────────────────────────────────────────────────────────────


def plain(value):
    """``value`` as plain JSON values: tuples become lists, numpy scalars Python numbers, exactly
    as an episode file stores them."""
    return json.loads(json.dumps(value, default=_number))


def _number(x):
    try:
        return x.item()
    except AttributeError:
        return float(x)


def packet_bytes(packet: dict) -> int:
    """A packet's size, as ``Packet.bytes()`` counts it: its compact, key-sorted JSON."""
    return len(json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def new_cursor():
    """Where :func:`emit` has got to in each of an episode's lists (2 agents)."""
    return {k: [0, 0] for k in ("samples", "trajectories", "updates", "geometry_events")}


def _snapshot(post):
    return dict(
        version=int(post["version"]),
        damping=float(post["damping"]),
        policy=[float(v) for v in post["policy"]],
        stages=[
            dict(
                id=int(s["id"]),
                bounds=[float(v) for v in s["bounds"]],
                centers=s["centers"],
                weights=s["weights"],
                **({"local_residual": s["local_residual"]} if "local_residual" in s else {}),
            )
            for s in post["stages"]
        ],
    )


def emit(view, cursor):
    """The messages for everything in ``view`` past ``cursor``, which it advances.

    ``view`` holds an episode's ``samples``, ``trajectories``, ``updates`` and
    ``geometry_events`` (per agent), and ``packets``, a dict from identity to the packet as a
    dict (``asdict``). It is an episode record, or the runner's lists as they stand at a control
    boundary."""
    out = []
    for i in range(2):
        samples = view["samples"][i]
        for s in samples[cursor["samples"][i] :]:
            out.append(
                plain(
                    (
                        "pose",
                        i,
                        s["time"],
                        s["x"],
                        int(s["stage"]),
                        int(s["version"]),
                        bool(s["done"]),
                    )
                )
            )
        cursor["samples"][i] = len(samples)

        steps = view["trajectories"][i]
        for r in steps[cursor["trajectories"][i] :]:
            e = r["energy"]
            out.append(
                plain(
                    (
                        "step",
                        i,
                        r["end_time"],
                        r["u"],
                        e["work"],
                        e["heat"],
                        e["kinetic_change"],
                        e["residual"],
                        r.get("observation_shaping_jump", 0.0),
                    )
                )
            )
        cursor["trajectories"][i] = len(steps)

        updates = view["updates"][i]
        for u in updates[cursor["updates"][i] :]:
            for identity in u["packets"]:
                m = view["packets"][identity]
                out.append(
                    plain(
                        (
                            "packet",
                            identity,
                            int(m["sender"]),
                            int(m["recipient"]),
                            list(m["q"])[:2],
                            packet_bytes(plain(m)),
                            m["timestamp"],
                        )
                    )
                )
            d = u["diagnostic"]
            out.append(
                plain(
                    (
                        "update",
                        i,
                        u["time"],
                        int(u["tick"]),
                        bool(d["accepted"]),
                        str(d["reason"]),
                        d.get("shaping_jump", 0.0),
                        list(u["packets"]),
                        _snapshot(u["post"]),
                    )
                )
            )
        cursor["updates"][i] = len(updates)

        geometry = view["geometry_events"][i]
        for g in geometry[cursor["geometry_events"][i] :]:
            out.append(plain(("geometry", i, g["time"], g["kind"], g["obstacle"])))
        cursor["geometry_events"][i] = len(geometry)
    return out


def outcomes(record):
    """The ``outcome`` messages of a finished episode."""
    return [
        plain(
            (
                "outcome",
                int(a["agent"]),
                bool(a["success"]),
                bool(a["collision"]),
                a["start_time"],
                a["finish_time"],
                a["goal_gap"],
                a["cost"],
                a["components"],
                a["bytes"],
            )
        )
        for a in record["agents"]
    ]


def messages(record):
    """Every message of a finished episode (a T1 episode file's record), in stream order."""
    view = dict(
        samples=record["samples"],
        trajectories=record["trajectories"],
        updates=record["updates"],
        geometry_events=record["geometry_events"],
        packets={p["identity"]: p for p in record["packets"]},
    )
    return emit(view, new_cursor()) + outcomes(record)


def run(demo, arm, seed, steps, *, scene="severe", sink=None, out=None):
    """Run one arm live, exactly as the demo's own ``run`` does, sending each control boundary's
    new messages to ``sink(list)`` as they happen, then the outcomes. With ``out``, the episode is
    also written where the demo writes it (``out/episodes/<seed>_<arm>.json.gz``), so the run can
    be replayed. Returns the episode record."""
    from pathlib import Path

    import torch

    from .evaluation import dump
    from .staggered_demo import execute

    torch.set_num_threads(1)
    spec, channel, communication, kwargs = job(demo, arm, seed, steps, scene)
    cursor, packets = new_cursor(), {}
    sink = sink or (lambda _messages: None)

    def boundary(b):
        view = dict(
            samples=b["samples"],
            trajectories=b["trajectories"],
            updates=b["updates"],
            geometry_events=[getattr(a, "geometry_events", []) for a in b["agents"]],
            packets=packets,
        )
        batch = emit(view, cursor)
        if batch:
            sink(batch)

    def delivery(_agent, _x, m, _world, _tick):
        packets[m.identity] = asdict(m)

    r = execute(
        spec,
        channel,
        communication,
        boundary_callback=boundary,
        delivery_callback=delivery,
        **kwargs,
    )
    view = dict(
        samples=r["samples"],
        trajectories=r["trajectories"],
        updates=r["updates"],
        geometry_events=r["geometry_events"],
        packets=packets,
    )
    sink(emit(view, cursor) + outcomes(r))
    if demo == "hazard":  # as hazard_demo.run labels its episodes
        r["arm"] = arm
        r["channel"] = channel
    if out is not None:
        (Path(out) / "episodes").mkdir(parents=True, exist_ok=True)
        dump(Path(out) / "episodes" / f"{seed}_{arm}.json.gz", r)
    return r


def run_dir(demo, root, scene="severe"):
    """Where a ``grl-snam coph`` run under ``root`` keeps this demo's episodes' directory:
    ``root`` for crossing, ``root/<scene>`` for hazard (as ``hazard_demo.main`` lays it out)."""
    from pathlib import Path

    return Path(root) if demo == "crossing" else Path(root) / scene


def serve(queue, demo, arm, seed, steps, scene="severe", out=None):
    """:func:`run` in a worker process: puts ``("batch", arm, messages)`` on ``queue`` at each
    control boundary, then ``("done", arm, None)``, or ``("error", arm, traceback)``."""
    import traceback

    try:
        run(
            demo,
            arm,
            seed,
            steps,
            scene=scene,
            sink=lambda batch: queue.put(("batch", arm, batch)),
            out=out,
        )
        queue.put(("done", arm, None))
    except BaseException:  # noqa: BLE001 - reported to the viewer, which exits with it
        queue.put(("error", arm, traceback.format_exc()))
