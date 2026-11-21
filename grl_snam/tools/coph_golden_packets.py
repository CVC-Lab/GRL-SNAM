"""G9–G10 of libcvc's CoPH golden: packets, their byte count, and their assimilation.

``grl_snam.tools.coph_golden`` writes these sections into ``coph_golden.json`` (libcvc's
``NAV-COPH-ROADMAP.md`` §4.4, §5; C4). They pin the demos' learner (``CrossingLearner``:
``MazeLearner`` on ``TerrainLearner`` on ``CooperativeLearner``) at the snapshots of G5–G8's
scripted crossing run (``coph_golden_learner``).

- G9 (``packets.exports``): ``export`` at the three snapshots, with the block bootstraps' resample
  indices injected (roadmap D8). It records both bootstraps (``CooperativeLearner``'s and
  ``TerrainLearner``'s), the anchors and covariance each gives, H and the task energy, and the
  packet. The packet is recorded in its *wire* form: the export with its anchors' values and
  gradients, its covariance, H and V_task rounded to multiples of 2^-24, as G5–G8 round what an
  update commits. ``bytes()``, the length of the packet's compact JSON (D9), is BIT on the wire
  packet. On the raw export it is not: whether a covariance entry's shortest repr has 16 or 17
  digits depends on its last bits (VENDORED.md).
- ``packets.repr``: doubles and Python's ``repr`` of each, the float rule the JSON uses.
- G10 (``packets.assimilate``): a receiver, the last snapshot's learner as agent '1',
  assimilates 37 packets in order. They are the wire packets, an earlier export from the west
  end, copies from 9 other senders (over the received cap) and malformed variants. Then come its
  field fit with the received messages, and its update.

As in G4–G8, every geometric decision that goes through a BLAS dot product (``numpy.linalg.norm``)
is kept more than ``coph_golden.MARGIN`` from its threshold, and the receiver's update is checked
against copies with the committed values nudged by 1e-12 each way.
"""

from __future__ import annotations

import contextlib
import copy
import json
from dataclasses import asdict, replace

import numpy as np
import torch

from grl_snam.coph.cooperative import bootstrap_field, field_vector
from grl_snam.coph.learner import Learner
from grl_snam.coph.maze_learning import MazePacket
from grl_snam.coph.model import tensor
from grl_snam.coph.shapes import is_slab, slab_distance
from grl_snam.tools import coph_golden_learner as gl

RECIPIENT = "1"
RECEIVER = 2  # G10's receiver: learner.snapshots[2] (tick 112), as agent RECIPIENT
OVER_CAP = tuple(str(k) for k in range(2, 11))  # 9 more senders: 10 against a cap of 8

# Doubles whose repr exercises the rule: fixed between 1e-4 and 1e16, scientific outside it with a
# signed exponent of at least two digits, the shortest round-trip digits, '.0' on whole numbers.
REPR_VALUES = [
    0.0, -0.0, 1.0, -1.0, 0.1, 0.5, 2.5, 100.0, 1e-4, 0.00012345, 9.999999999999999e-05, 1e-05,
    1.5e-05, -2.5e-06, 1e-07, 1e-10, 1e15, 9999999999999998.0, 1e16, 1.2345678901234567e16,
    123456789012345.67, 2.0**53, 2.0**53 + 2, 1e22, 1e100, 1.5e300, 5e-324, 2.2250738585072014e-308,
    1.7976931348623157e308, 0.30000000000000004, 1 / 3, 2 / 3, 2.0**-24, 3 * 2.0**-24, 13.44, 7.68,
    11.52, 6.299999999999999, 4.799999999999999, 0.0025, 0.002499999999999995, 123.456, 1e-300,
]  # fmt: skip


# ---------------------------------------------------------------------------
# Draws: recorded from the learner's own generators, then injected
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def recorded_draws(out: list):
    """Record each ``numpy.random.default_rng`` made inside: ``out`` gets one
    ``{"seed", "draws"}`` per generator, ``draws`` the lists its ``integers`` returned."""
    original = np.random.default_rng

    def make(seed):
        g = original(seed)
        record = dict(seed=int(seed), draws=[])
        out.append(record)

        class Recorder:
            def integers(self, low, high, size):
                drawn = g.integers(low, high, size)
                record["draws"].append([int(i) for i in drawn])
                return drawn

        return Recorder()

    np.random.default_rng = make
    try:
        yield
    finally:
        np.random.default_rng = original


@contextlib.contextmanager
def injected_draws(generators: list):
    """Each ``numpy.random.default_rng`` made inside takes the next list of ``generators``; its
    ``integers`` returns that list's draws in order, whatever the seed."""
    original = np.random.default_rng
    queue = [list(g) for g in generators]

    def make(_seed):
        draws = queue.pop(0)

        class Injected:
            def integers(self, low, high, size):
                d = draws.pop(0)
                assert len(d) == size and all(low <= i < high for i in d), (d, low, high, size)
                return np.asarray(d, dtype=np.int64)

        return Injected()

    np.random.default_rng = make
    try:
        yield
    finally:
        np.random.default_rng = original


# ---------------------------------------------------------------------------
# Packets in the file
# ---------------------------------------------------------------------------


def canonical(p) -> str:
    """A packet as text that tells every field and every double's bits apart (NaN included)."""
    return json.dumps(asdict(p), sort_keys=True, allow_nan=True)


def wire(p):
    """The export as G9 records it: anchors' values and gradients, covariance, H and V_task on
    the 2^-24 grid. Everything else (positions, times, provenance, geometry) is unchanged."""
    g = gl._grid
    return replace(
        p,
        anchors=tuple((float(a[0]), float(a[1]), g(a[2]), g(a[3]), g(a[4])) for a in p.anchors),
        covariance=tuple(tuple(g(c) for c in row) for row in p.covariance),
        H=g(p.H),
        V_task=g(p.V_task),
    )


def _script_points(sc) -> dict:
    """q -> the index (21 tick + j) of the first script reading at q."""
    index = {}
    for tick, t in enumerate(sc):
        for j, r in enumerate(t["readings"]):
            index.setdefault((float(r[0]), float(r[1])), len(gl.LATTICE) * tick + j)
    return index


def _geometry_entry(g) -> list:
    from grl_snam.tools.coph_golden import _hexes, hexbits

    return [gl.OBSTACLES.index(tuple(g[:-5])), _hexes(g[-5:-3]), hexbits(g[-3]), hexbits(g[-2]),
            g[-1]]  # fmt: skip


def encode_packet(p, sc, points=None) -> dict:
    """A packet in the file. Doubles are bits; sources are the sender's reading indices plus any
    other uids; support points are script reading indices; geometry shapes are obstacle indices.
    """
    from grl_snam.tools.coph_golden import _hexes, _runs, hexbits

    points = _script_points(sc) if points is None else points
    n = len(gl.LATTICE)
    readings, extra = [], []
    for uid in p.sources:
        parts = uid.split(":")
        if not extra and len(parts) == 3 and parts[0] == p.sender:
            readings.append(n * int(parts[1]) + int(parts[2]))
        else:
            extra.append(uid)
    out = dict(
        schema=p.schema, sender=p.sender, recipient=p.recipient, identity=p.identity,
        timestamp=hexbits(p.timestamp), expires=hexbits(p.expires), version=p.version,
        convention=p.convention, q=_hexes(p.q), p=_hexes(p.p), v=_hexes(p.v), H=hexbits(p.H),
        V_task=hexbits(p.V_task), goal=_hexes(p.goal), bounds=_hexes(p.bounds),
        anchors=[_hexes(a) for a in p.anchors], covariance=[_hexes(r) for r in p.covariance],
        sources=dict(agent=p.sender, readings=_runs(sorted(readings)), extra=extra),
        stage=p.stage,
        support_points=_runs([points[(float(q[0]), float(q[1]))] for q in p.support_points]),
        uncertainty=p.uncertainty, acknowledged_sources=list(p.acknowledged_sources),
        geometry_schema=p.geometry_schema, geometry=[_geometry_entry(g) for g in p.geometry],
    )  # fmt: skip
    xy = [(float(r[0]), float(r[1])) for t in sc for r in t["readings"]]
    assert canonical(decode_packet(out, xy)) == canonical(p), "the packet does not round-trip"
    return out


def decode_packet(d: dict, xy: list) -> MazePacket:
    """``encode_packet``'s inverse. ``xy[i]`` is script reading i's position."""
    b = gl._bits
    n = len(gl.LATTICE)
    src = d["sources"]
    sources = sorted(f"{src['agent']}:{i // n}:{i % n}" for i in gl._expand(src["readings"]))
    return MazePacket(
        schema=d["schema"], sender=d["sender"], recipient=d["recipient"], identity=d["identity"],
        timestamp=b(d["timestamp"]), expires=b(d["expires"]), version=d["version"],
        convention=d["convention"], q=tuple(b(d["q"])), p=tuple(b(d["p"])), v=tuple(b(d["v"])),
        H=b(d["H"]), V_task=b(d["V_task"]), goal=tuple(b(d["goal"])), bounds=tuple(b(d["bounds"])),
        anchors=tuple(tuple(b(a)) for a in d["anchors"]),
        covariance=tuple(tuple(b(r)) for r in d["covariance"]),
        sources=tuple(sources) + tuple(src["extra"]), stage=d["stage"],
        support_points=tuple(xy[i] for i in gl._expand(d["support_points"])),
        uncertainty=d["uncertainty"], acknowledged_sources=tuple(d["acknowledged_sources"]),
        geometry_schema=d["geometry_schema"],
        geometry=tuple((*gl.OBSTACLES[k], *b(s), b(r), b(t), source)
                       for k, s, r, t, source in d["geometry"]),
    )  # fmt: skip


def encode_reports(reports: dict) -> list:
    """A learner's local geometry reports (MazeLearner.local_geometry_reports), in order."""
    return [_geometry_entry(v) for v in reports.values()]


def decode_reports(entries: list) -> dict:
    b = gl._bits
    out = {}
    for k, s, r, t, source in entries:
        shape = gl.OBSTACLES[k]
        out[tuple(shape)] = (*shape, *b(s), b(r), b(t), source)
    return out


# ---------------------------------------------------------------------------
# G9: exports
# ---------------------------------------------------------------------------


def _fitting(stage) -> list:
    return [r for r in stage.readings if int(r.uid.split(":")[-1]) % 5]


def _export_margins(agent, x) -> None:
    """Refuse an export whose anchor choices sit within ``MARGIN`` of a norm threshold: the
    candidates' support (0.65 from a fitting reading) and the risk-ordered anchors' support and
    spacing (more than 0.3 apart), up to where ``export`` stops looking."""
    from grl_snam.tools.coph_golden import _margin

    s, cfg = agent.stages[-1], agent.cfg
    fitting = [np.asarray(r.q) for r in _fitting(s)]

    def support(q):
        if s.contains(q, cfg):
            for f in fitting:
                _margin("anchor support", float(np.linalg.norm(np.asarray(q) - f)) - 0.65)

    for q in agent.anchor_candidates(x, s):
        support(q)
    points = []
    for r in sorted(_fitting(s), key=lambda r: (-r.value, r.uid)):
        support(r.q)
        for p in points:
            _margin("anchor spacing", float(np.linalg.norm(np.asarray(r.q) - p)) - 0.3)
        if s.supported(r.q, cfg) and all(np.linalg.norm(np.asarray(r.q) - p) > 0.3 for p in points):
            points.append(r.q)
        if len(points) == agent.communication.anchors:
            break


def _anchor_set(stage, cfg, mean, ensemble, points, scale, floor):
    """Anchors (q, value, gradient) and the ensemble's covariance at ``points``, as export does."""
    vector = field_vector(mean, stage.centers, points, cfg.rbf_width)
    vectors = np.array([field_vector(w, stage.centers, points, cfg.rbf_width) for w in ensemble])
    cov = scale * (np.cov(vectors, rowvar=False) + np.eye(len(vector)) * floor)
    return vector.reshape(-1, 3), cov


def export_case(agent, x, sc, check=True) -> tuple[dict, MazePacket, MazePacket]:
    """G9 at one snapshot: the case, the raw export and its wire form."""
    from grl_snam.tools.coph_golden import _hexes, hexbits

    if check:
        _export_margins(agent, x)
    generators: list = []
    with recorded_draws(generators):
        packet = copy.deepcopy(agent).export(x, RECIPIENT)
    assert packet is not None and len(generators) == 2, "both bootstraps run"
    draws = [g["draws"] for g in generators]
    with injected_draws(draws):
        again = copy.deepcopy(agent).export(x, RECIPIENT)
    assert canonical(again) == canonical(packet), "injected draws reproduce the export"

    s, cfg, comm = agent.stages[-1], agent.cfg, agent.communication
    with injected_draws(draws):
        mean, ensemble, rs = bootstrap_field(s, cfg, comm, agent.now, 0, agent.dynamic)
        _, ensemble2, _ = bootstrap_field(s, cfg, comm, agent.now, 0, agent.dynamic)
    blocks: dict = {}
    for r in rs:
        k = int(r.time / (cfg.dt * cfg.update_every))
        blocks[k] = blocks.get(k, 0) + 1
    # CooperativeLearner.export: the candidates around x that the stage supports.
    candidates = [np.asarray(q) for q in agent.anchor_candidates(x, s)]
    chosen = [q for q in candidates if s.supported(q, cfg)][: comm.anchors]
    z1, cov1 = _anchor_set(s, cfg, mean, ensemble, np.array(chosen), comm.covariance_scale,
                           comm.covariance_floor)  # fmt: skip
    # TerrainLearner.export: the supported readings with the most risk, 0.3 apart.
    picked = []
    for r in sorted(rs, key=lambda r: (-r.value, r.uid)):
        if s.supported(r.q, cfg) and all(
            np.linalg.norm(np.asarray(r.q) - np.asarray(p.q)) > 0.3 for p in picked
        ):
            picked.append(r)
        if len(picked) == comm.anchors:
            break
    path = "terrain" if picked else "cooperative"
    z2, cov2 = (None, None)
    if picked:
        z2, cov2 = _anchor_set(s, cfg, mean, ensemble2, [r.q for r in picked], 1.0,
                               comm.covariance_floor)  # fmt: skip
        assert [tuple(map(float, a[2:])) for a in packet.anchors] == [
            tuple(map(float, z)) for z in z2
        ]
        assert np.array_equal(np.array(packet.covariance), cov2)
    else:
        assert np.array_equal(np.array(packet.covariance), cov1)
    w = wire(packet)
    points = _script_points(sc)
    case = dict(
        recipient=RECIPIENT,
        geometry=encode_reports(agent.local_geometry_reports),
        seeds=[g["seed"] for g in generators],
        draws=draws,
        blocks=[[k, c] for k, c in blocks.items()],
        candidates=[[_hexes(q), bool(s.supported(q, cfg))] for q in candidates],
        picked=[points[(float(r.q[0]), float(r.q[1]))] for r in picked],
        path=path,
        packet=encode_packet(w, sc, points),
        bytes=w.bytes(),
        out=dict(
            mean=_hexes(mean.tolist()),
            cooperative=dict(ensemble=[_hexes(e.tolist()) for e in ensemble],
                             anchors=[_hexes(z) for z in z1],
                             covariance=[_hexes(r) for r in cov1]),
            terrain=None if not picked else dict(
                ensemble=[_hexes(e.tolist()) for e in ensemble2],
                anchors=[_hexes(z) for z in z2], covariance=[_hexes(r) for r in cov2]),
            H=hexbits(packet.H), V_task=hexbits(packet.V_task),
        ),
    )  # fmt: skip
    return case, packet, w


# ---------------------------------------------------------------------------
# G10: assimilation
# ---------------------------------------------------------------------------


def relabel_learner(agent, new: str):
    """A copy of ``agent`` as agent ``new``: its readings' uids and its geometry sources carry
    the new id."""
    a = copy.deepcopy(agent)
    old = a.agent_id
    a.agent_id = new
    for s in a.stages:
        s.readings = [replace(r, uid=new + r.uid[len(old) :]) for r in s.readings]
    a.local_geometry_reports = {
        k: (*v[:-1], new + v[-1][len(old) :]) for k, v in a.local_geometry_reports.items()
    }
    return a


def relabel(p, new: str):
    """A copy of packet ``p`` from sender ``new``: identity, sources and geometry sources too."""
    old = p.sender
    return replace(
        p, sender=new, identity=new + p.identity[len(old) :],
        sources=tuple(new + s[len(old) :] for s in p.sources),
        geometry=tuple((*g[:-1], new + g[-1][len(old) :]) for g in p.geometry),
    )  # fmt: skip


def _with_anchor(p, index, **changes):
    names = ("x", "y", "value", "gx", "gy")
    anchors = [list(a) for a in p.anchors]
    for k, v in changes.items():
        anchors[index][names.index(k)] = v
    return replace(p, anchors=tuple(map(tuple, anchors)))


def g10_packets(early, p1, p2, p3) -> list:
    """G10's packets, in the order the receiver assimilates them, with a name for each."""
    cov = [list(r) for r in p3.covariance]
    asymmetric = copy.deepcopy(cov)
    asymmetric[0][1] += 0.01
    n = len(cov)
    five = replace(
        p3, anchors=p3.anchors + p3.anchors[:1],
        covariance=tuple(tuple(cov[i % n][j % n] + (0.0025 if i == j and i >= n else 0.0)
                               for j in range(n + 3)) for i in range(n + 3)),
    )  # fmt: skip
    cases = [
        ("early_route_only", early),  # its anchors are behind the receiver: route evidence only
        ("first", p1),
        ("newer", p2),
        ("older_again", p1),
        ("expired", replace(p3, expires=13.0)),
        ("from_the_future", replace(p3, timestamp=13.5)),
        ("nan_timestamp", replace(p3, timestamp=float("nan"))),
        ("schema", replace(p3, schema=2)),
        ("convention", replace(p3, convention="unit_mass; SI")),
        ("returned_sender", relabel(p3, RECIPIENT)),
        ("wrong_recipient", replace(p3, recipient="2")),
        ("foreign_source", replace(p3, sources=p3.sources + ("2:5:1",))),
        ("own_source", replace(p3, sources=p3.sources + (f"{RECIPIENT}:112:3",))),
        ("asymmetric", replace(p3, covariance=tuple(map(tuple, asymmetric)))),
        ("indefinite", replace(p3, covariance=tuple(tuple(-float(i == j) for j in range(n))
                                                    for i in range(n)))),
        ("covariance_shape", replace(p3, covariance=tuple(tuple(r[:9]) for r in cov[:9]))),
        ("five_anchors", five),
        ("negative_value", _with_anchor(p3, 0, value=-0.1)),
        ("nan_gradient", _with_anchor(p3, 1, gx=float("nan"))),
        ("p_not_v", replace(p3, v=(p3.v[0] + 0.1, p3.v[1]))),
        ("out_of_bounds", _with_anchor(p3, 0, x=7.9)),
        ("unsupported_anchor", _with_anchor(p3, 0, x=4.85, y=2.05)),
        ("geometry_schema", replace(p3, geometry_schema=2)),
        ("geometry_source", replace(p3, geometry=tuple((*g[:-1], "2:geometry:0")
                                                       for g in p3.geometry))),
        ("geometry_late", replace(p3, geometry=tuple((*g[:-2], 99.0, g[-1]) for g in p3.geometry))),
        ("geometry_far", replace(p3, geometry=tuple((*g[:-5], g[-5] - 3.0, *g[-4:])
                                                    for g in p3.geometry))),
        ("geometry_overflow", replace(p3, geometry=p3.geometry * 33)),
        ("latest", p3),
    ]  # fmt: skip
    for k, sender in enumerate(OVER_CAP):
        cases.append((f"sender_{sender}", relabel((p1, p2, p3)[k % 3], sender)))
    return cases


def _assimilate_margins(p) -> None:
    """Refuse a packet whose route-evidence support (an anchor within 0.65 of a support point)
    or geometry check (the sensor within its radius of the shape) sits within ``MARGIN``."""
    from grl_snam.tools.coph_golden import _margin

    support = np.asarray(p.support_points, dtype=float)
    for a in p.anchors:
        if len(support) and np.isfinite(a[:2]).all():
            d = np.min(np.linalg.norm(support - np.asarray(a[:2], dtype=float), axis=1))
            _margin("route support", float(d) - 0.65)
    for o in p.geometry:
        shape, sensor = o[:-5], np.asarray(o[-5:-3], dtype=float)
        if is_slab(shape):
            gap = float(slab_distance(sensor, shape))
        else:
            gap = float(np.linalg.norm(np.asarray(o[:2]) - sensor)) - o[2]
        _margin("geometry gap", gap - (o[-3] + 1e-9))


def message_fit_case(agent) -> dict:
    """The receiver's field fit with its received messages (CooperativeLearner._field_fit), stage
    by stage: the messages and anchors it uses, the whitened Gauss–Newton system and its step."""
    from grl_snam.tools.coph_golden import _hexes

    cfg = agent.cfg
    base, cov = Learner._field_fit(copy.deepcopy(agent))
    adjusted, _ = copy.deepcopy(agent)._field_fit()
    stages = []
    for k, (s, w) in enumerate(zip(agent.stages, base)):
        rows, targets, used = [], [], []
        for m in agent.received.values():
            ids = [i for i, a in enumerate(m.anchors) if s.contains(a[:2], cfg)]
            if not ids:
                continue
            used.append([m.sender, ids])
            points = tensor([m.anchors[i][:2] for i in ids])
            z = tensor([m.anchors[i][2:] for i in ids])

            # As CooperativeLearner._field_fit: value and gradient of the risk at the anchors.
            def vector(v, points=points):
                f = gl.features(points, s.centers, cfg.rbf_width)
                raw = f @ v
                sigmoid = torch.sigmoid(raw)
                df = -(points[:, None, :] - s.centers) / cfg.rbf_width**2 * f[:, :, None]
                val = torch.nn.functional.softplus(raw) / 4
                grad = sigmoid[:, None] / 4 * torch.einsum("nmd,m->nd", df, v)
                return torch.cat((val[:, None], grad), 1).reshape(-1)

            J = torch.autograd.functional.jacobian(vector, w)
            idx = [3 * i + j for i in ids for j in range(3)]
            C = tensor(np.asarray(m.covariance)[np.ix_(idx, idx)])
            C = C + torch.eye(len(idx), dtype=torch.float64) * (
                0.01 + 0.005 * (agent.now - m.timestamp)
            )
            L = torch.linalg.cholesky(C)
            rows.append(torch.linalg.solve_triangular(L, J, upper=False))
            residual = (z.reshape(-1) - vector(w))[:, None]
            targets.append(torch.linalg.solve_triangular(L, residual, upper=False).flatten())
        if rows:
            A, y = torch.cat(rows), torch.cat(targets)
            local = (
                torch.linalg.inv(cov[k])
                if cov[k] is not None
                else torch.eye(len(w), dtype=torch.float64)
            )
            gram, rhs = A.T @ A, A.T @ y
            delta = torch.linalg.solve(local + gram, rhs)
            after = w + delta.clamp(-0.5, 0.5)
            out = dict(gram=_hexes(gram.reshape(-1).tolist()), rhs=_hexes(rhs.tolist()),
                       delta=_hexes(delta.tolist()), weights=_hexes(after.tolist()))  # fmt: skip
        else:
            after = w
            out = dict(gram=None, rhs=None, delta=None, weights=_hexes(w.tolist()))
        assert torch.equal(
            after, adjusted[k]
        ), "message_fit_case is not CooperativeLearner._field_fit"
        stages.append(dict(messages=used, rows=sum(3 * len(ids) for _, ids in used), out=out))
    return dict(received=list(agent.received), stages=stages)


def receiver_state(agent) -> dict:
    """What assimilation leaves behind that the field fit, the router and later packets read."""
    from grl_snam.tools.coph_golden import _hexes, hexbits

    nav = agent.navigation
    return dict(
        received=list(agent.received),
        last_sequences=[[k, hexbits(t), st] for k, (t, st) in agent.last_sequences.items()],
        remote_terrain=[[k[0], k[1]] for k in agent.remote_terrain],
        remote_cells=[[k[0], int(k[1]), int(k[2]), hexbits(v[0]), _hexes(v[1]), v[2]]
                      for k, v in agent.remote_cells.items()],
        remote_geometry=[[gl.OBSTACLES.index(tuple(k)), hexbits(v["expires"]), v["packet"],
                          v["source"], hexbits(v["observation_time"]), v["sender"]]
                         for k, v in agent.remote_geometry_reports.items()],
        reported_geometry=[gl.OBSTACLES.index(tuple(o)) for o in nav["reported_geometry"]],
        geometry=[gl.OBSTACLES.index(o) for o in agent.geometry_store],
        terrain_anchors=len(nav["terrain_anchors"]),
    )  # fmt: skip


def receiver_cases(receiver, named, sc, check=True) -> dict:
    """``receiver`` assimilates the ``(name, packet)`` list in order; then its field fit (after
    'latest' and after the last packet) and its update at its snapshot's state."""
    from grl_snam.tools.coph_golden import _hexes

    points = _script_points(sc)
    cases, fits = [], []
    for name, p in named:
        if check:
            _assimilate_margins(p)
        event = receiver.assimilate(p)
        cases.append(dict(
            name=name, packet=encode_packet(p, sc, points), reason=event["reason"],
            scope=event.get("scope"), received=list(receiver.received),
        ))  # fmt: skip
        if name == "latest":
            fits.append(dict(after=name, **message_fit_case(receiver)))
    fits.append(dict(after=cases[-1]["name"], **message_fit_case(receiver)))
    state = receiver_state(receiver)
    before = copy.deepcopy(receiver)
    x = sc[gl.SNAPSHOTS[RECEIVER][1]]["state"]
    report, _ = gl.traced_update(receiver, x)
    mine = gl.decisions(report)
    if check:
        for sign in (1.0, -1.0):
            other, _ = gl.traced_update(gl._nudged(before, sign), x)
            if gl.decisions(other) != mine:
                raise AssertionError(f"G10: a {gl.NUDGE} nudge changes the receiver's update")
    update = gl.update_case(before, report)
    update.update(
        messages=report["messages"],
        eligibility=[list(e) for e in mine["eligibility"]],
        plan=[[p["stage"], p["eligible"], p["horizon"], p["reason"], p["pilot_steps"]]
              for p in mine["plan"]],
        steps=mine["steps"],
    )  # fmt: skip
    update["out"]["weights"] = [_hexes(s.weights.tolist()) for s in receiver.stages]
    return dict(cases=cases, state=state, field_fit=fits, update=update)


def assimilate_case(found, sc, wires, early, check=True) -> dict:
    """G10: the receiver and ``g10_packets``, through ``receiver_cases``."""
    tick = gl.SNAPSHOTS[RECEIVER][1]
    receiver = relabel_learner(found[tick][0], RECIPIENT)
    inputs = dict(snapshot=RECEIVER, agent_id=RECIPIENT,
                  geometry=encode_reports(receiver.local_geometry_reports))  # fmt: skip
    return dict(receiver=inputs, **receiver_cases(receiver, g10_packets(early, *wires), sc, check))


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------


def packet_cases(check=True, sim=None) -> dict:
    """G9, the repr cases and G10. ``sim`` is a ``coph_golden_learner.simulate(check)`` result."""
    from grl_snam.tools.coph_golden import hexbits

    cfg, sc, found = sim if sim is not None else gl.simulate(check)
    exports, raws, wires = [], [], []
    for k, (_, tick) in enumerate(gl.SNAPSHOTS):
        agent = found[tick][0]
        case, raw, w = export_case(agent, np.array(sc[tick]["state"]), sc, check)
        exports.append(dict(snapshot=k, **case))
        raws.append(raw)
        wires.append(w)
    early_tick = gl.EXTRA_TICKS[0]
    _, _, early = export_case(found[early_tick][0], np.array(sc[early_tick]["state"]), sc, check)
    first = wires[0]
    return dict(
        tier=gl.TIER,
        recipient=RECIPIENT,
        repr=[[hexbits(v), repr(v)] for v in REPR_VALUES],
        json=json.dumps(asdict(first), sort_keys=True, separators=(",", ":"), allow_nan=False),
        exports=exports,
        assimilate=assimilate_case(found, sc, wires, early, check),
    )


PACKET_RULES = """\
G9 and G10 pin the demos' packets: CrossingLearner.export and .assimilate (MazeLearner on
TerrainLearner on CooperativeLearner), at the snapshots of G5-G8 (learner.snapshots; rebuilt as
learner_rules says). The communication config is learner.world.communication (the profile's,
with ttl 300 and byte_budget 4e6). Recipient and receiver are agent '1'.

A packet (CooperativeLearner's Packet, then MazePacket's two fields) has, in this order: schema,
sender, recipient, identity, timestamp, expires, version, convention, q, p, v, H, V_task, goal,
bounds, anchors, covariance, sources, stage, support_points, uncertainty, acknowledged_sources,
geometry_schema, geometry. In the file: doubles are bits; sources = sorted strings
'<agent>:<tick>:<j>' for the reading indices i (tick, j = divmod(i, 21)), then extra in order;
support_points = the positions of those script readings, in order; a geometry entry is [obstacle
index, sensor (x, y), sensor radius, time, source] for the tuple (*shape, x, y, radius, time,
source), a slab's shape being (x, y, r, hx, hy).

bytes() = the length of json.dumps(asdict(packet), sort_keys=True, separators=(',', ':'),
allow_nan=False): keys sorted by code point (H and V_task first), tuples as arrays, strings with
JSON's escapes, ints as ints, and every double as Python's repr. repr writes the shortest digits
that read back as the double (the closest such string when two have that length), fixed when the
decimal exponent e of its first digit is -4 <= e < 16 (with '.0' when it is whole), else
d.ddde+XX / d.ddde-XX with at least two exponent digits ('1e-05', '1e+16'); -0.0 keeps its sign.
repr[] lists doubles and their repr; json is export 0's wire packet in full.

G9, exports[] (snapshot k's learner exports to '1' at x = learner.script[tick].state):
- CooperativeLearner.export: the readings are the newest stage's with j % 5 != 0, in order; they
  are grouped into blocks by int(time / (dt update_every)), in order of first appearance (blocks:
  [key, count]). fit(rs) solves (A^T A + fit_ridge I) w = A^T y + fit_ridge w0, A the RBF
  features (G5's) of rs, y = log(expm1(4 clip(value, 0.001, 20))), w0 = -1/16 each. mean =
  fit(all). The ensemble: for each of ensemble_size members, draws[0][member] are block indices
  (with replacement, as many as there are blocks); the member is fit(the readings of those blocks,
  concatenated in draw order). In the demos the indices come from numpy's default_rng seeded with
  seeds[0] (the first 8 hex digits of sha256('<agent>:<repr(now)>:<stage id>')); here they are
  injected, and the C++ draws its own from libcvc's RNG in production.
  candidates: x + (0, 0), (0.4, 0), (0, 0.4), (0, -0.4), each with whether the stage supports it
  (inside its box, and within 0.65 of a fitting reading); the first `anchors` supported are the
  points. field_vector(w, points) = for each point, (r(q), dr/dq) with r G3's risk
  (softplus(phi(q) . w) / 4). anchors = (q, field_vector(mean, q)); covariance = covariance_scale
  (cov(ensemble vectors) + covariance_floor I), cov numpy's (rows are members, n - 1 = 7).
- TerrainLearner.export then replaces anchors and covariance when it can (path 'terrain'): a second
  bootstrap with draws[1] (seed 12345 + int(1000 now)); the readings in order of (-value, uid),
  each kept when the stage supports it and it is more than 0.3 from every point kept so far, until
  `anchors` (picked: their script reading indices); anchors at them from the same mean, and
  covariance = cov(the second ensemble's vectors) + covariance_floor I (no scale). With none
  picked, CooperativeLearner's stay (path 'cooperative').
- The rest: identity '<agent>:<now, 9 decimals>:<stage id>', timestamp now, expires now + ttl,
  version, the convention string, q = x[:2], p = v = x[2:], H = |v|^2 / 2 + G3's potential at q
  (target: the navigation target, else the goal; mean's weights; the policy; the geometry store's
  obstacles), V_task = |q - target|^2 / 2, goal, bounds = the newest stage's box (x0, x1, y0, y1),
  sources = the sorted uids of the readings, stage = its id, support_points = the readings'
  positions, first occurrences in order, uncertainty 'block_bootstrap; calibration_scale=' +
  repr(covariance_scale), no acknowledged sources; geometry_schema 1 and geometry = the learner's
  local geometry reports (geometry: its inputs, in insertion order).
- packet is the wire packet: the export with each anchor's value and gradient, the covariance, H
  and V_task rounded to the nearest multiple of 2^-24. bytes = its bytes() (BIT).
  out (FLOAT64, tier): mean; cooperative {ensemble weights, anchors' (value, gx, gy), covariance};
  terrain {the same} or null; H and V_task (unrounded).

G10, assimilate (the receiver is learner.snapshots[2]'s learner as agent '1': its readings' uids
start '1:', its local geometry reports are receiver.geometry, nothing is received). cases[] are
assimilated in order (MazeLearner.assimilate, then TerrainLearner's, then CooperativeLearner's);
reason, scope and the received senders after each are BIT.
- MazeLearner: geometry_schema must be 1 (else incompatible_geometry). The geometry is valid when
  it has at most 32 entries and each has 8 or 10 values, all finite but the source, radius > 0,
  sensor radius > 0, 0 <= time <= timestamp + 1e-9, source starting '<sender>:geometry:', and the
  sensor within radius + 1e-9 of the shape (slab: slab_distance; disk: |c - sensor| - r); else
  invalid_geometry.
- CooperativeLearner, the first that fails: schema 1 and the convention (incompatible_convention);
  the sender is not the receiver and no source starts '<receiver>:' (returned_evidence); recipient
  (wrong_recipient); sources non-empty, each starting '<sender>:' (invalid_provenance); timestamp
  and expires finite, timestamp <= now + 1e-9, expires >= now, expires >= timestamp (invalid_age);
  (timestamp, stage) > the sender's last accepted pair (duplicate_or_reordered); then
  invalid_payload unless every value is finite, 1 <= #anchors <= `anchors`, covariance is
  3#anchors square, numpy-allclose to its transpose and positive definite, q, p and v have 2
  values, p allclose to v, every anchor value >= 0 and #support_points <= sample_cap; every anchor
  inside bounds (invalid_support); some anchor inside a kept stage's box (unsupported). Eligible:
  received[sender] = packet (an existing sender keeps its place), last[sender] = (timestamp,
  stage), and while more than received_cap are held the oldest timestamp goes (the first such).
- TerrainLearner, for eligible or unsupported: route support is valid when there are support
  points and every anchor is within 0.65 of one; the key is (sender, stage). Invalid support, or a
  held packet at the key with timestamp >= this one, undoes CooperativeLearner's changes (reason
  invalid_remote_support or duplicate_or_reordered). Else remote_terrain[key] = packet, and for each
  anchor the cell (sender, round_half_even(x / 0.2), round_half_even(y / 0.2)) = (expires, anchor,
  identity, sources), the oldest-expiring cells dropped past 128 and the oldest packets past 8;
  last[sender] = (timestamp, stage); reason eligible with scope retained_and_remote_route (or
  remote_route_only when CooperativeLearner said unsupported). The navigation context's
  terrain_anchors become the unexpired cells' anchors (in order) then the learner's own.
- MazeLearner, when eligible: each entry's shape gets a remote report (expires, packet identity,
  source, observation time, sender); reported_geometry = the sorted union of local and remote
  shapes; the shapes that intersect a kept stage join the geometry store (sorted, capped).
- state: what is held afterwards (remote_cells: [sender, ix, iy, expires, anchor, identity];
  terrain_anchors: how many).
- field_fit[], after 'latest' and after the last case (FLOAT64): CooperativeLearner._field_fit. G5's
  fit gives w_k and covariance_k per stage; for each received packet (in order) the anchors inside
  stage k's box (messages: [sender, anchor indices]) give the residual z - f(w_k), f(w) = (value,
  gx, gy) per anchor as field_vector, z the packet's; J = df/dw (analytic: value' = sigmoid(raw) /
  4 phi; gradient' = sigmoid(raw) (1 - sigmoid(raw)) / 4 phi (sum_m dphi_m w_m) + sigmoid(raw) / 4
  dphi, dphi_m = -(q - c_m) / rbf_width^2 phi_m); C = the packet's covariance on those anchors +
  (0.01 + 0.005 (now - timestamp)) I; L = chol(C); rows L^-1 J, targets L^-1 residual, stacked over
  packets. gram = A^T A, rhs = A^T y, delta = (covariance_k^-1 + gram)^-1 rhs, weights = w_k +
  clamp(delta, -0.5, 0.5); a stage with no anchors keeps w_k. rows: the residual's length.
  out: gram, rhs, delta, weights.
- update: the receiver's update at learner.script[112].state, as G8 (decisions BIT) with the
  messages it holds (messages: their identities), each stage's eligibility and plan [stage,
  eligible, horizon, reason, pilot_steps] and the counterfactual steps; out adds each kept stage's
  weights after the update.
"""
