"""A run file: a ``grl-snam ariadne coph`` run packed so that it plays back without torch.

The viewer replays a run from its episode files, but reading those builds each arm's world (the
corridors and the true risk field) from the vendored demos, which import torch, and the files
are large (an arm's is 15-46 MB gzipped, nearly all of it the learner's update records). A web
page (libcvc's ariglx-web) has no torch. A run file holds what the viewer draws, made where
torch is:

- per arm, its spec, its label and its messages (:func:`grl_snam.coph.feed.messages`, about
  0.1-0.4 MB gzipped), from which :class:`~grl_snam.coph.timeline.Timeline` rebuilds everything
  it answers;
- per arm, its floor's corridors and true risk field sampled on the floor's grid, and its
  nameplate's pixels (the viewer draws those with Pillow);
- per arm and agent, :func:`~grl_snam.coph.timeline.learning`'s series (the Learning window's
  charts), which a page would otherwise spend seconds on at start;
- ``records.Config()``'s values, for :data:`~grl_snam.coph.timeline.CFG`.

It is one ``.npz`` (``np.savez_compressed``): numbers as arrays, the rest as UTF-8 JSON in uint8
arrays, so that loading it needs no pickle. ``grl-snam ariadne coph DEMO --bake FILE`` writes one
from a run's episodes; ``--from FILE`` plays it back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .timeline import LEARNING, Floor, Timeline

SCHEMA = "grl_snam.coph_run/v1"
#: A run file's name ends so.
SUFFIX = ".coph.npz"


def _json(value) -> np.ndarray:
    return np.frombuffer(json.dumps(value, separators=(",", ":")).encode(), np.uint8)


def _unjson(array) -> object:
    return json.loads(np.asarray(array, np.uint8).tobytes().decode())


@dataclass
class Arm:
    """One arm of a run file."""

    arm: str
    label: str
    spec: dict
    messages: list
    truth: np.ndarray  # (h, w) at px_per_m, row 0 the south edge
    corridor: np.ndarray  # (h, w) bool
    plate: np.ndarray | None = None  # uint8 RGB, row 0 its north edge, or None
    learning: list = field(default_factory=list)  # per agent: {name: array} (timeline.LEARNING)

    def timeline(self, demo) -> Timeline:
        return Timeline(demo, self.spec, self.arm, label=self.label).ingest(self.messages)


@dataclass
class Run:
    """A run file's contents (:func:`read`)."""

    demo: str
    seed: int
    scene: str | None
    steps: int
    px_per_m: int
    config: dict
    arms: list
    meta: dict = field(default_factory=dict)

    def timelines(self) -> list:
        return [a.timeline(self.demo) for a in self.arms]

    def floors(self) -> list:
        return [
            Floor(a.spec, px_per_m=self.px_per_m, corridor=a.corridor, truth=a.truth)
            for a in self.arms
        ]


def write(path, run: Run) -> Path:
    """Write ``run`` to ``path`` (its name should end in :data:`SUFFIX`); the path."""
    path = Path(path)
    meta = dict(
        run.meta,
        schema=SCHEMA,
        demo=run.demo,
        seed=int(run.seed),
        scene=run.scene,
        steps=int(run.steps),
        px_per_m=int(run.px_per_m),
        config=run.config,
        learning=list(LEARNING),
        arms=[dict(arm=a.arm, label=a.label, plate=a.plate is not None) for a in run.arms],
    )
    arrays = {"meta": _json(meta)}
    for k, a in enumerate(run.arms):
        arrays[f"spec{k}"] = _json(a.spec)
        arrays[f"messages{k}"] = _json(a.messages)
        arrays[f"truth{k}"] = np.asarray(a.truth, np.float32)
        arrays[f"corridor{k}"] = np.asarray(a.corridor, bool)
        if a.plate is not None:
            arrays[f"plate{k}"] = np.ascontiguousarray(a.plate, np.uint8)
        for i, series in enumerate(a.learning):
            arrays[f"learning{k}_{i}"] = np.stack([np.asarray(series[n], float) for n in LEARNING],
                                                  axis=1).reshape(-1, len(LEARNING))  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrays)
    return path


def read(path) -> Run:
    """The run file at ``path``. Raises ``ValueError`` if it is not one this code reads."""
    with np.load(path, allow_pickle=False) as z:
        if "meta" not in z.files:
            raise ValueError(f"{path} is not a run file (no meta)")
        meta = _unjson(z["meta"])
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"{path} is {meta.get('schema')!r}, not a run file ({SCHEMA})")
        names = meta["learning"]
        arms = []
        for k, info in enumerate(meta["arms"]):
            learning, i = [], 0
            while f"learning{k}_{i}" in z.files:
                rows = z[f"learning{k}_{i}"]
                learning.append({n: rows[:, j].astype(float) for j, n in enumerate(names)})
                i += 1
            arms.append(
                Arm(
                    arm=info["arm"],
                    label=info["label"],
                    spec=_unjson(z[f"spec{k}"]),
                    messages=_unjson(z[f"messages{k}"]),
                    truth=z[f"truth{k}"].astype(float),
                    corridor=z[f"corridor{k}"].astype(bool),
                    plate=z[f"plate{k}"] if info.get("plate") else None,
                    learning=learning,
                )
            )
    return Run(
        demo=meta["demo"],
        seed=int(meta["seed"]),
        scene=meta.get("scene"),
        steps=int(meta["steps"]),
        px_per_m=int(meta["px_per_m"]),
        config=meta["config"],
        arms=arms,
        meta={k: v for k, v in meta.items() if k not in ("arms", "learning")},
    )
