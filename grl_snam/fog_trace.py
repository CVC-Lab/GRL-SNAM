"""The measured trace: what the simulation *did*, replayed rather than redone.

A trace is the demo's only artifact. The simulation runs exactly once, in
:mod:`grl_snam.tools.fog_record`; every renderer afterwards — the live
VolRover3 demo, the offscreen capture, and any future web view — reads this
and recomputes nothing. That discipline is not bureaucracy: the last time a
renderer re-derived state it already had, it estimated heading from
frame-to-frame displacement and overwrote the measured speed with playback
ground speed, so the video quietly disagreed with the simulation it claimed
to show.

The bundle is two files:

``trace.npz``
    one row per world tick (pose, speed, clearance, belief version, flags),
    plus belief/dynamic-layer snapshots packed with :func:`numpy.packbits` at
    the ticks where belief actually changed, plus the route polyline at each
    replan.

``trace.json``
    the manifest — ``fixed_dt``, bounds, the story spec hash, seed, torch
    version, captions, and the headline numbers quoted on stage. Storing the
    summary at record time means the end cards can never drift from the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading_rad: float
    speed_mps: float
    tick: int
    alpha: float


def _lerp_angle(a: float, b: float, f: float) -> float:
    """Shortest-arc interpolation; a vehicle crossing +-pi must not spin the
    long way round for one frame."""
    d = (b - a + np.pi) % (2.0 * np.pi) - np.pi
    return float(a + d * f)


class Trace:
    """Reader for a recorded fog-of-war run."""

    def __init__(self, rows: dict, snaps: dict, routes: list, manifest: dict):
        self.rows = rows
        self.snaps = snaps
        self.routes = routes
        self.manifest = manifest

        self.fixed_dt = float(manifest["fixed_dt"])
        self.story_key = manifest["story"]
        self.summary = manifest.get("summary", {})
        self.captions = [tuple(c) for c in manifest.get("captions", [])]
        self.shape = tuple(manifest["shape"])
        self.bounds = tuple(manifest["bounds"])
        self.n_ticks = int(len(rows["tick"]))

    # ── io ──────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, trace_dir: str | Path) -> Trace:
        d = Path(trace_dir)
        manifest = json.loads((d / "trace.json").read_text())
        z = np.load(d / "trace.npz")
        row_keys = [k for k in z.files if k.startswith("row_")]
        rows = {k[4:]: z[k] for k in row_keys}
        snaps = {
            "tick": z["snap_tick"],
            "occ": z["snap_occ"],
            "dyn": z["snap_dyn"],
        }
        offs = z["route_offsets"]
        pts = z["route_points"]
        routes = [pts[offs[i] : offs[i + 1]] for i in range(len(offs) - 1)]
        return cls(rows, snaps, routes, manifest)

    # ── queries ─────────────────────────────────────────────────────────────
    @property
    def duration_s(self) -> float:
        return self.n_ticks * self.fixed_dt

    def _tick_index(self, t: float) -> tuple[int, float]:
        """World time -> (row index, fraction into the next row).

        Snaps to the nearest tick when ``t`` is within a rounding error of a
        boundary. Without this the renderer sits one tick behind at every
        exact quantum: 29 * 0.06 is 1.7399999999999998, which divided back by
        0.06 gives 28.999999999999996 and truncates to 28 at alpha ~ 1.0. The
        pose would be a whole frame stale precisely at the moments the trace
        is authoritative about.
        """
        if t <= 0.0:
            return 0, 0.0
        raw = t / self.fixed_dt
        nearest = round(raw)
        if abs(raw - nearest) < 1e-6:
            raw = float(nearest)
        i = int(raw)
        if i >= self.n_ticks - 1:
            return self.n_ticks - 1, 0.0
        return i, float(raw - i)

    def row(self, index: int) -> dict:
        i = max(0, min(int(index), self.n_ticks - 1))
        return {k: v[i] for k, v in self.rows.items()}

    def pose_at(self, t: float) -> Pose:
        """Interpolated pose at world time ``t``. At an exact tick boundary
        this returns the recorded row unchanged — pinned by test."""
        i, f = self._tick_index(t)
        r, x, y = self.rows, self.rows["x"], self.rows["y"]
        if f == 0.0:
            return Pose(
                float(x[i]),
                float(y[i]),
                float(r["heading_rad"][i]),
                float(r["speed_mps"][i]),
                i,
                0.0,
            )
        j = i + 1
        return Pose(
            float(x[i] + (x[j] - x[i]) * f),
            float(y[i] + (y[j] - y[i]) * f),
            _lerp_angle(float(r["heading_rad"][i]), float(r["heading_rad"][j]), f),
            float(r["speed_mps"][i] + (r["speed_mps"][j] - r["speed_mps"][i]) * f),
            i,
            f,
        )

    def snapshot_index_at(self, t: float) -> int:
        """Which belief snapshot is current at ``t``. Belief is a step
        function: it changes only when the sensor changed its mind."""
        i, _ = self._tick_index(t)
        st = self.snaps["tick"]
        if len(st) == 0:
            return -1
        return int(np.searchsorted(st, i, side="right") - 1)

    def belief_at(self, t: float) -> tuple[np.ndarray, np.ndarray, int]:
        """``(believed_occupancy, dynamic_marks, snapshot_index)``."""
        k = self.snapshot_index_at(t)
        ny, nx = self.shape
        if k < 0:
            empty = np.zeros((ny, nx), bool)
            return empty, empty.copy(), -1
        occ = np.unpackbits(self.snaps["occ"][k], count=ny * nx).astype(bool).reshape(ny, nx)
        dyn = np.unpackbits(self.snaps["dyn"][k], count=ny * nx).astype(bool).reshape(ny, nx)
        return occ, dyn, k

    def route_at(self, t: float) -> np.ndarray:
        k = self.snapshot_index_at(t)
        if k < 0 or k >= len(self.routes):
            return np.empty((0, 2), np.float32)
        return self.routes[k]

    def track_upto(self, t: float) -> np.ndarray:
        """The driven path so far — the yellow trail."""
        i, _ = self._tick_index(t)
        return np.stack([self.rows["x"][: i + 1], self.rows["y"][: i + 1]], axis=-1)

    def caption_at(self, t: float) -> str | None:
        for t0, t1, text in self.captions:
            if t0 <= t < t1:
                return text
        return None

    def scaled_captions(self, speed: float) -> list[tuple[float, float, str]]:
        """Captions in *playback* seconds, for burning in at encode time."""
        s = max(speed, 1e-9)
        return [(t0 / s, min(t1, self.duration_s) / s, txt) for t0, t1, txt in self.captions]

    def to_metrics(self, t: float):
        """A :class:`~grl_snam.metrics.NavMetrics` for the recorded tick, so the
        HUD goes through the same ``hud_lines`` the live demos use — one HUD,
        no second implementation to drift."""
        from grl_snam.metrics import NavMetrics

        i, _ = self._tick_index(t)
        r = self.rows
        return NavMetrics(
            x=float(r["x"][i]),
            y=float(r["y"][i]),
            speed_mps=float(r["speed_mps"][i]),
            heading_rad=float(r["heading_rad"][i]),
            clearance_m=float(r["clearance_m"][i]),
            goal_dist_m=float(r["goal_dist_m"][i]),
            goal_index=int(r["goal_index"][i]),
        )
