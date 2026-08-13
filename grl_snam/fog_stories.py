"""The three fog-of-war stories, as data.

This module is the single source of truth for what "story 2" *is*. The
recorder, the live VolRover3 demo, the offscreen capture and the tests all
build their scenario from here, so they cannot disagree about the geometry,
the events, or the timing.

The stories, and what each one is meant to prove:

1. ``ghost`` — **the map has a wall reality no longer has.** Belief keeps the
   obstacle until a ray actually crosses where it stood, so the agent detours
   around nothing, then clears it and replans straight through. This is the
   cost of a stale map, shown rather than asserted.
2. ``blocker`` — **reality grows a wall the map does not know about.** It is
   discovered on approach and forces a replan mid-drive. This is where the
   vehicle dynamics bite: a car with a turning radius cannot simply reverse
   its velocity vector the way a point mass can.
3. ``unit`` — **a transient obstacle that decays.** A moving unit is marked,
   routed around, and then expires instead of smearing a permanent ghost
   along its path.

Pure numpy; torch is imported lazily inside :func:`build_scenario` so that
importing this module (for the CLI, the tests, or a renderer) costs nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from grl_snam.scenario import Event

# Cell rectangles are (r0, r1, c0, c1) half-open, matching numpy slicing and
# the Event(kind="add_rect") argument order.
Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class Story:
    """One scripted fog-of-war run."""

    key: str
    title: str
    subtitle: str

    # ── the world ───────────────────────────────────────────────────────────
    n: int = 96
    bounds: tuple[float, float, float, float] = (-100.0, -100.0, 100.0, 100.0)
    scale: float = 0.05
    dt: float = 0.06
    nsub: int = 2
    vmax: float = 0.9
    rr: float = 0.15
    d_hat: float = 0.35

    # ── the scenario ────────────────────────────────────────────────────────
    start: tuple[float, float] = (-60.0, 0.0)
    waypoints: tuple[tuple[float, float], ...] = ((60.0, 0.0),)
    truth_rects: tuple[Rect, ...] = ()
    prior_rects: tuple[Rect, ...] = ()
    events: tuple[Event, ...] = ()
    sensor: dict = field(default_factory=lambda: dict(range_m=35.0, n_rays=240))
    sense_every: int = 5
    unknown: str = "optimistic"
    unit_ttl_s: float = 4.0
    dynamics: str = "bicycle"
    max_steps: int = 8000

    # reach_tol is normalized, so the metric radius is reach_tol/scale. The
    # SdfNavigator default of 0.8 means 16 m here: the vehicle stops visibly
    # short of the goal marker and the video reads as broken. 0.15 -> 3 m.
    reach_tol: float = 0.15

    # ── presentation ────────────────────────────────────────────────────────
    # (world_t_start, world_t_end, text); burned in at encode time.
    captions: tuple[tuple[float, float, str], ...] = ()
    cam: str = "map"

    # A path from an earlier full-knowledge run, drawn underneath for
    # comparison. Not part of the simulation -- presentation only.
    reference_xy: object | None = None
    # Full knowledge: render lit, with no fog tiers. The baseline variant.
    no_fog: bool = False
    # Route inflation in METRES (cells are not a fixed size across rasters).
    inflate_m: float = 6.0
    # Vehicles that physically exist in truth and move (see Mover).
    movers: tuple = ()
    # A target that moves. When set, it overrides the final waypoint.
    moving_goal: object | None = None

    def cell_to_world(self, r: float, c: float) -> tuple[float, float]:
        mnx, mny, mxx, mxy = self.bounds
        x = mnx + c / (self.n - 1) * (mxx - mnx)
        y = mny + r / (self.n - 1) * (mxy - mny)
        return float(x), float(y)

    def rect_world(self, rect: Rect) -> tuple[float, float, float, float]:
        """Cell rect -> ``(x0, y0, x1, y1)`` in metres, for the renderers."""
        r0, r1, c0, c1 = rect
        x0, y0 = self.cell_to_world(r0, c0)
        # r1/c1 are exclusive, so the far edge is the start of the next cell.
        x1, y1 = self.cell_to_world(r1, c1)
        return x0, y0, x1, y1

    def meta(self) -> dict:
        """The CoefMLP/navigator meta dict this story's scenario runs under."""
        return dict(
            scale=self.scale,
            center=(0.0, 0.0),
            region=float(self.bounds[2]),
            rr=self.rr,
            d_hat=self.d_hat,
            dt=self.dt,
            nsub=self.nsub,
            vmax=self.vmax,
            bounds=list(self.bounds),
        )

    def truth_grid(self) -> np.ndarray:
        return _grid(self.n, self.truth_rects)

    def prior_grid(self) -> np.ndarray | None:
        return _grid(self.n, self.prior_rects) if self.prior_rects else None


def _grid(n: int, rects: Sequence[Rect]) -> np.ndarray:
    g = np.zeros((n, n), bool)
    for r0, r1, c0, c1 in rects:
        g[r0:r1, c0:c1] = True
    return g


@dataclass(frozen=True)
class Mover:
    """A vehicle that physically exists and moves — not a scripted mark.

    A `unit_at` Event stamps the *belief* layer directly, which is fine for
    "something was reported here" but cheats: the agent is told. A Mover is
    part of TRUTH. It occludes, the sensor has to actually see it, and until
    it does the agent routes straight at it. That is the honest version, and
    it is what makes a second vehicle interesting rather than decorative.

    ``path`` is a world-coordinate polyline walked at ``speed_mps``; with
    ``loop`` the mover ping-pongs along it forever.
    """

    key: str
    path: tuple[tuple[float, float], ...]
    speed_mps: float = 6.0
    half_m: float = 4.0  # half-extent of the square footprint
    start_s: float = 0.0
    loop: bool = True

    def position_at(self, t: float) -> tuple[float, float]:
        """Where it is at world time ``t`` — constant speed along the path."""
        pts = np.asarray(self.path, np.float64)
        if len(pts) == 1:
            return float(pts[0][0]), float(pts[0][1])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        if total <= 0.0:
            return float(pts[0][0]), float(pts[0][1])
        d = max(0.0, t - self.start_s) * self.speed_mps
        if self.loop:
            # ping-pong: out and back, so a mover stays in frame
            d = d % (2.0 * total)
            if d > total:
                d = 2.0 * total - d
        else:
            d = min(d, total)
        for i, s in enumerate(seg):
            if d <= s or i == len(seg) - 1:
                f = 0.0 if s <= 0 else min(d / s, 1.0)
                a, b = pts[i], pts[i + 1]
                q = a + f * (b - a)
                return float(q[0]), float(q[1])
            d -= s
        return float(pts[-1][0]), float(pts[-1][1])


@dataclass(frozen=True)
class MovingGoal:
    """A target that does not wait to be reached.

    Chasing a moving goal is a different control problem from reaching a fixed
    one — the navigator's ``track_goal`` retargets in place, so the route spine
    is replanned toward a point that has moved rather than restarted.
    """

    path: tuple[tuple[float, float], ...]
    speed_mps: float = 4.0
    start_s: float = 0.0
    loop: bool = True

    def position_at(self, t: float) -> tuple[float, float]:
        return Mover(
            key="goal",
            path=self.path,
            speed_mps=self.speed_mps,
            start_s=self.start_s,
            loop=self.loop,
        ).position_at(t)


def unit_track(
    *, step0: int, every: int, count: int, r0: int, c0: int, r1: int, c1: int
) -> tuple[Event, ...]:
    """A unit sweeping from cell (r0, c0) to (r1, c1), one mark every ``every``
    steps. Marking repeatedly is the point: a single mark is a one-second blip
    that nobody watching will notice, and the decay only reads as decay if
    there is a trail to decay *from*."""
    out = []
    for i in range(count):
        f = i / max(count - 1, 1)
        r = int(round(r0 + f * (r1 - r0)))
        c = int(round(c0 + f * (c1 - c0)))
        out.append(Event(step=step0 + i * every, kind="unit_at", args=(r, c)))
    return tuple(out)


def city_blocks(n: int, *, rows: int = 3, cols: int = 3, gap: int = 8, margin: int = 12):
    """A grid of rectangular blocks with streets between them.

    The escalation from "one wall across the route" to a scene with topology:
    there are several ways through, so the route choice becomes a decision
    rather than a detour, and a stale map costs a wrong TURN rather than a
    wrong metre.
    """
    out = []
    span = n - 2 * margin
    # Both axes need their own pitch AND their own gap. Deriving the block size
    # from the row pitch only made the columns abut, so a 2x3 grid rendered as
    # two solid slabs with no street between them -- and a scene about choosing
    # a route through streets has to have streets.
    rp = span // max(rows, 1)
    cp = span // max(cols, 1)
    rb = max(2, rp - gap)
    cb = max(2, cp - gap)
    for i in range(rows):
        for j in range(cols):
            r0 = margin + i * rp
            c0 = margin + j * cp
            out.append((r0, min(r0 + rb, n), c0, min(c0 + cb, n)))
    return tuple(out)


def corridor(n: int, *, row: int, gap_col: int, gap_width: int = 6, thickness: int = 3):
    """A wall across the map with a single doorway — the cheapest way to make
    topology matter: miss the gap and there is no way through at all."""
    return (
        (row, row + thickness, 0, max(0, gap_col - gap_width // 2)),
        (row, row + thickness, min(n, gap_col + gap_width // 2), n),
    )


STORIES: dict[str, Story] = {
    "ghost": Story(
        key="ghost",
        title="Stale map",
        subtitle="a wall the world no longer has",
        # Reality: open field. The map: a wall square across the route.
        prior_rects=((20, 76, 47, 50),),
        sensor=dict(range_m=35.0, n_rays=240),
        sense_every=5,
        captions=(
            (0.0, 4.0, "MAP: wall ahead.   TRUTH: open field."),
            (4.0, 7.0, "Belief is all it has - so it detours."),
            (7.0, 11.0, "SENSOR: nothing there. Map updated, route replanned."),
            (11.0, 99.0, "Straight to the goal through a wall that was never there."),
        ),
    ),
    "blocker": Story(
        key="blocker",
        title="New blocker",
        subtitle="discovered on approach, mid-drive",
        # Reality grows a ~30 m wall across the route at step 40.
        events=(Event(step=40, kind="add_rect", args=(41, 56, 46, 50)),),
        sensor=dict(range_m=45.0, n_rays=240),
        sense_every=3,
        captions=(
            (0.0, 3.0, "Open field. Route is a straight line."),
            (3.0, 6.5, "A blocker appears - the map does not know yet."),
            (6.5, 10.0, "Discovered on approach. Replan, and turn within the radius."),
            (10.0, 99.0, "Scored against TRUTH: zero steps inside the wall."),
        ),
    ),
    "city": Story(
        key="city",
        title="City blocks",
        subtitle="topology, not just a detour",
        truth_rects=city_blocks(96, rows=3, cols=3, gap=9, margin=14),
        # The map knows the city -- but one block has been demolished and
        # another has appeared, so the shortest believed route is wrong twice.
        prior_rects=city_blocks(96, rows=3, cols=3, gap=9, margin=14)[:-1] + ((30, 42, 62, 74),),
        sensor=dict(range_m=38.0, n_rays=240),
        sense_every=3,
        start=(-78.0, -60.0),
        waypoints=((78.0, 62.0),),
        max_steps=3000,
        captions=(
            (0.0, 4.0, "A city block grid - several ways through."),
            (4.0, 9.0, "The map is out of date in two places."),
            (9.0, 16.0, "Each discovery changes the TURN, not just the lane."),
            (16.0, 999.0, "Re-routing through streets it can actually see."),
        ),
    ),
    "traffic": Story(
        key="traffic",
        title="Moving vehicles",
        subtitle="obstacles that have to be seen to be known",
        truth_rects=city_blocks(96, rows=2, cols=3, gap=10, margin=18),
        prior_rects=city_blocks(96, rows=2, cols=3, gap=10, margin=18),
        movers=(
            Mover(key="v1", path=((6.0, -78.0), (6.0, 78.0)), speed_mps=9.0, half_m=5.0),
            Mover(
                key="v2",
                path=((-30.0, 66.0), (62.0, -34.0)),
                speed_mps=7.0,
                half_m=5.0,
                start_s=2.0,
            ),
        ),
        sensor=dict(range_m=42.0, n_rays=240),
        sense_every=3,
        start=(-76.0, 0.0),
        waypoints=((76.0, 6.0),),
        unit_ttl_s=1.5,
        max_steps=3000,
        captions=(
            (0.0, 4.0, "The map is correct. The traffic is not on it."),
            (4.0, 10.0, "Two vehicles, discovered only when they are SEEN."),
            (10.0, 18.0, "Tracked in a layer that expires - no permanent smear."),
            (18.0, 999.0, "Zero contact, against a world that keeps moving."),
        ),
    ),
    "pursuit": Story(
        key="pursuit",
        title="Moving target",
        subtitle="the goal does not wait",
        truth_rects=city_blocks(96, rows=2, cols=2, gap=12, margin=22),
        prior_rects=city_blocks(96, rows=2, cols=2, gap=12, margin=22),
        moving_goal=MovingGoal(
            path=((70.0, 0.0), (70.0, 66.0), (0.0, 76.0), (-56.0, 40.0)),
            speed_mps=5.5,
            loop=False,
        ),
        sensor=dict(range_m=40.0, n_rays=240),
        sense_every=3,
        start=(-76.0, -30.0),
        waypoints=((70.0, 0.0),),
        max_steps=3000,
        captions=(
            (0.0, 4.0, "The target is moving."),
            (4.0, 11.0, "Retargeted in place - the route follows, it does not restart."),
            (11.0, 999.0, "Closing on a goal that keeps running."),
        ),
    ),
    "unit": Story(
        key="unit",
        title="Moving unit",
        subtitle="a transient obstacle that decays",
        events=unit_track(step0=30, every=3, count=30, r0=70, c0=40, r1=30, c1=56),
        sensor=dict(range_m=40.0, n_rays=240),
        sense_every=2,
        unit_ttl_s=2.0,
        captions=(
            (0.0, 4.0, "A unit crosses the route."),
            (4.0, 9.0, "Marked and avoided - in a layer that expires."),
            (9.0, 99.0, "No permanent smear: the marks decay, the map stays clean."),
        ),
    ),
}


def build_scenario(story: Story, model=None, *, seed: int = 0, truth_occ=None, prior_occ=None):
    """Turn a :class:`Story` into a runnable :class:`~grl_snam.scenario.FogScenario`.

    The one place a story becomes a simulation. ``model`` defaults to a
    seed-initialised :class:`sdf_nav.CoefMLP` (the navigator's coefficients are
    biased toward a known-good regime, so an untrained net drives adequately —
    the demo is about belief and dynamics, not about the learned coefficients).
    """
    import torch  # noqa: PLC0415 -- lazy: importing this module must stay cheap

    import sdf_nav
    from grl_snam.scenario import FogScenario

    if model is None:
        torch.manual_seed(seed)
        model = sdf_nav.CoefMLP()
        model.eval()

    # truth_occ/prior_occ let a caller supply a rasterized world (e.g. a real
    # city) instead of the story's declarative rectangles. Kept OUT of Story
    # itself: it is frozen and content-hashed for staleness, and a megabyte of
    # occupancy does not belong in a spec hash.
    return FogScenario(
        story.truth_grid() if truth_occ is None else truth_occ.astype(bool),
        story.bounds,
        story.scale,
        model,
        story.meta(),
        waypoints=[tuple(w) for w in story.waypoints],
        prior_occ=story.prior_grid() if prior_occ is None else prior_occ,
        events=list(story.events),
        unknown=story.unknown,
        sense_every=story.sense_every,
        sensor=dict(story.sensor),
        dynamics=story.dynamics,
        unit_ttl_s=story.unit_ttl_s,
        reach_tol=story.reach_tol,
        inflate_m=story.inflate_m,
        movers=story.movers,
        moving_goal=story.moving_goal,
    ).start(story.start)


def shrunk(story: Story, *, n: int = 48, max_steps: int = 120) -> Story:
    """A small, fast variant of a story for tests. Rects scale with the grid so
    the geometry stays proportionally the same."""
    f = n / story.n

    def sc(rect: Rect) -> Rect:
        return tuple(int(round(v * f)) for v in rect)  # type: ignore[return-value]

    return replace(
        story,
        n=n,
        max_steps=max_steps,
        truth_rects=tuple(sc(r) for r in story.truth_rects),
        prior_rects=tuple(sc(r) for r in story.prior_rects),
        events=tuple(
            Event(
                step=e.step,
                kind=e.kind,
                args=(
                    sc(e.args)
                    if e.kind.endswith("_rect")
                    else tuple(int(round(v * f)) for v in e.args)
                ),
            )
            for e in story.events
        ),
    )
