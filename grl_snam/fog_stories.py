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


def build_scenario(story: Story, model=None, *, seed: int = 0):
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

    return FogScenario(
        story.truth_grid(),
        story.bounds,
        story.scale,
        model,
        story.meta(),
        waypoints=[tuple(w) for w in story.waypoints],
        prior_occ=story.prior_grid(),
        events=list(story.events),
        unknown=story.unknown,
        sense_every=story.sense_every,
        sensor=dict(story.sensor),
        dynamics=story.dynamics,
        unit_ttl_s=story.unit_ttl_s,
        reach_tol=story.reach_tol,
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
