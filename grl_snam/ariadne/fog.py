"""The fog-of-war stories (``grl-snam fog record / capture / play``) replayed in an Ariadne document.

    python -m grl_snam.ariadne fog ghost              # record if needed, then play in a window
    python -m grl_snam.ariadne fog city --record      # record again first
    python -m grl_snam.ariadne fog blocker --offscreen --frames 200 --png blocker.png
    python -m grl_snam.ariadne fog unit --headless --frames 50 --max-steps 120

The story (one of ``grl_snam.fog_stories.STORIES``: ghost, blocker, city, traffic, pursuit,
unit) is recorded with ``grl_snam.tools.fog_record.record`` into ``--trace`` (default
``traces/<story>``) when that trace is missing or stale, as ``grl-snam fog record`` does. The
scene is ``grl_snam.fog_scene``'s, the one the capture and the live player draw: the fog tiers,
what the agent believes and what it has not found, its route, track and sensor ring, seen from
above. The document (``docs/player.ari``) adds the transport (play, speed, loop, scrubbing), the
recording's captions as text, and the vehicle's HUD lines; ``--headless`` replays the readouts
with nothing drawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

from grl_snam.ariadne import _host
from grl_snam.ariadne._player import Player


def _args(p):
    from grl_snam.fog_stories import STORIES

    p.add_argument("story", choices=sorted(STORIES), help="which fog story")
    p.add_argument("--trace", default=None, help="the trace directory (default traces/<story>)")
    p.add_argument("--record", action="store_true", help="record again even if a trace exists")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="cap the recording (ticks)")


def ensure_trace(story_key, trace_dir, *, record_again=False, seed=0, max_steps=None):
    """The story's trace directory, recorded first when it is missing, stale or asked for."""
    from grl_snam.fog_stories import STORIES
    from grl_snam.tools.fog_record import is_stale, record

    trace_dir = Path(trace_dir)
    if (
        record_again
        or not (trace_dir / "trace.json").exists()
        or is_stale(trace_dir, STORIES[story_key])
    ):
        print(f"grl_snam fog: recording {story_key} into {trace_dir} ...", flush=True)
        record(story_key, trace_dir, seed=seed, max_steps=max_steps)
    return trace_dir


def main(argv=None) -> int:
    args = _host.parse(argv, _args)
    from grl_snam import fog_scene
    from grl_snam.fog_stories import STORIES
    from grl_snam.fog_trace import Trace

    story = STORIES[args.story]
    trace_dir = ensure_trace(
        args.story,
        args.trace or Path("traces") / story.key,
        record_again=args.record,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    trace = Trace.load(trace_dir)
    host = _host.open_host("player.ari", "fog", args)
    drawn = host.mode != "headless"
    state = None
    hx, hy = fog_scene.plate_half(story)
    if drawn:
        state = fog_scene.build(host.lab, story, trace)
        _host.track_camera(host, 2.0 * fog_scene.CAR_L)  # the car icon is drawn larger than life
    camera = _host.CameraSwitch(host, lambda: host.camera.frameMap(0.0, 0.0, hy * 1.05, hx * 1.05))

    def draw(t):
        m = trace.to_metrics(t)
        if drawn:
            fog_scene.apply(host.lab, story, trace, t, state)
            host.lab.pump()
            camera.update((m.x, m.y, 1.4))
        _host.publish_metrics(host, m)
        host.set_table(
            "agents",
            [
                {
                    "key": story.key,
                    "speed": f"{m.speed_mps:.1f}",
                    "goal": f"{m.goal_dist_m:.0f}",
                    "mode": "fog",
                }
            ],
        )
        host.set("play.caption", trace.caption_at(t) or "")

    player = Player(host, trace.duration_s, draw, title=f"{story.title} ({story.key})")
    host.on_frame(player.tick)
    return _host.finish(host, args)


if __name__ == "__main__":
    sys.exit(main())
