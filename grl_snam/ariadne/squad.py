"""GRL-SNAM's squad recordings on real Austin, replayed in an Ariadne document with Humvees.

    python -m grl_snam.ariadne squad finale-rendezvous    # act one: eight vehicles converge
    python -m grl_snam.ariadne squad finale-pursuit       # act two: pairs chase moving targets
    python -m grl_snam.ariadne squad convoy               # each vehicle follows the one ahead
    python -m grl_snam.ariadne squad pursuit              # three pursuers per runner
    python -m grl_snam.ariadne squad convoy --offscreen --frames 300 --png convoy.png

Each run is recorded once, as ``grl_snam.tools.finale_record`` / ``convoy_record`` /
``pursuit_record`` record it, into ``--out`` (default ``traces/<run>``) when it is missing or
``--record`` asks; that needs the Austin bundle (``scene-austin-south``) and takes minutes. The
replay draws the recorded poses, never re-simulating: Austin's terrain and buildings, a Humvee
at its real size per agent (seated on the terrain at its recorded heading) with a pole in its
colour so it can be found from afar, its goal (a pyramid, moving in a chase) and its track.
The document (``docs/player.ari``) is the transport, the captions, a row per agent and the
first agent's HUD. This replaces finale_capture's oversized boxes, beacons and in-scene HUD with
true-scale vehicles and panels; its fog decal and picture-in-picture are not drawn here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from grl_snam.ariadne import _host
from grl_snam.ariadne._player import Player

RUNS = {
    "finale-rendezvous": ("grl_snam.tools.finale_record", "record_rendezvous",
                          "Finale, act one: a blind rendezvous across Austin"),
    "finale-pursuit": ("grl_snam.tools.finale_record", "record_pursuit",
                       "Finale, act two: two vehicles per moving target"),
    "convoy": ("grl_snam.tools.convoy_record", "record_convoy",
               "A convoy: each vehicle chases the one in front"),
    "pursuit": ("grl_snam.tools.pursuit_record", "record_pursuit_v2",
                "Pursuit: three vehicles per runner"),
}  # fmt: skip


def _args(p):
    p.add_argument("run", choices=sorted(RUNS), help="which recording")
    p.add_argument("--out", default=None, help="the recording's directory (default traces/<run>)")
    p.add_argument("--record", action="store_true", help="record again even if it exists")
    p.add_argument("--bundle", default=None, help="the Austin scene bundle (default: installed)")


def ensure_recording(run, out, bundle, *, record_again=False):
    """The run's directory, recorded first when it has no squad.json or ``record_again``."""
    import importlib

    out = Path(out)
    if record_again or not (out / "squad.json").exists():
        if bundle is None:
            raise RuntimeError(
                f"recording {run} needs the Austin scene bundle (scene-austin-south)"
            )
        module, fn, _ = RUNS[run]
        print(f"grl_snam squad: recording {run} into {out} (minutes) ...", flush=True)
        getattr(importlib.import_module(module), fn)(str(bundle), out)
    return out


def load(out):
    from grl_snam.fog_trace import Trace

    spec = json.loads((Path(out) / "squad.json").read_text())
    return spec, {a["key"]: Trace.load(Path(out) / a["key"]) for a in spec["agents"]}


def add_city(host, bundle):
    """Austin's terrain and buildings in the document's scene, matte; the terrain sampler."""
    from pycvc_gl.scenes import load_geometry_bundle

    sample = load_geometry_bundle(
        host.lab, str(bundle), terrain_color=(0.42, 0.46, 0.36), building_color=(0.80, 0.78, 0.74)
    )
    for name, (amb, dif, spec) in (("terrain", (0.4, 0.85, 0.0)), ("buildings", (0.3, 0.8, 0.04))):
        gn = host.scene.geometry_node(name)
        gn.setAmbient(amb)
        gn.setDiffuse(dif)
        gn.setSpecular(spec)
    return sample


def _pyramid(size):
    s = size
    v = [0.0, 0.0, 0.0, -s / 2, -s / 2, s, s / 2, -s / 2, s, s / 2, s / 2, s, -s / 2, s / 2, s]
    return v, [0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 1, 2, 3, 1, 3, 4]


def main(argv=None) -> int:
    args = _host.parse(argv, _args)
    bundle = _host.bundle_or_none(args.bundle)
    out = ensure_recording(
        args.run, args.out or Path("traces") / args.run, bundle, record_again=args.record
    )
    spec, traces = load(out)
    keys = [a["key"] for a in spec["agents"]]
    colors = {a["key"]: tuple(a["color"]) for a in spec["agents"]}
    duration = max(tr.duration_s for tr in traces.values())
    host = _host.open_host("player.ari", "squad", args)
    drawn = host.mode != "headless"
    sample = None
    if drawn:
        from pycvc_gl.vehicle import seat_transform

        from grl_snam.fog_scene import _style

        if bundle is not None:
            sample = add_city(host, bundle)
        model = _host.humvee_model(bundle)
        for k in keys:
            _host.add_vehicle(host, f"car_{k}", model)
            host.lab.add_mesh(f"pole_{k}", *_host.humvee_box(0.35, 0.35, 7.0), color=colors[k])
            host.lab.add_mesh(f"goal_{k}", *_pyramid(6.0), color=colors[k])
            host.lab.add_path(f"track_{k}", [(0, 0, 0), (0.1, 0, 0)], colors[k])
        _host.track_camera(host)
    camera = _host.CameraSwitch(host, lambda: host.frame_bounds((-600, -600, 0), (600, 600, 60)))
    frame = {"n": 0}

    def ground(x, y):
        return sample(x, y) if sample else 0.0

    def draw(t):
        rows = []
        for i, k in enumerate(keys):
            tr = traces[k]
            p = tr.pose_at(min(t, tr.duration_s))
            m = tr.to_metrics(min(t, tr.duration_s))
            rows.append(
                {
                    "key": k,
                    "speed": f"{m.speed_mps:.1f}",
                    "goal": f"{m.goal_dist_m:.0f}",
                    "mode": "arrived" if m.goal_dist_m < 8.0 else "driving",
                }
            )
            if i == 0:
                _host.publish_metrics(host, m)
            if not drawn:
                continue
            z = ground(p.x, p.y)
            host.scene.getGraphics(f"car_{k}").setTransform(
                seat_transform(p.x, p.y, p.heading_rad, sample if sample else None)
            )
            host.scene.getGraphics(f"pole_{k}").setTransform(
                [1, 0, 0, p.x, 0, 1, 0, p.y, 0, 0, 1, z + 2.0, 0, 0, 0, 1]
            )
            g = tr.goal_at(min(t, tr.duration_s))
            if g is not None:
                host.scene.getGraphics(f"goal_{k}").setTransform(
                    [1, 0, 0, g[0], 0, 1, 0, g[1], 0, 0, 1, ground(*g) + 8.0, 0, 0, 0, 1]
                )
            if frame["n"] % 6 == 0:
                pts = tr.track_upto(min(t, tr.duration_s))
                if len(pts) >= 2:
                    step = max(1, len(pts) // 300)
                    host.lab.add_path(
                        f"track_{k}",
                        [(float(x), float(y), ground(x, y) + 0.6) for x, y in pts[::step]],
                        colors[k],
                    )
                    _style(host.scene, f"track_{k}", color=colors[k], mode="lines", line_width=3)
        host.set_table("agents", rows)
        host.set("play.caption", traces[keys[0]].caption_at(t) or "")
        if drawn:
            p0 = traces[keys[0]].pose_at(min(t, traces[keys[0]].duration_s))
            camera.update((p0.x, p0.y, ground(p0.x, p0.y)))
        frame["n"] += 1

    player = Player(host, duration, draw, title=RUNS[args.run][2])
    host.on_frame(player.tick)
    return _host.finish(host, args)


if __name__ == "__main__":
    sys.exit(main())
