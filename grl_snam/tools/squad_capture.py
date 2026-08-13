"""Render a squad run: N agents, N routes, N tracks, one world.

The fog shown is the **team's** combined coverage, because a single frame
cannot show four private maps at once. Each agent still plans on its own
belief — the clip captions say so, and the per-agent sensor rings make the
difference legible: coverage is the union, knowledge is not.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from grl_snam import fog_scene
from grl_snam.clock import WorldClock
from grl_snam.fog_trace import Trace
from grl_snam.tools.fog_capture import (
    _filters,
    open_encoder,
    open_renderer,
    plate_crop_rgb,
)


def _load(bundle: Path):
    spec = json.loads((bundle / "squad.json").read_text())
    traces = {a["key"]: Trace.load(bundle / a["key"]) for a in spec["agents"]}
    return spec, traces


def build(lab, story, spec, traces) -> dict:
    scene = lab._scene
    v, t = fog_scene.quad(
        -fog_scene.PLATE_HALF_X,
        -fog_scene.PLATE_HALF_Y,
        fog_scene.PLATE_HALF_X,
        fog_scene.PLATE_HALF_Y,
        0.0,
    )
    lab.add_mesh("ground", v, t, fog_scene.GROUND)

    for a in spec["agents"]:
        k, col = a["key"], tuple(a["color"])
        sx, sy = a["start"]
        gx, gy = a["goal"]
        lab.add_markers(f"start_{k}", [(sx, sy, 1.2)], col)
        fog_scene._style(scene, f"start_{k}", mode="points", point_size=13)
        lab.add_markers(f"goal_{k}", [(gx, gy, 1.2)], col)
        fog_scene._style(scene, f"goal_{k}", mode="points", point_size=19)
        lab.add_path(f"route_{k}", [(sx, sy, 1.0), (gx, gy, 1.0)], col)
        fog_scene._style(scene, f"route_{k}", mode="lines", line_width=3)
        lab.add_path(f"track_{k}", [(sx, sy, 1.1), (sx + 0.1, sy, 1.1)], col)
        fog_scene._style(scene, f"track_{k}", mode="lines", line_width=3)
        v, t = fog_scene.quad(
            -fog_scene.CAR_L / 2,
            -fog_scene.CAR_W / 2,
            fog_scene.CAR_L / 2,
            fog_scene.CAR_W / 2,
            1.4,
        )
        lab.add_mesh(f"car_{k}", v, t, col)
        rng = traces[k].sensor_range_m
        if rng > 0:
            lab.add_path(f"fov_{k}", fog_scene.ring_points(sx, sy, rng, 1.6), col)
            fog_scene._style(scene, f"fov_{k}", mode="lines", line_width=1)

    fog_scene._hide_chrome(lab)
    return {
        "snap": -2,
        "origins": {a["key"]: (a["start"][0], a["start"][1]) for a in spec["agents"]},
        "frame": 0,
        "track_every": 3,
        "nodes": set(),
    }


def apply(lab, story, spec, traces, t, state):
    scene = lab._scene
    keys = [a["key"] for a in spec["agents"]]

    # Fog = the union of what the TEAM has seen. One frame cannot show four
    # private maps; the per-agent rings keep the distinction visible.
    fov_idx = tuple(traces[k].fov_index_at(t) for k in keys)
    if fov_idx != state["snap"]:
        state["snap"] = fov_idx
        seen_any = None
        vis_any = None
        occ_any = None
        for k in keys:
            fov = traces[k].fov_at(t)
            if fov is None:
                continue
            vis, seen = fov
            seen_any = seen if seen_any is None else (seen_any | seen)
            vis_any = vis if vis_any is None else (vis_any | vis)
            occ, _dyn, _ = traces[k].belief_at(t)
            occ_any = occ if occ_any is None else (occ_any | occ)
        if seen_any is not None:
            truth = story.truth_grid()
            for name, mask, color, z in (
                ("fog_seen", seen_any & ~vis_any, fog_scene.FOG_REMEMBERED, 0.28),
                ("fog_now", vis_any, fog_scene.FOG_VISIBLE, 0.30),
                ("ghost", occ_any & ~truth, fog_scene.GHOST, 0.5),
                ("wall", occ_any & truth, fog_scene.WALL, 0.6),
            ):
                v, tri = fog_scene.cells_mesh(mask, story, z, inset=0.0 if "fog" in name else 0.12)
                if v is None:
                    if name in state["nodes"]:
                        scene.geometry_node(name).setVisible(False)
                    continue
                lab.add_mesh(name, v, tri, color)
                fog_scene._style(scene, name, color=color)
                scene.geometry_node(name).setVisible(True)
                state["nodes"].add(name)
            und = truth & ~occ_any
            v, idx = fog_scene.outline_segments(und, story, 0.9)
            if v is not None:
                fog_scene.add_segments(lab, "silhouette", v, idx, fog_scene.SILHOUETTE)
                fog_scene._style(
                    scene, "silhouette", color=fog_scene.SILHOUETTE, mode="lines", line_width=2
                )
                state["nodes"].add("silhouette")
            elif "silhouette" in state["nodes"]:
                scene.geometry_node("silhouette").setVisible(False)

            for k in keys:
                r = traces[k].route_at(t)
                if len(r) >= 2:
                    col = tuple(next(a["color"] for a in spec["agents"] if a["key"] == k))
                    lab.add_path(f"route_{k}", [(float(x), float(y), 1.0) for x, y in r], col)
                    fog_scene._style(scene, f"route_{k}", color=col, mode="lines", line_width=3)

    state["frame"] += 1
    for a in spec["agents"]:
        k = a["key"]
        pose = traces[k].pose_at(t)
        import math

        c, s = math.cos(pose.heading_rad), math.sin(pose.heading_rad)
        scene.geometry_node(f"car_{k}").setTransform(
            [c, -s, 0.0, pose.x, s, c, 0.0, pose.y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )
        ox, oy = state["origins"][k]
        if scene.hasGraphics(f"fov_{k}"):
            scene.geometry_node(f"fov_{k}").setTransform(
                [1, 0, 0, pose.x - ox, 0, 1, 0, pose.y - oy, 0, 0, 1, 0, 0, 0, 0, 1]
            )
        if state["frame"] % state["track_every"] == 0:
            pts = traces[k].track_upto(t)
            if len(pts) >= 2:
                step = max(1, len(pts) // 300)
                col = tuple(a["color"])
                lab.add_path(f"track_{k}", [(float(x), float(y), 1.1) for x, y in pts[::step]], col)
                fog_scene._style(scene, f"track_{k}", color=col, mode="lines", line_width=3)


def capture_squad(
    bundle: str | Path,
    out: str | Path,
    story,
    *,
    fps: int = 20,
    speed: float = 0.5,
    size: tuple[int, int] = (960, 540),
    render_size: tuple[int, int] = (1280, 720),
    captions: bool = True,
) -> Path:
    from pycvc_gl.lab import Lab

    bundle = Path(bundle)
    spec, traces = _load(bundle)
    out = Path(out)
    lab = Lab()
    state = build(lab, story, spec, traces)
    renderer = open_renderer(lab, render_size[0], render_size[1])
    if renderer is None:
        raise RuntimeError("squad capture needs cvcGL.SceneRenderer")

    any_trace = next(iter(traces.values()))
    clock = WorldClock(fixed_dt=any_trace.fixed_dt, mode="replay")
    n_frames = max(1, int(any_trace.duration_s / max(speed, 1e-9) * fps))
    caps = any_trace.scaled_captions(speed) if captions else None

    first = renderer.frameRGB()
    crop = plate_crop_rgb(first, renderer.frameWidth(), renderer.frameHeight())
    enc = open_encoder(
        out,
        fps=fps,
        frame_size=(renderer.frameWidth(), renderer.frameHeight()),
        size=size,
        crop=crop,
        captions=caps,
    )
    try:
        for k in range(n_frames):
            t = k / fps * speed
            clock.seek_time(t)
            apply(lab, story, spec, traces, t, state)
            lab.pump()
            enc.stdin.write(renderer.frameRGB())
    finally:
        enc.stdin.close()
        enc.wait()
        renderer.close()
    shutil.rmtree(bundle / "_frames", ignore_errors=True)
    return out


_ = (np, _filters)  # re-exported helpers kept importable for tests
