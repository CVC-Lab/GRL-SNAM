"""The whole learned-nav pipeline in one call: from nothing but a **world model**
(a scene bundle = ``terrain.json`` + ``buildings.glb``) through SDF construction and
self-supervised training to a rendered demo video. This is ``grl-snam pipeline``.
"""

from __future__ import annotations

import os

from . import capture, sdf, train


def run(
    bundle: str,
    *,
    source: str = "edt",
    region: float = 430.0,
    steps: int = 1500,
    out_dir: str | None = None,
    minutes: float = 3.0,
    fps: int = 15,
    hud: bool = True,
    drive: str = "multigoal",
) -> dict:
    """world model -> SDF -> trained policy -> demo video. Returns the artifact paths.

    ``drive`` selects the demo video: ``multigoal`` (dynamic re-targeting) or
    ``single`` (a fixed A->B). Reuses an existing ``nav_sdf.npz`` / checkpoint if
    present in ``out_dir`` unless absent."""
    out_dir = out_dir or bundle
    os.makedirs(out_dir, exist_ok=True)
    sdf_npz = os.path.join(out_dir, "nav_sdf.npz")
    ckpt = os.path.join(out_dir, "coef_sdf.pt")
    video = os.path.join(out_dir, "grl_snam_%s.mp4" % drive)

    print("== [1/3] build SDF from the world model ==")
    sdf.build(bundle, source=source, region=region, out=sdf_npz)
    print("== [2/3] train the navigation policy (self-supervised) ==")
    train.train_sdf(sdf_npz, ckpt, steps=steps)
    print("== [3/3] drive + render the demo video (%s) ==" % drive)
    if drive == "single":
        # a reachable A->B is scene-specific; reuse the multigoal selector to find one
        from ..nav import select_reachable_goals

        field, model, meta = capture._load(bundle, ckpt, sdf_npz)
        start, goals = select_reachable_goals(field, model, meta, n_corners=1)
        capture.capture_drive(
            bundle,
            ckpt,
            tuple(start),
            tuple(goals[0]),
            video,
            sdf_npz=sdf_npz,
            minutes=minutes,
            fps=fps,
            hud=hud,
        )
    else:
        capture.capture_multigoal(
            bundle, ckpt, video, sdf_npz=sdf_npz, minutes=minutes, fps=fps, hud=hud
        )
    print("== pipeline done ==")
    return {"sdf": sdf_npz, "checkpoint": ckpt, "video": video}
