"""Offscreen mp4 of a recorded fog story.

Uses a standalone :class:`pycvc_gl.lab.Lab` and the shared
:mod:`grl_snam.fog_scene`, so it needs no VolRover3 host — the video still
gets made if the live demo is broken the night before.

Framing note: there is no camera binding in the installed ``pycvc_gl``
(``render_png`` calls ``ResetCamera``), so the map is auto-framed and lands
inside a black border. Rather than hard-code a crop, the first frame is used
as a calibration frame: the ground plate is a known solid colour, so its pixel
extent is measured and handed to ffmpeg. That survives any change to the
camera defaults.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from grl_snam import fog_scene
from grl_snam.clock import WorldClock
from grl_snam.fog_stories import STORIES
from grl_snam.fog_trace import Trace

_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font() -> str | None:
    return next((f for f in _FONTS if Path(f).exists()), None)


def plate_crop(png: str | Path, *, pad: int = 2) -> tuple[int, int, int, int] | None:
    """Measure the ground plate's pixel box in a rendered frame.

    Returns ``(w, h, x, y)`` in ffmpeg's crop order, or None if the plate
    could not be found (in which case the caller keeps the full frame).
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    im = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
    target = np.array([int(round(c * 255)) for c in fog_scene.GROUND], np.int16)
    # The plate is flat-shaded but lighting shifts it slightly; match loosely
    # and exclude pure black (the letterbox).
    near = (np.abs(im - target).sum(-1) < 60) & (im.sum(-1) > 20)
    rows, cols = np.nonzero(near)
    if len(rows) < 100:
        return None
    y0, y1 = int(rows.min()) + pad, int(rows.max()) - pad
    x0, x1 = int(cols.min()) + pad, int(cols.max()) - pad
    w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2  # h264 wants even dimensions
    if w < 64 or h < 64:
        return None
    return w, h, x0, y0


def encode(
    frames_dir: str | Path,
    out_mp4: str | Path,
    *,
    fps: int,
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    captions: list[tuple[float, float, str]] | None = None,
) -> Path:
    """ffmpeg the frame sequence, cropping to the plate and burning captions."""
    filters = []
    if crop:
        filters.append("crop={}:{}:{}:{}".format(*crop))
    filters.append(f"scale={size[0]}:{size[1]}:flags=lanczos")
    font = _font()
    for t0, t1, text in captions or []:
        if not font:
            break
        safe = text.replace(":", r"\:").replace("'", "")
        filters.append(
            f"drawtext=fontfile={font}:text='{safe}':fontcolor=white:fontsize=26"
            f":x=(w-text_w)/2:y=h-70:box=1:boxcolor=black@0.6:boxborderw=12"
            f":enable='between(t,{t0:.3f},{t1:.3f})'"
        )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(Path(frames_dir) / "f_%05d.png"),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_mp4),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return Path(out_mp4)


def capture(
    story_key: str,
    trace_dir: str | Path | None = None,
    out: str | Path | None = None,
    *,
    fps: int = 20,
    speed: float = 0.5,
    size: tuple[int, int] = (960, 540),
    render_size: tuple[int, int] = (1920, 1080),
    captions: bool = True,
    keep_frames: bool = False,
    progress=None,
) -> Path:
    """Replay a recorded story to an mp4. Renders nothing that is not measured."""
    from pycvc_gl.lab import Lab  # lazy: needs the compiled bindings

    story = STORIES[story_key]
    trace = Trace.load(trace_dir or Path("traces") / story_key)
    out = Path(out or f"fog_{story_key}.mp4")
    frames = out.parent / f"_frames_{story_key}"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True, exist_ok=True)

    lab = Lab()
    state = fog_scene.build(lab, story, trace)
    # replay: the clock is driven by the frame index, not by wall time, so the
    # video is frame-exact regardless of how long rendering takes.
    clock = WorldClock(fixed_dt=trace.fixed_dt, mode="replay")

    n_frames = max(1, int(trace.duration_s / max(speed, 1e-9) * fps))
    crop = None
    for k in range(n_frames):
        t = k / fps * speed
        clock.seek_time(t)
        fog_scene.apply(lab, story, trace, t, state)
        lab.pump()
        path = frames / f"f_{k:05d}.png"
        lab.render_png(str(path), render_size[0], render_size[1])
        if k == 0:
            crop = plate_crop(path)
        if progress and k % 20 == 0:
            progress(k, n_frames)

    encode(
        frames,
        out,
        fps=fps,
        size=size,
        crop=crop,
        captions=trace.scaled_captions(speed) if captions else None,
    )
    if not keep_frames:
        shutil.rmtree(frames, ignore_errors=True)
    return out


def concat(mp4s: list[str | Path], out: str | Path) -> Path:
    """Join the three stories into the single artifact shown on the day."""
    listing = Path(out).with_suffix(".txt")
    listing.write_text("".join(f"file '{Path(m).resolve()}'\n" for m in mp4s))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)],
        check=True,
    )  # fmt: skip
    listing.unlink(missing_ok=True)
    return Path(out)
