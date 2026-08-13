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
    return _crop_from_image(im, pad)


def _crop_from_image(im: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
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


def _filters(
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    captions: list[tuple[float, float, str]] | None,
    *,
    vflip: bool = False,
) -> str:
    filters = []
    if vflip:
        # VTK hands back the framebuffer bottom-up; PNGs come out the right way
        # because the writer flips, raw pixels do not.
        filters.append("vflip")
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
    return ",".join(filters)


def encode(
    frames_dir: str | Path,
    out_mp4: str | Path,
    *,
    fps: int,
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    captions: list[tuple[float, float, str]] | None = None,
) -> Path:
    """ffmpeg a directory of PNGs, cropping to the plate and burning captions."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(Path(frames_dir) / "f_%05d.png"),
        "-vf", _filters(size, crop, captions),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_mp4),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return Path(out_mp4)


def open_encoder(
    out_mp4: str | Path,
    *,
    fps: int,
    frame_size: tuple[int, int],
    size: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    captions: list[tuple[float, float, str]] | None = None,
):
    """An ffmpeg process taking raw RGB frames on stdin.

    Skips a PNG encode per frame here and a PNG decode per frame in ffmpeg, and
    leaves no temp directory of thousands of files behind. Pair with
    ``SceneRenderer.frameRGB()``.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{frame_size[0]}x{frame_size[1]}",
        "-framerate", str(fps),
        "-i", "-",
        "-vf", _filters(size, crop, captions, vflip=True),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_mp4),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def plate_crop_rgb(
    buf: bytes, width: int, height: int, *, pad: int = 2
) -> tuple[int, int, int, int] | None:
    """plate_crop for a raw bottom-up RGB frame, so the calibration frame does
    not have to become a PNG just to be measured."""
    a = np.frombuffer(buf, np.uint8)
    if a.size != width * height * 3:
        return None
    im = a.reshape(height, width, 3)[::-1].astype(np.int16)  # bottom-up -> top-down
    return _crop_from_image(im, pad)


def open_renderer(lab, width: int, height: int, *, offscreen: bool = True):
    """A render target for `lab`'s scene, persistent where the bindings allow.

    ``cvcGL.SceneRenderer`` holds one GL context open for the whole sequence;
    the older ``Lab.render_png`` builds and destroys a context per call, which
    measured 631 ms/frame against 31 ms — a 20x difference that is the gap
    between "render overnight" and "play it live".

    Returns ``None`` when the installed bindings predate SceneRenderer, and the
    caller falls back. Kept as a capability check rather than a version check
    because the deployed pycvc-gl column lags the source.
    """
    SceneRenderer = getattr(_pycvc_gl(), "SceneRenderer", None)
    if SceneRenderer is None:
        return None
    return SceneRenderer(lab._scene, width, height, offscreen)


def _pycvc_gl():
    import pycvc_gl

    return pycvc_gl


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

    renderer = open_renderer(lab, render_size[0], render_size[1])
    n_frames = max(1, int(trace.duration_s / max(speed, 1e-9) * fps))
    caps = trace.scaled_captions(speed) if captions else None

    def step(k: int) -> None:
        t = k / fps * speed
        clock.seek_time(t)
        fog_scene.apply(lab, story, trace, t, state)
        lab.pump()
        if progress and k % 20 == 0:
            progress(k, n_frames)

    # Fast path: render straight into the encoder's stdin. No PNG per frame on
    # this side, no PNG decode on ffmpeg's, and no temp directory at all.
    if renderer is not None and not keep_frames:
        first = renderer.frameRGB()  # frame 0 doubles as the crop calibration
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
            step(0)
            enc.stdin.write(renderer.frameRGB())
            for k in range(1, n_frames):
                step(k)
                enc.stdin.write(renderer.frameRGB())
        finally:
            enc.stdin.close()
            enc.wait()
            renderer.close()
        shutil.rmtree(frames, ignore_errors=True)
        return out

    # PNG path: the fallback for bindings without SceneRenderer, and what
    # --keep-frames uses when someone wants the stills.
    crop = None
    try:
        for k in range(n_frames):
            step(k)
            path = frames / f"f_{k:05d}.png"
            if renderer is not None:
                # ResetCamera runs once, at construction; from here the framing
                # is ours and stays put frame to frame.
                renderer.writePNG(str(path))
            else:
                lab.render_png(str(path), render_size[0], render_size[1])
            if k == 0:
                crop = plate_crop(path)
    finally:
        if renderer is not None:
            # Release the GL context while the scene is still alive; leaving it
            # to interpreter shutdown segfaults on some offscreen backends.
            renderer.close()

    encode(frames, out, fps=fps, size=size, crop=crop, captions=caps)
    if not keep_frames:
        shutil.rmtree(frames, ignore_errors=True)
    return out


def play(
    story_key: str,
    trace_dir: str | Path | None = None,
    *,
    speed: float = 0.5,
    size: tuple[int, int] = (1280, 720),
    loop: bool = False,
    max_seconds: float | None = None,
) -> int:
    """Play a recorded story in a window, paced by the world clock.

    The same scene builder and the same trace the capture uses — a preview that
    renders a *different* picture from the recording is worse than no preview.
    Needs SceneRenderer: the older path can only produce a blocking window that
    owns the event loop, which cannot be stepped by a clock.

    Returns the number of frames drawn.
    """
    import time

    from pycvc_gl.lab import Lab

    story = STORIES[story_key]
    trace = Trace.load(trace_dir or Path("traces") / story_key)
    lab = Lab()
    state = fog_scene.build(lab, story, trace)

    renderer = open_renderer(lab, size[0], size[1], offscreen=False)
    if renderer is None:
        raise RuntimeError(
            "live playback needs cvcGL.SceneRenderer (transfix/libcvc#184); "
            "the installed pycvc_gl only has the one-shot render_png"
        )

    # live, not replay: this clock is driven by real elapsed time, and `scale`
    # is the demo's slow-motion knob.
    clock = WorldClock(fixed_dt=trace.fixed_dt, scale=speed, mode="live")
    frames = 0
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            wall_dt, last = now - last, now
            step = clock.advance(wall_dt)
            # alpha carries the sub-quantum remainder, so motion stays smooth
            # even when the frame rate is not a multiple of the sim rate.
            t = clock.t() + step.alpha * clock.fixed_dt
            if t >= trace.duration_s:
                if not loop:
                    break
                clock.reset()
                t = 0.0
            fog_scene.apply(lab, story, trace, t, state)
            lab.pump()
            renderer.render()
            renderer.processUIEvents()
            frames += 1
            if renderer.windowClosed():
                break
            if max_seconds is not None and (time.monotonic() - (now - wall_dt)) > max_seconds:
                break
    finally:
        renderer.close()
    return frames


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
