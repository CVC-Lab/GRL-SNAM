"""``grl-snam`` — the unified command-line interface.

One entry point for every workflow, from the world model to a running demo:

\b
  grl-snam selftest                      numerical correctness check (no data)
  grl-snam obstacles   BUNDLE            world model -> circular obstacles (.npz)
  grl-snam build-sdf   BUNDLE            world model -> navigation SDF (.npz)
  grl-snam train       SDF.npz           self-supervised SDF-coefficient training
  grl-snam capture drive|multigoal ...   drive the policy -> mp4 (offscreen, HUD)
  grl-snam pipeline    BUNDLE            world model -> SDF -> train -> video (all)
  grl-snam fog list|record|capture|play|all   fog-of-war demo: record -> replay
  grl-snam demo        NAME              run a demo live inside VolRover3
  grl-snam lab-demo    [PNG]             standalone lab visualization
  grl-snam eval        [args...]         CoefEnergyNet visual eval (legacy trainer)
  grl-snam train-coef  [args...]         CoefEnergyNet dataset training (legacy)

The heavy work lives in :mod:`grl_snam.tools`; the volrover demos in
:mod:`grl_snam.demos`. Environment-specific deps (``pycvc_gl``, VTK, ``vrhost``) load
lazily, so ``grl-snam --help`` works anywhere.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import click

from . import __version__


@click.group(help=__doc__, context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(version=__version__, prog_name="grl-snam")
def main() -> None:
    pass


# ── correctness + standalone viz ─────────────────────────────────────────────
@main.command()
def selftest() -> None:
    """Exercise the surrogate navigation dynamics (goal-seeking / finiteness / determinism)."""
    from .selftest import main as _selftest

    raise SystemExit(_selftest())


@main.command("lab-demo")
@click.argument("png", required=False)
def lab_demo(png: str | None) -> None:
    """Standalone lab demo (terrain + agent track + marker). PNG => offscreen snapshot."""
    from .demos import lab

    raise SystemExit(lab.main([png] if png else []))


# ── world model -> obstacles / SDF ───────────────────────────────────────────
@main.command()
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--out", default=None, help="output .npz (default <bundle>/obstacles.npz)")
@click.option("--grid", default=512, show_default=True, help="occupancy raster resolution")
@click.option(
    "--block", default=2, show_default=True, help="obstacle coarsening (finer=smaller circles)"
)
def obstacles(bundle: str, out: str | None, grid: int, block: int) -> None:
    """World model (terrain + buildings) -> circular obstacle set (.npz)."""
    from .tools import obstacles as _obs

    _obs.extract_to_npz(bundle, out, grid=grid, block=block)


@main.command("build-sdf")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.option("--source", type=click.Choice(["edt", "cvc"]), default="edt", show_default=True)
@click.option(
    "--region", default=430.0, show_default=True, help="working-region half-extent (world)"
)
@click.option("--grid", default=512, show_default=True, help="2-D field resolution")
@click.option("-o", "--out", default=None, help="output .npz (default <bundle>/nav_sdf.npz)")
def build_sdf(bundle: str, source: str, region: float, grid: int, out: str | None) -> None:
    """World model -> navigation SDF (.npz). --source cvc uses CVC's mesh-exact 3-D SDF."""
    from .tools import sdf as _sdf

    _sdf.build(bundle, source=source, region=region, grid=grid, out=out)


# ── training ─────────────────────────────────────────────────────────────────
@main.command()
@click.argument("sdf_npz", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default="coef_sdf.pt", show_default=True)
@click.option("--steps", default=1500, show_default=True)
@click.option("--batch", default=128, show_default=True)
@click.option("--lr", default=3e-4, show_default=True)
@click.option("--threads", default=6, show_default=True)
@click.option("--seed", default=0, show_default=True)
def train(
    sdf_npz: str, out: str, steps: int, batch: int, lr: float, threads: int, seed: int
) -> None:
    """Self-supervised training of the SDF navigation coefficients from a nav_sdf.npz."""
    from .tools import train as _train

    _train.train_sdf(sdf_npz, out, steps=steps, batch=batch, lr=lr, threads=threads, seed=seed)


# ── drive + render ───────────────────────────────────────────────────────────
@main.group()
def capture() -> None:
    """Drive the learned policy and render an mp4 OFFSCREEN (no window), with a live HUD."""


@capture.command("drive")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("--start", nargs=2, type=float, required=True)
@click.option("--goal", nargs=2, type=float, required=True)
@click.option(
    "--sdf", "sdf_npz", default=None, help="prebuilt nav_sdf.npz (else built from occupancy)"
)
@click.option("-o", "--out", default="drive.mp4", show_default=True)
@click.option("--minutes", default=1.0, show_default=True)
@click.option("--no-hud", is_flag=True, help="disable the metrics HUD overlay")
def capture_drive_cmd(bundle, checkpoint, start, goal, sdf_npz, out, minutes, no_hud) -> None:
    """A single fixed A->B run."""
    from .tools import capture as _cap

    _cap.capture_drive(
        bundle, checkpoint, start, goal, out, sdf_npz=sdf_npz, minutes=minutes, hud=not no_hud
    )


@capture.command("multigoal")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("--sdf", "sdf_npz", default=None, help="prebuilt nav_sdf.npz")
@click.option("-o", "--out", default="multigoal.mp4", show_default=True)
@click.option("--minutes", default=3.0, show_default=True)
@click.option("--no-hud", is_flag=True, help="disable the metrics HUD overlay")
def capture_multigoal_cmd(bundle, checkpoint, sdf_npz, out, minutes, no_hud) -> None:
    """The dynamic multi-goal free-drive (goals re-targeted live, drone chase cam)."""
    from .tools import capture as _cap

    _cap.capture_multigoal(
        bundle, checkpoint, out, sdf_npz=sdf_npz, minutes=minutes, hud=not no_hud
    )


# ── fog of war ───────────────────────────────────────────────────────────────
@main.group()
def fog() -> None:
    """Fog-of-war demo: record a measured trace, then replay it live or to mp4.

    The simulation runs exactly ONCE, in `record`. Everything after that replays
    what was measured -- so the window, the video and the quoted numbers cannot
    disagree with each other or with the run.
    """


@fog.command("list")
def fog_list_cmd() -> None:
    """The available stories, with the summary of any recorded trace."""
    import json

    from .fog_stories import STORIES

    for key, story in STORIES.items():
        line = f"{key:9s} {story.title} - {story.subtitle}"
        manifest = Path("traces") / key / "trace.json"
        if manifest.exists():
            s = json.loads(manifest.read_text()).get("summary", {})
            line += (
                f"\n          recorded: {s.get('steps')} ticks, "
                f"{s.get('world_seconds')} world s, {s.get('map_updates')} map updates, "
                f"{s.get('penetration_steps')} collisions, "
                f"{s.get('detour_peak_m')} m detour peak"
            )
        click.echo(line)


@fog.command("record")
@click.argument("story", type=click.Choice(["ghost", "blocker", "unit"]))
@click.option("-o", "--out", "out_dir", default=None, help="trace dir (default traces/<story>)")
@click.option("--seed", default=0, show_default=True)
@click.option("--max-steps", default=None, type=int)
def fog_record_cmd(story, out_dir, seed, max_steps) -> None:
    """Run the scenario once and write the measured trace bundle."""
    from .tools import fog_record

    out = fog_record.record(story, out_dir, seed=seed, max_steps=max_steps)
    click.echo(f"trace={out}")


@fog.command("capture")
@click.argument("story", type=click.Choice(["ghost", "blocker", "unit"]))
@click.option("--trace", "trace_dir", default=None, help="trace dir (default traces/<story>)")
@click.option("-o", "--out", default=None, help="mp4 (default fog_<story>.mp4)")
@click.option("--fps", default=20, show_default=True)
@click.option("--speed", default=0.5, show_default=True, help="world seconds per played second")
@click.option("--no-captions", is_flag=True)
@click.option("--keep-frames", is_flag=True, help="keep the PNG stills (uses the slower path)")
def fog_capture_cmd(story, trace_dir, out, fps, speed, no_captions, keep_frames) -> None:
    """Replay a recorded trace to an mp4 (offscreen; no window needed)."""
    from .tools import fog_capture

    path = fog_capture.capture(
        story,
        trace_dir,
        out,
        fps=fps,
        speed=speed,
        captions=not no_captions,
        keep_frames=keep_frames,
    )
    click.echo(f"video={path}")


@fog.command("play")
@click.argument("story", type=click.Choice(["ghost", "blocker", "unit"]))
@click.option("--trace", "trace_dir", default=None)
@click.option("--speed", default=0.5, show_default=True)
@click.option("--loop", is_flag=True)
def fog_play_cmd(story, trace_dir, speed, loop) -> None:
    """Play a recorded trace in a window, paced by the world clock."""
    from .tools import fog_capture

    n = fog_capture.play(story, trace_dir, speed=speed, loop=loop)
    click.echo(f"frames={n}")


@fog.command("all")
@click.option("-o", "--out-dir", default="fog", show_default=True)
@click.option("--fps", default=20, show_default=True)
@click.option("--speed", default=0.5, show_default=True)
def fog_all_cmd(out_dir, fps, speed) -> None:
    """Record and capture all three stories, then concatenate the reel."""
    from .fog_stories import STORIES
    from .tools import fog_capture, fog_record

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp4s = []
    for key in STORIES:
        trace = fog_record.record(key, out / "traces" / key)
        mp4s.append(fog_capture.capture(key, trace, out / f"fog_{key}.mp4", fps=fps, speed=speed))
        click.echo(f"  {key} -> {mp4s[-1]}")
    reel = fog_capture.concat(mp4s, out / "fog_demo.mp4")
    click.echo(f"reel={reel}")


# ── the finale ───────────────────────────────────────────────────────────────
@main.command()
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--out-dir", default="finale", show_default=True)
@click.option("--fps", default=24, show_default=True)
@click.option("--speed", default=6.0, show_default=True, help="world seconds per clip second")
@click.option("--width", default=1600, show_default=True)
@click.option("--record/--no-record", default=True, help="re-record the traces first")
def finale(bundle, out_dir, fps, speed, width, record) -> None:
    """Eight vehicles across real city geometry: rendezvous, then pursuit.

    BUNDLE is a scene bundle directory (terrain + buildings). There is no
    default: the geometry lives outside this repository.
    """
    from .tools import finale_capture, finale_record
    from .tools.austin import occupancy

    out = Path(out_dir)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    if record:
        click.echo("recording (this is the slow part) ...")
        finale_record.record_both(
            bundle,
            out / "traces",
            progress=lambda k, n: click.echo(f"  tick {k}") if k % 400 == 0 else None,
        )
    occ, bounds = occupancy(bundle)
    size = (int(width) // 2 * 2, int(width * 9 / 16) // 2 * 2)
    # The pursuit frames its targets with the vehicles, so the closing gap is
    # on screen. The rendezvous does not: its goals are a kilometre away at the
    # start, and framing them would hold the whole map -- and eight specks --
    # for the entire clip. The minimap already answers "where are they going".
    acts = (("finale_rendezvous", False), ("finale_pursuit", True))

    # One camera move across both acts. The elevation/bearing schedule is split
    # in proportion to each act's LENGTH, so it runs continuously in time rather
    # than restarting -- and the camera state is handed from one act to the next
    # so the second picks up exactly where the first left off.
    durs = [finale_capture.act_duration_s(out / "traces" / a) for a, _g in acts]
    total = sum(durs) or 1.0
    edges, acc = [], 0.0
    for d in durs:
        edges.append((acc / total, (acc + d) / total))
        acc += d

    cam_state = None
    for (act, frame_goals), u_range in zip(acts, edges):
        mp4, cam_state = finale_capture.capture_finale(
            out / "traces" / act, bundle, out / f"{act}.mp4",
            fps=fps, speed=speed, size=size, occ=occ, world_bounds=bounds,
            frame_goals=frame_goals, camera_in=cam_state, u_range=u_range,
            progress=lambda f, n, a=act: click.echo(f"  {a} {f}/{n}") if f % 100 == 0 else None,
        )  # fmt: skip
        click.echo(f"  {act} -> {mp4}")


# ── full pipeline ────────────────────────────────────────────────────────────
@main.command()
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.option("--source", type=click.Choice(["edt", "cvc"]), default="edt", show_default=True)
@click.option("--steps", default=1500, show_default=True, help="training steps")
@click.option("--minutes", default=3.0, show_default=True, help="demo video length")
@click.option(
    "--drive", type=click.Choice(["multigoal", "single"]), default="multigoal", show_default=True
)
@click.option("--out-dir", default=None, help="where to write nav_sdf.npz / checkpoint / video")
def pipeline(bundle, source, steps, minutes, drive, out_dir) -> None:
    """Run the WHOLE pipeline: world model -> SDF -> train -> demo video."""
    from .tools import pipeline as _pipe

    res = _pipe.run(
        bundle, source=source, steps=steps, minutes=minutes, drive=drive, out_dir=out_dir
    )
    click.echo("artifacts: " + ", ".join(f"{k}={v}" for k, v in res.items()))


# ── live-in-VolRover demos ───────────────────────────────────────────────────
@main.command()
@click.argument("name", required=False)
@click.option("--bundle", default=None, help="scene bundle (sets GRL_SNAM_SCENE_BUNDLE)")
@click.option("--checkpoint", default=None, help="trained .pt (sets GRL_SNAM_CHECKPOINT)")
@click.option("--list", "list_", is_flag=True, help="list available demos")
def demo(name: str | None, bundle: str | None, checkpoint: str | None, list_: bool) -> None:
    """Run a demo live inside VolRover3 (shells out to `volrover3 --run-job`).

    Set VOLROVER3_BIN if `volrover3` is not on PATH. NAME is one of the registered
    demos (see `--list`)."""
    from . import demos

    if list_ or not name:
        click.echo("available demos (grl-snam demo NAME):")
        for key, desc in demos.registry().items():
            click.echo(f"  {key:18s} {desc}")
        return
    path = demos.demo_path(name)
    if path is None:
        raise click.ClickException(f"unknown demo {name!r}; try `grl-snam demo --list`")
    env = os.environ.copy()
    if bundle:
        env["GRL_SNAM_SCENE_BUNDLE"] = bundle
    if checkpoint:
        env["GRL_SNAM_CHECKPOINT"] = checkpoint
    binary = os.environ.get("VOLROVER3_BIN", "volrover3")
    click.echo(f"launching: {binary} --run-job {path}")
    import subprocess

    raise SystemExit(subprocess.call([binary, "--run-job", path], env=env))


# ── legacy CoefEnergyNet trainer/evaluator (kept; future real-time HUD source) ─
def _forward(module_name: str, args) -> None:
    mod = importlib.import_module(module_name)
    old = sys.argv
    sys.argv = [module_name, *args]
    try:
        mod.main()
    finally:
        sys.argv = old


@main.command("eval", context_settings=dict(ignore_unknown_options=True))
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def eval_cmd(args) -> None:
    """CoefEnergyNet visual eval (online correction/finetune, GIF/MP4). Forwards args to eval_coef_energy."""
    _forward("eval_coef_energy", args)


@main.command("train-coef", context_settings=dict(ignore_unknown_options=True))
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def train_coef_cmd(args) -> None:
    """CoefEnergyNet dataset training. Forwards args to train_coef_energy."""
    _forward("train_coef_energy", args)


@main.command("train-geo", context_settings=dict(ignore_unknown_options=True))
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def train_geo_cmd(args) -> None:
    """Train a CoefEnergyNet directly on a scene's obstacles.npz (circle-obstacle track).
    Forwards args to scripts.train_on_geometry."""
    _forward("scripts.train_on_geometry", args)


if __name__ == "__main__":
    main()
