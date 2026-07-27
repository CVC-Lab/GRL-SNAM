"""``grl-snam`` — the unified command-line interface.

One entry point for every workflow, from the world model to a running demo:

\b
  grl-snam selftest                      numerical correctness check (no data)
  grl-snam obstacles   BUNDLE            world model -> circular obstacles (.npz)
  grl-snam build-sdf   BUNDLE            world model -> navigation SDF (.npz)
  grl-snam train       SDF.npz           self-supervised SDF-coefficient training
  grl-snam capture drive|multigoal ...   drive the policy -> mp4 (offscreen, HUD)
  grl-snam pipeline    BUNDLE            world model -> SDF -> train -> video (all)
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
