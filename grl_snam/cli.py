"""``grl-snam`` — the unified command-line interface.

One entry point for every workflow, from the world model to a running demo:

\b
  grl-snam selftest                      numerical correctness check (no data)
  grl-snam obstacles   BUNDLE            world model -> circular obstacles (.npz)
  grl-snam build-sdf   BUNDLE            world model -> navigation SDF (.npz)
  grl-snam material-raster BUNDLE        satellite -> material.json (id grid + grip/risk)
  grl-snam austin-material-routes BUNDLE pinned public Austin bundle -> routes.json
  grl-snam austin-material-train ...     train belief-material heads on pinned routes
  grl-snam austin-material-evaluate ...  four-arm held-out material evaluation
  grl-snam train       SDF.npz           self-supervised SDF-coefficient training
  grl-snam arms-eval   --arm N=P ...     trained / untrained / constant on paired seeds
  grl-snam learn-report HIST...          did it learn? claims, each pass or fail
  grl-snam capture drive|multigoal ...   drive the policy -> mp4 (offscreen, HUD)
  grl-snam pipeline    BUNDLE            world model -> SDF -> train -> video (all)
  grl-snam fog list|record|capture|play|all   fog-of-war demo: record -> replay
  grl-snam coph crossing|hazard --out DIR   CoPH-Stagewise cooperative demos -> results.json
  grl-snam finale     BUNDLE            8 vehicles on real city geometry, in 3-D
  grl-snam demo        NAME              run a demo live (standalone window; --volrover3 to embed)
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


@main.command("material-demo")
@click.option("--out-dir", default="material_demo", show_default=True, help="Output directory.")
@click.option(
    "--plant",
    default=None,
    help="run both stories on libcvc's sim_world and this plant: kinematic, dynamic[:preset] "
    "or jolt[:preset] (RoboRacers by default)",
)
@click.option("--mu", "ground_mu", default=0.9, show_default=True, help="--plant: the grip")
@click.option(
    "--governor/--no-governor",
    default=None,
    help="--plant: libcvc's risk governor, so the material's risk slows the vehicles (default: on "
    "for the dynamic and Jolt plants where the running pycvc has it); --no-governor shows the "
    "forces alone",
)
def material_demo(out_dir: str, plant, ground_mu, governor) -> None:
    """Material-aware navigation demo (pure Python): two stories run with and
    without a MaterialGrid, rendered as top-down trajectory PNGs + stats.
    Set GRL_SNAM_MATERIAL_BACKEND=native to run through the C++ kernels."""
    from grl_snam.tools.material_demo import main as run

    raise SystemExit(run(out_dir, plant=plant, ground_mu=ground_mu, governor=governor))


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


@main.command("material-raster")
@click.argument("bundle", required=False, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--satellite",
    type=click.Path(exists=True, dir_okay=False),
    help="orthophoto to segment (default: <BUNDLE>/satellite.png)",
)
@click.option(
    "--rows", type=int, default=None, help="grid rows (default: from <BUNDLE>/terrain.json)"
)
@click.option(
    "--cols", type=int, default=None, help="grid cols (default: from <BUNDLE>/terrain.json)"
)
@click.option(
    "--bounds",
    nargs=4,
    type=float,
    default=None,
    metavar="MIN_X MIN_Y MAX_X MAX_Y",
    help="world extent (default: from terrain.json)",
)
@click.option(
    "-o", "--out", default=None, help="output material.json (default: <BUNDLE>/material.json)"
)
@click.option("--preview", default=None, help="also write a color-coded tag-map PNG for review")
def material_raster(
    bundle: str | None,
    satellite: str | None,
    rows: int | None,
    cols: int | None,
    bounds,
    out: str | None,
    preview: str | None,
) -> None:
    """Satellite orthophoto -> scene material.json (material-id grid + grip/risk).

    \b
    Classifies a north-up orthophoto into the shared 13-class material palette on the world grid and
    derives per-material grip (mu) + terrain risk — the material input the nav-stats scorecard buckets
    time by and the drive reads for grip/reroute. Give a scene BUNDLE (a dir with terrain.json +
    satellite.png) to take the grid + bounds + image from it, or drop BUNDLE and pass --satellite +
    --rows + --cols + --bounds to segment any other satellite data.
    """
    import json as _json
    import os as _os

    from .tools import material_raster as _mr

    terr = _json.load(open(_os.path.join(bundle, "terrain.json"))) if bundle else None
    sat = satellite or (_os.path.join(bundle, "satellite.png") if bundle else None)
    R = rows if rows is not None else (int(terr["rows"]) if terr else None)
    C = cols if cols is not None else (int(terr["cols"]) if terr else None)
    bnds = bounds if bounds else (terr["bounds"] if terr else None)
    op = out or (_os.path.join(bundle, "material.json") if bundle else None)
    if not (sat and R and C and bnds and op):
        raise click.UsageError(
            "provide a BUNDLE (terrain.json + satellite.png), or --satellite + --rows + --cols + "
            "--bounds + --out"
        )
    dist = _mr.generate(sat, R, C, bnds, op, preview)
    click.echo(f"wrote {op}  ({R}x{C} grid)")
    for name, frac in dist.items():
        click.echo(f"  {name:20s} {100 * frac:5.1f}%")


@main.command("austin-material-routes")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--out", required=True, type=click.Path(dir_okay=False))
@click.option("--seed", default=20261015, show_default=True, type=int)
@click.option("--train", "train_count", default=12, show_default=True, type=int)
@click.option("--validation", "validation_count", default=6, show_default=True, type=int)
@click.option("--test", "test_count", default=30, show_default=True, type=int)
@click.option("--hard-train", default=2, show_default=True, type=int)
@click.option("--hard-validation", default=2, show_default=True, type=int)
@click.option("--hard-test", default=6, show_default=True, type=int)
def austin_material_routes(
    bundle: str,
    out: str,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    hard_train: int,
    hard_validation: int,
    hard_test: int,
) -> None:
    """Generate split-locked routes from the pinned public Austin bundle."""
    from .material_routes import write_material_route_manifest
    from .tools.austin_material_routes import generate_austin_material_route_manifest

    counts = {"train": train_count, "validation": validation_count, "test": test_count}
    hard_counts = {
        "train": hard_train,
        "validation": hard_validation,
        "test": hard_test,
    }
    manifest = generate_austin_material_route_manifest(
        bundle,
        counts=counts,
        hard_counts=hard_counts,
        seed=seed,
    )
    write_material_route_manifest(out, manifest)
    click.echo(
        f"wrote {out}  "
        f"(train={train_count}, validation={validation_count}, test={test_count}; "
        f"hard={hard_train}/{hard_validation}/{hard_test})"
    )


@main.command(
    "austin-material-train",
    context_settings=dict(ignore_unknown_options=True, help_option_names=[]),
    add_help_option=False,
    short_help="Train belief-material heads from a public geometry checkpoint.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def austin_material_train(args) -> None:
    """Warm-start geometry and train on the pinned Austin train split."""
    from .tools.austin_material_training import main as run

    raise SystemExit(run(list(args)))


@main.command(
    "austin-material-evaluate",
    context_settings=dict(ignore_unknown_options=True, help_option_names=[]),
    add_help_option=False,
    short_help="Evaluate fixed-lambda, force-only, waypoint, and learned material arms.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def austin_material_evaluate(args) -> None:
    """Run the pinned, sensor-limited Austin protocol; all flags pass to the evaluator."""
    from .tools.austin_material_evaluation import main as run

    raise SystemExit(run(list(args)))


# ── training ─────────────────────────────────────────────────────────────────
@main.command()
@click.argument("sdf_npz", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default="coef_sdf.pt", show_default=True)
@click.option("--steps", default=1500, show_default=True)
@click.option("--batch", default=128, show_default=True)
@click.option("--lr", default=3e-4, show_default=True)
@click.option("--threads", default=6, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--device", default=None, help="cpu or cuda (default: cuda when present)")
@click.option("--history", default=None, help="write the learning history (JSON lines) here")
@click.option(
    "--checkpoint-every",
    default=0,
    show_default=True,
    help="save <out>.stepNNNNNN.pt every N steps",
)
@click.option(
    "--val-frac",
    default=0.0,
    show_default=True,
    help="hold out this fraction of the region's blocks; score held-out pairs at each checkpoint "
    "and keep <out>.best_heldout.pt on held-out reach (0 = off)",
)
@click.option("--val-seed", default=0, show_default=True, help="--val-frac: the split's seed")
@click.option("--val-blocks", default=8, show_default=True, help="--val-frac: blocks per axis")
@click.option("--val-pairs", default=256, show_default=True, help="--val-frac: held-out pairs")
@click.option("--val-ticks", default=200, show_default=True, help="--val-frac: drive ticks")
def train(
    sdf_npz: str,
    out: str,
    steps: int,
    batch: int,
    lr: float,
    threads: int,
    seed: int,
    device,
    history,
    checkpoint_every: int,
    val_frac: float,
    val_seed: int,
    val_blocks: int,
    val_pairs: int,
    val_ticks: int,
) -> None:
    """Self-supervised training of the SDF navigation coefficients from a nav_sdf.npz."""
    from .tools import train as _train

    _train.train_sdf(
        sdf_npz, out, steps=steps, batch=batch, lr=lr, threads=threads, seed=seed, device=device,
        history=history, checkpoint_every=checkpoint_every, val_frac=val_frac, val_seed=val_seed,
        val_blocks=val_blocks, val_pairs=val_pairs, val_ticks=val_ticks,
    )  # fmt: skip


@main.command(
    "coef-train",
    # Pass every arg straight through to the tool's own argparse (the single source of truth for
    # the ~18 flags) instead of re-declaring them here and risking drift. Disabling Click's -h/--help
    # lets `grl-snam coef-train --help` reach argparse's full flag listing.
    context_settings=dict(ignore_unknown_options=True, help_option_names=[]),
    add_help_option=False,
    short_help="Train a nav CoefMLP (geometry / grip / risk+lam) and export a deployable .cvcnav.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def coef_train_cmd(args) -> None:
    """Train the coefficient net (torch) and export the versioned .cvcnav a pure-C++ host loads via
    coef_mlp::load — the DEPLOYABLE nav policy. Geometry by default; grip/risk+learned-lam with
    `--rollout bicycle --w-risk <w> --learned-lam` (a risk net deploys through the material drive:
    --grip + --material on the host). Thin wrapper over grl_snam.tools.coef_train: all flags pass
    straight through, so `grl-snam coef-train --help` prints the full list (--rollout / --steps /
    --w-risk / --lam-soft / --learned-lam / --curriculum / --cuda / --out / --seed / --score ...).
    """
    from .tools.coef_train import main as _run

    raise SystemExit(_run(list(args)))


@main.command(
    "arms-eval",
    context_settings=dict(ignore_unknown_options=True, help_option_names=[]),
    add_help_option=False,
    short_help="Score policies (trained, untrained, constant) on paired held-out seeds.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def arms_eval_cmd(args) -> None:
    """Score each --arm NAME=PATH|const:... on the same held-out seeds through coef_eval and
    scorecard_eval, and write the grl_snam.arms/v1 table (per seed, mean, CI, paired difference).
    All flags pass to grl_snam.tools.arms_eval; `grl-snam arms-eval --help` lists them."""
    from .tools.arms_eval import main as _run

    raise SystemExit(_run(list(args)))


@main.command(
    "learn-report",
    context_settings=dict(ignore_unknown_options=True, help_option_names=[]),
    add_help_option=False,
    short_help="Did the learner learn? Claims from histories and arms, each pass or fail.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def learn_report_cmd(args) -> None:
    """HIST... [--arms A.json] --json R.json: the held-out loss, selection, task-metric and
    planted-latent claims, each marked pass or fail (grl_snam.learning.report). All flags pass to
    it; `grl-snam learn-report --help` lists them."""
    from .learning.report import main as _run

    raise SystemExit(_run(list(args)))


@main.command(
    "ariadne",
    # Every arg passes straight through to grl_snam.ariadne's own parsers, as coef-train does, so
    # `grl-snam ariadne --list` and `grl-snam ariadne <port> --help` reach them.
    context_settings=dict(ignore_unknown_options=True, help_option_names=[]),
    add_help_option=False,
    short_help="Run a demo as a Python host of an Ariadne document (live/fog/squad/fleet/...).",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def ariadne_cmd(args) -> None:
    """Run one of GRL-SNAM's demos as a Python host of an Ariadne (.ari) document: `grl-snam
    ariadne --list` names the ports (live, fog, squad, fleet, material), `grl-snam ariadne <port>
    --help` explains one. The same as `python -m grl_snam.ariadne`; needs pycvc_gl's Python host
    (pycvc_gl.ariadne). See grl_snam/ariadne/README.md.
    """
    from .ariadne.__main__ import main as _run

    raise SystemExit(_run(list(args)))


# ── drive + render ───────────────────────────────────────────────────────────
@main.group()
def capture() -> None:
    """Drive the learned policy and render an mp4 OFFSCREEN (no window), with a live HUD."""


def _plant_options(max_steps: int):
    """The plant options `capture drive` and `capture multigoal` share."""

    def deco(f):
        for opt in reversed(
            [
                click.option(
                    "--plant",
                    type=click.Choice(["point", "kinematic", "dynamic", "jolt"]),
                    default="point",
                    show_default=True,
                    help="what drives: the Python point-mass surrogate, or the policy on "
                    "libcvc's sim_world driving nav's kinematic bicycle, the dynamic "
                    "single-track vehicle or a Jolt vehicle (needs a pycvc with the plants)",
                ),
                click.option(
                    "--vehicle",
                    type=click.Choice(["humvee", "roboracer"]),
                    default="humvee",
                    show_default=True,
                    help="dynamic / jolt: the vehicle",
                ),
                click.option(
                    "--mu",
                    "ground_mu",
                    default=0.9,
                    show_default=True,
                    help="dynamic / jolt: the ground's grip (0.3 is wet ice-ish)",
                ),
                click.option(
                    "--routes",
                    is_flag=True,
                    help="dynamic / jolt: follow libcvc's vehicle routes instead of the "
                    "navigator's carrot",
                ),
                click.option(
                    "--fog",
                    is_flag=True,
                    help="plants: the map starts unknown; the vehicle maps the buildings "
                    "with its sensor as it drives (they are in the physics from the start)",
                ),
                click.option(
                    "--weights",
                    default=None,
                    type=click.Path(exists=True, dir_okay=False),
                    help="plants: drive this .cvcnav (e.g. trained through Jolt) instead of "
                    "CHECKPOINT's policy; CHECKPOINT still frames the run",
                ),
                click.option(
                    "--reach-m",
                    default=15.0,
                    show_default=True,
                    help="plants: arrival radius in metres",
                ),
                click.option(
                    "--max-steps",
                    default=max_steps,
                    show_default=True,
                    help="ticks per leg before giving up",
                ),
                click.option(
                    "--speed",
                    default=1.0,
                    show_default=True,
                    help="plants: world seconds per played second (raised to fit --minutes)",
                ),
                click.option("--caption", default=None, help="a line drawn along the bottom"),
            ]
        ):
            f = opt(f)
        return f

    return deco


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
@click.option("--cam", type=click.Choice(["chase", "drone"]), default="chase", show_default=True)
@_plant_options(max_steps=3000)
def capture_drive_cmd(bundle, checkpoint, start, goal, sdf_npz, out, minutes, no_hud, cam, **plant):
    """A single fixed A->B run. On a plant (--plant kinematic|dynamic|jolt) the whole run is
    kept and played by the clock, the HUD adds the vehicle's slip, tire use, lateral g and
    energy books, and OUT's .json records how it ended."""
    from .tools import capture as _cap

    _cap.capture_drive(
        bundle,
        checkpoint,
        start,
        goal,
        out,
        sdf_npz=sdf_npz,
        minutes=minutes,
        hud=not no_hud,
        cam=cam,
        **plant,
    )


@capture.command("multigoal")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("--sdf", "sdf_npz", default=None, help="prebuilt nav_sdf.npz")
@click.option("-o", "--out", default="multigoal.mp4", show_default=True)
@click.option("--minutes", default=3.0, show_default=True)
@click.option("--no-hud", is_flag=True, help="disable the metrics HUD overlay")
@click.option("--goals", "n_goals", default=4, show_default=True, help="how many goals (1-4)")
@_plant_options(max_steps=1300)
def capture_multigoal_cmd(bundle, checkpoint, sdf_npz, out, minutes, no_hud, n_goals, **plant):
    """The dynamic multi-goal free-drive (goals re-targeted live, drone chase cam), on
    --plant (see `capture drive`)."""
    from .tools import capture as _cap

    _cap.capture_multigoal(
        bundle,
        checkpoint,
        out,
        sdf_npz=sdf_npz,
        minutes=minutes,
        hud=not no_hud,
        n_goals=n_goals,
        **plant,
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
@click.argument(
    "story", type=click.Choice(["ghost", "blocker", "unit", "city", "traffic", "pursuit"])
)
@click.option("-o", "--out", "out_dir", default=None, help="trace dir (default traces/<story>)")
@click.option("--seed", default=0, show_default=True)
@click.option("--max-steps", default=None, type=int)
@click.option(
    "--plant",
    default=None,
    help="drive the story on libcvc's sim_world and this plant: kinematic, dynamic[:preset] "
    "or jolt[:preset] (RoboRacers by default). Default: the Python scenario",
)
@click.option("--mu", "ground_mu", default=0.9, show_default=True, help="--plant: the grip")
def fog_record_cmd(story, out_dir, seed, max_steps, plant, ground_mu) -> None:
    """Run the scenario once and write the measured trace bundle."""
    from .tools import fog_record

    out = fog_record.record(
        story, out_dir, seed=seed, max_steps=max_steps, plant=plant, ground_mu=ground_mu
    )
    click.echo(f"trace={out}")


@fog.command("capture")
@click.argument(
    "story", type=click.Choice(["ghost", "blocker", "unit", "city", "traffic", "pursuit"])
)
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
@click.argument(
    "story", type=click.Choice(["ghost", "blocker", "unit", "city", "traffic", "pursuit"])
)
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
@click.option("--plant", default=None, help="record every story on this plant (see fog record)")
@click.option("--mu", "ground_mu", default=0.9, show_default=True, help="--plant: the grip")
def fog_all_cmd(out_dir, fps, speed, plant, ground_mu) -> None:
    """Record and capture every story, then concatenate the reel."""
    from .fog_stories import STORIES
    from .tools import fog_capture, fog_record

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp4s = []
    for key in STORIES:
        trace = fog_record.record(key, out / "traces" / key, plant=plant, ground_mu=ground_mu)
        mp4s.append(fog_capture.capture(key, trace, out / f"fog_{key}.mp4", fps=fps, speed=speed))
        click.echo(f"  {key} -> {mp4s[-1]}")
    reel = fog_capture.concat(mp4s, out / "fog_demo.mp4")
    click.echo(f"reel={reel}")


# ── CoPH-Stagewise (vendored in grl_snam.coph) ───────────────────────────────
@main.group()
def coph() -> None:
    """CoPH-Stagewise's cooperative demos, vendored from CoPH-Sense (grl_snam/coph/VENDORED.md).

    A run writes its episodes, a manifest with the oracle predictions frozen before
    any episode ran, results.json and REPORT.md into --out, which must not exist yet.
    The full demos take 20-30 min on a many-core box; --seeds 1 --steps 120 is a smoke.
    """


def _coph_options(default_seeds: int, default_steps: int, default_workers: int):
    def deco(f):
        for opt in reversed(
            [
                click.option("--out", required=True, type=click.Path(file_okay=False)),
                click.option(
                    "--seeds",
                    default=default_seeds,
                    show_default=True,
                    type=click.IntRange(min=1),
                    help="number of noise seeds, counted from the demo's first seed",
                ),
                click.option(
                    "--steps",
                    default=default_steps,
                    show_default=True,
                    type=click.IntRange(min=1),
                    help="horizon in 0.12 s control steps",
                ),
                click.option("--workers", default=default_workers, show_default=True),
                click.option(
                    "--summary-only",
                    is_flag=True,
                    help="only rewrite results.json and REPORT.md from an existing --out",
                ),
            ]
        ):
            f = opt(f)
        return f

    return deco


def _coph_threads() -> None:
    # Upstream's reproduction recipe: one BLAS/OpenMP thread per worker process (each worker
    # also calls torch.set_num_threads(1)). Set before the workers import numpy and torch.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


@coph.command("crossing")
@_coph_options(default_seeds=2, default_steps=2500, default_workers=8)
def coph_crossing(out, seeds, steps, workers, summary_only) -> None:
    """Crossing traverses: opposite starts, swapped goals, two hidden doors; 4 arms per seed."""
    _coph_threads()
    from .coph import crossing_demo

    first = crossing_demo.SEEDS[0]
    argv = ["--out", out, "--steps", str(steps), "--workers", str(workers)]
    argv += ["--seeds", *(str(first + k) for k in range(seeds))]
    r = crossing_demo.main(argv + (["--render-only"] if summary_only else []))
    click.echo(f"goals {r['goals']}/{r['missions']}, collisions {r['collisions']}")
    for seed, e in r["per_seed"].items():
        click.echo(
            f"  seed {seed}: one-way saving {e.get('one_way_saving', float('nan')):.1f} m, "
            f"reciprocal saving {e.get('reciprocal_saving', float('nan')):.1f} m"
        )
    click.echo(f"results={Path(out) / 'results.json'}")


@coph.command("hazard")
@_coph_options(default_seeds=1, default_steps=1800, default_workers=12)
def coph_hazard(out, seeds, steps, workers, summary_only) -> None:
    """Graded hazards: a risk-tolerant scout and a risk-averse carrier; 4 scenes x 8 arms."""
    _coph_threads()
    from .coph import hazard_demo

    argv = ["--out", out]
    if summary_only:
        r = hazard_demo.main(argv + ["--render"])
    else:
        argv += ["--steps", str(steps), "--workers", str(workers)]
        # 79000 is upstream's hazard seed (hazard_demo.specification's default).
        r = hazard_demo.main(argv + ["--seeds", *(str(79000 + k) for k in range(seeds))])
    click.echo(f"goals {r['goals']}/{r['missions']}, collisions {r['collisions']}")
    click.echo(
        "carrier worst-case excess cost: "
        + ", ".join(f"{a} {v:.1f}" for a, v in r["worst_case_excess"].items())
    )
    click.echo(f"results={Path(out) / 'results.json'}")


# ── the finale ───────────────────────────────────────────────────────────────
@main.command()
@click.argument("bundle", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--out-dir", default="finale", show_default=True)
@click.option("--fps", default=24, show_default=True)
@click.option("--speed", default=6.0, show_default=True, help="world seconds per clip second")
@click.option("--width", default=1600, show_default=True)
@click.option("--record/--no-record", default=True, help="re-record the traces first")
@click.option(
    "--plant",
    default=None,
    help="record on libcvc's sim_world and this plant: kinematic, dynamic[:preset], "
    "jolt[:preset] or a preset (humvee, roboracer). Default: the Python squad",
)
@click.option("--max-steps", default=None, type=int, help="cap each act's ticks (a short run)")
def finale(bundle, out_dir, fps, speed, width, record, plant, max_steps) -> None:
    """Eight vehicles across real city geometry: rendezvous, then pursuit.

    BUNDLE is a scene bundle directory (terrain + buildings). There is no
    default: the geometry lives outside this repository. With --plant the squads
    run natively, so the traces carry no map updates or route spines to draw.
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
            plant=plant,
            max_steps=max_steps,
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
@click.option(
    "--plant",
    type=click.Choice(["point", "kinematic", "dynamic", "jolt"]),
    default="point",
    show_default=True,
    help="what drives in the video (see `grl-snam capture drive --help`)",
)
@click.option(
    "--vehicle", type=click.Choice(["humvee", "roboracer"]), default="humvee", show_default=True
)
@click.option("--mu", "ground_mu", default=0.9, show_default=True, help="dynamic / jolt: grip")
def pipeline(bundle, source, steps, minutes, drive, out_dir, plant, vehicle, ground_mu) -> None:
    """Run the WHOLE pipeline: world model -> SDF -> train -> demo video."""
    from .tools import pipeline as _pipe

    res = _pipe.run(
        bundle,
        source=source,
        steps=steps,
        minutes=minutes,
        drive=drive,
        out_dir=out_dir,
        plant=plant,
        vehicle=vehicle,
        ground_mu=ground_mu,
    )
    click.echo("artifacts: " + ", ".join(f"{k}={v}" for k, v in res.items()))


# ── live-in-VolRover demos ───────────────────────────────────────────────────
@main.command()
@click.argument("name", required=False)
@click.option("--bundle", default=None, help="scene bundle (sets GRL_SNAM_SCENE_BUNDLE)")
@click.option("--checkpoint", default=None, help="trained .pt (sets GRL_SNAM_CHECKPOINT)")
@click.option(
    "--standalone/--volrover3",
    default=True,
    help="standalone pycvc_gl window (default; needs a display) vs. inside VolRover3",
)
@click.option("--width", default=1280, show_default=True, help="standalone window width")
@click.option("--height", default=800, show_default=True, help="standalone window height")
@click.option("--list", "list_", is_flag=True, help="list available demos")
def demo(
    name: str | None,
    bundle: str | None,
    checkpoint: str | None,
    standalone: bool,
    width: int,
    height: int,
    list_: bool,
) -> None:
    """Run a demo live. By default it runs standalone in its own pycvc_gl window
    (no VolRover3 — just `cvcpkg install`ed bindings + a display). Pass --volrover3
    to launch it inside VolRover3 instead (shells out to `volrover3 --run-job`; set
    VOLROVER3_BIN if `volrover3` is not on PATH).

    NAME is one of the registered demos (see `--list`)."""
    from . import demos

    if list_ or not name:
        click.echo("available demos (grl-snam demo NAME):")
        for key, desc in demos.registry().items():
            click.echo(f"  {key:18s} {desc}")
        return

    # Bundle / checkpoint overrides apply to BOTH paths through the environment
    # (the demos read GRL_SNAM_SCENE_BUNDLE / GRL_SNAM_CHECKPOINT at setup()).
    if bundle:
        os.environ["GRL_SNAM_SCENE_BUNDLE"] = bundle
    if checkpoint:
        os.environ["GRL_SNAM_CHECKPOINT"] = checkpoint

    if standalone:
        module = demos.demo_module(name)
        if module is None:
            raise click.ClickException(f"unknown demo {name!r}; try `grl-snam demo --list`")
        # Some demos build their scene through VolRover3's live host (the `vrhost`
        # module — VolRover3's embedded Python and entry point into the running
        # app) and have no standalone path. Stop with a clear pointer to
        # --volrover3 instead of an opaque ImportError from deep inside setup().
        import importlib.util as _ilu

        if getattr(module, "REQUIRES_VOLROVER3", False) and _ilu.find_spec("vrhost") is None:
            raise click.ClickException(
                f"the {name!r} demo runs only inside VolRover3: it needs `vrhost` "
                f"(VolRover3's host module and entry point into the running app), "
                f"which isn't importable here.\n"
                f"Launch it in VolRover3 instead:\n"
                f"    grl-snam demo {name} --volrover3"
            )
        from .demos._common import run_standalone

        click.echo(f"running {name} standalone (pycvc_gl window; no VolRover3) — close it to quit")
        run_standalone(module, width=width, height=height)
        return

    path = demos.demo_path(name)
    if path is None:
        raise click.ClickException(f"unknown demo {name!r}; try `grl-snam demo --list`")
    binary = os.environ.get("VOLROVER3_BIN", "volrover3")
    click.echo(f"launching: {binary} --run-job {path}")
    import subprocess

    raise SystemExit(subprocess.call([binary, "--run-job", path], env=os.environ.copy()))


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
