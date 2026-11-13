"""Score policies side by side on paired held-out seeds: the arms of a learning claim.

    python -m grl_snam.tools.arms_eval \\
        --arm trained=run/coef.best.cvcnav \\
        --arm untrained=run/coef.step000000.cvcnav \\
        --arm constant=const:hist=run/history.jsonl \\
        --seeds 20 --out arms.json

A trained policy proves it learned something only against the policies it could be confused
with: the untrained net it started from (a run's step-0 checkpoint) and a constant one (the same
coefficients everywhere, :mod:`grl_snam.learning.arms`). Every arm runs on the same seeds, so
each seed is a matched pair, and the difference that decides is the paired one.

Two task cases, each one unit per seed (``--seed-base`` .. ``+ --seeds``, held out from the
trainers' own draws), the default:

- ``coef``: :func:`grl_snam.tools.coef_eval.evaluate` on the ice-band city, one draw of
  ``--agents`` starts and goals per seed: reach, penetration (clearance below ``-0.5 * rr``),
  mean worst clearance and the final gap to the goal.
- ``scorecard``: :func:`grl_snam.tools.scorecard_eval.episodes`, one city scene per seed, reduced
  to its one-episode scorecard: arrival, success, penetration %, path ratio, time to goal (over
  the arrivers; none when nobody arrived) and stall steps.

Two loss cases (``--cases objective-sdf`` / ``objective-coef``), the trainer's own objective on
its held-out units, so the loss family is tested against the same arms as the task
(:mod:`grl_snam.learning.objective`; every term lower is better):

- ``objective-sdf``: ``grl-snam train``'s terms (loss, goal, col, reg) per held-out pair of
  ``--sdf``, ``--pairs`` pairs drawn with ``--pair-seed`` inside the run's held-out blocks
  (``--split-history``, the run's history, or ``--val-frac``/``--val-seed``/``--val-blocks``).
  The rollout settings are the first ``.pt`` arm's ``meta``.
- ``objective-coef``: coef-train's terms (loss, goal, coll) per held-out start of the city on
  ``--grid``, ``--heldout-agents`` drawn with ``--heldout-seed``, ``--horizon`` steps of
  ``--rollout`` (surrogate or bicycle, with the trainer's ``--w-coll`` and window unless given).

    python -m grl_snam.tools.arms_eval --cases objective-sdf --sdf nav_sdf.npz \\
        --split-history run/history.jsonl \\
        --arm trained=run/coef_sdf.best_heldout.pt --arm untrained=run/coef_sdf.step000000.pt \\
        --arm constant=const:hist=run/history.jsonl --out objective.json

A run with several training seeds is one table per seed (``--label``); ``grl-snam learn-report``
takes them all (``--arms`` repeated) and pools the runs.

The output is ``grl_snam.arms/v1`` (:func:`grl_snam.learning.arms.arms_table`): per metric and
arm, the per-seed values, mean, 95% bootstrap CI and the paired difference to the reference arm
(``--reference``, default the first). ``grl-snam learn-report --arms`` reads it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grl_snam.learning import objective
from grl_snam.learning.arms import ArmSpec, arms_table, load_arm_model, parse_arm

#: Per case: each metric's better direction ("up": higher is better), and the ones claimed on.
DIRECTION = {
    "coef": {"reach": "up", "pen": "down", "clear": "up", "gap": "down"},
    "scorecard": {
        "arrival_rate": "up",
        "success_rate": "up",
        "mean_penetration_pct": "down",
        "mean_path_ratio": "down",
        "mean_time_to_goal_s": "down",
        "mean_stall_steps": "down",
        "veh_contacts_per_run": "down",
    },
    "objective-sdf": {t: "down" for t in objective.SDF_TERMS},
    "objective-coef": {t: "down" for t in objective.COEF_TERMS},
}
PRIMARY = {
    "coef": ["reach", "pen"],
    "scorecard": ["arrival_rate", "mean_penetration_pct"],
    "objective-sdf": list(objective.SDF_TERMS),
    "objective-coef": list(objective.COEF_TERMS),
}
CASES = tuple(DIRECTION)
#: The cases run when ``--cases`` is not given: the task metrics.
DEFAULT_CASES = ("coef", "scorecard")
#: Per case: the claim family (``learn-report``'s), and what one unit is.
FAMILY = {"coef": "task", "scorecard": "task", "objective-sdf": "loss", "objective-coef": "loss"}
UNIT = {"coef": "seed", "scorecard": "seed", "objective-sdf": "pair", "objective-coef": "start"}


def score_coef(models: dict, seeds, *, grid=96, mu=0.18, agents=192, ticks=90) -> dict:
    """``{metric: {arm: {seed: value}}}`` from :func:`coef_eval.evaluate`, one draw per seed."""
    from grl_snam.tools import coef_eval

    field, meta, rand_on, friction = coef_eval.ice_band_world(grid, mu)
    out: dict = {m: {a: {} for a in models} for m in DIRECTION["coef"]}
    for a, model in models.items():
        for s in seeds:
            got = coef_eval.evaluate(
                model, field, meta, rand_on, friction=friction, n=agents, ticks=ticks, seed=s
            )
            for m in DIRECTION["coef"]:
                out[m][a][s] = got[m]
    return out


def score_scorecard(
    models: dict, seeds, *, agents=64, steps=400, grid=96, contact_r=0.0, material=True
) -> dict:
    """``{metric: {arm: {seed: value}}}`` from :func:`scorecard_eval.episodes`, one scene per
    seed, each reduced to its one-episode scorecard."""
    from grl_snam.scorecard import aggregate_nav
    from grl_snam.tools import scorecard_eval

    names = [m for m in DIRECTION["scorecard"] if contact_r > 0 or m != "veh_contacts_per_run"]
    out: dict = {m: {a: {} for a in models} for m in names}
    seeds = list(seeds)
    for a, model in models.items():
        eps = scorecard_eval.episodes(
            model,
            seeds,
            agents=agents,
            steps=steps,
            grid=grid,
            contact_r=contact_r,
            material=material,
        )
        for s, ep in zip(seeds, eps, strict=True):
            card = aggregate_nav([ep]).to_dict()
            for m in names:
                v = card[m]
                # Time to goal is over the arrivers: with none it is undefined, not 0 (the best).
                if m == "mean_time_to_goal_s" and card["arrival_rate"] <= 0.0:
                    v = None
                out[m][a][s] = v
    return out


def checkpoint_meta(arms: list[ArmSpec]) -> dict | None:
    """The rollout settings of the ``.pt`` arms (``grl-snam train`` checkpoints carry them), or
    ``None`` when no arm has them. Arms trained under different settings are refused."""
    import torch

    keys = ("rr", "d_hat", "dt", "vmax", "horizon")
    found = None
    for a in arms:
        if a.kind != "pt":
            continue
        ck = torch.load(a.spec, map_location="cpu")
        meta = ck.get("meta") if isinstance(ck, dict) else None
        if not meta:
            continue
        got = {k: meta[k] for k in keys}
        if found is None:
            found = (a.name, got)
        elif got != found[1]:
            raise ValueError(f"arms {found[0]!r} and {a.name!r} were trained with other settings")
    return None if found is None else found[1]


def score_objective_sdf(
    models: dict,
    *,
    sdf: str,
    meta: dict | None = None,
    split_history: str | None = None,
    val_frac: float | None = None,
    val_seed: int = 0,
    val_blocks: int = 8,
    pairs: int = 1024,
    pair_seed: int = 777,
    notes: dict | None = None,
) -> dict:
    """``{term: {arm: {pair: value}}}``: ``grl-snam train``'s objective per held-out pair."""
    from grl_snam.tools import sdf_eval

    scene = sdf_eval.load_scene(sdf)
    m = objective.sdf_meta(scene, meta)
    blocks, held = objective.sdf_split(
        split_history, val_frac=val_frac, val_seed=val_seed, val_blocks=val_blocks
    )
    starts, goals = objective.sdf_heldout_pairs(scene, blocks, held, pairs, pair_seed, rr=m["rr"])
    out: dict = {t: {} for t in objective.SDF_TERMS}
    for a, model in models.items():
        for t, per in objective.per_unit(
            objective.sdf_terms(model, scene, starts, goals, **m)
        ).items():
            out[t][a] = per
    if notes is not None:
        notes["objective-sdf"] = [
            f"{pairs} held-out pairs, seed {pair_seed}",
            f"held-out blocks {held} of {blocks}x{blocks}",
            "rollout " + ", ".join(f"{k} {v:g}" for k, v in m.items()),
        ]
    return out


def score_objective_coef(
    models: dict,
    *,
    grid: int = 96,
    horizon: int = 28,
    agents: int = 200,
    seed: int = 123,
    rollout: str = "surrogate",
    w_coll: float | None = None,
    window: int | None = None,
    notes: dict | None = None,
) -> dict:
    """``{term: {arm: {start: value}}}``: coef-train's objective per held-out start."""
    scene = objective.coef_scene(grid)
    kw = dict(grid=grid, horizon=horizon, n=agents, seed=seed, rollout=rollout, w_coll=w_coll)
    out: dict = {t: {} for t in objective.COEF_TERMS}
    for a, model in models.items():
        got = objective.coef_terms(model, window=window, scene=scene, **kw)
        for t, per in objective.per_unit(got).items():
            out[t][a] = per
    if notes is not None:
        d = objective.COEF_DEFAULTS[rollout]
        notes["objective-coef"] = [
            f"{agents} held-out starts, seed {seed}, grid {grid}",
            f"{rollout} rollout, horizon {horizon}, window {window or d['window']}, "
            f"w_coll {d['w_coll'] if w_coll is None else w_coll:g}",
        ]
    return out


def load_models(arms: list[ArmSpec]) -> dict:
    """Each arm's policy; a constant arm takes its input layout from the first policy arm."""
    models: dict = {}
    for a in arms:
        if a.kind != "const":
            models[a.name] = load_arm_model(a)
    like = next(iter(models.values()), None)
    for a in arms:
        if a.kind == "const":
            models[a.name] = load_arm_model(a, like=like)
    return {a.name: models[a.name] for a in arms}


def evaluate_arms(arms: list[ArmSpec], seeds, cases=DEFAULT_CASES, notes=None, **kw) -> dict:
    """``{case: {metric: {arm: {unit: value}}}}`` for every arm on every unit. ``kw`` are the
    cases' settings: ``grid``, ``mu``, ``agents``, ``ticks`` (coef); ``scene_agents``,
    ``scene_steps``, ``scene_grid``, ``contact_r`` (scorecard); ``sdf``, ``split_history``,
    ``val_frac``, ``val_seed``, ``val_blocks``, ``pairs``, ``pair_seed`` (objective-sdf);
    ``grid``, ``horizon``, ``heldout_agents``, ``heldout_seed``, ``rollout``, ``w_coll``,
    ``window`` (objective-coef). ``notes`` (a dict) receives each loss case's description."""
    bad = [c for c in cases if c not in CASES]
    if bad:
        raise ValueError(f"unknown case {bad[0]!r} (cases: {', '.join(CASES)})")
    if "objective-sdf" in cases and not kw.get("sdf"):
        raise ValueError("objective-sdf needs sdf= (a nav_sdf.npz)")
    models = load_models(arms)
    seeds = list(seeds)
    values: dict = {}
    for case in cases:
        if case == "coef":
            values[case] = score_coef(
                models,
                seeds,
                grid=kw.get("grid", 96),
                mu=kw.get("mu", 0.18),
                agents=kw.get("agents", 192),
                ticks=kw.get("ticks", 90),
            )
        elif case == "scorecard":
            values[case] = score_scorecard(
                models,
                seeds,
                agents=kw.get("scene_agents", 64),
                steps=kw.get("scene_steps", 400),
                grid=kw.get("scene_grid", 96),
                contact_r=kw.get("contact_r", 0.0),
            )
        elif case == "objective-sdf":
            values[case] = score_objective_sdf(
                models,
                sdf=kw["sdf"],
                meta=checkpoint_meta(arms),
                split_history=kw.get("split_history"),
                val_frac=kw.get("val_frac"),
                val_seed=kw.get("val_seed", 0),
                val_blocks=kw.get("val_blocks", 8),
                pairs=kw.get("pairs", 1024),
                pair_seed=kw.get("pair_seed", 777),
                notes=notes,
            )
        else:
            values[case] = score_objective_coef(
                models,
                grid=kw.get("grid", 96),
                horizon=kw.get("horizon", 28),
                agents=kw.get("heldout_agents", 200),
                seed=kw.get("heldout_seed", 123),
                rollout=kw.get("rollout", "surrogate"),
                w_coll=kw.get("w_coll"),
                window=kw.get("window"),
                notes=notes,
            )
    return values


def markdown(table: dict) -> str:
    """The table as the A/B harness prints it: mean [CI], and the paired difference."""
    names = [a["name"] for a in table["arms"]]
    lines = []
    for res in table["results"]:
        unit = res.get("unit", "seed")
        lines += [
            f"\n## {res['case']}  ({res['episodes']} {unit}s, paired; diff to {table['reference']})\n",
            "| metric | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for metric, per_arm in res["metrics"].items():
            cells = []
            for a in names:
                e = per_arm.get(a)
                if e is None:
                    cells.append("-")
                    continue
                lo, hi = e["ci95"]
                cell = f"{e['mean']:.3f} [{lo:.3f}, {hi:.3f}]"
                d = e.get("paired_diff")
                if d is not None:
                    dlo, dhi = d["ci95"]
                    sig = "" if dlo <= 0.0 <= dhi else " *"
                    cell += f"<br>diff {d['mean']:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{sig}"
                cells.append(cell)
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append(
        "\nmean [95% bootstrap CI]; diff: paired difference to the reference (* = CI excludes 0)"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="grl-snam arms-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        required=True,
        metavar="NAME=PATH|const:...",
        help="an arm: a .cvcnav, a .pt, or const:alpha,beta,gamma | const:alpha=..,.. | "
        "const:bias | const:hist=history.jsonl (repeatable; the first is the reference)",
    )
    ap.add_argument("--seeds", type=int, default=20, help="paired seeds per task case (20)")
    ap.add_argument("--seed-base", type=int, default=1000, help="the first seed (1000)")
    ap.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help=f"comma-separated, of {', '.join(CASES)} ({','.join(DEFAULT_CASES)})",
    )
    ap.add_argument("--label", help="the run this table scores (e.g. its training seed)")
    ap.add_argument("--reference", help="the arm paired differences are taken to (the first)")
    ap.add_argument("--boot", type=int, default=5000, help="bootstrap resamples (5000)")
    ap.add_argument("--boot-seed", type=int, default=1, help="the bootstrap's seed (1)")
    ap.add_argument("--out", help="write grl_snam.arms/v1 JSON here")
    g = ap.add_argument_group("coef case (coef_eval's ice-band city)")
    g.add_argument("--grid", type=int, default=96)
    g.add_argument("--mu", type=float, default=0.18, help="the ice band's grip")
    g.add_argument("--agents", type=int, default=192, help="starts and goals per seed")
    g.add_argument("--ticks", type=int, default=90, help="drive ticks per seed")
    g = ap.add_argument_group("scorecard case (scorecard_eval's city scenes)")
    g.add_argument("--scene-agents", type=int, default=64)
    g.add_argument("--scene-steps", type=int, default=400)
    g.add_argument("--scene-grid", type=int, default=96)
    g.add_argument("--contact-r", type=float, default=0.0, help="vehicle contact radius (m)")
    g = ap.add_argument_group("objective-sdf case (grl-snam train's objective, held-out pairs)")
    g.add_argument("--sdf", help="the nav_sdf.npz the run trained on")
    g.add_argument("--split-history", help="the run's history: its held-out blocks")
    g.add_argument("--val-frac", type=float, help="else: the run's --val-frac")
    g.add_argument("--val-seed", type=int, default=0, help="the run's --val-seed (0)")
    g.add_argument("--val-blocks", type=int, default=8, help="the run's --val-blocks (8)")
    g.add_argument("--pairs", type=int, default=1024, help="held-out pairs (1024)")
    g.add_argument("--pair-seed", type=int, default=777, help="their seed (777)")
    g = ap.add_argument_group("objective-coef case (coef-train's objective, held-out starts)")
    g.add_argument("--heldout-seed", type=int, default=123, help="held-out starts' seed (123)")
    g.add_argument("--heldout-agents", type=int, default=200, help="held-out starts (200)")
    g.add_argument("--horizon", type=int, default=28, help="rollout steps (28)")
    g.add_argument("--rollout", choices=objective.ROLLOUTS, default="surrogate")
    g.add_argument("--w-coll", type=float, help="collision weight (the trainer's: 6 / 3)")
    g.add_argument("--window", type=int, help="BPTT window (the trainer's: 7 / 6)")
    args = ap.parse_args(argv)

    arms = [parse_arm(a) for a in args.arm]
    if len({a.name for a in arms}) != len(arms):
        ap.error("arm names must differ")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    bad = [c for c in cases if c not in CASES]
    if bad:
        ap.error(f"unknown cases {bad} (cases: {', '.join(CASES)})")
    if "objective-sdf" in cases and not args.sdf:
        ap.error("objective-sdf needs --sdf")
    if "objective-sdf" in cases and not (args.split_history or args.val_frac):
        ap.error("objective-sdf needs the run's held-out split: --split-history or --val-frac")
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    notes: dict = {}
    values = evaluate_arms(
        arms,
        seeds,
        cases,
        notes=notes,
        grid=args.grid,
        mu=args.mu,
        agents=args.agents,
        ticks=args.ticks,
        scene_agents=args.scene_agents,
        scene_steps=args.scene_steps,
        scene_grid=args.scene_grid,
        contact_r=args.contact_r,
        sdf=args.sdf,
        split_history=args.split_history,
        val_frac=args.val_frac,
        val_seed=args.val_seed,
        val_blocks=args.val_blocks,
        pairs=args.pairs,
        pair_seed=args.pair_seed,
        horizon=args.horizon,
        heldout_agents=args.heldout_agents,
        heldout_seed=args.heldout_seed,
        rollout=args.rollout,
        w_coll=args.w_coll,
        window=args.window,
    )
    settings = {k: v for k, v in vars(args).items() if k not in ("arm", "out", "label")}
    meta = {"tool": "grl_snam.tools.arms_eval", "settings": settings}
    if args.label:
        meta["label"] = args.label
    table = arms_table(
        values,
        arms,
        reference=args.reference,
        direction={m: d for c in cases for m, d in DIRECTION[c].items()},
        primary={c: PRIMARY[c] for c in cases},
        case_flags={c: notes.get(c, [f"seeds {seeds[0]}..{seeds[-1]}"]) for c in cases},
        family={c: FAMILY[c] for c in cases},
        unit={c: UNIT[c] for c in cases},
        n_boot=args.boot,
        seed=args.boot_seed,
        meta=meta,
    )
    print(markdown(table))
    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
