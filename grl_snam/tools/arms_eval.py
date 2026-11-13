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

Two cases, each one unit per seed (``--seed-base`` .. ``+ --seeds``, held out from the trainers'
own draws):

- ``coef``: :func:`grl_snam.tools.coef_eval.evaluate` on the ice-band city, one draw of
  ``--agents`` starts and goals per seed: reach, penetration (clearance below ``-0.5 * rr``),
  mean worst clearance and the final gap to the goal.
- ``scorecard``: :func:`grl_snam.tools.scorecard_eval.episodes`, one city scene per seed, reduced
  to its one-episode scorecard: arrival, success, penetration %, path ratio, time to goal (over
  the arrivers; none when nobody arrived) and stall steps.

The output is ``grl_snam.arms/v1`` (:func:`grl_snam.learning.arms.arms_table`): per metric and
arm, the per-seed values, mean, 95% bootstrap CI and the paired difference to the reference arm
(``--reference``, default the first). ``grl-snam learn-report --arms`` reads it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
}
PRIMARY = {"coef": ["reach", "pen"], "scorecard": ["arrival_rate", "mean_penetration_pct"]}
CASES = tuple(DIRECTION)


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


def evaluate_arms(arms: list[ArmSpec], seeds, cases=CASES, **kw) -> dict:
    """``{case: {metric: {arm: {seed: value}}}}`` for every arm on every seed. ``kw`` are the
    cases' settings: ``grid``, ``mu``, ``agents``, ``ticks`` (coef) and ``scene_agents``,
    ``scene_steps``, ``scene_grid``, ``contact_r`` (scorecard)."""
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
        else:
            raise ValueError(f"unknown case {case!r} (cases: {', '.join(CASES)})")
    return values


def markdown(table: dict) -> str:
    """The table as the A/B harness prints it: mean [CI], and the paired difference."""
    names = [a["name"] for a in table["arms"]]
    lines = []
    for res in table["results"]:
        lines += [
            f"\n## {res['case']}  ({res['episodes']} paired seeds; diff to {table['reference']})\n",
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
    ap.add_argument("--seeds", type=int, default=20, help="paired seeds per case (20)")
    ap.add_argument("--seed-base", type=int, default=1000, help="the first seed (1000)")
    ap.add_argument(
        "--cases", default=",".join(CASES), help=f"comma-separated ({', '.join(CASES)})"
    )
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
    args = ap.parse_args(argv)

    arms = [parse_arm(a) for a in args.arm]
    if len({a.name for a in arms}) != len(arms):
        ap.error("arm names must differ")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    bad = [c for c in cases if c not in CASES]
    if bad:
        ap.error(f"unknown cases {bad} (cases: {', '.join(CASES)})")
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    values = evaluate_arms(
        arms,
        seeds,
        cases,
        grid=args.grid,
        mu=args.mu,
        agents=args.agents,
        ticks=args.ticks,
        scene_agents=args.scene_agents,
        scene_steps=args.scene_steps,
        scene_grid=args.scene_grid,
        contact_r=args.contact_r,
    )
    settings = {k: v for k, v in vars(args).items() if k not in ("arm", "out")}
    table = arms_table(
        values,
        arms,
        reference=args.reference,
        direction={m: d for c in cases for m, d in DIRECTION[c].items()},
        primary={c: PRIMARY[c] for c in cases},
        case_flags={c: [f"seeds {seeds[0]}..{seeds[-1]}"] for c in cases},
        n_boot=args.boot,
        seed=args.boot_seed,
        meta={"tool": "grl_snam.tools.arms_eval", "settings": settings},
    )
    print(markdown(table))
    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
