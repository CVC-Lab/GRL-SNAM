"""Base-policy scorecard eval — rank a CoefMLP checkpoint by its RF-free navigation
fitness over a scene corpus.

Runs the vectorized :class:`~grl_snam.swarm.Swarm` with its opt-in base nav_stats
collector across a corpus of city scenes, reduces each episode to an
:class:`~grl_snam.scorecard.EpisodeStats`, and aggregates them into one
:class:`~grl_snam.scorecard.NavScorecard` — the single fitness row grl-snam uses to
rank its own base-policy checkpoints (arrival, economy, safety, material), RF-free.
This is the base half of the two-layer nav-stats design's training bridge (the DBG
campaign layers an ``rf_scorecard`` on top); the scorecard field-mirrors the C++
``cvc::nav::nav_scorecard`` and the native ``sim_world`` collector, so a checkpoint
scores identically whichever path evaluates it.

    python -m grl_snam.tools.scorecard_eval --scenes 8                    # the bias net
    python -m grl_snam.tools.scorecard_eval --checkpoint run/coef.pt --json card.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import sdf_nav
from grl_snam import planner
from grl_snam.fog_stories import STORIES, shrunk
from grl_snam.material_palette import terrain_risk_share
from grl_snam.scorecard import aggregate_nav
from grl_snam.squad import AgentSpec
from grl_snam.swarm import Swarm


def load_model(checkpoint: str | None):
    """A CoefMLP: the trained net from ``checkpoint`` (a ``grl-snam train`` .pt with a
    ``model_state_dict``, or a bare state dict), or the fresh bias net when none given."""
    m = sdf_nav.CoefMLP()
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu")
        sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        m.load_state_dict(sd)
    m.eval()
    return m


def scene_specs(grid: int, seed: int, n_agents: int):
    """One city scene: the shared story/truth + ``n_agents`` starts+goals drawn from
    the largest free component (the same construction the swarm parity tests use)."""
    story = shrunk(STORIES["city"], n=grid, max_steps=10_000_000)
    truth = story.truth_grid()
    labels, sizes = planner.free_components(truth, 2)
    best = max(sizes, key=sizes.get)
    rows, cols = np.nonzero(labels == best)
    mnx, mny, mxx, mxy = story.bounds
    ny, nx = truth.shape

    def w(r, c):
        return (mnx + c / (nx - 1) * (mxx - mnx), mny + r / (ny - 1) * (mxy - mny))

    rng = np.random.default_rng(seed)
    s = rng.integers(0, len(rows), n_agents)
    g = rng.integers(0, len(rows), n_agents)
    specs = [
        AgentSpec(f"a{i}", w(rows[s[i]], cols[s[i]]), w(rows[g[i]], cols[g[i]]))
        for i in range(n_agents)
    ]
    return story, truth, specs


def evaluate(
    model,
    *,
    scenes: int = 8,
    agents: int = 64,
    steps: int = 400,
    grid: int = 96,
    contact_r: float = 0.0,
    checkpoint_label: str = "",
    material: bool = True,
):
    """Run ``model`` over ``scenes`` city scenes (seeds ``0..scenes-1``), collecting one
    base ``EpisodeStats`` per scene via the Swarm collector, and aggregate them into a
    ``NavScorecard``. Each scene runs to ``all_reached`` or ``steps``, whichever first.

    ``material`` (default on) attaches a per-scene discrete material-id raster so the
    ``material_time_share`` / terrain-risk buckets are populated. For a BASE net this is a
    stats-only classifier — it does NOT change navigation, so arrival/economy/safety are
    unaffected. For a **risk net** (``model.use_risk``) the matching continuous material
    grid is ALSO attached (``material=`` on the Swarm) so the drive builds the risk feature
    and applies the reroute force it was trained with — i.e. the net is scored as deployed.
    Pass ``material=False`` for a corpus with no material terrain (a risk net then has no
    risk column source and cannot be scored)."""
    from ..material import city_material_grid, city_material_ids

    # A risk net needs the material grid for its risk column; a lam-head net needs it for the
    # reroute force it learned to modulate. Either way the drive requires a material grid.
    needs_grid = getattr(model, "use_risk", False) or getattr(model, "use_lam", False)
    if needs_grid and not material:
        raise ValueError("scoring a use_risk/use_lam net needs material=True (its drive source)")

    episodes = []
    for seed in range(scenes):
        story, truth, specs = scene_specs(grid, seed, agents)
        mids = mgrid = None
        if material:
            meta = story.meta()
            if needs_grid:
                # matched grid (drive force + risk feature/lam) + id raster (stats), same terrain
                mgrid, mids = city_material_grid(
                    truth, story.bounds, meta["center"], meta["scale"], seed=seed
                )
            else:
                mids = city_material_ids(
                    truth, story.bounds, meta["center"], meta["scale"], seed=seed
                )
        sw = Swarm(
            story,
            specs,
            model=model,
            truth_occ=truth,
            prior_occ=truth,
            collect_stats=True,
            contact_radius_m=contact_r,
            material_ids=mids,
            material=mgrid,
        )
        for _ in range(steps):
            sw.step()
            if sw.all_reached:
                break
        episodes.append(sw.episode_stats())
    return aggregate_nav(episodes, checkpoint_label)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="CoefMLP .pt (grl-snam train output / bare state dict); default = the bias net",
    )
    ap.add_argument("--scenes", type=int, default=8, help="scene-corpus size (seeds 0..N-1)")
    ap.add_argument("--agents", type=int, default=64, help="agents per scene")
    ap.add_argument(
        "--steps", type=int, default=400, help="max ticks per scene (or until all reached)"
    )
    ap.add_argument("--grid", type=int, default=96, help="scene grid resolution")
    ap.add_argument(
        "--contact-r",
        type=float,
        default=0.0,
        help="pairwise vehicle-contact radius in world metres (0 = skip the O(N^2) pass)",
    )
    ap.add_argument(
        "--json", type=str, default="", help="also write the scorecard JSON to this path"
    )
    args = ap.parse_args(argv)

    model = load_model(args.checkpoint or None)
    label = args.checkpoint or "bias-net"
    card = evaluate(
        model,
        scenes=args.scenes,
        agents=args.agents,
        steps=args.steps,
        grid=args.grid,
        contact_r=args.contact_r,
        checkpoint_label=label,
    )
    d = card.to_dict()
    # A compact human line + the full JSON (the machine record a training campaign ranks on).
    print(
        f"[{label}] scenes={args.scenes} runs={d['n_vehicle_runs']} "
        f"success={d['success_rate']:.3f} arrival={d['arrival_rate']:.3f} "
        f"path_ratio={d['mean_path_ratio']:.3f} pen%={d['mean_penetration_pct']:.3f} "
        f"contacts/run={d['veh_contacts_per_run']:.3f} "
        f"risk-time%={100.0 * terrain_risk_share(d['material_time_share']):.1f}"
    )
    print(json.dumps(d, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            f.write(card.to_json())
    return card


if __name__ == "__main__":
    main()
