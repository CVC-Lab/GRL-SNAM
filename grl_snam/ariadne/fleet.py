"""A live squad of Humvees on real Austin, on the plant you pick, in an Ariadne document.

    python -m grl_snam.ariadne fleet                       # Jolt Humvees when pycvc has them
    python -m grl_snam.ariadne fleet --plant kinematic     # nav's bicycle, in libcvc's sim_world
    python -m grl_snam.ariadne fleet --plant torch         # GRL-SNAM's own path (the one that trains)
    python -m grl_snam.ariadne fleet --agents 8 --offscreen --frames 300 --png fleet.png

Unlike the recorded players this runs the squad live: ``grl_snam.squad.Squad`` on Austin's
occupancy (``grl_snam.tools.austin``), each agent with its own belief, starting blind, driving to a
goal across the city. ``--plant`` (and the document's Plant combo) picks the runtime:

- ``torch`` (the default): GRL-SNAM's Python squad, one FogScenario per agent, the path that
  trains;
- ``humvee``: libcvc's sim_world with each agent driving a Jolt-simulated Humvee
  (``Squad(plant="humvee")``: engine, tyres, the buildings as boxes);
- ``kinematic``: the same sim_world on nav's kinematic bicycle.

Measured on Austin with 4 agents over 18 s of sim: torch moves all four at about 10 m/s; the
kinematic sim_world moves all four but at 14-18 m/s (the story's vmax, 0.9 normalized); on Jolt
Humvees two of the four stay stuck in wall-escape mode. The navigator is sized for the story's
bodies, not a 4.7 m Humvee (NATIVE_CVC_NAV.md says a Humvee does not fit the city story's
streets), so Jolt is a choice here rather than the default. libcvc's own
``examples/ariadne/python/nav.py``, whose Humvees follow routes planned for their turning radius,
runs Jolt by default.

The policy is the published ``grl-snam-weights`` checkpoint when installed (else the untrained
seed). Each agent is a Humvee at its real size, seated on the terrain at its heading, with a pole
in its colour, its goal and its track. The panels give a row per agent, the first agent's HUD,
and New goals / Restart; an agent that arrives waits for the rest.
"""

from __future__ import annotations

import sys

import numpy as np

from grl_snam.ariadne import _host

PLANTS = ("torch", "humvee", "kinematic")


def _args(p):
    p.add_argument("--plant", choices=PLANTS + ("auto",), default="auto")
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--bundle", default=None, help="the Austin scene bundle (default: installed)")
    p.add_argument("--seed", type=int, default=0)


def load_model():
    """The published CoefMLP (grl-snam-weights), or None (the untrained seed) without it."""
    import os

    from grl_snam.demos._common import default_nav_weights_pt

    path = default_nav_weights_pt()
    if not path or not os.path.exists(path):
        return None
    import torch

    import grl_snam.sdf_nav as sdf_nav

    ck = torch.load(path, map_location="cpu")
    model = sdf_nav.CoefMLP()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


class Fleet:
    """A squad of ``n`` agents on Austin with random far-apart starts and goals."""

    def __init__(self, occ, bounds, story, model, n, plant, rng):
        from grl_snam.squad import AgentSpec, Squad
        from grl_snam.tools.finale_record import COLORS, _free_snapper

        snap = _free_snapper(occ, bounds)
        half = 0.85 * (bounds[2] - bounds[0]) / 2
        agents = []
        for i in range(n):
            for _ in range(50):
                s = snap(*rng.uniform(-half, half, 2))
                g = snap(*rng.uniform(-half, half, 2))
                if np.hypot(g[0] - s[0], g[1] - s[1]) > 300.0 and all(
                    np.hypot(s[0] - a.start[0], s[1] - a.start[1]) > 15.0 for a in agents
                ):
                    break
            agents.append(AgentSpec(f"v{i}", s, g, COLORS[i % len(COLORS)]))
        self.agents = agents
        self.plant = plant
        self.squad = Squad(
            story,
            agents,
            model,
            truth_occ=occ,
            prior_occ=None,
            plant=None if plant == "torch" else plant,
        )
        self.recs = {}
        self.tracks = {a.key: [] for a in agents}
        self.ticks = 0

    def step(self, n):
        for _ in range(n):
            if self.squad.done:
                return
            self.recs = self.squad.step()
            self.ticks += 1
            if self.ticks % 4 == 0:
                for k, r in self.recs.items():
                    self.tracks[k] = (self.tracks[k] + [(r.x, r.y)])[-400:]


def main(argv=None) -> int:
    args = _host.parse(argv, _args)
    from grl_snam.nav_native import HAS_PLANT
    from grl_snam.tools.austin import austin_story, occupancy

    bundle = _host.bundle_or_none(args.bundle)
    if bundle is None:
        raise RuntimeError(
            "fleet needs the Austin scene bundle (cvcpkg install "
            "scene-austin-south, or --bundle / GRL_SNAM_SCENE_BUNDLE)"
        )
    plant = args.plant if args.plant != "auto" else "torch"
    occ, bounds = occupancy(bundle)
    story = austin_story(occ, bounds, key="fleet", fog=True)[0]
    model = load_model()
    rng = np.random.default_rng(args.seed)
    host = _host.open_host("fleet.ari", "fleet", args)
    host.set_many({"agents": args.agents, "plant": plant}, prefix="fleet")
    drawn = host.mode != "headless"
    st = {"fleet": None, "nodes": []}
    sample = None
    if drawn:
        from grl_snam.ariadne.squad import _pyramid, add_city

        sample = add_city(host, bundle)
        model_mesh = _host.humvee_model(bundle)
        _host.track_camera(host)

    def build():
        n = max(1, host.get_int("fleet.agents", args.agents))
        kind = host.get("fleet.plant", plant)
        if kind != "torch" and not HAS_PLANT:
            kind = "torch"
        st["fleet"] = Fleet(occ, bounds, story, model, n, kind, rng)
        if drawn:
            for name in st["nodes"]:
                host.scene.removeGraphics(name)
            st["nodes"] = []
            for a in st["fleet"].agents:
                k = a.key
                _host.add_vehicle(host, f"car_{k}", model_mesh)
                host.lab.add_mesh(f"pole_{k}", *_host.humvee_box(0.35, 0.35, 7.0), color=a.color)
                host.lab.add_mesh(f"goal_{k}", *_pyramid(6.0), color=a.color)
                gx, gy = a.goal
                host.scene.getGraphics(f"goal_{k}").setTransform(
                    [1, 0, 0, gx, 0, 1, 0, gy, 0, 0, 1, sample(gx, gy) + 8.0, 0, 0, 0, 1]
                )
                host.lab.add_path(f"track_{k}", [(0, 0, 0), (0.1, 0, 0)], a.color)
                st["nodes"] += [f"car_{k}", f"pole_{k}", f"goal_{k}", f"track_{k}"]

    host.on("restart", build)
    host.on("shuffle", build)

    @host.watch("fleet.plant")
    @host.watch("fleet.agents")
    def _changed(_v):
        if st["fleet"] is not None:
            build()

    build()
    camera = _host.CameraSwitch(host, lambda: host.frame_bounds((-600, -600, 0), (600, 600, 60)))

    @host.on_frame
    def tick(dt):
        fleet = st["fleet"]
        if not host.get_bool("view.paused"):
            fleet.step(max(1, host.get_int("fleet.ticks", 2)))
        rows = []
        for i, a in enumerate(fleet.agents):
            r = fleet.recs.get(a.key)
            if r is None:
                continue
            rows.append(
                {
                    "key": a.key,
                    "speed": f"{abs(r.speed_mps):.1f}",
                    "goal": f"{r.goal_dist_m:.0f}",
                    "mode": "arrived" if r.goal_dist_m < 10.0 else r.mode,
                }
            )
            if not drawn:
                continue
            from pycvc_gl.vehicle import seat_transform

            from grl_snam.fog_scene import _style

            z = sample(r.x, r.y)
            host.scene.getGraphics(f"car_{a.key}").setTransform(
                seat_transform(r.x, r.y, r.heading_rad, sample)
            )
            host.scene.getGraphics(f"pole_{a.key}").setTransform(
                [1, 0, 0, r.x, 0, 1, 0, r.y, 0, 0, 1, z + 2.0, 0, 0, 0, 1]
            )
            pts = fleet.tracks[a.key]
            if host.frame_count % 6 == 0 and len(pts) >= 2:
                host.lab.add_path(
                    f"track_{a.key}", [(x, y, sample(x, y) + 0.6) for x, y in pts], a.color
                )
                _style(host.scene, f"track_{a.key}", color=a.color, mode="lines", line_width=3)
        host.set_table("agents", rows)
        host.set_many(
            {
                "plant": fleet.plant,
                "time": f"{fleet.ticks * story.dt:.1f}",
                "arrived": f"{sum(r['mode'] == 'arrived' for r in rows)} / {len(rows)}",
            },
            prefix="stats",
        )
        r0 = fleet.recs.get(fleet.agents[0].key)
        if drawn and r0 is not None:
            camera.update((r0.x, r0.y, sample(r0.x, r0.y)))

    return _host.finish(host, args)


if __name__ == "__main__":
    sys.exit(main())
