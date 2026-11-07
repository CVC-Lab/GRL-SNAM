"""Pure-Python material-aware navigation demo.

Two side-by-side stories, each run twice (material off / on), each rendered
as a top-down PNG with every trajectory drawn over the terrain, plus printed
stats:

* **scenario** — the full fog-of-war stack (belief + planner + drive): a mud
  field straddling the direct route and a hard-hazard water strip with a dry
  gap. With material, the planner pays the risk surcharge and routes through
  the gap; the forces and witness gate polish locally.
* **swarm** — the reactive vectorized swarm (no planner): mud blobs and a
  water lake sitting just off a lane of vehicles. The forces + gate alone
  detour every vehicle; hazard cells are never entered.

Run: ``grl-snam material-demo [--out-dir material_demo]``

Everything is torch + numpy + matplotlib — no pycvc needed (set
``GRL_SNAM_MATERIAL_BACKEND=native`` to run the same demo through the C++
kernels). The libcvc twin of the swarm story is
``examples/nav_material_demo.cpp`` (pure C++, zero Python).

``--plant`` (``kinematic``, ``dynamic[:preset]``, ``jolt[:preset]``) runs both stories on
libcvc's sim_world instead, its agents driving that plant with the material's forces and
witness gate attached (``NativeSimWorld.set_material_grid``). The native world has no route
spine, so the scenario story's planner surcharge does not apply there: its agent detours on
the forces and the gate alone, as the swarm's do. Those forces steer a vehicle off risky ground
but slow it only by what they take from its goal pull's surplus over its top speed, which on a
physics plant is all of it: the dynamic vehicle driving square at the water crosses it at its top
speed, its run the same to the millimetre as without material. So on a physics plant libcvc's risk
governor is on as well, where the running pycvc has it (``governor``), and the material's risk
slows the vehicles there. The kinematic bicycle keeps the forces alone (the Python stories' drive)
unless ``governor`` is True: it creeps through the water under the governor rather than stopping.
"""

from __future__ import annotations

import numpy as np

N = 96
BOUNDS = (-100.0, -100.0, 100.0, 100.0)
SCALE = 0.05


def _meta():
    return dict(
        scale=SCALE,
        center=(0.0, 0.0),
        region=100.0,
        rr=0.15,
        d_hat=0.35,
        dt=0.06,
        nsub=2,
        vmax=0.9,
        bounds=list(BOUNDS),
    )


def _model(seed=0):
    import torch

    import grl_snam.sdf_nav as sdf_nav

    torch.manual_seed(seed)
    m = sdf_nav.CoefMLP()
    m.eval()
    return m


def _cell_of(x, y, n=N):
    c = int(round((x - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]) * (n - 1)))
    r = int(round((y - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1]) * (n - 1)))
    return min(max(r, 0), n - 1), min(max(c, 0), n - 1)


def _blob(arr, wy, wx, rad_m, value, n=N):
    yy = np.linspace(BOUNDS[1], BOUNDS[3], n)[:, None]
    xx = np.linspace(BOUNDS[0], BOUNDS[2], n)[None, :]
    mask = (yy - wy) ** 2 + (xx - wx) ** 2 <= rad_m * rad_m
    arr[mask] = value
    return mask


def _render(path, title, risk, hard, occ, runs, n=N):
    """Terrain + trajectories -> PNG. runs = [(label, positions[N,2], color)]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    n = risk.shape[0]
    img = np.ones((n, n, 3)) * np.array([0.93, 0.93, 0.89])
    mud = np.clip(risk, 0, 1)[..., None]
    img = img * (1 - mud) + mud * np.array([0.45, 0.30, 0.16])
    img[hard.astype(bool)] = (0.28, 0.43, 0.78)
    img[occ.astype(bool)] = (0.12, 0.12, 0.12)
    ax.imshow(img, origin="lower", extent=[BOUNDS[0], BOUNDS[2], BOUNDS[1], BOUNDS[3]])
    for label, pts, color in runs:
        ax.plot(pts[:, 0], pts[:, 1], ".", ms=1.5, color=color, label=label)
    ax.set_title(title)
    ax.legend(loc="upper right", markerscale=8)
    ax.set_xlim(BOUNDS[0], BOUNDS[2])
    ax.set_ylim(BOUNDS[1], BOUNDS[3])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _governed(governor, plant):
    """Whether a run on ``plant`` uses libcvc's risk governor: ``governor`` if given, else on a
    physics plant (dynamic, Jolt) where the running pycvc has it."""
    from grl_snam import nav_native

    if governor is not None:
        return bool(governor)
    return nav_native.HAS_RISK_GOVERNOR and not str(plant).startswith("kinematic")


GOVERNOR_LOOKAHEAD_M = 10.0


def _plant_run(truth, material, start, goal, plant, ground_mu, max_steps, governor=None):
    """One agent from ``start`` to ``goal`` on ``plant`` (libcvc's sim_world, the map known),
    with ``material`` attached and the risk governor per :func:`_governed`; returns its
    positions (N, 2) and whether it arrived."""
    import grl_snam.sdf_nav as sdf_nav
    from grl_snam.plant_nav import PlantNavigator

    meta = _meta()
    phi, nx, ny = sdf_nav.build_sdf(truth, BOUNDS, SCALE)
    field = sdf_nav.SDFField(phi, nx, ny, BOUNDS, meta["center"], SCALE)
    nav = PlantNavigator(
        field, _model(), meta, truth, plant, reach_tol=0.15, ground_mu=ground_mu,
        material=material,
        risk_governor={"lookahead_m": GOVERNOR_LOOKAHEAD_M} if _governed(governor, plant) else False,
    )  # fmt: skip
    nav.start(start, goal)
    ms, _, _, _ = nav.drive_to_goal(max_steps=max_steps)
    return np.array([[m.x, m.y] for m in ms]), bool(ms[-1].reached)


def scenario_story(out_dir, plant=None, ground_mu=0.9, governor=None):
    """Fog scenario: mud field + hazard strip with a gap; planner + forces. On ``plant`` one
    agent on libcvc's sim_world drives it with the forces, the gate and the governor."""
    risk, hard, plain, mat = scenario_runs(plant, ground_mu, governor)
    on = "" if plant is None else f" on {plant}, mu {ground_mu}" + _gov_note(governor, plant)
    print("scenario (belief + planner + drive):" if plant is None else f"scenario{on}:")
    print("                       material OFF   material ON")
    print(f"  waypoint reached     {plain[1]:>9}      {mat[1]:>9}")
    print(f"  mud exposure         {plain[2]:>9.1f}      {mat[2]:>9.1f}")
    print(f"  hazard-cell steps    {plain[3]:>9}      {mat[3]:>9}")
    path = _render(
        f"{out_dir}/material_scenario.png",
        (
            "FogScenario: planner + forces route around mud, through the dry gap"
            if plant is None
            else f"sim_world{on}: forces + gate around mud and the water"
        ),
        risk,
        hard,
        np.zeros((N, N), bool),
        [("material OFF", plain[0], "#c0392b"), ("material ON", mat[0], "#1e8449")],
    )
    print(f"  wrote {path}")
    return mat[3] == 0


def _gov_note(governor, plant):
    return ", risk governor on" if _governed(governor, plant) else ", forces only"


def scenario_runs(plant=None, ground_mu=0.9, governor=None):
    """The scenario story's world and both runs: ``(risk, hard, plain, mat)``, each run
    ``(positions (N, 2), waypoints reached, mud exposure, hazard-cell steps)``. On ``plant``
    the agent is libcvc's sim_world driving that plant (see :func:`_plant_run`)."""
    from grl_snam.material import MaterialGrid
    from grl_snam.scenario import FogScenario

    truth = np.zeros((N, N), bool)
    risk = np.zeros((N, N), np.float32)
    hard = np.zeros((N, N), bool)
    risk[38:58, 34:52] = 0.95  # mud field on the direct route
    # water strip crossing the route, dry gap OFF the direct line (rows 58..66,
    # y ~ +22..+39): the planner must pay the hard surcharge or detour up.
    hard[16:58, 60:64] = True
    hard[66:82, 60:64] = True
    risk[hard] = 1.0

    def run(material):
        if plant is not None:
            pts, arrived = _plant_run(
                truth, material, (-75.0, 0.0), (75.0, 0.0), plant, ground_mu, 4000, governor
            )
            return pts, int(arrived), *_exposure(pts)
        sc = FogScenario(
            truth,
            BOUNDS,
            SCALE,
            _model(),
            _meta(),
            waypoints=[(75.0, 0.0)],
            material=material,
        ).start((-75.0, 0.0))
        res = sc.run(max_steps=4000)
        pts = res.positions
        return pts, res.waypoints_reached, *_exposure(pts)

    def _exposure(pts):
        exposure = 0.0
        hazard_hits = 0
        for x, y in pts:
            r, c = _cell_of(x, y)
            exposure += float(risk[r, c])
            hazard_hits += int(hard[r, c])
        return exposure, hazard_hits

    plain = run(None)
    grid = MaterialGrid(risk, hard, BOUNDS, (0, 0), SCALE)
    mat = run(grid)
    return risk, hard, plain, mat


def swarm_story(out_dir, plant=None, ground_mu=0.9, governor=None):
    """Reactive swarm (no planner): forces + gate detour a lane of vehicles. On ``plant`` the
    same swarm runs on libcvc's sim_world and drives that plant's vehicles."""
    risk, hard, plain, mat = swarm_runs(plant=plant, ground_mu=ground_mu, governor=governor)
    n = risk.shape[0]
    print(
        "swarm (reactive, no planner):"
        if plant is None
        else f"swarm on {plant}, mu {ground_mu}{_gov_note(governor, plant)}:"
    )
    print("                       material OFF   material ON")
    print(f"  agents reached       {plain[1]:>7}/6      {mat[1]:>7}/6")
    print(f"  mud exposure         {plain[2]:>9.1f}      {mat[2]:>9.1f}")
    print(f"  hazard-cell steps    {plain[3]:>9}      {mat[3]:>9}")
    path = _render(
        f"{out_dir}/material_swarm.png",
        (
            "Swarm: reactive detours around mud + a water lake"
            if plant is None
            else f"sim_world swarm on {plant}: detours around mud + a water lake"
        ),
        risk,
        hard,
        np.zeros((n, n), bool),
        [("material OFF", plain[0], "#c0392b"), ("material ON", mat[0], "#1e8449")],
    )
    print(f"  wrote {path}")
    return mat[3] == 0


def swarm_runs(steps=700, plant=None, ground_mu=0.9, governor=None):
    """The swarm story's world and both runs, as :func:`scenario_runs` gives them (reached is
    the agents that arrived, of 6). On ``plant`` the swarm runs on libcvc's sim_world."""
    import torch

    from grl_snam.fog_stories import STORIES, shrunk
    from grl_snam.material import MaterialGrid
    from grl_snam.squad import AgentSpec
    from grl_snam.swarm import Swarm

    n = 64
    story = shrunk(STORIES["city"], n=n, max_steps=10_000_000)
    risk = np.zeros((n, n), np.float32)
    hard = np.zeros((n, n), bool)
    rr_, cc_ = np.mgrid[0:n, 0:n]
    # Blob centres sit BETWEEN lanes and the radii keep the lane chords
    # shallow (a few metres): the reactive field deflects around lateral
    # gradients; deep head-on overlap is the planner's job (documented).
    hard |= (rr_ - 39.0) ** 2 + (cc_ - 28.0) ** 2 <= 12.0  # lake between y=18/30, mid-course
    risk[(rr_ - 31.5) ** 2 + (cc_ - 40.0) ** 2 <= 12.0] = 0.95  # mud between y=-6/6
    risk[hard] = 1.0

    specs = [
        AgentSpec(f"a{i}", (-45.0, -30.0 + 12.0 * i), (45.0, -30.0 + 12.0 * i)) for i in range(6)
    ]

    def run(material, steps=steps):
        torch.manual_seed(0)
        s = Swarm(
            story, specs, _model(), seed=0, truth_occ=np.zeros((n, n), bool), material=material
        )
        if plant is not None:
            return _run_native(s, material, steps)
        pts = []
        exposure = 0.0
        hazard_hits = 0
        for _ in range(steps):
            s.step()
            w = s.n2w(s.o).cpu().numpy()
            pts.append(w.copy())
            for x, y in w:
                r, c = _cell_of(x, y, n)
                exposure += float(risk[r, c])
                hazard_hits += int(hard[r, c])
            if bool(s.reached.all()):
                break
        reached = int(s.reached.sum().item())
        return np.concatenate(pts), reached, exposure, hazard_hits

    def _run_native(s, material, steps):
        import os
        import tempfile

        from grl_snam import nav_native
        from grl_snam.tools.coef_export import write_coef_mlp

        path = os.path.join(tempfile.mkdtemp(), "material_swarm.cvcnav")
        write_coef_mlp(s.model, path)
        truth = np.zeros((n, n), bool)
        pl = nav_native.make_plant(plant, bounds=story.bounds, obstacles=truth, ground_mu=ground_mu)
        w = nav_native.sim_world_from_swarm(
            s, path, truth=truth, freeze_sense=True, plant=pl, material=material
        )
        if _governed(governor, plant):
            w.set_risk_governor(True, 0.0, 0.3, GOVERNOR_LOOKAHEAD_M * story.scale, 3)
        pts = []
        exposure = 0.0
        hazard_hits = 0
        reached = np.zeros(len(specs), bool)
        for _ in range(steps):
            w.step()
            pos, _h, _sp, _md, rch = w.snapshot()
            pts.append(np.asarray(pos, np.float64).copy())
            for x, y in pts[-1]:
                r, c = _cell_of(x, y, n)
                exposure += float(risk[r, c])
                hazard_hits += int(hard[r, c])
            reached |= np.asarray(rch, bool)
            if reached.all():
                break
        return np.concatenate(pts), int(reached.sum()), exposure, hazard_hits

    plain = run(None)
    grid = MaterialGrid(risk, hard, story.bounds, (0, 0), story.scale)
    mat = run(grid)
    return risk, hard, plain, mat


def main(out_dir: str = "material_demo", plant=None, ground_mu: float = 0.9, governor=None) -> int:
    import os

    os.makedirs(out_dir, exist_ok=True)
    ok1 = scenario_story(out_dir, plant, ground_mu, governor)
    ok2 = swarm_story(out_dir, plant, ground_mu, governor)
    if plant is None:
        print("done — pure Python (set GRL_SNAM_MATERIAL_BACKEND=native for the C++ kernels).")
    else:
        print(f"done — on libcvc's sim_world and {plant}.")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
