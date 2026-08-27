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

    import sdf_nav

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


def scenario_story(out_dir):
    """Fog scenario: mud field + hazard strip with a gap; planner + forces."""
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
        exposure = 0.0
        hazard_hits = 0
        for x, y in pts:
            r, c = _cell_of(x, y)
            exposure += float(risk[r, c])
            hazard_hits += int(hard[r, c])
        return pts, res.waypoints_reached, exposure, hazard_hits

    plain = run(None)
    grid = MaterialGrid(risk, hard, BOUNDS, (0, 0), SCALE)
    mat = run(grid)

    print("scenario (belief + planner + drive):")
    print("                       material OFF   material ON")
    print(f"  waypoint reached     {plain[1]:>9}      {mat[1]:>9}")
    print(f"  mud exposure         {plain[2]:>9.1f}      {mat[2]:>9.1f}")
    print(f"  hazard-cell steps    {plain[3]:>9}      {mat[3]:>9}")
    path = _render(
        f"{out_dir}/material_scenario.png",
        "FogScenario: planner + forces route around mud, through the dry gap",
        risk,
        hard,
        np.zeros((N, N), bool),
        [("material OFF", plain[0], "#c0392b"), ("material ON", mat[0], "#1e8449")],
    )
    print(f"  wrote {path}")
    return mat[3] == 0


def swarm_story(out_dir):
    """Reactive swarm (no planner): forces + gate detour a lane of vehicles."""
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

    def run(material, steps=700):
        torch.manual_seed(0)
        s = Swarm(
            story, specs, _model(), seed=0, truth_occ=np.zeros((n, n), bool), material=material
        )
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

    plain = run(None)
    grid = MaterialGrid(risk, hard, story.bounds, (0, 0), story.scale)
    mat = run(grid)

    print("swarm (reactive, no planner):")
    print("                       material OFF   material ON")
    print(f"  agents reached       {plain[1]:>7}/6      {mat[1]:>7}/6")
    print(f"  mud exposure         {plain[2]:>9.1f}      {mat[2]:>9.1f}")
    print(f"  hazard-cell steps    {plain[3]:>9}      {mat[3]:>9}")
    path = _render(
        f"{out_dir}/material_swarm.png",
        "Swarm: reactive detours around mud + a water lake",
        risk,
        hard,
        np.zeros((n, n), bool),
        [("material OFF", plain[0], "#c0392b"), ("material ON", mat[0], "#1e8449")],
    )
    print(f"  wrote {path}")
    return mat[3] == 0


def main(out_dir: str = "material_demo") -> int:
    import os

    os.makedirs(out_dir, exist_ok=True)
    ok1 = scenario_story(out_dir)
    ok2 = swarm_story(out_dir)
    print("done — pure Python (set GRL_SNAM_MATERIAL_BACKEND=native for the C++ kernels).")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
