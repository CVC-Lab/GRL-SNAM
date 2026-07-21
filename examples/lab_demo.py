"""Demo: build a synthetic GRL-SNAM scene in the lab and show / snapshot it.

General GRL-SNAM only — a wavy terrain, a couple of obstacle meshes, two
agent trajectories, agent markers, and a risk field. A DBG adapter would
instead feed a real movement_bundle.v1 through the same Lab API.

Run (with pycvc / pycvc_gl on PYTHONPATH and a display):
    python examples/lab_demo.py           # interactive window
    python examples/lab_demo.py out.png   # offscreen snapshot
"""

import math
import sys

from grl_snam_lab import Lab


def _wavy_terrain(n=40, amp=3.0):
    return [
        [amp * math.sin(0.3 * r) * math.cos(0.3 * c) for c in range(n)]
        for r in range(n)
    ]


def build_demo() -> Lab:
    lab = Lab()
    lab.add_terrain(_wavy_terrain(), bounds=(-100, -100, 100, 100), color=(0.3, 0.5, 0.3))

    # two boxy "obstacles" as simple prisms (just top faces here for brevity)
    lab.add_mesh(
        "obstacle_a",
        [-40, -40, 5, -20, -40, 5, -20, -20, 5, -40, -20, 5],
        [0, 1, 2, 0, 2, 3],
        color=(0.6, 0.6, 0.65),
    )
    lab.add_mesh(
        "obstacle_b",
        [20, 10, 8, 45, 10, 8, 45, 35, 8, 20, 35, 8],
        [0, 1, 2, 0, 2, 3],
        color=(0.6, 0.6, 0.65),
    )

    # two agent trajectories crossing the field
    lab.add_path(
        "agent0",
        [(-90 + i * 4, -30 + 20 * math.sin(0.15 * i), 6) for i in range(45)],
        color=(1.0, 0.2, 0.2),
    )
    lab.add_path(
        "agent1",
        [(-90 + i * 4, 40 - 15 * math.sin(0.2 * i), 6) for i in range(45)],
        color=(0.2, 0.4, 1.0),
    )
    lab.add_markers("agents", [(-90, -30, 6), (-90, 40, 6)])

    # a risk field (higher near the obstacles); flat 8x8x4 grid
    nx, ny, nz = 8, 8, 4
    field = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                field.append(float((i - 4) ** 2 + (j - 4) ** 2 + k))
    lab.add_field("risk", field, dims=(nx, ny, nz), bounds=(-100, -100, 0, 100, 100, 30))

    lab.pump()
    return lab


if __name__ == "__main__":
    lab = build_demo()
    print(f"GRL-SNAM lab demo: {lab.num_nodes()} scene nodes")
    if len(sys.argv) > 1:
        lab.render_png(sys.argv[1])
        print(f"wrote {sys.argv[1]}")
    else:
        lab.show()
