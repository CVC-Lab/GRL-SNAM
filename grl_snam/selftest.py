"""grl-snam-selftest — a console numerical correctness demo.

Exercises the GRL-SNAM surrogate navigation dynamics (``integrate_surrogate_v2``,
the radius-aware differentiable rollout at the heart of the planner) on small,
seeded, self-contained cases and asserts correctness properties that hold by
construction — no dataset, no training, ~1 second:

  1. GOAL-SEEKING (obstacle-free): a damped spring toward the goal must reduce the
     distance to the goal over the rollout.
  2. FINITE through the IPC BARRIER: with an obstacle present, the barrier path
     (log/clamp) must produce finite, bounded output (no NaN/Inf).
  3. DETERMINISM: identical inputs must yield bit-identical outputs (reproducible).
  4. SHAPES: (o, v, min_clear) come back as (B,2), (B,2), (B,).

This is the correctness counterpart to the visual grl_snam_lab demo. It needs
torch (a declared GRL-SNAM dependency); the applied full-gym end-to-end demo lives
in a separate, private downstream project.

Run: ``grl-snam-selftest`` (installed console script) or ``python -m grl_snam.selftest``.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        import torch
    except ImportError:
        print(
            "grl-snam-selftest: requires torch (a GRL-SNAM runtime dependency).\n"
            "  Install GRL-SNAM with its deps, e.g. `pip install grl-snam` or `poetry install`."
        )
        return 2

    from grl_snam.dynamics import integrate_surrogate_v2

    torch.manual_seed(0)
    f64 = torch.float64  # double precision for a crisp numerical check
    B = 1

    def coeffs(N: int):
        return dict(
            alphas=torch.ones(B, N, dtype=f64),  # barrier strength per obstacle
            beta=torch.ones(B, dtype=f64),  # goal-spring stiffness
            gamma=torch.ones(B, dtype=f64),  # velocity damping
            d_hat=torch.full((B,), 2.0, dtype=f64),  # barrier activation distance
            dt=torch.full((B,), 0.05, dtype=f64),
            H=torch.full((B,), 400, dtype=torch.long),
        )

    o0 = torch.tensor([[-10.0, 0.0]], dtype=f64)
    v0 = torch.zeros(B, 2, dtype=f64)
    goal = torch.tensor([[10.0, 0.0]], dtype=f64)

    print("grl-snam-selftest: exercising integrate_surrogate_v2 (surrogate navigation dynamics)")
    checks: list[tuple[str, bool, str]] = []

    # ── 1. Goal-seeking, obstacle-free (N=0) ────────────────────────────────
    empty2 = torch.zeros(B, 0, 2, dtype=f64)
    empty1 = torch.zeros(B, 0, dtype=f64)
    emptym = torch.zeros(B, 0, dtype=torch.bool)
    o, v, clr = integrate_surrogate_v2(o0, v0, goal, empty2, empty1, emptym, **coeffs(0))
    d0 = torch.linalg.norm(o0 - goal).item()
    d1 = torch.linalg.norm(o - goal).item()
    checks.append(
        (
            "shapes (o,v,min_clear)",
            tuple(o.shape) == (B, 2) and tuple(v.shape) == (B, 2) and tuple(clr.shape) == (B,),
            f"o{tuple(o.shape)} v{tuple(v.shape)} clr{tuple(clr.shape)}",
        )
    )
    checks.append(
        ("goal-seeking reduces distance", d1 < d0, f"|o0-goal|={d0:.3f} -> |oT-goal|={d1:.3f}")
    )
    checks.append(("goal-seeking output finite", bool(torch.isfinite(o).all()), f"oT={o.tolist()}"))

    # ── 2. Finite through the IPC barrier (N=1, obstacle beside the path) ────
    C = torch.tensor([[[0.0, 6.0]]], dtype=f64)  # off the x-axis direct path
    R = torch.full((B, 1), 2.0, dtype=f64)
    mask = torch.ones(B, 1, dtype=torch.bool)
    ob_o, ob_v, ob_clr = integrate_surrogate_v2(
        o0, v0, goal, C, R, mask, **coeffs(1), robot_radius=0.5
    )
    barrier_finite = bool(
        torch.isfinite(ob_o).all() and torch.isfinite(ob_v).all() and torch.isfinite(ob_clr).all()
    )
    checks.append(
        (
            "IPC barrier path stays finite",
            barrier_finite,
            f"min_clear={ob_clr.item():.4f}, oT={ob_o.tolist()}",
        )
    )

    # ── 3. Determinism (identical inputs -> identical outputs) ──────────────
    o_a, v_a, c_a = integrate_surrogate_v2(o0, v0, goal, C, R, mask, **coeffs(1), robot_radius=0.5)
    o_b, v_b, c_b = integrate_surrogate_v2(o0, v0, goal, C, R, mask, **coeffs(1), robot_radius=0.5)
    deterministic = bool(torch.equal(o_a, o_b) and torch.equal(v_a, v_b) and torch.equal(c_a, c_b))
    checks.append(("deterministic (reproducible)", deterministic, "two runs bit-identical"))

    # ── report ──────────────────────────────────────────────────────────────
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:34} {detail}")
        ok = ok and passed
    print("grl-snam-selftest:", "OK — surrogate dynamics correct" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
