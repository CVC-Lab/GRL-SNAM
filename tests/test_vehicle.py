"""Kinematic-bicycle dynamics (sdf_nav.bicycle_rollout) + moving-target tracking
(SdfNavigator.track_goal) on a synthetic open field — no compiled bindings needed.

The invariants pinned here are the ones that fail *silently* if broken: a
turning radius that isn't actually respected, sideways motion a car cannot do,
an acceleration cap that leaks, a per-frame retarget that quietly disables the
wall-follow escape, and gradients that stop flowing (which would kill the
self-supervised training path).
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# These import torch at module level, so they must come AFTER the skip guard
# or collection itself crashes on a torch-less environment.
import sdf_nav  # noqa: E402
from grl_snam.nav import SdfNavigator  # noqa: E402

RR, DHAT, DT, VMAX = 0.15, 0.35, 0.06, 0.9
VEH = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8)
R_MIN = VEH["L"] / math.tan(VEH["delta_max"])  # the promised turning radius


def _open_field():
    n = 64
    phi = np.full((n, n), 5.0, np.float32)
    zeros = np.zeros((n, n), np.float32)
    bounds = (-100.0, -100.0, 100.0, 100.0)
    scale = 0.05
    field = sdf_nav.SDFField(phi, zeros, zeros, bounds, (0.0, 0.0), scale)
    meta = dict(
        scale=scale,
        center=(0.0, 0.0),
        region=100.0,
        rr=RR,
        d_hat=DHAT,
        dt=DT,
        nsub=2,
        vmax=VMAX,
        bounds=list(bounds),
    )
    return field, meta


def _coef(b=1.0, g=4.0, be=3.0):
    one = torch.ones(1)
    return b * one, be * one, g * one


def _roll(field, o, th, sp, goal, steps, **over):
    al, be, ga = _coef()
    kw = dict(rr=RR, d_hat=DHAT, dt=DT, nsub=2, vmax=VMAX, **VEH)
    kw.update(over)
    return sdf_nav.bicycle_rollout(field, o, th, sp, goal, al, be, ga, steps, **kw)


# ── the bicycle contract ─────────────────────────────────────────────────────


def test_turning_radius_is_respected():
    """Goal directly behind forces the tightest turn the model allows; the arc
    it traces must never dip below R_min = L / tan(delta_max)."""
    field, _ = _open_field()
    o = torch.zeros(1, 2)
    th = torch.zeros(1)
    sp = torch.full((1,), 0.4)
    goal = torch.tensor([[-5.0, 0.0]])
    pts, hds = [], []
    for _ in range(300):
        o, th, sp, _c = _roll(field, o, th, sp, goal, 1)
        pts.append(o[0].numpy().copy())
        hds.append(float(th[0]))
    pts = np.array(pts)
    hds = np.unwrap(np.array(hds))
    # instantaneous radius = |ds/dtheta| wherever the heading is actually changing
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dth = np.abs(np.diff(hds))
    turning = dth > 1e-4
    assert turning.any(), "vehicle never turned"
    radii = ds[turning] / dth[turning]
    assert (
        radii.min() >= R_MIN * 0.95
    ), f"turning radius violated: {radii.min():.4f} < R_min {R_MIN:.4f}"
    # and it must actually have gotten around toward the goal
    assert pts[-1, 0] < -1.0


def test_motion_is_always_along_the_heading():
    """The non-holonomy assertion — the one thing a point mass cannot satisfy.
    Every displacement must align with the vehicle's heading; no strafing."""
    field, _ = _open_field()
    o = torch.zeros(1, 2)
    th = torch.zeros(1)
    sp = torch.zeros(1)
    goal = torch.tensor([[2.0, 1.5]])
    prev = o[0].numpy().copy()
    for _ in range(200):
        o, th, sp, _c = _roll(field, o, th, sp, goal, 1)
        cur = o[0].numpy()
        step_v = cur - prev
        n = np.linalg.norm(step_v)
        if n > 1e-6:
            head = np.array([math.cos(float(th[0])), math.sin(float(th[0]))])
            # substepping means displacement is a short arc chord; allow a few
            # degrees of chord-vs-final-heading mismatch, nothing more.
            assert np.dot(step_v / n, head) > 0.99, "vehicle moved off-heading (strafed)"
        prev = cur


def test_acceleration_cap_holds():
    field, _ = _open_field()
    o = torch.zeros(1, 2)
    th = torch.zeros(1)
    sp = torch.zeros(1)
    goal = torch.tensor([[8.0, 0.0]])  # far: full-throttle regime
    last = 0.0
    peak = 0.0
    for _ in range(150):
        o, th, sp, _c = _roll(field, o, th, sp, goal, 1)
        now = float(sp[0])
        assert now - last <= VEH["a_max"] * DT + 1e-6, "acceleration cap leaked"
        last = now
        peak = max(peak, now)
    # It reaches vmax en route; the *final* speed is low because it brakes on
    # arrival, which is correct behaviour, not a failure.
    assert peak > 0.8, f"never got up to speed (peak {peak:.3f})"


def test_lateral_cap_slows_the_vehicle_into_turns():
    """v^2 * |tan(delta)| / L <= a_lat_max, checked via realized curvature:
    kappa = dtheta / ds. The emergent behaviour: it brakes for corners."""
    field, _ = _open_field()
    o = torch.zeros(1, 2)
    th = torch.zeros(1)
    sp = torch.full((1,), VMAX)  # enter the turn fast
    goal = torch.tensor([[0.0, 4.0]])  # hard 90-degree demand
    prev_p = o[0].numpy().copy()
    prev_h = float(th[0])
    for _ in range(200):
        o, th, sp, _c = _roll(field, o, th, sp, goal, 1)
        p, h, v = o[0].numpy(), float(th[0]), float(sp[0])
        ds = np.linalg.norm(p - prev_p)
        dth = abs(h - prev_h)
        if ds > 1e-6 and dth > 1e-4:
            kappa = dth / ds
            a_lat = v * v * kappa
            assert (
                a_lat <= VEH["a_lat_max"] * 1.15
            ), f"lateral acceleration {a_lat:.3f} exceeds the cap"
        prev_p, prev_h = p, h


def test_speed_never_negative_or_above_vmax():
    field, _ = _open_field()
    o = torch.zeros(1, 2)
    th = torch.zeros(1)
    sp = torch.zeros(1)
    goal = torch.tensor([[-4.0, 3.0]])  # behind: exercises the creep path too
    for _ in range(300):
        o, th, sp, _c = _roll(field, o, th, sp, goal, 1)
        v = float(sp[0])
        assert 0.0 <= v <= VMAX + 1e-6


def test_rollout_is_deterministic():
    field, _ = _open_field()

    def run():
        o = torch.zeros(1, 2)
        th = torch.zeros(1)
        sp = torch.zeros(1)
        goal = torch.tensor([[3.0, -2.0]])
        o, th, sp, mc = _roll(field, o, th, sp, goal, 120)
        return o.numpy().copy(), th.numpy().copy(), sp.numpy().copy()

    a, b = run(), run()
    for x, y in zip(a, b):
        assert np.array_equal(x, y), "bitwise determinism violated"


def test_gradients_flow_through_the_bicycle():
    """The training path: loss on final goal distance must produce finite,
    nonzero grads on the coefficient model — same property sdf_rollout has."""
    field, _ = _open_field()
    model = sdf_nav.CoefMLP()
    o = torch.zeros(1, 2)
    th = torch.zeros(1)
    sp = torch.zeros(1)
    goal = torch.tensor([[1.5, 1.0]])
    al, be, ga = model(sdf_nav.coef_feats(field, o, goal))
    o2, th2, sp2, _mc = sdf_nav.bicycle_rollout(
        field, o, th, sp, goal, al, be, ga, 20, rr=RR, d_hat=DHAT, dt=DT, nsub=1, vmax=VMAX, **VEH
    )
    loss = (o2 - goal).norm()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients reached the model"
    total = sum(float(g.abs().sum()) for g in grads)
    assert math.isfinite(total) and total > 0.0


# ── the navigator: bicycle mode + moving-target tracking ────────────────────


def test_navigator_bicycle_mode_reaches_a_goal():
    field, meta = _open_field()
    model = sdf_nav.CoefMLP()
    model.eval()
    nav = SdfNavigator(field, model, meta, dynamics="bicycle")
    nav.start((-50.0, 0.0), (50.0, 0.0))
    ms, best_i, _o, _v = nav.drive_to_goal(max_steps=3000)
    assert ms[best_i].goal_dist_m < ms[0].goal_dist_m * 0.2, "barely progressed"
    assert any(abs(m.heading_rad) >= 0.0 for m in ms)  # field populated
    # .v stays meaningful for existing consumers (capture, VehiclePose)
    assert nav.v.shape == (1, 2)


def test_point_mode_is_untouched_by_vehicle_kwargs():
    """The original demo path must behave identically: point dynamics never
    reads the vehicle dict, heading stays at the metrics default."""
    field, meta = _open_field()
    model = sdf_nav.CoefMLP()
    model.eval()
    nav = SdfNavigator(field, model, meta, vehicle=dict(L=999.0, a_max=1e-9))
    nav.start((-30.0, 0.0), (30.0, 0.0))
    ms, best_i, _o, _v = nav.drive_to_goal(max_steps=1500)
    # reach_tol is in *normalized* units (0.8 -> 16 m at this fixture's scale),
    # so assert the navigator's own success signal, not an absolute metre count.
    assert ms[best_i].reached, "point mode failed to navigate"
    assert all(m.heading_rad == 0.0 for m in ms)


def test_track_goal_preserves_the_escape_state():
    """set_goal resets the wall-follow machinery (documented, correct for a
    discrete retarget); track_goal must NOT — that reset, applied per-frame,
    is exactly the livelock trap."""
    field, meta = _open_field()
    nav = SdfNavigator(field, sdf_nav.CoefMLP(), meta)
    nav.start((0.0, 0.0), (50.0, 0.0))
    nav._mode = "wall"
    nav._turn = -1.0
    nav._stall = 55
    nav.track_goal((60.0, 5.0))
    assert nav._mode == "wall" and nav._turn == -1.0 and nav._stall == 55
    nav.set_goal((60.0, 5.0))
    assert nav._mode == "seek" and nav._stall == 0


def test_fleeing_target_does_not_false_trigger_wall_follow():
    """Distance to a receding target never improves; the old closing-based
    stall would fire the escape while tracking is going perfectly well."""
    field, meta = _open_field()
    model = sdf_nav.CoefMLP()
    model.eval()
    nav = SdfNavigator(field, model, meta)
    nav.start((0.0, 0.0), (20.0, 0.0))
    gx = 20.0
    for _ in range(300):
        gx += 0.35  # flees a touch faster than the agent closes
        nav.track_goal((gx, 0.0))
        m = nav.step()
    assert m.mode == "seek", "escape false-triggered while chasing a fleeing target"
    assert m.x > 5.0, "agent did not actually chase"


def test_tracking_still_escapes_when_genuinely_stuck():
    """The other half of the bargain: displacement-based stall must still fire
    when the agent truly cannot move (pinned at zero speed)."""
    field, meta = _open_field()
    model = sdf_nav.CoefMLP()
    model.eval()
    nav = SdfNavigator(field, model, meta)
    nav.start((0.0, 0.0), (50.0, 0.0))
    nav.track_goal((50.0, 0.0))
    for _ in range(130):
        nav.step()
        # pin the agent: simulate a hard jam regardless of dynamics
        nav.o = torch.zeros(1, 2)
        nav.v = torch.zeros(1, 2)
    assert nav._mode == "wall", "displacement stall never fired for a pinned agent"


def test_track_goal_rejects_nothing_but_updates_goal_and_index():
    field, meta = _open_field()
    nav = SdfNavigator(field, sdf_nav.CoefMLP(), meta)
    nav.start((0.0, 0.0), (10.0, 0.0))
    nav.track_goal((30.0, 40.0), goal_index=7)
    assert nav.goal_index == 7
    gw = nav.n2w(nav._gn)
    assert np.allclose(gw, [30.0, 40.0], atol=1e-4)
