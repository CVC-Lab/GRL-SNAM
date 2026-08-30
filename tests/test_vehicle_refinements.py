"""The three optional vehicle refinements on ``sdf_nav.bicycle_rollout``:
multi-disc footprint, inner-wheel steering lock, and material-coupled grip.

Two classes of invariant are pinned here, and the first matters more than the
second.  **Legacy parity**: each feature must be bit-for-bit inert when left at
its ``None`` default, because every golden trace, every stored ``.cvcnav``
weight and the bit-identical C++ twin depend on it.  **Effect**: each feature
must actually do the thing it claims, measured as a behaviour rather than a
formula echo, because a parameter that is inert in BOTH directions would also
pass the first class of test.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sdf_nav  # noqa: E402
from grl_snam.material import FrictionField  # noqa: E402

RR, DHAT, DT, VMAX = 0.15, 0.35, 0.06, 0.9
VEH = dict(L=0.035, delta_max=0.6, a_max=1.5, a_lat_max=1.0, k_steer=0.8)
L = VEH["L"]
TRACK = 0.6 * L  # a car's track is ~0.6 of its wheelbase
BODY = (0.0, 0.5 * L, L)  # rear axle, mid, front axle
HALF_W = 0.012  # ~half-width + margin, vs the rr=0.15 single disc

BOUNDS = (-100.0, -100.0, 100.0, 100.0)
SCALE = 0.05
N = 201  # 1.0 m cells


def _field(occ=None):
    occ = np.zeros((N, N), bool) if occ is None else occ
    phi, nx_g, ny_g = sdf_nav.build_sdf(occ, BOUNDS, SCALE)
    return sdf_nav.SDFField(phi, nx_g, ny_g, BOUNDS, (0.0, 0.0), SCALE)


def _wall_at_col(c0):
    """Occupancy filling every column >= c0 -- a wall face at that column."""
    occ = np.zeros((N, N), bool)
    occ[:, c0:] = True
    return occ


def _state(x=0.0, y=0.0, th=0.0, sp=0.0, b=1):
    return (
        torch.tensor([[float(x), float(y)]] * b),
        torch.full((b,), float(th)),
        torch.full((b,), float(sp)),
    )


def _coef(b=1, al=1.0, be=3.0, ga=4.0):
    return (torch.full((b,), al), torch.full((b,), be), torch.full((b,), ga))


def _roll(field, o, th, sp, goal, steps=1, **kw):
    al, be, ga = _coef(o.shape[0])
    return sdf_nav.bicycle_rollout(
        field, o.clone(), th.clone(), sp.clone(), goal, al, be, ga, steps,
        rr=RR, d_hat=DHAT, dt=DT, nsub=2, vmax=VMAX, **{**VEH, **kw},
    )


# --------------------------------------------------------------------------
# 1. Legacy parity -- each feature inert at its default
# --------------------------------------------------------------------------


def test_single_disc_footprint_is_bit_identical_to_legacy():
    """One disc of radius rr at offset 0 IS the legacy single-sample path.

    This is the refactor gate: it proves the multi-disc branch reduces exactly,
    so any later divergence is the footprint doing its job, not a rewrite bug.
    """
    f = _field(_wall_at_col(150))
    o, th, sp = _state(x=2.0, sp=0.4)
    goal = torch.tensor([[3.0, 0.0]])
    base = _roll(f, o, th, sp, goal, steps=12)
    same = _roll(f, o, th, sp, goal, steps=12, body_offsets=(0.0,), body_rr=RR)
    for a, b in zip(base, same):
        assert torch.equal(a, b), "single-disc multi path diverged from legacy"


def test_unit_friction_is_bit_identical_to_legacy():
    """mu == 1 is the reference dry surface the constants are quoted against."""
    f = _field(_wall_at_col(150))
    o, th, sp = _state(x=2.0, sp=0.4)
    goal = torch.tensor([[3.0, 0.0]])
    mu1 = FrictionField.uniform((N, N), BOUNDS, (0.0, 0.0), SCALE, mu=1.0)
    base = _roll(f, o, th, sp, goal, steps=12)
    same = _roll(f, o, th, sp, goal, steps=12, friction=mu1)
    for a, b in zip(base, same):
        assert torch.equal(a, b), "uniform mu=1 perturbed the legacy trace"


def test_zero_track_width_leaves_the_steer_limit_alone():
    f = _field()
    o, th, sp = _state(sp=0.5)
    goal = torch.tensor([[0.0, 2.0]])
    base = _roll(f, o, th, sp, goal, steps=15)
    same = _roll(f, o, th, sp, goal, steps=15, track_width=0.0)
    for a, b in zip(base, same):
        assert torch.allclose(a, b, atol=1e-6), "t=0 changed the steer limit"


# --------------------------------------------------------------------------
# 2. Footprint
# --------------------------------------------------------------------------


def test_min_over_discs_binds_at_the_nose():
    """At equal disc radius, three discs must never report MORE clearance than
    one, and strictly less when the nose is the part nearest the wall."""
    f = _field(_wall_at_col(150))
    o, th, sp = _state(x=2.0, th=0.0, sp=0.3)  # heading +x, straight at the wall
    goal = torch.tensor([[3.0, 0.0]])
    _, _, _, clr_one = _roll(f, o, th, sp, goal, steps=1)
    _, _, _, clr_three = _roll(f, o, th, sp, goal, steps=1, body_offsets=BODY, body_rr=RR)
    assert clr_three.item() < clr_one.item(), "the nose disc saw nothing extra"
    assert clr_three.item() == pytest.approx(clr_one.item() - L, abs=2e-3), (
        "the binding disc should sit one wheelbase closer to the wall"
    )


def test_tight_footprint_recovers_clearance_the_disc_gives_away():
    """The reach lever: a body-width footprint reports the clearance a 4.3-
    wheelbase disc was throwing away, so gaps the disc refuses become passable."""
    f = _field(_wall_at_col(150))
    o, th, sp = _state(x=2.0, sp=0.3)
    goal = torch.tensor([[3.0, 0.0]])
    _, _, _, clr_disc = _roll(f, o, th, sp, goal, steps=1)
    _, _, _, clr_body = _roll(f, o, th, sp, goal, steps=1, body_offsets=BODY, body_rr=HALF_W)
    assert clr_body.item() > clr_disc.item()
    # rr - HALF_W recovered, minus the wheelbase the nose gives back
    assert clr_body.item() == pytest.approx(clr_disc.item() + (RR - HALF_W) - L, abs=2e-3)


def test_barrier_force_is_the_sum_over_discs():
    """Three COINCIDENT discs must be exactly one disc at three times the
    barrier coefficient. That identity is the precise statement of "the force
    is the SUM", and unlike a trajectory comparison it cannot be satisfied by
    accident -- which matters because the a_max clamp hides force differences
    whenever the goal spring is also saturating the actuator."""
    f = _field(_wall_at_col(150))
    o, th, sp = _state(x=2.30, sp=0.10)
    goal = torch.tensor([[2.30, 0.0]])  # goal at the start: isolate the barrier

    def run(offsets, al):
        return sdf_nav.bicycle_rollout(
            f, o.clone(), th.clone(), sp.clone(), goal,
            torch.full((1,), al), torch.zeros(1), torch.zeros(1), 3,
            rr=RR, d_hat=DHAT, dt=DT, nsub=2, vmax=VMAX,
            body_offsets=offsets, body_rr=HALF_W, **VEH,
        )

    for a, b in zip(run((0.0,), 0.15), run((0.0, 0.0, 0.0), 0.05)):
        assert torch.allclose(a, b, atol=1e-6), "discs are not summing"


# --------------------------------------------------------------------------
# 3. Inner-wheel steering lock
# --------------------------------------------------------------------------


def test_inner_wheel_lock_shrinks_the_virtual_steer_limit():
    eff = sdf_nav.ackermann_delta_max(L, VEH["delta_max"], TRACK)
    assert eff < VEH["delta_max"]
    assert eff == pytest.approx(0.5157, abs=1e-3)  # 14% less steer
    r_min_bicycle = L / math.tan(VEH["delta_max"])
    r_min_true = L / math.tan(eff)
    assert r_min_true == pytest.approx(r_min_bicycle + 0.5 * TRACK, rel=1e-9)
    assert r_min_true / r_min_bicycle == pytest.approx(1.205, abs=5e-3)  # 20% wider


def test_locked_vehicle_turns_on_a_wider_radius():
    """The limit has to bind on the TRAJECTORY, not just in the constant.

    Measured as arc/dtheta rather than heading-after-N-steps: at low speed both
    vehicles finish the turn and the headings converge, while above
    ``sp^2 > a_lat L / tan(delta_max)`` the lateral cap binds first and hides
    the lock entirely. The radius is the quantity that separates them wherever
    the steer limit is what is actually binding.
    """
    f = _field()
    o, th, sp = _state(sp=0.22)
    goal = torch.tensor([[0.0, 0.35]])

    def radius(**kw):
        oT, thT, _, _ = _roll(f, o, th, sp, goal, steps=6, **kw)
        return float(torch.linalg.norm(oT - o)) / abs(thT.item())

    assert radius(track_width=TRACK) > radius(), "lock did not widen the arc"


def test_wheel_angles_bracket_the_virtual_angle():
    dl, dr = sdf_nav.ackermann_wheel_angles(0.3, L, TRACK)
    assert abs(dl) > 0.3 > abs(dr), "inner wheel must out-steer the virtual one"
    # both wheel axes must meet the rear axle line at ONE centre
    r_from_left = L / math.tan(dl) + 0.5 * TRACK
    r_from_right = L / math.tan(dr) - 0.5 * TRACK
    assert r_from_left == pytest.approx(r_from_right, rel=1e-9)
    assert sdf_nav.ackermann_wheel_angles(0.0, L, TRACK) == (0.0, 0.0)


def test_wheel_angles_are_batched_and_differentiable():
    d = torch.tensor([0.2, -0.2, 0.0], requires_grad=True)
    dl, dr = sdf_nav.ackermann_wheel_angles(d, L, TRACK)
    assert dl.shape == d.shape
    assert torch.sign(dl[0]) == 1 and torch.sign(dl[1]) == -1
    (dl.sum() + dr.sum()).backward()
    assert torch.isfinite(d.grad).all()


# --------------------------------------------------------------------------
# 4. Material-coupled grip
# --------------------------------------------------------------------------


def _ice(mu=0.15):
    return FrictionField.uniform((N, N), BOUNDS, (0.0, 0.0), SCALE, mu=mu)


def test_ice_understeers_rather_than_slowing_down():
    """The headline behaviour, and NOT the intuitive one.

    A kinematic bicycle has no lateral velocity state, so it cannot fishtail.
    On ice ``d_cap = atan(mu a_lat L / v^2)`` collapses, the vehicle simply
    cannot bend its path, and it ploughs on toward the outside of the turn --
    which means it ends up FASTER than the dry vehicle, not slower, because the
    dry one brakes hard for a corner it is actually taking. Asserting "ice is
    slower" would pass in some regimes and is the wrong physics; the invariant
    that always holds is that ice tracks the carrot worse.
    """
    f = _field()
    o, th, sp = _state(sp=0.8)
    goal = torch.tensor([[0.0, 0.5]])
    _, th_dry, sp_dry, _ = _roll(f, o, th, sp, goal, steps=20)
    _, th_ice, sp_ice, _ = _roll(f, o, th, sp, goal, steps=20, friction=_ice())
    assert abs(th_ice.item()) < 0.5 * abs(th_dry.item()), "ice did not understeer"
    assert sp_ice.item() > sp_dry.item(), "ice should carry speed, not shed it"


def test_ice_lengthens_the_stopping_distance():
    """The governor's v_stop = sqrt(2 mu a_max (d - rr/2)) collapses with grip,
    so a vehicle that meets a wall on ice gets closer before it can stop."""
    f = _field(_wall_at_col(150))
    o, th, sp = _state(x=1.6, th=0.0, sp=VMAX)  # driving hard at the wall
    goal = torch.tensor([[4.0, 0.0]])  # goal behind the wall: keeps the throttle on
    _, _, _, clr_dry = _roll(f, o, th, sp, goal, steps=40)
    _, _, _, clr_ice = _roll(f, o, th, sp, goal, steps=40, friction=_ice())
    assert clr_ice.item() < clr_dry.item(), "ice did not shorten the stop margin"


def test_grip_couples_steering_and_braking_together():
    """Ice is dangerous because turning AND braking fail at once. Wiring only
    one of the two limits would leave the vehicle an escape via the other, so
    both couplings are pinned here rather than trusting the shared ``mu``."""
    # steering half: cannot bend the path
    f = _field()
    o, th, sp = _state(sp=0.8)
    goal = torch.tensor([[0.0, 0.5]])
    _, th_dry, _, _ = _roll(f, o, th, sp, goal, steps=20)
    _, th_ice, _, _ = _roll(f, o, th, sp, goal, steps=20, friction=_ice())
    assert abs(th_ice.item()) < abs(th_dry.item())
    # braking half: cannot stop in the clearance it has
    fw = _field(_wall_at_col(150))
    ow, thw, spw = _state(x=1.6, sp=VMAX)
    gw = torch.tensor([[4.0, 0.0]])
    _, _, _, clr_dry = _roll(fw, ow, thw, spw, gw, steps=40)
    _, _, _, clr_ice = _roll(fw, ow, thw, spw, gw, steps=40, friction=_ice())
    assert clr_ice.item() < clr_dry.item()


def test_grip_is_sampled_at_the_vehicle_not_globally():
    """A patch of ice must bite only where the vehicle is standing on it --
    otherwise this is a global constant wearing a raster's clothes. Both runs
    use the SAME carrot geometry relative to the vehicle, so the only variable
    left is which surface it is on."""
    mu = FrictionField.uniform((N, N), BOUNDS, (0.0, 0.0), SCALE, mu=1.0)
    mu.stamp(0, N, 0, 100, 0.15)  # ice only on the world's left half (x < 0)
    assert mu.sample(torch.tensor([[-2.0, 0.0]])).item() < 0.2
    assert mu.sample(torch.tensor([[2.0, 0.0]])).item() == pytest.approx(1.0)
    f = _field()

    def yaw(x):
        o, th, sp = _state(x=x, sp=0.8)
        _, thT, _, _ = _roll(f, o, th, sp, torch.tensor([[x, 0.5]]), steps=20, friction=mu)
        return abs(thT.item())

    assert yaw(-2.0) < 0.5 * yaw(2.0), "the ice patch bit everywhere, or nowhere"


# --------------------------------------------------------------------------
# 5. End-to-end wiring
# --------------------------------------------------------------------------


def _navigator(**vehicle):
    from grl_snam.nav import SdfNavigator

    meta = dict(
        scale=SCALE, center=(0.0, 0.0), region=100.0,
        rr=RR, d_hat=DHAT, dt=DT, vmax=VMAX, nsub=2,
        bounds=BOUNDS,
    )
    model = sdf_nav.CoefMLP()
    nav = SdfNavigator(_field(_wall_at_col(150)), model, meta, dynamics="bicycle",
                       vehicle=vehicle or None)
    nav.o = torch.tensor([[2.0, 0.0]])
    nav.th, nav.sp = torch.zeros(1), torch.full((1,), 0.3)
    return nav


def test_navigator_defaults_are_inert():
    """The refinements ride in VEHICLE_DEFAULTS and an attribute; left unset,
    SdfNavigator must drive exactly as it did before they existed."""
    from grl_snam.nav import SdfNavigator

    assert SdfNavigator.VEHICLE_DEFAULTS["body_offsets"] is None
    assert SdfNavigator.VEHICLE_DEFAULTS["track_width"] is None
    a, b = _navigator(), _navigator()
    assert a.friction is None
    torch.manual_seed(0)
    for _ in range(5):
        a.step()
    torch.manual_seed(0)
    for _ in range(5):
        b.step()
    assert torch.equal(a.o, b.o) and torch.equal(a.th, b.th)


def test_navigator_threads_footprint_and_grip_into_the_drive():
    plain, tuned = _navigator(), _navigator(body_offsets=BODY, body_rr=HALF_W, track_width=TRACK)
    tuned.friction = _ice(0.3)
    for _ in range(8):
        plain.step()
        tuned.step()
    assert not torch.equal(plain.o, tuned.o), "navigator dropped the refinements"


def test_native_drive_refuses_the_unported_refinements():
    """Silently ignoring a footprint in C++ while honouring it in torch is the
    exact "fast twin that moves differently" failure; the parity gates cannot
    catch it because they feed both paths the same params."""
    nav_native = pytest.importorskip("grl_snam.nav_native")
    params = dict(
        rr=RR, d_hat=DHAT, dt=DT, vmax=VMAX, **VEH,
        nsub=2, allow_reverse=True, body_offsets=BODY, body_rr=HALF_W,
    )
    with pytest.raises(NotImplementedError, match="body_offsets"):
        nav_native.bicycle_rollout(
            None, None, None, None, None, None, None, None,
            bounds=BOUNDS, center=(0.0, 0.0), scale=SCALE, params=params,
        )


# --------------------------------------------------------------------------
# 6. The training path must survive all three
# --------------------------------------------------------------------------


def _grad_run(x, sp0, al0, be0, ga0, steps, **kw):
    f = _field(_wall_at_col(150))
    o, th, sp = _state(x=x, sp=sp0)
    al = torch.tensor([al0], requires_grad=True)
    be = torch.tensor([be0], requires_grad=True)
    ga = torch.tensor([ga0], requires_grad=True)
    oT, _, spT, _ = sdf_nav.bicycle_rollout(
        f, o, th, sp, torch.tensor([[x, 0.0]]), al, be, ga, steps,
        rr=RR, d_hat=DHAT, dt=DT, nsub=2, vmax=VMAX, **kw, **VEH,
    )
    (oT.sum() + spT.sum()).backward()
    return al, be, ga


def test_gradients_still_flow_with_every_refinement_on():
    """Self-supervised training backprops through the rollout; a feature that
    silently detached the graph would look fine everywhere else in this file.

    Run with the goal spring and damping off and a small ``al``, because a
    saturated ``a_max`` clamp legitimately zeroes the barrier gradient -- that
    is the clamp doing its job, and it would mask a genuinely dead graph.
    """
    al, be, ga = _grad_run(
        2.30, 0.10, 0.05, 0.0, 0.0, 4,
        body_offsets=BODY, body_rr=HALF_W, track_width=TRACK, friction=_ice(0.4),
    )
    for name, t in (("al", al), ("be", be), ("ga", ga)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"{name} grad broke"
    assert al.grad.abs().item() > 0, "barrier coefficient got no gradient"


def test_no_nan_gradient_when_the_footprint_reaches_the_stop_margin():
    """Regression: sqrt'(0) is infinite and clamp_min's subgradient below its
    floor is 0, so ``v_stop`` evaluated 0 * inf = NaN and poisoned every
    coefficient gradient. The legacy rr-disc is fat enough that the vehicle is
    turned away long before it reaches d = rr/2; a body-width footprint gets
    there in ordinary driving, which is what exposed this."""
    for x in (2.25, 2.35, 2.40):
        al, be, ga = _grad_run(
            x, 0.4, 1.0, 3.0, 4.0, 8,
            body_offsets=BODY, body_rr=HALF_W, track_width=TRACK, friction=_ice(0.4),
        )
        for name, t in (("al", al), ("be", be), ("ga", ga)):
            assert torch.isfinite(t.grad).all(), f"NaN {name} gradient at x={x}"


# --------------------------------------------------------------------------
# 7. Grip as a coefficient feature (anticipation)
# --------------------------------------------------------------------------


def test_coef_feats_defaults_to_five_columns():
    """Every trained .cvcnav weight file is a 5-input net; the default must not
    move, or all of them stop loading."""
    f = _field(_wall_at_col(150))
    o = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    goal = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    feat = sdf_nav.coef_feats(f, o, goal)
    assert feat.shape == (2, 5)
    assert torch.equal(feat, sdf_nav.coef_feats(f, o, goal, friction=None))


def test_coef_feats_appends_sampled_grip():
    f = _field(_wall_at_col(150))
    o = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    goal = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    mu = _ice(0.25)
    feat = sdf_nav.coef_feats(f, o, goal, friction=mu)
    assert feat.shape == (2, 6)
    assert torch.equal(feat[:, :5], sdf_nav.coef_feats(f, o, goal))
    assert torch.allclose(feat[:, 5], mu.sample(o))


def test_widened_net_is_output_identical_to_the_trained_one():
    """The whole point: widening must NOT be a retrain. A fresh 6-input init
    lands outside the basin the shipped seed occupies (base training is known to
    collapse reach against it), so the mu column starts at zero and the widened
    net computes exactly the old function until fine-tuning moves it."""
    torch.manual_seed(0)
    base = sdf_nav.CoefMLP()
    wide = sdf_nav.widen_coef_mlp(base)
    assert wide.in_dim == 6 and base.in_dim == 5
    feat5 = torch.randn(16, 5)
    for mu in (0.0, 0.15, 1.0, 7.3):  # any mu at all: the column is dead
        feat6 = torch.cat([feat5, torch.full((16, 1), mu)], -1)
        for a, b in zip(base(feat5), wide(feat6)):
            assert torch.equal(a, b), f"widening changed the function at mu={mu}"


def test_widened_net_can_still_learn_from_grip():
    """Zeroed, not detached — fine-tuning has to be able to discover a use for
    mu, otherwise the widening is decorative."""
    torch.manual_seed(0)
    wide = sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP())
    feat = torch.randn(8, 6)
    feat[:, 5] = 0.3
    sum(x.sum() for x in wide(feat)).backward()
    g = wide.net[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g[:, 5].abs().sum().item() > 0, "no gradient reaches the mu column"


def test_widen_rejects_an_already_widened_net():
    with pytest.raises(ValueError, match="in_dim=6"):
        sdf_nav.widen_coef_mlp(sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP()))


def test_train_accepts_a_widened_model_and_rejects_a_mismatch():
    """The fine-tune seam: coef_train.train must be able to CONTINUE from a
    widened net (training from scratch is what collapses reach), and must refuse
    a friction field the model has no column for rather than silently dropping
    it."""
    from grl_snam.tools import coef_train

    wide = sdf_nav.widen_coef_mlp(sdf_nav.CoefMLP())
    out = coef_train.train(steps=1, horizon=2, n=8, grid=32, window=2, model=wide,
                           friction=_ice(0.5))
    assert out is wide and wide.in_dim == 6
    with pytest.raises(ValueError, match="widen_coef_mlp"):
        coef_train.train(steps=1, horizon=2, n=8, grid=32, window=2, friction=_ice(0.5))
