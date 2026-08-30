# Vehicle refinements — footprint, steering lock, grip

Four optional knobs on `sdf_nav.bicycle_rollout`. Every one is **inert at its
default**, bit-for-bit, so an untouched caller gets the trajectory it always got
and every stored `.cvcnav` weight keeps working. The C++/CUDA twin is
`cvc::nav` — see libcvc `docs/NAV_VEHICLE.md`.

## First: do not write an "Ackermann rollout"

The kinematic bicycle **already is** the exact kinematic reduction of an
Ackermann axle. `delta` is the *virtual* centre-wheel angle that the inner and
outer wheel angles bracket:

    cot(delta_outer) - cot(delta_inner) = track / L

Ackermann geometry decides what each *wheel* does; it does not change where the
*body* goes. A separate "Ackermann rollout" would produce identical
trajectories. Per-wheel angles are a **rendering** quantity —
`sdf_nav.ackermann_wheel_angles(delta, L, track_width)` returns them for posing
wheel pivots in a viewer. Do not feed them back into the rollout.

What actually differs from a point agent is the footprint, the inner-wheel lock,
and grip.

## The knobs

| parameter | default | what it does |
|---|---|---|
| `body_offsets`, `body_rr` | `None` | multi-disc footprint: clearance = MIN over discs, barrier force = SUM |
| `body_gain` | `1.0` | scales that summed force — **set it to `1/len(body_offsets)`** |
| `track_width` | `None` | inner-wheel steering lock |
| `friction` | `None` | a `material.FrictionField`; `mu = 1` is the reference dry surface |

### Footprint

```python
L = 0.035
o, th, sp, clr = sdf_nav.bicycle_rollout(
    field, o, th, sp, goal, al, be, ga, steps,
    rr=rr, d_hat=d_hat, dt=dt, vmax=vmax,
    body_offsets=(0.0, L / 2, L),   # rear axle, mid, front axle
    body_rr=0.012,                  # ~half-width, normalized
    body_gain=1 / 3,                # 1/len(body_offsets) — see below
    **VEHICLE,
)
```

Offsets are longitudinal, in normalized units, measured along the heading from
the rear axle (the rollout's reference point). Clearance becomes the min over
discs, so the nose is pushed off a wall the rear axle cannot see, and the
governor / creep / nose-blocked margins switch from `rr` to `body_rr` with it —
leaving those sized for the fat disc is what made a tighter footprint buy
nothing.

**`body_gain` is not optional in practice.** The summed barrier is a K-times
gain on `alpha`, which the coefficient net learned for ONE sample point.
Uncorrected the vehicle does not break, it becomes *timid*: more standoff, fewer
collisions, longer to arrive — so it misses any fixed budget. City story, 5
seeds × 4 agents:

| arm | reach @700 | reach @1600 | pen/agent | clearance |
|---|---|---|---|---|
| disc 0.150 (legacy) | 45% | 75% | 2.9 | 2.92 m |
| fp3 0.150, gain 1 | 0% | 30% | 2.8 | 3.65 m |
| **fp3 0.150, gain 1/3** | **35%** | **60%** | **2.8** | **3.65 m** |
| fp3 0.075, gain 1/3 | 50% | 60% | 7.2 | 2.59 m |

Use **full radius with gain `1/n`**: it keeps the lower collision rate and ~0.7 m
more standoff. The last row is the trap — smaller discs look best at a tight
budget, but by 1600 ticks they reach the same as full radius while carrying 2.5×
the collisions. That extra reach was borrowed from safety, not earned.

### Steering lock

```python
..., track_width=0.6 * L, ...
```

`delta` is the virtual angle; on a real axle the **inner** wheel reaches the
mechanical lock first, so the achievable virtual angle is
`atan(L / (L/tan(delta_max) + t/2))` — `sdf_nav.ackermann_delta_max()` computes
it. At `t = 0.6 L` that is **14% less steer and a 20% larger `R_min`**: a real
constraint, not a rounding correction.

### Grip

```python
from grl_snam.material import FrictionField

mu = FrictionField.uniform(occ.shape, bounds, center, scale, mu=1.0)
mu.stamp(r0, r1, c0, c1, 0.15)          # an ice patch
..., friction=mu, ...
```

`mu = 1` is the reference dry surface the vehicle constants are already quoted
against, so a uniform-1 field is exactly the pre-grip rollout. Rough table: ice
0.15, mud 0.35, grass 0.45, gravel 0.6, wet 0.75, dry 1.0.

Grip is a **separate one-plane sampler**, not a seventh `MaterialField` channel,
because risk and grip are independent surface properties — ice is innocuous to
look at and lethal to drive on, rubble is the reverse. Folding them together
would make the one case worth simulating unrepresentable, and it would cost the
bit-identical C++ `material_sample` twin.

Both actuator limits are grip-limited, so `a_max` **and** `a_lat_max` scale with
mu together. That coupling is what makes ice read as ice: the corner cap and the
stopping governor collapse at the same moment.

## Two behaviours that look like bugs and are not

**Ice understeers, so a free turn gets FASTER, not slower.** A kinematic bicycle
has no lateral velocity state and cannot skid. On ice `d_cap` collapses, the
vehicle simply cannot bend its path, and it ploughs on toward the outside of the
turn — while the *dry* vehicle brakes for a corner it is actually taking.
"Ice is slower" is true only when the corner is forced (a wall, a convoy slot).
Do not assert it as an invariant.

**The DYNAMICS still cannot anticipate, by design.** `a_max`/`a_lat_max` scale
by mu at the current pose, so entering ice at speed means the stopping governor
already budgeted for grip the vehicle no longer has. That is the intended
failure mode of the plant. Anticipation has to come from the coefficients, which
is what the sixth feature is for — and that feature looks *ahead*, not
underfoot; see below.

## Anticipation without invalidating trained weights

`coef_feats(field, o, goal, friction=mu)` appends a **sixth** feature: the
**worst grip between here and the carrot** — `min` over a short probe out to
`mu_lookahead` (default 0.3 normalized, never past the carrot).

It looks ahead rather than underfoot for a reason that took a failed experiment
to see. The first version sampled mu at `o`, which reports the surface the
vehicle is standing ON and never the one it is about to hit — so it could not
support anticipation *even in principle*, and training predictably learned
nothing from it. The dynamics already react to current mu; what the coefficients
need is the part the dynamics cannot see yet.

Adding a sixth feature would normally invalidate every trained `.cvcnav` file,
and training this policy from a fresh init is known to collapse reach against
the shipped seed. So do not retrain from scratch:

```python
wide = sdf_nav.widen_coef_mlp(trained_5_feature_net)   # in_dim 5 -> 6
```

The new mu column is **zero**, so the widened net computes exactly the same
function of the original five features — identical outputs, bit-for-bit, for any
mu. Fine-tuning therefore starts inside the basin that already works, and the
only thing it can do is discover a use for mu.

```python
from grl_snam.tools import coef_train
tuned = coef_train.train_bicycle(model=wide, friction=mu, steps=300)
```

`train_bicycle` exists because `train` integrates `sdf_rollout` — a holonomic
*point*, which has no actuator envelope for grip to limit, so mu there is an
input the loss has no reason to use. Through the vehicle, entering ice too fast
breaches the barrier, so the intended mechanism was that the existing collision
penalty would flow back to the coefficient that set the approach speed.

**That was measured and it does not happen** — see Status below. Treat this loop
as a working seam, not a result; a loss term that rewards slowing before a mu
transition is probably what it needs.

**`grid` must be big enough to contain buildings.** `shrunk` scales the city's
rects with the grid: fraction of random starts inside the barrier band is 0.00
at `n=32` and 0.31 at `n=96`. Below ~64 the barrier is identically zero, `alpha`
scales nothing, its gradient is exactly 0, and training silently fits the goal
spring alone — it looks like it is working and teaches nothing about walls.

## Wiring into the navigator and squads

The three *vehicle* parameters ride in `SdfNavigator.VEHICLE_DEFAULTS`, so a
`Squad`'s batched rollout picks them up with no extra plumbing:

```python
nav = SdfNavigator(field, model, meta, dynamics="bicycle",
                   vehicle=dict(body_offsets=(0.0, L/2, L), body_rr=0.012,
                                body_gain=1/3, track_width=0.6*L))
nav.friction = mu          # a WORLD field, so it is an attribute, not a vehicle param
```

`SdfNavigator` reads the feature width off the model, so attaching a friction
field to an ordinary 5-input net still feeds it exactly five features.

The native path carries them too — `nav_native.bicycle_rollout` forwards
`body_offsets` / `body_rr` / `body_gain` / `track_width` / `friction` from
`params`. It only widens the call when one is actually set, so callers on a
pycvc predating the binding are unaffected; when one *is* set against an old
binding the `TypeError` is deliberate, because a native drive that quietly
ignores a constraint is the failure this whole port exists to prevent.

## Testing

`tests/test_vehicle_refinements.py` (28 tests) pins two classes of invariant,
and the first matters more:

- **Inertness.** Each knob must be bit-for-bit inert at its default —
  `torch.equal`, not a tolerance — because golden traces and stored weights
  depend on the drive not moving. A one-disc footprint IS the legacy sample; a
  uniform `mu=1` field IS the legacy actuator envelope.
- **Effect.** Each knob must actually do what it claims, asserted as a
  *behaviour*. A parameter inert in both directions would pass the first class
  of test just as well.

Some assertions are subtler than they look, and the comments say why:

- The `body_gain` identity uses `al` **below** the `a_max` clamp. At the clamp,
  1× and 3× the barrier give an identical trajectory and the comparison is
  vacuous.
- It compares **position**, not speed: with no goal spring the speed is set by
  the governor alone and matches either way, for reasons unrelated to the knob.
- The steering-lock test measures **arc/dtheta**, not heading after N steps. At
  low speed both vehicles finish the turn and the headings converge; above
  `sp² > a_lat·L/tan(delta_max)` the lateral cap binds first and hides the lock.
- There is a NaN-gradient regression test: a tighter footprint lets the vehicle
  actually reach `d = rr/2`, where `sqrt'(0) × clamp'(0) = 0 × inf` used to
  poison every coefficient gradient.

```bash
python -m pytest tests/test_vehicle_refinements.py -q
```

The native parity test **skips** rather than passes on a pycvc that predates the
binding. That is deliberate: a green tick from an absent feature reads as
coverage this repo does not have.

Before pushing, run the linters CI runs — they are not installable here
(`black`/`ruff` are absent from `dbg-deps` and from the cvcpkg catalog for this
platform tuple), so use a throwaway venv:

```bash
python -m black --check grl_snam tests
ruff check . --exclude _libcvc-deps
```

## Status

Done and measured: all four knobs, on CPU, CUDA, and through the SWIG binding;
`widen_coef_mlp`'s equivalence; the `body_gain` numbers above.

Not done, and worth knowing before quoting this feature:

- **mu-in-features is a working seam, not a demonstrated win — this was tried
  and it did not pay off.** A widened net fine-tuned through the vehicle on an
  ice-bearing city, against the seed it was widened from:

  | model | reach | pen/agent | final gap |
  |---|---|---|---|
  | A seed (5-feature, blind) | 14.6% | 1.56 | 3.524 |
  | B widened, untuned | 14.6% | 1.56 | 3.524 |
  | C widened + grip-tuned | 18.8% | 1.71 | 3.845 |

  B matching A exactly is the control working — `widen_coef_mlp` really is the
  identity at init, confirmed in a live drive and not just on synthetic
  features. But C's extra reach comes with *worse* penetration and a *worse*
  final gap, and 20 training steps produce the same numbers as 250. Across
  lr 1e-3 / 2e-4 / 5e-5 the loss lands at 4.986 / 4.988 / 4.997 — a 20x range
  with no effect, which says the loss is insensitive to these parameters rather
  than that the tuning is off; smaller lr just stays nearer the seed.

  So the honest reading is that the collision penalty does **not** by itself
  teach the policy to slow before a mu transition. Making it learn that probably
  needs a term that rewards it directly — penalising speed carried into low mu —
  rather than hoping the existing penalty finds it. The plumbing, the
  equivalence proof and the training seam are all in place for whoever tries.
- **The native binding is compile-validated, not runtime-validated** — see
  libcvc `docs/NAV_VEHICLE.md`. It ships when pycvc is next republished.
- **No scenario turns the footprint on by default.** It is available, not
  adopted; the `body_gain` table is the evidence you would need to justify
  adopting it.
