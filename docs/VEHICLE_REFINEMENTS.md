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
tuned = coef_train.train_bicycle(model=wide, friction=mu, steps=300, w_coll=3.0)
```

From the command line, `--rollout bicycle` selects this trainer and `--w-coll`
sets the dial:

```bash
python -m grl_snam.tools.coef_train --rollout bicycle --w-coll 3 --out coef_mlp.cvcnav
```

There is deliberately **no flag for friction or the footprint**: both need a
world rather than a scalar, so they stay programmatic. (`--rollout bicycle` used
to be accepted and then ignored on the torch backend — it always trained the
surrogate, printed a `reach_rate` and wrote a file, so the failure was
invisible. Fixed, with a test on the routing.)

`train_bicycle` exists because `train` integrates `sdf_rollout` — a holonomic
*point*, which has no actuator envelope for grip to limit, so mu there is an
input the loss has no reason to use. Through the vehicle, entering ice too fast
breaches the barrier, so the collision penalty can flow back to the coefficient
that set the approach speed.

### The loss had to be rescaled before any of that could work

The first version of this loop learned nothing, and the reason was the
objective, not the feature. `goal_dist + w_coll * mean(relu(rr - phi))` puts a
world-scale distance (~5.5) against a breach depth bounded by `rr` = 0.15 —
about **36×** apart on units alone. Worse, a breach depth is nonzero only for an
agent *already inside* geometry: measured over random free-space starts, **0.00%
of the batch**, so averaging divided the term by another ~100×. Collision ended
up **0.2–0.7% of the loss**. Training minimised goal distance alone, and since
slowing costs goal distance it was correctly learning *not* to slow down.

That is not a tuning problem — balancing it needed `w_coll` ≈ 3,700 on ice and
≈ 55,000 on dry, and no single constant serves both. Both terms are now O(1):
goal distance is divided by the world half-extent, and the penalty is a **margin
shortfall**, `relu(d_safe − clearance) / d_safe`, which is nonzero for anything
merely *near* geometry (21% of random starts, against 0.00%).

`clearance` is the `min` over the **body** when the vehicle carries a footprint,
so the quantity being optimised is the same one `FogScenario.body_clearance_m`
reports — what is trained and what is published cannot drift apart.

### The safety-for-reach trade, measured

With the loss balanced, `w_coll` spans the trade with small numbers. Three
training seeds each, 120 steps, widened 6-feature net, ice-bearing city;
penetration is per agent, and the raw per-seed column is there because at least
one row is not summarised honestly by its mean:

| `w_coll` | reach | penetration | final gap | raw penetration |
|---|---|---|---|---|
| *seed* | 14.6% | 1.56 | 3.524 | — |
| 0 | 18.9% ±0.2 | 2.05 ±0.17 | 3.845 | 2.25, 2.05, 1.83 |
| 1 | 20.1% ±0.2 | 1.54 ±0.01 | 3.525 | 1.55, 1.55, 1.53 |
| **3** | **20.1% ±0.2** | **1.52 ±0.03** | **3.524** | 1.48, 1.54, 1.54 |
| 10 | 11.6% ±5.8 | 0.57 ±0.70 | 4.773 | **0.06, 0.10, 1.56** |
| 30 | 5.2% ±2.7 | 0.09 ±0.06 | 5.522 | 0.05, 0.05, 0.17 |

Read it as three regimes, not a smooth curve:

* **`w_coll` = 1–3 is a reliable Pareto improvement** over the shipped seed:
  +5.5 points of reach, slightly *better* penetration, the same final gap, and a
  ±0.2 spread across seeds. This is the default, and it is not buying safety
  with reach — it is getting both.
* **`w_coll` = 10 is bimodal, not noisy.** Two seeds converge on a genuinely
  cautious policy (0.06, 0.10); the third never leaves the seed (1.56). The
  ±0.70 is a convergence coin-flip being averaged, and quoting the mean — or
  worse, the best seed — would misrepresent it. If you want this operating
  point, train several seeds and select, or use `w_coll` = 30.
* **`w_coll` = 30 reliably reaches penetration ~0.09** — a 94% cut — and costs
  most of the reach (14.6% → 5.2%).

`w_coll` is therefore a **dial, not a hyperparameter to tune away**. It selects
an operating point on a safety-for-reach curve, and every headline metric this
project publishes — `reach_rate`, `ScenarioResult.reached` — measures only the
reach side. A model trained at `w_coll` = 30 will look like a two-thirds
*regression* on every published number while being 17× safer. Report
`body_penetration_steps` and `body_clearance_m` alongside reach, or the trade is
invisible and the safer policy loses on the scoreboard.

### Training WITH the footprint: the answer for footprint scenarios

The loss uses the `min` over the body when `veh` carries a footprint, so what is
optimised is the same quantity `FogScenario.body_clearance_m` reports.

Three seeds, `w_coll` = 3, ice-bearing city. **Both arms are driven and scored
with the 3-disc footprint** — the only difference is what the training loss
looked at. Penetration is body penetration (min over discs):

| arm | reach | body penetration | raw |
|---|---|---|---|
| seed (untrained) | 15.1% | 1.52 | — |
| trained on the POINT | 19.4% | 1.60 | 1.75, 1.60, 1.46 |
| **trained on the BODY** | **20.7%** | **1.44** | 1.46, 1.47, 1.40 |

Training on the body metric is a Pareto improvement over training on the point:
more reach *and* less body penetration, a **10% cut**. Note also that the
point-trained policy is *worse than the untrained seed* on body penetration
(1.60 against 1.52) while looking better on reach — it is optimising a quantity
that is not the one being scored, and paying for it exactly where you would
expect. And it is four times steadier: ±0.03 against ±0.12, because a
point-trained policy's body penetration depends on which part of the body
happens to overhang, which varies by seed.

So: **train with the same footprint you deploy.** The effect is real but modest,
and it is worth having the honest size of it.

> **This table was published wrong once.** The first version reported a 42% cut
> (14.47 → 8.35) because it counted clearance below `+0.5 * rr` — the *stopping
> margin*, a near miss — while the `w_coll` table above counted clearance below
> `-0.5 * rr`, actual overlap. The looser threshold fires about ten times as
> often, so the two tables were not comparable even though both said
> "penetration". `coef_eval.pen_threshold` now names the definition and a test
> pins its sign. If you are comparing penetration numbers across documents,
> check they used the same ruler.

Note this is a training result, not a scenario result — training uses the
analytic SDF, which has sub-cell information. Scoring the metric *in a scenario*
still needs the lattice resolution described below.

### Why mu cannot earn its keep in THIS vehicle model

The wash above is not a tuning failure or a bad world. It is structural, and
three worlds were built before that became clear:

1. **Ice-band city** — ice covered the narrow streets, so a clearance-seeking
   policy already slowed inside it. mu explained nothing extra.
2. **One wall in open space** — training degenerated (clearance exceeded
   `d_safe` almost everywhere, the collision term went to ~0, and the net
   learned `be` = 19.5 / `ga` = 0.001, "max spring, no damping"), and the eval
   was actuator-saturated, so five seeds returned identical numbers.
3. **City with ice ONLY in open areas** — the honest separation, and it shows
   why the separation cannot pay. Sweeping how far the ice reaches, with
   penetration measured on the untrained policy:

   | ice coverage | ice confined to clearance ≥ | penetration |
   |---|---|---|
   | 40% | 1.232 | **0.00** |
   | 60% | 0.640 | **0.00** |
   | 86% | 0.316 | **0.00** |
   | 93% | 0.211 | 0.13 |
   | 100% | 0.105 | 1.30 |

**Ice causes no collisions at all until it reaches the low-clearance cells.**
The mechanism:

* the stopping governor already scales `v_stop` by **current** mu, so the
  vehicle can always stop for the grip it currently has;
* a kinematic bicycle **cannot skid** — low mu lengthens stopping distance, it
  never causes loss of control (see "Ice understeers" above);
* therefore mu only matters within about one stopping distance of geometry;
* and that is exactly where `phi` — the first coefficient feature — is already
  informative and the barrier is already active.

The only residual gain would come from slowing *earlier than clearance warrants*,
which costs reach everywhere else. That is the trade `w_coll` already exposes,
more directly and without a sixth feature.

**What would change this:** a dynamics where low mu causes loss of *control*
rather than longer stopping — a dynamic bicycle with lateral slip, where
entering a corner too fast on ice skids unrecoverably. The kinematic bicycle
cannot represent that by construction. **Do not run a material-aware training
campaign on this vehicle model**; build slip into the dynamics first, then the
feature has something to predict that the geometry does not already say.

**`grid` must be big enough to contain buildings.** `shrunk` scales the city's
rects with the grid: fraction of random starts inside the barrier band is 0.00
at `n=32` and 0.31 at `n=96`. Below ~64 the barrier is identically zero, `alpha`
scales nothing, its gradient is exactly 0, and training silently fits the goal
spring alone — it looks like it is working and teaches nothing about walls.

## Reporting these: every published metric is a reach metric

**The refinements trade reach for safety, and the metrics this project reports
are all reach metrics.** Quote reach alone and every one of them looks like a
regression — the footprint costs 45% -> 35%, an anticipating policy costs more
still. That is not the feature failing; it is the scoreboard measuring one side
of a trade.

Report all three, always:

| metric | what it is | why alone it misleads |
|---|---|---|
| `reached` / reach rate | did the agent arrive within the budget | the side the trade SPENDS |
| `body_penetration_steps` | ticks with any part of the BODY in truth | the side it BUYS |
| `body_clearance_m` | metres from the body surface to the nearest static obstacle | the margin, which the boolean cannot show |

`truth_penetration` is retained but is a **point test at the rear axle**. It
cannot see a nose clipping a corner while that point stays clear — precisely the
collision a body-shaped vehicle has and a disc-shaped one is assumed not to. A
footprint measured with it posts an unchanged safety number and a worse reach
number, i.e. reads as a pure loss. Existing recorded traces keep it so their
numbers stay comparable; **new results should quote the body metrics.**

`body_clearance_m` is the one to lead with in a deliverable. Two configurations
can both post zero penetrations while one runs half the margin, and only the
continuous metric shows it. A count answers "did we crash"; the margin answers
"how close were we", which is the question a safety claim actually rests on.

### The metric has a hard resolution requirement

A bitmap carries **no sub-cell information**. Its distance field steps in whole
cells — measured over a 200 m world:

| grid | cell | signed distance approaching a wall |
|---|---|---|
| 96 | 2.105 m | 6.316, 4.211, 2.105, −2.105 |
| 201 | 1.000 m | 3.0, 2.0, 1.0, −1.0 |
| 601 | 0.333 m | 1.0, 0.667, 0.333, −0.333 |

So a body of half-width ~0.21 m can only register an overlap once cells are at
or below the body size. **Against the city story's 2.1 m cells,
`body_penetration` is exactly `truth_penetration` under a different name** — it
cannot resolve a 0.42 m-wide vehicle, and no method built on that bitmap can.

`FogScenario.body_metric_resolvable()` answers this; **check it before quoting a
body number.** A vehicle-scale lattice (≲ 0.2 m cells for this vehicle) is
required, which is the same constraint the L-System export lattice hit — 0.5 m
is about the coarsest grid on which a door exists at all.

This is why the footprint has no adopted scenario yet: the worlds it would be
demonstrated in cannot yet measure its benefit. A vehicle-scale story is the
prerequisite, not the demo.

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

- **mu-in-features cannot pay in this vehicle model — structural, not a tuning
  failure.** See "Why mu cannot earn its keep" above: ice causes no collisions
  until it reaches cells the clearance feature already flags, because the
  governor scales stopping distance by current mu and a kinematic bicycle cannot
  skid. Three worlds were built to test it. The prerequisite for material-aware
  anticipation is a dynamics with lateral slip, not more training.
- **mu-in-features is a working seam, not a demonstrated win.** The first
  attempt failed for a reason that turned out to be the *loss*, not the feature
  — see "The loss had to be rescaled" above — and rescaling it produced a real,
  reliable gain over the seed (14.6% -> 20.1% reach at slightly better
  penetration). But an ablation against a blind 5-feature net attributes that
  gain entirely to the objective: with mu visible, 20.1% / 1.52; blind, 19.6% /
  1.58. The feature is plumbed, tested on both the torch and native paths, and
  provably output-identical at init via `widen_coef_mlp` — and still not shown
  to buy anything. Do not quote it as anticipation.

  The control is worth keeping in mind when reading that: a widened net,
  untuned, reproduces the seed exactly (14.6% / 1.56 / 3.524) in a live drive,
  so `widen_coef_mlp` really is the identity and any difference measured after
  fine-tuning is the fine-tuning.
- **The native binding is compile-validated, not runtime-validated** — see
  libcvc `docs/NAV_VEHICLE.md`. It ships when pycvc is next republished.
- **No scenario turns the footprint on by default**, and the blocker is now
  identified rather than merely noted: the worlds it would be demonstrated in
  **cannot measure its benefit**. At 2.1 m cells the body metric degenerates to
  the point test (see the resolution requirement above), so a vehicle-scale
  lattice is the prerequisite for a footprint scenario — not the scenario.
  Training for one, though, is unblocked and worthwhile today: see "Training
  WITH the footprint" above, which cuts body penetration 10% and is markedly
  steadier across seeds.
- **The native trainer cannot train against a footprint.** `diff::bike_veh` has
  no body fields, so `train_bicycle(veh=...)` is torch-only. Worth closing on
  the C++ side, though at a 10% effect it is not urgent.
