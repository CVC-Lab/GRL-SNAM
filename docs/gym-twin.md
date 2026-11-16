# The gym twin: training through libcvc's world gym

`grl_snam.gym_twin` is a differentiable twin of libcvc's world gym (`pycvc.WorldGym`,
`cvc::ariadne::WorldGym`). The gym drives a world document's vehicles headless on Jolt or on the
dynamic models, with no differentiation. The twin is the same world as a torch model:
- the gym's four-wheel vehicle model;
- the deployed drive's control law;
- the gym's own signed distance and ground.

A policy trains by backpropagating through the twin and is judged by driving the gym itself. This
is the pattern libcvc's P8 used for the Jolt vehicle (NAV-VEHICLE-PHYSICS-ROADMAP): Jolt stays the
plant for evaluation, and a differentiable model stands beside it.

## What it reads

It reads everything from the gym, through libcvc's twin exports (libcvc `WorldGym::fields()` and
`WorldGym::plants()`; ARIADNE.md, "A differentiable twin"):

| From the gym | In the twin |
|---|---|
| `fields()`: the navigator's grid, its signed distance (m) and gradient, the ground's height and grip | an `SDFField` in nav's normalized frame for the drive and `coef_feats`; a bilinear `(mu, height)` ground for the plant |
| `fields()["navigator"]`: nav's scale, tick, arrival radius, its vehicle | the drive's settings (`bike_control`'s `rr`, `d_hat`, `vmax`, `L`, `delta_max`, `a_max`, `a_lat_max`, `k_steer`) |
| `plants()`: each agent's four-wheel model in SI (`Dyn4Params`), its step limits, substeps, ground reference, state | the plant (`dyn4_plant`), batched over the agents |
| `observe()`: the goals | the drive's goals |
| `plants()`' `plan` (with `plant_arm=True`): the limits nav@1's plant arm plans each command with | the vehicle's envelope, wheelbase and footprint discs in the drive's settings; `sdf_nav.PlantArm` (latency, `a_brake`, the steering's slew and lock); which agents the next tick parks |

A tick of the twin (`GymTwin.step`) is the drive's commands from the state and the policy's
coefficients (`bike_control`, normalized). It is followed by the plant's substeps under those
commands in SI, as libcvc's `dyn_4w_plant` holds nav's command over its substeps.

```python
from grl_snam.gym_twin import GymTwin, mlp_policy, progress_loss, clearance_loss, drive_gym
from grl_snam.sdf_nav import CoefMLP

gym = pycvc.WorldGym("world.ari", agents=[{"plant": "dynamic4w"}] * 3)
gym.reset(seed)
twin = GymTwin(gym)                    # the world and the agents, as they are now
model = CoefMLP().double()
states, cmds = twin.rollout(mlp_policy(model), twin.state(gym), T=30)
loss = progress_loss(twin, states) + clearance_loss(twin, states)
loss.backward()                        # into the policy, through the plant and the drive

drive_gym(gym, twin, mlp_policy(model), ticks=600)   # the gym judges it
```

## Where it is the gym, and where it is not

| | |
|---|---|
| **The plant on `dynamic4w`** | The gym's own model. Away from a collision, `GymTwin.replay` of a tick from the gym's state, with the command the gym's plant took in each substep, is the gym's next state to 1e-9 (gate 1). libcvc checks the same in C++ bit for bit |
| **The plant on `jolt`** | The four-wheel model with the vehicle's own numbers, its state read back from the chassis; the fitted model's residual applies (gate 2's numbers) |
| **The driver** | The gym's dynamic plants take nav's acceleration through the Jolt plant's driver (libcvc's `dyn_driver`), which has no twin. The twin applies the drive's acceleration raw |
| **Contact** | The gym's chassis collide in Jolt. The four-wheel twin has no contact port (`dyn_contact` couples to the single-track model only), so a rollout into a wall is not the gym's. The drive's barrier and `clearance_loss` keep rollouts off walls |
| **Grade** | The gym's four-wheel model runs `grade_force`, which has no torch twin. The twin turns it off, so it is exact on flat ground only |
| **The controller** | The gym's navigator (nav@1) runs its carrot FSM, planner and governors around this control law. The twin takes the goal as given and the coefficients from the policy. Judged in the gym, a policy drives through the twin's controller (`drive_gym`), not nav@1 |
| **The plant arm** | On a vehicle plant, nav@1's drive gives its command through libcvc's plant arm (`plant_drive`), after the control law. `GymTwin(gym, plant_arm=True)` has it (gate 5); the default is the control law alone, as the training gates ran |
| **Other plants** | The bicycle and the single-track model are refused for now |

## Gates

Run with `pytest tests/test_gym_twin.py` (skipped without a pycvc that has the twin exports, or
without Jolt). Measured 2026-11-14 on prettyhatemachine against libcvc's branch with the exports;
torch 2.14 CPU, float64.

1. **Replay** (`test_gate1_replay_is_the_gyms_tick`): 30 ticks of a two-Humvee `dynamic4w` gym on
   a flat square, every tick replayed from the gym's state with the command its plant took. Each
   tick is the gym's to 1e-9 relative. **Passes.**
2. **Residuals** (`test_gate2_the_twins_own_tick_is_close_off_contact`, `tick_residuals`): the
   twin's own tick (its controller, then its plant with no driver) against the gym's for the same
   commands. 400 ticks on open flat ground, the constant coefficients driving, no collisions:

   | plant | position (m/tick) median / p95 / max | heading (rad) median / max | speed (m/s) median / max |
   |---|---|---|---|
   | `dynamic4w` | 0.0001 / 0.0002 / 0.0004 | 0.00000 / 0.00005 | 0.0027 / 0.0159 |
   | `jolt` | 0.0022 / 0.0030 / 0.0033 | 0.00005 / 0.00056 | 0.075 / 0.120 |

   On `dynamic4w` what remains is the driver; on `jolt`, the model's fit to Jolt as well. Near
   walls and on slopes the residual grows (contact, grade); not yet measured there.
3. **Gradients** (`test_gate3_gradients_are_finite_and_match_finite_differences`): a 12-tick
   rollout's loss in the coefficients, at speed with a goal 15 m ahead. Beta's and gamma's
   gradients are nonzero and match central differences to 1e-6 relative; alpha's is zero, with
   no wall in reach. **Passes**, with the plant arm (gate 5) and without.

   In open ground far from the goal all three gradients are exactly zero. The drive accelerates at
   its limit (`a_max`) and steers at the goal, whatever the coefficients. They act only near walls
   (alpha's barrier) and when closing on the goal (beta's pull, gamma's damping), so a rollout
   teaches the policy only where it reaches one of those.
4. **Training.** The first design (`experiments/gym_twin/train_gate.py`) **fails**. The second
   (`experiments/gym_twin/train_gate2.py`) **passes on all three training seeds** against the
   untrained init and the constant coefficients, and ties nav@1 (below).

   **The first design.** A CoefMLP trained through the twin drives worse in the gym than its
   untrained init.
   - The run (2026-11-14, `--iters 150`, the script's other defaults):
     - 3 Humvees on `dynamic4w`, on flat 160 m squares with 8 box obstacles;
     - training: 6 world layouts (seeds 0-5), episode seeds from 0, a random 0-120 gym ticks
       driven before each 30-tick rollout;
     - judging: 4 held-out layouts (100-103) × 10 held-out seeds (5000-5009), 600 ticks each;
     - 31 minutes on one CPU core.

   | arm (judged in the gym) | distance left (m) | progress (m) | arrived |
   |---|---|---|---|
   | nav@1 (the gym's navigator) | 6.4 [5.0, 7.7] | 75.3 [67.7, 83.0] | 1.000 [1.000, 1.000] |
   | constant (1, 3, 4) | 12.5 [9.5, 15.6] | 69.1 [61.6, 76.7] | 0.900 [0.833, 0.958] |
   | untrained CoefMLP | 9.8 [7.3, 12.3] | 71.9 [65.4, 78.5] | 0.967 [0.933, 0.992] |
   | trained through the twin | 17.1 [13.3, 21.0] | 64.5 [56.8, 72.6] | 0.867 [0.792, 0.933] |

   Paired over the 40 episodes:

   | | progress (m) | arrived |
   |---|---|---|
   | trained − untrained | −7.33 [−10.90, −4.02] | −0.100 [−0.175, −0.042] |
   | trained − constant | −4.59 [−6.68, −2.68] | −0.033 [−0.067, −0.008] |
   | trained − nav@1 | −10.73 [−14.38, −7.27] | −0.133 [−0.208, −0.067] |

   No arm collided.

   **Why, as far as the run shows:** only 36 of the 150 iterations had a nonzero gradient, and
   the clearance term never engaged (no rollout came within 2 m of a wall). As gate 3 found, the
   coefficients act only near walls and when closing on the goal. Here only the second happened,
   so the training saw only the approach to the goal. The loss is the mean distance left over the
   30 ticks, which rewards closing fast; it says nothing about stopping inside the reach. The
   likely effect is less braking on the approach, which costs arrivals over 600 ticks. That
   mechanism is not measured yet.

   **The second design** (`train_gate2.py`, 2026-11-15) changes exactly those four things:
   - **Where rollouts start.** Each iteration drives a training episode in the gym with the
     current policy (400 ticks) and records every tick's state. Rollouts start from the recorded
     states where the coefficients act: within 8 m of a wall, or within 30 m of the goal and not
     yet there. Four per agent are batched, each with its agent's goal (`GymTwin.rollout`'s
     `goal`). Over the run that gave 147 wall starts, 1,049 approach starts and 4 random ones.
   - **The loss** is `arrival_loss`: the distance outside the reach at the end of the rollout,
     plus the speed left there in proportion to how far inside the reach it is, plus a small pull
     while outside it; and the clearance barrier.
   - **The horizon** is 90 ticks (5.4 s), long enough to brake to a stop from the approach.
   - **A curve:** every 25 iterations the policy is judged in the gym on a probe set (layouts
     200-201 × seeds 7000-7002), disjoint from the final set.

   The worlds are denser, so walls are met: 18 boxes on 120 m squares. Training uses layouts 0-7;
   the final judging uses layouts 100-103 × seeds 5000-5009, 600 ticks each. The libcvc build was
   transfix/libcvc#1275 (the 0.06 s tick; physics at 100 Hz). Torch seed 0, 100 iterations, 58
   minutes on one core.

   98 of the 100 iterations had a gradient (the first design had 36 of 150), and the clearance
   term engaged in 48. The probe curve (distance left after 600 ticks, mean over 6 episodes):

   | iteration | 0 | 25 | 50 | 75 | 100 |
   |---|---|---|---|---|---|
   | left (m) | 20.1 | 15.2 | 8.2 | 8.3 | 8.2 |

   Judged in the gym on the 40 held-out episodes:

   | arm | distance left (m) | progress (m) | arrived |
   |---|---|---|---|
   | nav@1 | 3.49 [2.99, 4.06] | 41.4 [35.4, 47.5] | 1.000 |
   | constant (1, 3, 4) | 5.57 [4.37, 6.93] | 39.4 [33.6, 45.1] | 1.000 |
   | untrained CoefMLP | 5.19 [4.19, 6.28] | 39.7 [33.9, 45.5] | 1.000 |
   | trained through the twin | 2.96 [2.78, 3.15] | 42.0 [35.7, 48.3] | 1.000 |

   Paired over the 40 episodes:

   | | distance left (m) | progress (m) |
   |---|---|---|
   | trained − untrained | −2.23 [−3.26, −1.26] | +2.23 [+1.26, +3.27] |
   | trained − constant | −2.61 [−3.89, −1.46] | +2.61 [+1.46, +3.90] |
   | trained − nav@1 | −0.53 [−1.06, −0.06] | +0.53 [+0.06, +1.06] |

   No arm collided.

   **What the pass is, and is not:**
   - **The gain is in stopping, not in getting there.** Every arm arrived in every episode, so
     arrival is at its ceiling on these worlds and tells the arms apart not at all. The trained
     policy ends inside the 3 m reach on average (2.96 m) and with a narrow interval; the
     untrained init stops about 2 m further out. That is what the stop term pays for.
   - **It ties nav@1.** Only seed 0 ends ahead of it (below). nav@1 is the gym's navigator,
     with its carrot FSM and parking, not this control law.

   **Over three training seeds** (torch seeds 0, 1, 2; the same worlds, the same 40 held-out
   episodes), distance left at the end, paired:

   | seed | trained − untrained (m) | trained − constant (m) | trained − nav@1 (m) | probe left at 0 → 100 (m) |
   |---|---|---|---|---|
   | 0 | −2.23 [−3.26, −1.26] | −2.61 [−3.89, −1.46] | −0.53 [−1.06, −0.06] | 20.1 → 8.2 |
   | 1 | −1.83 [−2.90, −0.85] | −1.80 [−2.86, −0.84] | +0.28 [−0.29, +0.86] | 20.0 → 7.0 |
   | 2 | −1.68 [−2.72, −0.75] | −1.92 [−3.04, −0.91] | +0.16 [−0.42, +0.78] | 20.5 → 5.3 |

   - Every seed beats its untrained init and the constant coefficients, with intervals clear of
     zero: by 1.9 m on average, 1.7 m in the worst seed.
   - Against nav@1, one seed is ahead and two are tied.
   - 98, 99 and 98 of 100 iterations had a gradient.
   - The probe curves rise on every seed, but not steadily: seed 2 dipped at iteration 50 before
     recovering. Each run took 52-58 minutes on one core.
   - **Judged through the twin's controller** (`drive_gym`), not nav@1 with the policy exported to
     `.cvcnav`.
   - **Easy worlds:** flat, so the twin's plant is the gym's (gate 1), and the policy never
     collided; the hard cases (grade, contact, crowding) are not in this set.

5. **The plant arm** (`test_gate5_the_plant_arm_is_the_command_the_plant_took`,
   `plant_arm_departures`). nav@1 drives the gym; the twin is given, each tick, the gym's state
   before it, the drive the gym reports (`plants()`' `drive`: the coefficients and the carrot) and
   whether the tick parks the agent (`plan`'s `parked`). Its command is compared with the one the
   gym's plant took (`steer_command`, `accel_command`). An agent-tick departs when the steering
   differs by more than 1e-4 rad or the acceleration by more than 0.01 m/s².

   The run: three Humvees on `dynamic4w` on the flat 160 m square, from x = −70 m to goals 85,
   100 and 115 m east, toward the square's east edge. It ends 20 ticks after all three are parked
   and stopped (253 ticks). The agent-ticks are split three ways:
   - **en route**: more than 60 m from the goal, not parked (317);
   - **arriving**: within 60 m, not parked (190);
   - **parked** (252): these are compared on the acceleration (the park brake). The steering is
     counted apart, for the reason below.

   Measured 2026-11-15 against libcvc's branch with the plan (transfix/libcvc#1286, on
   #1284 and #1275):

   | twin | en route | arriving | parked (acceleration) | parked (steering) | largest outside parking: steering (rad) / acceleration (m/s²) |
   |---|---|---|---|---|---|
   | the control law alone (the default) | 0.9 % (3) | 10.5 % (20) | 99.2 % (250) | 55.6 % | 0.0084 / 6.4 |
   | `plant_arm=True` | 0 % | 0.5 % (1) | 0 % | 55.6 % | 4·10⁻⁷ / 0.65 |

   **Passes.** What the control law alone misses:
   - en route, the first ticks' steering, which the actuator's slew holds to 0.021 rad a tick
     (0.35 rad/s);
   - arriving, braking past `a_max`: down to −6 m/s² (`a_brake`) where the law stops at −2, when
     the stopping governor (with the latency and the footprint) binds against the edge;
   - parked, the park brake.

   The one arriving tick that still departs (0.65 m/s²) is the first tick the stopping governor
   binds against the edge. From the next tick on, the two agree to 10⁻⁴ m/s²; the cause is not
   found. The parked steering departs with or without the plant arm, and no twin in double can
   remove it: nav@1 holds a parked agent's carrot 2 cm ahead of it. Pure pursuit's angle there
   is float rounding times 2 L / 2 cm (~330), so the gym's float steering and the twin's double
   one differ by up to the actuator's slew.

   **nav@1 overshoots its goals here.** It arrives at 15–17 m/s: the goal's pull brakes it only
   in the last ~20 m, at `a_max`. It is parked as it passes within the 3 m reach, and the park
   brake stops it 9–13 m past the goal. The gym's own arrival test is then never met, so the
   episode does not end; this gate stops it instead.

   A twin rollout parks too (`test_the_plant_arm_parks_a_rollout_at_its_goal`): from the tick
   that ends within the reach of its goal, a row brakes to a stop, at most at `a_brake`.

## Running the training gate

```bash
PYTHONPATH=<libcvc build>/bindings/pycvc:. \
  python experiments/gym_twin/train_gate2.py --iters 100 --probe-every 25 --torch-seed 0 --out gate2.json
# the first design, as it ran:
PYTHONPATH=<libcvc build>/bindings/pycvc:. \
  python experiments/gym_twin/train_gate.py --iters 150 --out gate.json
```

About 35 s an iteration on one core (`OMP_NUM_THREADS=1`), most of it the 90-tick float64 rollout
and its backward pass.

It writes per-episode measures, summaries and paired differences, with
`grl_snam.learning.stats`' bootstrap. The worlds are flat squares with box obstacles, one layout
per world seed; training and judging use disjoint world seeds and episode seeds.
