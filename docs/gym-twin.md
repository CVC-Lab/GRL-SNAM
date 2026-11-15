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
   no wall in reach. **Passes.**

   In open ground far from the goal all three gradients are exactly zero. The drive accelerates at
   its limit (`a_max`) and steers at the goal, whatever the coefficients. They act only near walls
   (alpha's barrier) and when closing on the goal (beta's pull, gamma's damping), so a rollout
   teaches the policy only where it reaches one of those.
4. **Training** (`experiments/gym_twin/train_gate.py`): **fails on its first run.** A CoefMLP
   trained through the twin drives worse in the gym than its untrained init.
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

   **What to change before the next run:**
   - start rollouts where the coefficients act (near walls and goals);
   - a loss that pays for arriving and stopping, not for closing fast;
   - longer horizons over the approach;
   - judge on held-out seeds during training, so the curve shows what each step did.

## Running the training gate

```bash
PYTHONPATH=<libcvc build>/bindings/pycvc:. \
  python experiments/gym_twin/train_gate.py --iters 150 --out gate.json
```

It writes per-episode measures, summaries and paired differences, with
`grl_snam.learning.stats`' bootstrap. The worlds are flat squares with box obstacles, one layout
per world seed; training and judging use disjoint world seeds and episode seeds.
