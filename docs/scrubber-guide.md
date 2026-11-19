# The checkpoint scrubber: a short guide

`grl-snam ariadne scrub` deploys every checkpoint a training run saved. The Pages site's scrub page
is the same thing in a browser. This guide says what each window shows, and what to look for to
tell a policy that is learning something useful from one that is not.

## What you are looking at

Every checkpoint drives the same scenario: the same world, start, goal, seed, plant and number of
ticks. So any difference between two checkpoints comes from the policy, not from the scene.

- **Two cars.** The orange car is checkpoint k, the one you picked. The blue car is step 0, the
  untrained net. They drive side by side over the same episode.
- **The plant.** The cars drive libcvc's Jolt Humvee by default (`--plant humvee`). The Humvee
  collides with the buildings: a car that drives into one stops against it. The old default,
  nav's kinematic bicycle (`--plant kinematic`), has no collision response, so its cars drove
  through buildings.

## The windows

- **Scrub** picks what you look at and charts it.
  - *Set* and *Run* pick a data set (a retrain) and a run in it (one training seed).
  - *Checkpoint*, the slider, *Previous* and *Next* pick checkpoint k.
  - *Metric* picks the deployed number to chart. The options are reached the goal, time, path
    length, ticks penetrating, minimum and mean clearance, and distance left to the goal.
  - The chart shows that metric against training step, one point per checkpoint. Step 0's value
    is the dashed line, and the cursor marks checkpoint k. Click a point to pick it.
  - The table below the chart shows every metric, one row per checkpoint.
- **Training** is the held-out curve the training run logged. *Curve* picks the term. It has the
  same cursor, and a click on it picks the checkpoint nearest that step.
- **Episode** is the play head: the position slider scrubs through the episode, and *Camera*
  either follows checkpoint k's car or orbits the world from above.
- **Stats** has one row per car: clearance (minimum and mean), ticks penetrating, path length,
  time elapsed, wall time per tick, and whether it reached the goal. Below the rows is the
  clearance chart over the episode.
- **Ledger** has one row per car. It shows:
  - the policy's coefficients (`alpha` the wall barrier's gain, `beta` the goal's, `gamma` the
    damping, and the `lam_*` and `mu` heads when the net has them);
  - the potential's terms;
  - the energy books.

  Each comes summed over the run and for this tick.
- **Simulation** pauses, steps and restarts the episode.

## What converging on useful behaviour looks like

A falling training loss is not enough. Look for all of the following together.

1. **The deployed metric leaves step 0's line, and stays away from it.** Reached flips to yes, time,
   path and distance to the goal fall, and ticks penetrating stays at 0, all compared with the
   dashed line. One good checkpoint proves nothing; a run of them in a row does.
2. **The outcomes are ordered by training step.** Later checkpoints do at least as well as earlier
   ones, and neighbours agree with each other. When the chart jumps between two outcomes from one
   checkpoint to the next, the policy is not settling: that is noise, not learning.
3. **The held-out curve and the deployed metric move together.** In *Training*, the held-out
   curve falls and flattens. At the same steps, the deployed metric in *Scrub* improves. If the
   curve falls while the deployed metric stays flat, the objective is not the task.
4. **Every seed agrees.** Step through *Run*: the same trend should show on every seed of a set,
   not just on the seed you looked at first. *Set* compares retrains at the same step.
5. **The coefficients settle, and they respond to the situation.** In *Ledger*, `alpha`, `beta`
   and `gamma` stop changing between neighbouring checkpoints, and they end up away from step 0's.
   Within an episode they should also vary with where the car is. Coefficients that stay at one
   value all episode mean training only moved the mean, which a constant net would do as well.
6. **The car drives better, not just faster.** The orange car takes a shorter, cleaner route than
   the blue one. It keeps its clearance above 0 in *Stats*' chart, and it does not gain reach by
   scraping walls.

## Warning signs

- **The best checkpoint is step 0:** training never improved on the untrained net.
- **Ticks penetrating above 0, or clearance falling while reach rises:** the policy is trading
  safety for progress.
- **Outcomes scattered across training steps** instead of ordered by them.
- **A held-out curve that falls while the deployed metric stands still.**

## What today's runs show

The Pages page shows the learning check's three retrains: `retrain`, `fixed/retrain` and
`world-units/retrain`, five seeds each, on libcvc's generated city.
[`learning-validation.md`](learning-validation.md) has the full measure. In short:

- Training curves fall in the later retrains.
- Deployed outcomes are not yet ordered by training step, and no set shows criteria 1–6 together.
- No checkpoint reaches the goal on either plant.
- On kinematic, every checkpoint drives through buildings for 355 to 636 ticks of the 1200.
- On the Humvee, two of the sets come to a single outcome whatever the step: the car meets a
  building and stays against it. A policy that had learned to route around buildings would not
  do that.

So treat the page as the instrument to watch the next retrain with, not as evidence that the
current policy learned. The numbers, per set and plant, are in the scrub section of
[`grl_snam/ariadne/README.md`](../grl_snam/ariadne/README.md).
