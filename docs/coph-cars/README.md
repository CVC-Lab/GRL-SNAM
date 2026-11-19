# CoPH drives the cars: the first paired-seed runs (exploratory)

`grl_snam.ariadne.coph_cars` (T4, demo 2b of libcvc's `NAV-COPH-ROADMAP.md` §3.3), measured on
2026-11-19 on Windows. The interpreter was cvcpkg's Python 3.11.13 with torch 2.8.0 (CPU), and
pycvc and pycvc_gl came from a local libcvc Release build at f3cecaf2 (2026-11-15).

```bash
python -m grl_snam.ariadne coph-cars --arms --seeds 1 2 3 --cars 4 --frames 2000 --workers 3 --out runs/random
python -m grl_snam.ariadne coph-cars --arms --layout convoy --seeds 1 2 3 --cars 4 --frames 2000 --workers 3 --out runs/convoy
```

- [REPORT-random.md](REPORT-random.md) (`summary-random.json`): random starts and goals, the
  template's own.
- [REPORT-convoy.md](REPORT-convoy.md) (`summary-convoy.json`): a column of four cars on one route.

Both use four Jolt RoboRacers at `s` = 1 on a 21 m flat square. A car arrives within 0.80 m of
its goal. Each episode stopped when all four cars had arrived, after 27-36 s simulated and about
2.5 min of controller time per episode.

**What they show.** These are three seeds, and nothing here is a navigation claim.
- **The allocation.** About a quarter of CoPH's force is not taken: the allocation residual,
  `sqrt(sum |e|^2 / sum |u|^2)`, is 0.22 on random starts and 0.25-0.27 in the convoy. The cars
  run at about 0.5 m/s, the allocation's speed floor, where the lateral force a RoboRacer can take
  is small. The unallocated work `e . dq` is small (0.018 over 12 cars), because `e` is mostly
  across the motion.
- **The ledger.** It does not close on a car:
  - The ledger residual (kinetic change - work + heat, with heat booked by CoPH's plant at the
    learner's damping) is 9.7-10.2, against work of 8.1-8.3.
  - The damping fit ends at its lower bound, 0.15, in all 12 cars of every arm.
  - So the cars dissipate less than the least damping CoPH's plant allows.
- **The field and the routing.**
  - **The fit is right.** A one-car check sampled the learner's `r` at the car every 1.2 s
    along a convoy route. It was within 0.015 of the material plane `r~` at 26 of 27 points,
    0.569 against 0.570 at the patch's peak. The other point, just after a stage advance at
    3.0 s, read 0.152 where `r~` was 0.020.
  - **The cars do not avoid the patches.** The cars drive through them, and `pi_risk` stays at
    about 1.005.
  - **Why.** The profile routes only through sensed footprints, a 1.25 m disc, so it cannot plan
    around a patch about 2 m across. The demos route on a surveyed corridor network, which a car's
    world does not have.
- **The arms.** They are indistinguishable on random starts: the same arrivals, times and paths,
  and risk cost within 0.01. Only 70 of 1,054 CoPH packets were eligible. The rest were refused as
  `unsupported`: none of their anchors lay in the receiver's kept stages.
  - **In the convoy,** most packets were eligible: 688 of 1,284 for CoPH, 868 of 1,278 for raw
    readings.
  - Mean risk cost against communication off: -0.014 for CoPH and -0.048 for raw readings, on
    costs of about 43.
  - Only 2 of the 12 seed-car pairs differ by more than 0.03. Raw readings cut 0.24 and 0.35;
    CoPH cut 0.25 in one of them. Arrivals and paths were the same.

`window.png` is the template's window (`python -m grl_snam.ariadne coph-cars --offscreen --frames
300 --seed 2 --set view.camera=overview --set view.raster_m=12`). It shows four RoboRacers on the
template's Humvee-sized flat square, with their predicted paths, the coefficient plots and, in
Rasters, car 0's `V` around it.
