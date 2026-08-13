# Getting the tick loop to 30 Hz

Measured 2026-08-13 on a 32-core box (loadavg ~106 throughout, so every number
here is pessimistic). Workload: 8 agents, real Austin, 384x384 occupancy grid,
fog on, `Squad.step()`.

    REAL: 1375.5 ms/tick = 0.73 Hz.   30 Hz needs 33.3 ms/tick => 41x.

    component         calls  per tick   ms/call   ms/tick  % tick
    build_sdf           104      1.73     457.6     793.2    57.7
    astar               104      1.73     259.8     450.3    32.7
    bicycle_rollout     480      8.00       5.9      47.4     3.4
    simplify            104      1.73      25.5      44.1     3.2
    everything else                                  40.4     2.9

## Four things that are not obvious from that table

**The mean is the wrong statistic.** All eight agents share `step_i`, so all
eight sense, rebuild and replan on the *same* tick: 15 of 60 ticks carry the
entire cost, ~5,240 ms on a sense tick against ~88 ms on the other 45. A 30 Hz
real-time loop is bound by the worst tick, not the mean, so staggering agents
(offset agent `i` by `i mod sense_every`) is a correctness requirement for
real-time even though it saves nothing on average.

**Neither hot function is slow because of its algorithm.** `_edt1d_rows` is an
O(n) separable Felzenszwalb transform whose inner while-loops almost never
re-iterate; it issues ~92,000 numpy dispatches per `build_sdf` on 384-element
vectors at 5-9 us each, for an effective 0.32 Mcell/s. `astar` costs 17-22 us
per expanded node, of which ~5 us is `occ[r,c]` numpy scalar indexing (620 ns
x8), ~2.0 us heap traffic and ~2.2 us `dict.get`. Both are paying interpreter
and dispatch tax, not asymptotics. That is why numpy has no headroom left: the
EDT already went through a 6x vectorisation pass and is *still* the top cost.

**The A* is run before the decision to use it.** `_replan_route`
(`scenario.py:270-294`) plans at :283 and only then checks validity at :289.
On 618 recorded replans of the Austin fog run, **541 (87.5%) produced a
bit-for-bit identical committed route**. `route_valid` costs 0.508 ms against
`plan()`'s 137 ms - a 270x ratio - so the check is essentially free and is
being done in the wrong order.

**The SDF is built to be sampled ~21 times.** The 147,456-cell field is read at
exactly 4-5 point samples per agent per tick (`nav.py:230`, `:232-244`, `:256`,
`:158`) and nowhere else; the planner uses the occupancy raster directly.
Bilinear sampling touches ~84 distinct cells, **0.057%** of what was built.

## A hypothesis that measurement killed

"Only 0.15% of cells change between rebuilds, so an incremental EDT should be
~1000x." **False, and worth recording so nobody retries it.** A distance
transform is global: one new obstacle cell in a 99.7%-empty optimistic map wins
a huge Voronoi region. Measured, inserting a single cell dirtied 1 intermediate
column but changed **281-384 of 384 rows** of the column-transform output in 6
of 8 sweeps, so the row pass stays full. Exact dirty-region restriction caps at
about 1.4x, for days of work on the fiddliest code in the stack.

The redundancy is real, but it has to be harvested at the *call* level (don't
rebuild at all) rather than the *cell* level (rebuild less).

## CUDA: no, with a crossover number

Both candidate kernels were ported and measured on this box's GTX 1650.

| | CUDA | single-core C++ | verdict |
|---|---|---|---|
| batched EDT, 384^2 | 0.440 ms/agent | 9.285 ms | GPU ~= 21 CPU cores |
| A* (GPU wavefront vs heap) | 18.4 ms | 2.84 ms | **GPU 6.5x worse** |

The box has 16 usable cores, so the GPU is worth 1.3-1.6x the whole CPU on one
sub-component - flat from 128^2 to 768^2. Worse, if the field must return to
host memory, D2H (0.301 ms) + compute (0.440 ms) = 0.741 ms already loses to
the 16-core CPU's 0.580 ms, so **a GPU EDT with a host consumer never wins at
any agent count.** A* goes the wrong way outright: it expands 8.3k nodes where
a wavefront touches 112k free cells x 416 sweeps, and 500 dependent kernel
launches alone cost 2.325 ms - 82% of an entire CPU A*.

CUDA becomes worth it above **~60 agents at 384^2** (~27 at 512^2, ~14 at
768^2), and only with results kept device-resident. Note also that exactly one
machine in the fleet has CUDA, so a CUDA-only fast path does not run anywhere
else. The CPU port reaches the target on its own; CUDA would be a second
implementation of a solved problem.

## Cython / numba / pybind11 / C++

C++ in the existing `cvc` library, per project policy. The FFI is not the
problem and never was: marshalling a 384^2 uint8 array in costs 9.6 us and a
zero-copy view out 0.6 us, against the 457 ms it saves. libcvc already ships
the expensive part - a working SWIG module with a zero-copy numpy `ArrayView`
typemap, per-interpreter cvcpkg columns GRL-SNAM's CI already installs, and
`WINDOWS_EXPORT_ALL_SYMBOLS ON`.

Note `cvc::sdf` is **not** what this loop needs: it is a triangle-mesh-to-3D
volume SDF behind `CVC_ENABLE_SDF` (`inc/cvc/utility/algorithm.h:56`). It
shares only a name with a 2-D EDT over a boolean raster. libcvc has no A*, no
Dijkstra, no distance transform over occupancy - all of it is new code, in new
files under the existing library (`inc/cvc/nav/grid_nav.h`,
`src/cvc/nav/grid_nav.cpp`), exposed through the **existing** `pycvc.i`, never
a new SWIG module.

## Staged plan

Each stage stands alone and is ordered by value per effort. Every saving is
booked once; where several changes attack the same milliseconds they are
alternatives, not addends.

| stage | change | where | ms/tick after | Hz | effort |
|---|---|---|---|---|---|
| 0 | profiling harness + golden-trace oracle | python | 1375.5 | 0.73 | days |
| 1 | exact gates: plan-only-if-needed, rebuild-only-if-a-sample-could-move, dedupe double-computed arrays | python | 450-900 | 1.1-2.2 | days |
| 2 | batch the 8 agents' rollouts into one B=8 call; vectorise `_route_subgoal` | python | 390-840 | 1.2-2.6 | days |
| 3 | `cvc::nav` C++ kernels: EDT, A*, line-of-sight, inflate | C++ | ~35 | ~28 | weeks |
| 4 | thread across agents (they are independent) + stagger sense ticks | C++ | 10-12 | 80-100 | days |

Stage 1 alone cuts the speedup stages 3-4 must deliver from 41x to about 8x.
Stage 4's headroom is roughly **150 agents at 384^2** inside a 33.3 ms budget.

Measured component ratios behind stage 3, from independent implementations by
two investigators that agreed: EDT 457.6 -> 9.3-10.9 ms (43-65x, bit-identical
in float64); A* 259.8 -> 2.8-4.9 ms (58-77x, byte-identical paths);
`_line_of_sight` 126 us -> 0.103 us so `simplify` 44.1 -> ~0.06 ms/tick.
`bicycle_rollout` is ~1,600 flops per agent-step wrapped in 405 torch calls on
1-element tensors - torch costs ~2,900x the arithmetic - and batching to B=8 was
verified **bit-identical** over 30 chained steps.

## Fidelity guardrails

A fast digital twin that moves differently is worthless. These are the specific
ways this port can silently change trajectories:

1. **EDT precision.** Keep the parabola envelope arithmetic in float64. On
   equidistant sites the tie-break `s <= z[k]` can pick a different argmin in
   float32; distances stay equal but the *gradient* flips on medial axes, which
   flips the wall normal and the steering bias (`sdf_nav.py:367-368`).
2. **A* tie-breaking.** Python's `heapq` compares the whole tuple
   `(f, g, (r,c), parent)`. A C++ heap must reproduce that order exactly or it
   returns a different, equally optimal, path.
3. **No `-ffast-math`, no `-ffp-contract=fast`.** `g` is an exact sum of 1.0 and
   sqrt(2) terms and `h` is libm `hypot`; plain IEEE double reproduces CPython
   bit-for-bit, but contraction or reassociation breaks it.
4. **Copy `_nearest_free` verbatim,** including its top-left-biased scan order.
   It is not actually a nearest search, and "cleaning it up" moves start/goal
   cells and changes every route.
5. **`route_valid` walks Bresenham; `astar` is 8-connected with corner-cut
   prevention.** A route can validate through a diagonal squeeze A* would
   refuse, so the stage-1 gate needs a low-rate unconditional replan if
   `no_route` detection matters.
6. **SWIG runtime must be pycvc's 4.2.0/data4,** not cvcpkg's 4.4.1/data5, or
   cross-module type sharing breaks silently.

The oracle is a **golden trace**: record positions, headings,
`waypoints_reached`, `truth_penetration_steps` and rebuild ticks on a fixed
seed, then assert bit-identity after every change. Keep each Python
implementation as the reference and add a CI test asserting the C++ and Python
paths agree on a fixed grid set.

## What not to do

- **Octile heuristic.** 1.7x, admissible, provably identical path *cost* - but a
  different optimal path among equal-cost ties, so trajectories change. Not
  worth re-validating a twin for 3 ms once A* is already at 7.5.
- **Jump Point Search.** Validated over 2,020 random grids with zero cost
  mismatches, and it would cut `simplify` 10-20x as a side effect - but it
  changes which optimal path is returned. Hold it until `simplify` is the
  binding constraint, which stage 3 makes unlikely.
- **Incremental/dirty-region EDT.** See above: measurably ~1.4x, not ~1000x.
- **Rewriting in Cython or numba.** The win is in C++ that libcvc can also use;
  a second toolchain buys a fraction of it and another build surface.
