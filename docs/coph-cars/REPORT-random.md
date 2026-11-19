# CoPH drives the cars (T4, demo 2b): exploratory

roboracer at s = 1 on a flat square, 4 cars, layout random, paired seeds [1, 2, 3]; CoPH steps every 2 ticks; a car arrives within 0.80 m. Each seed gives every arm the same starts, goals and risk patches.

## The ledger and the allocation (per unit mass, CoPH's frame, summed over cars and seeds)

| arm | allocation residual | work (u_used . dq) | unallocated work (e . dq) | heat | kinetic change | ledger residual | worst step residual | shaping jumps |
|---|---|---|---|---|---|---|---|---|
| off | 0.224 | 8.313 | 0.018 | 16.165 | 1.891 | 9.743 | 0.0144 | -0.140 |
| coph | 0.224 | 8.314 | 0.018 | 16.166 | 1.890 | 9.742 | 0.0144 | -0.142 |
| raw | 0.224 | 8.315 | 0.018 | 16.165 | 1.891 | 9.741 | 0.0144 | -0.139 |

The allocation residual is sqrt(sum |e|^2 / sum |u|^2) over the ticks: the share of CoPH's force the cars did not take. The ledger residual is kinetic change - work + heat, with heat booked by CoPH's plant at each learner's damping: zero on CoPH's own plant, so here it measures how far a car's motion is from a damped point mass under the force it took.

## Outcomes

| arm | arrived | mean arrival (s) | mean path (m) | cars touching | excess risk (mean / max) | bytes per car | updates accepted / rejected |
|---|---|---|---|---|---|---|---|
| off | 12/12 | 27.1 | 13.5 | 0 | 0.000 / 0.000 | 0 | 411 / 0 |
| coph | 12/12 | 27.1 | 13.5 | 0 | 0.001 / 0.005 | 1098680 | 411 / 0 |
| raw | 12/12 | 27.1 | 13.5 | 0 | 0.002 / 0.011 | 1310140 | 411 / 0 |

## Learning and packets

| arm | final damping (mean; at the 0.15 floor) | final pi_risk (mean) | packets by the receiver's verdict |
|---|---|---|---|
| off | 0.150; 12/12 | 1.005 | none sent |
| coph | 0.150; 12/12 | 1.005 | eligible 70, unsupported 984 |
| raw | 0.150; 12/12 | 1.005 | eligible 95, unsupported 959 |

A receiver refuses a packet as `unsupported` when none of its anchors lies in the receiver's kept stages (`CooperativeLearner.assimilate`).

Excess risk is each car's risk cost (4 r~ dt, summed) minus the lowest any arm got on the same seed and car. Nothing here is a navigation claim: the seeds are few, and the formulation treats a fully actuated point (roadmap §3.3.2, §8).
