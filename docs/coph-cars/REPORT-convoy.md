# CoPH drives the cars (T4, demo 2b): exploratory

roboracer at s = 1 on a flat square, 4 cars, layout convoy, paired seeds [1, 2, 3]; CoPH steps every 2 ticks; a car arrives within 0.80 m. Each seed gives every arm the same starts, goals and risk patches.

## The ledger and the allocation (per unit mass, CoPH's frame, summed over cars and seeds)

| arm | allocation residual | work (u_used . dq) | unallocated work (e . dq) | heat | kinetic change | ledger residual | worst step residual | shaping jumps |
|---|---|---|---|---|---|---|---|---|
| off | 0.272 | 8.139 | 0.017 | 16.570 | 1.774 | 10.205 | 0.0161 | -0.337 |
| coph | 0.272 | 8.148 | 0.018 | 16.590 | 1.772 | 10.214 | 0.0161 | -0.346 |
| raw | 0.253 | 8.136 | 0.018 | 16.610 | 1.772 | 10.246 | 0.0164 | -0.206 |

The allocation residual is sqrt(sum |e|^2 / sum |u|^2) over the ticks: the share of CoPH's force the cars did not take. The ledger residual is kinetic change - work + heat, with heat booked by CoPH's plant at each learner's damping: zero on CoPH's own plant, so here it measures how far a car's motion is from a damped point mass under the force it took.

## Outcomes

| arm | arrived | mean arrival (s) | mean path (m) | cars touching | excess risk (mean / max) | bytes per car | updates accepted / rejected |
|---|---|---|---|---|---|---|---|
| off | 12/12 | 30.4 | 14.4 | 0 | 0.051 / 0.353 | 0 | 461 / 0 |
| coph | 12/12 | 30.3 | 14.4 | 0 | 0.038 / 0.249 | 1349309 | 460 / 0 |
| raw | 12/12 | 30.3 | 14.4 | 0 | 0.003 / 0.016 | 1601851 | 459 / 0 |

## Learning and packets

| arm | final damping (mean; at the 0.15 floor) | final pi_risk (mean) | packets by the receiver's verdict |
|---|---|---|---|
| off | 0.150; 12/12 | 1.005 | none sent |
| coph | 0.150; 12/12 | 1.005 | eligible 688, unsupported 596 |
| raw | 0.150; 12/12 | 1.005 | eligible 868, unsupported 410 |

A receiver refuses a packet as `unsupported` when none of its anchors lies in the receiver's kept stages (`CooperativeLearner.assimilate`).

Excess risk is each car's risk cost (4 r~ dt, summed) minus the lowest any arm got on the same seed and car. Nothing here is a navigation claim: the seeds are few, and the formulation treats a fully actuated point (roadmap §3.3.2, §8).
