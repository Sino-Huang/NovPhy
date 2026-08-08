# Temporal Frontier Witness Table

| Partition | Motion regime | Delta | States | Weighted error mean | Compute mean | Truncation rate | Primary selections |
|---|---|---:|---:|---:|---:|---:|---:|
| controller-train | quiescent | 1 | 212211 | 7.67315444529e-09 | 1 | 0 | 27989 |
| controller-train | quiescent | 5 | 212211 | 3.7056891508e-08 | 0.225826025355 | 0.0791335039183 | 114810 |
| controller-train | quiescent | 15 | 212211 | 1.03632200217e-07 | 0.111979386244 | 0.251372454774 | 69412 |
| controller-train | transitional | 1 | 175050 | 4.33744294549e-09 | 1 | 0 | 8450 |
| controller-train | transitional | 5 | 175050 | 2.1548242415e-08 | 0.200362372656 | 0.00278206226792 | 110429 |
| controller-train | transitional | 15 | 175050 | 6.57672633204e-08 | 0.0689453968389 | 0.0407654955727 | 56171 |
| controller-train | high_motion | 1 | 45353 | 1.82435602971e-08 | 1 | 0 | 7317 |
| controller-train | high_motion | 5 | 45353 | 3.05588484817e-07 | 0.2 | 0 | 33175 |
| controller-train | high_motion | 15 | 45353 | 2.52401553498e-06 | 0.0666666666666 | 0 | 4861 |
| evaluation | quiescent | 1 | 31533 | 8.37673235889e-09 | 1 | 0 | 5294 |
| evaluation | quiescent | 5 | 31533 | 4.04681260406e-08 | 0.224618124927 | 0.0752862080995 | 16035 |
| evaluation | quiescent | 15 | 31533 | 1.13582102124e-07 | 0.109865864675 | 0.241651603082 | 10204 |
| evaluation | transitional | 1 | 24197 | 5.08464622105e-09 | 1 | 0 | 1737 |
| evaluation | transitional | 5 | 24197 | 2.52937734028e-08 | 0.20037676847 | 0.00305823035914 | 14837 |
| evaluation | transitional | 15 | 24197 | 7.68434175798e-08 | 0.0690039743104 | 0.0391784105468 | 7623 |
| evaluation | high_motion | 1 | 6334 | 2.15722522481e-08 | 1 | 0 | 1348 |
| evaluation | high_motion | 5 | 6334 | 3.29791562448e-07 | 0.2 | 0 | 4283 |
| evaluation | high_motion | 15 | 6334 | 2.64456543185e-06 | 0.0666666666667 | 0 | 703 |

Frontier verdict: **not_supported**
Bootstrap replicates: 1000 (seed 20260807)
Fixed-delta intersection: `[1, 5, 15]`

Scope: continuous temporal carrier only. Symbolic and physical axes are unavailable.
