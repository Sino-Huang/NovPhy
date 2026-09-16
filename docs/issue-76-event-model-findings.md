# Event-ranking development screen: retained failure

The source-frozen event-ranking stage completed all 39,000 scheduled optimizer
updates across three paired seeds, two independently trained dynamics arms, and
their matched controllers.  All 2,600 assigned event records were retained;
2,560 had passed the preceding capture audit.  No fresh or final data were
opened.

The development readiness result is false.  On 33 reset-comparable
model-selection lineages, 10 contained at least one observed clearing action.
Adaptive hybrid, same-hybrid fixed, adaptive pure, and fixed pure each selected
one clear.  The training-only action prior selected three.  Adaptive hybrid
therefore failed every required relative-utility check.  It did pass complete
finite inventory, nonzero utility, nondegenerate mode and horizon use, and the
matched-compute ceiling; its measured hybrid/pure linear-MAC ratio was 0.870.

| Development system | Clears / 33 | Clears / 10 informative |
| --- | ---: | ---: |
| Adaptive hybrid | 1 | 1 |
| Same-hybrid selected fixed pair | 1 | 1 |
| Selected primary pure-continuous | 1 | 1 |
| Selected no-model prior | 3 | 3 |

The failure is structural rather than a resource stop.  Every learned system
selected candidate ordinal 1 in all 68 comparable training groups and all 31
comparable calibration groups; adaptive hybrid had zero training and
calibration clear hits.  Across valid reference images, the original shared
parser/history carrier's mean per-feature standard deviation was approximately
`2.15e-5` (maximum `3.67e-4`).  The outcome head therefore had effectively no
level-specific visual signal and converged to a nearly constant ordinal score.
Increasing the same optimizer budget cannot repair that representation.

An outcome-independent parser diagnostic already frozen before this screen
provides the next justified branch.  Its balanced parser checkpoints have mean
per-feature standard deviation approximately `3.88e-2` on the same reference
images and passed their predeclared semantic sensitivity/specificity test on
all three seeds.  A retrospective engineering-only probe confirmed that this
representation can fit every informative training lineage and no longer
collapses to one action ordinal.  That probe is not a readiness result.

The failed event plan, report, individual selections, work logs, and completed
checkpoints remain unchanged.  Do not lower its thresholds, reinterpret its
one clear as support, or reuse its exposed model-selection outcomes for another
readiness claim.  The next candidate must retrain both dynamics arms from
scratch under the shared balanced perception contract and use a newly disjoint
development screen before any fresh protocol can be considered.
