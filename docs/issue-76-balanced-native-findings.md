# Balanced-native engineering findings

The v2 execution plus its explicitly frozen norm-reduction recovery completed
all 15 jobs on 2026-09-16. The parent and numerical-recovery reports are in
`data/issue-76-balanced-native-v2/report.json` and
`data/issue-76-balanced-native-v2/norm-recovery/report.json`. The original v1
failure and v2 pre-optimizer failure remain retained; neither was relabeled as
a successful fit.

All model and Adam tensors in the completed checkpoints are finite. Each of
the six predictors completed 12,000 scheduled updates, 11,904 applied and 96
skipped. Each controller completed 4,000, 3,968 applied and 32 skipped. Each
shared history fit completed 1,000, 992 applied and 8 skipped. Total scheduled
optimizer updates were 99,000, with 98,208 applied and 792 retained skips.
Recorded v2 active fitting time, including the preserved recovery allowance,
was 19,557.8 seconds. No job reached its resource limit.

The balanced perception surface fixes the prior cross-level carrier collapse.
First-carrier mean feature standard deviations across assigned trajectories
were 0.07235, 0.07510, and 0.07024 for the three seeds, versus approximately
2.15e-5 for the older original-parser engineering reference. The diagnostic
adaptive rollouts had zero trace failures in both arms for all three seeds.

The native hybrid controller remains mode-degenerate: every diagnostic choice
was continuous, although horizons 50 and 750 both occurred. Hybrid counts were
420/47, 570/37, and 1,080/3 for horizons 50/750. Thus this stage establishes
usable varying carriers, finite trained dynamics, and completed controllers,
but does not establish non-degenerate symbolic adaptation or higher gameplay
utility. No advancement or fresh-access claim follows from it.

The next engineering stage must use these completed balanced dynamics for
event-ranking/readout and adaptive selection on the already exposed event
cohort. That cohort cannot support a new readiness claim. Any readiness test
still requires a newly frozen development cohort disjoint from every prior
lineage, followed by the separately authorized fresh non-final gate.
