# Native refit v2: completed, but not ready for advancement

The September 13 automated refit completed all three common parser/history
fits, six independent dynamics fits, six controllers, and 600 assigned
development diagnostic records. The original report validates exactly. This is
development evidence, not the once-only fresh non-final experiment.

Publication: `data/issue-76-native-refit/report.json`. The accompanying
`diagnostic-report.json.gz` preserves the complete original 122,066,334-byte
report, including every per-transition work record. The compact report retains
all assigned rows, failures, metrics, costs, and exact joint mode/horizon counts.
Linear MACs remain linear MACs, not full FLOPs or matched deployment compute.

Reproduce publication with the initialized `novphy` environment and `env.sh`:

```bash
python -m scripts.publish_issue_76_native_refit --publish
python -m scripts.publish_issue_76_native_refit
```

## What the completed diagnostic shows

Each arm/seed has 45/50 calibration endpoints and 45/50 model-selection
endpoints. The frozen 90% model-selection endpoint-coverage check passes.
Unavailable assigned episodes remain in the report. There were no nonfinite
prediction exceptions; finite but very inaccurate predictions are not successes.

Mean recursive carrier MSE on the 45 available model-selection lineages, at the
same 11,250-native-step (4.5-second) endpoint:

| Paired seed | Adaptive hybrid | Same hybrid, fixed 750 continuous | Independent pure, fixed 750 continuous |
| --- | ---: | ---: | ---: |
| 760930001 | 4.482638 | 0.059464 | 0.00051439 |
| 760930002 | 0.057298 | 0.057298 | 0.00007595 |
| 760930003 | 1.766953 | 0.032411 | 0.00023078 |

Calibration selects fixed-750-micro as each hybrid's lowest-MSE fixed policy,
and fixed-750-continuous for each independent pure model. Adaptive hybrid
executes only continuous mode on every scored model-selection trajectory:
225 short steps per lineage for seeds 1/3, and 15 long steps for seed 2.
Thus this diagnostic does not demonstrate useful adaptive description switching.
The independent fixed-long-horizon control is substantially stronger in this
diagnostic; comparing only against the unstable adaptive pure controller would
hide that result. No gameplay or fresh-test superiority is inferred from MSE.

## Post-publication CPU diagnosis (no new fitting or captures)

A same-checkpoint replay on the first calibration lineage with an available
endpoint, `issue-76-development-052`, reproduced seed 1's failure: adaptive
MSE 4.482642 versus fixed-750-continuous 0.059464. The CPU replay took 5.16s.
An initial probe of the first usable calibration lineage had no exact endpoint;
it was not substituted as truth or retried in the engine.

The first training lineage from each family (001, 071, 141, 211, 281) has true
initial block counts `[1, 5, 6, 7, 4]`. Their agent-image pixel MSEs relative to
001, with RGB scaled to [0,1], are `[0, .001893, .005426, .006693, .002233]`.
Yet predicted counts are approximately `[4.005, 4.005, 4.006, 4.007, 4.006]`
for seed 1, similarly 4.005–4.007 for seed 2 and 4.032–4.035 for seed 3.
Maximum predicted-center differences across these different inputs are below
.00035. This 4.07s CPU probe supports near-collapse of shared perception; it
does not establish the cause or justify treating low carrier error as physical
competence.

An audit of every saved training-only controller target (62,935 rows per arm
and seed; 7.49s CPU wall time) found **zero symbolic-mode target labels** for
all three hybrid controllers. Initial-episode targets select horizon 750 in
230/248 cases, while hybrid controllers 1/3 predict horizon 50 in all 248.
Their overall teacher-label accuracies are .653/.486/.629; the pure controllers'
are .709/.640/.722. Both target utility and distillation need attention.

Candidate causes still to distinguish: family-blocked training order and
perception collapse; short observed-context training versus long recursive
deployment; controller target construction; and failure to fit those targets.
Do not force arbitrary symbolic use, weaken the gate, or claim a repair before
a prospectively specified, symmetrically applied experiment verifies it.

## Preserve outcome and authorization boundaries

The original collection's legacy success field is zero. The separately
validated native-terminal audit records **6/350** fixed-policy collection wins:
`data/issue-76-native-outcome-audit/report.json`. These are neither trained-policy
wins nor fresh evidence. The compact publication links that correction without
rewriting the original report or collection records.

No original model, optimizer, membership, endpoint, failure, or budget record
was rewritten. No fresh evaluation was opened. This publication supplies no
supported advancement disposition and does not authorize #64 or #65. The next
step is a bounded, prospective diagnosis/repair of training readiness, not a
fresh sweep of this failed adaptive candidate.
