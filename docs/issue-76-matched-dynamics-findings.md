# Exact-example mixed dynamics: qualification failed

The source-bound experiment frozen at commit `03fe55e` and recorded in
[#76 before fitting](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5652693937)
completed all six predictors and all 30 paired training-lineage diagnostics.
Every fit used 6,000 scheduled updates: 5,952 applied, 48 skipped, and exactly
190,464 sampled starts. No controller training, new capture, or fresh evaluation
was performed. The qualification criteria were not changed.

## Prespecified contrasts

Mean recursive carrier MSE on the same five first-assigned training lineages,
at the exact 11,250-fixed-step endpoint:

| Seed suffix | Arm / fixed policy | Parent | Mixed | Mixed wins versus unchanged carrier | Qualified |
| --- | --- | ---: | ---: | ---: | --- |
| 001 | Hybrid / 750-micro | .063690 | .127440 | 0/5 | No |
| 002 | Hybrid / 750-micro | .049205 | .055028 | 0/5 | No |
| 003 | Hybrid / 750-micro | .052419 | .042298 | 4/5 | No |
| 001 | Pure / 750-continuous | .023283 | .010239 | 5/5 | Yes |
| 002 | Pure / 750-continuous | .024581 | .013220 | 5/5 | Yes |
| 003 | Pure / 750-continuous | .026889 | .012185 | 5/5 | Yes |

Seeds are 760930001–3. Hybrid required at least a 20% mean reduction and 4/5
unchanged-carrier wins separately in every seed. Seed 003 improved by about
19.3%, below its frozen requirement; seeds 001 and 002 worsened. Pure required
no more than a 10% mean regression and instead improved about 46–56% in every
seed. Both arms share the exact same parent carrier targets, so these
within-arm parent/new MSE comparisons do not change representation underneath
the metric. These are training diagnostics, not held-out generalization or
gameplay results.

The full publication retains all fixed policies, including unfavorable ones.
For example, hybrid seed 001 fixed-50-micro mean MSE increased from 8.86346 to
612.53673, while fixed-50-continuous increased from 10.14361 to 124.40672.
Finite optimizer losses do not imply stable recursive prediction.

## Interpretation and next action

Regrouping the exact same examples helped the independently trained pure arm
on this probe but did not repair the hybrid predictor. Do not promote this
candidate to controller refitting or fresh evaluation, select only seed 003,
relax the 20% criterion, or extend its optimizer budget. Its final checkpoints
remain preserved as negative development evidence.

This experiment does not establish why the hybrid behaves differently. The
remaining candidates include competing gradients through its symbolic heads,
interference between its three mode-conditioned dynamics tasks, and inadequate
control of recursive drift beyond the four training offsets. The continuous
mode's deterioration in seed 001 means this is not solely a failure observed
while executing the micro adapter. A read-only diagnostic should distinguish
these candidates before another prospective training change. Forcing symbolic
controller selections would not repair the demonstrated predictor errors.

The useful-mode, strong pure/fixed/prior, gameplay, uncertainty, and matched-cost
advancement requirements remain unmet. No advancement or #64/#65 authorization
is claimed.

## Reproduction and measured cost

`data/issue-76-matched-dynamics/report.json` contains the frozen source plan,
all 30 parent/new records, six decisions, and ten completed budget records.
`validation.json` records a complete independent recomputation of all 30
records, exact summary equality, source-plan equality, all six checkpoint
counts, and completed budget/memory checks. The validation command is:

```
python -m scripts.validate_issue_76_matched_dynamics
```

Initialize `novphy` and source `env.sh` first. The validation output is immutable;
its timing is run-specific, so the published validation file is not overwritten
by a subsequent invocation.

New predictor work totaled approximately 1,729.5 active seconds. Metadata and
carrier preparation totaled 15.1 seconds; scoring 17.7 seconds; the separate
no-fit validation 17.9 seconds. Peak recorded CPU RSS was 2,669.5 MiB; allocated
GPU memory was about 61 MiB. Reused common representation training remains an
additional standalone-model cost, retained in the source plan and its parent
protocol; it is not included in these new-work numbers or treated as free.
All final checkpoint files remain under
`.local-artifacts/issue-76-matched-dynamics-v1/checkpoints/`.
