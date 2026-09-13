# Fixed-policy development results

The complete calibration and model-selection matrices have been scored and
independently verified using the explicitly amended source-backend method.
The frozen controls show no clear hybrid prediction advantage, and two families
fail the endpoint-coverage prerequisite. This is not a fresh #76 experiment or
an advancement pass. All earlier failed training qualifications remain unchanged.
The original strict CPU/CUDA comparison failure is also retained, not relabeled
as passed.

## Model-selection result

Calibration selected the fixed 50/50 local/endpoint parameter midpoint and
`fixed-750-continuous` for **both** arms, once across all three seeds. Choices
were frozen and pushed before model-selection access. All other recipes and
fixed policies were retained in the [complete report](../data/issue-76-fixed-development/report.json).

| Endpoint prediction measure | Hybrid-trained, continuous | Independent pure |
| --- | ---: | ---: |
| Equal-family/equal-seed carrier MSE | 0.005335875 | 0.005350131 |
| Pig-count absolute error | 0.042038 | 0.035663 |
| Block-count absolute error | 0.849725 | 0.845527 |
| Present-object center MSE | 0.021181 | 0.022178 |

The hybrid-minus-pure carrier-MSE difference is −0.000014257, with a descriptive
95% paired-lineage bootstrap interval of [−0.000263412, +0.000242088]. The three
per-seed differences are −0.000047534, −0.000013182 and +0.000017946. This does
not establish a seed-robust or practically meaningful hybrid advantage.
The 10,000 resamples carry all seeds of each lineage together, preserve family
strata and use the prospectively frozen seed 760940001.

The hybrid's difference from the unchanged-carrier reference is −0.041656770,
interval [−0.042796992, −0.040563004]. This is prediction recovery, not a win
against a no-model gameplay action prior. The same-hybrid continuous contrast
is exactly zero because calibration chose continuous execution itself; no
adaptive switching was performed or validated.

There are 50 assigned and 45 endpoint-available model-selection lineages:

| Family | Assigned | Exact 11,250-step endpoint |
| --- | ---: | ---: |
| type010101 | 10 | 9 |
| type010102 | 10 | 8 |
| type010103 | 10 | 8 |
| type010204 | 10 | 10 |
| type010105 | 10 | 10 |

The overall 90% availability does not rescue the two 80% family strata. The
five missing model-selection endpoints are intact first-shot segments with
observed physical termination at 2.4372–3.6444 seconds, not failed predictions.
Physical termination/stability is not equivalent to winning. No segment was
padded, replaced or silently evaluated at a different requested time.

## Implemented and checked

The evaluator now supports batched, uninterrupted fixed-pair curves at all five
frozen physical endpoints; separate exact observed-context local predictions;
masked engine-target field metrics; unavailable assignments; and cumulative
per-member prediction failures. It does not expose a controller, optimizer,
collector or fresh-data role. An unchanged-carrier reference is explicitly not
a gameplay action prior.

Calibration selection requires the complete ordered model/member matrix and
common endpoint availability. It chooses one recipe/pair per arm across seeds
using equal-family, equal-seed endpoint error, disqualifies prediction failures,
and breaks ties by the frozen recipe/pair order. These are offline prediction
controls, not strongest gameplay policies.

The initial seventeen tests passed, including independent single-member rollout agreement,
exact first-shot indexing, persistent failures, engine masks/counts, exact work
counters, equal-family selection, retention of an adverse seed, and rejection
of incomplete or incorrectly bound inputs.

## Bounded synthetic CUDA smoke

The source-bound [smoke plan](../data/issue-76-fixed-development/smoke-plan.json)
was written before execution. It used 50 synthetic records, seed 760940001,
batch size 50, and the first paired hybrid/pure checkpoints in the frozen
inventory. All nine hybrid and three pure policies reached all five exact
endpoints with finite predictions. No real development record was loaded.

The [smoke report](../data/issue-76-fixed-development/smoke-report.json) records:

- All 48 saved checkpoint files and six parameter-midpoint instances passed
  source, completion, capacity, finite-state and completed-budget checks.
- Checkpoint validation: 9.704 seconds measured wall time; 645.3 MiB peak RSS.
- Synthetic scoring phase: 2.740 seconds measured wall time; 1,305.8 MiB peak
  RSS and 18.2 MiB peak allocated CUDA memory.
- 57,000 recursive and 3,000 local transitions; 109,224,480,000 linear MACs.
  There were no controller calls, optimizer updates or new captures.

The phase times are conservative wall-time charges, not CPU process time,
CUDA kernel time, or batch-one deployment latency. MACs are not full FLOPs.
An earlier read-only completion check additionally took 8.451 seconds wall time;
it also read no development records. The detailed source checkpoint receipts
are retained at
`.local-artifacts/issue-76-fixed-development-v1/smoke-checkpoints.json`.

## Execution and retained verification history

The [execution plan](../data/issue-76-fixed-development/execution-plan.json) was
committed and pushed before calibration access. Its ledger preserves 141
canonical #76 cost receipts, including failed/nonselected branches, and excludes
seven exact public receipt mirrors from double counting. These receipts record
86,112.187 seconds of prior job wall-time charges, not kernel hours or full FLOPs;
unmetered earlier engineering and prior-issue work are not represented as zero.

All 54 calibration model instances completed in 67.748 seconds of phase wall
time, retaining 50 assignments and 45 exact endpoints per seed. This is 2,700
assigned model/member rows. Peak RSS was 1,339.0 MiB and peak allocated CUDA
memory was 18.1 MiB. The three preparation phases took 1.084, 1.089 and 1.100
seconds. Original scores are retained without a rerun.

Validation first exposed and corrected a JSON object-key ordering assumption;
see the [technical correction](issue-76-fixed-development-json-correction.md).
At that stage 28 tests passed. The corrected independent CPU validator completed the
first three cells, then stopped on repaired-representation/pure seed 760930002,
member `issue-76-development-265`, fixed-50 continuous, elapsed native step 6000:

- Stored block-count absolute error: 0.0340266227722168.
- Independent CPU reconstruction: 0.0339810848236084.
- Independent CUDA reconstruction with separate NumPy field formulas:
  0.03402674198150635, within the original tolerance of the stored score.
- Maximum CPU/CUDA carrier-component difference at that batched point:
  0.00013303756713867188.

The strict CPU failure is retained; no numerical tolerance was widened. The
same-backend check establishes the cause at this one point, not validation of
the entire matrix. The cumulative validation budget retains 18.188 seconds,
including the failed attempts. The focused arithmetic diagnostic additionally
took 1.060 seconds wall time. Its compact evidence is in the
[calibration status receipt](../data/issue-76-fixed-development/calibration-status.json).

The subsequent [backend correction](issue-76-fixed-development-backend-correction.md)
distinguishes verification of the executed CUDA computation from backend
invariance. It retains the same independent host loop, NumPy field formulas,
all inventories and numerical tolerances, but executes transitions on their
original arithmetic backend. This qualified method verified all 54 cells and
2,700 assigned rows in each role: 78,084 available recursive curve points in
calibration and 79,380 in model selection, plus their available local metrics
and unchanged-carrier references. There were no prediction failures in either
full matrix. Neither this fact nor zero latent bound excess is a physical
validity or gameplay claim.

Model-selection scoring took 68.632 seconds wall time; its three preparation
phases took 1.171, 1.147 and 1.165 seconds. Cumulative validation and publication,
including failed validation attempts, took 193.314 seconds under the unchanged
900-second allowance. The full role-specific transition/MAC counts, source
paths, costs and access chronology are in the report. The original plans and
provisional failure receipt remain archived.

Twenty-nine isolated evaluator regressions and three reporting tests pass:

```bash
python -m scripts.test_issue_76_fixed_development
python -m unittest tests.test_issue_76_fixed_development_publication
```

## Remaining advancement evidence

These one-recorded-action traces cannot establish counterfactual action ranking,
strongest gameplay policies, useful adaptation, or a gameplay advantage over
the strong fixed action prior. Do not retrain a controller merely to create
symbolic counts or reinterpret this near-tie as an adaptive-system success.
Task-action utility and useful switching need separately declared development
evidence before a new gameplay/fresh protocol can be justified. No further
fitting or new collection was performed under this diagnostic. #64/#65 and
the full #76 advancement gate remain unauthorized.
