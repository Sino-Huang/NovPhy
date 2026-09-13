# Controller-free fixed-policy development evaluation

## Purpose and unchanged boundaries

The latest training diagnostics establish substantial recursive-error recovery
but retain failed local/endpoint qualification checks. Their dispositions stay
false. This separate diagnostic will test transfer to already assigned,
previously opened native development partitions; it does not adopt an
unqualified checkpoint for gameplay, refit a controller or open fresh data.
The #76 advancement gate still requires actual adaptive gameplay, strongest
independent pure and same-hybrid fixed controls, a strong action prior, useful
adaptation, practical margins with uncertainty and matched compute.

The existing native `score_entry` includes adaptive prediction and requires a
fitted controller; those controllers belong to an older predictor recipe.
Do not reuse them as if fitted to the new predictors. This evaluation is
fixed-policy only and must not infer adaptive or action-ranking performance.

## Membership and retained predictor pool

Use every one of the 50 calibration and 50 model-selection assignments in
`issue-76-repaired-representation-v1/data-index.json`, preserving order,
base-cluster identity and all five families. The metadata index has 49 usable
calibration assignments and 50 usable model-selection assignments; actual
endpoint availability is checked only at scoring. Retain the unusable member
and every missing endpoint. No training or fresh/final member is added.

Include all three paired seeds and both arms from all eight retained recipes:
repaired representation, matched dynamics, boundary dynamics, full-duration
dynamics, lower rate, boundary focus, local anchor and endpoint anchor.
Also include the one independently validated 50/50 local/endpoint parameter
midpoint, with no other weights. This is 48 existing saved models and six
temporary midpoint instances. Every one of the 48 checkpoint files exists;
completion/source/shape validation is still required before scoring.
All recipes use the same repaired shared representation and cached carriers.
Keep each recipe's nine hybrid fixed pairs and three pure fixed horizons.
No pure recipe may be dropped because its training diagnostic failed.

## Access order and measurements

Freeze the metadata inventory first. Implement and test the scoring source
and immutable execution plan before evaluating any member. Score the entire
calibration assignment list for all models/policies before freezing choices.
For each arm, choose one common recipe/fixed-pair combination across all three
seeds by equal-family, equal-seed mean endpoint carrier MSE on calibration;
ties follow listed recipe order then existing pair order. Exclude a choice
from eligibility if it has any prediction failure, but retain its full results.
Do not select different recipes or pairs by seed or model-selection outcome.

Only after that complete calibration freeze, score all assigned model-selection
members and every fixed policy, reporting the frozen selections separately.
These are strongest controls for this offline prediction metric, not claims
of strongest gameplay policies. Preserve the complete model/policy matrix
for later prospectively specified task-level baseline work.

Initialize from the first shot's actual pre-intervention observation and known
executed action. Use the exact common 11,250-native-step endpoint. No truth
resets, earlier substitute endpoint, terminal padding or target interpolation.
Report recursive and observed-context local errors separately at common
elapsed times 750, 1,500, 3,000, 6,000 and 11,250 native steps when available.
The observed-context diagnostic is not used for deployed action selection.

Preserve the existing field metrics: aggregate carrier, visual and memory
MSE; presence MSE and present-object center MSE against engine evaluation
targets; pig and block count absolute errors; absolute carrier-bound excess.
Do not call latent bound excess a physical violation. The unchanged-carrier
reference is a prediction reference, not a no-model action prior. These
single recorded-action traces cannot establish counterfactual ranking,
multi-shot gameplay success or adaptation headroom by themselves.

## Reporting, costs and validation

Report all assigned and available denominators per role/family/seed, with
the inherited 90% model-selection endpoint-coverage requirement left intact.
Use common available-member comparisons, retain failures, and never use
availability filtering to favor a model. Record deterministic prediction
failures with penalty MSE 1e9 and disqualify them from calibration selection.
Report per-seed effects and paired lineage-cluster bootstrap intervals
(10,000 resamples, seed 760940001, 95% intervals) descriptively; three seeds
do not establish a seed-robust advancement claim. Repeated policies/frames
on a lineage are not independent replicates.

The execution plan will enforce at most 900 active CUDA seconds per role,
300 CPU seconds per role/seed for preparation, and 900 CPU seconds for
independent validation, 12 GiB RSS, 8 GiB allocated CUDA and 1 GiB new derived
artifacts. Batch sizes and actual wall times must be recorded. Batched wall
time is not batch-one deployment latency. Charge exact linear-MAC proxies
and transition counts; do not call MACs full FLOPs or claim matched deployment
compute. Preserve all parent/representation costs with shared work counted
once. No new collection, optimizer work or fresh access is permitted.

Validate source/role inventories, source checkpoint completion, exact curves,
selection ordering and decisions before publishing conclusions. This document
and the metadata inventory do not authorize score execution on their own:
the scoring implementation, tests, source-bound execution plan and bounded
smoke must be frozen first. Every failed training qualification and every
unmet gameplay/advancement requirement remains explicitly recorded.
