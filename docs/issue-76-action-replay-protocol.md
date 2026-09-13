# Training-only counterfactual action-replay diagnostic

## Purpose and boundaries

The native action-support audit found one first-shot intervention across all
248 usable training lineages. This separate diagnostic will test whether the
existing predictors can rank alternative actions and whether those alternatives
offer actual task headroom. It is not a new optimizer run, a learned gameplay
deployment, a precision study or a fresh #76 advancement test. Prior negative
results and failed qualifications remain unchanged.

## Metadata inventory

Use the first **assigned** training lineage of each of the five existing normal
families, in original family order: native development members 001, 071, 141,
211 and 281. Do not select by usability, success, predicted error or solvability.
Preserve exact source XML, scenario identity, generation seed, engine seed,
base-cluster identity and training exposure role. New rollout identities must
distinguish interventions without pretending they are independent lineages.

For each lineage, record the original first action `(-80, 10)` as a separate
reference, then the historical 12-action grid in its original ordinal order:
x in `{-10,-60,-110,-160}` within each y stratum `{-80,0,80}`. Every shot has
release 600 ms and tap 0 ms. There are 65 assigned one-shot replays. Only the
12 grid actions participate in the ranking pool; the original-action reference
is not an extra candidate from which to choose a favorable result.

Use the existing native development player and live episode interface with a
prospectively fixed selector, no learned model choosing executed actions. The
same original action is used for public actuator readiness before every
pre-decision snapshot. The selected replay action is applied only afterward.
Preserve the genuine earlier decision snapshot and actual prelaunch/capture
observations; do not initialize candidate scoring from an action-dependent
post-decision observation. Keep all native clocks and decision-to-launch gaps.
The collector must use verified native terminal events, not a later game UI
lifecycle state, to report level-clear evidence.

## Resource envelope and staged access

One isolated worker, private ports/workspace and kernel-allocated Xvnc display;
no `-nographics` capture. Total collection allowance: 14,400 wall seconds,
32 GiB new artifacts and 4 GiB worker-tree RSS. Each replay is bounded by 420
wall seconds, including startup, and a 180-second shot wait. Preserve the
existing speed-1, 0.0004-second native step, RGB every 50 native steps, maximum
30,000 steps/12 physical seconds and 601 RGB frames per shot. No technical
retries, extra seeds, replacement levels or outcome-conditioned omissions.
Interrupted assignments and all unattempted assignments remain recorded.

The first source lineage's reference and candidate 01 form the bounded rendered
smoke: two of the 65 assignments, at most 1,200 seconds, not extra replacement
data. Smoke continuation depends on capture/provenance, initial-state
comparability and resource checks—not on winning or predictor quality. All
captured failed/partial attempts and reviewable frames must be retained.

The original collection recorded about 66.8 GB across 350 assigned episodes;
this is a planning reference, not a measured cost for the new replay design.
Offline preparation is bounded by 1,800 CPU wall seconds and scoring by 900
CUDA wall seconds, with no optimizer updates. Charge actual preparation,
perception/history, predictor transitions and validation work separately;
linear MACs are not full FLOPs or matched deployment compute.

## Task interpretation

Use the existing task utility: 1000 times active pig count plus active block
count, with native engine labels used only as evaluation targets. A genuinely
observed physical terminal can supply settled task counts; it does not supply
an observed 4.5-second carrier or automatically mean gameplay success. Censored,
missing and failed outcomes receive the predeclared 1e9 ranking failure cost.
Report absolute count regret alongside normalized regret, top1/top3, ties,
all-failed states, task-discriminating lineages and actual native level clears.
An all-failed or all-tied group does not establish useful ranking performance.

Five training lineages and repeated model seeds are not independent evidence of
a population-level or seed-robust advantage. Strong independent pure, same-
hybrid fixed controls and a strong action prior remain required for future
gameplay work. Do not reuse old controllers as if fitted to the midpoint models.

## Execution is still disabled

This inventory freezes identities and scope, **not** capture authorization.
Before rendering, finish and test the source-bound execution driver and the
initial-state comparability checks, then freeze their complete execution plan.
Verify the bounded rendered smoke before the remaining assignments. Freeze the
offline scoring implementation, checkpoint pool and common decision-anchor
construction before predictor scores. No model scoring is authorized by this
document alone. Any later fitting needs a separate prospective training plan;
fresh evaluation still requires every #76 readiness, statistical and compute
gate. Nothing here unlocks #64/#65.
