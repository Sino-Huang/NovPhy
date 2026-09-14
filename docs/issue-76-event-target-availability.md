# Event-target availability in existing development data

This metadata audit reuses the completed 350-lineage native development plan,
recorded segment summaries, and previously verified native outcome audit.
It performs no new capture, fitting, fresh access, or native-corpus validation.
The [machine-readable inventory](../data/issue-76-event-target-availability.json)
retains all assigned identities, source roles, family strata, and failures.

## Available supervision

| Exposure role | Assigned lineages | Without collection failure | Recorded segments | Clear | Level fail | Right-censored | Stable without clear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 250 | 248 | 284 | 5 | 29 | 138 | 112 |
| Calibration | 50 | 49 | 51 | 1 | 4 | 31 | 15 |
| Model selection | 50 | 50 | 61 | 0 | 10 | 26 | 25 |

The four outcome columns count segments from episodes without collection
failure, not independent replicates. There are five distinct clear training
lineages: 002, 028, 032, 046 (type010101) and 300 (type010105). The sole
calibration clear is 335 (type010105); model selection has no observed clears.
The other three training families contain no native-clear example under these
assigned actions. These are properties of this collection, not proof that the
families are unsolvable or intrinsically unfavorable to either model.

## Decision-time and action-support gaps

All 350 legacy episode results have zero explicit decision records. Their
collector prepares the assigned action, executes it, then persists the aligned
capture interval. The retained first physical sample must not be silently
relabeled as the earlier genuine decision observation. Existing frames remain
usable for appropriately defined observed-context/transition tasks; this audit
does not invalidate them or invent missing causal inputs.

Training action support is 248 segments at (-80,10), 20 at (-60,45), and 16 at
(-10,80). The latter actions occur later in sequential gameplay states rather
than as randomized alternatives from the same decision state. Calibration has
49/1/1 and model selection 50/7/4 segments at those respective actions. Thus
frame count and repeated transitions do not supply broad independent
decision-state/action coverage.

The original-grid and angular replay collections add genuine pre-decision
observations and multiple assigned actions, but reuse the same five training
lineages. They add no calibration or model-selection lineages. The angular
pilot provides two clear lineages and still fails its frozen viability screen;
its positives cannot be treated as a new independent confirmation cohort.

## Target semantics that must not be conflated

An intact right-censored 12-second segment has no observed clear within that
observed interval. It does not prove never clearing, supply a terminal-time
regression target, or establish what happened after collection stopped.
Likewise an early `stable_entered` stopping event is not a level clear and does
not establish absence of a later clear at an unobserved common deadline.
Time-to-first-segment-stop, clear-by-observed-deadline, and eventual multi-shot
gameplay success are distinct targets with different availability masks.
The legacy segment's 12 seconds start at capture, not at an explicit saved
decision; the decision-to-capture gap cannot be invented as zero.

## Development decision

Do not fit a nominal decision-time event head on these records while claiming
the missing input and action coverage are present. Keep original roles and
negative results intact. The next source-bound development protocol must
provide genuine pre-decision RGB/history, outcome-independent action coverage
across independent training and development-check lineages, and explicit
event/censoring target availability. Shared changes must apply symmetrically
to independently trained hybrid and pure-continuous arms. Numerical collection,
training, compute, stopping, and readiness criteria must be frozen before that
workflow runs. No classifier fit or metadata audit authorizes fresh evaluation
or satisfies #76's adaptive-gameplay advancement gate.
