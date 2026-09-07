# Task scoring, action ranking, and bounded CEM pilot (#70)

The completed calibration run failed its objective audit. Across 2,400 actual
endpoints the frozen parser estimated approximately 0.69 pigs both when a pig
remained and when it had been removed, and approximately 2.61 blocks across
actual counts 0–5. On the 81 count-discriminating states, top-3 was 0.420 and
regret 0.741, failing the frozen 0.80/0.10 requirements. The pilot remains blocked.

Reproduce the count diagnosis without inference or new collection:

```bash
python -u -m scripts.run_issue_70_action_design \
  --diagnose-parser 2>&1 | tee -a data/issue-70-parser-diagnosis.log
```

The legacy parser also has a proven architectural limitation: its shared linear
presence head on `image_features + object_query` gives all objects the same
image-dependent logit change, plus fixed object biases. The separately versioned
`SlotConditionedVisualPredicateParser` adds nonlinear image/object fusion. A
synthetic independent-removal regression is learnable with v2 and not v1.
This is an architecture repair, not a trained replacement or proof of real-data
accuracy. The v1 serializer rejects v2, and v1 checkpoints/behavior remain intact.

Before resuming gameplay, a new training workflow must train/validate the repaired
parser on training-only data, validate counts on calibration, rebuild the carrier
bundles, and retrain matched world models. Inserting an untrained or differently
trained parser underneath existing world-model checkpoints is not a valid fix.
The commands below preserve and publish the current negative diagnostic result.

This is a new exploratory experiment. The original #63/#69 results are preserved.
It asks whether a deployment-available task score represents useful outcomes,
whether model predictions preserve those rankings, and only then whether CEM
helps in a small fresh non-final gameplay pilot. It does not authorize #64.

## Commands

Activate the `novphy` conda environment and source `env.sh` first. The agent has
tested preparation, the synthetic dry run, and a bounded real endpoint/checkpoint
smoke test. Full endpoint preparation, model scoring and gameplay are operator
commands. Every long stage reports foreground progress; completed artifacts are
immutable and resumable. No new hashes or repeated full-corpus integrity scans
are added. Original checkpoint loading is isolated in short-lived processes.

```bash
python -u -m scripts.run_issue_70_action_design --dry-run

python -u -m scripts.run_issue_70_action_design \
  --prepare 2>&1 | tee -a data/issue-70-prepare.log

python -u -m scripts.run_issue_70_action_design \
  --prepare-endpoints --device cuda \
  2>&1 | tee -a data/issue-70-endpoints.log

python -u -m scripts.run_issue_70_action_design \
  --audit-objective \
  2>&1 | tee -a data/issue-70-objective.log

python -u -m scripts.run_issue_70_action_design \
  --score-models --device cuda \
  2>&1 | tee -a data/issue-70-ranking.log

python -u -m scripts.run_issue_70_action_design \
  --freeze-pilot \
  2>&1 | tee -a data/issue-70-freeze.log
```

Check the freeze log's `allowed=True/False`. **Only if it is true**, run:

```bash
python -u -m scripts.run_issue_70_action_design \
  --run-pilot --device cuda --start-display \
  2>&1 | tee -a data/issue-70-pilot.log
```

Then publish and validate, whether the pilot gate passed or failed:

```bash
python -u -m scripts.run_issue_70_action_design \
  --publish 2>&1 | tee -a data/issue-70-publish.log

python -u -m scripts.run_issue_70_action_design \
  --validate 2>&1 | tee -a data/issue-70-validate.log
```

If the objective audit fails, model scoring remains available as a diagnostic;
it cannot override that failed gate. Publication then reports
`not_ready_for_pilot`. Do not change the gate to obtain a passing run. Bring back
the objective/ranking reports so the next change can address the measured cause.

The optional `--smoke-test --device cuda` reparses the 12 candidates of the first
calibration state and scores them with one corrected checkpoint. It writes no
experiment artifacts. `--dry-run` uses synthetic carriers and models, exercises
all four planners and three re-observation steps, and needs no Unity or data.

## What the stages measure

The score is `1000 * expected remaining pigs + expected remaining blocks`.
The frozen parser vocabulary identifies pig/block slots. Presence probabilities
are bounded to [0,1] for utility evaluation, but the recursively predicted
carrier is never clamped. Carrier-range excess is reported separately. No
best-candidate target, desired endpoint from a demonstration, future game state,
or realized candidate label enters this scorer.

Stage A parses actual recorded endpoints from every #68 calibration candidate.
It compares that deployment representation's score with authoritative remaining
counts. Thus even a correct count formula can fail when the visual parser does
not correctly distinguish object removal. Original fractional progress costs
(contacts, displacement, support changes) are reported as a secondary diagnostic;
they are not silently equated to the remaining-count objective. Stable support
changes without removal need not improve this particular objective.

The objective gate requires at least 20 count-discriminating calibration states,
mean normalized regret <=0.10 and top-3 fraction >=0.80 on those states, with no
new scoring failures. All-state results and ties remain visible; this diagnostic
subset is not a replacement or collection filter.

Stage B scores the same legal 12-action inventory with the original three #63
models and the nine matched #67 models. Ensemble costs are the mean member cost
plus a nonnegative disagreement coefficient from {0, 0.25, 0.5, 1, 2}. H15 is
called 15 times using self-conditioned carriers and a fixed action. The future
capture duration never determines model rollout length.

The original #63 references used four million training examples; the matched
#67 teacher-forced/u2/u4 comparisons used eight million. The pilot comparison
against the original model is therefore an operational comparison, not evidence
that unrolling alone caused any gameplay difference.

Prediction accuracy is compared with the recorded observation at fixed-step
offset 225, or an earlier stable terminal treated as absorbing. Task ranking
uses actual settled outcomes. These are distinct endpoints, and neither is fed
to the planner. The observed endpoint cache records the actual horizon offset.
The absorbing-terminal comparison is an explicit approximation; stable stopping
does not itself mean gameplay success.

Reports include:

- Best actual action's top-1/top-3 inclusion, deterministic rank and tie-ambiguity
  interval; all predicted ties and all realized ties are counted separately.
- Per-state normalized regret `(selected-min)/(max-min)`; all realized ties
  receive zero. Prediction failures receive regret one, including on tied states.
  Retained capture failures remain candidates with the existing cost of 1e9.
- Remaining-count errors, aligned-horizon carrier MSE, range excess, disagreement,
  inference time, model evaluations and failures.

Stage B freezes the minimum-regret corrected and original configurations and
the strongest fixed-action prior using calibration only. It permits Stage C
only if Stage A passed, the selected corrected model has no prediction failures,
and its mean count-regret is strictly lower than that prior's. This is an
exploratory feasibility gate, not a statistical advancement claim.

## Gameplay and audit

The pilot prospectively samples 12 new levels, balanced between `type010101`
and `type010102`, from seeds 700500000–700500011. Source identities and seeds are
checked against prior cohorts and the opened #57 inventory before gameplay.
Each level is played by original-model CEM, corrected-model CEM, corrected-model
12-action grid search, and the frozen no-model action prior. Every level and
execution failure is retained; one attempt per level/system, no replacements.

CEM evaluates four candidates in each of three iterations, retains the lowest
cost action seen across all iterations, and fixes RedBird tap time to zero.
Grid search evaluates 12 actions. Both share the same 12-candidate budget and
30-second planning limit per shot. An ensemble requires three times the model
evaluations of a single model; actual compute is reported, rather than claiming
equal FLOPs across different ensemble sizes.

The runner executes one selected shot, captures the engine and agent frames
until stopping, then reparses a real observation before planning again. There
are at most three shots, capped by the number of authored birds: the first
generator family has one bird; the second has three. One-bird trials test action
selection; three-bird trials can additionally demonstrate closed-loop replanning.
The slingshot anchor comes from the existing interface alignment procedure, not
the model. Actual shot actions are checked against the saved planner decisions.

Videos and their gallery are saved under `data/issue-70-pilot-audit/`. Video
playback is 50 captured frames/second, matching the existing fixed-step audit
convention. Failed trials retain any available partial frames/video. Full
per-candidate predictions and CEM iteration records accompany every decision.
Pilot summaries report paired level success differences and descriptive 95%
bootstrap intervals; 12 levels do not provide a confirmatory success claim.

## Artifacts and next boundary

Private artifacts: `.local-artifacts/issue-70-action-design-v1/`, including
`objective-audit.json`, `ranking-diagnostics.json`, `pilot-freeze.json`, source
inventory, per-model scores and per-trial captured evidence. Public result:
`data/runtime_evidence/issue-70/action-design-v1.json`.

Validation recomputes the objective audit, rankings, selection and publication
from the complete saved inventories. For a completed pilot it checks paired
source membership, recorded actions, captured termination and video existence.
It does not repeat neural inference or deep-hash the corpus.

The old model-selection role is already opened. This redesign uses calibration
only and is exploratory. Even a promising pilot needs a separately frozen,
prospectively powered fresh confirmation before #64 -> #65 can resume. A failed
objective or ranking gate is a useful result identifying what to fix next.
