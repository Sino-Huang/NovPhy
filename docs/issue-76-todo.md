# #76 requirement checklist and completion record

This checklist covers the current issue body and all five preceding comments:
[coverage amendment](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5603214096),
[#75 disposition](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5603383149),
[prospective execution contract](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5603606530),
[operator handoff](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5604363972),
and [smoke/implementation verification](https://github.com/Sino-Huang/NovPhy/issues/76#issuecomment-5604380134).

Checked means the stated deliverable is done, not that an unexecuted experiment
passed. The current branch is a **pre-access readiness stop**, not a fresh
negative experiment. Gated activities below remain unchecked and must not be
launched simply to clear this checklist.

## Completed diagnostic and metadata work

- [x] Read and retain #73 diagnosis, #74 independently trained continuous/hybrid readiness, and the exactly validated #75 negative disposition.
- [x] Preserve #15 and #70/#71/#72/#74/#75 evidence; do not silently retune, replace checkpoints, or weaken advancement gates.
- [x] Separate training effects, symbolic execution, adaptation, and gameplay; disclose the shared semantically supervised CNN and absence of a uniquely joint-mechanism claim.
- [x] Prospectively freeze 24 calibration states (12 per family, evenly spaced sorted identities), all 12 actions, paired seeds 20260908/20260909/20260910, exact source/target membership, and claim scope before new diagnostic scores.
- [x] Execute all 10,368 fixed candidate records: independently trained continuous h1/h5/h15 and hybrid continuous/micro/macro at each matching h.
- [x] Report recursive predictions at common 15/30/60/150/225 fixed-step endpoints without truth resets; keep observed-context local-step errors separate.
- [x] Retain unchanged CNN/carrier/action encoding/objective, shared B=1 initial perception, canonical target batch/predecessor semantics, archived offset-225 targets, and explicit stable absorption/unavailable handling.
- [x] Record actual launch/collision/settlement timing and target availability; distinguish offset-225 prediction from settled replay outcomes and complete-shot gameplay.
- [x] Report predeclared presence/kind/position/motion/time and availability errors, aggregate carrier error and count errors with target-only masks; no post-outcome rescaling or invented physical violations.
- [x] Report settled ranking regret/top-k, ties, selected absolute progress, headroom, prediction failures and all seeds.
- [x] Retain original and covered #74/#75 adaptive controls at offset 225; do not interpolate skipped adaptive endpoints or call poor controller performance proof of poor dynamics.
- [x] Publish the 33 planned training/symbolic/adaptation contrasts with descriptive paired-state uncertainty, without treating candidates or seed repeats as independent lineages.
- [x] Record fixed/local/parser/controller/decoder/adapter work, historical training references and wall times; disclose that linear MACs are not full FLOPs or matched end-to-end compute.
- [x] Enforce the frozen work/memory/storage caps with progress/ETA and deterministic resume. Completed run: 983.47 active seconds, 769,148,024 bytes, no budget stop, no new fitting or capture.
- [x] Produce the metadata-only 45-cell scenario/novelty matrix, 80 source-bound templates and 40 workbook-mapped normal/novel pairs, including generator constraints, authored slots, mechanisms and exposure limitations.
- [x] Explicitly flag platform-slot overflow, unsupported appearances/classes/forces/events, right-side action constraints and insufficient observability; do not drop entities or claim metadata proves runtime compatibility.
- [x] Publish a costed metadata-selected rolling/falling/sliding and appearance-pair recommendation, plus magnetic/gravity inspection needs; distinguish zero-shot generalization from equal-budget adaptation.
- [x] Preserve all scheduled diagnostic records and provide CSV/SVG/HTML, 24 existing source video links and 72 source frame links with manifests/timing; no new rendered evidence claimed.
- [x] Run no-write dry-run and bounded real CUDA smoke before the operator's full run; leave long commands with logs and progress rather than launching fresh collection.
- [x] Fix CSV newline validation through an explicit exact-source amendment; preserve the original plan, records and publication. Full CUDA validation passed, and all 22 issue-specific tests passed.
- [x] Publish local `readiness_or_precision_insufficient` with zero fresh/final access and no #64/#65 authorization; explicitly distinguish this from a fresh negative experiment.

## Completed closeout work — existing evidence only

- [x] Add the missing paired model-versus-frozen-no-model comparator effects and uncertainty; retain ordinal09 selection optimism and uniform expected regret as a separate reference.
- [x] Explicitly report adaptive hybrid versus the strongest eligible independent continuous control selected on predecessor development data; do not choose a favorable new baseline from the 24 diagnostic states.
- [x] Write a final claim-by-claim findings report covering training, symbolic execution, adaptation, strongest baselines, absolute progress, timing/perception limits, compute and uncertainty; do not turn descriptive contrasts into fresh confirmation.
- [x] Validate the supplemental analysis against the preserved diagnostic and predecessor evidence, with regression tests and an exact repeatable operator command; record that it completes reporting after the diagnostic, not a new prospective test.
- [x] Reconcile every original acceptance item with either executed evidence or a concrete conditional/not-run disposition; retain the distinction between a proposal and an executable expanded protocol.
- [x] Publish the completed run, CSV repair, final findings, validation and explicit stop-branch handoff in #76; supersede the stale waiting-for-operator status.
- [x] Record the evidence-consolidation handoff to #35–#38 while explicitly keeping #64/#65 blocked/not run; do not imply downstream tickets are completed.

### Closeout completion update — 2026-09-10

This supersedes the earlier waiting-for-operator comments: the full diagnostic
and reporting closeout have now completed and passed exact validation. #76 is
left open for review of the readiness-stop disposition, not waiting for another
experiment. No commit, push, archive or fresh access was performed.

The separate supplement preserves the original 33 training/symbolic/adaptation
contrasts and adds 34 paired comparisons: all 16 systems versus ordinal09 and
uniform expected regret, plus both adaptive hybrid policies versus independent
continuous h5 selected in #75. Pairing averages seed differences within each
state before resampling; all states, seeds and family summaries are retained.
This is reporting completion after diagnostic inspection, not fresh confirmation
or retroactive preregistration.

Key descriptive results (positive = reference regret minus tested regret):

- Original hybrid versus ordinal09: -0.069444, 95% paired-state interval [-0.208333, 0.055556].
- Covered hybrid versus ordinal09: -0.069444, interval [-0.180556, 0.013889].
- Each adaptive hybrid versus independent continuous h5: -0.027778, interval [-0.083333, 0.027778].
- Hybrid-trained continuous h5 lowers offset225 carrier MSE but worsens ranking regret; this is not a gameplay advantage.

The conclusion remains `readiness_or_precision_insufficient` before fresh
access. The #75 negative result and #64/#65 restrictions are unchanged.
Only 10/24 states have informative action regret and 2/24 have a pig-removing/
level-clearing candidate; no claim of broad NovPhy or multi-shot competence is
made. Existing-data component effects and their limitations are reported separately.

Evidence and repeatable commands:

- `docs/issue-76-findings.md`: final claim-by-claim interpretation and limits.
- `data/issue-76-closeout/{summary.json,comparisons.csv,findings.md}`: source-bound supplement and exact numerical/component decisions.
- `scripts/run_issue_76_closeout.py`: `--dry-run`, `--publish --device cuda`, `--validate --device cuda`.
- `data/issue-76-closeout-{publish,validate}.log`: successful original-plus-supplement validation.
- Original `.local-artifacts/issue-76-dynamics-diagnostic-v1/` and `data/issue-76-review/` remain unchanged; the explicit CSV source-repair receipt remains bound and validated.
- All 31 focused tests passed: nine supplemental pairing/selection/report tests plus 22 diagnostic/CSV-repair tests. No-write dry-run and real publication/validation passed.
- The [#35 evidence-consolidation handoff](https://github.com/Sino-Huang/NovPhy/issues/35#issuecomment-5615811266) points to this checklist and the retained artifacts, and routes later archival/reporting/completion accounting through #36/#37/#38. Those tickets are not claimed complete.

## Conditional future work — not executed, not required to pretend a stop is success

- [ ] Obtain explicit direction for material broader collector/perception/carrier/action/history engineering; cost any shared representation/training change symmetrically before implementation. Current proposal is not an approved refit plan.
- [ ] Before any rendered expansion smoke, freeze exact templates, per-cell counts, generator/model seeds, development roles, endpoint/action/compute/failure budgets and global exposure bindings. Current template recommendations and upper cost bounds are not this protocol.
- [ ] Execute approved graphics-enabled compatibility/power smoke with all failures retained, separate engine ports/work directories and bounded RAM; no `-nographics` capture. No new compatibility smoke has occurred.
- [ ] Before fresh access, freeze a viable candidate, population, lineage-disjoint paired membership, strongest baselines, legal action/search semantics, re-observation/max-shots/timeouts/retries and failure penalties.
- [ ] Freeze fresh practical/validity margins, confidence/multiplicity/decision hierarchy, useful-mode criteria, paired clustered success power/precision and matched full training/deployment compute/resource budgets.
- [ ] Obtain commit/push/archive authority and archive code/checkpoint/repair assets before fresh access; a local source snapshot is not an archived release.
- [ ] Generate the once-only fresh non-final inventory only after the freeze; keep normal/novel variants in one partition/analysis cluster and prove disjointness from all fitting, opened development and prior final lineages.
- [ ] Run all frozen systems on all assigned fresh units without solvability screening, replacements, favorable seeds or privileged runtime oracle hints; measure real gameplay success/shots, ranking, equal-time prediction/available validity, mode use and full costs.
- [ ] If wider novel evaluation is authorized, report normal/novel performance and paired novelty effects with weighting/multiplicity/uncertainty; keep zero-shot and few-shot protocols separate and all deferred cells untested.
- [ ] Advance to #64/#65 only after the exact supported disposition and every primary/adaptation/strongest-baseline/useful-gameplay/validity/compute gate. The current stop supplies no such authorization.

The issue permits completion via a validated readiness stop. Closing that branch
requires the reporting and handoff above; it does not make these conditional
activities executed or authorize them. No additional tickets, commits, pushes or
new experimental cohorts are implied by this checklist.
