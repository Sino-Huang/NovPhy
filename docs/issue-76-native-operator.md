# #76 native collection/refit operator handoff

The next long-run workflow is implemented and tested. **The ten-item scientific
workflow is not finished:** no new research model has been fitted, no learned
policy has been evaluated in live gameplay, and no fresh/sealed inventory has
been opened. The commands below produce the development data, paired checkpoints
and offline prerequisite diagnostics needed for that next decision.

## Completed engineering evidence

The separately frozen censored-policy smoke completed all three new assignments:
normal single-force, multiple-force and sliding, one shot each. Their captures
have13,147/15,129/5,281 native samples and264/304/107 RGB observations. All three
are genuine stable capture stops, **not gameplay wins**. The new reader also
validated an earlier intact censored prefix read-only, without relabeling it or
granting fitting eligibility. All previous failures remain unchanged.

The canonical allowance is now fully used: eight actual captures across C2,
the native continuation and this smoke. Cumulative capture cost is493.495 active
seconds,1,064,075,879 attempt-artifact bytes and1,872.863MiB peak aggregate RSS.
There are no retries, replacement lineages or further engineering captures to
run under that allowance. Player/source caches are separate from attempt bytes.

Local playback: `data/issue-76-canonical-engineering-media/index.html`.
The original C1 aligned-player gallery remains separate and retained.
C4's validated numeric report is
`data/issue-76-smoke-censored/attempts-3/report.json`.

The refit dry-run detected four missing slots using training metadata, before
any fit. The [capacity amendment](issue-76-native-capacity-amendment.md) retains
every assignment and uses22 slots/288 visual values plus64 memory values. Both
independent predictors therefore use the same352-value carrier. Earlier236/300
models, checkpoints and native captures remain unchanged.

CUDA synthetic optimizer/diagnostic smokes exercised common perception/history,
all nine hybrid mode/horizon combinations, pure predictors, both controllers,
exact equal-time rollouts, publication/validation and no-replay completed resume.
They did not fit on a research lineage. Temporary synthetic weights were removed;
versioned receipts remain in `data/issue-76-native-refit-synthetic-smoke/`.
The latest receipt is `attempt-004.json` and matches the frozen refit sources.
The final regression runs passed182 issue-focused tests and29 native/history
tests (211 distinct tests across the two runs), in addition to the earlier10
Unity native-player fixtures. The long research commands themselves remain unrun.

## Already prepared in this workspace

Do not rerun preparation or change frozen sources/software during these runs.

- Collection: `.local-artifacts/issue-76-native-development-v1/plan.json`.
  Exactly350 normal base lineages: five families ×70, with250 training,
  50 calibration and50 model-selection lineages; at most490 authored-bird shots.
- Refit: `.local-artifacts/issue-76-native-refit-v2/plan.json`.
  Three paired seeds, common weights fitted once per seed, independently fitted
  hybrid/pure predictors and prediction-mode/horizon controllers.
- Engineering test-cost reservation:
  `.local-artifacts/issue-76-native-refit-v2/engineering-cost-reservation.json`.
  Four completed CUDA synthetic smokes consumed6.5244s common,3.9814s predictor
  and10.2298s controller/diagnostic active model-job time. Those costs are divided
  symmetrically among the corresponding real-job budgets, **inside** the existing
  3h/6h/3h ceilings. They are not extra allowances. Do not run additional GPU
  smokes after this reservation without updating the cost audit.

The local frozen players, original source snapshots, recovered Unity projects
and earlier evidence are still prerequisites. Do not delete their `.local-artifacts`
or cache directories. A source-only clone is not yet a portable archived release.

## Run the long jobs

From the repository root, initialize the required environment:

```bash
cd /p/Project/NovPhy
source /home/sukaih/miniconda3/etc/profile.d/conda.sh
conda activate novphy
source env.sh
set -o pipefail
```

First collect the frozen development episodes. This is the long data-generation
command; it has **not** been run by the agent:

```bash
{ time python -u -m scripts.run_issue_76_development --stage development --collect; } 2>&1 | tee -a data/issue-76-development-collection.log
```

It prints episode count, native chunk/RGB counts, elapsed active time, RSS,
artifact bytes and ETA. It runs one graphics-enabled isolated worker, with
private runtime/display/ports/XDG. No `-nographics`, replacements or retries.
The raw capture allowance is12 hours; CPU shard preparation later uses only its
unspent portion. Each shot has a12-second native window and180-second manifest
wait; each episode has420 seconds wall. Attempt-data/RSS limits are256GiB/12GiB.

After it returns—even after a resource stop—publish and validate the retained
collection. These commands also log per-episode progress and ETA:

```bash
{ time python -u -m scripts.issue_76_development_report --publish; } 2>&1 | tee -a data/issue-76-development-report.log
{ time python -u -m scripts.issue_76_development_report --validate; } 2>&1 | tee -a data/issue-76-development-validation.log
```

Then validate the cost reservation and run the refit. This is the long fitting
command; it also has **not** been run on research data:

```bash
python -u -m scripts.issue_76_reserve_fit_cost --validate
{ time python -u -m scripts.run_issue_76_native_refit --run-refit --device cuda; } 2>&1 | tee -a data/issue-76-native-refit.log
python -u -m scripts.run_issue_76_native_refit --validate --device cuda
```

`--run-refit` performs shard preparation, common visual/history fitting and
carrier caching, both predictor fits, both controller fits, and offline
calibration/model-selection diagnostics. It logs shard/cache counts,
optimizer update counts, skipped assignments, loss, elapsed/budget/ETA and
memory. It stops before fitting if fewer than90% of assigned lineages are usable
in any role/family. Corrupt/missing data remains an explicitly failed assignment,
not a replacement or a fabricated terminal.

The shell's `time` output records whole-command wall/CPU time, including startup
and validation overhead. Persistent JSON budgets separately record active
model-job time (including input/cache work), preprocessing and memory. CUDA
allocated/reserved figures are PyTorch metrics, not complete driver memory or
total hardware FLOPs. Validation overhead remains visible in the command logs.

## Resume, limits and outputs

Rerun the same collection or refit command to resume; do not create a new output
root or edit a plan. Collection retains completed/interrupted assignments without
replay. Fitting skips completed stages without changing their cost records.
Ctrl-C during fitting requests a clean pause at an update/shard boundary;
the same command resumes the optimizer counter and remaining budget. An unclean
hard kill or failed optimizer job requires an audit, not automatic replay.
Do not run concurrent copies of the workflow.

Resource stops are terminal for that allowance. Preserve partial checkpoints,
raw traces and unattempted members; do not reset budget files. Timeouts remain
gameplay failures even when their intact observed windows can supply masked
training endpoints. Shared compute is charged symmetrically; unused time is
not transferred between arms. No old checkpoint initializes the new models.

Key outputs:

| Output | Location |
| --- | --- |
| Collection results and complete/raw failed evidence | `.local-artifacts/issue-76-native-development-v1/` |
| Published collection report | `data/issue-76-development-censored/attempts-N/report.json` |
| Prepared data and role/family coverage | `.local-artifacts/issue-76-native-refit-v2/data-index.json` |
| Common and independent model/controller checkpoints | `.local-artifacts/issue-76-native-refit-v2/checkpoints/` |
| Persistent per-stage costs | `.local-artifacts/issue-76-native-refit-v2/budgets/` |
| Offline prerequisite diagnostic | `.local-artifacts/issue-76-native-refit-v2/diagnostic-report.json` |

Individual refit stages are also available: `--prepare-data`, `--fit-common`,
`--fit-predictors`, `--fit-controllers`, `--diagnose`. Use them only in that order
with the existing plan/budgets. The ordinary publication is not an independent
rerun of GPU inference; it validates the saved source-bound records and costs.

## What remains after these commands

The new controllers choose prediction mode/horizon; the offline diagnostic is
not a live gameplay-policy evaluation. It reports equal-time carrier/task-field
errors, selected-mode work, strongest fixed baselines selected on calibration,
and available hybrid symbolic probes. Pure symbolic validity is explicitly
unavailable; an unchanged-carrier reference is not mislabeled a no-model gameplay
baseline. No action-ranking power or model-superiority claim follows from it.

The ten requested requirements currently stand as follows:

| Item | Disposition |
| --- | --- |
| 1. Direction, shared engineering and symmetric costing | Approved; collection/refit implementation and cost reservation complete; real fits operator-pending |
| 2. Exact prospective templates/counts/seeds/roles/budgets | Engineering and development/refit plans frozen before their data/fits; fresh freeze still pending |
| 3. Graphics-enabled compatibility/power smoke | Bounded collection smoke passed with all old failures retained; useful-gameplay power not established |
| 4. Viable candidate, population, strongest baselines and legal live search | Depends on real fitted results and live-policy integration/verification; not complete |
| 5. Fresh statistical margins, confidence/multiplicity and matched full costs | Not frozen or certified; offline diagnostics cannot substitute |
| 6. Commit/push and large-asset archival | Scoped source push authorized; durable archive destination still required; no archived release claim |
| 7. Once-only lineage-disjoint fresh inventory | Not generated |
| 8. All-system fresh gameplay/ranking/equal-time evaluation | Not run |
| 9. Wider novelty/adaptation comparisons | Untested; zero-shot/few-shot remain separate |
| 10. Advance #64/#65 only after supported gates | Not authorized or advanced |

After the long jobs, inspect the actual diagnostics before choosing a candidate
or changing any protocol. A viable result, legal live-gameplay/strongest-baseline/
useful-mode/validity evidence, durable code/checkpoint/repair/player archival and
the full fresh statistical freeze are still required. Supply the durable archive
destination; neither these local snapshots nor source pushes satisfy that gate.
