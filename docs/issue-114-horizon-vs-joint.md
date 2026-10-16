# Issue #114: horizon-only versus joint request selection

Parent: #108. Gate B passed; this experiment uses the #104 sealed #109 evaluation split. Grid is primary; offset, angle and power are DESCRIPTIVE. The #85 conventions and #111 deterministic, tie-free scoring contract apply. Earlier published evidence and the ICLR submission remain unchanged.

| Deliverable | Location |
| --- | --- |
| Runner | `scripts/run_horizon_vs_joint.py` |
| Shared selectors | `world_model/training/request_selectors.py` |
| Selector tests | `tests/test_issue_114_request_selectors.py` |
| Artifacts | `.local-artifacts/issue-114-horizon-vs-joint-v1/` |
| Freeze | `8f86925d`, 2026-10-14T11:29:59Z; smoke passed before training/scoring |
| Selector implementation | `82a5878e` |
| Validation command | `python -u -m scripts.run_horizon_vs_joint --validate` |
| Validation evidence | Exit 0 (278 s reported): five generated artifacts byte-compared; all 311 grid cells re-scored exactly for J-cont, T-cont and J-as-cont, seed 20260908. See `verification.json`. |

## Execution disclosure

An operational duplicate-launch incident affects literal scored-once compliance. The async shell launch returned completion while its `nohup` children remained alive. A second scoring batch and a diagnostic batch overlapped the original frozen shards. Eight duplicate workers were terminated; the original four continued. At least two fixed-arm prefix records were written twice, and other prefix records may also have been overwritten. No checkpoint, training label, lambda, arm, statistical contrast, or frozen token rule changed in response to evaluation results. There were no retries of typed model failures.

`execution-incident.json` preserves process identities, handles, inferred launch/cancellation times, the available overwrite evidence and its limitations. Exact discarded GPU active time and all pre-overwrite timing headers were not retained. The discarded allocation is conservatively charged as an **estimated 8200 process/GPU-allocation seconds**, separately labelled in `ledger/operational-discarded-upper-bound.json`; it is not a measured CUDA-active-time claim.

Deterministic recomputation is not independent new evidence. The numerical G1–G6/token rules remain the frozen rules, but their passing does **not** establish literal one-look compliance. Consequently, the report distinguishes numerical dispositions from execution-integrity qualification. Wall-time points include concurrent-worker contention and must not be interpreted as isolated-device latency benchmarks.

The post-freeze runner amendment is reporting-only: incident disclosure, control annotations in every results table, wrong-anchor top-3/compute summaries derived from already-retained rows, and explicit endpoint-clamp/all-lambda-interpolant interpretation. Training, rollout/scoring, frontier interpolation, decomposition, guards and token functions are unchanged. The plan and its original runner hash remain intact.

## Protocol

### Predictors, controller pool and requests

- Predictors: #112 v3 retrain LC hybrid and LC continuous C* fold `all`, Gate-B checkpoint hashes, seeds 20260908/09/10. Policy seed matches predictor seed.
- Pool: #101's E99 `data/pool.pt`, the #109 policy split, 120 levels and 3480 complete branches. The first 60-frame window of every shot is used; the unit is the policy level. Policy levels are disjoint from predictor-fit and evaluation levels.
- Requests: delta-major `{1,5,15} × {continuous,micro,macro}`. Index 0/3/6 is `(1/5/15, continuous)`.
- Evaluation: unchanged Gate-B anchors and mixed cells: 311 grid, 76 offset, 78 angle and 47 power, each across three seeds.
- Cost: #111 tie-free endpoint cost, evaluated against the starting anchor under `deterministic_scoring()`. No engine calls.

### Work 1: restricted argmax on J

J is the existing #101 prediction-error policy at lambda `1e-4`. The same logits are masked to an allowed subset; there is no training:

- T-continuous / T-micro / T-macro: choose delta at one fixed description.
- D-1 / D-5 / D-15: choose description at one fixed delta.
- Ties go to the lower request index; feasibility remains the policy's `delta <= remaining` mask.

D-5 and D-15 remain feasible because their fixed delta divides the 225-step endpoint. A general restriction without a feasible request raises rather than silently selecting a disallowed request.

### Work 2: trained continuous-description ablation

**J-cont** uses exactly J's `242 → 128 → 9` policy, initialization and two-round DAgger/distillation recipe. Six noncontinuous logits are masked to negative infinity. Its DP and DAgger choices are restricted to `R_cont = {(1,continuous),(5,continuous),(15,continuous)}` on the **same LC hybrid predictor**, not a different trunk. Labels still index the nine-logit policy, and the compute regularizer counts the full policy.

All families use the #77/#101 prediction-error DP: delta-weighted one-step carrier MSE plus lambda times normalized linear MACs, backward DP, first window per shot. Round 1 overwrites visited contexts with round-0 imagined states; aggregation trains on both rounds. Each round performs 1800 AdamW updates, batch 128, learning rate `1e-3`, CPU sampler `seed + 7700 + round`, policy initialization `seed + 7700`. Remaining time uses the ICLR `min(remaining,60)/60` feature.

The smoke reproduced #101 J's round-0 labels and both distillation checkpoints bit for bit, checked restricted-to-all-nine equals unrestricted labels/rollouts, and checked selector/replay equivalence on three **fit-split** grid cells.

### Work 3: lambda sweeps and matched compute

All three families use eight lambdas:

`{1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`.

- **J:** #101's four `pe` policies at `{1e-5,1e-4,1e-3,1e-2}` are reused read-only; the remaining four lambdas are trained with the same recipe/pool/seeds.
- **J-cont:** all eight lambdas are relabelled/retrained on the hybrid trunk with the six-logit mask.
- **C-J:** all eight lambdas are relabelled/retrained on the LC continuous C* predictor; policy architecture `242 → 131 → 3`, bit-identical to `MatchedController(True)` in the smoke. Pool, seeds and update count match J.

The `1e-4` points transfer the **ICLR operating recipe**, not a promise of numerically reproducing old-checkpoint MACs. The historical #74 values (118.7M hybrid, 356.0M continuous-adaptive MACs per candidate/state) were measured with earlier predictors/carriers. They cannot be interpreted as the MACs of the LC/E99 policies. The current matched budgets are explicitly the newly measured J and C-J `1e-4` points, as required by Work 3.

Pre-freeze policy-window probes at `{0,1e-6,0.3,1,10}` showed near-saturation outside the chosen grid. These are training-data label probes, not held-out selection; the grid was frozen before the new policies trained.

Each lambda reports:

- Member-clustered tie-free AUC and top-3 intervals.
- Linear MACs per candidate rollout to step 225, including the policy calls.
- CUDA-synchronized elapsed host wall time per state/cell, where all candidates are batched.
- All-tied cells and tied success/failure pair counts.

Top-3 hits when any successful candidate occurs among the three lowest-cost finite candidates, ordered by cost then ordinal. The chance reference is `1 - choose(n-k,3)/choose(n,3)` (one when `n <= 3`).

Frontier interpolation uses **every** lambda point sorted by pooled mean MACs, piecewise linear in log10 MACs, without outcome-based pruning. Interpolation weights are common across target members/seeds and are held fixed for paired intervals; they do not account for uncertainty in the MAC budget itself. The two budgets are J's and C-J's `1e-4` MACs. Above a family's largest-MAC point, that endpoint is used and explicitly flagged as underspending the budget. Below its smallest-MAC point, the estimate is undefined—no extrapolation. A clamped comparison is an at-most-budget contrast, not exact equal expenditure.

`frontier.json` exports all eight J-cont and C-J policy points, all inventories, AUC/top-3/ties/MACs/wall times, policy checkpoint paths/hashes, fixed/restricted references, controls and interpolation pieces for #103. Checkpoints remain local, not git payloads.

### Work 4: exact compute decomposition and counterfactual

For first minus second, let `n` be predictor calls per candidate, `c` their mean predictor MACs, `p` policy calls and `m` MACs per policy call. Then:

`MAC1 − MAC2 = (n1 − n2) c2 + n1 (c1 − c2) + (p1 m1 − p2 m2)`.

The terms respectively isolate call count/horizon, per-call cost, and policy work. Against C-J, the per-call term is further split into J's description surcharge above **hybrid-continuous** and the hybrid-versus-continuous-predictor trunk difference. This trunk term matters: C* is capacity matched, not equal in per-call linear MACs.

Per-call predictor MACs are 1,432,448 (hybrid continuous), 1,795,456 (micro), 1,464,704 (macro), and 1,852,200 (C* continuous). Policy calls cost 32,128 MACs for J/J-cont and 32,095 for C-J. These counts exclude nonlinear, indexing/masking and memory work; they are not full FLOPs.

**J-as-cont** replays each candidate's recorded J schedule on the hybrid predictor with its delta unchanged and description replaced by continuous. It does not call the policy. Thus its predictor-only compute isolates the horizon schedule; the policy overhead and description surcharge are shown separately. A J-typed-failure candidate retains its failure rather than inventing a continuation. A replay endpoint can independently become nonfinite and is retained as such.

### Work 5: controls and disposition

Every results table carries the no-model action prior and the wrong-anchor diagnostic. The prior uses fit-split action success frequencies under the Gate-B rule. Wrong-anchor controls reuse 480 fit-split anchors and the 16 grid candidates for J/J-cont/C-J at `1e-4`. Their AUCs are averaged across wrong anchors for each target evaluation member/seed. They are **grid-only**, and are diagnostic references rather than lambda frontier operating points; dashes in unrelated decomposition/schedule columns are not missing model scores.

Intervals are percentile member/level-clustered bootstraps, 10,000 PCG64(7201) draws, DESCRIPTIVE. The margin is delta = 0.02. For each contrast, `supported` requires mean >= 0.02 and lower > 0; `not_supported_by_this_experiment` requires upper < 0.02; otherwise `readiness_or_precision_insufficient`.

The primary claim is **adaptive alpha adds decision value beyond adaptive delta at matched compute**: J minus J-cont on grid at both declared budgets. It is supported only if both contrasts are supported; not supported if either upper bound is below the margin; otherwise precision/readiness is insufficient. J minus C-J uses the same rule as a secondary matched-compute contrast. Unmatched `1e-4` contrasts include J−J-cont and J−T-continuous in every inventory.

Frozen guards: G1 split isolation, G2 deterministic scoring policy, G3 freeze/training/scoring chronology, G4 checkpoint binding, G5 replication against #101 and Gate B, G6 schedule-subset compliance. Execution-integrity disclosure is separate; the accidental duplicate launches do not justify changing these rules or manufacturing a supported token.

The frozen failure wording turns only the primary and secondary dispositions into `readiness_or_precision_insufficient`. That rule is kept. Since the phase-A fix, a failed guard also adds a `guard_failure` block to `compute.json`, `summary.json` and `findings.md`. The block names the failed guards and states that no other token in the publication (fixed-lambda, matched-budget, restricted-arm, inventory and wrong-anchor contrasts) is interpretable. Every retained guard passes, so the block is absent and no published file changed.

## Published results on the sealed evaluation split

The full per-inventory table, every lambda point, tie counts, matched contrasts, schedules and controls are in `findings.md` and `compute.json`. Grid has 311 level clusters × 3 seeds (933 cells; 14,928 candidates and 18,945 success/failure pairs). The following are the `1e-4` operating points; intervals are member-clustered and DESCRIPTIVE.

| Policy / control | Grid AUC | Top-3 hit rate | Linear MACs / candidate | Calls / candidate | Wall seconds / state |
| --- | --- | --- | --- | --- | --- |
| J | 0.7057 [0.6846, 0.7267] | 0.5102 [0.4620, 0.5606] | 184.0M | 121.1 | 0.339 |
| J-cont | 0.6856 [0.6647, 0.7061] | 0.4662 [0.4223, 0.5123] | 122.7M | 83.8 | 0.227 |
| C-J | 0.6999 [0.6827, 0.7170] | 0.4469 [0.4148, 0.4802] | 234.7M | 124.5 | 0.168 |
| T-continuous | 0.7025 [0.6813, 0.7238] | 0.5348 [0.4909, 0.5809] | 196.1M | 133.9 | 0.358 |
| J-as-cont (recorded schedule replay) | 0.7021 [0.6805, 0.7231] | 0.4920 [0.4480, 0.5370] | 173.5M | 121.1 | 0.100 |
| No-model action prior | 0.7203 [0.6965, 0.7442] | 0.4180 [0.3633, 0.4727] | 0 model MACs | 0 predictor calls | not timed |
| Wrong-anchor J (grid diagnostic) | 0.6721 [0.6506, 0.6935] | 0.4231 [0.3774, 0.4704] | 181.5M | — | 0.243 |

All 24 grid policy/lambda points have zero all-tied cells and zero tied success/failure pairs (0/18,945 each). The prior has 42/18,945 tied pairs and no all-tied cells. Wall times are synchronized host elapsed times under contention, not isolated-device throughput; zero prior model work does not assert zero CPU evaluation time.

### Frozen matched-budget dispositions

| Declared budget | J − J-cont AUC | Frozen token | J − C-J AUC | Frozen token |
| --- | --- | --- | --- | --- |
| J `1e-4`: 183,958,616 MACs / candidate | +0.0201 [0.0154, 0.0248] | supported | +0.0111 [0.0050, 0.0174] | not_supported_by_this_experiment |
| C-J `1e-4`: 234,671,220 MACs / candidate | +0.0201 [0.0154, 0.0248] | supported | +0.0058 [0.0000, 0.0117] | not_supported_by_this_experiment |
| No-model action prior AUC | 0.7203 [0.6965, 0.7442] | reference | — | — |
| Wrong-anchor J AUC | 0.6721 [0.6506, 0.6935] | reference | — | — |

**The frozen primary token is `supported`, with important qualifications.** J-cont's largest observed expenditure is only 122.7M, so it is clamped at both budgets. J is also clamped at the C-J budget. These are **at-most-budget comparisons, not exact equal expenditure**; both primary estimates are the same endpoint contrast, not independent replications. At the J budget C-J interpolates as 0.797 × lambda `1e-3` + 0.203 × lambda `1e-4`. At the C-J budget it uses its actual `1e-4` point.

The frozen curve is an all-lambda interpolant, **not the best AUC achievable below a budget**. Cheaper points can rank better: at lambda `1e-1`, J has grid AUC 0.7700 at 22.1M MACs and J-cont has 0.7574 at 22.1M. No Pareto-dominance or exact-matched-expenditure claim follows from the primary token. Literal scored-once compliance also failed, as disclosed above; this publication is not a pristine one-look confirmatory execution.

The narrower grid checks do not support a delta = 0.02 gain from alpha selection: J−T-continuous is +0.0032 [−0.0075, 0.0137], and J−J-as-cont is +0.0037 [−0.0021, 0.0095], both `not_supported_by_this_experiment`. J−prior is −0.0146 [−0.0388, 0.0088], likewise not supported. Wrong-anchor J degradation is +0.0336 [0.0299, 0.0373] (`supported`), but J-cont's +0.0152 [0.0112, 0.0192] does not meet the margin. These diagnostics constrain interpretation rather than changing the predeclared primary rule.

### Compute decomposition at lambda `1e-4`

Values are first minus second, millions of linear MACs per candidate and state. All terms sum to the observed difference (numerical residual below 0.000001 MAC).

| Contrast / control | Total | Calls / horizon | Per-call alpha | Trunk | Policy |
| --- | --- | --- | --- | --- | --- |
| J − J-cont | +61.241M | +53.460M | +6.583M | 0 | +1.199M |
| J − C-J | −50.713M | −6.353M | +6.583M | −50.836M | −0.106M |
| J − J-as-cont | +10.474M | 0 | +6.583M | 0 | +3.891M |
| No-model action prior: AUC 0.7203 | 0 model MACs | — | — | — | — |
| Wrong-anchor J: AUC 0.6721 | diagnostic | — | — | — | — |

The J/C-J saving is principally the lower hybrid-continuous per-call trunk MACs, not savings from micro/macro: the alpha surcharge is positive. J spends more than J-cont mainly through more transition calls. The replay preserves J's 121.1 calls but removes policy work and the description surcharge.

### Recorded experiment compute

The published ledger charges **9.57 GPU-allocation hours and 9.57 summed process-wall hours**, comprising 3.12 h retained training, 4.15 h retained scoring, 0.015 h smoke and a separately labelled 2.28 h conservative discarded-process upper bound. Reused #101 J training is another 0.60 h, disclosed separately. Engine time is zero. The experiment remains within the frozen 12 GPU-hour / 24 process-wall-hour caps; no acceptance item was cut. This is process-allocation accounting (two workers per GPU), not a measurement of CUDA utilization or elapsed fleet wall time.

`verification.json` separately records the two publication processes and two exercised validation processes: 923.01 observed process-wall seconds, conservatively charged as one GPU's allocation throughout. Including this reporting/verification overhead gives **9.83 newly allocated GPU hours and 9.83 summed process-wall hours** (plus the separately disclosed 0.60 reused #101 hours), still within the caps. It is kept outside the experiment ledger so recording the validation does not alter the artifacts whose bytes it verifies.


## Shared selector API for #105

`request_policy.py` and the #96 runner are hash-bound by #101's frozen plan. They remain unchanged so #101 validation is not invalidated. The new module is the single restricted-selection implementation for #114/#105:

```python
from world_model.training import request_selectors as rs

spec = rs.restricted_spec(PAIRS, [(1, "continuous"), (5, "micro"), (15, "macro")])
# rs.diagonal_spec(PAIRS), rs.anti_diagonal_spec(PAIRS)
# rs.temporal_spec(PAIRS, "continuous"), rs.description_spec(PAIRS, 5)
# rs.monotone_spec(), rs.replay_spec(per_candidate_schedules)
z_end, timing = rs.rollout_endpoints(
    predictor, policy, z0, actions, spec,
    select=t96.select_pairs, record_schedules=True,
)
```

`monotone_spec()` is path dependent. Given previous `(delta_p, alpha_p)`, allow `(delta, alpha)` iff `(delta − delta_p) × (alpha_index − alpha_index_p) >= 0`, with `continuous < micro < macro`; equal delta or description is unrestricted, and the first request is free. As delta grows the description cannot refine; as delta shrinks it cannot coarsen. This is a consecutive-request constraint, not a global monotone function across all previously visited deltas.

`MaskedRequestPolicy`, `new_masked_policy` and `restricted_window_rows` implement full-head masked-logit training and the subset DP. Thirteen selector tests cover ties/feasibility, all-nine equality with J (selection and full rollout), T/D equivalence, diagonal/antidiagonal masks, monotone constraints, per-row replay and restricted labels/distillation.

## Resource authorization and chronology

The binding conventions require a declared frozen compute cap. This plan declared 12 GPU-allocation hours and 24 summed process-wall hours; the pre-freeze projection was 5.8 GPU hours, approximately 2.9 elapsed hours on two GPUs. The cap is an experiment stop/accounting limit, not permission to omit acceptance items; no acceptance scope has been reduced.

Training and scoring used assigned GPUs 0 and 1 (two processes per GPU); GPU-allocation/process wall accounting is conservative, not utilization-weighted. Hardware is RTX 5090 rather than the historical RTX 3090. Reused #101 training is separately disclosed, not double-charged as newly executed training. Terminal artifacts, timing records and the execution incident remain in the output directory. Delegated decisions were posted on #114 before publication.

## Post-freeze runner edits (disclosure)

Commit `3834c53e` changed `scripts/run_horizon_vs_joint.py` after the publication. The edits concern reporting and input validation only. No evidence changed, and every published file is byte-identical (`--validate` exits 0).

- **Failed-guard reporting.** `guard_failure()` adds an explicit statement whenever a guard fails: secondary tokens are then not interpretable. The frozen rule, which invalidates the primary and secondary dispositions only, is kept.
- **Input check.** `load_plan` checks the pinned #101 runner through `scripts/post_freeze_amendments.py`. It accepts exactly the frozen bytes or the disclosed #101 amendment. The #103 plan pins this runner, and the same registry lists its amended bytes.
