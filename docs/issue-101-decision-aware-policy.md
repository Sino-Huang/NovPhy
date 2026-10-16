# Issue #101: decision-aware request policy

Parent: #108. The binding shared rules of #85 apply, and #101 adopts the #111 scoring contract. Gate B (#108), #104, #109 and #112 records are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_decision_aware_policy.py` |
| Shared module | `world_model/training/request_policy.py` (API below; reused by #114, #116, #119 and #121) |
| Artifacts | `.local-artifacts/issue-101-decision-aware-policy-v1/` (`findings.md`, `summary.json`, `compute.json`, `selection.json`, `models/*/lam-*/seed-*/report.json`, `records/`, `records-manifest.json`) |
| Freeze | Plan frozen 2026-10-14T10:04:00Z (commit 9bff8308), after a passed smoke and before any data, training or evaluation score. The module bytes are unchanged since the freeze. The runner changed after the publication for guard propagation, the record manifest and reporting only (see "Post-freeze runner edits"). |
| Validation | `python -u -m scripts.run_decision_aware_policy --validate` exits 0. `pytest tests/test_issue_101_decision_aware_policy.py` passes (15 tests). |

Intervals are percentile bootstraps clustered by level (member): 10 000 draws, PCG64 7201, DESCRIPTIVE. The margin is δ = 0.02. Zero engine seconds.

## Result: primary `not_supported_by_this_experiment`

The decision-valued policy J_dec does not beat the best fixed request in hindsight on grid:

| Contrast (grid) | Estimate |
| --- | --- |
| J_dec − F_hindsight (F_hindsight = F-15-continuous) | **−0.0163 [−0.0203, −0.0122]** |

This covers 933 cells (311 levels × 3 seeds), so the result is `not_supported_by_this_experiment`. All five guards pass.

J_dec is `J-dec-1e-01`, the λ chosen on the policy split. J (ICLR) is `J-pe-1e-04`. J_pg is `J-pg-1e-01`.

| Contrast (AUC difference, paired by level and seed) | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| J_dec − F_hindsight | **−0.016 [−0.020, −0.012]** | −0.152 [−0.176, −0.128] | −0.157 [−0.180, −0.135] | −0.019 [−0.035, −0.006] |
| J_dec − F* (cross-inventory r†, see below) | +0.001 [−0.005, +0.008] | −0.152 [−0.176, −0.128] | −0.157 [−0.180, −0.135] | −0.019 [−0.035, −0.006] |
| J_dec − M_dec (jointness) | +0.001 [−0.001, +0.002] | +0.001 [−0.004, +0.006] | +0.003 [−0.002, +0.009] | +0.000 [−0.001, +0.001] |
| J_dec − OL_dec (state dependence) | −0.003 [−0.006, +0.000] | −0.023 [−0.032, −0.014] | −0.023 [−0.032, −0.013] | −0.012 [−0.025, −0.003] |
| J_dec − J (ICLR) | **+0.040 [+0.029, +0.052] supported** | −0.062 [−0.114, −0.009] | −0.078 [−0.118, −0.037] | −0.013 [−0.031, +0.002] |
| J − F_hindsight | −0.057 [−0.068, −0.046] | −0.090 [−0.126, −0.055] | −0.079 [−0.106, −0.054] | −0.006 [−0.014, +0.001] |
| J − M | −0.001 [−0.003, +0.002] | +0.038 [+0.014, +0.063] supported | −0.002 [−0.018, +0.015] | +0.031 [+0.015, +0.050] supported |
| J − OL | +0.013 [+0.010, +0.016] | +0.080 [+0.061, +0.098] supported | +0.050 [+0.034, +0.065] supported | +0.011 [+0.003, +0.019] |
| J_pg − F_hindsight | −0.001 [−0.003, +0.001] | −0.033 [−0.060, −0.006] | −0.064 [−0.087, −0.039] | −0.008 [−0.015, 0.000] |
| J_pg − J_dec | +0.015 [+0.011, +0.019] | +0.119 [+0.107, +0.131] supported | +0.093 [+0.081, +0.107] supported | +0.012 [−0.000, +0.026] |

Tokens follow the δ rule: `supported` iff the mean is ≥ δ and the lower bound is > 0. Every unmarked cell above is `not_supported_by_this_experiment`, except J_pg − J_dec on power, which is `readiness_or_precision_insufficient`.

F_hindsight is F-15-continuous on grid and F-5-micro on the other three inventories. F* is F-5-micro on every inventory. Offset, angle and power are DESCRIPTIVE.

**F* is cross-inventory selection, not independent cross-fitting.** #101 Work 3 asks for "cross-fit r†". The runner applies #96's literal rule: the best fixed request on the other three inventories. The same evaluation levels appear in several inventories, however. In the Gate-B seed-20260908 record, the grid shares 65 levels with angle, 57 with offset and 31 with power. F* is therefore not member-disjoint cross-fitting, and the rule does not remove selection optimism. F* is selected once, before the bootstrap, and F_hindsight is selected on the target inventory itself. Every J − F* and J − F_hindsight interval is therefore conditional on that selection. A member-disjoint estimate would need a separately declared analysis. It is not part of the frozen results. The primary J_dec − F_hindsight disposition is unaffected, because F_hindsight is labelled hindsight.

### λ sweep (grid): MAC vs AUC for each policy type

| type | λ | AUC | MACs / candidate rollout | − F_hindsight | policy-split AUC (seed mean) |
| --- | --- | --- | --- | --- | --- |
| pe (ICLR DP) | 1e-5 | 0.704 [0.684, 0.724] | 181M | −0.058 | 0.710 |
| pe | **1e-4** (J) | 0.706 [0.685, 0.727] | 184M | −0.057 | 0.721 |
| pe | 1e-3 | 0.734 [0.714, 0.755] | 34M | −0.028 | 0.780 |
| pe | 1e-2 | 0.713 [0.693, 0.733] | 50M | −0.049 | 0.722 |
| dec (decision DP) | 1e-4 | 0.681 [0.662, 0.700] | 46M | −0.082 | 0.662 |
| dec | 1e-3 | 0.690 [0.671, 0.709] | 39M | −0.073 | 0.665 |
| dec | 1e-2 | 0.706 [0.686, 0.726] | 47M | −0.056 | 0.683 |
| dec | **1e-1** (J_dec) | 0.746 [0.728, 0.764] | 23M | −0.016 | 0.734 |
| pg (policy gradient) | 1e-4 | 0.764 [0.745, 0.783] | 45M | +0.002 | 0.776 |
| pg | 1e-3 | 0.753 [0.734, 0.772] | 26M | −0.009 | 0.775 |
| pg | 1e-2 | 0.761 [0.742, 0.780] | 22M | −0.001 | 0.784 |
| pg | **1e-1** (J_pg) | 0.761 [0.742, 0.780] | 22M | −0.001 | 0.784 |

Fixed requests for reference (grid AUC, MACs per candidate rollout):

| Request | AUC | MACs |
| --- | --- | --- |
| F-1-continuous | 0.677 | 322M |
| F-5-continuous | 0.749 | 65M |
| F-15-continuous | 0.762 | 22M |
| F-15-macro (Gate-B unit) | 0.744 | 22M |
| C* C-F-15 | 0.783 | 28M |

The no-model action prior scores 0.720.

### Reading

- **Training on decision value does not lift the policy above the best fixed request.**
  - Under the decision-valued DP labels, the AUC rises with λ. The best decision policy (λ = 0.1) is a 15-step policy that varies α by state: 48 % of steps are not the modal request (15, continuous), and schedules depend on the candidate in 732 of 933 cells. It sits 0.016 below F-15-continuous.
  - The decision labels pull the policy toward short Δ when the MAC term is small, and short Δ ranks worse here. This is the same direction as the ICLR J.
- **J_dec beats the ICLR J on grid: +0.040, `supported`.**
  - The gain matches J_dec's coarser schedules. J (λ = 1e-4) is 79 % (1, macro) and costs 184M MACs per rollout; J_dec costs 23M.
  - On offset and angle, J_dec ranks worse than J.
- **Jointness and state dependence add no ≥ δ ranking advantage for these selected policies.**
  - J_dec − M_dec ≈ 0 on every inventory.
  - J_dec − OL_dec ≤ 0 everywhere. Open-loop draws from J_dec's own request frequencies rank as well on grid, and better on offset, angle and power.
  - For the ICLR J, J − OL is positive (supported on offset and angle) and J − M is supported on offset and power. Against a better fixed request, however, J trails F_hindsight by 0.057 on grid.
  - These are contrasts of the selected J and J_dec. They do not show that jointness or state dependence is worthless in general.
- **The policy-gradient alternative collapses to a fixed request.**
  - At λ ≥ 1e-2, each seed requests one pair at every step, for every candidate (0 of 933 cells differ): (15, macro) for seed 20260908, and (15, continuous) for seeds 20260909 and 20260910. The grid schedule table pools the three seeds into a 2 : 1 mix, but no seed mixes requests.
  - J_pg is within 0.003 of F_hindsight (−0.001 [−0.003, +0.001]). This is a small difference, not a demonstrated equivalence. J_pg − J_dec is +0.015 [+0.011, +0.019] on grid: a positive difference below δ (`not_supported_by_this_experiment`). No ≥ 0.02 advantage of J_pg over the selected decision policy is demonstrated.
  - Optimizing the per-cell AUC end to end (Work 2) does not recover state-dependent request value either. The gap between DP and PG comes from choosing the coarsest Δ consistently, not from state dependence.
- **The trained policy still depends on the anchor.** In the wrong-anchor control, J_dec from the 480 fit-split anchors scores 0.618, against 0.746 from the right anchor. Right − wrong is +0.128 [0.117, 0.139] (`supported`), and rank stability is 0.60.
- Every hybrid policy trails C* (C-F-15, 0.783): J_dec − C* = −0.037 on grid. This repeats the Gate-B finding that the matched continuous model ranks better.

## Protocol

### Pool (orchestrator decision)

The pool is the #109 `policy` split as captured by the #104 campaign (stream `issue109-policy`):
- 120 levels and 3480 complete branches, with 0 typed failures. 199 branches remove the pig.
- Grid has 16 candidates per level and angle has 13.
- `split_roles.policy` = "controller (policy) training". The split replaces the 3 #77 controller lineages, as the owner's comment asks.
- The policy roster is disjoint from the fit and evaluation rosters (checked in the smoke and in guard G1). The fit split stays predictor-only, and the evaluation split is held out.

Data built for the pool:
- **Shots:** E99 carriers (`s99.carriers_for`, positions 0..min(frames − 1, 285)) for all 3480 shots, parsed under `deterministic_scoring()`. Each executed action equals its inventory action.
- **Anchors:** per level, the E99 batch-1 parse of frame 0 of its first complete candidate (the retrain rule).
- **Cells:** per level × {grid, angle}, the complete verdict-bearing candidates in ordinal order, mixed cells only. That gives 94 grid and 73 angle cells over 108 levels.

### Predictors

- The LC hybrid fold 'all' Gate-B handoff checkpoints, seeds 20260908, 20260909 and 20260910, hash-checked. Policy seed s uses predictor seed s.
- C* (LC continuous, C-F-15) is scored only as a fixed reference arm. No continuous-family policy is trained, because no acceptance item needs one.

### Policy kinds

All three kinds use the same 242 → 128 → 9 MLP, initialized with `torch.manual_seed(seed + 7700)`, the request set PAIRS, and the clamp60 remaining-frames feature.

- **pe (prediction-error DP, ICLR control).**
  - Labels: the #77 `label_lineage` / `controller_labels` DP with λ as a parameter, on the first 60-frame window of every pool shot.
  - Training: 2 DAgger rounds × 1800 steps, batch 128, AdamW 1e-3. The sampler is `Generator(seed + 7700 + round)` and the unit is the policy level (round r trains on rounds 0..r).
  - Each round has 208 800 label rows.
- **dec (decision-valued DP).**
  - Rollout: every candidate of every mixed policy cell is rolled out from the level anchor under a continuation. Round 0 uses F-15-macro (with a same-α fallback near the end); round 1 (DAgger) uses the round-0 policy.
  - Q value: at each visited state, each feasible request i is tried, and the rollout then continues to step 225. Q(i) is the candidate's pairwise softplus ranking loss (T = 0.1) against the other candidates' continuation costs, plus λ × (work of the remaining transitions) / work(15, continuous).
  - Label: argmin Q. This is a one-step lookahead over a base policy, i.e. approximate DP.
  - **Surrogate qualification.** Work 1 asks for "the change in within-cell ranking loss ... plus λ·MAC". The owner's delegation comment and the frozen plan instead adopt the candidate-normalized loss above, and the code implements that declared surrogate. Dropping the candidate's baseline loss does not change the argmin. The candidate-normalized loss is not, however, numerically the mean full-cell ranking-loss change. For a successful candidate, the full-cell change is the candidate's change divided by the number of successes; for a failed candidate, it is divided by the number of failures. With λ fixed, this changes the ranking/compute trade-off. The #112 reference, `decision_dynamics.pairwise_ranking`, averages over all success–failure pairs. In 140 of the 167 mixed policy cells there is only one success, so the difference matters. Every dec result here describes this surrogate. These results do not show that decision-valued DP labels in general fail, and they do not show that the original cell-loss-delta requirement is met exactly. Testing a class-normalized formulation would need its own frozen plan.
  - Training: the same distillation recipe as pe.
  - Round 0 has 36 795 rows. Round 1 has 37k–467k rows, depending on λ and seed.
- **pg (policy gradient, Work 2).**
  - REINFORCE with reward = per-cell tie-free AUC − λ × the mean work units per candidate. The rollouts are imagined, with no engine calls.
  - 3600 AdamW updates (lr 3e-4, clip 1), each on 8 cells × 8 sampled schedule draws. The advantage is the reward minus the cell's mean over its draws.
- **λ sweep.**
  - pe: {1e-5, 1e-4, 1e-3, 1e-2}; 1e-4 is the ICLR value and gives J.
  - dec and pg: {1e-4, 1e-3, 1e-2, 1e-1}. This grid was set from the smoke label distribution on four policy levels; it runs from a negligible MAC term to a dominant one.
- **Selection (policy split only, frozen rule).**
  - For dec and pg, the λ with the largest seed-mean policy-split pooled AUC wins. This AUC comes from greedy J rollouts on the 167 training cells, and ties go to the larger λ. pe is fixed at 1e-4.
  - Selected: dec 1e-1 (0.734 vs 0.683, 0.665, 0.662). pg 1e-1 (0.784, tied with 1e-2). The selection was written after training and before the first score (G3).

### Scoring (once, in processes that did not train)

- **Arms** follow #96 semantics:
  - the nine fixed requests and C-F-15;
  - J for every (kind, λ), i.e. 12 policies;
  - M (factorized selector, t96) of J and of J_dec;
  - OL of J and of J_dec. Each OL arm draws from its J arm's pooled per-seed request frequencies over every cell of all four inventories and every candidate, with one PCG64(7202) stream in the frozen order (inventory, level, seed).
  - F* (cross-inventory r†, the #96 rule; not member-disjoint, see above) and F_hindsight are selected by member-mean AUC, as in t96.
  - Controls: the no-model action prior (Gate-B rule) and J_dec's wrong-anchor control (480 fit anchors × 16 grid candidates).
- **Cells and anchors:** the Gate-B cells (mixed only: grid 311, offset 76, angle 78, power 47) and the Gate-B anchor store, unchanged.
- **Scoring path:** `request_policy.rollout_endpoints`, the same ops in the same order as `m99.rollout_endpoints`, plus the OL schedule kind. It runs per cell under `deterministic_scoring()`.
- **Cost and AUC:** tie-free cost via `EndpointCosts.row`; cell AUC via `m99.view`.
- **Guards:**
  - G1 isolation;
  - G2 scoring policy;
  - G3 chronology (freeze ≤ training ≤ selection ≤ first score);
  - G4 policy checkpoint hashes equal their training reports;
  - G5 the F-15-macro and C-F-15 rows equal the Gate-B unit and C* records exactly (all 2 × 3 × 512 cell blocks).

  All pass. The frozen `guards.failure` rule says that a failed guard turns every token into `readiness_or_precision_insufficient`. The runner as published applied that only to the primary token. Since the phase-A fix, it also applies the rule to every inventory contrast and to the wrong-anchor token. With all guards passing, this changes no published value.
- **Validation:** `--validate` re-derives every table and the selection from the records and training reports. It byte-compares `compute.json`, `summary.json`, `findings.md` and `records-manifest.json`, the content hashes of all 81 raw records. It also re-scores J_dec seed 20260908 on all 311 grid cells. Rows and schedules match exactly.

### Smoke (pre-freeze)

The smoke used the #77 artefacts, fit-split cells, and four policy levels. It ran no evaluation-split model.

- **`RequestPolicy(PAIRS)` equals `MatchedController(False)`.** Initialization and forward are bit-identical on the #77 seed-20260908 round-0 controller.
- **`window_rows` equals #77 `label_lineage` bit for bit.** This holds in both rounds (observed and DAgger contexts) on the #77 seed-20260908 hybrid checkpoint and its three N1 controller lineages (005, 010, 015), with the #77 code re-run on this hardware. The stored #77 labels (computed on the #85 RTX 3090) differ in 38 labels; this comparison is informational.
- **`distil_round` equals #77 `train_control` bit for bit.** Both rounds were trained on the stored #77 label files of those lineages, 1800 steps each.
- **`rollout_endpoints` equals `m99.rollout_endpoints`** on three fit grid cells. The fixed, joint and factorized kinds match exactly, and a constant schedule equals the fixed request.
- **Decision labels match a brute-force recomputation.** An independent single-row loop gives |ΔQ| ≤ 3.2e-6 on 12 rows. The smoke also recorded the label distribution per λ, the policy-gradient steps and the timings.

## Shared module API (`world_model/training/request_policy.py`)

Parameters for later tickets. The defaults reproduce the ICLR/#77 path bit for bit.

| Item | What it is |
| --- | --- |
| `RequestPolicy(pairs, width=128, remaining_feature="clamp60")` | **Request set R** = `pairs`; logits index `policy.pairs`; Δ > remaining is masked to −inf. **Remaining-frames feature** ∈ `REMAINING_FEATURES` (`clamp60` = ICLR `min(κ, 60)/60`, `over225`, `log1p`; add an entry `(columns, fn)` for #119 variants). State-dict compatible with `MatchedController`. |
| `new_policy(pairs, seed, *, width, remaining_feature, device)` | The #77 init (`torch.manual_seed(seed + 7700)`). |
| `policy_macs(policy)` | Linear MACs of one policy call. |
| `first_window(carriers, action, end, window_length=60)` | **Label window length** (s99 rule, start 0, terminal absorption). |
| `prediction_error_labels(model, policy, z, action, length, contexts=None, *, compute_weight=1e-4)` | **Label kind "prediction error"** (ICLR eq. dp); **λ** = `compute_weight`; the MAC term is `λ · work(model, pair, policy) / work(model, REFERENCE_PAIR)`. |
| `dagger_contexts(model, policy, z, action, length)` | The #77 DAgger visited contexts (B = 1). |
| `window_rows(model, policy, windows, *, aggregated, compute_weight, device)` | #77 `label_lineage` on a list of windows → rows `{z, action, remaining, labels}`. |
| `decision_rows(model, policy, cells, *, continuation, cost_fn, compute_weight, endpoint=225, temperature=0.1, chunk, device)` | **Label kind "decision"**. `cells`: `{z0, actions [n,5], verdicts [n]}`. `continuation`: `{"kind": "fixed", "pair_index": k}` or `{"kind": "policy"}`. `cost_fn(z_end, z0)` → per-row cost. Returns rows plus `cell`, `candidate`, `position`, `q`, `continuation_costs`. Helpers: `continue_rollout`, `fallback_table`, `partner_terms`, `candidate_rank_loss`. |
| `distil_round(policy, rows_per_unit, round_index, *, seed, steps=1800, batch_size=128, learning_rate=1e-3, device)` | The #77 training round. Pair with `aggregate_rounds(history)` for DAgger aggregation. `sample` = the #71 sampler. |
| `policy_gradient(model, policy, cells, *, seed, cost_fn, compute_weight, steps, cells_per_step=8, samples=8, learning_rate=3e-4, grad_clip=1.0, endpoint, device, on_step)` | REINFORCE on per-cell AUC − λ · work. Helpers: `sampled_rollout`, `cell_auc_tensor`. |
| `rollout_endpoints(model, policy, z0, actions, spec, *, select, endpoint=225, record_schedules=False)` | The scoring rollout: kinds `fixed` / `schedule` / any `select` kind (pass `t96.select_pairs`). Returns z_end plus calls, MACs, GPU seconds and per-row schedules. Helpers: `schedule_counts`, `open_loop_schedule` (#96 OL draw). |

Runner-level reuse:
- **Pool and data:** `run_decision_aware_policy.complete_branches()`, `policy_cell_entries()`, `load_pool(OUTPUT)` (anchors, first windows, cells), `read_shot(OUTPUT, key)` (full 0..285 E99 carriers for 225-frame windows).
- **Trained policies:** `load_policy(OUTPUT, kind, λ, seed)`.

## Decisions under delegation (posted on #101)

1. **Controller pool.** The #109 policy split (orchestrator decision).
2. **Shared module.** `world_model/training/request_policy.py` (orchestrator decision).
3. **Compute cap.** 40 GPU-h and 60 h of process wall time summed, on 4 GPUs (orchestrator decision). The projection was 2.4 h elapsed and the observed total is 2.96 GPU-h, so no scope cut was needed.
4. **Predictor.** LC hybrid, Gate-B seeds. C* is a fixed reference arm only, and no continuous policy is trained.
5. **pe labels.** The #77 path unchanged: first window, 2 rounds × 1800 steps, unit = level.
6. **dec labels.** One-step lookahead over a continuation policy (F-15-macro, then the round-0 policy), with a softplus pairwise ranking loss at T = 0.1. The ticket's "or regret" option is not used.
7. **pg recipe.** 3600 updates, 8 × 8 draws, lr 3e-4. The policy starts from the same initialization as pe and dec.
8. **λ grids and selection.** The grids and the policy-split selection rule are as above.
9. **Arms, OL streams, controls and guards.** As above. The Gate-B anchors and cells are reused.

## Compute and disclosures

- Observed GPU time is 10 638 s in total (2.96 GPU-h), with process wall time equal to GPU time:

  | Phase | GPU s |
  | --- | --- |
  | Smoke | 23 |
  | Data (4 workers) | 1 920 |
  | Assemble | 6 |
  | Training (36 groups on 4 GPUs) | 6 378 |
  | Scoring (81 records on 4 GPUs) | 2 312 |

  Elapsed time was about 1 h for data, training and scoring. Validate takes 75 s. Engine time is 0 s.
- The hardware is RTX 5090 instead of the #85 RTX 3090. The bit-for-bit replication is against the #77 code re-run on this hardware.
- **Known before the freeze:** Gate B (unit 0.7444, C* 0.7834, prior 0.7203) and every #96, #100 and #112 result. No policy had been trained on #112 predictors, and no policy output on an evaluation capture existed.
- **λ grid caveat.** The dec/pg λ grid was set from smoke label distributions on four policy levels (training data).
- **Selection caveat.** The policy-split selection AUCs are in-sample, because the training labels used those verdicts. The selection compares λ values within a kind only.
- **Training is not bit-reproducible.** Training ran under deterministic flags, but bit reproducibility of training is not claimed: the CUDA sampler of `pg` and the GPU kernels are assumed deterministic, not verified. The published records and policies are the retained ones.

## Review qualifications (phase-A review)

- **Raw records.** The 81 records (26 arms × 3 seeds plus 3 wrong-anchor records, 196 MiB) stay local in `records/`. The committed `records-manifest.json` binds each of them by size and sha256. `--validate` recomputes that manifest from the files and byte-compares it with the committed copy. The records were not regenerated.
- **G3 chronology.** G3 reads `trained_at`, which `train_group` writes when a group completes; it is not a start receipt. The retained chronology is: freeze 10:04:00, training completions 10:15:41–10:43:48, selection 10:43:51, first score 10:43:53 UTC. This supports selection before scoring. G3 alone does not prove that training began after the freeze. No contrary chronology was observed.
- **Selection table.** The selection is reconstructed from the AUC aggregates in the committed training reports, not from retained per-cell policy-split rows. This shows that the reported aggregates agree; it is not an independent audit of each selection AUC.
- **Checkpoint retrieval.** The 36 policy checkpoints and the Gate-B predictors are bound by sha256 in the training reports and the Gate-B handoff. Their payloads stay in the local `.local-artifacts/` tree of the main checkout (`models/<kind>/lam-<λ>/seed-<s>/policy.pt`) and are not committed.

## Post-freeze runner edits (disclosure)

Commit `a48dea6c` changed `scripts/run_decision_aware_policy.py` after the publication. Every edit concerns reporting, guard propagation or validation, and no evidence changed. The records, models, selection and plan are untouched, and `compute.json` and `summary.json` are byte-identical.

- **Guard propagation.** `invalidate()` applies the frozen `guards.failure` rule to every inventory contrast and to the wrong-anchor token (spec review finding 1).
- **Record manifest.** `--publish` writes `records-manifest.json` (81 records, sizes and sha256), and `--validate` byte-compares it.
- **Reporting.** `findings.md` gains a paragraph that F* is cross-inventory selection, not member-disjoint cross-fitting, with selection-conditional intervals. This is the only change to a published file.
- **Downstream pins.** The #107, #114, #105, #103 and #117 plans pin this runner by sha256. `scripts/post_freeze_amendments.py` lists the amended bytes, and their `load_plan` accepts exactly the frozen or the amended bytes. Each of those runners' `--validate` still regenerates its publication byte for byte through this module.
