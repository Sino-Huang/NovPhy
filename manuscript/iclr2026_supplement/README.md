# Supplementary outcome records

These files repackage records that were already retained: per-unit outcome records, split memberships and published aggregate tables. No model, engine run, training run or new statistic was run to produce them. `build_supplement.py` reads the retained artifacts and writes every file. It then re-derives each published value it can from the packaged rows and records the comparison in `checks.csv`. All source paths are relative to the NovPhy repository root. Ticket numbers (for example #96) identify the internal experiment record that produced each artifact.

Regenerate with `python build_supplement.py`. The script needs Python ≥ 3.10 and NumPy. It expects to sit inside the NovPhy repository, whose root is `../..` relative to this directory. Output is byte-identical across runs. The script exits non-zero if any check fails or if any output contains an absolute machine path or a repository URL.

**Check result: 821 of 821 checks match** (see `checks.csv`).

## Contents

| File | Rows | What it is |
|---|---|---|
| `dev_issue96_cells.csv` | 3,186 | Development ranking cells behind the submission-time J − F* tables (#96): one row per (inventory, level, seed, arm). |
| `dev_issue96_verdicts.csv` | 882 | Engine verdicts behind those cells: one row per (inventory, level, candidate). |
| `clevrer_membership.csv` | 15,000 | CLEVRER pixel study (#106): every video ID 0–14999, with its role and whether it was included. |
| `physion_membership.csv` | 3,146 | Physion-Dominoes pixel study (#106): every included trial per release split, with overlap flags for testing trials. |
| `physion_overlap_pairs.csv` | 136 | Physion physical-overlap audit (#106): one row per (testing trial, fitting-split trial) pair that shares every physical parameter except colour. |
| `physion_split_overlap_counts.csv` | 3 | Physion split-pair overlap counts (#106). |
| `novphy_sealed_levels.csv` | 1,000 | NovPhy sealed split: every predictor-fit, policy-fit and evaluation level, with family and the inventory coverage that local records list. |
| `live_issue102_closed_loop_table.csv` | 48 | Byte copy of the published live retry-from-start table (#102): all sets, all arms, all columns. |
| `live_issue102_outcome_counts.csv` | 48 | Per-arm episode outcome counts of the live evaluation (#102), from its `compute.json`. |
| `live_issue102_typed_failures.csv` | 17 | Every live execution that ended in a typed engine failure (#102). |
| `frontier_issue103_table.csv` | 560 | Byte copy of the published ranking/compute frontier table (#103): both sets, all four inventories, all policies, all columns. |
| `checks.csv` | 821 | Every published-versus-recomputed comparison. |
| `build_supplement.py` | — | The packaging script. |

## Records not included

The rebuttal promised per-episode and per-level outcome records. The record families below are not part of this supplement. Where the published aggregate built from them exists, it is included; the right-hand column names the file.

| Record family not included | Units | Published aggregate included |
|---|---|---|
| Live retry-from-start episodes (#102), one record per (arm, level, seed) episode | 17,600 episodes on the 400 evaluation levels (1,200 per seeded arm; 400 each for `prior` and `oracle`), drawing on 4,645 live executions | `live_issue102_closed_loop_table.csv`, `live_issue102_outcome_counts.csv`, `live_issue102_typed_failures.csv` |
| Level-by-candidate engine verdicts for the evaluation split | 11,680 launches on 400 levels | Mixed-cell flags per level and inventory in `novphy_sealed_levels.csv` |
| Sealed-split ranking cells, one record per (policy, inventory, level, seed): main ranking table and Appendix D | 1,536 cells per policy (grid 933, offset 228, angle 234, power 141) for each of the 71 policies | `frontier_issue103_table.csv` |
| Submitted-checkpoint rerun cells on the evaluation split (#104) | 1,536 cells for each of 18 arms | None in this supplement |

The only per-unit records included are the development records behind the submission-time tables (#96, below).

## Per-unit development records (#96)

### `dev_issue96_cells.csv` — development ranking cells (submitted-checkpoint family)

- **Population.** All 177 scheduled cells, without filtering. A cell is one (inventory, level, seed). The cells cover the 15 development levels of the #77 N1 cohort; angle covers 14 of them (see below). Each cell has a row for each of the 18 arms, giving 3,186 rows, one per retained decision record. Cells whose levels have only successes or only failures are kept and marked `mixed_level = 0`.
- **In-sample caveat.** These are development (in-sample) diagnostics, not identity-disjoint evaluation: 11 of the 15 levels have `exposure_role = training`. They are separate from the evaluation-split results.
- **Excluded level.** The angle inventory excludes `issue-77-n1-005`, whose angle oracle anchor #89 declared typed-unmeasurable before #96 was frozen. Its verdict rows appear in `dev_issue96_verdicts.csv` with `in_issue96_universe = 0`.
- **Published-aggregate population.** The published aggregates use mixed cells only (`mixed_level = 1`): grid 42 cells on 14 levels, offset 27 on 9, angle 21 on 7, power 24 on 8. AUC means are level-clustered: the mean over levels of the per-level mean of the defined cell AUCs. Intervals are level-clustered percentile bootstraps (10,000 draws, PCG64 seed 7201, quantiles 0.025/0.975), labelled descriptive. Inventories are never pooled.
- **Sources.**
  - Decision records: `.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/records/decision--<inventory>--<level>--seed<seed>--<arm>.json`.
  - Arm and cell definitions: `.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/plan.json`.
  - Published aggregates: `.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/summary.json`.
  - Published per-cell rows (mixed cells): `.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/compute.json .inventories.<inventory>.rows`.

| Column | Meaning (unit) |
|---|---|
| `inventory` | Candidate launch set: `grid` (#93 4×4 Cartesian drag grid, 16 candidates; primary), `offset` (#93 offset-angle sweep at radius 80 px, 11), `angle` (#87 radius-80 launch-angle sweep, 9–13), `power` (#94 angle × launch-power sweep, 20; outside the training action support, secondary). |
| `level` | Development level identifier (#77 N1 cohort member). |
| `state` | Canonical engine state of the level from which the decision anchor and verdicts are taken. |
| `exposure_role` | Role of the level in the checkpoint's data (`training` or `calibration`), from `.local-artifacts/issue-93-second-parameterization-v1/plan.json .states`. |
| `seed` | Checkpoint training seed (20260908, 20260909, 20260910). |
| `arm` | Request schedule; definitions in `plan.json .universe.arm_definitions`. `F-<Δ>-<α>`: hybrid checkpoint, fixed request (Δ ∈ {1, 5, 15} carrier steps; α ∈ {continuous, micro, macro}) at every step. `J`: hybrid checkpoint, joint argmax over the nine request logits at every step. `M`: factorized selector from the same logits. `T`: temporal-only (α fixed at J's modal α). `D`: description-only (Δ fixed at J's modal Δ). `OL`: open-loop draw from J's request frequencies. `C-F-<Δ>`: continuous checkpoint, fixed (Δ, continuous). `C-J`: continuous checkpoint, horizon-only adaptive. |
| `checkpoint_family` | `hybrid` or `continuous`. |
| `candidates` | Number of candidates in the inventory for this level. |
| `verdict_slots` | Candidates with a retained engine verdict. |
| `verdict_successes`, `verdict_failures` | Verdict-bearing candidates with verdict 1 / 0 (success = engine pig removal by the single shot). |
| `nonfinite_predictions` | Verdict-bearing candidates whose predicted cost was non-finite. These are typed failures and are excluded from ranking. |
| `admissible` | Candidates with both a verdict and a finite predicted cost. This is the AUC and top-1 population. |
| `mixed_level` | 1 if the level's verdicts contain at least one success and one failure in this inventory. Only these cells enter the published aggregates. |
| `cell_auc` | Within-cell ordering AUC: the probability that a successful admissible candidate has a lower predicted cost than a failed one, counting ties as 0.5. Predicted cost is the frozen count cost of the predicted endpoint carrier after 225 carrier steps; lower is better. Empty when the cell has no success or no failure among admissible candidates. |
| `top1_hit` | 1 if the lowest-cost admissible candidate (ties to the lower ordinal) is successful. |
| `chance_top1` | Successes / admissible (the top-1 rate of a uniform random pick). |
| `all_tied` | 1 if every admissible candidate has the same predicted cost. |
| `chosen_ordinal` | Candidate the arm selected over all finite-cost candidates (minimum predicted cost, ties to the lower ordinal), as stored in the record. |
| `chosen_verdict` | Verdict of `chosen_ordinal`. Empty if that candidate has no verdict. |
| `decision_failure` | Typed decision failure stored in the record. All rows are empty: there were none. |
| `source_record` | Relative path of the decision record. |

### `dev_issue96_verdicts.csv` — development engine verdicts (level × candidate)

- **Population.** Every candidate of every inventory on the 15 development levels: 882 rows (grid 240, offset 165, angle 177, power 300). Typed execution failures are included.
- **Sources.**
  - Grid and offset: `.local-artifacts/issue-93-second-parameterization-v1/records/oracle--<level>--{g,o}<ordinal>.json`.
  - Power: `.local-artifacts/issue-94-launch-power-v1/records/oracle--<level>--p<ordinal>.json`.
  - Angle: `.local-artifacts/issue-87-closed-loop-oracle-v1/records/oracle--<state>--a<ordinal>.json`, overridden by the #89 completion record when that record has an outcome (`.local-artifacts/issue-89-oracle-completion-v1/records/`).
  - Candidate actions: the frozen inventories in the corresponding `plan.json` files.
- **Counts.**
  - In the #96 universe: 41 successes, 818 failures and 19 typed execution failures, all on `issue-77-n1-005` (grid 8, offset 6, power 5).
  - The excluded angle level adds 4 rows: 1 success, 2 failures and 1 typed failure.

| Column | Meaning (unit) |
|---|---|
| `inventory`, `level`, `state` | As above. |
| `ordinal` | Candidate index within the inventory. Matches the decision records' `ordinal`. |
| `branch_identity` | Candidate identifier. |
| `drag_x`, `drag_y` | Slingshot drag vector as recorded in the frozen inventory (pixels). |
| `release_time_ms`, `tap_time_ms` | Release and tap times (ms). |
| `verdict` | 1 = the executed single shot removed the pig (engine-side `pig_removed` channel; `first_shot_success`), 0 = it did not. Empty = no verdict. |
| `typed_failure` | Failure class when `verdict` is empty (`execution_failure`). |
| `typed_failure_detail` | Recorded failure message. |
| `verdict_record` | Relative path of the record that supplied the verdict or failure. |
| `in_issue96_universe` | 1 if the row belongs to the #96 cell universe. 0 only for the 4 angle rows of the excluded level. |

Headline #96 values, reproduced exactly from these rows:

| Inventory | Mixed cells (levels) | J AUC [95% CI] | F* (AUC) | J − F* [95% CI] | J / F* top-1 hits |
|---|---|---|---|---|---|
| grid | 42 (14) | 0.4706 [0.4158, 0.5256] | F(5,macro) (0.4006) | +0.0700 [−0.0057, +0.1437] | 0/42, 0/42 |
| offset | 27 (9) | 0.5074 [0.4556, 0.5593] | F(1,macro) (0.5037) | +0.0037 [−0.0796, +0.0926] | 0/27, 1/27 |
| angle | 21 (7) | 0.5655 [0.5198, 0.6250] | F(1,macro) (0.5060) | +0.0595 [+0.0079, +0.1131] | 0/21, 0/21 |
| power | 24 (8) | 0.5287 [0.4693, 0.5906] | F(5,macro) (0.5014) | +0.0273 [−0.0691, +0.1281] | 0/24, 0/24 |

F* is the cross-fit fixed request: the F arm with the highest mean AUC over the other three inventories. All #96 intervals are descriptive, and the #96 precision guards failed (`summary.json .guards_pass = false`).

## Split membership

### `clevrer_membership.csv` — CLEVRER pixel study (#106)

- **Population.** Every CLEVRER video ID in the two released annotation splits used: `train` 0–9999 (fitting) and `validation` 10000–14999 (evaluation). That is 15,000 rows: 6,285 + 3,132 included, 3,715 + 1,868 excluded.
- **Source.** `.local-artifacts/issue-106-pixel-benchmarks-v1/plan.json .membership.clevrer.<split>.scenes` (included IDs) and `.roles.clevrer`.
- **How excluded IDs are derived.** An excluded ID is an ID in the split's range that is not in the recorded `scenes` list. The plan records only the excluded count (`skipped_repeated_key_or_contract`). Per video, it does not record which membership condition failed.
- **Membership rule.** Frozen in `world_model/data/pixel_benchmarks.py:clevrer_member`. All of the following must hold:
  - at least one object;
  - pairwise-distinct (colour, material) keys, so each object maps to a unique carrier slot;
  - every shape, colour and material is in the carrier vocabulary;
  - frames 0–127 are present contiguously.

| Column | Meaning |
|---|---|
| `video_id` | CLEVRER scene index. |
| `release_split` | `train` or `validation` (CLEVRER release split). |
| `role` | Use in this study: `train` = encoder, predictor and policy fitting; `validation` = rollout-error and QA evaluation. |
| `included` | 1 if the video is a member, 0 if it was excluded. |
| `exclusion_reason` | `failed_membership_rule` for excluded videos (the specific failed condition is not recorded); empty for included videos. |

### `physion_membership.csv`, `physion_overlap_pairs.csv`, `physion_split_overlap_counts.csv` — Physion-Dominoes (#106)

**Population.** All included Dominoes trials of the three releases: dynamics 2,000, readout 996 and testing 150 (3,146 rows). No trial was excluded: the plan's `excluded_names` is empty for every split.

**Sources.**
- `.local-artifacts/issue-106-pixel-benchmarks-v1/plan.json .membership.physion` (trials, roles, frozen digest overlap).
- `.local-artifacts/issue-106-pixel-benchmarks-v1/leakage-audit.json` (post-freeze physical-overlap audit).
- `.local-artifacts/issue-106-pixel-benchmarks-v1/summary.json .split_overlap`.

**Overlap audit definitions.** These come from `leakage-audit.json` and are labelled descriptive (post-freeze review).
- **Physical parameter set:** `initial_position`, `initial_rotation`, `scale`, `mass`, `dynamic_friction`, `static_friction`, `bounciness`, `push_force`, `push_position`, `push_time`, `model_names`. This is the full parameter set minus the appearance-only `color`.
- **Near-identical:** same native frame count, identical per-frame target-contact labels, and maximum absolute position difference ≤ 0.001 world units. This threshold was set after the review.

**Result.** 20 of the 150 testing trials share a physical parameter set with a fitting-split trial: 15 with dynamics, 17 with readout, 12 with both. Four testing trials have a near-identical trajectory.

**Name overlap.** The 996 readout trials have the same file names as 996 dynamics trials. They share no full parameter set (`shared_parameter_sets = 0`), and 45 dynamics–readout pairs share a physical parameter set.

`physion_membership.csv`:

| Column | Meaning |
|---|---|
| `release_split` | `dynamics`, `readout` or `testing`. |
| `trial` | Trial name (file stem). |
| `release_path` | Path inside the extracted Physion release, as recorded with the local data prefix removed. |
| `role` | Use in this study: dynamics = encoder, predictor and policy fitting; readout = object-contact-prediction readout fit; testing = rollout error and OCP test. |
| `included` | 1 for every row. |
| `shares_physical_parameters_with_dynamics`, `shares_physical_parameters_with_readout` | Testing rows only: 1 if the trial shares its physical parameter set with at least one trial of that split. |
| `near_identical_trajectory_in_fit_split` | Testing rows only: 1 if the audit lists the trial as near-identical to a fitting-split trial. |

`physion_overlap_pairs.csv` has one row per testing trial and fitting-split partner sharing a physical parameter set (136 pairs; 95 with dynamics, 41 with readout):

| Column | Meaning (unit) |
|---|---|
| `testing_trial` | Testing trial name. |
| `fit_split` | `dynamics` or `readout`. |
| `fit_trial` | Partner trial name. |
| `fit_release_path` | Partner path inside the extracted release. |
| `testing_native_frames`, `fit_native_frames` | Native frame counts. |
| `max_abs_position`, `max_abs_velocity` | Maximum absolute position and velocity difference over the common frames, as recorded in the audit (position in world units). |
| `contact_labels_identical` | 1 if the per-frame target-contact labels are identical. |
| `near_identical` | 1 if the pair meets the near-identical rule above. |

`physion_split_overlap_counts.csv` has one row per split pair, giving the counts of shared trial names, shared layouts (`initial_position`, `initial_rotation`, `scale`), shared full parameter sets (including colour) and shared physical parameter sets.

### `novphy_sealed_levels.csv` — NovPhy sealed split (#109 roster as captured by the #104 campaign)

- **Population.** 1,000 levels in three disjoint roles: 480 predictor-fit, 120 policy-fit and 400 evaluation. Each role is split equally over the two generator families `type010103` and `type010105` (240 / 60 / 200 per family). No level appears in more than one role.
- **Sources.** The cohort plan that defines the roster is not included. The level lists below are taken from the records that enumerate them:

| Role | Source of the level list | Inventory coverage given |
|---|---|---|
| `predictor_fit` | `.local-artifacts/issue-112-v3-retrain-v1/fit-verdicts.json` (fit-stream rows of the campaign verdict file, 13,919 complete branches) | `fit_grid_branches` and `fit_angle_branches`: complete captured candidates per inventory. The fit split has grid (16) and angle (13) candidates only. One level, `issue-109-sc-fit-type010105-049`, has 15 grid branches: the campaign's single fit-stream typed failure (13,920 scheduled; see `docs/issue-104-power-cohort.md`). |
| `policy_fit` | `.local-artifacts/issue-113-macro-predicates-v1/summaries-manifest-run.json .summaries.policy` (3,480 captured branches) | `policy_branches`: captured candidates per level (29 on every level). The per-inventory split of these candidates is not recorded in this source. |
| `evaluation` | `.local-artifacts/issue-102-closed-loop-v1/plan.json .levels.N1` (400 levels) | `eval_<inventory>_mixed`: 1 if the level is a mixed cell (at least one success and one failure) of that inventory. These flags come from the cell keys of the six Gate-B records `.local-artifacts/issue-108-gate-b-v1/records/*.json`, which are identical across the six. `eval_grid_any_recorded_pass`: `.local-artifacts/issue-102-closed-loop-v1/plan.json .levels.N1-mixed`. |

- **Non-grid coverage.** Grid (16 candidates) was captured on all 400 evaluation levels. Offset, angle and power were captured on 120 of them (`.local-artifacts/issue-108-gate-b-v1/plan.json .evaluation.accounting.levels_per_inventory`). The identities of those 120 levels are not listed in the packaged sources, so the file gives mixed-cell flags instead. Mixed offset/angle/power cells occur on 115 distinct levels, all within the 120.

| Column | Meaning |
|---|---|
| `level` | Level identity (`issue-109-sc-<split>-<family>-<index>`). |
| `role` | `predictor_fit`, `policy_fit` or `evaluation`. |
| `family` | Generator family token of the identity. For predictor-fit levels it is checked against the recorded `family` field. |
| `fit_grid_branches`, `fit_angle_branches` | Predictor-fit levels only (see above). |
| `policy_branches` | Policy-fit levels only (see above). |
| `eval_grid_mixed`, `eval_offset_mixed`, `eval_angle_mixed`, `eval_power_mixed` | Evaluation levels only. Grid mixed levels: 311; offset 76; angle 78; power 47. |
| `eval_grid_any_recorded_pass` | Evaluation levels only: 1 if at least one recorded grid candidate passed (the live table's `N1-mixed` set; 311 levels, identical to the grid mixed set). |
| `source` | Relative source path(s). |

## Live retry-from-start evaluation (#102): local aggregates

**Protocol** (`.local-artifacts/issue-102-closed-loop-v1/plan.json .semantics`):
- An episode is up to 3 shots on one single-bird level, resetting it to its start state before each shot.
- Before each shot the planner ranks the frozen 16-launch grid, excludes candidates it already executed and failed in the episode, and executes the best remaining one live.
- Pass = live engine pig removal.
- An episode stops at the first pass, after 3 shots, or at a typed engine failure, which counts as not passed.
- Each distinct (level, candidate) chosen by any arm was executed once, and its verdict is shared by every episode that chose it.

**Sets.**
- `N1`: the 400 evaluation levels (primary).
- `N1-mixed`: the 311 N1 levels with at least one recorded passing grid candidate.
- `N2-appearance`: 40 novelty levels, reported separately.

**Cells and denominators.**
- Statistics are computed over (level, seed) cells.
- Model arms and `random` have one episode per cell: 1,200 on N1.
- Seed-free arms (`prior`, `oracle`) have one episode per level (400 on N1), repeated over the three seeds of each level (1,200 cells).
- `oracle` executes only recorded passing candidates and takes no shot when a level has none. It is not deployable.

### `live_issue102_closed_loop_table.csv`

Byte-identical copy of `.local-artifacts/issue-102-closed-loop-v1/closed-loop-table.csv`; column names are unchanged. Intervals are level-clustered percentile bootstraps (10,000 draws, PCG64 seed 7201), labelled descriptive.

| Column | Meaning (unit) |
|---|---|
| `set`, `arm`, `group` | Set; arm identifier; arm description. Arm sources and checkpoints are in `plan.json .arms`. |
| `episodes` | Episodes of the arm in the set. |
| `pass1`, `pass2`, `pass3` (+ `_low`, `_high`) | Fraction of cells passed within 1, 2, 3 shots, with 95% interval. |
| `shots_to_pass` (+ `_low`, `_high`) | Mean shot index of the pass, over passed cells only. |
| `native_clear3` | Fraction of cells with an engine level-clear terminal within 3 shots (descriptive). |
| `recorded_pass3` | Pass@3 the same planner would get from the recorded campaign verdicts instead of live execution (declared cross-check). |
| `mean_decision_seconds`, `mean_rollout_seconds` | Mean host decision time and CUDA-synchronized rollout time per decision (ranking all 16 candidates; s). |
| `mean_linear_macs_per_decision`, `mean_linear_macs_per_candidate` | Mean linear multiply-accumulates per decision and per candidate rollout. |
| `minus_prior`, `minus_random` (+ `_low`, `_high`, `_token`) | Paired pass@3 difference against the no-model prior and against random, with interval and the frozen δ = 0.02 token. Every token is `readiness_or_precision_insufficient` because guard G4 failed: the live decision frame of 5,106 complete executions was checked against the anchor tolerance of 1e-3, and the maximum difference was 0.0032 (`compute.json .guards.G4_decision_frame`). |

`n/a` marks an undefined value: the `prior` row against itself, and the `random` row against itself. Those rows have an empty token.

### `live_issue102_outcome_counts.csv`

Transcribed from `.local-artifacts/issue-102-closed-loop-v1/compute.json .tables.<set>.arms.<arm>`.

| Column | Meaning |
|---|---|
| `set`, `arm`, `group` | As above. |
| `levels`, `cells` | Levels and (level, seed) cells in the set. |
| `episodes` | Executed episodes of the arm (cells for seeded arms; levels for `prior` and `oracle`). |
| `outcome_passed`, `outcome_failed`, `outcome_typed_failure`, `outcome_no_candidate`, `outcome_incomplete` | Episode end states. They sum to `episodes`. `no_candidate` occurs only for `oracle`, on levels without a recorded passing candidate. |
| `passed_at_shot1`, `passed_at_shot2`, `passed_at_shot3` | Passed episodes by the shot that passed. |
| `decisions_shot1`, `decisions_shot2`, `decisions_shot3` | Decisions (rankings) made at each shot index. |
| `shots_executed_mean` | Level-clustered mean number of shots executed per cell. |
| `replay_differs_cells` | Cells whose recorded-verdict replay outcome differs from the live outcome. |

### `live_issue102_typed_failures.csv`

Every live execution that ended in a typed engine failure: 17 in total, all on N1 (16 `supervisor_interrupted`, 1 `other`). Source: `.local-artifacts/issue-102-closed-loop-v1/compute.json .cross_check.typed_failures`. Columns: `execution` (`<level>-<candidate suffix>`), `level`, `candidate_suffix` (campaign candidate suffix, `cNN`), `set`, `failure_class`.

## Ranking and compute frontier (#103): local aggregate

### `frontier_issue103_table.csv`

Byte-identical copy of `.local-artifacts/issue-103-engine-truth-frontier-v1/frontier-table.csv`; column names are unchanged.

**Sets.**
- `primary`: the Gate-B mixed cells of the 400-level evaluation split. Cells are grid 933, offset 228, angle 234 and power 141 (mixed levels × 3 seeds); 71 policies.
- `secondary`: the #96 development universe described above. Cells are grid 42, offset 27, angle 21 and power 24; 69 policies.

**Further sources and definitions.** Policy definitions are in `.local-artifacts/issue-103-engine-truth-frontier-v1/plan.json .policies` (id, arm, group, source ticket) and `.groups`. The no-model prior row has source `gate_b rule`. Statistics are member- (level-) clustered percentile bootstraps (10,000 draws, PCG64 seed 7201), labelled descriptive.

| Column | Meaning (unit) |
|---|---|
| `set`, `inventory`, `policy`, `group`, `source` | Set; inventory; policy id; policy group (`fixed`, `fixed_continuous` = C*, `horizon_adaptive` = C−J, `J`, `J_cont`, `J_dec`, `J_pg`, `J_restricted`, `factorized_open_loop`, `hierarchical`, `tied_axes`, `external_78`, `prior`); ticket whose retained records were scored. |
| `auc`, `auc_low`, `auc_high` | Tie-free ordering AUC (level-mean of cell AUCs) with 95% interval. |
| `top3`, `top3_low`, `top3_high` | Offline top-3 hit rate: a cell is a hit if a success is among the three lowest-cost admissible candidates (ties by ordinal). With 95% interval. |
| `chance_top3` | Mean over cells of 1 − C(n−k, 3)/C(n, 3), where n is admissible candidates and k successes. |
| `macs_per_candidate`, `transitions_per_candidate`, `policy_calls_per_candidate` | Linear MACs, predictor transitions and policy calls per candidate rollout to step 225, pooled over seeds and cells. |
| `wall_seconds_per_state` | Wall seconds per state (all candidates rolled out together). Primary set: the frozen seed-20260908 timing pass, which is the confirmatory wall axis; the post-freeze three-seed re-timing is reported only under `compute.json .posthoc_wall`. Secondary set: this scoring, all seeds. 0 for the prior (no model work). |
| `recorded_seconds_per_state` | GPU seconds per state recorded in the scored records. |
| `cells`, `all_tied_cells`, `tied_pairs`, `pairs` | Cells; cells with all admissible costs tied; tied and total (success, failure) pairs. |
| `nonfinite_slots`, `typed_failures` | Non-finite predicted costs; cells without a usable record. |
| `on_auc_macs`, `on_auc_wall`, `on_top3_macs`, `on_top3_wall` | 1 if the policy is on the point Pareto set for that (y, x) pair. |
| `share_auc_macs`, `share_auc_wall`, `share_top3_macs`, `share_top3_wall` | Bootstrap frontier share: the fraction of the 10,000 level-resampling draws in which the policy is on the Pareto set (x held fixed). |
| `verdict_auc_macs` | AUC × MACs verdict text: on the frontier with its share, or dominated by Q with the paired Q − P difference, interval and δ token. |

## `checks.csv` — comparisons

One row per compared quantity. Columns:

- `file`: the packaged file the check uses.
- `quantity`
- `population`
- `published_source`: relative path and JSON key.
- `published_value`
- `recomputed_value`: computed from the packaged rows only.
- `match`: 1 if equal; floats must agree to within 1e-12, and lists or strings must be identical.

**821 of 821 match.**

| Packaged file | Checks | What is compared |
|---|---|---|
| `dev_issue96_cells.csv` | 574 | For each inventory and each of the 18 arms: AUC mean, both interval bounds (bootstrap re-run with the published settings), cell and level counts. All 9 published contrasts per inventory: mean, interval bounds, cells, undefined cells. J and F* top-1 hits, top-1 cells, chance rate. Mixed-cell and mixed-level counts. A cell-by-cell comparison against the published per-cell rows (0 mismatches over 114 cells × 18 arms). Scheduled-cell (177) and record (3,186) counts. |
| `dev_issue96_verdicts.csv` | 8 | Successful verdict slots on mixed levels, and levels, per inventory. |
| `clevrer_membership.csv` | 4 | Included and excluded counts per split against the plan. |
| `physion_membership.csv` | 17 | Included trials per split against the plan and the audit. Shared trial names per split pair. Testing trials sharing physical parameters with dynamics, readout, both, either: against the audit and against `summary.json`. |
| `physion_overlap_pairs.csv` | 3 | Near-identical testing trials, closest pair, set of shared testing trials. |
| `novphy_sealed_levels.csv` | 17 | Levels per role and per family against the campaign plan (`.local-artifacts/issue-104-power-cohort-v1/plan.json`). Complete fit branches (13,919, `.local-artifacts/issue-120-cost-alignment-v1/compute.json .fit.branches`). Policy branches (3,480) and policy levels (120, `.local-artifacts/issue-101-decision-aware-policy-v1/compute.json .pool`). Mixed levels per inventory (311/76/78/47, `.local-artifacts/issue-108-gate-b-v1/plan.json`). N1-mixed count (311). |
| `live_issue102_closed_loop_table.csv` | 2 | Row count. Every one of the 1,344 non-key cells re-rendered from `compute.json` with the runner's formatting: 0 mismatches. |
| `live_issue102_outcome_counts.csv` | 192 | Per set and arm: outcomes sum to episodes; passed-at-shot sums to passed; pass@1 and pass@3 recomputed as passed / episodes equal the published table to its 4 printed decimals. |
| `live_issue102_typed_failures.csv` | 2 | Typed-failure counts per set against `compute.json .cross_check.by_set`. |
| `frontier_issue103_table.csv` | 2 | Row count. Every row re-rendered from `compute.json` (keyed by set, inventory, policy): 0 differing rows. |

In addition, `build_supplement.py` copies the two published tables byte for byte. It also confirms that the Gate-B records agree on their mixed cells and that the three NovPhy roles are disjoint.
