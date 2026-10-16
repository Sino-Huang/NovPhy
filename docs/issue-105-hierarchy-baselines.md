# Issue #105: independent axes vs hierarchy

Parent: #108. The binding shared rules of #85 apply, and #105 adopts the #111 scoring contract. Gate B (#108), #101, #104, #109, #112, #114 and #78 records are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_hierarchy_baselines.py` |
| Hierarchical model | `world_model/training/hierarchical_dynamics.py` |
| Selector kinds | `world_model/training/request_selectors.py` (#114, 82a5878e; hash-bound, not edited here) |
| Artifacts | `.local-artifacts/issue-105-hierarchy-baselines-v1/` (`findings.md`, `summary.json`, `compute.json`, `plan.json`, `smoke.json`, `ledger/`, `records/`, `training/`, `models/`, `data/`) |
| Freeze | Plan frozen 2026-10-14T11:43:40Z (commit 0601a9e0), after a passed smoke and before any training, truth build or evaluation score. The module bytes are unchanged since the freeze. The runner changed after the publication for failed-guard reporting and the amendment-aware input check only (see "Post-freeze runner edits"). |
| Validation | `python -u -m scripts.run_hierarchy_baselines --validate` exits 0. `pytest tests/test_issue_105_hierarchy_baselines.py` passes (9 tests). |

Gate B passed, so this is the confirmatory reading on the #104 sealed #109 evaluation split. It uses the Gate-B cells (mixed only: grid 311, offset 76, angle 78, power 47 levels, × 3 seeds) and the Gate-B anchors. Grid is primary. Offset, angle, power and the rollout error are DESCRIPTIVE. Intervals are percentile bootstraps clustered by level: 10 000 draws, PCG64 7201, DESCRIPTIVE. The margin is δ = 0.02. Zero engine seconds.

## Result: `not_supported_by_this_experiment`

Independent axes do not add ranking value over tied axes on grid. Each of the three pre-declared contrasts has an upper bound below δ, and all seven guards pass.

| AUC difference (paired by level and seed) | grid | offset | angle | power |
| --- | --- | --- | --- | --- |
| **J − J_diag** | **−0.007 [−0.016, +0.003]** n.s. | −0.022 [−0.062, +0.014] n.s. | −0.025 [−0.058, +0.007] n.s. | +0.020 [−0.001, +0.046] r/p |
| **J − J_monotone** | **+0.003 [+0.001, +0.004]** n.s. | −0.002 [−0.006, +0.001] n.s. | −0.004 [−0.009, +0.000] n.s. | −0.002 [−0.004, +0.001] n.s. |
| **J − H\*** (H\* = H-5-micro) | **−0.001 [−0.019, +0.017]** n.s. | +0.057 [−0.020, +0.132] r/p | +0.104 [+0.047, +0.158] supported | +0.003 [−0.012, +0.018] n.s. |
| J − J_anti (control) | +0.006 [+0.002, +0.010] n.s. | +0.137 [+0.111, +0.163] supported | +0.102 [+0.080, +0.124] supported | +0.018 [+0.007, +0.029] r/p |
| J_diag − J_anti | +0.012 [+0.003, +0.022] r/p | +0.159 [+0.128, +0.192] supported | +0.127 [+0.099, +0.154] supported | −0.002 [−0.027, +0.018] n.s. |
| J_dec − J_dec_diag | +0.051 [+0.036, +0.067] supported | +0.071 [+0.050, +0.093] supported | +0.063 [+0.042, +0.084] supported | +0.045 [+0.005, +0.090] supported |
| J_dec − J_dec_monotone | +0.000 | 0.000 | 0.000 | 0.000 |
| J_dec − H\* | +0.040 [+0.030, +0.049] supported | −0.005 r/p | +0.027 r/p | −0.010 n.s. |
| J_dec − J_dec_anti | −0.016 [−0.020, −0.012] n.s. | −0.127 n.s. | −0.099 n.s. | −0.010 n.s. |

n.s. = `not_supported_by_this_experiment`; r/p = `readiness_or_precision_insufficient`.

- J is the #101 ICLR prediction-error policy (`J-pe-1e-04`). J_dec is the #101 decision-valued policy (`J-dec-1e-01`). Both run on the Gate-B LC hybrid predictor.
- H\* is the hierarchical level chosen by the t96 cross-fit rule. It is H-5-micro on every inventory.

Grid AUC per arm (tie-free cost; MACs per candidate rollout):

| Arm | grid AUC | MACs |
| --- | --- | --- |
| J | 0.706 [0.685, 0.727] | 184M |
| J_diag {(1,cont),(5,micro),(15,macro)} | 0.713 [0.692, 0.733] | 111M |
| J_monotone | 0.703 [0.682, 0.724] | 187M |
| J_anti {(1,macro),(5,micro),(15,cont)} | 0.700 [0.679, 0.721] | 221M |
| J_dec | 0.746 [0.728, 0.764] | 23M |
| J_dec_diag | 0.695 [0.679, 0.711] | 27M |
| J_dec_anti | 0.762 [0.743, 0.781] (= F-15-continuous) | 22M |
| Hierarchical H-1-continuous | 0.508 [0.495, 0.521] | 411M |
| Hierarchical H-5-micro (H\*) | 0.706 [0.688, 0.725] | 59M |
| Hierarchical H-15-macro | 0.416 [0.402, 0.430] | 8M |
| #78 TAWM-1 / -5 / -15 (#71 carrier) | 0.459 / 0.601 / 0.610 | 417M / 83M / 28M |
| #78 VLWM-1 / -5 / -15 (#71 carrier) | 0.448 / 0.552 / 0.642 | 417M / 83M / 28M |
| #78 TAWM-adaptive | 0.450 [0.430, 0.469] | 236M |
| F-15-continuous (F_hindsight) | 0.762 [0.743, 0.781] | 22M |
| F-15-macro (Gate-B unit) | 0.744 [0.727, 0.762] | 22M |
| C\* (LC continuous C-F-15) | 0.783 [0.768, 0.799] | 28M |
| No-model action prior | 0.720 [0.697, 0.744] | — |

`unsupported` #78 cells:
- VLWM-adaptive: #78 trained a horizon controller for TAWM only.
- Diagonal, monotone and anti-diagonal arms of TAWM/VLWM: the ports are continuous-only and have no α axis.
- Wrong-anchor control of TAWM/VLWM: their carrier is the #71 parser's, and the fit-split anchor pool exists only as E99 carriers.

Rollout error at t = 225 uses the #100 `engine_position_mse` on every mixed grid candidate's own branch (lower is better; 0 nonfinite endpoints):

| Arm | error |
| --- | --- |
| J | 0.0317 [0.0303, 0.0331] |
| J_diag | 0.0361 [0.0343, 0.0381] |
| J_monotone | 0.0314 [0.0300, 0.0328] |
| F-1-macro (lowest) | 0.0281 [0.0271, 0.0291] |
| F-5-micro / F-15-macro | 0.0375 / 0.0474 |
| H-5-micro | 0.1096 [0.1054, 0.1139] |
| H-15-macro | 186 (diverged, finite) |
| H-1-continuous | 4354 (diverged, finite) |
| TAWM-15 / VLWM-15 | 0.163 / 0.280 |
| TAWM-1 / VLWM-1 / TAWM-adaptive | 5312 / 12240 / 763 (diverged) |

Paired by level:

| Contrast | estimate |
| --- | --- |
| J − J_diag | −0.0044 [−0.0053, −0.0036] |
| J − J_monotone | +0.0003 [+0.0002, +0.0004] |
| J − H\* | −0.0779 [−0.0812, −0.0747] |

### Reading

- **Tying horizon to description costs J nothing on grid.**
  - J_diag ranks as well as J: it is +0.007 better, at 60 % of J's MACs.
  - J_monotone is J to within 0.003. J's schedules are already almost monotone: its modal request is (1, macro), and only 13.5 % of its steps are on the diagonal.
  - The anti-diagonal control ranks about like J on grid, but clearly worse on offset and angle. The diagonal also beats the anti-diagonal on offset and angle (supported). The pairing of short horizons with fine descriptions is the useful one, not the reverse.
- **The test is weak in one direction: J itself carries little request value here.**
  - J trails the best fixed request (F-15-continuous) by 0.057 on grid, and it trails the no-model action prior by 0.015.
  - A tied-axes restriction cannot lose value that the unrestricted policy does not have. "Not supported" therefore means no measurable value of independence for the ICLR J on this split. It does not show that tied axes are as good as a policy that exploits independence.
- **For J_dec, the diagonal restriction hurts. The anti-diagonal is numerically higher, but below δ.**
  - J_dec − J_dec_diag is +0.05 to +0.07 on every inventory (supported). J_dec_anti collapses to (15, continuous) at almost every step. On grid it is numerically higher than J_dec by 0.016 (J_dec − J_dec_anti = −0.0163 [−0.0203, −0.0122]). That is below δ, so no ≥ 0.02 advantage is established.
  - The gain comes from reaching the coarse-horizon continuous request (15, cont), which lies off the diagonal. A hierarchy with continuous description at the coarsest level would reach it too. This reading is DESCRIPTIVE, and the disposition rule uses J only.
- **The trained hierarchical model is usable only at its middle level.**
  - H-5-micro ranks like J on grid (J − H\* −0.001), loses on angle (+0.104 for J) and is anchor-dependent (right − wrong +0.136, supported).
  - Its rollout error is three times J's (0.110 vs 0.032).
  - The Δ = 1 and Δ = 15 levels diverge under 225-frame recursion. The diverged endpoints are finite (no typed failures), but they are far from the engine. H-15-macro's endpoint cost does not depend on the anchor (wrong-anchor stability 1.00).
  - The same LC recipe keeps the hybrid stable. Training every level only at its own request (3000 updates per level, no shared trunk) leaves the coarse and fine levels unstable in recursion.
- **The #78 ports, on their own #71 carrier and their #77 N1 training, do not transfer to this split.**
  - On grid, every #78 port is below 0.65.
  - The Δ = 1 variants and TAWM-adaptive diverge.

## Protocol

### Arms

- **Restricted J (Work 1).** These take the same logits as the #101 policies, take the argmax over a subset, and involve no training. They use #114's `request_selectors`:
  - J_diag = `diagonal_spec`;
  - J_anti = `anti_diagonal_spec`;
  - J_monotone = `monotone_spec`. This kind is path-dependent. After the row's previous request (Δp, αp), a request (Δ, α) is allowed iff sign(Δ − Δp)·sign(α − αp) ≥ 0, with α ordered cont < micro < macro. So α may only coarsen as Δ grows. The first request is free.
  - Scoring runs through `rs.rollout_endpoints(..., select=t96.select_pairs)`. The same restrictions are applied to J_dec.
- **Hierarchical predictor (Work 2).** Three residual levels on the 236-d E99 carrier, each answering one diagonal request: (1, cont), (5, micro), (15, macro). Off-diagonal requests raise an error.
  - Each level is Linear(241, 202), 3 FiLM blocks, LayerNorm and Linear(202, 236). Level 1 adds the hybrid's micro head and adapter; level 2 adds the macro head and adapter.
  - Coupling is top-down (H-JEPA / CW-VAE style): level k's FiLM code is Linear(236, 128) of the displacement that level k+1 predicts from the same state. The top level uses a learned constant code.
  - It has 1,840,322 parameters against 1,837,690 for the hybrid (+0.14 %). The width is the nearest-total rule of `capacity_contract`.
- **Training.** The #112 v3 LC recipe exactly as the Gate-B retrain (`rt.implementation()`):
  - #77 local loss, long-horizon supervision to 225 and the decision-contrastive ranking loss;
  - truncation 45 and skip-on-nonfinite;
  - AdamW 1e-4, batch 64, 9000 updates;
  - update k trains level k mod 3;
  - data: the retrain's #109 fit-split payload (13 919 shots, 680 ranking cells), hash-checked;
  - seeds 20260908/09/10.

  All three seeds are usable: complete, 0 skipped updates, no #117 excursion.
- **#78 (Work 3).** The published TAWM and VLWM checkpoints, hash-bound, plus #78's TAWM horizon controller (TAWM-adaptive). They run from the #71 (F77) parse of the same decision frames as the E99 anchors, because they were trained on #71 carriers.
- **Controls.** The fixed requests and C\* (rows replicate #101 exactly, G5), the no-model action prior (Gate-B rule), the wrong-anchor control (480 fit anchors) for J_diag and each hierarchical level, and the anti-diagonal arm.

### Scoring

- The tie-free cost is `EndpointCosts.row` against the anchor. Cell AUC is `m99.view`.
- Rollout-error truth is the #99 `targets` presence and projected centres at observed position min(225, last frame) of each candidate's branch. The smoke checks it against `s99.shot_labels`; it was built after the freeze.
- Everything runs under `deterministic_scoring()`, in two processes on GPUs 2 and 3 that did not train, launched once (PIDs recorded in `logs/score-pids.txt`).
- **Disposition rule (frozen):**
  - `supported` iff J − J_diag, J − J_monotone and J − H\* are each supported;
  - `not_supported_by_this_experiment` iff at least one has upper bound < δ;
  - otherwise `readiness_or_precision_insufficient`;
  - a failed guard gives the stop token. This frozen wording covers the disposition and the J_dec reading only. Since the phase-A fix, a failed guard also adds a `guard_failure` block to `compute.json`, `summary.json` and `findings.md`. The block names the failed guards and states that no other token in the publication (the contrast tokens beside the disposition, and the inventory, rollout-error and wrong-anchor contrasts) is interpretable. The frozen rule itself is unchanged. All guards pass, so the block is absent and no published file changed.
- **Guards** (all pass):
  - G1 isolation;
  - G2 scoring policy;
  - G3 chronology;
  - G4 checkpoint hashes (#101 policies, hierarchical, #78);
  - G5 the fixed, C\*, J and J_dec rows and schedules equal #101's exactly;
  - G6 restriction integrity (0 out-of-set steps, 0 monotone violations);
  - G7 nonfinite share ≤ 0.10.
- **Validate** re-derives every table, byte-compares `compute.json`, `summary.json` and `findings.md`, and re-scores J_diag, J_monotone and the three hierarchical levels on all 311 grid cells for seed 20260908. Rows, engine errors and schedules match exactly.

### Smoke (pre-freeze)

The smoke checked the following:
- capacity matching and the interface (6 off-diagonal requests rejected);
- 12 full-cost training updates;
- `hd.fixed_rollout` equals `m99.rollout_endpoints`;
- `rs` J equals `rp` J;
- restricted and monotone schedules stay in their sets;
- the #78 checkpoints load and the F77 parse is deterministic;
- the engine truth equals `s99.shot_labels`.

A rehearsal of every post-freeze mode then ran on 24 fit levels with a 12-update model, in a deleted throwaway output (disclosed in the plan, 1092 GPU s).

## Decisions under delegation (posted on #105)

1. **J.** #101's `J-pe-1e-04` on the Gate-B LC predictor. The #77/#96 controllers expect F77 carriers. `J-dec-1e-01` is reported DESCRIPTIVE.
2. **Restricted kinds.** #114's shared module. Monotone uses the path-dependent reading.
3. **Hierarchical model.** The architecture and coupling above.
4. **Matched baseline.** The Gate-B #112 v3 LC hybrid replaces the ticket's #77 hybrid, because it is the predictor of every confirmatory #108 reading. It is matched in parameters, recipe and updates.
5. **H\*.** Selected by the cross-fit rule. J minus each level is also reported.
6. **Rollout error.** The #100 engine metric at t = 225 on the evaluation grid branches.
7. **#78.** Scored on the #71 carrier where the interface allows. The rest are `unsupported`, with reasons.
8. **Controls and guards.** As above.
9. **Compute cap.** 12 GPU-h and 24 h process wall on 2 GPUs. No scope cut.

## Compute and disclosures

- GPU and process wall time by phase:

  | Phase | Time |
  | --- | --- |
  | Smoke | 86 s |
  | Rehearsal | 1 092 s |
  | Hierarchical training | 5 724 GPU s (3 seeds) |
  | F77 anchors | 11 s |
  | Engine truth (16 CPU processes) | 1 594 s wall |
  | Scoring (96 records on 2 GPUs) | about 3 470 s, 30 min elapsed |

  Total: 10 386 GPU s and 11 982 process wall s (`compute.json`). Engine time is 0 s.
- The run paused for about an hour after training, on the orchestrator's one-ticket-at-a-time instruction. Scoring started only after #114 published.
- Hardware: RTX 5090 instead of the #85 RTX 3090.
- **Known before the freeze:** Gate B, every #101 result (J 0.706, J_dec 0.746, F_hindsight 0.762, J modal (1, macro)), and #96/#78/#100/#112. No restricted arm, hierarchical model or #78 port output on an evaluation capture existed.
- **The hierarchical model is one matched instance of a tied-axes model.** Its Δ = 1 and Δ = 15 levels diverge under recursion. A different hierarchy (shared trunk, other coupling, continuous description at the coarse level) could behave differently.
- **H\* and F\* are cross-inventory selections, not member-disjoint cross-fits.** As in #101, the same evaluation levels appear in several inventories. The selection is made once, before the bootstrap, so intervals that involve H\* or F\* are conditional on it.

## Post-freeze runner edits (disclosure)

Commit `28b3764c` changed `scripts/run_hierarchy_baselines.py` after the publication. The edits concern reporting and input validation only. No evidence changed, and every published file is byte-identical (`--validate` exits 0).

- **Failed-guard reporting.** `guard_failure()` adds an explicit statement whenever a guard fails: secondary tokens are then not interpretable. The frozen rule, which invalidates the disposition and the J_dec reading only, is kept.
- **Input check.** `load_plan` checks the pinned #101 runner through `scripts/post_freeze_amendments.py`. It accepts exactly the frozen bytes or the disclosed #101 amendment. The #103 plan pins this runner, and the same registry lists its amended bytes.
