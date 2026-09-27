# Issue-99 v2: guard revision of the attribution and slot-encoder publications — findings

- identity `issue-99-guard-revision-v2`, plan v2 frozen 2026-09-27T05:43:17Z **after** both v1 publications
- validation command: `python -u -m scripts.run_decision_chain_guard_revision --validate`; 0 GPU and 0 engine seconds; every number below except the two revised guards is read from the sha256-bound v1 publications
- disclosure: v2 was frozen after both v1 publications, with every v1 outcome known (all AUCs, contrasts, the Gate A result and the counterfactual token readings written in the v1 record and the #99 comment). The v2 guard definitions come from the diagnoses of the two v1 guard failures and from the parse tolerance the v1 attribution smoke froze before any outcome; no AUC or contrast entered their definition. v2 changes no estimand, cost, threshold, cohort, record or decision rule, and v1 is retained byte-identical (sha256-bound below) under the #87 versioned-protocol practice.

## Revised guards

| guard | v1 definition | v2 definition | reason |
|---|---|---|---|
| attribution.G2 | every executed shot's position-0 agent frame is byte-identical to the member's sealed anchor | every member has at least one verdict-bearing shot whose position-0 agent frame is byte-identical to its sealed anchor, and (v1 G3, unchanged) every shot of the member has the same engine-projected position-0 carrier within 1e-06; together every shot starts from the sealed anchor's physical state | the guard protects the physical start state; v1 failed on 4/859 frames that differ by 196-207 rendered pixels while the engine state is identical (spread 0.0). Byte-mismatched frames and their parsed position-0 deltas stay reported descriptively |
| fix.G3 | R0 parsed-endpoint costs (stage ap) replicate the attribution records within 1e-3 (count cost) | R0 parsed presence of every vocabulary slot at positions 0 and 225, the pig displacement and the tie-free cost replicate the attribution evidence within 0.001 (PARSE_BATCH_TOLERANCE, frozen in the v1 attribution smoke before any outcome) | the guard protects 'R0 is the same parse'; the count cost multiplies pig presence by 1000, so v1 read a 1.34e-4 batch-size float difference as 0.134 |

| publication | guard | v1 pass | v2 pass | revised |
|---|---|---|---|---|
| attribution | G1 | True | True | False |
| attribution | G2 | False | True | True |
| attribution | G3 | True | True | False |
| attribution | G4 | True | True | False |
| attribution | caps | True | True | False |
| fix | G1 | True | True | False |
| fix | G2 | True | True | False |
| fix | G3 | False | True | True |
| fix | G4 | True | True | False |
| fix | caps | True | True | False |

- attribution G2 v2: 15/15 members have a byte-identical shot; engine spread max 0.0; byte-mismatched shots 4: angle/issue-77-n1-010/o07 parsed position-0 max |delta| 0.00e+00, pig presence delta +0.00e+00; offset/issue-77-n1-010/o03 parsed position-0 max |delta| 0.00e+00, pig presence delta +0.00e+00; offset/issue-77-n1-010/o08 parsed position-0 max |delta| 0.00e+00, pig presence delta +0.00e+00; grid/issue-77-n1-016/o06 parsed position-0 max |delta| 6.07e-04, pig presence delta -2.02e-05
- fix G3 v2: 240 held-out shots; max |delta| presence 6.65e-04, pig displacement 0.00e+00, tie-free 1.34e-04 (tolerance 0.001)

## Dispositions (v1 -> v2, unchanged v1 decision rules)

| question | v1 token | v2 token |
|---|---|---|
| Q1_attribution | readiness_or_precision_insufficient | **supported** |
| Q2_tie_free_cost | readiness_or_precision_insufficient | **supported** |
| Q3_target | readiness_or_precision_insufficient | **supported** |
| Q4_encoder_effect | readiness_or_precision_insufficient | **not_supported_by_this_experiment** |

- Q1: Gate A dominant stage perception; drops cost 0.2143 [0.0714, 0.3571], dynamics -0.0842 [-0.2685, 0.0962], perception 0.2925 [0.0922, 0.4891]
- Q2: stage-c tied-cell share on mixed grid cells under the tie-free cost, max over 14 requests 0.000 (target < 0.1)
- Q3: arm E requests meeting AUC >= 0.65 with lower > 0.55 on held-out grid: C-F-1 (1/12): C-F-1 0.7259 [0.5778, 0.8444]
- Q4: request-mean E - R0 on held-out grid 0.0025 [-0.0204, 0.0185] (margin 0.02)

## Reading limits (carried from v1, not rules)

- Q3 rests on 1 of 12 requests over 3 held-out member clusters (10 distinct bootstrap resamples); no multiplicity adjustment was declared
- under the count cost E:C-F-1 is tied in 1.000 of held-out grid cells: its ranking comes only from the tie-free cost's continuous residue
- the target is not evidence that the fix helps: E - current (request-mean, held-out grid) -0.1401 [-0.1667, -0.0963]; the current #77 checkpoints meet the target on 7 requests without any fix
- attribution extraction wall 3715.9 s (v1 attribution compute.json compute.extraction.wall_seconds is an unpopulated field (0.0); v2 reports the extraction wall from the v1 ledger (extraction_seconds_elapsed))

## Bound v1 artifacts

| artifact | sha256 |
|---|---|
| `.local-artifacts/issue-99-decision-chain-attribution-v1/plan.json` | `sha256:c8d5f21e1190104d8ebdfd9fed7c1b078674bd871cf5eb33b00bee741c656e01` |
| `.local-artifacts/issue-99-decision-chain-attribution-v1/smoke.json` | `sha256:dddfc3fc6369a90c95ae7ac9074fe50fa1374471a706bc37f07bdc88d9884be4` |
| `.local-artifacts/issue-99-decision-chain-attribution-v1/ledger.json` | `sha256:7ea664f646ff69713da0281e042dae140032b1aa87495178bed2235e9b3b7e67` |
| `.local-artifacts/issue-99-decision-chain-attribution-v1/compute.json` | `sha256:e016d2bbad543d030611bce4280a5fba4a382a2e6559886a793d25ef255bfc5d` |
| `.local-artifacts/issue-99-decision-chain-attribution-v1/summary.json` | `sha256:5cb7168a1424230950c975e9e4c480ab8494663be643ef33f42d9d54f153b2a3` |
| `.local-artifacts/issue-99-decision-chain-attribution-v1/findings.md` | `sha256:b0be11ad59bfcb3d1e0cd809e1f5a4e6ea10aa11b055a1052ba8b3e934de69b4` |
| `.local-artifacts/issue-99-decision-chain-attribution-v1/comparisons.csv` | `sha256:1a0dc276ae9f74f94a2a9f8c4a4dd2f67399e8001f8fc83c4170b5d8da59b899` |
| `.local-artifacts/issue-99-slot-encoder-fix-v1/plan.json` | `sha256:cce9fb1b3e2a134d0b1d41f85f57f106f592c8a7a59cb8ee1a980cf7aebc9098` |
| `.local-artifacts/issue-99-slot-encoder-fix-v1/smoke.json` | `sha256:6d56e2246c0252e07372711e8b6358d318abe1dcb1b20a41d8f328c8fd67ac6b` |
| `.local-artifacts/issue-99-slot-encoder-fix-v1/ledger.json` | `sha256:d1c78f784c59244c03a5c04e826ad8b5d4980925dff38966b2223b261bc93798` |
| `.local-artifacts/issue-99-slot-encoder-fix-v1/compute.json` | `sha256:5481d831be79efe684a45a509e4feac92a85d98a9e40e2ec056d50e8099ec0d1` |
| `.local-artifacts/issue-99-slot-encoder-fix-v1/summary.json` | `sha256:e49baecf86bcfc4f04e76c6e01ce430b0859d36dd8382f528b03f798466570bb` |
| `.local-artifacts/issue-99-slot-encoder-fix-v1/findings.md` | `sha256:d23e837d78259fa90ae3b24872b4635f30cba2bac63c44252eeeb7daf5482739` |
| `.local-artifacts/issue-99-slot-encoder-fix-v1/comparisons.csv` | `sha256:9653f7b571683a575fb0fbfbc480e346b77c69abdfaeac1da7e553ef2a288b3c` |
| `scripts/run_decision_chain_attribution.py` | `sha256:9e017a24b0e6b126ec68eb91586f454e05e1c259430d7057281977c349fcc43c` |
| `scripts/run_slot_encoder_fix.py` | `sha256:e256d0908c459b403826b07a39fa4bfa7d50692ede7db692b000d322aa422ddf` |
| `world_model/training/spatial_slot_parser.py` | `sha256:4a1fb427c258880007d7d7f1e98400f1ea94db758c2b4b0dd81bfc7f5bd289f6` |

## Claim boundary

decision-only re-scoring of retained engine-verdicted executions and the frozen #77 N1 checkpoints; stage d reads post-launch frames and is a diagnostic, not a decision-time ranker; development/exposed N1 members only (held-out = not fit by #77 predictors or controllers, still exposed through earlier tickets); no engine access, rendering or retraining; inventories and cohorts are never pooled; every interval is DESCRIPTIVE; decision-only ranking of retained engine-verdicted executions on the 4 held-out N1 members (never fit here or by #77; exposed through earlier tickets); a new slot encoder and predictors trained on fit lineages of the same two families; no engine access, rendering or capture; no closed-loop or competence claim; every interval is DESCRIPTIVE; v2 revises two guards only.
