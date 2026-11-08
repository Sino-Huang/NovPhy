# Author handoff — upload prerequisites for the ICLR 2027 discussion-phase revision

Not part of the PDF or the supplement. Items the authors must act on before uploading the revision (deadline Nov 18 2026 AoE).

## 1. Records promised in the rebuttal but not in the supplement

The rebuttal promised per-level outcome records, per-episode live records for every arm, level-by-candidate success/failure records, and exact split membership. The supplement (`manuscript/iclr2026_supplement/`) contains the submitted-era development cells/verdicts (#96), CLEVRER/Physion membership, the 1,000-level NovPhy role list, live per-arm tables, outcome counts and typed failures, and the full frontier table (821/821 aggregate checks pass). The Reproducibility Statement says the following families are not included. Either restore them and extend the supplement, or acknowledge the gap in the OpenReview reply; do not imply the promise is fulfilled.

The SupplementPackager's restore instructions (verbatim):

```
The rebuttal promised per-episode and per-level outcome records. Four of the promised families are **not on this machine**. They are absent from the local `.local-artifacts/` tree and from every branch of the repository's git history. Each was produced on the experiment server. The manifests that list them are local, and each manifest entry carries a SHA-256 hash, so restored files can be verified.

| Missing record family | Manifest that lists it (relative path) | Files listed | What packaging would need |
|---|---|---|---|
| Live retry-from-start episodes (#102), one row per (arm, level, seed): 16 model/reference arms + random + oracle, 400 evaluation levels | `.local-artifacts/issue-102-closed-loop-v1/manifest.json` | 15,510 in total. Decision records `records/decisions/shot{1,2,3}/`: 132. Live verdicts `records/verdicts/`: 5,123. Engine results `engine/results/`: 5,123. Engine receipts `engine/receipts/`: 5,123. Engine run logs: 4. Anchor stores: 2. Post-hoc decision frames: 1. Gallery manifest: 1. The evaluation verdict file below: 1. | The decision records and live verdicts. |
| Level-by-candidate engine verdicts for the evaluation split (11,680 launches) | `.local-artifacts/issue-103-engine-truth-frontier-v1/manifest.json` (role `verdicts`). The same file is also listed in `.local-artifacts/issue-102-closed-loop-v1/manifest.json` (role `recorded_104_rows`) and is bound by SHA-256 in `.local-artifacts/issue-102-closed-loop-v1/plan.json .inputs` (`issue_104_branches`). | 1: `.local-artifacts/issue-104-power-cohort-campaign-v1/branches.json`. The evaluation roster and candidate inventories are in `.local-artifacts/issue-109-sealed-cohort-v1/plan.json`, also absent and bound by SHA-256 in the same `.inputs`. | Both files. |
| Sealed-split ranking cells, one row per (arm, inventory, level, seed): the main ranking table and Appendix D | `.local-artifacts/issue-103-engine-truth-frontier-v1/manifest.json`. `.local-artifacts/issue-101-decision-aware-policy-v1/records-manifest.json` also lists the #101 records. | From the #103 manifest, 715 files in total: #103 `records/` 414 (204 secondary, 70 timing, 140 post-hoc timing); #114 `records/` 126; #101 `records/` 78; #105 `records/` 84; anchor and verdict files. The #101 records manifest lists 81 #101 record files. The 6 Gate-B records `.local-artifacts/issue-108-gate-b-v1/records/{unit,c_star}--seed*.json` **are** local, but they hold predicted costs only. They give no cell AUC without the two verdict/roster files above. | The record files plus the verdict and roster files above. |
| Submitted-checkpoint rerun cells on the evaluation split (#104) | `.local-artifacts/issue-104-heldout-contrasts-v1/records-manifest.json` | 54 (`records/<arm>--seed<seed>.json`, 18 arms × 3 seeds) | The record files plus the verdict and roster files above. |

**Why there is no automatic packaging.** `build_supplement.py` does not package these families automatically when the files are restored. Their schemas cannot be confirmed locally: no file of any of these record types is present to test against. Writing a reader from the runners' source alone would mean guessing the stored layout.

**TODO (authors, before upload; not for the PDF).**

1. Copy the files listed above from the experiment server to their manifest paths.
2. Verify each restored file against its manifest SHA-256.
3. Extend `build_supplement.py` with readers for those record types.
4. Re-derive the published aggregates as `checks.csv` does for #96:
   - per-arm pass@1/pass@3 against `live_issue102_closed_loop_table.csv`;
   - per-arm pooled cell AUC and top-3 hit rate against `frontier_issue103_table.csv`;
   - #104 contrasts against `.local-artifacts/issue-104-heldout-contrasts-v1/comparisons.csv`.

Until then, the per-unit records the rebuttal promised for these families are not part of this supplement.
```

If the records are restored and packaged, change the second sentence of the Reproducibility Statement (W21 in `manuscript/paper-contract.md`) to list them.

## 2. Promised references not computed

- Persistence and constant-velocity reference rows on the pixel CLEVRER/Physion protocol (rebuttal: "scoring-only rows on stored observations") were not computed (the revision ran no new scoring). The PDF says: "Persistence and constant-velocity references were not evaluated on this protocol; no comparison with them is claimed." Either run the scoring-only computation on the stored #106 observations and add the rows, or explain the omission in the OpenReview reply.
- A physically group-disjoint Physion evaluation was not run (the rebuttal already said so); the 20/150 physical-overlap qualification is attached to the result.

## 3. AI-use disclosure

The AI-use section now reads (decision m1 R3-2): "Generative AI tools were used in preparing both the submitted version and this revision of the manuscript: to draft and edit text, create and modify figures, check text against experiment records, and identify and format references; during this revision they also assisted with scientific framing and presentation decisions. The authors are responsible for the manuscript's claims, citations, figures, and disclosures." Before upload, add any research-stage uses of generative AI (experiment design, implementation, analysis) that this manuscript-preparation disclosure does not cover; ICLR 2027 requires the disclosure in the paper and in the submission form.

## 4. Rebuttal text that the revision changed

The posted rebuttal says "Figures 1 and 3 will be separated into readable interface and result panels". In the revision the submitted Figure 2 (description levels) moved to Appendix A, so the figure numbers shifted: new Figure 1 = interface, Figure 2 = model overview, Figure 3 = AUC–MAC frontier. Mention this in the OpenReview revision note. The revision also reports one finding not in the rebuttal: on the sealed NovPhy grid, the primary model's Δ=1 requests have the lowest rollout error (the rebuttal's C2 cited only the pixel study).

## 5. Venue checks not performed

Page fitting was done (main text ends on p.10). Not done: anonymity audit of the supplement code (`build_supplement.py` reads `NovPhy/.local-artifacts/...` relative paths), venue template version (the sources still use the ICLR 2026 style files and header "Under review as a conference paper at ICLR 2026"; the ICLR 2027 style files are at https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).

## 6. Final review status (2026-11-08)

Full review r4 + targeted re-check r5 (`manuscript/review-log.md`): absolute readiness **WEAK REJECT** at the ICLR 2027 bar; relative progress vs the submitted version **improved**; page limit met (main text ends p.10). Open after the directed revision:
- I-F01 (CRITICAL, author action): promised per-level/per-episode records not supplied (§1 above).
- I-F02 (CRITICAL, WONTFIX here): persistence/constant-velocity rows need new scoring (§2 above).
- I-P03 (MAJOR): value is narrow — one decision benchmark, within distribution, predictors trained for 9,000 updates that the records show are undertrained per request. Text now frames C3 as a budget-conditional counter-observation; strengthening it needs new experiments (a converged/longer-trained predictor rerun of U−C*, C*−J, J−J_cont and the readout control; a second decision benchmark).
- I-P09 (MINOR): ICLR 2026 template (see §5 above); re-check pagination after switching.
- Venue risk: the revised central claims reverse the submitted ones; ICLR 2027 lets ACs ignore revisions "significantly different from the original paper". The OpenReview revision note should list the withdrawn claims and point to Appendix F.
