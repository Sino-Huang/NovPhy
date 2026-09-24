# Manuscript Content Brief

## CURRENT STATUS (2026-09-24, round-6 storyline doc sync; binding for a fresh session)

**Thesis = selection, not ordering (Story Lock A-hybrid)** (see `writing_outline_boundary_paper.md` v2.6 §0–§2; canonical numbers with status tags in `CONTEXT.md` "Current state"; review dispositions in `iclr2026/review-log.md`). Title: *Solvable but Not Selected: Frozen World-Model Rankers Against a Measured Engine Ceiling in NovPhy*. Thesis sentence: under a per-state ceiling measured in the engine, the three frozen rankers' pooled top-1 is at or below inventory-matched chance in point estimate on three candidate inventories over the same 15 N1 source members and on a held-out split, while member-clustered ordering AUC runs from 0.4180 to 0.6178 with the inventory. Headline numbers: ceilings 7/14, 9/15, 14/15 (member unit; descriptive intervals); pooled top-1 0/108 vs 0.0780 (outcome pre-specified; chance reading post hoc), 7/81 vs 0.1030 and 8/126 vs 0.0804 (pre-declared descriptive), held-out 13/612 vs 0.0407 (pre-registered); adapted few-shot 29/612 reported separately; AUC_m 0.4180 (headline choice post hoc) / 0.5512 `readiness_or_precision_insufficient` / 0.6178 `not_supported_by_this_experiment` (frozen decision statistic). Fig. 1 = `fig_hero_selection_ordering` (ceiling / pooled top-1 minus chance / AUC per inventory); the old N1 band figure is `fig:app:bands`. r6 gate: 42 pp total, Conclusion ends p. 9, Repro/Ethics p. 9 (uncounted), references p. 10, 0 undefined, 0 [TODO].

**Post-WP5 update (WriterGamma, same day; overrides the table below where they differ).** Main text is now targeted at the verified 9-page ICLR 2026 limit, and the compile gate confirmed a 9-page body (references start p.11; Reproducibility/Ethics uncounted). The following moved to the appendix: the registry (now `tab:app:registry`, merged with the provenance table; `tab:syn:registry` deleted), the frontier (`tab:app:frontier`), all §5 subsections and floats (`sec:app:anat`, `tab/fig:exp:*` renamed to `tab/fig:app:*`), the related-work subsections and matrix (`tab:app:matrix`), the estimand figure and table (`fig/tab:app:estimands`), the §4 element list, and §7(b)/(c). The main text keeps compressed summaries with pointers. Also new: `tab:app:audit` (number→artifact audit table), a completed reproducibility statement, and a rendered Ethics statement. The empty checklist and Acknowledgments headings are removed. All tex-side items from the doc sync are closed (item 1 resolved by the citation pass, item 7 by the ceiling-map pass). Still open: the LLM-usage disclosure (author decision) and round-2 implementation of review-log.md.

Per-section status after round 3 (historical; r6 rewrote abstract, §1, §3–§4 estimand/status sentences, §6, §7–§8, related-work delta and App. fig:app:bands/H.2/Table 17 — see `changelog.md` r6-authoring and outline v2.6 §2; the rows below are superseded where they differ):

| # | Tex section (file) | Status | Notes |
|---|---|---|---|
| -- | Abstract (`_abstract`) | REVISED r3-followup (Thesis2) | N1 "at or below chance"; cross-split "both AUC point estimates sit below 0.5" with 216 of 612 cells per condition, 6 state clusters, held-out = frozen zero-shot, few-shot adapted; inverted ranking 15/612 and 21/612 vs chance 0.0407 (I30); lookahead scope on the verbatim mechanism sentence; significance names the delta (I25/I26/I29/I36). |
| 1 | Introduction (`_introduction`) | REVISED r3-followup (Thesis2) | Teaser caption lead "at or below chance on solvable N1 states" (I37); spine (c) point-estimate wording and full inverted-ranker result with paired intervals (I36/I30); lookahead "between 0.25% and 25.6% of the outcome horizon", DERIVED (I35); bullet 2 "below 0.5 in point estimate". |
| 2 | Related Work (`_related_work`) | REVISED r3 (Evidence) | World-in-World claim cites `zhang2026worldinworld`, PHYRE claims cite `bakhtin2019phyre` (I33); delta sentence (I25). |
| 3 | Setting and Formulation (`_method`) | REVISED r3-followup (Evidence2) | Request paragraph states the Δ² lookahead mapping (Δ applications × Δ agent frames of 50 native steps; 1/25/225) (I38); estimand paragraph two sentences (I10). |
| 4 | Pre-Registered Protocol (`_method`) | REVISED r2 (Evidence) | Pre-registration scoped; verbatim restatement post-hoc sentence (I3a). |
| 5 | Boundary Anatomy (`_experiments`) | REVISED r4 (Evidence) | Two paragraphs: construct-validity disclosure (experiments.tex:34) + shortened demotion notice pointing to row C2 of `tab:app:registry` (experiments.tex:43); summary paragraph cut (I10). Ticket tokens stay out (I11). |
| 6 | Ranking-Failure Measurement (`_synthesis`) | REVISED r3-followup (Thesis2) | Point-estimate cross-split wording (I36); inverted-ranker result with paired intervals, 17 clusters, descriptive (I30); lookahead "between 0.25% and 25.6%" (I35); hybrid-only sweep sentence; stage names. |
| 7–8 | Discussion / Conclusion (`_discussion`) | REVISED r3-followup (Thesis2) | §7 lookahead depth + 13/13 coverage + stability confound (I32); conclusion: point-estimate wording, compact inverted-ranker result, verbatim mechanism clause behind the 1–25-frame qualifier (I36/I30/I39). |
| -- | Reproducibility / Ethics (`_discussion`) | REVISED (WP5) | On p. 9, uncounted; LLM-usage disclosure is an author decision. |
| A– | Appendix (`_appendix`) | REVISED r3-followup (Evidence2 + Main) | App. H.1 + audit "Rollout depth" row "between 0.25% and 25.6%", DERIVED (I35); H.1 inverted-ranker sentence, new audit row "Inverted ranker (post hoc)", new repro row "#91 inverted-ranker probe" (I30); C19 "600-step" label (I28); earlier r3 items (I9/I13/I27/I28/I29/I32) unchanged. Hero figure: 1-member hatched mark (I34), panel (b) AUC rows (I3). |

---


---

_Historical record (pre-#91 boundary-anatomy era) and the closed tex-items list were removed 2026-09-23 per owner directive; see git history and `changelog.md`._

## OPEN TEX ITEMS (r6 authoring 2026-09-24; supersedes r5)

r6 closed the r5-era pointer items (I43–I46 RESOLVED r5-followup; their text moved in r6, pointers added in `iclr2026/review-log.md`) and restated the headline around pooled top-1 (I25 PARTIAL). Earlier closures (I3, I9–I11, I13, I26–I42) stand as logged in `review-log.md` and `changelog.md`. The following remain open:

1. **I16 (owner action, carried):** anonymized artifact link; the Reproducibility Statement still cites commits `2b4d897` and `b8aba59` of a non-public repository.
2. **I25 residual (Axis 1 value limit):** the engine-side second action coordinate is still untested; in the engine every inventory is a set of launch angles at saturated speed. §7 names a launch-power sweep inside the 18.49 px clamp and a second ranker family as the tests that could widen the scope (discussion.tex:68). Not a wording item.
3. **Cold-read carry-overs (r6, not fixed for budget/jargon):** N1/member/state/cell unit map; "12 systems"; a04 jargon; status-token density; chance 0.0780 vs 1/13; point-estimate-null attack (a stronger ranker beating chance; #93 intervals admit above-chance; angle bias in one architecture ≠ world-model rankers in general).
4. **I8 scope note:** pretrained-world-model experiment declined. Disclosed limitation, not a tex task.
