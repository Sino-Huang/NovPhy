# Manuscript Content Brief

## CURRENT STATUS (2026-09-23, round-4 doc sync; binding for a fresh session)

**Thesis = the measured ranking failure, with AUC as the headline** (see `writing_outline_boundary_paper.md` v2.4 §0–§1; canonical numbers are in `CONTEXT.md` "Current state"; review dispositions are in `iclr2026/review-log.md`). Title: *Solvable but Mis-Ranked: An Engine-Truth Audit of Frozen World-Model Action Selection in NovPhy*. r4 gate (Main): 39 pp total, body §1–§8 ends p. 9 l.442 (§5 compression moved §8 to start on p. 8 l.423), Repro/Ethics p. 9 (uncounted), references p. 10, figures unchanged. Positioning (r4, I25): PHYRE certifies solvability and scores multi-attempt task success by AUCCESS (up to 100 attempts); our delta is a measured per-state ceiling on a frozen 13-candidate inventory plus a per-candidate engine-truth AUC in a single-shot cascade regime, not a new success-metric family.

**Post-WP5 update (WriterGamma, same day; overrides the table below where they differ).** Main text is now targeted at the verified 9-page ICLR 2026 limit, and the compile gate confirmed a 9-page body (references start p.11; Reproducibility/Ethics uncounted). The following moved to the appendix: the registry (now `tab:app:registry`, merged with the provenance table; `tab:syn:registry` deleted), the frontier (`tab:app:frontier`), all §5 subsections and floats (`sec:app:anat`, `tab/fig:exp:*` renamed to `tab/fig:app:*`), the related-work subsections and matrix (`tab:app:matrix`), the estimand figure and table (`fig/tab:app:estimands`), the §4 element list, and §7(b)/(c). The main text keeps compressed summaries with pointers. Also new: `tab:app:audit` (number→artifact audit table), a completed reproducibility statement, and a rendered Ethics statement. The empty checklist and Acknowledgments headings are removed. All tex-side items from the doc sync are closed (item 1 resolved by the citation pass, item 7 by the ceiling-map pass). Still open: the LLM-usage disclosure (author decision) and round-2 implementation of review-log.md.

Per-section status after round 3 (r2 review 4/10; r3 authoring tally 12 RESOLVED, 3 PARTIAL, 0 REJECTED):

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

## OPEN TEX ITEMS (r5 authoring 2026-09-24, #93 + r4 items; supersedes round 4)

Round-3 Batch A+B closed I3, I9, I11, I13, I26–I29 and I31–I34; the r3 follow-up closed I30 and I35–I39, and r3-followup authoring closed I40; round 4 closed I10 and I25 (r4 reviewer: I10 verified, I25 three residues, I40 residue, new I41/I42). r5 authoring 2026-09-24 fixed I25(a)(b)(c), I40, I41, I42 and landed the #93 second-parameterization appendix (C22 `not_supported_by_this_experiment`, C23 `readiness_or_precision_insufficient`, reading-matrix row 5: appendix H.2 plus body pointers at abstract.tex:30, introduction.tex:90, discussion.tex:58, method.tex:49), all pending r5 reviewer verification (tex locations in `iclr2026/review-log.md`). The following remain open:

1. **I16 (author action, still open):** anonymized artifact link (Fig. 1 gallery pointer now lives only in the appendix repro/audit tables; the reproducibility statement still cites commit `2b4d897` of a non-public repository).
2. **I11:** CLOSED 2026-09-23 (Main): the §5 construct-validity paragraph now reads "(proxy audit, status no_defensible_binding)" and "(engine-event channel)" (experiments.tex:34).
3. **I10:** CLOSED r4 2026-09-23 (pending reviewer verification): §5 is two paragraphs (experiments.tex:34, 43); the summary paragraph is deleted.
4. **I30:** CLOSED 2026-09-23 (r3 follow-up): owner approved; issue-91-inverted-ranker-v1 executed; inverted top-1 15/612 and 21/612 vs chance 0.0407, intervals cover zero, printed in abstract/§1/§6/§8, App. H.1, tab:app:audit, tab:app:repro. I35–I39 CLOSED the same pass.
5. **I8 scope note:** the pretrained-world-model experiment is declined (no new experiments). Disclosed limitation, not a tex task.
6. **I25:** r5-authoring 2026-09-24 (pending reviewer verification): (a) §8 denominator sentence scoped to "World-in-World-style reporting" (discussion.tex:77); (b) abstract delta = per-candidate engine-truth AUC under a per-state ceiling (abstract.tex:31); (c) PHYRE §4.4 OPTIMAL oracle-ranking agent cited as nearest precedent (appendix.tex:424). The Axis 1 value limit stays (not a tex item).
7. **I40:** r5-authoring: outcome horizon renamed "carrier frames" at every ratio site and the agent-frame = carrier-frame identity restated (method.tex:44; introduction.tex:85; synthesis.tex:92; discussion.tex:52; appendix.tex:581, :767).
8. **I41:** r5-authoring: App. E lead "that Section 5 demotes" (appendix.tex:149); audit group header "Section 5 disclosure" (appendix.tex:786).
9. **I42:** r5-authoring: abstract PHYRE sentence rewritten with "Whereas", parentheticals separated, one 46-word sentence → 20 + 13 words plus citations (abstract.tex:31).
10. **#93:** C22/C23 in `tab:app:registry`, audit row, repro row, "ten runners" (discussion.tex:84); r5 review minted I43–I46 against it.
11. **I43 (MAJOR):** r5-followup-authoring 2026-09-24: body pointers now carry the verdict ("not supported on a drag grid, though top-1 stays at or below chance"; abstract.tex:30, introduction.tex:91). §7 prints C22 0.6178 not supported, C23 0.5512 undecided, and top-1 8/126 vs 0.0804 and 7/81 vs 0.1030 (discussion.tex:61). Pending r5-followup review.
12. **I44:** r5-followup-authoring: H.2 closing verdict (appendix.tex:637–639); §7 states both inventories are launch-angle sets at saturated speed in the engine. Pending review.
13. **I45:** r5-followup-authoring: App. A inventory convention scoped to the N1 closed-loop and proxy-era records, with H.2 exception pointers (appendix.tex:46, :53). Pending review.
14. **I46:** r5-followup-authoring: audit #93 row completed (appendix.tex:776). Pending review.
