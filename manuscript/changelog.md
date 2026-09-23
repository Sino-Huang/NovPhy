# Changelog

Single canonical change log for the manuscript. File-local conventions (per-file header comments in `iclr2026/*.tex`) remain, but anything that changes the story, the file set, or the doc set lands here. Reviews are logged separately in `iclr2026/review-log.md`.

## 2026-09-23 · r4: I25 (abstract, §1, §6, related work, appendix) + I10 (§5); docs sync
- Changed: I25 (MAJOR) - PHYRE AUCCESS named and characterized (multi-attempt, task-level, log-weighted area under the success curve, up to 100 attempts; arXiv:1908.05656 §3.2, §4.2; tier solvability §3.1/App. B) at abstract.tex:31, introduction.tex:93 (contribution 1; "releases simulation results for 100,000 actions" over-claim removed, PHYRE §4.2 fn.3 states release as future), introduction.tex:99 (Girdhar 2020 forward-predictor PHYRE ranking acknowledged as objective-mismatch precedent), synthesis.tex:112, related_work.tex:24–25, appendix.tex:419–420. Delta stated narrowly: measured per-state ceiling on a frozen 13-candidate inventory + per-candidate engine-truth AUC in a single-shot cascade regime, not a new success-metric family. Words: abstract −4, intro 0, synthesis 0, related work −2, appendix +1. Why: I25 (r3/r3-followup most damaging); facts local://phyrefacts.md
- Changed: I10 (MINOR) - §5 three paragraphs → two: construct-validity disclosure unchanged (experiments.tex:34), demotion notice shortened with pointer to row C2 of tab:app:registry (experiments.tex:43), summary paragraph duplicating §1 deleted (content in appendix/registry C1–C10); header comment experiments.tex:27. −7 source lines, about −300 words. Why: I10 residue per r3/r3-followup
- Changed: gate 39 pp, body ends p. 9 l.442 (§5 compression moved §8 to start on p. 8 l.423), Repro/Ethics p. 9, references p. 10; figures unchanged. review-log r4 tags under r3 I10/I25 + r4 authoring tally (additive); CONTEXT.md, content_brief.md, outline v2.4 synced. Why: r4 sync
- Deferred: I16 anonymized artifact link (owner action); r4 reviewer verification of I10/I25.
- Rejected: none.

## 2026-09-23 · r3-followup Main: I40 naming fix, nine-runners repro fix, gate
- Changed: I40 (r3-followup MINOR, naming drift) fixed - \S3 lookahead renamed to carrier frames (method.tex:43), the $k < \Delta$ clamp now reads "agent frames", and the cross-split AUC cluster unit unified to "state clusters" (abstract.tex:26; introduction.tex:52, :84; synthesis.tex:32, :87; discussion.tex:74; appendix.tex:712 audit row); stale "~0.3-7%" % comment at appendix.tex:567 marked superseded by the derived 0.25-25.6% range. Why: R3Reviewer follow-up minted I40; wording-only, page-neutral.
- Changed: Reproducibility Statement runner count "seven runners" -> "nine runners" with run_state_difficulty_auc and run_inverted_ranker_probe listed (discussion.tex:80). Why: factual count after the two post-hoc probes landed.
- Changed: `iclr2026/review-log.md` follow-up section gains the I40 [RESOLVED r3-followup-authoring ...] tag (additive). Gate: exit 0, 39 pp (appendix +1 page from the new audit/repro rows), body ends p. 9, Repro/Ethics p. 9, references p. 10. R3Reviewer verdict on the follow-up PDF: 6/10 BORDERLINE (r3 criterion met); open: I25, I10, I16, I40(now fixed).

## 2026-09-23 · r3-followup Docs: review-log r3 tags, CONTEXT, content_brief, research_evidence, outline v2.3
- Changed: `iclr2026/review-log.md` r3 entry gains [RESOLVED r3-followup …] tag lines under I30 and I35–I39, each citing tex file:line checked this pass; r3-followup authoring tally appended before "## r2" (6 RESOLVED, 0 REJECTED; open I10, I25, I16). Additive only (+8 lines). Why: r3 follow-up landed (I35–I39, I30 executed)
- Changed: CONTEXT.md current state (inverted-ranker digits, Δ² mapping, 0.25–25.6% range, point-estimate wording, gate 39 pp, pending list); content_brief.md CURRENT STATUS rows (abstract, §1, §3, §6, §7–8, App.) + OPEN TEX ITEMS (I30/I35–I39 closed, I25 added). Why: follow-up sync
- Changed: research_evidence.md rows for issue-91-state-difficulty-auc-v1 (missing) and issue-91-inverted-ranker-v1; WP1b row "601-frame" → "600-step" and derived lookahead range; outline pointer v2.2 → v2.3. Why: follow-up sync; I28 label consistency; I35
- Changed: `writing_outline_boundary_paper.md` → v2.3 (status, §0 thesis, round-3 wording bullet drops "systematically below chance", §2 rows abstract/§1/§3/§6/§8/App., §5 gates). Why: follow-up sync; old wording rule contradicted I36
- Deferred: I10 §5 surface; I25 AUCCESS precedent; I16 (author).
- Rejected: editing the r2-section I30 [PARTIAL r3] tag and the r3 tally line sitting before "## r1" (r2 section is closed to edits).

## 2026-09-23 · r3 follow-up (Main): inverted-ranker analysis + gate
- Changed: new `scripts/run_inverted_ranker_probe.py` + `.local-artifacts/issue-91-inverted-ranker-v1/` (plan.json, summary.json, findings.md, records/inverted_cells.csv); `--validate` exit 0, byte-identical; zero engine seconds; guards reproduce argmin 13/612 and 29/612. Why: I30 owner-approved
- Changed: binding digits: inverted (max-cost, ties to lower ordinal) top-1 zero-shot 15/612 = 0.0245, few-shot 21/612 = 0.0343, chance 0.0407; paired −0.0162 [−0.0392, +0.0026] / −0.0064 [−0.0270, +0.0118]; 17 state clusters; descriptive. Why: I30
- Changed: csv LF line terminator fixed in the probe so `--validate` byte-compare passes. Why: validation
- Changed: gate green: 39 pp (appendix +1: audit + repro rows), body ends p. 9, Repro/Ethics p. 9, refs p. 10; 0 Float too large, 0 undefined refs, 0 rendered [TODO]. Why: page-limit gate
- Deferred: none.
- Rejected: none.

## 2026-09-23 · r3 Main tail fixes
- Changed: experiments.tex:34 ticket tokens #82/#85 → "(proxy audit, …)" / "(engine-event channel)" (I11 closed). Why: Docs-verified residue
- Changed: appendix C19 "601-frame" → "600-step". Why: I28 label consistency
- Changed: review-log I11 tag flipped to RESOLVED; r3 authoring tally updated to 13 RESOLVED / 2 PARTIAL. Why: Docs-verified residues
- Deferred: none.
- Rejected: none.

## 2026-09-23 · r3 follow-up (Thesis2): abstract; §1 introduction (Fig. 1 caption, spine (c), contribution bullet 2); §6 synthesis; §8 conclusion
- Changed: §1 spine (c) and §6 lookahead sentence "about 0.3--7%" → "between 0.25% and 25.6% of the outcome horizon", % comment marks DERIVED (1/400.9, 25/97.8; lookahead findings.md lines 9, 14); digits agreed with Evidence2 (App. H.1). Why: I35
- Changed: every "systematically below chance" (abstract, §1 ×2, §6, §8) → descriptive "both AUC point estimates sit below 0.5" (bullet: "below 0.5 in point estimate"). Why: I36
- Changed: inverted-ranker result added (abstract/§8 compact; §1/§6 full with paired intervals, 17 clusters, descriptive; % summary.json per_condition). Why: I30 (owner-approved, executed)
- Changed: Fig. 1 caption lead "is below chance" → "is at or below chance". Why: I37
- Changed: §8 "the failure sits in" → verbatim "selection fails at within-state action discrimination along the candidate sweep". Why: I39
- Cuts (offsets): abstract "no success in 70 slots", "held out from it/adapted on it" → parentheticals; §1 variance-split sentence (kept in §6), "No single launch angle substitutes" sentence, 216-cell basis sentence folded into a parenthetical; §6 "No constant launch angle rescues…" sentence, "one lookahead arm selects 2 of 36 … counterexample" tail (in Table 1), bootstrap clause compressed; §8 "rather than treat aggregate prediction quality…" tail.
- Rejected: inline Δ² mapping in §1 (I38) — not word-neutral; method.tex carries it.

## 2026-09-23 · r3 follow-up (Evidence2): method.tex §3.1 request paragraph; appendix.tex App. H.1, tab:app:repro, tab:app:audit
- Changed: App. H.1 prose and audit "Rollout depth" row: "about 0.3--7%" → "between 0.25% and 25.6% of the outcome horizon", % comment marks it DERIVED (1/400.9, 25/97.8, lookahead findings.md lines 9, 14). Why: I35
- Changed: method.tex request paragraph states the Δ² lookahead mapping (Δ applications × Δ agent frames, 50 native steps per frame; 1/25/225), with % comments to runner loop, findings.md line 9, plan.json line 272; offset by cutting "the number of fixed steps the model advances" gloss and the "In the closed-loop record ... roll only 1 or 25" clause. Why: I38
- Changed: App. H.1 gains one inverted-ranker sentence; tab:app:audit new row "Inverted ranker (post hoc)"; tab:app:repro new row "#91 inverted-ranker probe". Why: I30 (owner-approved, executed)
- Deferred: none
- Rejected: registry C-row edits (no row quoted 0.3--7)

## 2026-09-23 · r3 Docs: review-log r3 dispositions, outline v2.2, CONTEXT, content_brief, research_evidence pointer
- Changed: `iclr2026/review-log.md` r2 entry gains [RESOLVED r3 …]/[PARTIAL r3 …] tags on I25–I34 and on the r1-partial residues I3, I9, I10, I11, I13, each citing a tex file:line checked in this pass; r3 authoring tally line appended before the r1 heading: 12 RESOLVED, 3 PARTIAL (I10, I11, I30), 0 REJECTED; I16 untouched. Why: round-3 Batch A landed
- Changed: `writing_outline_boundary_paper.md` → v2.2 (§0 thesis, round-3 wording bullet in §1, §2 rows for abstract/§1/§3/§6/§7/§8/App., C19 row, teaser line, §5 gates). Why: round-3 sync
- Changed: CONTEXT.md current state; content_brief.md CURRENT STATUS table + OPEN TEX ITEMS (round 3); research_evidence.md outline pointer v2.0 → v2.2 and registry home → `tab:app:registry`. Why: round-3 sync; pointer was stale
- Deferred: I11 residue (#82/#85 in experiments.tex:34); I10 residue (§5 three paragraphs); I30 inverted-ranker test (owner decision); I16 (author).
- Rejected: RESOLVED for I10 and I11 (tex still shows the named residues); RESOLVED for I30 (the reviewer's demand was a test, round 3 changed wording only).

## 2026-09-23 · r3 Main gate: build + page fit
- Changed: gate after Batch A green: 38 pp, body §1–§8 ends p. 9, Repro/Ethics p. 9 (uncounted), references p. 10 (same footprint as r2). Why: round-3 page-limit gate
- Changed: gate fallout routed to Thesis, discussion.tex trimmed ~80 words so Repro+Ethics fit p. 9. Why: gate
- Deferred: none.
- Rejected: none.

## 2026-09-23 · r3 Evidence: method, experiments, related work, appendix, figures/build_figures.py hero
- Changed: hero panel (a) title "N1: engine-truth successes vs. chosen ordinals"; panel (b) adds held-out zero-shot 0.2039 and adapted few-shot 0.2732 rows (216 cells, 6 clusters each), asserted against cross-pool summary.json; footprint unchanged. Why: I3, I26
- Changed: 005-a07 hatched mark redrawn 1 member tall on the members axis, legend "(excluded; 1 member)", geometry asserted. Why: I34
- Changed: Table 14 caption + App. H.1 name the 13th slot (issue-77-n1-005-a07). Why: I27
- Changed: C19 `supported` bound to the frozen 0.5-margin rule, 0.0780 reading post hoc; App. A token rule names its one disclosed exception. Why: I28
- Changed: App. H.1 rollout depth (1/25 carrier frames vs 97.8–400.9 agent frames, ~0.3–7%), 8/13 and 13/13 coverage; method (Δ,α) paragraph compressed with rollout-depth sentence. Why: I29, I10
- Changed: H.1/C19 h15 wording: opposite directions (hybrid ≈7.1→2.3, continuous ≈5.1→6.1), only hybrid-fixed-h15-e225 finds the success. Why: I32
- Changed: App. E.3 horizon-separation sentence replaced. Why: I9. fig:app:ceiling caption labels 42/1,224 as combined count. Why: I13. Related work: World-in-World → zhang2026worldinworld, PHYRE → bakhtin2019phyre. Why: I33
- Changed: §5 demotion notice ticket tokens removed (Thesis draft); §3.1 estimand paragraph two sentences, "defined but unmeasured" dropped. Why: I11, I10
- Changed: tab:app:audit rows Rollout depth (new), Lookahead control (frozen-margin result + h15 means), Cross-split scope (216 of 612, 6-cluster bootstrap). Why: audit rule
- Deferred: #82/#85 tokens in the experiments.tex:34 construct-validity paragraph (no draft sent).
- Rejected: none recorded.

## 2026-09-23 · r3 Thesis: abstract, introduction, synthesis, discussion
- Changed: abstract + §1 significance name the delta (engine-truth per-candidate AUC under a measured per-state ceiling, single-shot cascade regime); girdhar2020forward added in the abstract. Why: I25
- Changed: 216 of 612 scored cells per condition, 6 member clusters, member-clustered bootstrap in abstract/§1/§6/§8; held-out scoped to the frozen zero-shot arm, few-shot "adapted on it". Why: I26
- Changed: rollout-depth disclosure in abstract/§1/§6/§7/conclusion; mechanism clause kept, lookahead scope attached; §7 prints 13/13 coverage and the stability confound. Why: I29
- Changed: cross-split "systematically below chance" vs N1 "at or below chance" (abstract/§1/§6/§8, contribution 2); wording only. Why: I30
- Changed: "The hybrid ranker orders actions by the sweep coordinate" (0.9986; continuous 0.4924/0.6130; pooled 0.8776). Why: I31
- Changed: §7 flat-shift sentence replaced with Table-14-exact wording agreed with Evidence. Why: I32
- Changed: fig:teaser caption leads with engine-truth selection AUC; non-overlap demoted to "a post-hoc observation on N1"; gallery pointer cut from the caption. Why: I3
- Changed: §6 "#87/#89" → stage names; §7 "(Q1 …)" → "with reason readiness_or_precision_insufficient". Why: I11
- Changed: discussion trimmed ~80 words for the gate. Why: Main gate
- Deferred: I30 inverted-ranker test (owner decision; no new analysis this round).
- Rejected: none recorded.

## 2026-09-23 · r2 Docs: review-log, outline v2.1, CONTEXT, content_brief
- Changed: `iclr2026/review-log.md` r2 dispositions for I1–I24, each verified in the tex: 23 RESOLVED, 1 PARTIAL (I16). I6 was flipped to RESOLVED after Main's residue fix at synthesis:67 and discussion:47. Gate B is recorded: 38 pp, body ends p. 9, references p. 10. Why: r2 Batch A+B landed
- Changed: `writing_outline_boundary_paper.md` → v2.1. Covers the AUC headline, post-hoc disclosure, the mechanism sentence without state-difficulty claims, 3 contributions, the new title, stage names, and the objective-mismatch paragraph. Why: round-2 sync
- Changed: CONTEXT.md current state, content_brief.md CURRENT STATUS + OPEN TEX ITEMS. Why: round-2 sync
- Deferred: I16 anonymized link (author action).
- Rejected: none.

## 2026-09-23 · r2 Evidence: method §3/§4, experiments summary, appendix (anatomy, positioning, registry, lookahead, repro, audit), figures/build_figures.py hero
- Changed: §4 pre-registration scoped + verbatim restatement post-hoc sentence, and C18 now notes that the band/supports were post-outcome. Why: I3a
- Changed: fig:teaser (a) now shows member-unit counts {2:1,3:1,4:3,5:1,6:1}, with 005-a07 hatched. §3 adds "exactly one successful candidate". Panel (b) has separate axes, and the zero-/few-shot bars are split with no combined count. Why: I5, I22, I13
- Changed: new sec:app:lookahead + tab:app:lookahead with all 16 arms against per-arm chance 0.0780 and (12/13)^36 = 0.056, plus the carrier-MSE dose-response. C19 is rewritten. Why: I6
- Changed: §3 model paragraph (depth 3, 236-dim carrier, 1,837,690/1,862,396 params, 9,000 updates, 2,409 lineages), offset in the same file. Also an audit row. Why: I8
- Changed: the "only positive family contrast rests on one seed" wording, and C2b restricted to the original families. Why: I9. appendix:329 + C2 now say "unsupported: carried by one seed on an invalidated proxy". Why: I18. appendix:411 now uses the PHYRE/World-in-World sentence. Why: I2. #83/#84 sentences moved to the appendix. Why: I10. C20 gets 19/26. Why: I14. The a04 audit row now reads 3/7 members. Why: I15. C21 now says "descriptive" and includes 2/153. Why: I19, I23
- Changed: added a run_state_difficulty_auc row to tab:app:repro, plus new rows in tab:app:audit (mechanism, per-member successes, state-difficulty AUC, lookahead, second seed, cross-split AUCs, architecture). Why: I4 and the audit rule
- Deferred: synthesis:67/discussion:47 "below the frozen 0.5 margin" (Batch A files, reported to Main); the I8 pretrained-WM experiment (no new experiments).
- Rejected: promoting fig:app:ceiling to the body (not page-neutral).

## 2026-09-23 · r2 Scout: related work (bib + related_work.tex + related-work .md docs)
- Changed: new "Decision-relevant prediction" paragraph (objective mismatch), with bib keys `lambert2020objective` and `girdhar2020forward`. Why: I7
- Changed: PHYRE-solvability positioning sentence (the World-in-World closed-loop paragraph keeps the per-state ceiling as our addition). The two related-work .md docs were updated. Why: I2
- Rejected: none recorded.

## 2026-09-23 · r2 Thesis: abstract, introduction, synthesis, discussion, root \title
- Changed: headline statistic = AUC; cross-split AUCs per condition (0.2039 / 0.2732) printed as held-out replication; disjoint supports demoted to an N1 observation. Why: I3(b,c)
- Changed: §6 post-hoc disclosure of the restatement (member unit, success band, disjoint supports, AUC as headline); state-difficulty AUC (0.5758 / 0.5918 / [0.2653, 0.8958]) + verbatim post-hoc sentence; "learn state difficulty" / "or state-difficulty estimation" cut everywhere. Why: I3(a), I4
- Changed: full variance split with 0.636 residual, "action main effect", per-system Spearman 0.4924/0.6130/0.9986. Why: I4
- Changed: denominator claim repositioned to closed-loop world-model success reporting (World-in-World), PHYRE solvability + 100k cache as precedent (Scout-verified); falsified "Benchmark papers report..." sentence deleted (intro, synthesis). tab:syn:denominator left column relabelled. Why: I2
- Changed: one success per ceiling state + member counts {2:1,3:1,4:3,5:1,6:1} (§1 caption, §6). Why: I5
- Changed: abstract scoped to three ~1.8M-parameter frozen predictors. Why: I8
- Changed: contributions 1–2 cut/folded into a protocol item; "every reported number names its estimand" removed; intro ¶1–2 compressed; discussion (d) #83/#84 sentences cut. Why: I10
- Changed: stages named oracle pass / completion pass / timeout audit / restatement; accounting provenance moved to tab:syn:oracle caption (binding row cut). Why: I11
- Changed: top-1 paired interval dropped; 7-cluster bootstrap caveat. Why: I12. Counterexample leads with frozen 13/612; 42/1,224 only as "combined count across two separately reported conditions". Why: I13. 19/26 byte-identical frames, "effectively deterministic". Why: I14. a04 3/7 members (5/12 states). Why: I15. "+0.0887 unsupported: carried by one seed on an invalidated proxy". Why: I18. Interval-exclusion wording. Why: I19. "108 model cells (plus 36 prior cells)". Why: I20. 0/70 = candidate slots. Why: I21. 2/153 analogue systems. Why: I23. Title → "Solvable but Mis-Ranked: ...". Why: I24
- Deferred: §4 matching disclosure, §3 model paragraph, appendix:411 sentence, fig:teaser panel (a) rebuild, #83/#84 sentences placement → Evidence batch.
- Rejected: none.

## 2026-09-23 — #91 round 1: ICLR main-track rebuild (thesis = measured ranking failure)

- WP1 thesis restructure: abstract rewritten to the scoped ranking-failure claim (round-2 framing wording); introduction spine (a) proxy-era reading + invalidation cascade, (b) engine-truth audit + scoped zero-selection with counterexample, (c) mechanism, (d) scope/limits; new contribution list; synthesis re-centred on `tab:syn:oracle` + denominator-contrast table; discussion §7(a) resolved-chain rewrite (#90 as protocol working), §8 significance skeleton.
- WP2 factual corrections (all 8): h=15 "null on N1, exactly zero on original families"; cost superlative corrected to continuous_adaptive 2.0147 s + arm disclosed in walls table; regime-separation D negative only at e=120 (qualifier added); Table 6 relabelled exposure/few-shot contrast (τ_ad unmeasured); withheld h=15 intervals printed; `data/issue-76-closeout/` provenance path fixed; TAWM degeneracy (0.3998, identical Top1/Top3) disclosed; per-seed decompositions (0, 0, +0.2662 / +0.2905, −0.0215, +0.0141) printed wherever the aggregates appear, abstract included. Extra correction beyond the ticket: #78 cohort is 21 states (4 N1 + 17 N2), table columns relabelled accordingly.
- WP3: registry rows C14–C21 (pre-applied by #92 WP4/WP5) verified against the digest; C2/C8 demoted; C3 corrected.
- WP4 figures: five real vector PDFs, all values asserted against artifacts by `iclr2026/figures/build_figures.py` — `fig_hero_ordinal_masses.pdf` (teaser, `fig:teaser`), `estimand_taxonomy.pdf`, `fig_error_growth.pdf`, `fig_novelty_boundary.pdf`, `fig_ceiling_map.pdf` (`fig:app:ceiling`). The family×horizon teaser grid was NOT built (dead thesis; per-cell values #81-gated).
- WP5: ICLR 2026 submission limit verified at 9 pages main text (Author Guide; the .sty is silent; recorded in root tex comment). Body cut ~21 → 9 pages: claim registry (`tab:app:registry`), frontier (`tab:app:frontier`), §5 subsections + floats (`sec:app:anat`), related-work subsections + matrix (`tab:app:matrix`), estimand figure/table, §4 element list moved to the appendix; `tab:app:audit` number→artifact longtable added; reproducibility statement completed (7 `--validate` runners, 24 h/3 h GPU/4-worker caps, 4,187 tracked paths through `2b4d897`); Ethics statement rendered; checklist emptied (ICLR 2026 requires none).
- Docs sync: `writing_outline_boundary_paper.md` → v2.0; `research_evidence.md` gained #82–#92 rows; `CONTEXT.md` current-state block; `content_brief.md` CURRENT STATUS table; `critics/2026-09-19-critic-1.md` fully dispositioned (9/9 RESOLVED, incl. the description-mode causal-lever point in discussion §7(a3)).
- Citation pass: Henderson AAAI'18 + Agarwal NeurIPS'21 added (source-verified) for the evaluation-caution arc; `zhang2026worldinworld` entry confirmed; denominator-positioning paragraph at `sec:app:rel:eval` (softened per verification: PHYRE/I-PHYRE/World-in-World checked full-text; NovPhy excluded — denominator reporting unverifiable).
- Round-1 review: 4/10 weak reject, 24 findings I1–I24 — see `iclr2026/review-log.md` + `review_r1_full_report.json`; 22/22 spot-checked numbers match artifacts; all 8 WP2 corrections confirmed.
- Post-review gate fixes: registry + repro tables converted to longtable (I1 — rows C10–C21 had stopped rendering); matrix resizeboxed; broken `.local-artifacts/issue-15-confirmatory-v2/` roots corrected to `data/runtime_evidence/issue-15/` (I16 partial; anonymized artifact link remains an author action); C16 + discussion carry the #90 token `indeterminate` (I17); prior-ordinal-8 claims scoped to ceiling cells; xurl + breakable typewriter tokens (appendix overfulls ≤ 27 pt, none content-destroying).

## 2026-09-23 — State-difficulty AUC analysis (resolves review I4's open question)

- NovPhy repo: `scripts/run_state_difficulty_auc.py` + `.local-artifacts/issue-91-state-difficulty-auc-v1/` (sha256-pinned inputs; `--validate` exits 0; retained records only, zero engine seconds).
- Result: state-level AUC 0.5758 (23 states; higher predicted cost on non-ceiling states), member-deduplicated 0.5918, member-clustered descriptive interval [0.2653, 0.8958] covers 0.5 → "the predictors learn state difficulty" and "not at state-difficulty estimation" are NOT supported. Round 2 cuts both clauses; the within-state discrimination half of the mechanism sentence survives. Post-hoc disclosure required (analysis specified after #87/#89 outcomes).

## 2026-09-23 — Housekeeping (owner directive: no confusing deprecated material)

- Retired to git history: `critics/` (all 11 files; the 09-19 closeout is recorded in the round-1 entry above), `teaser_figure_brief.md` (superseded; the hero-figure contract lives in `issue91_exec_digest.md` and `content_brief.md`), `iclr2026/.tex-build/` (stale pre-rebuild build with the old title), `iclr2026_conference_checklist.tex` (ICLR 2026 requires no checklist; root tex `\input` removed).
- Trimmed marked-superseded tails: `writing_outline_boundary_paper.md` v1.2 content, `content_brief.md` historical record + the fully-closed OPEN TEX ITEMS list. Both now carry a one-line pointer to git history.
- Reference artifacts added for session portability: `issue91_exec_digest.md` (binding #91 numbers/wording), `review_r1_full_report.json` (round-1 review, full).
