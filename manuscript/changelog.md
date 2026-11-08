# Changelog

## 2026-11-08 · Author-directed framing/presentation revision, with m2 and m3 review
- Changed: abstract and §1 now open with the fixed-stride/fixed-description problem and action-sparse physics motivation, then separate the request-interface claim from the learned-selection hypothesis. C1–C4 lead with sharing (+0.0848), prediction–ranking dissociation, fixed-request controls and direct prediction; no withdrawn superiority, 3×-compute or novel-physics claim was restored.
- Changed: §6 replaces repeated question/answer openers with direct takeaways. §6.2 adds matched per-horizon sharing +0.0324 [0.0219, 0.0431] and seed accounting. §6.3 leads with the same-hybrid $F_{\mathrm{hind}}$ versus $J$ comparison, quantifies $J-D_{15}=-0.0562$ [-0.0685, -0.0441], restores the offset rerun, and frames the fixed-$\Delta=15$ result as a budget-conditional counter-observation. §6.1 states CLEVRER video-ID disjointness. §6.5 compares direct prediction with both the 0.885 full-budget specialist and $U$'s 0.744.
- Changed: §7–§8 synthesize interface value separately from selector value and remove caveats already stated in §3/§6. Q&A Table 2 was shortened to remove a 16.3pt overfull box.
- Review: m2 checked claim/comparator/scope alignment; m3 checked motivation, structure, presentation and style. Rejected suggestions that would upgrade the direct-prediction conclusion using the G-action-failed longer-training control, use an absolute no-prior-method claim, or hide the at-most-budget/seed qualifications.
- Checked: `latexmk -pdf` succeeds; 66-page PDF, main text including Conclusion ends on p.10 and references start on p.11; final log has no undefined references/citations, LaTeX errors or overfull boxes. Rendered and inspected pp.1–3 and 7–10. `sha256sum -c --quiet ../iclr2026_as_submitted.sha256` from the snapshot directory passes all recorded files. No new experiment or scoring was run.
- Deferred: existing author-action items in `AUTHOR_HANDOFF.md` (record packaging, missing pixel reference scoring, research-stage AI disclosure and venue-template update) are unchanged.

## 2026-11-08 · W2 heading-keep fix: §6, Appendix D, Appendix E
- Changed: `\nopagebreak` inserted immediately after every `\begin{revblock}` that directly follows a \section/\subsection heading (§6: 6 places; App D: 8; App E: 6, including the longtable revblock under E.1). No text, label or number changed. Why: I-P08 (colour whatsit after a heading creates a legal page break); Main r5 instruction.

## 2026-11-08 · W1 layout fix: `\nopagebreak` after revblocks that open directly after a heading
- Changed: `\begin{revblock}\nopagebreak` at §5 (method.tex:100), App A (notation.tex:7) and App C (appendix_method.tex:7). Other revblocks in my files open before a `\subsection` or mid-section, not after a heading. No other change.
  Why: Main (stranded §5 heading; colour whatsit after `\section` creates a legal break).


## 2026-11-08 · W5 final where-printed pass: Appendix H audit table
- Changed: live row drops §1 (C4 no longer prints live pass rates) and §7 (prints no live number); per-seed row adds §6.3 (pointer "one of three seeds favours the hybrid"); novelty row notes §7's rounded prior 0.60. Why: Main r5 request after W1/W2/W4 r5 edits.
- Checked, no change: abstract (400, +0.0848, +0.0201), §1 C1–C4 (C1 full-budget clause has no number; 0.783, 0.885, +0.0201, 0.02, U−prior, 0.976/0.744, −0.0044), §5 (time base, 1,000/480/120/400, 3,480, 42,344, 859, 240, 384/456, 10,000), §6.2–§6.5 (§6.4 no longer prints +61.241M, which App D still prints), §7 (U−prior, J−prior, 1,000/9,000, 0.885, 16,929, 1.88×, 0.813, F1 0.157/0.113/0.015, 0.9039, novelty), App D wall times and decomposition, App F 16.74/1.63. Time-base row never cited §7.

## 2026-11-08 · W4 r5 re-check fixes: §1, §2, §7, §8, App G
- Changed: C1 "; at full budget, dedicated predictors rank most coarse requests better (Section 6.2)". Why: I-F18 regression.
- Changed: §7 "Other" restores "so no model acts on a changed state" ("every" dropped). Why: Main r5.
- Changed: App G opening rev sentence → "fixed (Δ=15, cont) requests on the primary continuous predictor". Why: I-P01 / W22-amended.
- Changed: `\nopagebreak` directly after `\begin{revblock}` that follows `\section` in §2, §7 and §8. No float follows any of these headings; the colour special that `revblock` inserts in vertical mode after the heading creates a legal page break (whatsit before the first paragraph's glue), which can strand the heading. Why: I-P08.

## 2026-11-08 · W2 r5 re-check fixes: §6
- Changed: Table 3 (`tab:res:rollout`, now `[t]`), Table 4 (`tab:res:ranking`) and Figure 3 (`fig:res:frontier`) moved in source to just before their \subsection commands. Why: I-P08 (stranded headings). §6 heading/Table 2 checked: the opener paragraph sits between them, so no change.
- Changed: §6.3 U−C* sentence adds "(pooled over seeds; one of three seeds favours the hybrid, Table `tab:app:res:perseed`)". Why: I-P12.

## 2026-11-08 · W1 layout fix I-P08: float placement in §4–§5
- Changed: `tab:setup:arms` float moved from directly after `\section{Experimental Setup}` to directly before it; the Figure 3 float (`\input{iclr2026_conference_fig_model}`) moved from directly after `\section{Method}` to directly before it. Both keep `[t]`; no text, label or number changed. §3 has no float after its heading.
  Why: I-P08 (heading stranded at page foot by a float right after `\section`).


## 2026-11-08 · W5 I-P13: Appendix H pixel-protocol row
- Changed: pixel-study protocol row drops "(16.8 in App. D)" and "(45.9)"; App D now prints 16.83 and 45.88 px, matching App B. Why: I-P13 (Main relay).

## 2026-11-08 · W4 coordinated cut: §1, §7
- Changed: §1 ¶1 simulator analogy compressed to one short sentence (blue); ¶3 grid/ordering-AUC/split/prior merged into two sentences (definitions kept); declaration paragraph folded into a final C4 sentence pointing to §5; C4 "Live pass rates… narrower trunk (App F)" dropped (§6 states both). §7 "What the evidence supports" and "What it does not support" merged into "Scope of the evidence" (interface clause kept; sharing pointer dropped, in §6.2); time-base table pointer dropped; Live and One-benchmark items merged into "Other". W1, W8, W9, W11, W18, W22-amended, W23–W26 verbatim.
  Why: Main coordinated page-fit cut (−8 lines share; m1 Q12).

## 2026-11-08 · W2 coordinated cut: §6
- Changed: §6.5 dropped A1 pixel clause (App E keeps it); §6.3 J−prior, U−prior and right-minus-wrong anchor merged into one references sentence; Tab 3, Tab 4 (dropped "λ as in Section 4"; "---: not in this accounting"), Fig 3 (W13 verbatim kept) and Tab 5 captions shortened. Why: Main coordinated cut (−5 lines share).

## 2026-11-08 · W1 coordinated page-fit cut: §3–§5
- Changed: §3 surrogate lead-in joined to its display with a colon; context/additivity paragraph shortened to two sentences (DP minimizes G exactly given fixed contexts; G is not the recursive rollout cost); inventory/mixed-outcome definitions merged into the ordering-AUC sentence. §4 "Per-call costs are in Appendix C" dropped; "What changed" 5 → 4 sentences (data, scoring and evaluation merged). §5 arm key (`tab:setup:arms`) moved to directly after the §5 heading ([t]).
  Why: Main coordinated page-fit (−5 lines for W1). W-sentences, labels and numbers unchanged.


## 2026-11-08 · W2 directed revision (m1 R4): §6, Appendix D, Appendix E
- Changed: §6.3 opener and `tab:res:qa` Q3 scoped to rankers on the primary hybrid and continuous predictors (9,000 updates each); added full-budget single-request specialists (up to 0.885, `sec:app:res:sharing`). Why: I-P01 / W22-amended.
- Changed: §6.5 direct prediction: A2 "matched to the shared predictor in total updates rather than updates per request"; readout control in W24 wording ("scores slightly lower … −0.0044 [−0.0081, −0.0014]"); App E paragraph same, "adds no measured ranking value" → "provides no evidence of incremental ranking value". Why: I-P02/W24, I-P04.
- Changed: App D sharing paragraph: (15, macro) specialist 0.885 [0.872, 0.898]; "no experiment retrains it longer" → #140 sentence (16,929 updates, 1.88×, 1.654× U time, 0.813 [0.798, 0.829], failed G-action, selector comparisons not rerun). Added `tab:app:res:perseed` (#108 summary `per_seed_grid_auc`, `c_star_per_seed_grid_auc`; descriptive) and a pointer paragraph. App E U-native row/text: A4 compute was the target, 0.879× achieved. Why: W25, W28/R4-1.
- Changed: "pre-registered" → "pre-declared" (Tab 4 caption, App E opener, H-JEPA paragraph, #127 claim). Why: I-P06/W27.
- Changed: §6.5 heading now follows the `tab:res:live` float in source. Why: I-P08 (orphaned heading); needs PDF check.
- Line compensation: §6.2 dropped "On the 311 mixed-outcome grid levels," (W6 wording restored); §6.3 description paragraph dropped the audit-head clause (kept in §7); §6.4 dropped "+61.241M" (kept in App D.10); §6.5 dropped "but only descriptively" (the descriptive-maxima clause remains); two-appendix pointer → `sec:app:results`.

## 2026-11-08 · W5: Appendix H rows for W24/W25/W28 and I-A08 NN-px
- Changed: nearest-fit row back to NN-px 0.9039 [0.8838, 0.9231] everywhere (App E reverted rounding); NN-car 0.777 [0.750, 0.804] (App E). Why: I-A08 (Main relay).
- Changed: direct-prediction row adds #141 classifiers 1,855,301 parameters each, R(ẑ225) − A2 +0.0008 [−0.0019, +0.0034] (program findings line 106; #141 findings line 16), and the 472-dimensional input (2 × 236 carrier, marked as not stated in the #141 records). Why: new App E text.
- Added: three `tab:app:audit` rows. Full-budget specialist 0.885 [0.872, 0.898] and 1,000 of 9,000 updates (§1, §6.3, §7, App D), from #117 findings line 59 (0.8850 [0.8717, 0.8976]). U-native 0.813 [0.798, 0.829], 16,929 updates (1.88×; record 1.881), 1.654× U's and 0.879× A4's measured training time (§7, App D, App E), from #140 findings lines 3, 7, 17–22 and program findings line 49. Per-seed table `tab:app:res:perseed` (U 0.714/0.801/0.718, C* 0.826/0.733/0.791 with intervals), from #108 summary `per_seed_grid_auc` and `c_star_per_seed_grid_auc`. Why: Main request (W24, W25, W28 directed revision); every value checked against the record.

## 2026-11-08 · W4 directed revision (m1 R4; W22-amended, W24–W28): Abstract, §1, §7, App G
- Changed: abstract "400 held-out, within-distribution NovPhy levels" (W28); W22-amended abstract sentence; direct-prediction sentence → W28 ("…at the tested budgets, and adding the rollout endpoint slightly lowers AUC"). Why: m1 R4-1/R4-2, I-P04, I-P11.
- Changed: §1 C3 opens with the two W22-amended sentences (0.783 on the continuous predictor; specialists 0.885), keeps W22's closest-selector sentence with $J_{\mathrm{pg}}$ defined in place ("trained on policy-fit AUC"; I-P05), then W26 verbatim with neitz2018adaptive, chatterjee2025improving, then the same-checkpoint contrast. C4 direct-prediction/readout sentences → W24 verbatim (U in math mode; section pointers folded into the closing parenthesis); prior, live and compute qualifications kept; "pre-registered" → "pre-declared". Why: m1 R4-2/R4-3, I-P01, I-P02, I-P05, I-P06 (W27).
- Changed (length compensation in §1): C1 qualifier cut (now carried by C3/C4's 0.885); ¶2 prior-work clause compressed; ¶3 "References include…" → one sentence defining the no-model prior; pixel-study sentence folded into C2 with citations.
- Changed: §7 Budget item → W25 verbatim, moved first among limitations; "G-action" given an App E pointer (`sec:app:ext:guard`); "pre-registered" → "pre-declared"; supports paragraph tightened. App G "pre-declared guard". Why: m1 R4-2, R4-4.
- Not done: I-P08 page-2 slack is a float/page-break effect not traceable to text in my files without the PDF; Figure 1 stays `[t]` after ¶1.

## 2026-11-08 · W1 directed revision (m1 R4; W27, W28): §5
- Changed: W27 verbatim after the first "pre-declared" (grid primary inventory); W28 verbatim after the bootstrap sentence. "pre-declared" is the only term in my files (no "pre-registered" remained). Why: m1 R4, contract W27/W28.
- Changed (line compensation): separate pixel-study paragraph merged into the time-base paragraph (own parser/predictors/policies, five seeds, §6.1 caveats, units); dropped "(full key in Appendix A)", "from its earlier training corpus", "Ordering AUC is averaged over cells", the t=225 gloss; "the predictor's ranking loss" → "the ranking loss". Why: net main-text change ≤ 0.
- Rejected: adding J_pg to the Table 1 J row (would wrap the cell); App A already defines J_pg.


## 2026-11-08 · W2 integrity-audit fixes: §6, Appendix D, Appendix E
- Changed: I-A01 readout control described as two A2-architecture classifiers (1,855,301 params, 472-d input) trained separately on [z0, ẑ225] and [z0, z0] — Tab 4 caption, App E rows (a) and "Direct prediction" paragraph (adds F1 − A2 +0.0008 [−0.0019, +0.0034], #127 findings l.106). I-A02 "The pre-declared same-checkpoint contrast meets the frozen δ rule". I-A04 Tab 13 caption: J-flat is #116's own 225-frame-window policy (374.9M MACs on R_9), not J. I-A05 App E NN-px 0.9039 [0.8838, 0.9231]; App D parser centre error 16.83 / 45.88 px. I-A06 App D Tab 14 rows/caption $J_{\mathrm{cont}}^{\mathrm{NovPhy}}$, Fig 3 caption legend gloss, App D "sealed NovPhy split". I-A07 Physion overlap pointers → `sec:app:proto:pixel` (§6.1, App D). I-A09 `zhang2026hjepa` on the H-JEPA row and paragraph; HWM/LeWM rows labelled re-implementations in the H-JEPA codebase. I-A10 App D scaling attribution reworded. I-A12 W4 ablation pointer "(Appendix D.3; exploratory, development lineages)". Tab 12 overfull: shortened question/contrast labels.
  Why: review-log r3 IntegrityAudit I-A01, I-A02, I-A04–I-A07, I-A09, I-A10, I-A12; Main.
- Line compensation in §6: opener trimmed; "(Section 6.5)" pointer dropped from §6.3; §6.5 guard sentence compressed (scope kept: 5 levels = 3 evaluation + 2 novelty-set).

## 2026-11-08 · W4 integrity-audit fixes: Abstract, §1, App G, appendix front matter
- Changed: §1 C4 readout control → "a same-architecture classifier trained with U's predicted endpoint as an extra input scores −0.0044 [−0.0081, −0.0014] AUC against one trained without it"; "0.976 against U's 0.744". Why: I-A01.
- Changed: "single-request specialists" in abstract and §1 C1; "sealed NovPhy split" in appendix.tex summary and App G (former "The sealed evaluation"). Why: I-A06.
- Changed: §1 C4 pointer → Sections 6.3, 6.5. Why: I-A07.
- Changed: App G hierarchy paragraph: "We reproduced end-to-end hierarchical JEPA (zhang2026hjepa) and trained its codebase's re-implementations of HWM and LeWorldModel on NovPhy. Every H-JEPA-family arm we trained failed a pre-registered guard…". Why: I-A09.
- Changed: App G "among world-model request rankers, a fixed coarse request ranks best". Why: I-A11.
- Net main-text length paid inside C1/C4 ("each", "withdrawn claims:" and "for U" dropped). The I-F20 sentence keeps Main's "evaluation split" verbatim.

## 2026-11-08 · W1 integrity-audit fixes: §5, Appendices A–C
- Changed: App A arm key readout row → two separately trained classifiers with A2's architecture and recipe (472-value input, 1,855,301 parameters each) on [z0, ẑ225] and [z0, z0]; "neither is A2 itself". Why: I-A01.
- Changed: policy-fit outcome uses now include J_pg (REINFORCE, λ selection) and the remaining-frame (κ) input variant: §5 sentence, App B item (iv), App A glossary. Why: I-A03.
- Changed: §5 CLEVRER unit 0.039 s → 0.0390625 s; paid by dropping "(no novelty)" and "in Appendix A" from §5 (net ≤ 0 lines). Why: I-A05.
- Changed: App C stability adds continuous-predictor counts (190/92/88 large-gradient updates; 109, 0, 6 after update 500; #112-v3 findings l.33–35). Why: Main (audit note).
- Rejected: adding the rerun's secondary count-cost reading to §3 (costs a main-text line; App B already states it).


## 2026-11-08 · W5 I-A08: Appendix H rows synced to App E three-decimal AUCs
- Changed: nearest-fit row now lists NN-px 0.9039 [0.8838, 0.9231] where §5, §7 and App B print it, and App E's 0.904 [0.884, 0.923] and NN-car 0.777 [0.750, 0.804]; direct-prediction row drops the four-decimal parentheticals (0.9757, 0.9946, 0.9765, 0.9809), which are no longer printed anywhere; external row now reads DIAMOND 0.718, ablations 0.493 / 0.539 / 0.493, "AUC at 3 decimals".
  Why: integrity audit I-A08 (Main relay).
- Deferred: App E still prints NN-px 0.904 at last read (snapshot before W2's pending fix). If W2 switches it to 0.9039, the App E clause of the nearest-fit row needs the same change.

## 2026-11-08 · W5: Appendix H audit row for J_pg λ=10⁻⁴
- Added: `tab:app:audit` row for the App D J_pg λ=10⁻⁴ sentence and its `tab:app:res:frontier` row (0.764 [0.745, 0.783]; J_pg − F_hind +0.0021 [−0.0030, +0.0070]; 29.7 transitions; 74% at (5,cont); 34/933 cells), anchored to #101 `compute.json` `inventories.grid.lambda_sweep.pg` (J-pg-1e-04: auc, minus_F_hindsight) and `inventories.grid.schedules."J-pg-1e-04"` (pair_counts 328179/39887/74640, non-modal share 0.2587, 34 differing cells), and #103 `frontier-table.csv` line 48 (29.656).
  Why: Main request; closes the deferred item of the layout entry below.
- Changed: selector-collapse row now also anchors J_pg − F_hind −0.0011 to #101 `inventories.grid.contrasts."J_pg-F_hindsight"` (first J-pg-1e-01), read directly.
  Why: same request (value now verified in the record, previously taken from W2's log).
- Checked: no audit row uses hard-coded table or figure numbers (all `\ref`); Table 2 (`tab:res:qa`) prints only 0.784, already in the compute row; Table 1 (`tab:setup:arms`) λ values in the costs row.

## 2026-11-08 · W5 layout: Appendix H overfull boxes
- Changed (layout only, no content): submitted-checkpoint roots in `tab:app:checkpoints` and `tab:app:repro` moved from hand-broken `\texttt` to `\nolinkurl`; `\_` made breakable inside the three table groups (local `\protected\def`) so long `\texttt` identifiers and commands wrap; repro columns 0.16/0.25/0.33/0.20 and audit columns 0.16/0.35/0.35/0.08 of `\textwidth` (sums 0.94, inside the tabcolsep budget); `\allowbreak` in the two λ-grid sets (costs row), the CLEVRER command in the prose, and the ticket range "#127–#141".
  Why: Main request (30 overfull \hbox warnings in App H).
- Deferred: W2 fix round 2 added App D J_pg λ=10⁻⁴ numbers (0.764 [0.745, 0.783], +0.0021 [−0.0030, +0.0070], 29.7 transitions, 74%, 34/933) not yet in `tab:app:audit` (layout-only brief).

## 2026-11-08 · W2 fix round 2: §6, Appendix D, Appendix E
- Changed (§6): I-R27 "The best sweep point and the policy-selected J_pg reach F_hind by collapsing to a fixed coarse horizon"; "the decision-valued selector" → "J_dec", "policy-gradient selector … selected on the policy split" shortened to pay the line; I-F20 alignment "the contrast declared before the primary stack was first scored" (U−C*); I-R29 "3,132 CLEVRER validation"; I-R30 "sealed NovPhy split" (§6 opener); I-R09 Q2 "lead on most coarse ones" (question shortened to "One predictor for all requests?"); I-R28 Q&A table first column 0.30\linewidth.
  Why: review-log r2 I-R09, I-R27–I-R30; Main fix round 2 + I-F20 note; net 0 main-text lines intended.
- Changed (App D): I-R27 collapse sentence re-scoped + J_pg λ=10⁻⁴ sentence (0.764 [0.745, 0.783]; −F_hind +0.0021 [−0.0030, +0.0070]; 29.7 transitions/candidate; 74% of steps (5,cont), rest Δ=15; 34/933 cells differ), verified in #101 compute.json grid `pg` λ=1e-4 record and `schedules."J-pg-1e-04"` (pair_counts 328179/39887/74640) and frontier-table.csv; I-R28 Tab 12 `\tabcolsep` 3pt, λ/window note moved to caption, two row labels shortened; I-R30 opener "sealed NovPhy split".
- Changed (App E): I-R06 decision — AUC columns and AUCs in prose at 3 decimals (rounded from #127/#123/#125 summary.json full precision: A4 0.99448→0.994, THICK upper 0.51648→0.516, HJ-px 0.49967→0.500, HJ-car-native lower 0.51850→0.518, HWM-px 0.50010→0.500, LeWM-car 0.49451→0.495); contrasts stay at 4 decimals; disclosure sentence removed.
- Changed (follow-up): `tab:res:qa` set to `lll` with one-line rows; questions shortened ("Q1. Rollout error?", "Q2. One shared predictor?", "Q3. Selection vs fixed?", "Q4. Compute saving?", "Q5. Live; rollout vs direct?"); Q1 answer "Changes with granularity; …", Q3 answer drops the leading "No;". Why: Main follow-up (Q&A rows wrapped, ≈ +5 lines).

## 2026-11-08 · W4 fix round 2: Abstract, §1, §7, appendix statements
- Changed: §1 pre-declaration sentence replaced by Main's I-F20 sentence (U−C* before the primary stack was first scored; later experiments incl. J−J_cont before their own scoring), with "(the best fixed coarse request above)" to define C* in place; C3 names $J_{\mathrm{cont}}^{\mathrm{NovPhy}}$ so the sentence's symbol has a referent; C4 "Neither the pre-registered request U=(15,macro)"; withdrawal pointer folded into C4 to pay the line. "Ranks a level's 16 grid launches" defines grid; "object-slot latent states" → "object-state vectors"; "Separate … We make four claims." cut.
  Why: I-F20, I-F05 remainder, method-reviewer note; 0-line budget.
- Changed: abstract "AUC, a small gain at higher compute"; "dedicated predictors given more updates" (≈ 176 words). Why: I-F21.
- Changed: §7 W9 parenthesis followed by ", below the no-model prior's 0.60" (App D / #128: 0.6022 [0.5459, 0.6559]); "so for U and J a scene-blind ranking…"; "as the controls of Section 6.3 require," cut. Why: I-F22.
- Changed: Reproducibility Statement "The supplement also lists …", parenthesis split into its own sentence. Why: I-F24.

## 2026-11-08 · W1 fix round 2: §3–§5, Appendices A–C
- Changed: Tab 2 (`tab:setup:arms`) cells: "$C\text{-}F(\Delta)$; $C^*{=}C\text{-}F(15)$", "$J$ restricted to one α/Δ; replay", "live; oracle nondeployable", shorter cells and refitted widths. Why: I-M11. App A full arm key → `longtable` (breaks across pages; blue via group colour). Why: I-M16. Glossary tie-free cost → "every post-submission scoring, including the primary cost of the submitted-checkpoint rerun". Why: I-M17.
- Changed: Eq. `eq:auc` loses the trailing y-conditions (moved to the next sentence); Eq. `eq:lc` names the ranking term ℒ_rank, defined in the following sentence; App C DP display defines ω after the display. Why: I-M18. §5 "scored once," → "scored". Why: I-M19. §5 time-base sentence adds the pixel-study units (CLEVRER one video frame, 0.039 s; Physion three native frames, 0.1 s). Why: I-M20.
- Changed: "sealed split" / "sealed cohort" → "sealed NovPhy split" in §4, App B (2) and App C. Why: Main (W8 name).


## 2026-11-08 · W5 fix: Appendix H stability and time-base rows
- Changed: stability row of `tab:app:audit` now reads "pre-clip gradient norm above 10^3 on 104, 108 and 108 of all 9,000 updates … and on 1, 0 and 0 updates after update 500" (wording of App C l.84).
  Why: Main relay of method-reviewer item (old wording read as counts after update 500).
- Checked: time-base row cites `tab:setup:timebase` only; no row cites `tab:app:timebase` (removed in the resync pass).

## 2026-11-08 · W5 resync: Appendix H audit table (`iclr2026_conference_appendix_repro.tex`)
- Changed: `tab:app:audit` rows resynced to the page-fit and fix-round text (now 47 rows): split row (abstract prints 400 only; Reproducibility Statement added); Fig 1 row (caption now 50 steps/0.02 s, endpoint 225, settled ≈553, 45-frame schematic; ticks 0/80/120/553 kept as artwork); Fig 2 row points to App A; costs row (J_dec selection 0.734 vs 0.683, J_pg λ tie rule; #101 doc 51–61, 137–138); architecture row (§4 "12 earlier NovPhy levels"); stability row (104/108/108 total, 1/0/0 after update 500, C* 39 + 8 updates; retrain findings 30–35, #112 doc 230–244); perception row (0/240 vs 240/240, 60 per level; #99 plan claim_boundary); time-base row (table now in App A); rollout (a), short-window, tied-axes, macro, sharing, compute, contrasts rows re-pointed (§2 and §7 no longer print the tied-axis or W5 numbers; §6.3 no longer prints macro F1; decomposition only in App D; wall 0.242/0.014 added); direct-prediction row adds A2 1.84M; external row adds U 1,294 GPU-s, native budgets 54,250/123,858/200,000/409,600, 18.87×, #141 MACs 22.8M/21.0M/1.84M vs 22.0M, H-JEPA sensitivities, ablations, readout range, LeWM IDM rows and ceiling 0.5698; not-run/calibration row (six arms; U 0.0229 per seed, C* 0.0334).
  Why: Main resync request (IRC, 2026-11-08); W1/W2/W3/W4 fix and page-fit rounds.
- Added: row for the time-base sources W1 removed from App A; row for the selector collapse (J λ=10⁻¹ 15.0 transitions/calls, 0.770, C*−J(λ=10⁻¹) +0.0134 [+0.0065, +0.0202]; J_pg 0.761, J_pg−F_hind −0.0011 [−0.0031, +0.0009], 0/933 cells differ).
  Why: Main resync request.
- Deferred: J_pg per-seed requests ((15,macro) for 20260908, (15,cont) otherwise) seen only as pooled schedules in #101 findings 68–77; C*−J(λ=10⁻¹) field taken from W2's logged anchor, not re-read. Conflicts still open: App C "about 19 times" vs record "19x the v3 windows"; 16.83/45.88 vs 16.8/45.9 px.
- Rejected: none.

## 2026-11-08 · W4 page-fit pass: Abstract, §1 (+ Fig 1 caption), §2, §7, §8
- Changed: abstract 193 → ~174 words (level glosses, "separate … pixel studies", "while spending" cut). §1: Fig 1 caption drops the frame list and merges (b)/(c); ¶3 merges the ranking and AUC sentences, drops "Angry-Birds-style" and the 11,680-launch clause (in §5); C1–C4 keep headline numbers and qualifiers; J−prior, NN-px 0.9039, the parsing-check detail and the "frozen rule" clause left to §6/§7; compute clause shortened; "other comparisons are descriptive" sentence folded. §2 sentences merged (temporal list, MPC/MuSix/VisualPredicator, PHYRE DQN, external baselines). §7 supports paragraph one sentence; pixel consequence, novelty consequence and budget item shortened; W8, W9, W11 verbatim; macro F1 values kept. §8 middle sentence cut.
  Why: Main page-fit instruction + `local://briefs/pagefit.md` (§1 −12, §2 −5, §7 −5, abstract ≤175 words, §8 −1).
- Kept: W1, W18, W22, W23 verbatim; all labels; Main's `\end{revblock}` in `_related_work.tex`.
- Changed (second cut, ~3–4 lines): §1 carrier gloss shortened, "with their own predictors" and C3 margin wording trimmed; §7 pixel consequence sentence cut (W11 kept), Live and Budget items shortened, untested undertraining interpretation dropped, one-benchmark item shortened.
  Why: Main (Conclusion spilled 2 lines onto p. 11).

## 2026-11-08 · W2 page-fit pass: §6 (`iclr2026_conference_results.tex`)
- Changed: rerun sentence drops the offset contrast (grid, angle, power + App F.2 pointer kept); the three audit-head macro F1 values replaced by a pointer to §7 (which carries them); W6/W4 ablation pointer shortened to App D only; §6.4 tightened (frontier members incl. unselected J λ=10⁻¹ kept; C-F(15) dominance sentence now only via §6.3 C*−J and App D); `tab:res:qa` caption and Q2/Q5 answers shortened; captions of Tables 4–6 and Fig 4 cut to population + metric + interval + denominator (W13 verbatim; Tab 5 λ now pointed to §4; "nonlinear and memory work excluded" kept in §6.4 text only).
  Why: Main page-fit instruction + `local://briefs/pagefit.md` (§6 −12 lines).
- Kept: W2, W4, W5 (amended), W6, W7, W11, W19, W20 scope in §6.4, the §6.3 distinction (C* 0.783, F(15,cont) 0.762, unselected J λ=10⁻¹ 0.770, J_pg 0.761) and every negative control.
- Changed (follow-up, −2–3 lines): `tab:res:rollout` caption cut to population + metric + interval (dropped the "J rows use own schedule" clause; I-R16's "not the NovPhy J" kept as "their own models"); §6.2 parenthetical "(nine and three times those budgets)" removed and its content moved into the App D `tab:app:res:sharing` caption.
  Why: Main follow-up instruction (main text ~2 lines over).

## 2026-11-08 · W1 page-fit pass: §3–§5, Appendices A and C
- Changed: `tab:setup:timebase` merged into the App A time-base table (both labels on it; caption carries the encoder-on-N1 note); §5 keeps one sentence on the primary unit + the submitted mixed-unit fact + W10 verbatim, pointing to App A. `tab:setup:arms` rebuilt as a two-half, one-line-per-row key (12 entries in 6 rows; full key stays in App A `tab:app:arms`). §3: 50-step clause dropped (Fig 1 caption and §5 carry it), surrogate/context and tie-free sentences compressed. §4: per-call MAC sentence moved (black) to App C; second-round sentence and continuous-predictor sentence compressed.
  Why: Main page-fit brief (`local://briefs/pagefit.md`; m1 Q12/R3-1).
- Kept: W1, W2 (rule), W3, W8, W10, W19 core, W20 scope phrase; all labels.


## 2026-11-08 · W4 fix round 1: Abstract, §1 (+ Fig 1 caption), §2, §7, §8, App G, appendix statements
- Changed: abstract cut to ~193 words; "no-model" dropped from the control list (I-F04); W22 abstract sentence verbatim (I-F19); "single-request predictors"/"dedicated predictors given more updates" glosses (I-F05, I-F18); first motivation sentence cut (I-F06).
  Why: Main fix round 1; I-F04, I-F05, I-F06, I-F18, I-F19; contract W22.
- Changed: §1 ¶1 analogy two sentences shorter (merged time/model resolution sentence, blue). Fig 1 float sets colour inside the float; caption ≈ 6 lines with W23 verbatim. ¶2 prior-work list → one cited sentence (Neitz, Chatterjee–Khardon, H-JEPA), TAWM per-step claim removed, "symbolic modes" → "micro and macro levels", title clause added. ¶3 glosses (split "before any was recorded", 11,680 engine-executed launches, no-model prior defined); "substantially stronger than the submission's" cut. ¶4 + contribution list replaced by four claim bullets C1–C4 (W22 §1 verbatim in C3; both orderings in C2; prior contrasts and NN-px in C4) and one sentence on which contrasts were declared before evaluation.
  Why: I-F05, I-F06, I-F08, I-F09, I-F10, I-F15, I-F16, I-F17, I-F19; contract §4 float rule, W22, W23.
- Changed: §2 TAWM description corrected; Chatterjee–Khardon placed on the prediction side with reason; "In most of these models"; THICK moved to hierarchies; H-JEPA/HWM cited; tied-axis sentences cut (App G keeps them); Girdhar characterized fully, our result scoped "within distribution".
  Why: I-F06, I-F08, I-F11, I-F12; citation map.
- Changed: §7 W5 repeat, repeated non-claims (selector, compute, live, direct prediction) and submitted-model time-base sentence removed; no-model-prior item added to "does not support"; one consequence clause per limitation; budget item names per-horizon predictors. §8 first two sentences cut; closes with a test-against-references rule.
  Why: Main item 4; I-F04, I-F06, I-F18.
- Changed: App G: every [TODO: cite] replaced (chatterjee2025improving, zhang2026hjepa, zhang2026hierarchical with HWM described from its arXiv abstract, balestriero2025lejepa); SALT described from its ICML 2026 workshop abstract ("no public text" removed); "Evaluation zeitgeist" → "Evaluation platforms"; "set the current standard", "asks where … stops", "outside this success-rate comparison", "do appear individually" removed; "output-description mode" → "description level"; selector sentence names (Δ=15, cont) on the continuous predictor; LeWorldModel no longer said to use SIGReg (not verified).
  Why: I-F10, I-F13, I-F14, I-F19; Main item 2.
- Changed: Reproducibility Statement supplement sentences → W21 verbatim + Main's sentence listing the 1,000 sealed NovPhy levels with roles, per-arm live counts, typed live failures and the full AUC–MAC table (replaces "Complete NovPhy split membership is not included."); LLM section → new W17 verbatim.
  Why: I-F01 (Main decisions, incl. supplement update per `iclr2026_supplement/README.md`), I-F07; contract W17, W21.
- Deferred: none ([TODO] markers in W4 files: 0).
- Rejected: I-F02 (WONTFIX per Main); hedged causal sentence for the Δ=1/Δ=15 dissociation in §1 (optional per Main; cut for length); predicate names in the Fig 1 caption (carried by the revised artwork, caption length cap).

## 2026-11-08 · W1 fix round 1: §3–§5, Fig 2 (moved to App A), Fig 3 caption, Appendices A–C
- Changed: label/outcome usage corrected in §3–§4 and App B "Label and outcome usage" (encoder = engine labels on 12 N1 lineages, no outcomes; predictor = engine predicate labels + predictor-fit outcomes; prior; J_dec = policy-fit outcomes; development outcomes chose recipe, U, C*). Why: I-M01. Tab 1 caption/N1 row and App A glossary/table name encoder training on N1 frames. Why: I-M02. "except the submitted checkpoints, which keep the count cost" removed; "What changed" adds count→tie-free ranking cost; App B count-cost note corrected. Why: I-M03.
- Changed: Fig 2 (`fig:levels`) moved to App A with its submitted caption (black) + blue qualifier; §3 points to Fig 1(b) and App A. Why: Main (m1 R3-1), I-M04(i). §5 W11 caveats, live-guard and scored-once sentences replaced by pointers; Tab 2 widths; §4 training/policy trimmed; Fig 3 caption shortened. Why: I-M04, Main length target.
- Changed: `\ifshowrevisions\color{blue}\fi` inside every new table (Tabs 1–2, App A tables). Why: I-M05. App C stability counts (1/0/0 after update 500; second excursion burst). Why: I-M06. 0/240 vs 240/240 same-population comparison (§4, App C). Why: I-M07. J vs J_dec differences stated (§4, App C 0.734 vs 0.683; Tab 2 J row). Why: I-M08.
- Changed (renames, I-M09): task cost Ĵ → $\widehat{\mathrm{cost}}(a)$; pig presence/centre → $\mathrm{pres}_{\mathrm{pig}}$/$\mathbf{p}_{\mathrm{pig}}$; schedule length J → n; MAC ratio W_π/W_ref → ω(r); adapter U_α → A_α; L_LC → L_primary; App C DP/J_dec index t → k, c(t,r) → g_k(r); §5 defines t as carrier frames from the decision frame; App B unused h removed.
- Changed: plain-word names for base corpus / N1 / development states in main text, record names kept in App A (I-M10); Tab 2 sharing row, "separate predictor" type, pre-declared C*, non-circular caption; App A arm table adds J_pg, M, OL, J_diag, J_anti, J_monotone, H(Δ,α)/H* (I-M11); grid primary + descriptive others + no multiplicity adjustment in §5 (I-M12); "continuous object-state vector", "in carrier space" (I-M13); "linear" MACs, width 456 vs 384, App B "matched" → similar not equal (I-M14); live guard sentence → pointer (I-M15); App A floats [H], overfull row fixed, repository paths removed from App A (I-M16).
- Rejected: renaming the split "sealed evaluation split" (W8 fixes "sealed NovPhy split"; that one name is used throughout).


## 2026-11-08 · W3 fix round 2: Appendix F (`iclr2026_conference_appendix_development.tex`)
- Changed: F.1 J−OL paragraph: removed the "development plan's secondary rule … would meet the margin" sentence; replaced by "J−OL compares J with its own open-loop schedule, not with a fixed request, and does not bear on the withdrawn claim." Why: I14.
- Changed: `tab:app:dev:rerun` caption: guards fail under each cost (tie-free grid half-width > 0.02; count cost fails its own tie and precision guards, #104 findings.md:120,129,131) → every contrast precision-insufficient. Why: I14/Main.
- Changed: `tab:app:dev:s1` to \scriptsize, \tabcolsep 2pt, headers shortened (Mixed, Slots, Top-1; compound formats moved to caption); `tab:app:dev:rerun` \tabcolsep 2pt. Why: I15 (36.4pt / 6.4pt overfull).
- Changed: F.3 hardware sentence drops "under an exclusive lock" (not found in the #77 runner; #77 plan.json not in the checkout). Why: Main r2.
- Checked: every new/changed float carries `revblock` inside the float (contract §4 float rule); submitted Table 1 body stays black.
- Deferred: none. Rejected: none.

## 2026-11-08 · W2 fix round 1: §6, Appendix D, Appendix E
- Changed (§6): I-R01 scope "among world-model request rankers … block (vi) higher still" in §6.3 and `tab:res:qa` Q3; I-R02 one sentence on J (λ=10⁻¹, unselected; Δ=15 every step; 0.770; C*−J +0.0134 [+0.0065, +0.0202], #103 compute.json `sets.primary.grid.frontiers.auc_wall.verdicts."J-1e-01"`) and J_pg (policy-split λ=10⁻¹; 0.761; one fixed Δ=15 request per seed; −0.0011 [−0.0031, +0.0009], #101 findings); I-R03 U−C* and J−prior clause + C* marked pre-declared matched-recipe baseline in Tab 5; I-R10 amended W5 placed after the NovPhy sentence; I-R07, I-R08, I-R09, I-R11 ([t]-aligned halves + 1em gutter; `tab:res:qa`/`tab:res:rollout` to [!htb]), I-R12 (rerun 2 sentences → App F.2 pointer; decomposition values and wall-time → App D; W6 footnote → Tab 12 caption), I-R13 (qualification paragraph split in two, budgets named), I-R14, I-R15, I-R16, I-R26 (caption pointer); live guard sentence scoped (25 frames on 5 levels: 3 evaluation, 2 novelty-set; 5,106 includes the novelty set).
  Why: review-log r1 I-R01–I-R16, I-R26; Main fix-round decisions (W5 amended by Main; footnote move).
- Changed (App D): I-R17 F(5,cont)/F(5,micro) rows + angle sentence; I-R18; I-R19 (oracle half; 0.000); I-R20 seed counts; I-R02 J λ=10⁻¹/J_pg collapse sentence and "unselected" label in Tab 16; Tab 13 "decision-valued selector" and J_dec−J row notes λ/window differences (method-reviewer note a); pixel parser = same architecture trained per benchmark (note b); wall-time detail (0.242/0.014, 16.74; 0.138/0.085, 1.63) moved here; live guard scoped.
- Changed (App E): I-R04 (U 0.0229, C* 0.0334 in E.4; F1/F1-D inherit U's G-action); I-R06 rounding sentence; I-R21 licensing scope + "reference only" defined; I-R22 (two LeWM-IDM rows, #123 ablations/readouts, "Six" not run, scope to #123); I-R23 per-port sensitivity; I-R24 budget sentence in caption; I-R25 column widths with \newline intervals; I-R26 #141 MACs 22.8M/1.84M; all 11 `[TODO: cite]` replaced per `local://briefs/citation-map.md`.
- Deferred: I-R05 figure items (legend "selected λ", frontier line style, J_pg points) need the FigFrontier executor; caption fixed.
- Rejected: new J_pg row in Tab 5 (Main default: one sentence).

## 2026-11-08 · W5: Appendix H `sec:app:repro` (`iclr2026_conference_appendix_repro.tex`)
- Changed: H merged into one section "Reproduction Commands and Number-to-Artifact Audit" (blue title) with subsections Checkpoints, Validation commands and last observed status, Number-to-Artifact Audit (`sec:app:audit` kept as a subsection label). Labels `tab:app:checkpoints`, `tab:app:repro`, `tab:app:audit` kept.
  Why: brief W5; contract §8 row H.
- Changed: `tab:app:checkpoints` gains blue primary-stack rows (E99 encoder sha256 cdbe677a…; #112 hybrid and continuous predictors, 3 seeds each, retrain freeze b1cd71fe, hashes from `gate_b_handoff.json`; #101 and #114 policies; #106 pixel study); #74/#77/#79 rows kept black under a blue "Submitted checkpoints (Appendix F)" header; caption rewritten (blue).
  Why: brief W5 item 1.
- Changed: `tab:app:repro` is a 4-column longtable with a blue status column; 32 rows match the 2026-10-23 check (23 exit 0, 9 stopped, causes from results.json/#126); added rows #109 cohort, #104 campaign, #99, #100, #112, #108, #101, #114, #103, #102, #117 ×3, #121, #119, #115, #113, #120, #116, #105, #107, #110, #106, #78, #123, #125, #127 program (validation.json pass, 2026-10-30) and children #128–#141 with their own exit-0 dates; #104 rerun and #109 rescore under the submitted block. W12 verbatim plus a causes paragraph. CLEVRER "exits 0" sentence and the small-artifact "every root" sentence rescoped (blue).
  Why: brief W5 item 2; contract §3 W12; #126.
- Changed: `tab:app:audit` rewritten (blue) to 45 rows grouped by table/paragraph: protocol, split, model, recipe, time base, pixel protocol, statistics, every §6 table and paragraph group, App D/E groups, App F groups, the reproduction check. Retained submitted rows: `tab:res:rollout` row retargeted to `tab:app:dev:rollout` (values black); `fig:teaser` → `fig:interface`(a) (frames 0/80/120/553, 50 steps = 0.02 s) plus a `fig:levels` row (frame 554); `sec:exp:fixed`+`fig:res:grid` → `fig:app:dev:grid`; `sec:exp:selection` (J−r†) → S1 tables at 4 decimals; `sec:exp:selection` compute → `sec:app:dev:compute` ("80 against 189 calls" dropped: no longer printed); `sec:exp:qual` → `fig:app:dev:qual`; "Ranker architecture" → `sec:app:impl`. Synced with the W3 fix round (count-cost column, rerun J−M/T/D/OL, Physion long-horizon LOSO, trunk widths 456/384).
  Why: brief W5 items 3–4; stale refs `fig:teaser`, `fig:res:grid`, `sec:exp:fixed`, `sec:exp:selection` ×2, `sec:exp:qual`.
- Deferred: source conflict to Main/W1: App C "about 19 times as many windows" vs the record's "about 19x the v3 windows". Rounding conflict to W1/W2: CLEVRER/Physion centre error 16.83/45.88 px (App B) vs 16.8/45.9 (App D). `.local-artifacts/issue-77-n1-dynamics-v1/` absent from this checkout (submitted anchors kept: plan, shards). #78 and #141 command invocations inferred from runner modes.
- Rejected: listing every cell of large tables in the audit (frontier, inventories, live, pixel, external, S1): rows cite the field per column instead, so W2/W3 value moves do not desynchronize H; keeping "The external-baseline validator was verified on both cpu and cuda" unchanged (submitted sentence, now matched by the #78 row); a separate #87 repro row (data gallery, no validator recorded).

## 2026-11-08 · W3 fix round 1: Appendix F (`iclr2026_conference_appendix_development.tex`)
- Changed: F.2 `tab:app:dev:rerun` gains a count-cost J−F* column (offset −0.1152, angle −0.0874, power −0.0323, issue-104 findings.md:157,166,175); prose states the claim fails under both costs. Why: I1.
- Added: rerun J−OL +0.0401 [+0.0247, +0.0556] beside the development J−OL in F.1 (with the #96-secondary-rule reading stated); one F.2 sentence with rerun grid J−M/J−T/J−D (findings.md:49–52), all precision-insufficient. Why: I2.
- Changed: F opener scoped to "every submitted record"; LOSO tagged post-submission exploratory refit; compute correction tagged post-submission evaluation-split record. Why: I3.
- Changed: "frozen before any evaluation score existed" → "before any score of this rerun existed". Why: I4.
- Changed: guard wording: grid G6 (≤0.02) fails → every token insufficient; non-grid half-widths also exceed 0.02; rerun plan tightened the 0.05 bound; angle/power "descriptively lower, intervals exclude zero". Why: I5.
- Changed: Tab rollout caption causal sentence → "consistent with the short-window recipe", with Physion-Dominoes long-horizon LOSO values added (findings.md:294,314). Why: I6.
- Changed: compute withdrawal gets reason (iv) trunk widths 456 vs 384; wall-time ratios now directional (J/C−J 16.74, 1.63). Why: I7.
- Changed: restructure to F.1 development diagnostics, F.2 rerun, new F.3 `sec:app:dev:model` (facts, withdrawn compute, CLEVRER details, implementation); early pig-blind-parser sentence at the top of F. Why: I8.
- Changed: #74 compute-accounting paragraph replaced by a blue three-sentence version (RTX 3090, identical minibatches, linear MACs + wall time); dropped undefined "calibration lineages" clause. Why: I9.
- Changed: F* defined in `tab:app:dev:s1` caption; Notation defines "fully tied", cites Eq. `eq:taskcost` with decision-frame reference, maps "transferred fixed request" = F*. Why: I10.
- Changed: NovPhy rollout levels → "4 N1 levels held out from fitting but used as development diagnostics". Why: I11.
- Changed: F.2 states anchors are parsed by the submitted parser and that F* is re-selected on the evaluation split (offset F(1,macro) vs F(5,cont)). Why: I12.
- Changed: pointer to same-checkpoint controls after verbatim W16; "In this rerun, F* and F_hind are such choices." after W3. Why: I13.
- Deferred: none. Build not run by this unit.
- Rejected: none of I1–I13.

## 2026-11-08 · W4: Abstract, §1 (+ Fig 1 `fig:interface`), §2, §7, §8, Appendix G, appendix front summary and statements, main-tex front matter
- Changed: abstract rewritten whole (blue) around D3 + C1–C4: interface, sealed split numbers (H11), C1 with "at matched training updates per request" (H4), W4 abstract phrase, no-selector-advantage, C3 qualified (H6), direct prediction (H10); thesis verbatim as last sentence.
  Why: brief W4; m1 Q15/D3; contract §3 W4.
- Changed: §1 ¶1 kept black (submitted analogy), trimmed ("Imagining…" sentence and pfaff2021learning clause cut); W18 footnote on first sentence. New Fig 1 float `fig:interface` (fig_interface.pdf) after ¶1, blue caption (frames 0/80/120/553, 50 steps = 0.02 s, schematic panel c). ¶2–¶4 + contribution list new (blue): MDP defined, W1 verbatim, prior per-step temporal adaptation (neitz2018adaptive, Chatterjee–Khardon [TODO], nhu2025tawm), narrowed distinction (C4 last sentence), question + C4 numbers + ordering-AUC definition, findings incl. explicit non-claims; submitted "To our knowledge, no world model chooses…" removed.
  Why: brief W4; reviews MSwm W2 (novelty/citations), MMed presentation; m1 N1, N8, Q4.
- Changed: §2 rewritten (blue) into three paragraphs: temporal adaptation (Neitz, Chatterjee–Khardon, TAWM, keyframes, THICK/SUNTA, options); hierarchies and predicates with H15 tied-axis clause; physical reasoning and direct success prediction (PHYRE DQN, Girdhar 2020 per citation map, NovPhy, App E pointer).
  Why: brief W4; contract §7 row §2; m1 N8.
- Changed: §7 retitled "Discussion and Limitations" (blue); supports / does-not-support (every D5 non-claim) / limitations (time base, D7 audit F1, W11, W8 verbatim, W9 verbatim, live guard + single-bird protocol, budget as untested interpretation, one benchmark family). §8 body blue: thesis verbatim + what the interface enables.
  Why: brief W4; contract §2 D5, D7, §3 W5/W8/W9/W11.
- Changed: App G: positioning matrix `tab:app:matrix` and its "This paper: Yes×5" row removed and replaced by prose; universal "no method reports…" and "None of the four…" sentences removed; added Neitz/Chatterjee–Khardon paragraph, new subsection `sec:app:rel:hier` (H-JEPA, end-to-end H-JEPA, HWM, LeJEPA/SIGReg, LeWM, H15 and App E pointers), TAWM marked ICML 2025, SALT row reference replaced. Unchanged sentences kept black.
  Why: brief W4; m1 N8 (matrix cells not all substantiable: SALT row "?"; columns served the withdrawn boundary-curve argument).
- Changed: appendix.tex: blue "Summary of revisions" paragraph after `\appendix`; Reproducibility Statement rewritten with W12 verbatim and an accurate supplement description; Ethics Statement first sentence updated (NovPhy, CLEVRER CC0, Physion-Dominoes MIT); LLM section body = W17 verbatim. `\input` list unchanged.
  Why: brief W4; contract §8 row "—", §3 W12/W17, m1 N10.
- Removed: pre-abstract teaser float `fig:teaser` from `iclr2026_conference.tex`; file-map comment line updated accordingly. Title and preamble untouched.
  Why: brief W4 (Fig 1 replaces teaser; m1 D8).
- Deferred: [TODO: cite] markers for Chatterjee and Khardon 2025 (NeurIPS), Zhang et al. 2026 (arXiv 2610.06805), HWM, Balestriero and LeCun 2025 (LeJEPA). Bib entry `nhu2025tawm` still says "ICML 2025 Workshop track" (bib outside write set; scout). Stale refs to removed `fig:teaser`/`fig:res:grid` remain in `appendix_repro.tex` (W5).
- Rejected: keeping `tab:app:matrix` with fixed caption ref (cells unverifiable, N8); printing "1,000 levels" (not a contract §6 row); compute numbers in abstract (optional, length); renaming the LLM-use section title (submitted heading kept); printing A2/U as 0.9757/0.7444 in §1 (§6 and `tab:res:ranking` print 0.976/0.744; contract §5 same-rounding rule).

## 2026-11-08 · W1 length pass: §3–§5, Fig 2/Fig 3 captions, Appendices A and C
- Changed: §3–§5 compressed toward ≤3.3 pp. Moved verbatim (black) to App C new subsection "Formulation and model equations": eq:kernel with its sentence and the PHYRE sentence, semi-MDP sentence, eq:carrier, eq:code/input/film/residual with the head sentence, eq:loss with its definitions, eq:distill. eq:surrogate and eq:objective merged into one align. Encoder/predictor/training/policy/variants paragraphs condensed (counts, stability, MACs, hyperparameters already in App C). §5 shortened: exposure, inventory, pixel membership and fidelity, guard and live details left to App B. Arm-key table cut to 12 one-line rows; full definitions moved into App A `tab:app:arms`. Time-base table compacted (full version in App A). Fig 2 and Fig 3 captions shortened (Fig 2 caption now wholly blue).
  Why: Main length directive (main text p1–15 against the 10-page limit; §3+§4+§5 ≤ 3.3 pp).
- Kept: W1, W2, W3, W8, W10, W19 core, W20 scope phrase; all externally referenced equation labels.
- Deferred: none. Rejected: shrinking figures or table fonts below \footnotesize.


## 2026-11-08 · W2 length pass: §6 (`iclr2026_conference_results.tex`), Appendix D
- Changed: §6 cut from ≈5.5 pp to an estimated ≈3.3–3.5 pp: one evidence paragraph per subsection (6.3 adds one qualification paragraph); J−J_cont W2 sentences kept verbatim, three qualifications merged into one sentence; macro ≤2 sentences; submitted-checkpoint rerun in 2 sentences; 6.4 one paragraph; 6.5 one live paragraph + W7. `tab:res:ranking` split into two side-by-side halves (blocks i–iii | iv–vi), short row names with caption footnotes; all floats \footnotesize, tabcolsep 3pt; captions shortened to population/metric/interval/denominator (W13 kept verbatim).
  Why: Main instruction (length pass, build at 15 pp main text).
- Changed: text uses table 3-decimal values for U (0.744), A2 (0.976), A1 (0.995); δ-level contrasts keep recorded 4 decimals.
  Why: Main instruction (consistent U rounding).
- Moved to App D: same-checkpoint qualification detail (122.7M clamp, endpoint pair, incident, D_15 gap) into D.5; MAC decomposition and per-call MACs into D.10. U−C*, oracle-feedback numbers, pixel downstream readouts, direct-predictor readout values and NN-px hedge now only in App D/E.
- Rejected: dropping W4/W5/W6/W11/W7/W13 content or fonts below \footnotesize (Main instruction).

## 2026-11-08 · W2: §6 Results (`iclr2026_conference_results.tex`), Appendix D (`..._appendix_results.tex`), Appendix E (`..._appendix_external.tex`)
- Changed: §6 rewritten whole (all blue): orientation + `tab:res:qa`; 6.1 `sec:res:rollout` (`tab:res:rollout` two blocks, W4, W5, W11); 6.2 `sec:res:sharing` (W6 + budget footnote); 6.3 `sec:res:ranking` (`tab:res:ranking` blocks i–vi; C*−J, J_dec−F_hind, U−C*; J−J_cont with W2 and three qualifications; narrow controls; D_15; macro audit; tied axes; references; submitted-checkpoint rerun incl. angle/power + withdrawal; external pointer); 6.4 `sec:res:compute` (`fig:res:frontier` = `fig_frontier`, W13 caption, signed decomposition, 0.784, frontier members, wall 16.74/1.63, withdrawal, W20 scope); 6.5 `sec:res:live` (`tab:res:live`, W19 guard/351/351, W7 with H10 numbers).
  Why: unit brief W2-results + paper-contract §3, §6 H3–H10/H14/H15/H18/H20, §7 row §6, §9.
- Removed from §6: submitted Table 1, `fig:res:grid`, `fig:res:qual` and labels `sec:exp:granularity/fixed/selection/qual`; Appendix F (W3) receives the submitted material.
  Why: brief (submitted §6.2–6.4 leave the main text); withdrawn claims R2/R4.
- Added: Appendix D (`sec:app:results`, 12 subsections `sec:app:res:*`, tables `tab:app:res:{inventories,fixed,sharing,controls,scaling,frontier,live,pixel}`) and Appendix E (`sec:app:external`, `tab:app:ext`, subsections `sec:app:ext:{scored,notrun,agents,guard,verdict}`; W14, W15 verbatim; program verdict "not licensed").
  Why: contract §8 rows D–E.
- Deferred: visible `[TODO: cite …]` markers in App E for Phy-Q, SlotFormer, γ-models, DIAMOND, HWM, DDLP, Diffusion Forcing, DeepPHY, Datalab, Eagle's Wing, PigShooter (citation scout per brief). Stale `\ref`s to removed labels remain in `appendix_repro.tex` (`sec:exp:fixed/selection/qual`, `tab:res:rollout` row text) and `appendix_related.tex` (`fig:res:grid`) — other units' files, reported to Main.
- Rejected: printing block (vi) readout-on-U AUC as 0.977 (brief) — 3-decimal rounding of the artifact mean 0.97649 is 0.976; printing U+head parameter count (not in readable artifact); persistence/CV rows (never run; W11 sentence instead); a sentence quantifying horizon-vs-description AUC spread in App D (would require computing new differences).

## 2026-11-08 · W1: §3 Problem Formulation, §4 Method, §5 Experimental Setup, Fig 2/Fig 3 captions, Appendices A (notation), B (protocol), C (method)
- Changed: §3 adds W1 (MDP framing), N4 (same carrier; description-feedback mode; not a hierarchy, compression or compute saving), macro names steady-state/structure-unstable; schedule cost split into per-step fixed-context surrogate `eq:surrogate` + additive `eq:objective` with explicit "not the recursive rollout cost"; count cost `eq:select` replaced in §3 by tie-free cost `eq:taskcost` (scoring_harness.py:47-51,118-120), count cost moved to App B; blanket "enters neither training nor ranking" replaced by W3 plus the four declared outcome-supervision sites; r† sentence replaced by inventory/mixed-level definitions and arm-key pointer.
  Why: contract §3 W1/W3, §6 H1/H12/H13, m1 N1/N3/N4, brief W1.
- Changed: Fig 2 caption gains "illustrates targets, not evidence that predicted predicates stay accurate" (blue). Fig 3 float now includes `fig_model_overview_rev` with a new blue caption (decision-frame-only encoding, recursive feedback, heads only when α≠cont, outcome supervision, live retry-from-start; "never in closed loop" removed).
  Why: m1 Q5; Main code-check message (predicate names, head input, runtime carrier).
- Changed: §4 rewritten to the primary stack: repaired encoder (spatial_slot_parser.py:30-62; #99 plan `encoder`), carrier motion/elapsed (deployment_temporal.py:694,716-719), LC objective `eq:lc` (decision_dynamics.py:201-311; stable_decision_dynamics.py:27-78; run_decision_relevant_dynamics.py:481-496; #112-v3 plan `recipe.training`, `data`), aggregation rule (request_policy.py:111-175,333-357), J_cont/J_dec/T_α/D_Δ/J_as-cont (#101, #114 plans), continuous predictor and C−J, N3 "what changed" paragraph. Removed: base-corpus stride sentence (moved to N3 paragraph/App A), submitted compute-comparison sentence, teaser (b1) pointer, "without a curriculum".
  Why: contract D6, H2, H13, H21; brief W1.
- Changed: §5 replaced (blue): sealed split + W8, inventories, pixel study, time base + W10 + `tab:setup:timebase`, `tab:setup:arms`, metrics, statistics + W2 + scored-once exception, compute scope (W20), live retry-from-start (W19).
  Why: contract §7 row §5, R2-4, R2-6, R2-10, Q14.
- Changed: App A written (notation, glossary, time base with sources, arm key with record names); App B keeps true submitted-era protocol text black, adds split/exposure, inventories, scoring path, count cost, statistics and guard lists (#108/#101/#114/#102/#104), live and pixel protocols; App C written (encoder, predictors, per-call MACs, LC recipe and stability, policies, exact DP and aggregation, labeller scale, compute scope, pixel-study models).
  Why: contract §8 rows A–C.
- Deleted from App B (submitted protocol only; for Appendix F coverage): "Evaluation states are held-out lineages … sorted identity order"; the N1 13-candidate inventory sentence (a00 + a01–a12); the #77 seed bases (764,000,000+ …, N2 reserved ranges); the "13 candidate actions per state, one reference drag plus 12 release-angle actions" clause.
- Deferred: none ([TODO] markers: 0).
- Rejected: printing #104 rerun numbers or compute/wall numbers in §3–§5 (owned by W2/W3); "parameter-matched" for the continuous predictor (contract §5 "similar size").


## 2026-11-08 · W3: Appendix F `sec:app:development` (`iclr2026_conference_appendix_development.tex`)
- Changed: App F restructured to contract §8 row F: new blue section title "Submitted-Checkpoint Evidence"; W16 opener verbatim; checkpoint definition and one-time notation map (r† = F*, "hybrid adaptive" = submitted J, horizon-only baseline = C-J, count vs tie-free cost).
  Why: brief W3 / contract §3 W16, §5, §8.
- Added: F.1 `sec:app:dev:insample` with subsubsections: submitted Table 1 as `tab:app:dev:rollout` (relabelled blue caption: CLEVRER scenes 10000–10011 in fit and eval, Physion 12 within-release trials, NovPhy 4 levels, short-window Δ=1 divergence) + #117 LOSO paragraph (exploratory); `fig:app:dev:grid` (fig_results_grid, caption 14/9/7/8 mixed levels, AUC 0.40–0.63, descriptive); S1 tables `tab:app:dev:s1` and `tab:app:dev:s1arms` (18 arms × 4 inventories, every cell re-checked against issue-96 summary.json); `fig:app:dev:qual` with the submitted §6.4 paragraph (black where still true; "as in the aggregate" and "same behaviour holds" replaced in blue) + J−OL qualifier; withdrawn compute comparison printed only with the issue-103 correction and reasons; submitted model facts (2,409 = 2,400 + 9 lineages, mixed time base, no elapsed-channel ablation, 0/859 parser).
  Why: brief W3 items (a)–(f); contract H12, H18, H21.
- Changed: CLEVRER Replication Details and Implementation Details demoted to F.1 subsubsections (labels `sec:app:clevrer`, `sec:app:impl` kept); blue in-sample sentence after membership; "Parameter counts are matched" replaced by similar-size sentence; compute-comparison pointer now to `sec:app:dev:compute` and marked withdrawn; duplicated last sentence of the #74 compute-accounting paragraph removed.
  Why: contract §5 ("never parameter-matched"), §2 non-claims; Prep-unit note on the duplicate.
- Added: F.2 `sec:app:dev:rerun` with `tab:app:dev:rerun` (311/76/78/47, J−F* [CI], half-widths, AUC J/F*), W3 outcome-usage text, grid J/F*/F_hind, count-cost secondary (395/933 tied), conclusion that the adaptivity claim and the +0.004 offset gain do not replicate.
  Why: brief W3; contract H18.
- Deferred: none. Build not run by this unit (no shell).
- Rejected: recolouring the heatmap/qualitative artwork labels "r†"/"policy" (artwork not recoloured, mapping given in captions); printing #117 1.5× sharing ratios (not needed for the ordering statement).

## 2026-11-08 · Prep unit: preamble, comment cleanup, appendix split (all `iclr2026_conference*.tex`)
- Added: revision-markup macros in the preamble after `\usepackage[table]{xcolor}` (`\ifshowrevisions`/`\showrevisionstrue`, `\rev{}`, `revblock`), with a one-line comment pointing to paper-contract §4.
- Added: appendix files A–H (`iclr2026_conference_appendix_{notation,protocol,method,results,external,development,related,repro}.tex`); A, C, D, E hold only a header comment (no PDF output).
- Changed: `iclr2026_conference_appendix.tex` now holds `\appendix`, `\input`s A–H in order, then Reproducibility Statement, Ethics Statement, and the LLM-use section (moved from the front of the appendix to the end).
- Changed (verbatim moves): Protocol Mechanics → B, except its compute-accounting paragraph (#74 fits, parameter counts) → F as the last paragraph of Implementation Details; `tab:app:checkpoints` → H (after the Reproduction Commands heading); CLEVRER Replication Details → F; Extended Related Work + `tab:app:matrix` → G; Implementation Details → F; Reproduction Commands + Number-to-Artifact Audit → H. Labels unchanged.
- Changed: main-tex file-map header replaced by a short current map pointing to `manuscript/paper-contract.md`; longtable comment retagged.
- Removed: every `% ==== BRIEF … /BRIEF` block and the BRIEF-LEGEND block; stale steering comments (title/abstract-frozen and owner directives, "matches" directive, PROSE PASS, VOCAB-PASS, TAU-AD markers incl. three trailing `%` comments, #91-WP4/WP5 history, page-limit history, storyline headers in intro/related/method, float-drift directive, BRIEF-only code-trace key in method.tex).
- Removed: Estimand Taxonomy Details section with `fig:app:estimands` and `tab:app:estimands` (contract §8/§11); `iclr2026_conference_related_work_table.tex` deleted from the working tree (never input; preserved in the as-submitted snapshot).
  Why: paper-contract §4, §8, §11 step 1 (Prep unit).
- Proof: `grep '==== BRIEF'` returns 0 matches in every tex; no `\ref` to a removed label remains; main-tex comment-stripped text equals `local://submitted_stripped/iclr2026_conference.tex` except the four macro lines; other files differ from their snapshots only by the moves/deletion above and blank lines. Build NOT run by this unit (no shell tool available); page count and undefined-reference status pending Main's `latexmk`.
- Deferred: none created. Noted for W3: the moved compute-accounting paragraph repeats its last sentence verbatim (present in the submission; left untouched). Appendix letters shift (Protocol prints as A) until A has content.
- Rejected: placing the compute paragraph as a loose paragraph at the top of F (it would print under the preceding appendix section); kept the method.tex `\varnothing` fallback comment, the xurl/longtable package comments, ICLR template comments, and the camera-ready acknowledgments comment as still-true technical comments.

## 2026-09-26 · §6.4 qualitative analysis added
- Added: §6.4 "How the policy chooses requests" with new `fig:res:qual` (task agent QualFigure: `figures/build_qualitative_requests.py` → `fig_qualitative_requests.{pdf,png}`). (a) the policy's request schedule for each of the 12 candidate launches of the running example (angle set, level of Figs. 1–2, seed 20260908, chosen a priori; a11 not in the frozen inventory); (b) each launch's rank by predicted cost under the policy, r† = (1, macro) and hindsight-best (5, macro). Text: eight distinct schedules among 12 candidates; the successful a6 ranks 6/12 (policy), 12/12 (r†), 1/12 (hindsight), and the policy chooses a5, which fails; across 2,634 rollouts in 177 level–seed cells, schedules differ between candidates in 77% of cells; Δ=5 share rises 2% → 16% and macro 59% → 83% from the first 25 frames to after frame 100, while single-frame steps stay the majority. Verdicts joined through the runner's own `load_sources()` / `build_universe()`.
  Why: owner asked for qualitative analysis in the main body; page overrun accepted by owner (main text now ends two lines into p. 10).
- Proof: `latexmk -pdf` clean, no undefined/multiply-defined references.

## 2026-09-26 · Results, Discussion and Conclusion rewritten for the granularity storyline; ranking-failure record moved to the appendix
- Added: §6 Results (`iclr2026_conference_results.tex`) in three subsections that follow the Intro findings: 6.1 granularity changes rollout error (new `tab:res:rollout`: position MSE at 225 / 120 frames for baseline Δ=1/5/15, hybrid Δ=1/5/15 and hybrid (15, micro/macro); three to five orders of magnitude; the shared hybrid stays within 1.5× of the dedicated baseline at Δ=15); 6.2 no fixed request suits every launch set (new `fig:res:grid`, `figures/build_results_grid.py` → `fig_results_grid.*`, task agent ResultsGrid: 3×3 AUC heatmaps per launch set with hindsight-best and cross-fitted r† marked; per-level spread 0.44–0.51; r† never equals the set's best and trails it by 0.03–0.13); 6.3 choosing the request per prediction (J−r† on all four sets with descriptive intervals, interval excludes 0 on the angle sweep; J vs hindsight-best −0.069/−0.087/−0.042/+0.001; policy uses 6–7 of 9 requests per seed; compute: regret 0.350 vs 0.367, 118.7M vs 356.0M MAC, 1.0 s vs 2.0 s, 80 vs 189 calls — "matches" kept per owner, no further qualifying facts). Sources: issue-77 diagnostic / issue-79 / issue-84 `summary.json`, issue-96 `comparisons.csv` + `frequencies.json`, issue-74 `readiness.json`, issue-92 trace.
- Changed: §7 Discussion + §8 Conclusion (`iclr2026_conference_discussion.tex`) rewritten: granularity as a decision; relation to hierarchies (axes vary separately: macro best everywhere, horizon 1 vs 5) and to adaptive stepping; one limitations-and-next-steps paragraph (decision-only; 7–14 levels per set and wide intervals; macro only on NovPhy; annotation carriers on CLEVRER/Physion; frozen supervised encoder, no EMA; hand-defined levels; error+compute objective). Conclusion restates formulation, model, policy, testbed and findings.
- Moved: former main-text §6 (heuristic anatomy, `iclr2026_conference_experiments.tex`), §7 (ranking-failure measurement, `iclr2026_conference_synthesis.tex`, with Figure `fig:selection-ordering` and its tables/figures) and the former Discussion/Conclusion (now `iclr2026_conference_legacy_discussion.tex`, labels `sec:app:legacydisc`, `sec:app:legacyconc`) unchanged except for demoted headings into new Appendix `sec:app:record` ("The Ranking-Failure Record"). Main-tex file map and label ownership updated.
- Proof: `latexmk -pdf` clean, no undefined/multiply-defined references; main text ends mid p. 9 (within the 9-page cap); References start p. 10; 63 pp total.

## 2026-09-26 · Owner-finalized abstract and teaser; figure moves; Introduction, §2 Related Work and §3 Problem Formulation rewritten (chat-reviewed)
- Changed: abstract (`iclr2026_conference_abstract.tex`) replaced by the owner's final text; the owner declared it frozen.
- Changed: Fig. 1 teaser (`fig:teaser`, `figures/build_scene_figures.py::teaser_granularity`, `fig_teaser_granularity.*`) redesigned over several rounds: (a) real a06 launch photos at carrier frames 0/80/120/553.38, schematic MDP track, and the ACTUAL imagined request trace of candidate 6 (#92 hybrid-adaptive-e600 record); (b1) recursive position MSE Δ=1 vs Δ=15 on NovPhy, CLEVRER, Physion-Dominoes (three seeds); (b2) rank of fixed (5, macro) among nine requests (1/1/9/4) and J−F* ordering-AUC differences (#96). Caption defines ordering AUC and states decision-only. Owner declared it frozen.
- Moved: `fig:selection-ordering` from the end of the Introduction to the synthesis section, beside its only reference.
- Changed: model overview (`iclr2026_conference_fig_model.tex`, `figures/build_model_figure.py`) redrawn: frozen E → carrier → F_θ with three FiLM blocks, request lane with π_ψ (242→128→9, Δ>κ mask), predicate lane entering once at the input, training strip; caption leads with "Model overview." and states no EMA target encoder and decision-only evaluation.
- Changed: §1 (`iclr2026_conference_introduction.tex`) rewritten in four paragraphs modeled on arXiv 2409.15915: planning with world models and physical-simulation practice (adaptive step size, level of detail) → the gap (MDP one step, one representation; one axis at a time; H-JEPA coupling) → "To close this gap" with three key innovations (per-prediction request within the MDP; one world model for all requests; DP-distilled request policy, verified against `cnn_hybrid.py` / `matched_dynamics.py` / `run_issue_77_n1_train.py`) → findings with figure pointers. No separate contributions list; qualifiers deferred to §5 by owner directive (§5 still owes: 3/4 J−F* intervals include 0, J trails F_hindsight, compute on development states, decision-only).
- Changed: §2 (`iclr2026_conference_related_work.tex`) compressed from >1 page to ~3/8 page in two paragraphs (adapting granularity incl. the H-JEPA / SALT / Wang 2023 / MuSix defense and the scoped gap; action-sparse physics and NovPhy as unused testbed). Merged from an orchestrator draft and an independent task-agent draft. `tab:rel:gap` removed from the main text (file `iclr2026_conference_related_work_table.tex` left unreferenced); Kim 2023 citation dropped from §2 per owner.
- Changed: §3 (`iclr2026_conference_method.tex`) restructured into four paragraphs that formalize the three innovations: action-sparse MDP + new Δ-step kernel eq:kernel; requests on one shared 236-d latent with a concrete example per description level; request-selection problem with the schedule cost eq:objective (moved from §4 eq:dpcost); predictions to decisions with Ĵ, y and the new ordering-AUC definition eq:auc and the cross-fit fixed-request reference. Removed from the main text: the Δ² lookahead (eq:lookahead) and §3.1 Estimands (kept in Appendix `sec:app:estimands`). §4 Request policy now references eq:objective; launch-set details moved to the Evaluation Protocol (angle sweep corrected to 9–13 candidates per state per #96 `plan.json`).
- Added: `fig:levels` (task agent LevelsFigure): `figures/build_description_levels.py` → `fig_description_levels.{pdf,png}`, placed after §3 ¶2, which now gives a concrete example per description level. Source: original N1 capture issue-77-n1-001-a06, agent frame 165, engine-projected centres/velocities and the Boolean contact/supports/macro training targets from `issue-77-n1-dynamics-v1/shards/lineage-n1-001.pt` (window 25; settled contrast frame 554, window 27). Caption states these are engine labels used as training targets, not model outputs.
  Why: owner asked for an example of cont/micro/macro. The frozen-model readout variant was rejected: on the teaser's a06 oracle capture the frozen parser (issue-70 parser v6) misses bird/pig at frame 80 and mislocalizes the block, and the seed-20260908 hybrid micro head never exceeds 0.059 gated contact/supports over all 555 frames; macro (frame 80) steady 0.188 / unstable 0.041. Only N1 lineage 009 yields gated micro edges >0.5, still with mislocalized centres. Encoding path validated (a06 frame 0 carrier equals the #96 sealed anchor, max |Δ| = 0). Open: this readout weakness must be disclosed in §5.
- Changed: §4 Method (`iclr2026_conference_method.tex`) restructured into six paragraphs with no repetition of §1/§3: overview + the sense of "JEPA-style" (old ¶1+¶7 merged); encoder and carrier (feature list moved out); request-conditioned predictor (now cites perez2018film and peebles2023scalable; states the joint-code rationale and the per-call MAC ratio: micro +25%, macro +2%, re-verified with `linear_macs`); training (round-robin first, eq:loss compressed to one line, stride-unit disclosure kept); request policy (solution of eq:objective only: DP, distillation, DAgger-style round 2); "Horizon-only baseline" (renamed; states it is the Fig. 1(b1) model and the §1 horizon-only adaptive baseline, and that the compute comparison uses the issue-74 training on the 2,400 base lineages / 600 policy lineages). Symbol clashes fixed: compute is MAC(r) in eq:objective, c(r) is the request code; loss rollout ẑ_j and M steps. Hyperparameters moved to new Appendix `sec:app:impl` (Implementation Details).
  Why: owner-approved plan; owner asked that §4 not repeat earlier sections. §4 shrinks from ≈1.6 to ≈1.25 pages including Figure 3.
- Changed: §5 rewritten and retitled "Experimental Setup" (label `sec:mth:protocol` kept), ≈0.63 page, five run-in paragraphs aligned with the three Intro findings: benchmarks and models (CLEVRER / Physion-Dominoes carriers from released annotations, macro only on NovPhy, three seeds, per-benchmark reporting); rollout error (four held-out NovPhy levels × 13 launches to 225 frames; 48 contexts in 12 scenes to 120 frames); ranking candidate launches (four launch sets over 15 levels, mixed-level rule leaving 7–14 levels, endpoint 225, cross-fitted r† and hindsight-best fixed request); compute (200 development states × 12 candidates, regret on the engine-realized end-of-window count cost, linear MACs per candidate-state); statistics and scope (frozen plans, clustered percentile bootstrap, descriptive, decision-only). Sources: issue-77 diagnostic, issue-79, issue-84, issue-96 and issue-74 plan/summary files.
- Moved: the former §5 paragraphs (program and strata, ceiling / top-1 / chance with eq:ceiling, registration status, closed-loop scope) verbatim to new Appendix subsection `sec:app:legacyeval` ("Evaluation contract of the ranking-failure record"); five references that meant the old contract now point there or to `sec:app:protocol`.
  Why: owner directive — no membership/type details in the main text, shorter section; keep "matches" in the compute finding and add no further qualifying facts later.
- Removed: synthesis paragraph "τ_ad … four executed readings, none informative" (contradicted the abstract; owner directive).
- Changed: stale references fixed (`sec:mth:estimands` → `sec:app:estimands` in the appendix; `$h{=}…$` → `$\Delta{=}…$` in synthesis/discussion; appendix now defines h = Δ once); §4 back-reference to "the controller" in §1 removed.
- Added bib (pending/verified as marked in the bib): hairer1993solving, mirtich2000timewarp, carlson1997simulation, pfaff2021learning, yang2024learning, bruce2024genie, cen2025worldvla, talvitie2014model, venkatraman2015improving, janner2019when, perez2018film, peebles2023scalable. Open: `hairer1993solving` currently holds the DBLP record of a SIAM Review book review (renders "Alexander, 1990"); a corrected book entry was proposed to the owner.
- Proof: `latexmk -pdf` clean, no undefined references/citations, 58 pp.

## 2026-09-26 · Abstract rewrite v2 (owner feedback: background → gap/significance → what we did → findings → elevated ending)
- Changed: `iclr2026_conference_abstract.tex` rewritten by paper-writer AbstractWriter2 following `skill://academic-writing`. Structure: 5 moves. (1) World models fix Δ/α at design time; action-sparse persistent-effect cascades cannot be corrected once started. (2) Phase-granularity hypothesis; prior work varies one axis, ties axes to hierarchy, selects among models, or changes action timing; gap (to our knowledge: joint, per decision, one shared latent); significance: mid-cascade, what to predict is the only lever. (3) K1 engine-step MDP + per-decision request r; K2 granularity-conditioned JEPA-style predictor; K3 DP-distilled request policy; NovPhy testbed. (4) G1a (0.6346 / 0.4006, post-hoc descriptive); selection not yet shown to help (S1, S2, no closed loop). (5) "We argue" ending for embodied agents (throwing/pouring robots), framed as testable claim. 250 words (texcount rule, math excluded). BRIEF block replaced; cuts listed there. Removed from abstract: E1 proxy number (0.1299), E2 chance-level clause, the S2 interval, and the micro/macro feedback detail.
  Why: owner judged v1 stiff and missing the problem-modeling and granularity-as-variable novelty in the background.
- Proof: `latexmk -pdf` clean, 59 pp, no undefined references; abstract fits p. 1.

## 2026-09-26 · Granularity reframing, round 2: abstract, §3 Problem Formulation, §4 Method (+ equations), §5 Evaluation Protocol, Fig. 1 teaser redesign, model-overview figure, G1 correction
- Corrected (data): the round-1 §1 claim "the nine fixed requests span 0.5098 in ordering AUC" was wrong. In #96, 0.5098 is the member-mean of per-cell max−min AUC across the nine requests (`grid_spread`), not a span of pooled per-request AUCs (grid range 0.4006–0.5126). Outline §2 G1 split into G1a (best fixed request changes with the candidate set: F(5,macro) best on offset 0.5944 and angle 0.6346, worst on drag grid 0.4006 where F(1,macro) 0.5126 leads) and G1b (per-cell spread 0.5098, no null reference, kept as a secondary with caveat). §1 ¶5 and the abstract rewritten to G1a; the teaser panel (b) plots the 36 pooled per-request AUCs.
- Changed: abstract (`iclr2026_conference_abstract.tex`) fully rewritten (244 words): fixed granularity → action-sparse cascade hypothesis → request r=(Δ,α) as per-decision variable → JEPA-style mechanism → NovPhy/pre-registered → G1a → negative/insufficient selection results → proxy/ceiling lesson. Old ranking-failure BRIEF deleted.
- Changed: §1 (`iclr2026_conference_introduction.tex`) ¶4 gains the "on N1 captures, one frame spans 50 engine steps" scope; ¶5 G1 sentences corrected as above.
- Changed: §3–§5 (`iclr2026_conference_method.tex`) rewritten: §3 Problem Formulation with eq:mdp, eq:requests, eq:policy/eq:rollout, eq:select, eq:lookahead; §3.1 Estimands with eq:tau-tr/ex/ad; §4 Method (new label sec:mth:model) with eq:carrier/code/input/film/residual, training objective eq:loss (local + recursive + λsym=0.1 symbolic + λbd=0.01 carrier-bound; round-robin requests; stride-unit disclosure: 2,400 of 2,409 lineages are stride-one base corpus where Δ counts engine steps, N1 shards stride 50), DP-distilled request policy eq:dpcost/eq:dp/eq:distill (step cost Δ·(1/d)‖·‖² + λ·MAC_π(r)/MAC(15,cont); MAC_π includes the controller; d=236; λ=1e-4; DAgger-style round 2), capacity-matched baseline (1,862,396 vs 1,837,690); §5 Evaluation Protocol compressed (kept sec:mth:protocol) with eq:ceiling. Mechanical: "Equation~\eqref" → "\ref" (8 occurrences; the ICLR style file renders \eqref as "equation~…").
- Added: `iclr2026_conference_fig_model.tex` + `figures/fig_model_overview.{pdf,svg,png}` + `figures/build_model_figure.py` (task agent ModelFig): model overview (frozen E → carrier → 3 FiLM blocks with joint c(r), predicate feedback adapters, 9-way masked policy with one schematic chosen cell, candidate-launch input, training-signals strip); 5.5 × 3.0 in, 7 pt minimum text.
- Changed: Fig. 1 (`fig:teaser` in `iclr2026_conference.tex`) redesigned by task agent TeaserFig: (a) genuine cascade strip start/flight/first contact/settled for the successful a06 launch (frames from the #87 gallery `shot.webm`, uniformly cropped) with the request-space question; no phase-specific request claimed; (b) four 3×3 heatmaps of pooled fixed-request ordering AUC from #96 comparisons.csv (36 values verified against the CSV), best cell outlined. `teaser_scene()` kept intact. `tab:app:audit` fig:teaser row updated (audit table only).
- Proof: `latexmk -pdf` clean; 0 undefined refs/citations; 0 "equation equation" remnants; 59 pp total; References start p. 14. Rendered pages/figures inspected (model overview, teaser, table page).
- Open (next round): main text ≈ 4 pages over the 9-page cap — §3–§4 written at full precision (≈3.5 pp) and §2 ≈1.3 pp; a compression pass is planned together with the §6/§7 and §7-(old §6) restructure. Abstract phrasing to revisit on owner review: "puts the policy's AUC 0.070 above the fixed request best on other sets". Appendix §F SALT footnote still outdated; CONTEXT.md/research_evidence.md docs sync still pending; kim2023imagine/lecun2022path still pending human bib validation.

## 2026-09-26 · Granularity reframing, round 1: outline v3.0, §1 Introduction, §2 Related Work, `tab:rel:gap`, 14 new bib entries
- Changed: `writing_outline_boundary_paper.md` → v3.0 granularity outline (file name kept; supersedes the v2.x ranking-failure thesis and the CONTEXT.md thesis bullet). Thesis: granularity (Δ, α) as a per-decision variable of one shared latent world model; contributions K1 formulation, K2 architecture, K3 controller training, K4 pre-registered NovPhy evaluation with negative/insufficient results.
  Why: owner directive (2026-09-26). Target reader is JEPA/world-model researchers. Novelty = problem modeling + granularity variable + architecture/training; ceiling material demoted to evaluation support.
- Added: `novelty_verification.md` (related-work-scout). Claim A (problem modeling) holds with a qualifier, B (NovPhy unused as a world-model testbed; 3 S2 / 2 OpenAlex citations) holds, C (joint per-decision granularity over one shared latent) holds with a qualifier. Closest threats: SALT, MuSix, Wang et al. 2023; H-JEPA pre-empted by name.
- Changed: §1 (`iclr2026_conference_introduction.tex`) rewritten. ¶1 fixed granularity in world models; ¶2 H-JEPA coupling + single-axis relaxations + scoped novelty sentence; ¶3 action-sparse persistent-effect setting, PHYRE contextual-bandit framing, engine-step MDP with per-decision request; ¶4 mechanism (JEPA-style latent predictor over a frozen object-slot encoder, joint FiLM code, predicate feedback, DP-distilled controller); ¶5 findings (G1 fixed-pair AUC spread 0.5098 / F* worst 0.4006, post hoc, ordering only; G2 carrier stability; S1 not supported; S2 insufficient; controller never closed-loop; proxy 0.1299; fixed-request rankers at or below chance in point estimate). New contribution list K1–K4. `fig:selection-ordering` float left byte-identical, to be relocated in round 3.
  Why: new storyline; paper-writer IntroWriter.
- Changed: §2 (`iclr2026_conference_related_work.tex`) rewritten as a gap defense. Seven run-in paragraphs (fixed contract; temporal only; description only; both axes coupled or fixed incl. H-JEPA and SALT; per-state selection over another variable incl. MuSix/AdaJEPA; action-sparse physical settings incl. PHYRE, I-PHYRE, AIBIRDS, options/action persistence/delayed MDPs, NovPhy usage; evaluation precedents) plus a scoped "to our knowledge" closing sentence.
  Why: owner priority on the related-work defense; paper-writer RelWriter. ~707 words, over the ~550 budget; cut candidates are the setting paragraph and the coupled-axes list.
- Added: `iclr2026_conference_related_work_table.tex` → `tab:rel:gap` (task agent GapTable), 13 rows × 5 properties, caption above; `\usepackage{array}` added to the main preamble.
- Added: 14 bib entries appended under a "NEW CITATIONS … PENDING HUMAN VALIDATION" separator at the end of `iclr2026_conference.bib`, each with a `% verified:` URL: lecun2022path, metelli2020control, katsikopoulos2003mdp, walsh2009learning, saxena2021clockwork, neitz2018adaptive, jayaraman2019tap, pertsch2020keyin, kim2019vta, gregor2019tdvae, hafner2022director, mattes2024hieros, renz2015aibirds, kim2023imagine. For human check: kim2023imagine may have a NeurIPS 2023 version (kept as arXiv); lecun2022path verified via a Wayback capture of OpenReview.
- Proof: `latexmk -pdf` clean, 0 undefined citations/references, 57 pp; Table 1 on p. 4 renders inside the margins.
- Open: main text now runs through p. 11 (References p. 12), about 2 pages over the cap. Cuts come in rounds 2–3 (§3 duplicates the intro setting; §5/§6 restructure). The abstract, §3 and Fig. 1 are stale against the new §1 until later rounds. Appendix §F SALT "no public text" footnote is outdated (public abstract at icml.cc/virtual/2026/83751).

## 2026-09-25 · Fig. 1 pure-continuous teaser text: caption + BRIEF (`iclr2026_conference.tex` l.145–171), `tab:app:audit` Fig. 1 row (appendix l.2502), CONTEXT.md l.12, outline l.18 and l.56
- Changed: `fig:teaser` caption, hybrid (Δ, α) interface lead → plain opening "A world model's chosen launch can fail even when a winning launch exists." then panels (a) one real state, 12 admissible launches; (b) separately trained continuous-only model, one fixed request (Δ=1 model frame per step, continuous-latent description), picks a09 by lowest predicted cost; (c) separate first-shot engine runs, a09 misses, a06 the only success of 12 removes the target; frames are cropped engine screenshots; one example, not a rate, no test of request choice or of horizon/description effect.
  Why: user instruction naming this caption (override of the logged two-tier architecture caption); figure redesigned to three panels (a/b/c) with a pure-continuous ranker.
- Changed: BRIEF FIG 0 [中]/[EN] → pure-continuous ranker, three panels, no 3×3 request grid or hybrid interface, not a second-shot retry, no causal Δ/description claim.
  Why: same.
- Changed: `tab:app:audit` Fig. 1 row → pure-continuous ranker (`continuous-fixed-h1`, seed 20260908, Δ=1, α=continuous), engine seed 764100001 added; artifact `cnn_hybrid.py` → `matched_dynamics.py` (continuous-only request set), `summary.json` (engine seed) added. 12 slots, a09 miss, only a06 success, one-state boundary kept.
  Why: Main brief; old row described the hybrid nine-pair interface. Checked `slot_join.csv` row (12 candidates, successful ordinal 6, chosen 9, top1_hit False), `build_scene_figures.py:_evidence` (seed 764100001), `run_issue_77_n1_train.py` imports `ContinuousDynamics` from `matched_dynamics.py`.
- Changed: CONTEXT.md l.12, outline l.18 and l.56 Fig. 1 descriptions → the same three-panel pure-continuous figure; old 5.5 × 2.32 in size dropped (final size not yet known).
  Why: Main brief.
- Deferred: none. Flagged, not edited: `iclr2026_conference.tex` l.34 file map still says "conceptual/gameplay teaser"; CONTEXT.md l.11 "the two real-frame figures".
- Rejected: naming the state's training exposure role in the caption (true per slot_join `exposure_role=training`, but outside the word budget and not a figure claim).
- Figure provenance (Main-reported, not rerun here): source script `iclr2026/figures/build_scene_figures.py:teaser_scene`. Helper `figures/fig_teaser_helper.png` served only as a layout guide. The real frames `figures/source/{001_common_decision,001_a09_final,001_a06_final}.png` are checked against the #87 manifest by `_evidence()`. All three use the same crop x70–470, y110–300 and the same original-pixel target-area inset x306–349, y232–277. No AI imagery is used as outcome evidence.
- Outputs (Main-reported): `fig_teaser_scene.{svg,pdf,png}` at 5.5 × 2.72 in. SVG text is editable and the PDF embeds TrueType fonts.
- Proof (Main-reported, not rerun here): a smoke run of `teaser_scene()` in the `novphy` env with env.sh passed. latexmk built the PDF (55 pp), and page 1 was rendered at 2100 px and inspected. Both label overlaps were fixed.
- Open risk (pre-existing, not fixed): main text runs to p. 10 and References start on p. 11, over the ICLR 9-page main-text cap. No broad prose cut was made.

## 2026-09-25 · Fig. 1 architecture-teaser docs sync: CONTEXT.md l.12, writing_outline_boundary_paper.md l.18 and l.56 (no TeX, figure or code edits)
- Changed: `CONTEXT.md` Fig. 1 bullet, three real frames → two tiers. (a) hybrid fixed-request architecture: observed state → frozen CNN slot parser → shared 236-D continuous carrier (+ launch a_i) → endpoint target/block cost → fixed-pair ranker; Δ = per-step span 1/5/15, α = description/readout that conditions the same carrier. (b) one real N1 start forked into alternative first shots: 12 admissible slots, a06 only engine success (target removed; circled object = ball on ramp), a09 continuous/Δ=1 ranker pick misses (target intact). Boundary line: no deployed or validated adaptive pair policy, no causal α/horizon claim, no rate.
  Why: user-requested architecture teaser now built (Main brief); the old bullet described the superseded three-frame figure. Facts checked against the caption and BRIEF (`iclr2026_conference.tex` l.145–166), `tab:app:audit` Fig. 1 row (appendix l.2502) and `build_scene_figures.py:teaser_scene` (boxes, arrows, labels).
- Changed: outline §2 row 1 figure parenthetical and §4 Fig. 1 line → the same description and boundaries.
  Why: same.
- Figure history (Main-reported, not rerun): Oracle reviewed the first drawn draft, verdict FAIL: unsupported α→cost arrow, nine-pair action-cost overclaim, retry-like three-frame strip, invisible a06 success, overlapping 6 pt labels, colour ambiguity. Task agent's fixes: α → carrier only, fixed-pair action ranking, fork layout, blue "a06 · TARGET REMOVED" header with a neutral circle on the actual ball, rust a09 selected miss, 7 pt minimum at final width.
- Related prose fix: the false "α affects only the scored output" claim (method l.71, discussion l.141) was corrected in the previous entry (§3 ¶2, §7 ¶5). Not re-edited here.
- Historical vs current: the three-frame teaser (figure-redesign docs sync entry, incl. its "(three real frames; no schematic)" audit-row line) and the candidate-arc fan schematic are both superseded. `tab:app:audit` now reads "(one fixed-request interface and forked first-shot engine outcomes)". Old entries left intact.
- Proof (Main-reported, not rerun here): `build_scene_figures.py` regenerates pdf/svg/png at 5.5 × 2.32 in with unchanged gallery raster frames; SVG labels are editable vector text; font check passed (builder sets DejaVu Sans, l.116). latexmk succeeded, 55 pp; page 1 inspected at scale-to 1800.
- Open risk (unchanged, not fixed): main text runs through p. 10 and References start p. 11, against the 9-page cap.
- Deferred: none. Flagged for Main, not edited: `iclr2026_conference.tex` l.34 file-map comment still says "conceptual/gameplay teaser" (TeX, out of scope). `CONTEXT.md` l.11 "the two real-frame figures" and l.17 Superseded list do not name the three-frame teaser (l.12 carries it).
- Rejected: rewriting older entries; widening the sync beyond the three assigned lines.

## 2026-09-25 · §3 ¶2 (method l66→71 + BRIEF), §7 ¶5 description-mode (discussion l139→141 + BRIEF)
- Changed: §3 ¶2 "All nine pairs are readouts over one continuous latent carrier. Switching to a symbolic readout … changes only which output is scored, not what is rolled forward." → hybrid's nine pairs condition one continuous carrier (pair code on each transition; micro/macro symbolic features fed back into it), so a symbolic mode changes the rollout and the scored output; state stays continuous latent, no separate symbolic dynamics; pure-continuous models accept only α = continuous. Δ/Δ² values untouched.
  Why: Oracle teaser review S2 + code evidence `world_model/training/cnn_hybrid.py:75-86` (pair-code FiLM, micro/macro adapters), `matched_dynamics.py:12,40` (CONTINUOUS_PAIRS only). Old sentence contradicted the code and the Fig. 1 architecture arrows.
- Changed: §7 ¶5 "narrow causal lever by design … changes only which readout is scored" → mode axis bounded by the architecture: α conditions the carrier (pair code + symbolic feedback) and picks the scored readout, state stays continuous latent, no separate symbolic dynamics engine, so a null on the tested mode-axis contrasts does not show granularity irrelevant.
  Why: same cause (Oracle S2, cnn_hybrid.py:75-86).
- Changed: both BRIEFs ([中]/[EN]) plus the §7 VOCAB-PASS note, changed claim lines only: keep-sentence now "all nine pairs condition one continuous latent carrier" + mechanism; new boundary "never say α changes only the scored output".
  Why: comments must describe the prose.
- Deferred: none. Not in scope, reported to Main: appendix l.1421 BRIEF ("request picks the scored readout and how many steps") and method §3.1 τ_ex gloss ("executed readout inside one hybrid checkpoint") are incomplete but do not claim "only".
- Rejected: keeping "readouts over one carrier" verbatim as the old BRIEF required; "readout" framing implies α does not enter the transition. ~+56 words net against the open page-cap risk; no compression done outside assigned paragraphs.

## 2026-09-25 · figure-redesign docs sync: CONTEXT.md, writing_outline_boundary_paper.md, research_evidence.md (no TeX, figure or code edits)
- Changed: `CONTEXT.md` Fig. 1 bullet → current figure guide: Fig. 1 `fig_teaser_scene` (three real frames, state `issue-77-n1-001-a00`: shared start, ranker choice a09 miss, a06 engine removal); Fig. 2 `fig_hero_selection_ordering` (four N1 inventories: ceiling, top-1 minus matched chance, AUC; launch-power support confound); Fig. 3 `fig_hero_ordinal_masses` = `fig:syn:bands` (post hoc, N1-only; breadth pool also succeeds at 10–12); Fig. 4 `fig_qualitative_case` close-up audit; appendix list. Headline bullet gains the corrected held-out cell-unit AUC.
  Why: Main brief; the old bullet described the pre-redesign hero, a breadth-pool/few-shot row the figure no longer plots, and `fig:app:bands`.
- Changed: `research_evidence.md` WP3 row: unlabelled "cross-split AUC 0.2732 [0.2182, 0.3239]" → zero-shot 0.5015 [0.4751, 0.5275], few-shot 0.5073 [0.4645, 0.5481], 216 of 612 scored cells each, 6 state clusters; old values kept as SUPERSEDED.
  Why: new evidence, #98 tie-scoring correction (`issue-92-cross-pool-audit-v1/findings.md` l.57–62); verified in `summary.json` `cross_split.per_condition.{zero-shot,few-shot}.auc.cell_unit` (means 0.50147 / 0.50733, DESCRIPTIVE, seed 7201). Separate conditions, not N1.
- Changed: outline §2 rows 1 and App, §4 list → current body figures; `fig:app:bands` marked SUPERSEDED.
  Why: Main brief.
- Superseded (old entries left intact): the candidate-arc fan teaser in the layout entry's "Figure 1 rebuilt as a selling-point schematic" line and the "conceptual/real-frame teaser" of the Visual revision entry. The current teaser draws no trajectories. "0.2039 / 0.2732 not quoted anywhere new" still holds; both are pre-#98.
- Reproducibility (Main-reported, not rerun here): `build_figures.py` (calls `build_scene_figures.teaser_scene/qualitative_case`) writes pdf/svg/png for 8 stems: fig_teaser_scene, fig_hero_selection_ordering, fig_hero_ordinal_masses, fig_qualitative_case, estimand_taxonomy, fig_error_growth, fig_novelty_boundary, fig_ceiling_map. Scene builder asserts the slot join and a06/a09 verdicts and byte-compares `figures/source/` frames with the #87 gallery. latexmk succeeded, 55 pp; teaser, body figures and appendix PNGs visually inspected.
- Open risk (unresolved, not fixed): main text runs through printed p. 10 and References start p. 11, against the 9-page ICLR 2026 main-text cap in the TeX preamble. No prose or figure squeezed. Not submission-ready on pagination.
- Resolved (this session; TeX by FixFigureAuditRows, verified in appendix l.2502–2504, parent compile succeeded): `tab:app:audit` Fig. 1 row reads "(three real frames; no schematic)". The Fig. 2 row lists the four N1 inventories plus post-hoc speed-only 0.6889 and no breadth-pool or held-out values. The Fig. 3 row is N1-only, and its caption pointer names the breadth counterexample at ordinals 10--12.
- Rejected: editing old changelog entries to strike the fan teaser (history is append-only).
- Addendum (post-sync correction, Main brief): `CONTEXT.md` thesis bullet "three candidate inventories … AUC 0.4180 to 0.6178" → four N1 inventories, AUC 0.4180–0.7441, launch-power 0/72 support confound, post-hoc speed-only 0.6889. False "every inventory … saturated speed" → first three inventories saturated; launch power varies pull radius and speed, four of five radii inside the clamp.
  Why: Main brief; stale since #94/#95. Figure asserts 0.4180…0.7441 and 0.6889 (`build_figures.py`); radii per appendix `sec:app:launchpower`.
- Addendum: `issue91_exec_digest.md` l.14 single cross-split AUC 0.2732 [0.2182, 0.3239] → per-condition zero-shot 0.5015 [0.4751, 0.5275], few-shot 0.5073 [0.4645, 0.5481], 216 scored cells each (#98); old value kept, marked SUPERSEDED. l.15 breadth successful ordinals "0–6" → "0–6 and 10–12".
  Why: Main brief overrides the earlier "digest owner-controlled" note (living digest). Verified in `issue-92-cross-pool-audit-v1/summary.json` `cross_split.per_condition.*.auc.cell_unit` and `bounded.ordinal_histogram` {0:1, 1:1, 2:3, 3:3, 4:5, 5:3, 6:4, 10:3, 11:3, 12:6}.
- Addendum: `CONTEXT.md` Headline bullet gains the missing launch-power row: ceiling 8/15 [0.2667, 0.8000], top-1 0/72 vs 0.0521 (support-confounded), AUC_m 0.7441 [0.6923, 0.8026] `not_supported_by_this_experiment`. Other values untouched. Blockers bullet: the continuation is now "a launch-power sweep fully inside the 18.49 px clamp", distinguishing it from #94, which has four of five radii inside.
  Why: Main brief. Values per appendix l.1751, l.1876, l.2528.
- Addendum: re-checked against the appendix TeX (parent-reported 55-pp compile, not rerun here). l.13 status stays Resolved (appendix l.2502–2504 match the current Fig. 1–3). `CONTEXT.md` Headline: "outside the rankers' N1 launch-power pool" → "outside the frozen rankers' N1 training pool", the appendix term (l.1876, l.2554); 17 px lies inside the #71 range, so "N1" stays. Blockers bullet already matched; unchanged.
  Why: Main brief.

## 2026-09-25 · Route-C spine pass: §1 list, §3 opening, §6 τ_ad paragraph, §7 pointer (§8 skipped)
- Changed: §1 contribution list gains the specification item as item 1 (spec wording verbatim); other three items untouched. Paid in §1 by: figure-division sentence compressed; ¶2 Δ/α value sets dropped (now enumerated once, in item 1); ¶3 restated "carried by one seed on an invalidated proxy" tail cut; ¶4 opening merged ("An oracle pass replaces the proxy: …"), em-dash pair → parentheses, lookahead-arm and "frozen model ranker" wording tightened, duplicate oracle-detail pointer cut; ¶5 "when measured by running the launch in the physics engine" and the C24-prediction clause cut (clause kept verbatim in item 3); ¶6 saturation scope folded into the scope sentence.
  Why: owner-approved Route-C spec item 1 (`/tmp/routeC_spec.md`).
- Changed: §3 opens with the spec's specification sentence; paid by replacing the nested ranker gloss with "(defined below)", dropping "(the same states are reused across candidate lists)", and defining the ceiling as "A starting state is a ceiling state when at least one launch in its candidate list succeeds at first shot."
  Why: spec item 2.
- Changed: §6 gains one paragraph (after `fig:syn:bands`, i.e. after the table-reading paragraph) carrying the four τ_ad readings with the spec's numbers, the selector-exercised facts and the independence statement, pointing to `sec:app:tauad`. Paid by: ¶1 headline preview reduced to an AUC gloss and duplicate oracle-detail pointer cut; ¶2 zero-width gloss shortened and duplicate launch-power support sentence cut (kept in ¶3); ¶3 em-dash pair → parentheses, degeneracy reason stated once (¶2), restatement-choice sentence tightened; ¶6 gallery clause cut (caption carries it); mechanism clause drops "the frozen rankers'".
  Why: spec item 3.
- Changed: §7 #96 sentence shortened to a pointer to §6 (guard values and interval no longer repeated; guard-failure reason and sign-reversal boundary kept).
  Why: spec item 5.
- Word delta (whitespace tokens, comments excluded): §1 −5, §3 −2, §6 −2, §7 −23, §8 0; total −32.
- Deferred: none created. Flagged for Main: §6 last sentence still says "registry C1--C28" (registry now C1--C31); §8 ¶2 "The protocol is the second contribution" now clashes with the four-item §1 list; §1 ¶3 "Both contrasts were scored against … the replay-cost proxy" vs the confirmatory contrast being teacher-forced endpoint carrier MSE; §6 cites the deployed controllers against 0.335 (`hybrid_continuous_h15`, readiness.json) while Appendix development quotes the `continuous_h5` comparator at 0.329997.
- Rejected: §8 specification clause (no §8 ¶1 redundancy can pay for it without cutting a qualifier, and no ¶1 sentence hosts it without implying the ceilings are defined over requests); spec's "count-cost proxy" wording in the §6 paragraph (body term is the replay-cost proxy, so "the invalidated proxy" is used); cutting the §8 "under a training-range support confound" duplicate (qualifier adjacent to 0/72).

## 2026-09-25 · #96 τ_ad within-checkpoint contrast: four executed readings (prose half)
- Result recorded: ticket #96, a pre-registered within-checkpoint, decision-only selection-schedule contrast on the engine-verdict channel (root `.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/`, freeze `dd806eec`, publish `0e7f921c`, `--validate` exit 0), returned `readiness_or_precision_insufficient`: θ_grid 0.069983 [-0.005697, 0.143651], half-width 0.0747, margin 0.02; guards G4 (ties 17/42 = 0.4048 vs 0.20 cap) and G6 (half-width 0.0747 vs 0.05 cap) failed; J − F(1, macro) −0.0420 [−0.1295, 0.0475] reverses sign.
- Changed: every "three executed readings / contrasts / read three times" site (all `% TAU-AD-FIX:` sites plus their BRIEF comments) now states four executed readings, none informative, the fourth being #96; each changed site carries a `% TAU-AD-96:` marker. `method.tex` estimand sentence keeps the pointer form and now points to `sec:app:tauad`; the intro sentence changes the count only (word-neutral).
  Why: owner-approved #96 spec §A (`/tmp/taud96_spec.md`).
- Changed: new Appendix H subsection `\subsection{A pre-registered within-checkpoint selection-schedule contrast}` (`sec:app:tauad`) after the launch-power subsection: question, design, primary, guards, grid secondaries, per-inventory values, AUC implementation, boundary. Table body left as `% TAU-AD-96-TABLE-PLACEHOLDER` for the table writer.
  Why: spec §B (prose half).
- Changed: Discussion falsifiability paragraph gains the single spec §D sentence (+58 body words; intro and method edits word-neutral).
  Why: spec §D.
- Changed: `CONTEXT.md` bridge parenthetical upgraded to the four-reading form. `writing_outline_boundary_paper.md` has no count-bearing site (row C7 unchanged).
- Unchanged claims: no adaptive-granularity benefit, no coupling measurement and no optimality claim added; "no closed-loop, non-degenerate τ_ad contrast exists" kept; 0.2039 / 0.2732 not quoted anywhere new; title, abstract, `tab:syn:crosspool`, `tab:syn:oracle`, figures untouched.
- Files touched: `iclr2026_conference_introduction.tex`, `iclr2026_conference_method.tex`, `iclr2026_conference_discussion.tex`, `iclr2026_conference_appendix.tex`, `iclr2026_conference.tex` (comments only), `CONTEXT.md`, `changelog.md`.
- NOT edited: `issue91_exec_digest.md` (remains owner-controlled).
- Deferred: registry rows C29–C31 and caption range, `tab:app:tauad` body, `tab:app:repro` #96 row, `tab:app:audit` #96 rows, Reproducibility Statement runner count (table writer, spec §C).
- Rejected: the canonical long four-reading form in the intro and method body sentences (would break the +60-word body budget; the appendix sites carry it).

## 2026-09-25 · τ_ad correction: blanket "unmeasured" replaced by three executed uninformative readings
- Changed: 8 prose sites (A0–A7) in introduction, method, and appendix replaced the false blanket claim that the adaptation effect τ_ad is "unmeasured" / "isolated by no reported contrast" with the corrected fact: three executed contrasts read τ_ad without an informative result.
  - #92 WP1b decision-only adaptive arms: 0/36 vs best fixed arm 2/36, floored decision-only outcome. Artifact: `.local-artifacts/issue-92-lookahead-control-v1/comparisons.csv` + `records/`.
  - #15 v2 sealed confirmatory: candidate `joint_pair` vs comparator `fixed_pair`, gain −5.2e-4 [−7.1, −3.6]e-4, disposition `not_supported_by_this_experiment`, n=6, teacher-forced against a single fixed pair. Artifact: `data/runtime_evidence/issue-15/cohort-v2-oracle-symbol-confirmatory-v2-summary.json` `budget_decisions[1]`.
  - #74 deployed controllers: per-seed regrets 0.329995/0.39/0.33 (mean 0.349998) vs `hybrid_continuous_h15` 0.335 at every seed, scored on the invalidated count-cost proxy; linear MACs 294.06/362.04/198.62 G vs 51.57 G (3.9–7.0×). Artifact: `.local-artifacts/issue-74-matched-dynamics-v1/readiness.json`.
  What stays true: no closed-loop, non-degenerate τ_ad contrast exists (the adaptive controller never enters the engine-truth closed-loop record; #57's selector requested one pair on 44/44 decisions; #78's adaptive port is byte-identical to its fixed arms).
- Changed: 7 BRIEF comment sites (B) in conference.tex, method.tex, and appendix.tex updated to match the corrected fact, in the bilingual style of the surrounding blocks.
- Changed: 3 plan documents (C): `writing_outline_boundary_paper.md` row C7 "(τ_ad unmeasured)" → "(τ_ad is not the contrast read in this row)"; `CONTEXT.md` bridge parenthetical updated; this changelog entry added.
- Files touched: `iclr2026_conference_introduction.tex`, `iclr2026_conference_method.tex`, `iclr2026_conference_appendix.tex`, `iclr2026_conference.tex`, `writing_outline_boundary_paper.md`, `CONTEXT.md`, `changelog.md`.
- NOT edited: `issue91_exec_digest.md` (owner-controlled; WP2 item 4 needs the owner's amendment).
  Why: owner-approved τ_ad correction spec (2026-09-25).
- Deferred: none.
- Rejected: none.

## 2026-09-25 · owner-requested BRIEF pass: bilingual intent notes on every section and paragraph
- Added: one `% ==== BRIEF <unit> ====` ... `% ==== /BRIEF ====` block directly above every section, subsection, prose paragraph and float caption in all nine iclr2026 tex files (194 blocks; appendix 140). Fields per block: 定位 (role in the reader's journey), 主张 (the sentence the unit must land), 证据 (the numbers that carry it), 手法/边界 (craft and forbidden readings), each mirrored in English as Role / Claim / Evidence / Craft and Boundaries.
- Voice: Fei-Fei Li (open on the field-level question, show why the measurement had to exist, close forward on what the field adopts) over Kaiming He (one bare declarative sentence, claim first, plain nouns, exact numbers). Legend and voice spec: the preamble of `iclr2026_conference.tex`. The notes are advisory; they never override the prose they sit above.
- Unchanged: no claim, number, status token, citation or prose character. Comment-stripped diff is identical for all ten tex files (pre-edit snapshots in /tmp/brief-audit); `latexmk -pdf` exit 0, 50 pp, 0 "??", 0 undefined references, PDF byte-identical to the pre-pass build.
- Method: the 140 appendix blocks were drafted by eight parallel subagents against one shared spec plus exemplars, returned as JSON, and inserted by a single writer in document order; a tag-to-`\label` audit maps every float brief to its own float, and a coverage audit finds no section, subsection, paragraph or float without a block.
- Selling-point layer (owner feedback 2026-09-25): the original pitch in `docs/research_proposal.md` --- granularity is one decision over two
  axes (temporal horizon x description mode), imported into learned world models from the multiscale/kinetic tradition --- is now carried as a
  plain-language 卖点 / Pitch line in the TITLE, ABSTRACT, §1 ¶1 and ¶2, the §1 contribution list, §3 HEAD and ¶2, §3.1, §6 HEAD, §7 ¶5,
  §8 HEAD and the appendix estimand-taxonomy, estimand-figure and extended-related-work briefs. Each states the same bridge: the title
  promises a decision, the body measures that decision; τ_ad is isolated by no reported contrast and the joint controller never ran closed-loop,
  so no coupling or adaptive-granularity gain may be read out of this manuscript. Comment-only change (content-identical modulo comments, 50 pp).
- Oracle consult (2026-09-25, owner request) on the accessible two-axis selling point. Verdict: the owner is right that the idea is missing from the prose (the words
  "horizon" and "description" never appear in the abstract; the result sections carry none of the axis vocabulary), but the two-axis idea must NOT be sold as this
  paper's result. Correction applied to the BRIEF layer: the earlier bridge "the body delivers a measurement of that decision" was FALSE --- the measured decision is
  which candidate to pick (Fig. 1, §6), while the (Δ, α) pick is τ_ad and is isolated by no reported contrast. The corrected bridge, now in the legend and every Pitch
- Oracle language options landed in the briefs as candidates, each marked ALLOWED or FORBIDDEN: abstract S1--S2 (A1, A2), §1 ¶1 motivation sentences, §1 ¶2 replacement
  sentences, contribution-1 bold lead, §8 opening (O1) and alternative close (O2). Budget note: the abstract is 398 words against a 390 cap, so additions must be paid
- Readability layer (owner directive 2026-09-25, comment-only): every front-matter unit now carries a 易读性 / Readability field naming the assumed reader ---
  an embodied-agent researcher who does not know NovPhy, the ranker, the replay-cost proxy or this paper's statistics conventions. Coverage: the ABSTRACT,
  all eleven §1 units, a freeze note on the TITLE and a gloss rule on the teaser caption. Each field lists the terms that must be defined at first use
  (ceiling, inventory, top-1, chance, proxy; AUC only after glossing it as "how well it ranks, not what it picks"), the plain-word substitutions
- Harness note for future bulk comment writes: a comment block that begins with a bare `% ====...` fence and continues line by line was rewritten in this workspace into `__omp_magic(...)` lines, which broke the build; texted fences (`% ==== BRIEF <unit> ====`) and shorter blocks were unaffected. After any bulk comment write, check `grep -c __omp_magic *.tex` is 0 and re-run latexmk.

- Changed: the Reproducibility and Ethics statements moved out of the main body to the closing appendix sections (Appendix K/L, appendix.tex:974/:980; cut from discussion.tex). Main text now ends with §8 Conclusion on p. 10, references start p. 11. Why: owner directive; the style_example ICLR paper keeps no such statements in the main body
- Changed: Tables 1 and 2 (`tab:syn:oracle`, `tab:syn:crosspool`) merged into one side-by-side `[H]` float (labels, rows and caption facts unchanged; header "AUC (cell-unit)" -> "AUC", cell-unit basis stays in the caption). Saves ~0.25 page. Why: owner directive (side-by-side tables)
- Changed: Figure 3 (`fig:syn:bands`) compacted in `build_figures.py:hero()` (7.0x6.6 -> 7.0x5.5, hspace 1.05 -> 0.75) with caption ~200 -> ~140 words (interval brackets remain in Table 16), and its declaration moved up beside its referencing paragraph so the `[t]` float lands at the top of p. 7 inside §6; Figure 4 (`fig:syn:qualitative`) pinned `[H]`. Previously Fig. 3 float-paged after the Conclusion and Fig. 4 drifted to p. 11, so Figure 4 preceded Figure 3. Why: owner directive (layout)
- Changed: Figure 1 (`fig:teaser`) rebuilt as a selling-point schematic in `build_figures.py:teaser_scene()`: a candidate-arc fan coloured by the measured N1 ordinal structure (successes at ordinals 2-6, ranker choices at 7-12) with a06 hitting the target and a09 falling short, a verdict strip ("solvable (ceiling 7/14), yet ranker top-1 0/108 <= chance"), and the authentic a06/a09 frames with verdict badges. New caption (main file). Figure 4 refined in `qualitative_case()`: tighter FRAME_CROP (40,120,580,330), object highlights (slingshot/ball/target), corner verdict badges drawn as vector marks, branch arrows; caption notes the annotation semantics. Shared helpers `_frame_ax/_badge/_circ/_lab`; all artifact assertions retained (slot join, manifest cells, engine seed). Why: owner directive (teaser must be a schematic illustration of the storyline selling point; qualitative figure not refined)
- Gate: build_figures.py clean (8 PDFs, assertions pass); latexmk -pdf exit 0, 50 pp, 0 "??", 0 undefined references, 0 "Float too large"; the 47.8 pt merged-table overfull fixed (tabcolsep 4 pt + shorter header); remaining overfulls are pre-existing appendix rows. Main body pp. 1-10 (was 1-11). Why: verification

## 2026-09-25 · owner-requested prose pass: expert-style rewrite of every tex (no number/claim changes)
- Gate: numeric-token diff main body + appendix = no value removed from the paper (only de-duplicated instances; every removed token still present at its primary site or table); hedge/citation scan unchanged (post hoc, in point estimate, pre-registered, degenerate all retained; 16/16 citations); abstract word-neutral (token sentence +4 offset by -7 trims elsewhere, stays within the Main +8 exception); latexmk -pdf exit 0, 50 pp, 0 "??", 0 undefined; pages 1-10 render-checked (Fig 2 still p. 4, Fig 3 p. 7, Fig 4 p. 8, Conclusion ends p. 10). Why: verification

## Visual revision · conceptual teaser, authentic case and measurement tables
- Packaged unchanged N1 member-001 gallery frames (`figures/source/`), retained the six quantitative PDFs and exported an editable conceptual/real-frame teaser plus a three-frame matched qualitative case. The latter asserts the N1 slot join and executed a06/a09 engine-event outcomes; the ranker seed is not the engine seed. SVG labels are vector text, gameplay frames remain embedded raster.
- Moved the teaser before the abstract, kept selection-vs-ordering as a distinct §1 plot, and moved the qualified ordinal-bands plot into §6 beside the matched case. The single case illustrates an observed miss, not a population rate or mechanism.
- Replaced the dense body measurement table with four inventory rows and a separate two-condition cross-pool summary, retained every interval/status/control in `tab:app:oracle-detail`, and condensed the denominator table. Main tables remain within §6; `\clearpage` still prevents main floats from entering references. The text and captions retain the launch-power support-gap warning and post-hoc/pre-registered distinctions.

## 2026-09-24 · owner decisions: close-out (no further experiments)

- Rejected: printing the 5/27 and single-member (member 011) recounts (owner: pooled wording); Options B (protocol-led) and C (pre-registration narrative).

- Rejected: printing the Storyline recounts (offset continuous-fixed-h1 top-1 5/27; all 8 grid hits from one member) — owner chose pooled wording instead; Options B (protocol-led) and C (pre-registration narrative) as the frame.

- Rejected: narrowing the caption (Main preferred adding the numbers).

- Rejected: none.

- Rejected: none.

- Rejected: none.

- Changed: new numbered analysis #93, approved by the owner in session (release commit b8aba59; frozen plan commit 6239a04; artifact root .local-artifacts/issue-93-second-parameterization-v1/). Dispositions verbatim: C22 (Arm B drag grid) not_supported_by_this_experiment, member-clustered AUC 0.6178 [0.5126, 0.7139]; C23 (Arm A offset sweep) readiness_or_precision_insufficient, 0.5512 [0.4235, 0.6685]. Reading-matrix row 5 applied: descriptive appendix report, body scope sentences unchanged except a pointer. Why: #93 owner-approved analysis
- Changed: appendix.tex:615–663 new \subsection{Second candidate parameterization} (sec:app:secondparam) + tab:app:secondparam: inventories, states, rankers, estimands, frozen rule, dispositions, row 5 verbatim, WP0 drag-clamp finding (saturated speed 10; 234/234 retained launches 9.9992–9.9997), training support (grid out of release-1000 support in radius on the input side only), execution, per-arm table; every number carries a % comment into the #93 root (appendix.tex:619–628). Why: #93 brief item 1
- Changed: registry rows C22/C23 (appendix.tex:558–561), caption C1--C21 → C1--C23 (appendix.tex:518), row-to-family paragraph (appendix.tex:497), header comment (appendix.tex:6–9); synthesis.tex:115 C1--C23; method.tex:61 rows C16--C23. Why: #93 brief item 2
- Changed: audit row "Second parameterization (#93)" (appendix.tex:769); repro row (appendix.tex:713) and #93 in the closed-loop-runner paragraph with release commit b8aba59 (appendix.tex:719); Reproducibility Statement "nine runners" → "ten runners" with run_second_parameterization_probe listed, #93 root at b8aba59 (discussion.tex:84). Offset: artifact-type list after "4,187 paths" cut (kept in App. repro). Why: #93 brief items 3–4
- Changed: body pointers: method.tex:49 (one clause, offset "have no symbolic modules and" cut); introduction.tex:90 (offset: "The claim is narrow by construction." cut); discussion.tex:58 with both member-clustered AUCs (offset: "over which percentile intervals are wide," cut); abstract.tex:30 short pointer (fits: gate green). Each pointer states the drag-grid AUC is above chance. Mechanism sentence "selection fails at within-state action discrimination along the candidate sweep" unchanged at every site. Why: #93 brief items 5–6
- Changed: gate 41 pp (appendix +2), latexmk exit 0, 0 Float too large, 0 undefined, 0 rendered [TODO], body ends p. 9, Repro/Ethics p. 9, references p. 10. Figures unchanged. Why: page-limit gate
- Rejected: editing issue91_exec_digest.md or "along the candidate sweep" (owner did not amend the digest); pooling arms with each other or with N1; any state-difficulty or significance reading of C22.

- Rejected: none.

- Rejected: none.

- Rejected: a §1 sentence (body at the page limit).

- Rejected: none.

- Rejected: none.

- Rejected: none.

- Changed: Reproducibility Statement runner count "seven runners" -> "nine runners" with run_state_difficulty_auc and run_inverted_ranker_probe listed (discussion.tex:80). Why: factual count after the two post-hoc probes landed.

- Changed: csv LF line terminator fixed in the probe so `--validate` byte-compare passes. Why: validation
- Changed: gate green: 39 pp (appendix +1: audit + repro rows), body ends p. 9, Repro/Ethics p. 9, refs p. 10; 0 Float too large, 0 undefined refs, 0 rendered [TODO]. Why: page-limit gate
- Deferred: none.
- Rejected: none.

- Deferred: none.
- Rejected: none.

- Cuts (offsets): abstract "no success in 70 slots", "held out from it/adapted on it" → parentheticals; §1 variance-split sentence (kept in §6), "No single launch angle substitutes" sentence, 216-cell basis sentence folded into a parenthetical; §6 "No constant launch angle rescues…" sentence, "one lookahead arm selects 2 of 36 … counterexample" tail (in Table 1), bootstrap clause compressed; §8 "rather than treat aggregate prediction quality…" tail.

- Deferred: none
- Rejected: registry C-row edits (no row quoted 0.3--7)

- Changed: gate fallout routed to Thesis, discussion.tex trimmed ~80 words so Repro+Ethics fit p. 9. Why: gate
- Deferred: none.
- Rejected: none.

- Changed: tab:app:audit rows Rollout depth (new), Lookahead control (frozen-margin result + h15 means), Cross-split scope (216 of 612, 6-cluster bootstrap). Why: audit rule
- Deferred: #82/#85 tokens in the experiments.tex:34 construct-validity paragraph (no draft sent).
- Rejected: none recorded.

- Changed: discussion trimmed ~80 words for the gate. Why: Main gate
- Rejected: none recorded.

- Rejected: none.

- Changed: §4 pre-registration scoped + verbatim restatement post-hoc sentence, and C18 now notes that the band/supports were post-outcome. Why: I3a
- Rejected: promoting fig:app:ceiling to the body (not page-neutral).

- Rejected: none recorded.

- Deferred: §4 matching disclosure, §3 model paragraph, appendix:411 sentence, fig:teaser panel (a) rebuild, #83/#84 sentences placement → Evidence batch.
- Rejected: none.

- WP2 factual corrections (all 8): h=15 "null on N1, exactly zero on original families"; cost superlative corrected to continuous_adaptive 2.0147 s + arm disclosed in walls table; regime-separation D negative only at e=120 (qualifier added); Table 6 relabelled exposure/few-shot contrast (τ_ad unmeasured); withheld h=15 intervals printed; `data/issue-76-closeout/` provenance path fixed; TAWM degeneracy (0.3998, identical Top1/Top3) disclosed; per-seed decompositions (0, 0, +0.2662 / +0.2905, −0.0215, +0.0141) printed wherever the aggregates appear, abstract included. Extra correction beyond the ticket: #78 cohort is 21 states (4 N1 + 17 N2), table columns relabelled accordingly.
- WP3: registry rows C14–C21 (pre-applied by #92 WP4/WP5) verified against the digest; C2/C8 demoted; C3 corrected.
- WP4 figures: five real vector PDFs, all values asserted against artifacts by `iclr2026/figures/build_figures.py` — `fig_hero_ordinal_masses.pdf` (teaser, `fig:teaser`), `estimand_taxonomy.pdf`, `fig_error_growth.pdf`, `fig_novelty_boundary.pdf`, `fig_ceiling_map.pdf` (`fig:app:ceiling`). The family×horizon teaser grid was NOT built (dead thesis; per-cell values #81-gated).
- WP5: ICLR 2026 submission limit verified at 9 pages main text (Author Guide; the .sty is silent; recorded in root tex comment). Body cut ~21 → 9 pages: claim registry (`tab:app:registry`), frontier (`tab:app:frontier`), §5 subsections + floats (`sec:app:anat`), related-work subsections + matrix (`tab:app:matrix`), estimand figure/table, §4 element list moved to the appendix; `tab:app:audit` number→artifact longtable added; reproducibility statement completed (7 `--validate` runners, 24 h/3 h GPU/4-worker caps, 4,187 tracked paths through `2b4d897`); Ethics statement rendered; checklist emptied (ICLR 2026 requires none).
- Citation pass: Henderson AAAI'18 + Agarwal NeurIPS'21 added (source-verified) for the evaluation-caution arc; `zhang2026worldinworld` entry confirmed; denominator-positioning paragraph at `sec:app:rel:eval` (softened per verification: PHYRE/I-PHYRE/World-in-World checked full-text; NovPhy excluded — denominator reporting unverifiable).

- NovPhy repo: `scripts/run_state_difficulty_auc.py` + `.local-artifacts/issue-91-state-difficulty-auc-v1/` (sha256-pinned inputs; `--validate` exits 0; retained records only, zero engine seconds).
- Result: state-level AUC 0.5758 (23 states; higher predicted cost on non-ceiling states), member-deduplicated 0.5918, member-clustered descriptive interval [0.2653, 0.8958] covers 0.5 → "the predictors learn state difficulty" and "not at state-difficulty estimation" are NOT supported. Round 2 cuts both clauses; the within-state discrimination half of the mechanism sentence survives. Post-hoc disclosure required (analysis specified after #87/#89 outcomes).

## 2026-09-23 — Housekeeping (owner directive: no confusing deprecated material)

- Trimmed marked-superseded tails: `writing_outline_boundary_paper.md` v1.2 content, `content_brief.md` historical record + the fully-closed OPEN TEX ITEMS list. Both now carry a one-line pointer to git history.
