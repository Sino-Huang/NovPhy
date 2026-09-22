# Writing Outline — Boundary-Anatomy Paper (v1.2, post-audit + second sync)

Status: PLANNING artifact for the reframed manuscript. v1.2 incorporates the second 2026-09-21 sync wave: #79's full component decomposition (S1/S2 replicate, S3 fails at h15, Q2 regime direction reversed), #80's executed prevalence-gate stop (`readiness_or_precision_insufficient`; 0/45 first-shot successes, full run barred by the frozen precondition), #81 unblocked, and the three new follow-up tickets #82/#83/#84. v1.1 incorporated the ManuscriptAuditor review (19 findings) and the executed outcomes of #78 (closed) and #79. supersedes the provisional six-section outline inside `research_evidence.md`. Source evidence boundary: `Sino-Huang/NovPhy` issues #77–#84 (env@254720f + commits 0a0dd73, ac2a4e1, 8399f57, 9e369c4) plus issue-76-closeout evidence in this repository. Drafted 2026-09-21; revised 2026-09-21 (v1.1, v1.2).

## 0. Why this reframing, in one paragraph

The executed program no longer supports a method-advantage paper: the #15 confirmatory returned `not_supported_by_this_experiment` at both compute budgets, #74 found no hybrid superiority over the selected continuous policy, and #75's corrective pilot was negative. What the program uniquely owns is the **precise, family- and horizon-resolved boundary of when adaptive granularity helps**, under pre-registered matched-system discipline the 2025–2026 adaptive-abstraction literature (TAWM, VLWM, THICK, SALT) does not report. The paper's claim: where the benefit appears (and where it inverts), what the mechanism looks like, and what it costs — a calibration target for a method class currently claiming benefits without boundary curves.

## 1. The reviewer test this outline is built around

A reviewer's first-impression question is: *did I learn something I did not know, or is this "obviously so"?* Every section must serve a numbered learnable; each learnable is stated WITH its scope (a learnable without its bound is an overclaim):

- **L1 (family-conditional benefit).** The hybrid training effect is not a property of the method but of the (family, horizon) cell: +0.0887 [+0.0155, +0.2118] regret improvement at h=1 on the N1 rolling/sliding families (4 held-out states); null-to-negative at h=1 (−0.0278 [−0.0694, 0.0000]) and significantly negative at h=5 (−0.0556 [−0.1111, −0.0139]) on the original single/multi-force families (24 states, issue-76 closeout); h=15 exactly zero in both; positive at the short-horizon MSE level out-of-family (CLEVRER, #79). Aggregate "does X help" claims average away this boundary — horizon- AND family-resolved reporting is the lesson.
- **L2 (instability, as hypothesis not mechanism).** Recursive carrier MSE diverges in magnitude at late endpoints (t≥150 on NovPhy; e=120 on CLEVRER, both arms), consistent with — but not proof of — a mechanism in which long-horizon recursion lacks a stable signal. This is a stated hypothesis because the h15 macro symbolic-EXECUTION effect (+0.0944 [+0.0231, +0.1878], N1) is measured at exactly the horizon where carrier MSE is unstable: readout-level signal survives carrier-level instability, so "no stable signal for any policy" would overclaim. CLEVRER #79 sharpens this: the growth-shape component of the signature was NOT supported, so instability-growth is not universal either.
- **L3 (estimand separation — strongest learnable).** Training effect vs symbolic-execution effect vs adaptation effect are separable estimands with different signs: h=1 training effect excludes zero (+0.0887, N1) while h=1 symbolic-execution contrasts straddle it (micro +0.0348 [−0.0102, +0.1147]; macro +0.0239 [−0.0199, +0.0938]); macro execution at h15 is positive (+0.0944). "Symbols help training" and "symbolic readouts help execution" are different claims with different evidence.
- **L4 (novelty directions — strongest learnable).** Appearance novelty moves the comparison in opposite, system-specific directions (type010102 zero-shot: hybrid-continuous h15 −0.2693 [−0.5412, −0.0050] vs hybrid-macro h15 +0.4242 [+0.0915, +0.7657], both intervals excluding zero). No uniform novelty claim survives.
- **L5 (adaptation is not a free rescue, family-scoped).** Adaptation on the novel side degrades longer-horizon continuous ranking in BOTH families (h5: +0.1850 [+0.1012, +0.2688] type010102 and +0.1544 [+0.0062, +0.3025] type010101; h15: +0.0672 and +0.1080, both excluding zero), while the h=1 improvement excludes zero only in type010102 (−0.1970 [−0.2688, −0.1253]; type010101 −0.0866 [−0.2422, +0.0690] straddles).
- **L6 (cost inversion is horizon- and reference-dependent).** Per-state walls from #74 (200 calibration states, offset-225 rollouts — a different family from the N1 t=600 endpoints): continuous_h1 ≈1.79 s, continuous_h5 ≈0.37 s, continuous_h15 ≈0.13 s, hybrid adaptive ≈1.01 s. At h=1 the pure-continuous arm is the expensive one, but hybrid adaptive is still ~2.7× the selected comparator continuous_h5 — "adaptive = costly" is horizon-dependent, never cheapest.
- **L7 (the redirect, tempered).** Adaptive-temporal methods reproduce the decay signature under our contract where tested (#78: TAWM h1-vs-h15 +0.2253 [+0.0816, +0.3716] seed 20260908; VLWM indeterminate +0.0591 [−0.0251, +0.1482]; the plain continuous reference ALSO shows +0.1605 [−0.0125, +0.3256]), but #78's method-class question returned `readiness_or_precision_insufficient` at 4 held-out states. The defensible redirect: abstraction claims should report horizon- and family-resolved matched-system curves; our curves are a calibration datapoint, not a verdict on the class.

## 2. Title candidates and pitch

1. *When Does Adaptive Granularity Help World Models? A Pre-Registered Boundary Anatomy of Joint Horizon–Description Selection* (recommended — question first, method second).
2. *Granularity as a Decision: The Family- and Horizon-Resolved Boundary of Joint Selection in World Models* (retains continuity with the drafted title).
3. ~~*Short Horizons Only*~~ — WITHDRAWN: its precondition (a method-class-wide boundary confirmed by #78) was not met; #78 returned `readiness_or_precision_insufficient`.

One-sentence pitch: a pre-registered, matched-system study mapping WHERE and WHEN joint prediction-granularity selection helps — family- and horizon-conditional effects, mechanistic instability evidence, and full cost accounting — providing boundary curves the adaptive-abstraction literature currently lacks.

## 3. Section-by-section outline

### Abstract (rewrite of the existing two-sided abstract)

- Keep: granularity-as-decision framing, teacher-forced estimand qualifier, exact #15 numbers and `not_supported_by_this_experiment` disposition.
- Add: the boundary payload with its scopes (L1 family-conditionality with both signs; L3 estimand separation; L4 directions; L6 cost), and the tempered redirect (L7).
- Every number must appear in the claim registry (§5) first.

### 1. Introduction (retarget of the drafted 9-paragraph introduction)

- Paragraphs 1–7 are the reuse base (world-model definition; decision-dependent demands; single-axis relaxations; control-schedule distinction; setting; 3×3 surface; controller+carrier) — but reuse is subject to a retarget check, not verbatim adoption: they were calibrated under the all-negative framing (content_brief records this), and the boundary paper's positive payload now arrives in ¶8, so ¶5–7 emphasis may need rebalancing toward heterogeneity ("when", not "whether").
- Replace ¶8–9 with the **three-act boundary structure**: (a) confirmatory negative (#15) — there is no general advantage to find; (b) the anatomy (#74 + #76-closeout + #77) — the effect is family- and horizon-conditional with mechanism and cost; (c) the external/replication results (#78/#79, executed) — the signature question is precision-insufficient in-domain, and out-of-family replication is partial.
- Contribution list: formulation; the estimand taxonomy (L3) as a reusable protocol contribution; matched-system boundary anatomy (L1–L6); external-method and out-of-family tests with honest dispositions (#78/#79).
- Close with L7 as the significance claim — calibration for the field, stated as such.

### 2. Related Work — organized by the two axes plus evaluation zeitgeist

- Temporal axis: TAWM, VLWM, THICK, tempoRL/options line. Description axis: VisualPredicate, STRIPS-WM, neuro-symbolic world models. Joint/state+temporal: SALT (offline-RL skills; closest phrase-level neighbor — distinguish skills-vs-prediction-granularity). Evaluation zeitgeist: World-In-World (closed-loop utility), WorldBench (concept-isolated evaluation).
- Positioning table: method × {temporal adaptive, description adaptive, pre-registered protocol, matched-system comparison, family/horizon-resolved boundary curve}. The last column is empty across the surveyed class — but this claim MUST be verified per-paper before drafting (see §7 prerequisite R1; `related_work_citation_map.md` currently has NO rows for TAWM/THICK/SALT/VisualPredicate/World-In-World/WorldBench).
- Bib note: VLWM (du2026vlwm), CLEVRER (yi2020clevrer), V-JEPA 2 (assran2025vjepa2) are already in `iclr2026_conference.bib`; TAWM, THICK, SALT, VisualPredicate, World-In-World, WorldBench still need entries.

### 3. Setting and formulation (reuse drafted material)

- Action-sparse persistent-effect environments; NovPhy as test setting; the 3×3 request surface over the continuous carrier; requested vs effective horizon; description mode as readout selection.
- The estimand taxonomy as a named contribution: training effect / symbolic-execution effect / adaptation effect (#76 Phase A0.1 separation). Flag where the paper switches estimands (see 5.x headers).

### 4. The pre-registered matched-system protocol (new section; a contribution in itself)

- Frozen plans with typed terminal failures; membership sealed before outcomes; paired descriptive bootstrap; three-seed paired checkpoints; equal-update training with full work-reporting (`training_compute_matched=false` disclosed); the disposition vocabulary; every attempted cell reported (208/208, 186/208, 34/104, 480 CLEVRER windows, 0 typed failures on #79).
- What this buys: the boundary curves are not post-hoc cell selection — and the paper must show the #76-closeout original-family rows (which hurt the hybrid) right next to the N1 rows (which help it), precisely because a reviewer who finds the imported closeout table will otherwise read the spine as cell selection.

### 5. Boundary anatomy (the experimental core; each subsection names its estimand and family in the header)

- **5.1 Confirmatory endpoint result (#15; estimand: teacher-forced endpoint carrier-MSE gain; instance-held-out same-physics).** Two budgets, strongest-eligible comparator each, six sealed rollouts; zero gain / −5.2×10⁻⁴ [−7.1×10⁻⁴, −3.6×10⁻⁴]; no violation increase; `not_supported_by_this_experiment`.
- **5.2 Development anatomy on the original families (#74 + #76 closeout; estimand: ranking regret + offset-225 carrier MSE; 200-state and 24-state development cohorts).** `continuous_h5` selected strongest (selection optimism disclosed); hybrid adaptive −0.0200 [−0.045, +0.003] vs continuous_h5; the closeout's family table: h1 −0.0278 [−0.0694, 0.0000], h5 −0.0556 [−0.1111, −0.0139], h15 0.0000 — the hybrid is null-to-worse on ITS OWN training families; cost table (L6).
- **5.3 Horizon-resolved effects on new normal-mechanics families (#77 N1; estimand: ranking regret vs t=600 end-of-window cost; 4 held-out states, rolling/sliding).** h1 +0.0887 [+0.0155, +0.2118]; h5/h15 null; h15 macro execution +0.0944; error-growth figure (diverging magnitudes at t≥150, both arms); headroom (prior/uniform regret 0.492/0.461). The 5.2-vs-5.3 contrast IS the family-conditionality result (L1).
- **5.4 Novelty boundary (#77 N2; estimands: zero-shot and few-shot ranking regret, reported separately).** L4 pair of opposite-direction effects; L5 adaptation table with per-family intervals; n2n contention disclosure (34/104, retained).
- **5.5 External-method boundary (#78, EXECUTED; estimand: ranking regret on the #77 frozen states; mechanism-class comparison).** TAWM/VLWM ports under the identical contract with structural-equivalence disclosure (TAWM cell shares the #77 continuous reference recipe); decay-signature contrasts (TAWM +0.2253/+0.2576/+0.2177 across seeds; VLWM +0.0591/+0.1000/+0.0494; continuous reference +0.1605/+0.2671/+0.1354); work-reported comparisons vs continuous_h5 (+0.0664/+0.0325/+0.0743); dispositions Q1 `readiness_or_precision_insufficient`, Q2 `supported`, Q3 `supported`. The honest reading: everyone decays; distinguishing method class from implementation needs more states than 4.
- **5.6 Out-of-family replication (#79, EXECUTED; estimand: recursive position MSE at {15,30,60,120} CLEVRER frames; 12 validation scenes 10000–10011 × 40 windows, 48 units/seed, 0 typed failures, ~0.43 GPU-h).** Component-wise (frozen S1/S2/S3 rules): **S1 short-horizon separation REPLICATES** (h1/e15 training effect +0.00366 > margin 0.0010); **S2 horizon-ordering REPLICATES** (h15 difference −0.00006 < h1 difference — separation shrinks with horizon, matching the NovPhy shape); **S3 growth-shape FAILS at h15** (continuous-arm recursive MSE non-monotone: 0.00132 → 0.00166 → 0.00133 → 0.01156 over e ∈ {15,30,60,120}; h1 and h5 grow monotonically, h1 reaching 13.53 at e=120). Q1 signature `not_supported_by_this_experiment` (driven entirely by S3); Q2 contact-regime `not_supported_by_this_experiment` with direction OPPOSITE the NovPhy scale-separation story (all D point estimates negative: h1 −0.108, h5 −0.0076, h15 −0.0020). Macro arm unsupported, dropped in Phase 0. #83 decomposes the S3 ambiguity (regime property vs estimand/endpoint artifact); both families reported separately — no falsification claim either way.
- **5.7 Reactive closed-loop bound (#80, EXECUTED with typed stop).** Demoted to a bounded Discussion paragraph, and now with a known outcome: the mandatory prevalence gate FAILED — pooled first-shot success prevalence **0.0000 (0/45 valid closed-loop executions)** against the 0.10 floor; the full 288-cell matrix is barred by the frozen precondition, and the typed `readiness_or_precision_insufficient` stop IS the deliverable (the paired-difference questions are not scored — the gate stop precedes scoring by design). 3 typed `decision_failure: prior_candidate_absent` cells recorded. The #61 runner extension (fixed-h5, hybrid fixed-mode, ordinal-prior arm) landed additively first. #82's oracle-ceiling diagnostic (no new gameplay) resolves the stop's 2-way ambiguity: state-difficulty floor vs h1 ranking failure. Writing note: report the stop as a protocol-consistent bound, never as evidence about planner quality — the oracle ceiling (#82) determines whether ANY planner could have cleared the floor on this membership.

### 6. The boundary picture (synthesis, fed by #81 — pending; fold into §5/§7 if #81 stalls)

- One figure the teaser does not already show: the family × horizon boundary grid (improvement sign per cell across N1-rolling, N1-sliding, original-010101, original-010102, CLEVRER-MSE) — the single visual that carries L1.
- Work frontier (regret vs per-state wall) and mechanism table (growth shapes, estimand-separated effects, regime dependence, typed-failure concentrations).
- Claim registry as a published table.

### 7. Discussion

- Redirect (L7, tempered): matched-compute, family/horizon-resolved reporting as a field requirement; our curves as calibration.
- Honest limitations: single benchmark family program (4/45 cells runtime-tested; 41 unsupported with named blockers); descriptive intervals (4 held-out N1 states; 24 closeout states; 48 CLEVRER units/seed); t=600 end-of-window right-censored semantics; teacher-forced #15 estimand; instability limits recursive claims; n2n contention; sealed gameplay never opened; #78's 4-state precision insufficiency.
- Falsifiability, WRITTEN FROM EXECUTED RESULTS: what would change the conclusions — (i) a matched-system external method with non-decaying advantage on more states (extends #78, currently precision-insufficient); (ii) a third family breaking the two-family component tie (#84: NovPhy full signature vs CLEVRER 2/3 — S1/S2 replicate, S3 and the regime direction do not); (iii) #83 localizing the S3 non-monotonicity to regime property vs estimand/endpoint artifact; (iv) #82 resolving the #80 zero floor to state-difficulty vs ranking failure. All four are filed and executing; none is prose-blocked.

### 8. Conclusion

- The boundary statement compressed from ALL of L1–L7 (three sentences), the protocol contribution, and the open cells with named blockers.

## 4. Evidence inventory (updated for #78/#79 execution)

| Evidence | Status | Source |
|---|---|---|
| #15 confirmatory negative | Available | issue-15 confirmatory-v2 summary + compact report |
| #74 matched-systems anatomy + costs | Available | `issue-74-matched-dynamics-v1/readiness.json` |
| #76 closeout development anatomy (original families) | Available | `data/runtime_evidence/issue-76-closeout/` (this repo) |
| #77 N1 horizon-resolved effects | Available | `issue-77-n1-diagnostic-v1` |
| #77 N2 novelty boundary | Available | `issue-77-n2-eval-v1` |
| Readiness record 4/45 | Available | `docs/issue-76-novelty-level-evaluation-design.md` §1 |
| #78 external temporal-adaptation baselines | **EXECUTED (closed 2026-09-21)** | `issue-78-external-temporal-baselines-v1` |
| #79 CLEVRER out-of-family replication | **EXECUTED (issue open, dispositions published)** | `issue-79-clevrer-boundary-v1` (commit 0a0dd73) |
| #80 reactive closed-loop bound | **EXECUTED (typed stop: prevalence gate 0/45; full matrix barred)** | `issue-80-reactive-diagnostic-v1` (commits ac2a4e1, 8399f57, 9e369c4) |
| #81 pooled synthesis + registry | UNBLOCKED (all three upstreams terminally dispositioned); not yet executed | ADD-EXP issue #81 |
| #82 oracle-ceiling zero-floor diagnostic | PENDING (new 2026-09-21) | ADD-EXP issue #82 |
| #83 CLEVRER h15 non-monotonicity decomposition | PENDING (new 2026-09-21) | ADD-EXP issue #83 |
| #84 third-family replication tie-breaker | PENDING (new 2026-09-21; Physion/ShapeStacks short-list, readiness-based selection) | ADD-EXP issue #84 |

## 5. Claim registry (draft; each row must cite artifact rows before drafting)

| # | Claim | Status | Disposition / scope |
|---|---|---|---|
| C1 | Joint granularity selection shows no general endpoint advantage under the frozen two-budget protocol | Verified | `not_supported_by_this_experiment` (#15) |
| C2 | A hybrid h=1 training effect exists on the N1 rolling/sliding families | Verified (descriptive) | supported, bounded to 4 held-out states, 2 families |
| C2b | On the original single/multi-force families the hybrid training effect is null-to-negative at h1 and negative at h5 | Verified (descriptive) | supported as a boundary condition (24 closeout states); the family-conditionality of C2/C2b is itself the claim |
| C3 | Hybrid training effects at h15 are exactly zero in both family sets; h5 is null on N1 and negative on original families | Verified (descriptive) | supported, scope = both sets, per-family stated |
| C4 | Symbolic-execution effects at h=1 straddle zero; macro execution at h15 is positive (N1) | Verified (descriptive) | supported, bounded to N1 |
| C5 | Recursive carrier MSE diverges in magnitude at late endpoints, both arms, both families | Verified | supported, as instability observation; growth-shape NOT universal (#79) |
| C6 | Appearance novelty moves zero-shot contrasts in system-specific opposite directions (type010102) | Verified (descriptive) | supported, zero-shot only |
| C6b | Few-shot cross-side changes are interval-excluding in type010102 (+0.1953 / −0.1892) | Verified (descriptive) | supported, reported separately from C6 |
| C7 | Adaptation on the novel side degrades h5/h15 continuous ranking in both families; improves h1 only in type010102 | Verified (descriptive) | supported with per-family scope |
| C8 | The horizon-decay signature belongs to the adaptive-temporal method class | **TESTED** | `readiness_or_precision_insufficient` (#78, 4 states); TAWM/VLWM/reference all decay |
| C9 | The NovPhy boundary signature replicates out-of-family on CLEVRER | **TESTED** | `not_supported_by_this_experiment` overall (#79); component-wise: S1 separation REPLICATES, S2 ordering REPLICATES, S3 growth-shape fails at h15 (non-monotone), Q2 regime direction reversed |
| C10 | The h1 effect has closed-loop reactive payoff | **TESTED (gate stop)** | `readiness_or_precision_insufficient` (#80): prevalence 0.0000 < 0.10 floor over 45 valid executions; paired-difference questions unscored by design; #82 oracle ceiling pending |
| C14 | The CLEVRER h15 non-monotonicity is a regime property (post-settlement) rather than an estimand/endpoint artifact | PENDING #83 | untested; frozen-checkpoint decomposition |
| C15 | A third physics family breaks the two-family component tie (NovPhy full signature vs CLEVRER 2/3) | PENDING #84 | untested; either direction publishable |
| C16 | The #80 zero floor is a state-difficulty floor (oracle prevalence 0) rather than h1 ranking failure | PENDING #82 | untested; analysis-only |
| C11 | At matched development scoring, hybrid adaptive shows no regret advantage over the selected continuous_h5 | Verified (descriptive) | #74 contrast −0.0200 [−0.045, +0.003]; selection optimism disclosed |
| C12 | Per-state inference cost is horizon- and system-dependent; hybrid adaptive is never the cheapest arm | Verified | #74 walls (1.79/1.37/0.13/1.01 s); #74-family scope stated |
| C13 | Headroom: prior and uniform regret baselines leave ranking signal on every frozen state | Verified | #77/#78 headroom tables (e.g. prior 0.492 / uniform 0.461 pooled N1) |

## 6. Figures and tables plan

- Teaser v2: the family × horizon boundary grid as hero (§6's single figure, promoted), with a work-frontier inset; keep Okabe-Ito role-color rules and prohibited-claims discipline; rewrite `teaser_figure_brief.md` as v2 first (its current §C reserved-blank-result field is obsolete).
- Fig. 2: estimand taxonomy diagram (training / symbolic-execution / adaptation).
- Fig. 3: recursive error growth vs elapsed endpoint, both families + CLEVRER (shows agreement at short times, divergence in magnitude later).
- Fig. 4: novelty boundary (zero-shot and few-shot panels, per-family intervals).
- Table 1: related-work positioning matrix (after R1 verification). Table 2: cost/work frontier. Table 3: claim registry.

## 7. Reuse map and prerequisites (updated)

- Abstract ~60% reusable; Introduction ¶1–7 reuse-with-retarget-check (all-negative calibration noted); ¶8–9 + contributions rewritten.
- **R1 (blocking for §2 and the Abstract's redirect sentence):** verify the positioning-table premise per paper (TAWM/VLWM/THICK/SALT report no pre-registered matched-system family/horizon-resolved boundary curves) and add the six missing citation-map rows; without R1 the "empty column is ours" claim is unverified.
- Teaser brief v2 rewrite; `research_evidence.md`/`CONTEXT.md`/`reviewer_expectations_2026.md` synced (2026-09-21 pass + post-audit fixes); bib additions (TAWM, THICK, SALT, VisualPredicate, World-In-World, WorldBench).

## 8. Writing order and gates (updated for #78/#79 execution)

1. **Now:** Sections 3, 4, 5.1–5.6, 7 (falsifiability from executed results incl. the #80 stop), registry C1–C9+C11–C13, teaser v2 brief, family×horizon grid figure. The #80 Discussion paragraph now has its known outcome (typed stop) and is draftable.
2. **Gate A — CLEARED:** #78 executed; §5.5 and C8 draft from published artifacts.
3. **Gate B — CLEARED:** #79 executed (component detail published) and #80 executed with its typed stop; §5.6, C9, and the #80 Discussion paragraph draft from published artifacts.
4. **Gate C (#81, UNBLOCKED):** all three upstream tickets are terminally dispositioned — #81 is executable now; synthesis grid + final registry table; then the full critic round before submission.
5. **Live follow-ups (#82/#83/#84):** their outcomes amend C14–C16 and the §7 falsifiability sentences only; they do not gate the Now set.

## 9. House rules carried over

- Claim-status vocabulary (Verified / Specified / Blocked / Unavailable); no softening of `not_supported_by_this_experiment`; no "no significant difference" euphemisms; no generalization beyond recorded scope; descriptive-interval labelling wherever the source artifacts say so; **numbers in prose must appear in the claim registry first** (audited in v1.1: all §1/§3 numbers now have registry rows); estimand and family named in every 5.x header; the #76-closeout negative rows appear alongside the N1 positive rows wherever C2 is quoted.
