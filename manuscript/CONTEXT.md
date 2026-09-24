# Manuscript Context

This glossary is the manuscript-local vocabulary for work that references the published NovPhy benchmark and the in-progress BG-NS-JEPA program.

## Current state for a fresh session (2026-09-24, round 7 #94/#95 integration, Story Lock v2)

- **r7 update (overrides the r6 bullets below where they differ).** Title *Solvable, Yet Selected No Better Than Chance: Frozen World-Model Rankers Against a Measured Engine Ceiling in NovPhy*. Four inventories over the same 15 N1 members; the fourth, the #94 angle × launch-power sweep (pre-registered; commits 6c07c94 / 88a1a21), has ceiling 8/15 [0.2667, 0.8000], pooled top-1 0/72 vs 0.0521 (0/24 per system), paired −0.0521 [−0.0563, −0.0500] (degenerate; C24 supported), AUC_m 0.7441 [0.6923, 0.8026] (C25 not_supported_by_this_experiment). #95 post-hoc controls (frozen 2026-09-24T08:21:37Z, commits 19fc865c / 3742c7e0): release inert (C26 supported, C27 not_supported_by_this_experiment, AUC_m 0.7456), speed-only baseline AUC_m 0.6889 [0.6424, 0.7599] (seen before the freeze, no disposition), within-power AUC 0.5949 [0.4444, 0.7454] (C28 readiness_or_precision_insufficient). N1 paired interval −0.0778 [−0.0797, −0.0769] is degenerate; the held-out split is the only non-degenerate record with the interval below 0. Clause: "no more often than an inventory-matched uniform draw on every inventory tested (in point estimate on the offset sweep and drag grid)", companion "three sets of launch angles at saturated speed and one angle × launch-power sweep"; "pull radius saturates at 18.49 px". Lesson (post hoc): coordinate-only baseline and within-coordinate AUC beside ordering AUC. Registry C1–C28; new App. H.3 `sec:app:launchpower`. r7 gate: 46 pp, Conclusion ends p. 9, Repro/Ethics p. 9, references p. 10, 0 undefined, 0 [TODO]. Binding plan: outline v2.7; Story Lock v2 + digest amended r7 (`changelog.md` "r7 Story Lock v2").
- **r7 review status:** r6 6/10. r7-authoring tags in `iclr2026/review-log.md` r6 section: I47, I48, I49, I50, I51, I52, I53 RESOLVED; I25 PARTIAL (field-practice reach, training-range confound); I16 open (owner). Abstract words 390→390 (0). Pending: r7 re-review; anonymized link (I16; repro now also cites 88a1a21 and 3742c7e0).

### r6 state (historical where the r7 bullets differ)

- **Thesis = selection, not ordering.** Under a per-state ceiling measured in the engine, three frozen ~1.8M-parameter NovPhy rankers, pooled per inventory, pick the engine-verified success no more often than an inventory-matched uniform draw (in point estimate) on three candidate inventories over the same 15 N1 source members and on a held-out split, while their member-clustered ordering AUC runs from 0.4180 to 0.6178 with the inventory. For these rankers an ordering statistic does not certify selection. In the engine every inventory tested is a set of launch angles at saturated speed (pull radius clamped above 18.49 px).
- **Manuscript:** `iclr2026/`, title *Solvable but Not Selected: Frozen World-Model Rankers Against a Measured Engine Ceiling in NovPhy*. r6 gate: 42 pp total, Conclusion ends p. 9, Repro/Ethics p. 9 (uncounted), references p. 10, 0 undefined, 0 [TODO]. Binding plan: `writing_outline_boundary_paper.md` v2.6; Story Lock and owner decisions: `changelog.md` "r6 Story Lock"; digest amended r6 (`issue91_exec_digest.md`).
- **Fig. 1 (`fig:teaser`):** `figures/fig_hero_selection_ordering.pdf` (`build_figures.py:hero_selection_ordering()`): (a) member-unit ceilings per inventory plus the type010101 breadth pool; (b) pooled top-1 minus inventory-matched chance per record (hatched adapted few-shot row, not frozen-system evidence); (c) member-clustered AUC vs 0.5 with #93 frozen margins. One row per record, never pooled. The former hero (N1 ordinal bands) is now `fig:app:bands`, a post-hoc observation on N1.
- **Headline numbers (status):** ceilings 7/14 [0.2143, 0.7857] original 13-candidate sweep (#87 q1 pre-registered; member unit post hoc), 9/15 [0.3333, 0.8667] offset sweep, 14/15 [0.8000, 1.0000] drag grid (pre-declared descriptive), breadth pool 25/46 [0.3913, 0.6957] (pre-registered). Pooled top-1: 0/108 vs 0.0780, paired −0.0778 [−0.0797, −0.0769] (outcome pre-specified, #87 q2 margin 0.5; chance reading post hoc); 7/81 vs 0.1030, −0.0166 [−0.0781, +0.0575]; 8/126 vs 0.0804, −0.0169 [−0.0938, +0.1190] (pre-declared descriptive); held-out frozen zero-shot 13/612 vs 0.0407, −0.0195 [−0.0368, −0.0045] (pre-registered); adapted few-shot 29/612, +0.0067 [−0.0202, +0.0368] (never pooled). Ordering: AUC_m 0.4180 [0.3135, 0.5252] N1 (headline choice post hoc; cell-unit 0.4367 [0.4045, 0.4710]); 0.5512 [0.4235, 0.6685] offset `readiness_or_precision_insufficient`; 0.6178 [0.5126, 0.7139] grid `not_supported_by_this_experiment` (frozen decision statistic, primary arm). Grid median ρ radius −0.7882, drag_y −0.8125; 120/135 choices at ordinal 15 (interpretation post hoc). Miss-all 0.567 / 0.3732 / 0.3074, listed, never multiplied. Controls: 16 lookahead arms (15 at 0/36, one 2/36), second seed 26/26, inverted ranker 15/612 and 21/612 (post hoc). All intervals descriptive.
- **Review:** r5 6/10 (borderline). I43–I46 RESOLVED r5-followup; I25 PARTIAL r6-authoring (abstract and §1 now lead with pooled top-1; Axis 1 value limit remains: no engine-side second action coordinate tested); I16 open (owner). Two r6 cold reads reproduce the locked Δbelief.
- **Blockers:** none gate the prose. Scoped continuations named in §7 only: a launch-power sweep inside the 18.49 px clamp and a second ranker family.
- **Still pending:** anonymized artifact link (I16; repro cites commits 2b4d897 and b8aba59 of a non-public repository); r6 re-review.

## Core Terms

**NovPhy benchmark**

The published physical-reasoning benchmark for open-world AI systems. It is not the BG-NS-JEPA program.

_Avoid_: Calling published NovPhy findings BG-NS-JEPA results.

**BG-NS-JEPA**

The in-progress world-model program that selects requested horizons and description modes for persistent physical cascades. Its controller is implemented and evaluated at bounded model-selection scope, but no joint-controller advantage has been demonstrated.

_Avoid_: Published method, demonstrated advantage, completed benchmark result.

**Action-sparse persistent-effect environment**

An environment in which actions occur infrequently while their physical effects continue autonomously for many fixed steps.

_Avoid_: Long-horizon environment when action sparsity and effect persistence are the relevant properties.

**Rollout**

One independently executed single-shot simulation of a scenario specification under one recorded intervention, ending with one declared termination reason.

_Avoid_: Episode, video, multi-shot game.

**Fixed step**

The authoritative discrete simulation-time coordinate for physical state changes and event occurrence.

_Avoid_: Render frame, image frame.

**Cohort release**

An immutable, versioned publication of a cohort with its collection plan, partitions, provenance, and accepted derivation references. Completion of a cohort release does not authorize final scoring or manuscript claims.

_Avoid_: Final-evaluation authorization, manuscript authorization, mutable collection directory.

**Exposure role**

The declared permission for a scenario lineage to influence training, calibration, model selection, or final evaluation.

_Avoid_: Informal train/test label, folder split.

**Final evaluation**

The sealed role for frozen final metrics. For the issue-15 protocol, the seed-4505 final partition was authorized, collected (six complete sealed rollouts), consumed once, and evaluated under the frozen two-budget protocol with the bounded negative disposition `not_supported_by_this_experiment`. Fresh sealed gameplay evaluation (#64/#65) remains unauthorized; the #76 advancement gate is unmet (pre-access `readiness_or_precision_insufficient` stop, recorded as not achievable).

_Avoid_: Unrun, generally unavailable, pending-authorization wording for the executed issue-15 result.

**Continuous carrier**

The continuous predictive latent designated to carry a BG-NS-JEPA rollout between prediction decisions. Issue 60 implements a deployment-aligned temporal carrier with aligned prior context for motion and explicit motion-availability masks. Symbolic readouts do not replace it. [carrier construction](../world_model/data/deployment_temporal.py#L630-L705) [carrier tests](../tests/test_deployment_temporal_carrier.py#L180-L207)

_Avoid_: Symbolic rollout state, controller output.

**Deployment temporal carrier**

Implemented method infrastructure from issue 60, included in merge `6119b6c`. It accepts agent observations only. Canonical engine state is excluded from model and planner input except for declared source-bound supervision or alignment diagnosis. Complete trajectories are atomic, and each scenario lineage may belong to exactly one exposure role. This is not an empirical result. [input and inference contract](../world_model/data/deployment_temporal.py#L93-L120) [trajectory contract](../world_model/data/deployment_temporal.py#L311-L515) [input-isolation tests](../tests/test_deployment_temporal_carrier.py#L209-L326)

_Avoid_: Controller result, canonical-state input, independently sampled decision.

**Requested horizon**

The declared number of fixed steps that one prediction decision asks the world model to advance. It is distinct from an effective horizon shortened by terminal clamping during scoring.

_Avoid_: Observed duration, capture stride.

**Description mode**

The requested prediction description `continuous`, `micro`, or `macro`. The cohort-v2 pair surface is the 3 by 3 cross-product of these modes and horizons `(1, 5, 15)`.

_Avoid_: Dataset modality, label availability.

**Joint pair controller**

The controller that selects requested horizon and description mode as one coupled decision. A distilled joint-pair controller and a parameter-matched two-head controller have equal scores in bounded model selection. This provides no demonstrated advantage.

_Avoid_: Final-evaluation winner, established joint-versus-factorized result.

**Policy baseline**

One of four controller-free issue-9 policies. Each policy and exposure-role cell covers six states. These policy results are not compute-matched to issue 10.

_Avoid_: Controller baseline, compute-matched controller comparison.

**Trajectory-optimal label**

A bounded cohort-v2 label used by the controller workflow to select a trajectory-optimal pair.

_Avoid_: Terminal-outcome label, evidence of controller effectiveness.

**Terminal outcome**

The engine-defined final consequence of one recorded intervention, such as cleared, failed, settled nonterminal, or unsettled nonterminal. It is a property to predict from a fixed shot, not an action selected by an evaluated agent.

_Avoid_: Agent task success, free-form macro predicate.

**Terminal-outcome accuracy**

The accuracy of a common final-state readout against the engine-defined terminal outcome. It is a Specified endpoint and remains Unavailable: it was not the executed issue-15 primary estimand (which was authoritative endpoint carrier MSE under teacher forcing), and the common readout and coordinate decoder are uncompleted.

_Avoid_: Implemented result, coordinate displacement error, agent success rate.

**Macro predicate**

A coarse state or event predicate. In central cohort-v2, only `steady-state` and `structure-unstable` are accepted source-bound macro labels. The issue-7 pair-measurement surface contains 378 macro-mode available records. [issue-7 pair-measurement summary](../data/runtime_evidence/issue-7/cohort-v2-pair-measurement-summary.json#L1) `cascade-active`, `collapsed`, and `pigs-cleared` remain excluded.

_Avoid_: Automatically valid target, evidence of macro generalization.

**Physical violation v1**

The versioned endpoint measurement vocabulary for only `excess_penetration` and `unsupported_stationary_or_floating_body`. It does not support a dense-path plausibility claim. `illegal_contact` remains Unavailable.

_Avoid_: Dense-path metric, malformed data, inferred impossibility.

**Issue-11 aggregation**

One round over six rollouts and 109 controller decisions using aligned ground-truth-expert carrier continuation. It leaves the source cohort unchanged and reports zero deltas against the oracle-state baseline.

_Avoid_: Model closed-loop rollout, terminal-outcome accuracy, controller effectiveness.

**Issue-57 held-out gameplay evidence**

Verified bounded negative evidence from five systems on five held-out levels with three seeds each. All 75 trials were included, and every system recorded `0/15` successes. The result is a complete zero-success floor with disposition `not_supported_by_this_experiment`. Adaptive granularity was not materially exercised because adaptive CEM/MPC requested `continuous-h15` on all 44 recorded decisions. This does not establish equivalence, impossibility, causal training-data insufficiency, controller efficacy or inefficacy, or the manuscript's central claim. [issue-57 summary](../data/runtime_evidence/issue-57/cohort-v2-gameplay-success-summary-v2.json#L13-L15) [usage](../data/runtime_evidence/issue-57/cohort-v2-gameplay-success-summary-v2.json#L42-L56) [system results and trial matrix](../data/runtime_evidence/issue-57/cohort-v2-gameplay-success-summary-v2.json#L133-L225)

_Avoid_: Controller equivalence, impossibility result, central-claim result.

**Issue-61 retraining and evaluation tooling**

Completed implementation/tooling work, closed at commit `40ab258`. It delivers nested complete-lineage retraining manifests, matched source/deployment carrier matrix tooling, exact h1/h15 exposure and checkpoint validation, prediction, recursive, ranking, physical, and compute evaluation tooling, explicit legacy and retrained h1/h15 and adaptive gameplay systems, and a public no-write dry run. It is not a retraining result, model-selection result, data-scaling finding, gameplay result, or adaptive-granularity benefit.

_Avoid_: Retraining result, gameplay result, demonstrated effect.

**Boundary-anatomy reframing (SUPERSEDED 2026-09-23)**

The 2026-09-21 framing (the paper reports when adaptive granularity helps). Superseded by the ranking-failure thesis after #82/#85 construct-invalidated the t=600 proxy that scored it. Its sections survive only as the heuristic measurement-validity case study (§5).

_Avoid_: Drafting from it; any positive boundary claim.

**Ranking-failure thesis**

The binding #91 framing, restated r6 as "selection, not ordering": the frozen rankers' pooled top-1 is at or below inventory-matched chance under a measured per-state ceiling on every launch-angle inventory tested, while ordering AUC moves with the inventory. Registry rows C16–C23.

_Avoid_: "not dynamics prediction" (false dichotomy); "no system ever"; field-wide causal explanations.

**Source member**

The unit of analysis for the closed-loop ceiling: one physical initial state with its candidate table. 15 members cover 24 states, including 9 byte-identical duplicate pairs.

_Avoid_: Using states (12/23) as the headline unit.

**Action ceiling / oracle-conditioned denominator**

The fraction of source members for which engine replay finds at least one successful candidate (7/14 on N1; 25/46 on the bounded-transfer breadth pool). It is the denominator that turns a zero success rate into a ranking-failure measurement.

_Avoid_: Calling it a gameplay success rate or a sealed-benchmark result.

**Selection validity**

Pooled top-1 against inventory-matched chance (the selection estimand) and engine-truth AUC (the ordering estimand, a supporting statistic) of the frozen rankers' predicted-cost ordering against per-candidate verdicts on ceiling states.

_Avoid_: Significance language; the member-clustered AUC interval covers 0.5.

**Counterexample split**

The cross-split type010101 audit where predictors select engine-truth successes in 42 of 1,224 cells (5 of 17 states). It must accompany every N1 zero-selection sentence.

_Avoid_: Pooling zero-shot and few-shot; reading the counterexample as universality.

**Horizon-resolved training effect (DEMOTED)**

The issue-77 N1 diagnostic contrast `hybrid_continuous_h1` vs `continuous_h1`: +0.0887 [+0.0155, +0.2118], per-seed 0, 0, +0.2662, on the construct-invalidated t=600 proxy. Demoted by #92 WP5; first-person motivation only.

_Avoid_: Presenting it as evidence; quoting it without the per-seed decomposition.

**Appearance-novelty boundary**

The issue-77 N2 evaluation over matched normal/novel appearance pairs: zero-shot and few-shot conditions reported separately; type010102 effects move in opposite system-specific directions (hybrid-continuous h15 −0.2693 vs hybrid-macro h15 +0.4242). Heuristic (proxy-scored).

_Avoid_: Any uniform "novelty helps/hurts hybrids" claim.

**ADD-EXP and follow-up tickets (#78–#92)**

#78 closed (method-class `readiness_or_precision_insufficient`); #79 executed (S1/S2 replicate, S3 fails, Q2 reversed at e=120); #80 typed stop (0/45); #81 never executed (superseded); #82 `no_defensible_binding` (0.1299 vs 0.50); #83 Q1 supported / Q2 not_supported; #84 Q1 not_supported / Q2 supported; #85 agreement 0.1064, 0/47; #86 `readiness_or_precision_insufficient`; #87 pre-completion ceiling 0.3043; #88 no consistent rule; #89 completion 12/23 (state unit); #90 `indeterminate`, missingness disclosed; #92 final restatement. All closed. Final values: `research_evidence.md`.

_Avoid_: Treating pending tickets as evidence, or letting them reopen #64/#65.

**Issues #62 through #65**

#62 completed successor-cohort collection; #63 closed `not_supported_by_this_experiment`; #64/#65 remain open, unauthorized fresh sealed gameplay work — the #76 advancement gate is unmet (pre-access stop). They do not supply a positive adaptive-granularity result.

_Avoid_: Completed experiment, benchmark result, demonstrated effect.

**Issues #35 through #38**

Open research-completion work. Issue 35 is the prespecified replicate matrix and statistical analysis, which must include the issue-57 gameplay evidence and the issue-17 parser imbalance. Issue 36 is the reproducible archive. Issue 37 is the consolidated final report. Issue 38 is the terminal research-program audit. All four are incomplete.

_Avoid_: Completed analysis, archived release, final report.

**Oracle supervision**

Training-time, source-bound labels or alignment diagnostics derived from engine evidence under a versioned contract. Oracle or canonical engine state is excluded from model and planner input.

_Avoid_: Test-time controller input, visual guess.

## Claim-Status Vocabulary

**Verified**

Supported by a published record or source-bound artifact within the recorded scope.

_Avoid_: Proven generally, validated beyond the artifact.

**Specified**

Defined by a contract or plan but not established as a completed empirical result.

_Avoid_: Implemented result.

**Blocked**

Cannot proceed because an explicit prerequisite or authorization is unmet. Issue-15 final metrics were derived and consumed (bounded negative). What remains Blocked: fresh sealed gameplay (#64/#65) behind the unmet #76 advancement gate.

_Avoid_: Negative result, Unavailable, unrun.

**Unavailable**

The required evidence is absent or the concept is excluded for the relevant record. It is neither true nor false.

_Avoid_: False, missing-as-negative.
