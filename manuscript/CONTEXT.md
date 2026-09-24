# Manuscript Context

This glossary is the manuscript-local vocabulary for work that references the published NovPhy benchmark and the in-progress BG-NS-JEPA program.

## Current state for a fresh session (2026-09-23, round 4 after r3 follow-up review)

- **Thesis = the measured ranking failure, AUC as headline.** On the development N1 membership an engine-truth closed-loop audit finds a successful single-shot action for 7 of 14 source members (exactly one successful candidate per ceiling state). Three small (~1.8M-parameter) frozen rankers order candidates at or below chance on N1: AUC 0.4367 [0.4045, 0.4710] (108 cells), member-clustered 0.4180 [0.3135, 0.5252]. On the type010101 split both AUC point estimates sit below 0.5, each over 216 of 612 scored cells per condition with a 6-state-cluster bootstrap: 0.2039 [0.0762, 0.3324] for the frozen zero-shot arm (held out) and 0.2732 [0.2182, 0.3239] for the few-shot arm (adapted on the split), never pooled. Inverting the ranking (max cost, ties to the lower ordinal; post hoc, issue-91-inverted-ranker-v1) does not recover the successes: top-1 15/612 = 0.0245 zero-shot and 21/612 = 0.0343 few-shot against chance 0.0407, paired −0.0162 [−0.0392, +0.0026] and −0.0064 [−0.0270, +0.0118] over 17 state clusters, so no support for a cost-sign reclassification. A fixed-horizon ranker applies its carrier Δ times of Δ agent frames each (Δ² = 1, 25, 225 frames); the primary rankers look 1 (h1) or 25 (h5) frames ahead against outcomes resolving 97.8–400.9 agent frames after the decision, between 0.25% and 25.6% of the outcome horizon (derived: 1/400.9, 25/97.8). Mechanism sentence, always with that scope: "selection fails at within-state action discrimination along the candidate sweep". The hybrid ranker alone orders by the sweep coordinate (0.9986; continuous 0.4924/0.6130). There is no claim about state-difficulty estimation in either direction (post-hoc state-level AUC 0.5758, [0.2653, 0.8958] covers 0.5). The restatement statistics (member unit, band, disjoint supports, AUC choice) are disclosed as post hoc in §4 and §6, and the non-overlap of supports is a post-hoc N1 observation. The +0.0887 contrast reads "unsupported: carried by one seed on an invalidated proxy".
- **Manuscript:** `iclr2026/`, title *Solvable but Mis-Ranked: An Engine-Truth Audit of Frozen World-Model Action Selection in NovPhy*. r4 gate (Main): 39 pp total, body §1–§8 ends p. 9 l.442 (§5 compression moved §8 to start on p. 8 l.423), Repro/Ethics p. 9 (uncounted), references start p. 10, figures unchanged. Fig. 1 caption leads "Engine-truth selection AUC is at or below chance on solvable N1 states"; panel (b) carries the cross-split AUC rows. There are 3 contributions, prose uses stage names (oracle pass / completion pass / timeout audit / restatement), and the objective-mismatch paragraph cites `lambert2020objective` and `girdhar2020forward`. Per-section status is in `content_brief.md`, the binding plan is `writing_outline_boundary_paper.md` v2.4, and the evidence is in `research_evidence.md`.
- **Review:** r1 4/10, 24 findings (23 RESOLVED r2, I16 PARTIAL). r2 4/10 (weak reject), findings I25–I34 plus r1 partials. r3 4/10 (weak reject; "a 6 is defensible once I35 is corrected"), new findings I35 (CRITICAL) and I36–I39. r3-followup 6/10 (borderline), minted I40. Authoring tallies (`iclr2026/review-log.md`): r3 13 RESOLVED, 2 PARTIAL (I10, I30); r3-followup 6 RESOLVED (I30, I35–I39), 0 REJECTED, plus I40 RESOLVED; r4 2 RESOLVED (I10, I25), 0 REJECTED, pending r4 reviewer verification.
- **PHYRE/AUCCESS positioning (r4, I25):** PHYRE certifies solvability by construction and scores multi-attempt task success by AUCCESS, a log-weighted area under the task-level success curve over up to 100 attempts. This paper's delta is narrower: a measured per-state ceiling on a frozen 13-candidate inventory plus a per-candidate engine-truth AUC in a single-shot cascade regime, not a new success-metric family. Sites: abstract.tex:31, introduction.tex:93/:99, synthesis.tex:112, related_work.tex:24–25, appendix.tex:419–420. r4 gate: 39 pp, body ends p. 9 l.442 (§8 starts p. 8 l.423), Repro/Ethics p. 9, references p. 10; §5 now two paragraphs (I10).
- **Canonical numbers:** ceiling member-unit 7/14 = 0.5000 [0.2143, 0.7857]; AUC 0.4367 [0.4045, 0.4710] (108 cells), member-clustered 0.4180 [0.3135, 0.5252]; top-1 0/108; top-3 6/108 vs 0.2340 (paired −0.1859 [−0.2308, −0.1279]); chosen {7:90, 8:12, 9:45, 10:10, 11:6, 12:44}; lookahead 15/16 arms 0/36, one arm 2/36; determinism 176/176 + 288/288; second seed 26/26; counterexample 42/1,224 in 5 of 17 states; zero-shot 13/612 vs 0.0407, few-shot 29/612 (never pooled); inverted top-1 15/612 and 21/612 vs 0.0407; breadth ceiling 25/46 = 0.5435 [0.3913, 0.6957]. All intervals descriptive.
- **Blockers:** all closed (#89 completion, #90 timeout audit, #92 restatement, #91 inverted-ranker probe). No measurement gates the prose.
- **Second candidate parameterization (#93, owner-approved, release b8aba59, frozen plan 6239a04):** C22 (Arm B drag grid) `not_supported_by_this_experiment`, member-clustered AUC 0.6178 [0.5126, 0.7139] (14 members); C23 (Arm A offset sweep) `readiness_or_precision_insufficient`, 0.5512 [0.4235, 0.6685] (9 members). Reading-matrix row 5: appendix-only report (Appendix H.2, `sec:app:secondparam`, `tab:app:secondparam`), body scope sentences unchanged except a pointer (abstract.tex:30, introduction.tex:90, discussion.tex:58, method.tex:49). Engine drag clamp: every candidate launches at saturated speed 10, so the grid's pull-radius axis exists only on the ranker-input side. The digest mechanism sentence ("along the candidate sweep") is unchanged. Never pool arms with each other or with N1.
- **r5 authoring (2026-09-24):** gate 41 pp, body ends p. 9, Repro/Ethics p. 9, references p. 10; r4 I25(a)(b)(c), I40, I41, I42 fixed; registry C1–C23; ten runners in the Reproducibility Statement.
- **Still pending:** anonymized artifact link (author, I16; repro cites commits 2b4d897 and b8aba59 of a non-public repository); r5 re-review of the #93 appendix and the r4 wording fixes.

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

The binding #91 framing: a scoped measurement that frozen model rankers fail at within-state action discrimination on states the engine shows are solvable. Registry rows C16–C21.

_Avoid_: "not dynamics prediction" (false dichotomy); "no system ever"; field-wide causal explanations.

**Source member**

The unit of analysis for the closed-loop ceiling: one physical initial state with its candidate table. 15 members cover 24 states, including 9 byte-identical duplicate pairs.

_Avoid_: Using states (12/23) as the headline unit.

**Action ceiling / oracle-conditioned denominator**

The fraction of source members for which engine replay finds at least one successful candidate (7/14 on N1; 25/46 on the bounded-transfer breadth pool). It is the denominator that turns a zero success rate into a ranking-failure measurement.

_Avoid_: Calling it a gameplay success rate or a sealed-benchmark result.

**Selection validity**

Engine-truth AUC and top-k of the frozen rankers' predicted-cost ordering against per-candidate verdicts on ceiling states.

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
