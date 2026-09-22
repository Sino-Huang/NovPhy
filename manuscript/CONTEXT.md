# Manuscript Context

This glossary is the manuscript-local vocabulary for work that references the published NovPhy benchmark and the in-progress BG-NS-JEPA program.

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

**Boundary-anatomy reframing**

The 2026-09-21 manuscript decision: the paper reports WHEN adaptive granularity helps (horizon-local training effect, nulls, system-specific novelty directions, cost inversion) instead of claiming a method advantage. The binding section plan and claim registry (C1–C10) live in `writing_outline_boundary_paper.md`.

_Avoid_: Method-advantage framing, "BG-NS-JEPA wins", softening the issue-15 negative.

**Horizon-resolved training effect**

The issue-77 N1 diagnostic contrast `hybrid_continuous_h1` vs `continuous_h1` (kind `training_effect`): +0.0887 [+0.0155, +0.2118], descriptive, 4 held-out states. Distinct from symbolic-execution effects, which straddle zero at h=1.

_Avoid_: Calling it a symbol-conditioning or controller advantage.

**Appearance-novelty boundary**

The issue-77 N2 evaluation over matched normal/novel appearance pairs: zero-shot and few-shot conditions reported separately; type010102 effects move in opposite system-specific directions (hybrid-continuous h15 −0.2693 vs hybrid-macro h15 +0.4242).

_Avoid_: Any uniform "novelty helps/hurts hybrids" claim.

**ADD-EXP boundary tickets (#78–#81)**

The 2026-09-21 boundary-evidence follow-ups: #78 external temporal-adaptation baselines **closed 2026-09-21** (method-class `readiness_or_precision_insufficient`; work-reported and training-effect `supported`); #79 CLEVRER out-of-family replication **executed with published dispositions** (component-wise: S1 short-horizon separation and S2 horizon-ordering REPLICATE; S3 growth-shape fails at h15 non-monotone; Q2 regime direction reversed); #80 reactive-control diagnostic **executed with a typed stop** (`readiness_or_precision_insufficient`: first-shot prevalence 0.0000 over 45 valid executions, full 288-cell matrix barred by the frozen precondition — never present the stop as planner-quality evidence); #81 pooled synthesis **unblocked** (all upstreams terminal) but not executed. New open follow-ups: #82 oracle-ceiling zero-floor diagnostic, #83 CLEVRER h15 non-monotonicity decomposition, #84 third-family replication tie-breaker. #78/#79/#80 outcomes feed registry rows C8/C9/C10; #82/#83/#84 feed C14–C16; #81 feeds the synthesis section itself.

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
