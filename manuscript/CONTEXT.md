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

The sealed role for frozen final metrics. Six rollouts have been collected and sealed, but no final-evaluation metric has been derived or consumed because authorization is pending.

_Avoid_: Unrun, Unavailable, authorized result.

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

The accuracy of a common final-state readout against the engine-defined terminal outcome. It is a Specified endpoint. No final-evaluation result is available while authorization is pending.

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

**Issues #62 through #65**

Open successor work. Issue 62 is successor multi-shot cohort generation and data work. Issue 63 is a matched carrier-alignment by training-coverage experiment with no result. Issue 64 is fresh sealed gameplay benchmark generation, contingent on a supported issue-63 candidate and a nonzero pilot, with no benchmark or result yet. Issue 65 is a future sealed matched gameplay test separating retraining and adaptive-granularity claims, with no result. Their outcomes remain `[TODO: result]`.

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

Cannot proceed because an explicit prerequisite or authorization is unmet. Final metric derivation and consumption are Blocked by pending final-evaluation authorization.

_Avoid_: Negative result, Unavailable, unrun.

**Unavailable**

The required evidence is absent or the concept is excluded for the relevant record. It is neither true nor false.

_Avoid_: False, missing-as-negative.
