# Research-ready cohort v2 capability profile — proposal

**Date:** 2026-08-20 (Australia/Melbourne)  
**Status:** proposal for scientific approval; not an authorization to collect, relabel, publish, train, evaluate, or change a GitHub issue  
**Purpose:** freeze the smallest cohort capability set that can honestly test GitHub issue #1's central joint-controller claim

## Recommendation

Define cohort v2 around one confirmatory experiment: an **oracle/privileged-symbol, observation-backed joint controller** chooses one exclusive `(requested horizon, description mode)` pair from continuous, micro, and macro modes. The central result is a matched-compute reduction in endpoint prediction error against the strongest frozen comparator, subject to a fixed endpoint safety margin. This is the claim in issue #1, “Solution” and “Implementation Decisions”; learned perception, reliability estimation, and SPSG are explicitly separable supporting hypotheses there.

The minimum symbolic and safety vocabulary should be:

- micro: `contact(A,B)` and `supports(A,B)` only;
- macro: `structure-unstable` and `steady-state` only;
- endpoint physical violations: `excess_penetration` and `unsupported_stationary_or_floating_body` only.

The central generalization regime should be **instance-held-out complete scenario lineages within one declared physics family and predicate ontology**. All four exposure roles are required. Template identities remain mandatory provenance, but template-held-out evaluation is outside cohort v2 unless separately approved. Agent RGB remains the observation used to form the continuous carrier; oracle micro/macro values come from source-bound engine derivations. Canonical observations remain access-restricted alignment/diagnostic evidence and are never model input.

Physical-regime labels, model-relative micro-usefulness, learned visual predicate parsing, SPSG/TPR, bounded physics-validated negatives, novel-material generalization, gravity-shift generalization, damage semantics, illegal-contact labels, planning, and cross-domain evaluation are **not requirements of the central experiment**. Some are named secondary experiments below; none may delay or silently redefine the central result.

This recommendation follows issue #1, especially “Solution,” user stories 15–20 and 32–45, and its “Implementation Decisions.” It deliberately narrows the older component bundle in `docs/research_proposal.md` §§4.2–6.7 and `docs/training_mechanism_and_architecture_specs.md` §§3–6, 10–11.

## Authority and evidence baseline

Primary sources used:

- GitHub issue [#1](https://github.com/Sino-Huang/NovPhy/issues/1), especially “Solution,” “Implementation Decisions,” “Testing Decisions,” and “Out of Scope.” This is the current model/experiment specification.
- GitHub issue [#18](https://github.com/Sino-Huang/NovPhy/issues/18), especially §§1–14, §16, and §17. This is the normative collection specification; `CONTEXT.md` is its terminology authority.
- GitHub issue [#33](https://github.com/Sino-Huang/NovPhy/issues/33) and `.claude/project-docs/evidence/issue-33-section-16-audit-20260820/README.md`. The audit, not issue completion comments, is the current capability-gap authority.
- `docs/data_generation_and_collection_spec.md` §§1–17, the repository copy of the collection design.
- `docs/data_contracts/physics_capture_v1.md`, especially “Authority and alignment,” “State records,” “support_v1,” “Event taxonomy,” and “physics_violation_engine_evidence_v1.”
- `docs/data_contracts/physics_macro_labels_v1.schema.json`, especially the vocabulary and semantic-status declarations.
- `docs/research_proposal.md` §§4.2–4.6 and 6.1–6.7; `docs/training_mechanism_and_architecture_specs.md` §§3–6 and 10–11; `docs/high_level_plans/bg_ns_jepa_research_execution.md` §§2–6. These retain experiment intent but conflict with the newer issue #1 in places listed below.
- Deprecated design history was checked only for conflicts: `docs/deprecated/high_level_plan/research_project_plan.md` and `docs/deprecated/implementation_plan/discussion_about_method_details.md`. The presentation itself marks these superseded where they differ (`docs/method_proposal_presentation/method_proposal_presentation.md`, “Source documents scanned”).

The issue #33 audit establishes a strict starting point. The authoritative audit README has SHA-256 `ed02915ae861f2268a830f61f5b9cfe1d6f16b8bc41afb6ca74caf46126d6841`. The immutable publication `representative-cohort-publication-v1:sha256:a6daf82d47f7001e8731068c68a91e83487fd2c26926b35ab2974bc75a93ecf8` and release `representative-cohort-release-v1:sha256:40b997354a256f889ef7dd007888b5ad8d84b5266883f3611500086f22b62ed2` accept only `physics_capture_v1`, relational supervision, `steady-state`, and `structure-unstable`. The release is not representative: its pilot records `coverage.audit.representative=false`. Physical-regime and physical-violation labels, access-separated observations, template-held-out evidence, fixed-step-stride authority, bounded negatives, deterministic replay, material/damage mappings, and three macro predicates remain unavailable or rejected. Those dispositions remain unavailable/rejected; this proposal does not convert them to false or accepted.

## Classification rule

Every capability is assigned exactly one of these statuses:

- **REQUIRED — CENTRAL:** absence blocks the central joint-controller experiment.
- **REQUIRED — SECONDARY: _name_:** required only if that named experiment is approved and run.
- **OPTIONAL:** may be retained or reported but cannot gate or strengthen the central conclusion.
- **OUT OF SCOPE:** must not be collected or claimed as a cohort-v2 research capability without a separately approved successor profile.

“Required” always means demonstrated by accepted source evidence and fail-closed ingestion, not merely present in a schema, filename, fixture, test, command, or completion comment. Issue #18 §§1, 11–16 and the issue #33 audit impose that distinction.

## Complete scientific/model capability profile

| Capability mentioned by issue #1 | Classification | Cohort-v2 disposition and minimum consequence | Source |
|---|---|---|---|
| Shared continuous rollout carrier | **REQUIRED — CENTRAL** | Every mode emits the same continuous carrier contract; hard symbol decoding never becomes rollout state. | #1, Implementation Decisions 3–6 |
| Exclusive joint action `(requested horizon, description mode)` | **REQUIRED — CENTRAL** | Exactly one of continuous/micro/macro is selected per decision. Requested and terminal-clamped effective horizon remain distinct. | #1, Solution; Implementation Decisions 2, 8, 16 |
| Pair-conditioned shared predictor with mode-specific input/readout behavior | **REQUIRED — CENTRAL** | Horizon and mode must change carrier computation; symbolic content and mode identity must be independently ablatable. Exact layer layout is not a data capability. | #1, Implementation Decisions 4–9; Testing Decisions |
| Continuous description mode | **REQUIRED — CENTRAL** | Uses the agent-observation-derived carrier and action context; remains the no-symbol comparator surface. | #1, Implementation Decision 5 |
| Micro description mode | **REQUIRED — CENTRAL** | Uses only accepted `contact` and `supports` content plus source kinematics/geometry features. | #1, Implementation Decision 5; #18 §12.1 |
| Macro description mode | **REQUIRED — CENTRAL** | Uses only accepted `structure-unstable` and `steady-state` content and predicts a declared event endpoint. | #1, Implementation Decision 5; #18 §12.2 |
| Exhaustive, provenance-bound evaluation of every admissible pair | **REQUIRED — CENTRAL** | Every eligible state receives every supported pair score; unavailable pairs fail closed with a capability reason. | #1, Implementation Decisions 10–11; Testing Decisions |
| Trajectory-level dynamic-programming controller labels | **REQUIRED — CENTRAL** | Backward DP, duration-weighted segment costs, deterministic ties, and complete-path coverage replace the older per-state argmin teacher. | #1, Solution; Implementation Decisions 13–14 |
| Myopic per-state teacher | **REQUIRED — SECONDARY: sequential-label ablation** | Retained only to measure the effect of DP; never the default label source. | #1, Implementation Decision 13 |
| Initial oracle-state distillation and limited closed-loop aggregation | **REQUIRED — CENTRAL** | DP labels are generated on oracle trajectories; one or two predeclared aggregation rounds use **training-role lineages only**. No calibration/model-selection/final lineage may enter learned parameters. | #1, Implementation Decision 22; #18 §6 |
| Fixed, temporal-only, description-only, independently optimized axes, and matched-capacity two-head comparators | **REQUIRED — CENTRAL** | Comparator construction and the strongest comparator are frozen before final access. This is necessary to test non-separable pair utility rather than output-head shape. | #1, user stories 3–7; Implementation Decisions 19–20, 26 |
| Matched policy-dependent compute and full end-to-end compute | **REQUIRED — CENTRAL** | Charge controller, selected adapters/readouts, graph work when executed, transitions, and infilling when executed; report shared perception separately in full-system compute. Wall clock is secondary. | #1, Implementation Decisions 17–18 |
| Endpoint prediction error and fixed endpoint safety margin | **REQUIRED — CENTRAL** | The predeclared gain threshold and two-predicate violation margin are fixed from non-final pilot/calibration evidence before final evaluation. Failure at all declared budgets falsifies the claim. | #1, Solution; Implementation Decision 26 |
| Dense-path physical plausibility and charged infilling | **REQUIRED — SECONDARY: dense-path plausibility** | No dense-path result exists unless a declared infiller runs and its compute is charged. It is not needed for the endpoint-only central rule. | #1, Implementation Decisions 15, 18 |
| Separately trained pair experts | **OPTIONAL** | Diagnostic upper bound for shared-training interference; cannot become the central comparator after final access. | #1, Implementation Decision 21 |
| Recurrent controller | **OPTIONAL** | Architecture ablation only. | #1, Out of Scope/default-method exclusions; `docs/research_proposal.md` §6.3 |
| Physical-regime descriptor/gate | **REQUIRED — SECONDARY: regime-alignment diagnostic** | Not a central input, target, gate, or selection-accuracy requirement. If run, it is an engine-derived label distinct from usefulness. | #1, user stories 15–20; Implementation Decisions 8–9; #18 §12.4 |
| Model-relative micro-relation usefulness and reliability estimator | **REQUIRED — SECONDARY: reliability-gating** | Usefulness is generated by a frozen-model out-of-sample with/without-micro comparison; it may weight micro loss or supply a compressed feature. Remove the estimator if neither use adds held-out value. | #1, Implementation Decisions 8–9; #18 §12.4 |
| Learned visual predicate parser and learned-vs-oracle gap | **REQUIRED — SECONDARY: learned-symbol stress test** | Parser is a stress test, not the paper's central novelty. It requires agent observations, oracle parser targets, object correspondence, and canonical alignment diagnostics. | #1, Implementation Decisions 23; user stories 39–41 |
| SPSG/GINE/TPR interface | **REQUIRED — SECONDARY: symbolic-interface ablation** | Compare no-symbol, ordered flat predicates, directed GNN, and SPSG at matched capacity. No role-binding generalization claim follows automatically. | #1, Implementation Decision 24; Out of Scope |
| Physics-validated contrastive negatives | **REQUIRED — SECONDARY: SPSG contrastive-loss ablation** | Cohort v2 permits only explicit validated anti-support under a frozen bounded-negative plan. Reversed-gravity and massless-material counterfactuals are out with their corresponding studies. A simulator or capture failure is never a negative. | `docs/research_proposal.md` §4.2; #18 §8; narrowed by this proposal |
| Symbol-noise, predicate-flip, and predicate-drift studies | **REQUIRED — SECONDARY: learned-symbol stress test** | Apply only to learned parsing/fine-tuning; they do not gate the oracle central experiment. | #1, user stories 40–41; `docs/research_proposal.md` §§4.6, 6.2 |
| End-to-end discrete controller relaxation or extractor fine-tuning | **OPTIONAL** | Stage-4 variants with separately frozen acceptance criteria; not part of default methodology. | #1, Out of Scope; `docs/research_proposal.md` §6.4 |
| Terminal-anchor, relational projection, free-streaming/interaction split, PDDL serialization | **OPTIONAL** | Mechanism/interface ablations only; no central capability or planner claim depends on them. | `docs/method_proposal_presentation/method_proposal_presentation.md`, “Optional mechanisms” |
| Novel physical laws, unseen predicate vocabularies, ontology changes | **OUT OF SCOPE** | The central claim is limited to new configurations inside one physics family and ontology. | #1, Implementation Decision 25; Out of Scope |
| Planning, task success, shots-to-success | **OUT OF SCOPE** | Explicitly removed from the model and experimental design. | #1, Implementation Decision 27; Out of Scope |
| Cross-domain Physhion/CLEVRER evaluation | **OUT OF SCOPE** | Requires a later separately approved capability profile after NovPhy evidence. | #1's narrowed claim; conflict with `docs/research_proposal.md` §6.4 |

## Complete cohort/data capability profile

| Capability mentioned by issue #18 | Classification | Cohort-v2 disposition and minimum consequence | Source |
|---|---|---|---|
| Benchmark condition → template → instance → specification → lineage hierarchy | **REQUIRED — CENTRAL** | Every lineage has immutable complete identities and belongs to exactly one partition and exposure role. Template identity is retained even though template-held-out scoring is not central. | #18 §§3, 5–6 |
| Deterministic scenario realization and explicit `legacy_static` provenance | **REQUIRED — CENTRAL** | Generated instances must reproduce from declared inputs. Legacy instances remain explicitly legacy and never acquire inferred template/generation facts. | #18 §5 |
| Version-bounded deterministic replay | **REQUIRED — CENTRAL** | QA replay must reproduce scenario, intervention, initial state, identities, and deterministic artifact semantics inside the frozen engine/player/protocol/generator/plan envelope. A new version requires a new pilot decision. | #18 §14 |
| Benchmark-agent action as an intervention source | **OPTIONAL** | Include only when an action and provenance are available. Its absence is an explicit unavailable source disposition, not a failed central capability. Geometry-aware and targeted sources remain central. | #18 §8 |
| Single-shot reset/intervention/termination semantics | **REQUIRED — CENTRAL** | Identical declared initial state per lineage, one pre-intervention frame, exactly one intervention, and a final frame covering one declared termination. | #18 §7 |
| Frozen, outcome-independent collection plan | **REQUIRED — CENTRAL** | Freeze identities, order, retry/stopping rules, roles, strata, and quotas before outcomes. No cherry-picking, outcome-based replacement, or post-hoc quota filling. | #18 §8 |
| Central target coverage strata | **REQUIRED — CENTRAL** | Prospectively cover `no-contact/miss`, `collision`, `persistent support`, `support change`, `destruction`, and `stability transitions`. These exercise the central micro/macro/safety vocabulary. | #18 §8; narrowed by this proposal |
| Pig removal, explosion, level clear, level fail target strata | **REQUIRED — SECONDARY: task/outcome-event extension** | Retain any naturally observed event as raw evidence, but do not make these targeted production quotas for central v2. Approval of `pigs-cleared`, explosion, or task-outcome experiments makes the applicable target mandatory. | #18 §8; conflict with the minimal issue #1 claim |
| Bounded negative evidence | **REQUIRED — SECONDARY: SPSG contrastive-loss ablation** | Frozen negative specification and cap; failures never become negatives. | #18 §8 |
| Engine-authoritative state, lifecycle, identities, contacts, events, geometry, kinematics, mass, baseline gravity/world values | **REQUIRED — CENTRAL** | Complete source evidence must be captured without RGB reconstruction. Baseline gravity vector/applicability is required for the floating label; changing gravity is not. Raw `life` remains uninterpreted. | #18 §§4, 9, 13 |
| Material mapping and novel-material identities | **REQUIRED — SECONDARY: novel-material generalization** | Do not infer material from names, paths, sprites, RGB, or class. Recommend excluding this secondary from cohort v2. | #18 §§9, 12.3; issue #33 condition 5 |
| Damage mapping | **OUT OF SCOPE** | Preserve raw `life`; do not relabel it as damage/health. A future damage study requires a new accepted engine mapping and profile. | #18 §§9, 12.3 |
| Gravity-shift scenarios/generalization | **OUT OF SCOPE** | Record baseline gravity/world parameters for provenance and violation availability, but do not vary gravity or claim unseen-physics/OOD generalization. | #1, Out of Scope; conflict with older experiment docs |
| Raw contacts at every fixed step and fixed-step event authority | **REQUIRED — CENTRAL** | Contacts/events are complete across transition intervals regardless of frame capture stride; render frame/time is provenance only for event occurrence. | #18 §§4, 9–10 |
| Positive-integer fixed-step capture stride | **REQUIRED — CENTRAL** | Freeze prospectively; final terminal frame is the only declared exception. A target-FPS command is not stride authority. | #18 §10 and §17 |
| Atomic whole-rollout validation, quarantine, typed failures, transient-only retries | **REQUIRED — CENTRAL** | No partial/reconstructed admission; permanent semantic/evidence defects are not retried for a luckier outcome. | #18 §11 |
| Agent observation | **REQUIRED — CENTRAL** | Synchronized post-transform RGB used by the observation-backed continuous carrier. Its configuration and identity belong to the lineage. | #18 §§4, 9; #1 compute and oracle-symbol decisions |
| Canonical observation | **REQUIRED — CENTRAL (alignment/QA only)** | Access-restricted synchronized pre-transform observation for alignment/capture diagnosis; never training/model input in the central experiment. Distinct access policy must be tested. | `CONTEXT.md`; #18 §§4, 9 |
| Contact relation | **REQUIRED — CENTRAL** | Symmetric; true only from validated non-trigger raw contact, false only with complete evidence, otherwise unavailable. | `CONTEXT.md`; #18 §12.1 |
| Support relation | **REQUIRED — CENTRAL** | Directed, tri-state, and evidence-window complete. `support_v1` is positive authority; explicit negatives require additional window completeness. | #18 §12.1; `docs/data_contracts/physics_capture_v1.md`, “support_v1” |
| `structure-unstable` and `steady-state` | **REQUIRED — CENTRAL** | The only macro predicates admitted to the central vocabulary. They are currently `engine_verified`, but v2 still requires representative positive/negative/boundary evidence bound to its own release. | #18 §12.2; macro schema; issue #33 condition 5 |
| `cascade-active` | **REQUIRED — SECONDARY: extended macro-event prediction** | Excluded from central training/scoring until a new representative semantic gate accepts positive, negative, boundary, and unavailable cases. | #18 §12.2; current adjudication |
| `collapsed` and `pigs-cleared` | **OUT OF SCOPE** | Task/outcome-heavy predicates are not needed for the central controller claim. Existing all-negative evidence is not semantic acceptance. | #18 §12.2; issue #33 condition 5 |
| `excess_penetration` | **REQUIRED — CENTRAL** | One component of the endpoint safety margin; available only with complete geometry/contact/separation evidence and a frozen tolerance. | #18 §13 |
| `unsupported_stationary_or_floating_body` | **REQUIRED — CENTRAL** | Second component of the endpoint safety margin; requires gravity applicability, lifecycle, motion window, support/contact evidence, and world context. | #18 §13 |
| `illegal_contact` | **OUT OF SCOPE** | Remains unavailable without a legal-contact ontology, mapping, exemptions, and complete evidence contract. It is never an assumed negative. | #18 §§2, 13 |
| Physical-regime gate | **REQUIRED — SECONDARY: regime-alignment diagnostic** | Not required as a central model label. See the acceptance-dependency conflict below for physical-violation adjudication. | #18 §12.4; #1 Implementation Decisions 8–9 |
| Micro-relation usefulness | **REQUIRED — SECONDARY: reliability-gating** | Separate model-relative derivation bound to frozen checkpoint/objective and lineage-disjoint held-out evidence. | #18 §12.4; #1 Implementation Decisions 8–9 |
| `training`, `calibration`, `model_selection`, `final_evaluation` roles | **REQUIRED — CENTRAL** | All four are required for a confirmatory result; permissions are enforced at lineage level. | #18 §6 and §14 |
| Instance-held-out split | **REQUIRED — CENTRAL** | Primary generalization result; level instances and complete scenario lineages are disjoint across roles, while templates may recur. | #18 §6.1; #1 Implementation Decision 25 |
| Template-held-out split | **OUT OF SCOPE** | Template identities remain required provenance, but no v2 template-held-out score or claim is made. A separately approved structural-generalization experiment requires its own manifest and quotas. | #18 §6.2; #1's narrower primary split |
| Capability-complete pilot, prospective quotas, immutable release, source-bound derivations | **REQUIRED — CENTRAL** | A new release is published; existing pilot/release artifacts stay immutable. Only the central capability set must be complete. | #18 §14 and §16 |
| Role-separated final-evaluation workflow/access manifest | **REQUIRED — CENTRAL** | Freeze metrics, checkpoints, derivations, stopping rules, comparator, effect threshold, and safety margin before final access; audit every access. | #18 §14; #1 Testing Decisions |
| Public fail-closed downstream ingestion | **REQUIRED — CENTRAL** | Smoke-ingest every central primary/derivation/access capability with identity, timing, availability, and role assertions. Rejection of a missing capability is not evidence of successful ingestion. | #18 §16.6; issue #33 condition 6 |

## Exact central label semantics

### Micro

1. **`contact(A,B)`** is symmetric and stored canonically as an unordered pair. It is true at a fixed step only when at least one validated non-trigger raw contact joins two active identities. A false value requires complete raw-contact enumeration for that fixed step and the evaluated active pair. Missing, truncated, overflowed, or misaligned contact evidence yields unavailable, not false. This follows `CONTEXT.md`, #18 §12.1, and `docs/data_contracts/physics_capture_v1.md`, “State records.”
2. **`supports(A,B)`** is directed `supporter -> supported`. Positive truth uses the versioned support derivation and its cited persistence/geometry facts. Under current `support_v1`, the same canonical non-trigger pair must be retained in two consecutive fixed steps, `abs(normal_a_to_b.y) >= 0.5`, and the supported body's vertical-center offset must be at least `0.0001` Unity units in both samples. An explicit false value requires the complete assessment window; absence of a positive `support_v1` edge alone is insufficient. This follows #18 §12.1 and the frozen `support_v1` section of the capture contract.

`velocity-bin`, material, and `damaged` examples in `docs/research_proposal.md` §4.3 are not accepted v2 micro predicates. Velocity and geometry remain continuous engine features; raw `life` remains uninterpreted.

### Macro

1. **`structure-unstable`** is the central active/transition macro signal.
2. **`steady-state`** is the central stable endpoint macro signal.

Both are `engine_verified` in `docs/data_contracts/physics_macro_labels_v1.schema.json`. The three other schema predicates remain pending/rejected and are not silently emitted as central false values. The macro derivation remains a separately versioned, source-bound artifact projected by fixed-step bracketing, per #18 §12.2.

The two-predicate vocabulary is sufficient to instantiate a macro description with an unstable-to-stable event endpoint. It avoids making cascade density, geometric collapse, pig clearance, task outcome, or material semantics prerequisites for the non-separable horizon/mode claim.

### Central endpoint safety margin

For each scored endpoint, define the central violation vector as:

`V_endpoint = (excess_penetration, unsupported_stationary_or_floating_body)`.

A central score is admissible only when **both values are available** under the same immutable release and accepted derivation versions. The confirmatory safety statistic should report each predicate separately and a frozen aggregate such as `any(V_endpoint)`. The aggregate rule, tolerance, stability window, and allowed margin are frozen on calibration evidence before final access. An unavailable component makes the endpoint safety score unavailable; it never counts as zero.

`illegal_contact` is excluded. Dense-path violations are a different secondary metric and require charged infilling. These choices implement #1's all-mode endpoint plausibility rule while respecting #18 §13.

### Physical-regime dependency decision

A physical-regime label is **not scientifically required** to train or score the central controller. Issue #1 explicitly separates physical diagnostics from pair-relative utility, and the central confirmatory rule does not require regime-aligned selection accuracy.

There is, however, a current acceptance conflict: `.claude/project-docs/evidence/representative-pilot-20260820/physical-violation-adjudication-v1.json` rejects both candidate labels when an accepted physical-regime derivation is absent, whereas issue #18 §13 defines per-label completeness in terms of geometry, gravity, lifecycle, motion, support/contact, and world context. Before v2 collection, approve one of these prospective options:

- **Recommended:** define those per-label completeness/stability facts inside the new physical-violation derivation, keeping the physical-regime label optional and separate.
- Alternatively, retain a minimal engine-derived regime artifact as an operational dependency of physical-violation acceptance while forbidding it as a central controller input, target, or reported regime-selection metric.

Neither option may revise the existing adjudication artifact or retrofit the current cohort.

## Observation decision

Learned **symbol extraction** is not part of the central experiment, but visual observation is still part of the continuous world-model input:

- agent observation: required central post-transform RGB, synchronized to its engine frame record and available to the model;
- canonical observation: required central collection/QA evidence, pre-transform and access-restricted, used only to verify synchronization, camera/transform behavior, and capture defects;
- oracle contact/support/macro/violation values: engine-derived sidecar derivations, never inferred from either image stream;
- object matching, parser targets, confidence calibration, and learned-vs-oracle predicate metrics: required only for the named learned-symbol stress test.

This resolves issue #1's oracle-first decision without dropping issue #18 §§4 and 9's access-separated observation evidence.

## Exposure roles and generalization

| Role | Required central use | Prohibited use |
|---|---|---|
| `training` | Predictor/controller training, DP-label consumption, and predeclared closed-loop aggregation | Cannot contain a lineage assigned to any other role in the same regime |
| `calibration` | Pilot acceptance, tolerances/windows, compute calibration, practical-effect threshold, and safety-margin selection | Cannot update learned parameters; cannot be reported as final evaluation |
| `model_selection` | Architecture/checkpoint/ablation choice and strongest-comparator selection | Cannot update learned parameters; cannot be final evaluation |
| `final_evaluation` | One frozen confirmatory read after workflow authorization | No influence on training, calibration, comparator choice, thresholds, stopping, or model selection |

The primary split is one separately identified **instance-held-out** manifest. Complete scenario lineages—not frames, transitions, rollouts, reruns, seeds, or observation variants—are the atomic boundary. The final set contains unseen level instances/configurations from the same declared physics family and predicate ontology. Template reuse is allowed by this regime, but each template identity must be recorded so the claim cannot be misreported as template-held-out.

Template-held-out evaluation is out of v2. If later approved, it needs a separate manifest, disjoint held-out template identities and all descendants, separate quotas, and a separately named claim; its result cannot be merged with the instance-held-out score. This follows #18 §6 and #1 Implementation Decision 25.

## Prospective central coverage and intervention profile

The central plan needs coverage that directly exercises the six accepted labels and their evidence windows:

| Central stratum | Why required |
|---|---|
| `no-contact/miss` | Complete negative contact evidence and continuous-mode behavior |
| `collision` | Positive contact, contact boundaries, interaction-active dynamics, penetration evidence |
| `persistent support` | Positive support and stationary supported negatives for floating |
| `support change` | Support formation/loss boundaries and unsupported-body cases |
| `destruction` | Causal lifecycle/identity stress and unstable structure behavior; no damage label implied |
| `stability transitions` | Positive/negative/boundary evidence for `structure-unstable` and `steady-state` |

Geometry-aware feasible shots and targeted rare-interaction shots are required. Provenanced benchmark-agent replay actions are optional because #18 §8 conditions them on availability. **Version-bounded deterministic artifact replay is different and remains required central QA.**

Pig removal, explosion, level clear, and level fail may occur and must remain in raw event/failure/termination accounting, but they are not central target quotas. A secondary task/outcome-event extension must approve the relevant targets prospectively. The central closed termination vocabulary should minimally support post-intervention stable termination and rollout ceiling; any engine terminal outcome that occurs remains retained evidence and must not be hidden. The exact supported termination classes and quotas require approval before the new pilot.

## Minimum representative evidence per accepted label

This is a **semantic acceptance floor**, not a production sample-size calculation. For each accepted label, the capability-complete pilot must contain at least two source-bound positive witness windows and two source-bound negative witness windows spanning at least two independent non-final scenario lineages, two level instances, and two scenario templates, plus at least two boundary windows that straddle the decision threshold/transition. It must also exercise an intentionally incomplete case that correctly becomes unavailable or whole-rollout invalid according to the contract. No final-evaluation lineage may be used to define or accept semantics.

All cases and intervention/coverage assignments are declared before outcomes. Boundary windows retain the fixed step before, at, and after the transition where meaningful. Exact production quotas and numeric tolerances remain open until the capability-complete pilot, then are frozen prospectively as required by issue #18 §17.

| Accepted label | Positive evidence | Negative evidence | Boundary evidence | Unavailable/invalidation check |
|---|---|---|---|---|
| `contact(A,B)` | Active entities joined by retained non-trigger raw contact IDs at the evaluated fixed step | Both active; complete fixed-step contact enumeration; no joining raw contact | Contact begin and end windows, including separated-before/contact-at/separated-after where produced | Remove/truncate/overflow fixed-step contact evidence or break identity/alignment; result unavailable or rollout invalid, never false |
| `supports(A,B)` | Source-bound `support_v1` edge with its two cited fixed steps/contact IDs and complete geometry/persistence facts | Complete support-assessment window establishes non-support for the evaluated directed pair | Support formation and loss; include cases around the two-step persistence and normal/vertical-offset thresholds | Remove predecessor/contact/geometry/lifecycle evidence; result unavailable, never inferred from missing edge or RGB |
| `structure-unstable` | Accepted fixed-step interval with engine-backed unstable transition/state evidence | Complete stable window in which the predicate is false | Retain frame records bracketing entry to and exit from instability, with cited fixed-step events/states | Remove required source event/state interval or stale-bind the derivation; reject/unavailable |
| `steady-state` | Complete debounced stable window and accepted stable endpoint evidence | Complete motion/contact-active or explicitly unstable window | Retain pre-debounce, threshold-crossing, and post-debounce fixed steps around `stable_entered`/`stable_exited` | Missing stability history or terminal coverage produces unavailable/rejection, not false |
| `excess_penetration` | Complete collider/contact geometry and separation below the frozen negative tolerance, with coordinate convention | Same complete evidence establishes separation at or above tolerance | Representative contacts on both sides of and near the frozen tolerance; cite exact contact, fixed step, geometry, and separation | Missing geometry, coordinate declaration, contact completeness, or finite values produces unavailable or invalidation |
| `unsupported_stationary_or_floating_body` | Active dynamic body under applicable nonzero gravity; complete lifecycle/motion window establishes stationary; complete contact/support/world context establishes no valid support | Complete evidence establishes valid support or non-stationarity | Support removal/formation and motion just inside/outside the frozen stability threshold/window under recorded gravity | Missing gravity applicability, lifecycle, motion history, support/contact completeness, or world context produces unavailable or invalidation |

For secondary labels, the same floor applies if approved. In addition, micro-usefulness needs positive benefit, no-benefit/harm, and threshold-boundary cases bound to one frozen preliminary checkpoint/objective on lineage-disjoint held-out evidence; a physical-regime label needs representative examples on both sides of every declared motion/contact threshold; a learned parser needs positive/negative/boundary oracle targets and access-separated observations without using final-evaluation data for calibration.

## Required artifact and acceptance bundle

A central-v2 release is research-ready only if all of the following exist and pass for the **approved central capability list**:

1. frozen collection and production plans with the central strata, intervention identities, retry/termination rules, capture stride, roles, instance-held-out manifest, and quotas;
2. capability-complete representative pilot with deterministic version-bounded replay, access-separated observations, all label evidence floors, and visible unavailable/rejected cases;
3. atomic accepted rollouts plus immutable typed failure/quarantine accounting;
4. immutable cohort release binding scenario manifests, primary engine/observation traces, provenance, partitions/roles, quality report, and exact digests;
5. separately versioned accepted derivations for contact/support supervision, the two macro predicates, and the two violation predicates;
6. frozen final-evaluation workflow/access manifest;
7. public fail-closed ingestion evidence requiring every central capability, including observation and access restrictions;
8. no known systematic exporter defect in any central capability.

The current pilot and release remain valuable immutable evidence for their accepted scope but cannot be upgraded in place. This follows issue #18 §§11, 14, and 16 and every condition in the issue #33 audit.

## Conflicts that approval must resolve

1. **Controller teacher:** `docs/research_proposal.md` §§4.4/6.3 and the presentation describe per-state argmin labels; issue #1 requires trajectory-level dynamic programming and retains myopic labels only as an ablation. Adopt issue #1.
2. **Exclusive mode action:** `docs/research_proposal.md` §4.3 says macro prediction and continuous infilling may be active simultaneously; issue #1 says exactly one description mode is active per decision, with optional charged infilling treated separately. Adopt issue #1.
3. **Reliability target:** older docs use a KE/contact oracle regime as the learned gate target. Issue #1 requires model-relative micro-constraint usefulness and separates it from engine physical regime. Adopt issue #1; keep both secondary.
4. **Central result:** older docs require regime-aligned selection, three violation rates, novel-material/gravity OOD, and sometimes planning. Issue #1 narrows the confirmatory rule to endpoint error at matched compute under a fixed all-mode endpoint violation margin and explicitly excludes planning/unseen physics. Adopt issue #1.
5. **Macro vocabulary:** older execution documents assume five predicates and event/outcome F1. Current accepted semantics include only `structure-unstable` and `steady-state`; `cascade-active`, `collapsed`, and `pigs-cleared` are rejected/pending. Central v2 uses only the accepted two.
6. **Material/damage:** older node features and negative sampling assume material and health/damage. Issue #18 §12.3 and the issue #33 audit show mappings are unavailable. Material becomes conditional on a named secondary; damage is out.
7. **Learned perception:** older “full model” staging can make the visual parser look required. Issue #1 calls it a later stress test whose failure narrows rather than redefines the central claim. Central v2 remains oracle-symbol but observation-backed.
8. **Generalization:** issue #18 defines both instance- and template-held-out regimes, while issue #1 requires complete held-out scenarios/configurations but not template novelty. Declare only instance-held-out applicable to central v2; retain template identities and make any template-held-out claim separate.
9. **Coverage breadth:** issue #18 §8 lists task/outcome strata for a general enriched cohort. The minimal issue #1 claim needs six central strata; pig removal, explosion, clear, and fail are secondary dispositions, not silent omissions. A later GitHub specification update would be needed after approval.
10. **Violation/regime acceptance:** the current physical-violation adjudication requires an accepted physical-regime derivation, while the normative per-label rules can be read as complete without a separately reported regime label. Resolve prospectively before pilot freeze; do not edit existing evidence.
11. **Deprecated documents:** the deprecated high-level and implementation plans still discuss phase gating, planning, material/gravity OOD, and broader scene features. They are design history only and must not govern cohort-v2 acceptance.

## Scientific decisions requiring approval

1. **Central claim boundary:** approve the oracle/privileged-symbol, observation-backed, exclusive three-mode joint-controller experiment as the only cohort-v2 confirmatory claim.
2. **Vocabulary:** approve micro `{contact, supports}` and macro `{structure-unstable, steady-state}` as sufficient; reject `velocity-bin`, damage, `cascade-active`, `collapsed`, and `pigs-cleared` as central requirements.
3. **Safety margin:** approve `{excess_penetration, unsupported_stationary_or_floating_body}`, require both available at every scored endpoint, and exclude `illegal_contact`.
4. **Physical-regime dependency:** approve the recommended per-violation completeness design without a central physical-regime label, or require a minimal regime artifact only as an operational derivation dependency.
5. **Perception/observations:** approve oracle predicates for the central result, agent RGB for the continuous carrier, canonical RGB for restricted alignment/QA only, and learned visual parsing as a named secondary.
6. **Generalization:** approve instance-held-out within one physics family as the sole central split and template-held-out as out of cohort v2 pending a separate claim.
7. **Novel material question:** approve the recommendation that novel-material generalization does **not** remain part of cohort v2. If retained, name it a secondary experiment and accept the material-mapping, material-stratified split, representative evidence, and additional collection cost it requires.
8. **Other scope:** approve damage, gravity-shift OOD, illegal contact, planning, and cross-domain evaluation as out of cohort v2; approve SPSG/reliability/dense-path/learned-parser work as secondary only.
9. **Coverage/replay:** approve the six central strata, deterministic version-bounded replay as required QA, benchmark-agent replay actions as optional, and bounded negatives only for the SPSG secondary.
10. **Evidence floor and termination vocabulary:** approve the two-lineage/two-instance positive-negative-boundary floor per accepted label and prospectively choose the exact supported central termination classes and numeric quotas from the capability-complete pilot.

After these decisions, the next work should be a GitHub specification reconciliation and a new prospective pilot/collection plan. This proposal itself changes no issue, code, pilot, cohort, release, derivation, or access state.
