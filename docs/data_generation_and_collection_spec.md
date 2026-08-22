# Data Generation and Collection Research Specification

**Status:** normative research specification.  
**Scope:** generation, collection, validation, partitioning, release, and source-bound supervision for NovPhy research data.  
**Normative language:** **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are requirements terms.  
**Terminology authority:** canonical terms have the meanings defined in root `CONTEXT.md`.

This specification complements the frozen `physics_capture_v1` contract in `docs/data_contracts/physics_capture_v1.md` and the BG-NS-JEPA method documents, especially `docs/research_proposal.md`, `docs/training_mechanism_and_architecture_specs.md`, and `docs/high_level_plans/bg_ns_jepa_research_execution.md`. Where this specification is stricter than an existing artifact validator, the stricter rule governs admission to a research cohort; it does not silently change the frozen artifact contract.

### Approved central cohort-v2 scope

An artifact that requests `central_v2` status MUST reference `docs/data_contracts/cohort_v2_capabilities_v1.json` by its exact declared identity `cohort-v2-capabilities-v1`. The declaration's authorities are GitHub issues #1, #18, #33, #42, and #43; deleted local evidence files are not authorities.

The declaration is the normative scope overlay for the central joint-controller cohort. It classifies every named capability as central, named-secondary, optional, or out of scope. Producers, collection plans, releases, derivations, and consumers MUST reject a `central_v2` request with an unknown capability, an omitted central capability, a promoted non-central capability, or the wrong declared identity. A release, derivation, or consumer MUST additionally reject central status while any required central capability is unavailable.

The central label vocabulary is exactly micro `{contact, supports}`, macro `{structure-unstable, steady-state}`, and endpoint violations `{excess_penetration, unsupported_stationary_or_floating_body}`. Central collection also requires synchronized agent observation, access-restricted canonical observation, version-bounded deterministic replay, all four exposure roles, the instance-held-out split, the six prospective strata `no-contact/miss`, `collision`, `persistent support`, `support change`, `destruction`, and `stability transitions`, and the provenance/ingestion capabilities enumerated by the declaration.

Template-held-out evaluation, a central physical-regime label, bounded negatives, material/damage supervision, gravity-shift generalization, `illegal_contact`, planning, and cross-domain claims are non-central with the exact dispositions in the declaration. Template identity, baseline gravity applicability, and raw `life` remain required source facts where the central provenance or violation contracts need them; they do not establish template-held-out, gravity-shift, or damage capabilities.

The named-secondary capabilities are bounded-negative evidence, material labels, `cascade-active`, learned symbol parsing, micro-relation usefulness, and the physical-regime gate; each is required only for the experiment named in the declaration. Benchmark-agent intervention replay is optional. Template-held-out evaluation, damage labels, gravity-shift generalization, `collapsed`, `pigs-cleared`, `illegal_contact`, planning, and cross-domain evaluation are out of scope for central v2. None may gate, strengthen, or be reported as part of the central claim without a separately approved successor declaration.

For each accepted central label, representative semantic evidence MUST contain at least two positive witnesses and two negative witnesses spanning at least two non-final scenario lineages, two level instances, and two scenario templates, plus at least two boundary windows and an unavailable/invalidation check. `unavailable` is not false. Availability MUST NOT be inferred from a filename, RGB content, a fixture, command success, or closed-issue status. Existing pilot, cohort, release, derivation, and audit artifacts remain immutable and do not acquire central-v2 status from this declaration.

## 1. Purpose and scientific claim boundary

The purpose of this specification is to ensure that NovPhy data can support controlled research on BG-NS-JEPA's continuous, micro-relational, and macro-event descriptions without leakage, post-hoc sampling, ambiguous timing, or unsupported oracle claims.

A conforming collection MAY support claims about prediction, physical plausibility, representation usefulness, exposure-controlled generalization, and the joint choice of prediction horizon and description level. It MUST NOT, by itself, be treated as evidence that BG-NS-JEPA works, that a joint controller outperforms alternatives, that symbolic supervision is useful, or that any physical-violation detector is valid. Those claims require the role-separated experiments and acceptance evidence specified by the BG-NS-JEPA method documents.

The continuous latent remains the sole model rollout state carrier. Engine state, contacts, events, and synchronized observations define collected evidence; model latents and model-specific training examples are derivations from that evidence.

### Acceptance criteria

- Every published claim MUST identify the cohort release, exposure regime, derivation artifact versions, and evaluation workflow used.
- No continuous-only, RGB-only, fixture-only, command-success, or smoke-only result MAY be cited as evidence for micro/macro supervision, physical-violation validity, representative coverage, or the joint description-level claim.
- A successful smoke level MUST be treated only as bounded operational evidence for the capabilities it actually exercised.

## 2. Explicit non-goals

This specification does not:

- prescribe a model architecture, optimizer, training schedule, or implementation plan;
- authorize player publication, cohort collection, training, or final evaluation;
- redefine the frozen `physics_capture_v1` schema or event taxonomy;
- retroactively enrich `legacy_rgb_v1` data or infer missing engine facts from RGB;
- make PDDL or any other symbolic serialization the primary data representation;
- define a universal physical-plausibility theory;
- define an `illegal contact` oracle before its legal-contact ontology and required evidence are accepted;
- permit outcome-conditioned intervention selection, retry selection, quota filling, or release curation;
- treat a generated command, a passing validator, a staged archive, or a smoke rollout as a representative cohort.

## 3. Canonical hierarchy and artifact boundaries

The canonical hierarchy is:

1. A **benchmark condition** declares one novelty-level and novelty-type pair.
2. A **scenario template** is a reusable structural source associated with one or more benchmark conditions.
3. A **level instance** is one concrete generated or legacy-static layout whose complete contents form a **scenario specification**.
4. A **scenario lineage** contains that scenario specification and every rollout, seed, intervention, observation configuration, and rerun derived from it.
5. A **scenario collection** contains the independently executed rollouts collected from one level instance under one declared intervention plan.
6. A **cohort** contains accepted, partitioned complete rollouts governed by one data contract and provenance envelope.
7. A **cohort release** is the immutable publication boundary for the cohort, its collection plan, partitions, provenance, and accepted derivation references.

Artifact boundaries are normative:

- The scenario specification MUST be immutable within its scenario lineage.
- A rollout MUST be a primary collection artifact and MUST NOT contain model-specific training examples.
- An engine trace and its synchronized observation trace MUST remain primary evidence.
- Oracle labels, parser targets, physical-violation labels, physical-regime gates, micro-relation usefulness labels, and model latents MUST be separate derivation artifacts.
- A derivation artifact MUST identify its source cohort release and exact source records. It MUST NOT rewrite or replace the source engine trace or observation trace.
- A scenario lineage MUST belong to exactly one dataset partition and one exposure role within an exposure regime.

### Acceptance criteria

- No rollout identity appears under more than one dataset partition or exposure role in the same exposure regime.
- No source-bound derivation artifact can be validated against a different cohort release or changed source record set.
- Training-example regeneration does not alter the cohort release.

## 4. Source of truth and temporal alignment

The simulation engine is authoritative for physical state changes, object lifecycle, contacts, events, and fixed-step time. The engine trace MUST be the source of truth for physical supervision. The observation trace MUST be treated as synchronized sensor evidence, not physical truth.

Each frame record MUST describe the complete post-step engine state and synchronized observations at one fixed step. Exact agent/canonical RGB alignment MUST use the request-72 `synchronized_observation_endpoint` and `observation_trace_manifest_v1`; desktop screenshots and ordinary screenshot requests MUST NOT be represented as exact canonical observations. The frozen `physics_capture_v1` synchronized RGB/state response remains valid only within its declared v1 scope.

Event occurrence MUST be located by fixed step and fixed time. Event render frame and render time MAY be retained as provenance, but MUST NOT be used as event occurrence authority or as an event-to-state join key. Events MUST be projected to frame records by fixed-step bracketing.

### Acceptance criteria

- Every accepted enriched frame record has one exact synchronized state/observation identity.
- Every event maps deterministically to a fixed-step transition interval, including events serialized with a later common render frame.
- A render-frame-only event join fails validation for research use.

## 5. Benchmark-compatible deterministic scenario generation

Scenario generation MUST preserve the lineage `benchmark condition -> scenario template -> level instance -> scenario specification -> scenario lineage`. Every generated level instance MUST record generator identity, generator version, input scenario template identity, benchmark condition, generation seed, deterministic parameter realization, and content identity sufficient to reproduce or detect drift in the complete scenario specification.

Given identical generator version, scenario template, benchmark condition, seed, and declared generation inputs, generation MUST produce the same scenario specification. Randomness MUST be local to the declared generation operation and MUST NOT depend on process scheduling, prior generation order, or outcome data.

`legacy_static` MUST be used as the generation-mode value for an existing benchmark layout that was not produced by the deterministic generator. A `legacy_static` level instance MUST still receive benchmark condition, scenario template or explicitly unavailable template lineage, level-instance identity, scenario-specification identity, and provenance. It MUST NOT be misrepresented as generated.

The currently staged `novelty_level_0/type2/Levels/3_9_6_1.xml` content is staged for bounded runtime/smoke work only. Its `type2` location MUST NOT be interpreted as a complete benchmark novelty-type taxonomy, representative production inventory, accepted scenario-template family, or production-ready generation source. Until representative `type2` lineage and semantics are accepted, `type2` MUST have `smoke_only` eligibility and MUST NOT enter training, calibration, model selection, or final evaluation cohorts.

Current collection planning deterministically partitions discovered level paths within novelty/type buckets and supports an explicitly scoped inventory. That behavior is implemented. Canonical benchmark condition, scenario template, level-instance, scenario-lineage, exposure-role, and generation-mode records required by this specification are not established by that path and MUST be added before research-conforming generation.

### Acceptance criteria

- Repeated generation from identical declared inputs yields an identical scenario specification.
- Every level instance resolves to exactly one benchmark condition and exactly one scenario lineage.
- `legacy_static` records are distinguishable from generated records without inspecting path names.
- A staged `type2` smoke artifact is rejected from every research exposure role while its status is `smoke_only`.

## 6. Exposure roles and split regimes

Each scenario lineage MUST receive one exposure role chosen from:

- `training`: MAY influence learned parameters;
- `calibration`: MAY influence thresholds, calibration maps, and declared pilot decisions, but MUST NOT influence learned model parameters unless separately assigned to training in a different, predeclared experiment;
- `model_selection`: MAY influence architecture, checkpoint, hyperparameter, and ablation selection, but MUST NOT be reported as final evaluation;
- `final_evaluation`: MUST remain inaccessible to training, calibration, and model-selection decisions until the final-evaluation workflow is frozen.

Every release intended to test generalization MUST declare one of these split regimes:

### 6.1 Instance-held-out

Scenario templates MAY occur across exposure roles, but level instances and scenario lineages MUST be disjoint. No seed, intervention rerun, observation configuration, or derived rollout from a held-out level instance MAY occur in another role.

### 6.2 Template-held-out

Scenario templates assigned to held-out roles MUST be disjoint from scenario templates assigned to training, calibration, and model selection. All level instances and scenario lineages derived from a held-out scenario template inherit its held-out boundary.

Where both regimes are reported, they MUST use separately identified partition manifests and MUST NOT be conflated into one score. Splitting by frame record, transition record, rollout, or intervention while sharing a scenario lineage across roles is prohibited.

### Acceptance criteria

- An automated lineage audit finds no cross-role scenario lineage in either regime.
- The template-held-out audit finds no scenario-template identity shared across the prohibited role boundary.
- Final-evaluation records are unread by ordinary training, calibration, and model-selection workflows before release authorization.

## 7. Single-shot rollout semantics

Each rollout MUST be one independently executed single-shot simulation.

Before each rollout, the engine MUST reset to the declared scenario specification and clear prior runtime entities, contacts, events, timers, score, camera state where scenario-defined, and recorder state. The reset MUST NOT reuse post-shot state from another rollout. All rollouts used to vary interventions within one scenario lineage MUST begin from an identical declared initial engine state. Any seed or other input that changes the initial engine state defines a different scenario specification and therefore a different scenario lineage.

The first frame record MUST be captured after reset and stabilization requirements are satisfied and before the intervention. Exactly one intervention MUST then be applied. No second shot, policy continuation, or hidden corrective input is permitted in the same rollout.

A rollout MUST terminate with exactly one declared termination reason from the collection plan's closed termination vocabulary. Termination MUST be caused by an engine terminal outcome, a declared post-intervention stable condition, bird exhaustion where applicable, or the declared rollout ceiling. Operator interruption, capture failure, and invalid data are collection failures, not scientific termination reasons.

The final frame record MUST be the post-step state at or after the fixed step that establishes the termination reason. A terminal event after the last retained frame record does not satisfy this collection specification, even if an older artifact contract can represent such an event.

Current collectors implement shot-local reset hooks, pre-shot guarding, one shot per planned action, bounded capture, and artifact validation. They do not by themselves prove scenario-exact reset, declared fixed-step capture stride, complete final-terminal-frame retention, or this specification's closed termination semantics.

### Acceptance criteria

- All rollouts in one scenario lineage begin from the identical declared initial engine state, including rollouts with different interventions.
- Any seed-controlled initial-state variation is assigned to a different scenario specification and scenario lineage.
- Every accepted rollout has one pre-intervention initial frame record, exactly one intervention, one termination reason, and one final frame record covering termination.
- Any rollout containing a second shot is rejected as a whole.

## 8. Frozen intervention plans and coverage

Before collection begins, the collection plan MUST freeze all interventions for every scenario lineage, including deterministic ordering, retry limits, stopping rules, and coverage strata. Plan identity and version MUST be assigned before observing rollout outcomes. A plan version MUST NOT change because an intervention succeeded, failed, produced a desired class, or filled an outcome bucket.

The intervention plan MUST be hybrid and MUST combine:

- geometry-aware, stratified feasible shots derived prospectively from the scenario specification and interface constraints;
- targeted shots for predeclared rare-interaction coverage strata; and
- replayed benchmark-agent actions where such actions and their provenance are available.

The plan MUST declare target coverage for `no-contact/miss`, `collision`, `persistent support`, `support change`, `destruction`, `pig removal`, `explosion` where applicable, `stability transitions`, `level clear`, and `level fail`. A target category MAY be marked unsupported only when the scenario specification or accepted engine capabilities make it inapplicable, and that reason MUST appear in the collection plan and production quality report.

For `central_v2`, the declaration narrows this general enriched-cohort inventory. Geometry-aware feasible shots and targeted rare-interaction shots are required; benchmark-agent actions are optional and their absence MUST have an explicit unavailable source disposition. The required central target strata are exactly `no-contact/miss`, `collision`, `persistent support`, `support change`, `destruction`, and `stability transitions`. Pig removal, explosion, level clear, and level fail remain raw events or termination evidence when they occur, but are target quotas only for an approved named-secondary experiment.

Each intervention MUST be represented in both:

- interface terms: the command supplied through the benchmark interface, including all shot and tap/release fields needed to replay it; and
- engine-relative terms: the normalized action relative to the recorded slingshot and engine coordinate conventions.

The mapping between the two representations MUST be deterministic and retained as provenance.

Coverage strata MUST be predeclared categories of intended interaction or non-interaction, not post-hoc result labels. Planning assignments MUST be defined without consulting the realized outcome of the same rollout. Realized outcomes MAY be reported against the frozen target categories, but MUST NOT cause post-hoc retention, deletion, replacement, cherry-picking, intervention mutation, or plan-version mutation.

Physics-validated negatives MUST be generated or selected only under a frozen negative specification. Negatives MUST have explicit semantic or physical justification, MUST remain distinct from simulator or capture failures, and MUST be bounded by a declared negative cap. The collection MUST NOT keep generating negatives until a desired model result or class balance appears.

Bounded negative evidence is required only for the named `spsg_contrastive_loss_ablation` secondary experiment. A central-v2 plan that does not run that secondary MUST declare bounded negatives non-central and MUST NOT treat their absence as a central capability failure.

### Acceptance criteria

- The collection plan bytes and plan version are unchanged before and after execution.
- Every rollout maps to one predeclared intervention and coverage stratum.
- The frozen plan contains every intervention source required by its exact capability declaration and declares the disposition of every applicable target coverage category.
- Accepted, rejected, and failed rollouts remain visible in outcome-independent accounting; realized outcomes do not control retention.
- Rejected simulator/capture attempts do not become negative training examples.
- Negative count never exceeds the declared cap and cannot be increased from observed outcomes without a new prospective collection plan.

## 9. Required raw capture

Each accepted enriched rollout MUST capture, without image-based reconstruction:

1. **All causal entities:** every authored or spawned causal entity, visible or invisible, dynamic or static where it can affect physics, interventions, outcomes, or oracle semantics.
2. **Identity:** cohort release candidate identity, collection-plan identity, benchmark condition, scenario template, level instance, scenario specification, scenario lineage, rollout, capture, shot, seed, intervention, observation configuration, rerun/attempt, object instance ID, and scenario object ID where applicable.
3. **Lifecycle:** spawn/activation, active/inactive state, destruction/removal, and terminal participation needed to explain entity presence across frame records.
4. **Object and world causal properties:** object class/type, geometry, transform, body presence, pose, velocity, angular velocity, mass, raw `life`, gravity/world parameters, and other authored properties that can change dynamics or oracle semantics. Material and damage properties MUST be captured only through verified mappings and declared capabilities; otherwise they are unavailable. Raw `life` MUST be preserved as emitted and MUST NOT be relabeled as health or damage without such a mapping.
5. **Raw contacts and events:** every non-trigger raw contact point for every simulated fixed step in each collected transition interval, even when frame records use a larger capture stride; collider identities; canonical participants; point, normal, separation, relative velocity, available impulses; and the complete frozen event taxonomy payloads.
6. **Coordinates, cameras, and transforms:** world and observation coordinate declarations, units, camera state, viewport, transforms between engine/world and observation coordinates, and synchronization identity.
7. **Observations:** the agent observation after benchmark-defined transforms and the access-restricted canonical observation used only for alignment and capture diagnosis.
8. **Terminal evidence:** the final terminal frame record and the fixed-step events establishing termination.

The `physics_capture_v1` sidecars currently provide synchronized scene nodes, kinematics, raw contacts, derived `support_v1`, macro events, coordinates, and exact RGB references within their frozen scope. They do not currently establish all scenario-lineage, exposure-role, camera/transform, material, damage, world-property, scenario object ID, or final-terminal-frame requirements above. Missing required fields MUST be reported as unsupported, not inferred from path conventions or RGB.

`observation_trace_manifest_v1` provides the central-v2 observation boundary: separately identified pre-transform canonical and post-transform agent PNGs, complete engine-authored camera/viewport/coordinate/unit/world-to-observation metadata, exact source-frame synchronization, observation-configuration-bound lineage identity, exposure-boundary validation, and diagnostic-only canonical access. Its representative issue #46 evidence is under `data/runtime_evidence/issue-46`; this capability evidence does not by itself create a cohort release.

### Acceptance criteria

- Entity enumeration reconciles authored entities, spawned entities, lifecycle events, and every contact/event participant.
- Every object/contact/event identity resolves within the rollout.
- Agent observation and canonical observation access policies are testably distinct.
- Missing required causal properties make the rollout ineligible for capabilities that depend on them; absent verified material or damage mappings yield unavailable properties rather than inferred raw facts.

## 10. Fixed-step authority and capture stride

Fixed step is the sole authoritative discrete simulation-time coordinate. Every collection plan MUST declare one positive-integer capture stride measured in fixed steps. A frame record MUST be selected by fixed step according to that stride, not by render cadence, wall-clock pacing, or target video frame rate.

For consecutive retained frame records at fixed steps `S_i` and `S_{i+1}`, all contacts and events with occurrence fixed steps after `S_i` and through `S_{i+1}` belong to that transition interval. Raw contacts required by evidence derivations MUST be retained for every simulated fixed step in the interval; capture stride MAY reduce frame records but MUST NOT create gaps in required contact evidence. Same-fixed-step events form one atomic cluster; serialization order within the cluster MUST NOT imply causality.

Render frame and render time are provenance only for temporal authority. Video production MAY resample observations for presentation, but presentation frames MUST NOT change frame-record or transition-record identity.

Current capture records monotonic fixed-step/fixed-time values, but existing collection pacing is target-FPS based. Therefore a declared fixed-step capture stride is required but currently unsupported as the collection authority.

### Acceptance criteria

- Retained frame-record fixed steps follow the declared capture stride, except for the explicitly retained final terminal frame record.
- Raw-contact evidence is complete at every simulated fixed step covered by each transition interval, independent of frame-record capture stride.
- Every contact/event occurrence belongs to exactly one transition interval or to the pre-initial/post-final invalid region.
- Changing render cadence without changing engine execution does not change transition assignment.

## 11. Atomic validation, retries, and failures

The rollout is the atomic acceptance unit. Validation MUST cover identity, provenance, required capabilities, temporal monotonicity, alignment, entity references, lifecycle, finite numeric data, contact/event payloads, final terminal evidence, and all required artifacts. Partial frame records, sidecars, observations, or labels MUST NOT be admitted independently.

Retries MUST be limited to transient operational failures declared in the collection plan, such as bounded startup, transport, or temporary capture availability failures. A retry MUST rerun the entire rollout from reset and MUST preserve the planned scenario lineage and intervention identity while receiving a distinct attempt identity.

Permanent content, schema, semantic, or evidence defects MUST NOT be retried as if transient. Missing collision payload facts are permanent defects. Required `contact_ids`, relative speed, or raw-contact evidence MUST NOT be reconstructed from RGB, later states, geometry guesses, or neighboring rollouts.

Every failed attempt MUST produce or be referenced by a failure manifest recording typed failure, permanence classification, attempt identity, retry decision, and quarantine location. Invalid artifacts MUST be quarantined outside accepted cohort membership. Exhausted or permanent failures MUST appear in the cohort failure manifest and quota accounting; they MUST NOT disappear through resampling.

Current physics collection uses temporary shot publication, whole-artifact validation, quarantine/recovery metadata, and retry attempts. Those mechanisms are implemented foundations. Restriction to transient-only retries and complete research-level failure/quota manifests MUST be verified against the collection plan before production use.

### Acceptance criteria

- Mutation or removal of any required rollout component rejects the whole rollout.
- A permanent collision-payload defect produces no retry intended to obtain a luckier outcome and no reconstructed payload.
- Accepted and quarantined attempt identities are disjoint and auditable.
- Retry exhaustion changes only failure accounting, never the frozen intervention plan or exposure assignment.

## 12. Supervision contracts

### 12.1 Contact and support

The contact relation MUST be symmetric: if validated evidence establishes contact between entities A and B at a fixed step, the same fact holds for B and A. Storage MAY use one canonical unordered pair.

Support relation supervision MUST be evidence-aware and tri-state:

- `true` only when the versioned support derivation has complete required raw-contact persistence and geometry evidence;
- `false` only when the complete evidence window needed to assess the relation is present and the relation does not hold;
- `unavailable` when predecessor history, raw contacts, geometry, lifecycle identity, retention, or another required fact is absent.

`unavailable` MUST NOT be converted to false. The frozen `support_v1` rule in `physics_capture_v1` is the current authority for positive support edges. Consumers requiring explicit negative support targets MUST additionally prove evidence-window completeness.

### 12.2 Macro labels

`physics_macro_labels_v1` is the authority for fixed-step macro/outcome derivations. It MUST remain a separate, source-bound derivation artifact and MUST project events by fixed-step bracketing. The older combined macro/oracle artifact family is legacy-only and MUST NOT be used as current BG-NS-JEPA supervision or silently translated into `physics_macro_labels_v1`.

The implemented fixture-only `physics_macro_labels_v1` software provides deterministic derivation, validation, source binding, fixed-step intervals, and explicit availability. Representative semantic acceptance is still pending for `cascade-active`, `collapsed`, and `pigs-cleared`; their semantic status MUST remain `hypothesis_pending_representative_validation` until the representative gate passes. Predicates with that status MUST NOT be used as training targets, scoring inputs, model-selection signals, or reported research metrics until representative validation promotes them to an accepted semantic status in a new authoritative derivation version.

### 12.3 Material and damage

Material and damage supervision MUST use versioned mappings verified against authored engine values and runtime behavior. Display names, sprite appearance, path fragments, broad object classes, and commented or historical damage approximations MUST NOT define labels. Until canonical mappings are accepted, material- and damage-dependent labels are unavailable.

### 12.4 Physical-regime gate and usefulness

The physical-regime gate MUST be a versioned engine-derived label describing declared motion and contact conditions at a frame record. It MUST remain distinct from micro-relation usefulness, which is model-relative and can be established only by frozen-model out-of-sample comparison. Neither label MAY substitute for the other.

For `central_v2`, neither the physical-regime gate nor micro-relation usefulness is a required controller input, target, gate, or central metric. The central physical-violation derivation MUST encode and cite its own per-label completeness and stability facts; it MUST NOT substitute a physical-regime label for missing geometry, gravity, lifecycle, motion, support/contact, or world-context evidence. Either secondary capability requires its separately named experiment and cannot be promoted by a central artifact.

Engine state and observations are primary artifacts. Oracle labels, the physical-regime gate, micro-relation usefulness, parser targets, and latents are derivations. No derived latent or parser output MAY overwrite engine facts or canonical observations.

### Acceptance criteria

- Swapping contact participants leaves the contact-relation truth unchanged.
- Removing required support evidence changes the support target to `unavailable`, not false.
- Macro labels cite fixed-step source events and reject stale source sidecars.
- A reader or scoring workflow rejects any requested macro predicate whose semantic status is `hypothesis_pending_representative_validation`.
- A physical-regime gate can change without asserting that micro relations improve a model, and a usefulness result identifies the frozen model and held-out evidence used.

## 13. Physical-violation labels v1

The first accepted physical-violation vocabulary MUST contain only evidence-backed labels whose required facts are available in the source cohort release:

- **Excess penetration:** available only when complete collider/contact geometry and separation evidence exist under a declared coordinate convention and tolerance. It is true when penetration exceeds the declared tolerance, false when complete evidence establishes that it does not, and unavailable otherwise.
- **Unsupported stationary/floating body:** available only for an active body whose gravity applicability, lifecycle, motion over the declared stability window, support/contact evidence, and world context are complete. It is true when the body satisfies the declared stationary condition while lacking valid support under the accepted support contract, false when complete evidence establishes support or non-stationarity, and unavailable otherwise.

Malformed records, non-finite values, contradictory identity, incomplete required lifecycle, or invalid coordinate data invalidate collection. They MUST NOT be labeled as physical violations.

`illegal contact` MUST remain unavailable until a versioned legal-contact ontology, entity/material mapping, exemptions, and complete evidence contract are defined and accepted. Absence of such a definition MUST NOT be represented as false.

These labels MUST be derivation artifacts bound to an immutable cohort release. They MUST NOT be inferred from model predictions when evaluating source data quality.

### Acceptance criteria

- Each available violation value cites complete source evidence and the versioned derivation specification.
- Removing geometry, support-window, gravity, or lifecycle evidence produces `unavailable` or collection invalidation as specified, never an assumed negative.
- No v1 artifact emits an available `illegal contact` value.

## 14. Pilot gate, production, releases, and final evaluation

Before production collection, a capability-complete representative pilot MUST be accepted. The pilot MUST exercise every capability required by the intended research cohort, every supported rollout termination class, the planned identity/alignment path, intervention representation, fixed-step capture stride, lifecycle, contact/event payloads, canonical and agent observations, failure/quarantine path, and all supervision evidence classes intended for production. It MUST also demonstrate deterministic, version-bounded replay: replay is assessed only under the same declared engine/player, protocol, generator, scenario specification, collection-plan, and intervention versions, and any change to those bounds requires a new pilot determination. The versioned replay evidence and comparison semantics are defined by [`cohort_v2_replay_evidence_v1`](data_contracts/cohort_v2_replay_evidence_v1.md).

The pilot MUST contain representative benchmark conditions, scenario templates, level instances, interventions, coverage strata, and the physical interaction windows needed by every declared central label. A separately named physical-regime label is required only when its secondary experiment is approved. Fixture suites and smoke rollouts MAY support software confidence but MUST NOT substitute for this pilot.

Production quotas MUST be defined prospectively by benchmark condition, exposure role, split regime, scenario-template/level-instance coverage, intervention coverage stratum, and required capability coverage. Quotas MUST NOT be defined or backfilled by model score or desirable realized outcome. Permanent failures MUST remain visible alongside unmet quotas.

Each cohort release MUST be immutable. Corrections to raw rollout artifacts require a new cohort release; they MUST NOT be patched in place. New oracle labels, parser targets, physical-violation labels, physical-regime gates, micro-relation usefulness labels, and training examples MUST be separately versioned derivation artifacts bound to their exact source cohort release.

Final evaluation MUST use a role-separated workflow. The final-evaluation manifest, metrics, derivation versions, checkpoints, and stopping rules MUST be frozen before access. The workflow SHOULD separate data custodian, model-selection operator, and final evaluator roles; at minimum, technical access controls and audit records MUST prevent final-evaluation observations from influencing training, calibration, or model selection.

### Acceptance criteria

- The pilot report demonstrates every required capability on representative accepted rollouts and names unsupported capabilities explicitly.
- Version-bounded replay reproduces the declared scenario, intervention, initial engine state, identities, and deterministic artifact semantics under the pilot's fixed version envelope.
- Production cannot start while any required pilot capability or representative semantic gate is incomplete.
- Re-running derivations creates a new derivation version or byte-identical artifact; it never mutates the cohort release.
- Final-evaluation access occurs only after the frozen workflow manifest exists, and every access is auditable.

## 15. Current implementation status

The following statements distinguish implemented foundations from required but unsupported research behavior:

- **Implemented:** frozen `physics_capture_v1` sidecars and validation; synchronized RGB/state capture within that contract; `physics_capture_v2` fixed-step-stride authority and physical evidence; source-bound deterministic generated scenario manifests; request-72 synchronized agent/canonical observation traces and access audits; the issue-47 real four-role instance-held-out partition, lineage/template leakage audit, and pending-authorization final-access workflow; fixed-step clocks; scene nodes, kinematics, raw contacts, `support_v1`, macro events, bounded failures; collector-side temporary publication, validation, quarantine, and retry metadata; deterministic path-based bucket partitioning and scoped inventory; fixture-only `physics_macro_labels_v1` derivation and validation.
- **Implemented but insufficient:** the issue-specific #44–#47 representative evidence establishes only its exercised capabilities, identities, exposure assignments, and pre-authorization access boundary. It does not create a capability-complete central-v2 pilot, production coverage, or representative enriched cohort release.
- **Required but currently unsupported or unaccepted:** explicit `legacy_static` inventory evidence where applicable; frozen coverage-stratified intervention plans in canonical action forms; complete central causal-property capture; representative central macro-label semantics; accepted central physical-violation derivations; capability-complete pilot; production quotas; immutable cohort release workflow; authorized execution of the role-separated final evaluation.

## 16. Definition of done for GitHub issue #2

GitHub issue #2 is complete only when all of the following are true:

1. An immutable, versioned representative cohort release exists with its frozen collection plan, partitions, exposure roles, provenance, failure manifest, and accepted derivation references.
2. Every rollout admitted to that cohort release passes atomic whole-rollout validation under the required capture and provenance contracts; no partial or reconstructed rollout is admitted.
3. The capability-complete representative pilot and the production quality report demonstrate the required benchmark-condition, scenario-template, level-instance, intervention, target coverage category, termination, lifecycle, contact/event, observation, and capability-declared split coverage. They MUST report accepted coverage together with every failure, rejection, unavailable capability, unavailable label, and unmet quota reason.
4. Exposure isolation is demonstrated for both the declared dataset partitions and exposure roles, including the applicable instance-held-out and template-held-out audits and the role-separated final-evaluation boundary.
5. Every authoritative derivation artifact required by the exact capability declaration for the release's declared research use is published, source-bound to the immutable cohort release, semantically accepted, and valid. This includes applicable macro labels, support supervision, physical-regime gates, physical-violation labels, material/damage mappings, and any other declared oracle supervision. A `hypothesis_pending_representative_validation` predicate cannot satisfy this condition. A capability classified as named-secondary, optional, or out of scope does not gate central-v2 completion and cannot be silently promoted into the central release.
6. Downstream readers complete smoke ingestion of the released primary artifacts and every required derivation artifact through their public, fail-closed interfaces, preserving identity, temporal alignment, availability, and exposure restrictions.
7. No known systematic engine-export defect remains in any required capability. In particular, collision-bearing rollouts MUST export complete contract-valid collision payload and raw-contact evidence without reconstruction.

Specification completion, pipeline implementation, command success, fixture success, smoke success, or reaching a planned rollout count is insufficient to close issue #2. Issue #2 closes only after the immutable representative cohort release and all evidence above exist and pass acceptance.

## 17. Open numeric parameters

Only the following quantities are intentionally deferred to the representative pilot and collection plan:

- fixed-step capture stride;
- stability window;
- rollout ceiling;
- geometric, motion, and numeric tolerances;
- production and pilot quotas;
- bounded negative cap;
- transient retry counts.

Their decision process is settled: values MUST be chosen prospectively from capability-complete pilot evidence, recorded in the collection plan or versioned derivation specification before affected data are admitted, held fixed for the corresponding cohort release, and changed only through a new prospective plan or derivation version. Values MUST NOT be selected from final-evaluation outcomes or tuned to make a completed cohort pass.

## 18. ADR candidates

Only these architectural decisions are candidates for later ADRs:

1. **Engine-authoritative fixed-step timing** — fixed step is the only clock that preserves the occurrence order and interval assignment of physical state, contacts, and events independently of rendering.
2. **Single-shot rollout as atomic acceptance unit** — reset, intervention, trace, termination, and validation are scientifically inseparable and must be accepted or rejected together.
3. **Immutable cohort releases with sidecar derivations** — immutable primary evidence plus separately versioned source-bound derivations prevents silent relabeling while allowing supervision to evolve.

This specification does not create or authorize ADR files.

## 19. Current known blockers

The issue #33 audit remains the evidence baseline. Its immutable scoped release demonstrates a contract-valid collision path for its four admitted rollouts, but it is not a central-v2 release and MUST NOT be upgraded in place.

The current central-v2 blockers are:

1. **No capability-complete central-v2 pilot or immutable release:** the existing pilot records `coverage.audit.representative=false` and cannot satisfy the new declaration.
2. **Missing remaining central exporter evidence:** the complete source facts needed by both central physical-violation labels have not yet been demonstrated within a capability-complete central-v2 pilot and release. Fixed-step-stride authority and synchronized access-separated agent/canonical observations now have separate representative capability evidence, but are not cohort-release evidence.
3. **Missing central representative semantic evidence:** the two micro labels, two macro labels, and two endpoint-violation labels have not each met the declaration evidence floor on a new v2 release. Prior fixture, smoke, accepted-software, or rejected/unavailable evidence does not satisfy this gate.
4. **Missing central exposure, replay, and ingestion evidence:** real four-role instance-held-out partitions, version-bounded deterministic replay, frozen final access, and complete public central-v2 ingestion are not demonstrated together. Source-bound template/instance identities now have separate reviewed capability evidence, but are not a release-wide exposure audit.

Template-held-out evaluation, a physical-regime label, bounded negatives, material/damage mappings, gravity shifts, `cascade-active`, `collapsed`, `pigs-cleared`, `illegal_contact`, planning, and cross-domain evaluation are not central-v2 blockers. Their non-central disposition is not evidence that they are false or scientifically accepted.

Until these blockers are closed and the complete Section 16 evidence exists for the exact declaration, issue #2 and the cohort-v2 umbrella MUST remain open; production collection and downstream central claims MUST remain blocked.
