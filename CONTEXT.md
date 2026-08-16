# NovPhy Research Data

This context defines the language used to generate, collect, partition, and consume physics research data for NovPhy experiments.

## Language

**Scenario specification**:
A complete declarative description of one physics setup before simulation, including its intended novelty and layout identity.
_Avoid_: Scene, episode

**Benchmark condition**:
The declared pair of novelty level and novelty type under which level instances are generated and evaluated.
_Avoid_: Novelty, level type

**Scenario template**:
A reusable structural source from which related level instances are generated under one or more benchmark conditions.
_Avoid_: Level instance, scenario specification

**Level instance**:
One concrete generated or legacy-static benchmark layout whose complete contents form a scenario specification.
_Avoid_: Novelty level, template

**Scenario lineage**:
A scenario specification and every rollout, seed, intervention, observation configuration, and rerun derived from it. A lineage belongs to exactly one dataset partition.
_Avoid_: Sample group, related episodes

**Rollout**:
One independently executed single-shot simulation of a scenario specification under a recorded intervention, ending with a declared termination reason.
_Avoid_: Episode, multi-shot game, video

**Scenario collection**:
The set of independently executed rollouts collected from one level instance under a declared intervention plan.
_Avoid_: Episode, multi-shot game

**Intervention**:
The single recorded shot applied after a rollout's initial frame record, expressed in both interface and engine-relative terms.
_Avoid_: Untracked action, policy decision

**Collection plan**:
An outcome-independent declaration of scenario lineages, exposure roles, interventions, coverage strata, retry limits, and stopping rules to be collected together.
_Avoid_: Operator notes, mutable sampling script

**Coverage stratum**:
A predeclared category of physical interaction or non-interaction used to assess whether a collection plan covers its intended behavior.
_Avoid_: Post-hoc result bucket, class label

**Frame record**:
The complete post-step state and synchronized observations at one simulation time.
_Avoid_: Frame, snapshot, timestep

**Transition record**:
The intervention applied between two frame records together with the physical contacts and events produced by that simulation step.
_Avoid_: Action sample, event frame

**Fixed step**:
The authoritative discrete simulation-time coordinate for physical state changes and event occurrence.
_Avoid_: Render frame, image frame

**Capture stride**:
The declared positive integer number of fixed steps between consecutive frame records in a rollout.
_Avoid_: Frame rate, render cadence

**Engine trace**:
The authoritative sequence of physical states, contacts, and events emitted by the simulation engine for a rollout.
_Avoid_: Ground truth video, rendered truth

**Causal entity**:
An authored or spawned object that can affect physics, interventions, outcomes, or oracle semantics, whether or not it is visible.
_Avoid_: Visible object, renderable object

**Raw contact**:
A point-contact fact emitted by the engine for one fixed step, before relational interpretation.
_Avoid_: Support edge, collision label

**Contact relation**:
A symmetric oracle relation indicating that at least one validated non-trigger raw contact joins two active entities at a fixed step.
_Avoid_: Contact point, support relation

**Support relation**:
A versioned oracle relation derived from retained raw-contact persistence and geometry evidence.
_Avoid_: Raw contact, native engine support

**Observation trace**:
The sequence of rendered or otherwise sensor-like observations synchronized with an engine trace. It is evidence available to perception systems, not physical truth.
_Avoid_: Engine state, ground truth

**Agent observation**:
The observation exposed to an agent after every benchmark-defined representation transform.
_Avoid_: Canonical observation, raw screenshot

**Canonical observation**:
An access-restricted pre-transform observation retained only for alignment and capture diagnosis.
_Avoid_: Agent input, training image

**Training example**:
A model-specific input and target derived from accepted rollout records. It is not a primary collection artifact.
_Avoid_: Raw sample, rollout

**Object instance ID**:
The identity of one runtime object throughout a single rollout.
_Avoid_: Unity instance ID, global object ID

**Scenario object ID**:
The identity of one authored object across all rollouts in the same scenario lineage.
_Avoid_: Runtime object ID, cohort-wide object ID

**Unavailable**:
A label status meaning the required evidence is absent or the concept is undefined for that record; it is neither true nor false.
_Avoid_: False, missing-as-negative

**Physical-regime gate**:
A versioned engine-derived label describing whether declared motion and contact conditions hold at a frame record.
_Avoid_: Micro-relation usefulness, parser confidence

**Micro-relation usefulness**:
A model-relative label measuring whether micro-relational input improves a frozen model on out-of-sample evidence.
_Avoid_: Physical-regime gate, oracle truth

**Physical violation**:
A versioned oracle label for an evidence-backed breach of declared physical plausibility, distinct from malformed data that invalidates collection.
_Avoid_: Capture failure, unsupported heuristic

**Derivation artifact**:
A separately versioned, source-bound product computed from an immutable cohort release, such as oracle labels, parser targets, or model-relative labels.
_Avoid_: Rewritten raw trace, ad hoc training label

**Cohort**:
An accepted, partitioned set of complete rollouts governed by one declared data contract and provenance envelope.
_Avoid_: Dataset, data dump

**Cohort release**:
An immutable, versioned publication of a cohort together with its collection plan, partitions, provenance, and accepted derivation references.
_Avoid_: Latest dataset, mutable collection directory

**Exposure role**:
The declared permission governing whether a scenario lineage may influence training, calibration, model selection, or final evaluation.
_Avoid_: Folder split, informal train/test label
