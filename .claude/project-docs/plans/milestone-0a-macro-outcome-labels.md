# Milestone 0a Execution Specification — Deterministic Macro and Outcome Labels

**Status:** implementation-ready plan, 2026-08-13. Fixture-only implementation is authorized within the boundaries in §15; runtime execution, publication, cohort collection, and real-cohort writes are not authorized by this document.

**Authority:** this plan follows, in order, the completed T1 handoff, the appended current runtime verdict, the corrected BG-NS-JEPA roadmap, the research proposal and system overview, and the accepted continuous-only Milestone 1e/1f evidence. It supersedes `.claude/project-docs/plans/enriched-cohort-oracle-labels.md` for Milestone 0a semantics and sequencing. The older plan remains historical evidence.

## 1. Purpose and acceptance boundary

Milestone 0a defines a deterministic, engine-anchored macro/outcome label layer over accepted `physics_capture_v1` sidecars. It contains no learned or vision-derived signal and does not change the continuous latent $z$, which remains the sole rollout state carrier.

This plan separates three gates:

1. **Fixture software acceptance:** schema, derivation, validation, identity, canonical bytes, and reader projection pass on committed fixtures. This work may begin now.
2. **Representative semantic acceptance:** the provisional `cascade-active`, `collapsed`, and `pigs-cleared` interpretations agree with authorized representative enriched shots. This remains blocked.
3. **Milestone 0b:** oracle reliability labels and frame-aligned enriched supervision. This remains separate and blocked by data/authority.

Milestone 0a does **not** define the Milestone 0b oracle gate:

$$
\phi^*(x_t)=\mathbf{1}[\mathrm{KE}(x_t)<\epsilon_{\mathrm{KE}}]
\cdot\mathbf{1}[\mathrm{active\_contacts}(x_t)<\epsilon_{\mathrm{contact}}].
$$

No KE/contact threshold appears in a Milestone 0a predicate.

## 2. Corrected current state

- Runtime status is `repin_complete`; the accepted archive is `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.
- Exactly one full smoke accepted the archive. It cannot be repeated under the consumed authorization.
- Runtime publication and enriched-cohort collection were not authorized and did not occur.
- Commits `6c70ebd` and `6bfbb8a` introduced preliminary macro/outcome/oracle code before the accepted re-pin. The code remains useful scaffolding but is not accepted Milestone 0a evidence.
- The preliminary implementation must be superseded because it:
  1. groups events by `event.render_frame`, although real events are serialized later in one batch;
  2. combines Milestone 0a macro/outcome labels with Milestone 0b's thresholded oracle gate;
  3. identifies pigs using exact `object_class == "pig"`, while the exporter emits Unity tags such as `PigSmall`;
  4. converts a fixed-step debounce count into render-frame padding;
  5. lacks a real-smoke-faithful fixture.
- The continuous-only Milestone 1e/1f result remains reusable and unchanged. Its `not_supported` verdict is temporal-only; it is not evidence for or against the joint $(\Delta,\alpha)$ hypothesis.

## 3. Frozen input contract

The derivation consumes only sidecars that already pass `load_physics_capture` and the frozen `physics_capture_v1` validator.

### 3.1 Shared record clock and identity

Every header, state, and event carries:

- `schema_version`
- `capture_id`
- `shot_id`
- sidecar-local `sequence`
- `render_frame`, `render_time`
- `fixed_step`, `fixed_time`
- the complete coordinate/unit declaration

The validator requires one `(capture_id, shot_id, coordinates)` identity across both sidecars and strictly increasing sidecar-local sequences.

### 3.2 State records

Each state carries:

- `rgb_frame.relative_path`, its matching `render_frame`, dimensions, and synchronized source;
- `nodes`: `entity_id`, Unity instance ID, open-string `object_class`, `object_type`, polygon, world pose, life, and body presence/velocity/angular velocity/mass/kinetic energy;
- `raw_contacts`: deterministic contact identity, entity/collider pair, point, normal, separation, relative velocity, optional impulses, and non-trigger status;
- `support_edges`: directed supporter/supported identity with two contact IDs and two consecutive fixed-step evidence values under `support_v1`.

Milestone 0a uses node identities/classes, node presence across accepted states, support-edge sets, state clocks, and RGB identity. KE and contact activity remain available to Milestone 0b but are not 0a predicate inputs.

### 3.3 Event records

The closed taxonomy is:

`bird_launched`, `collision`, `explosion`, `entity_destroyed`, `pig_removed`, `bird_exhausted`, `stable_entered`, `stable_exited`, `level_cleared`, `level_failed`.

Each event carries `event_id`, `event_type`, unique participants, a taxonomy-specific payload, and the common clock. The validator requires deterministic fixed-step order; contiguous event sequence/IDs; unique collision pairs per fixed step; no repeated one-shot, terminal, or entity-lifecycle event; mutually exclusive clear/fail; and alternating stability transitions.

### 3.4 Critical clock finding

`PhysicsCaptureProtocol.BuildEventsJson` stamps buffered events with the serialization snapshot's `render_frame` and `render_time`, but preserves each event's occurrence `fixed_step` and `fixed_time`.

The accepted smoke proves this is a real supported case:

- one state: `fixed_step=476`, `render_frame=15207`;
- 13 events: distinct fixed steps `267..441`;
- all 13 events: `render_frame=15207`.

Therefore:

- event occurrence authority is `fixed_step`/`fixed_time`;
- event `sequence`/`event_id` supplies deterministic identity and ordering;
- event `render_frame` is retained as source provenance only;
- no derivation or reader may join events to states by event `render_frame`.

## 4. Output artifact decision

Create a new sidecar family named `physics_macro_labels_v1`, with default filename `physics_macro_labels.jsonl`.

Reasons:

- the frozen `physics_capture_v1` sidecars must not be changed;
- Milestone 0a must not inherit Milestone 0b threshold authority;
- the existing `physics_derived_labels_v1` bytes and semantics predate the accepted runtime and must not be silently reinterpreted;
- a new reader opt-in makes mixed or stale artifacts fail visibly rather than appear compatible.

Implementation may reuse parsing, immutable types, canonical JSON, source-digest, atomic-write, and re-derivation patterns from the preliminary code. It must introduce a new version, sidecar path, validation path, and explicit opt-in. It must not rewrite legacy derived-label files.

## 5. Closed vocabulary

Each per-state predicate has a boolean value and an `availability` value. Allowed availability values are `available`, `unavailable_no_predecessor`, and `unavailable_insufficient_state_evidence`. An unavailable predicate serializes `value: null`; it is never silently converted to false for training.

### 5.1 `steady-state`

- **Native timing:** fixed-step intervals.
- **Rule:** true in each half-open interval `[stable_entered.fixed_step, stable_exited.fixed_step)`. A trailing `stable_entered` remains open through the last accepted state.
- **Pre-launch:** if a `bird_launched` event exists, states with `state.fixed_step < launch.fixed_step` are steady.
- **Absent events:** no launch and no stability transition produces no synthetic steady interval.
- **Absorbing:** no.
- **Status:** engine-verified when stability events exist. The engine's debounce is already embodied in the emitted transition; 0a does not add another debounce.

### 5.2 `structure-unstable`

- **Rule:** true iff `steady-state` is false and the directed support-edge set differs from the immediately previous accepted state.
- **First accepted state:** `value: null`, `availability: unavailable_no_predecessor`.
- **KE:** not used. A KE threshold belongs to Milestone 0b scale separation.
- **Absorbing:** no.
- **Status:** deterministic support-topology change, not a claim that every moving structure is unstable.

### 5.3 `cascade-active`

- **Causal event set:** `collision`, `explosion`, `entity_destroyed`, `pig_removed`.
- **Onset:** the first fixed-step cluster in the causal set at or after `bird_launched.fixed_step`. Without a launch, no cascade interval is created.
- **Termination:** the earlier of:
  1. the first later `stable_entered` cluster; or
  2. one fixed step after the last causal cluster.
- **Projection:** true for state fixed steps in `[onset, termination)`.
- **No fabricated render tail:** `debounce_fixed_steps` is not converted to render frames.
- **Absorbing:** no.
- **Status:** onset is engine-anchored; the one-step tail/termination is a provisional research interpretation requiring representative semantic acceptance.

### 5.4 `collapsed`

- **Absorbing:** yes.
- **Candidate entity:** an entity that was the `supported_id` of at least one earlier support edge.
- **Rule:** latch true only when the candidate later has no incoming support edge and either:
  1. an `entity_destroyed` or `pig_removed` event for that identity has occurred by the projection step; or
  2. the entity disappears from a later accepted state after previously being present.
- Support loss alone is insufficient and remains false.
- A one-state or otherwise insufficient trajectory yields `value: null`, `availability: unavailable_insufficient_state_evidence` until the rule can be evaluated.
- **Status:** hypothesis requiring representative enriched shots. The engine exports evidence, not a native `collapsed` flag.

### 5.5 `pigs-cleared`

- **Pig class set:** the versioned closed set `PigSmall`, `PigMedium`, `PigBig`, matching Unity tags used by the exporter.
- **Rule:** latch true when no current node belongs to the pig class set and every pig identity observed in an earlier accepted state has a `pig_removed` event at or before the projection step.
- A shot with no previously observed pig identity does not satisfy the predicate.
- `level_cleared` does not retroactively fabricate earlier `pigs-cleared` values.
- **Absorbing:** yes.
- **Status:** class mapping is engine-anchored; taxonomy completeness and event coverage require representative semantic acceptance.

### 5.6 Outcome and equilibrium labels

One shot-level outcome record is emitted:

- `cleared`: a `level_cleared` event exists; preserve its score.
- `failed`: a `level_failed` event exists; preserve its reason.
- `settled_nonterminal`: neither terminal event exists and the final accepted state projects inside a steady interval.
- `unsettled_nonterminal`: neither terminal event exists and the final accepted state is not steady.

The two nonterminal classes describe the capture window, not a level outcome or learned equilibrium pole.

`terminal_equilibrium` is:

- `stable_terminal` only when a terminal event projects to an existing accepted state that is steady;
- `not_observed` otherwise, including a terminal event after the last state.

An event after the last state may determine the shot outcome, but it never fabricates terminal per-state macro labels.

## 6. Fixed-step clustering and state projection

### 6.1 Atomic clusters

All events sharing a `fixed_step` form one atomic cluster. Taxonomy serialization order cannot create within-step causality.

Deterministic same-step rules:

- launch plus a causal event permits cascade onset at that fixed step;
- `stable_entered` closes a prior cascade before state projection at that step because intervals are half-open;
- terminal events affect only the outcome record;
- both `stable_entered` and `stable_exited` in one cluster are a hard rejection;
- mutually contradictory terminal events are already a frozen-contract rejection.

### 6.2 Projection

For each accepted state with fixed step `S`:

1. consume all event clusters with `fixed_step <= S`;
2. evaluate interval membership and absorbing latches at `S`;
3. emit exactly one frame-label record bound to that state and RGB frame.

Events before the first state affect the first projection. Events after the last state affect only shot outcome/provenance. State-to-RGB alignment remains exact through the state's `rgb_frame`; event-to-state alignment is fixed-step bracketing, never event render-frame equality.

## 7. Identity and alignment keys

- **Shot identity:** `(capture_id, shot_id)`; the coordinate declaration must also match the validated source capture.
- **State-label identity:** `(capture_id, shot_id, state_sequence, render_frame, fixed_step, rgb_relative_path)`.
- **Event citation identity:** `(capture_id, shot_id, event_sequence, event_id, fixed_step)`.

The smoke's external request sequences `1 -> 2` are provenance outside the frozen sidecars. The derived artifact must not invent or copy them as shot/event keys unless a future separately validated metadata contract supplies them.

## 8. `physics_macro_labels_v1` JSONL contract

Record order is fixed: one header, zero or more event intervals, one frame label per accepted state, then one outcome.

### 8.1 Header

Required fields:

- `record_type: "macro_label_header"`
- `schema_version: "physics_macro_labels_v1"`
- `capture_schema_version: "physics_capture_v1"`
- `capture_id`, `shot_id`
- `derivation_spec_version` and canonical `derivation_spec_digest`
- ordered macro vocabulary with absorbing and semantic-status declarations
- ordered pig class set
- source paths and SHA-256 values for `physics_state.jsonl` and `physics_events.jsonl`
- state/event/interval/frame-label counts
- explicit clock declaration: event occurrence uses `fixed_step`; event `render_frame` is provenance only

### 8.2 Event interval

Required fields:

- `record_type: "event_interval"`
- `interval_type` (`steady-state` or `cascade-active`)
- `start_fixed_step`, exclusive `end_fixed_step` or `null` for an engine-open steady interval
- ordered event citations supporting onset/termination
- `semantic_status` (`engine_verified` or `hypothesis_pending_representative_validation`)

### 8.3 Frame label

Required fields:

- `record_type: "frame_label"`
- complete state-label identity key
- one object per macro predicate: `value`, `availability`, ordered evidence citations
- ordered active macro-state names containing only predicates whose value is true

### 8.4 Outcome

Required fields:

- `record_type: "shot_outcome"`
- `outcome_class`, nullable `score`, nullable `reason`
- `terminal_event` citation or null
- `terminal_equilibrium`
- nullable terminal projecting state identity
- explicit `semantic_status`

Unknown fields are rejected in v1.

## 9. Deterministic bytes and persistence

- UTF-8 JSONL; ASCII key ordering; compact separators; finite JSON numbers only; LF line endings; exactly one trailing newline.
- Records and every evidence-citation list use the declared deterministic order.
- Repeating derivation from identical source bytes and spec produces byte-identical output.
- Source SHA-256 values bind labels to the exact frozen inputs and materially change validation: stale/tampered source bytes reject the label file.
- Write to a random same-directory temporary file, flush, `fsync`, then atomically rename. Cleanup temporary residue on failure.
- Fixture implementation writes only to temporary test directories. It must not write labels into a real cohort.

## 10. Fail-closed validation

| Condition | Result |
|---|---|
| Frozen sidecar malformed, incomplete, identity-mismatched, over limit, or noncanonical | Hard reject through existing validator |
| Event sequence/order/duplicate/lifecycle/stability/terminal invariant fails | Hard reject |
| Missing or duplicate state-to-RGB mapping | Hard reject |
| Duplicate state-label identity | Hard reject |
| Pig class set missing, reordered, duplicated, or different from v1 | Hard reject |
| Contradictory same-step stability cluster or ambiguous interval | Hard reject |
| Derived source digest stale | Hard reject |
| Unknown derived field, wrong record order/count, noncanonical order/bytes | Hard reject |
| Frame-label key differs from accepted state/RGB | Hard reject |
| Absorbing predicate reverts | Hard reject |
| Terminal event after last state | Valid outcome; `terminal_equilibrium=not_observed`; no per-state backfill |
| First-state `structure-unstable` lacks predecessor | Valid unavailable |
| `collapsed` lacks sufficient multi-state evidence | Valid unavailable |
| Launch, stability, causal, pig-removal, or terminal event absent | Valid absence; do not synthesize the corresponding fact |

## 11. Edge cases the project can produce

Fixtures and validators must cover:

- all events sharing one render frame while fixed steps differ;
- fixed step 391 containing two `entity_destroyed`, one `pig_removed`, and `bird_exhausted` events atomically;
- collision events citing one or multiple sorted unique contact IDs;
- absent launch, stability, causal, pig-removal, and terminal event types;
- alternating stability transitions and rejection of invalid repetitions;
- a sparse one-state capture with many earlier events;
- a terminal event after the final state;
- rejection of duplicate collision pairs per fixed step and repeated entity-lifecycle events.

The same-step duplicate-ingestion finding remains outside scope unless a focused fixture proves that accepted source semantics can make the derived label incorrect.

## 12. Representative fixtures

The implementation session must add:

1. a canonical multi-state golden fixture with event fixed-step intervals;
2. an accepted-smoke-faithful one-state/13-event fixture in which all event render frames equal the serialization frame;
3. positive and negative cases for every predicate;
4. a same-step heterogeneous event-cluster fixture;
5. absent launch/stability/terminal fixtures;
6. a delayed terminal-after-last-state fixture;
7. `PigSmall`, `PigMedium`, and `PigBig` coverage;
8. support loss without destruction/disappearance as a negative collapse case;
9. support loss plus destruction and support loss plus disappearance as positive collapse cases;
10. malformed, duplicate, stale-source, unknown-field, absorbing-reversion, and canonical-byte fixtures.

The archived accepted smoke may be reduced into a committed fixture with a provenance note and source digest. It must not be rerecorded.

## 13. Machine-readable acceptance evidence

Fixture/software evidence belongs under:

`.claude/project-docs/evidence/milestone-0a-macro-outcome-labels/`

Required artifacts:

- `acceptance-manifest.json`: schema/spec/source digests, fixture list, test counts, byte-repeat result, validation matrix results, protected-scope statement, and `semantic_acceptance: pending_authorized_representative_shots`;
- `commands.log`: exact commands, exit codes, and complete output paths;
- `fixture-provenance.json`: accepted-smoke fixture source path/digests and transformation note;
- `canonical-bytes.json`: repeated output digests and equality result;
- `scope-receipt.json`: no runtime/player/cohort/protected-root write.

Planned final commands for the implementation session:

```bash
python -m unittest -v tests.test_physics_macro_labels
python -m unittest -v tests.test_physics_capture_contract tests.test_world_model_data
python scripts/derive_physics_macro_labels.py --target tests/fixtures/physics_capture_v1_macro --output-dir <tmp-a> --json
python scripts/derive_physics_macro_labels.py --target tests/fixtures/physics_capture_v1_macro --output-dir <tmp-b> --json
```

The last two outputs must compare byte-identically through the artifact validator. These commands are specified here, not executed in this planning session.

A later `representative-semantic-acceptance.json` must name authorized shot identities, covered level/regime set, per-predicate review results, unresolved cases, and owner authorization. It cannot be manufactured from fixtures or the consumed smoke.

## 14. Reuse of completed continuous Milestone 1e/1f work

Reuse unchanged:

- the approved `delta={1,5,15}` pair-grid representation;
- state-key, artifact, shard, scoring, frontier, and reproducibility infrastructure;
- continuous checkpoint/scores and their temporal-only claim boundary;
- explicit unavailable-reason handling.

Extend later, only after representative macro-label acceptance:

- declare `macro` available for eligible enriched states;
- add macro readout targets and event/outcome metrics;
- score `(delta, macro)` pairs on the same identity and duration contracts;
- regenerate best-pair labels and the oracle-symbol ceiling rather than editing continuous artifacts.

`micro` remains blocked on accepted object/contact predicate supervision and Milestone 0b reliability data. The temporal-only `not_supported` result is retained unchanged and is not reinterpreted.

## 15. Dependencies, authority, and non-goals

### Allowed before publication/cohort authorization

- new schema/types/deriver/validator and CLI over committed fixtures;
- faithful transformation of archived accepted-smoke evidence into a fixture;
- correction of opt-in reader event projection to fixed-step bracketing;
- fixture-only health/validation summaries;
- focused unit/regression tests and machine-readable fixture evidence.

### Forbidden without separate authorization

- launching the player or Unity runtime;
- rerunning the consumed full smoke;
- publishing the archive;
- collecting any enriched cohort;
- writing labels into a real cohort;
- running a real cohort health report or training run;
- claiming representative semantic acceptance, micro/macro pair results, oracle gate, physical violations, full oracle-symbol ceiling, or joint-alpha support;
- changing the frozen schema/taxonomy/consumer to simplify derivation;
- modifying protected rollout roots, the canonical Unity project, or unrelated artifacts.

Milestone 2 perception, controller distillation, SPSG expansion, and full joint-alpha claims remain outside this milestone.

## 16. Bounded unresolved questions

| Question | Software-contract default | Evidence required to settle research semantics |
|---|---|---|
| Does support loss plus destruction/disappearance reliably mean `collapsed`? | Use the closed conservative rule; mark hypothesis | Authorized multi-state collapse and non-collapse shots, reviewed against engine states/events |
| Is one step after the last causal cluster the right cascade end when no stability entry intervenes? | Use `[first causal, min(stable entry, last causal+1))`; mark hypothesis | Authorized shots with dense state sampling through collision/collapse/settling; compare alternative debounce rules |
| Does fixed-step bracketing project correctly across sparse captures? | Apply all clusters `<= state.fixed_step`; after-last events affect outcome only | Authorized multi-state captures with events between consecutive snapshots |
| Are pig tags permanently limited to the v1 set? | Pin `{PigSmall,PigMedium,PigBig}` in the spec and reject drift | Exporter/tag inventory and representative captures after any Unity content change |

These questions do not block fixture software implementation. They do block research-valid semantic acceptance and downstream claims.

## 17. Execution tasks and dependency order

1. **Contract foundation:** add `physics_macro_labels_v1` schema, immutable types, semantic-status/availability enums, canonical serialization, and source binding.
2. **Fixed-step derivation:** implement atomic event clusters, interval construction, state projection, absorbing latches, and shot outcome.
3. **Validation/persistence:** parse strictly, re-derive, reject stale/noncanonical/misaligned artifacts, and write atomically.
4. **Fixtures:** add multi-state synthetic and accepted-smoke-faithful fixtures with positive, negative, unavailable, delayed, and malformed cases.
5. **Opt-in reader correction:** expose macro labels only when explicitly requested and project events by fixed-step bracket; leave default RGB/action samples unchanged.
6. **Final fixture verification:** run the specified focused/regression commands once, emit the acceptance evidence, and stop.

No task in this sequence depends on runtime publication or cohort collection. Representative semantic acceptance is a later authority-gated task.

## 18. Exact next dependency-ready action

Open a new fixture-only implementation session in this worktree. Its first inspection is:

1. `scripts/physics_label_derivation.py:319-516` — identify every render-frame event dependency and preliminary mixed 0a/0b field;
2. `world_model/data/supervision.py:95-165` — identify the current render-frame event grouping and opt-in label join;
3. the accepted smoke sidecars under `.claude/project-docs/evidence/runtime-repin-gate-20260810/session-5-full-smoke/shot_001/` — construct the faithful fixed-step fixture without running the player.

The implementation session must create the new schema/types/deriver/validator, faithful and synthetic fixtures, and corrected opt-in fixed-step reader projection. It must run focused fixture tests only and stop before any real cohort operation.

Acceptance criteria for that session:

- every label is computed only from documented frozen fields or emitted as unavailable;
- no oracle-gate or learned/vision signal appears in `physics_macro_labels_v1`;
- frozen and legacy sidecars remain byte-unchanged;
- state/RGB identities and fixed-step event citations round-trip exactly;
- repeated derivation is byte-identical;
- every hard-reject and unavailable case in §10 is test-pinned;
- machine-readable fixture acceptance evidence is complete;
- the handoff states that representative semantic acceptance, publication, cohort collection, Milestone 0b, and full symbolic Milestone 1 remain blocked.
