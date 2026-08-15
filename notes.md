# Notes: BG-NS-JEPA Milestone 0a Reconciliation

## Verified Starting State
- Worktree: `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4`
- Branch: `physics-unity-2019.4`
- Starting HEAD: `2dfc439ac8414add241fa4ef32415b02aad3c037`
- Upstream: `origin/physics-unity-2019.4`
- Tracked tree: clean at session start.
- Untracked artifacts pre-existed under `.claude/`, `.omo/`, and `sciencebirdsgames/physics-v1/`; preserve them unless an exact path is intentionally adopted as a session-owned deliverable.

## Authority Findings

- Current runtime authority is `.handoff/2026-08-12-t1-runtime-repin-complete.md` plus the appended `Current verdict — third continuation` in `runtime-gate-result.md`: status `repin_complete`, implementation `d5be336be778103ac2ae883d4d946a1df3eaf540`, closure `5a34ec9e0a9f4f45ae7060de65c43a6d322d444c` and `2dfc439ac8414add241fa4ef32415b02aad3c037`.
- Staged receipt, accepted smoke report, and accepted-shot metadata all name archive `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`.
- Publication and cohort collection were not authorized and did not occur. The one-shot full smoke cannot be repeated.
- The high-level roadmap is stale: it names archive `429cac1d…` and states that no world-model/training code exists.
- The accepted legacy temporal evidence is continuous-only: `delta={1,5,15}`, `micro`/`macro` unavailable as `symbolic_supervision_unavailable`, two deterministic 3,600-step CUDA runs, 556,959 states and 1,670,877 scores each, identical checkpoint bytes, numeric aggregate differences within `rtol <= 1e-2`, best-pair agreement `1.0`, verdict `not_supported`. It is not the full oracle-symbol or joint-alpha result.
- Subordinate current-state evidence (not higher authority) shows preliminary macro/oracle derivation, reader integration, and health reporting already exist on this branch. They must be audited against current accepted runtime semantics rather than counted as accepted Milestone 0a closure.

## Schema and Evidence Inventory

- Frozen record clock on state headers, states, and events: `schema_version`, `capture_id`, `shot_id`, sidecar-local `sequence`, `render_frame`, `render_time`, `fixed_step`, `fixed_time`, `coordinates`.
- State records additionally carry `rgb_frame` (`relative_path`, matching `render_frame`, width, height, synchronized source), `nodes`, `raw_contacts`, and `support_edges`.
- Node facts: `entity_id`, `unity_instance_id`, open-string `object_class`, open-string `object_type`, screen polygon, world pose, nullable life, and body presence/velocity/angular velocity/mass/kinetic energy.
- Contact facts: deterministic `contact_id`, entity/collider pairs, point, normal, separation, relative velocity, optional impulses, non-trigger marker.
- Support facts: directed supporter/supported identities with exactly two contact IDs and consecutive fixed-step evidence under frozen `support_v1`.
- Event taxonomy: `bird_launched`, `collision`, `explosion`, `entity_destroyed`, `pig_removed`, `bird_exhausted`, `stable_entered`, `stable_exited`, `level_cleared`, `level_failed`.
- Event validation orders by `(fixed_step, render_frame, taxonomy rank, participants, event_id)`; sequence/event IDs are contiguous; same-step duplicate collision pairs, repeated one-shot events, repeated entity lifecycle events, nonalternating stability transitions, and dual terminal events fail closed.
- Critical producer fact: `PhysicsCaptureProtocol.BuildEventsJson` stamps every buffered event with the serialization snapshot's `render_frame`/`render_time`, while preserving each event's occurrence `fixed_step`/`fixed_time`. Therefore event `render_frame` is not an occurrence clock and must not drive macro intervals.
- Accepted smoke: one state at render frame 15207/fixed step 476 and 13 buffered events at fixed steps 267–441, all stamped render frame 15207. This proves delayed/batched render-frame emission is a real supported case.
- Accepted smoke same-step cluster at fixed step 391 contains two `entity_destroyed`, one `pig_removed`, and `bird_exhausted`; same-step heterogeneous events are real and must be treated atomically.
- Accepted smoke node taxonomy includes `object_class: PigSmall`, contradicting preliminary code's exact `object_class == "pig"` assumption. The frozen schema does not enumerate pig classes.
- Existing preliminary derivation (`scripts/physics_label_derivation.py`) writes a separate canonical JSONL sidecar and source digests, but it combines Milestone 0a with the Milestone 0b oracle gate, groups events by render frame, converts fixed-step debounce counts into render-frame padding, and depends on unverified pig/support-collapse assumptions.
- Existing reader (`world_model/data/supervision.py`) also groups events by render frame and joins preliminary labels only by render frame. It is reusable scaffolding but not authority-correct for event occurrence semantics.

## Semantic Decisions

- Native macro timing must be fixed-step event intervals. Per-RGB labels are deterministic projections onto validated state records, not event occurrence records.
- Event fixed-step clusters are atomic. Event sequence/event IDs remain exact provenance and deterministic tie/order evidence; derivation must not create within-step semantics that depend on taxonomy serialization order.
- A state at fixed step `s` receives all event-cluster effects with occurrence fixed step `<= s`. Event `render_frame` is retained only as source provenance and must never choose onset/termination.
- The Milestone 0a artifact should be separate from `physics_capture_v1` and separate from Milestone 0b's thresholded oracle gate. Proposed family: `physics_macro_labels_v1` with header, fixed-step interval records, per-state projections, and one terminal outcome/equilibrium record.
- Exact keys:
  - shot: `(capture_id, shot_id)`;
  - state/RGB: `(capture_id, shot_id, state_sequence, render_frame, fixed_step, rgb_relative_path)`;
  - event: `(capture_id, shot_id, event_sequence, event_id, fixed_step)`.
- The source contract has no capture-request sequence inside the sidecars; no such key may be invented. External request provenance may be carried only as optional separately validated metadata.
- `steady-state` and direct terminal outcome semantics are engine-verified. `structure-unstable`, `collapsed`, and pre-terminal `pigs-cleared` need bounded operational rules plus representative authorized enriched-shot evidence before research-valid acceptance.
- Fixture-only schema, validator, canonicalization, and semantic correction can be implemented without runtime publication or cohort authorization, provided it does not run the player, write a real cohort, publish an archive, or claim representative semantic acceptance.

## Final Validation Evidence

- `git diff --check`: exit 0; no output.
- Document assertion gate: 29 assertions passed, 0 failed.
- Corrected roadmap: 305 lines, 20,739 bytes, sha256 `8b7397803060a3d732f0b6d235b9486a6adb4c93fbc5780283cdc6cf8412945d`.
- Initial Milestone 0a specification: 431 lines, 25,556 bytes, sha256 `bd66e994b4b4b5c029437e8535f146b699bd49b072e0e03ac885c5ebab6e7546`.
- Required facts/terms present: accepted archive/runtime status, accepted-smoke evidence, continuous-only temporal evidence and claim boundary, fixed-step bracketing, pig tags, unavailable states, outcome/equilibrium semantics, evidence paths, and authority gates.
- Stale terms absent: previous `429cac1d…` digest, the old no-world-model claim, and event alignment by exact event `render_frame`.
- Git status continued to show only pre-existing unrelated untracked artifacts plus the four session-owned planning/deliverable files; no unrelated artifact was modified or removed.
- Independent critic review `critics/2026-08-13-critic-1.md` found one blocking wording contradiction: the specification's opening status prohibited all implementation while later sections authorized fixture-only implementation. The status line was corrected to authorize fixture-only implementation within §15 while preserving runtime/publication/cohort/real-data prohibitions.
- Focused correction validation passed all prior 29 assertions plus three authority checks; `git diff --check` remained clean.
- Corrected Milestone 0a specification: 431 lines, 25,647 bytes, sha256 `e1ad5ef0a4656f8aeb5ab2aade1e37114545382a7d0e3b455339c63ce57013a5`.
- Follow-up critic review `critics/2026-08-13-critic-2.md` found no blocking, important, or optional issue and returned `ACCEPT — safe for the fixture-only implementation`.
