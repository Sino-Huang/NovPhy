---
slug: world-model-physics-instrumentation
status: approved-plan-written
intent: clear
review_required: false
pending-action: start .omo/plans/world-model-physics-instrumentation.md in a worker session
approach: Instrument the Unity 2019.3 Science Birds source to produce a versioned, authoritative per-frame physics-state stream and event stream; expose it through a versioned game protocol and Python bridge; have the existing fresh-engine rollout collector atomically persist, validate, and report those sidecars alongside engine-synchronized RGB frames; rebuild and verify the Linux player that the full-dataset launcher clones per worker; then extend the completed data pipeline with an opt-in validated supervision reader. Schedule this work after the approved image/action dataloader plan has completed.
---

# Draft: world-model-physics-instrumentation

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
1 | Versioned Unity physical-state exporter emits stable scene-node snapshots with frame and fixed-step alignment | active | tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs:32-368; GameWorld/ABGameWorld.cs:372-395
2 | Unity contact/support and velocity/energy augmentation emits raw physical facts without image inference | active | GameWorld/ABGameObject.cs:24-35,93-156,258-298; Unity 2019.4 Physics2D APIs
3 | Unity macro-event recorder emits authoritative lifecycle, collision, explosion, and terminal/stability events | active | GameWorld/ABGameWorld.cs:665-789; GameWorld/ABGameObject.cs:102-287
4 | Versioned protocol and Python bridge retrieve aligned physics records from the built player | active | src/webui/bridge.py:35-40,123-125; sciencebirdsagents/Client/agent_client.py:310-341
5 | Existing fresh-engine collection and full-dataset launcher persist, validate, inventory, and resume enriched rollouts safely | active | scripts/collect_rollouts.py:1184-1360,1602-1775; scripts/prepare_rollout_dataset.py:616-733; scripts/collect_full_rollout_training_dataset.sh:232-325
6 | Rebuilt Linux player and end-to-end collection prove schema/version/temporal alignment without disrupting the active cohort | active | README.md:342-346,372-385; sciencebirdsgames/Linux.zip

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
Scheduling | Implement only after `world-model-data-pipeline` is complete | User explicitly sets this dependency; avoids entangling active data collection | yes
Authority | Use Unity-side state and event export, never infer contact/support/energy from RGB frames | Engine already has authoritative Rigidbody/collision facts; RGB inference would be weaker | yes
Support semantics | Preserve raw contacts and derive support from persistent contact, upward normal, and vertical ordering | Unity does not expose a native support predicate; raw facts retain auditability | yes
Time alignment | Record Unity render-frame index, `Time.time`, physics fixed-step index, and `Time.fixedTime` on every state/event record | Existing GT sampling is in `Update` while motion state changes in `FixedUpdate` | yes
Deployment | Rebuild the Unity 2019.3 project into Linux player artifacts, then let the existing launcher clone that enriched player per worker | Workers clone `sciencebirdsgames/Linux`; Python alone cannot modify the executable schema | yes
Data layout | Each accepted `shot_NNN` has separate versioned `physics_state.jsonl` and `physics_events.jsonl` sidecars | User approved the appendable sidecar contract; it avoids rewriting metadata while keeping sparse events separate | no

## Findings (cited - path:lines)
- The current deployed Linux assembly contains `SymbolicGameState`, `AIBirdsConnection`, `ShootAndRecordGroundTruth`, `GroundTruthWithoutScreenshot`, `ABGameObject`, and stability/launch tracking symbols; source-template capability is not merely an unrelated example. Evidence: `.omo/ulw-research/20260727-223250/wave-2-binary-parity-and-physics.md`.
- Existing symbolic state supplies instance IDs, polygons, types/labels, color information, and life, but omits velocity, mass, contacts, support, and macro-event fields. `SymbolicGameState.cs:32-368`; `GTObject.cs:27-96`.
- `ABGameObject` already owns `Rigidbody2D`, stores last velocity in `FixedUpdate`, reads collision relative velocity/mass, and implements destruction. `ABGameObject.cs:24-35,93-156,258-298`.
- Unity 2019.4 documents the needed physical facts: velocity, mass, contact-point collider pair/normal/point/relative velocity/impulses, and allocation-aware `Collider2D.GetContacts`. Research synthesis source 5.
- The active collector uses desktop PNG capture plus ordinary `bridge.shoot`; it consults one-shot symbolic state only for slingshot anchoring. `scripts/collect_rollouts.py:350-364,1184-1335,1465-1502`; `src/webui/bridge.py:123-125`.
- Batch ground truth exists in the legacy client and Unity handler but the modern `ScienceBirdsBridge` does not expose request 38. `sciencebirdsagents/Client/agent_client.py:310-341`; `AIBirdsConnection.cs:327-505`; `src/webui/bridge.py:35-40`.
- The task-template Unity project is version 2019.3.4f1, while workers clone a built Linux player. `tasks/task_template_designer/ProjectSettings/ProjectVersion.txt:1-2`; `scripts/prepare_rollout_dataset.py:663-667`.

## Decisions (with rationale)
- Keep the instrumentation branch separate from the dataloader implementation and do not alter the active raw collection root or currently executing Linux player.
- Treat ground-truth scene nodes as oracle engine records; retain current rich IDs and geometric contours rather than replacing them with image-derived detections.
- Emit raw contacts and separately derived support edges, with the derivation rule and raw evidence versioned in the artifact.
- Compute kinetic energy from emitted physics values rather than attempting to preserve the commented-out damage approximation in `ABGameObject.cs:199-244`.
- Add a new versioned request/response rather than silently changing request 38/62 semantics; preserve existing bridge behavior for current users.
- Require a dedicated live engine smoke test before claiming binary/source protocol parity, because assembly string metadata alone cannot validate a payload.

## Scope IN
- Unity source instrumentation, Linux-player rebuild/distribution, versioned physics-state and macro-event data contract, bridge/client support, collector and full-launcher controls, catalog validation, the narrow post-instrumentation data-pipeline supervision reader, resume/inventory behavior, documentation, and automated/unit plus dedicated-engine integration testing.

## Scope OUT (Must NOT have)
- No modification of the active rollout cohort, no retroactive labels for already-collected RGB-only episodes, no image-inferred physical labels, no learned scene-graph model, no trainer/controller implementation, and no external physics engine. The only data-pipeline change is the explicit opt-in sidecar reader recorded in the approved plan.

## Resolved owner decision
- Each accepted `shot_NNN` writes versioned `physics_state.jsonl` and `physics_events.jsonl` sidecars. The two files are required only for the new enriched capture contract; legacy RGB-only episodes retain their existing validation contract.

## Approval gate
status: approved-plan-written
The approved plan is `.omo/plans/world-model-physics-instrumentation.md`. It remains explicitly sequenced after the data-pipeline plan and does not authorize instrumentation yet.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
