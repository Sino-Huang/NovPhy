# Debug Journal - Todo 8 request-70 contract
Started: 2026-08-06T00:00:00+10:00
Goal: Make Unity request 70 emit deterministic records compatible with the frozen physics_capture_v1 schema while preserving requests 38/62.

## Environment snapshot
- Runtime: exact Unity 2019.4.41f2 editor and staged Linux Mono player.
- Entry: PhysicsCaptureV1Protocol.BuildCaptureEnvelope and PhysicsCaptureDirectSocket.Serve.
- Ports: request 70 on isolated port 2004; no Todo 8 smoke/build/player process at baseline.
- Worktree: shared dirty Unity migration overlay; assigned files are untracked overlay files and must be preserved.
- References read: debugging/SKILL.md, methodology/00-setup.md, methodology/02-investigate.md.

## Hypotheses
1. [CONFIRMED BY LIVE FIELD SET] Unity state serialization omits frozen record-clock identity and sequence fields. Distinguishing evidence: request-70 state keys omit capture_id/sequence; Task 1 record_clock requires them. Fix: runtime capture context.
2. [CONFIRMED BY SOURCE/SCHEMA] Unity serializes recorder-internal contact/support/event names rather than frozen schema names and payloads. Distinguishing evidence: BuildContactsJson/BuildSupportJson/BuildEventsJson keys differ from schema definitions. Fix: contract serializer.
3. [REFUTED] The Python bridge or collector strips valid producer fields. Distinguishing evidence: bridge exposes parsed envelope state unchanged and collector fails immediately at first_state["capture_id"]. Fix if true: bridge parser.
4. [REFUTED] The staged archive is stale relative to a schema-correct serializer. Distinguishing evidence: current source itself lacks required keys and live staged fields match it. Fix if true: rebuild only.

## Toggle proof
- Pending RED test: current serializer must fail frozen required-field comparison at capture_id.
- GREEN toggle: contract serializer must satisfy the same comparison; restoring current serializer restores failure.

## Artifacts to revert
- [ ] `.debug-journal.md` - copy to evidence then remove.
- [ ] `Assets/Tests/Editor/PhysicsCaptureProtocolTests.cs` - promote focused regression.
- [ ] `Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs` - promote minimal request-70 serializer fix.
- [ ] `Assets/Scripts/GroundTruth/PhysicalSnapshotRuntime.cs` - promote request-70 capture context only if required.
- [ ] `.omo/evidence/world-model-physics-instrumentation/task-8-protocol-contract-*` - retain evidence.
- [ ] final `sciencebirdsgames/physics-v1` rebuild - retain staged output only.
- [ ] task-owned player/Java/temp roots/ports/Unity locks - remove before DoneClaim.

## Findings
- Live state fields: coordinates,fixed_step,fixed_time,nodes,raw_contacts,render_frame,render_time,schema_version,support_edges.
- Task 1 requires state capture_id, sequence and synchronized rgb_frame; collector supplies shot_id/record_type/path/dimensions but reads capture_id before writing.
- Contact/support/event serializer fields differ from frozen schema and must be mapped at the producer boundary.

## Final fix
- Completed PIN -> RED -> GREEN -> toggle RED -> toggle GREEN.

## Todo 8 action-event remediation debug round (2026-08-06)

### Hypotheses
1. [OPEN] Scene readiness/current bird is not yet on the sling despite request-62 slingshot presence. Distinguishing evidence: request-62 contains Slingshot but request-70 `nodes` has no Bird before/after the shot; fix is an additional readiness wait or level reset.
2. [OPEN] Action coordinate conversion is accepted by request code but yields a no-op release. Distinguishing evidence: `shoot` returns 1 but a known-valid alternative coordinate/hold/tap changes Bird nodes or emits `bird_launched`; fix is the smallest corrected action geometry.
3. [OPEN] Recorder/event cursor starts or is drained at a timing boundary that misses launch. Distinguishing evidence: immediate and delayed request-70 calls after the same action differ in Bird/event presence; fix is capture timing/cursor ordering.
4. [OPEN] Staged archive/build mismatch prevents the runtime action path. Distinguishing evidence: archive static/provenance differs from the live source or a fresh verified archive clone shows the same failure; fix is archive selection only if proven.

### Probe artifacts to revert/clean
- [ ] `.omo/evidence/world-model-physics-instrumentation/task-8-action-probe-round2/` - bounded direct action matrix; retain final report only.
- [ ] temporary Xvnc/Java processes and ports created by the probe - terminate and verify.

### Oracle Triple synthesis
- Obvious/boundary/invariant agreement: request 31/41 acknowledges HUD input but never calls `PhysicalSnapshotRuntime.BeginShotCallback`; only legacy request 38 arms the authoritative recorder.
- Timing agreement: request 70 correctly reports failure code 4 while the request-38 recorder batch is not finalized; retrying that producer-owned state reaches the finalized event batch.
- Decisive toggle: identical request-62-derived coordinates through request 31/41 yielded `events=[]`; request 38 yielded ordered `stable_entered,bird_launched,stable_exited,entity_destroyed,bird_exhausted,stable_entered,level_failed`.
- Root cause confirmed: wrong action opcode plus immediate request-70 timing. Fix: request 38 and bounded finalization retry.

### Action-event live verification
- Fresh happy report: `task-8-action-live/happy.json`, exit 0, request code 38, response 1, seven nonempty authoritative events, synchronized render frame 16667.
- Independent strict validation: `task-8-action-live-happy-validator.json`, one state, seven events, 6299-byte sidecar, monotonic fixed steps and sequences.
- Fail-closed reports: `task-8-action-live/frame-mismatch.json`, `missing-sidecar.json`, and `request-failure.json`, each exit 1 with null accepted shot and unchanged protected receipts.
- Cleanup/provenance: `task-8-action-final-verification.json`, archive `c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992`, all run ports free, PIDs absent, temporary clones absent.

## Root cause (confirmed 2026-08-06T00:33:53+10:00)
- Mechanism: `PhysicsCaptureV1Protocol` inserted recorder-internal state/contact/support/event fields directly into the request-70 envelope. The collector preserves those records and supplies only shot/file metadata, so missing producer-owned identity, sequence, synchronized RGB metadata, frozen field names, event clocks, IDs, and payloads reached the frozen parser unchanged and caused rejection.
- Evidence: `task-8-protocol-contract-red.xml` reproduces missing `capture_id`; `task-8-live-blocker.log` records the same pre-fix wire field set.
- Toggle proof: removing the corrected state `capture_id` produced `task-8-protocol-toggle-red.xml` with the same failure; restoring it produced `task-8-protocol-toggle-green.xml`, Passed 1/1.
- Fix scope: `PhysicalSnapshotRuntime.cs` owns stable per-level capture identity and monotonic state sequence; `PhysicsCaptureProtocol.cs` maps runtime snapshots to the frozen producer contract; `PhysicsCaptureProtocolTests.cs` locks required fields and identity/sequence behavior.

### Red phase (2026-08-06T00:20:24+10:00)
- Test: `PhysicsCaptureProtocolTests.Request70RecordsMatchFrozenPhysicsCaptureV1ProducerContract`.
- Output: `request-70 producer is missing required field: capture_id` in `task-8-protocol-contract-red.xml`.

### Green phase (2026-08-06T00:28:08+10:00)
- Focused toggle: `task-8-protocol-toggle-green.xml`, Passed 1/1.
- Protocol fixture: `task-8-protocol-fixture.xml`, Passed 8/8.
- Related physical fixtures: `task-8-physical-fixtures.xml`, Passed 9/9.
- Complete partitioned EditMode inventory: `task-8-editmode-partition-receipt.json`, Passed 35/35 across five fixtures.
- Legacy PIN: `task-8-protocol-contract-pin-final.xml`, Passed 1/1 with `LEGACY_GT_SHA256=db0fb151f55403a7a49ae13605fe1c818ad8c00befef9a290cfaddd694eed420`.

## Live verification (2026-08-06T00:33:53+10:00)
- Rebuilt archive SHA-256: `c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992`.
- Static verifier: `task-8-stage-static.json`, checksum and payload verification true.
- Runtime verifier: `task-8-runtime.json`, request 62 true with one feature batch and request 70 true at render frame 9435.
- Writer/schema probe: `task-8-writer-probe.json` plus `task-8-writer-probe/shot_001`, accepted one state with real PNG, capture identity, sequence 1, and frozen parser/validator success.
- Wrong SHA: `task-8-wrong-sha.exit` is 1 and stderr records archive mismatch.
- Protected roots: `task-8-protected-receipt.txt` records canonical project, production player, and active-data comparisons all exit 0.
