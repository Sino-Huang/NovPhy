# world-model-physics-instrumentation - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** A separately staged Science Birds Linux player and opt-in collector mode that create engine-synchronized RGB frames plus authoritative per-frame physical state and sparse event sidecars for each accepted shot. The resulting new cohort has audit-ready scene nodes, raw contacts, derived support edges, velocity, kinetic energy, macro events, and an opt-in data-pipeline reader for those records.

**Why this approach:** Unity already owns the relevant Rigidbody2D and collision facts, so labels must be exported at the engine rather than inferred from RGB. A versioned direct-socket protocol and separate JSONL sidecars preserve legacy behavior while making partial enriched attempts safely rejectable.

**What it will NOT do:** It will not modify the active rollout root, retrofit labels onto old RGB-only episodes, add a learned scene-graph/world-model component, or use an external physics engine.

**Effort:** XL
**Risk:** High - Unity 2019.3 player rebuilding and exact engine/RGB synchronization are external build/runtime boundaries.
**Decisions to sanity-check:** `physics_state.jsonl` and `physics_events.jsonl` are mandatory only for a new `physics_capture_v1` cohort; support is explicitly derived from retained raw contacts; a staged enriched player is promoted only after a live smoke test.

Your next move: start the approved plan in a worker session. Full execution detail follows below.

---

> TL;DR (machine): XL / high-risk, staged Unity instrumentation and direct-socket capture provide engine-aligned RGB, versioned state/event sidecars, and a narrow opt-in supervision reader without touching the current cohort.

## Scope
### Must have
- Run only after `.omo/plans/world-model-data-pipeline.md` has completed; then extend its explicit capture-contract hook with the narrow, tested `physics_capture_v1` reader in Todo 10. Do not add a trainer, model, controller, or symbolic curriculum.
- Define `physics_capture_v1`: an engine-synchronized RGB image and same-render-frame state record, plus separate `physics_state.jsonl` and `physics_events.jsonl` per accepted shot.
- Export stable-lifetime scene-node IDs, current symbolic geometry/type/life, Rigidbody2D velocity/mass/angular velocity, and kinetic energy `0.5 * mass * (vx^2 + vy^2)` in Unity units.
- Export all non-trigger raw contact points; derive versioned directed support edges from retained contact evidence; record lifecycle/collision/terminal/stability macro events.
- Add a versioned direct Python-to-Unity protocol path that preserves request 38 and request 62 byte-for-byte for existing callers.
- Add collector, validator, launcher, staged-player provenance, tests, dedicated live smoke test, and documentation support for enriched episodes.
### Must NOT have (guardrails, anti-slop, scope boundaries)
- Do not write into `data/novphy_rollouts_dataset_20260708_171531`, alter a running player, or make enriched capture the default.
- Do not call an RGB frame exactly aligned unless it was returned by the new synchronized endpoint; do not assign image-inferred contacts, support, velocity, energy, or events.
- Do not overwrite `sciencebirdsgames/Linux` during development or promote an enriched player before its reproducible build and live-protocol checks pass.
- Do not replace the legacy episode predicate: RGB-only cohorts remain valid under their declared capture contract, while enriched-only consumers explicitly request `physics_capture_v1`.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after. Use Unity 2019.3 EditMode tests for pure schema/contact/event logic and Python `unittest` in `tests/` for protocol, artifact, validator, and launcher behavior.
- Evidence: `.omo/evidence/world-model-physics-instrumentation/task-<N>-<name>.{txt,json,xml}`. Persist commands, fixture hashes, test output, staged-player SHA-256, and the live smoke manifest; never treat console text alone as proof.
- Required non-test evidence: a dedicated temporary output root containing one accepted `physics_capture_v1` shot whose PNG, state records, event stream, manifest provenance, and schema validation are all checked by a machine-readable smoke report.

## Execution strategy
### Parallel execution waves
Wave 1 establishes the contract and engine exporter. Wave 2 adds transport, collector persistence, validation, and staged deployment. Wave 3 proves a rebuilt player through automated and live checks, then documents the public research-data interface.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | data-pipeline plan complete | 2, 3, 4, 5, 6, 7, 9 | none |
| 2 | 1 | 3, 4, 7 | 9 |
| 3 | 1, 2 | 4, 7 | 9 |
| 4 | 1, 2, 3 | 5, 6, 7 | 9 |
| 5 | 1, 4 | 6, 7 | 9 |
| 6 | 1, 4, 5 | 7, 8 | 9 |
| 7 | 1-6 | 8, F1-F4 | 9 |
| 8 | 7 | F1-F4 | 9 |
| 9 | 1, 2, 4, 6, 8, 10 | F1-F4 | 7 |
| 10 | 1, 5, world-model-data-pipeline | 9, F1-F4 | 7 |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. Freeze the `physics_capture_v1` schema, temporal contract, and fixtures.
  What to do / Must NOT do: Add a versioned JSON Schema/contract document and golden JSONL fixtures under a new `docs/data_contracts/` plus test fixtures. Require a state header and every state/event record to carry `schema_version`, `capture_id`, `shot_id`, monotonic `sequence`, Unity render frame/time, fixed-step/fixed time, and coordinate/unit declaration. Define state-node fields (lifetime `entity_id`, Unity instance ID, class/type, screen polygon, world pose, life, velocity, angular velocity, mass, kinetic energy); raw contact fields (two IDs/collider IDs, point, canonical normal, separation, relative velocity, impulse when available); and events (ID, ordered timestamp, taxonomy, participants, payload). Set support v1 precisely: only non-trigger contacts persistent for two consecutive fixed steps, `abs(normal_y) >= 0.5`, and a positive vertical-centre ordering of at least `1e-4` Unity units create one `supporter -> supported` edge; sort ties by `entity_id`, retain ground/static contacts with a synthetic `world:static:<collider-id>` ID, and never derive support from absent contacts. Define one-shot launch, destruction, explosion, per-unordered-pair-per-fixed-step collision, pig removal, bird exhaustion, clear/fail, and debounced stable-enter/exit events. Define the only RGB alignment guarantee: every enriched PNG is returned in the same endpoint response as its snapshot and has identical Unity `render_frame`; desktop captures are never labeled exact. Must NOT encode sidecars in `metadata.json` or claim SI units.
  Parallelization: Wave 1 | Blocked by: `world-model-data-pipeline` completion | Blocks: 2-9.
  References (executor has NO interview context - be exhaustive): `.omo/drafts/world-model-physics-instrumentation.md`; `.omo/ulw-research/20260727-223250/SYNTHESIS.md`; `tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs:32-368`; Unity 2019.4 `Rigidbody2D.velocity`, `Rigidbody2D.mass`, `ContactPoint2D`, and `Collider2D.GetContacts` documentation cited in the synthesis.
  Acceptance criteria (agent-executable): a Python schema/fixture test parses every valid fixture, rejects missing schema/version/time/ID fields, rejects a support edge without two raw-contact samples, and verifies deterministic record/contact/event ordering.
  QA scenarios (name the exact tool + invocation): happy: `python -m unittest tests.test_physics_capture_contract -v`; failure: run its malformed-schema, duplicate-sequence, nonpersistent-support, and mismatched-render-frame cases. Evidence `.omo/evidence/world-model-physics-instrumentation/task-1-contract.txt`.
  Commit: Y | `feat(physics-contract): define versioned engine supervision sidecars`

- [ ] 2. Implement Unity-side physical snapshot export with stable IDs and explicit clocks.
  What to do / Must NOT do: Add focused Unity exporter/registry components alongside `SymbolicGameState` that reset per level, assign `entity_id = <unity-instance-id>:<spawn-ordinal>` for the object's lifetime, and emit the current symbolic-node fields plus Rigidbody2D data at an end-of-render-frame snapshot. Record both `Time.frameCount`/`Time.time` and a monotonic fixed-step counter/`Time.fixedTime`; expose no stale `lastVelocity` as current velocity. Compute kinetic energy from current mass and linear velocity, marking absent bodies/static entities explicitly rather than inventing zero mass. Reuse existing symbolic geometry/type extraction, but leave request 62 payload and development/noise behavior unchanged.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 3, 4, 7.
  References (executor has NO interview context - be exhaustive): `tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs:32-368`; `tasks/task_template_designer/Assets/Scripts/GameWorld/ABGameObject.cs:24-35,54-99,124-244`; `tasks/task_template_designer/Assets/Scripts/GameWorld/ABGameWorld.cs:372-395`; `tasks/task_template_designer/ProjectSettings/ProjectVersion.txt:1-2`.
  Acceptance criteria (agent-executable): EditMode tests instantiate a dynamic and static fixture, assert unique lifetime IDs, current velocity/mass/energy formula, exact render/fixed clock fields, deterministic node order, and unchanged legacy `SymbolicGameState.GetGTJson()` fixture output.
  QA scenarios (name the exact tool + invocation): happy: `"$UNITY_2019_3" -batchmode -nographics -projectPath tasks/task_template_designer -runTests -testPlatform EditMode -testResults .omo/evidence/world-model-physics-instrumentation/task-2-unity.xml -quit`; failure: the same tests cover destroyed/recreated instance-ID reuse and absent Rigidbody2D. Evidence `.omo/evidence/world-model-physics-instrumentation/task-2-unity.xml`.
  Commit: Y | `feat(unity): export aligned physical scene snapshots`

- [ ] 3. Export raw contacts, derived support, and macro events from authoritative callbacks.
  What to do / Must NOT do: Build a bounded per-shot recorder reset on level load and finalized on terminal/timeout. Sample non-trigger `Collider2D.GetContacts` into canonical unordered contact records on each fixed step, preserve all raw facts, and apply support v1 only after the required persistence history. Hook existing launch, collision, death/destruction, pig-removal, TNT/explosion, bird exhaustion, level clear/fail, and stability transitions to append exactly one event according to task 1's taxonomy and stable event ordering. Enforce bounded memory (`max records/bytes` supplied by the capture request); on overflow, timeout, or truncated finalization emit a typed capture failure that makes the shot invalid. Must NOT use image analysis, Unity trigger contacts as physical contacts, commented damage/energy approximations, or repeated `Update` polling as duplicate events.
  Parallelization: Wave 1 | Blocked by: 1, 2 | Blocks: 4, 7.
  References (executor has NO interview context - be exhaustive): `ABGameObject.cs:93-156,258-298`; `ABGameWorld.cs:372-403,665-789`; `tasks/task_template_designer/Assets/Scripts/AIBirdsConnection.cs:327-505,1581-1610`; `.omo/ulw-research/20260727-223250/cause-disappearance.md`; Unity contact API sources in task 1.
  Acceptance criteria (agent-executable): EditMode tests cover two-step support creation/removal, ground/static contact, trigger exclusion, pair ordering/deduplication, collision-per-fixed-step behavior, one-shot launch/destroy/terminal events, and overflow/timeout failure envelopes.
  QA scenarios (name the exact tool + invocation): happy: `"$UNITY_2019_3" -batchmode -nographics -projectPath tasks/task_template_designer -runTests -testPlatform EditMode -testResults .omo/evidence/world-model-physics-instrumentation/task-3-physics.xml -quit`; failure: use fixture contacts that last one tick, reverse normal orientation, and exceed recorder capacity. Evidence `.omo/evidence/world-model-physics-instrumentation/task-3-physics.xml`.
  Commit: Y | `feat(unity): record contacts support and physics macro events`

- [ ] 4. Add a versioned direct-socket enriched-capture protocol and bridge API.
  What to do / Must NOT do: First enumerate the live Unity request dispatch and confirm that the modern `ScienceBirdsBridge` is a direct TCP peer, not dependent on an unavailable Java proxy. Allocate and document unused request code `70` as `GET_PHYSICS_CAPTURE_V1`; its response must be one length-delimited binary envelope containing protocol version, failure code/message or one PNG byte stream plus the exact same-render-frame state/event batch. Use `WaitForEndOfFrame` so RGB and state share `render_frame`; return bounded complete records only, not a mutable live buffer. Extend `RequestCode` and `ScienceBirdsBridge` with strict length/version/JSON parsing and typed exceptions. Preserve request 38/62 framing and behavior exactly; reject unknown/malformed/oversize responses without desynchronizing the socket.
  Parallelization: Wave 2 | Blocked by: 1, 2, 3 | Blocks: 5-7.
  References (executor has NO interview context - be exhaustive): `src/webui/bridge.py:35-40,123-176`; `tests/test_webui_bridge.py`; `sciencebirdsagents/Client/agent_client.py:310-341`; `tasks/task_template_designer/Assets/Scripts/AIBirdsConnection.cs:327-505`; `.omo/ulw-research/20260727-223250/wave-1-local-protocol-and-source.md`.
  Acceptance criteria (agent-executable): fake-socket tests assert request byte `70`, envelope round-trip, PNG/state same-frame equality, exact legacy 38/62 fixtures, and typed rejection on bad magic/version/length/JSON/overflow; Unity integration test returns an actual image and batch from the new handler.
  QA scenarios (name the exact tool + invocation): happy: `python -m unittest tests.test_webui_bridge.PhysicsCaptureV1Tests -v`; failure: `python -m unittest tests.test_webui_bridge.PhysicsCaptureV1MalformedEnvelopeTests -v`. Evidence `.omo/evidence/world-model-physics-instrumentation/task-4-bridge.txt`.
  Commit: Y | `feat(bridge): add versioned synchronized physics capture`

- [ ] 5. Persist enriched shot artifacts atomically and validate their capture contract.
  What to do / Must NOT do: Add an explicit `--physics-capture-v1` collector mode, default off. In this mode consume only the new bridge endpoint for rollout frames, write each attempt into a sibling `shot_NNN.tmp` directory, stream ordered state/event JSONL sidecars, flush/close them, validate schema/frame correspondence, then atomically rename to `shot_NNN`; quarantine invalid completed attempts and remove/ignore incomplete temporary directories on resume. Add `capture_contract: physics_capture_v1`, schema versions, protocol/player/archive checksums, and sidecar relative paths/counts to metadata/manifest. Extend `validate_rollout_artifact` and the canonical episode predicate with an explicit capture-contract switch: legacy RGB-only remains accepted under its old contract, while enriched-required rejects missing/truncated/extra/out-of-order sidecars, duplicated sequence keys, and state/PNG render-frame mismatch. Must NOT read partially written sidecars or claim an ordinary desktop frame is synchronized.
  Parallelization: Wave 2 | Blocked by: 1, 4 | Blocks: 6, 7.
  References (executor has NO interview context - be exhaustive): `scripts/collect_rollouts.py:828-920,992-1056,1184-1360,1580-1598,1744-1782`; `scripts/prepare_rollout_dataset.py:326-438`; `tests/test_collect_rollouts.py`; `tests/test_prepare_rollout_dataset.py`; `.omo/ulw-research/20260727-223250/observation-manifest.md`.
  Acceptance criteria (agent-executable): unit tests prove a successful enriched attempt has exactly one PNG/state record pair per frame and two closed JSONL sidecars; partial tmp directories, malformed JSONL, missing event file, out-of-order keys, and stale/extra frame files are excluded; an RGB-only fixture still passes its unchanged predicate.
  QA scenarios (name the exact tool + invocation): happy: `python -m unittest tests.test_collect_rollouts.PhysicsCapturePersistenceTests tests.test_prepare_rollout_dataset.PhysicsCaptureValidationTests -v`; failure: run their interrupted-write and corrupt-sidecar cases. Evidence `.omo/evidence/world-model-physics-instrumentation/task-5-artifacts.txt`.
  Commit: Y | `feat(collector): persist validated physics supervision sidecars`

- [ ] 6. Add opt-in launcher controls and staged-player provenance without touching the active cohort.
  What to do / Must NOT do: Extend `scripts/collect_full_rollout_training_dataset.sh` and `scripts/prepare_rollout_dataset.py` with a documented `PHYSICS_CAPTURE_V1=1`/CLI equivalent that requires an explicit staged enriched-player directory or archive, an output root different from the active cohort, and a fresh `physics_capture_v1` contract in the plan artifact. Propagate the flag and immutable player archive SHA-256 to every worker-local copy; fail before collection if the staged player/version/protocol smoke marker is absent. Preserve current defaults, train/dev behavior, locks, `RESUME`, and worker cloning; an unset flag must generate byte-equivalent normal collection commands. Do not add test-split collection or mutate source player assets.
  Parallelization: Wave 2 | Blocked by: 1, 4, 5 | Blocks: 7, 8.
  References (executor has NO interview context - be exhaustive): `scripts/collect_full_rollout_training_dataset.sh:25-64,151-251,325-340`; `scripts/prepare_rollout_dataset.py:468-528,549-605,616-733`; `.omo/ulw-research/20260727-223250/wave-2-binary-parity-and-physics.md`.
  Acceptance criteria (agent-executable): command-generation tests prove default commands remain legacy-compatible, enriched mode rejects the active root/missing stage/failed marker, and every enriched worker command includes the same staged-player digest and collector switch.
  QA scenarios (name the exact tool + invocation): happy: `python -m unittest tests.test_prepare_rollout_dataset.PhysicsLauncherTests -v`; failure: `PHYSICS_CAPTURE_V1=1 OUT_ROOT=data/novphy_rollouts_dataset_20260708_171531 bash scripts/collect_full_rollout_training_dataset.sh --help` must exit nonzero before any write, captured in the test harness. Evidence `.omo/evidence/world-model-physics-instrumentation/task-6-launcher.txt`.
  Commit: Y | `feat(dataset): stage opt-in physics rollout collection`

- [ ] 7. Build, package, and test a reproducible Unity 2019.3 enriched Linux player.
  What to do / Must NOT do: Add a Unity Editor build entry point pinned to `2019.3.4f1`, build into a versioned staging path outside `sciencebirdsgames/Linux`, package an archive with a manifest recording Unity/project revision/schema/protocol versions and SHA-256s, and add an explicit rollback rule: failed build/test leaves the current player untouched. Rebuild only after tasks 2-4 pass; unpack the staged archive to a temporary worker clone and execute legacy request 62 plus request 70 compatibility tests. Do not use Unity 2019.4 APIs unavailable in 2019.3, overwrite the production player, or promote on a static assembly-symbol check alone.
  Parallelization: Wave 3 | Blocked by: 1-6 | Blocks: 8, F1-F4.
  References (executor has NO interview context - be exhaustive): `tasks/task_template_designer/ProjectSettings/ProjectVersion.txt:1-2`; `README.md:342-346,372-385`; `tasks/task_template_designer/Assets/Scripts/Editor/`; `sciencebirdsgames/Linux.zip`; `scripts/prepare_rollout_dataset.py:663-667`.
  Acceptance criteria (agent-executable): CI/local script invokes the pinned Unity executable, produces a staged archive and provenance manifest, verifies checksums after unpack, and proves the staged player handles legacy and v1 protocol fixtures; failure preserves hashes of `sciencebirdsgames/Linux` and the active data root.
  QA scenarios (name the exact tool + invocation): happy: `"$UNITY_2019_3" -batchmode -nographics -projectPath tasks/task_template_designer -executeMethod NovPhyBuild.BuildPhysicsLinux -quit && python scripts/verify_physics_player.py --stage sciencebirdsgames/physics-v1`; failure: run `python scripts/verify_physics_player.py --stage sciencebirdsgames/physics-v1 --expect-sha deadbeef` and require failure. Evidence `.omo/evidence/world-model-physics-instrumentation/task-7-build.txt`.
  Commit: Y | `build(unity): package staged physics capture player`

- [ ] 8. Run the dedicated live engine smoke test and establish promotion criteria.
  What to do / Must NOT do: Add a noninteractive smoke driver that starts only a temporary staged-player clone on isolated ports/display, loads a known level, performs one action through request 70, and validates a new temporary output root with the enriched predicate. Assert exact PNG/state render-frame equality, monotonic state/event clocks, nonempty entity set, valid energy calculation, deterministic support/event ordering, archived player provenance, and byte-for-byte unchanged active root/player. Persist a concise machine-readable report and the accepted shot artifacts. Promotion is allowed only when this command and all task 7 checks pass; otherwise the stage is rejected and no collection begins.
  Parallelization: Wave 3 | Blocked by: 7 | Blocks: F1-F4.
  References (executor has NO interview context - be exhaustive): `scripts/collect_rollouts.py:1184-1360`; `src/webui/bridge.py:67-176`; `scripts/collect_full_rollout_training_dataset.sh:151-340`; task 1 contract; task 7 staged-build interface.
  Acceptance criteria (agent-executable): the smoke script exits zero only after its JSON report validates the stated invariants and an accepted `physics_capture_v1` shot exists; an injected wrong render frame, missing sidecar, or failed request produces nonzero and quarantined/no accepted shot.
  QA scenarios (name the exact tool + invocation): happy: `python scripts/smoke_physics_capture.py --stage sciencebirdsgames/physics-v1 --output-dir "$(mktemp -d)/physics-smoke" --report .omo/evidence/world-model-physics-instrumentation/task-8-smoke.json`; failure: `python scripts/smoke_physics_capture.py --stage sciencebirdsgames/physics-v1 --inject-frame-mismatch --output-dir "$(mktemp -d)/physics-smoke"` must fail. Evidence `.omo/evidence/world-model-physics-instrumentation/task-8-smoke.json`.
  Commit: Y | `test(physics-capture): prove staged player end-to-end`

- [ ] 9. Document the research-data contract and loader compatibility boundary.
  What to do / Must NOT do: Update `docs/` with the schema, engine authority, support/event definitions, units, timing guarantee, provenance, legacy-vs-enriched compatibility, sampling/buffer failure behavior, the optional data-pipeline supervision payload from Todo 10, and the exact staged collection/promotion/rollback commands. State that a caller must opt into `capture_contract=physics_capture_v1` and requested capabilities to require sidecars; it may otherwise consume canonical RGB-only episodes as before. Include a compact example query and validation command. Must NOT promise retroactive annotation, expose an implementation-only absolute path, or imply physical labels are visual ground truth.
  Parallelization: Wave 3 | Blocked by: 1, 2, 4, 6, 8, 10 | Blocks: F1-F4.
  References (executor has NO interview context - be exhaustive): `docs/research_proposal.md`; `.omo/ulw-research/20260727-223250/SYNTHESIS.md`; `.omo/plans/world-model-data-pipeline.md`; task 1 contract, task 8 smoke report, and task 10 reader API.
  Acceptance criteria (agent-executable): a documentation test/example validates against the real JSON Schema, references the generated staged-player provenance/report, and a link/path checker finds no stale capture command or claim that RGB-only data has physics sidecars.
  QA scenarios (name the exact tool + invocation): happy: `python scripts/verify_physics_capture_docs.py docs`; failure: its fixture with a removed required timing/provenance clause must fail. Evidence `.omo/evidence/world-model-physics-instrumentation/task-9-docs.txt`.
  Commit: Y | `docs(physics-capture): publish engine supervision contract`

- [ ] 10. Add the post-instrumentation `physics_capture_v1` supervision reader to the data pipeline.
  What to do / Must NOT do: After Todo 5's shared artifact validator accepts `physics_capture_v1`, extend the already-implemented `world_model/data` capture-contract registry and `EpisodeCatalog` to validate and expose only schema-valid sidecars. Add typed immutable `PhysicsFrameSupervision` and `PhysicsEvent` records plus a `PhysicsSupervisionRequest(required_capabilities, include_raw_contacts, include_events)` to parse each selected shot's `physics_state.jsonl` and `physics_events.jsonl`. Match each requested RGB frame to exactly one state record by `(shot_id, render_frame)`; return variable-cardinality node/contact/support/event collections as immutable nested records under an opt-in `supervision` sample field, and leave the existing RGB/action sample and collator unchanged by default. Required capabilities must fail closed for RGB-only, incomplete, stale-schema, or nonaligned records. Preserve raw contacts independently of derived support, expose velocity/mass/kinetic-energy units/provenance as recorded, and never manufacture labels or convert variable graph data into padded tensors. Do not change temporal-only curriculum/ablation semantics, add a model/trainer, or make enriched supervision the default.
  Parallelization: Wave 3 | Blocked by: 1, 5, completed `world-model-data-pipeline` | Blocks: 9, F1-F4.
  References (executor has NO interview context - be exhaustive): `.omo/plans/world-model-data-pipeline.md:20-43,72-86,96-118`; `scripts/rollout_artifacts.py` and `world_model/data/{catalog.py,dataset.py,inspect.py}` created by that plan; task 1 schema; task 5 validator/persistence contract; `scripts/collect_rollouts.py:828-920,1184-1360`.
  Acceptance criteria (agent-executable): targeted tests create a valid enriched fixture and prove `EpisodeCatalog.build(..., capture_contract="physics_capture_v1", required_capabilities=(...))` accepts it only after shared validation; the dataset returns frame-exact immutable supervision records with raw contacts and derived support distinct; default RGB samples never open sidecars; and legacy RGB-only catalogs remain unchanged.
  QA scenarios (name the exact tool + invocation): happy: `python -m unittest tests.test_world_model_data.PhysicsSupervisionReaderTests -v`; failure: `python -m unittest tests.test_world_model_data.PhysicsSupervisionReaderTests.test_rejects_missing_corrupt_stale_or_render_misaligned_sidecars -v` and `python -m unittest tests.test_world_model_data.PhysicsSupervisionReaderTests.test_required_capability_rejects_legacy_rgb_episode -v`. Evidence `.omo/evidence/world-model-physics-instrumentation/task-10-loader.txt`.
  Commit: Y | `feat(world-model-data): read validated physics supervision sidecars`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  Verify every todo's outputs, guardrails, schema fields, test receipts, commit boundaries, and dependency order against this plan; reject missing sidecar/temporal/provenance proof. Evidence `.omo/evidence/world-model-physics-instrumentation/f1-plan-compliance.txt`.
- [ ] F2. Code quality review
  Review Unity/C#/Python/shell changes for legacy protocol compatibility, bounded-memory behavior, atomic artifact lifecycle, deterministic ordering, and 2019.3 API compatibility. Evidence `.omo/evidence/world-model-physics-instrumentation/f2-code-quality.txt`.
- [ ] F3. Real manual QA
  Execute task 8's dedicated staged-player smoke flow plus legacy request 62 interaction and inspect machine-produced output only; verify the active cohort/player checksums did not change. Evidence `.omo/evidence/world-model-physics-instrumentation/f3-live-qa.json`.
- [ ] F4. Scope fidelity
  Compare changed paths and manifests to scope: only new cohort/stage artifacts, no active-root modification, no RGB inference, no trainer/model work, and no silent legacy contract replacement. Evidence `.omo/evidence/world-model-physics-instrumentation/f4-scope.txt`.

## Commit strategy
- Keep commits independently reviewable in todo order. Do not mix generated player archives or smoke output with source commits; release/stage provenance is an explicit build artifact produced only after source tests pass.
- Suggested sequence: contract; Unity state; Unity relations/events; protocol/bridge; collector/validator; launcher; staged build; smoke; docs.

## Success criteria
- A fresh staged `physics_capture_v1` episode has an engine-synchronized PNG and exact same-render-frame state record for every frame, closed/validated state and event sidecars, authoritative scene/contact/velocity/energy/event facts, deterministic support edges, and complete provenance.
- An opt-in world-model data sample can expose only schema-valid, exact-frame `physics_capture_v1` supervision records while its default RGB/action path, temporal curriculum, and legacy RGB-only catalog behavior remain unchanged.
- Existing request 38/62 callers, normal collector commands, and canonical RGB-only data still pass their preexisting test fixtures.
- Enriched capture refuses unsafe roots, unstaged/unverified players, partial sidecars, malformed schema, protocol incompatibility, and temporal mismatch before producing an accepted episode.
- The pinned Unity 2019.3 build, Python tests, staged-player verification, and live smoke report all pass before any promotion or full enriched rollout collection.
