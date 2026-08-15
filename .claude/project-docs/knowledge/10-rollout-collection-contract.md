# Rollout Collection Contract

## Episode Selection

Episode selection is configuration-first:

1. Select a level below `sciencebirdsgames/Linux/9001_Data/StreamingAssets/Levels/`.
2. Generate root `sciencebirdsgames/Linux/config.xml` with `sciencebirdsagents/Utils/PrepareTestConfig.py`.
3. Start a fresh engine.

`config.xml` is global mutable state. Collection must restore it or record the intended resulting configuration. The `novelty_level_N` directory identifies the level family; Unity `ui_level` is separate and dataset collection uses `ui_level=1`.

Scenario suffixes: `01` single force, `02` multiple forces, `03` rolling, `04` falling, `05` sliding.

## Safe Protocol Transitions

Do not use request 51 (`load_level`) or 52 (`restart_level`) as recovery paths. Use bounded transitions:

- new-set states: `ready_for_new_set()`;
- menu/end states: novelty-information preflight followed by request 53;
- unchanged states: debounce repeated transition commands;
- persistent failure: stop, retain evidence, and start a fresh engine for the next permitted attempt.

`PLAYING`, `shoot_response=1`, and process exit zero are necessary signals, not rollout-quality proof.

## Action Contract

The canonical action is final-release based. The verified TCP path has no live finger/drag command.

- `drag_release` and `release` are aliases for the final endpoint.
- `coordinate_frame` defaults to `slingshot_relative`; `coordinate_frame="absolute"` converts with `dx = x - sling_x`, `dy = sling_y - y` using the current slingshot center.
- Existing list, tensor, and `Point2D` action forms remain valid.
- `SBEnvironmentWrapperOpenAI` preserves existing discrete/continuous action spaces; dict actions are an additional manual path before `action_space.contains(...)`.
- `drag_start` is semantic/logging context.
- `drag_release` and `drag_hold_release` identify the final release endpoint.
- `holdTime`/`releaseTime` map to protocol release time; `tapTime` is post-release delay.
- Coordinates use bottom-left game space. Convert Y only at the canvas/socket boundary: `canvas_y = frame_height - 1 - game_y`.
- Resolve the slingshot anchor from request-62 GeoJSON shape data before relative actions.
- Do not claim that a visual guide proves a live protocol drag.

Default actions launch rightward through negative horizontal release offsets. Relation-novelty level 5 requires both negative and positive horizontal signs; plan and resume validation must preserve that requirement.

## Valid Visual Capture

Training-quality capture requires:

- `capture_source: capture_desktop_rollout`;
- a fresh engine per rollout;
- `ui_level: 1`;
- Xvnc-backed rendered desktop capture;
- a pre-shot baseline;
- one consistent cropped 640x480 surface for guard, saved pre-shot image, rollout frames, and post-shot validation;
- accepted-versus-attempt accounting;
- artifact validation before accepted count increments.

Reject and quarantine menu/static captures, uniform TCP screenshots, no-motion captures, missing artifacts, and low-motion captures that do not pass the bounded retry policy. Symbolic renderings can be diagnostics but must never be stored or labeled as pixel rollout frames.

Save the pre-shot frame before firing the deferred shot. Event order is baseline grab, one-shot callback, then post-shot grab; record `shoot_frame_index`. `duration_seconds` is the minimum post-shot duration; continue until visual settling or an explicit maximum cap and record the stop reason and settle thresholds. Review videos may contain overlays, but raw training frames remain unmodified.

`--actions-from-log` replays only stored `trials[*].action` values, disables re-anchoring, and must reproduce the logged action list exactly. Replay equality is an action/provenance assertion, not a guarantee of byte-identical rendered frames.

## Fresh Engine and Worker Isolation

Use Xvnc with GLX/Mesa only after a bounded pre-use probe confirms the host still exposes the required graphical path. Each attempt owns and cleans its process group; terminating only the Java wrapper is insufficient. Detect the visible non-black game viewport on the full Xvnc desktop and crop 640x480 from that origin; use the historical fixed crop only as an explicitly recorded fallback.

Parallel workers require one display and isolated engine copy per worker, distinct port families, process-group cleanup, guarded temporary-root removal, and an isolated/firewalled host with `NOVPHY_ALLOW_NETWORK_LISTENERS=1`. The Java wrapper uses `--game-start-port`, not `--port`.

## Dataset Planning and Resume

The planner deterministically partitions source levels by stable SHA-256 key, preserves split disjointness before cap selection, schedules only permitted splits, preserves unsafe existing outputs as surplus, chooses absent paths instead of overwriting incompatible outputs, creates output directories exclusively, and maintains a locked failure ledger.

Generated plans schedule novelty levels 1 through 8 in deterministic round-robin order before serial emission or worker assignment. Level 0 is excluded from the novelty schedule, and the configured XML remains Unity `ui_level=1`. The current capped contract selects 100 train and 20 dev episodes per discovered normal/novel `(novelty_level, level_type)` bucket; test remains unscheduled. Any future cap change belongs in the collection plan and its digest rather than silently changing this contract.

Resume accepts an episode only when its manifest proves the exact planned capture contract and required artifacts/action logs. Strict numeric fields reject booleans and non-integral substitutes.

Source-level split claims come from plan-backed integration tests. Active-root inspection is a read-only health check and cannot prove leakage freedom or source provenance.

Before large collection, run bounded QA, inspect manifests/action logs/per-shot metadata/quarantine, verify cleanup and ports, and confirm accepted artifacts are gameplay-valid.
