# Fix NovPhy Rollout Static/Menu Shot Bug

## TL;DR
> **Summary**: Investigate and fix invalid rollout samples where later shots are static or capture the SELECT LEVEL menu despite `shoot_response=1`. The plan locks the known bad artifact with a validator, proves the root cause with runtime evidence, then adds collector-side pre/post shot validity gates and bounded retry/quarantine behavior.
> **Deliverables**:
> - Machine-readable rollout artifact validator for static/menu/invalid shots.
> - Regression tests for known bad `shot_002` and `shot_003` artifacts.
> - Collector-side pre-shot gameplay guard and post-shot acceptance gate.
> - Bounded retry/quarantine metadata for invalid shots.
> - Bounded real Xvnc/Java QA run proving accepted shots are gameplay rollouts.
> **Effort**: Medium
> **Parallel**: YES - 4 waves
> **Critical Path**: Task 1 → Task 2 → Task 4 → Task 6 → Final Verification

## Context

### Original Request
The user reported that in `data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1`, `shot_001` looks OK, `shot_002` looks static because `frames/frame_000000.png` and `frames/frame_000043.png` appear the same, and `shot_003` is in the SELECT LEVEL menu rather than the gameplay scene. The user asked for careful Java engine/WebUI mechanism investigation and a plan to solve the bug.

### Interview Summary
- The work must be planned first; implementation happens later via `/start-work`.
- TDD is required for production changes.
- Full dataset rerun is out of scope until a bounded fix is proven.
- WebUI behavior is comparison evidence, not permission to rewrite WebUI routes.

### Research Findings
- Frame inspection confirmed `shot_002` is gameplay/static-looking and `shot_003/frame_000000.png` is SELECT LEVEL.
- Artifact metadata:
  - `shot_001`: `shoot_response=1`, `frame_count=35`, final state sample `PLAYING`, score `1770`, `max_frame_delta=962`.
  - `shot_002`: `shoot_response=1`, `frame_count=44`, final state sample `PLAYING`, score `1210`, `max_frame_delta=22`; tiny localized motion only.
  - `shot_003`: `shoot_response=1`, `frame_count=29`, final state sample `PLAYING`, score `0`, `max_frame_delta=0`, `max_pre_shot_delta=0`; visual frames are SELECT LEVEL.
- `scripts/collect_rollouts.py::collect_rollouts()` same-engine loop does not inherently re-prepare/restart/load/select between shots; it optionally calls a `reset_rollout` callback, then shoots and captures.
- `scripts/collect_rollouts.py::collect_fresh_engine_rollouts()` starts a fresh engine per action, configures, prepares, and can call xdotool `select_level_in_display(ui_level)`.
- `scripts/manual_agent.py::prepare_for_play()` and WebUI startup use protocol transitions: `ready_for_new_set()` for new-set states and `get_novelty_info()` + `load_next_available_level()` for menu/end states.
- `src/webui/server.py` `/api/shot` does not re-prepare before each shot.
- `src/webui/server.py` `/api/load-level` and `/api/restart` route to `load_next_available_level()`; `src/webui/README.md` says protocol `51`/`52` can hang, so they must not become default recovery.

### Metis Review (gaps addressed)
- Do not trust `shoot_response=1` as proof of valid gameplay.
- Separate root-cause investigation, validity gating, and bounded recovery.
- Do not default to protocol `51`/`52`.
- Do not silently discard, overwrite, or accept invalid shots.
- Do not change WebUI route semantics unless separately scoped.
- Require machine-readable invalid reasons and bounded retry/quarantine.
- Require bounded real Xvnc/Java QA before any full dataset run.

## Work Objectives

### Core Objective
Ensure rollout collection never accepts static/menu/non-gameplay shots as valid dataset samples, and make any recovery/retry/quarantine decision explicit in metadata.

### Deliverables
- New artifact validator module or CLI function that classifies shot directories using metadata and frame evidence.
- Tests proving known bad artifacts are rejected/suspicious before collection fixes.
- Collector pre-shot guard that verifies gameplay surface before shooting.
- Collector post-shot acceptance gate that rejects menu captures and suspicious no-motion captures.
- Bounded retry/quarantine behavior with explicit metadata fields.
- Real Xvnc/Java bounded QA evidence under `.omo/evidence/`.

### Definition of Done (verifiable conditions with commands)
```sh
source ~/cd_novphy && python -m unittest tests.test_collect_rollouts tests.test_manual_agent tests.test_prepare_rollout_dataset
source ~/cd_novphy && python -m py_compile scripts/collect_rollouts.py scripts/manual_agent.py tests/test_collect_rollouts.py tests/test_manual_agent.py
GIT_MASTER=1 git diff --check -- scripts/collect_rollouts.py scripts/manual_agent.py tests/test_collect_rollouts.py tests/test_manual_agent.py .omo/knowledges/fresh-engine-rollout-collection.md
```
Expected: all commands exit `0`; LSP diagnostics may still be unavailable if `basedpyright-langserver` is not installed. Task 7's one-level bounded Xvnc/Java QA command must also exit `0` before the implementation is considered complete.

### Must Have
- Known bad `shot_003` must be detected as invalid with reason `menu_detected` or `capture_target_invalid`.
- `shot_002` must be detected as invalid or suspicious with reason such as `no_frame_motion`, `no_bird_motion`, or `low_motion_suspicious`; the threshold must be documented and tested against `shot_001` as valid.
- Accepted new shots must have gameplay pre-shot evidence and non-menu post-shot evidence.
- Invalid shots must not silently count as valid collected shots.
- Retry attempts and invalid reasons must be written to metadata/action logs.

### Must NOT Have
- No full dataset rerun before bounded QA passes.
- No default use of protocol `load_level(51)` or `restart_level(52)`.
- No broad Java engine rewrite.
- No WebUI `/api/load-level` or `/api/restart` semantic changes unless a separate user decision expands scope.
- No manual-only acceptance criteria.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: TDD with existing `unittest` framework.
- QA policy: Every task has agent-executed scenarios.
- Evidence: `.omo/evidence/task-{N}-{slug}.{ext}`.
- Runtime QA: bounded Xvnc/Java collector run only, never full dataset.

## Execution Strategy

### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Tasks 1-3 (artifact validator, mechanism instrumentation plan/test seams, WebUI comparison docs)
Wave 2: Tasks 4-6 (collector pre-shot guard, post-shot gate, retry/quarantine metadata)
Wave 3: Tasks 7-8 (bounded runtime QA, documentation/knowledge update)
Wave 4: Final Verification Wave

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 4, 5, 6, 7.
- Task 2 blocks Tasks 4, 6, 7.
- Task 3 informs Tasks 4 and 8 but does not block implementation if completed after Task 1.
- Task 4 blocks Tasks 5, 6, 7.
- Task 5 blocks Tasks 6, 7.
- Task 6 blocks Task 7.
- Task 7 blocks final verification.
- Task 8 can run after Tasks 1-7.

### Agent Dispatch Summary
- Wave 1 → 3 tasks → categories: `deep`, `quick`, `writing`
- Wave 2 → 3 tasks → categories: `deep`, `quick`, `unspecified-high`
- Wave 3 → 2 tasks → categories: `unspecified-high`, `writing`
- Final → 4 tasks → categories: oracle / unspecified-high / deep

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Build rollout artifact validity validator and lock known bad samples

  **What to do**: Add a small validator in `scripts/collect_rollouts.py` or a helper module imported by it that can classify an existing `shot_*/` directory using `metadata.json`, `pre_shot.png`, and sampled frames. It must detect at minimum: gameplay-valid, menu-detected, no-frame-motion, low-motion-suspicious, and missing-artifact. Add `unittest` coverage in `tests/test_collect_rollouts.py` using the concrete artifact path `data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1`. The test must assert `shot_001` valid, `shot_003` invalid menu/static, and `shot_002` invalid or suspicious.
  **Must NOT do**: Do not depend on a manual visual assertion. Do not use brittle single-pixel matching as the only menu detector. Do not mutate the existing dataset artifact.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: combines image/metadata analysis with regression design.
  - Skills: [`debugging`] - root-cause evidence and regression-first discipline.
  - Omitted: [`playwright`] - no browser UI is required for artifact validation.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 5, 6, 7 | Blocked By: none

  **References**:
  - Artifact: `data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1/shot_001/metadata.json` - valid baseline with `max_frame_delta=962`.
  - Artifact: `data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1/shot_002/metadata.json` - suspicious low motion with `max_frame_delta=22`.
  - Artifact: `data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1/shot_003/pre_shot.png` - SELECT LEVEL menu evidence.
  - Pattern: `scripts/collect_rollouts.py` - existing metadata/frame-delta fields and rollout capture helpers.
  - Test: `tests/test_collect_rollouts.py` - existing collector unit test style.

  **Acceptance Criteria**:
  - [ ] `source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_known_bad_dataset_artifact_classifies_static_and_menu_shots` exits `0` after implementation and fails before validator/fix exists.
  - [ ] Validator result for `shot_001` includes `accepted: true` and no invalid reason.
  - [ ] Validator result for `shot_002` includes `accepted: false` or `suspicious: true` with `invalid_reason`/`warning_reason` in `no_frame_motion|no_bird_motion|low_motion_suspicious`.
  - [ ] Validator result for `shot_003` includes `accepted: false` and `invalid_reason` in `menu_detected|capture_target_invalid|no_frame_motion`.

  **QA Scenarios**:
  ```
  Scenario: Known good and bad artifacts classify deterministically
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_known_bad_dataset_artifact_classifies_static_and_menu_shots
    Expected: exits 0; shot_001 valid, shot_002 suspicious/invalid, shot_003 invalid menu/static.
    Evidence: .omo/evidence/task-1-artifact-validator.txt

  Scenario: Missing shot directory fails closed
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_rollout_artifact_validator_rejects_missing_frames
    Expected: exits 0; missing frames return accepted=false and invalid_reason=missing_artifact.
    Evidence: .omo/evidence/task-1-artifact-validator-missing.txt
  ```

  **Commit**: YES | Message: `test(rollouts): classify invalid rollout artifacts` | Files: [`scripts/collect_rollouts.py`, `tests/test_collect_rollouts.py`]

- [x] 2. Add runtime state evidence instrumentation before changing recovery behavior

  **What to do**: Extend collector metadata for attempted shots to record protocol state before pre-shot capture, after recovery, after `shoot`, and after capture; also record `current_level` when safely available, recovery action, and capture surface classification. Tests should use fake bridge/capture to assert metadata is recorded even when the final surface is invalid.
  **Must NOT do**: Do not assume protocol state is authoritative. Do not enable protocol `51`/`52`.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: localized metadata additions and tests.
  - Skills: [`debugging`] - instrumentation must support root-cause confirmation.
  - Omitted: [`playwright`] - no browser flow required.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 6, 7 | Blocked By: none

  **References**:
  - Pattern: `scripts/collect_rollouts.py` - existing per-shot `metadata.json` and manifest writing.
  - API: `src/webui/bridge.py` - `get_game_state()`, `get_current_level()`, `ready_for_new_set()`, `load_next_available_level()`.
  - Test: `tests/test_collect_rollouts.py` - fake bridge and metadata assertions.

  **Acceptance Criteria**:
  - [ ] Per-shot metadata includes keys: `pre_shot_protocol_state`, `post_recovery_protocol_state`, `shoot_response`, `post_capture_protocol_state`, `recovery_action`, and `artifact_validation`.
  - [ ] Unit test proves metadata is still written when artifact validation rejects a shot.

  **QA Scenarios**:
  ```
  Scenario: Valid shot records protocol evidence
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_collect_rollouts_records_protocol_state_evidence
    Expected: exits 0; metadata contains all required state/recovery fields.
    Evidence: .omo/evidence/task-2-state-evidence.txt

  Scenario: Invalid shot records rejection evidence
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_invalid_rollout_records_state_and_reason
    Expected: exits 0; invalid metadata includes artifact_validation.accepted=false and invalid_reason.
    Evidence: .omo/evidence/task-2-invalid-state-evidence.txt
  ```

  **Commit**: YES | Message: `feat(rollouts): record shot state evidence` | Files: [`scripts/collect_rollouts.py`, `tests/test_collect_rollouts.py`]

- [x] 3. Document Java/WebUI mechanism boundaries without changing WebUI behavior

  **What to do**: Update `.omo/knowledges/fresh-engine-rollout-collection.md` with the mechanism comparison: WebUI startup uses protocol-only recovery; `/api/shot` does not re-prepare; `/api/load-level` and `/api/restart` use protocol `53`; protocol `51`/`52` exist but can hang. Include this as implementation context, not as a WebUI change request.
  **Must NOT do**: Do not edit WebUI source routes. Do not claim WebUI is bug-free.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: documentation of mechanism and guardrails.
  - Skills: [] - no special skill needed.
  - Omitted: [`debugging`] - no runtime debugging required in this task.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: none | Blocked By: none

  **References**:
  - Pattern: `src/webui/server.py` - startup recovery and route behavior.
  - API: `src/webui/bridge.py` - protocol methods `51`, `52`, `53`, `68`, `69`.
  - Doc: `src/webui/README.md` - rationale for avoiding `51`/`52`.
  - Knowledge: `.omo/knowledges/fresh-engine-rollout-collection.md` - existing rollout history.

  **Acceptance Criteria**:
  - [ ] Knowledge note includes explicit guardrail: collector fix must not default to protocol `51`/`52`.
  - [ ] Knowledge note includes explicit guardrail: WebUI route semantics are out of scope.

  **QA Scenarios**:
  ```
  Scenario: Knowledge note contains WebUI/protocol guardrails
    Tool: Bash
    Steps: GIT_MASTER=1 git diff -- .omo/knowledges/fresh-engine-rollout-collection.md
    Expected: diff contains protocol 51/52 guardrail and WebUI out-of-scope note.
    Evidence: .omo/evidence/task-3-knowledge-diff.txt

  Scenario: WebUI source untouched
    Tool: Bash
    Steps: GIT_MASTER=1 git diff --name-only -- src/webui/server.py src/webui/bridge.py scripts/webui.sh
    Expected: no output.
    Evidence: .omo/evidence/task-3-webui-untouched.txt
  ```

  **Commit**: YES | Message: `docs(rollouts): record engine recovery guardrails` | Files: [`.omo/knowledges/fresh-engine-rollout-collection.md`]

- [x] 4. Add pre-shot gameplay guard for collector attempts

  **What to do**: Before each shot attempt that will be accepted into the dataset, require a pre-shot guard that combines protocol state and visual artifact classification. If protocol state is not `PLAYING` or pre-shot frame is menu/invalid, run bounded recovery using existing safe protocol path: `prepare_for_play()` style transitions with `ready_for_new_set()` and `get_novelty_info()` + `load_next_available_level()`. In desktop/fresh-engine mode, do not call `select_level_in_display()` repeatedly unless explicitly configured and bounded. Tests must simulate protocol `PLAYING` with menu-looking frame and assert the guard rejects/retries rather than shooting blindly.
  **Must NOT do**: Do not trust protocol `PLAYING` alone. Do not use `load_level(51)` or `restart_level(52)` by default. Do not loop forever.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: collector state machine and fake-runtime tests.
  - Skills: [`debugging`] - root-cause confirmation and minimal fix discipline.
  - Omitted: [`playwright`] - no browser QA needed.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 5, 6, 7 | Blocked By: 1, 2

  **References**:
  - Pattern: `scripts/manual_agent.py::prepare_for_play()` - safe recovery state machine.
  - Pattern: `scripts/collect_rollouts.py::collect_rollouts()` - same-engine loop requiring guard.
  - Pattern: `scripts/collect_rollouts.py::collect_fresh_engine_rollouts()` - fresh-engine per-attempt preparation.
  - Test: `tests/test_manual_agent.py` - state transition tests for LOST and NEWTRIAL.

  **Acceptance Criteria**:
  - [ ] Unit test with fake protocol `PLAYING` plus menu frame fails before fix and passes after, proving pre-shot guard rejects menu surface.
  - [ ] Unit test with `NEWTRIAL -> PLAYING` proves safe recovery still reaches shooting path.
  - [ ] Recovery attempts are bounded by a configurable count/default, and timeout raises/records `recovery_failed`.

  **QA Scenarios**:
  ```
  Scenario: Pre-shot guard rejects protocol-playing menu surface
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_pre_shot_guard_rejects_menu_surface_even_when_protocol_playing
    Expected: exits 0; shot is not accepted and invalid_reason=menu_detected or capture_target_invalid.
    Evidence: .omo/evidence/task-4-preshot-menu-guard.txt

  Scenario: Pre-shot guard recovers NEWTRIAL to gameplay
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_pre_shot_guard_recovers_new_trial_before_shooting
    Expected: exits 0; ready_for_new_set called once, shot proceeds only after gameplay-valid pre-shot.
    Evidence: .omo/evidence/task-4-preshot-recovery.txt
  ```

  **Commit**: YES | Message: `fix(rollouts): guard gameplay before shooting` | Files: [`scripts/collect_rollouts.py`, `tests/test_collect_rollouts.py`]

- [x] 5. Add post-shot acceptance gate and invalid-shot classification

  **What to do**: After capture, run the validator from Task 1 on the newly captured shot. Accepted shots must pass menu/static checks. Invalid shots must be marked with `artifact_validation.accepted=false`, `invalid_reason`, and `retryable` decision. Suspicious low-motion shots must be configurable: default should reject if it resembles the known `shot_002` pattern and no stronger gameplay/bird-motion evidence exists.
  **Must NOT do**: Do not delete invalid shot directories silently. Do not count invalid shots as successful rollouts.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: localized post-capture validation wiring once validator exists.
  - Skills: [`debugging`] - failure evidence must be explicit.
  - Omitted: [`playwright`] - no browser required.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6, 7 | Blocked By: 1, 4

  **References**:
  - Pattern: `scripts/collect_rollouts.py` - per-shot metadata writing and manifest entries.
  - Artifact: known bad `shot_003` - menu invalid baseline.
  - Artifact: known suspicious `shot_002` - low-motion threshold baseline.

  **Acceptance Criteria**:
  - [ ] Invalid captured shot writes metadata with `accepted=false` and `invalid_reason` before any retry/quarantine.
  - [ ] Manifest/action log distinguish attempted shots from accepted shots.
  - [ ] Accepted count excludes invalid attempts.

  **QA Scenarios**:
  ```
  Scenario: Post-shot menu capture rejected
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_post_shot_gate_rejects_menu_capture
    Expected: exits 0; manifest accepted count excludes menu attempt, metadata records invalid_reason=menu_detected.
    Evidence: .omo/evidence/task-5-postshot-menu.txt

  Scenario: Post-shot low-motion capture rejected or flagged
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_post_shot_gate_flags_low_motion_capture
    Expected: exits 0; low-motion pattern comparable to shot_002 is not silently accepted.
    Evidence: .omo/evidence/task-5-postshot-low-motion.txt
  ```

  **Commit**: YES | Message: `fix(rollouts): reject invalid captured shots` | Files: [`scripts/collect_rollouts.py`, `tests/test_collect_rollouts.py`]

- [x] 6. Add bounded retry/quarantine behavior for invalid attempts

  **What to do**: When pre/post validation rejects an attempt, retry using the safest existing lifecycle for the active mode. Fresh-engine mode should start a new engine attempt up to `--fresh-engine-attempts`; same-engine mode should use bounded safe recovery or fail/quarantine when recovery cannot prove gameplay. Add metadata fields: `attempt_status`, `accepted`, `invalid_reason`, `retry_attempt`, `recovery_action`, and `quarantined_path` when applicable. Invalid attempts should be moved or copied to a deterministic `invalid_attempts/` subdirectory under the shot or episode root, not overwritten silently.
  **Must NOT do**: Do not unbounded-retry. Do not hide invalid artifacts. Do not bias action logs by omitting failed attempts.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: coordinates metadata, filesystem behavior, and lifecycle tests.
  - Skills: [`debugging`] - runtime failure semantics.
  - Omitted: [`playwright`] - no browser needed.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7 | Blocked By: 2, 4, 5

  **References**:
  - Pattern: `scripts/collect_rollouts.py::collect_fresh_engine_rollouts()` - existing per-action fresh-engine retry semantics.
  - Pattern: `scripts/prepare_rollout_dataset.py` - generated script failure ledger for per-level failures.
  - Test: `tests/test_collect_rollouts.py` - retry tests for fresh-engine prepare timeout.

  **Acceptance Criteria**:
  - [ ] Test proves invalid first attempt is retained/quarantined and second valid attempt becomes accepted shot.
  - [ ] Test proves retry exhaustion returns nonzero/raises and records `recovery_failed` or final invalid reason.
  - [ ] Action log contains both attempts or contains accepted trial plus explicit invalid attempt ledger.

  **QA Scenarios**:
  ```
  Scenario: Invalid first attempt retries and preserves evidence
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_invalid_attempt_retries_and_quarantines_evidence
    Expected: exits 0; invalid_attempts exists, accepted shot metadata references retry_attempt=2.
    Evidence: .omo/evidence/task-6-retry-quarantine.txt

  Scenario: Retry exhaustion fails closed
    Tool: Bash
    Steps: source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_invalid_attempt_retry_exhaustion_fails_closed
    Expected: exits 0; no invalid attempt counted as accepted, final reason recorded.
    Evidence: .omo/evidence/task-6-retry-exhaustion.txt
  ```

  **Commit**: YES | Message: `fix(rollouts): retry and quarantine invalid attempts` | Files: [`scripts/collect_rollouts.py`, `tests/test_collect_rollouts.py`]

- [x] 7. Run bounded real Xvnc/Java collector QA on the reported level

  **What to do**: Run a bounded collector check, not a full dataset run. Use a fresh display and isolated output directory under `data/rollout_bugfix_check_<timestamp>` or `/tmp/opencode/rollout_bugfix_check_<timestamp>`. Target the same installed level/config path used by the bad artifact. Collect a small count such as 3-5 shots. Save logs, manifest, metadata, validator output, and representative frame paths to `.omo/evidence/`.
  **Must NOT do**: Do not run the full dataset. Do not reuse the existing bad output directory. Do not require manual frame inspection.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: real runtime QA with process cleanup.
  - Skills: [`debugging`] - actual runtime evidence and artifact cleanup discipline.
  - Omitted: [`playwright`] - not browser UI unless explicitly comparing WebUI.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final Verification | Blocked By: 4, 5, 6

  **References**:
  - Xvnc setup pattern only: `scripts/collect_full_rollout_training_dataset.sh` - reference for display startup/cleanup conventions; do not execute this full train/dev launcher for bounded QA.
  - Runtime path: `sciencebirdsgames/Linux/config.xml` - level config mutated by dataset helper.
  - Artifact: bad episode path - target for comparison.

  **Acceptance Criteria**:
  - [ ] Bounded run produces no accepted shot with SELECT LEVEL frames.
  - [ ] Bounded run produces no accepted shot with `max_frame_delta=0` unless explicitly classified as valid by stronger evidence; default should reject.
  - [ ] Validator CLI/check returns `accepted=true` for every accepted shot and logs invalid attempts separately.
  - [ ] Bounded QA command exits `0`; any nonzero exit is a failed QA result requiring investigation and rerun.
  - [ ] Bounded QA command targets exactly one configured level, not the full train/dev launcher.
  - [ ] No stale Xvnc/Java collector process remains after command exits.

  **QA Scenarios**:
  ```
  Scenario: Bounded one-level fresh collector run rejects menu/static attempts
    Tool: Bash
    Steps: Run exactly this bounded script, not `scripts/collect_full_rollout_training_dataset.sh`:
      ```sh
      set -euo pipefail
      source ~/cd_novphy
      mkdir -p .omo/evidence /tmp/opencode/rollout_bugfix_check
      python scripts/prepare_rollout_dataset.py write-config \
        --manifest data/novphy_rollouts_dataset_plan/partitions.json \
        --split train \
        --level-path 9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml
      Xvnc :153 -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 \
        >/tmp/opencode/rollout_bugfix_check/xvnc.log 2>&1 &
      XVNC_PID=$!
      trap 'kill "$XVNC_PID" 2>/dev/null || true; wait "$XVNC_PID" 2>/dev/null || true' EXIT
      sleep 2
      DISPLAY=:153 LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}" \
        timeout -s INT 15m python scripts/collect_rollouts.py \
          --output-dir /tmp/opencode/rollout_bugfix_check/out \
          --capture-source desktop \
          --fresh-engine-per-rollout \
          --ui-level 1 \
          --count 3 \
          --fps 30 \
          --duration 5 \
          --connect-timeout 60 \
          --prepare-timeout 90 \
          --read-timeout 420 \
          --speed 1 \
        > .omo/evidence/task-7-runtime.log 2>&1
      STATUS=$?
      kill "$XVNC_PID" 2>/dev/null || true
      wait "$XVNC_PID" 2>/dev/null || true
      trap - EXIT
      exit "$STATUS"
      ```
    Expected: exits 0; output directory contains exactly the bounded one-level check; validator output for every accepted shot is accepted=true; invalid attempts, if any, are logged separately with machine-readable reasons.
    Evidence: .omo/evidence/task-7-runtime.log

  Scenario: Runtime cleanup leaves no matching processes
    Tool: Bash
    Steps: ps -eo pid,ppid,stat,etime,cmd | rg 'rollout_bugfix_check|Xvnc :153|game_playing_interface|collect_rollouts.py' || true
    Expected: no running collector, Java, or Xvnc process for the bounded QA display/output remains.
    Evidence: .omo/evidence/task-7-process-cleanup.txt
  ```

  **Commit**: NO | Message: `n/a` | Files: [runtime evidence only]

- [x] 8. Update rollout knowledge and operator guidance

  **What to do**: Update `.omo/knowledges/fresh-engine-rollout-collection.md` with the root cause, validator rules, retry/quarantine semantics, bounded QA command, and command operators should run after fix. Include warning not to trust pre-fix artifacts containing accepted static/menu shots.
  **Must NOT do**: Do not claim the entire existing dataset is valid. Do not instruct a full rerun unless bounded QA has passed.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: operational documentation.
  - Skills: [] - no special skill needed.
  - Omitted: [`debugging`] - runtime debugging already complete.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: Final Verification | Blocked By: 1, 5, 6, 7

  **References**:
  - Knowledge: `.omo/knowledges/fresh-engine-rollout-collection.md` - existing collection history.
  - Evidence: `.omo/evidence/task-7-runtime.log` - bounded runtime QA results.

  **Acceptance Criteria**:
  - [ ] Knowledge note includes invalid-shot reasons and retry/quarantine behavior.
  - [ ] Knowledge note includes exact bounded verification command used and result.
  - [ ] Knowledge note warns existing pre-fix dataset samples may need validation before use.

  **QA Scenarios**:
  ```
  Scenario: Knowledge note documents fix and QA evidence
    Tool: Bash
    Steps: GIT_MASTER=1 git diff -- .omo/knowledges/fresh-engine-rollout-collection.md
    Expected: diff contains root cause, validation rules, bounded QA command, and pre-fix dataset warning.
    Evidence: .omo/evidence/task-8-knowledge-diff.txt

  Scenario: No full-rerun instruction added before bounded QA evidence
    Tool: Bash
    Steps: rg -n 'full dataset|novphy_rollouts_dataset' .omo/knowledges/fresh-engine-rollout-collection.md
    Expected: any full-dataset command is explicitly guarded by bounded-QA-passed language.
    Evidence: .omo/evidence/task-8-no-premature-rerun.txt
  ```

  **Commit**: YES | Message: `docs(rollouts): document invalid shot recovery` | Files: [`.omo/knowledges/fresh-engine-rollout-collection.md`]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high (+ debugging; Playwright only if WebUI comparison is explicitly exercised)
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Prefer 3 small commits:
  1. `test(rollouts): classify invalid rollout artifacts`
  2. `fix(rollouts): guard and reject invalid shot attempts`
  3. `docs(rollouts): document invalid shot recovery`
- Do not commit runtime data under `data/` unless the user explicitly asks.
- Do not commit `.omo/evidence/` unless project convention requires it; keep it as local evidence.

## Success Criteria
- Known `shot_003` menu artifact is rejected by machine-readable validation.
- Known `shot_002` low-motion artifact is not silently accepted.
- New bounded real Xvnc/Java run produces accepted shots that are gameplay-valid and no accepted menu/static frames.
- Metadata/action logs explain every invalid attempt and retry.
- WebUI route semantics and protocol `51`/`52` defaults remain unchanged.
- Full train/dev dataset rerun is deferred until this plan’s bounded QA and final verification pass.
