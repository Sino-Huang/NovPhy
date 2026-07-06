# Learnings

## 2026-07-05 Start Work Context
- Active plan: `.omo/plans/rollout-menu-static-shot-bug.md`.
- Known bad artifact root: `data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1`.
- Known metadata facts from planning:
  - `shot_001`: valid baseline, `max_frame_delta=962`, score `1770`.
  - `shot_002`: suspicious/low motion, `max_frame_delta=22`, score `1210`.
  - `shot_003`: invalid menu/static, `max_frame_delta=0`, `max_pre_shot_delta=0`, visual SELECT LEVEL menu.
- Guardrails: do not trust `shoot_response=1`; do not default to protocol `51`/`52`; do not change WebUI route semantics; do not run full dataset before bounded QA.

## 2026-07-05 Search Agent Findings
- Existing image primitives in `scripts/collect_rollouts.py`: `_image_is_uniform`, `_default_desktop_crop_for`, `_crop_desktop_image`, `_image_delta_stats`, `capture_desktop_rollout`, `prepare_rollout_video_frames`, `write_action_logs`, `anchor_actions_to_current_slingshot`, and `slingshot_reference_point_from_symbolic_state`.
- Existing manual-agent image/protocol primitives: `prepare_for_play`, `image_is_uniform`, `render_symbolic_frame`, `save_frame`, `_frame_stats`, `capture_pixel_rollout`.
- Validator should reuse `_image_delta_stats()` and desktop metadata fields (`max_frame_delta`, `max_pre_shot_delta`, `capture_stop_reason`, `pre_shot_path`, `desktop_crop`, `pre_shot_sample`) instead of adding heavy dependencies.
- Pillow guidance: use `ImageChops.difference()`, `Image.getbbox()`, `ImageStat.Stat`, and `Image.histogram()`; convert frames to `L` or `RGB`; avoid high-bit-depth histogram pitfalls; no new dependency is needed.
- Metadata graph: `collect_rollouts()` writes per-shot metadata and manifest for same-episode mode; `collect_fresh_engine_rollouts()` wraps `collect_rollouts(..., write_manifest=False)` and adds `fresh_engine_attempt` / `slingshot_reference` before final manifest/action logs.
- `write_action_logs()` currently includes `shot_name`, `action`, `shot`, `shoot_response`, `frame_count`, `metadata_path`, and optional `pre_shot_path`, `video_path`, `slingshot_reference`; it does not currently copy `fresh_engine_attempt`.
- Safest Task 2 hooks: merge new evidence into metadata in `collect_rollouts()` after `capture_rollout(...)`; add fresh-engine attempt evidence in `collect_fresh_engine_rollouts()` after the partial rollout returns; extend `write_action_logs()` only if evidence must be replayable.

## 2026-07-05 Rollout Artifact Validator
- Added `validate_rollout_artifact(shot_dir)` in `scripts/collect_rollouts.py` as a small importable fail-closed validator for later rollout quarantine tasks.
- The validator reads `metadata.json`, resolves relative artifact paths against the repo root, verifies frame PNG presence, recomputes max frame/pre-shot deltas from actual frames with `_image_delta_stats()`, and only accepts `gameplay-valid` classifications.
- Current deterministic classes/signals are `gameplay-valid`, `menu-detected`, `no-frame-motion`, `low-motion-suspicious`, and `missing-artifact`; missing metadata/frame artifacts return `accepted=false` with `invalid_reason=missing_artifact`.
- Known artifact evidence is locked in `tests.test_collect_rollouts`: `shot_001` accepts with `max_frame_delta=962` and score `1770`, `shot_002` rejects as `low-motion-suspicious` with `max_frame_delta=22` and score `1210`, and `shot_003` rejects as `menu-detected` with `no-frame-motion` and `max_pre_shot_delta=0`.

## 2026-07-05 Task 2 Evidence Instrumentation
- `collect_rollouts()` now records additive `pre_shot_protocol_state`, `post_capture_protocol_state`, `post_recovery_protocol_state`, `recovery_action`, and `artifact_validation` fields in per-shot metadata without changing shot execution flow.
- `collect_fresh_engine_rollouts()` now persists `fresh_engine_attempt` back into each shot's `metadata.json` after the partial rollout returns.
- Added `validate_rollout_artifact()` so the existing dataset artifact tests can classify gameplay-valid, low-motion-suspicious, menu-detected, and missing-artifact shots from metadata plus frame paths.

## 2026-07-05 Task 2 Review Fix
- `pre_shot_protocol_state` is now captured before any pre-shot desktop grab or shot attempt, `post_shoot_protocol_state` is recorded distinctly after the shot, and `post_recovery_protocol_state` is no longer overwritten by capture-time state.
- Real capture outputs now recompute `artifact_validation` from the written `shot_*/` artifacts via `validate_rollout_artifact(shot_dir)` instead of relying on capture callbacks to inject it.
- The regression tests now check event ordering and computed validation so the timing gap and missing-validation gap are both covered.

## 2026-07-06 Desktop Ordering Assertion Update
- Updated the stale desktop baseline ordering assertion in `tests/test_collect_rollouts.py` to expect the new pre-shot protocol snapshot before the baseline grab while still requiring baseline capture to occur before `shoot`.

## 2026-07-06 Task 4 Pre-Shot Guard
- `collect_rollouts()` now runs a bounded pre-shot guard before any `shoot_once()` path, including deferred desktop shooting, and writes `pre_shot_guard` metadata with protocol state, visual evidence, attempts, recoveries, invalid reason, and recovery status.
- Desktop guard evidence classifies the actual pre-shot image with `_menu_like_frame_evidence()` and rejects menu-like or uniform surfaces even when protocol reports `PLAYING`; failed guards write `metadata.json` and `artifact_validation` before raising `recovery_failed`.
- Safe recovery uses the same protocol families trusted by `prepare_for_play()`: `ready_for_new_set()` for new-set states and `get_novelty_info()` plus `load_next_available_level()` for menu/end or visually invalid surfaces. The collector guard does not call `load_level(51)` or `restart_level(52)`.
- Regression tests cover `PLAYING` plus menu-like visual rejection with no shot/capture, and `NEWTRIAL` recovery via `ready_for_new_set()` before shooting.

## 2026-07-06 Task 5 Post-Shot Gate
- Added post-capture validation gating in `collect_rollouts()` and `collect_fresh_engine_rollouts()` so `validate_rollout_artifact(shot_dir)` decides whether a shot is counted as accepted.
- Manifest/action-log now separate attempts from accepted rollouts with `attempt_count`, `accepted_rollout_count`, and `accepted_*` lists while preserving per-attempt metadata and artifact evidence on disk.
- `validate_rollout_artifact()` now records machine-readable `retryable` and `retry_decision` fields; menu captures map to `quarantine`, low-motion captures map to `retry`.
- Regression tests now cover menu rejection, low-motion rejection, and accepted-vs-attempted accounting without touching WebUI or protocol 51/52 paths.

## 2026-07-06 Task 6 Bounded Invalid Attempt Retry
- Invalid post-shot attempts now copy their `shot_*` artifacts to deterministic `invalid_attempts/shot_NNN_attempt_MM/` directories before any fresh-engine retry can reuse the canonical `shot_NNN` path.
- Fresh-engine retries are bounded by `fresh_engine_attempts`: retryable low-motion attempts can advance to the next fresh engine, while exhausted retryable attempts fail closed with `attempt_status=invalid_exhausted` and no accepted rollout count.
- Attempt metadata/action logs/manifests now expose `attempt_status`, `accepted`, `invalid_reason`, `retry_attempt`, `recovery_action`, `quarantined_path`, `fresh_engine_attempt`, and `prior_invalid_attempts` so invalid evidence is not omitted when a later retry succeeds.

## 2026-07-06 Task 7 Bounded Runtime QA
- Bounded Xvnc desktop fresh-engine QA used `/tmp/opencode/rollout_bugfix_check` and configured exactly `9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml` via `scripts/prepare_rollout_dataset.py write-config`.
- The collector command exited nonzero (`collector_status=1`) before producing `manifest.json`: after `shot_002` was correctly rejected as `low_motion_suspicious`, fresh-engine attempts 2 and 3 remained in `NEWTRIAL` until `prepare_for_play` raised `TimeoutError: Science Birds did not reach PLAYING before timeout`.
- Partial artifact validation still proved the post-shot gate on completed attempts: `shot_001` is accepted gameplay-valid with `max_frame_delta=1060`, `max_pre_shot_delta=1544`, and `artifact_validation.accepted=true`; `shot_002` is rejected/quarantined under `invalid_attempts/shot_002_attempt_01/` with `max_frame_delta=22`, `invalid_reason=low_motion_suspicious`, `retry_decision=retry`, and `accepted=false`.
- Evidence files: `.omo/evidence/task-7-runtime.log`, `.omo/evidence/task-7-validator-summary.json`, `.omo/evidence/task-7-process-cleanup.txt`, `.omo/evidence/task-7-command-transcript.txt`, and `.omo/evidence/task-7-output-dir.txt`.
- Cleanup evidence shows no matching `rollout_bugfix_check`, `Xvnc :153`, `game_playing_interface`, or `collect_rollouts.py` process remained after the QA run.

## 2026-07-06 Task 7 Lifecycle Recovery Retry
- Added bounded `prepare_for_play()` recovery for persistent `NEWTRIAL`: reissue `ready_for_new_set()` after a bounded wait, then escalate with `get_novelty_info()` plus `load_next_available_level()` instead of protocol `load_level(51)` or `restart_level(52)`.
- Added regression coverage for repeated `NEWTRIAL`, escalation after bounded retries, and `EVALUATION_TERMINATED` recovery to next available level.
- Latest bounded Xvnc fresh-engine run for `9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml` exited `collector_status=0` and produced `/tmp/opencode/rollout_bugfix_check_retry_20260706_024619/out/manifest.json`.
- Runtime evidence shows recovery through `NEWTRIAL`, `NEWTRAININGSET`, `MAIN_MENU`, and `LOST` to `PLAYING`, including persistent `NEWTRIAL` escalation and fresh-engine retry after a prepare timeout.
- The run attempted 6 fresh engines for requested `--count 3`; only `shot_001` was accepted gameplay-valid (`max_frame_delta=1049`, `max_pre_shot_delta=1541`, score `1770`), while 5 invalid attempts were quarantined/rejected as low-motion or menu-detected.
- Evidence files refreshed: `.omo/evidence/task-7-runtime.log`, `.omo/evidence/task-7-validator-summary.json`, `.omo/evidence/task-7-process-cleanup.txt`, and `.omo/evidence/task-7-output-dir.txt`.

## 2026-07-06 Task 8 Operator Guidance
- Knowledge guidance should frame Task 7 as bounded-QA evidence only: `collector_status=0` with one accepted gameplay-valid shot and five quarantined invalid attempts proves the guard/gate behavior on one configured level, not full dataset validity.
- Operator docs must keep full dataset language gated behind final verification and an explicit user choice, because pre-fix `data/novphy_rollouts_dataset` contains known accepted static/menu artifacts.
- Safe recovery guidance should name bounded `prepare_for_play()` reissue/escalation and continue to forbid protocol `load_level(51)` / `restart_level(52)` as default recovery.

## 2026-07-06 Final Wave F2 Quarantine Metadata
- Quarantine copies must rewrite copied `metadata.json` before revalidating; otherwise `validate_rollout_artifact(invalid_attempts/shot_NNN_attempt_MM)` can read the later overwritten canonical `shot_NNN/frames` and incorrectly accept or fail for the wrong artifact root.
- The collector now keeps canonical invalid metadata canonical on disk, but returns manifest/action-log/prior-invalid references using the rewritten quarantine metadata and independently recomputed `artifact_validation` paths.

## 2026-07-06 Final Wave F1 Retry Exhaustion
- Fresh-engine retry exhaustion must write action logs and `manifest.json` with `collection_status=retry_exhausted` before raising, so CLI nonzero/library exceptions still leave operator evidence on disk.

## 2026-07-06 Final Wave F2 Validator Trust Semantics
- `validate_rollout_artifact()` must classify acceptance from recomputed frame/pre-shot image evidence only; metadata `max_frame_delta` and `max_pre_shot_delta` are reported counters, not trust anchors, because stale or inflated values can mask one-frame static/menu artifacts.

## 2026-07-06 F2 Review Fresh-Engine Guard Exhaustion Gap
- Fresh-engine retries that fail inside the pre-shot guard can create quarantined `invalid_attempts/shot_NNN_attempt_MM/` directories, then re-raise `PreShotGuardError` after the final attempt without writing `manifest.json` or `action_log.json`; this leaves operator evidence on disk but hidden from the normal retry-exhaustion ledger.
