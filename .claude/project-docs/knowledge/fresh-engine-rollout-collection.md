# Fresh Engine Rollout Collection

## Safe Resume Contract

Date: 2026-07-21

Dataset resume treats an episode as complete only when its manifest proves the exact planned desktop contract: `capture_source` is `capture_desktop_rollout`, numeric non-boolean `target_fps` and `duration_seconds` equal the plan, `ui_level` is integer `1`, and all existing replay, error, attempt, aggregate-count, status, and artifact-validation checks pass. Resume uses exclusive episode-directory creation to preserve unsafe existing outputs; non-resume preserves the direct collection command shape.

The full-dataset launcher canonicalizes `OUT_ROOT` with `realpath -m` and uses that one identity for output reporting, planner output, active collector detection, and the adjacent collection lock. `RESUME=1` validates the explicit existing root before creating anything. Port inspection accepts only an awk-produced `occupied` or `available` result and fails closed for awk errors or unexpected output.

## Directional Action Sampling

Date: 2026-07-22

`scripts/manual_agent.py::generate_diverse_drag_release_actions()` samples default angles from 5 to 80 degrees and derives `dx` as `-round(cos(angle) * strength)`. Default generated actions therefore always have a negative horizontal release offset: the sling is pulled left, and the bird launches right. An audit of 34,668 accepted actions in `data/novphy_rollouts_dataset_20260708_171531` found 34,668 negative `drag_release.x` values, zero positive values, and zero zero-valued offsets (range: -105 through -14). Right-to-left initial launches require custom/replayed actions with positive horizontal release values or a generator configured with angles beyond 90 degrees.

## Level-5 Mixed Launch Coverage

Date: 2026-07-22

New plans route only `novelty_level_5` entries through `scripts/collect_rollouts.py --bidirectional-launches`. Generated actions remain standard rightward launches at zero-based even positions and horizontally mirrored leftward launches at odd positions, so an even action count is a 50/50 split and an odd count has one additional standard action. Other novelty levels, direct collector defaults, and action-log replay remain unchanged. This preserves mixed coverage for the relation novelty without changing existing artifacts or resume eligibility.

Date: 2026-06-22

The original `scripts/collect_rollouts.py` was only partially suitable for diverse rollouts: it generated diverse drag/release actions, but fired them sequentially into one evolving engine session and used the TCP screenshot capture path. On this migrated server, TCP screenshots can be uniform gray, and sequential shots do not rerun the same episode from the same initial state.

The collector now supports a fresh-engine replay mode plus desktop capture:

```sh
source ~/cd_novphy
Xvnc :142 -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >/tmp/novphy_rollout_xvnc.log 2>&1 &
DISPLAY=:142 LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}" \
  python scripts/collect_rollouts.py \
    --output-dir data/rollout_feature_check_20260622_1615_fixed \
    --count 2 \
    --fps 1 \
    --duration 1 \
    --connect-timeout 30 \
    --prepare-timeout 60 \
    --read-timeout 120 \
    --speed 1 \
    --capture-source desktop \
    --fresh-engine-per-rollout \
    --ui-level 1 \
    --ui-settle-seconds 5
```

Validated artifact:

- Stronger multi-frame manifest: `data/rollout_feature_check_20260622_1625_multiframe/manifest.json`
- Rollout 1 frames: `data/rollout_feature_check_20260622_1625_multiframe/shot_001/frames/`
- Rollout 2 frames: `data/rollout_feature_check_20260622_1625_multiframe/shot_002/frames/`
- Initial single-frame manifest: `data/rollout_feature_check_20260622_1615_fixed/manifest.json`

Validation result:

- The multi-frame `manifest.json` reports `replay_mode: fresh-engine-per-rollout`, `rollout_count: 2`, `target_fps: 2.0`, and `duration_seconds: 2.0`.
- The two actions are different drag/release actions.
- Both rollouts saved 4 frames each.
- Later frames `shot_001/frames/frame_000003.png` and `shot_002/frames/frame_000003.png` were visually confirmed as in-level Science Birds observations, not title/menu/blank frames.
- Per-shot metadata records `state: PLAYING`, `score: 0`, `uniform: false`, and thousands of unique colors for each frame.

Important behavior:

- Default `--capture-source protocol` keeps the previous TCP screenshot collector behavior.
- Use `--capture-source desktop --fresh-engine-per-rollout --ui-level <N>` for this server when repeated same-level visual rollouts are needed.
- `--ui-settle-seconds` waits both before and after the xdotool level-selection sequence. This is necessary because the backend can report `PLAYING` before the fresh Unity window is ready for reliable UI clicks.

## Dynamic rollout proof update

The initial multi-frame artifact proved only non-blank in-level frames. It was not enough to prove post-action rollout dynamics because the generated actions still used the default `drag_start` and there was no pre-shot baseline.

The collector now resolves the real slingshot reference from request-62 symbolic state in fresh-engine mode and anchors slingshot-relative actions to that point before shooting. Desktop capture also records a `pre_shot.png` baseline before `bridge.shoot(...)`, a `pre_shot_sample` with state and score, per-frame `pre_shot_delta`, `max_pre_shot_delta`, `max_pre_shot_delta_bbox`, consecutive `frame_delta`, and post-shot state/score samples.

Superseded validation artifacts:

- `data/rollout_feature_check_20260622_preshot_sample/manifest.json` proved pre-shot baselines and dynamics, but used canvas-Y as the anchored action `drag_start`. The code now uses bottom-left game coordinates for `drag_start`, matching the action contract.

Final validated corrected-anchor artifact:

- Manifest: `data/rollout_feature_check_20260622_corrected_anchor4/manifest.json`
- Mode: `fresh-engine-per-rollout`, `capture_source: capture_desktop_rollout`, `ui_level: 1`
- Rollouts: 4 distinct drag/release actions, 49-56 desktop frames each, `target_fps: 8.0`, `duration_seconds: 7.0`
- All actions anchored to `drag_start [97, 227]` with `slingshot_reference {gameX: 97, gameY: 227, canvasX: 97, canvasY: 252}`.
- Each rollout records `pre_shot_sample {state: PLAYING, score: 0}`, `pre_shot.png`, nonzero `max_pre_shot_delta`, and nonzero `max_frame_delta`.
- Strongest semantic action-effect evidence: `shot_004` changes from pre-shot score `0` to post-shot score `230`.

Final verification commands:

```sh
source ~/cd_novphy && python -m unittest tests.test_collect_rollouts tests.test_manual_agent
source ~/cd_novphy && python -m py_compile scripts/collect_rollouts.py tests/test_collect_rollouts.py scripts/manual_agent.py tests/test_manual_agent.py
```

Both commands passed on 2026-06-22. `basedpyright-langserver` was not installed, so LSP diagnostics could not run.

Final collection command shape:

```sh
source ~/cd_novphy
OUT="data/rollout_feature_check_20260622_corrected_anchor4"
Xvnc :149 -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >/tmp/novphy_rollout_xvnc_149.log 2>&1 &
XVNC_PID=$!
sleep 2
DISPLAY=:149 LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}" \
  python scripts/collect_rollouts.py \
    --output-dir "$OUT" \
    --capture-source desktop \
    --fresh-engine-per-rollout \
    --ui-level 1 \
    --ui-settle-seconds 5 \
    --count 4 \
    --fps 8 \
    --duration 7 \
    --connect-timeout 40 \
    --prepare-timeout 80 \
    --read-timeout 420
STATUS=$?
kill "$XVNC_PID" 2>/dev/null || true
wait "$XVNC_PID" 2>/dev/null || true
exit "$STATUS"
```

## Cropped review frames and MP4 update

Date: 2026-06-24

Desktop capture originally saved the full 1024x768 Xvnc desktop. Runtime measurement of saved Science Birds frames showed the actual game observation occupied bbox `(32, 64, 672, 544)`, a 640x480 viewport, leaving black desktop padding on the right and bottom. `scripts/collect_rollouts.py::capture_desktop_rollout()` now crops desktop frames and `pre_shot.png` to that viewport when the grabbed desktop is large enough.

`scripts/collect_rollouts.py::collect_rollouts()` now attempts to write `rollout.mp4` in every `shot_*/` directory from the saved `frames/frame_%06d.png` sequence. It uses `ffmpeg` at the rollout FPS, records `video_path` in both per-shot metadata and the manifest when successful, and records `video_error` without discarding frames if the encoder is unavailable.

The collector CLI already defaults to high-FPS collection with `--fps 30`. Fresh-engine collection now also has `--engine-settle-seconds` with a default of `20.0` to avoid connecting to the Java interface before it is ready to answer `configure`.

Validation artifact:

- `data/review_rollout_mp4_function_check_20260624/manifest.json`
- `data/review_rollout_mp4_function_check_20260624/shot_001/rollout.mp4`

Validation result:

- Source frame: real saved desktop Science Birds frame from `data/review_rollouts_20260624_novelty0_type010101/shot_001/frames/frame_000000.png`.
- Cropped output frame size: 640x480.
- MP4 probe: `width=640`, `height=480`, `r_frame_rate=30/1`, `nb_frames=3`.
- Unit coverage: `tests.test_collect_rollouts` includes crop, MP4 success, MP4 encoder failure, parser default FPS, and engine startup-settle tests.

Runtime caveat observed on 2026-06-24:

- A manual Java/Xvnc startup followed by a bridge `configure()` succeeds after the Java interface reaches `Waiting for agent`.
- A fresh collector-run surface check still timed out at `configure()` before capture in this session, even with a 35-second settle. No stale process or port owner remained afterward. The crop/video implementation was therefore verified against real saved frames and real `ffmpeg`, not a new live Unity rollout.

## High-FPS frame-count debug update

Date: 2026-06-24

The initial MP4 validation artifact reported `r_frame_rate=30/1` with only 3 frames because that check intentionally called the production capture function with `max_frames=3`. That verified encoding/crop only; it was not a representative high-FPS rollout.

Two runtime issues affected live high-FPS rollout collection:

1. `capture_desktop_rollout()` queried game state and score inside the frame loop. Those TCP calls can be slow enough to throttle capture, so desktop rollouts now sample state/score once after the frame loop instead of once per frame.
2. Connecting to port `2004` starts the Unity game-window side of the Java proxy. Sending `configure()` immediately after socket connect can timeout while the proxy is still waiting for the Unity window on port `9001`. The fresh-engine collector now supports a post-connect `--agent-settle-seconds` delay.

`scripts/manual_agent.py::start_engine()` now writes owned Java output to `/tmp/novphy_game_engine_*.log` instead of discarding it, so this startup phase is inspectable.

High-FPS review artifacts collected after the state-sampling fix:

- `data/highfps_review_20260624_novelty0_type010101/shot_001/rollout.mp4`
- `data/highfps_review_20260624_novelty1_type010101/shot_001/rollout.mp4`
- `data/highfps_review_20260624_novelty3_type010303/shot_001/rollout.mp4`

Verification summary:

- `novelty_level_0/type010101`: 66 frames, 640x480, `30/1` FPS, MP4 duration `2.200000`.
- `novelty_level_1/type010101`: 67 frames, 640x480, `30/1` FPS, MP4 duration `2.233333`.
- `novelty_level_3/type010303`: 57 frames, 640x480, `30/1` FPS, MP4 duration `1.900000`.
- Each manifest records `desktop_crop [32, 64, 672, 544]` and one post-capture state sample.
- `sciencebirdsgames/Linux/config.xml` was restored to `novelty_level_0/type010102` after collection.
- `novelty_level_8/type010805` was attempted but repeatedly stayed in `LOST` and timed out before capture, so it was not included in the final high-FPS review set.

## Pre-Shot Guard Crop Alignment

Date: 2026-07-06

Reported artifact `data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00002_0_1_010101_0_1/shot_001/frames/frame_000000.png` showed the SELECT LEVEL menu even though the pre-shot guard metadata reported `classification: gameplay-candidate`. The root cause was a surface mismatch: `capture_desktop_rollout()` cropped Xvnc screenshots to the 640x480 game viewport with `DEFAULT_DESKTOP_GAME_CROP`, but `_run_pre_shot_guard()` classified the uncropped 1024x768 desktop. A full desktop with black padding around a menu-like game crop could therefore pass the guard and only be rejected after capture.

The guard now normalizes/crops pre-shot desktop grabs before classifying and saving guard evidence, so the pre-shot guard, `pre_shot.png`, rollout frames, and validator all refer to the same 640x480 game surface. Regression coverage: `tests.test_collect_rollouts.CollectRolloutsTest.test_pre_shot_guard_rejects_menu_inside_default_desktop_crop` fails without the crop alignment and passes with it.

Action-side finding from the same investigation: fresh-engine collection is still the correct mode for dataset generation because `scripts/prepare_rollout_dataset.py` emits `--capture-source desktop --fresh-engine-per-rollout --ui-level 1`. In that path, each attempt re-reads request-62 symbolic state and anchors slingshot-relative actions to the current slingshot/bird-on-sling before shooting. The benchmark wrappers follow the same contract: actions are `[dx, dy, tap_time]` relative to the current sling center, with the engine advancing to the next bird after each shot. Low/no-motion captures are not proof of a second-bird action bug by themselves; they are already classified as `low_motion_suspicious`, marked retryable, and retried/quarantined by the post-shot validator.

## Same-episode action overlay videos

Date: 2026-06-29

The user review found that post-release rollout videos were useful, but did not clearly show the starting point or action details. The Science Birds TCP protocol sends the final shot command rather than exposing a live mouse-drag animation, so review videos now add explicit pre-action context:

- `collect_rollouts()` writes `action_log.json` and `action_log.jsonl` in the episode output directory.
- Each `shot_*/metadata.json` records the action, shot, overlay text, `pre_action_frame_count`, `video_frame_count`, and `video_frames_dir`.
- MP4s are encoded from `shot_*/video_frames/frame_%06d.png`, not raw `shot_*/frames/`.
- `video_frames/` starts with a pre-shot lead-in copied from `pre_shot.png`, then appends the captured rollout frames.
- Every review frame has a top text banner with drag/release mode, drag coordinates, release coordinates, socket shot coordinates, tap time, and release time.
- Every review frame also draws a visual action guide: cyan start marker, yellow drag vector, red release marker.
- Raw captured frames remain unchanged under `shot_*/frames/`.

Same-episode varied-trial validation artifact:

- `data/same_episode_action_overlay_30fps_20260629_144507/manifest.json`
- `data/same_episode_action_overlay_30fps_20260629_144507/action_log.json`
- `data/same_episode_action_overlay_30fps_20260629_144507/action_log.jsonl`
- `data/same_episode_action_overlay_30fps_20260629_144507/shot_001/rollout.mp4`
- `data/same_episode_action_overlay_30fps_20260629_144507/shot_002/rollout.mp4`
- `data/same_episode_action_overlay_30fps_20260629_144507/shot_003/rollout.mp4`

Validation result:

- Replay mode: `same-episode-varied-trials`.
- Top-level manifest has `rollout_count: 3`.
- `action_log.json` has `trial_count: 3`; `action_log.jsonl` has three lines.
- `shot_001`: 46 raw frames, 22 pre-action frames, 68 video frames, 640x480, `30/1` FPS, MP4 duration `2.266667`.
- `shot_002`: 47 raw frames, 22 pre-action frames, 69 video frames, 640x480, `30/1` FPS, MP4 duration `2.300000`.
- `shot_003`: 46 raw frames, 22 pre-action frames, 68 video frames, 640x480, `30/1` FPS, MP4 duration `2.266667`.
- Visual inspection of `shot_001/video_frames/frame_000000.png` confirmed the top text banner and visible action guide overlay.

Command shape for same-episode trials:

```sh
source ~/cd_novphy
OUT="data/same_episode_action_overlay_$(date +%Y%m%d_%H%M%S)"
Xvnc :174 -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >/tmp/novphy_same_episode_xvnc_174.log 2>&1 &
XVNC_PID=$!
sleep 2
DISPLAY=:174 LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}" \
  python scripts/collect_rollouts.py \
    --output-dir "$OUT" \
    --capture-source desktop \
    --start-engine \
    --count 3 \
    --fps 30 \
    --duration 2 \
    --connect-timeout 60 \
    --prepare-timeout 90 \
    --read-timeout 420 \
    --speed 1
STATUS=$?
kill "$XVNC_PID" 2>/dev/null || true
wait "$XVNC_PID" 2>/dev/null || true
exit "$STATUS"
```

## WebUI-aligned action guide update

Date: 2026-06-29

The rollout overlay guide was updated to match the WebUI coordinate convention:

- Action math is canonical in WebUI-style bottom-left game coordinates.
- `action_to_shot()` now exposes both game endpoint fields (`gameX`, `gameY`) and bridge/socket fields (`x`, `y`).
- Only the final socket/drawing boundary flips y with `frame_height - 1 - gameY`.
- The review overlay guide start is drawn from `drag_start` converted to canvas coordinates.
- The review overlay guide end is drawn from the normalized game shot endpoint converted to canvas coordinates, so it equals the actual socket shot pixel.
- The yellow guide shows the slingshot pull/release direction, not the bird's travel direction. Pulling left/down launches the bird right/up.
- The green guide shows the intended bird launch direction. This was added after user feedback that the yellow pull vector made shots look like they were going right-to-left.
- Generated review actions now use human-like nonzero `holdTime`; default generated holds are `600` or `900` ms instead of instant release.
- Default generated launch angles now favor right/upward shots toward right-side buildings and objects.
- Missing or zero hold time is normalized to a default `releaseTime` of `600` ms in `action_to_shot()`.

Validation artifact:

- `data/webui_aligned_action_overlay_30fps_20260629_231631/manifest.json`
- `data/webui_aligned_action_overlay_30fps_20260629_231631/action_log.json`
- `data/webui_aligned_action_overlay_30fps_20260629_231631/shot_001/rollout.mp4`
- `data/webui_aligned_action_overlay_30fps_20260629_231631/shot_002/rollout.mp4`
- `data/webui_aligned_action_overlay_30fps_20260629_231631/shot_003/rollout.mp4`

Validation result:

- `rollout_count: 3`, `replay_mode: same-episode-varied-trials`, `trial_count: 3`.
- All three shots had `holdTime: 600` and `releaseTime: 600`.
- For each rollout, `guide_end_canvas == (shot.x, shot.y)`, proving the visual guide endpoint matches the actual socket shot pixel.
- MP4s are 640x480 at `30/1` FPS with 68, 68, and 63 frames.
- Visual inspection of `shot_001/video_frames/frame_000000.png` confirmed top text plus cyan/yellow/red action guide overlay.

Right-launch overlay validation artifact:

- `data/right_launch_action_overlay_contrast_20260630_010205/manifest.json`
- `data/right_launch_action_overlay_contrast_20260630_010205/shot_001/rollout.mp4`

Validation result:

- The generated pull/release vector was `[-45, 4]`, meaning the bird is pulled left/down from the sling.
- The intended launch vector was `(345, 224)` from start `(300, 220)`, so launch is right/up.
- `holdTime` and `releaseTime` were both `600` ms.
- Visual inspection confirmed the top text banner, clear yellow pull/release guide, and clear green right/up launch guide.

## Corrected same-episode sling anchoring

Date: 2026-06-30

The artifact `data/right_launch_action_overlay_contrast_20260630_010205/shot_001/rollout.mp4` still used the synthetic default `drag_start [300, 220]` in same-episode mode. Frame inspection showed the actual visible sling was near `(97, 252)` in cropped canvas pixels, so the marker and shot were far to the right of the sling.

`collect_rollouts()` now resolves the current request-62 symbolic slingshot reference when available and anchors slingshot-relative actions before shooting, logging, or drawing overlays. Fresh-engine mode already did this; the fix extends it to same-episode `--start-engine` collection.

Corrected validation artifact:

- `data/corrected_sling_right_launch_30fps_20260630_135930/manifest.json`
- `data/corrected_sling_right_launch_30fps_20260630_135930/shot_001/rollout.mp4`

Validation result:

- Action was anchored to `drag_start [97, 227]` with `slingshot_reference {gameX: 97, gameY: 227, canvasX: 97, canvasY: 252}`.
- Overlay start canvas coordinate was `(97, 252)`, matching the visible sling/fork region.
- Pull/release vector was `[-45, 4]`; intended launch point was `(142, 231)`, so launch is right/up from the actual sling.
- Actual socket shot was `(52, 256)`, generated from game endpoint `(52, 223)`.
- Visual inspection confirmed the cyan marker is aligned to the visible slingshot and the green launch guide points right/up.
- MP4 probe: 640x480, `30/1` FPS, 58 frames, duration `1.933333`.

## Stronger pulls and longer holds

Date: 2026-06-30

Default generated review actions were increased from short pulls/holds to more human-like aiming pulls:

- Default pull strengths: `80, 105, 130, 155` pixels.
- Default hold times: `1000, 1400` ms.
- Generated review actions still favor right/up launch angles.

Validation artifact:

- `data/strong_pull_long_hold_30fps_20260630_141927/manifest.json`
- `data/strong_pull_long_hold_30fps_20260630_141927/shot_001/rollout.mp4`
- `data/strong_pull_long_hold_30fps_20260630_141927/shot_002/rollout.mp4`

Validation result:

- `shot_001`: anchored start `[97, 227]`, pull `[-80, 7]`, pull distance `80.31`, launch `(177, 234)`, `holdTime/releaseTime = 1000`, MP4 640x480 `30/1` FPS, 68 frames.
- `shot_002`: anchored start `[97, 227]`, pull `[-75, 27]`, pull distance `79.71`, launch `(172, 254)`, `holdTime/releaseTime = 1000`, MP4 640x480 `30/1` FPS, 71 frames.
- Visual inspection confirmed the cyan start marker is on the visible sling and the green launch guide points right/up.

## Phased pre-drag review video lead-in

Date: 2026-06-30

The review MP4 lead-in used to draw the full pull/release/launch guide on `frame_000000.png`. That made videos look like the bird was already dragged when playback started. `prepare_rollout_video_frames()` now splits pre-shot lead-in frames into explicit phases:

- `pre_drag`: neutral frames copied from `pre_shot.png` with only `phase=pre_drag pre_shot_baseline` in the banner and no pull/release/launch guide.
- `aim_hold`: staged guide frames drawn on `pre_shot.png`, ramping from a partial pull guide to the full pull/release/launch guide.
- `rollout`: captured post-shot frames with the full overlay.

Each `shot_*/metadata.json` now records `pre_drag_frame_count`, `aim_hold_frame_count`, and `video_phase_counts` in addition to the existing `pre_action_frame_count` and `video_frame_count`.

Validation result:

- Unit regression: `tests.test_collect_rollouts.CollectRolloutsTest.test_prepare_rollout_video_frames_starts_with_neutral_pre_drag_frame` asserts the first action-area frame matches `pre_shot.png` while the first aim frame differs.
- Manual QA regenerated frames from `data/strong_pull_long_hold_30fps_20260630_141927/shot_001` into `/tmp/novphy_phase_qa_shot_001`.
- Regenerated phase counts were `pre_drag: 11`, `aim_hold: 11`, `rollout: 46`, total `68` frames at 30 FPS.
- Visual inspection confirmed `frame_000000.png` is a clean pre-shot baseline frame with no guide, `frame_000011.png` begins the pull guide, and `frame_000021.png` shows the full guide before rollout frames.

## Deferred desktop shot capture

Date: 2026-07-03

User inspection of `data/strong_pull_long_hold_30fps_20260630_141927/shot_001/frames/frame_000000.png` and `shot_002/frames/frame_000000.png` showed that raw captured frames could start too late relative to the shot. The collector previously called `bridge.shoot(...)` before `capture_desktop_rollout(...)`, so desktop raw frame capture could begin after fast-shot execution had already removed or launched the bird.

Desktop capture now defers the shot into `capture_desktop_rollout()`:

- `collect_rollouts(..., shoot_before_capture=False)` passes a one-shot callback to the capture function instead of firing before capture starts.
- `capture_desktop_rollout(..., shoot=...)` grabs and saves `frames/frame_000000.png` first, then invokes the shot callback, records `shoot_response`, and records `shoot_frame_index`.
- CLI desktop capture passes `shoot_before_capture=False`; pixel/protocol capture keeps the old shoot-before-capture behavior.

Validation result:

- Unit regression: `test_capture_desktop_rollout_starts_capture_before_shoot_callback` asserts event order `grab-0`, `shoot`, `grab-1` and metadata `shoot_frame_index: 0`.
- Unit regression: `test_collect_rollouts_can_defer_shoot_until_desktop_capture_starts` asserts `collect_rollouts()` can route the shot through capture when `shoot_before_capture=False`.
- Real Xvnc artifact: `/tmp/novphy_deferred_capture_FcNZsj/shot_001/metadata.json` recorded `shoot_response: 1`, `shoot_frame_index: 0`, `frame_count: 13`, and video phase counts `pre_drag: 11`, `aim_hold: 11`, `rollout: 13`.
- Visual inspection confirmed raw `frames/frame_000000.png` is a pre-shot surface before the shot callback; raw `frames/frame_000001.png` catches the immediate post-shot launch transition.
- Limitation: the Science Birds TCP protocol sends a final shot command and does not expose a live drag-held mouse state. The pre-shot baseline can show birds queued beside an empty sling, and the first post-shot desktop frame may already be immediate launch rather than a visible held drag pose.

## Action log replay validation

Date: 2026-07-03

The collector now supports replaying an episode from a previous `action_log.json` with `--actions-from-log`. This loads only the saved `trials[*].action` objects and disables slingshot re-anchoring, so the replay uses the exact logged actions rather than regenerated actions or memory from a previous run.

Validation artifacts:

- Original episode: `data/action_log_replay_original_20260703_180602`
- Replay episode: `data/action_log_replay_replayed_20260703_180704`
- Replay command used `--actions-from-log data/action_log_replay_original_20260703_180602/action_log.json`.

Validation result:

- Original `action_log.json`: `trial_count: 2`, `action_log.jsonl` has 2 lines, both `shoot_response` values were `1`.
- Replay `action_log.json`: `trial_count: 2`, `action_log.jsonl` has 2 lines, both `shoot_response` values were `1`.
- The replayed `trials[*].action` list exactly matched the original `trials[*].action` list.
- `shot_001` MP4 probe matched across original and replay: 640x480, `30/1` FPS, 34 frames, duration `1.133333`.
- `shot_002` MP4 probe matched across original and replay: 640x480, `30/1` FPS, 27 frames, duration `0.900000`.

## Shot-completion desktop capture

Date: 2026-07-03

The fixed-duration desktop capture could still stop while the bird was airborne. In `data/action_log_replay_original_20260703_180602/shot_001/frames/frame_000011.png`, the bird was visibly mid-flight, but the original short capture ended at `frame_000011.png`. The cause was that `duration_seconds` was treated as a total wall-clock capture budget, including the blocking time spent inside `bridge.shoot(...)`, and there was no visual-settle requirement before advancing to the next shot.

Desktop capture with a deferred shot callback now treats `duration_seconds` as the minimum post-shot capture time, then keeps recording until frame deltas remain below the visual settle threshold for the quiet window, or until a max-duration cap is reached. The per-shot metadata records `capture_stop_reason`, `post_shot_capture_seconds`, `min_post_shot_duration_seconds`, `max_duration_seconds`, `settle_seconds`, and `settle_pixel_threshold`.

Validation artifacts:

- Original episode with completion capture: `data/action_log_completion_original_20260703_200446`
- Replay episode from that original log: `data/action_log_completion_replayed_20260703_200622`
- Replay command used `--actions-from-log data/action_log_completion_original_20260703_200446/action_log.json`.

Validation result:

- `frame_000011.png` in the original `shot_001` still shows the bird airborne, proving the old cutoff point was incomplete.
- Original and replay `trials[*].action` lists matched exactly; original and replay `trials[*].shot` lists also matched exactly.
- Original `shot_001`: 89 raw frames, `capture_stop_reason: settled`, `post_shot_capture_seconds: 4.07189`, MP4 640x480, `30/1` FPS, 111 frames, duration `3.700000`.
- Replay `shot_001`: 89 raw frames, `capture_stop_reason: settled`, `post_shot_capture_seconds: 4.070352`, MP4 640x480, `30/1` FPS, 111 frames, duration `3.700000`.
- Original `shot_002`: 82 raw frames, `capture_stop_reason: settled`, `post_shot_capture_seconds: 3.74332`, MP4 640x480, `30/1` FPS, 104 frames, duration `3.466667`.
- Replay `shot_002`: 86 raw frames, `capture_stop_reason: settled`, `post_shot_capture_seconds: 3.74305`, MP4 640x480, `30/1` FPS, 108 frames, duration `3.600000`.
- Visual inspection confirmed the final replay frames no longer show an airborne bird or active collapse; static trajectory lines may remain because Science Birds keeps trajectory marks visible after the shot.

## Dataset partition helper

Date: 2026-07-04

`scripts/prepare_rollout_dataset.py` prepares deterministic train/dev/test partitions over installed NovPhy XML levels and writes a dry-run train/dev collection script. The default `plan` command only scans levels and writes files; it does not start Unity, run rollouts, or mutate `sciencebirdsgames/Linux/config.xml`.

Use it from the repo root after initializing the project environment:

```sh
source ~/cd_novphy && \
python scripts/prepare_rollout_dataset.py plan \
  --output-dir data/rollout_dataset_plan \
  --command-output-root data/rollout_dataset \
  --count 2 \
  --fps 30 \
  --duration 5 \
  --display :149
```

This writes:

- `data/rollout_dataset_plan/partitions.json`: schema `novphy-rollout-dataset-partitions-v1`, split counts, and level entries grouped by `train`, `dev`, and `test`.
- `data/rollout_dataset_plan/collect_train_dev.sh`: executable command script for train/dev levels only.

The generated script intentionally leaves Xvnc startup as an explicit operator step. Start Xvnc first, then run selected lines or the whole script:

```sh
source ~/cd_novphy
Xvnc :149 -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >/tmp/novphy_rollout_xvnc_149.log 2>&1 &
data/rollout_dataset_plan/collect_train_dev.sh
```

For each train/dev level, the script first calls:

```sh
python scripts/prepare_rollout_dataset.py write-config \
  --manifest data/rollout_dataset_plan/partitions.json \
  --split train \
  --level-path 9001_Data/StreamingAssets/Levels/...
```

That is the explicit point where `sciencebirdsgames/Linux/config.xml` is mutated to contain exactly one level. The following `collect_rollouts.py` invocation then runs desktop capture with `--fresh-engine-per-rollout`, `--ui-level 1`, and the configured rollout count/FPS/duration.

Partitioning details:

- Discovery scans `sciencebirdsgames/Linux/9001_Data/StreamingAssets/Levels/novelty_level_*/type010*/Levels/*.xml`.
- Splits are computed independently per `novelty_level/type010...` bucket, using a stable SHA-256 key over `seed + relative_path`.
- Defaults are approximately 80/10/10 train/dev/test per bucket, with small buckets preserving train/dev/test when possible.
- Commands are generated only for train/dev by default; test levels remain held out.

Verification on 2026-07-04:

```sh
source ~/cd_novphy && python -m unittest tests.test_prepare_rollout_dataset
```

The helper test suite covers stable non-overlapping splits, train/dev-only command generation, and clear missing/empty level-root failures.

## Fresh-engine readiness retries

Date: 2026-07-05

During a full train/dev dataset run, one fresh-engine rollout failed before capture after repeatedly printing `State NEWTRIAL: ready_for_new_set` and timing out in `prepare_for_play()`. Earlier rollouts for the same level had reached `PLAYING`, so this was treated as a transient fresh-engine readiness failure rather than a partition/config problem.

`scripts/collect_rollouts.py` now supports bounded per-action retries in fresh-engine mode:

- CLI flag: `--fresh-engine-attempts`, default `3`.
- Each action starts a fresh engine per attempt.
- If startup/connect/configure/prepare/select/capture fails before success, the failed bridge/engine is cleaned up and the same action is retried in a new engine until attempts are exhausted.
- Existing generated `collect_train_dev.sh` scripts inherit the default because they call `collect_rollouts.py --fresh-engine-per-rollout` without overriding `--fresh-engine-attempts`.
- Successful fresh-engine manifests record `fresh_engine_attempts`; per-rollout entries record `fresh_engine_attempt` when retries are enabled.

Validation:

```sh
source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_collect_fresh_engine_rollouts_retries_prepare_timeout_with_new_engine
source ~/cd_novphy && python -m unittest tests.test_manual_agent tests.test_collect_rollouts tests.test_prepare_rollout_dataset
```

The regression test fails first without the retry parameter, then passes by simulating a first-attempt `TimeoutError("Science Birds did not reach PLAYING before timeout")` followed by a successful second fresh engine. A small live Xvnc/Java smoke with `--fresh-engine-attempts 2` also reached `PLAYING` and collected one desktop rollout after passing through `NEWTRIAL`/`NEWTRAININGSET`/`MAIN_MENU` readiness states.

## Full dataset failure ledger

Date: 2026-07-05

The generated full train/dev collection script no longer aborts the entire dataset at the first failed level. `scripts/prepare_rollout_dataset.py::generate_collection_commands()` still emits `set -euo pipefail`, but each per-level `write-config` plus `collect_rollouts.py --fresh-engine-per-rollout` pair is wrapped in one `if ...; then ... else ... fi` block.

Failure behavior:

- `PLAN_DIR/failed_levels.tsv` is initialized with columns `split`, `level_path`, `output_dir`, and `status`.
- If either config writing or rollout collection fails for a level, the generated script appends one TSV row, increments `failure_count`, prints a warning, and continues with the next train/dev level.
- After all requested levels have been attempted, the generated script exits `1` if any failures occurred, so automation still detects an incomplete dataset.
- If no level fails, the generated script exits `0` and prints the ledger path.
- `scripts/collect_full_rollout_training_dataset.sh` now documents this behavior in `--help` and at launch.

Verification on 2026-07-05:

```sh
source ~/cd_novphy && python -m unittest tests.test_prepare_rollout_dataset tests.test_collect_rollouts
source ~/cd_novphy && python -m py_compile scripts/prepare_rollout_dataset.py tests/test_prepare_rollout_dataset.py
bash -n scripts/collect_full_rollout_training_dataset.sh
source ~/cd_novphy && python scripts/prepare_rollout_dataset.py plan --output-dir /tmp/opencode/novphy_ledger_plan --command-output-root /tmp/opencode/novphy_ledger_out --count 1 --fps 1 --duration 1 --display :149
bash -n /tmp/opencode/novphy_ledger_plan/collect_train_dev.sh
```

The generated script was inspected and begins with `failure_ledger=/tmp/opencode/novphy_ledger_plan/failed_levels.tsv`, initializes the TSV header, wraps per-level commands in guarded blocks, and ends with a nonzero summary when `failure_count > 0`. `basedpyright-langserver` was still not installed, so Python LSP diagnostics could not run.

Post-review security hardening:

- A review pass found that generated status messages originally embedded `split` and `level_path` inside shell single-quoted `echo` strings.
- `generate_collection_commands()` now emits `printf` status messages and passes all dynamic values through `shlex.quote` as separate shell words.
- Regression coverage: `test_generate_collection_commands_quotes_status_messages_for_single_quote_level_path` builds a manifest containing a single quote in `relative_path`, writes the generated script, and asserts `bash -n` succeeds.
- Verification after the fix: `source ~/cd_novphy && python -m unittest tests.test_prepare_rollout_dataset tests.test_collect_rollouts` passed 49 tests, `py_compile` passed, launcher `bash -n` passed, and a regenerated full command script under `/tmp/opencode/novphy_ledger_plan_after_quote_fix` passed `bash -n`.

## Fresh-engine readiness debounce

Date: 2026-07-05

A bounded 10-minute observation run on a fresh Xvnc display did not reproduce the earlier `xdotool` / Xvnc client-limit failure (`Maximum number of clients reached`, `Failed creating new xdo instance`). It did reproduce a separate readiness inefficiency: some fresh engines remain in `NEWTRIAL` for the whole prepare timeout while `prepare_for_play()` sends `ready_for_new_set()` every poll.

Observation artifact:

- Log: `/tmp/opencode/novphy_observation_20260705_183403.log`
- Plan dir: `data/novphy_rollouts_observation_plan_20260705_183403`
- Output dir: `data/novphy_rollouts_observation_20260705_183403`
- Display: `:151`

`scripts/manual_agent.py::prepare_for_play()` now debounces transition commands for unchanged states:

- First `NEWTRIAL` / `NEWTRAININGSET` / `RESUMETRAINING` / `NEWTESTSET` poll sends `ready_for_new_set()`.
- Repeated polls of the same state print `State <STATE>: waiting` and do not resend the transition command.
- First `MAIN_MENU` / `EPISODE_MENU` / `LEVEL_SELECTION` / `WON` / `LOST` poll still sends `load_next_available_level()` with the novelty-info preflight.
- Repeated polls of the same menu/end state wait instead of resending the transition command.
- Passive states such as `LOADING` clear the debounce memory, so a later transition state can still be acted on.

Regression coverage:

- `tests.test_manual_agent.ManualAgentTest.test_prepare_for_play_does_not_spam_same_new_trial_transition` failed before the fix with `ready_calls == 3`, then passed after the debounce with one `ready_for_new_set()` and two passive waits.

Verification on 2026-07-05:

```sh
source ~/cd_novphy && python -m unittest tests.test_manual_agent.ManualAgentTest.test_prepare_for_play_does_not_spam_same_new_trial_transition
source ~/cd_novphy && python -m unittest tests.test_manual_agent tests.test_collect_rollouts
source ~/cd_novphy && python -m py_compile scripts/manual_agent.py scripts/collect_rollouts.py tests/test_manual_agent.py tests/test_collect_rollouts.py
GIT_MASTER=1 git diff --check -- scripts/manual_agent.py tests/test_manual_agent.py
```

The related test suites passed 52 tests. `basedpyright-langserver` was still not installed, so Python LSP diagnostics could not run.

## WebUI mechanism boundary for rollout collector fixes

Date: 2026-07-05

The WebUI is useful comparison evidence for the rollout collector, but it is not a rewrite target for the current rollout menu or static-shot plan. Preserve current WebUI route semantics while fixing collector behavior.

Current WebUI behavior checked in local code and README:

1. `src/webui/server.py::connect_bridge()` calls `_prepare_game_for_play()` only during startup or connect configuration. That recovery loop uses protocol transitions: `ready_for_new_set()` for `NEWTRAININGSET`, `RESUMETRAINING`, `NEWTRIAL`, and `NEWTESTSET`; `get_novelty_info()` plus `load_next_available_level()` for menu states. It exits when the engine reaches `PLAYING`.
2. `/api/shot` reads the shot payload and calls `bridge.shoot(...)`. It does not re-run `_prepare_game_for_play()` and does not re-prepare gameplay before each shot.
3. `/api/load-level` validates the requested minimum level, then currently routes to `load_next_available_level()`. `/api/restart` also routes to `load_next_available_level()`. In `src/webui/bridge.py`, that method sends protocol `53`.
4. `src/webui/bridge.py` still exposes explicit `load_level()` and `restart_level()` methods, which send protocol `51` and protocol `52`. Local WebUI README/research says those explicit paths can hang in this Unity build, especially from menu/startup or after loading NovPhy levels.

Collector guardrails:

1. Do not default the collector fix to protocol `51` or protocol `52`. Treat those as known risky comparison points unless later bounded runtime QA proves a narrow safe use.
2. Prefer protocol-only recovery already used by the collector/manual-agent path: new-set readiness via `ready_for_new_set()` and menu/end-state progression via novelty-info preflight plus `load_next_available_level()`.
3. WebUI route semantics are out of scope for this plan. Do not change `/api/shot`, `/api/load-level`, `/api/restart`, `src/webui/bridge.py`, `scripts/webui.sh`, or Java code while addressing the rollout collector bug.

## Collector pre-shot gameplay guard

Date: 2026-07-06

`scripts/collect_rollouts.py::collect_rollouts()` now guards every dataset shot before `shoot_once()` by combining protocol state and the actual desktop pre-shot surface when `pre_shot_grabber` is available.

- Protocol `PLAYING` is necessary but not sufficient for desktop capture; menu-like or uniform pre-shot images are rejected before shooting.
- Failed guards write `shot_*/metadata.json` with `pre_shot_guard.status: recovery_failed`, protocol/visual evidence, recovery attempts, recovery errors, and computed `artifact_validation` before raising.
- Bounded safe recovery mirrors `manual_agent.prepare_for_play()` semantics: `ready_for_new_set()` for `NEWTRIAL`/new-set states, and `get_novelty_info()` plus `load_next_available_level()` for menu/end or visually invalid surfaces.
- The collector guard intentionally does not call protocol `load_level(51)` or `restart_level(52)`.

Verification shape:

```sh
source "$HOME/cd_novphy" && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_pre_shot_guard_rejects_menu_surface_even_when_protocol_playing
source "$HOME/cd_novphy" && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_pre_shot_guard_recovers_new_trial_before_shooting
source "$HOME/cd_novphy" && python -m unittest tests.test_collect_rollouts
source "$HOME/cd_novphy" && python -m py_compile scripts/collect_rollouts.py tests/test_collect_rollouts.py
```

## Static/menu artifact fix and operator guidance

Date: 2026-07-06

Root cause: static, menu, and non-gameplay artifacts were accepted because the earlier collection path trusted `shoot_response=1` and protocol shot responses, then counted saved shot directories without validating the visual artifact. It also lacked accepted-vs-attempt separation, so a failed visual attempt could occupy `shot_002` or `shot_003` as if it were a usable rollout.

Known pre-fix bad artifact:

- `data/novphy_rollouts_dataset/train/novelty_level_0_type010101_00001_0_1_010101_0_1`
- `shot_001` is the valid baseline.
- `shot_002` is low-motion suspicious.
- `shot_003` is a SELECT LEVEL menu/static artifact.

Do not treat this pre-fix dataset path, or any artifact collected by the old path, as valid training data until it has been validated shot by shot. Do not trust `shoot_response=1` as proof of gameplay. The validator must decide acceptance from metadata plus frame evidence.

Validator outcomes now used by the collector:

- `gameplay-valid`: accepted. The shot has gameplay surface evidence and enough post-shot frame or pre-shot delta evidence.
- `menu_detected`: rejected and quarantined. Menu-like frames or SELECT LEVEL capture are not retryable acceptance candidates.
- `low_motion_suspicious`: rejected by default and marked retryable. This covers the known `shot_002` style, where motion is too small to trust as a gameplay rollout without stronger evidence.
- `no_frame_motion`: rejected. Static frames with no meaningful motion are not accepted.
- `missing_artifact`: rejected fail-closed when metadata, frames, or referenced artifacts are absent.

Runtime guards and gates:

- The pre-shot guard requires protocol and desktop evidence before shooting. Protocol `PLAYING` is necessary but not sufficient.
- The post-shot gate validates the newly written `shot_*/` directory before it can increment the accepted rollout count.
- `collect_rollouts.py` writes `attempt_count`, `accepted_rollout_count`, accepted rollout lists, and invalid attempt data so operators can distinguish attempts from usable samples.
- Invalid attempts are recorded under `invalid_attempts/` and do not increment accepted rollout count.
- Retryable low-motion attempts can start a fresh-engine retry while `fresh_engine_attempts` remain.
- Menu/static/non-retryable invalid attempts are quarantined, not silently overwritten or counted.
- Persistent `NEWTRIAL` after an invalid retry required bounded `prepare_for_play()` reissue and escalation. The safe lifecycle is bounded `ready_for_new_set()` reissue, then `get_novelty_info()` plus `load_next_available_level()` escalation when needed.
- Do not use protocol `load_level(51)` or `restart_level(52)` as the recovery path. They remain hang-risk commands for this Unity build.

Task 7 bounded runtime QA result:

- Run root: `/tmp/opencode/rollout_bugfix_check_retry_20260706_024619`
- Configured level: `9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml`
- Collector command status: `0`
- Manifest: `/tmp/opencode/rollout_bugfix_check_retry_20260706_024619/out/manifest.json`
- Scope: one configured reported level, not a full train/dev collection.
- Result: one accepted `gameplay-valid` shot, five invalid/quarantined attempts, no accepted menu/static shots.
- Counts: `attempt_count=6`, `accepted_rollout_count=1`, `invalid_attempt_count=5`, `quarantined_invalid_attempt_count=5`.
- Accepted shot: `shot_001`, `max_frame_delta=1049`, `max_pre_shot_delta=1541`, score `1770`.
- Invalid reasons observed: `low_motion_suspicious` and `menu_detected`, including `no-frame-motion` signals on menu captures.
- Runtime lifecycle evidence: recovered through `NEWTRIAL`, `NEWTRAININGSET`, `MAIN_MENU`, and `LOST` paths; one persistent `NEWTRIAL` attempt timed out, then the collector retried with a fresh engine.
- Cleanup evidence: no stale `rollout_bugfix_check`, `Xvnc :153`, `game_playing_interface`, or `collect_rollouts.py` process remained after the command exited.

Bounded QA command shape for operators:

```sh
set -euo pipefail
source ~/cd_novphy
RUN_ROOT="/tmp/opencode/rollout_bugfix_check_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT" .omo/evidence
python scripts/prepare_rollout_dataset.py write-config \
  --manifest data/novphy_rollouts_dataset_plan/partitions.json \
  --split train \
  --level-path 9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml
Xvnc :153 -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 \
  >"$RUN_ROOT/xvnc.log" 2>&1 &
XVNC_PID=$!
trap 'kill "$XVNC_PID" 2>/dev/null || true; wait "$XVNC_PID" 2>/dev/null || true' EXIT
sleep 2
DISPLAY=:153 LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}" \
  timeout -s INT 15m python scripts/collect_rollouts.py \
    --output-dir "$RUN_ROOT/out" \
    --capture-source desktop \
    --fresh-engine-per-rollout \
    --ui-level 1 \
    --count 3 \
    --fps 30 \
    --duration 5 \
    --connect-timeout 60 \
    --prepare-timeout 90 \
    --read-timeout 420 \
    --speed 1
STATUS=$?
kill "$XVNC_PID" 2>/dev/null || true
wait "$XVNC_PID" 2>/dev/null || true
trap - EXIT
exit "$STATUS"
```

Safe operator guidance:

- Before training on or using `data/novphy_rollouts_dataset`, validate pre-fix shots and exclude any shot whose validator result is not `gameplay-valid`.
- Use bounded QA on one or a few targeted levels before any large collection.
- Check `manifest.json`, `action_log.json`, per-shot `metadata.json`, and `invalid_attempts/` together. `rollout_count` or `shoot_response=1` alone is not enough.
- Treat a command exit of `0` plus accepted `gameplay-valid` shots as bounded-QA evidence only, not a promise that all levels or the full dataset are valid.
- Do not run a full train/dev dataset collection until the final verification wave passes and the user explicitly chooses to proceed.
- If final verification later approves a large collection, keep the same acceptance checks in place and review the invalid attempt ledger before using the dataset.

## Fresh Engine Process-Group Cleanup

Date: 2026-07-07

Large fresh-engine dataset runs can leave orphaned Unity children (`./9001.x86_64 ... --agent-port 2004 --dev`) even after the Java `game_playing_interface.jar` wrapper is stopped. Runtime inspection showed hundreds of PPID-1 Unity processes from prior rollout attempts, including user-reported PIDs `965979`, `967421`, `968797`, `974306`, `977147`, and `994093`.

`scripts.manual_agent.start_engine()` launches the Java wrapper with `start_new_session=True`; it now marks the returned process with `novphy_process_group = True`. `scripts.collect_rollouts.stop_owned_engine()` uses that marker to terminate the owned process group with `SIGTERM`, escalating to `SIGKILL` on timeout, and falls back to the previous PID-only terminate/kill behavior for unmarked injected or externally-owned processes.

Verification used a real process-group smoke: spawn a marked `bash` parent in a new session with a child `sleep`, call `stop_owned_engine()`, then `pgrep -g <pgid>` returns no remaining processes.

## Parallel Rollout Worker Port Isolation

Date: 2026-07-08

### Configurable worker port bases

The rollout planner defaults to agent/game base ports `2004` and `9001` with a fixed stride of `10`. `plan --agent-port-base` and `--game-port-base`, plus the full launcher environment variables `AGENT_PORT_BASE` and `GAME_PORT_BASE`, select an independent port family. Both values are decimal ports in `1..65535`; validation rejects a base whose final configured worker port would overflow and rejects any intersection between the derived agent and game port families. For example, `DISPLAY_ID=:152 AGENT_PORT_BASE=2034 GAME_PORT_BASE=9031 WORKERS=3` maps workers to `:152`/`2034`/`9031`, `:153`/`2044`/`9041`, and `:154`/`2054`/`9051`.

The launcher validates the configured bases and cross-family disjointness before output-root or lock filesystem effects, preflights the exact derived ports both before and after planning, and forwards the normalized bases to the planner. Default one-worker command scripts preserve their previous port-free collector invocation; a nondefault base makes the serial script explicitly pass its resolved agent and game ports. Port bases are intentionally not persisted in resume manifests because resume eligibility remains based solely on the existing completed episode contract.

A real `WORKERS=3` run originally generated distinct requested ports (`9001`, `9011`, `9021`) but the Java wrapper/Unity children still bound the default game-thread ports (`9001`, `9002`, etc.), producing `java.net.BindException: Address already in use` and leaving workers stuck without manifests. The root cause was that `scripts.manual_agent.start_engine(..., game_port=...)` passed `--port`, which is a Unity-side flag and is ignored by `game_playing_interface.jar`.

Bytecode inspection of `sciencebirdsgames/Linux/game_playing_interface.jar` proved the Java wrapper's actual flag is `--game-start-port`: `server.ABServer` parses it into `ABServer.gameStartPort`, and `server.ABServerManager.getFreePort()` starts allocation from that field. `scripts.manual_agent.start_engine()` now forwards `game_port` as `--game-start-port`, while `--agent-port` remains the wrapper's agent socket.

Bounded runtime proof used two temporary engine copies on isolated high ports, not a full dataset collection:

- Worker A: `--agent-port 2304 --game-start-port 9301`, `configure -> [1, 207, 0]`, log line `Starting new game thread with socket on port: 9301`.
- Worker B: `--agent-port 2314 --game-start-port 9311`, `configure -> [1, 207, 0]`, log line `Starting new game thread with socket on port: 9311`.
- No `BindException` appeared in either Java log.
- Default game port `9001` stayed closed during the high-port smoke.
- Cleanup left ports `2304`, `2314`, `9301`, `9311`, `9001`, and `9011` closed.

Current safe behavior:

- `WORKERS=1` remains the default serial behavior.
- `scripts/prepare_rollout_dataset.py plan --workers N` rejects `N < 1` as invalid.
- `WORKERS>1` is enabled for planning and generates per-worker X displays, temporary engine copies, output roots, agent ports, and Java game-start ports.
- Worker specs use `agent_port=2004 + index * 10` and `game_port=9001 + index * 10`; generated collection commands pass both `--engine-agent-port` and `--engine-game-port` into `scripts/collect_rollouts.py`.
- Generated worker functions remove their temporary engine copy with `trap 'rm -rf "$worker_root"' RETURN` when the worker exits.
- Generated parallel scripts keep the consumer-facing dataset layout as `OUT_ROOT/{train,dev}/...`; worker IDs are not inserted into final output paths.
- Generated parallel scripts track background worker PIDs, kill/wait them on `EXIT`, `INT`, or `TERM`, and clear the trap after all workers finish.
- `scripts/collect_full_rollout_training_dataset.sh` starts one Xvnc display per worker, then runs the generated parallel collection script.
- `scripts/collect_full_rollout_training_dataset.sh` rejects non-`:N` `DISPLAY_ID` values before Bash arithmetic expansion, preventing arithmetic command-substitution injection.
- Because the Java wrapper exposes unauthenticated agent/game sockets, `scripts/collect_full_rollout_training_dataset.sh` requires `NOVPHY_ALLOW_NETWORK_LISTENERS=1` for `WORKERS>1`; set it only on an isolated/firewalled host.
- `scripts.manual_agent.main()` now uses process-group-aware cleanup for engines it starts, matching the Java/Unity child cleanup strategy used by `scripts.collect_rollouts.stop_owned_engine()`.

Operator caution: the port-isolation path is now proven for startup/configure and Java/Unity socket binding, but full train/dev collection is still a large side-effecting operation. Use a bounded output root first when changing worker counts or capture parameters.

Validation commands run after the port-isolation fix:

```sh
source ~/cd_novphy && python -m unittest tests.test_prepare_rollout_dataset
source ~/cd_novphy && python -m unittest tests.test_manual_agent
source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_connect_or_start_engine_auto_starts_engine_after_connection_refusal tests.test_collect_rollouts.CollectRolloutsTest.test_connect_or_start_engine_forwards_custom_engine_ports tests.test_collect_rollouts.CollectRolloutsTest.test_main_wires_desktop_fresh_engine_baseline_capture
source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_stop_owned_engine_terminates_process_group_for_started_engine tests.test_collect_rollouts.CollectRolloutsTest.test_stop_owned_engine_falls_back_to_process_when_group_lookup_fails tests.test_collect_rollouts.CollectRolloutsTest.test_stop_owned_engine_escalates_process_group_after_timeout
source ~/cd_novphy && python -m py_compile scripts/prepare_rollout_dataset.py tests/test_prepare_rollout_dataset.py scripts/manual_agent.py tests/test_manual_agent.py scripts/collect_rollouts.py tests/test_collect_rollouts.py
source ~/cd_novphy && bash -n scripts/collect_full_rollout_training_dataset.sh
git diff --check
```

`basedpyright-langserver` was not installed, so LSP diagnostics could not run. The broad `tests.test_collect_rollouts` suite still has one unrelated fixture failure from previously mutated generated dataset artifacts: `test_known_dataset_artifacts_classify_gameplay_and_reported_menu_shots` expects `max_frame_delta=965`, while the current artifact reports `898`.

## Dynamic Xvnc Game Viewport Crop

Date: 2026-07-08

A `WORKERS=3` collection at `data/novphy_rollouts_dataset_20260708_122204` produced 640x480 saved frames with large black padding at the top and left. The reported frame
`train/novelty_level_0_type010101_00002_0_1_010101_0_1/shot_002/frames/frame_000001.png` had non-black content only inside bbox `(160, 80, 640, 480)`, so the historical fixed desktop crop `(32, 64, 672, 544)` was starting 160 px too far left and 80 px too high for that run.

The desktop crop path now detects the visible non-black game surface on the full Xvnc desktop first and crops a 640x480 viewport from that origin, falling back to the historical `(32, 64, 672, 544)` crop when detection is unavailable. This keeps old 1024x768 Xvnc layouts working while allowing shifted Unity windows such as inferred full-desktop crop `(192, 144, 832, 624)`.

Regression coverage:

```sh
source ~/cd_novphy && python -m unittest tests.test_collect_rollouts.CollectRolloutsTest.test_capture_desktop_rollout_detects_shifted_xvnc_game_viewport
```

Bounded image QA, without recollecting the full dataset, reconstructed the full-desktop placement from the bad saved frame and confirmed `_default_desktop_crop_for()` returns `(192, 144, 832, 624)`, the corrected crop is 640x480, and the non-black bbox moves to `(0, 0, 480, 400)` instead of being offset to `(160, 80, 640, 480)`.

## Novelty-Level Round-Robin Collection Scheduling

Date: 2026-07-15

Generated rollout collection commands now schedule task folders `novelty_level_1` through `novelty_level_8` in deterministic round-robin order. The planner preserves each novelty family's manifest order, then emits one entry from each available family before returning to the next entry in those families. This ordering is applied before serial command emission and before worker ordinal assignment, so both serial and parallel plans avoid a long prefix from one novelty family.

`novelty_level_0` is intentionally excluded from generated collection plans. Level 0 contains the non-novel baseline tasks and isn't part of this novelty rollout schedule. If a requested split contains only `novelty_level_0`, command generation raises a clear `Requested split contains no novelty levels 1 through 8` error instead of silently producing a baseline-only collection script.

The `novelty_level_N` folder name and Unity's `ui_level` are separate concepts. The folder identifies which novelty task XML is being scheduled. In serial plans, `write-config` writes exactly one selected XML to `sciencebirdsgames/Linux/config.xml`. In parallel plans, each worker writes exactly one selected XML to its isolated temporary engine copy's `config.xml` through the existing `--config-path "$worker_engine_dir/config.xml"`. In both modes, that XML is Unity's first and only configured UI slot, so the following collector command must retain `--ui-level 1`. The round-robin change does not select Unity UI slots 1 through 8.

Focused verification:

```sh
source ~/cd_novphy && python -m unittest tests.test_prepare_rollout_dataset
```

This ran 18 tests and finished `OK`. `py_compile` and `git diff --check` also passed.

Bounded real-plan QA inspected and rendered the existing `20260708_171531` plan without running collection. Its generated train command prefix is `novelty_level_1` through `novelty_level_8`, then the same sequence repeats. The plan schedules 11,200 train entries, contains no `novelty_level_0` collection command, retains `--ui-level 1` in all 11,200 collector invocations, and the rendered shell passes `bash -n`.

This change affects generated planning command order only. It does not recollect, rewrite, or otherwise modify existing artifacts under `data/novphy_rollouts_dataset_20260708_171531`.

## Capped Non-Destructive Rollout Cohort

Date: 2026-07-25

`scripts/prepare_rollout_dataset.py plan` now inventories an existing output root without writing to it. The default contract selects exactly 100 train and 20 dev episodes for each of the 80 discovered normal/novel `(novelty_level, level_type)` buckets: 9,600 selected episodes total, split evenly into 4,800 normal and 4,800 novel episodes. Test remains unscheduled.

Canonical existing episodes count only when their fresh-engine desktop manifest exactly matches `count=12`, `fps=30`, `duration=5`, `ui_level=1`, accepted-attempt validation, required action logs, and raw accepted-shot artifacts. Level-5 existing episodes additionally require accepted action-log horizontal release signs to begin negative and alternate negative/positive. Incompatible, partial, unreadable, escaping, and symlinked paths are preserved as surplus; the planner chooses a later absent path instead.

The emitted script contains only absent selections, reserves each output directory with exclusive `mkdir`, and appends failures through a locked ledger. It retains fresh-engine worker isolation and applies `--bidirectional-launches` only to `novelty_level_5`.

Non-destructive dry plan verification against `data/novphy_rollouts_dataset_20260708_171531` wrote only `/tmp/opencode/novphy_capped_dry_plan`: 9,600 selected, 2,200 canonical existing, and 7,400 newly scheduled; `bash -n` passed for the generated script. No rollout collector, engine, Java, Unity, Xvnc, or generated script was run.

## Partition-Safe Capped Plans

Date: 2026-07-25

The capped planner retains the v1 partition contract: `DEFAULT_SEED` is `novphy-rollout-dataset-v1`, and `partition_levels()` deterministically splits each novelty/type bucket into disjoint train, dev, and test populations before cap selection. `build_collection_plan()` selects only from the corresponding train or dev partition, so a path cannot be planned for both collection splits and test remains unscheduled.

`CollectionOptions` and generated scripts retain the runtime collector contract: `DISPLAY="$display_id"`, `LD_LIBRARY_PATH`, `--ui-settle-seconds 5`, `--connect-timeout 60`, `--prepare-timeout 90`, `--read-timeout 420`, and `--speed 1`, in addition to worker engine and port options. The full launcher names its controls `PARTITION_SEED`, `TRAIN_TARGET_PER_BUCKET`, and `DEV_TARGET_PER_BUCKET`.
