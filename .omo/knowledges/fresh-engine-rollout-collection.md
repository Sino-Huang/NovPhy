# Fresh Engine Rollout Collection

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
