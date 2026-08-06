## Manual rollout capture notes

- For preliminary rollout capture, the human-facing WebUI was not used as the driving surface; direct agent/runtime control was more reliable.
- `src.webui.bridge.ScienceBirdsBridge.load_level()` can hang on this runtime after startup, matching the existing WebUI workaround.
- `load_next_available_level()` and `restart_level()` were also unreliable for selecting/repeating an arbitrary NovPhy level from active PLAYING state.
- A reliable workaround is to rewrite `sciencebirdsgames/Linux/config.xml` temporarily so it contains only the chosen level, then start a fresh engine session per rollout.
- Direct ground-truth from bridge request `62` is GeoJSON-like: top-level list containing an item with `features`, where the slingshot is a feature with `properties.label == "Slingshot"` and polygon coordinates in screen-space.
- A workable slingshot anchor for drag/release capture was computed from the slingshot polygon bounding box using the same reference-point offsets already used elsewhere: `screen_x = min_x + 0.45 * width`, `screen_y = min_y + 0.35 * width`, then convert to bottom-left game coordinates with `game_y = frame_height - 1 - screen_y`.
- Sample artifacts were saved under `analysis/webui_capture_review/20260612T004349/` with one folder per rollout containing `frames/`, `rollout.mp4`, and `metadata.json`; top-level `manifest.json` summarizes the set.
- In this environment, raw screenshot requests (`11` and `61`) both returned uniform gray `(205,205,205)` frames even when the game state and score changed. The practical fix for review artifacts was to detect uniform screenshots and render a visible frame from symbolic state instead.
- `scripts/manual_agent.py` now saves PNG/PPM frames via a symbolic fallback renderer when the screenshot is uniform, so rollout frame sequences are no longer blank.
- `scripts/manual_agent.py::prepare_for_play()` also needed to treat `WON`/`LOST` like menu states and call `load_next_available_level()`; otherwise some fresh sessions could stall forever in `LOST`.
- A validated manual-agent rollout sample with non-blank frames was saved under `analysis/manual_agent_capture_review/20260612T022027/`.

## Actual rendered frame capture update

- `scripts/manual_agent.py::save_frame()` now refuses uniform TCP screenshots instead of converting them into symbolic fallback drawings. A uniform screenshot raises `RuntimeError: uniform Science Birds screenshot; refusing to save symbolic fallback as a rollout frame`.
- Regression coverage is in `tests/test_manual_agent.py::test_save_frame_rejects_uniform_screenshot_without_symbolic_fallback`.
- TCP screenshot request 11 can return uniform gray in this KDE/Wayland runtime because Unity logs `ReadPixels was called to read pixels from system frame buffer, while not inside drawing frame.` Treat that as a failed real-frame capture, not as data.
- For the 2026-06-13 recollection, stale Science Birds/WebUI runtimes on port 2004 had to be stopped first. A clean foreground Science Birds session on KDE virtual desktop 1 produced visible gameplay frames.
- Valid actual rendered rollout artifacts were saved under `analysis/manual_agent_capture_review/20260613T121106/`. The capture source is recorded as `spectacle-desktop-crop` with crop box `[0, 600, 650, 1250]`; frames visually show the slingshot, red birds, pig, ground, and level objects.

## High-FPS pixel rollout and action diversity update

- `scripts/manual_agent.py::capture_pixel_rollout()` records raw pixel screenshot sequences into `frames/frame_*.png` with `metadata.json` containing target FPS, elapsed timestamps, per-frame stats, state samples, and optional action signatures.
- The capture loop uses `ScienceBirdsBridge.screenshot()` directly and still rejects uniform frames; uniform TCP screenshots are failed pixel captures, not training data.
- The manual REPL command `rollout DIR [fps] [seconds]` captures timestamped pixel frames at the requested target FPS without requiring headless mode.
- `scripts/manual_agent.py::generate_diverse_drag_release_actions()` creates deterministic `drag_hold_release` action dictionaries spanning release angle, pull strength, tap time, and hold time.
- `scripts/manual_agent.py::deduplicate_similar_actions()` keeps the first action per binned signature `(strength, angle, tapTime, holdTime)`, which is a lightweight pre-rollout diversity filter before more expensive visual/outcome deduplication.
- Dry surface check: `python3 scripts/manual_agent.py --print-diverse-actions --diverse-action-count 3` prints JSON actions without connecting to Java.
- `scripts/collect_rollouts.py` is the standalone collection entry point. Use `python3 scripts/collect_rollouts.py --dry-run --output-dir /tmp/plan --count 3` to write only `action_plan.json`, or run it without `--dry-run` against a connected Science Birds engine to shoot each generated action and save `manifest.json` plus per-shot frame metadata.
- If no Science Birds engine is listening, `scripts/collect_rollouts.py` auto-starts `sciencebirdsgames/Linux/game_playing_interface.jar`, retries the connection, and terminates only that auto-started process after collection.
- The collector preflights output-directory writability before engine startup. Owned engine cleanup uses terminate + wait, with kill fallback if the engine ignores termination.
