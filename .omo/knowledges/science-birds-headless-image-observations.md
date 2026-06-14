# Science Birds Headless Image Observations

Date: 2026-06-15

## Confirmed Finding

Science Birds protocol `61` image observations are not reliable in true `--headless` mode in the current NovPhy/Science Birds build. Symbolic ground truth can still be valid, but the screenshot pixels returned through the agent observation API can be uniform gray.

## Evidence

- Python protocol path: `sciencebirdsagents/Client/agent_client.py::get_symbolic_state_with_screenshot()` sends `RequestCodes.GetGroundTruthWithScreenshot` and reads ground truth followed by image bytes.
- Unity protocol handler: `tasks/task_template_designer/Assets/Scripts/AIBirdsConnection.cs::GroundTruthWithScreenshot()` waits for end-of-frame, builds `SymbolicGameState`, calls `GetGTJson()`, then calls `GetScreenshotStr()`.
- Unity screenshot backend: `tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs::TakeScreenshot()` creates a `Texture2D(Screen.width, Screen.height, ...)` and calls `ReadPixels(...)` on the screen framebuffer.
- Headless artifact: `analysis/manual_agent_capture_review/20260613T224821_headless_agent_observation/manifest.json` launched with `--headless --dev` and produced five `(3, 480, 640)` frames with `unique_colors: 1`, `uniform: true`, while each frame still had `ground_truth_feature_count: 14`.
- Non-headless artifact: `analysis/manual_agent_capture_review/20260613T222822_agent_observation/manifest.json` used the same protocol-61 agent observation path and produced non-uniform gameplay frames, e.g. `unique_colors: 3951`, `uniform: false`, with valid ground truth.

## Interpretation

The issue is not Python image decoding or the protocol shape. It is the shared Unity screenshot backend reading the default screen framebuffer. In true headless mode that framebuffer is absent, not rendered, or cleared to a uniform color, so image observations become blank even though symbolic state remains available.

## Related AiBirds Finding

`https://github.com/BluemlJ/AiBirds` also uses screenshot-backed image observations rather than a headless-safe capture path:

- `src/envs/angry_birds.py` launches `Science Birds.exe` directly without headless flags.
- `AngryBirds.get_states()` calls `get_ground_truth_with_screenshot()` and crops/resizes the screenshot into the observation.
- `src/envs/ab/agent_client.py` exposes protocol `61` screenshot+GT and no-screenshot GT, but the Angry Birds environment uses the screenshot variant for image state.

Conclusion: AiBirds does not prove headless image observations are safe. Its image pipeline assumes a working graphical screenshot source.

## Recommended Options

1. Operational workaround: run image-observation agents with a graphical Science Birds process. On servers, use a graphics-capable virtual display/session rather than true headless mode.
2. Engine-side durable fix: replace the framebuffer `ReadPixels` path in `SymbolicGameState.TakeScreenshot()` with camera-to-`RenderTexture` capture, then read pixels from the render texture.
3. Wrapper guard: reject or warn when image observation mode is combined with explicit `--headless` unless the engine screenshot backend has been fixed and validated.

## Validation Targets

A correct fix should make protocol `61` return non-uniform gameplay pixels and valid ground truth in both graphical and intended server-run modes. Keep checks for `unique_colors > 1`, `uniform == false`, expected `(3, 480, 640)` observation shape, and non-empty ground-truth features.
