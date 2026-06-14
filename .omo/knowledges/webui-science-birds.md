# Science Birds WebUI

`src/webui` provides a dependency-free local WebUI for NovPhy Science Birds.

Architecture:

- `src/webui/bridge.py` is a standard-library TCP client for `game_playing_interface.jar` on `127.0.0.1:2004`.
- `src/webui/server.py` serves static files and JSON APIs with `http.server`.
- `src/webui/static/` renders raw RGB screenshots to a browser canvas and sends manual shot, level, restart, and zoom commands.

Run setup first:

```sh
python3 sciencebirdsagents/Utils/PrepareTestConfig.py --os Linux --novelty-level novelty_level_0 --level-type type010101 --max-levels 20
```

Start the WebUI:

```sh
python3 -m src.webui.server
```

Open:

```text
http://127.0.0.1:8766/
```

Verification performed on 2026-06-04:

- `python3 -m unittest tests.test_prepare_test_config tests.test_webui_bridge tests.test_webui_server` passed.
- `python3 -m src.webui.server --help` printed CLI help.
- Runtime smoke test launched WebUI with no args on default port `8766`, called `/api/start`, fetched `640x480` frames through `/api/frame`, exercised `/api/load-level` and `/api/restart`, then called `/api/stop`. Frames stayed in `PLAYING`, had non-uniform RGB bytes, and reported `numberOfLevels: 20` from the generated config.
- Science Birds explicit level selection request `51` (`selectlevel`) and the restart route can trigger Unity `NullReferenceException` paths and leave HTTP requests hanging. Startup, WebUI level loading, and WebUI restart should use request `53` (`LoadNextAvailableLevel`) instead; verified that it reaches `PLAYING` and returns a non-uniform `640x480` frame.
- Default WebUI port is `8766` because `8765` can be occupied by VS Code in this workspace.
- `game_playing_interface.jar` output is redirected to `subprocess.DEVNULL`; do not pipe it unless a reader drains the pipe, or long-running sessions can stall when Unity logs fill the buffer.
- WebUI drag aiming is browser-local: pointer drag on the canvas redraws an aim line and trajectory dots without backend calls, release fills final absolute shot fields, and `Send shot` posts through `/api/shot`. There is no live TCP drag/finger command in the verified protocol path.
- Include `tests.test_webui_static` in WebUI verification when changing static UI behavior.
- WebUI drag release also mirrors the agent wrapper action contract. It treats `drag_start` as a mimic slingshot center, sends `{"action_type":"drag_release","coordinate_frame":"slingshot_relative","drag_start":[x,y],"drag_release":[dx,dy],"tapTime":t}` to `/api/agent-action` immediately, and schedules the returned shot through `/api/shot` by default without waiting for the engine acknowledgement. Fast shot is checked by default so release uses protocol `41`. Uncheck the auto-execute control for validation-only transfers. `/api/agent-action` validates/translates only; it does not call an external env wrapper instance. It accepts `drag_release` and `drag_hold_release` action types.
- `/api/shot` accepts bottom-left game coordinates from the WebUI/agent-action path and converts Y to screenshot/canvas coordinates before calling `ScienceBirdsBridge.shoot(...)`; this keeps visible drag/release aiming aligned with the game engine's final shot command.

The WebUI does not embed the native Unity window. It displays screenshots from the game TCP protocol. The backend only stops the Java process it started.

## Trajectory Preview and Engine Lag Notes

Verification performed on 2026-06-08:

- WebUI trajectory preview now mirrors the Unity launch integration from `modules/benchmark/tasks/task_template_designer/Assets/Scripts/GameWorld/Characters/Birds/ABBird.cs`: capped drag radius 1 world unit, max launch speed 10, launch gravity 0.48, time step 0.02, and 500 integration steps. The backend exposes the active level camera `maxWidth` as `trajectoryWorldWidth` on `/api/frame`, so generated IratusAves levels such as `level-04.xml` use width `30` instead of the previous hardcoded `17.5`. The browser preview uses `buildTrajectoryPreviewPoints(...)` in `src/webui/static/app.js` rather than the old ad-hoc parabola, and `pointerup` sends/fills the same capped release point so long drags do not preview one trajectory while executing another.
- `python3 -m unittest tests.test_webui_static tests.test_webui_bridge tests.test_webui_server` passed after the trajectory change, including executable JavaScript trajectory coverage for a non-default camera width.
- `lsp_diagnostics src/webui/static/app.js` reported no diagnostics.
- When diagnosing Science Birds lag, first check the active runtime config at `sciencebirdsgames/Linux/config.xml`. Generated IratusAves configs store level paths in the `game_levels` `level_path` attribute, not in node text.
- The inspected active config pointed to one generated IratusAves level: `9001_Data/StreamingAssets/Levels/iratus_aves/Levels/level-04.xml`. That level had 124 counted objects: 63 `Block`, 55 `Platform`, 5 `TNT`, and 1 `Pig`.
- For comparison, sampled normal NovPhy `novelty_level_0/type010101` and `type010102` levels had only small single-digit object counts. A generated IratusAves level with over 100 objects is therefore a plausible primary source of Unity physics/rendering/stability lag.
- WebUI `/api/frame` is intentionally expensive: it sends screenshot request 11, reads `width * height * 3` raw RGB bytes, base64-encodes them for JSON, and also queries state, score, current level, and number of levels while holding the bridge lock. Frequent refresh/polling can make the interface feel sluggish even if the game itself is simulating normally.
- Unity screenshot and ground-truth-with-screenshot paths are also expensive: `SymbolicGameState.GetScreenshotStr()` calls `ReadPixels`, `Apply`, `EncodeToPNG`, and base64 conversion. Batch ground-truth recording after shots can capture screenshot plus symbolic state every `takeGroundTruthEveryNthFrames` frames, defaulting to 5.
- No active `python3 -m src.webui.server`, `game_playing_interface.jar`, `9001.x86_64`, or related Java process was found during the lag investigation, so stale live processes were not the observed cause at that moment.
- For engine-only smoothness tests without WebUI screenshot polling, use `scripts/manual_agent.sh`. In one terminal run `scripts/java_engine.sh`; in another run `scripts/manual_agent.sh`. The manual agent connects, configures training mode, loads a level, and then waits at a stdin prompt without polling frames. You can try the native Unity window directly or send typed commands such as `state`, `score`, `shoot X Y`, `zoom out`, `next`, and `quit`.

## NovPhy Built-in Level Variety

Verification performed on 2026-06-08:

- The root Linux runtime level tree contains 27,951 parseable NovPhy XML levels plus 5 IratusAves XML levels under `sciencebirdsgames/Linux/9001_Data/StreamingAssets/Levels`.
- NovPhy distribution by top-level folder: `novelty_level_0` has 13,951 levels; `novelty_level_1` through `novelty_level_8` each have 1,750 levels.
- Parsed NovPhy coverage is 9 novelty groups and 80 `type010*` groups. `novelty_level_0` alone includes 40 normal template/type groups across the five physical scenarios.
- Scenario counts by novelty: `novelty_level_1` through `novelty_level_8` each have 350 levels for each scenario: single force, multiple forces, rolling, falling, and sliding. `novelty_level_0` has 2,800 levels each for single force, multiple forces, rolling, and sliding, and 2,751 for falling.
- The XML files declare `utf-16` but are readable as UTF-8/ASCII text; for corpus analysis, strip the XML declaration before parsing with `xml.etree.ElementTree`.
- NovPhy levels are much lighter than the inspected IratusAves level. Example normal groups: `type010101` has object min/median/max `3/5/7`; `type010102` has `5/7/9`; `type010203` has `12/14/16`; `type010403` has `13/15/17`. The heaviest parsed NovPhy groups top out around 17 counted gameplay objects, compared with the inspected IratusAves level's 124 counted objects.
- Current active config was verified to contain 350 NovPhy `novelty_level_0/type010101` levels and no `iratus_aves` paths.
- Recommendation: avoid IratusAves for smooth interactive experiments unless specifically testing dense generated out-of-distribution structures. Use NovPhy subsets via `PrepareTestConfig.py`, e.g. `--novelty-level novelty_level_0 --level-type type010101 --max-levels 350`, or rotate selected `type010*` groups for broader layout/scenario coverage.

## Agent Drag/Hold/Release Action Space

Verification performed on 2026-06-08:

- The shared dict action contract now accepts both `action_type: "drag_release"` and `action_type: "drag_hold_release"`.
- `drag_hold_release` uses `drag_start`, `drag_release`, optional `holdTime`/`releaseTime`, and `tapTime`. `holdTime` maps to the Science Birds protocol release time (`t1`); `tapTime` remains the post-release tap delay (`t2`).
- `/api/agent-action` translates `drag_hold_release` into a `/api/shot` payload that includes `releaseTime` when non-zero, and `/api/shot` passes it through to `ScienceBirdsBridge.shoot(..., release_time=...)`.
- `scripts/manual_agent.py` supports stateful REPL commands: `drag X Y`, `hold [MS]`, and `release X Y [tap] [fast|safe]`. Coordinates are bottom-left game coordinates. Because the verified TCP protocol has no live drag command, the manual agent prints the normalized action but sends the final release shot only at `release`.
