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
- WebUI drag release also mirrors the agent wrapper action contract. It treats `drag_start` as a mimic slingshot center, sends `{"action_type":"drag_release","coordinate_frame":"slingshot_relative","drag_start":[x,y],"drag_release":[dx,dy],"tapTime":t}` to `/api/agent-action` immediately, and schedules the returned shot through `/api/shot` by default without waiting for the engine acknowledgement. Fast shot is checked by default so release uses protocol `41`. Uncheck the auto-execute control for validation-only transfers. `/api/agent-action` validates/translates only; it does not call an external env wrapper instance. It rejects action types other than `drag_release`.
- `/api/shot` accepts bottom-left game coordinates from the WebUI/agent-action path and converts Y to screenshot/canvas coordinates before calling `ScienceBirdsBridge.shoot(...)`; this keeps visible drag/release aiming aligned with the game engine's final shot command.

The WebUI does not embed the native Unity window. It displays screenshots from the game TCP protocol. The backend only stops the Java process it started.
