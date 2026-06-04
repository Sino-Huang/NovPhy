# NovPhy Science Birds WebUI

This folder contains a dependency-free local WebUI for manually playing the NovPhy Science Birds game through the existing Java game interface.

The browser does not embed the native Unity window. Instead, the Python backend talks to `game_playing_interface.jar` over the existing Science Birds TCP protocol, fetches raw RGB screenshots, and renders them into an HTML canvas.

## Prepare the root Linux engine

Run this once before starting the WebUI:

```sh
python3 sciencebirdsagents/Utils/PrepareTestConfig.py --os Linux --novelty-level novelty_level_0 --level-type type010101 --max-levels 20
```

Expected result: `sciencebirdsgames/Linux/config.xml`, `sciencebirdsgames/Linux/game_playing_interface.jar`, and `sciencebirdsgames/Linux/DB/` exist.

## Start the WebUI

```sh
python3 -m src.webui.server
```

Then open:

```text
http://127.0.0.1:8766/
```

Click **Start game** to launch `game_playing_interface.jar` and connect to it. If you already started the game interface separately, click **Connect** instead.

## Manual controls

- **Refresh frame** fetches a `640 x 480` RGB screenshot from the Java interface.
- **Load next** asks the game for the next available level. The backend validates the entered minimum level, but uses Science Birds protocol message `53` because this Unity build can hang when explicit level selection message `51` is sent from the menu/startup path.
- **Restart** reloads through the same next-available-level protocol path. The Unity `RestartLevel` / explicit selection path can hang in this build after loading NovPhy levels.
- **Zoom out / Zoom in** call the game interface zoom commands.
- Drag on the canvas to preview aim with a browser-local guide line and trajectory dots; release fills the shot `x` and `y` fields. The UI converts browser top-left coordinates into Science Birds bottom-left coordinates.
- On release, the WebUI also builds the same agent action dictionary accepted by the environment wrappers. The drag start is treated as a mimic slingshot center, so a drag from `[300, 220]` to `[250, 180]` becomes `{"action_type":"drag_release","coordinate_frame":"slingshot_relative","drag_start":[300,220],"drag_release":[-50,40]}`.
- `/api/agent-action` validates that shared action contract and returns the equivalent `/api/shot` payload. It does not call an external environment wrapper instance; a separate consumer is still required for that.
- The browser validates/transfers the agent action through `/api/agent-action` immediately after release. Auto-execution is enabled by default: the translated shot is scheduled through `/api/shot` without waiting for the engine acknowledgement. Uncheck the auto-execute control to validate/transfer only.
- **Send shot** sends the final Cartesian shot coordinates through `/api/shot`. **Fast shot** is enabled by default for responsive drag/release execution and uses protocol message `41`; uncheck it to use safe-shot protocol message `31`, which can wait for the engine's static-state acknowledgement. There is no live TCP drag command in this WebUI path.

## Headless game mode

If your environment supports headless screenshots, start the WebUI with:

```sh
python3 -m src.webui.server --game-headless
```

## Verification

Run the WebUI tests without starting Java:

```sh
python3 -m unittest tests.test_webui_static tests.test_webui_bridge tests.test_webui_server
```

Run the server help check:

```sh
python3 -m src.webui.server --help
```

If the backend cannot connect to port `2004`, check whether another Science Birds game interface is already running. The WebUI only stops the Java process it started; it does not kill unrelated Java processes.
