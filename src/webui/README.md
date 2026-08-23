# NovPhy Science Birds WebUI

This folder contains a dependency-free local WebUI for manually playing the NovPhy Science Birds game through the existing Java game interface.

The browser does not embed the native Unity window. Instead, the Python backend talks to `game_playing_interface.jar` over the existing Science Birds TCP protocol, fetches raw RGB screenshots, and renders them into an HTML canvas.

## Prepare the root Linux engine

Run this once before starting the WebUI:

```sh
python3 sciencebirdsagents/Utils/PrepareTestConfig.py --os Linux --novelty-level novelty_level_0 --level-type type010101 --max-levels 20
```

Expected result: `sciencebirdsgames/Linux/config.xml` is written after validating that the real Science Birds Java interface assets already exist at `sciencebirdsgames/Linux/game_playing_interface.jar` and `sciencebirdsgames/Linux/DB/`. Provision those runtime assets into the root engine directory before running the command.

## Start the WebUI

```sh
python3 -m src.webui.server
```

Then open:

```text
http://127.0.0.1:8766/
```

Click **Start game** to launch `game_playing_interface.jar` and connect to it. If you already started the game interface separately, click **Connect** instead.

`bash scripts/webui.sh --physics-v2-review` defaults the existing `--speed` option to `1` so issue-44 shots are observable. A later explicit option, such as `--speed 5`, overrides that review default.

## Review the failed issue-53 production run

Issue #53 has a separate retained-evidence review page. It does not use or
change the issue-44 collision/support state machine:

```sh
bash scripts/webui.sh \
  --issue-53-review-root .local-artifacts/issue-53-production-run \
  --review-output-dir .local-artifacts/issue-53-human-review-v2 \
  --speed 1
```

The playlist is loaded from the frozen execution and quality reports. Opening
an item plays its original `physics_capture_v2` fixed-step trace as a schematic.
The three final-evaluation items stay locked until the production authorization
identity is validated; their access records, decisions, and optional diagnostic
replays are written under the review output's `sealed/` directory.

This is the v2 review workflow. The original
`.local-artifacts/issue-53-human-review` directory remains v1 setup-defect
evidence and is never resumed or overwritten. Before a v2 diagnostic replay
can shoot, the page and result expose the shared screen-coordinate alignment
contract: accelerated zoom-out, stable anchor/scale at startup and speed 1,
and a retained production-anchor match within two pixels.

Each item permits at most one optional diagnostic replay after the retained
trace is opened. The replay uses the retained scenario XML, scenario manifest,
player envelope, and anchored socket command, captures live RGB frames and a
new request-71 trace, and encodes Chrome-compatible VP8 as `replay.webm`. Every replay is marked
`diagnostic_only`; it is ineligible for quotas, production accounting,
resampling, and release. Reviewer decisions are immutable and never modify the
production runtime or either production bundle.

## Manual controls

- **Refresh frame** fetches a `640 x 480` RGB screenshot from the Java interface.
- **Load next** asks the game for the next available level. The backend validates the entered minimum level, but uses Science Birds protocol message `53` because this Unity build can hang when explicit level selection message `51` is sent from the menu/startup path.
- **Restart** reloads through the same next-available-level protocol path. The Unity `RestartLevel` / explicit selection path can hang in this build after loading NovPhy levels.
- **Zoom out / Zoom in** call the game interface zoom commands.
- Drag on the canvas to preview aim with a browser-local guide line and trajectory dots; release fills the shot `x` and `y` fields. The UI converts browser top-left coordinates into Science Birds bottom-left coordinates.
- On release, the WebUI caps the release point and trajectory preview from the runtime slingshot reference and camera scale when symbolic state is available. The preview uses the authoritative Physics2D gravity and fixed-step integration, so browser drag strength and flight stay aligned with the engine trace even if pointer-down starts a little off-center. The generated `drag_hold_release` action dictionary uses that same slingshot-centered anchor and preserves the shared `holdTime` / `tapTime` fields.
- `/api/agent-action` validates that shared action contract and returns the equivalent `/api/shot` payload. It does not call an external environment wrapper instance; a separate consumer is still required for that.
- The browser validates/transfers the agent action through `/api/agent-action` immediately after release. Auto-execution is enabled by default: the translated shot is scheduled through `/api/shot` without waiting for the engine acknowledgement. Uncheck the auto-execute control to validate/transfer only.
- **Send shot** sends the final Cartesian shot coordinates through `/api/shot`, using the Hold time field as `/api/shot` `releaseTime` and Tap time as `tapTime`. **Fast shot** is enabled by default for responsive drag/release execution and uses protocol message `41`; uncheck it to use safe-shot protocol message `31`, which can wait for the engine's static-state acknowledgement. There is no live TCP drag command in this WebUI path; the browser preview stays local and only the final shot is synchronized.

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
