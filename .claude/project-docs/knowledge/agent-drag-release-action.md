# Agent Drag Release Actions

`SBEnvironmentWrapper.step()` originally accepted only final slingshot-relative release actions: `[dx, dy]`, `[dx, dy, tap_time]`, tensors with the same shape, or `Point2D`-like values.

The wrappers now also accept dict actions for agent-facing drag/release semantics:

```python
env.step({"drag_release": [-100, 50], "tap_time": 70})
env.step({"release": [120, 260], "coordinate_frame": "absolute", "tapTime": 70})
```

Rules:

- `drag_release` and `release` are aliases for the final release point.
- `coordinate_frame` defaults to `slingshot_relative`.
- `coordinate_frame="absolute"` converts board coordinates to relative action coordinates using the current slingshot center: `dx = x - sling_x`, `dy = sling_y - y`.
- `drag_start` may be supplied by agents for readability, but the wrapper still executes only the final release point because the verified TCP path has no live drag/finger command.
- Existing list/tensor/`Point2D` action compatibility is preserved.
- `SBEnvironmentWrapperOpenAI` keeps its existing discrete and continuous action spaces; dict actions are accepted as an extra manual path before `action_space.contains(...)`.
- The WebUI now emits the same dict contract after canvas release. It uses the drag start as a mimic slingshot center and validates the action through `/api/agent-action`; that endpoint returns an equivalent final `/api/shot` payload but does not directly call an external env wrapper instance.

Verification command:

```sh
python3 -m unittest tests.test_sb_action_semantics
```
