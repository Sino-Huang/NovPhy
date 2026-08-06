# Wave 1: Local Protocol and Engine Source

- The active Python bridge exposes request 62, `GET_GROUND_TRUTH_WITHOUT_SCREENSHOT`, and parses JSON from the engine (`src/webui/bridge.py:35-40,123-125,173-176`). The rollout collector already calls it to find the slingshot (`scripts/collect_rollouts.py:350-364,1667-1672`) but saves desktop PNG frames rather than state JSON (`scripts/collect_rollouts.py:1184-1335`).
- The legacy client has request 38, `GTshoot`, and receives a batch of ground-truth JSON records every requested N frames (`sciencebirdsagents/Client/agent_client.py:40-70,310-341`).
- The Unity template implements the matching `ShootAndRecordGroundTruth` handler (`tasks/task_template_designer/Assets/Scripts/AIBirdsConnection.cs:327-421,1581-1608`). `ABGameWorld.Update` records a `SymbolicGameState` every configured N Unity update frames (`tasks/task_template_designer/Assets/Scripts/GameWorld/ABGameWorld.cs:372-395`).
- `SymbolicGameState` emits object lifetime instance IDs, polygons, label/type, color map, and in development mode current life (`tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs:32-368`; `GTObject.cs:27-96`). It does not serialize velocity, mass, contacts, support, or macro events.

## EXPAND
- LEAD: Verify source-template parity with the deployed Linux binary — WHY: direct collection is useful only if shipped engine contains the handler — ANGLE: inspect `Assembly-CSharp.dll` metadata.
- LEAD: Determine whether Unity supplies the missing physical quantities — WHY: distinguishes exporter work from image inference — ANGLE: Unity 2019.3 primary API docs and local collision code.
