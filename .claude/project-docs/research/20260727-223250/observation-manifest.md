# Observation Manifest

| observation_id | source | evidence layer | observer group | observed_at | anchor | notes |
| --- | --- | --- | --- | --- | --- | --- |
| O1 | `tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs` | local source | orchestrator discovery | 2026-07-27 | file existence | Requires inspection; does not establish shipped-build availability. |
| O2 | `sciencebirdsgames/Linux.zip` | local artifact | orchestrator discovery | 2026-07-27 | file existence | Requires archive inspection. |
| O3 | `src/webui/bridge.py:35-40,123-125,173-176` | local source | collector/protocol | 2026-07-27 | request 62 and parser | Existing one-shot state path. |
| O4 | `sciencebirdsagents/Client/agent_client.py:310-341` | local source | legacy protocol | 2026-07-27 | request 38 batch reader | Existing batch GT wire contract. |
| O5 | `tasks/task_template_designer/Assets/Scripts/AIBirdsConnection.cs:327-505,1581-1610` | local source | engine protocol | 2026-07-27 | batch handler and registrations | Counterpart to O4. |
| O6 | `tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs:32-368` | local source | exporter schema | 2026-07-27 | object exporter | Omits physical relation/velocity fields. |
| O7 | `tasks/task_template_designer/Assets/Scripts/GameWorld/ABGameObject.cs:24-35,93-156,258-298` | local source | physics integration | 2026-07-27 | Rigidbody/collision access | Confirms engine has physical quantities. |
| O8 | `sciencebirdsgames/Linux.zip` -> `Assembly-CSharp.dll` strings | built artifact | deployed binary | 2026-07-27 | relevant class/method symbols | Metadata parity only, not live protocol proof. |
| O9 | Unity 2019.4 Rigidbody2D/ContactPoint2D/Collider2D docs | primary external docs | Unity API | 2026-07-27 | API descriptions | Version-compatible with local Unity 2019.3 project. |
