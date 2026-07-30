# Claim Graph

## Verified claims

| claim_id | statement | evidence | status |
| --- | --- | --- | --- |
| C1 | The deployed Science Birds build contains the existing symbolic-state and batch-ground-truth machinery. | Template source plus deployed `Assembly-CSharp.dll` metadata. | supported |
| C2 | The current exporter directly provides an object-level scene description but not velocity/contact/support/event fields. | `SymbolicGameState.cs` and `GTObject.cs`; bridge protocol. | supported |
| C3 | Velocity, mass, and contact geometry are available to a Unity-side exporter. | Local `ABGameObject.cs` plus Unity 2019.4 APIs. | supported |
| C4 | Support needs a relation rule over contacts; it is not a native field. | Unity contact API exposes geometry/normal/impulse, not an application support label. | supported |
| C5 | Macro events can be emitted authoritatively from current callbacks/state transitions, but are not logged today. | Local game world and collision code. | supported |

| claim_id | statement | type | risk | supporting observations | status |
| --- | --- | --- | --- | --- | --- |
| C1 | The task-template source has a usable scene-state/ground-truth surface. | code | high | O1 | supported |
| C2 | The deployed collector build exposes the same state surface. | code | high | O2 plus assembly metadata | supported, live protocol untested |
| C3 | Per-frame contacts/support can be emitted directly. | code/API | high | local source plus Unity APIs | partial: contacts direct, support derived |
| C4 | Per-frame velocity and kinetic energy can be emitted directly. | code/API | high | local source plus Unity APIs | partial: available but absent from wire schema |
| C5 | Macro-event labels can be emitted from authoritative callbacks/state. | code/API | high | local source | partial: anchors exist, event log absent |
