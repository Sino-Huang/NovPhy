# ULW-Research Synthesis: Science Birds Physical Labels

## Executive Summary

Science Birds already has an engine-authoritative route for **basic per-frame object state**. The deployed build contains the same batch-ground-truth machinery as the Unity task template, and the Python clients expose either a one-shot symbolic-state request or a legacy batch-ground-truth shot request. That state includes instance IDs, screen-space polygons, object labels/types, colors, and life. It is suitable as the starting node set of a scene graph, subject to a live payload smoke test and explicit control of development/noise mode.

The remaining desired labels are not available from the existing JSON schema. They do not require computer-vision inference or a third-party physics package: Unity 2D and the existing `ABGameObject` scripts already possess Rigidbody velocity/mass and collision/contact data. They require a small **Unity-side exporter plus protocol/collector extension** and a rebuild of the shipped game. A Python-only change cannot recover exact contacts, support, mass, or kinetic energy from the current PNGs/ground-truth payload.

## Findings

| Target | Existing direct capability | Needed for research-quality output | Verdict |
| --- | --- | --- | --- |
| Scene graph nodes | IDs, polygons, labels/types, colors, life via `SymbolicGameState` | Align emitted state with each image frame; record raw Unity frame/time and use non-noisy/development configuration deliberately | Mostly available now |
| Contact relations | Unity `ContactPoint2D` and `Collider2D.GetContacts` | Export object IDs, point, normal, separation, relative velocity, impulses per physics tick/frame | Engine exporter needed |
| Support relations | Contact geometry only | Derive a directed rule such as `A supports B` from persistent contact plus upward-facing normal and vertical ordering; retain raw contacts for audit | Derived, not native |
| Velocity and kinetic energy | `Rigidbody2D.velocity`, `mass`; `ABGameObject` stores `lastVelocity` | Export `(vx, vy, angularVelocity, mass)` and compute `0.5*m*||v||^2` downstream, with units and sampling clock | Engine exporter needed |
| Macro events | Launch, collision, death, TNT, clear/fail, unstable/stable anchors exist | Append a timestamped event log with stable IDs and explicit event taxonomy | Engine event exporter needed |

## Evidence and Caveats

- The active collector uses desktop capture and standard `bridge.shoot`, not the batch-GT shot protocol. It currently consults symbolic state only to anchor the slingshot. [Wave 1]
- The shipped assembly has all relevant class/method names, so the source template is not merely an unrelated example. Static metadata cannot prove exact payload format or runtime configuration. [Wave 2]
- The existing batch recorder samples in `Update`, whereas physics is stepped in `FixedUpdate`; exported records must include both Unity frame count and fixed-step/time to prevent false image-to-physics alignment. [Wave 2]
- Existing ground-truth handlers have noise/development-mode branches. Validate actual payload type granularity and noise behavior before treating it as an oracle label. [Wave 1]
- Unity documents that trigger colliders do not yield contact points through `GetContacts`; preserve trigger events separately if they matter. [Unity Collider2D.GetContacts]

## Recommended Research Direction

Treat this as an **engine instrumentation** workstream, separate from the currently approved image/action dataloader. Reuse the existing ground-truth transport for baseline scene nodes. For the extra fields, modify the Unity source at the point where `SymbolicGameState` is sampled, add a typed physical-state/event payload, expose it through a new versioned request (or a versioned batch-GT response), then rebuild the Linux player. Do not infer contacts/support/energy from images when the engine is the source of truth.

## Sources

1. Local `tasks/task_template_designer/Assets/Scripts/GroundTruth/SymbolicGameState.cs:32-368` and `GTObject.cs:27-96`.
2. Local `tasks/task_template_designer/Assets/Scripts/AIBirdsConnection.cs:327-505,1581-1610` and `GameWorld/ABGameWorld.cs:372-395,665-789`.
3. Local `tasks/task_template_designer/Assets/Scripts/GameWorld/ABGameObject.cs:24-35,93-156,258-298`.
4. Local `src/webui/bridge.py:35-40,123-125,173-176`; `sciencebirdsagents/Client/agent_client.py:310-341`; `scripts/collect_rollouts.py:350-364,1184-1335,1465-1502`.
5. Unity 2019.4 APIs: <https://docs.unity3d.com/2019.4/Documentation/ScriptReference/Rigidbody2D-velocity.html>, <https://docs.unity3d.com/2019.4/Documentation/ScriptReference/Rigidbody2D-mass.html>, <https://docs.unity3d.com/2019.4/Documentation/ScriptReference/ContactPoint2D.html>, <https://docs.unity3d.com/2019.4/Documentation/ScriptReference/Collider2D.GetContacts.html>.
