# Decisions — world-model-physics-instrumentation

Architectural choices and rationales discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## 2026-07-31T22:13:36+10:00 - Todo 1

- Freeze independent, sidecar-local strictly increasing sequences. State header/state records use `physics_state.jsonl`; sparse events use `physics_events.jsonl`; neither stream is embedded in `metadata.json`.
- Freeze Unity-derived units by name rather than calling them SI: `unity_unit`, `unity_mass_unit`, and their explicit compound velocity, impulse, and kinetic-energy units.
- Freeze `support_v1` as two retained consecutive fixed-step contacts, `abs(normal_y) >= 0.5`, and at least `0.0001` Unity-unit positive centre-height ordering in both samples. Missing evidence always means no support.
- Keep the public loader small and split immutable types, record decoding, and cross-record validation into focused modules. `scripts/physics_capture_parsing.py` is in the 200-250 warning band at 247 pure lines and must be split before future additions.

## 2026-08-01T00:17:04+10:00 - Todo 2

- Keep `SymbolicGameState.GetGTJson()`, request 62 dispatch/framing, and development/noise branches unchanged. Todo 2 adds a separate intermediate snapshot that Todo 4 can wrap with capture/shot identity and synchronized RGB metadata.
- Allocate dynamic lifetime IDs per Unity instance ID: the first observed lifetime is `<instance-id>:0`, and a different managed Unity object reusing that numeric ID increments the ordinal. Reset both current lifetimes and reuse ordinals at every scene level start and same-scene `DecodeLevel`.
- Use `world:static:<collider-id>` for static nodes. Dynamic Rigidbody2D bodies emit current velocity, angular velocity, mass, and `0.5*m*(vx^2+vy^2)`; static, kinematic, and absent bodies emit `present=false` with all physical values null.
- Todo 2 snapshots include schema version, coordinate/unit declaration, render/fixed clocks, and nodes only. They deliberately contain no RGB descriptor, raw contacts, support edges, events, transport envelope, or persistence behavior.
- Expose snapshot production through `ABGameWorld.CapturePhysicalSnapshot(Action<PhysicalSceneSnapshot>)`; it starts the end-of-render coroutine but deliberately adds no request code or socket framing before Todo 4.
