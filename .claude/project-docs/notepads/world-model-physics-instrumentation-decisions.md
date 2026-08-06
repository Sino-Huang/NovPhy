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

## 2026-08-03 - External blocker dependency closure

- Use `- [~]` only for blocked plan state, never as completion evidence. Todo 1 remains the sole `- [x]` item.
- Todo 2's Unity Hub license refresh blocks the entire remaining dependency graph. Todos 3-10 and F1-F4 are marked dependency-blocked and must be restored to `- [ ]` before dispatch after Todo 2 is unblocked.
- Keep Boulder paused while the user-only Hub action is outstanding; do not repeat equivalent batch probes or start downstream implementation against an unverified Unity migration.

## 2026-08-03 - Unity Personal minimum 2019.4 version

- Unity Support documents `2019.4.41f2` as the minimum Editor version that Personal users can activate under the current licensing system. The authorized `2019.4.40f1` target is therefore not recoverable by Hub refresh, exact-version Hub launch, command-line credentials, or manual `.alf`/`.ulf` activation.
- Revising the pinned migration target is a user decision. No replacement Editor may be provisioned or probed until the user authorizes `2019.4.41f2` or a later exact version.

## 2026-08-03 - Exact replacement Editor authorized

- The user authorized exact Unity `2019.4.41f2 (6b23d448b533)`. Official Linux archive: `https://download.unity3d.com/download_unity/6b23d448b533/LinuxEditorInstaller/Unity-2019.4.41f2.tar.xz`.
- Preserve `2019.4.40f1` as historical failed-license evidence. All new import, EditMode, build, smoke, and provenance gates use only `2019.4.41f2`.

## 2026-08-04 - Second refreshed-state fail-closed decision

- The single probe authorized after the second supported Hub Licenses refresh failed explicitly after live IPC and entitlement update. Stop before import and preserve Todo 2 as blocked.
- Do not repeat an equivalent probe or begin Todo 3. A future attempt requires another materially changed supported activation state and explicit authorization.

Final verification wave

- Pinned the wave to one exact commit and re-pinned twice rather than disclose findings, both times with explicit user authorisation. Final pin `e2d19ae`, archive `429cac1d`.
- Chose to fix `build_physics_player.sh` rather than accept Unity licensing identity in the published stage log, even though the log is not inside the archive and not part of archive identity.
- Chose to fix `ABBirdLaunchTests` rather than disclose it, because Todo 2 acceptance explicitly requires EditMode tests asserting launch velocity and a test that has never passed does not satisfy that.
- Did NOT weaken the protected-root receipt to make the smoke faster. The scan is already metadata-only; its cost is inherent to a 14.4-million-file root, and weakening it would reduce the guardrail this plan depends on. Raised the timeout bound instead.
- Did NOT overwrite the migration worktree `.omo/boulder.json`. It tracks a different completed work, `world-model-data-pipeline-019fa3ea`. Only the NovPhy boulder, which owns this plan, was updated.
- Left the pre-existing malformed ledger line 24 untouched. It belongs to `world-model-data-pipeline.md` and the ledger is append-only.
