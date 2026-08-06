# Learnings — world-model-physics-instrumentation

Conventions, patterns, and successful approaches discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## 2026-07-31T22:13:36+10:00 - Todo 1

- The completed data-pipeline boundary intentionally reserves `physics_capture_v1` as unsupported in `scripts/rollout_artifacts.py`; Todo 1 therefore provides a standalone strict sidecar parser without changing episode acceptance ahead of Todo 5.
- The repository had no committed fixture directory convention, so golden sidecars live at `tests/fixtures/physics_capture_v1/` and are loaded read-only by `tests/test_physics_capture_contract.py`.
- `/home/sukai/miniconda3/bin/python` remains the verified interpreter. The unchanged canonical baseline and post-change regression command each passed 113 tests.
- Exact RGB alignment is represented only by `rgb_frame.source=synchronized_endpoint` and equal record/PNG `render_frame`; desktop capture is explicitly excluded.

## 2026-08-01T00:17:04+10:00 - Todo 2

- A separate exporter can reuse the existing symbolic geometry/type extraction without touching legacy serialization by parsing `SymbolicGameState.GetGTJson(false)` and enriching matched live Unity objects.
- `WaitForEndOfFrame` is the correct runtime boundary after camera/UI rendering, but Unity 2019.3 does not invoke it in editor batch mode. The EditMode fixture therefore verifies the coroutine's yield boundary directly while the synchronous capture path verifies all values under `-batchmode -nographics`.
- `FindObjectsOfType` order is unspecified. Stable output requires lexical sorting by final `entity_id`, independent of Unity enumeration and legacy feature order.

## 2026-08-03 - Unity Hub desktop-session licensing

- Opening the privately extracted Unity Hub through SSH X11 forwarding can separate it from the desktop D-Bus/keyring session and make persisted authentication unavailable.
- The user resolved the visible Hub state by connecting through remote desktop and opening Unity Hub directly in that desktop session. Treat this as a material external-state change that permits one bounded batch license probe, not as license acceptance by itself.

## 2026-08-04 - Unity Hub refresh is not activation proof

- A second reported supported Licenses refresh again permitted one bounded probe but did not satisfy Editor activation: live IPC and entitlement refresh can succeed before `Missing or bad username or password` terminates the Editor.
- Accept activation only from the complete probe log with no activation, fallback, crash, timeout, or unresolved-entitlement signal; neither the user action nor exit code alone is sufficient.

Final verification wave

- `-runTests` with `-quit` makes Unity exit batchmode before running any test, writing no XML while still exiting 0. That reads as success and is not.
- Every headless EditMode run crashes at shutdown inside `CefBrowserMessageLoop::DoMessageLoopIteration`, exiting 134 or 139 AFTER writing a complete NUnit receipt. The XML is the authority, never the process exit code.
- EditMode does not raise `Awake` for `AddComponent`. `ABGameObject._rigidBody` stays null and `PhysicalSnapshotRuntime.Active` stays unbound, which silently turns every static record callback into a no-op.
- Running EditMode partitions by class is necessary, but a partial set silently hid a permanently failing test across many prior receipts. Enumerate every class.
- `provenance.json` embeds `project.git_head` and `project.git_tree`, so any commit changes archive identity even when no shipped byte changes. Budget a full rebuild, republish, and re-smoke for every re-pin.
- The smoke-timeout anomaly was never an engine, Xvfb, or socket problem. `smoke_protection.nested_manifest_digest` enumerates 14,432,052 files across 535 GB, twice per run.
