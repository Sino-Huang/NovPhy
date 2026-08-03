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
