# Physics Capture v1 Contract Boundary

Recorded 2026-07-31 for `world-model-physics-instrumentation` Todo 1.

- Canonical schema: `docs/data_contracts/physics_capture_v1.schema.json` (JSON Schema Draft 2020-12).
- Normative temporal/support/event rules: `docs/data_contracts/physics_capture_v1.md`.
- Golden sidecars: `tests/fixtures/physics_capture_v1/physics_state.jsonl` and `physics_events.jsonl`.
- Public strict loader: `scripts.physics_capture_contract.load_physics_capture(state_path, event_path)`.
- `physics_capture_v1` remains unsupported by the canonical episode validator until Todo 5 atomically persists and validates enriched shot artifacts. Do not remove that reservation early.
- Every JSONL record repeats capture/shot identity, sidecar-local sequence, render and fixed clocks, and the exact coordinate/unit declaration.
- The only exact image/state claim is same synchronized-endpoint response plus equal Unity `render_frame`; desktop captures are not exact.
- Support cannot be recomputed from snapshots without retained contacts. It requires two cited consecutive fixed-step contacts and the frozen `support_v1` normal/vertical rules.
- Parser files are deliberately dependency-free. `scripts/physics_capture_parsing.py` is 247 pure lines; split it before adding Todo 5 behavior.

Verification command:

```bash
/home/sukai/miniconda3/bin/python -m unittest tests.test_physics_capture_contract -v
```
