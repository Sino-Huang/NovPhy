# Decisions — world-model-physics-instrumentation

Architectural choices and rationales discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## 2026-07-31T22:13:36+10:00 - Todo 1

- Freeze independent, sidecar-local strictly increasing sequences. State header/state records use `physics_state.jsonl`; sparse events use `physics_events.jsonl`; neither stream is embedded in `metadata.json`.
- Freeze Unity-derived units by name rather than calling them SI: `unity_unit`, `unity_mass_unit`, and their explicit compound velocity, impulse, and kinetic-energy units.
- Freeze `support_v1` as two retained consecutive fixed-step contacts, `abs(normal_y) >= 0.5`, and at least `0.0001` Unity-unit positive centre-height ordering in both samples. Missing evidence always means no support.
- Keep the public loader small and split immutable types, record decoding, and cross-record validation into focused modules. `scripts/physics_capture_parsing.py` is in the 200-250 warning band at 247 pure lines and must be split before future additions.
