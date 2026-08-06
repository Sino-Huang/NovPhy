# World Model Curriculum Policy

`world_model/data/curriculum.py` is a pure policy layer over an immutable `EpisodeCatalog` snapshot. It does not read frames, sidecars, collector wall-clock metadata, or refresh catalogs.

- Schedules use contiguous ordered half-open stages and bind a fixed `total_steps`.
- Candidates are derived from episode, shot, start frame, and `TemporalWindowRequest`; window eligibility matches the dataset rule.
- `source_level_key` supplies novelty-level and scenario/type filtering. Missing provenance fails requested filters closed.
- Start progress is `start_frame / (frame_count - 1)` and stage ranges are normalized half-open intervals.
- `CurriculumState` records schedule and catalog digests, seed, step, total steps, and active stage. `validate_resume` rejects changes to binding values.
- The active `novphy` Python environment has no torch, so direct test collection is blocked at the existing dataset import. The curriculum test cases were run through a minimal dependency-isolated import while retaining real rollout catalog fixtures.

## 2026-07-29 Independent verification
- Exact-type validation matters at Python numeric boundaries: `True == 1` and `1.0 == 1`, so schedule/checkpoint step fields must check `type(value) is int` before equality.
- Schedule and catalog identities use schema-tagged canonical JSON with `allow_nan=False`, compact separators, and SHA-256. Allowed temporal/filter sets are sorted for identity while stage order remains semantic.
- The one-frame normalized start position is explicitly `0.0`; because temporal requests require positive horizon, a one-frame shot deterministically yields no candidates.
- Real plan-backed filtering uses only `source_level_key` path components. Missing provenance yields no candidates when novelty or scenario filters are requested.
- The policy has no production caller yet beyond tests and reads no filesystem or payload APIs; it is a pure in-memory projection over one `EpisodeCatalog` snapshot.

## 2026-07-29 Temporal ablation accounting
- An ablation manifest must accept a `CurriculumPolicy` and matching `CurriculumState`, call `validate_resume`, and rebuild the bound candidate view; checkpoint identity alone is not sample provenance.
- The four temporal-only preset declarations are `fixed_short=((1,1),)`, `fixed_long=((4,2),)`, and both `temporal_uniform` and `temporal_curriculum` over `((1,1),(2,1),(4,2))`. A preset name and its choices form one validated identity.
- Canonical SHA-256 identities use schema-tagged tuples, never Python `hash()`. Sample provenance includes ordered candidate IDs and `source_level_key`, preserving the distinction between candidate identity and source provenance.
- The selected compute budget is exactly `draw_count * base_window_cost + total_prediction_steps * predicted_frame_cost + temporal_controller_cost`; non-learned temporal policies always report controller cost zero.
- Sample matching is draw-count plus cost-rule identity equality. Compute matching is exact selected-budget equality and is intentionally independent; required unequal compute matches raise a typed error.

## 2026-07-29 End-to-end data-pipeline verification
- Plan-backed provenance is the only valid source-level identity: no-plan catalogs expose `provenance_available=False` and never derive keys from directory names.
- The end-to-end fixture binds train/dev/test catalogs, deterministic epoch sampling, curriculum resume state, collator outputs, sampled-provenance digest, and temporal-ablation manifest digest.
- Active-root inspection must avoid frame-record materialization: millions of immutable records are unnecessary for episode/shot/frame-length summaries and exceed a bounded health-check budget.
- Ordered concurrency is safe for independent episode validation because candidates and returned results remain in sorted input order.
