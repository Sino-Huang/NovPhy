# World Model Data Policy

## Catalog and Curriculum

`world_model/data/curriculum.py` is a pure policy layer over one immutable `EpisodeCatalog` snapshot. It must not read frames or sidecars, inspect collector wall-clock metadata, refresh catalogs, or infer provenance from output-directory names.

Schedules use ordered contiguous half-open stages with fixed `total_steps`. Candidate identity includes episode, shot, start frame, and `TemporalWindowRequest`; eligibility must match the dataset temporal-window rule. A one-frame shot has normalized progress `0.0` and no positive-horizon candidate.

## Provenance and Identity

`source_level_key` is the sole source-level identifier for novelty/scenario filtering.

- Requested filters fail closed when provenance is absent.
- No-plan catalogs report provenance unavailable.
- Duplicate source keys are rejected.
- Candidate identity and source provenance remain distinct.
- Schedule, catalog, policy, state, sample, and manifest identities use schema-tagged canonical JSON and SHA-256.
- Canonical serialization disallows NaN, uses compact separators, sorts unordered allowed sets, and retains semantic stage order.
- Never use Python `hash()` for durable identities.
- Strict integer fields use exact type checks; `True` and `1.0` do not satisfy `int` requirements.

## Resume and Ablation

`CurriculumState` binds schedule digest, catalog digest, seed, current step, total steps, and active stage. Resume rejects changed bindings.

An ablation manifest carries matching policy/state, validates resume compatibility, and rebuilds the bound candidate view. Checkpoint identity alone is not sample provenance. Temporal presets are declarations whose name and choices form one identity.

Compute accounting is:

`draw_count * base_window_cost + total_prediction_steps * predicted_frame_cost + temporal_controller_cost`

Non-learned temporal policies report zero controller cost. Sample matching and compute matching are intentionally separate checks.

## Todo 9 Boundary

Todo 9 is established by `WorldModelDataIntegrationTests` and a plan-backed synthetic fixture covering train/dev/test source disjointness, unavailable provenance, duplicate source-key rejection, partial-artifact exclusion, deterministic epoch sampling, collator output, resumed ablation manifests, and provenance/temporal-ablation digest binding.

The active-root inspector is read-only health information. It cannot prove source-level leakage freedom because it does not receive the collection plan.

## Performance

Active-root inspection should avoid materializing frame records when episode, shot, and frame-length summaries suffice. Independent episode validation may run concurrently only when candidate/result order remains deterministic.

Environment availability, dataset counts, completion claims, and test totals are dated evidence, not timeless policy.
