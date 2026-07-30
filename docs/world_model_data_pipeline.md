# World-Model Data Pipeline

`world_model.data` provides a read-only PyTorch input pipeline for completed
NovPhy rollout episodes. It covers cataloging, temporal RGB/action windows,
deterministic batching, temporal curricula, and temporal-only ablation
accounting. It does not implement a world model, trainer, learned controller,
symbolic labels, or physics instrumentation.

## Catalog Snapshots And Provenance

Build one immutable catalog per existing split:

```python
from pathlib import Path

from world_model.data import EpisodeCatalog, LEGACY_RGB_V1

catalog = EpisodeCatalog.build(
    root=Path("data/rollouts"),
    split="train",
    capture_contract=LEGACY_RGB_V1,
    collection_plan=Path("data/plan/collection_plan.json"),
)
```

Construction enumerates direct, non-symlink episode directories once and
accepts only episodes that satisfy the canonical collector contract. The
resulting episode tuple does not change if collection continues. Call
`catalog.refresh()` explicitly to create a new snapshot; existing datasets,
curriculum policies, states, and manifests remain bound to the original one.

`source_level_key` is trustworthy only when it comes from a supplied collection
plan's exact `selected` record. Use `check_source_key_disjointness()` across the
train, dev, and test catalogs before an experiment. Without a readable plan,
`catalog.provenance_available` is `False`, every source key is `None`, and no
source-level leakage-validation claim can be made. The pipeline never invents a
key from an episode directory name and never infers test examples from train or
dev.

## Capture Contracts And Capabilities

Every episode carries an immutable `CaptureContractDescriptor`: contract and
layout versions, optional player/protocol provenance, declared capability
names, and validated relative sidecar paths. Episodes without an explicit
descriptor use `legacy_rgb_v1`. Explicit unknown contracts do not fall back to
legacy behavior, and the reserved `physics_capture_v1` contract is rejected
until a validating reader exists.

Pass `required_capabilities` when building a catalog to negotiate supervision
requirements. A capability absent from the selected contract fails closed
before episode enumeration. Declared sidecar paths are provenance only in this
pipeline: they are not opened, parsed, tensorized, or used by the dataset,
curriculum, or ablation code.

## Sample Schema

`TemporalWindowDataset(catalog, TemporalWindowRequest(prediction_steps,
stride_frames), transform=None)` materializes eligible shot-local indices but
opens PNG files only in `__getitem__`. The default decoder returns RGB
`float32` tensors in `[0, 1]`, channel first, without implicit resize or
normalization. A sample contains:

| Field | Shape or value |
| --- | --- |
| `context_image` | `[3, H, W]` tensor |
| `target_images` | ordered list of `prediction_steps` tensors `[3, H, W]` |
| `action` | `[5]` tensor: start x/y, release x/y, hold time |
| `frame_indices` | context index followed by stride-spaced target indices |
| `prediction_steps` | requested target count |
| `stride_frames` | frame offset between targets |
| `horizon_frames` | `prediction_steps * stride_frames` |
| `provenance` | split, source key or `None`, episode, shot, capture descriptor, capabilities, sidecar paths |

Frames never cross accepted-shot boundaries. A missing or unreadable cataloged
PNG raises `FrameReadError`; the dataset does not rescan or fabricate a target.

## Sampling And Batch Schema

`EpochSampler(dataset, seed=..., draw_count=...)` uses local Torch generator
state only. Call `set_epoch(epoch)` explicitly. Identical snapshot, seed, epoch,
and draw count reproduce the same indices; no global random state, filesystem
refresh, rank, or world size participates.

`TemporalWindowCollator` stacks equal-geometry RGB contexts and fixed actions,
and pads only target sequences:

| Field | Shape or value |
| --- | --- |
| `context_image` | `[B, 3, H, W]` |
| `target_images` | `[B, T_max, 3, H, W]` |
| `target_mask` | boolean `[B, T_max]`; true only for real targets |
| `action` | `[B, 5]`; no action mask |
| `prediction_steps` | integer `[B]` |
| `provenance` | untouched list of per-sample provenance mappings |

## Curriculum State And Resume

A `CurriculumSchedule` is an ordered, fully covered set of half-open training
step ranges. Each stage declares allowed temporal requests and may filter by
plan-backed novelty level, scenario/type, or normalized start-frame range.
Source filters fail closed when plan provenance is unavailable.

`CurriculumState` records `global_step`, `total_steps`, `schedule_version`,
`schedule_digest`, `catalog_digest`, `sampler_seed`, and `active_stage_name`.
Resume through `CurriculumPolicy.validate_resume(state)` before rebuilding the
candidate view. Any catalog, schedule, seed, step binding, or stage drift is
rejected instead of silently selecting different samples.

## Temporal Ablation Manifests

The named presets are `fixed_short`, `fixed_long`, `temporal_uniform`, and
`temporal_curriculum`. They vary only prediction steps and frame stride while
the representation remains continuous. A `TemporalAblationManifest` records:

- `preset_name` and `preset_identity`;
- `catalog_digest`, `schedule_version`, `schedule_digest`, and
  `active_stage_name`;
- `sampling_seed`, `sampled_provenance_digest`, and `draw_count`;
- `cost_rule_identity`;
- `prediction_steps_distribution`, `stride_frames_distribution`, and
  `horizon_frames_distribution`;
- `total_prediction_steps` and `effective_prediction_steps`;
- `total_base_window_cost`, `total_predicted_frame_cost`,
  `temporal_controller_cost`, and `computed_budget_total`;
- the canonical manifest `digest`.

The non-learned temporal policies report temporal-controller cost as zero.
`sample_matched` means draw count and cost-rule identity are equal.
`compute_matched` independently means computed budget totals are exactly equal.
Neither label implies the other. Serialized comparisons expose these labels as
`sample_match` and `compute_match`, alongside the left and right computed budget
totals. Callers can require compute equality and receive a typed error when
totals differ.

## Read-Only Inspection

Inspect completed train/dev data without creating a report under the rollout
root:

```bash
/home/sukai/miniconda3/bin/python -m world_model.data.inspect \
  --root data/novphy_rollouts_dataset_20260708_171531 \
  --splits train dev \
  --json
```

Optional `--prediction-steps` and `--stride-frames` values report temporal
window feasibility; both must be supplied together. `--capture-contract` and
repeatable `--required-capability` flags negotiate the catalog contract. The
command reports canonically accepted/rejected episodes, accepted shots, typed
rejection counts, frame-length histograms, and requested-window counts. The
per-split JSON fields are `accepted_episodes`, `rejected_episodes`,
`accepted_shots`, and `rejection_counts`. It exits nonzero for invalid
arguments, unsupported requirements, or a requested split with zero accepted
episodes. Skipped live partial episodes are reported as rejections
rather than repaired or deleted.

The inspector validates frame-directory containment/readability, entry type,
symlink safety, contiguous names, frame metadata, and every frame file's read
access without materializing `FrameRecord` values. Normal catalog construction
uses the same canonical predicates while also materializing frame records.
Because the inspector does not accept a collection plan, novelty-level and
scenario composition are reported as `{ "status": "unavailable", "counts": {}
}` instead of being inferred from lossy episode directory names.

The inspector does not accept a collection plan, so its report is a rollout
health report, not source-level leakage validation. Run plan-backed catalog
disjointness separately for experimental split claims.

## Experimental Boundary

This package supports temporal RGB/action experiments over validated snapshots.
It does not read scene nodes, contacts, support relations, kinematics, macro
events, symbolic predicates, or inferred physical labels. Those payload
schemas, engine instrumentation, and validating readers are deferred to the
separate physics-instrumentation plan. The existence of reserved capability
names or sidecar paths is not evidence that such supervision is available.
