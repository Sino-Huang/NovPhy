# `physics_capture_v1`

Status: frozen state/event records with an additive engine-evidence component. Schema versions: `physics_capture_v1` and `physics_violation_engine_evidence_v1`. Artifact layout: the existing `physics_state.jsonl` and `physics_events.jsonl` plus, when the engine supplies it, `physics_violation_engine_evidence_v1.jsonl` inside each accepted shot directory. `metadata.json` references sidecar paths, counts, and checksums but never contains their records.

## Authority and alignment

All physical facts originate in Unity. Every enriched PNG and its state snapshot are returned together by the synchronized endpoint. The PNG descriptor's `render_frame` must equal the state record's Unity `render_frame`; this same-response equality is the only exact RGB alignment guarantee. Desktop screenshots and ordinary screenshot requests are not exact and must not be labeled `physics_capture_v1`.

Each state sidecar starts with one `state_header`, followed by `state` records. The event sidecar contains only `event` records. Every record repeats the full identity, clock, and coordinate declaration: `schema_version`, `capture_id`, `shot_id`, sidecar-local strictly increasing `sequence`, Unity render frame/time, monotonic fixed-step/fixed time, and the frozen coordinate/unit object. State records are sorted by `(render_frame, render_time, fixed_step, fixed_time)`. Event records are sorted by `(fixed_step, render_frame, event taxonomy rank, participants, event_id)` before sequence and event IDs are assigned.

Request 70 retains protocol version 1. Legacy success packets have three length-prefixed components (PNG, state, events) and decode with `evidence=None`. New success packets set the evidence-component flag and carry a fourth length-prefixed JSON component atomically in the same envelope. The component is either `null` when no finalized recorder exists or a closed `physics_violation_engine_evidence_v1` object. A legacy packet never gains inferred violation evidence.

`capture_id` identifies one endpoint capture. `shot_id` identifies the accepted shot. Dynamic `entity_id` is `<unity-instance-id>:<spawn-ordinal>` and remains stable for that lifetime. Static colliders use `world:static:<collider-id>`. Contact IDs are `contact:<fixed-step>:<entity-a>|<collider-a>:<entity-b>|<collider-b>:<point-index>`. Event IDs are `event:<eight-digit-sidecar-sequence>`. Support IDs are `support:<supporter-id>-><supported-id>`.

## Coordinates and Unity units

World coordinates use Unity 2D scene space (`unity_world_2d`), positive x right and positive y up. Screen polygons use RGB pixels (`rgb_pixel_2d`), origin top-left, positive x right and positive y down. Time is in seconds and angles are in degrees. Length, mass, velocity, impulse, and kinetic energy are explicitly named Unity-derived units in the schema. They are not SI quantities and no SI conversion is implied.

## State records

Nodes are lexically sorted by `entity_id`. A node contains its lifetime ID, Unity instance ID, symbolic class and type, screen polygon, Unity world position and rotation, life, and body state. `body.present=false` requires velocity, angular velocity, mass, and kinetic energy to be null. A present body records current values, and kinetic energy is exactly `0.5 * mass * (vx^2 + vy^2)` in the declared Unity-derived energy unit.

Raw contacts include every non-trigger contact point. Participants are canonicalized by lexical `entity_id`; swapping participants also swaps colliders and negates directional normal/relative velocity so `normal_a_to_b` always points from canonical A to B. Each contact contains collider IDs, point, separation, relative velocity, and nullable normal/tangent impulses. Contacts sort by `(entity_a_id, entity_b_id, collider_a_id, collider_b_id, point.x, point.y, contact_id)`.

## `support_v1`

A directed `supporter -> supported` edge exists only when the same canonical non-trigger contact pair is retained in two consecutive fixed steps, `abs(normal_a_to_b.y) >= 0.5`, and `supported.center_y - supporter.center_y >= 0.0001` Unity units in both samples. The edge cites exactly those two contact IDs and fixed steps. Ties sort by `(supporter_id, supported_id, support_id)`. Static/ground contacts remain raw contacts and use synthetic world IDs. Absent, dropped, truncated, or one-step contact history produces no support edge; support is never reconstructed from images or inferred without evidence.

## Event taxonomy

Taxonomy order is `bird_launched`, `collision`, `explosion`, `entity_destroyed`, `pig_removed`, `bird_exhausted`, `stable_entered`, `stable_exited`, `level_cleared`, `level_failed`. Participants are unique and lexically sorted.

| Event | Cardinality and payload |
| --- | --- |
| `bird_launched` | Once per shot; launched bird participant; `launch_velocity`. |
| `collision` | Once per unordered entity pair per fixed step; `contact_ids` and `relative_speed`. |
| `explosion` | Once per exploding entity; `radius_unity_units`. |
| `entity_destroyed` | Once per entity lifetime; `reason`. |
| `pig_removed` | Once per pig lifetime; `reason`. |
| `bird_exhausted` | Once per shot when no playable bird remains; `birds_remaining` is zero. |
| `stable_entered`, `stable_exited` | Emit only on a debounced transition, never repeatedly while unchanged; `debounce_fixed_steps` is positive. |
| `level_cleared`, `level_failed` | Mutually exclusive terminal event, at most one per shot; clear carries `score`, fail carries `reason`. |

## Bounded failure

The header records positive `max_state_records`, `max_event_records`, and `max_total_bytes`. Exceeding a record/byte bound, capture timeout, or truncated finalization returns a typed `capture_failure` envelope with one of `record_limit_exceeded`, `byte_limit_exceeded`, `capture_timeout`, or `truncated_finalization`. A failed envelope never makes a shot acceptable and must not be inserted into either JSONL sidecar. Complete sidecars contain no failure record.

## `physics_violation_engine_evidence_v1`

This sidecar is the sole availability source for later physics-violation derivation. Collectors may persist the engine object unchanged; they may not author completeness, gravity, support, or capability declarations. Its engine shot identity is distinct from the collector directory name, while `capture_id` and request `sequence` bind each cumulative evidence snapshot to its request-70 state.

`fixed_step_coverage` records first/last observed step, sample count, and explicit completeness. The closed incomplete reasons are `no_fixed_step_samples`, `fixed_step_gap`, `contact_sample_overflow`, `entity_sample_overflow`, and `sampling_failure`. Overflow or sampling failure therefore cannot silently become negative evidence. `minimum_contact_separation` is the capture-wide minimum non-trigger separation, with its contact ID and fixed step; Unity accumulates it before ordinary raw-contact retention pruning.

`terminal_trace` contains at most eight consecutive fixed steps and at most 128 tracked dynamic-entity lifetimes per step. Bounds, truncation, and failure reasons are explicit. Each entity row is producer-authored and includes `observed`, `present`, world position, Rigidbody2D `body_type`, `simulated`, and `gravity_scale`; each step records the contemporaneous global `Physics2D.gravity`. `support_v1` presence and its supporting IDs/contact IDs/fixed steps are copied from the recorder's authoritative support graph. Position remains a world-space position; velocity and angular-velocity units are not reused as displacement units.

## Compact failure envelope

This schema-valid example is a rejected capture result, not a sidecar record:

```json physics_capture_v1_example
{"schema_version":"physics_capture_v1","capture_id":"capture-example-001","shot_id":"shot_001","failure_code":"capture_timeout","message":"capture deadline reached","observed":30,"limit":30}
```

Validate the published documentation and this example from the repository root:

```bash
python scripts/verify_physics_capture_docs.py docs
```

## Cohorts, compatibility, and supervision

`physics_capture_v1` is opt-in. A consumer that requires sidecars must select `capture_contract=physics_capture_v1` and pass the required capabilities. The declared capabilities are `scene_nodes`, `raw_contacts`, `derived_support`, `kinematics`, and `macro_events`; state and event sidecars provide the capabilities declared by the immutable contract descriptor.

Legacy `legacy_rgb_v1` episodes remain canonical RGB/action episodes and can be consumed without opening physics sidecars. They do not gain labels later: no retroactive annotation is promised. A physical field in an enriched record is authoritative Unity-exported data, not a label inferred from RGB, a desktop screenshot, or visual analysis.

The optional Todo 10 reader leaves the default RGB/action sample unchanged. It validates the accepted enriched sidecars before returning immutable frame-exact `PhysicsFrameSupervision` records. `include_raw_contacts` and `include_events` control those variable-cardinality collections; derived support remains distinct from raw contacts. Invalid, stale, incomplete, nonaligned, or capability-incompatible sidecars fail closed. The default reader does not open sidecars.

```python
from pathlib import Path

from world_model.data import (
    EpisodeCatalog,
    PHYSICS_CAPTURE_V1,
    PhysicsSupervisionRequest,
    TemporalWindowDataset,
    TemporalWindowRequest,
)

capabilities = ("scene_nodes", "derived_support", "kinematics", "macro_events")
catalog = EpisodeCatalog.build(
    root=Path("data/physics_capture_v1_cohort"),
    split="train",
    capture_contract=PHYSICS_CAPTURE_V1,
    required_capabilities=capabilities,
)
dataset = TemporalWindowDataset(
    catalog,
    TemporalWindowRequest(prediction_steps=1, stride_frames=1),
    supervision=PhysicsSupervisionRequest(
        required_capabilities=capabilities,
        include_raw_contacts=True,
        include_events=True,
    ),
)
supervision = dataset[0]["supervision"]
```

## Staged provenance and operations

The accepted staged player is the archive named in `sciencebirdsgames/physics-v1/archive.sha256`. That receipt is the source of the staged archive digest. Its generated live-smoke provenance report is `.claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json`; it must record the same archive digest, player/protocol digests, an accepted shot, and unchanged protected roots.

Before collection, verify the stage and then rerun the smoke test. Promotion is permitted only after both commands succeed and their reports agree on the archive digest.

```bash
stage=sciencebirdsgames/physics-v1
expected_sha="$(awk 'NF == 2 {print $1}' "$stage/archive.sha256")"
python scripts/verify_physics_player.py --stage "$stage" --expect-sha "$expected_sha"

python scripts/smoke_physics_capture.py \
  --stage sciencebirdsgames/physics-v1 \
  --output-dir "$(mktemp -d)/physics-smoke" \
  --report .claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json
```

Collect a new cohort only after the staged selector has been promoted. This command is deliberately opt-in and uses a root distinct from the active legacy cohort:

```bash physics_capture_v1_collection
PHYSICS_CAPTURE_V1=1 \
PHYSICS_PLAYER_ARCHIVE=sciencebirdsgames/physics-v1/novphy-physics-player-2019.4.41f2.tar.gz \
PHYSICS_SMOKE_MARKER=.claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json \
RESUME=1 OUT_ROOT=data/physics_capture_v1_cohort NOVPHY_YES=1 \
scripts/collect_full_rollout_training_dataset.sh
```

Promotion selects the verified stage through an operator-owned symlink boundary. It requires a prior selector target, retains that target as `previous`, and atomically replaces only the selector. It never copies, renames, or overwrites `sciencebirdsgames/Linux`. Do not run this command during documentation validation.

```bash physics_capture_v1_promotion
set -eu
stage=sciencebirdsgames/physics-v1
selector=sciencebirdsgames/physics-selection
expected_sha="$(awk 'NF == 2 {print $1}' "$stage/archive.sha256")"
archive="$stage/novphy-physics-player-2019.4.41f2.tar.gz"
test "$(sha256sum "$archive" | awk '{print $1}')" = "$expected_sha"
test "$(awk '{print $1}' "$stage/archive.sha256")" = "$expected_sha"
python scripts/verify_physics_player.py --stage "$stage" --expect-sha "$expected_sha"
python - <<'PY'
import json
from pathlib import Path
receipt = Path("sciencebirdsgames/physics-v1/archive.sha256").read_text(encoding="ascii").split()
assert len(receipt) == 2
expected_sha = receipt[0]
report = json.loads(Path(".claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json").read_text(encoding="utf-8"))
assert report["status"] == "accepted"
assert report["protected_unchanged"] is True
assert report["provenance"]["archive_sha256"] == expected_sha
PY
test -L "$selector/current"
ln -s "$(readlink "$selector/current")" "$selector/previous.next"
mv -Tf "$selector/previous.next" "$selector/previous"
ln -s "../physics-v1" "$selector/next"
mv -Tf "$selector/next" "$selector/current"
test "$(readlink "$selector/current")" = "../physics-v1"
```

Rollback restores the retained selector target atomically. It does not invoke collection or modify the production player path:

```bash physics_capture_v1_rollback
set -eu
selector=sciencebirdsgames/physics-selection
test -L "$selector/previous"
target="$(readlink "$selector/previous")"
ln -s "$target" "$selector/rollback"
mv -Tf "$selector/rollback" "$selector/current"
test "$(readlink "$selector/current")" = "$target"
```

A failed build, verification, smoke report, or provenance digest comparison leaves the active cohort and production player untouched; retain the stage and failure report for diagnosis or rebuild a fresh stage with `scripts/build_physics_player.sh`.
