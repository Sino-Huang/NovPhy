# `physics_capture_v1`

Status: frozen. Schema version: `physics_capture_v1`. Artifact layout: two JSONL sidecars named `physics_state.jsonl` and `physics_events.jsonl` inside each accepted shot directory. `metadata.json` may reference their relative paths and counts in later collector work, but it never contains state or event records.

## Authority and alignment

All physical facts originate in Unity. Every enriched PNG and its state snapshot are returned together by the synchronized endpoint. The PNG descriptor's `render_frame` must equal the state record's Unity `render_frame`; this same-response equality is the only exact RGB alignment guarantee. Desktop screenshots and ordinary screenshot requests are not exact and must not be labeled `physics_capture_v1`.

Each state sidecar starts with one `state_header`, followed by `state` records. The event sidecar contains only `event` records. Every record repeats the full identity, clock, and coordinate declaration: `schema_version`, `capture_id`, `shot_id`, sidecar-local strictly increasing `sequence`, Unity render frame/time, monotonic fixed-step/fixed time, and the frozen coordinate/unit object. State records are sorted by `(render_frame, render_time, fixed_step, fixed_time)`. Event records are sorted by `(fixed_step, render_frame, event taxonomy rank, participants, event_id)` before sequence and event IDs are assigned.

`capture_id` identifies one endpoint capture. `shot_id` identifies the accepted shot. Dynamic `entity_id` is `<unity-instance-id>:<spawn-ordinal>` and remains stable for that lifetime. Static colliders use `world:static:<collider-id>`. Contact IDs are `contact:<fixed-step>:<entity-a>|<collider-a>:<entity-b>|<collider-b>:<point-index>`. Event IDs are `event:<eight-digit-sidecar-sequence>`. Support IDs are `support:<supporter-id>-><supported-id>`.

## Coordinates and Unity units

World coordinates use Unity 2D scene space, positive x right and positive y up. Screen polygons use RGB pixels, origin top-left, positive x right and positive y down. Time is in seconds and angles are in degrees. Length, mass, velocity, impulse, and kinetic energy are explicitly named Unity-derived units in the schema. They are not SI quantities and no SI conversion is implied.

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
