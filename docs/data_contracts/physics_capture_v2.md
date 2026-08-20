# `physics_capture_v2`

Status: prospective successor contract for cohort-v2 physical evidence. It does
not revise, decode, or reinterpret `physics_capture_v1` artifacts. A v2
capture is one JSON object stored as `physics_capture_v2.json` beside its
rollout metadata.

## Authority and scope

Unity is the only authority for physical evidence, engine identities, and the
configured fixed-step stride. Request 71 carries those values in a separate
`physics_capture_v2_engine_v1` envelope. The collector adds only the frozen-plan
`source_bindings`, changes the stored schema marker to `physics_capture_v2`,
and validates the complete object before persistence. The contract is for
non-observation evidence only: observations and their access controls are
specified separately. Material, damage, and gravity-shift capabilities are not
declared by this contract.

The root `source_bindings` contains opaque identities for the scenario
template, level instance, scenario lineage, rollout, and intervention. Their
definitions belong to the scenario-lineage contract; this contract only
requires that they are present and binds its state, contact, event, lifecycle,
and terminal records to the same capture and causal identities.

## Required evidence

- `configured_fixed_step_capture_stride` is a positive integer. Scheduled
  `frame_records` are separated by exactly that many fixed steps. When
  termination is off that grid, one final record is explicitly marked
  `forced_terminal: true`; no earlier record may carry that marker. Render
  cadence and target FPS are not accepted fields.
- `fixed_step_samples` covers every fixed step from the first through the last
  retained frame. The first sample must equal the declared
  `pre_intervention_fixed_step`. Each sample explicitly declares complete non-trigger raw
  contact enumeration, supplies direct Unity collider geometry, the world
  gravity vector, and one causal-entity row for every declared causal entity.
- Raw contacts cite declared causal entities and colliders, carry direct
  separation values, and may contain only finite numeric values. The root
  `minimum_contact_separation` is recomputed across every fixed-step contact;
  it explicitly records absence when the capture has no contacts. Collider
  geometry is retained as Unity-authored shape data and is never reconstructed
  from images.
- The root collider catalog freezes collider/entity identities. Every fixed-step
  sample separately retains that step's enabled/trigger state and direct world
  geometry, so moving geometry is never reconstructed from a later request.
  Each row has exactly one supported Unity shape: circle center/radius, box
  center/size/angle, polygon world-space paths, edge world-space points, or
  capsule center/size/direction/angle. Unsupported collider types reject the
  whole capture.
- Every entity row preserves lifecycle and explicit body presence. Present
  bodies carry body type, simulation state, gravity scale and applicability,
  position and rotation, linear velocity, and angular velocity. Support/contact
  context is repeated as deterministic contact, supported-by, and supports ID
  lists and must exactly match the same-step contact/support records. An absent
  body remains explicit rather than becoming a negative label.
- Events carry a declared macro-event type, finite payload, exact resolved
  participant cardinality (two for collision, one for entity events, zero for
  level/stability events), and fixed-step clock. Terminal evidence cites one
  of those events at the same fixed step, its reason equals that event type,
  and the final retained frame covers that exact fixed step.

Malformed evidence, a fixed-step gap, incomplete contacts, unresolved
identities, a non-finite number, missing geometry/gravity applicability, or
uncovered termination rejects the complete capture. Consumers must not infer a
value from filenames, RGB, fixtures, neighbouring steps, or a successful
command. A later violation derivation may mark a label unavailable when its
own narrower evidence window is absent; it must not convert that absence to
false.

The stored sidecar is bounded to 64 MiB, 100,000 fixed-step samples, 2,048
causal entities, 8,192 colliders, 32,768 contacts per step, 100,000 retained
frame records, and 100,000 events. Unity may enforce lower declared limits; an
overflow is a typed whole-capture failure and never a truncation.

## Exporter capability report

`physics_capture_v2_exporter_capability_report_v1` is a prospective report
format for actual Unity exporter probes. It binds exact engine, player,
protocol, and exporter-code SHA-256 identities to the captures it inspected.
It requires non-fixture `unity_exporter_probe` records for no-contact,
collision, support, support-change, and stable-terminal behavior, spanning at
least two non-final scenario lineages, two level instances, and two scenario
templates. Each required fact is either demonstrated by a cited capture digest
or explicitly unavailable. Report validation only verifies this accounting
contract; it is not pilot acceptance and cannot make a semantic label
accepted.
