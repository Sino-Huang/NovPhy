# Cohort-v2 source-bound scenario lineage v1

`scripts/cohort_v2_scenarios.py` adds prospective cohort-v2 planning artifacts
without changing `scenario_manifest_v1` or any existing pilot, cohort, release,
or derivation artifact.

## Scenario-template record

`scenario_template_v1` binds one declared `source_reference`, its exact XML-byte
content identity, and its declared benchmark conditions. Its identity is derived
from all recorded source facts. A consumer that has the source file must load the
record with `source_path`; different bytes fail validation. The source reference
is declared provenance, never inferred from a directory or filename.

For generator-backed templates, the record also embeds
`scenario_template_constraints_v1`: the exact constraints-workbook byte identity,
declared workbook reference, sheet and row, canonical generator template name,
and the explicit reference, minimum, and maximum coordinate pairs from columns
C–H. Materialization validates the workbook bytes and requires its request to
equal those recorded generator inputs. Legacy-static records have no generator
constraint binding.

The materialization boundary reports typed `ScenarioLineageError.reason` values
for `missing_template_identity`, `unresolved_source_provenance`, and
`content_drift`. These reasons cover absent identity, unresolved or inconsistent
declared source inputs, and changed template/workbook bytes respectively.

## Cohort-v2 scenario manifest

`cohort_v2_scenario_manifest_v1` embeds a validated template record and an
unchanged `scenario_manifest_v1`.

- Generated content must use that template-record identity and declare the same
  template-content identity in its generation inputs.
- Imported content remains `legacy_static`; it must cite the record's actual
  source reference and match its content identity. It never gains a seed or
  generated provenance.
- `smoke_only` content is rejected from this wrapper. It cannot become eligible
  through a template record, inventory role, path, or command result.

Creation requires the actual scenario XML. Loading can revalidate both XML and
the template source by supplying `xml_path` and `template_source_path`.

## Deterministic receipts

`deterministic_scenario_receipt_v1` records either an identical-input
reproduction or a changed declared-input case. The identical case binds the
scenario specification content identity and the *declared* initial-engine-state
identity. The changed case requires one changed declared-input value and distinct
scenario-specification and lineage identities. `generation_seed` is resolved
from generation provenance rather than the generator-specific `declared_inputs`
mapping. A changed-input comparison fails with typed `content_drift` when its
source-bound template records differ and `cross_lineage_reuse` when the changed
authority reuses a scenario-specification or lineage identity.

Those two receipt kinds establish deterministic content and declared-state
behavior only. A third `unity_reset_reproduction` receipt binds two independent
capture digests to the same source-bound scenario and requires their
`normalized-initial-engine-state-v1` identities to match. Request, rollout, and
capture IDs are excluded from that normalization; world, causal entities,
per-step collider geometry, body state, contacts, and supports are included. A
mismatch is `initial_state_mismatch`. The receipt is engine reproduction
evidence, not representative pilot or semantic-label acceptance.

## Reviewed central-v2 inventory

`central_v2_scenario_inventory_draft_v1` binds the approved central-v2 producer
scope claim and requires exactly one entry for each exposure role. Training,
calibration, and model selection entries are `planned_non_final`; final evaluation
is `sealed_final`. It rejects reused level instances or scenario lineages and
requires at least two source-bound non-final templates.

Inventory entries contain public scenario identities and the digest of the
canonical cohort-v2 scenario manifest artifact. Non-final entries contain an
ordinary manifest reference that validation resolves beneath its declared
manifest root, verifies by exact bytes, and compares with every public identity.
The final-evaluation entry instead contains an opaque sealed reference. Ordinary
validation never resolves that reference, and the entry never contains an
embedded manifest, generation seed, declared inputs, or parameter realization.

`central_v2_scenario_inventory_v1` can be created only from a valid draft and
binds that draft identity together with the approving issue-comment author and
URL. The reviewed inventory remains an administrative, prospective planning
artifact; it does not make a capability empirically accepted or grant access to
final-evaluation data. Typed failures distinguish `initial_state_mismatch`,
`cross_lineage_reuse`, `content_drift`, and `smoke_only`. Draft and reviewed
writers are immutable: a different payload at an existing path is refused.
