# Cohort-v2 partition and exposure manifest v1

`cohort_v2_partition_exposure_manifest_v1` freezes the central-v2 scenario
lineages approved in issue #45 into the instance-held-out exposure split required
by issues #42 and #47. The public artifact is
`data/runtime_evidence/issue-47/partition-exposure-manifest.json`.

## Partition membership and quotas

The manifest contains the `training`, `calibration`, `model_selection`, and
`final_evaluation` roles in that order. Each entry records its dataset
partition, scenario manifest, benchmark condition, scenario template, level
instance, scenario specification, scenario lineage, declared initial engine
state, sealing state, and lineage-membership quota.

The v1 quota is one prospectively assigned real lineage in each role. Its
`quota_scope` is `partition_lineage_membership`: production and pilot rollout
quotas remain intentionally deferred to the representative pilot and collection
plan under data-spec section 17 and issues #51–#52. Admission also enforces the
approved central evidence floor: at least two non-final lineages, two level
instances, and two scenario templates. The frozen v1 membership supplies three
non-final lineages, three non-final level instances, and two non-final templates.

Role permissions are closed:

- training may influence learned parameters;
- calibration may influence pilot, threshold, and tolerance values;
- model selection may influence configuration selection;
- final evaluation may influence only frozen final metrics after authorization.

## Lineage and template audit

Every rerun, intervention, generation seed, observation configuration,
observation variant, replay, and derivation artifact records its source scenario
lineage and must inherit that lineage's level instance, scenario template,
dataset partition, and exposure role. An observed artifact without a declared
provenance record is rejected. An input seed that changes the initial state
defines a different scenario specification and lineage under the base scenario
contract; that new lineage retains its prospectively assigned role.

Scenario-template identities are retained and reported. Template reuse across
roles is allowed for this instance-held-out regime. The audit creates neither a
template-held-out claim nor a template-held-out score.

`lineage-template-leakage-audit.json` records the real-manifest result and
rejected mutations for a missing role, duplicate lineage, held-out level reuse,
unknown lineage, replay leakage, derivation leakage, observation-variant
leakage, and undeclared artifact provenance.

## Final-evaluation workflow access

`final_evaluation_workflow_access_manifest_v1` binds the partition, workflow,
operator, final lineage inventory, and authorized artifact inventory. The
published issue-47 manifest is frozen with `authorization_state: pending` and
therefore grants no final access.

After a separate authorization, every access record must contain the exact
workflow identity, operator identity, artifact identity, source scenario
lineages, UTC access time, authorization identity, and final-evaluation consumer
role. The auditor rejects a wrong partition, stale final-lineage inventory,
pre-authorization access, non-final workflow, wrong workflow or operator,
undeclared artifact, changed source provenance, access before authorization, or
wrong authorization. Every rejected access raises a typed error carrying a
`final_evaluation_workflow_access_rejection_v1` audit record with the attempted
workflow, operator, artifact, source lineages, time, authorization, role, and
rejection reason.

Regenerate and validate the published evidence with:

```bash
python -m scripts.build_issue_47_evidence
python -c 'from pathlib import Path; from scripts.build_issue_47_evidence import validate_issue_47_evidence; root=Path.cwd(); print(validate_issue_47_evidence(root, root / "data/runtime_evidence/issue-47"))'
```
