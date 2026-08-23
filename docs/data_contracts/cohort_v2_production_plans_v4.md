# `cohort_v2_production_plans_v4`

Issue #53 plan v4 supersedes the incomplete stable-only v3 determination with
the user-approved mixed termination scope. Its immutable identities are
`cohort-v2-production-collection-plan-v4:issue-53:mixed-termination` and
`cohort-v2-production-parameter-plan-v4:issue-53:mixed-termination`.

All six interventions, their order and offsets, the three non-final lineages,
one-attempt policy, central-stratum quotas, and numeric parameters are retained.
Termination expectation is assignment-bound because the same relative action
has different valid engine outcomes on one-bird and three-bird lineages:

- training and model selection expect `level_fail` for
  `central-no-contact-miss` and `central-persistent-support`;
- every other role/intervention slot expects `stable_entered`.

The resulting production quotas are 0 `level_clear`, 4 `level_fail`, and 20
`stable_entered`. The map is derived from the 18 public non-final v3 outcomes.
No sealed or reviewed v3 final outcome is plan-derivation evidence.

Final evaluation uses a fresh sealed `type010102` lineage at seed 4504. The
public bundle contains only its sealed projection, inventory, partition, and
pending workflow.

Validate and dry-run without opening final data:

```sh
python -u -m scripts.build_issue_53_plan_v4 --validate
python -u -m scripts.capture_issue_53_evidence \
  --plan-root data/runtime_evidence/issue-53-plan-v4 \
  --dry-run
```
