# Cohort-v2 replay evidence v1

`cohort_v2_replay_evidence_v1` is the fail-closed deterministic replay contract for the central cohort-v2 experiment. It binds an original rollout and one independently launched replay to the same non-final scenario lineage, exposure role, source manifest, scenario content, collection plan, intervention, observation configuration, and version envelope while retaining distinct attempt, rollout, capture, and shot identities.

## Version envelope

Every determination records the exact Unity version, staged player profile, source commit and tree, declared player membership count, physics and observation protocol versions, engine/exporter/observation contracts, generator and importer authorities, and runtime code revision. Real collection refuses a dirty tracked worktree before recording that revision. All fields compare exactly. A changed field requires a new plan version and replay determination; it cannot pass an earlier envelope.

Source manifests, materialized scenario XML, source templates, the issue-47 partition manifest, and the issue-46 observation capability reference are retained in the bundle. Validation resolves every declared member and rejects missing, stale, cross-role, or extra bundle membership.

## Comparison rules

The current `cohort_v2_replay_exact_socket_comparison_rules_v1` declaration follows the issue-44 manual replay precedent: the original attempt freezes one exact socket command, and the replay applies that command without remapping it. Observation render provenance remains independent from the intervention authority.

Exact comparisons cover:

- scenario manifest, specification, content, template, level-instance, lineage, partition, exposure-role, plan, and intervention identities;
- the original interface action, its exact frozen socket command, and the engine-relative action;
- normalized initial engine state, coordinate convention, causal-entity catalog, collider catalog, configured fixed-step stride, semantic event/contact/support identities, terminal entity lifecycle, and termination reason;
- world, body, lifecycle, and collider state at every shared launch-relative fixed step; event payloads; and contact collider identities, points, and normals for every compared occurrence;
- observation configuration and access policy, with each agent/canonical trace independently checked for exact synchronization, complete camera/viewport/transform metadata, and its declared canonical-to-agent transform.

The version-bounded tolerances are reported per component:

- first launch-relative physical occurrences and termination may differ by at most one authoritative fixed step;
- repeated callbacks with the same physical semantic identity may differ in count while the semantic identity set and first occurrence remain bounded;
- minimum contact separation may differ by at most `0.001` Unity world units;

Cross-attempt camera framing and pixel equality are not required. Camera, viewport, transform, source-frame synchronization, agent/canonical identities, access roles, and within-attempt transforms remain mandatory and fail closed. Pixel equality can only become required through a prospectively declared successor observation contract.

## Representative evidence

The canonical publication target is `data/runtime_evidence/issue-48`. It contains two scenario collections and interventions across the frozen training and calibration roles, two non-final scenario lineages, two level instances, and two scenario templates. The training pair must demonstrate the collision and raw-contact strata. The calibration pair must demonstrate the stable stratum and retain matched persistent support relations.

Earlier determinations are audit history only. They are never rewritten or used to pass the canonical determination. No attempt may be retried within a plan, and benchmark-agent action provenance is not used or required.

## Commands

Validate the command and all authorities without launching Unity:

```sh
python -u -m scripts.capture_issue_48_evidence --dry-run
```

Run a new determination in fresh immutable destinations with progress logs on stdout:

```sh
python -u -m scripts.capture_issue_48_evidence \
  --runtime-root .local-artifacts/issue-48-replay-run-canonical \
  --output data/runtime_evidence/issue-48
```

The command prints progress for source freezing, each player launch, readiness, camera/slingshot stability, observation capture, intervention, terminal physics capture, comparison, and publication. It exits nonzero for any unavailable component or failed replay verdict.
