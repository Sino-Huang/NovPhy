# Cohort-v2 macro semantics v1

`cohort_v2_macro_derivation_v1` is the accepted central-v2 derivation for only `steady-state` and `structure-unstable`. It consumes validated `physics_capture_v2` fixed-step samples and events without modifying the primary capture or the frozen `physics_macro_labels_v1` family. `cascade-active`, `collapsed`, and `pigs-cleared` remain excluded and are not emitted as false labels.

## Frozen semantics

`steady-state` is a two-fixed-step debounced state. A fixed-step candidate is stable exactly when every dynamic body has squared linear speed at most `0.0001` and absolute angular speed at most `0.01` degrees per second. The initial label is unavailable until two consecutive candidates establish a state. Every later state change must match a same-step engine `stable_entered` or `stable_exited` event.

`structure-unstable` is available only with the immediately preceding complete fixed-step sample and an available `steady-state` label. It is true exactly when `steady-state` is false and the directed engine support-relation set differs from the predecessor. No predecessor produces `unavailable_no_predecessor`; an unavailable stability prerequisite remains unavailable.

Events use fixed step as their occurrence authority. Every predicate label records its exact current/predecessor state interval, events projected into that interval, derivation version, and predicate-specific evidence. Missing history, a fixed-step gap, an inconsistent stability event, or a stale source binding fails closed.

## Representative adjudication

The canonical bundle is `data/runtime_evidence/issue-49`. It is bound to the five accepted non-fixture Unity capability probes in `data/runtime_evidence/issue-44`, spanning two non-final scenario lineages, level instances, and scenario templates.

The frozen adjudication selects the `collision` and `support-change` probes from the two different source lineages. For each predicate it contains two positive and two negative witnesses across both lineages, two entry/exit or pre-threshold/threshold/post-threshold boundary windows, explicit unavailable records, and three rejection mutations. The published adjudication records zero disagreements and accepts both predicates as authoritative under this derivation version.

The bundle contains the derivation specification, prospective adjudication plan, adjudication report, five separately versioned derivations, and an exact membership manifest. Validation regenerates every label and artifact and rejects any changed value or extra/missing member.

## Commands

Validate sources, derivations, witness coverage, and rejection mutations without writing evidence:

```sh
python -u -m scripts.build_issue_49_evidence --dry-run
```

Publish once to a new immutable destination with progress logs on stdout:

```sh
python -u -m scripts.build_issue_49_evidence \
  --output data/runtime_evidence/issue-49
```

The publication command is an offline adjudication over existing real Unity captures; it does not launch Unity. It reports source validation, derived capture/label counts, witness-floor status, mutation-check status, and publication completion.
