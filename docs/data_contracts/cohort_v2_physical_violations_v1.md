# Cohort-v2 physical-violation semantics v1

`cohort_v2_physical_violation_derivation_v1` is the central-v2 derivation for only `excess_penetration` and `unsupported_stationary_or_floating_body`. It consumes validated `physics_capture_v2` records without modifying the primary captures or the frozen, rejected `physics_violation_labels_v1` family. `illegal_contact` remains excluded and is not emitted as false.

## Frozen semantics

At each retained fixed step, `excess_penetration` is true when a complete Unity-authored non-trigger contact has separation strictly less than `-0.006` Unity units. Complete enumeration with no such contact is false. Every value cites its exact capture, fixed step, contact and collider identities, coordinate convention, completeness facts, tolerance, and derivation version. Missing geometry, separation, coordinate, or contact-completeness evidence invalidates the whole source rollout.

For each causal entity, `unsupported_stationary_or_floating_body` uses a two-consecutive-fixed-step window. It is true only when both records show an active present body under applicable nonzero world gravity, squared linear speed at most `0.0001`, absolute angular speed at most `0.01` degrees per second, and no accepted support relation anywhere in the window. Complete support, nonstationarity, or inapplicable gravity is false. An incomplete initial window is `unavailable_incomplete_stability_window`; an inactive or absent endpoint is `unavailable_inactive_or_absent_body`. Every value cites the exact lifecycle, body, gravity, world, motion, support/contact, and fixed-step source records. No physical-regime label is consumed or emitted.

`any(violation)` is a frozen unavailable-preserving aggregate over excess penetration and the active-present body domain. Any true component makes it true. It is false only when every in-domain component is available and false. Otherwise it is `unavailable_component`. Inactive or absent entity labels remain separately `unavailable_inactive_or_absent_body` because the concept is undefined, but they are outside the aggregate's body domain; no unavailable input is converted to zero.

## Representative adjudication

The accepted issue #44 Unity probes already provide two excess-penetration positives, negatives, and threshold-crossing windows across two non-final scenario lineages, level instances, and scenario templates. They contain no valid unsupported-stationary positive: the pre-launch bird has gravity explicitly inapplicable, while the post-launch bird is moving. Those records must not be relabeled.

Issue #50 therefore freezes two additional source-bound calibration probe lineages before execution. Each template marks one dynamic block with the exact `unsupported_stationary_v1` capability-probe declaration. The player applies the marker only when `NOVPHY_ISSUE_50_CAPABILITY_PROBE=unsupported-stationary-v1`; otherwise it fails rather than silently collecting a different condition. The plan executes every declared intervention once, retains outcomes independently of their labels, and does not make the witnesses SPSG negative-training examples.

The canonical `data/runtime_evidence/issue-50` bundle records four accepted, zero rejected, and zero failed fresh-engine probes. Accepted determination 1 contains two true and two supported false witnesses across the two new lineages, two stability/support/motion boundary windows, initial-window unavailable evidence, five malformed-source invalidations, one cross-release binding rejection, exact re-derivation, unavailable-preserving aggregate checks, and the existing two-lineage excess-penetration floor. Exact validation passes across nine source captures and 26,999 derived labels.

## Commands

Validate the prospective plan, source identities, generated probe markers, derivation wiring, and the existing excess-penetration windows without building Unity or writing files:

```sh
python -u -m scripts.capture_issue_50_evidence --dry-run
```

After committing the implementation (the provenance-bound player build rejects dirty product source), build the player, collect all four fresh-engine probes, derive and adjudicate both labels, and publish the immutable bundle:

```sh
python -u -m scripts.capture_issue_50_evidence \
  --runtime-root .local-artifacts/issue-50-runtime \
  --output data/runtime_evidence/issue-50
```

The command prints phase changes, collector output, and a heartbeat at least every 15 seconds while a build or probe subprocess is still running. It never hides a rejected or failed attempt behind command success.
