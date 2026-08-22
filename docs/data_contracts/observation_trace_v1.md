# Synchronized observation trace v1

`observation_trace_manifest_v1` is the cohort-v2 boundary between engine capture and image-consuming workflows. Each retained frame record binds exactly one pre-transform canonical PNG and one post-transform agent PNG to a single request-72 source-frame identity, fixed step, render frame, camera, viewport, coordinate declaration, units declaration, and world-to-observation transform.

The only accepted capture source is `synchronized_observation_endpoint`. Request 11 screenshots, desktop captures, and images selected by RGB similarity are not admissible. Validators reproduce the declared transform byte-for-byte; missing images are unavailable and are never reconstructed from another source.

The observation configuration identity derives a distinct observation-bound scenario-lineage identity from the source scenario lineage. A source lineage, source frame, or observation artifact cannot cross exposure roles. Transform variants therefore cannot be used to relabel one lineage into another exposure boundary.

Agent observations are the sole image stream available to training, calibration, model selection, comparator selection, and final reported model-input workflows. Canonical observations may be read only by the `alignment_diagnosis` and `capture_diagnosis` purposes in a diagnostic workflow. Every attempted access can be recorded by `observation_access_audit_v1`; rejected attempts return no bytes.

The machine-readable manifest schema is `observation_trace_v1.schema.json`. Benchmark-defined transforms are closed by `observation_transforms_v1.json`.
